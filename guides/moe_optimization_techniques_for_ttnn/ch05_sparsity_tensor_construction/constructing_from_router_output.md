# Constructing the Sparsity Tensor from Router Output

This file walks through the complete pipeline for converting top-k router output into a TTNN sparsity tensor ready for `ttnn.sparse_matmul`. Each step is described in prose and then illustrated in the code example at the end.

All parameter values use Qwen3.5-35B: $E=256$, $k=8$, $N=8$, $E_d=32$, $H=7168$, $K_t=224$.

For the format contract (shape, dtype, layout) that the output of this pipeline must satisfy, see `sparsity_tensor_format.md`.

---

## Step 1: Router Output Format

The MoE router selects the $k$ most relevant experts for each token. Its output is a tensor of **top-k indices** of shape $[B \times S, k]$ containing integer values in $[0, E)$:

$$\text{routing indices} \in \mathbb{Z}^{B \times S \times k}, \quad \text{routing indices}[t, i] \in [0, E)$$

where $t$ indexes tokens (flattened over batch and sequence dimensions) and $i$ indexes the $k$ selected experts for that token.

**Filtering to local experts:** In a multi-device configuration, device $d$ holds experts with global indices in $[d \times E_d,\ (d+1) \times E_d)$. The local expert index is:

$$e_{\text{local}} = e_{\text{global}} - d \times E_d$$

To find which routing decisions are relevant to device $d$, filter `routing_indices` to entries in $[d \times E_d,\ (d+1) \times E_d)$ and subtract $d \times E_d$ to obtain local indices in $[0, E_d)$.

---

## Step 2: Construct Token-to-Expert Assignment

The assignment maps each token to each local expert, subject to capacity limits. The result is a boolean assignment mask of shape $[E_d, C]$:

$$\text{assignment}[e, s] = 1 \iff \text{token slot}\ s\ \text{of expert}\ e\ \text{is occupied}$$

**Expert capacity:** The maximum number of tokens that can be routed to a single expert is:

$$C = \left\lceil \frac{k \times B \times S}{E} \right\rceil$$

For Qwen3.5-35B with $B=4$, $S=2048$:

$$C = \left\lceil \frac{8 \times 4 \times 2048}{256} \right\rceil = \left\lceil \frac{65536}{256} \right\rceil = 256$$

For decode ($B=1$, $S=1$):

$$C = \left\lceil \frac{8 \times 1 \times 1}{256} \right\rceil = \left\lceil 0.03125 \right\rceil = 1$$

**Capacity overflow:** If more than $C$ tokens are routed to expert $e$, the excess tokens are dropped. Drop policy is typically first-come-first-served (tokens earlier in the batch fill slots first). This is the standard behavior of the capacity factor mechanism introduced in Chapter 1.

The assignment is constructed by a scatter operation: for each (token, expert) pair where the token selected that expert and a slot is available, set `assignment[local_expert, slot] = 1`.

---

## Step 3: Detect Active Expert Slots

After Step 2, `assignment` has shape $[E_d, C]$ with 1s at occupied slots. The activation tensor $A$ used in `sparse_matmul` has shape $[E_d \times C, H]$ (flattened multi-expert view), where row $e \times C + s$ corresponds to expert $e$, slot $s$.

A slot $(e, s)$ is active if `assignment[e, s] == 1`. The corresponding row in the flattened activation tensor is $e \times C + s$.

At this point, the active-slot information is still at per-row granularity. Step 4 collapses it to tile granularity.

---

## Step 4: Collapse to Tile Granularity

`ttnn.sparse_matmul` operates at tile-row granularity (32 rows at a time). The mask must represent which **tile rows** of the flattened $[E_d \times C, H]$ activation tensor are active.

**Tile row definition (flattened view):** Tile row $m$ covers activation rows $[m \times 32,\ (m+1) \times 32)$. Tile row $m$ is **active** if any of those activation rows is an active expert slot:

$$\text{tile active}[m] = \bigvee_{r = m \times 32}^{(m+1) \times 32 - 1} \text{assignment flat}[r]$$

where $\text{assignment flat}$ is `assignment` reshaped to $[E_d \times C]$.

The total number of tile rows is:

$$E_d \times M_t = E_d \times \left\lceil \frac{C}{32} \right\rceil$$

For Qwen3.5-35B with $E_d=32$, $C=1$: $E_d \times M_t = 32 \times 1 = 32$ tile rows.
For Qwen3.5-35B with $E_d=32$, $C=256$: $E_d \times M_t = 32 \times 8 = 256$ tile rows.

**Broadcasting across K_t columns:** Because sparsity is in the activation ($M$) dimension only — the $K$-dimension is always fully active — the mask for an active tile row $m$ has all $K_t = 224$ entries set to `1`, and an inactive tile row has all $K_t$ entries set to `0`.

Construct the mask by:
1. Computing `tile_active` as a 1D boolean vector of shape $[E_d \times M_t]$ (one flag per tile row).
2. Expanding to 2D by repeating across $K_t$ columns: `mask = tile_active.unsqueeze(1).expand(-1, K_t)`.

The result has shape $[E_d \times M_t, K_t]$ with dtype `torch.uint8` (cast from bool).

> **Warning:** Do not set partial tile rows to `0` just because they contain fewer than 32 real tokens. If a tile row contains even one real token, its mask value must be `1`. See `sparsity_tensor_format.md`, Section 7, and `common_pitfalls.md`, P2.

---

## Step 5: Transfer to Device

Transfer the PyTorch uint8 mask to the TTNN device with the correct dtype, layout, and memory config: use `ttnn.from_torch` with `dtype=ttnn.uint8`, `layout=ttnn.TILE_LAYOUT`, and `memory_config=ttnn.L1_MEMORY_CONFIG` (see `sparsity_tensor_placement.md`, Section 3 for the full snippet and placement rationale).

For decode sizes ($E_d \times M_t = 32$, $K_t = 224$), the mask occupies 7 tiles × 1024 bytes = 7168 bytes in L1. This is well within L1 capacity (1.5 MB per Tensix core on Wormhole B0).

> **Tip:** If mask construction happens on the hot path of the decode loop, consider pre-allocating the TTNN mask tensor once at model initialization and overwriting its buffer each step using `ttnn.copy_` rather than calling `ttnn.from_torch` on every step. See `sparsity_tensor_placement.md` for TTNN Trace integration.

---

## Complete Code Example

The following pseudocode implements the full pipeline from routing indices to a TTNN sparsity tensor. Shapes and dtypes are annotated at each step.

```python
import torch
import ttnn

# ── Qwen3.5-35B parameters ──────────────────────────────────────────────────
E    = 256    # total experts across all devices
k    = 8      # top-k experts per token
N    = 8      # number of devices
E_d  = E // N  # = 32, local experts per device
H    = 7168   # hidden dimension
K_t  = H // 32  # = 224, tile columns in activation tensor


def compute_expert_capacity(B: int, S: int) -> int:
    """C = ceil(k * B * S / E)"""
    import math
    return math.ceil(k * B * S / E)


def build_sparsity_tensor(
    routing_indices: torch.Tensor,  # shape [B*S, k], dtype torch.int64, values in [0, E)
    device_id: int,                 # which device (0-indexed)
    B: int,
    S: int,
    mesh_device,
) -> "ttnn.Tensor":
    """
    Convert top-k routing indices to a TTNN sparsity mask for device `device_id`.

    Returns a ttnn.Tensor of shape [E_d * M_t, K_t], dtype ttnn.uint8,
    layout TILE_LAYOUT, placed in L1.
    """
    import math

    T = B * S  # total tokens
    C = compute_expert_capacity(B, S)
    M_t = math.ceil(C / 32)

    # ── Step 1: Filter routing indices to local experts ──────────────────────
    # routing_indices: [T, k], int64, values in [0, E)
    global_lo = device_id * E_d          # e.g., 0 for device 0, 32 for device 1
    global_hi = (device_id + 1) * E_d   # exclusive upper bound

    # Boolean mask of which (token, slot) pairs are local to this device
    is_local = (routing_indices >= global_lo) & (routing_indices < global_hi)
    # is_local: [T, k], bool

    # Local expert indices (only meaningful where is_local == True)
    local_indices = routing_indices - global_lo  # [T, k], int64 (values 0..E_d-1 where local)

    # ── Step 2: Scatter into assignment mask [E_d, C] ────────────────────────
    # assignment[e, slot] = 1 if slot `slot` of local expert `e` is occupied.
    # Tokens are assigned to experts in order; the first C tokens routed to
    # expert e fill slots 0..C-1; overflow tokens are dropped.
    assignment = torch.zeros(E_d, C, dtype=torch.uint8)  # [E_d, C]

    # Slot counters: how many tokens have been assigned to each local expert so far
    slot_counter = torch.zeros(E_d, dtype=torch.int64)  # [E_d]

    for t in range(T):
        for i in range(k):
            if is_local[t, i]:
                e = local_indices[t, i].item()  # int in [0, E_d)
                slot = slot_counter[e].item()
                if slot < C:
                    assignment[e, slot] = 1
                    slot_counter[e] += 1
                # else: capacity overflow — token dropped

    # assignment: [E_d, C], uint8

    # ── Step 3: Flatten to [E_d * C] for tile-row computation ────────────────
    assignment_flat = assignment.reshape(E_d * C)  # [E_d * C], uint8
    # Row r in the activation tensor corresponds to assignment_flat[r]

    # ── Step 4: Collapse to tile-row granularity ──────────────────────────────
    # Pad to E_d * M_t * 32 rows (the padded activation height)
    padded_len = E_d * M_t * 32
    assignment_padded = torch.zeros(padded_len, dtype=torch.uint8)  # [E_d * M_t * 32]
    assignment_padded[:E_d * C] = assignment_flat

    # Reshape to [E_d * M_t, 32] — each row is one tile row's worth of token slots
    tile_rows = assignment_padded.reshape(E_d * M_t, 32)  # [E_d * M_t, 32]

    # A tile row is active if any of its 32 slots is occupied
    tile_active = (tile_rows.sum(dim=1) > 0).to(torch.uint8)  # [E_d * M_t]
    # tile_active: [32] for decode (M_t=1, E_d=32); [256] for prefill (M_t=8, E_d=32)

    # Broadcast across K_t columns — sparsity is in M only, not K
    mask_torch = tile_active.unsqueeze(1).expand(E_d * M_t, K_t).contiguous()
    # mask_torch: [E_d * M_t, K_t], uint8
    # e.g., decode: [32, 224]; prefill: [256, 224]

    # ── Step 5: Transfer to device ────────────────────────────────────────────
    mask_ttnn = ttnn.from_torch(
        mask_torch,
        dtype=ttnn.uint8,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    # mask_ttnn: shape [E_d * M_t, K_t], dtype uint8, TILE_LAYOUT, L1

    return mask_ttnn


# ── Example invocation ───────────────────────────────────────────────────────

# Simulate router output for B=1, S=1 (single-token decode)
B, S = 1, 1
T = B * S  # = 1

# Router selects k=8 experts for the single token
routing_indices = torch.randint(0, E, size=(T, k), dtype=torch.int64)
# routing_indices: [1, 8], int64

# Build sparsity tensor for device 0 (local experts 0–31)
# (In real code, mesh_device would be a ttnn.MeshDevice)
mask_ttnn = build_sparsity_tensor(
    routing_indices=routing_indices,
    device_id=0,
    B=B,
    S=S,
    mesh_device=None,  # replace with actual ttnn.MeshDevice in production
)
# Expected: mask_ttnn.shape == [32, 224], dtype uint8, TILE_LAYOUT, L1
# Expected: exactly 1 tile row is 1 (the local expert chosen by the token),
#           31 tile rows are 0 (inactive local experts)

print(f"Mask shape: {mask_ttnn.shape}")           # [32, 224]
print(f"Active tile rows: {(mask_ttnn[:, 0] > 0).sum()}")  # 0 or 1 depending on routing


# ── Vectorized production version (avoids Python loop) ───────────────────────

def build_sparsity_tensor_vectorized(
    routing_indices: torch.Tensor,  # [T, k], int64, values in [0, E)
    device_id: int,
    B: int,
    S: int,
    mesh_device,
) -> "ttnn.Tensor":
    """
    Vectorized implementation using scatter_add for efficiency.
    Equivalent output to build_sparsity_tensor but avoids Python loops.
    """
    import math

    T = B * S
    C = compute_expert_capacity(B, S)
    M_t = math.ceil(C / 32)

    global_lo = device_id * E_d
    global_hi = (device_id + 1) * E_d

    # Filter to local experts
    is_local = (routing_indices >= global_lo) & (routing_indices < global_hi)
    # is_local: [T, k], bool

    local_indices = (routing_indices - global_lo).clamp(0, E_d - 1)
    # local_indices: [T, k], int64 (clamped; only valid where is_local)

    # Build a hit-count tensor: how many tokens want to go to each local expert
    # Shape: [E_d], counting total token-expert activations
    hits = torch.zeros(E_d, dtype=torch.int64)
    for i in range(k):
        valid = is_local[:, i]  # [T], bool
        experts_this_slot = local_indices[:, i][valid]  # subset of [T]
        hits.scatter_add_(0, experts_this_slot, torch.ones_like(experts_this_slot))
    # hits[e] = number of tokens that selected local expert e

    # Active experts: those with at least 1 token (up to capacity C)
    # For each local expert, the number of active slots is min(hits[e], C)
    active_slots = hits.clamp(max=C)  # [E_d], int64

    # Build assignment [E_d, C]: first active_slots[e] entries in row e are 1
    slots_range = torch.arange(C).unsqueeze(0).expand(E_d, -1)  # [E_d, C]
    assignment = (slots_range < active_slots.unsqueeze(1)).to(torch.uint8)
    # assignment: [E_d, C], uint8

    # Flatten, pad, reshape to tile rows
    assignment_flat = assignment.reshape(E_d * C)
    padded_len = E_d * M_t * 32
    assignment_padded = torch.zeros(padded_len, dtype=torch.uint8)
    assignment_padded[:E_d * C] = assignment_flat

    tile_rows = assignment_padded.reshape(E_d * M_t, 32)
    tile_active = (tile_rows.sum(dim=1) > 0).to(torch.uint8)  # [E_d * M_t]

    mask_torch = tile_active.unsqueeze(1).expand(E_d * M_t, K_t).contiguous()
    # mask_torch: [E_d * M_t, K_t], uint8

    mask_ttnn = ttnn.from_torch(
        mask_torch,
        dtype=ttnn.uint8,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    return mask_ttnn
```

### Shape Trace (Decode, B=1, S=1, Device 0)

| Step | Tensor | Shape | dtype |
|------|--------|-------|-------|
| Router output | `routing_indices` | `[1, 8]` | `torch.int64` |
| Local filter | `is_local` | `[1, 8]` | `torch.bool` |
| Assignment | `assignment` | `[32, 1]` | `torch.uint8` |
| Flattened + padded | `assignment_padded` | `[1024]` | `torch.uint8` |
| Tile rows | `tile_rows` | `[32, 32]` | `torch.uint8` |
| Tile active flags | `tile_active` | `[32]` | `torch.uint8` |
| Mask (PyTorch) | `mask_torch` | `[32, 224]` | `torch.uint8` |
| Mask (TTNN) | `mask_ttnn` | `[32, 224]` | `ttnn.uint8`, TILE_LAYOUT, L1 |

### Shape Trace (Prefill, B=4, S=2048, Device 0)

| Step | Tensor | Shape | dtype |
|------|--------|-------|-------|
| Router output | `routing_indices` | `[8192, 8]` | `torch.int64` |
| Local filter | `is_local` | `[8192, 8]` | `torch.bool` |
| Assignment | `assignment` | `[32, 256]` | `torch.uint8` |
| Flattened + padded | `assignment_padded` | `[8192]` | `torch.uint8` |
| Tile rows | `tile_rows` | `[256, 32]` | `torch.uint8` |
| Tile active flags | `tile_active` | `[256]` | `torch.uint8` |
| Mask (PyTorch) | `mask_torch` | `[256, 224]` | `torch.uint8` |
| Mask (TTNN) | `mask_ttnn` | `[256, 224]` | `ttnn.uint8`, TILE_LAYOUT, L1 |

---

## References

- `sparsity_tensor_format.md` — format contract: required shape, dtype, layout, and partial-tile rules
- `sparsity_tensor_placement.md` — L1 vs. DRAM placement and TTNN Trace integration
- `common_pitfalls.md` — P2 (partial tile rows), P3 (stale mask), P5 (wrong dtype), P6 (recompilation)
- Chapter 1 — MoE routing, top-k selection, expert capacity definition
- Chapter 4, `sparse_matmul_internals.md` — how the kernel consumes the mask

---

**Next:** [sparsity_tensor_placement.md](./sparsity_tensor_placement.md)
