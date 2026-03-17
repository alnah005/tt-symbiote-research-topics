# sparse_matmul Internals

This file explains how `ttnn.sparse_matmul` differs from standard `ttnn.matmul` at the kernel level. It covers the sparsity mask structure, the tile-skip mechanism, the static-shape constraint and its rationale, the BF16/BFP8 dtype configuration for MoE, and the FLOP cost model.

All matmul vocabulary (`per_core_M`, `in0_block_w`, `out_subblock_h`, tile layout) is defined in Chapter 2 (`matmul_fundamentals_in_ttnn.md`) and is used here without re-derivation.

---

## 1. What sparse_matmul Does Differently

A standard `ttnn.matmul` over an activation tensor $A$ of shape $[C, H]$ (tiles: $[M_t, K_t]$) and a weight tensor $W$ of shape $[H, D]$ (tiles: $[K_t, N_t]$) computes every output tile in the $[M_t, N_t]$ output grid. For each output tile $(m, n)$, the kernel executes a full K-reduction inner loop:

$$O[m, n] = \sum_{k=0}^{K_t - 1} A[m, k] \cdot W[k, n]$$

When many rows of $A$ are padding zeros — as happens in MoE at low batch size — this inner loop still runs in full, executing FMAs against zero-valued tiles.

`ttnn.sparse_matmul` adds a **tile-skip mechanism**: before loading the tile pair $(A[m, k], W[k, n])$ for each K step, the kernel checks a bit in a pre-computed sparsity mask. If the mask bit for tile $(m, k)$ is zero, the entire K-step for that $(m, n)$ output is skipped: neither $A[m, k]$ nor the corresponding $W[k, n]$ tile is loaded, and the FMAs are not issued.

The key asymmetry: **activation tiles are the sparse dimension.** Weight tiles $W[k, n]$ are only read for K steps where the corresponding activation tile $A[m, k]$ is active. If no activation tile in row $m$ is active (i.e., the entire row-of-tiles $m$ is zero), none of the weight tiles for that $m$ are read either.

---

## 2. The Sparsity Mask

### 2.1 Structure

The sparsity mask is a per-tile boolean (or integer) tensor that encodes which activation tiles are non-zero. For an activation tensor $A$ of shape $[C, H]$ with tile dimensions $[M_t, K_t]$ where:

$$M_t = \left\lceil \frac{C}{32} \right\rceil, \qquad K_t = \left\lceil \frac{H}{32} \right\rceil = \left\lceil \frac{7168}{32} \right\rceil = 224$$

the mask has shape:

$$\text{mask shape} = [M_t, K_t] = \left[\left\lceil \frac{C}{32} \right\rceil,\; 224\right]$$

Each element `mask[m, k]` is 1 if tile $A[m, k]$ contains at least one non-zero element, and 0 if the tile is all zeros (pure padding).

### 2.2 Granularity

Sparsity in `sparse_matmul` is at the **tile level (32×32 elements)**, not the element level. A tile is either fully active (mask bit = 1, K-loop executes) or fully inactive (mask bit = 0, K-loop is skipped for that tile row). There is no sub-tile granularity: a tile with even one non-zero element is treated as fully active.

This has an important implication for MoE: in the canonical configuration, each row-of-tiles in $A$ corresponds to one or more token capacity slots. When an expert receives zero tokens, all 32 rows (one tile's worth) of its capacity slot in the activation tensor are zero. The mask marks the entire tile row as inactive, and the kernel skips all $K_t = 224$ K-steps for that tile row.

### 2.3 Constructing the Mask

The mask is derived from the routing output. After the gather step (see Chapter 3, `formulating_batched_matmul.md` §1.2), each expert's token slots are known. For expert $e$ with capacity slot $c$ in the padded $[E, C, H]$ activation tensor:

```python
import torch
import ttnn

# After gather: expert_tokens [E, C, H] — zero where no token was assigned
# Compute tile-level mask: [E, M_t, K_t]
# M_t = ceil(C / 32), K_t = ceil(H / 32) = 224

def build_sparsity_mask(expert_tokens: torch.Tensor, C: int, H: int = 7168) -> torch.Tensor:
    """
    expert_tokens: [E, C, H] float tensor (BF16 or float32)
    Returns mask: [E, M_t, K_t] uint8 tensor
    M_t = ceil(C / 32), K_t = ceil(H / 32)
    """
    E = expert_tokens.shape[0]
    M_t = -(-C // 32)   # ceil(C / 32)
    K_t = -(-H // 32)   # ceil(H / 32) = 224

    # Pad C and H to tile boundaries if needed
    C_pad = M_t * 32
    H_pad = K_t * 32
    if expert_tokens.shape[1] < C_pad or expert_tokens.shape[2] < H_pad:
        padded = torch.zeros(E, C_pad, H_pad, dtype=expert_tokens.dtype)
        padded[:, :C, :H] = expert_tokens
        expert_tokens = padded

    # Reshape to tile blocks and check for any non-zero element per tile
    # [E, M_t, 32, K_t, 32] → any over last two dims
    blocked = expert_tokens.view(E, M_t, 32, K_t, 32)
    mask = (blocked.abs().amax(dim=(2, 4)) > 0).to(torch.uint8)  # [E, M_t, K_t]
    return mask

# Transfer mask to device
mask_ttnn = ttnn.from_torch(
    mask_torch,           # [E, M_t, K_t] uint8
    dtype=ttnn.uint8,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

> **Note:** In MoE routing, a capacity slot is either filled (one real token) or empty (all zeros). The mask can therefore be constructed from the routing indices alone — without inspecting the actual activation values — by checking whether each `(expert_id, capacity_slot)` pair received a token. This is cheaper than scanning the full activation tensor.

---

## 3. The Tile-Skip Mechanism

Before loading each tile pair, the kernel executes the following logic (conceptually):

```
for m in range(M_t):
    for n in range(N_t):
        accumulator[m, n] = 0
        for k in range(K_t):
            if mask[m, k] == 0:
                continue           # skip: no FMA, no tile load
            load A[m, k]           # BF16 tile, 2048 bytes
            load W[k, n]           # BFP8 tile, 1088 bytes
            accumulator[m, n] += A[m, k] @ W[k, n]   # 32×32×32 = 32768 FMAs
        store O[m, n]
```

This is a simplified view. The actual kernel uses double-buffered DMA prefetch: the mask check happens ahead of the DMA issue, so skipped tiles never occupy DMA bandwidth. Weight tiles $W[k, n]$ are only requested from DRAM when the corresponding activation tile $A[m, k]$ is active.

The savings are multiplicative: skipping tile row $m$ eliminates $K_t \times N_t$ weight tile reads and $K_t \times N_t$ FMA blocks for that row (one per K-step per output-column tile). For a single inactive tile row, the savings are $K_t \times N_t \times 1088$ bytes of DRAM reads avoided and $K_t \times N_t \times 32768$ FMAs avoided, where $K_t = H/32 = 7168/32 = 224$.

---

## 4. The Static-Shape Constraint

### 4.1 What Must Be Static

The mask **shape** $[C/32, H/32]$ must be fixed across all forward passes that share a program cache entry. The mask **values** (which bits are 0 or 1) may change freely on every forward pass without triggering recompilation.

### 4.2 Why the Shape Must Be Static

TTNN's program caching compiles a kernel program once and reuses it for all subsequent calls with the same tensor shapes and config. The tile-skip control flow — the loop bounds over $M_t$ and $K_t$, the mask array size, the DMA descriptor table — is baked into the compiled program at the tile-count level.

If the mask shape changed (because $C$ changed), the compiled loop bounds would be wrong for the new shape, and the program would need to be recompiled. TTNN enforces this by making mask shape part of the program cache key, alongside activation shape and weight shape.

In practice, this means:

- The padded capacity $C$ must be fixed for all forward passes using a given compiled program.
- The actual number of active tokens may vary (routing assigns different numbers of tokens to different experts on each step), and this is fine: only the mask values change, not the mask shape.
- If you need to support multiple values of $C$ (e.g., for different batch sizes), maintain separate compiled programs per $C$ value.

### 4.3 Implication for MoE Deployment

In MoE serving with expert parallelism (T3K, 8 devices, $E_d = 32$ local experts per device), the padded capacity $C$ is computed from the maximum expected batch:

$$C = \left\lceil \frac{k \times B_{\max} \times S_{\max}}{E} \right\rceil$$

This value is fixed at deployment time. For a decode server handling single-token requests ($S=1$, $B_{\max} = 32$):

$$C = \left\lceil \frac{8 \times 32 \times 1}{256} \right\rceil = 1$$

$C = 1$ is padded to 32 (one tile) in tile layout. The mask shape is $[1, 224]$ for the gate/up projections (one tile row, 224 K-tile columns). This shape is fixed; the single mask bit per K-tile column varies depending on which expert received a token at each step.

---

## 5. BF16 Activations and BFP8 Weights in MoE

In the canonical MoE configuration:

- **Activations:** BF16 — each active tile is 2048 bytes ($32 \times 32 \times 2$).
- **Weights:** BFP8 — each weight tile is 1088 bytes ($32 \times 32 \times 1$ mantissa bytes + 64 bytes shared exponent per 16 elements, giving $1024 + 64 = 1088$ bytes).

`sparse_matmul` exploits sparsity in the **activation** tensor (the $A$ matrix), not the weight tensor. Weight tiles are always loaded in full for any active K step. This is appropriate for MoE: the weight tensor $W$ contains the expert FFN parameters, which are dense (no structural zeros); only the activation tensor, which holds the dispatched token buffer, has structural zeros from unoccupied capacity slots.

This means:

- BFP8 weight compression reduces weight tile size from 2048 to 1088 bytes — a 47% reduction in weight read bandwidth. The remaining fraction of bytes read relative to BF16 is $1088/2048 \approx 0.531$.
- Sparsity masking eliminates weight tile reads entirely for inactive activation tile rows.
- The two benefits are multiplicative: at $\rho = 0.03$ (3% of tile rows active), the total weight bytes read from DRAM are reduced by a factor of $0.531 \times 0.03 \approx 0.016$ relative to a dense BF16 baseline.

---

## 6. FLOP Cost Model

For a dense matmul of shape $[C, H] \times [H, D]$, the FLOP count is:

$$\text{FLOPs}_{\text{dense}} = 2 \times C \times H \times D \quad \text{[D UNVERIFIED — verify against Qwen3 Technical Report]}$$

For `sparse_matmul` with sparsity ratio $\rho$ (fraction of activation tiles that are active):

$$\text{FLOPs}_{\text{sparse}} = 2 \times \rho \times C \times H \times D \quad \text{[D UNVERIFIED]}$$

The memory reads scale similarly:

- Activation reads: $\rho \times C \times H \times 2$ bytes (only active BF16 tiles are loaded).
- Weight reads: $\rho \times C \times D \times 1.063$ bytes (BFP8 weight tiles only for active K steps, scaled by the BFP8 bytes-per-element factor). [D UNVERIFIED]
- Output writes: $C \times D \times 2$ bytes — the output buffer is always fully written (inactive rows contribute zero accumulation, but the store still occurs to maintain a contiguous output tensor for subsequent ops).

> **Note on output writes:** The output tile for an inactive row accumulates only zeros and writes a zero tile to DRAM. A future kernel variant could skip inactive output writes, but the current implementation always writes the full $[M_t, N_t]$ output. At high sparsity, this output-write cost is non-negligible relative to the reduced compute and activation-read costs.

---

## 7. Relationship Between Routing and Sparsity

In expert-parallel MoE on T3K ($N=8$ devices, $E=256$ total experts), each device holds $E_d = 32$ local experts. After the all-to-all dispatch (see Chapter 2, `collective_communication_background.md`), each device holds a token buffer of shape $[E_d, C, H] = [32, C, 7168]$ for its local experts.

An expert slot is empty (all-zero) if the routing assigned zero tokens to that expert in this forward pass. The fraction of empty expert slots determines the sparsity ratio:

$$\rho = \frac{\text{number of active expert slots on this device}}{E_d}$$

An important point: $\rho$ depends only on the number of active expert slots relative to $E_d$ — $M_t$ cancels from numerator and denominator. A detailed regime analysis, including worked examples for $B=1$, $B=8$, and $B=32$ decode and the prefill regime, is in `when_sparse_matmul_wins.md` §2.

---

## Summary

| Aspect | Detail |
|--------|--------|
| **Sparse dimension** | Activation tiles ($A$ matrix); weight tiles always loaded for active K steps |
| **Mask shape** | $[M_t, K_t] = [\lceil C/32 \rceil, 224]$ for gate/up projection |
| **Mask granularity** | Tile-level (32×32); one bit per tile, not per element |
| **Static constraint** | Mask shape fixed per compiled program; mask values may change freely |
| **FLOP reduction** | $2\rho CHD$ vs. $2CHD$ for dense case [D UNVERIFIED] |
| **DRAM reduction** | Activation and weight reads scale with $\rho$; output writes are always full |
| **Dtype** | BF16 activations, BFP8 weights; both unchanged from batched matmul |

---

## References

- Chapter 2, `matmul_fundamentals_in_ttnn.md` — Tile layout, BF16/BFP8 tile sizes, double-buffering, program caching.
- Chapter 3, `formulating_batched_matmul.md` — Gather-pad pattern, expert capacity $C$, padding to tile boundary.
- Chapter 3, `performance_profile_batched.md` §2.2 — Tile-level FLOP efficiency at decode; motivation for tile skipping.
- Chapter 4, `when_sparse_matmul_wins.md` — Quantitative crossover analysis.
- Chapter 4, `program_configs_sparse.md` — SparsityConfig and program config selection for sparse decode.
