# Sharding Strategies for T3K MoE

## Section 1: Activation Tensor Distribution

### Origin and Dispatch

In a pure EP configuration, device 0 holds the input token batch $[B, H]$ and the dispatch all-to-all sends activations to expert-holding devices. In a hybrid tensor-parallel + EP configuration, all devices process the same sequence in parallel and each holds a shard of the activation before the MoE layer; the all-to-all then re-distributes by expert assignment rather than by sequence position.

For Qwen3.5-35B at $B = 32$, $H = 7168$ (BF16):

$$[B, H] = [32, 7168] \Rightarrow 32 \times 7168 \times 2 = 458{,}752 \text{ bytes} \approx 0.46 \text{ MB}$$

This comfortably fits in L1 (1.5 MB per core on Wormhole B0), so activation tensors at decode should use `L1_MEMORY_CONFIG`.

### After Dispatch

After `ttnn.all_to_all` dispatch, each device holds a receive buffer $[T_d, H]$ containing the tokens assigned to its experts. Under uniform routing:

$$T_d \approx B \cdot \frac{k \cdot E_d}{E} = 32 \cdot \frac{8 \times 32}{256} = 32 \text{ tokens}$$

The expected per-device token count equals $B$ in the uniform-routing approximation, but the distribution is binomial and tail cases (devices receiving many more or fewer than $B$ tokens) are possible. The expert capacity $C$ sets the hard upper bound:

$$C = \left\lceil \frac{k \cdot B \cdot \text{CF}}{E} \right\rceil = \left\lceil \frac{8 \times 32 \times 1.25}{256} \right\rceil = \left\lceil \frac{320}{256} \right\rceil = \left\lceil 1.25 \right\rceil = 2$$

### Reshape for Expert-Local Computation

The received $[T_d, H]$ buffer is reshaped to $[E_d, C, H]$ to expose the expert-batch structure for the batched matmul:

$$[T_d, H] \rightarrow [E_d, C, H] = [32, C, H]$$

At $B = 32$: $[32, 7168] \rightarrow [32, 2, 7168]$

This reshape is zero-copy if the receive buffer is laid out contiguously with the expert axis outermost. Token overflow beyond $C$ is dropped (overflow tokens are clipped at dispatch, which is why $\text{CF} > 1.0$ is important for maintaining accuracy).

```python
import ttnn

# dispatched_tokens: [T_d, H] received from ttnn.all_to_all dispatch
E_d = 32
C = 2    # expert capacity at B=32 with CF=1.25
H = 7168

# Reshape for expert-local batched matmul
local_expert_batch = ttnn.reshape(dispatched_tokens, [E_d, C, H])
# Shape: [32, 2, 7168]; memory: L1 at decode

# At B=1: C=1, shape is [32, 1, 7168]
# At B=256: C=10, shape is [32, 10, 7168]
```

### Memory Placement by Regime

| Regime | Activation memory | Reason |
|---|---|---|
| Decode ($B \leq 32$) | `L1_MEMORY_CONFIG` | $[32, 2, 7168]$ ≈ 0.9 MB; fits in L1 |
| Prefill ($B = 256$) | `DRAM_MEMORY_CONFIG` | $[32, 10, 7168]$ ≈ 4.6 MB; exceeds L1 |
| Prefill (long seq) | `DRAM_MEMORY_CONFIG` | Mandatory; L1 overflow otherwise |

---

## Section 2: Expert Weight Tensor Placement

### Per-Device Expert Weight Shape

Each device holds $E_d = 32$ expert weight tensors. For a standard two-weight expert FFN (gate + up projections, or a single fused weight), the full per-device weight tensor is:

$$[E_d, H, D] = [32, H, D]$$

where $D$ is the FFN intermediate dimension [UNVERIFIED]. These weights live in DRAM on each device; they are too large to stage entirely in L1.

The DRAM placement is intentional: during the expert FFN computation, weight tiles are streamed from DRAM to L1 core-by-core via the Wormhole B0 NoC. The arithmetic intensity of the matmul must be high enough to hide this streaming latency — which is a core challenge at small $C$.

### DistributedTensorConfig for Weight Sharding

When creating the weight tensor on the `MeshDevice`, each device's shard is independent:

```python
import ttnn

# Create per-device expert weights
# Each device gets its own [32, H, D] weight tensor in DRAM
# No cross-device coordination needed during forward pass

E_d = 32
H = 7168
# D is the FFN intermediate dimension [UNVERIFIED]

# Weight tensor on each device: [E_d, H, D]
# Using DRAM_MEMORY_CONFIG because weights don't fit in L1
expert_weights_device = ttnn.as_tensor(
    expert_weights_host,         # [E_d, H, D] numpy/torch tensor for this device
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# For sparse_matmul, the weight tensor is accessed via the sparsity tensor's
# non-zero rows; DRAM placement is the same
```

### Access Pattern During Forward Pass

The expert FFN computation streams weight tiles from DRAM to L1 on a per-core basis. The memory access pattern for a single expert's matmul is:

1. Load activation tile $[C, \text{tile\_size}]$ from L1 receive buffer into compute core
2. Load corresponding weight tile $[\text{tile\_size}, D_{\text{tile}}]$ from DRAM
3. Accumulate into output tile in L1
4. Repeat across the $K_t = \lceil H / 32 \rceil = \lceil 7168 / 32 \rceil = 224$ tile columns

No cross-device weight communication occurs. Each device's DRAM weight shard is entirely self-contained.

---

## Section 3: Sparsity Tensor for Multi-Chip `sparse_matmul`

After dispatch, each device independently constructs its sparsity tensor from the received token counts. No cross-device coordination is needed; the dispatch buffer itself encodes which experts received tokens.

### Sparsity Tensor Shape

Following Chapter 5 ([sparsity_tensor_placement.md](../ch05_sparsity_tensor_construction/sparsity_tensor_placement.md)), the sparsity tensor shape for a device's local expert set is:

$$[E_d \times M_t, K_t] \quad \text{where} \quad M_t = \left\lceil \frac{C}{32} \right\rceil, \quad K_t = \left\lceil \frac{H}{32} \right\rceil$$

For Qwen3.5-35B decode at $B = 32$ ($C = 2$, $H = 7168$):

$$M_t = \left\lceil \frac{2}{32} \right\rceil = 1, \quad K_t = \left\lceil \frac{7168}{32} \right\rceil = 224$$

$$\text{Sparsity tensor shape} = [32 \times 1, 224] = [32, 224] \text{ uint8}$$

Size: $32 \times 224 = 7{,}168$ bytes — easily placed in L1.

### Per-Device Construction

```python
import ttnn
import torch

def build_local_sparsity_tensor(
    token_counts_per_expert: torch.Tensor,  # [E_d] int, tokens received per local expert
    E_d: int,
    C: int,
    H: int,
    device: ttnn.Device,
) -> ttnn.Tensor:
    """
    Construct the sparsity tensor for this device's E_d local experts.

    token_counts_per_expert[i] is the number of tokens received by local expert i.
    An expert is "active" (non-zero row in sparsity tensor) if it received >= 1 token.

    Returns uint8 tensor of shape [E_d * M_t, K_t] placed in L1.
    """
    tile_size = 32
    M_t = (C + tile_size - 1) // tile_size   # ceil(C / 32)
    K_t = (H + tile_size - 1) // tile_size   # ceil(H / 32)

    # One row per (expert, tile_row) pair; mark row as active if expert has tokens
    sparsity = torch.zeros(E_d * M_t, K_t, dtype=torch.uint8)
    for expert_idx in range(E_d):
        if token_counts_per_expert[expert_idx] > 0:
            # All K_t tile columns are active for this expert's M_t tile rows
            row_start = expert_idx * M_t
            row_end   = row_start + M_t
            sparsity[row_start:row_end, :] = 1

    # Place in L1; size is E_d * M_t * K_t bytes (uint8)
    return ttnn.as_tensor(
        sparsity,
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,  # 7,168 bytes at B=32 decode
    )
```

> **Tip:** `token_counts_per_expert` is derived from the dispatch routing table, which the router already computed before calling `ttnn.all_to_all`. No additional communication is needed; the routing decisions are already known on each device.

### Why No Cross-Device Coordination Is Needed

The dispatch phase already communicates the routing assignments globally. After dispatch, each device knows exactly which tokens it received for each of its local experts. The sparsity tensor is a purely local summary of that information — it does not depend on what other devices received. This is the key advantage of the EP design: the sparsity tensor is embarrassingly parallel across devices.

---

## Section 4: Replicated vs. Sharded Weight Tensors

Not all tensors in the MoE layer are sharded. The table below summarizes the placement strategy for each tensor type:

| Tensor | Shape | Placement | Reason |
|---|---|---|---|
| Expert FFN weights | $[E_d, H, D]$ [UNVERIFIED] | **Sharded** (DRAM, per device) | Different experts on each device; no replication needed |
| Routing weights $W_r$ | $[H, E] = [7168, 256]$ | **Replicated** (all devices) | Each device needs full routing for its token batch; 3.67 MB BF16 is small |
| Input activation | $[B, H]$ | Replicated or device-local | Depends on whether tensor-parallel or pure EP is used |
| Residual stream | $[B, H]$ | Replicated (EP) or sharded (TP) | Replicated under pure EP; sharded if tensor parallelism is layered in |
| Sparsity tensor | $[E_d M_t, K_t]$ | **Local** (L1, per device) | Constructed per-device; no replication or sharding needed |
| Router output probs | $[B, k]$ | Replicated | Small; computed before dispatch |

### Routing Weight Replication

The routing weight matrix $W_r \in \mathbb{R}^{H \times E}$ has size:

$$7168 \times 256 \times 2 \text{ bytes} = 3{,}670{,}016 \text{ bytes} \approx 3.67 \text{ MB}$$

This is small enough to replicate on all 8 devices. Each device needs the full routing matrix to compute the softmax scores for its local token batch. Under pure EP (no tensor parallelism within the attention layer), replication is the correct choice: it avoids an additional all-to-all for the routing step.

> **Warning:** If tensor parallelism is applied to the attention layer, the residual stream may be sharded across devices before reaching the MoE layer. In that case, an additional all-gather is needed to reconstruct $[B, H]$ before routing — or the router can be designed to work on sharded activations. This interaction is outside the scope of this chapter; consult the hybrid EP+TP design documentation.

---

## References

- Chapter 2: [ttnn_programming_model.md](../ch02_ttnn_wormhole_primer/ttnn_programming_model.md) — `DRAM_MEMORY_CONFIG`, `L1_MEMORY_CONFIG`
- Chapter 5: [sparsity_tensor_placement.md](../ch05_sparsity_tensor_construction/sparsity_tensor_placement.md) — sparsity tensor construction details
- Chapter 6: [decision_guide.md](../ch06_comparative_analysis/decision_guide.md) — hybrid strategy selection
- T3K guide: [collective_primitives.md](../../t3k_guide/ch02_ttnn_mesh_api/collective_primitives.md) — `ttnn.all_to_all`

---

**Next:** [program_configs_t3k.md](./program_configs_t3k.md)
