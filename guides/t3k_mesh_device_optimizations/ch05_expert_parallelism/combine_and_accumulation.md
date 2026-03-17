# Combine and Accumulation

> **Quick Reference — TTNN API Symbols Introduced**
>
> | Symbol | Description |
> |---|---|
> | `ttnn.all_to_all` (combine direction) | Routes expert outputs back to originating devices; same API as dispatch, opposite data flow direction; see `ch02_ttnn_mesh_api/collective_primitives.md` |

After each device computes its local expert outputs, the results must flow back to the tokens' originating devices, where they are combined using the router's weighting scores. This file covers the structure of the reverse all-to-all (A2A), the weighted accumulation formula for $k = 8$ expert outputs per token, in-place accumulation strategies using L1 and DRAM buffers, numerical precision considerations, and overlap opportunities between the combine communication and subsequent computation.

All values use Qwen3.5-35B constants: $E = 256$, $k = 8$, $N = 8$, $E_d = 32$, $H = 7168$.

---

## 1. Structure of the Reverse All-to-All

### Expert Output After Local Compute

After dispatch, device $d$ holds a receive buffer of shape $[C \times E_d, H] = [C \times 32, H]$ containing all tokens dispatched to it. Device $d$ runs its 32 local expert FFNs (feed-forward networks) on the received token embeddings and produces an output buffer of the same shape $[C \times 32, H]$.

Each row of the output corresponds to one (expert, capacity-slot) pair. Rows corresponding to padded (zero) input tokens produce zero outputs (or outputs that will be masked during accumulation).

### Reverse All-to-All

The reverse all-to-all routes each expert output back to the token's originating device. This is the combine phase. Its structure is symmetric to the dispatch all-to-all:

- **Dispatch:** originating device $\to$ device holding expert.
- **Combine:** device holding expert $\to$ originating device.

```python
# Reverse all-to-all: route expert outputs back to originating devices
# expert_output: [N, C * E_d, H] — packed expert outputs on this device
# combine_recv: [N, C * E_d, H] — will receive outputs from all other devices

combine_recv = ttnn.all_to_all(
    input_tensor=expert_output,
    mesh_device=mesh_device,
    topology=ttnn.Topology.Linear,   # T3K 1×8 linear mesh; see ch01_t3k_topology
    cluster_axis=1,
    num_links=num_links,
    memory_config=ttnn.L1_MEMORY_CONFIG,  # decode: L1; prefill: DRAM_MEMORY_CONFIG
)
# combine_recv[d] now contains the expert outputs produced by device d
# for tokens that originated on this device.
```

### Volume Symmetry

The combine all-to-all transfers the same data volume as the dispatch all-to-all, since expert outputs have the same shape $[C \times E_d, H]$ as the dispatch send buffer. Each device sends and receives the same number of bytes in both phases.

$$V_{\text{combine}} = V_{\text{dispatch}}$$

The formula and per-batch worked examples (6.4 MB at $B=32$, 3.2 MB at $B=1$) are derived in `token_routing_and_dispatch.md` Section 4.

### Output Buffer Before Accumulation

After the combine all-to-all completes, the originating device holds expert outputs organized as a $[B, k, H]$ logical buffer: $B$ tokens, $k = 8$ expert outputs per token, each of dimension $H = 7168$. Before accumulation, this is a set of 8 candidate output vectors per token, each carrying the result from a different expert.

---

## 2. Weighted Combination

### Routing Score Normalization

Qwen3.5-35B uses sigmoid-based routing scores. The raw scores $s_1, \ldots, s_k$ (from the router's top-$k$ output, saved during dispatch) are normalized to sum to 1.0 before accumulation:

$$w_{i}^{\text{norm}} = \frac{s_i}{\sum_{j=1}^{k} s_j}$$

For softmax-based routing (used in other MoE architectures), the scores are already approximately normalized but the normalization step is still applied for numerical consistency.

### Accumulation Formula

The combined output for token $b$ is:

$$\text{output}[b] = \sum_{i=1}^{k} w_i^{\text{norm}} \times \text{expert\_output}[b, i]$$

where $\text{expert\_output}[b, i] \in \mathbb{R}^H$ is the output of the $i$-th selected expert for token $b$.

Expanding for $k = 8$:

$$\text{output}[b] = w_1^{\text{norm}} \cdot e_1 + w_2^{\text{norm}} \cdot e_2 + \cdots + w_8^{\text{norm}} \cdot e_8$$

where each $e_i = \text{expert\_output}[b, i] \in \mathbb{R}^{7168}$.

### Accumulation Loop

```python
def weighted_combine(
    combine_recv: torch.Tensor,     # [B, k, H] — expert outputs after reverse A2A
    routing_scores: torch.Tensor,   # [B, k] — raw scores from router (saved at dispatch)
    output: torch.Tensor,           # [B, H] — output accumulation buffer (zeroed)
) -> torch.Tensor:
    """
    Weighted accumulation of k=8 expert outputs per token.
    Uses normalized routing scores (sigmoid-style as in Qwen3.5-35B).
    """
    B, k, H = combine_recv.shape
    # Normalize scores: w[b, i] = s[b, i] / sum_j(s[b, j])
    score_sum = routing_scores.sum(dim=-1, keepdim=True)  # [B, 1]
    w_norm = routing_scores / score_sum                    # [B, k]

    # Accumulate: output[b] = sum_i w_norm[b, i] * combine_recv[b, i, :]
    output.zero_()
    for i in range(k):
        # Broadcast multiply: [B, 1] * [B, H] -> [B, H]
        output += w_norm[:, i:i+1] * combine_recv[:, i, :]

    return output  # [B, H]
```

In TTNN, this loop is expressed as a fused batched multiply-accumulate rather than a Python loop. The loop is shown here for conceptual clarity.

---

## 3. In-Place Accumulation Strategies

### Option A: L1 Accumulation Buffer (Decode)

During decode, the output tensor $[B, H]$ is small enough to reside in L1 SRAM for the duration of accumulation:

- $B = 1$: $[1, 7168] \times 2$ bytes = $14{,}336$ bytes = **14 KB** — trivially fits in L1 (1.5 MB per core).
- $B = 32$: $[32, 7168] \times 2$ bytes = $458{,}752$ bytes = **448 KB** — fits per-core with HEIGHT_SHARDED layout across 80 Tensix cores: $448 \text{ KB} / 80 = 5.6$ KB per core, well within the 1.5 MB L1 per core.

**Procedure:**
1. Allocate the output buffer $[B, H]$ in L1 at the start of the MoE layer.
2. Initialize to zero.
3. For each of the $k = 8$ expert contributions, multiply by the normalized weight and accumulate in-place in L1.
4. After all 8 terms are accumulated, the output tensor is already in L1, ready for the next layer without a DRAM round-trip.

This eliminates $k = 8$ DRAM write-read cycles per token embedding, saving approximately:

$$\Delta t_{\text{DRAM}} \approx \frac{k \times B \times H \times 2}{\text{DRAM BW}} = \frac{8 \times 32 \times 7168 \times 2}{300 \text{ GB/s}} \approx 12.2 \mu\text{s}$$

(Wormhole B0 DRAM bandwidth ~300 GB/s [UNVERIFIED]; see `ch04_memory_config/wormhole_memory_hierarchy.md`.)

See `ch04_memory_config/decode_memory_strategy.md` for the full L1 budget analysis.

### Option B: DRAM Accumulation Buffer (Prefill or Large Batch)

When $B$ is large or memory pressure from other tensors leaves insufficient L1 headroom, accumulate directly to a DRAM buffer:

- Scatter-add: write each weighted expert output to the correct position in a $[B, H]$ DRAM tensor.
- After all $k$ terms are accumulated, the result is already in DRAM for the next layer (which typically uses DRAM-interleaved tensors in prefill).

**When to use DRAM accumulation:**
- Prefill phase (any $B$, $S \gg 1$): per `ch04_memory_config/prefill_memory_strategy.md`, all-to-all buffers and activations are already in DRAM; keeping the output in DRAM is consistent.
- Decode with $B > 32$: output tensor exceeds ~448 KB; L1 budget becomes tight.

```python
# DRAM accumulation (conceptual; TTNN handles memory placement via MemoryConfig)
output_dram = ttnn.zeros(
    [B, H],
    dtype=ttnn.bfloat16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    device=device,
)
for i in range(k):
    weight_i = w_norm[:, i:i+1]   # [B, 1]
    expert_out_i = combine_recv[:, i, :]  # [B, H]
    output_dram = ttnn.add(output_dram,
                           ttnn.multiply(weight_i, expert_out_i,
                                         memory_config=ttnn.DRAM_MEMORY_CONFIG),
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

---

## 4. Handling Top-k > 1 Multi-Expert Summation

### Ordering and Metadata

The combine all-to-all must preserve the correspondence between each expert output and its routing weight. The dispatch send buffer construction (see `token_routing_and_dispatch.md` Section 2) includes a small $[B, k]$ index tensor recording which expert index was sent where. This metadata is used at combine time to match received outputs to routing scores.

**Metadata structure:**

```python
# dispatch_meta[b, i] = (expert_idx, device_id, capacity_slot) for token b's i-th expert
# Saved at dispatch time; used at combine to assemble [B, k, H] from received outputs
dispatch_meta: list  # [B][k] of (expert_idx, device_id, slot)
```

After the combine all-to-all, the originating device uses `dispatch_meta` to reorder the received outputs into the $[B, k, H]$ buffer before calling `weighted_combine`.

### Numerical Precision

The accumulation of $k = 8$ BF16 terms introduces cumulative rounding error. A loose upper bound on the relative error per element is:

$$\epsilon_{\text{accum}} \lesssim k \times \epsilon_{\text{BF16}} = 8 \times 0.004 = 3.2\%$$

where $\epsilon_{\text{BF16}} \approx 0.004$ is the BF16 machine epsilon (unit roundoff). In practice, errors are far smaller than this bound because:

- Individual terms have varied signs and magnitudes; rounding errors partially cancel.
- The normalized weights $w_i^{\text{norm}}$ sum to 1.0, bounding the dynamic range.
- Empirical Pearson Correlation Coefficient (PCC) loss from BF16 accumulation of $k = 8$ expert outputs is negligible in production MoE deployments.

> **Tip:** If numerical fidelity is a concern, accumulate in FP32 and cast the result to BF16 before writing to the output tensor. The extra memory cost is $[B, H] \times 4$ bytes instead of $\times 2$ bytes — at $B = 32$: 896 KB vs. 448 KB. FP32 accumulation eliminates the cumulative BF16 rounding error entirely.

---

## 5. Overlap Opportunities

### Micro-Batch Pipelining

The combine all-to-all and the subsequent dispatch for the next micro-batch are independent operations if the output buffers are double-buffered. The pipeline is:

```
Step 1 (micro-batch m₀):   [dispatch A2A] → [expert compute] → [combine A2A] → [accumulate]
Step 2 (micro-batch m₁):   [dispatch A2A] ← can overlap with Step 1's combine A2A
```

With double-buffered send/receive buffers, device $d$ can begin packing the Step 2 dispatch buffer while the Step 1 combine all-to-all is still in flight. This hides combine communication latency under dispatch preparation for the next step.

**Condition:** The combine A2A and the Step 2 dispatch preparation must not share any buffer regions. With distinct L1 buffers allocated for odd and even micro-batches, this condition is satisfied.

### Overlapping Combine with Next-Layer Computation

Within a single micro-batch, the residual stream addition (`output + residual`) and the next layer's attention Q/K/V projection depend on the combine output. However, computations that do not depend on the combine result can proceed in parallel:

- **Independent:** Layer normalization parameters can be loaded; attention weight tiles can be prefetched into L1.
- **Dependent:** The residual add, the next attention matmul.

In TTNN, async dispatch for `ttnn.all_to_all` allows the host to enqueue subsequent operations while the collective is in flight. Operations on tensors that do not depend on `combine_recv` proceed immediately.

### Prefill-Phase Overlap

During prefill, the combine all-to-all is the dominant latency contributor (hundreds of milliseconds at $B=32, S=2048$). Overlap opportunities:

- Overlap combine A2A with the next transformer layer's attention Q/K/V projection: while combine completes for MoE layer $l$, compute Q/K/V projections for layer $l+1$ on the current hidden states.
- Requires: the MoE combine output is NOT needed for Q/K/V projections of the same layer (it feeds the residual add before layer $l+1$'s attention). This dependency is satisfied in standard transformer architecture.

The potential latency savings are approximately equal to the combine A2A duration, which at $B=4, S=2048$ is:

$$t_{\text{combine}} \approx \frac{117.4 \text{ MB}}{12.5 \text{ GB/s}} \approx 9.4 \text{ ms}$$

(using the buffer size from Chapter 3; `ch03_all_to_all_num_links/all_to_all_in_moe.md`). Hiding this saves $\sim9.4$ ms per MoE layer per prefill step.

> **Tip:** Enable async dispatch for `ttnn.all_to_all` in your TTNN program configuration to allow the runtime to pipeline combine and the next-layer prefetch automatically. Verify with the TTNN profiler that the combine and prefetch are indeed overlapping and not serialized by unexpected data dependencies.

### In-Place Residual Add

Once the weighted accumulation completes, the MoE output is added to the residual stream in-place:

```python
# residual: [B, H] — input to the MoE layer (saved before expert dispatch)
# output:   [B, H] — accumulated MoE output
# result is written directly to the residual tensor slot
residual = ttnn.add(
    residual,
    output,
    memory_config=ttnn.L1_MEMORY_CONFIG,  # decode
)
# residual now holds the post-MoE hidden states, ready for the next layer norm
```

Writing directly to the residual tensor's memory slot avoids a separate allocation, keeping the working set compact in L1 during decode.

---

## References

- `ch01_t3k_topology/t3k_physical_layout.md` — T3K topology, `ttnn.Topology.Linear`, `cluster_axis=1`
- `ch02_ttnn_mesh_api/collective_primitives.md` — `ttnn.all_to_all` API (dispatch and combine direction)
- `ch03_all_to_all_num_links/all_to_all_in_moe.md` — Combine volume derivation; buffer sizes at $B=4, S=2048$ (117.4 MB)
- `ch04_memory_config/decode_memory_strategy.md` — L1 budget for output accumulation buffer at $B=32$
- `ch04_memory_config/prefill_memory_strategy.md` — DRAM placement for combine buffers at large $S$
- `ch04_memory_config/wormhole_memory_hierarchy.md` — Wormhole B0 memory capacities; DRAM bandwidth ~300 GB/s [UNVERIFIED]
- `token_routing_and_dispatch.md` (this chapter) — Dispatch metadata structure; routing scores saved for combine
- `expert_placement_strategies.md` (this chapter) — Expert replication and how replica outputs are combined
- `expert_parallelism_strategies/ch04_expert_device_assignment/expert_replication.md` — Routing metadata for replicated expert combine
- [Fedus2022] Fedus, W., Zoph, B., Shazeer, N., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR, 2022.
- [Qwen3] Qwen Team, Alibaba Group, "Qwen3 Technical Report", 2025.

---

**Next:** [Chapter 6 — Profiling](../ch06_profiling/index.md)
