# Combine Phase

The combine phase converts per-expert output tensors back into a single `[seq_len, d_model]`
tensor in the original token order, weighted by the routing scores produced in the dispatch phase.

---

## Ops in Order

### 1. `ttnn.scatter` or Inverse Gather

```
Input:   [num_experts, expert_capacity, d_model]   per-expert outputs from down proj
         index tensor (inverse of the gather index from dispatch)
Output:  [seq_len, top_k, d_model]                token-ordered, one slot per expert per token
Tracy zone: MoE/combine/scatter
CSV op name: scatter
```

This op is the inverse of `ttnn.gather` from the dispatch phase. It places each expert's output
token back at its original sequence position. The output has a `top_k` dimension because each
token has contributions from `top_k` different experts that need to be merged.

Some TTNN MoE implementations do not use a distinct `scatter` op. Instead they construct the
inverse index at dispatch time and call `ttnn.gather` again with the inverse index. In the device
profiler CSV this will appear as a second `gather` entry rather than `scatter`.

### 2. Element-Wise Multiply by Router Scores

```
Input A: [seq_len, top_k, d_model]   token-ordered expert outputs
Input B: [seq_len, top_k]            routing scores from ttnn.topk (dispatch phase)
Output:  [seq_len, top_k, d_model]   weighted expert outputs
Tracy zone: MoE/combine/score_multiply
CSV op name: mul
```

Applies the softmax router score for each token-expert pair as a scalar weight. The router scores
tensor is broadcast across the `d_model` dimension. This is a memory-bandwidth-bound element-wise
op; latency scales with `seq_len × top_k × d_model × 2B`.

### 3. `ttnn.sum` — Accumulate Top-K Expert Outputs

```
Input:   [seq_len, top_k, d_model]
Output:  [seq_len, d_model]
Reduction: sum over the top_k dimension
Tracy zone: MoE/combine/topk_sum
CSV op name: sum (or reduce_sum)
```

Sums the weighted outputs from all `top_k` experts for each token, producing the final MoE layer
output. This is a reduce along `top_k=8`; it is memory-bandwidth bound and fast relative to the
matmul phase.

---

## T3K: Reduce-Scatter or All-Reduce

On T3K with expert parallelism, each chip has computed the down-projection outputs only for its
local expert shard. Before scatter and weighted-sum, partial outputs from all chips must be
aggregated.

```
Op:   reduce_scatter (preferred) or all_reduce
Tracy zone: MoE/combine/reduce_scatter
CSV op name: all_gather  (TTNN CCL ops may appear as all_gather in the CSV even for
                          reduce-scatter variants; check op shape and direction)
Input per chip:  partial outputs for all tokens from local experts
                 shape: [total_active_tokens, d_model]
Output per chip: full outputs for the token subset assigned to this chip
                 shape (reduce_scatter): [total_active_tokens / ep_degree, d_model]
                 shape (all_reduce):     [total_active_tokens, d_model]
```

**Reduce-scatter** is preferred over all-reduce because it reduces the per-chip output volume by
`ep_degree`, lowering DRAM write pressure in the subsequent scatter op. All-reduce is simpler to
implement but writes the full `[total_active_tokens, d_model]` tensor to every chip.

**Expected latency (T3K, seq_len=1024, d_model=7168, BF16):**

Transfer volume per chip for reduce-scatter:
```
bytes = total_active_tokens × d_model × 2B
      = 8192 × 7168 × 2 ≈ 117 MB
At ~100 GB/s inter-chip BW: ~1.2 ms per transfer
With 8 chips and ring topology: ~4–8 ms for full reduce-scatter
```

In practice, observed reduce-scatter latency at seq_len=1024 on T3K is **2–8 ms**, depending on
ring scheduling and whether the CCL implementation overlaps with local compute.

> **Tip:** If the combine-phase CCL latency exceeds the expert matmul latency in decode, the primary
> optimization lever is reducing `ep_degree` or switching to a smaller collective (e.g., using
> `ttnn.experimental.ccl.reduce_scatter_async` with compute overlap).

---

## Expected Latency for the Combine Phase

Configuration: Qwen 235B, seq_len=1024, top_k=8, d_model=7168, BF16.

| Op | Single chip (µs) | T3K (µs) |
|---|---|---|
| `ttnn.scatter` / inverse gather | 50–150 | 50–150 |
| Element-wise multiply (router scores) | 20–60 | 20–60 |
| `ttnn.sum` over top_k | 15–40 | 15–40 |
| **T3K reduce-scatter CCL** | — | **2000–8000** |
| **Combine phase total** | **~100–250 µs** | **~2100–8300 µs** |

**Relationship to seq_len:** all ops in the combine phase scale linearly with `seq_len`. Doubling
seq_len doubles the scatter index size, the multiply tensor volume, and the CCL transfer volume.
The CCL latency is bandwidth-limited at large seq_len and latency-limited at small seq_len; it
transitions between these regimes around seq_len=64–128 on T3K.

**Relationship to top_k:** the element-wise multiply and sum scale linearly with `top_k`. The
scatter op scales with `seq_len × top_k` (total active token slots). Increasing top_k from 8 to
16 approximately doubles combine phase latency for the non-CCL ops.

---

## Load Imbalance: Unequal Token Counts per Expert

Expert routing is rarely perfectly uniform. With 128 experts and 1024 tokens routing top_k=8,
the expected token count per expert is `1024 × 8 / 128 = 64`. In practice the distribution has
variance; some experts receive 80+ tokens while others receive fewer than 40.

During the scatter phase, this imbalance manifests as idle Tensix cores: the cores handling a
low-token expert finish early and wait for the cores handling the high-token expert. This effect
is captured in the `DEVICE KERNEL DURATION [ns]` column of the CSV as longer-than-expected
scatter times on runs with skewed routing.

**Diagnosing imbalance in the device profiler CSV:**

1. The `scatter` op duration is unusually high relative to the theoretical volume.
2. The ratio of p95 to p50 scatter duration across many iterations is > 1.3.
3. Inspecting the routing logit distribution shows several hot experts.

**Mitigation strategies** (documented in Chapter 7): auxiliary load-balancing loss in training;
expert capacity capping with token dropping; or Z-loss regularization on router logits.

---

## Next Steps

Continue to [`full_op_sequence_reference.md`](./full_op_sequence_reference.md) for the
consolidated op table covering every TTNN op in the MoE forward pass with shapes, durations,
and Tracy zone names.
