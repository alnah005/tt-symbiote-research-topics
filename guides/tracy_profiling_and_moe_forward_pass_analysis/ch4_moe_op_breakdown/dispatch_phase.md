# Dispatch Phase

The dispatch phase transforms a token tensor `[seq_len, d_model]` into a reordered tensor
`[total_active_tokens, d_model]` in which tokens destined for the same expert are contiguous.
On T3K, it also includes the all-to-all CCL op that ships tokens across chips to the chip
holding each expert's weight shard.

---

## Ops in Order

### 1. `ttnn.linear` — Router Projection

```
Input:  [seq_len, d_model]         e.g., [1024, 7168]
Weight: [d_model, num_experts]     e.g., [7168, 128]
Output: [seq_len, num_experts]     e.g., [1024, 128]
Tracy zone: MoE/dispatch/router_linear
CSV op name: matmul
```

The router is a single linear layer with no bias in most implementations (Qwen MoE, DeepSeek-V3).
It maps each token's hidden representation to a logit score for each expert. The TTNN op is a
standard `matmul`; it appears as `matmul` in the device profiler CSV.

### 2. `ttnn.softmax`

```
Input:  [seq_len, num_experts]
Output: [seq_len, num_experts]     probabilities sum to 1 per token
Tracy zone: MoE/dispatch/router_softmax
CSV op name: softmax
```

Converts logits to routing probabilities. Applied along the `num_experts` dimension.
Memory-bandwidth bound at typical seq_len values; not a significant latency contributor.

### 3. `ttnn.topk`

```
Input:  [seq_len, num_experts]
Output: values [seq_len, top_k],  indices [seq_len, top_k]
Tracy zone: MoE/dispatch/topk
CSV op name: topk
```

Selects the top-K expert indices and their corresponding routing scores for each token.
For Qwen 235B: top_k=8 out of 128 experts. This op is compute-light but can be latency-visible
because Wormhole does not have a dedicated sort unit; topk is implemented in Tensix with a partial
sort kernel over the 128-wide expert dimension.

### 4. Index Tensor Construction

This step builds the token-to-expert assignment map: for each of the `num_experts × expert_capacity`
slots, determine which token index fills it, and for each token, record its position in the
reordered output.

This is the most variable step in the dispatch phase. If implemented with Python loops over the
CPU, it contributes a host-side gap visible in Tracy but absent from the device profiler CSV.
If tensor-ized (e.g., using `ttnn.argsort` and scatter index ops), it contributes a device kernel
entry. Check Chapter 5 Pattern A for how to diagnose this in a trace.

```
Tracy zone: MoE/dispatch/index_construction  (if annotated)
CSV op name: none if CPU-only; argsort or scatter if tensor-ized
```

### 5. `ttnn.gather` — Token Reordering by Expert

```
Input:  [seq_len, d_model],  index tensor [total_active_tokens]
Output: [total_active_tokens, d_model]
Tracy zone: MoE/dispatch/gather
CSV op name: gather
```

Reorders the token tensor so that tokens assigned to the same expert are contiguous. The output
shape `[total_active_tokens, d_model]` depends on routing: `total_active_tokens = seq_len × top_k`
if every token selects exactly top_k experts with no capacity limit, or fewer if capacity capping
is applied. At seq_len=1024, top_k=8: `total_active_tokens` is at most 8192 but typically close
to that value in balanced routing.

---

## T3K: All-to-All CCL Op

On T3K (8-chip mesh with expert parallelism), each chip holds a shard of the expert weight
tensors. After local dispatch, tokens must be redistributed so that each chip receives exactly
the tokens that were routed to its local experts.

```
Op:   all_gather or all_to_all (CCL)
Tracy zone: MoE/dispatch/all_to_all
CSV op name: all_gather
Input per chip:  [total_active_tokens_local, d_model]
Output per chip: [total_active_tokens_for_local_experts, d_model]
```

**Expected latency:** 300–800 µs at seq_len=1024 on T3K, depending on token distribution balance
and ethernet link utilization. The transfer volume is `(seq_len × top_k × d_model × 2 bytes)` for
BF16 divided by `ep_degree`, routed over T3K's inter-chip ethernet links (~100 GB/s bidirectional
per chip).

**How to identify in a Tracy trace:**
- In the Tracy GUI, look for a wide zone labeled `all_gather` or `CCL::AllToAll` between the
  `MoE/dispatch/gather` zone and the first `MoE/expert_matmul` zone.
- In the device profiler CSV, filter for `all_gather` in the op name column. The
  `DEVICE KERNEL DURATION [ns]` for this op is the on-chip portion; true end-to-end latency
  includes ethernet fabric transit time not captured by the cycle counter.
- If the CCL op is not annotated, it appears as a gap between dispatch and expert matmul zones
  in the Tracy timeline. See Chapter 5 Pattern C.

---

## Latency Budget at seq_len=1024

All figures are indicative for Qwen 235B dims (d_model=7168, num_experts=128, top_k=8) on a
single Wormhole B0 at BF16. Re-profile on the target system.

| Op | Expected range (µs) | Primary driver |
|---|---|---|
| `ttnn.linear` (router) | 80–200 | seq_len × d_model × num_experts FLOPs; memory-bound |
| `ttnn.softmax` | 10–30 | Memory bandwidth over [1024, 128] tensor |
| `ttnn.topk` | 20–60 | Partial sort over 128 experts × 1024 tokens |
| Index construction | 5–50+ | 5–15 µs if tensor-ized; up to 50+ µs if CPU-side Python |
| `ttnn.gather` | 30–80 | DRAM bandwidth: `seq_len × top_k × d_model × 2B` bytes |
| **T3K all-to-all CCL** | **300–800** | Ethernet BW, token count, ep_degree |
| **Dispatch phase total (single chip)** | ~150–400 µs | — |
| **Dispatch phase total (T3K)** | ~450–1200 µs | — |

The dominant contributor on T3K is the all-to-all CCL. On a single chip, the gather op is
typically largest because it reads `seq_len × top_k × d_model` elements from DRAM.

---

## Output Shape: Why It Varies

The gather output `[total_active_tokens, d_model]` is not fixed even at fixed seq_len because:

1. **Capacity capping:** many implementations cap each expert at `expert_capacity = capacity_factor
   × seq_len × top_k / num_experts` tokens. Tokens exceeding this cap are dropped or handled by an
   overflow expert. The capacity cap sets a hard upper bound on `total_active_tokens`.

2. **Balanced vs. unbalanced routing:** in practice, routing is not perfectly uniform. Some experts
   receive more tokens and hit the cap; others receive fewer. The number of active slots is fixed
   by the capacity cap, but filled slots vary per run.

3. **Padding:** implementations that use batched matmul typically pad each expert's token slice to
   `expert_capacity` to produce a uniform `[num_experts, expert_capacity, d_model]` tensor
   (described in `expert_matmul_phase.md`).

---

## Next Steps

Continue to [`expert_matmul_phase.md`](./expert_matmul_phase.md) for the gate/up/down projection
matmul structure, expected FLOPs, and the prefill vs. decode latency relationship.
