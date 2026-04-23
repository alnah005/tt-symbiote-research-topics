# Parallel Prefix Scan Review

This file examines whether parallel prefix scan primitives in tt-metal can be applied to the DeltaNet recurrence, for either the decode path (T=1) or the prefill path (T>1). The conclusion is that parallel prefix scan is not applicable to DeltaNet decode under any configuration and provides no benefit for prefill at practical sequence lengths. The correct implementation for decode is the composed TTNN form (Chapter 2) or the fused kernel (Chapter 4); for prefill, a Python chunk loop calling TTNN matmuls is the correct first implementation.

---

## 1. Source Location

> **Note:** Search `ttnn/cpp/ttnn/operations/` and `tt_metal/impl/` for any of: `prefix_scan`, `parallel_scan`, `associative_scan`, `cumsum`. As of the survey, no dedicated parallel prefix scan op for recurrence has been found in tt-metal outside of the Mamba SSM context. If a cumulative sum (`ttnn.cumsum`) or inclusive scan op exists, document it here and assess whether it can be extended to matrix-valued scan elements; the expected answer is no, because the TTNN cumsum operates on scalar elements, not matrix-valued operands.

---

## 2. Why Parallel Prefix Scan Is Not Applicable for Decode

The DeltaNet decode path processes T=1 tokens per step. At T=1, there is exactly one recurrence step to execute:

```
S_t = g_t * S_{t-1} + k̃_t ⊗ (β_t * (v_t - S_{t-1}^T @ k̃_t))
```

A parallel prefix scan requires at least two elements to associate. With T=1, there is one element: the single state update at step t. There is nothing to parallelize. A parallel prefix scan over a sequence of length 1 is the identity operation — the scan output is the input.

> **Key Finding:** Parallel prefix scan is irrelevant for DeltaNet decode. At T=1, there is exactly one recurrence step per decode call. No scan parallelism exists. The composed TTNN form (Chapter 2) and the fused kernel (Chapter 4) are the complete solution space for decode.

This conclusion is independent of the model configuration (Qwen3.6-35B-A3B vs. Qwen3-9B) and independent of the number of attention heads. It follows from the definition of the decode path as T=1 token generation.

---

## 3. Why Parallel Prefix Scan Is Not the Recommended Approach for Prefill Either

At prefill time (T > 1), the DeltaNet model processes the full prompt using a chunked algorithm that separates intra-chunk and inter-chunk computation. The inter-chunk recurrence is associative and in principle could be parallelized with a scan. However, several considerations make the parallel scan approach impractical or unnecessary for the first implementation.

### 3.1 The Within-Chunk WY Decomposition Is Not Expressible as a Standard Associative Scan

The chunked DeltaNet prefill algorithm processes each chunk of C=64 tokens using a WY-like decomposition that handles intra-chunk token interactions. This within-chunk computation involves triangular matrix products and corrections that are not expressible as a simple associative operator `S_{c+1} = f(S_c, Δ_c)` applied independently to each chunk.

The associativity that does exist is at the inter-chunk level: once the within-chunk update `Δ_c` (a `[d_k, d_v]` correction matrix) is computed, the inter-chunk recurrence is:

```
S_{c+1} = G_c * S_c + Δ_c
```

where `G_c` is the product of per-token decay gates within chunk c, `G_c = ∏_{t in c} g_t`. This is a linear recurrence over matrix-valued elements and is associative — consecutive chunk updates can be composed.

But the within-chunk computation (which computes `Δ_c` given the tokens in chunk c and the current state `S_c`) requires `S_c` as an input. It is not independent of the current state in the same way that Mamba's per-step write is independent of the state. This means that even for the parallel scan over chunks, each chunk's computation has a data dependency on the previous chunk's output state. The parallel scan can parallelize the inter-chunk recurrence operator composition, but it cannot parallelize the within-chunk computation without reordering the data flow.

### 3.2 The Number of Chunks Is Small at Practical Sequence Lengths

For sequence length T and chunk size C=64:

- Number of chunks = T / C = T / 64
- At T=8192: 128 chunks
- At T=4096: 64 chunks
- At T=2048: 32 chunks

A Python loop over 128 chunks, where each iteration calls TTNN matmuls for the within-chunk WY step (dominant cost: a `[64, 128] × [128, 128]` matmul = 8×4×4 = 128 tiles per chunk), is not a bottleneck. The within-chunk matmul dominates per-chunk cost; the Python loop overhead (a few microseconds per iteration) is small compared to the within-chunk compute.

For T < 256K, the number of chunks stays below 4096 — the point at which loop overhead would become comparable to the compute cost of individual chunks. DeltaNet prefill at T > 256K is not a current requirement for Qwen3.6-35B-A3B inference.

### 3.3 Parallel Scan Memory Cost

A parallel prefix scan over T/C chunk-level state-correction pairs requires storing O(T/C × d_k × d_v) intermediate scan operands. For T=8192, C=64, d_k=d_v=128:

```
128 scan operands × 128 × 128 × 2 bytes BF16 = 128 × 32,768 bytes = 4,194,304 bytes ≈ 4 MB
```

4 MB of DRAM for scan operand storage is acceptable in the 12 GB DRAM budget. However, if a parallel scan implementation is pursued in the future, this allocation must be pre-allocated as a persistent buffer (not allocated inside the trace bracket, where dynamic allocation would break trace).

> **Note:** At T=8192 with C=64, a sequential Python chunk loop executing TTNN matmuls is the correct first prefill implementation. Parallelizing the inter-chunk scan is a future optimization, not a prerequisite for correctness or for the first performance target.

---

## 4. When Parallel Prefix Scan Would Be Relevant

For completeness, parallel prefix scan over DeltaNet chunks would become relevant under the following conditions:

1. **Very long sequences** (T > 256K) where the number of chunks (T/64 > 4096) makes a Python loop overhead comparable to within-chunk compute — unlikely for current Qwen3 inference workloads.
2. **The within-chunk WY computation is restructured** to separate the state-dependent and state-independent parts, enabling the state-independent part to be precomputed in parallel across chunks — a non-trivial algorithmic change.
3. **A tt-metal parallel scan primitive exists** for matrix-valued operands with the associative composition `(G_1, Δ_1) ⊕ (G_2, Δ_2) = (G_2 * G_1, G_2 * Δ_1 + Δ_2)` — this operator is associative and can in principle be implemented as a parallel scan; however, no such primitive currently exists in tt-metal.

None of these conditions hold for the current implementation target.

---

## 5. Summary

| Context | Parallel prefix scan applicable? | Reason |
|---|---|---|
| DeltaNet decode (T=1) | No | One recurrence step; nothing to parallelize |
| DeltaNet prefill, within-chunk (C=64 tokens) | No | WY decomposition has data dependency on current state; not expressible as standard associative scan |
| DeltaNet prefill, inter-chunk (T/C chunks) | In principle yes; not recommended for first implementation | Sequential Python loop over T/C ≤ 128 chunks is not a bottleneck; parallel scan overhead is not justified until T > 256K |
| Mamba SSM (for comparison) | Yes (used in Mamba prefill) | Mamba's write is independent of current state; standard linear recurrence is associative with scalar-times-matrix composition |

The correct prefill implementation is a Python loop over `T / C = T / 64` chunks, calling TTNN matmuls for the within-chunk WY step and TTNN ops for the inter-chunk state transfer. This is the approach described in Chapter 7's Task 7 and Chapter 6's prefill latency estimate.
