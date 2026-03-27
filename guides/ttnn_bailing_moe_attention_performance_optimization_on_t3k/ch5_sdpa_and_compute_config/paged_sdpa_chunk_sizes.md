# Paged SDPA Chunk Sizes: `q_chunk_size` and `k_chunk_size`

## Overview

`paged_sdpa_decode` is the TTNN kernel responsible for computing scaled dot-product attention during autoregressive decode. It reads the current token's Q vector, traverses a paged KV cache stored in DRAM, and produces the attention output. Two parameters — `q_chunk_size` and `k_chunk_size` — control how the kernel tiles this traversal across Q heads and KV sequence tokens respectively.

The Ling model passes both as zero:

```python
attn_output = ttnn.transformer.paged_sdpa_decode(
    input_tensor_q=q,
    input_tensor_k=paged_k_cache,
    input_tensor_v=paged_v_cache,
    page_table=page_table,
    q_chunk_size=0,
    k_chunk_size=0,
    compute_kernel_config=compute_kernel_config,
    ...
)
```

This document explains what zero means for each parameter, why it is the appropriate choice for Ling's 16/4 GQA configuration at decode batch=1, and what the performance consequences would be of using explicit non-zero values.

## What `q_chunk_size` Controls

### The Q Tiling Problem

At decode batch=1, the Q tensor has shape `(1, N_q, 1, H)` — one token, `N_q=16` heads, `head_dim=128`. The SDPA kernel must compute attention for each of the 16 Q heads. The kernel can process Q heads in one of two strategies:

1. **Single-chunk (all heads together):** Process all 16 Q heads in a single kernel invocation. The output accumulation buffer for the softmax normalisation holds attention scores for all 16 heads simultaneously in the destination registers.
2. **Multi-chunk (loop over head groups):** Process `q_chunk_size` Q heads at a time, looping until all heads are processed. Each iteration uses a smaller destination buffer, potentially reducing register pressure, but at the cost of loop overhead per chunk.

`q_chunk_size` is the stride of this loop. A value of `q_chunk_size=8` would cause the kernel to process Q heads 0–7 in the first iteration and heads 8–15 in the second.

### Semantics of `q_chunk_size=0`

A value of zero is interpreted by `paged_sdpa_decode` as a directive to process **all Q heads in a single chunk** — equivalently, the chunk size defaults to `N_q`. This is confirmed by the kernel's parameter-validation logic, which substitutes `N_q` for zero before constructing the tile loop:

```cpp
// From ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_decode_program_factory.cpp
// (schematic — exact line numbers may vary by tt-metal revision)
uint32_t effective_q_chunk = (q_chunk_size == 0) ? num_q_heads : q_chunk_size;
```

The effect is that a single kernel launch handles all 16 Q heads with no inter-chunk loop overhead. For Ling at decode batch=1 — where `N_q=16` and each head's data fits comfortably in the available L1 register space — this is correct and efficient.

### GQA Implication for `q_chunk_size`

In a GQA model, Q heads are grouped into `G = N_q / N_kv = 4` groups of `G=4` Q heads each, where all Q heads in a group share one KV head. When `q_chunk_size=0` (single chunk), the kernel processes all 16 Q heads together and must load each KV head `G=4` times — once per Q head in the group — or must restructure the inner loop to amortise KV loads across the group.

The `paged_sdpa_decode` kernel includes a GQA-aware inner loop that, when it detects `N_q > N_kv`, iterates over KV pages once per KV-head group and fans the result out to all `G` Q heads in that group. This means the number of DRAM page loads is proportional to `N_kv` (not `N_q`), regardless of the chunk configuration. The GQA fan-out is:

```
For each KV head k in [0, N_kv):
    Load KV page sequence for head k from DRAM
    Compute attention scores for all G Q heads assigned to KV head k
    Accumulate softmax-normalised output for each of the G Q heads
```

With `q_chunk_size=0`, all 16 Q heads' accumulators are live simultaneously during each KV-page traversal, which maximises register utilisation but also maximises the register file requirement. For `N_q=16`, `head_dim=128`, the accumulator footprint is:

```
At decode batch=1, Q has only 1 token (not a full 32-row tile).
Accumulator per Q head: 1 token × 128 head_dim FP32 values = 128 × 4 B = 512 bytes
Total accumulator for all N_q heads: 16 × 512 B = 8,192 bytes ≈ 8 KB
```

The Wormhole Tensix core has approximately 1.5 MB of L1 SRAM, of which around 512 KB is available for compute kernels (the remainder is reserved for input/output double-buffering and firmware). An 8 KB accumulator footprint is well within budget and does not cause register spilling to DRAM. This confirms that `q_chunk_size=0` is safe for `N_q=16`.

For a hypothetical model with `N_q=64` (e.g., LLaMA-70B style), the accumulator would be `64 × 4 × 128 B = 32 KB`, still manageable. The threshold at which register pressure forces chunking is model-dependent; for Ling it is not a concern.

## What `k_chunk_size` Controls

### The KV Sequence Tiling Problem

The KV sequence dimension `S` (total tokens in the KV cache for the current request) is the dominant dimension for SDPA at decode time. At long context lengths (e.g., S=8192 or S=32768), the KV cache spans many DRAM pages and the kernel must stream each page into L1, compute the partial dot product `q · k^T` and softmax numerics, and accumulate the output `v`-weighted sum.

`k_chunk_size` controls how many KV tokens are processed per inner-loop iteration before flushing the partial softmax accumulator and moving to the next chunk. More precisely, since the paged KV cache stores tokens in fixed-size pages of `page_size` tokens each, `k_chunk_size` determines how many tokens (equivalently, how many page_size-aligned blocks) are processed before the online softmax normaliser is updated.

A value of `k_chunk_size=64` (assuming `page_size=32`) would process 2 pages per inner iteration; the kernel would update the running softmax max `m` and sum `l` after every 64 tokens rather than after every page.

### Semantics of `k_chunk_size=0`

Like `q_chunk_size`, a value of zero instructs the kernel to use a default chunk size. For the KV dimension, the default is typically the kernel's native page size or a compile-time-selected tile size rather than the full sequence. From the kernel source (paraphrased):

```cpp
// From ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_decode_program_factory.cpp
// (schematic)
uint32_t effective_k_chunk = (k_chunk_size == 0) ? DEFAULT_K_CHUNK_SIZE : k_chunk_size;
// DEFAULT_K_CHUNK_SIZE is a compile-time constant; for Wormhole paged SDPA decode
// this is typically set to the page_size (e.g., 32 or 64 tokens) so that one
// DRAM page fetch corresponds to one k_chunk iteration.
```

In practice, `k_chunk_size=0` means the kernel will use its built-in default tiling, which is calibrated to match L1 double-buffering granularity for efficient DRAM streaming. This default is **not** "process the full sequence in one shot" — the full sequence cannot fit in L1 at any practical context length. Instead, the kernel processes one page (or a small multiple of pages) at a time and accumulates the online softmax state across pages.

The important point is that `k_chunk_size=0` does not disable chunking for the KV dimension; it defers to the kernel's internally optimised default. For Ling's use case, this default is appropriate.

### Performance Characteristics of `k_chunk_size` at Various Sequence Lengths

The inner KV loop executes `ceil(S / effective_k_chunk)` iterations. Each iteration involves:

1. A DRAM page read for K and V (bandwidth-bound at long context).
2. A `q · k^T` tile matmul for the `G` Q heads in the current group.
3. Online softmax update (compare new max, rescale running sum).
4. Accumulation of `v`-weighted output.

At short sequences (S ≤ 512, e.g., early decode steps), the KV data fits within a small number of pages and the loop terminates quickly. At long sequences (S = 8192–32768), DRAM bandwidth for KV streaming dominates latency. Adjusting `k_chunk_size` can influence the ratio of compute to DRAM latency:

Table: Effect of `k_chunk_size` on paged SDPA decode at selected sequence lengths (N_q=16, N_kv=4, H=128, page_size=32) [ESTIMATE]

| `k_chunk_size` | KV loop iters at S=2048 | DRAM prefetch efficiency | L1 KV buffer size | Recommended for |
|---|---|---|---|---|
| 0 (default) | ~64 (= S / page_size) | Kernel-optimised double-buffering | ~2 pages (8 KB) | All decode lengths — safe default |
| 32 | 64 | Standard — 1 page per iter | ~1 page (4 KB) | Fine-grained control, minimal L1 impact |
| 64 | 32 | Moderate — 2 pages per iter | ~2 pages (8 KB) | Slightly reduced loop overhead |
| 128 | 16 | Good — 4 pages per iter | ~4 pages (16 KB) | Long context (S > 4096) |
| 256 | 8 | Best for bandwidth | ~8 pages (32 KB) | Very long context (S > 16384), ample L1 |

Note: Page size of 32 tokens is assumed; actual page size is a compile-time constant in the KV cache allocator. Larger `k_chunk_size` increases L1 pressure for the KV staging buffer; at Ling's decode batch=1 with sparse L1 usage, this is not a concern for values up to 256.

## Correctness Analysis for Ling's GQA Configuration

### Head Count Alignment

The `paged_sdpa_decode` kernel requires that `N_q` be an integer multiple of `N_kv` (the GQA constraint). For Ling: `N_q / N_kv = 16 / 4 = 4` — satisfied. This means the kernel's GQA fan-out logic will create four groups of four Q heads, each group sharing one KV head. With `q_chunk_size=0`, all four groups are processed within a single kernel invocation, which is correct.

If `q_chunk_size` were set to a value that does not evenly divide `N_q` (e.g., `q_chunk_size=3` with `N_q=16`), the final chunk would be a partial chunk. The kernel handles this with tail masking; however, the shard layout of Q on the Tensix grid must be consistent with the chunk size. For `q_chunk_size=0` (single chunk), no partial-chunk issue arises.

### `head_dim=128` and Tile Alignment

`head_dim=128 = 4 × TILE_SIZE`. This is exactly divisible by the tile size, so no padding is needed in the head dimension. Every Q, K, and V tile within the paged cache is a full `(32, 32)` tile in BF16 TILE_LAYOUT. The kernel performs matmuls on full tiles throughout, with no tile-edge masking in the head dimension. This is the ideal condition for `paged_sdpa_decode` and requires no special configuration.

### Decode Sequence Length Range

At each decode step, S grows by one token. In practice, relevant decode lengths for long-form generation span:

```
S = 1        (first decode token, KV cache has only the prefill output)
S = 512      (moderate context, ~16 pages)
S = 2048     (standard context, ~64 pages)
S = 8192     (extended context, ~256 pages)
S = 32768    (long context, ~1024 pages)
```

For all these lengths, `k_chunk_size=0` (kernel-default tiling) correctly streams through the full sequence with online softmax normalisation. The correctness of the online softmax computation is independent of `k_chunk_size` — each chunk updates the running `(m, l)` pair using the standard safe-softmax recurrence, and the final output `O` is correctly normalised after the last chunk.

There is no minimum `k_chunk_size` required for numerical correctness; the online softmax algorithm is mathematically exact regardless of chunk granularity, provided the implementation correctly carries forward the running maximum and sum.

## Recommended Values

### Current Setting: `q_chunk_size=0`, `k_chunk_size=0`

The current configuration is **correct and appropriate** for Ling's 16/4 GQA layout at decode batch=1.

- `q_chunk_size=0` processes all 16 Q heads in a single chunk. This is valid because the accumulator footprint (≈8 KB) is far below the L1 budget, and it avoids multi-chunk loop overhead.
- `k_chunk_size=0` defers to the kernel's built-in default tiling for the KV sequence dimension, which is calibrated for efficient double-buffered DRAM streaming at the page granularity.

### When to Consider Explicit Non-Zero Values

**`q_chunk_size`:** Only consider a non-zero value if profiling reveals register spilling during SDPA (visible as unexpected DRAM traffic on a kernel that should be L1-resident). For `N_q=16` this is not expected to occur. If the model were scaled to `N_q=64`, trying `q_chunk_size=16` (4 chunks of 16 heads) would be the starting point.

**`k_chunk_size`:** If SDPA kernel time at long context (S > 8192) is dominated by loop-overhead rather than DRAM bandwidth — identifiable via Tracy by seeing many short kernel launches rather than a few long ones — then increasing `k_chunk_size` to 128 or 256 may reduce overhead. This is unlikely to matter at shorter decode sequences but worth measuring at production context lengths.

Table: Recommended `q_chunk_size` and `k_chunk_size` values for Ling at various operating points

| Scenario | `q_chunk_size` | `k_chunk_size` | Rationale |
|---|---|---|---|
| Decode batch=1, S ≤ 4096 (current) | 0 | 0 | Correct; kernel defaults are well-tuned |
| Decode batch=1, S = 8192–32768 | 0 | 128 | Reduce KV loop iterations; measure vs. default [ESTIMATE] |
| Decode batch > 1 (future) | 0 | 0 or 64 | Re-evaluate at higher batch; Q chunk may still be 0 |
| N_q scaled to 64 (hypothetical) | 16 | 0 | Limit accumulator footprint; 4 chunks of 16 Q heads |

### Validation Approach

To confirm that a non-zero `k_chunk_size` does not alter correctness:

```python
import torch
import ttnn

# Reference: run with k_chunk_size=0 (default)
out_ref = ttnn.transformer.paged_sdpa_decode(
    q, paged_k, paged_v, page_table,
    q_chunk_size=0, k_chunk_size=0,
    compute_kernel_config=compute_kernel_config,
)

# Candidate: run with k_chunk_size=128
out_cand = ttnn.transformer.paged_sdpa_decode(
    q, paged_k, paged_v, page_table,
    q_chunk_size=0, k_chunk_size=128,
    compute_kernel_config=compute_kernel_config,
)

ref_torch  = ttnn.to_torch(out_ref)
cand_torch = ttnn.to_torch(out_cand)
max_abs_diff = (ref_torch - cand_torch).abs().max().item()
# Expect max_abs_diff < 1e-3 for BF16 outputs; any non-zero diff is rounding only
print(f"Max absolute diff: {max_abs_diff:.6f}")
```

Because the online softmax recurrence is mathematically exact, `max_abs_diff` should be at or below BF16 rounding error (≈ 0.004 for values near 1.0). A larger difference would indicate a kernel bug in the chunked path.

---

**Next:** [Math Fidelity Trade-Off](./math_fidelity_tradeoff.md)
