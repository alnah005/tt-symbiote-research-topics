# Math Fidelity Trade-Off: `HiFi4` vs. `HiFi2` in SDPA

## Overview

The Ling model configures `paged_sdpa_decode` with the following compute kernel config:

```python
compute_kernel_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
```

`HiFi4` is the highest precision compute mode on Wormhole and is the conservative default for BFloat16 transformer models. However, it is not always necessary: `HiFi2` can deliver a meaningful throughput improvement by reducing FPU pipeline stalls in matrix operations. Whether the precision reduction is tolerable for attention — specifically for the QK dot product, the softmax exponential, and the output `V`-weighted accumulation — depends on the model's sensitivity to rounding error.

This file provides a first-principles analysis of the trade-off.

## TTNN Math Fidelity Levels and Wormhole Hardware Modes

### The Three Levels

TTNN exposes three math fidelity levels for Wormhole kernels:

```
MathFidelity.LoFi   — lowest precision, highest throughput
MathFidelity.HiFi2  — intermediate precision
MathFidelity.HiFi4  — highest precision, matches BF16 specification
```

These are not software rounding modes. They map directly to hardware configuration bits in the Wormhole Tensix FPU that control how the BFloat16 multiplier is pipelined.

### Wormhole FPU Architecture and Fidelity Mapping

The Wormhole Tensix FPU performs matrix operations (MMUL tiles) using a pipelined multiply-accumulate structure. BFloat16 has 7 stored mantissa bits plus 1 implicit leading bit, giving 8 significant bits total. Machine epsilon for BF16 is therefore `ε_BF16 = 2^{-(8-1)} = 2^{-7} ≈ 0.0078`. For a full-precision BF16 multiply `a × b`, both operands must contribute all 8 significant bits to the product. The hardware achieves this through a two-phase approach:

- **Phase 1 (high mantissa bits):** Multiply using the upper 4 bits of each mantissa.
- **Phase 2 (low mantissa bits):** Multiply using the lower 4 bits and add the cross-products.

Each phase requires one pass through the multiply pipeline. The fidelity levels control how many phases are executed:

Table: Wormhole math fidelity levels — hardware phases and effective mantissa width

| Fidelity | Multiply phases per BF16 tile op | Effective mantissa bits (out of 8 significant) | Throughput relative to HiFi4 |
|---|---|---|---|
| `LoFi` | 1 phase, upper 4 bits only | ~4 | ~2× faster [ESTIMATE] |
| `HiFi2` | 1 phase, upper 4 bits of A × all 8 bits of B | ~6–7 | ~1.5× faster [ESTIMATE] |
| `HiFi4` | 2 phases, full 8×8 mantissa product | 8 (full BF16, all significant bits) | 1× (baseline) |

The `HiFi2` designation refers to using two of the four 4-bit sub-multiplier stages, giving approximately 6–7 effective mantissa bits in the product before accumulation. This is a hardware asymmetry: one operand contributes its full 8-bit mantissa while the other contributes only its upper 4 bits.

The throughput gain comes from pipeline reuse: the FPU pipeline is shared across matrix tile operations, and reducing from two passes to one roughly halves the per-tile multiply latency for compute-bound kernels. For memory-bandwidth-bound kernels (where the FPU is waiting on DRAM reads), the gain may be smaller because the computation is not on the critical path.

### How These Map to SDPA Operations

`paged_sdpa_decode` performs three categories of floating-point computation that are affected by fidelity:

1. **QK dot product** — `q · k^T` for each Q head and each KV token: a tile matmul of `(1, H)` × `(H, S_chunk)`. This is the inner loop's most compute-intensive operation and the primary beneficiary of fidelity reduction.
2. **Softmax** — element-wise `exp(x - max)` followed by division by the running sum. The `exp` function uses a polynomial approximation on the Tensix SFPU (scalar FPU); `MathFidelity` does not directly govern SFPU precision (it governs the matrix FPU). However, `fp32_dest_acc_en` affects the precision of the accumulator that feeds into the softmax normalization step.
3. **Output accumulation** — `softmax(qk) · v`: another tile matmul of `(1, S_chunk)` × `(S_chunk, H)`. This is a second compute-intensive operation affected by fidelity.

Reducing from `HiFi4` to `HiFi2` affects operations (1) and (3). The softmax `exp` and normalisation are governed by `fp32_dest_acc_en` (discussed below) rather than `MathFidelity` directly.

## Role of `fp32_dest_acc_en` and `packer_l1_acc`

### `fp32_dest_acc_en=True`

The Wormhole Tensix FPU accumulates tile matmul partial sums in a destination register file. When `fp32_dest_acc_en=True`, this destination register operates in FP32 mode (32-bit IEEE float). When `False`, it operates in BF16 mode.

For attention, FP32 accumulation is significant in two places:

**QK dot product accumulation.** The dot product `q · k^T` for `head_dim=128` adds together 128 pairwise BF16 products. Without FP32 accumulation, each partial sum is rounded back to BF16 after every add (or after every 16 adds, depending on the pipeline). With FP32 accumulation, partial sums are held in 32 bits and rounded to BF16 only when the result is packed back to L1. For long vectors (H=128), catastrophic cancellation can occur in BF16 accumulation when positive and negative terms nearly cancel; FP32 accumulation suppresses this.

**V-weighted output accumulation.** The same argument applies to the output matmul `softmax(qk) · v`. Each output element is a sum of `S_chunk` terms. For large chunk sizes or long sequences, FP32 accumulation preserves more precision in the output before the final pack to BF16.

Keeping `fp32_dest_acc_en=True` when reducing `MathFidelity` to `HiFi2` is the standard practice: the throughput gain from `HiFi2` is in the multiply pipeline, not in the accumulator, so there is no throughput reason to disable FP32 accumulation. Doing so independently would degrade precision without performance benefit at `HiFi2`.

### `packer_l1_acc=True`

The packer is the hardware unit that writes the FPU's destination register (FP32, when `fp32_dest_acc_en=True`) back to L1 SRAM. When `packer_l1_acc=True`, the packer performs an additional read-modify-write: it reads the existing L1 value, adds the destination register value to it in FP32, and writes the result back. This enables in-place L1 accumulation for operations that span multiple FPU passes over the same output tile.

In `paged_sdpa_decode`, `packer_l1_acc=True` allows the running softmax-weighted output accumulator (the `O` tensor) to be maintained directly in L1 across KV page iterations without requiring an explicit read-add-write sequence in the kernel's inner loop. This reduces L1 traffic and kernel complexity.

Disabling `packer_l1_acc` would require the kernel to manually load the running `O` from L1, add the new contribution, and store it back — adding extra L1 memory traffic per KV chunk iteration. For long sequences with many KV chunks, this would be measurably slower. `packer_l1_acc=True` is therefore correct to keep regardless of the `MathFidelity` setting.

## Precision Analysis: `HiFi2` vs. `HiFi4` for BFloat16 Attention

### BF16 Baseline

Both `HiFi2` and `HiFi4` operate on BF16 inputs and produce BF16 outputs. The difference is the internal precision of the multiply pipeline, not the input or output representation. A BF16 number has 7 stored mantissa bits plus 1 implicit leading bit, yielding 8 significant bits and approximately 2–3 decimal digits of precision.

The maximum representable precision error of a single BF16 multiply at full precision (`HiFi4`) is bounded by BF16 machine epsilon: `ε_BF16 = 2^{-(8-1)} = 2^{-7} ≈ 0.0078`. The correct formulation is: 7 stored mantissa bits + 1 implicit = 8 significant bits; machine epsilon = 2^{-(significant_bits - 1)} = 2^{-7}. At `HiFi2`, the effective mantissa of the product uses ~6–7 significant bits, giving a per-multiply error up to `2^{-6} ≈ 0.016` to `2^{-5} ≈ 0.031` — roughly 2× to 4× larger than BF16 `HiFi4` error [ESTIMATE].

### Error Propagation in QK Dot Product

For a single Q-head, the raw attention logit for one KV token is:

```
score = (1 / sqrt(H)) * sum_{i=0}^{H-1} q[i] * k[i]
      = (1 / sqrt(128)) * sum_{i=0}^{127} q[i] * k[i]
      = (1 / 11.31) * (128 terms)
```

With `fp32_dest_acc_en=True`, the 128 partial products are accumulated in FP32 regardless of the fidelity level. The per-multiply error at `HiFi2` is amplified by the number of terms only if the errors are correlated; for typical attention weights (values near zero, random sign), errors accumulate with approximately `sqrt(H)` scaling (random walk). The resulting score error is bounded by:

```
Δscore ≈ (1 / sqrt(H)) * sqrt(H) * ε_multiply
        = ε_multiply
        ≈ 0.016–0.031   at HiFi2  [ESTIMATE, range reflects ~6–7 effective mantissa bits]
        ≈ 0.008         at HiFi4  [ESTIMATE, ≈ 2^{-7}]
```

Both are below the natural rounding noise of BF16 inputs, which introduces per-element quantisation error of ε_BF16 ≈ 0.008 before the dot product is even computed. In other words: the input values `q[i]` and `k[i]` are already rounded to BF16, introducing errors of this magnitude at every element. The additional multiply error from `HiFi2` is of the same order as the input quantisation error.

### Error Propagation Through Softmax

Softmax is the most precision-sensitive operation in attention:

```
p_i = exp(score_i - max_score) / sum_j exp(score_j - max_score)
```

The critical concern is the subtraction `score_i - max_score`, which can cause catastrophic cancellation when `score_i ≈ max_score`. However:

1. `max_score` is computed from `scores` which are themselves BF16-rounded; the subtraction cancellation is already present at BF16 regardless of fidelity.
2. The `exp` operation is performed on the Tensix SFPU, which uses FP32 internally and is not governed by `MathFidelity`.
3. For attention, scores are typically in the range `[-5, 5]` (after scaling by `1/sqrt(H)`), and BF16's 8 mantissa bits are sufficient to represent differences between scores at this range without dramatic cancellation.

The effective precision of the softmax output is therefore dominated by (a) BF16 input quantisation and (b) SFPU exp precision, neither of which is changed by lowering `MathFidelity` from `HiFi4` to `HiFi2`. The `HiFi2` error increase in the dot product (≈0.016–0.031 per score) will cause small shifts in the softmax distribution, but these are within the tolerance of BF16 attention outputs.

### Error Propagation in Output Accumulation

The output for head `h` is:

```
O[h] = sum_j p_j * v[h][j]
```

With `fp32_dest_acc_en=True`, the accumulation is in FP32. The multiply `p_j * v[h][j]` is the operation affected by fidelity. At `HiFi2`, the per-multiply error is ≈0.016–0.031 per term. For `S=2048` terms (at full KV context), the accumulated output error (random-walk) is:

```
ΔO ≈ sqrt(S) * ε_multiply * scale
   ≈ sqrt(2048) * 0.016 * |v_typical|
   ≈ 45.3 * 0.016 * 0.01    (typical V values ≈ 0.01 for normalised BF16 vectors)
   ≈ 0.007  [ESTIMATE, lower bound; upper bound ≈ 0.014 at ε=0.031]
```

Expressed as a fraction of typical output values (≈ 0.1–1.0), this is a relative error of less than 1%, which is within the noise floor of BF16 post-attention computations.

### Summary of Precision Impact

Table: Precision impact of `HiFi2` vs. `HiFi4` on BF16 paged SDPA decode (Ling, N_q=16, N_kv=4, H=128) [ESTIMATE]

| Operation | Affected by fidelity? | HiFi4 error | HiFi2 error | Within BF16 noise? |
|---|---|---|---|---|
| QK dot product (per score) | Yes — multiply | ~0.008 (≈ 2^{-7}) | ~0.016–0.031 (≈ 2^{-6} to 2^{-5}) | Yes — comparable to BF16 input quantisation |
| Softmax exp | No — SFPU, FP32 | N/A | N/A | N/A |
| Softmax normalisation | Indirectly | Minimal | Minimal | Yes |
| Output accumulation (S=2048) | Yes — multiply | ~0.004 | ~0.007–0.014 | Yes |
| Output accumulation (S=32768) | Yes — multiply | ~0.014 | ~0.028–0.057 | Borderline — verify at long context |

The analysis suggests that `HiFi2` is safe for typical decode sequence lengths (S ≤ 4096). At very long context (S > 16384), the accumulated output error at `HiFi2` grows to the borderline range; this should be validated empirically before deploying `HiFi2` for long-context workloads.

## Performance Delta: Expected Throughput Gain from `HiFi4` → `HiFi2`

### When Does Fidelity Matter for Throughput?

The throughput gain from `HiFi2` over `HiFi4` is realised only when the SDPA kernel is **compute-bound** — when the FPU multiply pipeline is on the critical path, not DRAM bandwidth.

At decode batch=1 with short sequences (S ≤ 512), `paged_sdpa_decode` is **DRAM-bandwidth-bound**: the KV cache pages must be streamed from DRAM, and the FPU sits largely idle between page fetches. In this regime, reducing fidelity does not improve throughput because the bottleneck is not the FPU.

At longer sequences (S ≥ 2048) or when the KV cache is in L1 (e.g., due to aggressive prefetching), the kernel can become **compute-bound**, and fidelity reduction will help.

For Ling on T3K at decode batch=1, the compute-to-bandwidth crossover point depends on the DRAM bandwidth per chip and the FPU throughput. A rough characterisation:

```
Wormhole DRAM bandwidth per chip (peak): ~192 GB/s  [ESTIMATE based on HBM spec]
Bytes per KV token per KV head (BF16):   2 * 128 * 2 = 512 B  (K and V, head_dim=128, 2 bytes)
KV bandwidth demand at N_kv=4, seq S:    4 * 512 * S / t_SDPA

Wormhole FPU throughput (HiFi4):         ~50 TFLOPS peak matrix [ESTIMATE]
FLOPs per KV token per Q head:           2 * 128 (QK) + 2 * 128 (OV) = 512 FLOPs
FLOPs for all N_q=16 heads, seq S:       16 * 512 * S = 8192 * S
```

The compute-bound crossover is at:

```
S_crossover = DRAM_bandwidth / (KV_bytes_per_token * N_kv)
            = 192e9 / (512 * 4)
            = 192e9 / 2048
            ≈ 93,750,000 tokens  (≈ 93.75 million tokens)  [ESTIMATE]
```

At ~93.75 million tokens, this crossover is astronomically far beyond any practical context length; Ling's typical decode context (S ≤ 8192) is far below this crossover. The conclusion is that `paged_sdpa_decode` at batch=1 is **DRAM-bandwidth-bound for all practical sequence lengths**, and the throughput gain from `HiFi2` at batch=1 is expected to be small (< 10%) [ESTIMATE].

The picture changes at higher batch sizes: with batch > 1, the Q tensor is larger, Q-side FLOPs scale with batch, and the kernel can become compute-bound at shorter sequences. If Ling is run at batch > 1 in the future, `HiFi2` becomes a stronger candidate.

Table: Expected throughput impact of `HiFi2` vs. `HiFi4` for `paged_sdpa_decode` on Wormhole (N_q=16, N_kv=4, H=128) [ESTIMATE]

| Scenario | Bottleneck | Expected speedup from HiFi2 | Notes |
|---|---|---|---|
| Batch=1, S=512 | DRAM bandwidth | < 5% | FPU not on critical path |
| Batch=1, S=4096 | DRAM bandwidth | 5–10% | Marginal compute contribution begins |
| Batch=1, S=32768 | DRAM bandwidth (still) | 5–15% | Longer compute exposure between fetches |
| Batch=8, S=2048 | Mixed / compute | 15–30% | FPU utilisation higher with larger Q |
| Batch=32, S=1024 | Compute | 30–50% | Full throughput benefit of HiFi2 |

## Recommendation

### Current Configuration (`HiFi4`): Correct and Safe

`HiFi4` is the right default for Ling at launch. It provides full BF16-specification precision and is risk-free. Given that SDPA at batch=1 is DRAM-bandwidth-bound, the throughput cost of `HiFi4` vs. `HiFi2` is minimal (< 10% [ESTIMATE]) and the precision benefit is a useful safety margin during model development and validation.

### Can `HiFi2` Be Used?

Based on the analysis above, `HiFi2` is **likely safe for Ling's attention at typical decode context lengths (S ≤ 4096)** provided:

1. `fp32_dest_acc_en=True` is kept (accumulator precision is not reduced alongside fidelity).
2. `packer_l1_acc=True` is kept (L1 accumulation correctness is maintained).
3. The change is validated with a numeric accuracy test before production deployment.

At very long context (S > 16384), the output accumulation error at `HiFi2` reaches the borderline range identified in the precision table above; empirical measurement is required before enabling `HiFi2` in that regime.

### How to Validate `HiFi2`

The recommended validation procedure is a precision comparison test that compares `HiFi2` and `HiFi4` outputs over a range of sequence lengths:

```python
import torch
import ttnn
import math

def build_compute_config(fidelity):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

def run_sdpa(q, paged_k, paged_v, page_table, fidelity):
    return ttnn.transformer.paged_sdpa_decode(
        input_tensor_q=q,
        input_tensor_k=paged_k,
        input_tensor_v=paged_v,
        page_table=page_table,
        q_chunk_size=0,
        k_chunk_size=0,
        compute_kernel_config=build_compute_config(fidelity),
    )

# Run both configs on the same inputs
out_hi4 = run_sdpa(q, paged_k, paged_v, page_table, ttnn.MathFidelity.HiFi4)
out_hi2 = run_sdpa(q, paged_k, paged_v, page_table, ttnn.MathFidelity.HiFi2)

ref  = ttnn.to_torch(out_hi4).float()
cand = ttnn.to_torch(out_hi2).float()

abs_diff = (ref - cand).abs()
rel_diff = abs_diff / (ref.abs() + 1e-6)

print(f"Max absolute diff:  {abs_diff.max().item():.6f}")
print(f"Mean absolute diff: {abs_diff.mean().item():.6f}")
print(f"Max relative diff:  {rel_diff.max().item():.6f}")
print(f"Mean relative diff: {rel_diff.mean().item():.6f}")

# Acceptance thresholds for Ling (BF16 outputs, GQA decode):
# max_abs_diff < 0.05   (5% of typical output range)
# mean_abs_diff < 0.005 (0.5% mean deviation)
# These are conservative; tighter thresholds may be achievable
MAX_ABS_THRESHOLD  = 0.05
MEAN_ABS_THRESHOLD = 0.005
assert abs_diff.max() < MAX_ABS_THRESHOLD,  "HiFi2 max error exceeds threshold"
assert abs_diff.mean() < MEAN_ABS_THRESHOLD, "HiFi2 mean error exceeds threshold"
print("HiFi2 precision check passed.")
```

In addition to numeric tolerance, the end-to-end language model output quality should be checked with perplexity measurement on a held-out corpus:

```bash
# Pseudo-command; actual implementation depends on Ling's evaluation harness
python eval_perplexity.py \
    --model ling_bailing_moe \
    --compute-fidelity hifi2 \
    --dataset wikitext-103 \
    --sequence-length 2048 \
    --compare-to hifi4
# Expected: perplexity increase < 0.5% for HiFi2 vs. HiFi4  [ESTIMATE]
```

A perplexity increase of more than 1% would indicate that attention precision is degrading the output quality and `HiFi4` should be retained.

### Summary of Recommendation

Table: Final recommendation for SDPA compute kernel config in Ling decode

| Parameter | Current value | Recommended value | Rationale |
|---|---|---|---|
| `math_fidelity` | `HiFi4` | `HiFi2` (after validation) | DRAM-bound at batch=1; gain is modest but cost-free at long context |
| `fp32_dest_acc_en` | `True` | `True` (keep) | FP32 accumulation is free in terms of fidelity; removing it degrades precision |
| `packer_l1_acc` | `True` | `True` (keep) | Required for correct in-place L1 softmax accumulation across KV chunks |

The recommended path is: keep `HiFi4` during development and correctness validation; run the precision comparison test above; if thresholds pass, switch to `HiFi2` in production and measure the throughput delta with Tracy or TTNN op timers (see Chapter 7, `ttnn_op_timers.md`). The expected speedup at batch=1 is 5–15% for SDPA alone; the system-level impact on decode step time will be proportionally smaller given that SDPA is one of several operations in the attention forward pass.

---

**Next:** [Chapter 6 — RoPE and QK Norm](../ch6_rope_and_qk_norm/index.md)
