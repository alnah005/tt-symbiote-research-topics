# Activation Dtype and Precision

This document covers the accuracy and L1 footprint tradeoffs between `ttnn.bfloat16` and `ttnn.bfloat8_b` as the `activation_dtype` for fused SiLU output, and provides a validation approach for confirming that BFP8_B does not degrade model output below an acceptable threshold.

---

## What `activation_dtype` Controls

`activation_dtype` is a field in `MatmulMultiCoreReuseMultiCastProgramConfig` and `MatmulMultiCoreReuseMultiCast1DProgramConfig`. It specifies the numeric format in which the fused activation output tensor is written to its destination buffer in L1 or DRAM.

It does not affect the precision of the FPU matrix multiply-accumulate itself. The matmul accumulation uses the dtype negotiated between the input tensors and the `dtype` parameter on `ttnn.matmul`. `activation_dtype` only controls the format of the tensor that exits the SFPU activation pass.

```python
import ttnn

# BF16 activation output: 2,048 bytes per tile
program_config_bf16 = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(8, 8),
    in0_block_w=2,
    out_subblock_h=1,
    out_subblock_w=4,
    per_core_M=1,
    per_core_N=8,
    transpose_mcast=False,
    fused_activation=(ttnn.UnaryOpType.SILU, True),
    activation_dtype=ttnn.bfloat16,       # 2 bytes per element
)

# BFP8_B activation output: 1,024 bytes per tile
program_config_bfp8 = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(8, 8),
    in0_block_w=2,
    out_subblock_h=1,
    out_subblock_w=4,
    per_core_M=1,
    per_core_N=8,
    transpose_mcast=False,
    fused_activation=(ttnn.UnaryOpType.SILU, True),
    activation_dtype=ttnn.bfloat8_b,      # 1 byte per element (block floating-point)
)
```

---

## BF16 Tile vs. bfloat8_b Tile

| Property | BF16 | bfloat8_b |
|---|---|---|
| Bits per element | 16 | 8 |
| Bytes per 32×32 tile | 2,048 | 1,024 |
| L1 footprint for 8 tokens × 2048 d_ff | 32,768 bytes (32 KB) | 16,384 bytes (16 KB) |
| L1 footprint for 128 tokens × 2048 d_ff | 524,288 bytes (512 KB) | 262,144 bytes (256 KB) |
| Exponent bits | 8 | 8 (shared per block of 16) |
| Mantissa bits | 7 | 0 (implicit, one exponent per block) |

bfloat8_b is a block floating-point format: elements within a block of 16 share a single 8-bit exponent, and each element contributes a sign bit and no explicit mantissa. The effective dynamic range is preserved for smooth, relatively uniform activation distributions; accuracy degrades where large magnitude variation exists within a 16-element block.

SiLU output has the following distribution properties relevant to bfloat8_b accuracy:

- SiLU(x) = x * sigmoid(x); the output is negative for negative inputs, reaching a minimum of approximately −0.278 near x ≈ −1.28, then increasing back toward zero as x → −∞, and growing linearly for large positive x.
- The output distribution is skewed: many small values near zero, with long positive tail depending on the input distribution.
- Within a 16-element block, value ranges can span 2–3 orders of magnitude for typical transformer hidden states. This is the regime where bfloat8_b quantization error is most visible.

---

## L1 Footprint Reduction

Wormhole B0 provides 1.5 MB of L1 per Tensix core across 80 cores (8×10 grid). For sharded SwiGLU computations with large hidden dims, the activation tensor is split across cores. Even with sharding, the per-core slice of the activation tensor competes with weight tiles and other intermediate buffers for L1 space.

Switching `activation_dtype` from `ttnn.bfloat16` to `ttnn.bfloat8_b` halves the per-core activation tile footprint. At `d_ff=2048` and `num_tokens=128`, the full activation tensor is 512 KB in BF16 and 256 KB in BFP8_B. When sharded across 8 cores, that is 64 KB vs. 32 KB per core — a difference that can determine whether the shard fits in L1 without spilling to DRAM.

This is primarily relevant for:

- Prefill workloads with large token counts.
- Dense FFN layers with `d_ff > 4096` where the activation tensor is large.
- Configurations that already have tight L1 budgets due to weight double-buffering.

At decode batch sizes (1–16 tokens), the absolute activation tensor size is small regardless of dtype (max 16 KB BFP8_B at `num_tokens=8`, `d_ff=2048`). The dtype choice at decode is driven by accuracy requirements, not L1 pressure.

---

## Accuracy Tradeoff: BF16 vs. bfloat8_b

The quantization error introduced by bfloat8_b `activation_dtype` propagates into the element-wise multiply (`ttnn.mul(gate, up)`) and from there into the down_proj matmul. It does not affect the gate_proj matmul itself.

Expected accuracy behavior:

| Workload | BFP8_B activation_dtype impact |
|---|---|
| BF16 inference, frozen weights | Small perplexity increase; typically < 0.5 perplexity points on standard benchmarks |
| BFP8 weight inference | Often negligible; quantization noise from weights dominates |
| Fine-tuning (gradient flow through SiLU output) | Not applicable — `activation_dtype` only affects inference kernel output; fine-tuning uses separate autograd path |
| Low-precision accumulation stacks (e.g., INT8 weights + BFP8 activations) | Accuracy budget may already be tight; validate carefully |

---

## When to Use BF16 `activation_dtype`

Use `activation_dtype=ttnn.bfloat16` when:

- The model is being fine-tuned or evaluated for perplexity on a precision-sensitive benchmark.
- The downstream element-wise multiply (`ttnn.mul`) accumulates small differences that must be preserved (e.g., residual-adjacent computations).
- BFP8_B accuracy validation has not been completed for the target model and dataset.
- L1 pressure is not a constraint (small token counts, large L1 shards available).

Use `activation_dtype=ttnn.bfloat8_b` when:

- The model runs in BFP8 weight inference mode and the activation quantization noise is within the accepted model accuracy budget.
- Large `d_ff` or large `num_tokens` create L1 pressure that is relieved by the 50% activation tile size reduction.
- An accuracy validation run (described below) has confirmed the perplexity delta is within tolerance.

---

## Accuracy Validation Approach

The recommended approach for validating bfloat8_b `activation_dtype` is a logit comparison against a BF16 reference run on a representative set of inputs.

```python
import ttnn
import torch

def run_ffn_fused(x, w1, w3, activation_dtype):
    """Single SwiGLU FFN forward pass with specified activation_dtype."""
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=x.shape[-2] // 32,
        per_core_N=w1.shape[-1] // 32 // 8,
        transpose_mcast=False,
        fused_activation=(ttnn.UnaryOpType.SILU, True),
        activation_dtype=activation_dtype,
    )
    gate = ttnn.matmul(x, w1, program_config=program_config, dtype=ttnn.bfloat16)
    up = ttnn.matmul(x, w3, dtype=ttnn.bfloat16)
    return ttnn.mul(gate, up)

# BF16 reference
hidden_bf16 = run_ffn_fused(x, w1, w3, activation_dtype=ttnn.bfloat16)

# BFP8_B candidate
hidden_bfp8 = run_ffn_fused(x, w1, w3, activation_dtype=ttnn.bfloat8_b)

# Convert to PyTorch for comparison
ref = ttnn.to_torch(hidden_bf16).float()
cand = ttnn.to_torch(hidden_bfp8).float()

# Per-element absolute error
abs_err = (ref - cand).abs()
print(f"Max absolute error:  {abs_err.max().item():.6f}")
print(f"Mean absolute error: {abs_err.mean().item():.6f}")
print(f"Relative error (L2): {torch.norm(ref - cand) / torch.norm(ref):.6f}")
```

Interpret results as follows:

| Metric | Acceptable threshold (BF16 inference) | Acceptable threshold (BFP8 weight inference) |
|---|---|---|
| Mean absolute error | < 0.01 | < 0.05 |
| Max absolute error | < 0.1 | < 0.2 |
| Relative L2 error | < 0.001 | < 0.01 |

These thresholds are indicative. The correct acceptance criterion is model-level: run the full model with `activation_dtype=ttnn.bfloat8_b` on a calibration set and measure perplexity against the BF16 baseline. A delta of less than 0.5 perplexity points on WikiText-2 or equivalent is a standard starting tolerance for BFP8 activation quantization in MoE inference.

If the perplexity delta exceeds tolerance, revert to `activation_dtype=ttnn.bfloat16`. The L1 footprint benefit of BFP8 is not worth accuracy regression in precision-sensitive deployments.

---

## Summary Table

| Scenario | Recommended `activation_dtype` | Rationale |
|---|---|---|
| BF16 inference, decode (1–16 tokens) | `ttnn.bfloat16` | L1 pressure low; preserve accuracy for element-wise mul |
| BFP8 weight inference, decode | `ttnn.bfloat8_b` | Consistent with weight precision; validate accuracy delta |
| BF16 inference, prefill (128+ tokens) | `ttnn.bfloat8_b` (if L1 pressure) | Large activation tensor; halved tile size helps sharding |
| Fine-tuning evaluation | `ttnn.bfloat16` | Accuracy-critical; do not introduce quantization noise |
| Unvalidated configuration | `ttnn.bfloat16` | Safe default; switch to BFP8 only after validation |

---

---

**Next:** [Chapter 6 — Performance Impact and Recommendations](../ch06_performance_impact_and_recommendations/index.md)
