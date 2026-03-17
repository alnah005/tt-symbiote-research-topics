# Chapter 5: Per-Projection Quantization Strategy — Gate, Up, and Down

## Overview

This chapter synthesizes the format fundamentals, API patterns, accuracy analysis, and
throughput trade-offs from Chapters 1–4 into a concrete, per-projection mixed-precision
strategy for Mixture of Experts (MoE) expert FFN weights.

Each of the three projections in a SwiGLU expert FFN — gate (w1), up (w3), and down (w2)
— plays a structurally different role in the computation. That structural difference
determines how much quantization error each projection can absorb before accuracy
degrades unacceptably. Chapter 5 explains the mechanistic reasoning behind assigning
different dtypes and compute kernel configs to each projection, specifies the TTNN
configuration for each, and provides validation criteria and code patterns for the
Qwen 235B-A22B use case.

## Learning Objectives

After completing this chapter, you will be able to:

1. Explain why gate and up projections tolerate `bfloat4_b` while the down projection
   requires `bfloat8_b`, using the SwiGLU data flow and residual stream semantics as the
   mechanistic basis.
2. Configure `WormholeComputeKernelConfig` correctly for each projection: LoFi for gate
   and up, HiFi2 for down — including the correct `fp32_dest_acc_en` value for each.
3. Organize mixed-precision expert weight tensors in DRAM with the correct shapes, dtypes,
   and memory configs.
4. Calculate the total DRAM footprint for Qwen 235B-A22B expert weights under mixed
   precision and compare it against the bfloat16 baseline.
5. Execute the step-by-step weight conversion procedure starting from a bfloat16 Qwen
   checkpoint, including per-layer PCC validation.

## Prerequisites

- Chapter 1, `bfloat4_b_format.md` and `bfloat8_b_format.md`: block floating-point
  format definitions, memory footprint, and compute throughput multipliers.
- Chapter 2, `compute_kernel_config.md` and `weight_conversion.md`: `WormholeComputeKernelConfig`
  fields, LoFi and HiFi2 config construction, and `ttnn.as_tensor` dtype conversion.
- Chapter 3, `projection_sensitivity.md`: empirical sensitivity ordering (down > gate > up)
  and the DeepSeek-V3 validation evidence.
- Chapter 4, `bandwidth_vs_accuracy_tradeoff.md`: Pareto-optimal configurations per
  projection type.

## Reference Table

The following table summarises the recommended configuration for each MoE expert
projection. The rationale column provides the compressed reason; each file in this
chapter elaborates the full mechanistic argument.

| Projection | Weight name | Dtype | MathFidelity | `fp32_dest_acc_en` | `packer_l1_acc` | Rationale |
|---|---|---|---|---|---|---|
| Gate | w1 | `ttnn.bfloat4_b` | LoFi | False | True | SiLU clips quantization noise; highest throughput needed for gate/up pair |
| Up | w3 | `ttnn.bfloat4_b` | LoFi | False | True | SwiGLU product dilutes uncorrelated errors; symmetric treatment with gate |
| Down | w2 | `ttnn.bfloat8_b` | HiFi2 | False | True | Output feeds residual stream directly; HiFi2 reduces accumulation rounding on top of weight quantization |

> **Tip:** Both LoFi and HiFi2 configs use `fp32_dest_acc_en=False`. Do not set
> `fp32_dest_acc_en=True` for either config — the authoritative kernel configurations
> for MoE expert projections use `False` in both cases.

This configuration is directly derived from the Pareto frontier documented in Chapter 4,
`bandwidth_vs_accuracy_tradeoff.md`, and the sensitivity ordering established in Chapter 3,
`projection_sensitivity.md`.

## Files in This Chapter

| File | Contents |
|---|---|
| [`gate_and_up_projection_strategy.md`](./gate_and_up_projection_strategy.md) | Mechanistic explanation for bfloat4_b + LoFi on gate and up; SiLU error compression; element-wise product dilution; validation criterion; code pattern |
| [`down_projection_strategy.md`](./down_projection_strategy.md) | Why bfloat8_b + HiFi2 is required for down projection; residual stream sensitivity; accumulation fidelity argument; validation criterion and fallback options |
| [`mixed_precision_memory_layout.md`](./mixed_precision_memory_layout.md) | DRAM layout for mixed-precision expert weight tensors; footprint calculation for Qwen 235B-A22B on a T3K system; tile alignment constraints |
| [`qwen_adaptation_guide.md`](./qwen_adaptation_guide.md) | Step-by-step weight conversion from Qwen 235B-A22B bfloat16 checkpoints; per-layer PCC validation; checkpoint caching |

## Next Steps

Begin with `gate_and_up_projection_strategy.md` to understand why the two projections
that feed into the SwiGLU multiplication path can safely use the most aggressive
quantization level available.
