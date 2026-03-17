# Chapter 7: Implementation and Validation Workflow

## Overview

This chapter provides an end-to-end guide for implementing mixed-precision expert weight
quantization in a TTNN Mixture of Experts (MoE) model. The previous six chapters
established the conceptual and analytical foundations: format properties, the TTNN
quantization API, memory footprint mechanics, arithmetic intensity analysis, per-projection
strategy design, and the comparative decision framework. Chapter 7 puts those foundations
into practice.

The chapter is organized as a five-step workflow, each step covered in a dedicated file.
Each step produces a concrete artifact — a measured baseline, a converted weight set, a
per-layer PCC (Pearson Cross-Correlation) report, a throughput profile, or a locked-in
configuration — that feeds into the next step.

## Learning Objectives

By the end of this chapter you will be able to:

1. Establish a numerically correct bfloat16 baseline against a CPU reference and measure
   PCC > 0.999.
2. Write a weight conversion script that iterates all MoE expert projections, applies the
   appropriate dtype and tile layout, and verifies conversion fidelity.
3. Build a per-layer PCC test harness that catches quantization accuracy regressions
   before they reach production.
4. Use TTNN's device profiler and Tracy traces to measure per-operation latency and
   identify throughput bottlenecks in the expert FFN (feed-forward network).
5. Follow a structured decision tree to tune compute kernel configurations, run calibration
   perplexity, and lock in a reproducible final configuration.

## The Five-Step Workflow

| Step | File | Produces |
|---|---|---|
| 1 — Establish bfloat16 baseline | `baseline_and_weight_conversion.md` | Baseline PCC measurement (target > 0.999 vs CPU) |
| 2 — Convert weights to target dtypes | `baseline_and_weight_conversion.md` | Converted weight set: bfloat4_b gate/up, bfloat8_b down |
| 3 — Validate per-layer PCC | `per_layer_pcc_validation.md` | PCC report: gate/up ≥ 0.96, down ≥ 0.975, full layer ≥ 0.97 |
| 4 — Profile throughput | `throughput_profiling.md` | Per-op latency breakdown; decode vs. prefill comparison |
| 5 — Tune and lock in final config | `iterative_tuning_guide.md` | Locked configuration; regression test suite |

Steps 1 and 2 are covered together in `baseline_and_weight_conversion.md` because the
baseline forward pass and the weight conversion share the same device setup and validation
pattern.

## Reference Configuration (Qwen 235B-A22B)

All code examples in this chapter use the Qwen 235B-A22B dimensions as the concrete
reference case. The mixed-precision target for gate and up projections is `bfloat4_b` with
LoFi compute kernel config; for the down projection, `bfloat8_b` with HiFi2.

Key model dimensions used throughout this chapter:

| Dimension | Value |
|---|---|
| `d_model` | 7168 |
| `d_ff` | 2048 |
| `num_experts` | 128 |
| `top_k` | 8 |

The LoFi and HiFi2 compute kernel configs referenced in all examples are defined as:

```python
import ttnn

# LoFi — gate and up projections (bfloat4_b weights)
COMPUTE_KERNEL_CONFIG_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,  # False for both LoFi and HiFi2
    packer_l1_acc=True,
)

# HiFi2 — down projection (bfloat8_b weights)
COMPUTE_KERNEL_CONFIG_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=False,  # False for both LoFi and HiFi2
    packer_l1_acc=True,
)
```

> **Warning:** `fp32_dest_acc_en=False` for both LoFi and HiFi2. Setting it to True
> changes the accumulation path and invalidates the PCC thresholds established in this
> chapter.

## Prerequisites

This chapter assumes knowledge of all previous chapters:

- **Chapter 1** (`ch01_quantization_formats/`) — bfloat16, bfloat8_b, bfloat4_b format
  definitions, tile layout constraint, size formulas.
- **Chapter 2** (`ch02_ttnn_quantization_api/`) — `ttnn.as_tensor`, `ttnn.linear`,
  `WormholeComputeKernelConfig`, `comp_pcc` usage patterns.
- **Chapter 3** (`ch03_memory_footprint/`) — per-expert memory calculations, DRAM
  footprint at system scale.
- **Chapter 4** (`ch04_arithmetic_intensity/`) — arithmetic intensity analysis, decode vs.
  prefill regime distinction, crossover batch size.
- **Chapter 5** (`ch05_per_projection_strategy/`) — per-projection dtype assignment
  rationale; Qwen adaptation guide with full conversion and validation code.
- **Chapter 6** (`ch06_comparative_study/`) — decision framework for choosing accuracy
  budget tier, deployment regime, and training history considerations.

## Navigation

- [`baseline_and_weight_conversion.md`](./baseline_and_weight_conversion.md) — Steps 1 and 2: baseline measurement and weight
  conversion with PCC verification.
- [`per_layer_pcc_validation.md`](./per_layer_pcc_validation.md) — Step 3: per-layer PCC test harness and diagnostic
  procedures.
- [`throughput_profiling.md`](./throughput_profiling.md) — Step 4: device profiler usage and expert FFN latency
  breakdown.
- [`iterative_tuning_guide.md`](./iterative_tuning_guide.md) — Step 5: tuning decision tree, calibration perplexity,
  and regression testing.

## Next Steps

Proceed to `baseline_and_weight_conversion.md` to establish the bfloat16 baseline and
run the weight conversion script.
