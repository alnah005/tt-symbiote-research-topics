# Guide: Compute Kernel Configuration for MoE on Wormhole B0

## What This Guide Teaches

This guide explains how to configure `ttnn.WormholeComputeKernelConfig` for Mixture-of-Experts (MoE) expert matmuls on Tenstorrent Wormhole B0 hardware. It covers what each field controls at the hardware level, why different MoE projection types require different configurations, and how to apply and validate a two-config pattern that measurably reduces decode-mode latency.

The motivation is concrete: DeepSeek-V3 uses `COMPUTE_KERNEL_CONFIG_LOFI` (with `packer_l1_acc=True`) for gate and up projections and `COMPUTE_KERNEL_CONFIG_HIFI2` for down projections. Qwen MoE implementations in tt-transformers do not specify any `compute_kernel_config`, defaulting to TTNN-chosen behavior that pays unnecessary DRAM bandwidth costs during accumulation. This guide explains the DeepSeek choices, characterizes the accuracy and throughput trade-offs, and provides a path to applying the same pattern to Qwen MoE expert matmuls.

---

## Audience

**You should read this guide if you are:**

- An ML or performance engineer optimizing MoE models (DeepSeek-V3, Qwen, or similar) on Wormhole B0 using TTNN
- Comfortable with `ttnn.matmul`, memory configs, and program configs, but have not previously tuned `compute_kernel_config`
- Debugging an accuracy regression after changing a kernel config, or trying to understand why two configs produce different PCC values

**You do not need to know in advance:**

- The internal Tensix FPU pipeline stages or how math fidelity is implemented in microcode
- How the Wormhole packer/unpacker pipeline differs from GPU tensor core pipelines
- Formal numerical analysis of floating-point rounding in reduced-precision multiply-accumulate

**After completing this guide you will be able to:**

- Set `math_fidelity`, `packer_l1_acc`, `fp32_dest_acc_en`, and `math_approx_mode` for each expert projection type with a documented rationale
- Predict the PCC and latency effect of each field change for decode-mode MoE workloads
- Apply the two-config pattern to a new MoE model and validate it via PCC sweep and per-op profiling
- Identify L1 budget constraints introduced by `packer_l1_acc=True` and resolve them

---

## Key Canonical Configs

These two configs, taken directly from DeepSeek-V3, are the primary reference point for the entire guide.

```python
import ttnn

# LOFI: used for gate (w1) and up (w3) projections
COMPUTE_KERNEL_CONFIG_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,   # no transcendental ops in matmul; set conservatively
    fp32_dest_acc_en=False,
    packer_l1_acc=True,       # eliminates DRAM round-trips during K-loop accumulation
)

# HIFI2: used for down (w2) projections
COMPUTE_KERNEL_CONFIG_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,    # acceptable for SFPU ops; irrelevant for pure matmul
    fp32_dest_acc_en=False,
    packer_l1_acc=True,       # same bandwidth benefit applies regardless of fidelity
)
```

**LOFI** is appropriate for gate and up projections because their outputs feed into SiLU/GELU activations that absorb small mantissa rounding errors. The `packer_l1_acc=True` flag eliminates per-iteration DRAM write-backs during K-loop accumulation — the primary bandwidth saving for decode-mode workloads.

**HIFI2** is required for the down projection because it accumulates directly into the residual stream, where rounding errors compound across transformer layers. Gate/up can tolerate LoFi rounding; down cannot. `packer_l1_acc=True` still applies because the bandwidth benefit is independent of fidelity level.

> **Warning:** Omitting `compute_kernel_config` entirely does not default to LOFI with `packer_l1_acc=True`. The TTNN device default uses `packer_l1_acc=False`, which pays DRAM read-modify-write costs for every K-loop iteration. For bandwidth-bound decode-mode expert matmuls this is a measurable latency penalty.

---

## Chapter Navigation

| Chapter | Description | Key Files |
|---|---|---|
| [Ch 1 — Compute Kernel Config Fundamentals](ch1_kernel_config_fundamentals/index.md) | Introduces `WormholeComputeKernelConfig` as the per-op compute handle on Wormhole B0; explains all four primary fields and what TTNN does when the config is omitted | [`wormhole_compute_kernel_config_api.md`](ch1_kernel_config_fundamentals/wormhole_compute_kernel_config_api.md), [`fp32_dest_acc_en.md`](ch1_kernel_config_fundamentals/fp32_dest_acc_en.md), [`math_fidelity_overview.md`](ch1_kernel_config_fundamentals/math_fidelity_overview.md) |
| [Ch 2 — Math Fidelity Levels — LoFi vs HiFi2 vs HiFi4](ch2_math_fidelity_levels/index.md) | Characterizes the four `MathFidelity` enum values: cycles per tile, effective mantissa bits, PCC impact on MoE expert matmul outputs, and the fidelity selection workflow | [`fidelity_precision_model.md`](ch2_math_fidelity_levels/fidelity_precision_model.md), [`fidelity_and_moe_accuracy.md`](ch2_math_fidelity_levels/fidelity_and_moe_accuracy.md), [`fidelity_selection_workflow.md`](ch2_math_fidelity_levels/fidelity_selection_workflow.md) |
| [Ch 3 — `packer_l1_acc` — Throughput Effect](ch3_packer_l1_acc/index.md) | Explains the Tensix packer pipeline stage and how `packer_l1_acc=True` eliminates DRAM round-trips during K-loop accumulation; quantifies the bandwidth reduction for decode-mode MoE | [`tensix_packer_pipeline.md`](ch3_packer_l1_acc/tensix_packer_pipeline.md), [`throughput_impact.md`](ch3_packer_l1_acc/throughput_impact.md), [`packer_l1_acc_constraints.md`](ch3_packer_l1_acc/packer_l1_acc_constraints.md) |
| [Ch 4 — `math_approx_mode` — Accuracy Trade-offs](ch4_math_approx_mode/index.md) | Covers the SFPU approximation flag; identifies which ops are affected (exp, reciprocal, sqrt, sigmoid, gelu, silu) and which are not (matmul FPU path); characterizes risk for MoE projections | [`sfpu_approx_operations.md`](ch4_math_approx_mode/sfpu_approx_operations.md), [`approx_mode_accuracy_risks.md`](ch4_math_approx_mode/approx_mode_accuracy_risks.md), [`approx_mode_for_moe.md`](ch4_math_approx_mode/approx_mode_for_moe.md) |
| [Ch 5 — MoE Expert Matmul Configuration](ch5_moe_expert_matmul_config/index.md) | Applies the parameter knowledge from Chapters 1–4 to the DeepSeek-V3 and Qwen MoE gate/up/down projections; shows the current Qwen gap and how to close it | [`deepseek_v3_config_analysis.md`](ch5_moe_expert_matmul_config/deepseek_v3_config_analysis.md), [`qwen_moe_current_state.md`](ch5_moe_expert_matmul_config/qwen_moe_current_state.md), [`applying_configs_to_qwen.md`](ch5_moe_expert_matmul_config/applying_configs_to_qwen.md) |
| [Ch 6 — Performance Benchmarking and Config Selection](ch6_benchmarking_and_selection/index.md) | Provides a systematic benchmarking methodology and a structured decision matrix for selecting configs in production deployments; includes a pre-deployment checklist | [`benchmarking_methodology.md`](ch6_benchmarking_and_selection/benchmarking_methodology.md), [`config_decision_matrix.md`](ch6_benchmarking_and_selection/config_decision_matrix.md), [`production_config_checklist.md`](ch6_benchmarking_and_selection/production_config_checklist.md) |

---

## Quick-Start Paths

Choose the path that matches your immediate goal. Each path lists the minimum reading to accomplish that goal; reading the full guide in order is still recommended for a complete understanding.

### New to the API — understand what `WormholeComputeKernelConfig` does

1. `ch1_kernel_config_fundamentals/index.md` — field overview and learning objectives
2. `ch1_kernel_config_fundamentals/wormhole_compute_kernel_config_api.md` — construction, defaults, and the two canonical configs in Python

### Debugging an accuracy regression — PCC dropped after changing a kernel config

1. `ch2_math_fidelity_levels/fidelity_and_moe_accuracy.md` — why different projections need different fidelity levels
2. `ch2_math_fidelity_levels/fidelity_selection_workflow.md` — parameterized PCC sweep to find the lowest safe fidelity
3. `ch4_math_approx_mode/approx_mode_accuracy_risks.md` — when approximation mode is a risk vs. irrelevant

### Optimizing for decode — reduce expert matmul latency at small batch sizes

1. `ch3_packer_l1_acc/index.md` — why `packer_l1_acc` is the highest-leverage single-field change for decode
2. `ch3_packer_l1_acc/throughput_impact.md` — quantitative bandwidth reduction and regime analysis
3. `ch3_packer_l1_acc/packer_l1_acc_constraints.md` — L1 budget constraints before enabling

### Applying configs to Qwen MoE — close the performance gap

1. `ch5_moe_expert_matmul_config/qwen_moe_current_state.md` — the current gap and its cost
2. `ch5_moe_expert_matmul_config/applying_configs_to_qwen.md` — step-by-step code change, PCC validation, and latency measurement
3. `ch6_benchmarking_and_selection/production_config_checklist.md` — pre-deployment checklist before merging

### DeepSeek or Qwen reference — understand why the production configs are set as they are

1. `ch5_moe_expert_matmul_config/deepseek_v3_config_analysis.md` — exact parameter values, projection-level assignment, rationale
2. `ch5_moe_expert_matmul_config/qwen_moe_current_state.md` — how Qwen differs from the DeepSeek pattern today

---

## Cross-Chapter Dependencies

The chapters are designed to be read in order. Each chapter builds on specific concepts from earlier ones. Reading a later chapter without the prerequisite concepts will leave the rationale unclear.

| Chapter | Depends on |
|---|---|
| Ch 1: Compute Kernel Config Fundamentals | None — introduces all four key fields from scratch |
| Ch 2: Math Fidelity Levels | Ch 1 — requires the `math_fidelity` field overview and FPU pipeline concept from `math_fidelity_overview.md` |
| Ch 3: `packer_l1_acc` Throughput Effect | Ch 1 (packer pipeline and `fp32_dest_acc_en` interaction), Ch 2 (bandwidth-bound vs. compute-bound framing) |
| Ch 4: `math_approx_mode` Accuracy Trade-offs | Ch 1 (`math_approx_mode` field introduced), Ch 2 (fidelity vs. approx mode distinction) |
| Ch 5: MoE Expert Matmul Configuration | Ch 1–4 all required — synthesizes all four field semantics into the two-config pattern |
| Ch 6: Performance Benchmarking and Config Selection | Ch 1–5 all required — uses DeepSeek and Qwen configs from Ch 5 as the concrete baseline |

**Specific forward references to be aware of:**

- Ch 1 (`math_fidelity_overview.md`) names the LoFi/HiFi2/HiFi4 levels but defers the throughput multiplier table and PCC data to Ch 2. Chapter 5 references specific PCC thresholds that are defined in Ch 2.
- Ch 1 (`fp32_dest_acc_en.md`) notes the interaction with `packer_l1_acc` and L1 space but defers the full L1 footprint analysis to Ch 3 (`packer_l1_acc_constraints.md`).
- Ch 3 (`throughput_impact.md`) establishes the bandwidth saving formula. Both Ch 5 and Ch 6 reference this formula without re-deriving it.
- Ch 4 (`approx_mode_for_moe.md`) gives the per-projection recommendation table for `math_approx_mode`. Chapter 5 (`applying_configs_to_qwen.md`) references those recommendations by name.

---

## Conventions Used in This Guide

- TTNN config fields are written in their exact Python attribute names: `math_fidelity`, `packer_l1_acc`, `fp32_dest_acc_en`, `math_approx_mode`.
- Tensor shapes are written as `[M, K]` x `[K, N]` with named dimensions where relevant (e.g., `[batch * seq, d_model]` x `[d_model, d_ff]`).
- Configuration constants are written in `SCREAMING_SNAKE_CASE` as they appear in the DeepSeek codebase: `COMPUTE_KERNEL_CONFIG_LOFI`, `COMPUTE_KERNEL_CONFIG_HIFI2`.
- DeepSeek-V3 source paths use repo-relative paths prefixed with the repository root: e.g., `models/demos/deepseek_v3/utils/config_helpers.py`.
- PCC values are scalars between 0 and 1. A value of 0.9995 means 99.95% Pearson correlation with the reference output. The standard tt-metal CI threshold for bfloat16 MoE is typically 0.999.
- All performance and hardware details in this guide refer to Wormhole B0. Blackhole and Grayskull differ and are not covered.

---

## Where to Start

If you are reading this guide for the first time, begin at [ch1_kernel_config_fundamentals/index.md](ch1_kernel_config_fundamentals/index.md).

If you already know the API and want the practical application directly, jump to [ch5_moe_expert_matmul_config/index.md](ch5_moe_expert_matmul_config/index.md).
