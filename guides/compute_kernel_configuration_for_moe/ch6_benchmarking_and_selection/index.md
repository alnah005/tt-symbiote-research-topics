# Chapter 6: Performance Benchmarking and Config Selection

## Overview

Chapters 1–5 built the conceptual and implementation foundation: what each field in `WormholeComputeKernelConfig` controls, how LoFi/HiFi2/HiFi4 trade precision for throughput, why `packer_l1_acc=True` eliminates DRAM round-trips, and how DeepSeek-V3 and Qwen MoE map their projection types to two canonical configs. This chapter converts that knowledge into a repeatable workflow.

The goal is practical: given a new MoE model or a new deployment regime, how do you arrive at the right `WormholeComputeKernelConfig` for each projection type, verify that it is correct, and ship it with confidence? This chapter provides a benchmarking methodology, a structured decision matrix, and a pre-deployment checklist.

---

## Learning Objectives

By the end of this chapter you should be able to:

1. Construct a standalone, reproducible benchmark that isolates the latency contribution of a single MoE expert matmul using `ttnn.device.profiler` or Tracy.
2. Sweep `math_fidelity`, `packer_l1_acc`, and `fp32_dest_acc_en` systematically and record both latency and PCC against a PyTorch float32 reference.
3. Apply the config decision matrix to a new model's gate, up, and down projections without having to re-derive the rationale from first principles.
4. Identify when the canonical two-config pattern needs to be adjusted — for high-PCC requirements, large d_ff/d_model ratios, or high top-K routing depth.
5. Complete the pre-deployment checklist before merging a compute kernel config change into production.

---

## Decision Flowchart

The flowchart below covers the common case: a SwiGLU/SiLU MoE model on Wormhole B0 with bfloat16 activations and weights.

```
Start: New MoE projection to configure
          |
          v
    What projection type?
    /          |          \
gate/up       down        other (attn, norm)
   |           |               |
   v           v          Not covered here;
LOFI config  What PCC       see model-level
(see matrix) threshold?     config docs
               |
        +----- +-----+
        |             |
    <= 0.9995      > 0.9995
        |             |
      HIFI2        HIFI4 +
    (see matrix)  fp32_dest_acc_en=True
        |             |
        v             v
   What regime?   Benchmark to
   /         \    verify; see
decode    prefill  benchmarking_
(M<=32)  (seq>=512) methodology.md
   |         |
   v         v
HIFI2:    Benchmark
packer_   LoFi vs HiFi2;
l1_acc=T  accept lowest
          fidelity above
          PCC threshold
```

In all branches, `packer_l1_acc=True` is the default. The only reason to deviate is an L1 allocation error at dispatch time, in which case reduce `per_core_N` or `out_subblock_w` before disabling `packer_l1_acc`.

---

## Prerequisites

- **Chapter 1** — `WormholeComputeKernelConfig` field definitions and the FPU/packer pipeline model.
- **Chapter 2** — Math fidelity throughput multipliers and PCC characterization for MoE projections.
- **Chapter 3** — `packer_l1_acc` bandwidth reduction formula and L1 overflow constraints.
- **Chapter 4** — `math_approx_mode` scope: SFPU-only, no effect on pure matmul FPU path.
- **Chapter 5** — The two-config pattern (LOFI for gate/up, HIFI2 for down) as the starting baseline.

---

## Files in This Chapter

| File | Contents |
|---|---|
| [benchmarking_methodology.md](benchmarking_methodology.md) | Standalone benchmark construction, profiling with `ttnn.device.profiler`, sweep dimensions, `packer_l1_acc` isolation test, PCC quantification |
| [config_decision_matrix.md](config_decision_matrix.md) | Structured decision rules by projection type and regime; d_ff/d_model ratio effect; top-K routing interaction |
| [production_config_checklist.md](production_config_checklist.md) | Pre-deployment checklist; heterogeneous expert handling; firmware version pinning; common L1 budget mistake |

Read in order. `config_decision_matrix.md` assumes familiarity with the benchmarking concepts in `benchmarking_methodology.md`. `production_config_checklist.md` assumes both.

---

## Next Steps

Begin with [benchmarking_methodology.md](benchmarking_methodology.md) to learn how to measure what you care about before consulting the decision matrix.
