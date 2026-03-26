# Chapter 7 — Profiling Methodology and Optimization Roadmap

## Purpose

This chapter translates the analysis from Chapters 1–6 into two concrete deliverables:

1. **Profiling methodology** (`profiling_methodology.md`): how to collect and interpret op-level latency data for `_forward_decode_paged` on T3K, with specific instrumentation targets identified in prior chapters.
2. **Optimization roadmap** (`bottleneck_ranking.md`): a prioritized list of the bottlenecks identified in Chapters 2–6, with estimated impact magnitude, root cause file and line references, and a cross-model comparison.
3. **Cross-model comparison** (`comparison_to_other_implementations.md`): side-by-side summary of `TTNNBailingMoEAttention`, `TTNNQwen3FullAttention`, and `TTNNGlm4MoeLiteAttention` on the key performance dimensions.

---

## Prerequisites

- A working `TTNNBailingMoEAttention._forward_decode_paged` running on a T3K 1×8 mesh
- A decode batch of at least B=1 (the minimum for single-step latency measurement)
- `ttnn` installed with profiling support (the `ttnn.synchronize_device` call is available in standard builds)

---

## Relationship to Prior Chapters

| Chapter | Bottleneck identified | Where to find details |
|---|---|---|
| 2 | Reduce-scatter in Q projection (unnecessary) | `chapter_02_collective_communication/sharding_alternatives.md` |
| 3 | 9 `to_memory_config` transitions per step | `chapter_03_memory_layout_transitions/avoidable_transitions.md` |
| 4 | `cur_pos_tt` `from_torch` each decode step | `chapter_04_host_device_roundtrips/cur_pos_roundtrip.md` |
| 4 | `get_cos_sin_for_decode` partial host op | `chapter_04_host_device_roundtrips/get_cos_sin_host_ops.md` |
| 5 | QK norm DRAM→L1 transition before `_apply_qk_norm` | `chapter_05_qk_normalization/distributed_alternative.md` |
| 6 | `fp32_dest_acc_en=True` limits dst tiles; HiFi2 candidate | `chapter_06_math_fidelity/accuracy_throughput_tradeoff.md` |

---

**Next:** [Profiling Methodology](profiling_methodology.md)
