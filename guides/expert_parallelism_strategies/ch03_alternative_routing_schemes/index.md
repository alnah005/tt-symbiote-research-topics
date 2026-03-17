# Chapter 3 — Alternative Expert Routing Schemes

## Overview

Chapters 1 and 2 established all-to-all collective communication as the canonical mechanism for MoE expert dispatch and combine. Chapter 2, `dispatch_combine_overhead.md`, quantified the cost and showed that on T3K (Tenstorrent 8-device mesh, 1×8 linear layout, `ttnn.Topology.Linear`), the two all-to-all collectives dominate MoE layer latency across all practical batch sizes when $k = N = 8$, with the communication-to-compute ratio approximately $9.79 \times 10^4 / D$ [D UNVERIFIED].

This chapter frames all-to-all as the **baseline** and characterizes three alternative routing architectures: expert sharding via all-gather, pipeline expert parallelism, and hierarchical two-level routing. For each scheme, it derives communication volume, identifies load-balance implications, and states when the scheme beats the baseline. The chapter concludes with a side-by-side comparison matrix and a decision procedure for choosing a scheme given hardware and workload parameters.

**Prerequisite:** Chapter 1 (`ch01_moe_fundamentals/`) in its entirety, and Chapter 2 (`ch02_all_to_all_primitives/`) in its entirety. Readers must be familiar with the MoE forward pass, dispatch/combine communication volumes, and the Qwen3.5-35B configuration ($E = 256$, $k = 8$, $N = 8$, $H = 7{,}168$, $E_d = 32$).

---

## What This Chapter Covers

| File | Topic | First used in later chapters |
|---|---|---|
| `expert_sharding.md` | Static assignment of experts to devices; all-gather-before-routing pattern; communication volume comparison between all-gather and all-to-all; when expert sharding wins | Chapter 4, `uniform_partitioning.md`; Chapter 8, `recommended_configuration.md` |
| `pipeline_expert_parallelism.md` | Pipeline arrangement of devices for sequential expert execution; bubble analysis; comparison with all-to-all latency; applicability to Qwen3.5-35B | Chapter 6, `pipeline_design.md` |
| `hierarchical_routing.md` | Two-level coarse/fine routing; communication reduction from group confinement; load-balance challenges at inference time; training dependency | Chapter 4, `load_aware_assignment.md`; Chapter 7, `auxiliary_loss_free_inference.md` |
| `scheme_comparison_matrix.md` | Summary table of all four schemes; ASCII decision flowchart; recommendation for Qwen3.5-35B on T3K | Chapter 8, `recommended_configuration.md` |

---

## New Notation Introduced in This Chapter

The following symbols extend the notation established in Chapters 1 and 2. Symbols inherited from earlier chapters are repeated for quick reference but are not redefined here.

| Symbol | Meaning | First appearance |
|---|---|---|
| $V_\text{gather}$ | Communication volume (bytes) of one all-gather round over $N-1$ hops | `expert_sharding.md` |
| $V_{a2a}$ | Communication volume (bytes) of one all-to-all dispatch or combine | `expert_sharding.md` |
| $G$ | Number of device groups in a hierarchical routing configuration | `hierarchical_routing.md` |
| $k_c$ | Coarse top-$k$: number of device groups selected by the coarse router per token | `hierarchical_routing.md` |
| $k_f$ | Fine top-$k$: number of experts selected within each selected group by the fine router per token | `hierarchical_routing.md` |
| $P$ | Pipeline depth: number of sequential expert stages a token traverses | `pipeline_expert_parallelism.md` |
| $M$ | Micro-batch size (tokens per micro-batch) in pipeline execution | `pipeline_expert_parallelism.md` |
| $\eta$ | Pipeline efficiency: fraction of time stages are active (no bubble) | `pipeline_expert_parallelism.md` |
| $L$ | Load imbalance factor: ratio of peak expert token count to average | inherited from Chapter 1 |
| $E_d$ | Experts per device under uniform assignment: $E_d = E/N = 32$ | inherited from Chapter 1 |
| $\text{BW}$ | Per-link Ethernet bandwidth: $\approx 12.5\,\text{GB/s}$ on T3K | inherited from Chapter 2 |
| $\alpha$ | Per-message network latency (seconds) | inherited from Chapter 2 |
| $\beta$ | Per-byte transmission time: $1/\text{BW}$ | inherited from Chapter 2 |

---

## Reading Order

1. **`expert_sharding.md`** — The simplest alternative to all-to-all. Read this first because it introduces the all-gather volume formula that reappears in the comparison matrix.

2. **`pipeline_expert_parallelism.md`** — A fundamentally different execution model. Read after `expert_sharding.md` because the pipeline discussion references all-to-all and all-gather as counterpoints.

3. **`hierarchical_routing.md`** — The most complex alternative; requires understanding flat routing (Chapter 1) and all-to-all (Chapter 2) as contrasts.

4. **`scheme_comparison_matrix.md`** — Synthesis file. Read last; it references all prior files in this chapter.

---

## Cross-Chapter Dependency Note

Results from this chapter feed directly into three later chapters:

- **Chapter 4, `ch04_expert_device_assignment/`:** The expert sharding analysis in `expert_sharding.md` motivates the uniform 32-experts-per-device assignment examined in Chapter 4. The hierarchical routing discussion in `hierarchical_routing.md` informs load-aware assignment strategies.

- **Chapter 6, `ch06_fused_dispatch_compute_combine/`:** The pipeline scheme described in `pipeline_expert_parallelism.md` is the conceptual precursor to Chapter 6's fused dispatch-compute-combine pipeline. Chapter 6, `pipeline_design.md` builds on but is distinct from the pipeline expert parallelism scheme here — Chapter 6 pipelines *micro-batches through a fixed all-to-all execution*, whereas this chapter surveys a scheme where *tokens pipeline through expert stages sequentially*.

- **Chapter 8, `ch08_qwen35b_t3k_strategy/`:** The scheme comparison matrix in `scheme_comparison_matrix.md` and the Qwen3.5-35B recommendation here feed directly into Chapter 8, `recommended_configuration.md`, which cites both the matrix and the crossover analysis.

The formal crossover derivation between communication-dominated and compute-dominated regimes is deferred to Chapter 6, `ch06_fused_dispatch_compute_combine/end_to_end_latency_model.md`. This chapter provides qualitative estimates and first-order symbolic formulas; readers requiring precise quantitative thresholds should consult Chapter 6.

---

## Relationship to Chapter 2 Baseline

Chapter 2 established the all-to-all baseline with the following key result:

$$V_{a2a} \approx \frac{N-1}{N} \times B \times k \times H \times 2 \text{ bytes per collective per device}$$

For Qwen3.5-35B on T3K ($N = k = 8$, $H = 7{,}168$, BF16):

$$V_{a2a} = \frac{7}{8} \times B \times 8 \times 7{,}168 \times 2 = B \times 7 \times 7{,}168 \times 2 = 100{,}352 \times B \text{ bytes}$$

**Arithmetic check:** $7 \times 7{,}168 \times 2 = 7 \times 14{,}336 = 100{,}352$ bytes per token. For $B = 32$: $100{,}352 \times 32 = 3{,}211{,}264$ bytes $\approx 3.06$ MiB. This matches the figure in Chapter 2, `dispatch_combine_overhead.md`.

All alternative schemes in this chapter are measured against this baseline.

---

## References

- [Ch1Index] Chapter 1, `ch01_moe_fundamentals/index.md` — notation, model parameters, routing problem overview.
- [Ch1Routing] Chapter 1, `ch01_moe_fundamentals/routing_problem.md` — dispatch/combine pattern, load imbalance.
- [Ch1Config] Chapter 1, `ch01_moe_fundamentals/qwen35b_config.md` — Qwen3.5-35B architectural constants.
- [Ch2Background] Chapter 2, `ch02_all_to_all_primitives/collective_communication_background.md` — all-to-all definition and bandwidth/latency model.
- [Ch2Overhead] Chapter 2, `ch02_all_to_all_primitives/dispatch_combine_overhead.md` — all-to-all volume formula and $D^*$ crossover result.
- [Ch4Assignment] Chapter 4, `ch04_expert_device_assignment/index.md` — expert-to-device assignment strategies.
- [Ch6Pipeline] Chapter 6, `ch06_fused_dispatch_compute_combine/pipeline_design.md` — micro-batch pipelining of dispatch/combine collectives.
- [Ch6Latency] Chapter 6, `ch06_fused_dispatch_compute_combine/end_to_end_latency_model.md` — formal crossover derivation.
- [Ch8Synthesis] Chapter 8, `ch08_qwen35b_t3k_strategy/recommended_configuration.md` — final configuration recommendations.
- [PlanDoc] Expert Parallelism Strategies — Research Guide Plan, `guides/expert_parallelism_strategies/plan.md`.
