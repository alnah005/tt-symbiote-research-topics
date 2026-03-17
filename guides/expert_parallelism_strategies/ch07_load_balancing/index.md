# Chapter 7 — Load Balancing for MoE Inference

## Overview

Chapters 1–6 established the Mixture-of-Experts (MoE) architecture, the all-to-all dispatch/combine communication primitives, alternative routing schemes, expert-to-device assignment strategies, routing weight optimization, and the fused dispatch-compute-combine pipeline. This chapter addresses the runtime consequence of non-uniform routing: **load imbalance**. When some experts receive far more tokens per step than others, those experts become bottlenecks, capacity overflows cause token drops, and output quality degrades. This chapter explains how to detect imbalance, how to handle overflow when it occurs, and how to apply dynamic mitigation strategies at inference time.

**Prerequisites:** All of Chapters 1 through 6 in their entirety. Readers must be familiar with:

- The MoE forward pass, routing probabilities $p_e$, expert capacity $C$, and the dispatch/combine pattern (Chapter 1, `ch01_moe_fundamentals/`)
- The all-to-all dispatch and combine primitives and their buffer layouts (Chapter 2, `ch02_all_to_all_primitives/`)
- Alternative routing schemes and their load-balance trade-offs (Chapter 3, `ch03_alternative_routing_schemes/`)
- Expert-to-device assignment, load imbalance metric $L$, and expert replication factors $r_e$ (Chapter 4, `ch04_expert_device_assignment/`)
- Routing weight optimization and softmax/sigmoid temperature effects (Chapter 5, `ch05_routing_weight_optimization/`)
- The fused dispatch-compute-combine pipeline assumptions (Chapter 6, `ch06_fused_dispatch_compute_combine/`)

---

## Goal Statement

Given:
- $E = 256$ total experts,
- $k = 8$ top-$k$ routing per token,
- $N = 8$ devices (T3K 1×8 linear mesh),
- hidden dimension $H = 7168$,
- experts per device $E_d = E / N = 32$,
- capacity factor $CF = 1.25$,
- batch size $B$ (number of tokens per forward pass),

understand:

1. How to quantify load imbalance in terms of the per-expert routing frequency $f_e$ and the Coefficient of Variation $CV$.
2. When and how token dropping occurs (expert capacity overflow) and its impact on output quality.
3. Which dynamic strategies — load-aware routing adjustment, temperature scaling, expert replication, and auxiliary loss — mitigate imbalance with acceptable runtime overhead.

---

## Quick-Reference Constants

The following constants apply throughout this chapter and are exact for Qwen3.5-35B on T3K.

| Symbol | Description | Value |
|---|---|---|
| $E$ | Total number of experts | 256 |
| $k$ | Top-$k$ experts selected per token | 8 |
| $N$ | Number of devices | 8 |
| $H$ | Hidden dimension | 7168 |
| $E_d$ | Experts per device, $E_d = E/N$ | 32 |
| $CF$ | Capacity factor | 1.25 |
| $f_{\text{avg}}$ | Average routing frequency per expert, $f_{\text{avg}} = k/E$ | $1/32 = 0.03125$ |
| $C$ | Expert capacity: $C = \lceil k \cdot B \cdot CF / E \rceil = \lceil 8 \cdot B \cdot 1.25 / 256 \rceil = \lceil B / 25.6 \rceil$ | derived; $B{=}1 \Rightarrow C{=}1$; $B{=}32 \Rightarrow C{=}2$ |
| $r_e$ | Minimum replication factor for expert $e$: $r_e = \max(1,\, \lceil 32 \cdot f_e \rceil)$ | derived |
| BF16 machine epsilon | $2^{-7} \approx 0.0078$ | 7 mantissa bits |
| T3K link bandwidth | 12.5 GB/s per Ethernet link | per direction |
| T3K avg hop count | 3.0 hops (1×8 linear mesh) | |
| Wormhole B0 Tensix cores | 80 cores, 1.5 MB L1/core | |

All formulas below use these values unless otherwise noted.

---

## Chapter Contents

This chapter consists of three files, each addressing one aspect of load balancing. Read them in the order listed below.

| File | Topic | What it covers |
|---|---|---|
| `load_imbalance_detection.md` | Quantifying and monitoring imbalance | Definition of $f_e$ and $CV$; hot-expert threshold; per-step overflow rate; routing metadata monitoring; expert utilization metric |
| `capacity_overflow_handling.md` | What happens when capacity is exceeded | Hard-drop and reassign policies; output weight renormalization; Poisson token-drop model; worked example at $B=32$ |
| `dynamic_routing_strategies.md` | Inference-time and training-time mitigation | Load-aware routing adjustment; temperature scaling; expert replication via $r_e$; auxiliary load-balancing loss; online vs. offline trade-offs; Qwen3.5-35B recommendation |

---

## Recommended Reading Order

1. **`load_imbalance_detection.md`** — Read first. Establishes the measurement framework ($f_e$, $CV$, hot-expert threshold, utilization metric) that the remaining two files depend on. Without this file, the quantitative claims in `capacity_overflow_handling.md` and `dynamic_routing_strategies.md` lack grounding.

2. **`capacity_overflow_handling.md`** — Read second. Uses the $f_e$ and $C$ definitions from the detection file to derive the token-drop mechanism, the two handling policies, the Poisson drop-rate model, and the worked example at $B = 32$.

3. **`dynamic_routing_strategies.md`** — Read last. Uses the overflow model from the previous file to motivate each mitigation strategy and quantifies the overhead of each approach. The $r_e$ formula here builds directly on Chapter 4, `ch04_expert_device_assignment/expert_replication.md`.

---

## Relationship to Other Chapters

- **Chapter 4, `ch04_expert_device_assignment/`:** The $r_e$ replication factor formula derived there ($r_e = \max(1, \lceil f_e \cdot E / k \rceil) = \max(1, \lceil 32 \cdot f_e \rceil)$) is used directly in `dynamic_routing_strategies.md`. Chapter 4's load-aware assignment and bin-packing strategies are the offline counterparts to this chapter's dynamic strategies.
- **Chapter 6, `ch06_fused_dispatch_compute_combine/`:** The fused pipeline's correctness assumptions depend on token counts not exceeding $C$ per expert. This chapter defines the conditions under which that assumption is violated and how to respond.
- **Chapter 8, `ch08_qwen35b_t3k_strategy/`:** The recommended configuration for Qwen3.5-35B synthesizes the monitoring setup (this chapter), the replication factors (Chapter 4), and the fused pipeline (Chapter 6).

---

## References

- [Ch1Index] Chapter 1, `ch01_moe_fundamentals/index.md` — notation, model parameters, routing problem overview.
- [Ch1Routing] Chapter 1, `ch01_moe_fundamentals/routing_problem.md` — dispatch/combine pattern, load imbalance, expert capacity concept.
- [Ch1Config] Chapter 1, `ch01_moe_fundamentals/qwen35b_config.md` — Qwen3.5-35B architectural constants.
- [Ch2Dispatch] Chapter 2, `ch02_all_to_all_primitives/all_to_all_dispatch.md` — dispatch buffer layout and send-count metadata.
- [Ch2Combine] Chapter 2, `ch02_all_to_all_primitives/all_to_all_combine.md` — combine buffer layout and routing weight accumulation.
- [Ch3Schemes] Chapter 3, `ch03_alternative_routing_schemes/scheme_comparison_matrix.md` — quantitative comparison of routing schemes and load-balance characteristics.
- [Ch4Replication] Chapter 4, `ch04_expert_device_assignment/expert_replication.md` — replication factor formula and dispatch interaction.
- [Ch4LoadAware] Chapter 4, `ch04_expert_device_assignment/load_aware_assignment.md` — routing frequency profiling and bin-packing formulation.
- [Ch5Weights] Chapter 5, `ch05_routing_weight_optimization/` — routing weight optimization; temperature effects on routing distribution.
- [Ch6Fused] Chapter 6, `ch06_fused_dispatch_compute_combine/` — fused pipeline design and capacity assumptions.
- [Ch8Config] Chapter 8, `ch08_qwen35b_t3k_strategy/recommended_configuration.md` — final Qwen3.5-35B T3K configuration.
- [Lepikhin2021] Lepikhin, D. et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding", ICLR, 2021.
- [Fedus2022] Fedus, W., Zoph, B., Shazeer, N., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR, 2022.
- [Zoph2022] Zoph, B. et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models", arXiv:2202.08906, 2022.
- [Qwen3] Qwen Team, Alibaba Group, "Qwen3 Technical Report", 2025.
- [PlanDoc] Expert Parallelism Strategies — Research Guide Plan, `guides/expert_parallelism_strategies/plan.md`.
