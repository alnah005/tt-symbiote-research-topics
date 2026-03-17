# Chapter 4 — Expert-to-Device Assignment for 256 Experts on 8 Devices

## Overview

Chapters 1–3 established the Mixture-of-Experts (MoE) architecture, the all-to-all dispatch/combine communication primitives, and the landscape of alternative routing schemes. This chapter addresses a question that cuts across all of those: **given that experts must be distributed across 8 devices, which expert goes where?**

The goal is to determine the expert-to-device assignment that minimizes peak device load — measured in tokens processed per forward pass — while respecting memory constraints. A poor assignment causes one or more devices to receive a disproportionate share of routed tokens, creating a load bottleneck that all devices must wait for. An assignment that ignores the T3K linear mesh topology can also inflate communication hop counts unnecessarily. This chapter analyzes three increasingly sophisticated assignment strategies and characterizes when each is worth the added complexity.

**Prerequisites:** This chapter requires all of Chapters 1, 2, and 3 in their entirety. Readers must be familiar with:

- The MoE forward pass, routing probabilities $p_e$, and expert capacity $C$ (Chapter 1, `ch01_moe_fundamentals/`)
- The all-to-all dispatch and combine primitives and their buffer layouts (Chapter 2, `ch02_all_to_all_primitives/`)
- Expert sharding, pipeline expert parallelism, hierarchical routing, and their trade-offs (Chapter 3, `ch03_alternative_routing_schemes/`)

---

## Goal Statement

Given:
- $E = 256$ total experts,
- $N = 8$ devices (T3K 1×8 linear mesh),
- top-$k = 8$ routing per token,
- per-expert routing frequency $f_e$ derived from a calibration dataset,
- a fixed per-device DRAM memory budget,

find an assignment $\sigma : \{0, \ldots, E-1\} \to \{0, \ldots, N-1\}$ such that:

$$\max_{d \in \{0,\ldots,N-1\}} \sum_{e:\,\sigma(e)=d} f_e \cdot B$$

is minimized subject to the constraint that each device holds at most $M_d$ bytes of expert weights (including any replication overhead).

Under uniform routing ($f_e = k/E$ for all $e$), any balanced assignment with $E/N = 32$ experts per device achieves the optimum. Under non-uniform routing, load-aware and replication-based strategies are needed.

---

## Assignment Strategies: Summary

| Strategy | Description | Pro | Con |
|---|---|---|---|
| **Uniform partitioning** | Round-robin by expert index: device $d$ holds experts $\{d,\, d+8,\, d+16,\, \ldots,\, d+248\}$ | Zero profiling cost; simple TTNN sharding | Ignores routing skew; hot-expert devices bottleneck the entire forward pass |
| **Load-aware assignment** | Profile per-expert frequency $f_e$; bin-pack experts onto devices to equalize load | Near-optimal load balance with no replication overhead | Requires a calibration dataset; reassignment involves cross-device weight migration |
| **With replication** | Replicate the top-$M$ most popular experts on multiple devices; dispatch routes tokens to any available replica | Absorbs load spikes; directly reduces peak device token count | Multiplies memory usage for replicated experts; dispatch must track replica locations |

---

## Chapter Notation

The following symbols are used throughout all files in this chapter. Symbols inherited from Chapters 1–3 are listed for quick reference; new symbols introduced in this chapter are marked **new**.

| Symbol | Meaning | Value for Qwen3.5-35B on T3K |
|---|---|---|
| $E$ | Total number of experts | 256 |
| $k$ | Top-$k$ selection count | 8 |
| $N$ | Number of devices | 8 |
| $B$ | Batch size: tokens per forward pass | varies |
| $S$ | Sequence length (tokens per sequence) | varies |
| $H$ | Hidden dimension | 7168 |
| $D$ | Expert FFN intermediate dimension | [UNVERIFIED — see `ch01_moe_fundamentals/qwen35b_config.md`] |
| $E_d$ | Experts per device under uniform assignment, $E_d = E/N$ | 32 |
| $p_e$ | Routing probability for expert $e$; scalar in $[0,1]$ representing the expected fraction of tokens that include expert $e$ in their top-$k$ selection | varies |
| $f_e$ | **new** Routing frequency for expert $e$: the empirically measured fraction of tokens (over a calibration dataset) that route to expert $e$. Satisfies $\sum_e f_e = k$ (each token selects $k$ experts). | varies |
| $C$ | Expert capacity: maximum tokens an expert can process in one forward pass, $C = \lceil k \cdot B \cdot S / E \rceil$; for $B=S=1$: $C=1$ | derived |
| $CF$ | Capacity factor: multiplier $\geq 1.0$ that scales expert capacity above the expected average; defined in Chapter 7, `ch07_load_balancing/capacity_factor_mechanics.md` | see Chapter 7 |
| $L$ | Load imbalance factor: ratio of peak device token count to mean device token count | $\geq 1$; 1.0 is perfect balance |
| $r_e$ | **new** Replication factor for expert $e$: number of devices that hold a copy of expert $e$'s weights | integer in $[1, N]$ |
| $M$ | **new** Number of hot experts selected for replication | integer in $[0, E]$ |
| $w_{ij}$ | **new** Co-activation frequency between experts $i$ and $j$: empirical probability that both $i$ and $j$ appear in the same token's top-$k$ selection | scalar in $[0,1]$ |

Device indices are 0-based: devices $0$ through $N-1 = 7$. Expert indices are 0-based: experts $0$ through $E-1 = 255$.

---

## Reading Order

Read the four files in the following order:

1. **`uniform_partitioning.md`** — The baseline strategy. Establishes the memory footprint per device, defines the load imbalance metric $L$, and explains why round-robin assignment is the natural TTNN default. Read this first; all later files compare against it.

2. **`load_aware_assignment.md`** — Expert popularity profiling and the bin-packing formulation. Introduces the greedy decreasing first (GDF) algorithm for near-optimal assignment without an integer linear program (ILP) solver. Requires understanding of the load imbalance metric defined in `uniform_partitioning.md`.

3. **`expert_replication.md`** — Replicating hot experts across multiple devices to absorb load spikes. Requires the replication factor formula and dispatch interaction discussion, both of which build on the load model from `load_aware_assignment.md`.

4. **`mesh_topology_constraints.md`** — T3K-specific placement considerations: locality-aware assignment to minimize inter-device hop counts given the 1×8 linear topology. Read last; it synthesizes all three prior strategies and adds the co-activation graph partitioning layer.

---

## Relationship to Later Chapters

Results from this chapter feed directly into two later chapters:

- **Chapter 6, `ch06_fused_dispatch_compute_combine/`:** The fused pipeline assumes a fixed expert assignment. Chapter 6, `pipeline_design.md` cites `uniform_partitioning.md` for the 32-expert-per-device assumption and `expert_replication.md` for the dispatch metadata augmentation needed to address replicas.

- **Chapter 7, `ch07_load_balancing/`:** The dynamic load rebalancing strategy in Chapter 7, `dynamic_load_rebalancing.md` uses the assignment strategies here as its response actions; monitoring signals a reassignment event, and the assignment algorithms defined here are then re-run.

- **Chapter 8, `ch08_qwen35b_t3k_strategy/`:** The recommended configuration in Chapter 8, `recommended_configuration.md` cites all four files of this chapter when justifying its expert assignment choice for Qwen3.5-35B.

---

## References

- [Ch1Index] Chapter 1, `ch01_moe_fundamentals/index.md` — notation, model parameters, routing problem overview.
- [Ch1Routing] Chapter 1, `ch01_moe_fundamentals/routing_problem.md` — dispatch/combine pattern, load imbalance, expert capacity concept.
- [Ch1Config] Chapter 1, `ch01_moe_fundamentals/qwen35b_config.md` — Qwen3.5-35B architectural constants.
- [Ch2Dispatch] Chapter 2, `ch02_all_to_all_primitives/all_to_all_dispatch.md` — dispatch buffer layout and send-count metadata.
- [Ch2Combine] Chapter 2, `ch02_all_to_all_primitives/all_to_all_combine.md` — combine buffer layout and routing weight accumulation.
- [Ch3Sharding] Chapter 3, `ch03_alternative_routing_schemes/expert_sharding.md` — static expert sharding baseline and all-gather volume comparison.
- [Ch3Matrix] Chapter 3, `ch03_alternative_routing_schemes/scheme_comparison_matrix.md` — quantitative comparison of routing schemes.
- [Ch7Capacity] Chapter 7, `ch07_load_balancing/capacity_factor_mechanics.md` — formal definition of expert capacity and capacity factor.
- [Ch7Dynamic] Chapter 7, `ch07_load_balancing/dynamic_load_rebalancing.md` — runtime reassignment triggered by load monitoring.
- [Ch8Config] Chapter 8, `ch08_qwen35b_t3k_strategy/recommended_configuration.md` — final Qwen3.5-35B T3K configuration.
- [Qwen3] Qwen Team, Alibaba Group, "Qwen3 Technical Report", 2025.
- [PlanDoc] Expert Parallelism Strategies — Research Guide Plan, `guides/expert_parallelism_strategies/plan.md`.
