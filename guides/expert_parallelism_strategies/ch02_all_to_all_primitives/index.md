# Chapter 2 — All-to-All Communication Primitives

## Overview

This chapter provides a rigorous treatment of the all-to-all collective communication operation and its role as the canonical mechanism for expert dispatch and combine in Mixture-of-Experts (MoE) inference. Chapter 1 established why tokens must travel across device boundaries and introduced dispatch and combine conceptually. Chapter 2 translates that conceptual picture into precise operational definitions, buffer layouts, and performance models.

**Prerequisite:** Chapter 1 (`ch01_moe_fundamentals/`) in its entirety. Readers must be familiar with the MoE forward pass equations, the dispatch/combine communication pattern, and the Qwen3.5-35B configuration ($E = 256$ experts, $k = 8$ top-k, $N = 8$ devices, $H = 7{,}168$ hidden dimension).

---

## What This Chapter Introduces

| File | Operations and Concepts |
|---|---|
| [`collective_communication_background.md`](./collective_communication_background.md) | Taxonomy of collective operations; precise definition of all-to-all; bandwidth and latency model on an $N$-device mesh; comparison with all-gather + local select |
| [`all_to_all_dispatch.md`](./all_to_all_dispatch.md) | `all_to_all_dispatch` TTNN semantics; pre-dispatch send-count computation; sparse token packing; tile alignment and shard layout; 4-device worked example |
| [`all_to_all_combine.md`](./all_to_all_combine.md) | `all_to_all_combine` TTNN semantics; weighted accumulation of expert outputs; ordering constraints; buffer layout symmetry with dispatch; floating-point accumulation considerations |
| [`dispatch_combine_overhead.md`](./dispatch_combine_overhead.md) | End-to-end MoE layer latency decomposition; arithmetic intensity of dispatch/combine vs. expert FFN; roofline sketch for T3K Ethernet; batch-size crossover threshold |

---

## Where These Operations Appear in the MoE Forward Pass

The two all-to-all operations bracket the expert feed-forward network (FFN) computation in every MoE layer. The sequence for a single MoE layer on an $N = 8$-device T3K (Tenstorrent 8-device mesh) is:

```text
1. Router forward pass
   Input:  token batch [B, H] on each originating device
   Output: routing indices [B, k], routing weights [B, k]

2. Pre-dispatch packing
   Input:  token batch [B, H], routing indices [B, k]
   Output: packed send buffers (one per destination device)

3. all_to_all_dispatch  <--- first all-to-all
   Input:  packed send buffers on all N devices
   Output: received token buffers on all N devices

4. Expert FFN computation (local, on each device's 32 experts)
   Input:  received token buffers [C * E_d, H]
   Output: expert output buffers [C * E_d, H]

5. all_to_all_combine   <--- second all-to-all
   Input:  expert output buffers on all N devices
   Output: received expert outputs on originating devices

6. Weighted accumulation (combine)
   Input:  received expert outputs, routing weights [B, k]
   Output: MoE layer output [B, H]

7. Residual add
   Output: x + y, shape [B, H]
```

Steps 3 and 5 are the all-to-all collectives defined in this chapter. Step 2 and the packing portion of step 6 are the pre- and post-processing that make those collectives efficient. Step 4 is the local expert computation that runs on each device between the two collectives; its latency relative to the collectives is the central concern of `dispatch_combine_overhead.md`.

The notation $C$ denotes expert capacity (maximum tokens an expert can process per forward pass) and $E_d = E/N = 256/8 = 32$ denotes the number of experts per device under uniform assignment. The formal definition of $C$ is deferred to Chapter 7, `capacity_factor_mechanics.md`; for this chapter it suffices to know that $C$ determines the allocated buffer size per expert in the dispatch and combine operations.

---

## Notation Used in This Chapter

The following symbols extend the notation established in Chapter 1.

| Symbol | Meaning |
|---|---|
| $E$ | Total number of experts: 256 |
| $k$ | Top-$k$ selection count: 8 |
| $N$ | Number of devices: 8 (T3K mesh) |
| $B$ | Batch size (tokens per device per forward pass) |
| $H$ | Hidden dimension: 7,168 |
| $D$ | Expert FFN intermediate dimension (unverified placeholder; see Chapter 1, `qwen35b_config.md`) |
| $E_d$ | Experts per device: $E/N = 32$ under uniform assignment |
| $C$ | Expert capacity; formally defined in Chapter 7, `capacity_factor_mechanics.md` |
| $CF$ | Capacity factor; multiplier $\geq 1.0$ on expected tokens per expert |
| $p_e$ | Per-token routing probability for expert $e$ |
| $\hat{w}_i$ | Renormalized routing weight for selected expert $i \in I$; $\hat{w}_i = p_i / \sum_{j \in I} p_j$ |
| $\text{BW}$ | Inter-device link bandwidth (bytes per second) |
| $\alpha$ | Per-message network latency (seconds) |
| $\beta$ | Per-byte transmission time: $1/\text{BW}$ (seconds per byte) |

---

## Reading Order

Read the four files in the following order:

1. **[`collective_communication_background.md`](./collective_communication_background.md)** — Establishes the collective communication vocabulary and the formal definition of all-to-all before any TTNN specifics are introduced. Readers already fluent in collective communication may skim the taxonomy sections and focus on the bandwidth/latency model.

2. **[`all_to_all_dispatch.md`](./all_to_all_dispatch.md)** — Specifies what `all_to_all_dispatch` does, how its inputs are prepared, and what its outputs look like. The worked example traces a concrete 4-device case end to end.

3. **[`all_to_all_combine.md`](./all_to_all_combine.md)** — Specifies `all_to_all_combine` as the inverse operation, covering the ordering constraints and weighted accumulation that transform raw expert outputs into the final MoE layer output.

4. **[`dispatch_combine_overhead.md`](./dispatch_combine_overhead.md)** — Quantifies the performance cost of the two collectives relative to expert FFN computation and identifies the batch-size regime where communication dominates.

---

## Relationship to Later Chapters

- The dispatch/combine operations defined here are the primitives on which Chapter 3, `ch03_alternative_routing_schemes/scheme_comparison_matrix.md` evaluates competing routing designs.
- Chapter 4, `ch04_expert_device_assignment/expert_replication.md` extends dispatch semantics to handle replicated experts; readers of Chapter 4 must have completed this chapter first.
- Chapter 5, `ch05_routing_weight_optimization/weight_normalization.md` discusses deferring renormalization to the combine step; the combine semantics defined here are the prerequisite.
- Chapter 6, `ch06_fused_dispatch_compute_combine/pipeline_design.md` integrates the two collectives into a pipelined execution plan; the latency model in `dispatch_combine_overhead.md` directly feeds Chapter 6's crossover-threshold analysis.

---

## References

- [Rabenseifner2004] Rabenseifner, R., "Optimization of Collective Reduction Operations", International Conference on Computational Science (ICCS), 2004.
- [Thakur2005] Thakur, R., Rabenseifner, R., Gropp, W., "Optimization of Collective Communication Operations in MPICH", International Journal of High Performance Computing Applications, 2005.
- [Fedus2022] Fedus, W., Zoph, B., Shazeer, N., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR, 2022.
- [Qwen3] Qwen Team, Alibaba Group, "Qwen3 Technical Report", 2025.
- [Ch1Index] Chapter 1, `ch01_moe_fundamentals/index.md` — Chapter 1 overview and notation reference.
- [Ch1Routing] Chapter 1, `ch01_moe_fundamentals/routing_problem.md` — dispatch/combine pattern introduced conceptually.
- [Ch7Capacity] Chapter 7, `ch07_load_balancing/capacity_factor_mechanics.md` — formal definition of expert capacity $C$.
