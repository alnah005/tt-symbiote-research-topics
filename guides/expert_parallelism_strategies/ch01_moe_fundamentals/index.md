# Chapter 1 — MoE Fundamentals and the Routing Problem

## Overview

This chapter establishes the conceptual and mathematical foundation for the rest of the guide. It introduces the Mixture-of-Experts (MoE) architecture, explains how the routing mechanism works, and identifies the core communication and load-balancing challenges that arise when experts are distributed across multiple devices. All subsequent chapters build directly on the vocabulary, notation, and problem framing developed here.

No prior chapters are required. Readers need only the background described in the guide's audience statement: familiarity with the standard transformer architecture (attention, feed-forward network, residual connections), awareness that data parallelism and tensor parallelism exist, and working knowledge of Python and tensor APIs such as TTNN (Tenstorrent's tensor operation library).

---

## What This Chapter Covers

| File | Topic |
|---|---|
| [`moe_architecture.md`](./moe_architecture.md) | The MoE layer structure, mathematical formulation of routing, and Qwen3.5-35B's specific configuration |
| [`routing_problem.md`](./routing_problem.md) | Why routing creates inter-device communication, how token imbalance arises, and the dispatch/combine pattern |
| [`qwen35b_config.md`](./qwen35b_config.md) | Concrete architectural constants for Qwen3.5-35B and why 256 experts with top-8 routing creates unique sharding pressure |

---

## Learning Objectives

After completing this chapter, readers will be able to:

1. **Define an MoE layer** precisely: identify the gating network, expert bank, and top-k selection mechanism, and explain how they interact during a forward pass.

2. **Write the routing equations** from scratch: given a token embedding $x$, derive the router logits $g = xW_r$, apply softmax normalization, and describe the top-k selection that produces the set of active expert indices $I$ and their associated routing weights $w_i$.

3. **Distinguish sparse MoE from dense MoE** and explain the computational savings that sparsity enables — and the new overhead it introduces.

4. **State the key architectural constants for Qwen3.5-35B**: total experts $E = 256$, top-k selection $k = 8$, hidden dimension $H$, expert feed-forward network (FFN) intermediate dimension $D$, number of MoE layers versus dense layers, and the model's total vs. active parameter counts.

5. **Explain the routing problem at the systems level**: articulate why distributing experts across $N = 8$ devices forces token embeddings to travel over inter-device links, identify the load-skew pathology that causes some devices to be overloaded while others sit idle, and describe the dispatch/combine communication pattern at a conceptual level.

6. **Identify the expert capacity concept** and explain its role in bounding per-expert token load, understanding that the formal definition and mechanics are covered in Chapter 7, `capacity_factor_mechanics.md`.

---

## Notation Used in This Chapter

The following symbols are used consistently across all files in this chapter and throughout the guide.

| Symbol | Meaning | Qwen3.5-35B value |
|---|---|---|
| $E$ | Total number of experts | 256 |
| $k$ | Top-k selection count | 8 |
| $N$ | Number of devices | 8 (T3K mesh) |
| $B$ | Batch size (tokens per forward pass) | varies |
| $H$ | Hidden dimension | see `qwen35b_config.md` |
| $D$ | Expert FFN intermediate dimension | see `qwen35b_config.md` |
| $p_e$ | Routing probability for expert $e$ | scalar in $[0, 1]$ |
| $\hat{w}_i$ | Renormalized routing weight for selected expert $i \in I$, equal to $p_i / \sum_{j \in I} p_j$ | varies (renormalized per token) |
| $I$ | Set of selected expert indices for a single token, $\|I\| = k$ | Distinct from $I_e$ (per-expert token index set in plan notation table) |
| $E_d$ | Number of experts per device under uniform assignment, $E_d = E/N$ | $256/8 = 32$ |
| $I_e$ | Set of token indices routed to expert $e$ | set of integers |
| $C$ | Expert capacity (max tokens per expert per forward pass) | derived |
| $CF$ | Capacity factor; scales the per-expert buffer size relative to the average expected token load. Formally defined in Chapter 7, `capacity_factor_mechanics.md`. | see Chapter 7, `capacity_factor_mechanics.md` |
| $L$ | Load imbalance factor: ratio of the most-loaded expert's token count to the expected average | dimensionless, $\geq 1$ |

Note: $I$ (the set of selected expert indices for a single token, $|I| = k$) is distinct from $I_e$ (the set of token indices routed to expert $e$, as used in `routing_problem.md`). Both symbols appear in this chapter.

Device indices are 0-based: devices $0$ through $N-1$. Expert indices are 0-based: experts $0$ through $E-1$.

---

## Reading Order

Read the three files in the following order:

1. **[`moe_architecture.md`](./moe_architecture.md)** — Establishes what an MoE layer is and how the routing math works. This is the prerequisite for everything else in this chapter.

2. **[`routing_problem.md`](./routing_problem.md)** — Explains the systems-level consequences of the routing mechanism: why tokens must cross device boundaries, how load imbalance arises, and what the dispatch/combine pattern looks like before we dive into its implementation.

3. **[`qwen35b_config.md`](./qwen35b_config.md)** — Grounds the abstract architecture in concrete numbers for the specific model this guide targets. Readers who are already deeply familiar with Qwen3.5-35B may read this file first as an orientation, but the notation introduced in [`moe_architecture.md`](./moe_architecture.md) is assumed throughout.

---

## Relationship to Later Chapters

Chapter 2 as a whole takes Chapter 1 as its sole prerequisite, and Chapter 1 contributes foundational concepts to every subsequent chapter. Specifically:

- The dispatch/combine pattern introduced conceptually in `routing_problem.md` is implemented in detail in Chapter 2, `ch02_all_to_all_primitives/all_to_all_dispatch.md` and `all_to_all_combine.md`.
- The expert capacity concept introduced in `routing_problem.md` is given its formal definition in Chapter 7, `capacity_factor_mechanics.md`.
- The Qwen3.5-35B constants from `qwen35b_config.md` appear in quantitative analyses throughout Chapters 2 through 8.

---

## References

- [Shazeer2017] Shazeer, N. et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", ICLR, 2017.
- [Fedus2022] Fedus, W., Zoph, B., Shazeer, N., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR, 2022.
- [Qwen3] Qwen Team, Alibaba Group, "Qwen3 Technical Report", 2025.
- [PlanDoc] Expert Parallelism Strategies — Research Guide Plan, `guides/expert_parallelism_strategies/plan.md`.
