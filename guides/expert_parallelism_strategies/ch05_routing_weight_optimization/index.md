# Chapter 5: Routing Weight Computation and Overhead Minimization

## Prerequisites

This chapter assumes familiarity with:

- **Chapter 1** (`ch01_moe_fundamentals/moe_architecture.md`, `ch01_moe_fundamentals/routing_problem.md`): Mixture-of-Experts (MoE) architecture fundamentals and the expert routing problem.
- **Chapter 2** (`ch02_all_to_all_primitives/all_to_all_dispatch.md`, `ch02_all_to_all_primitives/all_to_all_combine.md`): All-to-all dispatch and combine primitives on T3K hardware.

## Overview

Every MoE layer begins with a router sub-graph that must complete before any token can be dispatched to its assigned experts. Although the router is small relative to the expert feed-forward networks, it lies on the critical path of every forward pass: dispatch cannot start until routing decisions are fully materialized. For Qwen3.5-35B on T3K hardware — 256 experts, top-8 selection, 8 devices, hidden dimension 7168 — even modest inefficiencies in the router accumulate across all MoE layers.

This chapter dissects the router computation into its constituent operations, analyzes each for latency and memory overhead, and presents concrete strategies for minimizing total routing cost through algorithmic selection, kernel fusion, and deferred normalization.

**Key concepts introduced in this chapter:**

| Concept | Brief description |
|---|---|
| Logit computation | The linear projection $g = xW_r$ that produces per-expert scores |
| Softmax vs. sigmoid routing | Two normalization choices with different properties and implementation costs |
| Top-k complexity | Why partial selection is ~7× cheaper than full sort for $E=256$, $k=8$ |
| Weight normalization timing | Why deferring sigmoid weight normalization to the combine step reduces dispatch payload size |
| Kernel fusion | How fusing projection, top-k, and metadata preparation reduces critical-path latency |

## Chapter Files

| File | Description |
|---|---|
| `router_forward_pass.md` | End-to-end router forward pass: linear projection, sigmoid vs. softmax, top-k selection, auxiliary loss at training time, and numerical precision in BF16 |
| `topk_selection_efficiency.md` | Algorithmic and hardware analysis of top-k selection for $E=256$, $k=8$; partial selection vs. full sort; batched and tile-parallel strategies |
| `weight_normalization.md` | Why sigmoid routing requires explicit weight renormalization; timing options (before vs. after dispatch); recommended deferred approach for T3K |
| `router_kernel_fusion.md` | Kernel fusion opportunities: projection + top-k + index extraction; normalization + scatter metadata; double-buffering for latency hiding; INT8 quantization of $W_r$ |

## Qwen3.5-35B Router at a Glance

For concreteness, all analysis in this chapter uses the following constants derived from the Qwen3.5-35B model and T3K deployment configuration:

| Parameter | Value |
|---|---|
| Number of experts $E$ | 256 |
| Top-k selection $k$ | 8 |
| Hidden dimension $H$ | 7168 |
| Number of devices $N$ | 8 |
| Router weight matrix $W_r$ shape | $[7168, 256]$ |
| $W_r$ size in BF16 | $7168 \times 256 \times 2 = 3{,}670{,}016$ bytes $\approx 3.67$ MB |
| Average load per expert $f_{\text{avg}}$ | $k/E = 8/256 = 1/32$ |
| Capacity factor (CF) | 1.25 (default) |

The router forward pass for a token batch of size $B$ proceeds as follows:

$$g = xW_r \quad (x \in \mathbb{R}^{B \times H},\ W_r \in \mathbb{R}^{H \times E},\ g \in \mathbb{R}^{B \times E})$$

followed by per-expert sigmoid activation, top-8 selection, and (for sigmoid routing) explicit weight renormalization. The auxiliary load-balancing loss used during training is stripped from the inference computation graph entirely.

## Navigation

Read the files in order for a complete picture, or jump directly to a specific topic:

- For the full algorithmic walkthrough of the router: start with `router_forward_pass.md`.
- For hardware-level analysis of top-k cost: go to `topk_selection_efficiency.md`.
- For normalization timing and payload implications: see `weight_normalization.md`.
- For fusion and latency-hiding strategies: see `router_kernel_fusion.md`.

## References

- Qwen Technical Report (Qwen3.5-35B model card), Alibaba Cloud, 2025.
- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer," ICLR 2017.
- Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding," ICLR 2021.
- Tenstorrent, *TTNN Developer Guide*, 2024.
- Chapter 1 of this guide: `ch01_moe_fundamentals/`.
- Chapter 2 of this guide: `ch02_all_to_all_primitives/`.
