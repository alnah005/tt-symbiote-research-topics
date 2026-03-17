# Chapter 8: End-to-End Workflow and Troubleshooting

## Prerequisites

This chapter assumes familiarity with all material covered in Chapters 1 through 7 of this guide,
including:

- The Mixture-of-Experts (MoE) architecture and Qwen3.5-35B model parameters
- TTNN memory configs, program configs, and the sparse matmul primitive
- T3K mesh topology and all-to-all collective semantics
- Expert dispatch, token routing, and capacity factor mechanics
- BF16/INT8 quantization tradeoffs

Readers who have not completed Chapters 1–7 should do so before proceeding.

## Goal

This chapter provides a complete, practical walkthrough of everything required to run Qwen3.5-35B
MoE inference on a T3K 8-device cluster from first principles:

1. Loading a standard HuggingFace checkpoint and converting it to per-device expert shards.
2. Placing weights correctly across DRAM and L1 given the memory constraints of Wormhole B0.
3. Structuring the decode and prefill inference loops using TTNN primitives.
4. Diagnosing and resolving the most common runtime failures.

## Key Insight

Effective MoE inference on T3K is not achieved by optimizing any single component in isolation.
Weight placement, memory configuration, program configuration, and sparse matmul must all be
coordinated together. A correct weight layout with the wrong memory config will silently degrade
throughput; a correct memory config with mismatched program config will trigger allocation errors
or produce wrong results. This chapter shows how all four concerns fit together into a single
coherent workflow.

## Navigation

| Section | File | Description |
|---|---|---|
| 8.1 Model Loading and Weight Placement | [`model_loading_and_weight_placement.md`](./model_loading_and_weight_placement.md) | Checkpoint conversion, expert sharding across 8 devices, DRAM vs. L1 placement decisions |
| 8.2 Inference Loop Structure | [`inference_loop_structure.md`](./inference_loop_structure.md) | Full decode and prefill step breakdown, MoE layer internals, key TTNN API calls |
| 8.3 Troubleshooting Guide | [`troubleshooting_guide.md`](./troubleshooting_guide.md) | Six common errors with diagnosis and fix procedures |

## Chapter Constants (Qwen3.5-35B on T3K)

All numerical examples in this chapter use the following verified constants unless explicitly noted
otherwise.

| Symbol | Value | Description |
|---|---|---|
| $E$ | 256 | Total number of experts |
| $k$ | 8 | Top-k experts selected per token |
| $N$ | 8 | Number of T3K devices |
| $H$ | 7168 | Hidden dimension |
| $E_d$ | 32 | Experts per device ($E / N$) |
| $\text{CF}$ | 1.25 | Capacity factor |
| $\rho$ | $k/E = 3.1\%$ | Sparsity ratio |

Derived decode capacity (batch size $B$):

$$C = \left\lceil \frac{k \times B \times \text{CF}}{E} \right\rceil = \left\lceil \frac{B}{25.6} \right\rceil$$

Selected values: $B=1 \Rightarrow C=1$; $B=32 \Rightarrow C=2$.
