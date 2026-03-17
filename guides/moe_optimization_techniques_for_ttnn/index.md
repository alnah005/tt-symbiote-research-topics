# MoE Optimization Techniques for TTNN

## Overview

This guide covers how to optimize Mixture-of-Experts (MoE) transformer inference on Tenstorrent Wormhole B0 hardware using TTNN, with specific focus on the Qwen3.5-35B model deployed on a T3K 8-device mesh. The central question is: **which TTNN matmul strategy — batched matmul or `sparse_matmul` — should be used for MoE expert computation, and how should it be configured?**

The guide builds from MoE architecture fundamentals and Wormhole hardware concepts through the two matmul strategies, sparsity tensor construction, comparative analysis, multi-chip extension, and an end-to-end workflow reference.

**Target audience:** ML engineers and performance engineers deploying or optimizing MoE models on Tenstorrent hardware. Familiarity with PyTorch, basic TTNN op usage, and transformer architecture is assumed. No prior knowledge of TTNN internals, sparse matmul mechanics, or program config parameters is required.

---

## Chapter Index

| Chapter | Description |
|---|---|
| [Ch 1 — MoE Architecture Fundamentals](ch01_moe_architecture_fundamentals/index.md) | MoE layer structure, top-K routing, sparsity ratio, expert capacity, and why MoE is hard on accelerators |
| [Ch 2 — TTNN and Wormhole Hardware Primer](ch02_ttnn_wormhole_primer/index.md) | Tensix core architecture, L1 vs. DRAM memory hierarchy, TTNN programming model, and matmul tile constraints |
| [Ch 3 — Batched Matmul for MoE](ch03_batched_matmul_for_moe/index.md) | Gather-pad-matmul pattern, program config selection, and performance profile across batch/sequence sizes |
| [Ch 4 — sparse_matmul for MoE](ch04_sparse_matmul_for_moe/index.md) | Tile-skip mechanism, sparsity ratio thresholds, and program configs for decode-regime MoE |
| [Ch 5 — Sparsity Tensor Construction](ch05_sparsity_tensor_construction/index.md) | Format requirements, construction from router output, L1 vs. DRAM placement, and common pitfalls |
| [Ch 6 — Comparative Analysis](ch06_comparative_analysis/index.md) | Side-by-side performance comparison, decision flowchart, and the hybrid prefill/decode strategy |
| [Ch 7 — T3K Multi-Chip MoE Optimization](ch07_t3k_multi_chip_moe/index.md) | Expert parallelism across 8 chips, tensor sharding, and per-chip program config derivation |
| [Ch 8 — End-to-End Workflow and Troubleshooting](ch08_e2e_workflow_and_troubleshooting/index.md) | Model loading, inference loop structure, correctness validation, and troubleshooting reference |

---

## Chapter Summaries

### [Chapter 1 — MoE Architecture Fundamentals](ch01_moe_architecture_fundamentals/index.md)

Establishes the vocabulary needed for all subsequent optimization discussion. Covers the dispatch-combine pattern, top-K routing and why it produces sparse activation across the expert pool, the definition of `expert_capacity` and `sparsity_ratio` as used throughout this guide, and why MoE layers are harder to execute efficiently on accelerators than dense FFN layers. Previews the two TTNN strategies (batched matmul and `sparse_matmul`) that the guide evaluates.

### [Chapter 2 — TTNN and Wormhole Hardware Primer](ch02_ttnn_wormhole_primer/index.md)

Provides the minimal hardware background needed to reason about kernel performance. Covers the Tensix core structure (RISC-V cores, Math engines, NoC), the L1 SRAM vs. DRAM memory hierarchy and its impact on matmul tile scheduling, the TTNN programming model (tensor shapes, dtypes, memory configs, op dispatch), and the 32×32 tile constraint and output subblock parameters (`out_subblock_h`, `out_subblock_w`) that govern all program configs.

### [Chapter 3 — Batched Matmul for MoE](ch03_batched_matmul_for_moe/index.md)

Covers the batched matmul strategy: gathering tokens by expert assignment into a `[num_experts, expert_capacity, d_model]` tensor, selecting `MatmulMultiCoreReuseMultiCastProgramConfig`, and deriving key parameters (`per_core_M`, `per_core_N`, `in0_block_w`). Includes example configs for common shapes and a performance profile showing when batched matmul is preferred (high batch, high sequence length, balanced routing).

### [Chapter 4 — sparse_matmul for MoE](ch04_sparse_matmul_for_moe/index.md)

Introduces `ttnn.sparse_matmul`: how the tile-skip mechanism works at the kernel level, what sparsity ratio thresholds determine whether `sparse_matmul` outperforms batched matmul, and how program configs differ under sparsity constraints. Includes worked decode-regime configurations for `(batch=1, seq=1)`, `(batch=8, seq=1)`, and `(batch=32, seq=1)`.

### [Chapter 5 — Sparsity Tensor Construction](ch05_sparsity_tensor_construction/index.md)

Complete practical guide to building correct sparsity tensors. Covers the required shape, dtype (`uint8`), and layout; the step-by-step construction pipeline from router top-k indices; tile-alignment requirements; L1 vs. DRAM placement decisions; integration with `ttnn.Trace`; and six common construction pitfalls (two of which are silent correctness bugs).

### [Chapter 6 — Comparative Analysis](ch06_comparative_analysis/index.md)

Synthesizes findings from Chapters 3–5 into an actionable decision framework. For Qwen3.5-35B on T3K, the sparsity ratio $\rho = 3.1\%$ at decode (well below the $\rho < 0.1$ threshold), so `sparse_matmul` is always preferred during decode without profiling. The **recommended default is a hybrid strategy**: batched matmul for prefill, `sparse_matmul` for decode.

### [Chapter 7 — T3K Multi-Chip MoE Optimization](ch07_t3k_multi_chip_moe/index.md)

Extends single-chip optimization to the T3K 8-chip mesh. Covers how to partition experts across chips (expert parallelism degree), the all-to-all communication pattern for token dispatch and result reduction, sparsity tensor sharding for the multi-chip case, and how per-chip program configs change when `num_local_experts < num_experts`.

### [Chapter 8 — End-to-End Workflow and Troubleshooting](ch08_e2e_workflow_and_troubleshooting/index.md)

Practical walkthrough for running Qwen3.5-35B MoE inference on T3K from first principles: loading a HuggingFace checkpoint and converting to per-device expert shards, placing weights correctly across DRAM and L1, structuring the decode and prefill inference loops, validating correctness with PCC metrics, and diagnosing common runtime failures.

---

## Key Recommendation

> **For Qwen3.5-35B on T3K:** Use **batched matmul during prefill** and **`sparse_matmul` during decode**. At decode with $B=1$, only 1 of 32 local experts is active on average ($\rho \approx 3.1\%$), so `sparse_matmul` skips ~97% of tile reads and computations. At prefill, high expert utilization makes batched matmul's dense layout more efficient.

---

## Cross-Chapter Dependency Map

| Chapter | Depends On |
|---|---|
| Ch 1: MoE Architecture Fundamentals | None (foundational) |
| Ch 2: TTNN and Wormhole Primer | None (foundational) |
| Ch 3: Batched Matmul for MoE | Ch 1, Ch 2 |
| Ch 4: sparse_matmul for MoE | Ch 1, Ch 2, Ch 3 |
| Ch 5: Sparsity Tensor Construction | Ch 1, Ch 2, Ch 4 |
| Ch 6: Comparative Analysis | Ch 3, Ch 4, Ch 5 |
| Ch 7: T3K Multi-Chip MoE | Ch 3, Ch 4, Ch 5, Ch 6 |
| Ch 8: E2E Workflow and Troubleshooting | All previous chapters |

Read chapters in order 1 → 8 for a complete understanding.
