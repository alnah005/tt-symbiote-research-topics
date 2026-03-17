# Weight Quantization for MoE Experts on Wormhole B0

## Overview

Large Mixture-of-Experts models such as Qwen 235B-A22B and DeepSeek-V3 carry all expert
weights in DRAM regardless of which experts are active per token. At bfloat16, 128 experts
across a T3K 8-chip system consume 1,344 MB per chip; decode is memory-bound, so bandwidth
is the primary latency driver. This guide explains how to replace bfloat16 expert weights
with block floating-point formats (`bfloat8_b`, `bfloat4_b`) on Wormhole B0, reducing the
per-chip footprint to as little as 448 MB (3× reduction) while keeping production-quality
accuracy. The guide covers format mechanics, the TTNN quantization API, accuracy measurement,
throughput analysis, per-projection mixed-precision strategy, a cross-model comparative study,
and an end-to-end implementation workflow with PCC validation gates.

---

## Prerequisites

- TTNN fundamentals: tensor creation, `ttnn.linear`, `TILE_LAYOUT`, `DRAM_MEMORY_CONFIG`
- BF16 familiarity: IEEE-like 16-bit float as the unquantized baseline
- Wormhole B0 hardware awareness: DRAM bandwidth (~300 GB/s), 80 Tensix cores, T3K 8-chip topology

---

## Chapter Table

| # | Title | What you learn |
|---|---|---|
| 1 | Quantization Formats on Wormhole: bfloat16, bfloat8_b, bfloat4_b | Binary layout of each format, block floating-point semantics, TTNN dtype constants, tile memory footprints, throughput ceilings |
| 2 | TTNN Quantization API | `ttnn.as_tensor` dtype/layout arguments, `WormholeComputeKernelConfig` construction, LoFi and HiFi2 configs, T3K shard mapping |
| 3 | Accuracy Analysis for MoE Expert Quantization | PCC as the primary correctness metric, per-projection sensitivity ordering (down > gate > up), per-dtype PCC ranges, model-family differences |
| 4 | Throughput and Memory Bandwidth Impact | Roofline model for decode (memory-bound) vs. prefill (compute-bound), DRAM read volume reduction, MathFidelity pass counts, tile efficiency |
| 5 | Per-Projection Quantization Strategy | Why gate/up tolerate `bfloat4_b` + LoFi and down requires `bfloat8_b` + HiFi2; DRAM layout for mixed-precision expert tensors; Qwen weight conversion guide |
| 6 | Comparative Study: DeepSeek-V3 vs. Qwen | Production DeepSeek-V3 design vs. Qwen bfloat16 baseline; the 16ms decode gap; three-criterion decision framework; recommended Qwen starting point |
| 7 | Implementation and Validation Workflow | Five-step workflow: baseline measurement, weight conversion, per-layer PCC validation, throughput profiling, iterative tuning and regression locking |

---

## Key Facts

```
Models: Qwen 235B-A22B, DeepSeek-V3
  d_model = 7168 | d_ff = 2048 | num_experts = 128 | top_k = 8

Per-expert memory
  BF16:              3 × 7168 × 2048 × 2 bytes = 88,080,384 bytes ≈ 84.0 MB
  Mixed (BF4 g/u + BF8 down):                                      ≈ 28.0 MB  (3× reduction)
  All-BF8:                                                          ≈ 42.0 MB  (2× reduction)

T3K (8 chips, 16 experts/chip)
  BF16 total:   16 × 84.0 MB = 1,344 MB/chip
  Mixed total:  16 × 28.0 MB =   448 MB/chip

Tile memory (32×32 elements)
  bfloat16  = 2,048 bytes
  bfloat8_b = 1,024 bytes
  bfloat4_b =   512 bytes

Compute kernel configs (fp32_dest_acc_en=False for both)
  LOFI  — math_fidelity=LoFi,  math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
  HIFI2 — math_fidelity=HiFi2, math_approx_mode=True,  fp32_dest_acc_en=False, packer_l1_acc=True

PCC thresholds
  Production BF16 baseline : > 0.999  (layer-level vs. CPU reference)
  bfloat4_b gate/up        : ≥ 0.96
  bfloat8_b down           : ≥ 0.975
  Full MoE layer            : ≥ 0.97
```

---

## How to Read This Guide

**First-time readers:** work through Chapters 1–7 in order. Each chapter depends on the
vocabulary and constants established by the chapters before it.

**Practitioners who already understand block floating-point formats and the TTNN API:**
start at Chapter 5 for the per-projection strategy and Qwen adaptation guide, then use
Chapter 6 to calibrate the strategy against the DeepSeek-V3 production baseline, and
Chapter 7 for the step-by-step implementation and validation workflow.

---

## Navigation

Start here: [Chapter 1 — Quantization Formats on Wormhole](./ch01_quantization_formats/index.md)
