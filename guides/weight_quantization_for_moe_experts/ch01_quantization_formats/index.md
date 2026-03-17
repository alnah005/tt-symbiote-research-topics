## Prerequisites

Before reading this chapter, you should be familiar with:

- Basic PyTorch tensor operations and dtype concepts (`torch.float32`, `torch.bfloat16`)
- TTNN fundamentals: creating tensors, specifying layouts, and running matmul ops
- IEEE 754 floating-point representation at a conceptual level (sign, exponent, mantissa)
- What a Mixture-of-Experts (MoE) model is and why expert weight memory matters

If you are new to TTNN, review the [TTNN Getting Started guide](https://docs.tenstorrent.com/ttnn) before continuing.

---

# Chapter 1 — Quantization Formats on Wormhole: bfloat16, bfloat8_b, and bfloat4_b

## Overview

Tenstorrent's Wormhole architecture supports three primary floating-point formats for weight storage and compute: `bfloat16`, `bfloat8_b`, and `bfloat4_b`. Choosing among them is one of the highest-leverage decisions when deploying large MoE models, directly controlling memory footprint, DRAM bandwidth, and achievable TFLOPS.

This chapter explains what each format is, how it is physically laid out in hardware, what TTNN API constants and layout constraints apply, and how to reason about the tradeoffs for MoE expert weight matrices.

## Learning Objectives

By the end of this chapter you will be able to:

1. Describe the binary layout (bits, exponent width, mantissa width) of each format.
2. Explain what "block floating-point" means and why the `_b` suffix signals it.
3. State the TTNN `DataType` enum constant for each format and the required tensor layout.
4. Calculate the memory footprint of a specific expert weight tensor under each format.
5. Map each format to its approximate Wormhole throughput ceiling and explain the 2×/4× factors.
6. Identify which format DeepSeek-V3 uses for each expert projection and why.

## Navigation

| Section | File | Description |
|---|---|---|
| bfloat16 Format | [bfloat16_format.md](./bfloat16_format.md) | Binary layout, memory footprint, throughput, TTNN usage |
| bfloat8_b Format | [bfloat8_b_format.md](./bfloat8_b_format.md) | Block FP8, shared exponent, 2× compression |
| bfloat4_b Format | [bfloat4_b_format.md](./bfloat4_b_format.md) | Block FP4, 4× compression, precision tradeoffs |
| Hardware & dtype Support | [hardware_dtype_support.md](./hardware_dtype_support.md) | TTNN DataType enum, MathFidelity, bandwidth impact |

---

## Format Comparison Summary

The table below gives a quick reference across the three formats. Throughput numbers are for the Wormhole n150 (single chip) unless noted; n300 figures are approximately 1.77× higher due to the dual-chip configuration.

| Property | `bfloat16` | `bfloat8_b` | `bfloat4_b` |
|---|---|---|---|
| Bits per element | 16 | 8 | 4 |
| Sign bits | 1 | 1 (per element) | 1 (per element) |
| Exponent bits | 8 (per element) | shared per 32×32 tile | shared per 32×32 tile |
| Mantissa bits | 7 | ~7 (+ shared exp) | ~3 (+ shared exp) |
| Format family | IEEE-like | Block floating-point | Block floating-point |
| Dynamic range | ~3.4×10⁻³⁸ to ~3.4×10³⁸ | Tile-relative range | Tile-relative range |
| TTNN dtype constant | `ttnn.bfloat16` | `ttnn.bfloat8_b` | `ttnn.bfloat4_b` |
| Required layout | `TILE_LAYOUT` or `ROW_MAJOR_LAYOUT` | `TILE_LAYOUT` only | `TILE_LAYOUT` only |
| Bytes per element | 2.0 | 1.0 | 0.5 |
| Memory vs bfloat16 | 1× (baseline) | 0.5× (2× reduction) | 0.25× (4× reduction) |
| n150 peak throughput | 74 TFLOPS | 148 TFLOPS | ~296 TFLOPS (est.) |
| n300 peak throughput | 131 TFLOPS | 262 TFLOPS | ~524 TFLOPS (est.) |
| Typical PCC (MLP) | ~1.000 | ~0.975 | ~0.95–0.98 |
| DeepSeek-V3 usage | Activations | Down proj (w2) | Gate/up proj (w1, w3) |

> **Tip:** "Block floating-point" means a group of elements shares a single exponent. Within that group each element stores only its mantissa. The shared exponent is stored once per tile, so the per-element cost drops dramatically while preserving relative precision across values that have similar magnitudes — which is true of most trained weight matrices.

---

## Why This Matters for MoE Models

MoE models like DeepSeek-V3 route each token to a small subset of experts (e.g., 8 of 256). Even when only a handful of experts are active per forward pass, **all expert weights must reside in DRAM** and be streamed in for whichever experts are selected. The per-expert memory footprint and aggregate totals across 256 experts are shown in the format comparison table above. The chapters in this guide build toward the mixed-precision quantization strategy illustrated there. Chapter 1 gives you the format foundations.

---

## Next Steps

After completing this chapter, proceed to:

- **Chapter 2** — Casting and Converting Expert Weights in TTNN (how to load, cast, and validate quantized tensors)
- **Chapter 3** — MathFidelity and Compute Kernel Configuration (LoFi, HiFi2, HiFi4 in context)
