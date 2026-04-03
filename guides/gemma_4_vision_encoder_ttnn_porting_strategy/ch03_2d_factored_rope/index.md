# Chapter 3 — 2D Factored RoPE for Vision: Theory and TTNN Mapping

## Learning Objectives

After completing this chapter, you will be able to:

- Derive the standard 1D RoPE frequency schedule and rotation formula from first principles, and explain the role of `rope_theta`
- Extend the 1D formulation to 2D by splitting the head dimension across spatial axes (x, y) with independent frequency tables
- Walk through the HuggingFace reference implementation (`Gemma4VisionRotaryEmbedding`, `apply_multidimensional_rope`) line by line
- Identify the three gaps between current TTNN RoPE kernels and Gemma 4 vision requirements
- Select the appropriate implementation strategy (CPU precompute, TTNN composition, or custom kernel) for bringup versus optimized deployment

## Prerequisites

- Basic understanding of standard 1D RoPE as used in language models (Llama, Gemma text decoder)
- Completion of [Chapter 1 — Gemma 4 Vision Encoder Architecture Overview](../ch01_gemma4_vision_architecture/index.md) (attention module structure)
- Completion of [Chapter 2 — SigLIP vs. Gemma 4 Comparison](../ch02_siglip_vs_gemma4_comparison/index.md), specifically [`positional_encoding_shift.md`](../ch02_siglip_vs_gemma4_comparison/positional_encoding_shift.md)

## Chapter Contents

| File | Topic |
|------|-------|
| [`multidimensional_rope_theory.md`](./multidimensional_rope_theory.md) | 1D RoPE recap, extension to 2D factored RoPE, mathematical derivation, and the role of `rope_theta=100.0` |
| [`reference_implementation.md`](./reference_implementation.md) | Line-by-line walkthrough of the HuggingFace `Gemma4VisionRotaryEmbedding` and `apply_multidimensional_rope` |
| [`ttnn_rope_gap_analysis.md`](./ttnn_rope_gap_analysis.md) | Gap analysis between current TTNN RoPE and Gemma 4 vision requirements, with three ranked implementation strategies |

## Overview

The defining positional encoding mechanism of the Gemma 4 vision encoder is **2D factored Rotary Position Embedding**. Unlike the language model decoder, which applies 1D RoPE over a linear sequence of token positions, the vision encoder must encode spatial relationships along two independent axes: the horizontal (x) and vertical (y) dimensions of the patch grid.

The core idea is straightforward: split the head dimension in half, assign the first half to the x-axis and the second half to the y-axis, and apply standard RoPE independently to each half using the corresponding spatial coordinate. The result is a rotation matrix that encodes both the horizontal and vertical position of each patch, enabling attention to compute position-dependent similarity along both spatial axes simultaneously.

This chapter builds understanding in three stages:

1. **Theory** ([`multidimensional_rope_theory.md`](./multidimensional_rope_theory.md)) — Starting from the familiar 1D RoPE formula, we derive the 2D extension mathematically and explain why `rope_theta=100.0` is appropriate for the vision domain.

2. **Reference implementation** ([`reference_implementation.md`](./reference_implementation.md)) — We walk through the exact HuggingFace code that computes and applies 2D RoPE, including how patch grid coordinates are generated and how the cos/sin tables are structured.

3. **TTNN gap analysis** ([`ttnn_rope_gap_analysis.md`](./ttnn_rope_gap_analysis.md)) — We identify the specific gaps between existing TTNN RoPE kernels (optimized for 1D language model decoding) and the 2D vision requirements, then rank three implementation strategies by effort and performance.

The analysis in this chapter directly informs the new module implementation plan in [Chapter 6](../ch06_reuse_strategy/index.md) and the phased roadmap in [Chapter 7](../ch07_implementation_roadmap/index.md).
