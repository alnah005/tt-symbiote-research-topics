# Chapter 1 --- Standard RoPE and M-RoPE: Conceptual Foundations

## Overview

This chapter establishes the mathematical progression from standard 1D Rotary
Position Embedding (RoPE) through partial RoPE to Multimodal RoPE (M-RoPE),
covering all the notation, terminology, and dimensional reasoning needed before
any TTNN-specific material is introduced.

The central question this chapter answers is: what structural limitation of
standard RoPE motivates M-RoPE, and how exactly does M-RoPE resolve it by
partitioning the rotary dimension into three coordinate sub-groups? Readers who
can answer that question after this chapter are prepared to work through the
Qwen3.6-35B-A3B configuration in Chapter 2 and the TTNN implementation strategy
in Chapter 4.

## Learning Objectives

After reading this chapter you will be able to:

- Construct the inverse-frequency vector for any `rope_theta` and `rotary_dim`,
  and explain why pairs of dimensions rotate at different angular frequencies.
- Apply the rotate-half operation to a query or key vector given precomputed
  cos/sin values, and verify the result algebraically.
- Explain partial RoPE: derive `rotary_dim` from `head_dim` and
  `partial_rotary_factor`, and describe what happens to the non-rotated suffix.
- State the limitation of 1D position indexing for multimodal sequences and
  explain the M-RoPE solution in terms of the position ID triplet `(t, h, w)`.
- Read a `mrope_section = [s_t, s_h, s_w]` config value and identify which
  dimension pairs of the cos/sin table are governed by each coordinate.
- Work through the Qwen3.6 concrete example: `head_dim=128`,
  `partial_rotary_factor=0.5`, `rotary_dim=64`, `mrope_section=[11, 11, 10]`.

## Prerequisites

This chapter assumes the following background. Readers who are uncertain on any
item should review the linked definitions before proceeding.

| Prerequisite | What is assumed | Where to review |
|---|---|---|
| Standard RoPE | Familiarity with sinusoidal positional encoding; awareness that RoPE encodes relative positions through rotation of query/key vectors | Su et al. 2021, "RoFormer: Enhanced Transformer with Rotary Position Embedding" |
| Partial RoPE | Know that `partial_rotary_factor < 1.0` means only a prefix of the head dimension is rotated | This chapter, [`standard_rope_recap.md`](./standard_rope_recap.md) §Partial RoPE |
| `head_dim` vs. `rotary_dim` | `head_dim` is the full attention head dimension; `rotary_dim` is the prefix that receives rotation | This chapter, [`standard_rope_recap.md`](./standard_rope_recap.md) |
| TTNN basics | `ttnn.Tensor`, device placement, `ttnn.linear`, `ttnn.matmul` | tt-symbiote TTNN onboarding docs |
| Qwen3.6-35B-A3B architecture | The model uses the same `Qwen3_5MoeForCausalLM` class as Qwen3.5-35B-A3B with GQA and `partial_rotary_factor=0.5` for text layers | Chapter 2 |

## Reading Order

1. [`standard_rope_recap.md`](./standard_rope_recap.md) --- Frequency table
   construction, the rotate-half operation, partial RoPE, and the Qwen3.6
   concrete example.
2. [`mrope_motivation_and_design.md`](./mrope_motivation_and_design.md) --- The
   multimodal limitation of 1D RoPE, the M-RoPE position triplet, the
   `mrope_section` partition, and the 3D position ID tensor structure.
3. [`section_dimension_assignment.md`](./section_dimension_assignment.md) ---
   Deriving which frequency pairs belong to each section, the full Qwen3.6
   dimension map, and the shape of the effective cos/sin tensor.

## Key Terminology Used in This Chapter

All terms below are used consistently throughout this guide.

| Term | Definition |
|---|---|
| `rotary_dim` | Number of head dimensions that receive rotation: `floor(head_dim * partial_rotary_factor)` |
| `partial_rotary_factor` | Fraction of `head_dim` rotated; dimensions `[rotary_dim:]` pass through unchanged |
| `mrope_section` | List `[s_t, s_h, s_w]` partitioning `rotary_dim/2` pairs into temporal, height, width sub-groups; `s_t + s_h + s_w == rotary_dim / 2` |
| Position ID triplet | `(t, h, w)` — the three coordinate values for one token |
| 3D position IDs | `[3 x batch x seq_len]` integer tensor; axis 0 = temporal, axis 1 = height, axis 2 = width |
| Degenerate M-RoPE | M-RoPE where all three axes hold identical values; mathematically equivalent to standard 1D RoPE |
| rotate-half | The apply operation: `(x_i, x_{i + rotary_dim/2}) → (x_i·cos − x_{i + rotary_dim/2}·sin, x_i·sin + x_{i + rotary_dim/2}·cos)` — pairs dimension `i` with dimension `i + rotary_dim/2` (half-offset convention) |
| pairs | One complex rotation consuming 2 real dimensions; `rotary_dim` has `rotary_dim/2` pairs |

## Forward References

- **[Chapter 2 — M-RoPE in Qwen3.6-35B-A3B](../ch2_qwen36_mrope_config/index.md)** instantiates the
  abstract M-RoPE design developed here with the exact Qwen3.6-35B-A3B config
  fields (`rope_theta`, `mrope_section`, `partial_rotary_factor`) and traces the
  HuggingFace reference implementation.
- **[Chapter 3 — Text-Only Reduction](../ch3_text_only_reduction/index.md)** uses the degenerate
  M-RoPE case described in
  [`mrope_motivation_and_design.md`](./mrope_motivation_and_design.md) and the
  section partition from
  [`section_dimension_assignment.md`](./section_dimension_assignment.md) to
  prove that text-only batches produce numerically identical output to standard
  partial RoPE — with implications for whether the existing TTNN text inference
  path needs changes.
