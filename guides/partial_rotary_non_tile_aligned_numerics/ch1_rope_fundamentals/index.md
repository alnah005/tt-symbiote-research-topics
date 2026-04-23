# Chapter 1 — Partial RoPE Fundamentals and Tile Alignment Requirements

This chapter builds the conceptual foundation needed to understand the numerical correctness problem investigated in this guide. It covers the mathematics of partial Rotary Position Embedding (RoPE), the tile-alignment constraints imposed by the TTNN TILE layout, and the vocabulary used throughout all subsequent chapters. Readers who are already fluent in standard RoPE math and TTNN tile layout may skim the math file and focus on the tile alignment discussion.

---

## Learning Objectives

By the end of this chapter you will be able to:

- State the rotate-half formulation of RoPE and extend it correctly to the partial case where `rotary_dim < head_dim`.
- Explain why a non-tile-aligned `rotary_dim` forces the cos/sin tensor to be padded before it can be stored in TTNN TILE layout.
- Identify the silent correctness risk that zero-padding introduces when the downstream op does not respect `rotary_dim` as a logical boundary.
- Define every term in the chapter glossary precisely enough to use it in a bug report.

---

## Prerequisite Checklist

Work through the following before reading this chapter. Each item is assumed knowledge; it is not re-derived here.

- **Standard RoPE math.** You should be comfortable with the full-head formulation: a head vector $x \in \mathbb{R}^{d}$ is embedded by element-wise multiplication with $\cos\theta$ and $\sin\theta$ tensors derived from position and frequency. A good reference is the original RoFormer paper (Su et al., 2021).
- **`head_dim` vs. `rotary_dim`.** You should know that `head_dim` is the total size of each attention head, while `rotary_dim` is the number of elements within that head that receive positional encoding. These two quantities are equal in vanilla RoPE but differ in models like Qwen3 that use partial RoPE.
- **TTNN TILE layout basics.** You should know that TTNN offers two primary tensor layouts (ROW_MAJOR and TILE), and that TILE layout tiles data into 32x32 blocks. If you are unfamiliar, read the TTNN layout documentation before continuing.

---

## Glossary

| Term | Definition |
|---|---|
| `rotary_dim` | The number of elements per attention head that receive rotary positional encoding. Must satisfy `rotary_dim <= head_dim`. When `rotary_dim == head_dim`, the operation is full RoPE; otherwise it is partial RoPE. |
| `partial_rotary_factor` | A scalar in $(0, 1]$ such that `rotary_dim = int(partial_rotary_factor * head_dim)`. Used in model configs (e.g., Qwen3) to parameterize how much of each head is rotated. |
| tile-aligned | A dimension whose size is an exact multiple of 32. A tensor in TTNN TILE layout requires its last two dimensions to both be tile-aligned. |
| `nearest_32` | The function $\text{nearest\_32}(n) = \lceil n / 32 \rceil \times 32$. Returns the smallest multiple of 32 that is greater than or equal to $n$. Used to compute the padded size of a non-tile-aligned dimension. |
| rotate-half pairing | The pairing scheme used inside the rotated region: element $i$ in $[0, \text{rotary\_dim}/2)$ is paired with element $i + \text{rotary\_dim}/2$. These pairs jointly encode one sinusoidal frequency. The pairing must be computed within the rotated slice, not across the full `head_dim`. |

---

## Files in Reading Order

1. [`partial_rope_math.md`](./partial_rope_math.md) — The rotate-half formulation of standard and partial RoPE, with a concrete worked example for `rotary_dim=48, head_dim=128`.
2. [`tile_alignment_in_ttnn.md`](./tile_alignment_in_ttnn.md) — How TTNN TILE layout constrains tensor shapes, why non-tile-aligned `rotary_dim` requires padding, and the correctness risk that padding introduces.

---

## What's Next

Chapter 2 analyzes how `ttnn.experimental.rotary_embedding` actually enforces these constraints — specifically, whether it reads exactly `rotary_dim` elements from the cos/sin tensor or the full padded width.

**Next:** [Chapter 2 — How `ttnn.experimental.rotary_embedding` Processes cos/sin Shapes](../ch2_op_shape_contract/index.md)
