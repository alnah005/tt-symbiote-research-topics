# Tile Alignment in TTNN

This file explains why TTNN's TILE layout imposes a 32-element alignment requirement on tensor dimensions, how that requirement forces a non-tile-aligned `rotary_dim` to be padded before the cos/sin tensor can be stored in device memory, and why that padding creates a latent correctness risk that is invisible to the caller.

---

## TTNN TILE Layout Requirement

TTNN TILE layout stores tensors as a grid of 32x32 tiles. As a consequence, **the last two logical dimensions of any TILE-layout tensor must both be multiples of 32**:

$$\text{dim}[-1] \equiv 0 \pmod{32} \quad \text{and} \quad \text{dim}[-2] \equiv 0 \pmod{32}$$

The constants `TILE_HEIGHT = TILE_WIDTH = 32` are fixed by the hardware. There is no mechanism to store a tensor whose last dimension is, say, 48 in TILE layout without first padding it to 64.

> **Note:** ROW_MAJOR layout does not impose this constraint. However, most compute-intensive TTNN ops require TILE layout inputs for performance, so tensors that will be consumed by ops like `ttnn.experimental.rotary_embedding` must be converted — and conversion requires the dimension to be tile-aligned.

---

## The Problem: Non-Tile-Aligned `rotary_dim`

For example, a model with `head_dim=128` and `partial_rotary_factor=0.375` has `rotary_dim=48`. The resulting cos/sin tensor shape $[1, 1, \text{seq\_len}, 48]$ is **not tile-alignable** in the last dimension, since $48 \not\equiv 0 \pmod{32}$. Attempting to create it as a TILE-layout tensor will fail.

---

## The `nearest_32` Utility

`nearest_32` is defined in the [guide glossary](../index.md#glossary). For `rotary_dim=48`, $\text{nearest\_32}(48) = 64$. In tt-symbiote the utility is implemented as:

```python
def nearest_32(n: int) -> int:
    return math.ceil(n / 32) * 32
```

---

## `TTNNRotaryPositionEmbedding` Padding Behavior

When `rotary_dim % 32 != 0`, `TTNNRotaryPositionEmbedding` calls `ttnn.pad` to extend the cos/sin tensor from shape $[\ldots, \text{rotary\_dim}]$ to $[\ldots, \text{nearest\_32}(\text{rotary\_dim})]$, filling the new positions with zeros:

```python
# Pseudocode reflecting the actual padding path
if rotary_dim % 32 != 0:
    padded_dim = nearest_32(rotary_dim)
    pad_width = padded_dim - rotary_dim          # e.g., 64 - 48 = 16
    cos = ttnn.pad(cos, padding=((0, 0), ..., (0, pad_width)), value=0.0)
    sin = ttnn.pad(sin, padding=((0, 0), ..., (0, pad_width)), value=0.0)
    # cos, sin now have shape [..., padded_dim]  e.g., [..., 64]
```

After this padding the tensor is tile-alignable and can be placed in TILE layout on device.

---

## Intended Semantics of the Zeros

The zeros in positions $[\text{rotary\_dim},\; \text{nearest\_32}(\text{rotary\_dim}))$ are padding, not data. Their intended semantics are:

- "These positions represent no rotation" — i.e., $\cos\theta = 0$ and $\sin\theta = 0$ for those positions.

However, $\cos\theta = 0$ and $\sin\theta = 0$ is **not** a valid "no-op" for the rotate-half formula. The correct no-op is $\cos\theta = 1$ and $\sin\theta = 0$, which recovers $y = x$. Zero-padding produces $y = 0$ for any element multiplied by it.

This means the zero-padding strategy is only safe under one strict condition:

> **The downstream op must read exactly $\text{rotary\_dim}$ elements from the cos/sin tensor — not the full padded width $\text{nearest\_32}(\text{rotary\_dim})$.**

If the op reads the full padded width, elements $[\text{rotary\_dim}, \text{nearest\_32}(\text{rotary\_dim}))$ of $x$ will be zeroed out rather than passed through, silently corrupting the output.

> **[SILENT FAILURE]** There is no error, assertion, or shape mismatch that fires when a downstream op reads beyond `rotary_dim` into the zero-padded region. The op completes successfully and returns a result with the correct shape. The numerical error only becomes apparent through careful comparison against a reference implementation.

---

For a step-by-step element breakdown of this case, see [`partial_rope_math.md`](./partial_rope_math.md).

---

## Forward Reference

Whether `ttnn.experimental.rotary_embedding` actually reads only `rotary_dim` elements or the full padded width is the subject of Chapter 2. If the op reads the full padded width, the 16 elements in $[48, 64)$ will be silently zeroed on every forward pass.

---

**Next:** [Chapter 2 — How `ttnn.experimental.rotary_embedding` Processes cos/sin Shapes](../ch2_op_shape_contract/index.md)

---

## Change Log (B Review Pass 1)
- Corrected cos/sin tensor shape from `[batch, num_heads, seq_len, rotary_dim]` to `[1, 1, seq_len, rotary_dim]`; added note that cos/sin is broadcast across heads (item 1)
- Compression pass 1: collapsed duplicate rotary_dim=48 example tables; reduced nearest_32 section (removed 6-row examples table)
