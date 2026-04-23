# Partial RoPE Math

This file derives the rotate-half formulation of Rotary Position Embedding (RoPE) from first principles and then extends it to the partial case where only a prefix of each attention head is rotated. A concrete worked example with `rotary_dim=48, head_dim=128` is used throughout to ground the algebra. Understanding this math precisely is a prerequisite for diagnosing any numerical error in a partial RoPE implementation.

---

## Standard Rotate-Half Formulation

Given a head vector $x \in \mathbb{R}^{d}$ (where $d = \text{head\_dim}$) and position-dependent tensors $\cos\theta \in \mathbb{R}^{d}$ and $\sin\theta \in \mathbb{R}^{d}$, the standard RoPE output is:

$$y = x \odot \cos\theta + \text{rotate\_half}(x) \odot \sin\theta$$

where $\odot$ denotes element-wise multiplication and $\text{rotate\_half}$ is defined as:

$$\text{rotate\_half}(x) = \text{cat}\!\left([-x_{[d/2 : d]},\; x_{[0 : d/2]}],\; \text{dim}=-1\right)$$

In words: take the second half of $x$, negate it, prepend it to the first half of $x$. This implements a block-diagonal rotation matrix in $\mathbb{R}^{d}$ that pairs element $i$ with element $i + d/2$ for each $i \in [0, d/2)$.

In Python, using PyTorch notation:

```python
def rotate_half(x):
    # x: [..., head_dim]
    x1 = x[..., : x.shape[-1] // 2]   # [..., head_dim/2]
    x2 = x[..., x.shape[-1] // 2 :]   # [..., head_dim/2]
    return torch.cat([-x2, x1], dim=-1)  # [..., head_dim]

def apply_rope(x, cos, sin):
    # x, cos, sin: [..., head_dim]
    return x * cos + rotate_half(x) * sin
```

> **Note:** The rotate-half split is always computed with respect to the length of the slice being rotated, not with respect to `head_dim` as a whole. In full RoPE these are the same quantity; in partial RoPE they differ.

---

## Partial RoPE Extension

When $\text{rotary\_dim} < \text{head\_dim}$, only the first $\text{rotary\_dim}$ elements of each head receive positional encoding. The remaining $\text{head\_dim} - \text{rotary\_dim}$ elements pass through unchanged.

Let $x = [x_\text{rot} \;|\; x_\text{pass}]$ where:

- $x_\text{rot} = x_{[0:\text{rotary\_dim}]} \in \mathbb{R}^{\text{rotary\_dim}}$ — the slice that will be rotated.
- $x_\text{pass} = x_{[\text{rotary\_dim}:\text{head\_dim}]} \in \mathbb{R}^{\text{head\_dim} - \text{rotary\_dim}}$ — the passthrough slice.

The correct partial RoPE output is:

$$y = \text{cat}\!\left[\, x_\text{rot} \odot \cos\theta + \text{rotate\_half}(x_\text{rot}) \odot \sin\theta,\;\; x_\text{pass} \,\right]$$

where $\cos\theta, \sin\theta \in \mathbb{R}^{\text{rotary\_dim}}$ are defined only over the rotated region, and $\text{rotate\_half}$ is applied to $x_\text{rot}$ alone:

$$\text{rotate\_half}(x_\text{rot}) = \text{cat}\!\left([-x_\text{rot,\,[r/2:r]},\; x_\text{rot,\,[0:r/2]}],\; \text{dim}=-1\right)$$

where $r = \text{rotary\_dim}$.

In Python:

```python
def apply_partial_rope(x, cos, sin, rotary_dim):
    # x:   [..., head_dim]
    # cos, sin: [..., rotary_dim]
    x_rot  = x[..., :rotary_dim]           # [..., rotary_dim]
    x_pass = x[..., rotary_dim:]           # [..., head_dim - rotary_dim]

    x_rot_out = x_rot * cos + rotate_half(x_rot) * sin  # rotate_half uses rotary_dim

    return torch.cat([x_rot_out, x_pass], dim=-1)        # [..., head_dim]
```

> **Key Finding:** The `rotate_half` split must be $\text{rotary\_dim} / 2$, not $\text{head\_dim} / 2$. Using $\text{head\_dim} / 2$ as the split point would pair element 0 with element $\text{head\_dim}/2$, which lies in $x_\text{pass}$ when $\text{rotary\_dim} < \text{head\_dim} / 2$, or at least in the wrong position within the rotated region otherwise.

---

## Worked Example: `rotary_dim=48, head_dim=128`

The following quantities hold throughout this example:

| Symbol | Value |
|---|---|
| $\text{head\_dim}$ | 128 |
| $\text{rotary\_dim}$ | 48 |
| $\text{head\_dim} - \text{rotary\_dim}$ | 80 |
| $\text{rotary\_dim} / 2$ | 24 |

**Element assignment:**

- Elements $0$ through $47$ (48 total): rotated. These are $x_\text{rot}$.
- Elements $48$ through $127$ (80 total): passed through unchanged. These are $x_\text{pass}$.

**Rotate-half pairing within $x_\text{rot}$:**

The split point within the rotated slice is at index 24 (= $48 / 2$). The pairing is:

$$\forall\, i \in [0, 24):\quad (x[i],\; x[i + 24])$$

Concretely:

$$\text{rotate\_half}(x_\text{rot}) = [-x[24],\; -x[25],\; \ldots,\; -x[47],\; x[0],\; x[1],\; \ldots,\; x[23]]$$

The full output head vector $y \in \mathbb{R}^{128}$ is assembled as:

$$y[i] = \begin{cases} x[i] \cdot \cos\theta[i] - x[i+24] \cdot \sin\theta[i] & 0 \le i < 24 \\ x[i] \cdot \cos\theta[i] + x[i-24] \cdot \sin\theta[i] & 24 \le i < 48 \\ x[i] & 48 \le i < 128 \end{cases}$$

> **Warning:** The cos/sin tensor must have shape $[\ldots, 48]$ for this example — covering exactly `rotary_dim` elements. If the cos/sin tensor has shape $[\ldots, 64]$ (padded to the next multiple of 32) and the op applies the full 64-element cos/sin to the first 64 elements of $x$, elements 48 through 63 would be multiplied by the padding values (zeros) rather than passed through, and the rotate-half pairing would also be computed over width 64 instead of 48, corrupting elements 0 through 15 and 32 through 47.

---

## Key Invariant

The correctness of partial RoPE depends on a single invariant:

> The cos/sin tensor must cover **exactly** $\text{rotary\_dim}$ elements, and the rotate-half split must be computed as $\text{rotary\_dim} / 2$, operating entirely within the rotated slice.

Any deviation — whether from an incorrect tensor shape, a misaligned split, or a downstream op that reads beyond $\text{rotary\_dim}$ — produces silently wrong outputs for every token in every sequence processed by the affected attention head.

---

**Next:** [`tile_alignment_in_ttnn.md`](./tile_alignment_in_ttnn.md)
