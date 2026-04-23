# The Correct Partial RoPE Reference Output

This file defines exactly what the correct output should be for a partial RoPE operation with `rotary_dim=48` and `head_dim=128`. It provides the PyTorch reference implementation, derives the element-level formulas for each region of the output, and explains why no zero-padding scheme for cos/sin can make `ttnn.experimental.rotary_embedding` reproduce this reference.

---

## 1. Correct Output Definition

For an input head vector `x` of shape `[B, H, S, 128]`, the correct partial RoPE output is:

- **`output[..., 0:48]`** — rotated region: apply the rotate-half operation to `x[..., 0:48]` using `cos[0:48]` and `sin[0:48]`. The rotate-half split is at `rotary_dim / 2 = 24`. Element `i` in `[0, 24)` is paired with element `i + 24` in `[24, 48)`.
- **`output[..., 48:128]`** — passthrough region: exactly equal to `x[..., 48:128]`, unchanged.

The boundary at position 48 (`rotary_dim`) is absolute. No operation in the correct computation reads or writes elements beyond position 47 for the rotation step.

---

## 2. Element-Level Formulas

Let `c_i = cos[i]` and `s_i = sin[i]` for `i in [0, 48)`. The correct output at each position is:

### Rotated region: `i in [0, 24)` (left half of rotate-half)

$$\text{output}[i] = x[i] \cdot c_i + x[i + 24] \cdot (-s_i)$$

The paired element is `i + 24`, which is still within `[0, 48)`.

### Rotated region: `i in [24, 48)` (right half of rotate-half)

$$\text{output}[i] = x[i - 24] \cdot s_{i-24} + x[i] \cdot c_{i-24}$$

Equivalently, if we let `j = i - 24 in [0, 24)`:

$$\text{output}[j + 24] = x[j] \cdot s_j + x[j + 24] \cdot c_j$$

### Passthrough region: `i in [48, 128)`

$$\text{output}[i] = x[i]$$

No cos, sin, or any other value is applied. The passthrough region is 80 elements wide.

---

## 3. Concrete Example for Five Positions

| Position | Correct output | Notes |
|---|---|---|
| `output[0]` | `x[0]*c_0 + x[24]*(-s_0)` | Paired with `x[24]`; pairing offset is `rotary_dim/2 = 24` |
| `output[24]` | `x[0]*s_0 + x[24]*c_0` | Right-half of pair `(0, 24)`; symmetric formula |
| `output[47]` | `x[23]*s_23 + x[47]*c_23` | Last element in the rotated region |
| `output[48]` | `x[48]` | First passthrough element; cos/sin have no effect |
| `output[127]` | `x[127]` | Last passthrough element; cos/sin have no effect |

---

## 4. PyTorch Reference Implementation

The following code produces the correct partial RoPE output. It operates entirely on the first `rotary_dim` elements and concatenates the untouched passthrough region.

```python
def rotate_half(x):
    # why: standard rotate-half splits the input slice at its own midpoint
    half = x.shape[-1] // 2
    x1 = x[..., :half]   # [B, H, S, rotary_dim/2]  e.g. [B, H, S, 24]
    x2 = x[..., half:]   # [B, H, S, rotary_dim/2]  e.g. [B, H, S, 24]
    return torch.cat([-x2, x1], dim=-1)  # [B, H, S, rotary_dim]


def apply_partial_rope_reference(x, cos, sin, rotary_dim=48):
    # why: slice the input so rotate-half operates only within the rotated region
    x_rot  = x[..., :rotary_dim]   # [B, H, S, 48]  — rotated region
    x_pass = x[..., rotary_dim:]   # [B, H, S, 80]  — passthrough region

    # why: cos and sin have shape [1, 1, S, 48]; broadcast over B and H
    # NOTE: This implementation requires the standard RoPE frequency-duplication
    # property: cos[j+24] == cos[j] and sin[j+24] == sin[j] for all j in [0, 24).
    # Equivalently, the same 24 frequency values appear twice in the 48-element table
    # (first at positions [0, 24) and again at positions [24, 48)).
    # If cos/sin are constructed with 48 distinct monotonically increasing frequencies
    # (no duplication), the right-half outputs [24, 48) will be wrong.
    x_embed = (x_rot * cos) + (rotate_half(x_rot) * sin)
    # x_embed.shape: [B, H, S, 48]

    # why: concatenate the untouched passthrough; no rotation is applied here
    output = torch.cat([x_embed, x_pass], dim=-1)
    # output.shape: [B, H, S, 128]

    return output
```

The `rotate_half` function here splits `x_rot` (which has `shape[-1] = rotary_dim = 48`) at `rotary_dim / 2 = 24`, not at `head_dim / 2 = 64`. This is the critical difference from the TTNN kernel.

---

## 5. Why Zero-Padding cos/sin Cannot Replicate This

### 5a. The kernel's fixed split point

`ttnn.experimental.rotary_embedding` derives its rotate-half split point from the input's last dimension:

```
Wt      = input.padded_shape()[-1] / TILE_WIDTH = 128 / 32 = 4
half_Wt = 4 / 2 = 2  →  64 elements
```

The pairing offset is always 64. There is no runtime argument that changes this to 24.

### 5b. What the kernel computes with fully zero-padded cos/sin

Suppose cos/sin are constructed with the following layout (which is what double zero-padding produces):

```
cos[0:48]    = c_0, c_1, ..., c_47   (real values)
cos[48:128]  = 0.0, 0.0, ..., 0.0    (all zeros)
sin[0:48]    = s_0, s_1, ..., s_47
sin[48:128]  = 0.0, 0.0, ..., 0.0
```

The kernel computes:

```python
# For i in [0, 64):
output[i]      = x[i]      * cos[i]      + x[i + 64] * (-sin[i])
output[i + 64] = x[i]      * sin[i]      + x[i + 64] *   cos[i]
```

Substituting the zero values: see [`step_by_step_failure_trace.md §3c`](./step_by_step_failure_trace.md) for the element-by-element trace (representative positions `output[0]`, `output[24]`, `output[48]`, `output[64]`, `output[88]`, `output[127]` with kernel formula substitutions and corruption verdicts).

In every case the output is wrong. For positions `[0, 24)`, the pairing is across `head_dim/2=64` instead of `rotary_dim/2=24`. For positions `[24, 48)`, the error is structural: the kernel applies the first-half rotation formula (`output[i] = x[i]*c_i + x[i+64]*(-s_i)`) to positions that are the right half of the partial RoPE rotation — requiring different input elements, different trig subscripts, and a different combination rule. For positions `[48, 128)`, the passthrough semantics are violated: elements in `[48, 64)` are zeroed, and elements in `[64, 112)` receive incorrect linear combinations of input values from the left half, and elements in `[112, 128)` are zeroed (since `cos[i]=sin[i]=0` for `i in [48, 64)`).

### 5c. Can a different zero-pattern fix the pairing?

No. The kernel's pairing offset of 64 is determined at kernel-compile time by `Wt / 2`. It cannot be changed by supplying different values in the cos/sin tensor. The only way to make `output[0]` pair with `x[24]` is to restructure the cos/sin values such that the formula

$$\text{output}[0] = x[0] \cdot \text{cos}[0] + x[64] \cdot (-\text{sin}[0])$$

evaluates to the correct rotate-half result. That requires `sin[0]` to encode a rotation that implicitly uses `x[24]` — but `sin[0]` is a scalar applied to `x[64]`, not to `x[24]`. No scalar `sin[0]` can make `x[64] * (-sin[0])` equal `x[24] * (-s_0)` for arbitrary input.

For Strategy C (identity-filled cos/sin), the approach is to set `cos[24:64]=1.0` and `sin[24:64]=0.0` so that positions `[24, 64)` of the first half produce passthrough-like outputs. But elements `[24, 48)` of the **output** are **still wrong**: the kernel computes `output[24] = x[24]*1 + x[88]*0 = x[24]`, but position 24 is in the rotated region `[0, 48)` and its correct output is `x[0]*s_0 + x[24]*c_0`. The `x[24]` result happens to be the passthrough value of position 24 — it is not a correct rotation. The Strategy C construction reduces corruption to only positions `[24, 48)` in exchange for correctly handling the passthrough region; whether this tradeoff is acceptable and how to fully resolve it is analyzed in [`../ch4_implementation_strategies/strategy_c_precomputed_full_head_cos_sin.md`](../ch4_implementation_strategies/strategy_c_precomputed_full_head_cos_sin.md).

> **Key Finding:** Strategy C (Chapter 4) — precomputing a full `head_dim`-wide cos/sin table with identity values at passthrough positions — is the closest viable approach and correctly handles the passthrough region `[48, 128)`, but output positions `[24, 48)` still receive passthrough-like values rather than correct rotations. The full analysis of what Strategy C achieves and how the `[24, 48)` positions are addressed is in Chapter 4.

---

**Next:** [Chapter 4 — Correct Implementation Strategies](../ch4_implementation_strategies/index.md)
