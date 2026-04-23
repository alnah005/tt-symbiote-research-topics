# What the Golden Function Reveals

This file examines the Python golden function for `ttnn.experimental.rotary_embedding` defined in `ttnn/ttnn/operations/transformer.py`. The golden function is the reference implementation used to compute expected outputs for PCC (Pearson Correlation Coefficient) tests. Its shape assumptions corroborate the C++ constraints described in the previous two files and provide a clean, readable statement of the overall shape contract.

---

## 1. The `rotate_half` Helper

The golden function defines a local `rotate_half` helper that computes the right-half negation required by the rotate-half formula:

```python
def rotate_half(x):
    """Rotate the last dimension by half."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]   # elements [0, x.shape[-1]/2)
    x2 = x[..., half:]   # elements [x.shape[-1]/2, x.shape[-1])
    return torch.cat((-x2, x1), dim=-1)
```

The split point is `x.shape[-1] // 2` — the midpoint of **whatever tensor is passed in**. The function itself is agnostic to whether it receives a full `head_dim`-wide tensor or a `rotary_dim`-wide slice. The pairing convention is determined by the caller, not by `rotate_half` itself.

> **Key point:** The Python golden passes the **rotary-dim-sliced** input to `rotate_half`, not the full `head_dim`-wide tensor. See Section 2 for how this slicing is performed before `rotate_half` is called.

For `rotary_dim=48` (the Qwen3 example), `rotate_half` receives a `[..., 48]` tensor, so the split lands at index 24 = `rotary_dim/2`, not at 64 = `head_dim/2`.

$$\text{rotate\_half}(x_\text{rot})_i = \begin{cases} -x_\text{rot,\,i+24} & i \in [0, 24) \\ x_\text{rot,\,i-24} & i \in [24, 48) \end{cases}$$

This is the partial-dimension rotate-half operating within the rotated slice alone, consistent with the partial RoPE math derived in Ch1.

---

## 2. The Embedding Application

The main golden function applies the embedding as:

```python
def golden_rotary_embedding(
    input: torch.Tensor,
    cos_cached: torch.Tensor,
    sin_cached: torch.Tensor,
    token_index: int,
    rotary_dim: int,
) -> torch.Tensor:
    # Slice on sequence dimension AND restrict to rotary_dim elements
    cos_slice = cos_cached[:, :, token_index : token_index + 1, :rotary_dim]
    sin_slice = sin_cached[:, :, token_index : token_index + 1, :rotary_dim]

    # Separate the rotated and passthrough portions of the input
    x_rot  = input[..., :rotary_dim]   # [..., rotary_dim] — will be rotated
    x_pass = input[..., rotary_dim:]   # [..., head_dim - rotary_dim] — unchanged

    # rotate_half receives a [..., rotary_dim] tensor; split is at rotary_dim/2
    x_rot_out = (x_rot * cos_slice) + (rotate_half(x_rot) * sin_slice)

    return torch.cat([x_rot_out, x_pass], dim=-1)
```

Three observations:

1. **Sequence-and-dimension slicing:** `cos_cached[:, :, token_index : token_index + 1, :rotary_dim]` slices on both the sequence dimension (dim 2) and the last dimension (dim 3, restricted to `rotary_dim`). The resulting `cos_slice` has shape `[1, 1, 1, rotary_dim]`, not `[1, 1, 1, head_dim]`.

2. **Rotary-dim-sliced input into `rotate_half`:** `rotate_half` is called with `x_rot` — the `[..., rotary_dim]` slice — not the full `head_dim`-wide tensor. Therefore `x.shape[-1]` inside `rotate_half` equals `rotary_dim`, and the split lands at `rotary_dim/2 = 24` for the Qwen3 example. This is the `rotary_dim/2` convention described in Ch1.

3. **Passthrough concatenation:** `x_pass` is concatenated back unchanged, assembling the full `[batch, heads, 1, head_dim]` output.

> **Note:** The `cos_cached.shape[-1]` requirement depends on what shape the cache is built with. The golden pre-slices to `rotary_dim`, so the cache need only have `shape[-1] >= rotary_dim`. The C++ `invoke` and `validate` functions impose the stricter constraint `cos_cached.shape[-1] == head_dim` — this is the op-level shape contract, distinct from what the Python golden needs for its own computation. The Key Finding in Section 3 describes the op-level constraint.

---

## 3. Summary of the Shape Contract

The golden function makes the shape contract explicit and readable. Combining it with the C++ findings from the previous two files:

| Constraint | Location | Expression |
|---|---|---|
| `input.padded_shape()[-1] % 64 == 0` | `invoke` (C++) | `head_dim` must be a multiple of 64 |
| `cos.padded_shape()[-1] == input.padded_shape()[-1]` | `invoke` (C++) | `cos.shape[-1] == head_dim` |
| `sin.padded_shape()[-1] == input.padded_shape()[-1]` | `invoke` (C++) | `sin.shape[-1] == head_dim` |
| `cos.padded_shape()[-1] == X` | `validate` (C++) | same as invoke, enforced again |
| `rotate_half(x_rot)` uses `x_rot.shape[-1] // 2` | golden (Python) | split is at `rotary_dim / 2` (x_rot has shape `[..., rotary_dim]`) |
| `cos_slice = cos_cached[..., :rotary_dim]` | golden (Python) | takes only the first `rotary_dim` elements of cache |

All constraints point to the same requirement:

> **Key Finding:** `ttnn.experimental.rotary_embedding` requires `cos.shape[-1] == sin.shape[-1] == input.shape[-1] == head_dim`. This constraint is enforced by `TT_FATAL` in the C++ `invoke` and `validate` functions and is corroborated by the Python golden function. The `rotary_dim` argument does not relax this constraint. Passing cos/sin tensors with `shape[-1] == rotary_dim` or `shape[-1] == nearest_32(rotary_dim)` when `rotary_dim != head_dim` will cause a fatal error before any compute occurs.

---

## 4. What This Means for the PCC ~0.71 Bug

The golden function and the C++ constraints together explain why the PCC ~0.71 bug arises:

- A caller implementing partial RoPE for `rotary_dim=48`, `head_dim=128` might attempt to pass cos/sin with `shape[-1]=48` (raw) or `shape[-1]=64` (tile-padded). Both are rejected by `TT_FATAL` in `invoke`.
- If the caller works around the shape check by zero-padding cos to `shape[-1]=128` (padding positions $[48, 128)$ with zeros), the cos values at those positions are 0 instead of 1. The rotate-half formula then computes:

  $$x'_i = x_i \cdot 0 - x_{i+64} \cdot 0 = 0 \quad \text{for } i \in [48, 64)$$

  instead of the correct passthrough $x'_i = x_i$. This corrupts roughly half the elements in the right tail of each head, producing a PCC of approximately 0.71 rather than 1.0.

The root cause analysis is completed in Chapter 3. The fix (Strategy C) is analyzed in Chapter 4.

---

**Next:** [Chapter 3 — Root Cause Analysis of the PCC ~0.71 Bug](../ch3_bug_root_cause/index.md)
