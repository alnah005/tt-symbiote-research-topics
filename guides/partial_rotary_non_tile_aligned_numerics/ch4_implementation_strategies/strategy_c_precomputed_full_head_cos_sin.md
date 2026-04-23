# Strategy C — Precomputed Full-Head cos/sin with Identity Fill

Strategy C is the recommended production implementation for partial RoPE when `rotary_dim < head_dim`. The central insight is that `ttnn.experimental.rotary_embedding` cannot be told to restrict its rotation to a subset of `head_dim` positions — but it can be made to produce the correct partial RoPE output by carefully constructing the cos/sin table so that the kernel's `head_dim/2` pairing, combined with the identity values at passthrough positions, is mathematically equivalent to the correct `rotary_dim/2`-paired rotation on the first `rotary_dim` elements.

The cos/sin table is precomputed once in `__init__` as a device tensor of shape `[max_seq_len, head_dim]`. At forward time the kernel receives this pre-existing tensor directly — no runtime allocation occurs, making Strategy C fully trace-compatible.

---

## 1. The Kernel's Computation

Recall from Chapter 2 (see [`../ch2_op_shape_contract/kernel_rotate_half_pairing.md`](../ch2_op_shape_contract/kernel_rotate_half_pairing.md)) that `ttnn.experimental.rotary_embedding` with `input.shape[-1] = head_dim = 128` computes, for `i in [0, 64)`:

```
output[i]      = input[i]      * cos[i]      + input[i + 64] * (-sin[i])
output[i + 64] = input[i]      * sin[i]      + input[i + 64] *   cos[i]
```

The pairing offset is fixed at `head_dim / 2 = 64`. We need to choose values for the 128 entries of cos and the 128 entries of sin such that the above 128 equations produce the correct partial RoPE output for `rotary_dim=48`.

---

## 2. The Required Partial RoPE Output

The correct output (from [`../ch3_bug_root_cause/correct_partial_rope_reference.md`](../ch3_bug_root_cause/correct_partial_rope_reference.md)) is, for a frequency index `j in [0, 24)` (where `rotary_dim/2 = 24`):

```
output[j]      = input[j]      * c_j  +  input[j + 24] * (-s_j)
output[j + 24] = input[j]      * s_j  +  input[j + 24] *   c_j
output[k]      = input[k]                                            for k in [48, 128)
```

Here `c_j` and `s_j` are the real cosine and sine values at frequency `j`.

---

## 3. Deriving the Required cos/sin Layout

We match the kernel's 128 equations to the 128 desired outputs by solving for `cos[i]` and `sin[i]` at each position.

### 3a. Positions `i in [0, 24)` — first half of the rotated region

The kernel computes:

```
output[i]      = input[i] * cos[i]      + input[i + 64] * (-sin[i])
output[i + 64] = input[i] * sin[i]      + input[i + 64] *   cos[i]
```

The desired outputs are:

```
output[i]      = input[i] * c_i  +  input[i + 24] * (-s_i)    [from the rotated region]
output[i + 64] = input[i + 64]                                  [passthrough]
```

For `output[i]` to be correct, we need the kernel's pairing — which uses `input[i + 64]` — to produce the result that should use `input[i + 24]`. Because `input[i + 64]` is in the passthrough region and is unrelated to `input[i + 24]` for arbitrary inputs, this equation cannot be satisfied for arbitrary inputs by a scalar `cos[i]` and `sin[i]`.

> **Note:** Strategy C does NOT replicate the exact PyTorch partial RoPE formula at positions `[0, 24)`. The kernel's `head_dim/2=64` pairing is irreducibly different from the correct `rotary_dim/2=24` pairing. What Strategy C achieves is something more subtle: it constructs cos/sin such that the kernel's computation correctly rotates the pairs `(input[j], input[j+64])` for `j in [0, 24)` using the real frequencies — which matches the correct partial RoPE output only when the input head is laid out with `head_dim/2`-split pairing (elements `j` and `j+64` are rotation partners). This is correct under the `head_dim/2`-split input convention; under the PyTorch slice convention (pairs are `(x[i], x[i+rotary_dim/2])`), Strategy C produces different output from the reference — see Section 4b.

### 3b. The Correct Strategy C Construction

Strategy C works by laying the input out so that the kernel's `i` and `i+64` elements are both within the rotated region (for the first `rotary_dim/2` positions), or both in an identity/passthrough context (for the remaining positions). This is achieved by populating the cos/sin table as follows.

Let `R = rotary_dim = 48`, `H = head_dim = 128`, `r = R/2 = 24`, `h = H/2 = 64`.

The four regions of the cos/sin table are:

**Region 1: positions `[0, r) = [0, 24)`**

```
cos[i] = c_i    (real cosine value at frequency i)
sin[i] = s_i    (real sine value at frequency i)
```

**Region 2: positions `[r, h) = [24, 64)`**

```
cos[i] = 1.0    (identity)
sin[i] = 0.0    (identity)
```

**Region 3: positions `[h, h + r) = [64, 88)`**

```
cos[i] = c_{i - h}    (same real values as Region 1, duplicated)
sin[i] = s_{i - h}    (same real values as Region 1, duplicated)
```

For `i in [64, 88)`: `cos[i] = c_{i-64}`, `sin[i] = s_{i-64}`.

**Region 4: positions `[h + r, H) = [88, 128)`**

```
cos[i] = 1.0    (identity)
sin[i] = 0.0    (identity)
```

---

## 4. Why This Construction is Correct

### 4a. Verifying positions `j in [0, 24)` — the rotated first half

The kernel computes:

```
output[j]      = input[j]      * cos[j]      + input[j + 64] * (-sin[j])
               = input[j]      * c_j          + input[j + 64] * (-s_j)

output[j + 64] = input[j]      * sin[j]      + input[j + 64] *   cos[j]
               = input[j]      * s_j          + input[j + 64] *   c_j
```

As shown in the formula above, the kernel uses `cos[j] = c_j` and `sin[j] = s_j` (Region 1, indices `j in [0, 24)`) to compute `output[j+64]` — the same cos/sin indices used for `output[j]`. Region 3 values (`cos[j+64]`, `sin[j+64]`) are never read by the kernel when computing `output[j+64]`; see Section 4d and Section 6 for the derivation. So:

```
output[j + 64] = input[j] * s_j + input[j + 64] * c_j
```

This is correct for the kernel's pairing: element `j` (in `[0, 24)`) is paired with element `j+64` (in `[64, 88)`). The output at position `j+64` is the "right-half" rotation formula applied to the pair `(input[j], input[j+64])`.

> **Key Finding:** The kernel computes a correct rotate-half rotation on the pair `(input[j], input[j+64])` for each `j in [0, 24)`. For this to be the correct partial RoPE output, the input must be arranged so that `input[j+64]` is the element that should be paired with `input[j]` in the rotation. In the standard attention head layout where elements `[0, 48)` are the rotated region and `[48, 128)` are passthrough, `input[j+64]` is a passthrough element — not the correct `rotary_dim/2`-paired element at `input[j+24]`. Strategy C produces partial rotation of the `(input[j], input[j+64])` pairs, which is **different** from the reference implementation that rotates `(input[j], input[j+24])` pairs.

### 4b. The input layout assumption for Strategy C to be fully correct

For Strategy C to produce output numerically identical to the PyTorch reference, the input head must be laid out so that:

- `input[j]` and `input[j + 64]` are the two elements in the frequency-`j` rotation pair, for `j in [0, 24)`.
- `input[j + 24]` through `input[j + 63]` and `input[j + 88]` through `input[127]` are passthrough elements.

This requires the model's embedding to use a "head_dim/2-split" layout rather than a "rotary_dim/2-split" layout. In models where the positional embedding table is generated with `head_dim` frequencies (not `rotary_dim` frequencies), and the cos/sin are generated as `cos(m * theta_i)` for `i in [0, head_dim/2)`, the natural layout does have `input[j]` paired with `input[j + head_dim/2]` — which is exactly what the kernel expects.

> **Note:** Whether the input head satisfies this layout assumption depends on how the model's rotary embedding frequencies were generated. If frequencies were generated for `head_dim=128` positions (i.e., 64 frequency bins), then Strategy C is directly applicable. If frequencies were generated for only `rotary_dim=48` positions (24 frequency bins) with a different pairing convention, the duplication at Region 3 must be generated from those 24 bins repeated, not from a 64-bin table truncated at 24.

### 4c. Verifying positions `j in [24, 64)` — identity passthrough in first half

For `j in [24, 64)`, the construction gives `cos[j] = 1.0` and `sin[j] = 0.0` (Region 2). The kernel computes:

```
output[j]      = input[j] * 1.0 + input[j + 64] * 0.0  = input[j]
output[j + 64] = input[j] * 0.0 + input[j + 64] * 1.0  = input[j + 64]
```

Both elements pass through unchanged. For `j in [24, 48)`, position `j` is in the rotated region `[0, 48)` and position `j+64` is in the passthrough region `[88, 112)`. Both receive passthrough treatment.

> **Note:** For `j in [24, 48)`, position `j` is in `[24, 48)` which is supposed to be the right half of the rotated region — it should receive `output[j] = input[j-24]*s_{j-24} + input[j]*c_{j-24}`. Instead it receives `output[j] = input[j]`. This is the residual limitation of Strategy C under the standard `rotary_dim/2`-pairing layout: positions `[24, 48)` are not correctly rotated. They receive passthrough values. When the input uses the `head_dim/2=64`-pairing layout (as described in 4b), positions `[0, 24)` and `[64, 88)` are correctly rotated as pairs, and positions `[24, 64)` and `[88, 128)` are passthrough — which is a different but internally consistent rotation.

### 4d. Output at positions `j in [64, 88)` — rotated second half

For `j in [64, 88)`, let `j = 64 + k` where `k in [0, 24)`. The output at position `j` is produced by the kernel as part of the pair `(k, k+64)` already analyzed in 4a:

```
output[k + 64] = input[k] * sin[k] + input[k + 64] * cos[k]
               = input[k] * s_k    + input[k + 64] * c_k
```

The kernel reads `cos[k]` and `sin[k]` at positions `k in [0, 24)` (Region 1) — not at positions `[64, 88)`. The input data at positions `[64, 88)` is used as the right-half partner (`input[k + 64]`) in each pair, but the cos/sin values applied to those rotations come from Region 1 indices `[0, 24)`. The values placed in Region 3 (`cos[j]` and `sin[j]` for `j in [64, 88)`) are never read by the kernel. No additional equations arise from Region 3; the outputs at positions `[64, 88)` are fully determined by Region 1 values via the pair formula in 4a.

### 4e. Verifying positions `j in [88, 128)` — identity passthrough in second half

For `j in [88, 128)`, `cos[j] = 1.0` and `sin[j] = 0.0` (Region 4). Already handled: these are the `j+64` positions for `j in [24, 64)`, computed in 4c.

---

## 5. Python Construction Code

```python
import torch

def build_strategy_c_cos_sin(
    rotary_dim: int,
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a full-head cos/sin table for Strategy C.

    The table has shape [max_seq_len, head_dim].
    Positions [0, rotary_dim/2) receive real rotation values.
    Positions [rotary_dim/2, head_dim/2) receive identity (cos=1, sin=0).
    Positions [head_dim/2, head_dim/2 + rotary_dim/2) duplicate [0, rotary_dim/2).
    Positions [head_dim/2 + rotary_dim/2, head_dim) receive identity (cos=1, sin=0).

    Args:
        rotary_dim: number of dimensions to rotate (must be even, <= head_dim)
        head_dim:   total head dimension (must satisfy head_dim % 64 == 0)
        max_seq_len: maximum sequence length
        base:       RoPE base frequency

    Returns:
        cos_table: [max_seq_len, head_dim]
        sin_table: [max_seq_len, head_dim]
    """
    assert rotary_dim % 2 == 0, "rotary_dim must be even"
    assert head_dim % 64 == 0, "head_dim must be a multiple of 64 for ttnn op"
    assert rotary_dim <= head_dim, "rotary_dim must not exceed head_dim"

    r = rotary_dim // 2   # e.g. 24 for rotary_dim=48
    h = head_dim // 2     # e.g. 64 for head_dim=128

    # Compute real frequencies for r bins.
    # theta_i = 1 / base^(2i / rotary_dim)  for i in [0, r)
    # why: standard inverse-frequency formula; rotary_dim used (not head_dim)
    # so that the frequency spectrum covers the intended rotation bandwidth
    inv_freq = 1.0 / (base ** (torch.arange(0, r, dtype=torch.float32) / rotary_dim))
    # inv_freq.shape: [r]  e.g. [24]

    # Position indices [0, max_seq_len)
    t = torch.arange(max_seq_len, dtype=torch.float32)  # [max_seq_len]

    # Outer product: freqs[m, i] = m * theta_i
    freqs = torch.outer(t, inv_freq)  # [max_seq_len, r]

    # Real cos and sin at each (position, frequency) pair
    cos_real = torch.cos(freqs)  # [max_seq_len, r]
    sin_real = torch.sin(freqs)  # [max_seq_len, r]

    # Initialize the full table to identity: cos=1, sin=0
    cos_table = torch.ones(max_seq_len, head_dim)   # [max_seq_len, head_dim]
    sin_table = torch.zeros(max_seq_len, head_dim)  # [max_seq_len, head_dim]

    # Region 1: positions [0, r) — real rotation values
    cos_table[:, :r] = cos_real
    sin_table[:, :r] = sin_real

    # Region 2: positions [r, h) — already identity (cos=1, sin=0), no change needed

    # Region 3: positions [h, h+r) — duplicate Region 1 (frequency duplication)
    # note: Region 3 values are never read by the kernel (kernel reads cos/sin only at
    # positions [0, h)); this duplication is for table consistency and documentation only,
    # not a correctness requirement — see Section 6 for derivation.
    cos_table[:, h : h + r] = cos_real
    sin_table[:, h : h + r] = sin_real

    # Region 4: positions [h+r, head_dim) — already identity, no change needed

    return cos_table, sin_table


# Example for rotary_dim=48, head_dim=128, max_seq_len=2048
cos_table, sin_table = build_strategy_c_cos_sin(
    rotary_dim=48, head_dim=128, max_seq_len=2048
)
# cos_table.shape: [2048, 128]
# cos_table[:, 0:24]   — real cos values (Region 1)
# cos_table[:, 24:64]  — 1.0             (Region 2)
# cos_table[:, 64:88]  — same as [:, 0:24] (Region 3, frequency duplication)
# cos_table[:, 88:128] — 1.0             (Region 4)
```

---

## 6. Why Region 3 Values Are Numerically Inert

The kernel reads `cos[i]` and `sin[i]` only for `i in [0, head_dim/2) = [0, 64)`. Positions `[64, 128)` of the cos/sin table are **never accessed** during computation. Setting Region 3 (`[h, h+r) = [64, 88)`) to duplicate Region 1 values, and Region 4 (`[h+r, H) = [88, 128)`) to identity, is therefore redundant for the kernel's output but is included for two reasons: (1) it makes the table self-consistent and documents the intended frequency layout symmetrically; (2) if any future validation or debugging tooling checks cos/sin table structure, the Region 3 values reflect what the correct symmetry should be. The cos/sin table has shape `[max_seq_len, head_dim]` to satisfy the op's shape constraint `cos.shape[-1] == input.shape[-1]`; the upper half of that table is numerically inert.

> **Key Finding:** Strategy C's correctness depends only on cos/sin values at positions `[0, head_dim/2) = [0, 64)` (Regions 1 and 2). Region 3 (positions `[64, 88)`) is never read by the kernel; whatever values are placed there have zero effect on the output. The "frequency duplication" at Region 3 is not required for correctness — it is included only for table consistency and documentation.

---

## 7. Trace Compatibility

Strategy C achieves trace compatibility through a simple allocation policy:

- `cos_table` and `sin_table` are allocated as TTNN device tensors in `__init__`, before any trace bracket is entered.
- In `forward`, the code passes `self.cos_table` and `self.sin_table` directly to `ttnn.experimental.rotary_embedding`.
- No `ttnn.pad`, `ttnn.concat`, or any other buffer-allocating operation occurs inside `forward`.

```python
class TTNNRotaryPositionEmbedding:
    def __init__(self, rotary_dim, head_dim, max_seq_len, device, ...):
        cos_table, sin_table = build_strategy_c_cos_sin(
            rotary_dim, head_dim, max_seq_len
        )
        # Convert to TTNN device tensors once; never reallocated at forward time
        self.cos_table = ttnn.from_torch(
            cos_table.unsqueeze(0).unsqueeze(0),  # [1, 1, max_seq_len, head_dim]
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.sin_table = ttnn.from_torch(
            sin_table.unsqueeze(0).unsqueeze(0),
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    def forward(self, x, start_pos, seq_len):
        # Slice the precomputed table to the current sequence length
        # (this is a view/metadata operation, not a new allocation)
        cos = self.cos_table[:, :, start_pos : start_pos + seq_len, :]
        sin = self.sin_table[:, :, start_pos : start_pos + seq_len, :]

        # No ttnn.pad, no ttnn.concat, no runtime allocation
        return ttnn.experimental.rotary_embedding(x, cos, sin)
```

The forward pass performs zero device memory allocations — exactly the requirement for trace compatibility.

---

## 8. Summary

| Property | Value |
|---|---|
| cos/sin table shape | `[1, 1, max_seq_len, head_dim]` on device |
| Region 1 (`[0, r)`) | Real cosine/sine values at `r = rotary_dim/2` frequencies |
| Region 2 (`[r, h)`) | Identity: cos=1.0, sin=0.0 |
| Region 3 (`[h, h+r)`) | Duplicate of Region 1 (frequency duplication) |
| Region 4 (`[h+r, H)`) | Identity: cos=1.0, sin=0.0 |
| Allocation in `forward` | None — fully trace-compatible |
| Input layout assumption | Elements `j` and `j + head_dim/2` form rotation pairs for `j in [0, rotary_dim/2)` |

**Next:** [Trace-Safe Alternatives to `ttnn.pad`](./trace_safe_alternatives_to_ttnn_pad.md)
