# Recommended Fix — Strategy C: Precomputed Full-Head cos/sin with Identity Fill

The primary recommendation is Strategy C: replace the current `ttnn.pad`-based runtime padding in `TTNNRotaryPositionEmbedding` with a precomputed cos/sin table of shape `[max_seq_len, head_dim]`. The table fills passthrough positions with identity values (`cos=1.0, sin=0.0`) so that the kernel's fixed `head_dim/2`-split pairing produces the correct partial RoPE output for all `rotary_dim <= head_dim` configurations, regardless of tile alignment.

---

> **Key Finding:** Strategy C is the only approach that simultaneously fixes the numerical correctness bug, eliminates the need for runtime `ttnn.pad`, and is fully trace-compatible. It requires changes only to `__init__` (precompute the full-head table) and a minor change to `forward` (remove `ttnn.pad`; pass the precomputed table directly). No runtime overhead is added.

---

## Why Strategy C Is the Right Choice

The root cause established in Chapter 3 is that `ttnn.experimental.rotary_embedding` always splits the head at `head_dim/2` — it cannot be instructed to use `rotary_dim/2` as the pairing offset. Strategy C works with this constraint rather than against it: by placing identity values (`cos=1.0, sin=0.0`) at positions `[rotary_dim/2, head_dim/2)` in the cos/sin table, the kernel's computation at those positions reduces to `output[j] = input[j]` — a passthrough. Real rotation values at positions `[0, rotary_dim/2)` produce the correct rotation for elements in those positions.

Three properties make Strategy C the right production choice:

- **Correctness for all `rotary_dim`.** The construction is valid for any `rotary_dim` that is even and no greater than `head_dim`. There is no tile-alignment requirement on `rotary_dim`.
- **Trace compatibility.** The cos/sin table is a fixed device tensor allocated in `__init__`. No allocation, padding, or concatenation occurs inside `forward`, so the forward pass can run inside a Metal Trace bracket without restriction.
- **No runtime overhead.** The precomputed table is the same size as the table that would be constructed by the correct approach. Passing a pre-existing device tensor to the op is no more expensive than passing the (incorrectly) padded tensor.

---

## Step-by-Step Implementation

### Changes to `__init__`

Replace the current cos/sin precomputation and any tile-alignment padding with the following construction.

```python
def __init__(self, rotary_dim, head_dim, max_seq_len, device, base=10000.0, ...):
    # Preconditions — see precondition_policy.md for full rationale
    assert rotary_dim % 2 == 0, "rotary_dim must be even for rotate-half pairing"
    assert head_dim % 64 == 0, (
        "head_dim must be a multiple of 64; ttnn.experimental.rotary_embedding "
        "requires input.shape[-1] % (TILE_WIDTH * 2) == 0"
    )
    assert rotary_dim <= head_dim, "rotary_dim must not exceed head_dim"
    if rotary_dim % 32 != 0:
        import warnings
        warnings.warn(
            f"rotary_dim={rotary_dim} is not a multiple of 32. "
            "Strategy C handles this correctly, but verify that "
            "partial_rotary_factor is intentional for this model.",
            stacklevel=2,
        )

    rotary_half = rotary_dim // 2   # e.g. 24 for rotary_dim=48
    head_half   = head_dim // 2     # e.g. 64 for head_dim=128

    # --- Compute real frequencies for rotary_half bins ---
    # inv_freq[i] = 1 / base^(2i / rotary_dim) for i in [0, rotary_half)
    # Note: divide by rotary_half (== rotary_dim / 2) to get the factor-of-2 in the exponent.
    # Equivalently: arange * 2.0 / rotary_dim. The two forms are numerically identical.
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_half, dtype=torch.float32) / rotary_half)
    )  # shape: [rotary_half]

    # Positions [0, max_seq_len): each row m gives freqs m * theta_i
    t = torch.arange(max_seq_len, dtype=torch.float32)   # [max_seq_len]
    freqs = torch.outer(t, inv_freq)                     # [max_seq_len, rotary_half]

    # Real rotation values
    cos_real = torch.cos(freqs)   # [max_seq_len, rotary_half]  e.g. [2048, 24]
    sin_real = torch.sin(freqs)   # [max_seq_len, rotary_half]

    # --- Identity values for the passthrough region in the first half ---
    # Positions [rotary_half, head_half) receive identity: cos=1.0, sin=0.0
    # These make the kernel compute output[j] = input[j] * 1.0 + input[j+head_half] * 0.0
    # = input[j], i.e., passthrough for j in [rotary_half, head_half)
    cos_identity_first = torch.ones(max_seq_len, head_half - rotary_half)
    sin_identity_first = torch.zeros(max_seq_len, head_half - rotary_half)

    # --- Assemble the first half of the table (positions [0, head_half)) ---
    cos_first = torch.cat([cos_real, cos_identity_first], dim=-1)  # [max_seq_len, head_half]
    sin_first = torch.cat([sin_real, sin_identity_first], dim=-1)  # [max_seq_len, head_half]

    # --- Assemble the full table (positions [0, head_dim)) ---
    # The second half [head_half, head_dim) mirrors the first half.
    # IMPORTANT: Region 3 values (positions [head_half, head_half+rotary_half)) are
    # NEVER READ by the kernel — the kernel reads cos[j] for j in [0, head_half) only,
    # and uses those same cos[j] values for both output[j] and output[j+head_half].
    # The mirroring here is for table consistency and documentation; it does not
    # affect correctness. See Ch4 strategy_c_precomputed_full_head_cos_sin.md Section 6.
    cos_full = torch.cat([cos_first, cos_first], dim=-1)  # [max_seq_len, head_dim]
    sin_full = torch.cat([sin_first, sin_first], dim=-1)  # [max_seq_len, head_dim]

    # Add batch and head dimensions: [1, 1, max_seq_len, head_dim]
    cos_full = cos_full.unsqueeze(0).unsqueeze(0)
    sin_full = sin_full.unsqueeze(0).unsqueeze(0)

    # Transfer to device once; never reallocated in forward
    self.cos_table = ttnn.from_torch(
        cos_full,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    self.sin_table = ttnn.from_torch(
        sin_full,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
```

### Why the Second Half Mirrors the First Half

The kernel reads `cos[j]` and `sin[j]` for `j in [0, head_half)` only. It applies `cos[j]` to both `output[j]` and `output[j + head_half]` (the latter as the right-half partner in the pair). Values at positions `[head_half, head_dim)` in the cos/sin table are never accessed. The `torch.cat([cos_first, cos_first], dim=-1)` construction sets those positions to duplicate Region 1 and identity values — which is not required for correctness but reflects the correct symmetry of the intended frequency layout.

### Changes to `forward`

Remove the `ttnn.pad` call and pass the precomputed table directly:

```python
def forward(self, x, start_pos, seq_len):
    # Slice the precomputed full-head table to the current sequence position.
    # This is a metadata operation (view/slice), not a buffer allocation —
    # it is safe inside a Metal Trace bracket.
    cos = self.cos_table[:, :, start_pos : start_pos + seq_len, :]
    sin = self.sin_table[:, :, start_pos : start_pos + seq_len, :]

    # cos.shape: [1, 1, seq_len, head_dim]  — satisfies op's shape constraint
    # sin.shape: [1, 1, seq_len, head_dim]
    # No ttnn.pad, no ttnn.concat, no runtime allocation.
    return ttnn.experimental.rotary_embedding(x, cos, sin, start_pos)
```

The `ttnn.pad` call that previously appeared in `forward` is deleted entirely. The cos/sin tensors are already `head_dim`-wide from `__init__`.

---

## Precondition to Add

See [`precondition_policy.md`](./precondition_policy.md) for the complete specification of required assertions and warnings. In brief: assert `head_dim % 64 == 0`, `rotary_dim % 2 == 0`, and `rotary_dim <= head_dim`; emit a warning (not an error) when `rotary_dim % 32 != 0`. Do NOT add an `assert rotary_dim % 32 == 0` or `assert rotary_dim % 64 == 0` guard — Strategy C makes such a guard unnecessary.

---

## Summary of Changes

| Location | Change |
|---|---|
| `__init__` — cos/sin precomputation | Replace `rotary_dim`-wide table + conditional `nearest_32` padding with Strategy C construction using `rotary_half`, `head_half`, `cos_identity_first`, `sin_identity_first`, and `torch.cat` to produce `head_dim`-wide tables |
| `__init__` — preconditions | Add `assert head_dim % 64 == 0`; add `assert rotary_dim % 2 == 0`; add `assert rotary_dim <= head_dim`; add warning when `rotary_dim % 32 != 0` |
| `forward` | Delete `ttnn.pad` call; pass `self.cos_table` and `self.sin_table` directly (sliced to current seq position) |
| `forward` — no other changes | All other logic (sequence slicing, token_idx, output handling) is unchanged |

---

## What's Next

[`precondition_policy.md`](./precondition_policy.md) specifies which constraints are hard requirements, which should warn, and the short-term Strategy B option if Strategy C is not yet deployed.
