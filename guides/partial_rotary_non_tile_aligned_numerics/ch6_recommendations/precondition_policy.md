# Precondition Policy for TTNNRotaryPositionEmbedding

This file specifies exactly which constraints `TTNNRotaryPositionEmbedding` must enforce, which should emit warnings, and whether Strategy B (enforce `rotary_dim % 32 == 0`) remains appropriate after Strategy C is implemented. It directly answers the research question: "Should `TTNNRotaryPositionEmbedding` enforce `rotary_dim % 32 == 0`?"

---

> **Key Finding:** Do NOT enforce `rotary_dim % 32 == 0` as a hard error. Strategy C handles non-tile-aligned `rotary_dim` correctly. The constraint that must be enforced is `head_dim % 64 == 0` — this comes from the op's two-tile constraint and cannot be relaxed. Enforce `rotary_dim % 2 == 0` and `rotary_dim <= head_dim` as additional hard requirements. Emit a warning (not an error) when `rotary_dim % 32 != 0` to surface unexpected configurations early without blocking valid ones.

---

## Hard Requirements (Must Enforce with assert or ValueError)

### 1. head\_dim % 64 == 0

This constraint comes directly from `ttnn.experimental.rotary_embedding`, which asserts:

```
TT_FATAL(input_tensor.padded_shape()[-1] % (TILE_WIDTH * 2) == 0, ...)
```

where `TILE_WIDTH = 32`, so `TILE_WIDTH * 2 = 64`. The input tensor's last dimension is `head_dim`. No choice of cos/sin table can work around this constraint — it is enforced in C++ before any compute runs.

```python
assert head_dim % 64 == 0, (
    f"head_dim={head_dim} must be a multiple of 64. "
    "ttnn.experimental.rotary_embedding requires the input last dimension "
    "to satisfy head_dim % (TILE_WIDTH * 2) == 0."
)
```

### 2. rotary\_dim % 2 == 0

The rotate-half operation pairs element `i` with element `i + rotary_dim/2`. This pairing is only well-defined when `rotary_dim` is even. An odd `rotary_dim` leaves one element without a partner and the rotation is undefined.

```python
assert rotary_dim % 2 == 0, (
    f"rotary_dim={rotary_dim} must be even. "
    "The rotate-half pairing requires rotary_dim / 2 to be an integer."
)
```

### 3. rotary\_dim <= head\_dim

Partial RoPE applies rotation to the first `rotary_dim` elements of a head of size `head_dim`. If `rotary_dim > head_dim`, there are not enough elements to satisfy the rotation region, and the passthrough region would have negative size.

```python
assert rotary_dim <= head_dim, (
    f"rotary_dim={rotary_dim} must not exceed head_dim={head_dim}. "
    "Partial RoPE rotates only the first rotary_dim elements of each head."
)
```

---

## Warning (Should Warn, Must Not Error)

### rotary\_dim % 32 != 0

After Strategy C is implemented, non-tile-aligned `rotary_dim` is handled correctly. A hard error here would unnecessarily block valid configurations. However, non-tile-aligned `rotary_dim` values are unexpected in the current set of production models — if they appear, it is likely due to an unusual `partial_rotary_factor` that may warrant attention from the engineer bringing up the model.

```python
if rotary_dim % 32 != 0:
    import warnings
    warnings.warn(
        f"rotary_dim={rotary_dim} is not a multiple of 32. "
        "Strategy C will handle this correctly, but verify that "
        f"partial_rotary_factor={rotary_dim / head_dim:.4f} is the "
        "intended configuration for this model.",
        stacklevel=2,
    )
```

Do not promote this to an `assert`. The purpose of the warning is to surface unexpected configurations early — not to prohibit them.

---

## Constraint NOT to Enforce

### rotary\_dim % 32 == 0 (Do NOT Enforce)

This was the implicit assumption behind the zero-padding approach in the original `TTNNRotaryPositionEmbedding`. Now that Strategy C removes the tile-alignment requirement on `rotary_dim`, enforcing `rotary_dim % 32 == 0` as a hard error would:

1. Block valid model configurations where `partial_rotary_factor` yields a non-tile-aligned `rotary_dim`.
2. Mislead engineers into believing that tile alignment on `rotary_dim` is a fundamental mathematical requirement — it is not. The tile-alignment requirement that matters is on `head_dim`, not `rotary_dim`.

### rotary\_dim % 64 == 0 (Do NOT Enforce)

Similarly, enforcing the two-tile constraint on `rotary_dim` rather than `head_dim` would block configurations like `rotary_dim=32, head_dim=128`, which Strategy C handles correctly.

---

## Role of Strategy B After Strategy C Is Deployed

Strategy B enforces `rotary_dim % 64 == 0` as a hard precondition to prevent silent numerical corruption for configurations not yet handled by Strategy C. Once Strategy C is deployed, Strategy B's purpose is superseded:

- Strategy C already produces correct output for all `rotary_dim` values.
- The Strategy B guard would reject valid non-tile-aligned configurations that Strategy C can now handle.

Strategy B should be removed from `__init__` when Strategy C is deployed. The warning for `rotary_dim % 32 != 0` replaces it as the appropriate signaling mechanism.

### Short-Term Option: Deploy Strategy B First

If Strategy C cannot be deployed immediately, Strategy B is a safe interim measure. It converts Path B (silent PCC ~0.71 corruption) into Path A (an explicit error), preventing incorrect outputs from reaching users. The error message should cite Strategy C as the pending fix:

```python
# Short-term guard: remove when Strategy C is implemented
assert rotary_dim % 64 == 0, (
    f"rotary_dim={rotary_dim} is not a multiple of 64. "
    "The current TTNNRotaryPositionEmbedding implementation does not "
    "correctly handle non-tile-aligned rotary_dim. "
    "Use partial_rotary_factor such that rotary_dim % 64 == 0, "
    "or implement Strategy C (identity-filled precomputed cos/sin table) "
    "before using this configuration."
)
```

### Migration Path

1. **Now:** Deploy Strategy B guard (one line in `__init__`). Any new model with non-tile-aligned `rotary_dim` gets a clear error rather than silent corruption.
2. **Before the next non-tile-aligned model:** Implement Strategy C following the steps in [`recommended_fix.md`](./recommended_fix.md).
3. **After Strategy C is deployed:** Remove the Strategy B guard. Add the `rotary_dim % 32 != 0` warning in its place. Run the verification checklist in [`verification_checklist.md`](./verification_checklist.md) to confirm correctness.

---

## Complete Precondition Summary

| Constraint | Enforcement | Reason |
|---|---|---|
| `head_dim % 64 == 0` | Hard `assert` | Op's two-tile constraint; cannot be worked around |
| `rotary_dim % 2 == 0` | Hard `assert` | Rotate-half requires even element count |
| `rotary_dim <= head_dim` | Hard `assert` | Passthrough region would have negative size otherwise |
| `rotary_dim % 32 != 0` | `warnings.warn` | Unexpected but valid after Strategy C; surface for inspection |
| `rotary_dim % 32 == 0` | Do NOT enforce | Strategy C handles non-tile-aligned `rotary_dim` correctly |
| `rotary_dim % 64 == 0` | Do NOT enforce | Strategy C handles correctly; enforcing would be overly restrictive |
| Strategy B guard (`rotary_dim % 64 == 0` as hard error) | Short-term only; remove after Strategy C | Interim to prevent silent corruption before Strategy C is deployed |

---

## What's Next

[`verification_checklist.md`](./verification_checklist.md) provides five concrete test cases that confirm correctness of Strategy C across tile-aligned, non-tile-aligned, full-head, trace-compatibility, and edge-case configurations.
