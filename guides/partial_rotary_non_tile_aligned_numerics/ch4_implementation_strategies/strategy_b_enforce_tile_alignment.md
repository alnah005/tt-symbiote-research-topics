# Strategy B — Enforce Tile Alignment as a Precondition

Strategy B is not a fix for the partial RoPE correctness bug. It is a fail-fast guard: a single line added to `TTNNRotaryPositionEmbedding.__init__` that raises `ValueError` when a caller supplies a `rotary_dim` value that will silently produce wrong outputs. The value of Strategy B is that it converts an insidious silent failure (PCC ~0.71) into an immediate, actionable error that surfaces at model-initialization time rather than during inference.

---

## 1. The Guard

```python
class TTNNRotaryPositionEmbedding:
    def __init__(self, rotary_dim: int, head_dim: int, ...):
        if rotary_dim % 2 != 0:
            raise ValueError(
                f"rotary_dim must be even for rotate-half pairing; got rotary_dim={rotary_dim}"
            )
        if head_dim % 64 != 0:
            raise ValueError(
                f"head_dim must be a multiple of 64 for ttnn.experimental.rotary_embedding "
                f"(requires head_dim % (TILE_WIDTH * 2) == 0); got head_dim={head_dim}"
            )
        if rotary_dim > head_dim:
            raise ValueError(
                f"rotary_dim must not exceed head_dim; got rotary_dim={rotary_dim}, head_dim={head_dim}"
            )
        if rotary_dim != head_dim:
            # Non-partial case (rotary_dim == head_dim) is always correct with this op.
            # For partial RoPE (rotary_dim < head_dim), the op's fixed head_dim/2 pairing
            # is incompatible with the required rotary_dim/2 pairing unless Strategy C is used.
            raise ValueError(
                f"rotary_dim={rotary_dim} < head_dim={head_dim}: partial RoPE with "
                f"ttnn.experimental.rotary_embedding requires Strategy C (precomputed full-head "
                f"cos/sin table). See strategy_c_precomputed_full_head_cos_sin.md."
            )
        ...
```

A minimal version — enforcing only the op's own tile-alignment constraint on `head_dim` and blocking partial RoPE — is:

```python
if head_dim % 64 != 0:
    raise ValueError(
        f"head_dim must be a multiple of 64; got {head_dim}"
    )
if rotary_dim != head_dim:
    raise ValueError(
        f"Partial RoPE (rotary_dim={rotary_dim} < head_dim={head_dim}) is not supported "
        f"by the zero-padding approach. Use Strategy C instead."
    )
```

---

## 2. What Strategy B Enforces (and What It Does Not)

### What it enforces

- `rotary_dim % 2 == 0`: required for the rotate-half operation to be well-defined (split at `rotary_dim/2` must be an integer).
- `head_dim % 64 == 0`: required by the op itself (`TT_FATAL` otherwise).
- `rotary_dim == head_dim` (strictest form): the only configuration where `ttnn.experimental.rotary_embedding`'s `head_dim/2` pairing produces correct output with standard (non-duplicated) cos/sin.

### What it does not enforce

Strategy B does not enforce `rotary_dim % 64 == 0`. This is deliberate: `rotary_dim % 64 == 0` is not actually the right constraint. The op's tile-alignment requirement is on `head_dim`, not `rotary_dim`. A caller could have `rotary_dim=48` (not a multiple of 64) and `head_dim=128` (a multiple of 64) and Strategy C would still work correctly — the constraint to enforce in that case is that the caller use Strategy C rather than the zero-padding approach.

> **Note:** An earlier draft of this guide considered enforcing `rotary_dim % 64 == 0` as the guard condition. That would be wrong: it allows `rotary_dim=64, head_dim=128` through (which appears to work but only because `rotary_dim/2 = 32 = head_dim/2 / 2`, an accidental coincidence), and it rejects `rotary_dim=48, head_dim=48` (which would be fully correct since `rotary_dim == head_dim`). The correct invariant to check is `head_dim % 64 == 0` and, if partial RoPE is intended, that Strategy C is in use.

---

## 3. The Failure It Prevents

Without Strategy B, the sequence of events for `rotary_dim=48, head_dim=128` is:

1. `TTNNRotaryPositionEmbedding.__init__` completes without error.
2. At forward time, cos/sin are padded to `nearest_32(rotary_dim) = 64` — wrong target.
3. `ttnn.experimental.rotary_embedding` fires `TT_FATAL` because `cos.shape[-1]=64 != input.shape[-1]=128` (Path A), or in a variant where cos/sin are further padded to 128, the kernel produces outputs with PCC ~0.71 (Path B).
4. In Path B, the model produces numerically wrong outputs for every sequence position, with no warning.

With Strategy B, step 1 raises `ValueError` with a message that names both the configuration and the recommended fix. The engineer sees the error at model-load time, before any inference runs.

> **Key Finding:** Strategy B converts a PCC ~0.71 silent failure at inference time into a `ValueError` at model-initialization time. It does not produce correct partial RoPE output for non-full-head configurations; it simply refuses to proceed with a configuration that would produce wrong output.

---

## 4. When to Use Strategy B

Strategy B is appropriate in the following situation:

- The codebase has a `TTNNRotaryPositionEmbedding` class that is known to only be used with `rotary_dim == head_dim` in practice, but the constructor accepts `rotary_dim` as a parameter — creating a risk that a future caller supplies a partial RoPE configuration.
- The engineer wants to make this risk explicit without implementing the full Strategy C fix immediately.
- The team has determined that non-tile-aligned partial RoPE is out of scope for the current release, and wants the code to fail loudly rather than silently if such a configuration is attempted.

Strategy B is **not** appropriate as a standalone fix if the goal is to support partial RoPE (i.e., `rotary_dim < head_dim`). In that case, implement Strategy C.

---

## 5. Interaction with `head_dim % 64 == 0`

The op's own precondition — enforced by `TT_FATAL` inside `ttnn.experimental.rotary_embedding` — is `head_dim % 64 == 0`. Strategy B adds a Python-level guard that fires earlier (at `__init__` time) with a more informative error message. For `head_dim` values that violate this:

```python
# Example: head_dim=96, rotary_dim=96
# Without Strategy B: TT_FATAL fires deep in C++ at forward time
# With Strategy B:    ValueError fires at __init__ time with a clear message

>>> TTNNRotaryPositionEmbedding(rotary_dim=96, head_dim=96, ...)
ValueError: head_dim must be a multiple of 64; got 96
```

This is a separate concern from partial RoPE correctness. Both guards belong in `__init__`.

---

## 6. Summary

| Question | Answer |
|---|---|
| Does Strategy B produce correct partial RoPE output? | No — it raises `ValueError` for partial RoPE configurations |
| Does Strategy B fix the PCC ~0.71 bug? | No — it replaces silent corruption with an explicit error |
| What is the correct constraint to enforce? | `head_dim % 64 == 0`; and `rotary_dim == head_dim` if partial RoPE is unsupported |
| Should `rotary_dim % 64 == 0` be the guard? | No — this is the wrong constraint; see Section 2 above |
| When is Strategy B appropriate? | When partial RoPE is intentionally unsupported and fail-fast behavior is desired |

**Next:** [Strategy C — Precomputed Full-Head cos/sin](./strategy_c_precomputed_full_head_cos_sin.md)
