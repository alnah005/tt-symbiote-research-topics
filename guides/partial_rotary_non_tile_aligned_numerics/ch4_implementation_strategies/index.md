# Chapter 4 — Correct Implementation Strategies for Non-Tile-Aligned Partial RoPE

This chapter presents three strategies for obtaining numerically correct partial RoPE output when `rotary_dim` is not a multiple of 64 (or more generally, when `rotary_dim < head_dim`). Chapters 1–3 established that the current `TTNNRotaryPositionEmbedding` implementation is broken in this regime: the zero-padded cos/sin approach produces either `TT_FATAL` or silent PCC ~0.71 corruption. This chapter shows what to do instead.

---

> **Key Finding:** Strategy C — precomputing a full `[max_seq_len, head_dim]` cos/sin table with identity values at passthrough positions and duplicated frequencies at positions `[head_dim/2, head_dim/2 + rotary_dim/2)` — is the only strategy that is simultaneously correct (given the input head uses the `head_dim/2`-split pairing convention, where elements `j` and `j + head_dim/2` are rotation partners for `j in [0, rotary_dim/2)`) and trace-compatible. Under the PyTorch slice convention (pairs are `(x[i], x[i+rotary_dim/2])`), Strategy C produces different output from the reference. Strategies A and B address narrower concerns: Strategy A is correct for tile-aligned `rotary_dim` but requires runtime buffer allocation; Strategy B is a fail-fast guard that converts silent corruption into an explicit error.

---

## Recap of Chapters 1–3 Prerequisites

The following findings from earlier chapters motivate the strategies in this chapter.

- **Correct partial RoPE splits at `rotary_dim/2`, not `head_dim/2`:** elements `i` in `[0, rotary_dim/2)` pair with `i + rotary_dim/2`, and elements `[rotary_dim, head_dim)` pass through unchanged. (Ch1: [`../ch1_rope_fundamentals/partial_rope_math.md`](../ch1_rope_fundamentals/partial_rope_math.md))
- **`ttnn.experimental.rotary_embedding` enforces `cos.shape[-1] == input.shape[-1] == head_dim` via `TT_FATAL`:** supplying cos/sin of any other size causes the op to abort. (Ch2: [`../ch2_op_shape_contract/shape_validation_in_invoke.md`](../ch2_op_shape_contract/shape_validation_in_invoke.md))
- **The kernel's rotate-half pairing is always at `head_dim/2`:** for `head_dim=128` this is 64. There is no runtime argument that overrides this to `rotary_dim/2=24`. (Ch2: [`../ch2_op_shape_contract/kernel_rotate_half_pairing.md`](../ch2_op_shape_contract/kernel_rotate_half_pairing.md))
- **`head_dim % 64 == 0` is required:** the op asserts `input.shape[-1] % (TILE_WIDTH * 2) == 0`, i.e., `head_dim % 64 == 0`. (Ch2)
- **Zero-padding cos/sin cannot fix the pairing mismatch:** any cos/sin layout that places real values in `[0, rotary_dim)` and zeros in `[rotary_dim, head_dim)` produces wrong results at every output position. (Ch3: [`../ch3_bug_root_cause/correct_partial_rope_reference.md`](../ch3_bug_root_cause/correct_partial_rope_reference.md))

---

## Learning Objectives

By the end of this chapter you should be able to:

1. Identify which of the three strategies (A, B, C) is appropriate for a given combination of `rotary_dim`, `head_dim`, and trace-compatibility requirements.
2. Construct the full `head_dim`-wide cos/sin table required by Strategy C, including the frequency-duplication property at positions `[head_dim/2, head_dim/2 + rotary_dim/2)`.
3. Explain why `ttnn.pad` is trace-unsafe inside a trace bracket and what alternatives exist.
4. State the precondition that Strategy B enforces, and explain why it converts silent corruption into an explicit error without fixing the underlying bug.

---

## Decision Table

The following table summarizes the tradeoffs across the three strategies for the concrete case `rotary_dim=48, head_dim=128`.

| Property | Strategy A (Slice-Apply-Concat) | Strategy B (Enforce Alignment) | Strategy C (Precomputed Full-Head cos/sin) |
|---|---|---|---|
| Numerically correct output | Yes, if `rotary_dim % 64 == 0`; requires padded-slice variant otherwise | Not a fix; raises `ValueError` instead of corrupting | Yes, if `head_dim % 64 == 0` **and** input head uses `head_dim/2`-split pairing (elements `j` and `j + head_dim/2` are rotation partners). Produces a different output from the PyTorch slice convention (pairs `(x[i], x[i+rotary_dim/2])`). See `strategy_c_precomputed_full_head_cos_sin.md` Section 4b for the derivation of which convention applies. |
| Trace-compatible | No — requires `ttnn.pad` inside forward unless buffer is pre-allocated | N/A (guard fires before forward) | Yes — cos/sin allocated once in `__init__`; forward has no runtime allocation |
| Implementation complexity | Medium — requires slice, optional pad, apply, unpad, concat | Low — one guard line in `__init__` | Medium — requires careful cos/sin construction with frequency duplication |
| Works for arbitrary `rotary_dim` | Only if `rotary_dim % 64 == 0`, or with extra slice-padding logic | No; prevents use entirely | Yes |
| Requires changes to `__init__` | No | Yes (guard) | Yes (precompute cos/sin table) |
| Requires changes to `forward` | Yes (slice, apply, concat) | No | Minimal (pass precomputed table) |

---

## Files in Reading Order

1. [**`strategy_a_slice_apply_concat.md`**](./strategy_a_slice_apply_concat.md) — Slice input to `[..., rotary_dim]`, apply the op, then concat the passthrough region; shape analysis showing why `rotary_dim % 64 == 0` is required and how to handle the non-tile-aligned case with extra slice padding; trace-compatibility limitations.
2. [**`strategy_b_enforce_tile_alignment.md`**](./strategy_b_enforce_tile_alignment.md) — One-line `ValueError` guard in `__init__` that makes non-tile-aligned `rotary_dim` fail explicitly rather than silently; precondition analysis and when this strategy is appropriate.
3. [**`strategy_c_precomputed_full_head_cos_sin.md`**](./strategy_c_precomputed_full_head_cos_sin.md) — The recommended fix: precompute a `[max_seq_len, head_dim]` cos/sin table with identity values and duplicated frequencies; mathematical derivation of the correct layout; Python construction code; trace-compatibility proof.
4. [**`trace_safe_alternatives_to_ttnn_pad.md`**](./trace_safe_alternatives_to_ttnn_pad.md) — Why `ttnn.pad` is trace-unsafe inside a trace bracket; pre-allocated buffer alternatives; recommendation to prefer Strategy C over any runtime padding approach.

---

## What's Next

After reading the files in order above, you will have a complete implementation path for correct partial RoPE on TTNN hardware across tile-aligned and non-tile-aligned `rotary_dim` configurations. The recommended production approach is Strategy C; Strategies A and B are preserved here because they illuminate the design space and are appropriate for specific constrained situations.
