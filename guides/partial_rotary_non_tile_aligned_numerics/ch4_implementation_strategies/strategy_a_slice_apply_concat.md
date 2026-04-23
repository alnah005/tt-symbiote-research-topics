# Strategy A — Slice, Apply, Concat

Strategy A reframes the partial RoPE problem by operating only on the `rotary_dim`-wide slice of the input, applying `ttnn.experimental.rotary_embedding` to that slice, and then concatenating the untouched passthrough region back. The intuition is sound: if you can isolate the rotated region before calling the op, you avoid the mismatch between the kernel's `head_dim/2` split and the correct `rotary_dim/2` split. The problem, as this file shows, is that `ttnn.experimental.rotary_embedding` imposes its own tile-alignment constraint on the slice's last dimension — and that constraint is not automatically satisfied just because you sliced the input correctly.

---

## 1. Strategy A Outline

The three steps are:

1. **Slice:** extract `x_rot = input[..., :rotary_dim]` with shape `[B, H, S, rotary_dim]`.
2. **Apply:** call `ttnn.experimental.rotary_embedding(x_rot, cos, sin)` where cos and sin have shape `[1, 1, S, rotary_dim]`.
3. **Concat:** concatenate the op output with the untouched passthrough `input[..., rotary_dim:]` to recover the full `[B, H, S, head_dim]` output.

If this worked, it would correctly perform a rotate-half split at `rotary_dim/2` because the sliced input has `shape[-1] = rotary_dim`, and the kernel derives its pairing offset from `shape[-1] / 2 = rotary_dim / 2`. This is exactly what the correct implementation requires (see [`../ch3_bug_root_cause/correct_partial_rope_reference.md`](../ch3_bug_root_cause/correct_partial_rope_reference.md)).

---

## 2. Shape Analysis for `rotary_dim=48, head_dim=128`

After slicing, `x_rot` has shape `[B, H, S, 48]`. Before calling the op, TTNN validates:

```
TT_FATAL: input.padded_shape()[-1] % (TILE_WIDTH * 2) == 0
        → 48 % 64 == 0
        → False  → TT_FATAL
```

The op aborts immediately. Strategy A in its basic form does not work for `rotary_dim=48`.

> **Key Finding:** Strategy A requires `rotary_dim % 64 == 0` for the sliced-then-apply approach to be valid. When `rotary_dim=48`, the slice itself fails the op's tile-alignment check. `rotary_dim=64` (the next multiple of 64) would pass; `rotary_dim=48` does not.

The same constraint applies to cos/sin: they must have `shape[-1] = rotary_dim`, and `rotary_dim % 64 == 0` must hold.

---

## 3. The Padded-Slice Variant

For non-tile-aligned `rotary_dim` (such as 48), Strategy A can be extended with an extra padding step applied to the slice. The extended procedure is:

1. **Slice:** `x_rot = input[..., :rotary_dim]` with shape `[B, H, S, 48]`.
2. **Pad slice to tile boundary:** pad `x_rot` from `[..., 48]` to `[..., 64]`, filling positions `[48, 64)` with zeros. Call this `x_rot_padded`.
3. **Prepare cos/sin of shape `[..., 64]`:** real values at positions `[0, 48)`, zeros at positions `[48, 64)`.
4. **Apply:** `out_padded = ttnn.experimental.rotary_embedding(x_rot_padded, cos_64, sin_64)`. The op now sees `shape[-1] = 64`, which satisfies `64 % 64 == 0`.
5. **Unpad result:** slice `out_padded[..., :48]` to recover shape `[B, H, S, 48]`. The zero-padded positions `[48, 64)` of the output are discarded.
6. **Concat passthrough:** concatenate `out_rot` with `input[..., 48:]` to produce the final `[B, H, S, 128]` output.

### Why step 5 produces correct output

With `x_rot_padded.shape[-1] = 64`, the kernel's pairing offset is `64 / 2 = 32`. The cos/sin values are:

```
cos[0:24]  = real values c_0, ..., c_23
cos[24:32] = 0.0  (will receive identity via cos=0)
```

Wait — this is a subtlety. With pairing offset 32, the kernel computes:

```
output[i]      = x_rot_padded[i]      * cos[i]      + x_rot_padded[i + 32] * (-sin[i])
output[i + 32] = x_rot_padded[i]      * sin[i]      + x_rot_padded[i + 32] *   cos[i]
```

for `i in [0, 32)`. This is still not pairing at `rotary_dim/2 = 24`. For positions `i in [0, 32)`:

- `output[i] = x_rot[i] * c_i + x_rot_padded[i + 32] * (-s_i)` where `x_rot_padded[i+32]` is a **real** input value for `i in [0, 16)` (positions 32–47 are within the original 48-element slice) and **zero** (padded) for `i in [16, 32)` (positions 48–63 are zero-filled). Neither sub-range produces the correct partner `x_rot[i + 24]` required by the `rotary_dim/2=24` pairing; the offset 32 ≠ 24 regardless of whether the partner slot is real or zero.

> **Note:** The padded-slice variant with a 48→64 pad does NOT correctly compute partial RoPE for `rotary_dim=48` because the padded slice size is 64, giving a kernel pairing offset of 32 — which is neither `rotary_dim/2=24` nor the desired offset. Correct partial RoPE for `rotary_dim=48` requires the pairing offset to be exactly 24. No padding of the slice to a TTNN-compatible size will produce that offset unless the padded size is exactly 48 (which fails the tile check) or a multiple of 48 that is also a multiple of 64 (the LCM is 192, requiring a 48→192 pad, which is impractical).

### Correct application of Strategy A

Strategy A is correct and straightforward when `rotary_dim % 64 == 0`. In that case:

- The slice has `shape[-1] = rotary_dim`, which satisfies `rotary_dim % 64 == 0`.
- The kernel's pairing offset is `rotary_dim / 2`.
- cos/sin have `shape[-1] = rotary_dim`.
- No padding of the slice is needed.

For `rotary_dim=64, head_dim=128`:

```python
x_rot  = input[..., :64]   # [B, H, S, 64]  — satisfies 64 % 64 == 0
x_pass = input[..., 64:]   # [B, H, S, 64]

out_rot = ttnn.experimental.rotary_embedding(x_rot, cos_64, sin_64)
# kernel offset: 64/2=32  — but rotary_dim/2=32 also  — correct!

output = ttnn.concat([out_rot, x_pass], dim=-1)  # [B, H, S, 128]
```

Here the kernel offset of 32 equals `rotary_dim/2 = 32`, so the output is correct.

---

## 4. Trace Compatibility

Strategy A in any form requires runtime buffer operations inside the forward pass:

- `ttnn.slice` or indexing to extract `x_rot` — this may or may not allocate a new buffer depending on TTNN's implementation; in the worst case it does.
- `ttnn.pad` (in the padded-slice variant) always allocates a new device buffer. This is trace-unsafe inside a trace bracket.
- `ttnn.concat` to reassemble the output — the output buffer must also be pre-allocated for trace safety.

> **Key Finding:** Strategy A has the same trace-safety problem as the current implementation when `ttnn.pad` is used inside the forward pass. Even without `ttnn.pad`, the slice and concat operations introduce runtime allocations. Resolving trace compatibility requires pre-allocating all intermediate buffers in `__init__` — a significant engineering burden. For the full analysis of `ttnn.pad` trace-unsafety and the available alternatives, see [`trace_safe_alternatives_to_ttnn_pad.md`](./trace_safe_alternatives_to_ttnn_pad.md). Strategy C eliminates runtime allocation entirely.

---

## 5. Summary

| Condition | Strategy A outcome |
|---|---|
| `rotary_dim % 64 == 0` | Correct; no slice padding needed; still not trace-safe by default |
| `rotary_dim % 64 != 0` (e.g., 48) | Slice fails `TT_FATAL`; padded-slice variant does not produce correct output |
| Trace bracket | Unsafe unless all buffers pre-allocated in `__init__` |

Strategy A is best suited for offline or eager-mode inference where `rotary_dim` is a multiple of 64 and trace compatibility is not required.

**Next:** [Strategy B — Enforce Tile Alignment](./strategy_b_enforce_tile_alignment.md)
