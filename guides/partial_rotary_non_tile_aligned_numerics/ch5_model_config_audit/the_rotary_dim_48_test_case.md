# The rotary_dim=48 Test Case

This file reconstructs the synthetic test configuration — `partial_rotary_factor=0.375`, `head_dim=128`, `rotary_dim=48` — that exposes the PCC ~0.71 bug in `TTNNRotaryPositionEmbedding`. It traces the exact sequence of operations that occur in `TTNNRotaryPositionEmbedding.forward` for this configuration, identifies the two failure paths, and explains why observing PCC ~0.71 during warm-up (outside any Metal Trace) confirms that the bug is in the forward computation itself, not in trace capture or replay.

---

> **Key Finding:** The `rotary_dim=48` test case is a synthetic configuration, not a production-supported model. It exercises the zero-padding branch in `TTNNRotaryPositionEmbedding` directly. Depending on whether TTNN autoformat is active, execution either crashes with `TT_FATAL` (Path A) or completes with PCC ~0.71 (Path B). Either outcome confirms the same root cause: the cos/sin padding target is wrong. The PCC ~0.71 observed in warm-up (no trace) rules out trace capture as a confounding factor.

---

## Setup: Reconstructing the Synthetic Test

The test scenario corresponds to a hypothetical model with:

- `partial_rotary_factor = 0.375`
- `head_dim = 128`
- Derived `rotary_dim = int(0.375 * 128) = 48`
- `max_seq_len` chosen for the test, e.g., 2048
- Input tensor shape: `[batch, num_heads, seq_len, head_dim]` = e.g., `[1, 32, 1, 128]` for decode

The test runs one forward pass through `TTNNRotaryPositionEmbedding`, compares the output to a PyTorch reference, and reports PCC.

The PyTorch reference computes:

```python
import torch

def apply_rotary_partial(x, cos_real, sin_real, rotary_dim):
    """
    Correct partial RoPE reference.
    x: [..., head_dim]
    cos_real, sin_real: [..., rotary_dim]
    rotary_dim: number of dimensions to rotate
    """
    x_rot  = x[..., :rotary_dim]   # [..., 48]  — the region to rotate
    x_pass = x[..., rotary_dim:]   # [..., 80]  — passthrough, unchanged

    # Rotate-half within the rotary_dim slice only
    half = rotary_dim // 2  # 24
    x1 = x_rot[..., :half]   # [..., 24]  first half of rotated region
    x2 = x_rot[..., half:]   # [..., 24]  second half of rotated region
    rotated = torch.cat([
        x1 * cos_real[..., :half] - x2 * sin_real[..., :half],   # output[0:24]
        x1 * sin_real[..., :half] + x2 * cos_real[..., :half],   # output[24:48]
    ], dim=-1)

    return torch.cat([rotated, x_pass], dim=-1)  # [..., 128]
```

---

## Tracing TTNNRotaryPositionEmbedding.forward for rotary_dim=48

The following steps trace what happens inside `TTNNRotaryPositionEmbedding` when constructed and called with `rotary_dim=48, head_dim=128`.

### Step 1 — Precomputation in __init__

During construction, the class precomputes cos/sin on CPU and transfers to device:

```python
# In __init__:
# cos/sin precomputed for rotary_dim positions (correct shape for pure rotary math)
# shape: [1, 1, max_seq_len, 48]
cos_cache = precompute_cos(rotary_dim=48, max_seq_len=2048)
sin_cache = precompute_sin(rotary_dim=48, max_seq_len=2048)

# Transfer to device as TILE-layout tensors
# shape on device: [1, 1, 2048, 48] — but 48 is not tile-aligned
```

At this point no padding has occurred. The cos/sin tensors have `shape[-1]=48`.

### Step 2 — Runtime padding in forward (the bug location)

In `forward`, the class detects that `rotary_dim % 32 != 0` (`48 % 32 = 16`) and calls `ttnn.pad`:

```python
# In forward (current buggy implementation):
if rotary_dim % 32 != 0:
    # Pad cos/sin from 48 to nearest_32(48) = 64
    # fill_value=0 for the new positions [48, 64)
    cos_padded = ttnn.pad(cos_cache, [..., 64], fill_value=0.0)
    sin_padded = ttnn.pad(sin_cache, [..., 64], fill_value=0.0)
    # cos_padded.shape: [1, 1, max_seq_len, 64]
    # sin_padded.shape: [1, 1, max_seq_len, 64]
```

The target size `64 = nearest_32(48)` is the next tile boundary above 48. This is the wrong target. The op requires `cos.shape[-1] == head_dim = 128`.

### Step 3 — Call to ttnn.experimental.rotary_embedding

The padded cos/sin (shape `[..., 64]`) and the input tensor (shape `[..., 128]`) are passed to the op:

```python
output = ttnn.experimental.rotary_embedding(
    input,      # shape: [1, 32, 1, 128]  — head_dim=128
    cos_padded, # shape: [1, 1,  1, 64]   — WRONG: should be [..., 128]
    sin_padded, # shape: [1, 1,  1, 64]   — WRONG
    token_idx,
)
```

At this point execution diverges into two paths.

---

## Path A — TT_FATAL (Shape Mismatch)

In the standard TTNN validation path, `RotaryEmbeddingOperation::invoke` sets `X = input.padded_shape()[-1] = 128` and then asserts:

```
TT_FATAL(cos_cache.padded_shape()[-1] == X, "Cos dims must match input dims")
```

With `cos_cache.padded_shape()[-1] = 64` and `X = 128`, this assertion fails. The process halts with:

```
TT_FATAL: Cos dims must match input dims (64 != 128)
```

This is a hard crash. No output is produced. The failure is loud and immediately identifiable.

### Distinguishing characteristics of Path A

- Execution terminates before any compute kernel runs.
- The error message directly names the shape mismatch.
- No PCC measurement is possible — there is no output tensor.

---

## Path B — Autoformat Further Pads to 128 (Silent Corruption, PCC ~0.71)

If an autoformat layer intervenes before the device operation validates shapes — for example, if `run_with_autoformat` applies `AutoFormat::pad_to_tile_shape` to the cos/sin input — the cos/sin may be extended from `[..., 64]` to `[..., 128]` by zero-padding positions `[64, 128)`. In that case the op's shape assertion is satisfied (`cos.shape[-1] = 128 = X`) and the kernel runs.

The kernel now sees:

```
cos: positions [0, 48)  — real cosine values
     positions [48, 64) — zeros (from Step 2 padding)
     positions [64, 128) — zeros (from autoformat padding)

sin: positions [0, 48)  — real sine values
     positions [48, 64) — zeros (from Step 2 padding)
     positions [64, 128) — zeros (from autoformat padding)
```

The kernel applies its `head_dim/2 = 64` split: for each `j in [0, 64)`, it computes:

```
output[j]      = input[j]      * cos[j]      + input[j + 64] * (-sin[j])
output[j + 64] = input[j]      * sin[j]      + input[j + 64] *   cos[j]
```

For `j in [0, 48)`: `cos[j]` is a real value, `sin[j]` is a real value, `input[j+64]` is a passthrough element (should be unchanged) but it participates in the rotation with the wrong partner. The output at positions `[0, 48)` and `[64, 112)` is computed with the wrong pairing offset (64 instead of the required 24).

For `j in [48, 64)`: `cos[j] = 0` and `sin[j] = 0`. The kernel computes `output[j] = 0` and `output[j+64] = 0`. Positions `[48, 64)` and `[112, 128)` of the output are zeroed.

The result is:
- Positions `[0, 48)`: corrupted (wrong pairing offset, wrong partner elements)
- Positions `[48, 64)`: zeroed (should be passthrough)
- Positions `[64, 112)`: corrupted (wrong pairing offset)
- Positions `[112, 128)`: zeroed (should be passthrough)

All 128 output elements are wrong — 32 are zeroed (positions `[48, 64)` and `[112, 128)`) and the remaining 96 carry wrong-partner rotation values (positions `[0, 48)` and `[64, 112)`). The Pearson correlation with the correct output is approximately 0.71 — high enough to look plausible but far below the > 0.9999 target.

> **[SILENT FAILURE]** In Path B, `ttnn.experimental.rotary_embedding` returns an output tensor with no error. Loss metrics may appear plausible if they average over many elements. Only a direct PCC comparison against a PyTorch reference reveals the ~0.71 score and the element-level corruption pattern described above.

---

## Why PCC ~0.71 in Warm-Up Rules Out Trace as the Cause

The warm-up phase runs `TTNNRotaryPositionEmbedding.forward` outside any Metal Trace bracket — no `ttnn.begin_trace_capture` or `ttnn.end_trace_capture` is active. Because the PCC of ~0.71 is observed even in this trace-free warm-up pass, the numerical corruption is definitively in the forward computation itself, not in trace capture or trace replay mechanics.

If the bug were a trace issue, the warm-up pass would show correct PCC and only the traced run would produce incorrect output. The fact that warm-up is already wrong narrows the root cause to the `TTNNRotaryPositionEmbedding.forward` logic — specifically to the zero-padding step and the subsequent shape mismatch or wrong-paired rotation in the op.

This distinction is important for debugging: it rules out trace-related suspects (buffer reuse conflicts, static shape mismatches in trace replay, pre-allocated buffer wrong-size errors) and focuses attention on the cos/sin construction logic.

---

## Summary of Failure Paths for rotary_dim=48, head_dim=128

| Path | Trigger condition | Observable behavior | PCC |
|---|---|---|---|
| Path A | `TT_FATAL` fires before autoformat | Process crash, error message "Cos dims must match input dims" | N/A (no output) |
| Path B | Autoformat pads cos/sin from 64 to 128 before shape check | Silent completion, wrong output | ~0.71 |
| Correct (after Strategy C fix) | Identity-filled cos/sin of shape `[..., 128]` supplied | Correct partial RoPE output | > 0.9999 |

---

## What's Next

The full recommendation for fixing this bug — including the Strategy C construction code and the precondition policy — is in Chapter 6.

**Next:** [Chapter 6 — Recommendations and Implementation Guide](../ch6_recommendations/index.md)
