# Chapter 2 — How `ttnn.experimental.rotary_embedding` Processes cos/sin Shapes

This chapter examines the shape contract enforced by `ttnn.experimental.rotary_embedding` at every layer of the stack: the Python entry point, the C++ invoke and validate functions, the reader dataflow kernel, and the compute kernel. By the end of this chapter you will understand exactly what shape cos/sin must have for the op to accept them, and why that contract makes partial RoPE (rotation of only the first `rotary_dim` elements) invisible to the op natively.

---

## Learning Objectives

After reading this chapter you should be able to:

1. State the exact `TT_FATAL` constraint that `RotaryEmbeddingOperation::invoke` places on `cos_cache.shape[-1]` and explain why it ties cos/sin to `head_dim`, not `rotary_dim`.
2. Trace the path from a Python `ttnn.experimental.rotary_embedding(...)` call through `run_with_autoformat`, the C++ invoke, the device-operation validate, and into the compute kernel.
3. Explain how the compute kernel's `half_Wt` split implements rotate-half over the full `head_dim` and why there is no knob to restrict rotation to a `rotary_dim` subset.
4. Describe what the Python golden function reveals about the shape contract and how it corroborates the C++ constraints.
5. Articulate why padding cos/sin from `rotary_dim=48` to `nearest_32(48)=64` is incorrect when `head_dim=128` and identify which `TT_FATAL` would fire.

---

## Data-Flow Diagram

The diagram below shows the path a call takes from Python down to the compute kernel. Shape checks are annotated at each boundary.

```
Python caller
  │  cos_cache.shape[-1] == head_dim  (required; rotary_dim and nearest_32(rotary_dim) both rejected)
  │
  ▼
ttnn.experimental.rotary_embedding(input, cos_cache, sin_cache, rotary_dim=...)
  │  (transformer.py) — golden slices input to [:rotary_dim], applies cos/sin, concatenates passthrough unchanged
  │
  ▼
run_with_autoformat(...)
  │  AutoFormat::pad_to_tile_shape applied to cos/sin
  │  cos.shape[-1] must equal X = input.padded_shape()[-1] AFTER padding
  │
  ▼
RotaryEmbeddingOperation::invoke(...)             [rotary_embedding.cpp]
  │  TT_FATAL: input.padded_shape()[-1] % 64 == 0
  │  X = input.padded_shape()[-1]
  │  TT_FATAL: cos.padded_shape()[-1] == X  ← shape gate
  │
  ▼
RotaryEmbeddingDeviceOperation::validate(...)     [rotary_embedding_device_operation.cpp]
  │  TT_FATAL: cos.padded_shape()[-1] == X
  │
  ▼
Reader kernel: reader_rotary_embedding_interleaved_start_id.cpp
  │  half_Wt = Wt / 2  (Wt = X / TILE_WIDTH)
  │  rotated_input_curr_id = start_id + half_Wt
  │
  ▼
Compute kernel: rotary_embedding.cpp
     Wt = X / TILE_WIDTH
     half_Wt = Wt / 2
     pairs tile j with tile j + half_Wt  (full-head-dim rotate-half)
```

---

## Recap of Chapter 1 Prerequisites

This chapter assumes familiarity with partial RoPE math, tile alignment, and `nearest_32` from Chapter 1 (see [`../ch1_rope_fundamentals/index.md`](../ch1_rope_fundamentals/index.md)). The key starting point: cos/sin shape must equal `head_dim`, not `rotary_dim`; tile alignment pads a dimension of size `D` to `nearest_32(D) = ⌈D/32⌉ × 32`. Chapter 2 explains exactly why these constraints interact to cause failures when `rotary_dim` is non-tile-aligned.

---

## Files in Reading Order

Read the files in this chapter in the following order:

1. [**`shape_validation_in_invoke.md`**](./shape_validation_in_invoke.md) — The C++ shape gates in `invoke` and `validate`; why cos/sin must have `shape[-1] == head_dim`.
2. [**`kernel_rotate_half_pairing.md`**](./kernel_rotate_half_pairing.md) — How the compute and reader kernels implement rotate-half over the full `head_dim` with no `rotary_dim` knob.
3. [**`what_the_golden_function_reveals.md`**](./what_the_golden_function_reveals.md) — The Python golden function's shape contract and how it corroborates the C++ constraints.

---

## What's Next

After completing this chapter you will have a complete picture of the op's shape contract. Chapter 3 uses that understanding to perform a root-cause analysis of the PCC ~0.71 bug that appears when non-tile-aligned `rotary_dim` values are used naively.

Proceed to [Chapter 3 — Root Cause Analysis of the PCC ~0.71 Bug](../ch3_bug_root_cause/index.md) after finishing all three files above.
