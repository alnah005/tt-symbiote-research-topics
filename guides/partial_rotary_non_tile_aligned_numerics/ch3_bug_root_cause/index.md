# Chapter 3 — Root Cause Analysis of the PCC ~0.71 Bug

This chapter explains exactly why `TTNNRotaryPositionEmbedding` produces PCC ~0.71 against a PyTorch reference when `rotary_dim=48` and `head_dim=128`. By the end of this chapter you will be able to trace the failure from the Python forward call through the TTNN op down to the element-level numerical corruption, and you will understand why no zero-padding scheme for cos/sin can produce correct partial RoPE output from `ttnn.experimental.rotary_embedding`.

---

> **Key Finding:** The PCC ~0.71 bug is caused by two compounded errors. First, `TTNNRotaryPositionEmbedding` pads cos/sin to `nearest_32(rotary_dim)=64` instead of the required `head_dim=128`, causing `TT_FATAL` in the normal execution path. Second, in any hypothetical execution path where cos/sin are further padded to 128 (e.g., by an alternate autoformat route), the kernel's rotate-half split operates at `head_dim/2=64` rather than `rotary_dim/2=24`, so zeros at positions 48–127 of cos/sin, combined with the kernel's fixed pairing offset of 64, corrupt **all 128 output elements**: the 48-element rotation region `[0, 48)` is entirely corrupted: positions `[0, 24)` receive wrong-paired rotations (offset 64 instead of the required 24), and positions `[24, 48)` receive structurally wrong outputs (kernel applies first-half formula to right-half positions, using wrong input elements, wrong frequency indices, and wrong combination rule); the 80-element passthrough region `[48, 128)` is zeroed or receives incorrect linear combinations of input values

---

## Recap of Chapter 1 and 2 Prerequisites

- `rotary_dim=48, head_dim=128`: correct partial RoPE rotates elements `[0, 48)` using pairs `(i, i+24)`, and passes elements `[48, 128)` through unchanged. The cos/sin table has `shape[-1]=48`. (Ch1: [`../ch1_rope_fundamentals/partial_rope_math.md`](../ch1_rope_fundamentals/partial_rope_math.md))
- `ttnn.experimental.rotary_embedding` enforces `TT_FATAL: cos.padded_shape()[-1] == input.padded_shape()[-1]`. With `head_dim=128`, cos/sin must have `shape[-1]=128`. Supplying `shape[-1]=64` causes this assertion to fire. (Ch2: [`../ch2_op_shape_contract/shape_validation_in_invoke.md`](../ch2_op_shape_contract/shape_validation_in_invoke.md))
- The kernel's rotate-half split is always at `head_dim/2=64` tiles. There is no mechanism to restrict rotation to a `rotary_dim` subset. (Ch2: [`../ch2_op_shape_contract/kernel_rotate_half_pairing.md`](../ch2_op_shape_contract/kernel_rotate_half_pairing.md))

---

## Learning Objectives

By the end of this chapter you should be able to:

1. Identify the two failure paths — `TT_FATAL` (Path A) and silent numerical corruption (Path B) — that arise from the incorrect cos/sin padding target in `TTNNRotaryPositionEmbedding`, and explain under what conditions each path is reached.
2. Trace element-level compute for positions `output[0]`, `output[24]`, `output[48]`, `output[64]`, and `output[127]` through the buggy kernel to show exactly which values are corrupted and why.
3. State the correct reference output for `rotary_dim=48, head_dim=128` at the formula level, and verify it with a PyTorch code example.
4. Explain why no zero-padding scheme for cos/sin can make `ttnn.experimental.rotary_embedding` produce correct partial RoPE output when `rotary_dim < head_dim`.

---

## Files in Reading Order

1. [**`step_by_step_failure_trace.md`**](./step_by_step_failure_trace.md) — The exact sequence of operations in `TTNNRotaryPositionEmbedding.forward` for `rotary_dim=48, head_dim=128`; Path A (TT_FATAL) and Path B (silent corruption); element-level numerical example.
2. [**`correct_partial_rope_reference.md`**](./correct_partial_rope_reference.md) — The correct reference output; PyTorch code; why the kernel's `head_dim/2` split cannot be reconciled with partial RoPE via any zero-padding scheme.

---

## What's Next

After completing this chapter you will understand why `TTNNRotaryPositionEmbedding`'s current zero-padding approach cannot be salvaged for `rotary_dim < head_dim` configurations. Chapter 4 presents three implementation strategies that do produce correct output — including Strategy C (identity-filled precomputed cos/sin table), which is also fully trace-compatible.

Proceed to [Chapter 4 — Correct Implementation Strategies for Non-Tile-Aligned Partial RoPE](../ch4_implementation_strategies/index.md) after finishing both files above.
