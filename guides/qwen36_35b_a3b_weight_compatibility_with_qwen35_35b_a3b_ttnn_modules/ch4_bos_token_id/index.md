# Chapter 4: `bos_token_id` and Generation Loop Initialization

## Overview

This chapter traces the effect of the new `bos_token_id: 248044` field in
**Qwen3.6-35B-A3B** through two distinct code paths: HuggingFace's
`GenerationMixin.generate()` utility and TT-Symbiote's custom generation loop.
It establishes whether the field changes any tensor content passed to the TTNN
forward pass, identifies the failure mode when auto-prepend is not suppressed,
and provides a concrete safe recipe for both paths.

> **Key Finding:** `bos_token_id = 248044` is **out of range** for the
> embedding table. `vocab_size = 151,936` for Qwen3.6-35B-A3B; valid token IDs
> are in the range `[0, 151935]`. Token ID `248,044` exceeds this bound. If any
> code path prepends or uses this ID for an embedding lookup, it will raise an
> error on CPU/GPU or silently return garbage values on TTNN device. The field
> has **no effect** on TTNN tensors when pre-formed `input_ids` (from the
> tokenizer) are used. **Action required:** suppress BOS auto-prepend in both
> HuggingFace `generate()` calls and TT-Symbiote's generation loop.

## Prerequisites

- [Chapter 1](../ch1_config_diff/index.md) — establishes the config diff;
  `bos_token_id: 248044` is introduced as a new top-level field in
  [`new_and_modified_fields.md`](../ch1_config_diff/new_and_modified_fields.md)
- [Chapter 2](../ch2_weight_shapes/index.md) — confirms that `vocab_size =
  151,936` is identical between both checkpoints, establishing the embedding
  table bound
- [Chapter 3](../ch3_partial_rotary_factor/index.md) — no dependency, but
  reading order is Chapters 1–3 before this chapter

## Contents

| File | Description |
|---|---|
| [`hf_generation_usage.md`](./hf_generation_usage.md) | How `GenerationMixin.generate()` consumes `bos_token_id`; the auto-prepend condition; tokenizer-side behavior; Qwen3.5 vs Qwen3.6 comparison; mitigation |
| [`tt_symbiote_generation_loop.md`](./tt_symbiote_generation_loop.md) | How TT-Symbiote's generation loop initializes the first input token; KV cache, position ID, and paged KV cache impact; the out-of-range embedding ID on TTNN device; safe recipe |

## Navigation

- Previous chapter: [Chapter 3 — `partial_rotary_factor` Promotion and RoPE Resolution](../ch3_partial_rotary_factor/index.md)
- Next chapter: [Chapter 5 — MTP Head: Weight Loading and Inference Impact](../ch5_mtp_head/index.md)
