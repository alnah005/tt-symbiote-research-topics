# Chapter 6 — Weight Precision, DRAM Layout, and Weight Conversion

## Overview

This chapter covers how Qwen3.5 HuggingFace checkpoints are converted to the internal meta format used by the TTNN model, the dtype choices made for each weight category, and the MoE key protection mechanism that prevents expert weight corruption during conversion.

## Prerequisites

- **Chapter 4** (Decoder Block and Uniform Dispatch) — for the weight key prefix conventions (`linear_attn.*`, `attention.*`, `feed_forward.*`) used after conversion
- **Chapter 5** (Mixture of Experts) — for understanding MoE key structure (`experts.gate_up_proj`, `experts.down_proj`, `mlp.gate.*`, `mlp.shared_expert.*`)

## Reading Order

1. [`dtype_choices.md`](./dtype_choices.md) — Per-weight-category dtype selection with DRAM breakdown numbers. Read this first to understand why different parts of the model are stored at different precisions.

2. [`hf_to_meta_conversion.md`](./hf_to_meta_conversion.md) — The 5-step pipeline in `qwen35_utils.py` that transforms a raw HuggingFace `state_dict` into the key namespace and weight layouts expected by the TTNN modules. Covers the interleaved `q_proj` gate split and the absence of `reverse_permute` for Qwen3.5.

3. [`moe_key_protection.md`](./moe_key_protection.md) — Why 3D expert tensors must be extracted from the state dict before calling `split_hf_keys`, and the `gate_proj` rename problem that required the pop-protect-reinsert pattern.

## Relationship to Other Chapters

This chapter is primarily reference material. The dtype choices documented here explain the DRAM budget numbers discussed in Chapter 5 (`ch5_moe/dram_budget.md`) and the performance numbers in Chapter 7. Readers who want to understand the performance profile before the implementation details can skip directly to Chapter 7 and return here when specific dtype or conversion questions arise.
