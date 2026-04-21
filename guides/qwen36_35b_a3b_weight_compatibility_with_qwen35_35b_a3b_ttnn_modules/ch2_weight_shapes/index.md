# Chapter 2 — Weight Tensor Shape Analysis

## Overview

This chapter answers the core compatibility question at the tensor level: given that Qwen3.6-35B-A3B and Qwen3.5-35B-A3B share identical architecture hyperparameters (as established in Chapter 1), do their weight tensors actually carry the same shapes? If so, existing TTNN modules compiled and tested against Qwen3.5 checkpoints can consume Qwen3.6 checkpoints without modification.

The chapter enumerates every logical weight group in the shared backbone, confirms that each group's shape is governed solely by the hyperparameters documented in Chapter 1, and then separately catalogs the 11 new keys present only in the Qwen3.6 checkpoint (the Multi-Token Prediction head). Understanding both sets is required before writing or modifying any TTNN weight-loading code.

---

## Prerequisites

| Prerequisite | Why it matters |
|---|---|
| Familiarity with Chapter 1 Config Diff findings | Shape analysis depends on `hidden_size`, `num_attention_heads`, `head_dim`, `num_key_value_heads`, `moe_intermediate_size`, `intermediate_size`, and `num_experts` — all established there |
| Knowledge of HuggingFace safetensors format | Weight keys follow `safetensors` naming conventions; verification pseudocode uses the `safetensors` Python library |
| Awareness of GQA weight shapes | Qwen3.5/3.6 use Grouped Query Attention with 64 Q heads and 8 KV heads; Q, K, V projections therefore have asymmetric output dimensions |

---

## Summary Finding

> **All backbone weight shapes are identical between Qwen3.5-35B-A3B and Qwen3.6-35B-A3B.**
>
> Qwen3.6 adds exactly 11 weight keys under the prefix `model.future_prediction.0.*` belonging to the Multi-Token Prediction (MTP) head. These keys total approximately 440 million parameters. No key in this set overlaps with any backbone weight key, and no existing TTNN module is affected by their presence.

---

## Reading Order

1. [shared_weight_shapes.md](./shared_weight_shapes.md) — Complete enumeration of backbone weight keys and shapes, grouped by component; shape identity confirmation table; verification pseudocode.
2. [extra_weight_keys.md](./extra_weight_keys.md) — The 11 MTP head keys exclusive to Qwen3.6; parameter count; how `from_pretrained` and TT-Symbiote weight loading handle unexpected keys; safe loading recipe.
