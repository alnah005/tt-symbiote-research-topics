# Chapter 5: MTP Head — Weight Loading and Inference Impact

## Prerequisites

Chapters 1–4 of this guide:

- Ch1: `../ch1_config_diff/` — Config field diff; establishes `mtp_num_hidden_layers: 1` and `vocab_size = 151,936`
- Ch2: `../ch2_weight_shapes/` — Backbone weight shape identity between Qwen3.5 and Qwen3.6
- Ch3: `../ch3_partial_rotary_factor/` — `partial_rotary_factor: 0.25` promotion as a no-op redundancy
- Ch4: `../ch4_bos_token_id/` — `bos_token_id: 248044` suppression; no TTNN tensor impact when input_ids are pre-formed

## Chapter Overview

This chapter addresses the most structurally novel change introduced by Qwen3.6-35B-A3B: the Multi-Token Prediction (MTP) head declared by `mtp_num_hidden_layers: 1`. The chapter establishes:

1. Whether the MTP head is training-only or inference-active in standard HuggingFace usage
2. The exact set of weight keys introduced by the MTP head (`model.future_prediction[0].*`)
3. The safe loading recipe for TT-Symbiote's weight preprocessing pipeline
4. The overall impact on existing backbone TTNN modules

> **Key Finding:** The MTP head is **training-only** in standard HuggingFace; `model.generate()` never invokes it. `AutoModelForCausalLM.from_pretrained` loads MTP weights into the model object without error. For TT-Symbiote's weight preprocessing pipeline, MTP keys must be explicitly filtered out — they are not consumed by any existing TTNN module and passing them through could trigger unexpected behavior in the weight preprocessing hooks. No changes to backbone TTNN modules are required.

## Files in This Chapter

| File | Description |
|---|---|
| `mtp_architecture.md` | MTP head architecture: module structure, weight keys, forward-pass gate, `lm_head` sharing, and comparison to Qwen3.5 |
| `loading_recipe.md` | MTP weight loading scenarios, TT-Symbiote impact analysis, key filter implementation, validation step, and impact summary table |

## Navigation

- Previous: [Chapter 4 — BOS Token ID](../ch4_bos_token_id/)
- Next: (end of guide)
