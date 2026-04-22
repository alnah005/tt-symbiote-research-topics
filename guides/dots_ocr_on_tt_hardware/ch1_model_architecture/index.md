# Chapter 1 — dots.ocr Model Architecture

## Overview

This chapter establishes the complete model specification of `DotsOCRForCausalLM`, the architecture behind dots.ocr. It covers the text decoder, the vision encoder, and the precise relationship between dots.ocr and Qwen 2.5 VL — which is a relationship of architectural lineage, not of shared weights.

dots.ocr is a multilingual document parser purpose-built for OCR, layout detection, table parsing, and formula recognition. The model card's '1.7B LLM foundation' refers to the text decoder (~1.78B); the full model including the vision encoder (~1.22B) is approximately 3.0B total (see [`text_decoder_hyperparameters.md`](./text_decoder_hyperparameters.md) and [`vision_encoder_specs.md`](./vision_encoder_specs.md) for the full derivation).

## Reading Order

| File | Contents |
|------|----------|
| [`text_decoder_hyperparameters.md`](./text_decoder_hyperparameters.md) | Full walkthrough of the text decoder config: hidden size, GQA 12Q/2KV, SwiGLU, RoPE, vocabulary |
| [`vision_encoder_specs.md`](./vision_encoder_specs.md) | Full vision_config walkthrough: 42 ViT layers, post-norm, spatial merge, token count formula, parameter breakdown |
| [`relationship_to_qwen25vl.md`](./relationship_to_qwen25vl.md) | Architectural lineage, shared identifiers, key divergences, why dots.ocr is derived not a fine-tune |

Read in order. Each file assumes the previous.

## dots.ocr vs Qwen 2.5 VL 7B — Side-by-Side Summary

The table below compares the full config of dots.ocr against Qwen2.5-VL-7B, the model from which dots.ocr inherits its architectural pattern.

| Field | dots.ocr | Qwen2.5-VL-7B |
|---|---|---|
| `hidden_size` | 1536 | 3584 |
| `num_hidden_layers` | 28 | 28 |
| `num_attention_heads` | 12 | 28 |
| `num_key_value_heads` | 2 | 4 |
| `intermediate_size` | 8960 | 18944 |
| `vocab_size` | 151936 | 151936 |
| `image_token_id` | 151665 | 151655 |
| `max_position_embeddings` | 131072 | 32768 |
| Vision layers | 42 | 32 |
| Vision `hidden_size` | 1536 | 1280 |
| Vision `patch_size` | 14 | 14 |
| Vision `spatial_merge_size` | 2 | 2 |
| Vision `temporal_patch_size` | 1 | 2 |
| Total parameters | ~3.0B (~1.78B text decoder + ~1.22B vision) | ~7.6B |

See [`relationship_to_qwen25vl.md`](./relationship_to_qwen25vl.md) for full analysis of each divergence.

Chapter 2 covers how these configuration differences affect the TTNN port strategy.
