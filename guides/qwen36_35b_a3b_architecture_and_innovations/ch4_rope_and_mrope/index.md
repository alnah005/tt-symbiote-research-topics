# Chapter 4 — Partial Rotary Embedding and M-RoPE

## Overview

Qwen3.6-35B-A3B uses **two distinct position encoding schemes** depending on the layer type:

| Layer type | Applies every | Position encoding |
|---|---|---|
| Gated Attention | Every 4th layer (indices 0, 4, 8, …) | Partial RoPE (rotary_dim = 64 of head_dim = 256) |
| Gated DeltaNet | Remaining three out of four layers | **None** — Q and K are L2-normalized but carry no positional signal |

This split is fundamental to the hybrid architecture. Gated DeltaNet layers maintain recurrent state that implicitly tracks relative order; they do not need, and do not receive, any rotary positional encoding. Only the Gated Attention layers — which perform full softmax attention over the KV cache — require explicit position information.

Within Gated Attention, two related but distinct RoPE variants are active depending on whether the model is running text-only or multimodal inference:

- **Partial RoPE** — used in all Gated Attention layers; applies rotary encoding only to the first 64 of the 256 head dimensions.
- **M-RoPE (Multimodal RoPE)** — a superset of partial RoPE that assigns independent temporal, height, and width position IDs to vision tokens; degenerates to standard RoPE for text tokens.

---

## Learning Objectives

After reading this chapter you will be able to:

1. Explain why only Gated Attention layers receive RoPE and why partial application (rotary_dim = 64) outperforms full application (rotary_dim = 256) at long context.
2. Derive the cos/sin matrices from first principles given `rope_theta = 10,000,000` and `rotary_dim = 64`.
3. Describe the role of Q/K RMSNorm in Gated Attention and distinguish it from the L2 normalization in Gated DeltaNet.
4. Explain the mrope_section = [11, 11, 10] split and how M-RoPE degenerates to standard RoPE for text-only tokens.
5. Identify which components of the TTNN implementation require changes (none for text-only deployment).

---

## Files in This Chapter

- [partial_rotary_embedding.md](./partial_rotary_embedding.md) — Head dimensions, frequency spectrum, Q/K RMSNorm, and TTNN deployment notes.
- [mrope_multimodal_positions.md](./mrope_multimodal_positions.md) — mrope_section split, interleaved layout, per-token position IDs, and text-only equivalence.

---

## Key Parameters at a Glance

| Parameter | Value | Scope |
|---|---|---|
| `head_dim` (d_h) | 256 | Gated Attention Q/K/V heads |
| `num_attention_heads` (n_q) | 16 | Query heads per layer |
| `num_key_value_heads` (n_kv) | 2 | KV heads per layer (GQA) |
| `partial_rotary_factor` | 0.25 | Fraction of head_dim that receives RoPE |
| `rotary_dim` | 64 | = head_dim × partial_rotary_factor |
| `rope_theta` | 10,000,000 | Base frequency |
| `max_position_embeddings` | 262,144 | Supported context length |
| `mrope_section` | [11, 11, 10] | Rotary-pair split across temporal / height / width |
| `mrope_interleaved` | true | cos/sin pair layout |

---

## Navigation

- **Previous:** [Chapter 3 — Qwen3.5 vs Qwen3.6: Exact Differences](../ch3_qwen35_vs_qwen36_differences/index.md)
- **Next:** [Chapter 5 — Multi-Token Prediction](../ch5_multi_token_prediction/index.md)
