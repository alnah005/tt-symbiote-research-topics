# Chapter 3 — GatedAttention: Full-Attention Layers

## Overview

This chapter covers `GatedAttention`, the module that handles the 1/4 full-attention layers in Qwen3.5 (16 of 64 layers in the 27B dense model; 10 of 40 in the 35B-A3B MoE model). `GatedAttention` subclasses the standard `Attention` base class and adds two mechanisms that are unique to Qwen3.5:

1. **Partial RoPE** — only the first 64 of 256 head dimensions are rotated, using frequencies computed over `rotary_dim=64` rather than `head_dim=256`.
2. **Output gate** — the attention output is element-wise multiplied by `sigmoid(x @ q_proj_gate.weight)` before the WO projection.

The chapter focuses exclusively on the deltas from standard attention. Readers unfamiliar with GQA, KV caching, or the base `Attention` class should consult Chapter 1 (architecture hyperparameters) and the base class source before reading this chapter.

## Reading Order

| File | Topic |
|------|-------|
| [`partial_rope.md`](./partial_rope.md) | `partial_rotary_factor=0.25`, the three failure modes of standard RoPE, corrected cos/sin frequencies, 27B vs A3B differences |
| [`output_gate.md`](./output_gate.md) | `q_proj_gate` split, sigmoid gate, `pre_wo_hook` mechanism, memory config handling |
| [`forward_flow.md`](./forward_flow.md) | Complete forward pass: save/delegate/hook call sequence, per-head RMSNorm, GQA, KV cache |

## Prerequisites

- **Chapter 1** — attention hyperparameters: `n_heads`, `n_kv_heads`, `head_dim`, `partial_rotary_factor`, `rope_theta`, `norm_eps`
- **Chapter 2** — GQA concept (KV head expansion via `repeat_interleave`) used in both DeltaNet and full-attention layers

## Source Files

- `models/tt_transformers/tt/gated_attention.py` — `GatedAttention` class (primary implementation)
- `models/demos/qwen35/demo/demo_a3b.py` — `build_model()`, where the corrected cos/sin matrices are patched into `HfRotarySetup`
- `models/tt_transformers/tt/rope.py` — `RotarySetup` (27B) and `HfRotarySetup` (A3B)
- `models/demos/qwen35/reference/test_attention_pcc.py` — single-layer PCC validation script
- `models/demos/qwen35/tests/test_pcc.py` — 27B pytest PCC suite (`PCC_THRESHOLD = 0.99`)
- `models/demos/qwen35/tests/test_a3b_pcc.py` — A3B pytest PCC suite (`PCC_THRESHOLD = 0.99`)

## Key Numbers at a Glance

| Quantity | 27B | 35B-A3B |
|----------|-----|---------|
| Full-attention layers | 16 (of 64) | 10 (of 40) |
| Q heads / KV heads | 24 / 4 | 16 / 2 |
| `head_dim` | 256 | 256 |
| `partial_rotary_factor` | 0.25 | 0.25 |
| `rotary_dim` | 64 | 64 |
| `rope_theta` | 1,000,000 | 1,000,000 |
| RoPE class | `RotarySetup` (Meta-style) | `HfRotarySetup` (HF-style) |
| PCC threshold | 0.99 | 0.99 |
