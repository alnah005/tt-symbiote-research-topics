# Chapter 5 --- Heterogeneous Attention Module Design

## Overview

Gemma 4 31B presents a unique TTNNModule design challenge: its 60 decoder
layers contain two structurally different attention configurations that share
the same position in the decoder layer pipeline (between pre-attention RMSNorm
and the residual add) but differ in nearly every internal parameter. The 50
sliding layers use 16 KV heads with `head_dim=256`, a 1024-token window,
standard RoPE, and separate K/V projections. The 10 global layers use 4 KV
heads with `head_dim=512`, full causal attention, proportional RoPE with
partial rotation, and K=V weight sharing.

This chapter addresses the central design question: **how should the
TTNNModule class hierarchy be structured to handle these two attention types?**

The options range from a single unified class with conditional branches, to
completely separate classes, to a base-class-with-subclasses pattern. Each
approach has different implications for code maintainability, per-type
optimization, and the complexity of the decoder layer's dispatch logic.

Beyond the class design, this chapter provides the complete decode forward
pass for each attention type --- step-by-step tensor transformations with
shapes, TTNN pseudocode, and analysis of how paged KV cache and SDPA interact
with the sliding window constraint.

## Central Question

Should the TTNN implementation use:

1. **One class** --- `TTNNGemma4Attention` with a `layer_type` parameter and
   conditional branches in `forward`?
2. **Two classes** --- `TTNNGemma4SlidingAttention` and
   `TTNNGemma4GlobalAttention`, each fully self-contained?
3. **A base class with subclasses** --- `TTNNGemma4AttentionBase` containing
   shared logic (Q projection, O projection, V-norm), with
   `TTNNGemma4SlidingAttention` and `TTNNGemma4GlobalAttention` as subclasses
   that override the type-specific steps (KV projection, RoPE, SDPA)?

The answer depends on how much logic is truly shared between the two types,
how different the optimization paths are, and how the decoder layer dispatches
to the correct attention module. See
[`design_options.md`](./design_options.md) for the full analysis and
recommendation.

## Reading Order

1. [`design_options.md`](./design_options.md) --- Three design options
   analyzed with pros, cons, and a recommendation.
2. [`sliding_attention_forward.md`](./sliding_attention_forward.md) --- The
   complete decode forward pass for sliding-window attention with TTNN
   pseudocode.
3. [`global_attention_forward.md`](./global_attention_forward.md) --- The
   complete decode forward pass for global attention with K=V sharing and
   partial RoPE.
4. [`paged_sdpa_sliding_window.md`](./paged_sdpa_sliding_window.md) ---
   Investigation into how `paged_sdpa_decode` interacts with the
   `sliding_window_size` parameter.

## Prerequisites

This chapter builds on:

- [Chapter 2 --- Projection Weights and Tensor Shapes](../ch2_projection_shapes/index.md):
  all projection weight shapes and activation tensor shapes for both layer
  types.
- [Chapter 3 --- K=V Sharing and V-Norm](../ch3_kv_sharing_and_vnorm/index.md):
  the K=V sharing mechanism in global layers and V-norm implementation across
  all layers.
- [Chapter 4 --- Dual RoPE and Partial Rotary Embedding](../ch4_dual_rope/index.md):
  the two RoPE configurations and their TTNN mapping.
- [Windowed Attention Foundations and T3K Mapping](../../windowed_attention_foundations_and_t3k_mapping/index.md):
  circular buffer KV cache design, paged SDPA windowing strategies, and
  decode primitives.

## Key Parameters Quick Reference

| Parameter | Sliding (50 layers) | Global (10 layers) |
|-----------|--------------------|--------------------|
| `num_attention_heads` | 32 | 32 |
| `num_kv_heads` | 16 | 4 |
| `head_dim` | 256 | 512 |
| `hidden_size` | 5376 | 5376 |
| Q weight shape | [5376, 8192] | [5376, 16384] |
| K weight shape | [5376, 4096] | [5376, 2048] |
| V weight shape | [5376, 4096] | None (K=V sharing) |
| O weight shape | [8192, 5376] | [16384, 5376] |
| Fused QKV/QK shape | [5376, 16384] | [5376, 18432] |
| RoPE type | Standard (theta=10K) | p-RoPE (theta=1M) |
| Rotary dims | 256/256 (100%) | 128/512 (25%) |
| Window | 1024 tokens | Full causal |
| K=V sharing | No | Yes |
| V-norm | Yes (unscaled) | Yes (unscaled) |
| K-norm | Yes (scaled) | Yes (scaled) |
| GQA ratio (Q:KV) | 2:1 | 8:1 |

---

**Next:** [`design_options.md`](./design_options.md)
