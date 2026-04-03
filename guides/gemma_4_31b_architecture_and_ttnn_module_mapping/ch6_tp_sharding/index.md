# Chapter 6 --- Tensor-Parallel Sharding on T3K

## Overview

This chapter defines the optimal tensor-parallel (TP) sharding strategy for
Gemma 4 31B across the T3K 1x8 Wormhole mesh (8 devices, 12 GB DRAM each).
The central challenge is that TP=8 creates a clean split for query heads
(32 Q heads / 8 devices = 4 per device) and for sliding-layer KV heads
(16 KV heads / 8 devices = 2 per device), but produces a fractional result
for global-layer KV heads (4 KV heads / 8 devices = 0.5 per device). This
fractional split cannot be realized by assigning whole heads to devices, so
global layers require a different KV sharding strategy.

After reading this chapter you will know:

- Why TP=8 is the natural choice for T3K and where it breaks down for Gemma 4's
  global attention layers.
- Four candidate strategies for handling 4 global KV heads across 8 devices,
  with memory and CCL cost analysis for each.
- The recommended approach and its rationale.
- How column-parallel and row-parallel weight sharding maps to TTNN's
  distributed linear module classes.
- The per-device KV cache memory budget for both layer types at various
  sequence lengths.

## Central Challenge

Gemma 4 31B uses heterogeneous attention with two distinct KV head counts:

| Layer Type | Q Heads | KV Heads | head_dim | Q/device (TP=8) | KV/device (TP=8) |
|------------|---------|----------|----------|-----------------|-------------------|
| Sliding (50 layers) | 32 | 16 | 256 | 4 | 2 (clean) |
| Global (10 layers) | 32 | 4 | 512 | 4 | **0.5 (fractional)** |

The 32 Q heads divide cleanly into 4 per device for both layer types. The 16
sliding KV heads divide cleanly into 2 per device, giving a per-device GQA
ratio of 4Q:2KV = 2:1. But the 4 global KV heads cannot be evenly distributed
across 8 devices --- each device would need half a head, which is not a
meaningful unit for grouped-query attention.

This asymmetry is the defining TP challenge for Gemma 4 31B and must be
resolved before any implementation work begins. The resolution affects weight
sharding, KV cache layout, SDPA dispatch, and CCL communication patterns for
all 10 global layers.

## Reading Order

1. [`sharding_strategy_analysis.md`](./sharding_strategy_analysis.md) --- Four
   options for global KV head sharding, with memory and CCL analysis, and a
   recommendation.
2. [`weight_sharding.md`](./weight_sharding.md) --- Column-parallel and
   row-parallel weight sharding for all projections, per-device shapes, and
   TTNN linear module compatibility.
3. [`kv_cache_sharding.md`](./kv_cache_sharding.md) --- Per-device KV cache
   memory budget, page table configuration, and total DRAM budget across all 60
   layers.

## Prerequisites

This chapter builds on:

- [Chapter 1 --- Architecture Overview](../ch1_architecture_overview/index.md):
  head counts, head dimensions, hidden_size, and the 50/10 layer split.
- [Chapter 2 --- Projection Weights and Tensor Shapes](../ch2_projection_shapes/index.md):
  all weight shapes and per-layer parameter counts.
- [Chapter 5 --- Heterogeneous Attention Module Design](../ch5_attention_module_design/index.md):
  the attention module class hierarchy and forward pass dataflow.

## Key Constants

| Parameter | Value |
|-----------|-------|
| `hidden_size` | 5376 |
| `intermediate_size` | 21504 |
| `num_attention_heads` (Q) | 32 |
| `num_key_value_heads` (sliding KV) | 16 |
| `num_global_key_value_heads` (global KV) | 4 |
| `head_dim` (sliding) | 256 |
| `global_head_dim` (global) | 512 |
| `sliding_window` | 1024 tokens |
| `max_position_embeddings` | 262144 (256K) |
| T3K device count | 8 |
| DRAM per device | 12 GB |
| TP degree | 8 |

---

**Next:** [`sharding_strategy_analysis.md`](./sharding_strategy_analysis.md)
