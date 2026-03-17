# Expert Weight Memory Layout Optimization

## Overview

MoE inference on Wormhole B0 and T3K stalls on memory bandwidth, not compute. In the decode regime — where batch sizes are small and each forward pass reads tens of gigabytes of expert weights while performing relatively few FLOPs — how expert weight tensors are laid out in DRAM determines whether the hardware achieves near-peak bandwidth or loses 20–40% to NoC contention. This guide explains exactly how to map MoE expert weight tensors (Qwen 235B-A22B, DeepSeek-V3, Mixtral 8x7B) to DRAM-sharded `MemoryConfig` objects in TTNN, why DRAM-sharded layout reduces NoC hotspots, how tile-alignment rules constrain valid shard geometries, how quantization changes the tile footprint arithmetic, and how to measure the end-to-end bandwidth improvement with a reproducible benchmark. The scope is Wormhole B0 single-chip and T3K eight-chip configurations; all code targets the `ttnn` Python API.

---

## Prerequisites

- Comfortable creating TTNN tensors and running `ttnn.matmul`, `ttnn.to_device`, `ttnn.from_device`
- Familiar with `ttnn.DRAM_MEMORY_CONFIG` and `ttnn.L1_MEMORY_CONFIG`
- Basic understanding of MoE FFN structure: routing, expert selection, per-expert matrix multiply
- Awareness of Wormhole B0 hardware (DRAM hierarchy, Tensix core grid) at a conceptual level

---

## Chapter Table

| Chapter | Title | What you learn |
|---|---|---|
| 1 | TTNN Memory Architecture | Wormhole B0 DRAM topology (6 controllers, 12 banks, 12 GB), L1 per-core (1.5 MB), NoC hop model, `ttnn.MemoryConfig` API, interleaved vs sharded allocation |
| 2 | DRAM-Sharded Memory Layout | `ttnn.ShardSpec` fields (`grid`, `shape`, `orientation`), `CoreRange`/`CoreRangeSet` construction, HEIGHT_SHARDED vs WIDTH_SHARDED vs BLOCK_SHARDED strategies, constructing and verifying a DRAM-sharded `MemoryConfig` |
| 3 | Expert Weight Tensor Structure | Gate/up/down projection shapes for Mixtral 8x7B, DeepSeek-V3, and Qwen 235B-A22B; shard grid selection rules; dtype memory footprint (BF16, bfloat8_b); TILE_LAYOUT alignment requirement |
| 4 | Prefetch Patterns and Bandwidth | NoC packet model, DRAM controller column topology, why interleaved access creates hotspots, how DRAM-sharded layout eliminates them, double-buffering for DMA prefetch, roofline analysis for decode vs prefill |
| 5 | Tile Size Constraints | 32-element tile constraint on height and width, all five shard-shape alignment rules, page-alignment of shard byte sizes, common pitfalls and how to diagnose them |
| 6 | Performance Analysis and Trade-offs | Bandwidth efficiency gap (interleaved vs DRAM-sharded), decode regime rule of thumb (`batch_size × top_k ≤ 16`), reshard overhead at load time, four-regime trade-off matrix, T3K multi-chip interactions |
| 7 | Implementation and Validation | End-to-end code: weight loading, shard config construction, `ttnn.to_memory_config`, `ttnn.matmul` integration; PCC-based correctness verification; reproducible benchmark harness |

---

## Key Facts

Quick reference for practitioners. Full derivations are in the chapters indicated.

**Model parameters (Qwen 235B-A22B / DeepSeek-V3)**
- `d_model` = 7168, `d_ff` = 2048, `num_experts` = 128, `top_k` = 8

**Per-expert memory (BF16)**
- Gate projection: 7168 × 2048 × 2 bytes = 28.0 MB
- Up projection: 7168 × 2048 × 2 bytes = 28.0 MB
- Down projection: 2048 × 7168 × 2 bytes = 28.0 MB
- Total per expert (BF16): **84.0 MB**

**Effective_M crossover (compute-bound boundary)**
- Qwen 235B-A22B: effective_M ≈ 556 (= 437 × 2048 / 1611)
- Mixtral 8x7B: effective_M ≈ 451
- Below crossover: bandwidth-bound → DRAM sharding beneficial
- Above crossover: compute-bound → interleaved sufficient

**Decode regime rule**
- `batch_size × top_k ≤ 16` → DRAM sharding delivers −30 to −50% latency improvement
- Derivation: Ch6 `bandwidth_gain_analysis.md`

**L1 weight double-buffer formula**
- `2 × in0_block_w × per_core_N_t × tile_size_bytes`
- Must fit within 1.5 MB L1 per core
- See Ch4 `sharded_access_pattern.md`

**Reshard overhead (Mixtral 8x7B)**
- ~2.3 s one-time cost at model load: 24 tensors/layer × 32 layers × ~3 ms/tensor
- Negligible for deployment when performed at load time, not per-request

**T3K ethernet bandwidth**
- ~7 GB/s effective per link
- Expert dispatch across chips must account for this; Ch6 `tradeoff_matrix.md` covers multi-chip interaction

**Wormhole B0 hardware constants**
- DRAM controllers: 6 | GDDR6 banks: 12 | Peak DRAM BW: ~300 GB/s
- Tensix cores: 80 (8×10) | L1 per core: 1.5 MB | Ridge point: ~437 FLOP/byte
- BF16 tile: 2,048 bytes | bfloat8_b tile: 1,024 bytes | Page size: 32 bytes

---

## How to Read This Guide

**First-time readers:** Work through chapters 1–7 in order. Each chapter assumes the API vocabulary and hardware model established by the chapters before it. Chapter 7 contains runnable code that references concepts from all six preceding chapters; attempting it out of order will leave gaps.

**Practitioners already familiar with TTNN sharding:** Use the chapter table and key facts block above as a reference. The trade-off matrix in Ch6 `tradeoff_matrix.md` and the implementation checklist in Ch7 `index.md` are the two most-referenced artifacts in day-to-day work.

---

## Chapter Navigation

| Chapter | Description |
|---|---|
| [Ch 1 — TTNN Memory Architecture](ch01_ttnn_memory_architecture/index.md) | Wormhole B0 DRAM topology (6 controllers, 12 banks, 12 GB), L1 per-core (1.5 MB), NoC hop model, `ttnn.MemoryConfig` API, interleaved vs sharded allocation |
| [Ch 2 — DRAM-Sharded Memory Layout](ch02_dram_sharded_memory_layout/index.md) | `ttnn.ShardSpec` fields (`grid`, `shape`, `orientation`), `CoreRange`/`CoreRangeSet` construction, HEIGHT_SHARDED vs WIDTH_SHARDED vs BLOCK_SHARDED strategies, constructing and verifying a DRAM-sharded `MemoryConfig` |
| [Ch 3 — Expert Weight Tensor Structure](ch03_expert_weight_tensor_structure/index.md) | Gate/up/down projection shapes for Mixtral 8x7B, DeepSeek-V3, and Qwen 235B-A22B; shard grid selection rules; dtype memory footprint (BF16, bfloat8_b); TILE_LAYOUT alignment requirement |
| [Ch 4 — Prefetch Patterns and Bandwidth](ch04_prefetch_patterns_and_bandwidth/index.md) | NoC packet model, DRAM controller column topology, why interleaved access creates hotspots, how DRAM-sharded layout eliminates them, double-buffering for DMA prefetch, roofline analysis for decode vs prefill |
| [Ch 5 — Tile Size Constraints](ch05_tile_size_constraints/index.md) | 32-element tile constraint on height and width, all five shard-shape alignment rules, page-alignment of shard byte sizes, common pitfalls and how to diagnose them |
| [Ch 6 — Performance Analysis and Trade-offs](ch06_performance_analysis_and_tradeoffs/index.md) | Bandwidth efficiency gap (interleaved vs DRAM-sharded), decode regime rule of thumb (`batch_size × top_k ≤ 16`), reshard overhead at load time, four-regime trade-off matrix, T3K multi-chip interactions |
| [Ch 7 — Implementation and Validation](ch07_implementation_and_validation/index.md) | End-to-end code: weight loading, shard config construction, `ttnn.to_memory_config`, `ttnn.matmul` integration; PCC-based correctness verification; reproducible benchmark harness |
