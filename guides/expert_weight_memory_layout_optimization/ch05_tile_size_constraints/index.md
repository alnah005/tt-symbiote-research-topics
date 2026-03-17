# Chapter 5: Tile Size Constraints

## Overview

Every tensor in TILE_LAYOUT on Wormhole B0 must have dimensions that are exact multiples of 32 in both height and width. This requirement propagates directly into shard shape derivation: if a shard boundary falls mid-tile, TTNN raises an error at `ttnn.to_memory_config` time or produces a silently misaligned memory layout. This chapter specifies all tile-level alignment rules that govern valid shard configurations for expert weight tensors, explains the consequences of each rule, and provides a complete derivation checklist.

The chapter builds on the `ShardSpec` API from Chapter 2, `shard_spec_deep_dive.md`, and the concrete expert weight shapes established in Chapter 3, `projection_shapes.md`. After completing this chapter you will be able to derive a tile-valid shard configuration for any expert weight tensor without trial-and-error.

---

## Learning Objectives

1. **State the 32-element tile constraint** and explain why it applies to both the height and width dimensions of any TILE_LAYOUT shard independently.
2. **Compute tile memory footprints** for `bfloat16`, `bfloat8_b`, and `bfloat4_b` and use them to verify page alignment of shard byte sizes.
3. **Apply all five shard-shape alignment rules** to derive a valid `ShardSpec.shape` from a given tensor shape and core grid.
4. **Recognize and avoid the five common pitfalls** that produce alignment errors or silent performance degradation.
5. **Distinguish tile-count notation (`M_t`, `N_t`) from element notation** and translate between the two when constructing `ShardSpec` or reading program config output.

---

## Quick-Reference: Tile Alignment Checklist

Use this checklist before calling `ttnn.to_memory_config` with a sharded layout. All conditions must be true.

| # | Check | Formula | Pass condition |
|---|---|---|---|
| 1 | Shard height is tile-aligned | `shard_H % 32` | `== 0` |
| 2 | Shard width is tile-aligned | `shard_W % 32` | `== 0` |
| 3 | Tensor height divides evenly by shard height | `tensor_H / shard_H` | Integer, equals `num_cores_H` |
| 4 | Tensor width divides evenly by shard width | `tensor_W / shard_W` | Integer, equals `num_cores_W` |
| 5 | Shard byte size is page-aligned | `(shard_H * shard_W * bytes_per_element) % 32` | `== 0` |
| 6 | Tensor is in TILE_LAYOUT before sharding | `tensor.layout` | `ttnn.TILE_LAYOUT` |
| 7 | `ShardSpec.shape` is in elements, not tiles | `[shard_H, shard_W]` where `shard_H >= 32` | Both values are element counts |

---

## Prerequisites

| Chapter | Topics Covered |
|---|---|
| Chapter 1 | TILE_LAYOUT vs ROW_MAJOR_LAYOUT; 32×32 tile as atomic unit |
| Chapter 2 | `ShardSpec.shape` field semantics; `TensorMemoryLayout` strategies |
| Chapter 3 | Expert weight projection shapes; `d_model`, `d_ff` for Mixtral 8x7B and DeepSeek-V3 |

---

## Chapter Structure

| File | Contents |
|---|---|
| [`tile_fundamentals.md`](./tile_fundamentals.md) | 32×32 tile unit; zero-padding behavior; tile memory footprints by dtype; tile-count notation |
| [`shard_shape_alignment_rules.md`](./shard_shape_alignment_rules.md) | Five alignment rules with consequences; worked derivation for Mixtral gate projection |
| [`common_pitfalls.md`](./common_pitfalls.md) | Five pitfalls that produce errors or silent misalignment; diagnostic guidance |

---

## Key Constants (Wormhole B0) — Tile-Specific

| Parameter | Value |
|---|---|
| Tile dimensions | 32 × 32 elements |
| BF16 tile size | 2,048 bytes (32 × 32 × 2) |
| bfloat8_b tile size | 1,024 bytes (32 × 32 × 1) |
| bfloat4_b tile size | 512 bytes (32 × 32 × 0.5) |
| Page size (DRAM, Wormhole) | 32 bytes |

See Chapter 4, `index.md` for DRAM bandwidth, core grid, and L1 constants.

---

## Next Steps

After verifying that your shard configuration passes all checklist items above, proceed to Chapter 6, `index.md`, for performance analysis: how tile-valid shard configurations translate to measurable bandwidth gains in the decode regime.
