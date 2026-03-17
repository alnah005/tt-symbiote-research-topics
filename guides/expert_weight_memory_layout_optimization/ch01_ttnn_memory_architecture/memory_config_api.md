# The `ttnn.MemoryConfig` API

## Overview

Every tensor in TTNN carries a `MemoryConfig` that tells the runtime three things:

1. Which physical memory tier to use (DRAM or L1)
2. How to distribute the tensor's pages across that tier (interleaved or one of the sharding strategies)
3. If sharded, the exact grid of cores or banks and the per-shard dimensions

Understanding `MemoryConfig` is a prerequisite for all sharding work because it is the only mechanism for controlling these properties. There is no alternative path — placement is not inferred from tensor shape or op type.

---

## `ttnn.MemoryConfig` Constructor

```python
ttnn.MemoryConfig(
    memory_layout: ttnn.TensorMemoryLayout,
    buffer_type: ttnn.BufferType,
    shard_spec: ttnn.ShardSpec | None = None,
)
```

### Arguments

**`memory_layout`** (`ttnn.TensorMemoryLayout`)
Controls how the tensor's pages are distributed across the target memory tier. See the `TensorMemoryLayout` section below for the full set of values. For the default interleaved behavior, use `ttnn.TensorMemoryLayout.INTERLEAVED`.

**`buffer_type`** (`ttnn.BufferType`)
Selects the physical memory tier. Either `ttnn.BufferType.DRAM` or `ttnn.BufferType.L1`. This determines which allocator handles the buffer and which physical addresses are used.

**`shard_spec`** (`ttnn.ShardSpec`, optional)
Required when `memory_layout` is any sharded variant (`HEIGHT_SHARDED`, `WIDTH_SHARDED`, `BLOCK_SHARDED`). Must be `None` for `INTERLEAVED` and `SINGLE_BANK`. See Chapter 2 for a full treatment of `ShardSpec` construction.

---

## `ttnn.BufferType` Enum

```python
class ttnn.BufferType(Enum):
    DRAM   = ...   # Off-chip GDDR6; 12 GB total on Wormhole B0
    L1     = ...   # On-chip SRAM; 1.5 MB per Tensix core
```

`DRAM` buffers are allocated in the off-chip GDDR6 memory managed by the six DRAM controllers. They persist for the lifetime of the tensor object and are accessible from any Tensix core via the NoC.

`L1` buffers are allocated in the on-chip L1 SRAM of one or more Tensix cores. L1 buffers are typically short-lived (within a single op or a fused kernel sequence) because they consume the per-core 1.5 MB budget. Accessing an L1 buffer from a different core than the one that owns it requires a NoC transfer, though this is faster than a DRAM access.

> **Warning:** L1 buffers that are not explicitly freed before the next op that requires their memory will cause allocation failures at runtime. Use `ttnn.deallocate(tensor)` or scope L1 tensors carefully within a graph.

---

## `ttnn.TensorMemoryLayout` Enum

```python
class ttnn.TensorMemoryLayout(Enum):
    INTERLEAVED    = ...
    HEIGHT_SHARDED = ...
    WIDTH_SHARDED  = ...
    BLOCK_SHARDED  = ...
    SINGLE_BANK    = ...
```

### `INTERLEAVED`

The default layout. TTNN distributes the tensor's pages in round-robin order across all available banks of the target tier. For `buffer_type=DRAM`, this means pages are spread across all 12 GDDR6 banks. For `buffer_type=L1`, pages are spread across all cores in the device's L1 pool.

No `shard_spec` is required or permitted.

### `HEIGHT_SHARDED`

The tensor is partitioned along its height dimension (rows) into contiguous chunks. Each chunk (shard) is assigned to one core (for L1) or one DRAM bank group. All shards have the same width as the full tensor.

Requires a `shard_spec` that specifies the core grid and the per-shard shape.

### `WIDTH_SHARDED`

The tensor is partitioned along its width dimension (columns). Each shard has the same height as the full tensor. Suitable when the weight matrix is partitioned along the output feature dimension, which is the column-parallel convention.

Requires a `shard_spec`.

### `BLOCK_SHARDED`

The tensor is partitioned in two dimensions simultaneously: rows and columns. Each shard is a rectangular sub-block. Requires a 2D core grid (e.g., 4×8 or 8×8). Useful for very large weight tensors where neither pure height nor pure width sharding provides sufficient parallelism.

Requires a `shard_spec` with a 2D grid.

### `SINGLE_BANK`

The entire tensor is placed in a single bank. Primarily used for small tensors that must be co-located (e.g., bias vectors that are read once and reused). Rarely used for weight tensors in MoE.

No `shard_spec` is required.

---

## Predefined Configs: `DRAM_MEMORY_CONFIG` and `L1_MEMORY_CONFIG`

TTNN exports two convenience objects that cover the most common interleaved cases:

```python
# Equivalent to:
# ttnn.MemoryConfig(
#     memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
#     buffer_type=ttnn.BufferType.DRAM,
# )
ttnn.DRAM_MEMORY_CONFIG

# Equivalent to:
# ttnn.MemoryConfig(
#     memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
#     buffer_type=ttnn.BufferType.L1,
# )
ttnn.L1_MEMORY_CONFIG
```

You can verify this yourself:

```python
import ttnn
print(ttnn.DRAM_MEMORY_CONFIG)
# MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED, buffer_type=BufferType::DRAM, shard_spec=std::nullopt)

print(ttnn.L1_MEMORY_CONFIG)
# MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED, buffer_type=BufferType::L1, shard_spec=std::nullopt)
```

These objects are immutable. You cannot modify them in place; if you need a non-default layout with DRAM buffer type, construct a new `MemoryConfig` explicitly.

---

## Constructing a Custom `MemoryConfig`

### DRAM Interleaved (explicit form)

This is identical to `ttnn.DRAM_MEMORY_CONFIG` but written out explicitly. Useful when you want to be unambiguous in code that will be reviewed by someone unfamiliar with the predefined constants:

```python
import ttnn

dram_interleaved = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
    buffer_type=ttnn.BufferType.DRAM,
)
```

### DRAM with Non-Default Layout (WIDTH_SHARDED)

Sharded DRAM configs require a `ShardSpec`. The example below shows the structure; Chapter 2 covers how to compute valid shard shapes for a given tensor:

```python
import ttnn

# Define the core grid that will hold the shards.
# For DRAM-sharded, this references DRAM bank groups, not Tensix L1 cores.
core_range = ttnn.CoreRange(
    ttnn.CoreCoord(0, 0),
    ttnn.CoreCoord(0, 7),  # 1 column, 8 rows → 8 shards
)
core_range_set = ttnn.CoreRangeSet({core_range})

# Each shard covers the full height of the tensor; width is split 8 ways.
# For a [4096, 14336] tensor in TILE_LAYOUT (32-element tiles):
#   - full height = 4096 elements = 128 tiles of 32
#   - width per shard = 14336 / 8 = 1792 elements = 56 tiles of 32
shard_spec = ttnn.ShardSpec(
    grid=core_range_set,
    shape=[4096, 1792],         # [shard_height, shard_width] in elements
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
)

dram_width_sharded = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    buffer_type=ttnn.BufferType.DRAM,
    shard_spec=shard_spec,
)
```

### Inspecting a Tensor's Memory Config

After allocating a tensor or calling `ttnn.to_memory_config`, you can inspect its placement:

```python
weight = ttnn.from_torch(
    expert_weights_torch,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# Check the config
cfg = weight.memory_config()
print(cfg)
# MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED, buffer_type=BufferType::DRAM, shard_spec=std::nullopt)

# Check whether it is sharded
print(cfg.is_sharded())
# False

# After converting to sharded:
weight_sharded = ttnn.to_memory_config(weight, dram_width_sharded)
print(weight_sharded.memory_config().is_sharded())
# True
print(weight_sharded.memory_config().shard_spec())
# ShardSpec(grid=..., shape=[4096, 1792], orientation=ShardOrientation::RowMajor)
```

> **Tip:** `tensor.memory_config()` is the authoritative source of truth for where a tensor lives at any point in a program. Use it liberally when debugging unexpected op failures caused by mismatched input configs.

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Passing `shard_spec` with `INTERLEAVED` layout | Runtime error on `MemoryConfig` construction | Set `shard_spec=None` for interleaved |
| Omitting `shard_spec` with `HEIGHT_SHARDED` | Runtime assertion in allocator | Always provide `shard_spec` for sharded layouts |
| Shard shape not aligned to tile size (32 elements) | Silent shape mismatch or kernel assertion | Ensure both dimensions of `shard_spec.shape` are multiples of 32 |
| Using L1 config for a tensor that does not fit in available L1 | Out-of-memory error at allocation time | Move large tensors to DRAM; use L1 only after confirming per-core budget |

---

**Next:** [`interleaved_vs_sharded.md`](./interleaved_vs_sharded.md)
