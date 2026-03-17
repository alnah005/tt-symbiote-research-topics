# Mapping Weight Tensor Shapes to Valid Shard Grids

## Overview

Selecting a shard grid for an expert weight tensor requires satisfying two kinds of constraints simultaneously: integer divisibility of the sharded dimension by the number of cores, and tile-alignment of the resulting shard shape. This file presents the rules, the projection-specific strategies, the impact of large `num_experts`, and a worked example.

---

## Core Divisibility Constraint

A shard configuration is valid only when:

1. The sharded dimension divides evenly across `num_cores` (no remainder).
2. The resulting shard dimension is a multiple of 32. For a full explanation of why the tile side length is 32 and how TILE_LAYOUT enforces this constraint, see [`dtype_and_tile_layout.md`](./dtype_and_tile_layout.md).

Formally, for a tensor of shape `[M, N]`:

| Sharding Mode | Shard Shape | Constraint 1 | Constraint 2 |
|---|---|---|---|
| WIDTH_SHARDED | `[M, N // num_cores]` | `N % num_cores == 0` | `(N // num_cores) % 32 == 0` |
| HEIGHT_SHARDED | `[M // num_cores, N]` | `M % num_cores == 0` | `(M // num_cores) % 32 == 0` |
| BLOCK_SHARDED | `[M // grid_rows, N // grid_cols]` | both row and col divisibility | both dimensions % 32 == 0 |

> **Warning:** Constraint 2 (tile alignment) is an implicit requirement of TILE_LAYOUT and is not always surfaced as an explicit error at tensor-creation time. A shard width or height that is not a multiple of 32 will produce incorrect matmul results or a runtime assertion failure. Always verify both constraints before constructing a ShardSpec.

---

## Grid Size Selection Guidelines

Not all core counts divide cleanly into typical MoE weight dimensions. Prefer core counts that have hardware-level alignment benefits:

| num_cores | Alignment rationale |
|---|---|
| 6 | Matches Wormhole's 6 DRAM (Dynamic Random Access Memory) controllers; maximizes DRAM bandwidth utilization |
| 12 | Matches Wormhole's 12 DRAM banks (2 banks per controller); useful for larger shards |
| 8 | Matches the 8-column Tensix grid width on Wormhole; avoids partial-row grid shapes |
| 16 | 2×8 grid; balanced for tensors whose sharded dimension is divisible by 512 (16 × 32) |
| 32 | 4×8 grid; maximum fill for a single Wormhole chip |

> **Tip:** When multiple core counts satisfy the divisibility constraints, prefer the count that aligns with DRAM controller boundaries (6 or 12) for weight tensors that are read but not written during inference. The bandwidth gain from aligned DRAM access often outweighs the benefit of distributing compute across more cores.

---

## Projection-Specific Sharding Strategies

### Gate and Up Projections: `[d_model, d_ff]`

The preferred sharding mode for gate and up weight projections is **WIDTH_SHARDED along the d_ff dimension**.

Rationale:
- Each core receives all `d_model` input rows, so the full token embedding is available locally.
- Output features (`d_ff`) are split across cores; each core computes its partial output independently.
- No inter-core reduction is needed for the matmul itself (reduction happens in the activation/routing step).

```python
# Gate projection width sharding
# weight shape: [d_model, d_ff]
# Shard along d_ff (width)

num_cores = 8  # chosen to divide d_ff
shard_width = d_ff // num_cores       # must be % 32 == 0
shard_shape = [d_model, shard_width]

mem_config = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    buffer_type=ttnn.BufferType.L1,
    shard_spec=ttnn.ShardSpec(
        core_range_set,                # CoreRangeSet covering num_cores
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    ),
)
```

### Down Projection: `[d_ff, d_model]`

Two valid strategies exist; choice depends on the matmul access pattern:

**Option A — WIDTH_SHARDED along d_model:**
- Mirrors the gate/up strategy but applied to the transposed shape.
- Each core holds `[d_ff, d_model // num_cores]`.
- Works well when the preceding hidden-state tensor is replicated across cores.

**Option B — HEIGHT_SHARDED along d_ff:**
- Each core holds `[d_ff // num_cores, d_model]`.
- Works well when the hidden-state tensor is also height-sharded with a matching core layout.
- Requires `d_ff % num_cores == 0` and `(d_ff // num_cores) % 32 == 0`.

| Strategy | Shard Shape | Best When |
|---|---|---|
| WIDTH_SHARDED on d_model | `[d_ff, d_model // num_cores]` | Hidden state is replicated |
| HEIGHT_SHARDED on d_ff | `[d_ff // num_cores, d_model]` | Hidden state is height-sharded |

> **Warning:** Mismatching the sharding of the down projection weight with the sharding of the hidden-state activation tensor forces TTNN to insert an implicit tensor reshard or copy. This can dominate runtime on large models. Always align the weight and activation shard strategies for the same matmul.

---

## num_experts Interaction with Sharding

### Small num_experts (≤ 8)

When `num_experts` is small (e.g., Mixtral 8x7B with 8 experts), it is feasible to assign one expert per core or to shard the stacked `[num_experts, d_model, d_ff]` tensor across the expert axis.

```python
# Per-expert axis sharding: each core holds one expert's full weight matrix
# Stacked tensor shape: [8, 4096, 14336]
# 8 experts -> 8 cores, one expert per core
# Shard shape: [1, 4096, 14336] (viewed as [4096, 14336] per core)
```

Constraints: the per-expert shard must itself be tile-aligned (`d_model % 32 == 0`, `d_ff % 32 == 0`).

### Large num_experts (64–128)

For models like DeepSeek-MoE-16B (64 experts) or Qwen MoE (128 experts), assigning one expert per core is generally impractical:

- A single Wormhole chip has at most ~72 usable Tensix cores.
- 128 experts would require more cores than are available on one device.
- Expert-parallel distribution across devices handles this at the system level.

**Recommended approach for large num_experts:** shard within each individual expert weight matrix rather than across the expert axis.

```python
# Qwen MoE example: 128 experts, d_model=7168, d_ff=2048
# Each expert's gate/up: [7168, 2048]
# Shard within the matrix using WIDTH_SHARDED, 8 cores
shard_width = 2048 // 8   # = 256, 256 % 32 == 0 -> valid
shard_shape = [7168, 256]
# Dispatch one expert at a time to the 8-core grid
```

---

## Worked Example: Mixtral 8x7B Gate Projection

**Input parameters:**
- Projection: gate (w1)
- Shape: `[4096, 14336]`
- Target: WIDTH_SHARDED, 8 cores

**Step 1: Verify divisibility.**

```python
d_model = 4096
d_ff    = 14336
num_cores = 8

assert d_ff % num_cores == 0, "d_ff must be divisible by num_cores"
shard_width = d_ff // num_cores   # 14336 // 8 = 1792
```

**Step 2: Verify tile alignment.**

```python
assert shard_width % 32 == 0, "shard width must be tile-aligned (multiple of 32)"
# 1792 % 32 = 0 -> valid
```

**Step 3: Construct the shard spec.**

```python
import ttnn

# Valid grid options for 8 cores on Wormhole:
#   1x8 row:   CoreRange((0,0), (7,0))
#   8x1 column: CoreRange((0,0), (0,7))
core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))  # 1x8 row
core_range_set = ttnn.CoreRangeSet({core_range})

shard_shape = [d_model, shard_width]   # [4096, 1792]

mem_config = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    buffer_type=ttnn.BufferType.L1,
    shard_spec=ttnn.ShardSpec(
        core_range_set,
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    ),
)
```

**Step 4: Summary table.**

| Parameter | Value |
|---|---|
| Tensor shape | `[4096, 14336]` |
| Sharding mode | WIDTH_SHARDED |
| num_cores | 8 |
| Shard shape | `[4096, 1792]` |
| d_ff divisibility check | `14336 % 8 == 0` ✓ |
| Tile alignment check | `1792 % 32 == 0` ✓ |
| Grid option (1×8 row) | `CoreRange((0,0), (7,0))` |
| Grid option (8×1 column) | `CoreRange((0,0), (0,7))` |

---

## Validity Checklist

Use this checklist when deriving a shard grid for any expert weight tensor:

- [ ] Identify the sharding mode (WIDTH, HEIGHT, or BLOCK).
- [ ] Compute `shard_dim // num_cores`; confirm `shard_dim % num_cores == 0`.
- [ ] Confirm the resulting shard dimension is a multiple of 32.
- [ ] Confirm the non-sharded dimension is also a multiple of 32 (required for TILE_LAYOUT).
- [ ] Select a grid shape whose total core count equals `num_cores`.
- [ ] Confirm the activation tensor for the same matmul uses a compatible shard layout.

---

## Next Steps

Continue to [`dtype_and_tile_layout.md`](./dtype_and_tile_layout.md) for dtype size tables, the tile layout requirement for `ttnn.matmul`, padding implications, and byte-accurate per-expert memory footprint calculations.
