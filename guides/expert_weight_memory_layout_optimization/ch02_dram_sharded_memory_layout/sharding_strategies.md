# Sharding Strategies

TTNN provides three sharding strategies for distributing a tensor across a set of cores or DRAM banks. The right choice depends on which tensor dimensions are large, how the downstream matmul consumes the data, and the shape of the target core grid.

---

## HEIGHT_SHARDED (1D along Height)

Each shard is a contiguous row-block that spans the full tensor width. Shard `i` holds rows `[i*shard_h : (i+1)*shard_h, :]`.

```
Tensor [H, W]:
  Shard 0: rows [0 : shard_h,    :]
  Shard 1: rows [shard_h : 2*sh, :]
  ...
```

The grid is 1D (a single row of `num_shards` cores, or equivalently a single column). Each DRAM bank stores an independent, full-width slice of the tensor.

**Natural fit for:** computation that iterates over independent row-groups — M-dimension parallelism, where each compute core processes a different subset of input tokens.

**Expert weight use case:** gate/up projection weights of shape `[d_model, d_ff]` sharded along `d_model` (the row dimension). Each shard holds the projection rows corresponding to a subset of the input token dimension, enabling token-parallel routing where different experts handle different row slices.

> **Tip:** HEIGHT_SHARDED is straightforward to reason about when the height dimension is the primary bottleneck — for example, when `d_model` is large and `d_ff` is moderate.

---

## WIDTH_SHARDED (1D along Width)

Each shard is a contiguous column-block that spans the full tensor height. Shard `i` holds columns `[:, i*shard_w : (i+1)*shard_w]`.

```
Tensor [H, W]:
  Shard 0: cols [:, 0 : shard_w]
  Shard 1: cols [:, shard_w : 2*sw]
  ...
```

The grid is 1D (a single row of `num_shards` cores). Each DRAM bank stores a vertical stripe of the tensor — identical height, reduced width.

**Natural fit for:** output-dimension parallelism (N-dimension sharding). The matmul consumes each stripe independently, allowing partial outputs to be computed in parallel before a final reduction.

**Expert weight use case:** gate/up projection weights of shape `[d_model, d_ff]` sharded along `d_ff` (the column dimension). Each bank holds a subset of output features, distributing the large `d_ff` dimension (e.g., 14336 for Mixtral-style experts) across 8 DRAM banks.

> **Tip:** WIDTH_SHARDED is the preferred default for expert gate/up projections in Wormhole B0. Wormhole B0 has 6 DRAM controllers (12 GDDR6 banks, 2 per controller). The choice of 8 shards aligns with the 8-column Tensix grid width and produces tile-aligned shard sizes (e.g., 14336/8=1792 elements, which is divisible by 32).

---

## BLOCK_SHARDED (2D)

Each shard is a rectangular sub-block. The grid must be 2D (`num_cores_y` rows by `num_cores_x` columns). The shard at grid position `(row_i, col_j)` holds:

```
rows [row_i*sh : (row_i+1)*sh,  col_j*sw : (col_j+1)*sw]
```

Both the height and width dimensions are partitioned simultaneously. This spreads data access across a larger number of banks with smaller per-bank footprint.

**Natural fit for:** tensors where both M and N dimensions are large enough to benefit from 2D distribution. The 2D grid increases the number of banks engaged and can improve aggregate DRAM bandwidth utilization.

**Expert weight use case:** down projection weights of shape `[d_ff, d_model]` — for example, `[14336, 4096]`. Both dimensions are substantial. A `4x2` grid (4 columns, 2 rows of cores) would yield `shard_height = 14336/2 = 7168` and `shard_width = 4096/4 = 1024` — both tile-aligned.

```python
# BLOCK_SHARDED example: [14336, 4096] across a 4x2 grid (4 cols, 2 rows)
grid = ttnn.CoreRangeSet({
    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 1))
})
shard_spec = ttnn.ShardSpec(
    grid=grid,
    shape=[7168, 1024],   # 14336/2=7168, 4096/4=1024; both divisible by 32 ✓
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
)
```

> **Warning:** BLOCK_SHARDED requires that the grid dimensions (`num_cores_y`, `num_cores_x`) divide the tensor's height and width respectively, producing integer tile-aligned shard dimensions. An irregular grid (e.g., 3 cores for a height of 14336) will fail or produce incorrect padding unless handled explicitly.

---

## DRAM-Sharded vs L1-Sharded: Key Distinction

Both DRAM-sharded and L1-sharded buffers use `ShardSpec` and the same three strategies, but the physical location of each shard differs fundamentally.

**DRAM-sharded (`BufferType.DRAM`):**
- Shards physically reside in DRAM, distributed across DRAM banks.
- The `CoreRangeSet` in `ShardSpec` refers to DRAM controller positions in the NoC grid, not Tensix L1 cores.
- Reduces NoC contention compared to interleaved DRAM because each DRAM bank serves a dedicated, non-overlapping set of shards. With interleaving, every access touches all controllers in round-robin; with sharding, each access is directed to a single controller.
- Used for large weight tensors that remain in DRAM throughout the model's lifetime.

**L1-sharded (`BufferType.L1`):**
- Each shard lives in the L1 SRAM of the corresponding Tensix compute core (1.5 MB per core on Wormhole B0).
- The `CoreRangeSet` refers to actual Tensix core coordinates.
- Used for activations during compute — the canonical pattern is to move DRAM-sharded weights into L1-sharded activations immediately before a matmul, then deallocate.

---

## Strategy Comparison

| Strategy | Typical expert weight use case | NoC access pattern |
|---|---|---|
| HEIGHT_SHARDED | Gate/up projections, token-parallel routing | Each shard group reads from one DRAM region |
| WIDTH_SHARDED | Gate/up projections, feature-parallel output | Column stripes each in a separate bank |
| BLOCK_SHARDED | Down projection (large d_ff x d_model) | 2D tiles distributed across a grid of banks |

---

## Next Steps

Proceed to [constructing_dram_sharded_config.md](./constructing_dram_sharded_config.md) for a step-by-step guide to building and verifying a complete DRAM-sharded `MemoryConfig`, including common mistakes and an end-to-end code example.
