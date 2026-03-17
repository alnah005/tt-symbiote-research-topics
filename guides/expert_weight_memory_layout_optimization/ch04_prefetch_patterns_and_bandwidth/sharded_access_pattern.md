# Sharded Access Pattern

## DRAM-Sharded Layout

`TensorMemoryLayout.WIDTH_SHARDED` (and `HEIGHT_SHARDED`) with `BufferType.DRAM` produces a DRAM-sharded tensor. Instead of distributing tiles round-robin across all 12 GDDR6 banks, TTNN assigns a **contiguous shard** — a rectangular sub-tensor — to a specific core's slice of the DRAM address space. The shard boundaries align with the `ShardSpec` defined at tensor creation time.

The key property: each shard lives within one DRAM bank range. A Tensix core assigned to process shard N always reads from the same DRAM bank. The NoC path from that core to that bank is fixed, short, and predictable.

For a `WIDTH_SHARDED` weight tensor of shape `[M, K]` distributed over `S` shards:
- Shard shape: `[M, K // S]` (columns split evenly; `K // S` must be divisible by 32 for TILE_LAYOUT).
- Core N reads columns `[N * K//S, (N+1) * K//S)` exclusively.
- No two cores ever request data from the same DRAM bank simultaneously due to weight access.

---

## Locality Benefit

With DRAM-sharded layout, each Tensix core's DRAM read path is deterministic:

```
Core N  ──►  DRAM bank assigned to shard N  ──  fixed hop count
```

The NoC path length is set once at tensor allocation and never changes across kernel invocations. The minimum hop count from core N to its assigned DRAM column is determined by the shard-to-bank assignment produced by `ttnn.create_sharded_memory_config`. On an 8-column Tensix grid mapped to 6 DRAM columns, most cores are within 1–2 hops of their assigned bank.

Because the path is fixed:
1. No NoC link carries requests from multiple cores to the same bank.
2. The DMA engine can issue read requests for the next tile in the shard before the current tile's compute completes.
3. Latency variance is bounded; no stall spikes from NoC contention.

---

## ShardOrientation.ROW_MAJOR

```python
ttnn.ShardOrientation.ROW_MAJOR
```

With `ROW_MAJOR` orientation, shards are assigned to cores by scanning the `CoreRangeSet` in row-major order: core (0,0) gets shard 0, core (1,0) gets shard 1, ..., core (7,0) gets shard 7, then core (0,1) gets shard 8, and so on.

For a weight matrix where the K-dimension corresponds to columns, `ROW_MAJOR` assignment means adjacent cores in the same row hold adjacent column slices of the weight matrix. During a matmul where the activation is broadcast across the row, each core's weight shard is spatially adjacent to its neighbors. The memory access pattern matches the order in which tiles are consumed, making spatial prefetching by the DMA engine effective: after fetching tile column `j`, the engine prefetches column `j+1` from the same DRAM bank with no address recalculation.

`ROW_MAJOR` is the default and preferred orientation when the matmul reduction dimension (K) is distributed across columns.

---

## ShardOrientation.COL_MAJOR

```python
ttnn.ShardOrientation.COL_MAJOR
```

With `COL_MAJOR` orientation, shards are assigned to cores by scanning the `CoreRangeSet` in column-major order: core (0,0) gets shard 0, core (0,1) gets shard 1, ..., core (0,9) gets shard 9, then core (1,0) gets shard 10, etc.

This orientation is useful when the matmul K-dimension is partitioned across the column axis of the core grid. In that configuration, all cores in a given column share the same K-tile boundary. `COL_MAJOR` shard assignment ensures the shard boundary aligns with the K-tile boundary: core (r, c) always holds the weight rows corresponding to K-partition c, regardless of row. The result is that reduce operations across the K-dimension collect partial sums from cores that hold contiguous K-partitions, minimizing the all-reduce communication pattern.

---

## Double-Buffering Interaction

Double-buffering allocates two L1 buffers per core: buffer A and buffer B. While the compute engine processes the shard currently in buffer A, the DMA engine prefetches the next shard from DRAM into buffer B. When the compute engine finishes, it swaps A and B. This fully overlaps DRAM transfer latency with computation, provided:

1. The shard fits within half the available L1. For a 1.5 MB L1, each double-buffer slot is ≤ 768 KB.
2. The compute time for one shard exceeds the DRAM transfer time for the next shard. At ~300 GB/s, transferring 768 KB takes ~2.62 µs; the compute must take at least that long to achieve full overlap.

With DRAM-sharded layout, the DMA engine knows the exact DRAM address and size of the next shard before the current shard's compute begins. The prefetch request can be issued immediately at the start of compute, maximizing the overlap window. With interleaved layout, the address of the next tile depends on the tile index modulo number of banks — still computable, but adds address-calculation overhead and does not benefit from spatial prefetching.

---

## API Reference

```python
# WIDTH_SHARDED, ROW_MAJOR, 8 cores across a single row
shard_spec = ttnn.ShardSpec(
    core_range_set=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
    shard_shape=[weight_rows, weight_cols // 8],  # shard_shape[1] % 32 == 0 required
    shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
)
mem_config = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    buffer_type=ttnn.BufferType.DRAM,
    shard_spec=shard_spec,
)
weight_sharded = ttnn.to_memory_config(weight_tensor, mem_config)
```

---

**Next:** [`bandwidth_estimation.md`](./bandwidth_estimation.md)
