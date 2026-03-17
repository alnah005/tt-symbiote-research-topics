# ShardSpec Deep Dive

`ttnn.ShardSpec` is the struct that describes how a tensor is divided across cores or DRAM banks. Every sharded `MemoryConfig` must include a `ShardSpec`; TTNN raises an error if it is missing or if it is supplied for an INTERLEAVED config.

---

## The Three Fields

### `grid: ttnn.CoreRangeSet`

The set of cores that hold shards. For DRAM-sharded tensors, these "cores" refer to DRAM controller positions in the Wormhole NoC grid, not Tensix compute cores. The number of elements in the set equals the number of shards the tensor is divided into.

### `shape: List[int]`

The shard shape as `[shard_height, shard_width]` in elements — not tiles. For tile-layout tensors both dimensions must be multiples of 32.

> **Warning:** TTNN raises an error at buffer allocation time if either shard dimension is not a multiple of 32 for a tile-layout tensor. Verify divisibility before constructing the spec.

### `orientation: ttnn.ShardOrientation`

Controls which core receives which portion of the tensor:

- `ShardOrientation.ROW_MAJOR`: shards are assigned to cores left-to-right across each row, then advancing to the next row.
- `ShardOrientation.COL_MAJOR`: shards are assigned top-to-bottom down each column, then advancing to the next column.

For DRAM-sharded expert weights, `ROW_MAJOR` is the conventional default.

---

## Building the Grid: CoreCoord, CoreRange, CoreRangeSet

### `ttnn.CoreCoord(x, y)`

Identifies a single core in the Wormhole NoC grid. `x` is the column index; `y` is the row index.

### `ttnn.CoreRange(start: CoreCoord, end: CoreCoord)`

A rectangular range of cores, inclusive on both ends. The total core count is `(end.x - start.x + 1) * (end.y - start.y + 1)`.

```python
# 8 cores in a single row: columns 0-7, row 0
row_of_8 = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))

# 4x2 grid: columns 0-3, rows 0-1 (8 cores)
block_4x2 = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 1))
```

### `ttnn.CoreRangeSet`

A set of `CoreRange` objects. It can express non-contiguous grids by combining multiple ranges. The total shard count equals the total number of cores across all ranges.

```python
# Single rectangular range: 1 row of 8 cores (columns 0-7, row 0)
grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})

# Two ranges: first 4 and last 4 columns of row 0
grid = ttnn.CoreRangeSet({
    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),
    ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 0)),
})
```

The two-range example above produces the same 8-core set as the first example; the distinction matters when targeting non-contiguous DRAM banks or when hardware topology makes a single rectangle infeasible.

---

## Shard Shape Arithmetic

The relationship between `shard_shape` and the full tensor shape depends on the sharding strategy.

| Strategy | Constraint |
|---|---|
| HEIGHT_SHARDED | `shard_shape[0] * num_shards = tensor_height`; `shard_shape[1] = tensor_width` |
| WIDTH_SHARDED | `shard_shape[1] * num_shards = tensor_width`; `shard_shape[0] = tensor_height` |
| BLOCK_SHARDED | `shard_shape[0] * num_cores_y = tensor_height` AND `shard_shape[1] * num_cores_x = tensor_width` |

`num_shards` (for 1D strategies) equals the total number of cores in the `CoreRangeSet`. For BLOCK_SHARDED, `num_cores_y` and `num_cores_x` are derived from the grid dimensions.

---

## Worked Example: WIDTH_SHARDED across `num_cores` Cores

**Tensor:** shape `[M, N]`, tile layout.
**Goal:** distribute the `N`-column dimension across `num_cores` DRAM banks.

```
shard_width  = N / num_cores   # must be an integer and divisible by 32
shard_height = M               # full height per shard (WIDTH_SHARDED does not split height)
```

```python
import ttnn

shard_spec = ttnn.ShardSpec(
    grid=ttnn.CoreRangeSet({
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))
    }),
    shape=[M, N // num_cores],
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
)
```

> **Tip:** Always work out shard_width and shard_height on paper before constructing `ShardSpec`. Choose `num_cores` to be a divisor of both the target dimension and 32 simultaneously — pick from the set of common divisors of the tensor dimension and 32.

For a concrete expert weight worked example, see `constructing_dram_sharded_config.md`.

---

**Next:** [`sharding_strategies.md`](./sharding_strategies.md)
