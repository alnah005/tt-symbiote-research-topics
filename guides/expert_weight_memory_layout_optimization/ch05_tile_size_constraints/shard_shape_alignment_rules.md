# Shard Shape Alignment Rules

This file enumerates the five rules that every valid `ShardSpec` for a TILE_LAYOUT expert weight tensor must satisfy. Each rule is stated precisely, its consequence for violation is described, and a code snippet shows how to check the condition programmatically.

---

## Rule 1: Shard Height Must Be a Multiple of 32

For a tensor in TILE_LAYOUT, the shard height (the first element of `ShardSpec.shape`) must satisfy:

```
shard_H % 32 == 0
```

**Why it exists.** The TTNN shard mapping allocates complete tiles to each shard. A shard height of, say, 48 would place 1.5 tile rows in the shard — the second tile row is split between this shard and the next. The kernel has no mechanism to handle split tiles; the first shard either reads the partial tile incorrectly or the config raises an error at `ttnn.to_memory_config`.

**Consequence of violation.** Depending on the TTNN version and the specific op, you will see one of:
- `RuntimeError: Shard shape must be tile-aligned` at config construction time.
- Silent data corruption where the last partial tile row of one shard is read as if it were the first row of the next shard.

**Check:**
```python
def check_rule1(shard_H: int) -> bool:
    return shard_H % 32 == 0
```

---

## Rule 2: Shard Width Must Be a Multiple of 32

The shard width (the second element of `ShardSpec.shape`) must satisfy:

```
shard_W % 32 == 0
```

**Why it exists.** Identical reasoning to Rule 1, applied to the width dimension. A tile column is 32 elements wide; a shard boundary in the middle of a tile column is not addressable.

**Consequence of violation.** Same as Rule 1: error at config time or silent misalignment.

**Check:**
```python
def check_rule2(shard_W: int) -> bool:
    return shard_W % 32 == 0
```

> **Tip:** When choosing a shard width, prefer multiples of 32 that are also multiples of the downstream matmul's `in0_block_w` (expressed in elements = `in0_block_w_tiles * 32`). This is not required for correctness but improves tiling efficiency. See `common_pitfalls.md`, Pitfall 3.

---

## Rule 3: BLOCK_SHARDED — Both Dimensions Must Independently Satisfy Rules 1 and 2

For `TensorMemoryLayout.BLOCK_SHARDED`, the tensor is partitioned in both height and width simultaneously. The shard has shape `[shard_H, shard_W]` and is assigned to a 2D core grid of size `(num_cores_H, num_cores_W)`. Both dimensions are independently subject to the 32-multiple requirement:

```
shard_H % 32 == 0   (Rule 1, applied to the height sub-partition)
shard_W % 32 == 0   (Rule 2, applied to the width sub-partition)
```

**Why it must be checked independently.** It is possible for one dimension to be tile-aligned while the other is not, particularly when deriving shard dimensions from a non-square core grid. Both must be checked explicitly.

**Example.** A `[4096, 14336]` tensor on a 4×4 core grid (BLOCK_SHARDED):

```python
tensor_H, tensor_W = 4096, 14336
num_cores_H, num_cores_W = 4, 4

shard_H = tensor_H // num_cores_H  # 1024 — 1024 % 32 == 0  OK
shard_W = tensor_W // num_cores_W  # 3584 — 3584 % 32 == 0  OK

# Both pass. A different grid:
num_cores_H2, num_cores_W2 = 3, 4
shard_H2 = tensor_H // num_cores_H2  # 1365 — 1365 % 32 != 0  FAIL
```

---

## Rule 4: Tensor Dimension / Shard Dimension Must Equal the Core Count in That Direction

For `HEIGHT_SHARDED`:

```
tensor_H / shard_H == num_cores   (total core count in the shard grid)
tensor_H % shard_H == 0
```

For `WIDTH_SHARDED`:

```
tensor_W / shard_W == num_cores   (total core count in the shard grid)
tensor_W % shard_W == 0
```

For `BLOCK_SHARDED` (applied in each dimension independently):

```
tensor_H / shard_H == num_cores_H
tensor_W / shard_W == num_cores_W
```

**Why it exists.** The shard grid must tile the tensor exactly. If the division is not exact, some cores are assigned to shards that extend beyond the tensor boundary. TTNN does not implicitly extend or truncate: such a config produces an error or reads garbage from memory beyond the tensor allocation.

**Consequence of violation.** `RuntimeError` at `ttnn.to_memory_config`, or (in older TTNN versions) an OOB read from the buffer backing the tensor.

**Check:**
```python
def check_rule4_width_sharded(tensor_W: int, shard_W: int, num_cores: int) -> bool:
    return tensor_W % shard_W == 0 and tensor_W // shard_W == num_cores
```

---

## Rule 5: Shard Size in Bytes Must Be Page-Aligned (32 Bytes on Wormhole)

```
(shard_H * shard_W * bytes_per_element) % 32 == 0
```

**Why it exists.** The Wormhole B0 DRAM controller operates at 32-byte page granularity. A shard whose byte length is not a multiple of 32 forces partial-page reads on the final page of the shard, consuming extra bus cycles and reducing effective bandwidth.

**Consequence of violation.** Not a hard error, but a measurable bandwidth penalty. See Chapter 4, `bandwidth_estimation.md`, for the bandwidth model.

**Practical note.** As shown in `tile_fundamentals.md`, any shard where `shard_H` and `shard_W` are both multiples of 32 automatically satisfies this rule for `bfloat16` (2048-byte tiles) and `bfloat8_b` (1024-byte tiles). Verify explicitly only when working with `bfloat4_b` or non-standard dtypes.

**Check:**
```python
def check_rule5(shard_H: int, shard_W: int, bytes_per_element: float) -> bool:
    shard_bytes = int(shard_H * shard_W * bytes_per_element)
    return shard_bytes % 32 == 0
```

---

## Worked Example: Mixtral 8x7B Gate Projection, WIDTH_SHARDED Across 8 DRAM Banks

This example demonstrates that a `[4096, 14336]` tensor sharded WIDTH_SHARDED across 8 cores satisfies all five alignment rules simultaneously: both tensor dimensions are already 32-aligned (no padding needed), the derived shard width of 1792 is tile-aligned, and the shard byte size is a multiple of the 32-byte page size. The final result is:

- Shard shape: `[4096, 1792]` — `1792 % 32 = 0` ✓

See Chapter 3, `tensor_to_shard_grid_mapping.md` for the full derivation.

---

## Tile-Count Summary

For tile-count values by model (M_t, N_t), see `tile_fundamentals.md` in this chapter.

---

## Next Steps

With the five alignment rules defined, proceed to `common_pitfalls.md` for the five failure modes that arise in practice when these rules are not applied correctly or completely.
