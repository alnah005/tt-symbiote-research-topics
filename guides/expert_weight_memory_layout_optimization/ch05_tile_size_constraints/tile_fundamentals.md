# Tile Fundamentals

## The 32×32 Tile as the Atomic Compute Unit

On Wormhole B0, the matrix engine (FPU) operates on 32×32 element tiles. A tile is the smallest unit that the compute engine can issue as a single instruction. This has two consequences that are impossible to work around:

1. The compute engine cannot address a sub-tile region. A kernel that logically needs a 16×32 slice must still load a full 32×32 tile.
2. All TTNN kernels that operate in TILE_LAYOUT assume tile boundaries are correctly aligned. Misaligned shard boundaries mean a tile straddles two shards, which is either rejected at config time or silently produces incorrect reads.

Both height and width must independently be multiples of 32. A tensor with shape `[4096, 14336]` is valid because `4096 % 32 == 0` and `14336 % 32 == 0`. A tensor with shape `[4000, 14336]` would require zero-padding to `[4096, 14336]` before any TILE_LAYOUT operation can proceed.

---

## Automatic Zero-Padding

TTNN automatically zero-pads tensors to the next 32-multiple when you call `ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)` on a tensor whose dimensions are not already 32-aligned. The padding is appended in the trailing dimension positions:

```python
import ttnn

# A tensor with a non-aligned height
cpu_tensor = torch.randn(4000, 14336, dtype=torch.bfloat16)
tt_tensor = ttnn.from_torch(cpu_tensor, dtype=ttnn.bfloat16)
tt_tiled = ttnn.to_layout(tt_tensor, ttnn.TILE_LAYOUT)

# Effective shape after padding:
print(tt_tiled.shape)  # [4032, 14336]  (4000 padded to next 32-multiple: 4032)
```

The padding elements are zero and do not affect the numerical output of matmul (zero rows in the weight matrix contribute zero to the output). However, the zero-padded shape is the shape that all shard arithmetic must use. If the logical tensor height is 4000 but the effective tiled height is 4032, shard height derivations must use 4032, not 4000.

> **Warning:** Deriving shard shapes from the pre-padding logical shape is a frequent source of alignment errors. Always inspect `tensor.shape` after calling `ttnn.to_layout(..., ttnn.TILE_LAYOUT)` and use that padded shape in all subsequent `ShardSpec` calculations.

---

## Tile Memory Footprint by Dtype

Quick reference — bytes per 32×32 tile by dtype (Wormhole B0):

| Dtype | Bytes per element | Bytes per tile (32×32) |
|---|---|---|
| `bfloat16` | 2 | 2,048 |
| `bfloat8_b` | 1 | 1,024 |
| `bfloat4_b` | 0.5 | 512 |

See Chapter 3, `dtype_and_tile_layout.md` for the full derivation.

For a shard of shape `[shard_H, shard_W]` in dtype `D`, the shard occupies:

```
shard_bytes = shard_H * shard_W * bytes_per_element(D)
```

Because tile alignment guarantees `shard_H` and `shard_W` are each multiples of 32, the smallest valid shard (32×32, `bfloat4_b`) is 512 bytes — a multiple of the 32-byte DRAM page size. This means the page-alignment constraint (Rule 5 in `shard_shape_alignment_rules.md`) is automatically satisfied for `bfloat16` and `bfloat8_b`. For `bfloat4_b`, verify separately because half-byte packing can create edge cases with odd tile counts.

---

## Tile-Count Notation

Program configs and kernel documentation frequently express dimensions in tile counts rather than element counts. The two notations are:

```
M_t = height / 32      # number of tile rows
N_t = width  / 32      # number of tile columns
K_t = inner_dim / 32   # number of tile columns in the K (contraction) dimension
```

For Mixtral 8x7B gate projection `[4096, 14336]`:

```
M_t = 4096  / 32 = 128   tile rows
N_t = 14336 / 32 = 448   tile columns
```

`ShardSpec.shape` always takes element counts, not tile counts. If a kernel config parameter is named `in0_block_w` and its documentation says "in tiles", its value of `8` means 8 tiles = 256 elements. See `shard_shape_alignment_rules.md`, Pitfall 3 in `common_pitfalls.md`, for how confusing these notations causes suboptimal shard width selection.

---

## Page Alignment for Shard Sizes

Wormhole B0 DRAM uses a 32-byte page granularity. A read request that does not start or end on a 32-byte boundary causes the memory controller to fetch additional bytes to fill the page, wasting bandwidth.

For DRAM-sharded configurations, each shard should start on a 32-byte boundary and have a byte length that is a multiple of 32. The alignment check is:

```python
def is_page_aligned(shard_H: int, shard_W: int, bytes_per_element: float) -> bool:
    shard_bytes = int(shard_H * shard_W * bytes_per_element)
    return shard_bytes % 32 == 0
```

For any `bfloat16` shard where `shard_H` and `shard_W` are multiples of 32:

```
shard_bytes = (32k) * (32j) * 2 = 2048 * k * j
2048 % 32 == 0  =>  always page-aligned
```

The same arithmetic holds for `bfloat8_b` (1024-byte tiles) and `bfloat4_b` (512-byte tiles). In practice, tile alignment guarantees page alignment for all standard expert weight dtypes on Wormhole B0.

---

## Next Steps

With the tile unit, zero-padding behavior, memory footprints, and tile-count notation established, proceed to `shard_shape_alignment_rules.md` for the five concrete rules that translate these fundamentals into valid `ShardSpec.shape` values.
