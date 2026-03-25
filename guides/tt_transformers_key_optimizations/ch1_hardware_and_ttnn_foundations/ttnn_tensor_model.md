# TTNN Tensor Model

This file explains how TTNN represents tensors on Tenstorrent hardware — the layout formats, memory locations, sharding strategies, and data types that every tt-transformers operation depends on. If [tensix_architecture.md](./tensix_architecture.md) described the physical substrate, this file describes the programming model layered on top of it.

---

## TilizedLayout vs RowMajorLayout

TTNN supports two tensor storage layouts:

### RowMajorLayout

Elements are stored in row-major (C-contiguous) order, exactly as in a standard NumPy or PyTorch tensor. This is the layout for tensors that live on the host or are passed through non-compute paths. TTNN ops that accept RowMajorLayout inputs typically convert them to TilizedLayout internally before dispatching to the compute hardware.

### TilizedLayout

Elements are stored as a sequence of **32×32 tiles**. Within each tile, elements are organized as four 16×16 "faces" (matching the matrix engine's native input primitive described in [tensix_architecture.md](./tensix_architecture.md)). The tile order follows the row-major order of tiles across the tensor dimensions, but the intra-tile layout is always face-major.

**Why 32×32 tiles are the native computation unit:**

- The matrix FPU takes 32×32 tile inputs — a 32-row × 32-column block of values from SrcA and SrcB
- Storing tensors in tile order means the Unpacker can load a complete tile into SrcA/SrcB in one contiguous DMA operation — no strided access, no gather
- Weight tensors for LLM linear layers (e.g., 4096×4096) stored in TilizedLayout have all elements of tile `(i, j)` contiguous in memory, making NoC reads maximally efficient

**Face layout within a tile:**

A 32×32 tile is internally four 16×16 faces:

```
┌───────────┬───────────┐
│ Face 0    │ Face 1    │   rows  0–15
│ (rows0-15,│ (rows0-15,│
│  cols0-15)│  cols16-31│
├───────────┼───────────┤
│ Face 2    │ Face 3    │   rows 16–31
│ (rows16-31│ (rows16-31│
│  cols0-15)│  cols16-31│
└───────────┴───────────┘
```

In memory, Face 0 is stored first (all 256 elements), then Face 1, then Face 2, then Face 3. This matches the order in which the matrix FPU consumes them for its 16×16 × 16×16 sequential face multiplications.

### Layout Conversion

```python
import ttnn

# Host tensor in row-major
x_host = torch.randn(1, 1, 4096, 4096)

# Place on device in TilizedLayout (default for most ops)
x_device = ttnn.from_torch(
    x_host,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,   # TilizedLayout
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# Row-major is also available (used for certain element-wise ops or host-side processing)
x_row_major = ttnn.from_torch(
    x_host,
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
)
```

---

## BufferType: DRAM vs L1

A TTNN tensor's **BufferType** determines where in the memory hierarchy its data is allocated:

### `ttnn.BufferType.DRAM`

- Data is stored in off-chip DRAM
- Interleaved across DRAM banks by default (each tile is distributed round-robin across banks for balanced bandwidth)
- Accessible from any Tensix core via NoC
- **Default for most tensors**: model weights, KV caches, and any tensor that is too large for L1

### `ttnn.BufferType.L1`

- Data is stored in the L1 SRAM of specific Tensix cores
- Used in combination with sharding: a sharded tensor in L1 means each core holds its assigned portion of the tensor locally
- Accessible at single-cycle latency from that core's Unpacker — no NoC traffic needed to read the shard
- **Used for activations during compute**: the optimization goal described in [tensix_architecture.md](./tensix_architecture.md)

The buffer type is specified through a `MemoryConfig` object:

```python
# DRAM interleaved (default)
dram_config = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
    buffer_type=ttnn.BufferType.DRAM,
)

# L1 interleaved (less common; same structure as DRAM but uses L1 pool)
l1_interleaved_config = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
    buffer_type=ttnn.BufferType.L1,
)
```

---

## TensorMemoryLayout: Interleaved vs Sharded

The **TensorMemoryLayout** controls how a tensor's tiles are distributed across cores. This is the key axis of variation for performance optimization.

### `INTERLEAVED`

Tiles are distributed round-robin across all DRAM banks (or L1 pools if buffer type is L1). No core has exclusive ownership of any subset of tiles. Any core can read any tile, but each read costs a NoC transaction.

- Default for all tensors that are not explicitly sharded
- Flexible: works with any core count and tensor shape
- Higher NoC traffic under heavy access patterns

### `HEIGHT_SHARDED`

The tensor's height dimension (rows) is divided equally among cores. Each core owns a contiguous slice of rows (its "shard"), stored in that core's L1. The width dimension is not split — each core holds all columns for its row slice.

Example: a tensor `[1 × 1 × 512 × 4096]` height-sharded across 32 cores → each core holds `[1 × 1 × 16 × 4096]`.

Used when:
- The operation processes each row independently (e.g., RMSNorm, element-wise ops)
- Activations flow from a previous height-sharded matmul output

### `WIDTH_SHARDED`

The tensor's width dimension (columns) is divided among cores. Each core holds all rows but only a slice of the columns.

Example: a tensor `[1 × 1 × 32 × 4096]` width-sharded across 32 cores → each core holds `[1 × 1 × 32 × 128]`.

Used when:
- Column-parallel matmuls: each core computes a subset of output features
- Attention head grouping: each core group handles a subset of heads

### `BLOCK_SHARDED`

Both height and width dimensions are divided across a 2D core grid. Each core holds a rectangular block of the tensor.

Example: a tensor `[1 × 1 × 512 × 4096]` block-sharded across a 4×8 core grid → each core holds `[1 × 1 × 128 × 512]`.

Used when:
- 2D matmuls that parallelize over both M and N dimensions simultaneously
- When the op following the matmul also needs 2D sharding

### Summary Table

| Layout | Row split | Column split | Common Use |
|---|---|---|---|
| `INTERLEAVED` | No | No | Weights at rest, KV cache, default |
| `HEIGHT_SHARDED` | Yes | No | Batch-parallel ops, row-parallel activations |
| `WIDTH_SHARDED` | No | Yes | Column-parallel matmul outputs, head sharding |
| `BLOCK_SHARDED` | Yes | Yes | 2D matmuls, prefill activations |

---

## Creating a Sharded Memory Config

TTNN provides `ttnn.create_sharded_memory_config` to construct sharded `MemoryConfig` objects:

```python
import ttnn

# Height-sharded: shard activations across 32 cores
# Tensor shape: [1, 1, 512, 4096] → each core holds [1, 1, 16, 4096]
height_sharded_config = ttnn.create_sharded_memory_config(
    shape=(1, 1, 512, 4096),          # full tensor shape
    core_grid=ttnn.CoreGrid(y=4, x=8), # 32 cores in a 4×8 grid
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=False,
)

# Width-sharded: shard output features across 8 cores
# Tensor shape: [1, 1, 32, 1024] → each core holds [1, 1, 32, 128]
width_sharded_config = ttnn.create_sharded_memory_config(
    shape=(1, 1, 32, 1024),
    core_grid=ttnn.CoreGrid(y=1, x=8),
    strategy=ttnn.ShardStrategy.WIDTH,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=False,
)

# Block-sharded: 2D shard across a 4×8 grid
block_sharded_config = ttnn.create_sharded_memory_config(
    shape=(1, 1, 512, 4096),
    core_grid=ttnn.CoreGrid(y=4, x=8),
    strategy=ttnn.ShardStrategy.BLOCK,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=False,
)
```

To apply a sharded memory config to a tensor already on device, use `ttnn.to_memory_config`:

```python
# Move an interleaved tensor to a height-sharded L1 layout
x_sharded = ttnn.to_memory_config(x_device, height_sharded_config)
```

The `ttnn.interleaved_to_sharded` op also exists for explicit resharding. In practice, many TTNN ops accept an `output_mem_config` argument that instructs them to write their output directly into the specified sharded memory config, avoiding a separate resharding step.

---

## Data Types and Memory Footprint

TTNN supports several numeric data types, each with a different memory footprint and hardware support level on Wormhole. The choice of data type for weight matrices is one of the most impactful performance knobs in tt-transformers.

### BF16 (Brain Float 16)

- 16-bit: 1 sign, 8 exponent, 7 mantissa bits
- Same exponent range as FP32, reduced mantissa precision
- **2 bytes per element**
- Standard format for activations throughout tt-transformers
- Supported by the matrix FPU with full precision

### FP32 (Single Precision Float)

- 32-bit: 1 sign, 8 exponent, 23 mantissa bits
- **4 bytes per element**
- Used for accumulation in the Dst register file when `fp32_dest_acc_en=True`
- Not typically used for stored tensors in production inference (too large); used internally for high-precision accumulation

### BFP8_B (Block Floating Point 8)

- A **block** of 16 values shares one 8-bit exponent; each value has a 7-bit mantissa + 1 sign bit = **8 bits per value** (before shared exponent cost), giving ~8.5 bits/element total (see memory footprint table below for derivation)
- Effective storage: approximately **~1.0625 bytes per element** — roughly half the memory of BF16
- The "B" suffix indicates the Tenstorrent variant (Bfloat block format)
- Used for attention weight matrices (`wqkv`, `wo`) and KV cache in tt-transformers
- Native hardware support on Wormhole — the Unpacker can decode BFP8_B blocks directly

### BFP4_B (Block Floating Point 4)

- Same block structure as BFP8_B: 16 values share one 8-bit exponent; each value has a 3-bit mantissa (plus 1 sign bit = 4 bits per value)
- Effective storage: approximately **~0.5625 bytes per element** — roughly one quarter the memory of BF16
- Used for MLP weight matrices (FF1, FF2, FF3) in tt-transformers — the lower precision is tolerable at this layer
- Halving the weight bandwidth vs BFP8 was measured to give +22% decode throughput on Llama 3.1 8B (23 → 28 tokens/s/user on N150)

### Memory Footprint Comparison Table

For a 4096×4096 weight matrix:

| Data Type | Bits per element | Bytes per element (approx) | Total for 4096×4096 |
|---|---|---|---|
| FP32 | 32 | 4.0 | ~64 MB |
| BF16 | 16 | 2.0 | ~32 MB |
| BFP8_B | ~8.5 (7-bit mantissa + shared exp overhead) | ~1.06 | ~17 MB |
| BFP4_B | ~4.5 (3-bit mantissa + shared exp overhead) | ~0.56 | ~9 MB |

> Note: The shared exponent is 8 bits for every 16 values, adding 0.5 bits per element overhead to both BFP8 and BFP4.

### Specifying Data Types in TTNN

```python
# Load a weight tensor in BFP8 format
w_bfp8 = ttnn.from_torch(
    weight_tensor,
    dtype=ttnn.bfloat8_b,   # BFP8_B
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# Load a weight tensor in BFP4 format
w_bfp4 = ttnn.from_torch(
    weight_tensor,
    dtype=ttnn.bfloat4_b,   # BFP4_B
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

In tt-transformers, weight tensors are converted to their target dtype once during model loading and remain in that format on device across all inference steps (see Chapter 6 on weight caching).

---

## Key Takeaways

- TilizedLayout stores tensors as 32×32 tiles (composed of four 16×16 faces), matching the matrix FPU's native input primitive. All matmul inputs must be in TilizedLayout.
- `BufferType.DRAM` is the default for weights and KV caches; `BufferType.L1` is used for activations that are sharded to cores for low-latency access during compute.
- The four `TensorMemoryLayout` options — `INTERLEAVED`, `HEIGHT_SHARDED`, `WIDTH_SHARDED`, `BLOCK_SHARDED` — determine how tiles are distributed across cores. Sharded layouts eliminate NoC traffic for shard reads and are the key to turning memory-bound ops into compute-bound ops.
- `ttnn.create_sharded_memory_config` is the primary API for constructing sharded `MemoryConfig` objects; the resulting config can be passed as `output_mem_config` to ops or applied via `ttnn.to_memory_config`.
- BFP8_B and BFP4_B are Tenstorrent block floating-point formats with native hardware support. BFP4_B for MLP weights reduces memory footprint to ~25% of BF16, which is the dominant reason tt-transformers achieves higher decode throughput than a naive BF16 implementation.

---

## Further Reading

- TTNN Python API reference — `ttnn.MemoryConfig`, `ttnn.create_sharded_memory_config`, `ttnn.TensorMemoryLayout` (github.com/tenstorrent/tt-metal/blob/main/docs/source/ttnn/ttnn/tensor.rst)
- Tenstorrent block floating-point format: the BFP8_B and BFP4_B type definitions and block-encoding details (shared exponent structure, per-value mantissa packing) can be found in `tt_metal/include/tt_metal/common/bfloat8.hpp` and `tt_metal/include/tt_metal/common/bfloat4.hpp` in the tt-metal repository (github.com/tenstorrent/tt-metal). Note that `bfloat16.hpp` in the same directory is for standard BF16 (Brain Float 16) and does not contain the block floating-point format definitions. File names should be verified against the current repository, as paths may change across versions.
- [tensix_architecture.md](./tensix_architecture.md) — hardware constraints that motivate the layout and sharding choices described here
- [math_fidelity_and_data_formats.md](./math_fidelity_and_data_formats.md) — how data type selection pairs with math fidelity for optimal throughput

---

**Next:** [`math_fidelity_and_data_formats.md`](./math_fidelity_and_data_formats.md)
