# Reader Dataflow Kernel: Batched NOC Reads and Sub-Tile Extraction

The reader kernel (`reader_gdn_fused.cpp`) is responsible for fetching all inputs for each pair from DRAM (or L1) into circular buffers where the compute kernel can consume them. Its key optimization is **batched NOC reads**: all 44 reads required for a single pair are issued before a single `noc_async_read_barrier()`, reducing barrier overhead from 17 barriers (in a naive per-tensor approach) to 1.

The reader also performs **sub-tile extraction** -- reading individual rows and scalar values from within packed tiles. The `conv_out` tensor is `[1, B, qkv_dim_tp]` where Q, K, and V are concatenated along the last dimension. Rather than reading entire tiles and discarding most of the data, the reader reads only the specific 32-element rows corresponding to each batch element, then copies them into full tiles in the circular buffers.

## Compile-Time and Runtime Arguments

The reader receives 11 compile-time arguments and 12 runtime arguments per core, as detailed in [`kernel_dispatch.md`](./kernel_dispatch.md). The key compile-time values that drive the reader logic are:

- `Kt = 4`, `Vt = 4`: tile counts for key and value dimensions
- `STATE_IN_L1`: controls whether state reads use DRAM or L1 address generators
- `STATE_IS_SHARDED`: controls whether state reads use NOC or direct L1 memcpy
- `Nv_TP = 12`, `Nk_TP = 4`, `repeat_factor = 3`: head mapping parameters
- `key_tile_offset = 16`, `v_tile_offset = 32`: tile offsets into the packed `conv_out` tensor

The runtime arguments provide buffer addresses for all 10 input tensors plus `pair_start` and `num_pairs` for the core's pair assignment.

## Address Generators

The reader constructs `InterleavedAddrGenFast` address generators for each input tensor:

```cpp
const InterleavedAddrGenFast<true> conv_rd = {
    .bank_base_address = conv_out_addr, .page_size = tile_bytes,
    .data_format = DataFormat::Float16_b};
```

All input tensors except state use `InterleavedAddrGenFast<true>` (DRAM). The state address generator uses a constexpr template parameter derived from the compile-time flags:

```cpp
constexpr bool state_is_dram = (STATE_IN_L1 == 0) && (STATE_IS_SHARDED == 0);
const InterleavedAddrGenFast<state_is_dram> state_rd = { ... };
```

This compile-time branching means the compiler generates separate code paths for DRAM vs. L1 state, with no runtime overhead for the branch decision.

## Scratch Buffer Layout

The reader allocates a single tile (2048 bytes) from `cb_scratch` (`c_21`) as a staging area for sub-tile reads. The 1792 bytes of useful space within this tile are partitioned as:

```
Offset     Size    Content
[0..511]   512B    Q: 4 tiles x 128 bytes (2 face-halves per tile)
[512..1023] 512B   K: 4 tiles x 128 bytes
[1024..1535] 512B  V: 4 tiles x 128 bytes
[1536..1599] 64B   a scalar
[1600..1663] 64B   b scalar
[1664..1727] 64B   neg_exp_A scalar
[1728..1791] 64B   dt_bias scalar
```

The scratch is allocated once at kernel startup via `cb_reserve_back(cb_scratch, 1)` and the L1 address is captured with `get_write_ptr(cb_scratch)`. It persists across all pairs.

## Pair-to-Head Mapping

Each pair index `p` maps to a batch element and head indices:

```cpp
uint32_t batch_idx = p / Nv_TP;     // which batch element (0..31)
uint32_t v_head    = p % Nv_TP;     // which value head (0..11)
uint32_t k_head    = v_head / repeat_factor;  // which key head (0..3)
```

Since `repeat_factor = 3`, three consecutive value heads share the same key head. For example, value heads 0, 1, 2 all use key head 0; value heads 3, 4, 5 use key head 1; and so on. This means Q and K are read from the `k_head`-indexed position within `conv_out`, while V is read from the `v_head`-indexed position.

## Sub-Tile Row Extraction: `issue_row_reads`

The `conv_out` tensor stores data for all batch elements within tiles. In TILE_LAYOUT, a 32x32 bfloat16 tile (2048 bytes) is stored as four 16x16 faces:

```
Face layout (2048 bytes):
  [0..511]    Face 0: rows 0-15, cols 0-15
  [512..1023] Face 1: rows 0-15, cols 16-31
  [1024..1535] Face 2: rows 16-31, cols 0-15
  [1536..2047] Face 3: rows 16-31, cols 16-31
```

Within each face, rows are stored consecutively at 32 bytes per row (16 elements x 2 bytes each). To extract a single row (representing one batch element), the reader needs data from two faces -- one for columns 0-15 and one for columns 16-31.

The `issue_row_reads` function issues exactly 2 NOC reads per tile:

```cpp
template <bool is_dram>
FORCE_INLINE void issue_row_reads(
    const InterleavedAddrGenFast<is_dram>& addr_gen,
    uint32_t tile_id, uint32_t row, uint32_t scratch_slot
) {
    uint32_t face_base = (row < 16) ? 0 : 1024;
    uint32_t aligned_row = (row % 16) & ~1u;

    uint64_t src0 = addr_gen.get_noc_addr(tile_id, face_base + aligned_row * 32);
    noc_async_read(src0, scratch_slot, 64);

    uint64_t src1 = addr_gen.get_noc_addr(tile_id, face_base + 512 + aligned_row * 32);
    noc_async_read(src1, scratch_slot + 64, 64);
}
```

Key details:

- **Face selection**: `face_base = (row < 16) ? 0 : 1024` selects between the upper faces (0 and 1) and lower faces (2 and 3)
- **Row alignment**: `aligned_row = (row % 16) & ~1u` rounds down to the nearest even row. NOC reads are 64-byte aligned, and each 64-byte chunk contains 2 consecutive rows (2 x 16 elements x 2 bytes = 64 bytes). The target row is either the first or second row within this 64-byte block.
- **Two reads**: The first read gets columns 0-15 from the left face; the second gets columns 16-31 from the right face (offset by 512 bytes within the tile). Each read is 64 bytes.
- **128 bytes per tile**: The scratch slot for each tile is 128 bytes -- 64 bytes for each face half.

For Q and K (each `Kt=4` tiles), this produces `4 x 2 = 8` NOC reads. For V (also `Vt=4` tiles), another 8 reads. Total: 24 reads for Q/K/V row extraction.

## Sub-Tile Scalar Extraction: `issue_scalar_read`

The gate tensors (`a`, `b`, `neg_exp_A`, `dt_bias`) have shapes like `[1, B, Nv_TP]` where `Nv_TP=12`. Since the last dimension is less than 32, a single tile contains multiple scalars. The reader extracts one scalar per pair by reading the 64-byte aligned block containing the target element:

```cpp
template <bool is_dram>
FORCE_INLINE void issue_scalar_read(
    const InterleavedAddrGenFast<is_dram>& addr_gen,
    uint32_t tile_id, uint32_t row, uint32_t col, uint32_t scratch_slot
) {
    uint32_t face_base = (row < 16 ? 0 : 1024) + (col < 16 ? 0 : 512);
    uint32_t aligned_row = (row % 16) & ~1u;
    uint64_t src = addr_gen.get_noc_addr(tile_id, face_base + aligned_row * 32);
    noc_async_read(src, scratch_slot, 64);
}
```

This reads 64 bytes containing 2 rows x 16 columns of bfloat16 values. The specific scalar is extracted later in the local copy phase. For the per-pair scalars, `row = batch_idx` and `col = v_head`. For the constants (`neg_exp_A`, `dt_bias`), `row = 0` and `col = v_head`.

Total scalar reads: 4 per pair.

## State Reads

Each pair's recurrence state is `Kt x Vt = 16` tiles at `state_tiles = 16` x 2048 bytes = 32 KB. State reads use one of two paths based on compile-time flags:

**DRAM / L1 interleaved path** (`STATE_IS_SHARDED == 0`): reads via NOC with the state address generator:

```cpp
for (uint32_t s = 0; s < state_tiles; s++) {
    noc_async_read_tile(p * state_tiles + s, state_rd, wp_st + s * tile_bytes);
}
```

This issues 16 NOC reads per pair. The tile index is `p * state_tiles + s` -- pair `p`'s state starts at tile `p * 16`.

**HEIGHT_SHARDED path** (`STATE_IS_SHARDED == 1`): the state is already in L1 on this core, so no NOC read is needed. Instead, a direct L1 memcpy copies the data into the CB:

```cpp
uint32_t shard_byte_offset = pair * state_tiles * tile_bytes;
uint32_t src_addr = state_addr + shard_byte_offset;
volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addr);
volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wp_st);
for (uint32_t w = 0; w < num_words; w++) {
    dst[w] = src[w];
}
```

Note the `volatile tt_l1_ptr` qualifier -- this ensures the compiler does not optimize away the copy and that accesses go through the L1 pointer path.

## Total Read Count Per Pair

| Source | Reads | Description |
|--------|-------|-------------|
| Q rows | 8 | 4 tiles x 2 face-halves |
| K rows | 8 | 4 tiles x 2 face-halves |
| V rows | 8 | 4 tiles x 2 face-halves |
| Scalars (a, b, neg_exp_A, dt_bias) | 4 | 1 read each |
| State tiles | 16 | 16 full tile reads (DRAM path) |
| **Total** | **44** | All issued before single barrier |

All 44 reads are issued in sequence with no intervening barriers. After the last read, a single `noc_async_read_barrier()` waits for all of them to complete. This is critical for performance: each barrier causes a RISC-V stall while the NOC controller drains its queue, and issuing all reads first maximizes NOC utilization by allowing the hardware to pipeline read requests.

## Local Copy Phase

After the barrier, all data is in the scratch buffer (for sub-tile extracts) or directly in the CB (for state tiles). The reader then performs pure L1 copies to move data from scratch into the target CBs.

### `copy_row_to_tile`: Row Data

```cpp
FORCE_INLINE void copy_row_to_tile(
    uint32_t row, uint32_t scratch_slot, uint32_t dest_l1
) {
    uint32_t row_offset = ((row % 16) & 1u) * 32;

    volatile uint32_t* dst0 = reinterpret_cast<volatile uint32_t*>(dest_l1);
    volatile uint32_t* s0 = reinterpret_cast<volatile uint32_t*>(scratch_slot + row_offset);
    for (uint32_t i = 0; i < 8; i++) dst0[i] = s0[i];

    volatile uint32_t* dst1 = reinterpret_cast<volatile uint32_t*>(dest_l1 + 512);
    volatile uint32_t* s1 = reinterpret_cast<volatile uint32_t*>(scratch_slot + 64 + row_offset);
    for (uint32_t i = 0; i < 8; i++) dst1[i] = s1[i];
}
```

The `row_offset = ((row % 16) & 1u) * 32` selects the correct row within the 64-byte block that was read. If the row is even within the face, `row_offset = 0` (first row in the pair); if odd, `row_offset = 32` (second row). The copy writes 32 bytes to face 0 position and 32 bytes to face 1 position of the destination tile, placing the extracted row at row 0 of the output tile.

### `copy_scalar_to_tile`: Scalar Data

```cpp
FORCE_INLINE void copy_scalar_to_tile(
    uint32_t row, uint32_t col, uint32_t scratch_slot, uint32_t dest_l1
) {
    uint32_t row_offset = ((row % 16) & 1u) * 32;
    uint32_t face_col = col % 16;
    volatile uint16_t* src_val = reinterpret_cast<volatile uint16_t*>(
        scratch_slot + row_offset + face_col * 2);
    volatile uint16_t* dst_val = reinterpret_cast<volatile uint16_t*>(dest_l1);
    *dst_val = *src_val;
}
```

This extracts a single bfloat16 value from the 64-byte scratch block and writes it to position `[0,0]` of the destination tile. The compute kernel treats the entire tile as a scalar broadcast value, so only the `[0,0]` element matters.

### K Tile Zeroing

Before copying K row data, each K tile is zeroed:

```cpp
for (uint32_t kt = 0; kt < Kt; kt++) {
    uint32_t tile_addr = wp_k + kt * tile_bytes;
    zero_tile(tile_addr, tile_bytes);
    copy_row_to_tile(batch_idx, scratch_l1 + SCRATCH_K + kt * 128, tile_addr);
}
```

This zeroing is required because the K tile is later transposed to form `k_col` for the outer product in the recurrence. If non-zero garbage exists in other rows of the K tile, the transpose would produce incorrect column vectors. Q and V do not need zeroing because Q is only used as a `[1, Dk]` row vector (matmul only reads row 0) and V is used element-wise.

## Persistent Constants

Five values are read once at kernel startup, before the per-pair loop, and remain in their CBs for the entire kernel execution:

1. `cb_norm_w` (`c_14`, Vt=4 tiles): RMS norm weight vector `[1, 1, Dv]`
2. `cb_scale` (`c_15`, 1 tile): Q scale factor `Dk^(-0.5)`
3. `cb_rms_scale` (`c_31`, 1 tile): `sqrt(Dv)` for RMS normalization
4. `cb_rms_eps` (`c_20`, 1 tile): `Dv * eps` for numerical stability
5. `cb_reduce_scaler` (`c_19`, 1 tile): All-ones tile for reduce operations

The first four are read from DRAM via `noc_async_read_tile` calls, all batched under a single barrier:

```cpp
cb_reserve_back(cb_norm_w, Vt);
cb_reserve_back(cb_scale, 1);
cb_reserve_back(cb_rms_scale, 1);
cb_reserve_back(cb_rms_eps, 1);

// Issue all constant reads (7 tiles total)
for (uint32_t vt = 0; vt < Vt; vt++)
    noc_async_read_tile(vt, norm_w_rd, wp + vt * tile_bytes);
noc_async_read_tile(0, scale_rd, get_write_ptr(cb_scale));
noc_async_read_tile(0, rms_scale_rd, get_write_ptr(cb_rms_scale));
noc_async_read_tile(0, rms_eps_rd, get_write_ptr(cb_rms_eps));

noc_async_read_barrier();  // Single barrier for all 7 tile reads

cb_push_back(cb_norm_w, Vt);
cb_push_back(cb_scale, 1);
cb_push_back(cb_rms_scale, 1);
cb_push_back(cb_rms_eps, 1);
```

The `cb_reduce_scaler` is generated locally via the `generate_reduce_scaler` utility, which fills a tile with a packed bfloat16 constant (`0x3F803F80` = 1.0 in bfloat16, packed as two values). This requires no NOC read.

The compute kernel waits for these constants at the very beginning (`cb_wait_front`) and does not pop them until after the last pair, so they persist in L1 across all pairs without re-reads.

## CB Push Sequence

After the local copy phase for each pair, all per-pair CBs are pushed to the compute kernel:

```cpp
cb_push_back(cb_q_raw, Kt);
cb_push_back(cb_k_raw, Kt);
cb_push_back(cb_v, Vt);
cb_push_back(cb_a, 1);
cb_push_back(cb_b, 1);
cb_push_back(cb_neg_exp_A, 1);
cb_push_back(cb_dt_bias, 1);
cb_push_back(cb_state, state_tiles);
```

This signals the compute kernel (which is blocked on `cb_wait_front` calls) that data is ready. The reader then immediately begins issuing reads for the next pair while the compute kernel processes the current one, achieving pipeline overlap between data movement and computation.

---

**Previous:** [`kernel_dispatch.md`](./kernel_dispatch.md) | **Next:** [`compute_kernel.md`](./compute_kernel.md)
