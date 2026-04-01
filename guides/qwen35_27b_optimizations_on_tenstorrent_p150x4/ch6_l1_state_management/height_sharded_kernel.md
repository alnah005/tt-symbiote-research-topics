# HEIGHT_SHARDED L1 State Support in the Custom Kernel

Moving GDN recurrence state from DRAM to L1 INTERLEAVED eliminates long-latency DRAM access but still requires NOC transfers: the state tiles may reside on a different L1 bank than the compute core processing them. HEIGHT_SHARDED goes further by placing each core's state tiles in that core's own L1, enabling direct memory access with zero NOC overhead.

The fused GDN kernel (`reader_gdn_fused.cpp`, `writer_gdn_fused.cpp`) supports both modes through the `STATE_IS_SHARDED` compile-time argument. When `STATE_IS_SHARDED = 1` the reader and writer bypass the NOC entirely for state access, using `volatile tt_l1_ptr` pointer arithmetic instead. When `STATE_IS_SHARDED = 0` the kernel uses `noc_async_read_tile` / `noc_async_write_tile` with an `InterleavedAddrGenFast` address generator, which works for both DRAM and L1 INTERLEAVED layouts.

## Compile-Time Branching

The two kernels use different compile-time arg indices for `STATE_IS_SHARDED`. In the reader kernel (`reader_gdn_fused.cpp`, line 130):

```cpp
constexpr uint32_t STATE_IN_L1      = get_compile_time_arg_val(3);
constexpr uint32_t STATE_IS_SHARDED = get_compile_time_arg_val(10);
```

In the writer kernel (`writer_gdn_fused.cpp`, line 25):

```cpp
constexpr uint32_t STATE_IN_L1      = get_compile_time_arg_val(3);
constexpr uint32_t STATE_IS_SHARDED = get_compile_time_arg_val(6);
```

Both kernels use the same expression to select the `InterleavedAddrGenFast` template parameter (reader line 178, writer line 37):

```cpp
constexpr bool state_is_dram = (STATE_IN_L1 == 0) && (STATE_IS_SHARDED == 0);
const InterleavedAddrGenFast<state_is_dram> state_rd = {
    .bank_base_address = state_addr, .page_size = tile_bytes,
    .data_format = DataFormat::Float16_b};
```

When `STATE_IS_SHARDED = 1` the template resolves to `InterleavedAddrGenFast<false>` (L1 interleaved variant). However the address generator is never invoked for state when the sharded path is active — the `if constexpr` branches compile out the NOC paths entirely, leaving only the direct-copy path.

## Reader Kernel: HEIGHT_SHARDED State Path

In the reader kernel (`reader_gdn_fused.cpp`, lines 268-277), the HEIGHT_SHARDED path replaces 16 NOC tile reads with a direct L1-to-L1 memcpy:

```cpp
if constexpr (STATE_IS_SHARDED) {
    // HEIGHT_SHARDED: state is local on this core — direct L1 copy (no NOC)
    uint32_t shard_byte_offset = pair * state_tiles * tile_bytes;
    uint32_t src_addr = state_addr + shard_byte_offset;
    uint32_t num_words = (state_tiles * tile_bytes) >> 2;  // /4 for uint32
    volatile tt_l1_ptr uint32_t* src =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addr);
    volatile tt_l1_ptr uint32_t* dst =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wp_st);
    for (uint32_t w = 0; w < num_words; w++) {
        dst[w] = src[w];
    }
}
```

Key details of this path:

**Shard offset calculation.** The state for local pair `pair` within this core's shard starts at `state_addr + pair * state_tiles * tile_bytes`. Since `state_tiles = Kt * Vt = 16` and `tile_bytes = 2048`, each pair occupies 32,768 bytes (32 KB) of contiguous L1. The `pair` variable is local to this core (0-based), not the global pair index.

**The `volatile tt_l1_ptr` qualifier.** `volatile` prevents the compiler from reordering or eliding memory accesses. The `tt_l1_ptr` address-space qualifier tells the Tensix compiler that these pointers reference L1 memory, enabling correct address translation on the RISC-V cores.

**Word-granularity copy.** The copy operates on `uint32_t` words (4 bytes each). For 16 tiles of 2048 bytes: $16 \times 2048 / 4 = 8192$ word copies per pair. This avoids NOC contention and the overhead of setting up and waiting on asynchronous NOC transactions.

**Placement within the batched read flow.** The L1 memcpy executes in the same position as the 16 `noc_async_read_tile` calls in the DRAM/INTERLEAVED path (line 267 comment: "State: 16 full tile reads"). It runs before the single `noc_async_read_barrier()` that synchronizes the Q/K/V/scalar NOC reads (line 285). Because the memcpy is synchronous the state data is immediately available — it does not need the barrier.

## Writer Kernel: HEIGHT_SHARDED State Path

The writer kernel (`writer_gdn_fused.cpp`, lines 58-67) mirrors the reader's approach for state writeback:

```cpp
if constexpr (STATE_IS_SHARDED) {
    // HEIGHT_SHARDED: write to local L1 shard — direct copy (no NOC)
    uint32_t shard_byte_offset = pair * state_tiles * tile_bytes;
    uint32_t dst_addr = state_addr + shard_byte_offset;
    uint32_t num_words = (state_tiles * tile_bytes) >> 2;
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sp);
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
    for (uint32_t w = 0; w < num_words; w++) {
        dst[w] = src[w];
    }
}
```

The write path copies from `cb_state_out` (where the compute kernel placed the updated state) back to the shard's L1 address. The `noc_async_write_barrier()` that follows (line 75) covers the output tile NOC writes to DRAM; the state writeback itself is synchronous and complete before the barrier is reached.

## HEIGHT_SHARDED Config Construction

The Python test (`test_e2e_l1_hs.py`, lines 83-94) shows how the HEIGHT_SHARDED memory config is built:

```python
# rec_states shape: [B*Nv_TP, Dk, Dv] = [384, 128, 128]
# Flattened height: 384 * 128 = 49152 rows
total_rows = batch_size * args.gdn_nv_tp * args.gdn_dk  # 32 * 12 * 128 = 49152
shard_h = total_rows // NUM_HS_CORES  # 512
cr1 = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7))  # 88 cores
cr2 = ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(7, 8))   # 8 cores = 96 total
cg = ttnn.CoreRangeSet([cr1, cr2])
hs_cfg = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
    ttnn.ShardSpec(cg, [shard_h, args.gdn_dv], ttnn.ShardOrientation.ROW_MAJOR)
)
```

## DRAM vs L1 INTERLEAVED vs HEIGHT_SHARDED Comparison

| Aspect | DRAM (baseline) | L1 INTERLEAVED | HEIGHT_SHARDED |
|--------|-----------------|----------------|----------------|
| State location | DRAM banks | L1 banks (any core) | L1 on compute core |
| Read mechanism | `noc_async_read_tile` via DRAM NOC | `noc_async_read_tile` via L1 NOC | Direct `volatile tt_l1_ptr` memcpy |
| Write mechanism | `noc_async_write_tile` via DRAM NOC | `noc_async_write_tile` via L1 NOC | Direct `volatile tt_l1_ptr` memcpy |
| NOC transactions per pair (state) | 16 reads + 16 writes | 16 reads + 16 writes | 0 |
| Address generator | `InterleavedAddrGenFast<true>` | `InterleavedAddrGenFast<false>` | Not used for state |
| Capacity constraint | None | Limited by total L1 | Limited by per-core L1 |

## Output Tiles: Always DRAM

Regardless of the state access mode, output tiles are always written to DRAM via NOC. The writer kernel header (`writer_gdn_fused.cpp`, lines 6-8) is explicit:

```
// Writes output tiles to [1, B, value_dim_tp] layout (not [num_pairs, 1, Dv]),
// mapping pair -> (batch_idx, v_head) to place tiles at correct positions.
// Also writes updated recurrence state back to DRAM/L1.
```

The output tensor `[1, B, value_dim_tp]` feeds into the subsequent RMS norm and output projection, which expect DRAM-resident inputs. Only the state — consumed exclusively by the same kernel on the next decode step — benefits from L1 residency.

Validation status for all three state configurations (DRAM baseline, L1 INTERLEAVED, and HEIGHT_SHARDED) is summarized in [`sdpa_l1_conflict.md`](./sdpa_l1_conflict.md) §Current Status.

---

**Next:** [`sdpa_l1_conflict.md`](./sdpa_l1_conflict.md)
