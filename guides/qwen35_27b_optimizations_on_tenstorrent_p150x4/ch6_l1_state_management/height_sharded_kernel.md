# HEIGHT_SHARDED L1 State Support in the Custom Kernel

Moving GDN recurrence state from DRAM to L1 INTERLEAVED eliminates the long-latency DRAM access but still requires NOC transfers -- the state tiles may reside on different L1 banks than the compute core processing them. HEIGHT_SHARDED takes this a step further: each core's state tiles are placed in that core's own L1, enabling direct memory access with zero NOC overhead.

The fused GDN kernel supports both modes through the `STATE_IS_SHARDED` compile-time argument. When `STATE_IS_SHARDED = 1`, the reader and writer kernels bypass the NOC entirely for state access, using `volatile tt_l1_ptr` pointer-based memcpy instead. When `STATE_IS_SHARDED = 0`, the kernel falls back to `noc_async_read_tile` / `noc_async_write_tile` with an `InterleavedAddrGenFast` address generator, which works for both DRAM and L1 INTERLEAVED layouts.

## Compile-Time Branching

The kernel uses two compile-time constants to determine the state access path. In the reader kernel (`reader_gdn_fused.cpp`, line 130):

```cpp
constexpr uint32_t STATE_IN_L1     = get_compile_time_arg_val(3);
constexpr uint32_t STATE_IS_SHARDED = get_compile_time_arg_val(10);
```

In the writer kernel (`writer_gdn_fused.cpp`, line 25), `STATE_IS_SHARDED` uses a different compile-time arg index:

```cpp
constexpr uint32_t STATE_IN_L1     = get_compile_time_arg_val(3);
constexpr uint32_t STATE_IS_SHARDED = get_compile_time_arg_val(6);
```

These feed into the address generator template parameter:

```cpp
constexpr bool state_is_dram = (STATE_IN_L1 == 0) && (STATE_IS_SHARDED == 0);
const InterleavedAddrGenFast<state_is_dram> state_rd = {
    .bank_base_address = state_addr, .page_size = tile_bytes,
    .data_format = DataFormat::Float16_b};
```

When `STATE_IS_SHARDED = 1`, the `InterleavedAddrGenFast` is still constructed (template parameter resolves to `false` for L1) but is never used for state reads or writes -- the `if constexpr` branches compile out the NOC paths entirely.

## Reader Kernel: HEIGHT_SHARDED State Path

In the reader kernel (`reader_gdn_fused.cpp`, lines 268-277), the HEIGHT_SHARDED path replaces 16 NOC tile reads with a direct L1-to-L1 copy:

```cpp
if constexpr (STATE_IS_SHARDED) {
    // HEIGHT_SHARDED: state is local on this core -- direct L1 copy (no NOC)
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

- **Shard offset calculation.** The state for pair `p` within this core's shard starts at `state_addr + pair * state_tiles * tile_bytes`. Since `state_tiles = Kt * Vt = 16` and `tile_bytes = 2048`, each pair occupies 32,768 bytes (32 KB) of contiguous L1. The `pair` variable is local to this core (0-based), not the global pair index.

- **The `volatile tt_l1_ptr` qualifier.** The `volatile` keyword prevents the compiler from reordering or eliding the memory accesses. The `tt_l1_ptr` address space qualifier tells the Tensix compiler that these pointers reference L1 memory, enabling correct address translation on the RISC-V cores.

- **Word-granularity copy.** The copy operates on `uint32_t` words (4 bytes each). For 16 tiles of 2048 bytes, this is `16 * 2048 / 4 = 8192` word copies per pair. While not as fast as a DMA transfer, this avoids NOC contention and the overhead of setting up and waiting on asynchronous NOC transactions.

- **Placement within the batched read flow.** The L1 memcpy executes in the same position as the 16 `noc_async_read_tile` calls in the DRAM/INTERLEAVED path. It runs before the single `noc_async_read_barrier()` that synchronizes the Q/K/V/scalar NOC reads. Since the memcpy is synchronous, the state data is immediately available -- it does not need the barrier.

## Writer Kernel: HEIGHT_SHARDED State Path

The writer kernel (`writer_gdn_fused.cpp`, lines 58-66) mirrors the reader's approach for state writeback:

```cpp
if constexpr (STATE_IS_SHARDED) {
    // HEIGHT_SHARDED: write to local L1 shard -- direct copy (no NOC)
    uint32_t shard_byte_offset = pair * state_tiles * tile_bytes;
    uint32_t dst_addr = state_addr + shard_byte_offset;
    uint32_t num_words = (state_tiles * tile_bytes) >> 2;
    volatile tt_l1_ptr uint32_t* src =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sp);
    volatile tt_l1_ptr uint32_t* dst =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
    for (uint32_t w = 0; w < num_words; w++) {
        dst[w] = src[w];
    }
}
```

The write path copies from the `cb_state_out` circular buffer (where the compute kernel placed the updated state) back to the shard's L1 address. The `noc_async_write_barrier()` that follows still runs (it covers the output tile NOC writes to DRAM), but the state writeback itself is already complete before the barrier is reached.

## DRAM vs L1 INTERLEAVED vs HEIGHT_SHARDED Comparison

| Aspect | DRAM (baseline) | L1 INTERLEAVED | HEIGHT_SHARDED |
|--------|-----------------|----------------|----------------|
| State location | DRAM banks | L1 banks (any core) | L1 on compute core |
| Read mechanism | `noc_async_read_tile` via DRAM NOC | `noc_async_read_tile` via L1 NOC | Direct `volatile tt_l1_ptr` memcpy |
| Write mechanism | `noc_async_write_tile` via DRAM NOC | `noc_async_write_tile` via L1 NOC | Direct `volatile tt_l1_ptr` memcpy |
| NOC transactions per pair | 16 reads + 16 writes | 16 reads + 16 writes | 0 |
| Latency | High (DRAM access) | Medium (L1 NOC) | Low (local SRAM) |
| Address generator | `InterleavedAddrGenFast<true>` | `InterleavedAddrGenFast<false>` | Not used (offset arithmetic) |
| Capacity constraint | None (DRAM is large) | Limited by total L1 | Limited by per-core L1 |

## Output Tiles: Always DRAM

Regardless of the state access mode, the output tiles are always written to DRAM via NOC. The output tensor `[1, B, value_dim_tp]` feeds into the subsequent RMS norm and output projection, which expect DRAM-resident inputs. Only the state -- which is consumed exclusively by the same kernel on the next decode step -- benefits from L1 residency.

## Validation Status

HEIGHT_SHARDED state has been validated in isolation:

- **1-2 GDN layers with HEIGHT_SHARDED state** produce the correct "Paris" output in end-to-end generation (`test_e2e_l1_hs.py`).
- **L1 INTERLEAVED** (non-sharded) works correctly with up to 4 layers.
- The kernel compiles and runs correctly in both modes; the `if constexpr` branching ensures no dead code reaches the compiler for either path.

The remaining blocker for full deployment is the L1 address space conflict with SDPA circular buffers, covered in the next section.

---

**Previous:** [`l1_state_design.md`](./l1_state_design.md) | **Next:** [`sdpa_l1_conflict.md`](./sdpa_l1_conflict.md)
