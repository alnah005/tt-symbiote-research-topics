# Writer Kernel: Output and State Writeback

The writer kernel (`writer_gdn_fused.cpp`) is the simplest of the three kernel components. For each pair, it waits for the compute kernel to produce output tiles in `cb_out` (`c_16`) and updated state tiles in `cb_state_out` (`c_8`), then writes both to their target memory locations. It supports two state writeback paths: NOC writes for DRAM-interleaved or L1-interleaved state, and direct L1 memcpy for HEIGHT_SHARDED state.

## Compile-Time and Runtime Arguments

**Compile-time args** (7 values):

| Index | Value | Description |
|-------|-------|-------------|
| 0 | `Kt = 4` | Key dimension in tiles |
| 1 | `Vt = 4` | Value dimension in tiles |
| 2 | `tile_bytes = 2048` | Tile size in bytes |
| 3 | `STATE_IN_L1` | 1 if state is in L1 interleaved, 0 if DRAM |
| 4 | 0 | Unused (Nv_TP, kept for compatibility) |
| 5 | 0 | Unused (out_tiles_per_batch, kept for compatibility) |
| 6 | `STATE_IS_SHARDED` | 1 if state is HEIGHT_SHARDED in L1 |

**Runtime args** (4 values per core):

| Index | Value | Description |
|-------|-------|-------------|
| 0 | `out_addr` | DRAM address of output tensor `[num_pairs, 1, Dv]` |
| 1 | `state_addr` | DRAM or L1 address of state tensor `[num_pairs, Dk, Dv]` |
| 2 | `pair_start` | First pair index for this core |
| 3 | `num_pairs` | Number of pairs on this core |

## Address Generators

The writer constructs two `InterleavedAddrGenFast` address generators:

```cpp
// Output always goes to DRAM
constexpr bool is_dram = true;
const InterleavedAddrGenFast<is_dram> out_wr = {
    .bank_base_address = out_addr, .page_size = tile_bytes,
    .data_format = DataFormat::Float16_b};

// State destination depends on compile-time flags
constexpr bool state_is_dram = (STATE_IN_L1 == 0) && (STATE_IS_SHARDED == 0);
const InterleavedAddrGenFast<state_is_dram> state_wr = {
    .bank_base_address = state_addr, .page_size = tile_bytes,
    .data_format = DataFormat::Float16_b};
```

The output is always written to DRAM because it feeds into subsequent operations (RMS norm, SiLU gating, output projection) that expect DRAM-resident tensors. The state destination depends on the L1 state optimization (Chapter 6).

## Per-Pair Write Loop

The writer processes pairs in the same order as the reader and compute kernels, using `pair_start` to determine the global pair index for tile addressing:

```cpp
for (uint32_t pair = 0; pair < num_pairs; pair++) {
    uint32_t p = pair_start + pair;
    uint32_t out_tile_base = p * Vt;

    // Wait for compute to produce both outputs
    cb_wait_front(cb_out, Vt);
    cb_wait_front(cb_state_out, state_tiles);

    // Write output tiles
    uint32_t rp = get_read_ptr(cb_out);
    for (uint32_t vt = 0; vt < Vt; vt++) {
        noc_async_write_tile(out_tile_base + vt, out_wr, rp);
        rp += tile_bytes;
    }

    // Write state tiles (path depends on compile-time flags)
    // ...

    noc_async_write_barrier();
    cb_pop_front(cb_out, Vt);
    cb_pop_front(cb_state_out, state_tiles);
}
```

### Output Layout

The output tensor has shape `[num_pairs, 1, Dv]` in a sequential per-pair tile layout. Each pair writes Vt=4 tiles starting at tile index `p * Vt`. For pair 0, tiles 0-3; for pair 1, tiles 4-7; and so on. With `num_pairs = 384` and `Vt = 4`, the output contains 1536 tiles total (3 MB).

### State Writeback: DRAM / L1 Interleaved Path

When `STATE_IS_SHARDED == 0`, state is written via NOC:

```cpp
uint32_t sp = get_read_ptr(cb_state_out);
for (uint32_t s = 0; s < state_tiles; s++) {
    noc_async_write_tile(p * state_tiles + s, state_wr, sp);
    sp += tile_bytes;
}
```

Each pair writes 16 tiles (32 KB) to the state tensor at tile offset `p * state_tiles`. The `state_wr` address generator routes writes to either DRAM or L1 based on the compile-time `state_is_dram` template parameter.

### State Writeback: HEIGHT_SHARDED L1 Path

When `STATE_IS_SHARDED == 1`, the state shard is local to the compute core and no NOC write is needed:

```cpp
if constexpr (STATE_IS_SHARDED) {
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

Key details of the HEIGHT_SHARDED path:

- **No NOC involved**: The copy is a direct L1-to-L1 memory transfer on the same core. This eliminates NOC write latency and bandwidth consumption entirely.
- **`volatile tt_l1_ptr` qualifier**: Ensures the compiler does not optimize away the copy or reorder it relative to the NOC write barrier.
- **Offset calculation**: `pair * state_tiles * tile_bytes` computes the byte offset within the local shard. The `pair` index here is the **local** pair index (0 to `num_pairs-1` on this core), not the global pair index `p`.
- **Word-granularity copy**: The loop copies `(16 * 2048) / 4 = 8192` 32-bit words per pair (32 KB).

This path is the key enabler for the L1 state optimization described in Chapter 6. By keeping the state in HEIGHT_SHARDED L1, both the reader and writer avoid NOC transfers for state data, reducing per-pair data movement by 64 KB (32 KB read + 32 KB write).

## Barrier Strategy

Both the output write and state write (in the non-sharded path) are covered by a single `noc_async_write_barrier()` per pair:

```cpp
noc_async_write_barrier();
cb_pop_front(cb_out, Vt);
cb_pop_front(cb_state_out, state_tiles);
```

The barrier ensures all NOC writes for the pair have completed before the CBs are freed. This is important because `cb_pop_front` makes the CB space available for the next pair's data from the compute kernel -- if the NOC write has not completed, the reader/compute pipeline could overwrite the data being written to DRAM.

For the HEIGHT_SHARDED path, the L1 memcpy is synchronous (no NOC involvement), so the barrier only covers the output write. The state data is already committed to L1 by the time the barrier call executes.

## Write Volume Per Pair

| Destination | Tiles | Bytes | Method |
|-------------|-------|-------|--------|
| Output (DRAM) | Vt=4 | 8 KB | NOC write |
| State (DRAM path) | 16 | 32 KB | NOC write |
| State (L1 sharded path) | 16 | 32 KB | L1 memcpy |
| **Total per pair** | **20** | **40 KB** | |

For the full kernel (384 pairs across all cores), total write volume is:
- Output: `384 * 8 KB = 3 MB` (always to DRAM)
- State: `384 * 32 KB = 12 MB` (DRAM or L1 depending on configuration)

With the HEIGHT_SHARDED L1 path, the 12 MB state writeback becomes a local L1 operation, reducing DRAM write bandwidth by 12 MB per GDN layer per decode step.

---

**Previous:** [`compute_kernel.md`](./compute_kernel.md) | **Next:** [Chapter 5 — Prefill TTFT Optimization](../ch5_prefill_ttft_optimization/index.md)
