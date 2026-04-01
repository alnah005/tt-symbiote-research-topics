# Writer Kernel: Output and State Writeback

The writer kernel (`writer_gdn_fused.cpp`) waits for the compute kernel to produce output tiles in `cb_out` (`c_16`) and updated state tiles in `cb_state_out` (`c_8`), then writes both to their target memory locations. It supports two state writeback paths: NOC writes for DRAM-interleaved or L1-interleaved state, and direct L1 memcpy for HEIGHT_SHARDED state.

Compile-time and runtime argument details are in [`kernel_dispatch.md`](./kernel_dispatch.md). In brief: 7 compile-time args covering `Kt`, `Vt`, tile size, `STATE_IN_L1`, two compatibility placeholders, and `STATE_IS_SHARDED`; 4 runtime args per core covering output address, state address, `pair_start`, and `num_pairs`.

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

Output is always written to DRAM because it feeds into subsequent Python-side operations (RMS norm, SiLU gating, output projection). The state destination depends on which L1 state configuration is active.

## Per-Pair Write Loop

The writer processes pairs in the same order as the reader and compute kernels, using `pair_start` to determine the global pair index for tile addressing:

```cpp
for (uint32_t pair = 0; pair < num_pairs; pair++) {
    uint32_t p = pair_start + pair;
    uint32_t out_tile_base = p * Vt;

    cb_wait_front(cb_out,       Vt);
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
    cb_pop_front(cb_out,       Vt);
    cb_pop_front(cb_state_out, state_tiles);
}
```

### Output Layout

The output tensor has shape [num_pairs, 1, Dv] in a sequential per-pair tile layout. Each pair writes Vt=4 tiles starting at tile index `p * Vt`. For pair 0, tiles 0-3; for pair 1, tiles 4-7; and so on. With `num_pairs=384` and `Vt=4`, the output contains 1536 tiles total (3 MB).

### State Writeback: DRAM / L1 Interleaved Path

When `STATE_IS_SHARDED == 0`, state is written via NOC:

```cpp
uint32_t sp = get_read_ptr(cb_state_out);
for (uint32_t s = 0; s < state_tiles; s++) {
    noc_async_write_tile(p * state_tiles + s, state_wr, sp);
    sp += tile_bytes;
}
```

Each pair writes 16 tiles (32 KB) to the state tensor at tile offset `p * state_tiles`. The `state_wr` address generator routes writes to DRAM or L1 based on the compile-time `state_is_dram` template parameter.

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

(here `pair` is the core-local pair index (0 to `num_pairs-1`), not the global pair index `p`)

## Barrier Strategy

Both the output write and state write (non-sharded path) are covered by a single `noc_async_write_barrier()` per pair:

```cpp
noc_async_write_barrier();
cb_pop_front(cb_out,       Vt);
cb_pop_front(cb_state_out, state_tiles);
```

The barrier ensures all NOC writes for the pair have completed before the CBs are freed. This matters because `cb_pop_front` makes CB space available for the next pair's data from the compute kernel — if the NOC write has not completed, the pipeline could overwrite data still being written to DRAM.

For the HEIGHT_SHARDED path, the L1 memcpy is synchronous, so the barrier covers only the output write. The state data is already committed to L1 by the time the barrier call executes.

## Write Volume Per Pair

| Destination | Tiles | Bytes | Method |
|-------------|-------|-------|--------|
| Output (DRAM) | Vt=4 | 8 KB | NOC write |
| State (DRAM path) | 16 | 32 KB | NOC write |
| State (L1 sharded path) | 16 | 32 KB | L1 memcpy |
| **Total per pair** | **20** | **40 KB** | |

For the full kernel (384 pairs across all cores), total write volume is:

- Output: $384 \times 8\text{ KB} = 3\text{ MB}$ (always to DRAM)
- State: $384 \times 32\text{ KB} = 12\text{ MB}$ (DRAM or L1 depending on configuration)

With the HEIGHT_SHARDED L1 path, the 12 MB state writeback becomes a local L1 operation, reducing DRAM write bandwidth by 12 MB per GDN layer per decode step.

---

**Next:** [Chapter 5 — Prefill TTFT Optimization](../ch5_prefill_ttft_optimization/index.md)
