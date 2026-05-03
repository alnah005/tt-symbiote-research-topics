# Circular Buffer Deadlocks

Circular buffer (CB) deadlocks are the single most common source of user-visible hangs on Tenstorrent hardware. The CB model is straightforward in principle -- a producer pushes tiles, a consumer pops them -- but the API has several non-obvious contract requirements that, when violated, cause hangs rather than errors. This section documents the CB data structures, the exact code paths that spin, every known failure mode, and the remote CB variants that introduce additional complexity.

**Prerequisites:** [Chapter 1, `02_blocking_primitives_taxonomy.md`](../ch01_anatomy_of_a_hang/02_blocking_primitives_taxonomy.md) (CRBW/CWFW blocking primitives), familiarity with the Tensix core data flow model (BRISC reader, TRISC compute, NCRISC writer). The basic CB spin-loop code and exit conditions are documented in Chapter 1; this section focuses on the specific failure scenarios that make those loops infinite.

Reference files: `tt_metal/hw/inc/api/dataflow/dataflow_api.h` (lines 200-460), `tt_metal/hw/inc/internal/circular_buffer_interface.h`

---

## The `CBInterface` Data Structure

Every circular buffer on the device is managed through a `CBInterface` structure stored in the `cb_interface` array (see Chapter 1 for the basic structure). The key fields relevant to hang diagnosis:

```c++
// tt_metal/hw/inc/internal/circular_buffer_interface.h
struct LocalCBInterface {
    uint32_t fifo_size;          // Total CB size in bytes
    uint32_t fifo_limit;         // Upper bound address (inclusive)
    uint32_t fifo_page_size;     // Size of one tile/page in bytes
    uint32_t fifo_num_pages;     // Total capacity in tiles (fifo_size / fifo_page_size)

    uint32_t fifo_rd_ptr;        // Consumer read pointer (byte address)
    uint32_t fifo_wr_ptr;        // Producer write pointer (byte address)

    union {
        uint32_t tiles_acked_received_init;  // Zeroed during init
        struct {
            uint16_t tiles_acked;    // Consumer pop counter (lower 16 bits)
            uint16_t tiles_received; // Producer push counter (upper 16 bits)
        };
    };

    uint32_t fifo_wr_tile_ptr;   // Used by packer for in-order packing
};
```

Key observations:

- **`tiles_acked` and `tiles_received` are `uint16_t` values packed into a single `uint32_t`.** Both counters use wrapping arithmetic: `tiles_received - tiles_acked` gives the number of occupied tiles via unsigned `uint16_t` wraparound. The maximum in-flight tile count is 65535.
- **The TRISC Pack core updates `tiles_acked` as a 16-bit value** in the `llk_pop_tiles` path, which is why `cb_reserve_back` reads it with `(uint16_t)reg_read(...)`.
- **Important asymmetry:** `cb_reserve_back` calls `invalidate_l1_cache()` in its spin loop because it reads `pages_acked` which is updated by a different core. `cb_wait_front` does **not** call `invalidate_l1_cache()` -- it reads `pages_received` via `reg_read()`, which bypasses the L1 cache. This is a performance optimization, not a bug.

### The `cb_addr_shift` Mechanism

Compute cores (TRISCs) and data-movement cores (BRISC, NCRISC) interpret CB addresses differently:

```c++
#if defined(COMPILE_FOR_TRISC)
constexpr uint32_t cb_addr_shift = CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT;  // typically 4
#else
constexpr uint32_t cb_addr_shift = 0;
#endif
```

On data-movement cores, `cb_addr_shift` is 0 -- addresses are byte addresses. On compute cores, the shift transforms the byte address. **If a kernel is compiled for the wrong core type** (e.g., a compute kernel running on BRISC), the address shift will be wrong, and the CB pointers will be misinterpreted, leading to hangs.

---

## The Producer-Consumer Model

The CB protocol has four primary operations:

| Operation | Side | Effect | Waypoint |
|-----------|------|--------|----------|
| `cb_reserve_back(cb_id, n)` | Producer | Block until `n` pages of free space are available | `CRBW` |
| `cb_push_back(cb_id, n)` | Producer | Advance write pointer by `n` pages; increment `tiles_received` | -- |
| `cb_wait_front(cb_id, n)` | Consumer | Block until `n` pages are available to read | `CWFW` |
| `cb_pop_front(cb_id, n)` | Consumer | Advance read pointer by `n` pages; increment `tiles_acked` | -- |

The standard usage pattern:

```
Producer:                          Consumer:
  cb_reserve_back(cb, N)             cb_wait_front(cb, N)
  // write N tiles to CB             // read N tiles from CB
  cb_push_back(cb, N)                cb_pop_front(cb, N)
```

Both `cb_reserve_back` and `cb_wait_front` are blocking spin loops. The push and pop operations are non-blocking.

---

## Hang Cause 2.2.1: Consumer Never Pops (Producer Stuck at `CRBW`)

### Symptom

The producer core (typically BRISC or NCRISC) is stuck at waypoint `CRBW`. The consumer core (typically TRISC0 for compute input, or NCRISC for writer output) shows either waypoint `D` (already done) or its own wait waypoint.

### Root Cause

`cb_reserve_back` spins until there is enough free space. The free space formula is: `free_space = fifo_num_pages - (pages_received - pages_acked)`. If the consumer never calls `cb_pop_front` (or calls it with fewer tiles than the producer expects), `pages_acked` never increases and the producer spins forever.

**Buggy pattern (host-side setup):**
```c++
// WRONG: Producer pushes 1024 tiles, but consumer kernel only pops 512
// Host-side:
auto reader_kernel = CreateKernel(..., {.num_tiles = 1024});
auto compute_kernel = CreateKernel(..., {.num_tiles = 512});  // MISMATCH
```

**Corrected pattern:**
```c++
// CORRECT: Both use the same tile count
uint32_t total_tiles = 1024;
auto reader_kernel = CreateKernel(..., {.num_tiles = total_tiles});
auto compute_kernel = CreateKernel(..., {.num_tiles = total_tiles});
```

### Diagnosis Steps

1. Identify the producer core at `CRBW` and the CB index.
2. Read `pages_received` and `pages_acked` for that CB.
3. Compute `free_space = fifo_num_pages - (pages_received - pages_acked)`.
4. If `free_space < num_pages` and `pages_acked` is not changing, the consumer has stopped popping.
5. Check the consumer core's waypoint:
   - If `D`: consumer finished before popping enough tiles. **This is the most common case** -- a loop count mismatch between producer and consumer.
   - If `CWFW`: consumer is itself blocked waiting for input from yet another producer (cascade).

### Fix

Match the total number of tiles pushed by the producer to the total number popped by the consumer across the entire kernel execution.

### Prevention

- Calculate total tile counts for each CB at design time and verify that producers and consumers agree.
- Use the non-blocking `cb_pages_reservable_at_back` to add timeout-based debugging (see Debugging Aids below).

---

## Hang Cause 2.2.2: Producer Never Pushes (Consumer Stuck at `CWFW`)

### Symptom

The consumer core (typically TRISC0/Unpack or NCRISC/Writer) is stuck at waypoint `CWFW`. The producer core shows `D` (done), `NRBW` (stuck at a NOC read barrier), or its own `CRBW` (cascade from another CB).

### Root Cause

`cb_wait_front` spins until enough tiles are available. If the producer never calls `cb_push_back` (because it terminated early, or is itself hung), `pages_received` remains below `num_pages` and the consumer spins forever.

### Diagnosis Steps

1. Identify the consumer core at `CWFW` and the CB index.
2. Read `pages_received` and `pages_acked` for that CB.
3. If `(pages_received - pages_acked) < num_pages` and `pages_received` is not changing, the producer has stopped pushing.
4. Check the producer core:
   - If `D`: producer finished before pushing enough tiles (loop count mismatch).
   - If `NRBW`: producer is stuck at a NOC read barrier (compound NOC + CB hang).
   - If `CRBW` on a different CB: cascade -- the producer is itself starved of output buffer space.

### Fix

Ensure the producer pushes at least as many tiles as the consumer expects. When the producer is a data-movement kernel that reads from DRAM, verify that the total number of NOC reads matches the number of tiles the consumer expects.

### Prevention

Same as 2.2.1. Derive tile counts from the same source for both producer and consumer.

---

## Hang Cause 2.2.3: The Cumulative-Total Requirement Violation

### Symptom

The consumer hangs at `CWFW` partway through execution. The producer is pushing tiles correctly, but the consumer appears to stop accepting them after a certain number of iterations.

### Root Cause

The `cb_wait_front` API has a non-obvious contract requirement:

> "In case multiple calls of `cb_wait_front(n)` are issued without a paired `cb_pop_front()` call, `n` is expected to be incremented by the user to be equal to a cumulative total of tiles."

The `pages_received` value computed inside `cb_wait_front` is `((uint16_t)reg_read(pages_received_ptr)) - pages_acked`, where `pages_acked` is captured **once** at the beginning of the call and never updated during the spin loop. This means `pages_received` represents the total tiles available since the last `cb_pop_front`, not since the last `cb_wait_front`.

**Buggy pattern:**
```c++
// WRONG: Non-cumulative cb_wait_front calls
cb_wait_front(cb, 8);  // Wait for 8 tiles -- OK
// process first 8 tiles
cb_wait_front(cb, 8);  // Wait for 8 MORE tiles -- WRONG!
// This actually waits for 8 total, which is already satisfied!
// The second batch may not have arrived yet.
cb_pop_front(cb, 16);
```

**Corrected pattern (cumulative totals):**
```c++
// CORRECT: Cumulative totals
cb_wait_front(cb, 8);   // Wait for 8 tiles
// process first 8 tiles
cb_wait_front(cb, 16);  // Wait for 16 TOTAL tiles
// process next 8 tiles
cb_pop_front(cb, 16);
```

**Alternative corrected pattern (wait-pop pairs):**
```c++
// CORRECT: Wait-pop pairs
cb_wait_front(cb, 8);   // Wait for 8 tiles
// process first 8 tiles
cb_pop_front(cb, 8);    // Reset acked counter

cb_wait_front(cb, 8);   // Wait for 8 more (acked was reset)
// process next 8 tiles
cb_pop_front(cb, 8);
```

### Diagnosis Steps

1. Look for a pattern where `cb_wait_front` is called multiple times with the same argument between `cb_pop_front` calls.
2. The hang may not occur at the first invocation but after many iterations, when the accumulated error compounds.

### Fix

Change all `cb_wait_front` calls between `cb_pop_front` calls to use cumulative totals, or insert `cb_pop_front` between individual waits.

### Prevention

- Always use the cumulative-total pattern or the wait-pop pair pattern.
- Consider wrapping `cb_wait_front` in a helper that automatically tracks the cumulative count.

---

## Hang Cause 2.2.4: CB Size Not Evenly Divisible by Tile Count

### Symptom

The hang occurs after a specific, deterministic number of iterations. The producer or consumer hangs at `CRBW` or `CWFW` respectively.

### Root Cause

The `fifo_wr_ptr` and `fifo_rd_ptr` wrap around when they reach `fifo_limit`:

```c++
// In cb_push_back (dataflow_api.h):
get_local_cb_interface(operand).fifo_wr_ptr += num_words;
ASSERT(get_local_cb_interface(operand).fifo_wr_ptr <=
       get_local_cb_interface(operand).fifo_limit);
if (get_local_cb_interface(operand).fifo_wr_ptr ==
    get_local_cb_interface(operand).fifo_limit) {
    get_local_cb_interface(operand).fifo_wr_ptr -=
        get_local_cb_interface(operand).fifo_size;
}
```

The wrap condition checks for **exact equality** with `fifo_limit`. If `num_pages * fifo_page_size` does not evenly divide `fifo_size`, the write pointer overshoots `fifo_limit`, the equality check fails, and the pointer is never reset. On subsequent iterations, the pointer advances beyond the allocated memory region, corrupting adjacent L1 data.

Example: CB has `fifo_num_pages = 10` and the kernel uses `num_pages = 3`:
- Iteration 1: push 3, ptr at 3 * page_size
- Iteration 2: push 3, ptr at 6 * page_size
- Iteration 3: push 3, ptr at 9 * page_size
- Iteration 4: push 3, ptr at 12 * page_size -- but fifo_limit = 10 * page_size!

The ASSERT macro catches this when assertions are enabled. Without assertions (production builds), the behavior is silent pointer corruption.

### Diagnosis Steps

1. Deterministic hang at a specific iteration count is the key signal.
2. Check CB configuration: `fifo_size`, `fifo_page_size`, `fifo_num_pages`.
3. Calculate `fifo_num_pages % num_pages` -- if non-zero, this is the bug.
4. Enable watcher CB sanitization: `debug_valid_cb_addr` will detect out-of-bounds accesses.

### Fix

Adjust either the CB size or the tile count so that the tile count evenly divides the CB capacity. For example, if the CB has 10 pages and the kernel operates on 3 tiles at a time, change the CB to 9 or 12 pages.

### Prevention

- At program creation time, validate that `cb_size_in_tiles % num_tiles_per_call == 0` for every CB.
- Enable assertions during development to catch the ASSERT on pointer overshoot.

---

## Hang Cause 2.2.5: Mismatched Tile Counts Across Producers and Consumers

### Symptom

One kernel on a Tensix core hangs at `CRBW` or `CWFW` while its partner has already exited (waypoint `D`).

### Root Cause

Consider a three-kernel pipeline on a single Tensix:
- BRISC (reader): reads tiles from DRAM, pushes to CB_IN
- TRISC0/1/2 (compute): reads from CB_IN, computes, pushes to CB_OUT
- NCRISC (writer): reads from CB_OUT, writes to DRAM

If the reader pushes 1024 tiles but the compute kernel expects 2048, the compute kernel hangs at `CWFW` after consuming the 1024th tile. Meanwhile, the reader has exited (waypoint `D`). BRISC then hangs at `NTW` waiting for subordinates.

**Buggy pattern:**
```c++
// WRONG: Reader and compute disagree on tile count
// reader kernel:
for (uint32_t i = 0; i < 1024; i++) { cb_push_back(cb_in, 1); }
// compute kernel:
for (uint32_t i = 0; i < 2048; i++) { cb_wait_front(cb_in, 1); cb_pop_front(cb_in, 1); }
```

**Corrected pattern:**
```c++
// CORRECT: Both derive from the same total
uint32_t num_tiles = get_arg_val<uint32_t>(0);  // Same runtime arg for both
// reader kernel:
for (uint32_t i = 0; i < num_tiles; i++) { cb_push_back(cb_in, 1); }
// compute kernel:
for (uint32_t i = 0; i < num_tiles; i++) { cb_wait_front(cb_in, 1); cb_pop_front(cb_in, 1); }
```

### Diagnosis Steps

1. Identify which core is at `CWFW` or `CRBW`.
2. Check the partner core: if it is at `D` (done), the tile counts are mismatched.
3. Compare the total tile count in the kernel source code.

### Fix

Ensure that for every CB, the total tiles pushed equals the total tiles popped across the kernel's lifetime. Use runtime arguments to pass the tile count from a single source.

### Prevention

- Compute the total tile count as a function of the problem dimensions and verify consistency across all three kernels at program creation time.
- Use runtime arguments rather than hardcoding tile counts.

---

## Hang Cause 2.2.6: Cross-Core CB Deadlock (Cascade Pattern)

### Symptom

All five RISC-V cores on a Tensix are hung. The waypoints form a cascade:

```
TRISC0: CWFW (waiting for CB_IN tiles)
TRISC1: (stalled, no waypoint change -- blocked by TRISC0)
TRISC2: CRBW (waiting for CB_OUT space)
NCRISC: CWFW (waiting for CB_OUT tiles)
BRISC: NTW (waiting for subordinates)
```

### Root Cause

This is the "intra-Tensix CB pipeline" cascade pattern. The root cause is typically one of:

1. **BRISC (reader) hung at a NOC barrier** (`NRBW`): BRISC cannot push tiles to CB_IN because its NOC read from DRAM never completed. Without input tiles, the entire pipeline stalls downstream.

2. **NCRISC (writer) hung at a NOC barrier** (`NWBW`): NCRISC cannot pop tiles from CB_OUT because its NOC write to DRAM never completed. Without space in CB_OUT, TRISC2 stalls, and the stall cascades upstream.

3. **Compute pipeline deadlock**: TRISC0/1/2 have a circular dependency through the tile register acquire/commit protocol.

### Diagnosis Steps

1. Map the waypoints for all five cores.
2. Trace the CB dependency chain from the core with the most "upstream" waypoint.
3. The root cause is almost always at the **edges** of the pipeline (BRISC or NCRISC), not in the compute cores.

### Fix

Fix the NOC transaction or kernel logic at the root of the cascade.

### Prevention

- When debugging a multi-core hang, always trace to the root cause rather than fixing the first symptom you see. A `CWFW` on TRISC0 is almost never the root cause -- it is usually a consequence of a NOC or memory issue upstream.

---

## Hang Cause 2.2.7: `cb_push_back` Overflow (Silent Corruption Leading to Hang)

### Symptom

The hang occurs after apparently successful execution for some number of iterations. The symptoms are unpredictable: any core may hang at any waypoint, NOC sanitization violations appear at addresses that do not correspond to any programmed operation, or the next kernel invocation hangs immediately.

### Root Cause

If `cb_push_back` is called with more tiles than the CB can hold, the write pointer wraps and overwrites data the consumer has not yet read. More specifically, if the producer calls `cb_push_back(cb, n)` where `n` exceeds the current free space (without first calling `cb_reserve_back` to verify space), the `tiles_received` counter advances but the `fifo_wr_ptr` may wrap and overwrite live data.

The `cb_push_back` function is **non-blocking and has no safety check** -- it unconditionally advances the counter and pointer. The safety is supposed to come from the preceding `cb_reserve_back` call. If the producer skips `cb_reserve_back` or uses a smaller `n` in `cb_reserve_back` than in `cb_push_back`, the overflow is silent.

### Diagnosis Steps

1. Look for seemingly random hangs that follow successful execution
2. Check for inconsistencies between `cb_reserve_back` and `cb_push_back` arguments in the kernel source
3. Enable watcher CB sanitization to detect out-of-bounds memory accesses

### Fix

Ensure every `cb_push_back(cb, n)` is preceded by a `cb_reserve_back(cb, n)` with the **same `n` value**.

### Prevention

- Always pair `cb_reserve_back` and `cb_push_back` with identical tile counts.
- Enable CB sanitization during development to catch overflow-induced memory corruption.

---

## Hang Cause 2.2.8: Remote Circular Buffer Deadlocks

### Symptom

A core on one Tensix is stuck at `CRBW` or `CWFW` on a CB that is configured as a remote CB. The partner core is on a different Tensix, and its state may also be stuck.

### Root Cause

The `RemoteSenderCBInterface` and `RemoteReceiverCBInterface` extend the CB model across Tensix core boundaries. Remote CBs use L1 semaphores (written via NOC atomic increments) to coordinate between the sender and receiver cores.

```c++
// tt_metal/hw/inc/internal/circular_buffer_interface.h
struct RemoteSenderCBInterface {
    uint32_t config_ptr;
    uint32_t fifo_start_addr;
    uint32_t fifo_limit_page_aligned;
    uint32_t fifo_page_size;
    uint32_t fifo_wr_ptr;
    uint32_t receiver_noc_xy_ptr;      // Array of receiver NOC coordinates
    uint32_t aligned_pages_sent_ptr;   // Per-receiver pages_sent/pages_acked pairs
    uint32_t num_receivers;
};
```

Remote CBs introduce additional deadlock vectors:

1. **NOC-dependent counter updates**: Unlike local CBs where counter updates are L1 writes on the same core, remote CB counter updates require NOC transactions. If the NOC transaction fails, the counter is never updated.

2. **Per-receiver tracking**: The sender tracks `pages_sent` and `pages_acked` per receiver. If one receiver is slow or hung, the sender may fill the buffer and stall, blocking all other receivers. This is a fan-out deadlock that does not exist with local CBs.

3. **Setup ordering**: The `barrier_remote_cb_interface_setup` function on Blackhole (Hang Cause 2.1.11) ensures that remote CB interface configuration has landed before kernel execution. If this barrier fails, the remote core may read an uninitialized CB interface.

### Diagnosis Steps

1. Determine if the CB is a remote CB by checking the CB index: remote CBs typically start at `min_remote_cb_start_index` from the launch message.
2. Read the `RemoteSenderCBInterface` or `RemoteReceiverCBInterface` fields.
3. Check the NOC coordinates: do `receiver_noc_xy_ptr` and `sender_noc_x/y` point to valid, reachable cores?

### Fix

- If the NOC coordinates are wrong, fix the program configuration.
- If one receiver is hung, unblock the receiver first -- the sender's `CRBW` will resolve once the hung receiver pops tiles.

### Prevention

- Remote CBs add an entire NOC failure surface to CB operations. Use them only when cross-core data movement is necessary.
- On Blackhole, always ensure the `NABW` barrier completes before kernel execution begins.
- Validate NOC coordinates at program creation time.

---

## Non-Blocking Alternatives for Debugging

Both blocking CB operations have non-blocking counterparts useful for instrumentation:

```c++
bool cb_pages_reservable_at_back(int32_t operand, int32_t num_pages);
bool cb_pages_available_at_front(int32_t operand, int32_t num_pages);
```

These perform the same check without spinning. They can be used to implement custom timeout logic for debugging:

```c++
uint32_t timeout = 0;
while (!cb_pages_available_at_front(cb, N)) {
    if (++timeout > THRESHOLD) {
        DPRINT << "CB " << cb << " stuck: pages_received="
               << get_cb_tiles_received_ptr(cb)[0]
               << " pages_acked=" << get_cb_tiles_acked_ptr(cb)[0] << ENDL();
        break;
    }
}
```

---

## Summary: CB Deadlock Quick Reference

| Scenario | Stuck Waypoint | Stuck Core | Partner State | Root Cause |
|----------|---------------|------------|---------------|------------|
| 2.2.1 Consumer never pops | `CRBW` | Producer (BRISC/NCRISC) | `D` or `CWFW` | Loop count mismatch |
| 2.2.2 Producer never pushes | `CWFW` | Consumer (TRISC/NCRISC) | `D` or `NRBW` | Loop count mismatch or NOC hang |
| 2.2.3 Cumulative-total | `CWFW` | Consumer | Producer running | Non-cumulative `cb_wait_front` calls |
| 2.2.4 CB size divisibility | `CRBW` or `CWFW` | Either | Running or `D` | `fifo_num_pages % num_pages != 0` |
| 2.2.5 Tile count mismatch | `CRBW` or `CWFW` | Either | `D` | Producer/consumer tile count disagreement |
| 2.2.6 Cascade pattern | All five cores stuck | All | Cascaded waits | Root cause at pipeline edge (NOC) |
| 2.2.7 Push overflow | Varied | Varied | Varied | `cb_push_back` without `cb_reserve_back` |
| 2.2.8 Remote CB | `CRBW`/`CWFW`/`NSW` | Either core | Cross-Tensix partner stuck | NOC failure on counter update |

| CB Waypoint | Function | Meaning |
|-------------|----------|---------|
| `CRBW` | `cb_reserve_back` | Producer waiting for free space |
| `CRBD` | `cb_reserve_back` | Producer got space, loop exited |
| `CWFW` | `cb_wait_front` | Consumer waiting for tiles |
| `CWFD` | `cb_wait_front` | Consumer got tiles, loop exited |

> **Tip:** When a CB deadlock involves a cascade (2.2.6), always trace upstream to the pipeline edge. The root cause is almost never in the compute cores -- it is typically a NOC read failure on BRISC or a NOC write failure on NCRISC.

---

**Previous:** [`01_risc_synchronization_and_deadlocks.md`](./01_risc_synchronization_and_deadlocks.md) | **Next:** [`03_noc_address_sanitization_and_violations.md`](./03_noc_address_sanitization_and_violations.md)
