# 3.2 DRAM and NOC Backpressure

DRAM-related hangs are distinct from L1 corruption (Section 3.1) in that the memory contents are typically correct -- the hang arises from **bandwidth contention, bank collisions, and backpressure propagation** through the NOC. These hangs are particularly dangerous because they can be non-deterministic: the same workload may run successfully at low core utilization but hang reliably at full grid occupancy when DRAM bandwidth becomes the bottleneck.

Reference files: `tt_metal/hw/inc/internal/debug/sanitize.h` (lines 194-207, `debug_valid_dram_addr`), `tests/tt_metal/tt_metal/test_kernels/dataflow/dram_arbiter_hang.cpp`, `tt_metal/hw/inc/hostdev/dev_msgs.h` (`core_info_msg_t`)

---

## DRAM Architecture Background

### DRAM Channels and Banks

Each Tenstorrent architecture has a different number of DRAM channels and per-channel capacity:

| Architecture | DRAM Channels | Per-Channel Size | Total DRAM | Banks per Channel |
|---|---|---|---|---|
| Wormhole (WH) | 12 (6 physical x 2 sub-channels) | ~1 GB | ~12 GB | Multiple |
| Blackhole (BH) | 8 | ~4 GB | ~32 GB | Multiple |
| Quasar | 8+ | Varies | Varies | Multiple |

**Interleaved buffers** distribute data across DRAM channels by page: page 0 goes to channel 0, page 1 to channel 1, etc. (modulo the number of channels). This maximizes bandwidth when all channels are equally loaded, but creates **hotspot** problems when the access pattern is not uniformly distributed.

### DRAM Address Validation

The watcher validates DRAM addresses against per-core bounds stored in the mailbox:

```c++
// tt_metal/hw/inc/internal/debug/sanitize.h
inline uint16_t debug_valid_dram_addr(uint64_t addr, uint64_t len) {
    if (addr + len <= addr) {
        return DebugSanitizeNocAddrZeroLength;
    }
    core_info_msg_t tt_l1_ptr* core_info = GET_MAILBOX_ADDRESS_DEV(core_info);
    if (addr < core_info->noc_dram_addr_base) {
        return DebugSanitizeNocAddrUnderflow;
    }
    if (addr + len > core_info->noc_dram_addr_end) {
        return DebugSanitizeNocAddrOverflow;
    }
    return DebugSanitizeOK;
}
```

The bounds `noc_dram_addr_base` and `noc_dram_addr_end` are set by the host during device initialization and stored in `core_info_msg_t` within the mailbox. If these values are corrupted (see Section 3.1.8 and Hang Cause 3.2.10 below), DRAM address validation becomes unreliable.

---

## Hang Cause 3.2.1: DRAM Bandwidth Saturation Stall

### Symptom

Multiple cores hang simultaneously at `NRBW` (NOC read barrier wait) or `NWBW` (NOC write barrier wait). The hang is non-deterministic -- it may not reproduce at lower core counts or with different data layouts. No NOC sanitization violations are detected.

### Root Cause

When many cores simultaneously issue NOC reads from the same DRAM channel, the channel's bandwidth is saturated. The backpressure chain:

1. DRAM channel is saturated -- cannot accept more read requests.
2. The NOC slave interface on the DRAM core stops accepting new requests.
3. The NOC router buffers fill up.
4. The issuing core's NOC master interface cannot dispatch new transactions.
5. The NOC command buffer on the issuing core fills up (see also 3.2.4).
6. The kernel's `noc_async_read_barrier()` spins forever because no more transactions can complete while the pipeline is stalled.

In extreme cases, this can create a **circular dependency**: core A is waiting for a DRAM read to complete, but the DRAM channel is blocked by outstanding writes from core B, and core B is waiting for a DRAM read that is blocked behind core A's writes. This is a true deadlock, not just a performance bottleneck.

### Diagnosis Steps

1. Check whether the hang involves multiple cores and whether they all target the same DRAM channel(s).
2. Examine the interleaved buffer layout to determine the access pattern.
3. Count the number of outstanding NOC transactions per core:
   ```
   reads_issued  = NOC_STATUS(NIU_MST_RD_REQ_STARTED)
   reads_complete = NOC_STATUS(NIU_MST_RD_RESP_RECEIVED)
   outstanding = reads_issued - reads_complete
   ```
4. If `outstanding` is at the maximum for all cores targeting the same DRAM channel, bandwidth saturation is the cause.

**Buggy pattern:**
```c++
// All cores read from the same DRAM bank 0
uint64_t dram_noc_addr = get_noc_addr(dram_bank_0_x, dram_bank_0_y, offset);
noc_async_read(dram_noc_addr, local_addr, transfer_size);
noc_async_read_barrier();
```

**Corrected pattern:**
```c++
// Use interleaved addressing to spread reads across all DRAM banks
uint64_t dram_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, offset);
noc_async_read(dram_noc_addr, local_addr, transfer_size);
noc_async_read_barrier();
```

### Fix

- Reduce the number of concurrent cores accessing the same DRAM channel.
- Use larger transfer sizes (fewer, bigger DMA transactions) to amortize per-transaction overhead.
- Redistribute the data layout to balance load across all DRAM channels.

### Prevention

- Design interleaved buffer layouts with uniform channel utilization.
- Profile DRAM bandwidth usage before scaling to full grid.

---

## Hang Cause 3.2.2: DRAM Bank Collision Stalls

### Symptom

Similar to 3.2.1 (multiple cores stuck at NOC barriers), but the hang pattern correlates with specific page addresses rather than channel numbers. Even with good channel distribution, specific pages experience much higher latency.

### Root Cause

Within a single DRAM channel, the memory controller organizes data into banks. When multiple requests target the same bank, they are serialized by the bank arbiter. **Bank collisions** occur when many cores request pages that map to the same physical bank. The lower bits of the local DRAM address typically select the bank, so buffers allocated at specific alignments may consistently map to the same bank.

On the L1 side, Tenstorrent devices also use banked L1 for NOC access. When multiple remote NOC writes target the same L1 address range within the same bank, the L1 bank arbiter serializes them, creating backpressure that stalls the sending cores.

### Diagnosis Steps

1. Analyze the DRAM addresses accessed by each core -- look for patterns in the lower address bits.
2. Compare the bank-address bits across all concurrent requests.
3. For L1 bank collisions, check if multiple remote cores are writing to the same L1 semaphore simultaneously.

### Fix

- Adjust buffer allocation alignment to distribute across banks.
- Add padding to buffer base addresses to break bank-collision patterns.
- For L1 semaphore updates, ensure semaphore addresses are spread across L1 banks.

### Prevention

- Use page sizes that are multiples of the bank count times the bank alignment.
- Avoid allocating large buffers at power-of-2 alignments that map to the same bank.

---

## Hang Cause 3.2.3: NOC Backpressure Propagation Leading to Write Barrier Hang

### Symptom

A core hangs at `NWBW` (NOC write barrier wait). The `noc_nonposted_writes_num_issued` counter is greater than `noc_nonposted_writes_acked`. No new writes can be issued because the NOC command buffer is full.

### Root Cause

NOC non-posted writes require an acknowledgment from the target. When the target is a DRAM channel under heavy load, the acknowledgment is delayed:

```c++
// Simplified write barrier logic (see Ch2 Section 04 for full details)
while (noc_nonposted_writes_acked < noc_nonposted_writes_num_issued) {
    // spin
}
```

Under DRAM backpressure, write acknowledgments are delayed because:
1. The DRAM controller prioritizes in-flight reads (read-modify-write bank conflicts).
2. Write data must travel through the NOC, which is congested with read response data.
3. The NOC arbiter may starve write traffic in favor of higher-priority read completions.

### Diagnosis Steps

1. Read the NOC status registers on the hung core:
   ```
   writes_issued = NOC_STATUS(NIU_MST_NONPOSTED_WR_REQ_STARTED)
   writes_acked  = NOC_STATUS(NIU_MST_WR_ACK_RECEIVED)
   ```
2. If `writes_issued > writes_acked`, pending writes are not being acknowledged.
3. Identify the target of the pending writes (decode the NOC address from command buffer registers).
4. Check the DRAM channel's incoming request counters for congestion.

### Fix

- Insert explicit write barriers before issuing reads to the same DRAM channel.
- Use the mcast path reservation workaround documented in [Chapter 2, Section 04](../ch02_kernel_and_noc_hangs/04_noc_barrier_and_semaphore_hangs.md) if multicast writes are involved.

### Prevention

- Batch writes to the same DRAM channel and drain them with a barrier before switching to reads.
- Use posted writes where acknowledgment is not needed.

---

## Hang Cause 3.2.4: NOC Command Buffer Stalls

### Symptom

A core hangs attempting to issue a new NOC transaction (it spins waiting for a free command buffer slot), even though it has not yet called any barrier. The core is stuck before the `noc_async_read` or `noc_async_write` call returns.

### Root Cause

Each NOC endpoint has a limited number of command buffer slots (typically 4 per NOC). When all slots are occupied by transactions that have been issued but not yet completed (because the remote endpoint is not responding due to backpressure), the issuing core cannot submit new transactions.

The NOC API functions wait for a free command buffer slot before submitting:

```cpp
// Simplified NOC command buffer wait
while (NOC_CMD_BUF_READ_REG(noc_id, cmd_buf, NOC_CMD_CTRL) != NOC_CTRL_STATUS_READY) {
    // spin -- the command buffer slot is still occupied
}
```

If the DRAM controller or a remote L1 endpoint is saturated, the response that frees a command buffer slot never arrives, and this spin-loop becomes a hang.

This is a *cascading failure*: DRAM backpressure causes NOC endpoint backpressure, which causes command buffer fullness, which blocks the kernel from issuing any further transactions (including unrelated ones to different targets).

### Diagnosis Steps

1. The watcher may show waypoints like `NTW` (NOC transaction wait) or the core may be stuck inside the NOC API without any waypoint update.
2. Read the NOC command buffer status registers to determine which slots are occupied and what their target addresses are.
3. If the occupied slots target a DRAM bank, check for DRAM saturation (scenario 3.2.1).
4. If the occupied slots target another L1 core, check for L1 bank collisions (scenario 3.2.2) or a hung receiver core.

**Buggy code:**
```cpp
// Issues N reads without any barrier -- can fill all command buffers
for (uint32_t i = 0; i < num_pages; i++) {
    noc_async_read(src_noc_addrs[i], local_addrs[i], page_size);
}
noc_async_read_barrier();  // single barrier at the end
```

**Corrected code:**
```cpp
// Issue reads in batches with intermediate barriers
constexpr uint32_t batch_size = 8;
for (uint32_t i = 0; i < num_pages; i++) {
    noc_async_read(src_noc_addrs[i], local_addrs[i], page_size);
    if ((i + 1) % batch_size == 0) {
        noc_async_read_barrier();  // drain before issuing more
    }
}
noc_async_read_barrier();  // final drain
```

### Fix

Reduce the number of outstanding NOC transactions by adding intermediate barriers.

### Prevention

- Keep the number of outstanding NOC transactions per core below the command buffer depth.
- Use intermediate barriers to prevent command buffer exhaustion.

---

## Hang Cause 3.2.5: Posted vs. Non-Posted Write Backpressure Asymmetry

### Symptom

A core stuck at `NWBW` has a mismatch between `noc_nonposted_writes_num_issued` and `noc_nonposted_writes_acked`, but posted write counters show all posted writes have been accepted. The core appears to be blocking only on non-posted write acknowledgments.

### Root Cause

The NOC supports both posted writes (fire-and-forget, no ack) and non-posted writes (requires ack). The write barrier waits only for non-posted write acks. Non-posted writes compete for DRAM bandwidth with posted writes. If a core issues many posted writes (which are accepted immediately into the NOC) alongside non-posted writes, the posted writes may consume DRAM bandwidth that the non-posted writes need to complete, delaying their acks indefinitely.

More subtly, if a kernel mixes posted and non-posted writes but uses the wrong barrier type, it may wait for acks that were never expected.

### Diagnosis Steps

1. Read both posted and non-posted write counters from NOC status registers.
2. Determine the write type mix in the kernel.
3. Check if posted writes are creating bandwidth contention with non-posted writes to the same DRAM bank.

### Fix

- Use consistent write types. If you need acks (for ordering guarantees), use non-posted writes exclusively.
- If using posted writes for performance, ensure they do not create enough backpressure to block non-posted writes.

### Prevention

- Do not mix posted and non-posted writes to the same DRAM bank in tight loops.
- Use write barriers between phases of different write types.

---

## Hang Cause 3.2.6: Interleaved DRAM Buffer Hotspot

### Symptom

The workload hangs under full grid utilization but works at reduced core counts. Profiling reveals that certain DRAM channels receive disproportionately more traffic. The hang disappears when buffer sizes are adjusted.

### Root Cause

Interleaved buffers distribute pages round-robin across DRAM channels. If many cores all access the same "hot" pages (e.g., the first page of a weight tensor), those pages all map to specific channels.

**Buggy pattern:**
```c++
// All 80 cores read the same weight page from DRAM -- channel 0 becomes the bottleneck
uint32_t weight_page_id = 0;  // always page 0
noc_async_read_tile(weight_page_id, weight_tensor, l1_addr);
noc_async_read_barrier();
```

**Corrected pattern:**
```c++
// Designate one core per column as the DRAM reader, then multicast to other cores
if (is_designated_reader) {
    noc_async_read_tile(weight_page_id, weight_tensor, l1_addr);
    noc_async_read_barrier();
    uint64_t mcast_addr = get_noc_multicast_addr(col_start_x, col_start_y,
                                                   col_end_x, col_end_y, l1_addr);
    noc_async_write_multicast(l1_addr, mcast_addr, tile_size, num_peers, false);
    noc_async_write_barrier();
}
noc_semaphore_wait(semaphore_addr, expected_val);
```

### Diagnosis Steps

1. Profile the DRAM access pattern: which pages are accessed by which cores, and when.
2. Map page indices to channels (page_index % num_channels).
3. Count the per-channel access frequency.
4. Look for broadcast patterns where many cores read the same page.

### Fix

- Use multicast for shared data instead of having each core independently read from DRAM.
- Replicate hot data across channels.
- Consider sharded layouts (`ShardedBufferType`) instead of interleaved for patterns with known locality.

### Prevention

- Audit interleaved buffer access patterns for hotspot potential.
- Use the profiler to visualize per-channel DRAM bandwidth utilization.

---

## Hang Cause 3.2.7: DRAM Arbiter Hang (Hardware Test Pattern)

### Symptom

The test kernel `dram_arbiter_hang.cpp` reproduces a specific DRAM arbiter contention pattern. In production workloads, this manifests as a hang during sustained DRAM reads without intervening barriers.

### Root Cause

The DRAM arbiter hang test exercises a pattern where a single core issues many DRAM reads in a tight loop without waiting for any to complete:

```c++
// tests/tt_metal/tt_metal/test_kernels/dataflow/dram_arbiter_hang.cpp
void kernel_main() {
    uint32_t bank_base_address = get_arg_val<uint32_t>(0);
    uint32_t page_size = get_arg_val<uint32_t>(1);
    uint32_t dst_l1_addr = get_arg_val<uint32_t>(2);
    uint32_t num_pages_to_read = get_arg_val<uint32_t>(3);
    uint32_t num_iterations = get_arg_val<uint32_t>(4);

    constexpr auto s_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(s_args, bank_base_address, page_size);

    for (uint32_t iter_idx = 0; iter_idx < num_iterations; iter_idx++) {
        for (uint32_t page_id = 0; page_id < num_pages_to_read; page_id++) {
            noc_async_read_tile(page_id, s, dst_l1_addr);
        }
    }
    noc_async_read_barrier();
}
```

This test is used to validate that the DRAM arbiter does not deadlock under maximum load. When the test fails, it indicates a hardware-level DRAM arbiter bug or a firmware issue.

### Diagnosis Steps

1. Run the test with varying `num_pages_to_read` and `num_iterations`.
2. Check NOC status registers to see whether reads were issued but not completed.
3. Compare results across architectures -- the DRAM arbiter behavior differs between WH, BH, and Quasar.

### Fix

In production kernels, avoid the dangerous pattern by inserting periodic barriers and using different L1 destination addresses for concurrent reads.

### Prevention

- Never issue unbounded reads without intermediate barriers.
- Limit the number of outstanding NOC read transactions per core.

---

## Hang Cause 3.2.8: DRAM Address Bounds Violation

### Symptom

The watcher reports `DebugSanitizeNocAddrUnderflow` or `DebugSanitizeNocAddrOverflow` with a DRAM target core. The core enters a deliberate hang.

### Root Cause

A NOC transaction targets a DRAM core but the local DRAM address is outside the valid range defined by `core_info_msg_t.noc_dram_addr_base` and `core_info_msg_t.noc_dram_addr_end`. Common causes:
1. An interleaved buffer address calculation has an off-by-one error in the bank offset.
2. A page offset is computed with the wrong page size.
3. The DRAM address was computed for one architecture's memory layout but runs on a different architecture.

### Diagnosis Steps

1. The watcher log shows the exact NOC address, L1 address, and length.
2. Decode the NOC address to extract the DRAM core coordinates and local address.
3. Compare the local address against `noc_dram_addr_base` and `noc_dram_addr_end`.
4. Trace back to the host-side buffer allocation and kernel-side address calculation.

### Fix

Correct the address calculation. Use the `TensorAccessor` API for DRAM access, which handles bank calculations automatically.

### Prevention

- Validate DRAM addresses against per-architecture bounds in host-side test code.

---

## Hang Cause 3.2.9: Blackhole Inline Write Backpressure Hang

### Symptom

On Blackhole only, a core hangs when attempting to issue an inline write to a DRAM address. The watcher reports `DebugSanitizeInlineWriteDramUnsupported` (return code 11).

### Root Cause

On Blackhole, inline writes and atomics require all 4 memory ports to accept the transaction simultaneously. If one port has backpressure but another does not, the transaction cannot complete. This is documented in `dev_mem_map.h`:

```
// On Blackhole issuing inline writes and atomics requires all 4 memory
// ports to accept the transaction at the same time. If one port on the
// recipient has no back-pressure then the transaction will hang because
// there is no mechanism to allow one memory port to move ahead of another.
```

The firmware workaround emulates inline writes via a local L1 staging buffer (`MEM_L1_INLINE_BASE`). Additionally, inline writes to DRAM are explicitly prohibited. The sanitization check catches this:

```cpp
// From sanitize.h
void debug_throw_on_dram_addr(uint8_t noc_id, uint64_t addr, uint32_t len) {
    uint8_t x = (uint8_t)NOC_UNICAST_ADDR_X(addr);
    uint8_t y = (uint8_t)NOC_UNICAST_ADDR_Y(addr);
    bool is_virtual_coord = true;
    AddressableCoreType core_type = get_core_type(noc_id, x, y, is_virtual_coord);
    if (core_type == AddressableCoreType::DRAM) {
        debug_sanitize_post_addr_and_hang(
            noc_id, addr, 0, len,
            DEBUG_SANITIZE_NOC_UNICAST, DEBUG_SANITIZE_NOC_WRITE,
            DEBUG_SANITIZE_NOC_TARGET,
            DebugSanitizeInlineWriteDramUnsupported);
    }
}
```

### Diagnosis Steps

1. Check if the hang is BH-specific. If it works on WH but hangs on BH, suspect inline write backpressure.
2. Check the watcher for `DebugSanitizeInlineWriteDramUnsupported`.
3. Search the kernel for `noc_inline_dw_write` or similar inline write primitives targeting DRAM.

### Fix

Replace inline writes to DRAM with `noc_async_write` from a local L1 buffer.

### Prevention

- Use the `DEBUG_SANITIZE_NO_DRAM_ADDR` macro (compiled when watcher is enabled) to catch inline DRAM writes.
- Avoid inline writes in architecture-generic kernel code; use conditional compilation for BH.

---

## Hang Cause 3.2.10: Read-Write Ordering Violations Under DRAM Backpressure

### Symptom

Data corruption leading to a secondary hang. A core reads stale data from DRAM because a preceding write has not yet been committed. The stale data is used as an address or control value, causing a subsequent NOC operation to fail.

### Root Cause

NOC writes to DRAM may be buffered in the DRAM controller's write queue. A subsequent read may return old data if:
1. The write and read use different NOC instances (NOC_0 vs NOC_1).
2. The write is posted (no acknowledgment) and the read is issued before the write completes.
3. On BH, inline writes have special ordering concerns due to the 4-port memory interface.

### Diagnosis Steps

1. Look for patterns where the same DRAM address is written and then read.
2. Check whether a write barrier is issued between the write and subsequent read.
3. On BH, check whether inline writes are used to DRAM.

### Fix

- Always issue `noc_async_write_barrier()` between a DRAM write and a subsequent read to the same address.
- Ensure writes and reads to the same address use the same NOC instance.
- On BH, never use inline writes to DRAM targets.

### Prevention

- Enforce write-before-read barriers in all DRAM communication patterns.

---

## Hang Cause 3.2.11: Corrupted DRAM Address Bounds in core_info_msg_t

### Symptom

The watcher reports DRAM address violations on transactions that should be valid. Alternatively, the watcher does not catch invalid DRAM accesses because the bounds have been widened by corruption.

### Root Cause

The `core_info_msg_t` structure in the mailbox contains `noc_dram_addr_base` and `noc_dram_addr_end`. If the mailbox region is corrupted (see Section 3.1.8), these bounds may become incorrect:

- **Too narrow:** Valid DRAM accesses trigger false positive sanitization violations.
- **Too wide:** Invalid DRAM accesses pass validation and cause hardware-level stalls.

```c++
// tt_metal/hw/inc/hostdev/dev_msgs.h
struct core_info_msg_t {
    volatile uint64_t noc_pcie_addr_base;
    volatile uint64_t noc_pcie_addr_end;
    volatile uint64_t noc_dram_addr_base;  // if corrupted, DRAM bounds checking fails
    volatile uint64_t noc_dram_addr_end;   // if corrupted, DRAM bounds checking fails
    // ... other fields ...
};
```

### Diagnosis Steps

1. Read the `core_info_msg_t` from the mailbox and verify `noc_dram_addr_base` and `noc_dram_addr_end` match expected values.
2. If they are wrong, look for the root cause of mailbox corruption (Section 3.1.8).

### Fix

Fix the underlying mailbox corruption. If the corruption source cannot be identified, reinitialize the core.

### Prevention

- Enable watcher mailbox protection checks.
- Validate `core_info_msg_t` values periodically during long-running workloads.

---

## Summary Table

| ID | Hang Cause | Key Indicator | Deterministic? | Architecture |
|----|-----------|---------------|----------------|--------------|
| 3.2.1 | DRAM bandwidth saturation | Multiple cores at `NRBW`/`NWBW`, high outstanding tx count | No (load-dependent) | All |
| 3.2.2 | DRAM bank collision stalls | Latency variance correlated with address alignment | No | All |
| 3.2.3 | Write barrier hang under backpressure | `writes_issued > writes_acked`, congested DRAM channel | No | All |
| 3.2.4 | NOC command buffer stalls | Core stuck before barrier, all cmd buffer slots occupied | No | All |
| 3.2.5 | Posted vs. non-posted write asymmetry | Non-posted acks delayed by posted write bandwidth | No | All |
| 3.2.6 | Interleaved DRAM hotspot | Asymmetric per-channel traffic | No (workload-dependent) | All |
| 3.2.7 | DRAM arbiter hang (test pattern) | Unbounded reads without barrier | Yes (pattern-specific) | All |
| 3.2.8 | DRAM address bounds violation | `DebugSanitizeNocAddr{Underflow,Overflow}` on DRAM core | Yes | All |
| 3.2.9 | BH inline write backpressure | `DebugSanitizeInlineWriteDramUnsupported` (11) | Yes | Blackhole only |
| 3.2.10 | Read-write ordering under backpressure | Stale data from DRAM | No | All (worst on BH) |
| 3.2.11 | Corrupted DRAM bounds in mailbox | Wrong `noc_dram_addr_base/end` | Depends on corruption | All |

---

**Previous:** [`01_l1_memory_corruption_and_overflow.md`](./01_l1_memory_corruption_and_overflow.md)
**Next:** [`03_alignment_and_tile_size_mismatches.md`](./03_alignment_and_tile_size_mismatches.md)
