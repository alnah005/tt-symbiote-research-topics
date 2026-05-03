# NOC Barrier and Semaphore Hangs

NOC barriers and semaphores are the primary mechanisms for synchronizing data movement with computation. Barriers wait for outstanding NOC transactions to complete; semaphores wait for explicit signals from other cores. Both are spin loops that can hang indefinitely when the expected event never occurs. This section documents the barrier and semaphore hang causes, including a known hardware bug requiring a software workaround in the dispatch kernel, transaction ID barriers, and the critical write flush vs. barrier distinction.

**Prerequisites:** [Chapter 1, `02_blocking_primitives_taxonomy.md`](../ch01_anatomy_of_a_hang/02_blocking_primitives_taxonomy.md) (NRBW, NWBW, NSW, NSMW blocking primitives), [`03_noc_address_sanitization_and_violations.md`](./03_noc_address_sanitization_and_violations.md) (NOC address validation that precedes barrier execution).

Reference files: `tt_metal/hw/inc/api/dataflow/dataflow_api.h` (lines 1731-2504), `tt_metal/impl/dispatch/kernels/cq_dispatch.cpp`

---

## NOC Read Barrier Hang (`NRBW`)

`noc_async_read_barrier` spins until `NIU_MST_RD_RESP_RECEIVED == noc_reads_num_issued[noc]` (dedicated mode) or the equivalent per-RISC sum (dynamic mode). See [Chapter 1, `02_blocking_primitives_taxonomy.md` -- NRBW](../ch01_anatomy_of_a_hang/02_blocking_primitives_taxonomy.md#nrbw-noc_async_read_barrier----noc-read-barrier-wait) for the full code listing and exit conditions.

---

## Hang Cause 2.4.1: Read Barrier Hang from Invalid Address

### Symptom

A core is stuck at waypoint `NRBW`. The NOC hardware counter `NIU_MST_RD_RESP_RECEIVED` is less than `noc_reads_num_issued[noc]`. The difference indicates one or more reads that never received a response.

### Root Cause

If `noc_async_read` targets an address that does not correspond to a reachable endpoint -- wrong coordinates, address out of range, targeting a powered-down or harvested core -- the NOC hardware may drop the request. No response is generated, and the barrier waits forever.

Without watcher/sanitization enabled, this failure is completely silent.

### Diagnosis Steps

1. Read the NOC status registers: `NIU_MST_RD_RESP_RECEIVED` and `noc_reads_num_issued[noc]`.
2. Calculate the deficit: `noc_reads_num_issued - NIU_MST_RD_RESP_RECEIVED`. This is the number of missing responses.
3. If watcher was enabled, check the sanitize mailbox for an address violation.
4. If watcher was not enabled, dump the kernel source to identify the `noc_async_read` calls. Common patterns: incorrect bank ID in DRAM address calculation, stale buffer pointers, coordinate calculation errors.

### Fix

Correct the NOC address computation. Enable watcher to catch the violation at the source.

### Prevention

- Enable watcher during development.
- Use address construction helpers (`get_noc_addr`, `get_noc_addr_helper`) rather than manual address computation.

---

## Hang Cause 2.4.2: Read Barrier Hang from NOC Index Mismatch

### Symptom

Core stuck at `NRBW`. The NOC counter values appear normal for the checked NOC, but the reads were actually issued on a different NOC.

### Root Cause

Each NOC maintains independent hardware counters. If reads are issued on NOC 0 but the barrier checks NOC 1's counters, the barrier sees zero outstanding reads on NOC 1 and returns immediately, without waiting for the NOC 0 reads.

This is particularly dangerous in dynamic NOC mode on Blackhole, where BRISC and NCRISC can dynamically switch between NOCs. If a kernel uses dynamic NOC mode but the barrier hardcodes a NOC index, the counters may not align.

### Diagnosis Steps

1. Check the `noc_index` variable at the time of the barrier call.
2. Check the `noc_index` used in the preceding `noc_async_read` calls.
3. In dynamic NOC mode, check both NOCs' counters.

### Fix

Ensure the barrier uses the same NOC index as the reads.

### Prevention

Use the default parameter (`noc_async_read_barrier()` without arguments) to automatically use the current `noc_index`.

---

## Hang Cause 2.4.3: Missing Barrier Before Data Use

### Symptom

Data corruption or secondary hang. The kernel reads from an L1 buffer that was the target of a `noc_async_read`, but the data has not yet arrived. The kernel may push corrupted data into a CB, causing downstream compute hangs, or compute results based on stale data.

### Root Cause

`noc_async_read` is non-blocking -- it only issues the read request. The data is not guaranteed to be in L1 until `noc_async_read_barrier` completes. If the kernel accesses the destination buffer without an intervening barrier, it reads stale or partially-written data.

**Buggy pattern:**
```c++
// WRONG: Missing barrier between read and use
noc_async_read(src_noc_addr, dst_l1_addr, size);
cb_push_back(cb_out, 1);  // Data may not have arrived yet!
```

**Corrected pattern:**
```c++
// CORRECT: Barrier ensures data arrival
noc_async_read(src_noc_addr, dst_l1_addr, size);
noc_async_read_barrier();  // Wait for data
cb_push_back(cb_out, 1);   // Data is now in L1
```

### Diagnosis Steps

1. Enable `TT_METAL_NOC_DEBUG_DUMP=1` to automatically detect missing barriers (see Chapter 6).
2. Look for `noc_async_read` followed by CB or compute operations without an intervening `noc_async_read_barrier`.

### Fix

Insert `noc_async_read_barrier()` between the read issuance and the data use.

### Prevention

- Establish the pattern: every `noc_async_read` block must be followed by `noc_async_read_barrier` before any use of the destination data.
- Use `TT_METAL_NOC_DEBUG_DUMP=1` during development to catch `WRITE_FLUSH_BARRIER` and `READ_BARRIER` issues.

---

## NOC Write Barrier Hang (`NWBW`)

`noc_async_write_barrier` spins until `NIU_MST_WR_ACK_RECEIVED == noc_nonposted_writes_acked[noc]` (dedicated mode) or the equivalent per-RISC sum (dynamic mode). See [Chapter 1, `02_blocking_primitives_taxonomy.md` -- NWBW](../ch01_anatomy_of_a_hang/02_blocking_primitives_taxonomy.md#nwbw-noc_async_write_barrier----noc-write-barrier-wait) for the full code listing and exit conditions.

---

## Hang Cause 2.4.4: Write Barrier Hang from Multicast `num_dests` Mismatch

### Symptom

Core stuck at `NWBW`. The `NIU_MST_WR_ACK_RECEIVED` counter is advancing but never reaches `noc_nonposted_writes_acked[noc]`. The deficit corresponds exactly to a miscounted multicast destination.

### Root Cause

When a multicast write is issued, the software counter `noc_nonposted_writes_acked[noc]` is incremented by `num_dests` -- the expected number of acknowledgments. If `num_dests` does not match the actual number of cores in the multicast rectangle:

- **`num_dests` too high**: The barrier expects more acks than will ever arrive. The barrier hangs.
- **`num_dests` too low**: The barrier completes before all acks arrive. Subsequent barriers may then hang because the deficit carries over.

### Diagnosis Steps

1. Read `NIU_MST_WR_ACK_RECEIVED` and `noc_nonposted_writes_acked[noc]`.
2. Calculate the deficit.
3. Check if the deficit matches a specific multicast's `num_dests` value.
4. Calculate expected `num_dests` from multicast rectangle: `(x_end - x_start + 1) * (y_end - y_start + 1)`.

### Fix

Correct the `num_dests` value to match the actual multicast target count.

### Prevention

Compute `num_dests` from the actual multicast grid dimensions, not from hardcoded values. The `noc_async_write_multicast` API takes `num_dests` as an explicit parameter -- always derive it from the grid specification.

---

## Hang Cause 2.4.5: Blackhole Inline-Write Back-Pressure

### Symptom

On Blackhole. Core at `NWBW` or at a NOC command buffer wait (`NWPW`). The hang occurs after an inline write to L1.

### Root Cause

Inline writes on Blackhole use all four L1 memory ports simultaneously. Under contention, back-pressure builds and the pipeline stalls. From `risc_attribs.h`:

> "Inline writes use all 4 memory ports and may hang on Blackhole when there is back-pressure. This hang only manifests when the inline writes are issued to a L1 location."

The `InlineWriteDst` enum provides the workaround mechanism:

```c++
enum class InlineWriteDst : uint8_t { DEFAULT = 0, L1 = 1, REG = 2 };
```

When `dst_type == InlineWriteDst::L1`, the implementation falls back to `noc_async_write` instead of a hardware inline write.

### Diagnosis Steps

1. Confirm the architecture is Blackhole.
2. Check for `noc_inline_dw_write` calls targeting L1 addresses (not stream registers).
3. Look for `NWBW` or `NWPW` waypoints.

### Fix

Use `noc_inline_dw_write<InlineWriteDst::L1>` (which routes through `noc_async_write` on BH) or replace with a standard `noc_async_write`.

### Prevention

- On Blackhole, never use default `noc_inline_dw_write` for L1 targets. Always specify `InlineWriteDst::L1`.
- The API documentation warns: *"Note: On Blackhole, this API can only write to stream registers, writing to L1 will cause hangs!"*

---

## Hang Cause 2.4.6: The Mcast Path Reservation Hang (Hardware Bug with Software Workaround)

### Symptom

A multicast write hangs the NOC command buffer. Subsequent transactions on the same NOC are blocked. The core may be stuck at `NWBW` or `NWPW`. This occurs specifically when an unlinked multicast follows unicast transactions without an intervening write barrier.

### Root Cause

This is a **known hardware bug** documented in the dispatch kernel (`cq_dispatch.cpp`):

```c++
// tt_metal/impl/dispatch/kernels/cq_dispatch.cpp
auto wait_for_barrier = [&]() {
    if (!mcast) {
        return;
    }
    noc_nonposted_writes_num_issued[noc_index] += writes;
    noc_nonposted_writes_acked[noc_index] += mcasts;
    writes = 0;
    mcasts = 0;
    // Workaround mcast path reservation hangs by always waiting for a
    // write barrier before doing an mcast that isn't linked to a
    // previous mcast.
    noc_async_write_barrier();
};
```

The hardware issue: multicast transactions use a "path reservation" mechanism. If there are pending unicast transactions when the multicast path reservation is attempted, the path reservation can deadlock against the unicast transactions.

The workaround: **always issue a write barrier before performing a multicast that is not linked to a previous multicast.** Linked multicasts share the path reservation with the previous multicast and do not need a barrier between them.

### Diagnosis Steps

1. Identify a `NWBW` or `NWPW` hang during multicast operations.
2. Check if the preceding code path includes a mix of unicast and multicast writes without intervening barriers.
3. Verify that the `wait_for_barrier` pattern is being applied.

### Fix

Insert `noc_async_write_barrier()` before any multicast write that is not linked to a previous multicast on the same NOC.

### Prevention

- When writing dispatch or data-movement kernels that use multicast, always follow the pattern established in `cq_dispatch.cpp`.
- This workaround applies to all architectures that support multicast (WH, BH, QA).

---

## Semaphore Hangs (`NSW` and `NSMW`)

`noc_semaphore_wait` (`NSW`) spins until `*sem_addr == val` (exact equality). `noc_semaphore_wait_min` (`NSMW`) spins until `*sem_addr >= val` (threshold). See [Chapter 1, `02_blocking_primitives_taxonomy.md` -- NSW](../ch01_anatomy_of_a_hang/02_blocking_primitives_taxonomy.md#nsw-noc_semaphore_wait----noc-semaphore-wait-equality) and [NSMW](../ch01_anatomy_of_a_hang/02_blocking_primitives_taxonomy.md#nsmw-noc_semaphore_wait_min----noc-semaphore-wait-minimum) for the full code listings.

---

## Hang Cause 2.4.7: Semaphore Overshoot with `noc_semaphore_wait`

### Symptom

Core stuck at waypoint `NSW`. The semaphore value in L1 is non-zero but not equal to the expected value -- it may be *greater* than expected.

### Root Cause

Because `noc_semaphore_wait` checks for **exact equality** (`!= val`), if the semaphore is incremented past the target value, the condition is never satisfied.

Example:
- Core A waits for `*sem == 1` (expecting one signal from Core B)
- Core B increments the semaphore twice (from 0 to 2) before Core A reads it
- Core A sees `*sem == 2`, which `!= 1`, so it keeps waiting forever

**Buggy pattern:**
```c++
// WRONG: Exact match is fragile with multiple producers
noc_semaphore_wait(sem_addr, 1);
```

**Corrected pattern:**
```c++
// CORRECT: Threshold check is immune to overshoot
noc_semaphore_wait_min(sem_addr, 1);
```

### Diagnosis Steps

1. Read the semaphore value at `sem_addr`.
2. Compare with the expected `val`.
3. If `*sem_addr > val`, this is an overshoot.

### Fix

Replace `noc_semaphore_wait(sem, val)` with `noc_semaphore_wait_min(sem, val)`.

### Prevention

- Use `noc_semaphore_wait_min` for any semaphore that is incremented atomically by remote cores.
- Reserve `noc_semaphore_wait` (exact equality) only for cases where the semaphore is set to a specific value (via `noc_semaphore_set`) rather than incremented.

---

## Hang Cause 2.4.8: Semaphore Signal Never Arrives

### Symptom

Core stuck at `NSW` or `NSMW`. The semaphore value is stuck at its initial value (typically 0) and never changes.

### Root Cause

The core that is supposed to increment or set the semaphore is not doing so. Possible reasons:

1. **Signaling core is itself hung**: If Core B is stuck at its own blocking primitive, the signal never arrives.
2. **Wrong semaphore address**: The signaling core writes to a different L1 address than the one being waited on.
3. **NOC write delivering the semaphore fails**: The `noc_semaphore_inc` targets wrong coordinates or encounters inline-write back-pressure on Blackhole.
4. **Wrong initial value**: The semaphore was not reset between kernel iterations.
5. **Forgotten signal**: The kernel logic has a code path where the signaling core exits without incrementing the semaphore.

### Diagnosis Steps

1. Read the semaphore value at the waited address.
2. Identify which core is supposed to set/increment this semaphore.
3. Check that core's waypoint -- is it hung, done, or still running?
4. If `D`, it exited without sending the signal. Check the kernel source for early-exit paths.
5. Verify the semaphore address and NOC coordinates match between writer and waiter.

### Fix

- If the signaling core forgot to signal: add the missing `noc_semaphore_inc` or `noc_semaphore_set`.
- If the NOC write failed: fix the NOC address (see [`03_noc_address_sanitization_and_violations.md`](./03_noc_address_sanitization_and_violations.md)).
- If the initial value is wrong: reset the semaphore at the beginning of each iteration.

### Prevention

- Reset semaphores to 0 (via `noc_semaphore_set(sem, 0)`) at the beginning of each kernel invocation.
- Document the semaphore protocol: which core signals, which waits, what value, when.
- Define semaphore addresses centrally and share them across kernels via runtime arguments.

---

## Hang Cause 2.4.9: Semaphore Wrong Count in Multi-Iteration Protocols

### Symptom

The hang occurs after a specific number of iterations. The semaphore value reaches the correct value for the first N iterations but then falls behind.

### Root Cause

In multi-iteration protocols, the semaphore is typically incremented once per iteration by the producer and waited-on by the consumer. If the producer misses one increment (due to a conditional branch, an off-by-one in the loop bound, or a timing issue), the consumer will eventually wait for a value that is one higher than what was signaled.

### Diagnosis Steps

1. Read the semaphore value and compare with the expected value.
2. If the semaphore is exactly one less than expected, look for an off-by-one in the producer's loop.

### Fix

Match the producer's iteration count with the consumer's expected count.

### Prevention

- Pass the iteration count as a runtime argument to both producer and consumer from the same source.
- Use `noc_semaphore_wait_min` to allow for timing variations where the producer may be slightly ahead.

---

## Hang Cause 2.4.10: Semaphore Not Reset Between Iterations

### Symptom

The first kernel invocation works correctly. The second invocation hangs immediately at `NSW` or `NSMW`, or succeeds immediately when it should wait.

### Root Cause

If a semaphore is not reset to 0 between kernel invocations, the residual value affects the next invocation:
- Previous value at target: `noc_semaphore_wait` returns immediately (stale success)
- Previous value past target: `noc_semaphore_wait` returns immediately (for `wait_min`) or never returns (for `wait` with exact match)

**Buggy pattern:**
```c++
// WRONG: No reset between iterations
void kernel_main() {
    for (uint32_t iter = 0; iter < num_iters; iter++) {
        // ... work ...
        noc_semaphore_inc(remote_sem, 1);
        noc_semaphore_wait(local_sem, 1);  // Hangs on iter > 0!
    }
}
```

**Corrected pattern:**
```c++
// CORRECT: Reset at start of each iteration
void kernel_main() {
    for (uint32_t iter = 0; iter < num_iters; iter++) {
        noc_semaphore_set(local_sem, 0);    // Reset
        // ... work ...
        noc_semaphore_inc(remote_sem, 1);
        noc_semaphore_wait_min(local_sem, 1);  // Also use wait_min for safety
    }
}
```

### Diagnosis Steps

1. Check the semaphore value at the start of the kernel invocation. If non-zero, it was not reset.
2. Trace the reset logic: is `noc_semaphore_set(sem, 0)` called at the beginning of each iteration?

### Fix

Add `noc_semaphore_set(sem, 0)` at the beginning of the kernel or each iteration.

### Prevention

Include semaphore reset as part of the standard kernel prologue.

---

## The `noc_semaphore_inc` Function: Atomic Increment Mechanics

```c++
// tt_metal/hw/inc/api/dataflow/dataflow_api.h
template <bool posted = false>
FORCE_INLINE void noc_semaphore_inc(
    uint64_t addr, uint32_t incr,
    uint8_t noc_id = noc_index, uint8_t vc = NOC_UNICAST_WRITE_VC) {
    WAYPOINT("NSIW");
    DEBUG_SANITIZE_NOC_ADDR(noc_id, addr, 4);
    DEBUG_INSERT_DELAY(TransactionAtomic);
    noc_fast_atomic_increment<noc_mode>(
        noc_id, write_at_cmd_buf, addr, vc, incr,
        31 /*wrap*/, false /*linked*/, posted /*posted*/,
        MEM_NOC_ATOMIC_RET_VAL_ADDR);
    WAYPOINT("NSID");
}
```

Key details:
- The `wrap` parameter is 31, meaning the atomic value wraps at `(1 << 31)` -- effectively no wrap for typical usage.
- By default (`posted = false`), the increment is non-posted and contributes to the `noc_nonposted_writes_acked` counter.
- The `NSIW`/`NSID` waypoints are around the *issuance*, not a wait. A hang at `NSIW` means the NOC command buffer is full.

The multicast variant `noc_semaphore_inc_multicast` atomically increments a semaphore on a grid of cores simultaneously, with the same `num_dests` correctness requirements as multicast writes.

---

## Transaction ID Barriers

Transaction IDs (TRIDs) provide finer-grained barrier control on architectures that support them (Blackhole: max 255 TRIDs, Quasar: max 65535 TRIDs).

### `noc_async_read_barrier_with_trid` (`NBTW`)

```c++
void noc_async_read_barrier_with_trid(uint32_t trid, uint8_t noc = noc_index) {
    WAYPOINT("NBTW");
    while (!ncrisc_noc_read_with_transaction_id_flushed(noc, trid)) {
        continue;
    }
    invalidate_l1_cache();
    WAYPOINT("NBTD");
}
```

Exit condition: `NIU_MST_REQS_OUTSTANDING_ID(trid) == 0` -- no outstanding requests with this TRID.

### `noc_async_write_barrier_with_trid` (`NWTW`)

```c++
void noc_async_write_barrier_with_trid(uint32_t trid, uint8_t noc = noc_index) {
    WAYPOINT("NWTW");
    while (!ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc, trid)) {
        continue;
    }
    invalidate_l1_cache();
    WAYPOINT("NWTD");
}
```

### `noc_async_write_flushed_with_trid` (`NFTW`)

```c++
void noc_async_write_flushed_with_trid(uint32_t trid, uint8_t noc = noc_index) {
    WAYPOINT("NFTW");
    while (!ncrisc_noc_nonposted_write_with_transaction_id_sent(noc, trid)) {
        continue;
    }
    invalidate_l1_cache();
    WAYPOINT("NFTD");
}
```

---

## Hang Cause 2.4.11: TRID Barrier Hang from Reuse or Stale State

### Symptom

Core stuck at `NBTW` or `NWTW`. The TRID-specific outstanding request counter never reaches zero.

### Root Cause

Two distinct failure modes:

**Reuse before completion:** The kernel reuses a TRID before the previous transaction with that ID has completed. The outstanding count increases but the barrier waits for zero.

**Stale counter:** If the TRID was never assigned to any transaction in the current kernel, the hardware register may contain a stale value from a previous operation (the BH dynamic NOC contamination problem from Hang Cause 2.1.12).

### Diagnosis Steps

1. Read `NIU_MST_REQS_OUTSTANDING_ID(trid)` for the stuck TRID.
2. Verify the TRID value is within the valid range (BH: 0-254, Quasar: 0-65534).
3. If no transaction was assigned this TRID in the current kernel, the counter is stale.

### Fix

- If the transaction failed: fix the underlying NOC address error.
- If the TRID is stale: call `noc_async_read_barrier_with_trid_reset_mask` to clear the counter before using it.
- Always call the appropriate TRID barrier between reuses of the same transaction ID.

### Prevention

- Reset TRID counters at the beginning of each kernel when using dynamic NOC mode.
- Use a TRID allocation scheme that cycles through available IDs.

---

## Hang Cause 2.4.12: Blackhole Fabric Router Counter Workaround

### Symptom

A NOC write transaction with TRID hangs on Blackhole. The barrier never completes. This occurs specifically with writes that skip counter updates.

### Root Cause

Blackhole has a known issue (GitHub #28758) where the fabric router checks NOC write counters as part of its routing logic. If a write operation does not update the counter (`update_counter = false`), the fabric router's counter check finds a mismatch and enters a hang state.

The workaround:

```c++
// tt_metal/hw/inc/api/dataflow/dataflow_api.h
#ifdef ARCH_BLACKHOLE
    // Issue https://github.com/tenstorrent/tt-metal/issues/28758:
    // always update counter for blackhole as a temporary workaround
    // for avoiding hangs in fabric router
    constexpr bool update_counter_in_callee = true;
#else
    constexpr bool update_counter_in_callee = update_counter;
#endif
```

On Blackhole, `update_counter_in_callee` is always `true`, regardless of the template parameter.

> **Danger:** This is a temporary workaround. The comment states: "will remove this restriction once all inline write change to stream reg write." Custom kernel code that bypasses the standard APIs may not benefit from this workaround.

### Diagnosis Steps

1. Confirm the architecture is Blackhole.
2. Check if the write operation used `update_counter = false`.
3. The hang manifests in the fabric router, which may be difficult to observe directly.

### Fix

Ensure all BH writes update the NOC counter. Use the standard `noc_async_write_*` APIs on Blackhole, which automatically apply the counter workaround.

### Prevention

- Always use the standard `noc_async_write_*` APIs on Blackhole rather than low-level NOC register writes. The APIs automatically apply the `update_counter = true` workaround.
- When writing custom NOC transaction code, never set `update_counter = false` on Blackhole.
- Monitor GitHub #28758 for the permanent hardware/firmware fix that will remove this restriction.

---

## NOC Full Barrier Hang (`NFBW` through `NFBD`)

The full barrier is a sequence of five individual barriers:

```c++
void noc_async_full_barrier(uint8_t noc_idx = noc_index) {
    WAYPOINT("NFBW");  // Waiting for reads
    while (!ncrisc_noc_reads_flushed(noc_idx));
    WAYPOINT("NFCW");  // Waiting for non-posted writes sent
    while (!ncrisc_noc_nonposted_writes_sent(noc_idx));
    WAYPOINT("NFDW");  // Waiting for non-posted writes acked
    while (!ncrisc_noc_nonposted_writes_flushed(noc_idx));
    WAYPOINT("NFEW");  // Waiting for atomics acked
    while (!ncrisc_noc_nonposted_atomics_flushed(noc_idx));
    WAYPOINT("NFFW");  // Waiting for posted writes sent
    while (!ncrisc_noc_posted_writes_sent(noc_idx));
    WAYPOINT("NFBD");
}
```

Each sub-waypoint reveals which class of NOC transaction is incomplete:

| Sub-Waypoint | Stuck On |
|--------------|----------|
| `NFBW` | Outstanding reads not completed |
| `NFCW` | Non-posted writes not departed |
| `NFDW` | Non-posted writes not acknowledged |
| `NFEW` | Atomics not acknowledged |
| `NFFW` | Posted writes not departed |

### Diagnosis Steps

The specific sub-waypoint directly identifies the class of NOC transaction causing the hang:
- `NFBW`: Same as Hang Cause 2.4.1 (read barrier)
- `NFDW`: Same as Hang Cause 2.4.4 (write barrier)
- `NFEW`: Check outstanding atomics (similar to the `NABW` hang from Hang Cause 2.1.11)
- `NFCW`/`NFFW`: NOC departure hangs indicate command buffer congestion or hardware-level stalls

---

## NOC Write Flush vs. NOC Write Barrier

Understanding the distinction prevents unnecessary hangs from overly strict synchronization:

| Function | Waypoint | Waits For | Use Case |
|----------|----------|-----------|----------|
| `noc_async_write_barrier` | `NWBW` | All non-posted writes acknowledged by destination | Data ordering: "destination has the data" |
| `noc_async_writes_flushed` | `NWFW` | All non-posted writes sent from local NIU | Resource reuse: "local L1 buffer can be reused" |
| `noc_async_posted_writes_flushed` | `NPWW` | All posted writes sent from local NIU | Fire-and-forget writes |

> **Tip:** Use `noc_async_writes_flushed` when you only need to reuse the local source buffer. Use `noc_async_write_barrier` when you need to guarantee the remote destination has received the data (e.g., before sending a semaphore signal that tells the remote core its data is ready).

---

## Summary: Barrier and Semaphore Hang Diagnostic Table

| Scenario | Waypoint | Counter Check | Most Common Root Cause |
|----------|----------|---------------|------------------------|
| 2.4.1 Read to invalid addr | `NRBW` | `NIU_MST_RD_RESP_RECEIVED != noc_reads_num_issued` | Invalid read target address |
| 2.4.2 NOC index mismatch | `NRBW` | Wrong NOC's counters checked | Barrier on wrong NOC |
| 2.4.3 Missing barrier | `NRBW`/`NWBW` (secondary) | N/A (data corruption) | Missing `noc_async_read_barrier` |
| 2.4.4 num_dests mismatch | `NWBW` | `NIU_MST_WR_ACK_RECEIVED != noc_nonposted_writes_acked` | Wrong multicast target count |
| 2.4.5 BH inline-write | `NWBW`/`NWPW` | Write counter stuck | All 4 memory ports back-pressure |
| 2.4.6 Mcast path reservation | `NWBW`/`NWPW` | Path reservation deadlock | Missing barrier before unlinked mcast |
| 2.4.7 Semaphore overshoot | `NSW` | `*sem_addr > val` | Multiple producers or stale value |
| 2.4.8 Signal never arrives | `NSW`/`NSMW` | `*sem_addr` unchanged | Signaler hung, wrong address, or NOC failure |
| 2.4.9 Wrong iteration count | `NSW`/`NSMW` | `*sem_addr` falls behind | Off-by-one in producer loop |
| 2.4.10 Not reset | `NSW`/`NSMW` | Stale non-zero value | Missing `noc_semaphore_set(sem, 0)` |
| 2.4.11 TRID stale/reuse | `NBTW`/`NWTW` | TRID outstanding != 0 | Reuse before completion or stale counter |
| 2.4.12 BH fabric router | `NWTW` | Write counter mismatch | `update_counter = false` on BH |

| Waypoint | Blocking Primitive | Exit Condition |
|----------|-------------------|----------------|
| `NRBW` | `noc_async_read_barrier` | All read responses received |
| `NWBW` | `noc_async_write_barrier` | All non-posted writes acknowledged |
| `NWFW` | `noc_async_writes_flushed` | All non-posted writes departed |
| `NABW` | `noc_async_atomic_barrier` | All atomics acknowledged |
| `NSW` | `noc_semaphore_wait` | `*sem == val` (exact) |
| `NSMW` | `noc_semaphore_wait_min` | `*sem >= val` (threshold) |
| `NBTW` | `noc_async_read_barrier_with_trid` | TRID outstanding == 0 |
| `NWTW` | `noc_async_write_barrier_with_trid` | TRID writes acknowledged |
| `NFTW` | `noc_async_write_flushed_with_trid` | TRID writes departed |

> **Tip:** When BRISC shows `NRBW` or `NWBW` and all subordinates show `D`, the root cause is almost always a NOC address error in the BRISC kernel, not a subordinate issue. Check the BRISC kernel's NOC read/write addresses first.

---

**Previous:** [`03_noc_address_sanitization_and_violations.md`](./03_noc_address_sanitization_and_violations.md) | **Next:** [Chapter 3 -- Memory-Related Hang Causes](../ch03_memory_related_hangs/index.md)
