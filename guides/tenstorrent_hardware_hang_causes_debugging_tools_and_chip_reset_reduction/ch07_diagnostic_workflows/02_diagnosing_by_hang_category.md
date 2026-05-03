# 7.2 Diagnosing by Hang Category

[Previous: Initial Triage](./01_initial_triage.md) | [Next: Narrowing and Reproducing](./03_narrowing_and_reproducing.md)

---

Once the initial triage (Section 01) has routed you to a specific hang category, this section provides the step-by-step diagnosis procedure for each. These are not exhaustive catalogs of every possible scenario (those are in Chapters 2-5); rather, they are focused workflows that identify the root cause as quickly as possible, using the tools from Chapter 6 in the most effective sequence.

This section also includes comprehensive **error-message-to-action mapping tables** for every known watcher error. If you have a specific error message on screen, look it up in the relevant table below for immediate diagnosis steps.

**Prerequisites:** [Section 01, Initial Triage](./01_initial_triage.md) (to have arrived here via the decision tree), [Chapter 1, `02_blocking_primitives_taxonomy.md`](../ch01_anatomy_of_a_hang/02_blocking_primitives_taxonomy.md) (waypoint codes), [Chapter 6](../ch06_debugging_tools/) (tool configuration).

---

## Common Waypoint Pattern Table

Before diving into category-specific procedures, this table maps common multi-core waypoint patterns to their most likely diagnosis:

| Worker Core Pattern | Dispatch Core Pattern | Likely Diagnosis | Section |
|--------------------|-----------------------|-----------------|---------|
| One core `CRBW`, another core `CWFW` on same Tensix | Normal (`D`) | CB deadlock: producer/consumer mismatch | [Kernel CB Deadlock](#kernel-cb-deadlock-diagnosis) |
| Multiple cores `CWFW` | Normal (`D`) | Compute pipeline stall or reader hung | [Kernel CB Deadlock](#kernel-cb-deadlock-diagnosis) |
| One+ cores `NRBW` or `NWBW` | Normal (`D`) | NOC transaction never completed | [NOC Hang](#noc-hang-diagnosis) |
| All cores `NSW` or `NSMW` | Normal (`D`) | Semaphore protocol violation | [Semaphore Deadlock](#semaphore-deadlock-diagnosis) |
| Worker cores `D` (done) or running | Dispatch stuck at `HQW` | Prefetch waiting for host; host may be blocked | [Dispatch Hang](#dispatch-hang-diagnosis) |
| Worker cores still running | Dispatch stuck at `PWW`/`WCW` | Dispatch waiting for workers to complete | [Dispatch Hang](#dispatch-hang-diagnosis) |
| Worker cores `D` (done) | Dispatch stuck at `PWW`/`WCW` | Dispatch missed completion signal | [Dispatch Hang](#dispatch-hang-diagnosis) |
| Mix of stuck waypoints across many cores | Normal or stuck | Memory corruption cascading to multiple failures | [Memory Corruption](#memory-corruption-diagnosis) |
| Worker cores running on one chip, idle on others | Dispatch running | Multi-chip collective stalled | [Multi-Chip Hang](#multi-chip-hang-diagnosis) |

---

## NOC Error Message Mapping Table

These errors are reported by the watcher NOC sanitization feature. They appear in the watcher log and are also printed to the terminal with a `TT_THROW`. Each error message is emitted from `watcher_device_reader.cpp` via `DumpNocSanitizeStatus()`.

**Source code reference:** `tt_metal/impl/debug/watcher_device_reader.cpp`, lines 656-753; error codes defined in `tt_metal/hw/inc/hostdev/dev_msgs.h`, lines 229-244.

| Error Message (from watcher log) | Sanitize Code | Root Cause | Diagnosis Steps |
|---|---|---|---|
| `(NOC target address underflow).` | `DebugSanitizeNocAddrUnderflow` (3) | NOC address below valid range. Typically caused by a zero or negative offset applied to a base address. | 1. Read the full error for source RISC, NOC ID, read/write, local L1 address, target core, target address. 2. Identify the kernel from `k_ids`. 3. Check if the target address was computed from a runtime argument. 4. If DRAM, verify bank ID and offset calculations. Cross-ref: [Ch3](../ch03_memory_related_hangs/). |
| `(NOC target address overflow).` | `DebugSanitizeNocAddrOverflow` (4) | Address beyond valid range. Common when `base_addr + offset + len` exceeds the target memory size. | 1. Read full error for core/address details. 2. Check buffer allocation size vs. actual access size. 3. Verify `len` parameter in `noc_async_read`/`noc_async_write`. 4. Check for off-by-one in tile count calculations. Cross-ref: [Ch3](../ch03_memory_related_hangs/). |
| `(zero length transaction).` | `DebugSanitizeNocAddrZeroLength` (5) | DMA transfer with length = 0, which is never valid. | 1. Trace the `len` parameter through kernel code. 2. Common cause: empty tensor edge case not handled. 3. Check `cb_reserve_back`/`cb_wait_front` called with 0 tiles. |
| `(NOC target address did not map to any known Tensix/Ethernet/DRAM/PCIE core).` | `DebugSanitizeNocTargetInvalidXY` (6) | NOC X/Y coordinates do not correspond to any valid core. | 1. Read target coordinates from error. 2. Check physical vs. virtual coordinate confusion ([Ch1](../ch01_anatomy_of_a_hang/04_hang_causes_across_architectures.md)). 3. Verify NOC address encoding. 4. On WH check NOC0 vs NOC1 coordinate mapping. 5. On BH check virtual coordinate mapping. |
| `(multicast to non-worker core).` | `DebugSanitizeNocMulticastNonWorker` (7) | Multicast target range includes non-worker cores. Multicast is only valid to Tensix workers. | 1. Verify multicast range coordinates. 2. Check if harvested rows are incorrectly included. 3. Ensure grid range excludes Ethernet and DRAM columns/rows. |
| `(multicast invalid range).` | `DebugSanitizeNocMulticastInvalidRange` (8) | Multicast start > end coordinates, or range otherwise malformed. | 1. Check start_x <= end_x and start_y <= end_y. 2. On WH with NOC1, coordinates are mirrored. 3. Verify `CoreRange` construction in host code. |
| `(invalid address alignment in NOC transaction).` | `DebugSanitizeNocAlignment` (9) | Address not aligned to required boundary (`NOC_L1_READ_ALIGNMENT_BYTES` or `NOC_L1_WRITE_ALIGNMENT_BYTES`, typically 16B). | 1. Check DRAM address alignment. 2. Verify L1 buffer addresses are not offset inappropriately. Cross-ref: [Ch3, `03_alignment_and_tile_size_mismatches.md`](../ch03_memory_related_hangs/03_alignment_and_tile_size_mismatches.md). |
| `(mixing virtual and virtual coordinates in Mcast).` | `DebugSanitizeNocMixedVirtualandPhysical` (10) | Multicast mixes virtual with physical coordinates. | 1. Ensure all cores in the multicast range use the same coordinate system. 2. Typically caused by mixing `device->worker_core_from_logical_core()` with raw NOC coordinates. |
| `(inline dw writes do not support DRAM destination addresses).` | `DebugSanitizeInlineWriteDramUnsupported` (11) | Inline doubleword write targeting DRAM (unsupported). **Blackhole-specific.** | 1. Replace inline writes to DRAM with regular `noc_async_write`. 2. See [Ch1, architecture differences](../ch01_anatomy_of_a_hang/04_hang_causes_across_architectures.md). |
| `(NOC target overwrites mailboxes).` | `DebugSanitizeNocAddrMailbox` (12) | NOC write targets the mailbox region of L1, corrupting firmware state. | 1. Check target L1 address against mailbox range. 2. Usually a buffer allocation error overlapping the reserved mailbox region. |
| `(submitting a non-mcast transaction when there's a linked transaction).` | `DebugSanitizeNocLinkedTransactionViolation` (13) | Unicast submitted while a linked multicast is in progress. | 1. Review kernel code for mixed unicast/multicast patterns. 2. Ensure all transactions in a linked sequence are multicast. 3. Enable `TT_METAL_WATCHER_ENABLE_NOC_SANITIZE_LINKED_TRANSACTION=1`. |
| `(read or write past the end of local memory).` | `DebugSanitizeL1AddrOverflow` (14) | L1 address exceeds `MEM_L1_SIZE`. | 1. Check the RISC identified in the error. 2. Look for array accesses or pointer arithmetic exceeding buffer boundaries. 3. May indicate stack overflow. |
| `(ethernet send with L1 source overflow).` | `DebugSanitizeEthSrcL1AddrOverflow` (15) | Ethernet send reads from out-of-bounds local L1 address. | 1. Check source buffer address and length. 2. Verify Ethernet send buffer is properly allocated. Cross-ref: [Ch5](../ch05_multi_chip_and_ccl_hangs/). |
| `(ethernet send to core with L1 destination overflow).` | `DebugSanitizeEthDestL1AddrOverflow` (16) | Ethernet send targets out-of-bounds L1 on remote core. | 1. Check Ethernet routing configuration. 2. Verify destination L1 address on remote chip. Cross-ref: [Ch5](../ch05_multi_chip_and_ccl_hangs/). |
| `(NOC transaction overflows a circular buffer).` | `DebugSanitizeCBOutOfBounds` (17) | NOC transaction extends beyond CB boundaries. | 1. Check `ntiles` parameter -- must evenly divide CB size. 2. Verify tile size in NOC transaction matches CB's configured tile size. 3. Check for off-by-one in tile iteration loops. Cross-ref: [Ch2, CB deadlocks](../ch02_kernel_and_noc_hangs/02_circular_buffer_deadlocks.md). |
| `corrupted noc sanitization state - sanitization memory overwritten` | Corrupted sentinel fields | Watcher sanitization metadata overwritten by stray write. | 1. Indicates **memory corruption** bug. 2. Enable `DebugSanitizeNocAddrMailbox` checking. 3. Review all NOC write targets. 4. Check for stack overflow. |
| `corrupted noc sanitization state - unknown failure code` | Unknown `return_code` value | The `return_code` field contains an unrecognized value. Sanitization memory itself is corrupted. | Same as above: suspect stray writes into the mailbox/watcher L1 region. |

### NOC Error Diagnosis Procedure

For any NOC sanitize error:

1. **Read the full error message.** It contains: source RISC name, NOC ID (0 or 1), read/write direction, local L1 address, target core coordinates, target memory address, and the specific violation type.
2. **Identify the kernel.** The watcher line for the same core shows `k_ids` which maps to `kernel_names.txt` in `generated/watcher/`.
3. **Find the offending NOC call.** Using the RISC name and operation type (read/write/multicast), search the identified kernel for the matching `noc_async_read`, `noc_async_write`, or `noc_async_write_multicast` call.
4. **Trace the address computation.** The faulty address was computed from runtime arguments, compile-time constants, or CB interface pointers. Trace backward from the NOC call to find the source of the invalid address.
5. **If the error is intermittent,** the address may depend on runtime state (e.g., a loop counter, a semaphore value used as an offset). Use `DPRINT` to log the address values, or use watcher ring buffer (`WATCHER_RING_BUFFER_PUSH`) to capture the values without DPrint overhead.

---

## Assert Message Mapping Table

These errors are reported by the watcher assert feature. The assert type is read from the watcher mailbox.

**Source code reference:** `tt_metal/impl/debug/debug_helpers.hpp`, lines 101-123; assert types in `tt_metal/hw/inc/hostdev/dev_msgs.h`, lines 254-261.

| Assert Message | Assert Code | Root Cause | Diagnosis Steps |
|---|---|---|---|
| `tripped an assert on line <N>` | `DebugAssertTripped` (3) | `ASSERT()` in kernel code evaluated to false. | 1. Identify kernel from `k_ids`. 2. Use kernel ELF path and line number. Note: line number may be from inlined header. 3. Use `TT_METAL_WATCHER_NOINLINE=1` for accurate line numbers. 4. Check ring buffer for values logged before assert. |
| `detected an inter-kernel data race ... (missing NOC reads flushed barrier).` | `DebugAssertNCriscNOCReadsFlushedTripped` (4) | Kernel completed with outstanding NOC read transactions. | 1. Add `noc_async_read_barrier()` before kernel exit. 2. This is a correctness bug even if it does not hang. Cross-ref: [Ch2, NOC barriers](../ch02_kernel_and_noc_hangs/04_noc_barrier_and_semaphore_hangs.md). |
| `detected an inter-kernel data race ... (missing NOC non-posted writes sent barrier).` | `DebugAssertNCriscNOCNonpostedWritesSentTripped` (5) | Kernel completed with outstanding non-posted write transactions. | 1. Add `noc_async_write_barrier()` before exit. |
| `detected an inter-kernel data race ... (missing NOC non-posted atomics flushed barrier).` | `DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped` (6) | Kernel completed with outstanding atomic NOC operations. | 1. Add appropriate atomic barrier before exit. Common in kernels using semaphore increment via NOC atomics. |
| `detected an inter-kernel data race ... (missing NOC posted writes sent barrier).` | `DebugAssertNCriscNOCPostedWritesSentTripped` (7) | Kernel completed with outstanding posted write transactions. | 1. Add posted-write barrier before exit. |
| `accessed unique runtime arg index out of bounds.` | `DebugAssertRtaOutOfBounds` (8) | Kernel accessed runtime arg at index beyond allocated count. | 1. Compare number of runtime args set by host (`SetRuntimeArgs`) with kernel-side `get_arg_val<>()` indices. |
| `accessed common runtime arg index out of bounds.` | `DebugAssertCrtaOutOfBounds` (9) | Same for common (shared across all cores) runtime arguments. | 1. Check `SetCommonRuntimeArgs` on host side. |

### Assert Diagnosis Procedure

1. **Identify the processor.** The error includes which RISC (brisc, ncrisc, trisc0/1/2, erisc) tripped the assert.
2. **Identify the kernel.** Use `k_ids` from the same watcher line, cross-referenced with `kernel_names.txt`.
3. **For `DebugAssertTripped` (line-number asserts):** Use the kernel ELF path to find the source. Enable `TT_METAL_WATCHER_NOINLINE=1` and reproduce for accurate line numbers.
4. **For inter-kernel data race asserts (codes 4-7):** Add the missing barrier. These are always software bugs in the kernel.
5. **For runtime arg out-of-bounds (codes 8-9):** Reconcile host-side `SetRuntimeArgs` counts with kernel-side `get_arg_val` indices.

---

## Kernel CB Deadlock Diagnosis

**Entry criteria:** Watcher or triage shows one or more worker cores stuck at `CRBW` (CB Reserve Back Wait) or `CWFW` (CB Wait Front Wait).

### Step-by-Step Procedure

**Step 1: Identify the stuck core(s) and their waypoints.**

From the watcher log or tt-triage output, note:
- Which core(s) are stuck (logical coordinates `(x, y)`)
- Which RISC on each core is stuck (BRISC, NCRISC, TRISC0/1/2)
- The waypoint code (`CRBW` or `CWFW`)

```
Example watcher log fragment:
  Core (1,1): BRISC: CRBW  NCRISC: D  TRISC0: CWFW  TRISC1: R  TRISC2: R
  This shows: BRISC (reader) waiting for CB space, TRISC0 (unpack) waiting for CB data.
  NCRISC (writer) has finished (D). TRISCs 1/2 are running.
```

**Step 2: Map the RISC waypoint pattern to a specific scenario.**

| BRISC | NCRISC | TRISC0 | Pattern | Most Likely Scenario | Cross-Reference |
|-------|--------|--------|---------|---------------------|-----------------|
| CRBW | D | CWFW | Producer full, consumer waiting | Mismatched push/pop counts | Ch 2, Scenario 2.2.1 |
| D | CRBW | CWFW | Writer producing, compute waiting | Tile count mismatch | Ch 2, Scenario 2.2.5 |
| CRBW | CWFW | R | Both data-mover RISCs stuck | CB size indivisible by tile count | Ch 2, Scenario 2.2.4 |
| CRBW | D | R | Producer stuck, compute running | Downstream consumer (pack) not draining | Ch 2, Scenario 2.2.2 |

**Step 3: Identify the kernel running on each RISC.**

Map the kernel ID from the watcher log to a source file using `generated/watcher/kernel_names.txt`:

```bash
cat generated/watcher/kernel_names.txt
```

**Step 4: Determine which CB is involved.**

A core can use up to 32 circular buffers (CB indices 0-31). To identify which CB is causing the stall:

- If tt-triage `check_cb_inactive.py` is available, it directly reports which CBs are inactive.
- Otherwise, examine the kernel source code to find which CB index is used with `cb_reserve_back` or `cb_wait_front` calls.
- Typical pattern: CB 0-7 for reader-to-compute data, CB 16-23 for compute-to-writer data.

**Step 5: Verify producer-consumer consistency.**

Check the following in the kernel source code (see [Chapter 2, `02_circular_buffer_deadlocks.md`](../ch02_kernel_and_noc_hangs/02_circular_buffer_deadlocks.md) for detailed scenarios):

| Check | What to Verify | Scenario If Wrong |
|-------|---------------|-------------------|
| Push/pop count match | Does the producer `cb_push_back(cb, N)` exactly as many tiles as the consumer `cb_pop_front(cb, N)` per iteration? | Scenario 2.2.1: Mismatched push/pop counts |
| Loop iteration match | Does the reader iterate the same number of times as the compute kernel expects? | Scenario 2.2.5: Loop bound mismatch |
| CB size divisibility | Is the CB total size an even multiple of the `num_pages` argument to `cb_reserve_back`/`cb_wait_front`? | Scenario 2.2.4: CB size indivisible by tile count |
| Cumulative `cb_wait_front` | Are `cb_wait_front` calls using cumulative totals (8, 16, 24...) not repeated values (8, 8, 8...)? | Scenario 2.2.2: Incorrect cumulative semantics |
| `cb_addr_shift` | Is the kernel compiled for the correct core type (data mover vs. compute)? | Ch 2, `02_circular_buffer_deadlocks.md` (cb_addr_shift section) |

**Step 6: If source inspection does not reveal the bug, enable CB sanitization.**

```bash
export TT_METAL_WATCHER=120
# Ensure CB sanitize is NOT disabled:
# unset TT_METAL_WATCHER_DISABLE_CB_SANITIZE
```

Re-run the workload. The watcher will check every NOC transaction against active CB boundaries and report `DebugSanitizeCBOutOfBounds` if any transaction touches CB memory incorrectly.

### CB Deadlock Diagnosis Checklist

```
CB DEADLOCK DIAGNOSIS
=====================
[ ] Identify stuck cores and their RISC waypoints
[ ] Map RISC pattern to scenario table
[ ] Identify kernel ID and source file
[ ] Identify which CB index is involved
[ ] Check: push/pop count match?
[ ] Check: loop iteration count match?
[ ] Check: CB size divisible by tile count?
[ ] Check: cumulative vs repeated cb_wait_front?
[ ] Check: correct core type compilation?
[ ] Enable CB sanitization and re-run if still unclear
```

### Cross-Reference

- Full CB deadlock scenario catalog: [Chapter 2, `02_circular_buffer_deadlocks.md`](../ch02_kernel_and_noc_hangs/02_circular_buffer_deadlocks.md) (Scenarios 2.2.1 through 2.2.10+)
- Remote CB deadlocks (`RemoteSenderCBInterface`, `RemoteReceiverCBInterface`): [Chapter 2, `02_circular_buffer_deadlocks.md`](../ch02_kernel_and_noc_hangs/02_circular_buffer_deadlocks.md)
- CB sanitization tool details: [Chapter 6, `01_watcher_system.md`](../ch06_debugging_tools/01_watcher_system.md)

---

## NOC Hang Diagnosis

**Entry criteria:** Watcher or triage shows one or more cores stuck at `NRBW` (read barrier), `NWBW` (write barrier), or `RP2W` (command buffer wait). Alternatively, a sanitize violation was reported, or `check_cb_inactive` shows active CBs.

### Step-by-Step Procedure

**Step 1: Distinguish sanitize-detected hangs from barrier hangs.**

- If the watcher log contains a `SANITIZE` line: the hang is **deliberate** -- the watcher caught an illegal NOC transaction, wrote the violation to the mailbox, and the core entered `while(1){}`. Decode the violation using the [NOC Error Message Mapping Table](#noc-error-message-mapping-table) above -- you have the root cause.
- If no sanitize violation: the hang is at a **barrier** -- the core issued a NOC transaction that never completed.

**Step 2: For barrier hangs, check outstanding transaction counts.**

Use tt-triage `check_noc_status.py` or read the NOC status registers directly:

```bash
./tools/tt-triage.py --run=check_noc_status --dev=0 --verbosity=4
```

Look for mismatches between issued and completed transactions:
- `noc_reads_num_issued[noc] > NIU_MST_RD_RESP_RECEIVED` means a read never returned a response
- `noc_nonposted_writes_num_issued[noc] > noc_nonposted_writes_acked[noc]` means a write was never acknowledged

**Step 3: Identify the target of the stuck transaction.**

Use `check_noc_locations.py` to inspect pending NOC addresses:

```bash
./tools/tt-triage.py --run=check_noc_locations --dev=0 --verbosity=4
```

This reveals:
- Was the transaction targeting a valid core? (Wrong coordinates = [Ch2, Scenario 2.3.1])
- Was the target address within valid L1/DRAM range? (Out of range = [Ch2, Scenario 2.3.2])
- Was the address properly aligned? (Misalignment = [Ch3, `03_alignment_and_tile_size_mismatches.md`](../ch03_memory_related_hangs/03_alignment_and_tile_size_mismatches.md))

**Step 4: Enable NOC Debug Dump for missing-barrier detection.**

If the barrier hang may be caused by unflushed writes:

```bash
export TT_METAL_NOC_DEBUG_DUMP=1
```

Re-run the workload. The `NOCDebugState` tracking system detects:

| Issue Type | Meaning |
|------------|---------|
| `WRITE_FLUSH_BARRIER` | A write was issued but never flushed before a barrier |
| `READ_BARRIER` | A read barrier has an incorrect transaction count |
| `UNFLUSHED_WRITE_AT_END` | A kernel completed with outstanding unflushed writes |
| `WRITE_TO_LOCKED_CORE_LOCAL_MEM` | Write targets locked local memory |
| `WRITE_TO_LOCKED_CB` | Write targets a locked circular buffer |

**Step 5: Check for Blackhole-specific issues.**

On Blackhole hardware:
- **Inline-write back-pressure:** Inline writes to L1 using all four memory ports can stall the NOC pipeline. Workaround: use stream register writes via `risc_attribs.h` (see [Chapter 1, `04_hang_causes_across_architectures.md`](../ch01_anatomy_of_a_hang/04_hang_causes_across_architectures.md)).
- **Relaxed memory ordering:** Test with `TT_METAL_DISABLE_RELAXED_MEM_ORDERING=1` to check if memory ordering is the issue (see [Section 05](./05_distinguishing_hw_vs_sw_bugs.md#blackhole-specific-options)).

**Step 6: Check for multicast path reservation hangs (Wormhole).**

On Wormhole, multicast transactions acquire path reservations that must be released. If a core issues a multicast but fails to complete the sequence, the path reservation is never released, blocking all subsequent multicasts on that path.

**Step 7: If `RP2W` -- NOC command buffer hardware stall.**

`RP2W` means the core is waiting for `NOC_CMD_CTRL == NOC_CTRL_STATUS_READY`, indicating the NOC hardware command buffer is full or hung. This is typically a hardware-level issue:
- Check if the same hang occurs on a different chip (hardware fault vs. software trigger).
- If reproducible on multiple chips: likely a software pattern that saturates the NOC command buffer. Add barriers between transaction bursts.
- If only on one chip: likely a hardware defect. See [Section 05](./05_distinguishing_hw_vs_sw_bugs.md).

### NOC Hang Diagnosis Checklist

```
NOC HANG DIAGNOSIS
==================
[ ] Check for SANITIZE violation (deliberate hang)
[ ] If no sanitize: check NOC transaction counters (check_noc_status)
[ ] Check pending NOC addresses (check_noc_locations)
[ ] Enable NOC Debug Dump and re-run if needed
[ ] Check for BH-specific issues (inline write, relaxed ordering)
[ ] Check for WH multicast path reservation issues
[ ] If RP2W: test on different chip (HW vs SW)
```

### Cross-Reference

- NOC sanitization and violation details: [Chapter 2, `03_noc_address_sanitization_and_violations.md`](../ch02_kernel_and_noc_hangs/03_noc_address_sanitization_and_violations.md)
- NOC barrier and semaphore mechanics: [Chapter 2, `04_noc_barrier_and_semaphore_hangs.md`](../ch02_kernel_and_noc_hangs/04_noc_barrier_and_semaphore_hangs.md)
- DRAM backpressure causing barrier hangs: [Chapter 3, `02_dram_and_noc_backpressure.md`](../ch03_memory_related_hangs/02_dram_and_noc_backpressure.md)

---

## Dispatch Hang Diagnosis

**Entry criteria:** Dispatch cores show stuck waypoints (`HQW`, `PWW`, `WCW`, `CBRW`, `DCW`, `!CMD`), or the host received a fetch queue or completion queue timeout.

### Step-by-Step Procedure

**Step 1: Run dump_fast_dispatch for dispatch-specific state.**

```bash
./tools/tt-triage.py --run=dump_fast_dispatch --dev=0 --verbosity=4
```

This produces:
- Prefetch kernel state: is it waiting for host data (`HQW`), stalled (`STALLED`), or processing?
- Dispatch kernel state: is it waiting for prefetch, waiting for workers (`PWW`/`WCW`), or processing?
- Dispatch subordinate state: is it waiting for the master (`DCW`)?
- CQ fill levels: how many entries in the issue queue and completion queue?
- `cmd_ptr`: current command pointer (not advancing = dispatch stuck)
- `cb_fence`: indicates how much data dispatch has consumed
- `last_wait_count`: how many times dispatch entered a wait loop

**Step 2: Decode the dispatch waypoint.**

| Waypoint | Kernel | Meaning | Next Step |
|----------|--------|---------|-----------|
| `HQW` | Prefetch | Waiting for host to write commands to fetch queue | Check host-side: is it blocked before writing? Is `SystemMemoryManager` stuck? |
| `UAPW` | Prefetch | Waiting to read upstream data (relay topology) | Check upstream prefetcher state |
| `CNSW` | Prefetch | Waiting for "command not sent" acknowledgment | Check dispatch CB consumption |
| `PWW` / `PWD` | Dispatch | `process_wait`: waiting for worker notification semaphore | Workers may be hung; check worker waypoints |
| `WCW` / `WCD` | Dispatch | `write_and_check_completion_signal`: waiting for all workers | Same as PWW -- workers did not signal completion |
| `DCW` | Dispatch_S | Waiting for dispatch master notification | Check dispatch master state |
| `DAPW` / `DAPD` | Dispatch | Data-mover progress wait | Check data transfer completion |
| `CBRW` / `CBRD` | Dispatch | CB page release stall | CB between prefetch and dispatch is full; check prefetch consumption |
| `!CMD` | Any dispatch | Invalid command byte received | Likely memory corruption in CQ or command stream ([Ch4, Scenario 4.1.11](../ch04_dispatch_and_host_device_hangs/01_dispatch_architecture_and_hang_points.md)) |

**Step 3: For "dispatch waiting for workers" (PWW/WCW).**

This is the most common dispatch hang: the dispatch kernel sent go signals to workers, but one or more workers never signaled completion.

```
Procedure:
  a) Identify which workers are still running (not at waypoint 'D').
  b) For each stuck worker, apply the appropriate category diagnosis:
     - CRBW/CWFW -> Kernel CB Deadlock (above)
     - NRBW/NWBW -> NOC Hang (above)
     - NSW/NSMW -> Semaphore Deadlock (below)
  c) Fix the worker hang. The dispatch hang is a secondary symptom.
```

**Step 4: For "prefetch waiting for host" (HQW).**

The prefetch kernel consumed all available commands and is waiting for more. Check:
- Is the host blocked? (In `Synchronize`? In `fetch_queue_reserve_back`?)
- Has the host exited without sending the final command?
- Is there a CQ fill-level mismatch? (`dump_fast_dispatch` shows this)

This often indicates the host issued a `Synchronize()` and the device never wrote a completion signal -- the root cause is typically a worker hang that prevented dispatch from writing the completion.

**Step 5: For invalid command (!CMD).**

This is severe: the dispatch kernel received an unrecognized command byte. Causes:
- Memory corruption in the command queue or prefetch buffer
- Stale trace data being replayed after a program/layout change
- A bug in the host-side command encoding

Check if the hang is reproducible. If it happens only during trace replay, see [Chapter 4, `03_trace_replay_and_lightmetal.md`](../ch04_dispatch_and_host_device_hangs/03_trace_replay_and_lightmetal.md).

**Step 6: Use slow dispatch mode for isolation.**

Set `TT_METAL_SLOW_DISPATCH_MODE=1` and reproduce. If the hang disappears, the bug is in the dispatch infrastructure. If the hang persists, the bug is in the kernel/NOC logic.

### Dispatch Hang Diagnosis Checklist

```
DISPATCH HANG DIAGNOSIS
=======================
[ ] Run dump_fast_dispatch to read dispatch kernel state
[ ] Decode the dispatch waypoint code
[ ] If PWW/WCW: identify stuck worker cores and diagnose those
[ ] If HQW: check host-side blocking and CQ fill levels
[ ] If !CMD: check for memory corruption or stale trace data
[ ] Test with slow dispatch mode for isolation
[ ] Check for trace replay issues (if applicable)
```

### Cross-Reference

- Full dispatch hang scenario catalog: [Chapter 4, `01_dispatch_architecture_and_hang_points.md`](../ch04_dispatch_and_host_device_hangs/01_dispatch_architecture_and_hang_points.md) (Scenarios 4.1.1-4.1.14)
- Host synchronization and timeout: [Chapter 4, `02_host_synchronization_and_timeout_detection.md`](../ch04_dispatch_and_host_device_hangs/02_host_synchronization_and_timeout_detection.md)
- Trace replay hangs: [Chapter 4, `03_trace_replay_and_lightmetal.md`](../ch04_dispatch_and_host_device_hangs/03_trace_replay_and_lightmetal.md)

---

## Memory Corruption Diagnosis

**Entry criteria:** Watcher reports L1 corruption (`DumpL1Status` failure), `check_binary_integrity` or `check_core_magic` fails, or the hang symptoms are inconsistent/cascading (different cores stuck at different unrelated waypoints).

### Step-by-Step Procedure

**Step 1: Check for L1 address-0 corruption.**

The watcher's `DumpL1Status()` check reads the firmware launch address at L1 address 0. If this value has been overwritten, memory corruption has occurred. The tt-triage `check_core_magic` script detects when the mailbox's `core_magic_number` does not match the expected firmware type.

**Step 2: Check for stack overflow.**

If watcher `STACK_USAGE` tracking is enabled, check for stack overflow reports:

```bash
grep -i "stack" generated/watcher/watcher.log
```

Stack overflow on a RISC-V processor overwrites adjacent L1 data structures, causing secondary hangs. See [Chapter 3, `01_l1_memory_corruption_and_overflow.md`](../ch03_memory_related_hangs/01_l1_memory_corruption_and_overflow.md).

**Step 3: Enable CB sanitization to detect out-of-bounds access.**

```bash
export TT_METAL_WATCHER=120
# Ensure CB sanitize is NOT disabled:
# unset TT_METAL_WATCHER_DISABLE_CB_SANITIZE
```

Re-run the workload. The watcher iterates all 32 CBs per core on every NOC transaction to check if any transaction target falls within an active CB's address range.

**Step 4: Verify buffer allocation boundaries.**

If the hang involves NOC transactions to addresses that "look reasonable" but are slightly wrong, check:
- Host-side logs for OOM warnings from the L1 allocator (`free_list_opt.cpp`).
- L1 usage near capacity for the failing core.
- Use `TT_METAL_CLEAR_L1=1` to zero all L1 before each program -- this converts silent corruption into predictable failures.

**Step 5: Check for DRAM-related corruption.**

If NOC transactions target DRAM addresses:
- Verify DRAM address alignment requirements (see [Chapter 3, `03_alignment_and_tile_size_mismatches.md`](../ch03_memory_related_hangs/03_alignment_and_tile_size_mismatches.md)).
- Check for DRAM backpressure (see [Chapter 3, `02_dram_and_noc_backpressure.md`](../ch03_memory_related_hangs/02_dram_and_noc_backpressure.md)).

**Step 6: For cascading failures.**

When corruption is widespread, the initial corruption event may have happened much earlier. Use binary search with `Synchronize()` checkpoints ([Section 03](./03_narrowing_and_reproducing.md)) to find the first operation that corrupts L1. Then enable all sanitization and re-run with just that operation.

### Watcher Data Corruption Indicators

These errors indicate that the watcher's own diagnostic data structures have been corrupted, usually by a stray NOC write or stack overflow. They are symptoms, not root causes.

| Error Message | Root Cause |
|---|---|
| `Watcher data corrupted, unexpected processor index <N>` | Memory corruption overwrote the `which_risc` field |
| `Watcher data corrupted, unprintable character <N>` | Waypoint data contains non-printable bytes |
| `Watcher data corruption, unexpected run state: <N>` | Run state field contains unrecognized value |
| `Watcher data corruption, unexpected launch mode: <N>` | Launch mode is neither DEV nor HOST |
| `Watcher data corruption, unexpected brisc noc_id: <N>` | BRISC NOC ID is neither 0 nor 1 |
| `Watcher data corruption, unexpected kernel id: <N>` | Kernel ID exceeds number of registered kernels |

For all watcher data corruption errors: something else corrupted L1 first. Identify what could write to the mailbox region (stray NOC transactions, stack overflow, CB overflow). Enable all NOC sanitization features and reproduce.

### Memory Corruption Diagnosis Checklist

```
MEMORY CORRUPTION DIAGNOSIS
============================
[ ] Check for L1 address-0 corruption (watcher DumpL1Status)
[ ] Check for core magic mismatch (check_core_magic)
[ ] Check for stack overflow (watcher STACK_USAGE)
[ ] Enable CB sanitization and re-run
[ ] Check buffer allocation boundaries
[ ] Check DRAM address alignment
[ ] Use TT_METAL_CLEAR_L1=1 to expose initialization bugs
[ ] For cascading failures: binary search to find first corruption
```

### Cross-Reference

- Full L1 corruption catalog: [Chapter 3, `01_l1_memory_corruption_and_overflow.md`](../ch03_memory_related_hangs/01_l1_memory_corruption_and_overflow.md)
- Alignment violations: [Chapter 3, `03_alignment_and_tile_size_mismatches.md`](../ch03_memory_related_hangs/03_alignment_and_tile_size_mismatches.md)
- Allocation failures and silent OOM: [Chapter 3, `04_allocation_failures_and_silent_oom.md`](../ch03_memory_related_hangs/04_allocation_failures_and_silent_oom.md)

---

## Multi-Chip Hang Diagnosis

**Entry criteria:** The workload runs on multiple chips (T3K, Galaxy, or N300), and the hang involves Ethernet-connected devices. Symptoms: some chips appear idle while others are stuck, `check_eth_status.py` reports a link down, ERISC cores are stuck at fabric wait points, or a CCL collective operation is the running op.

### Step-by-Step Procedure

**Step 1: Check Ethernet link status on ALL devices.**

```bash
./tools/tt-triage.py --run=check_eth_status --dev=0 --verbosity=4
# Repeat for all devices: --dev=1, --dev=2, etc.
```

Key indicators:

| Indicator | Healthy Value | Stuck Indicator |
|-----------|---------------|-----------------|
| Port Status | Up | Down |
| Retrain Count | 0 | > 0 (link instability) |
| RX Link Up | Up | Down |
| Heartbeat | Active (changing) | Inactive (dead ERISC) |

If a link is down: all operations depending on that link will hang. Check `logical_core_to_eth_link_retraining_count` in the watcher output for retraining events.

**Step 2: Check fabric router state.**

```bash
python3 tools/triage/fabric_erisc_dumper.py --dev=0 --polling
```

Check flow control counters, heartbeat TX/RX, and router mux state.

**Step 3: Procedure for Ethernet Link Failure.**

If `check_eth_status` reports a link down:
1. Note which ERISC core and which port failed.
2. Check the remote end -- is the corresponding ERISC on the peer device also reporting link down?
3. Check watcher for `link_down = 1` in the ERISC mailbox.
4. Check for retraining: if retrain count > 0, the link is physically marginal. See [Section 05, Ethernet Retrain Checks](./05_distinguishing_hw_vs_sw_bugs.md#ethernet-retrain-checks).
5. For immediate workaround: `TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1`.

**Step 4: Procedure for EDM/Fabric Router Stalls.**

If ERISC cores are alive (heartbeat active) but stuck at fabric wait points:
1. Run `fabric_erisc_dumper.py` to inspect stream register values.
2. If sender BUF_SPACE_AVAILABLE is zero: the sender is blocked because the receiver has not consumed data.
3. Trace the chain to find the root cause: the receiving side's worker has not pulled data from the EDM output buffer.
4. Check for worker-to-EDM connection races ([Chapter 5, Scenario 5.1.2](../ch05_multi_chip_and_ccl_hangs/01_ethernet_and_fabric_fundamentals.md)).

**Step 5: Procedure for CCL Collective Hangs.**

**Signal:** The host is blocked inside a CCL operation, and multiple devices show CCL worker cores stuck at `NSW`.

1. **Determine the collective type and topology.** From the host stack or kernel names, identify which CCL operation, Ring or Linear topology, and number of devices.
2. **Check all devices simultaneously.** For each device, record CCL worker core waypoints, ERISC core states, and whether the device completed its local portion.
3. **Identify the "odd one out."** The device in a different state from all others is the likely root cause. If 7 of 8 devices are at `NSW` and 1 is idle or shows a NOC violation, the 1 device is the problem.
4. **Check for parameter mismatches.** All devices must pass identical parameters to the collective (tensor shape, gather dimension, topology). A mismatch on any single device causes a hang.
5. **Check the termination protocol.** CCL operations use a master-slave synchronization pattern. If the master is stuck at `noc_semaphore_wait(master_l1_semaphore_addr, num_workers_to_sync - 1)`, one or more slave workers did not complete.
6. **Check for semaphore cycling mismatches.** Async CCL operations use double-buffered semaphore handles. If a trace replay is involved, the host-side semaphore counter may be out of sync. See [Chapter 5, `02_ccl_collective_operation_hangs.md`](../ch05_multi_chip_and_ccl_hangs/02_ccl_collective_operation_hangs.md).

**Step 6: Procedure for Topology/Mesh Misconfiguration.**

If the hang occurs on the first or last device:
1. Verify mesh connectivity matches logical topology.
2. Check if Ring topology was specified on a Linear fabric (or vice versa).
3. Check submesh correctness -- verify included devices are actually connected via Ethernet.

**Step 7: Procedure for Galaxy-Scale Distributed Failures.**

1. Coordinate across all hosts. Ensure all have collected watcher and ERISC data before any reset.
2. Identify the host boundary -- which inter-host links are involved?
3. Check `dmesg` on every host for IOMMU faults, PCIe errors, or kernel panics.
4. Check fabric reliability mode -- if `STRICT_SYSTEM_HEALTH_SETUP_MODE` is active, verify device count matches topology expectations.
5. Trace the ring at the Galaxy level to find which device broke the ring.

### Multi-Chip Hang Diagnosis Checklist

```
MULTI-CHIP HANG DIAGNOSIS
==========================
[ ] Check Ethernet link status on ALL devices (check_eth_status)
[ ] Check ERISC heartbeats on all devices
[ ] Run fabric_erisc_dumper on suspected devices
[ ] For link failure: check both ends of the failing link
[ ] For CCL: check all devices simultaneously, find "odd one out"
[ ] For CCL: verify parameter consistency across all devices
[ ] For CCL: check termination protocol and semaphore cycling
[ ] For topology issues: verify physical-to-logical mapping
[ ] For Galaxy: coordinate across all hosts, check dmesg on each
[ ] Consider TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 as workaround
```

### Cross-Reference

- Ethernet and fabric fundamentals: [Chapter 5, `01_ethernet_and_fabric_fundamentals.md`](../ch05_multi_chip_and_ccl_hangs/01_ethernet_and_fabric_fundamentals.md)
- CCL collective operation hangs: [Chapter 5, `02_ccl_collective_operation_hangs.md`](../ch05_multi_chip_and_ccl_hangs/02_ccl_collective_operation_hangs.md)
- Topology and mesh configuration: [Chapter 5, `03_topology_and_mesh_configuration_hangs.md`](../ch05_multi_chip_and_ccl_hangs/03_topology_and_mesh_configuration_hangs.md)

---

## Semaphore Deadlock Diagnosis

**Entry criteria:** Watcher or triage shows one or more cores stuck at `NSW` (noc_semaphore_wait) or `NSMW` (noc_semaphore_wait_min).

### Step-by-Step Procedure

**Step 1: Identify the semaphore address.**

From the kernel source, find the L1 address of the semaphore being waited on.

**Step 2: Determine the expected signaling pattern.**

For each core waiting on a semaphore:
- Which core is supposed to increment it? (Via `noc_semaphore_inc`)
- What value is the waiting core expecting? (`noc_semaphore_wait` checks `== val`, `noc_semaphore_wait_min` checks `>= val`)
- Is the signaling core running? Check its waypoint.

**Step 3: Check for common semaphore protocol violations.**

| Violation | Symptom | Root Cause | How to Detect |
|-----------|---------|------------|---------------|
| Wrong target core | Core A waits, but Core B increments the semaphore on Core C | Incorrect NOC coordinates in `noc_semaphore_inc` | Check NOC target address in increment call |
| Wrong semaphore address | Core A waits on address X, but incrementer targets address Y | Mismatched semaphore allocation | Compare addresses in wait vs. increment calls |
| Wrong initial value | Semaphore starts at non-zero from a previous iteration | Missing `noc_semaphore_set` reset | Read semaphore at start of kernel; non-zero = stale |
| Wrong increment count | `noc_semaphore_wait(sem, 4)` but only 3 increments sent | Off-by-one in signaling loop | Check increment vs. wait values |
| `wait` vs. `wait_min` | Using `noc_semaphore_wait` (equality) when increments arrive in bursts | Value can skip over target | Switch to `noc_semaphore_wait_min` |
| Not reset between iterations | Semaphore from previous iteration lingers | Missing reset at iteration start | Check for `noc_semaphore_set` call |
| Signaling core exited | Core supposed to signal has already finished | Execution ordering bug | Check signaling core's waypoint -- if `D`, it exited |

**Step 4: If the semaphore is used for cross-chip coordination (CCL).**

Cross-chip semaphore increments travel over Ethernet. If the link drops, the increment is lost. Check Ethernet link status (see [Multi-Chip Hang Diagnosis](#multi-chip-hang-diagnosis)). Check that fabric sockets are properly initialized.

**Step 5: Read the current semaphore value.**

If you can access the hung device via `watcher_dump` or ttexalens:
- Value = 0, waiting for > 0: no increments received. Signaling core never ran or never reached the increment.
- Value = N, waiting for N+1: N increments arrived but N+1 did not. Signaling core may have hung before completing.

**Step 6: Use debug delays to expose races.**

If the semaphore deadlock is intermittent, use `TT_METAL_WRITE_DEBUG_DELAY_CORES` to slow down NOC writes on specific cores and make the race deterministic.

### Semaphore Deadlock Diagnosis Checklist

```
SEMAPHORE DEADLOCK DIAGNOSIS
==============================
[ ] Identify semaphore L1 address from callstack/source
[ ] Read current semaphore value (watcher_dump / ttexalens)
[ ] Compare to expected value (wait target)
[ ] Check: correct initial value (0)?
[ ] Check: correct target core in noc_semaphore_inc?
[ ] Check: correct L1 address in noc_semaphore_inc?
[ ] Check: increment count matches wait target?
[ ] Check: semaphore reset between iterations?
[ ] Check: wait vs wait_min semantics correct?
[ ] Check: signaling core still alive (not exited early)?
[ ] For multi-chip: check Ethernet link status
[ ] For intermittent: use write debug delays to expose race
```

### Cross-Reference

- Semaphore primitives and failure modes: [Chapter 2, `04_noc_barrier_and_semaphore_hangs.md`](../ch02_kernel_and_noc_hangs/04_noc_barrier_and_semaphore_hangs.md)
- CCL semaphore protocols: [Chapter 5, `02_ccl_collective_operation_hangs.md`](../ch05_multi_chip_and_ccl_hangs/02_ccl_collective_operation_hangs.md)
- Blocking primitives: [Chapter 1, `02_blocking_primitives_taxonomy.md`](../ch01_anatomy_of_a_hang/02_blocking_primitives_taxonomy.md) (NSW/NSMW entries)

---

## Host-Device Hang Diagnosis

**Entry criteria:** The device appears idle (all cores at waypoint `W` or `D`), but the host is blocked.

### Diagnosis Procedure

1. **Check the host call stack.** Attach GDB and identify where the host is blocked.
2. **Check for async dispatch queue deadlock.** If the host uses async dispatch, a deadlock can occur when the dispatch thread is blocked waiting for a device operation to complete, but the device has already completed and the completion signal was missed.
3. **Check the timeout value.** If `TT_METAL_OPERATION_TIMEOUT_SECONDS` is set very high (or unset), the host may simply not have timed out yet.
4. **Check for `synchronize_device` inside a trace.** If host code calls `ttnn.synchronize_device()` inside a traced region, the trace replay will block because host-device communication is not allowed inside a trace.

### Cross-Reference

- [Chapter 4, `02_host_synchronization_and_timeout_detection.md`](../ch04_dispatch_and_host_device_hangs/02_host_synchronization_and_timeout_detection.md)

---

## Waypoint-to-Category Quick Reference Card

Print this card for immediate category routing based on waypoint codes.

```
WAYPOINT-TO-CATEGORY ROUTING
==============================

WORKER CORE WAYPOINTS:
  CRBW  --> CB deadlock (producer waiting)     --> 7.2 CB Deadlock
  CWFW  --> CB deadlock (consumer waiting)     --> 7.2 CB Deadlock
  NRBW  --> NOC read barrier hang              --> 7.2 NOC Hang
  NWBW  --> NOC write barrier hang             --> 7.2 NOC Hang
  RP2W  --> NOC command buffer stall           --> 7.2 NOC Hang
  NSW   --> Semaphore wait (exact match)       --> 7.2 Semaphore Deadlock
  NSMW  --> Semaphore wait (minimum)           --> 7.2 Semaphore Deadlock
  GW    --> Go-Wait (waiting for dispatch)     --> 7.2 Dispatch Hang
  NTW   --> NCRISC/TRISC subordinate wait      --> Ch 2, 2.1.x
  NBTW  --> NOC barrier with TRID              --> 7.2 NOC Hang
  NWTW  --> NOC write barrier with TRID        --> 7.2 NOC Hang

DISPATCH CORE WAYPOINTS:
  HQW   --> Prefetch waiting for host          --> 7.2 Dispatch Hang
  PWW   --> Dispatch waiting for workers       --> 7.2 Dispatch Hang
  WCW   --> Dispatch completion check          --> 7.2 Dispatch Hang
  CBRW  --> Dispatch CB page release stall     --> 7.2 Dispatch Hang
  DCW   --> Dispatch subordinate sync          --> 7.2 Dispatch Hang
  DAPW  --> Dispatch data-mover wait           --> 7.2 Dispatch Hang
  !CMD  --> Invalid dispatch command           --> 7.2 Dispatch Hang

ETHERNET CORE WAYPOINTS:
  RW    --> ERISC router wait                  --> 7.2 Multi-Chip Hang
  SEW   --> ERISC semaphore/event wait         --> 7.2 Multi-Chip Hang

GENERAL STATE:
  I     --> Init (RISC initializing)
  W     --> Wait (firmware idle loop)
  R     --> Run (executing kernel code)
  D     --> Done (kernel execution complete)

NO WAYPOINT DATA:
  (empty or stale)  --> Watcher not enabled or core not reached
                        Use watcher_dump, check Step 4 of triage
```

---

## Diagnosis Routing Summary

```
HANG DETECTED
  |
  +-- Watcher/triage evidence available?
  |     |
  |     +-- SANITIZE violation --> Decode violation (Section 04), go to root cause chapter
  |     +-- ASSERT tripped --> Decode assert (Section 04), fix kernel code
  |     +-- CRBW / CWFW --> Kernel CB Deadlock (above)
  |     +-- NRBW / NWBW / RP2W --> NOC Hang (above)
  |     +-- NSW / NSMW --> Semaphore Deadlock (above) or Multi-Chip (if CCL)
  |     +-- HQW / PWW / WCW / DCW / CBRW / !CMD --> Dispatch Hang (above)
  |     +-- Link down / ETH anomaly / ERISC stuck --> Multi-Chip Hang (above)
  |     +-- L1 corruption / cascading failures --> Memory Corruption (above)
  |     +-- All cores DONE, host stuck --> Host-Device Hang (above)
  |
  +-- No clear evidence?
        --> Section 03: Narrowing and Reproducing
            (binary search, null_kernels, slow dispatch, etc.)
```

---

**Next:** [03_narrowing_and_reproducing.md](./03_narrowing_and_reproducing.md)
