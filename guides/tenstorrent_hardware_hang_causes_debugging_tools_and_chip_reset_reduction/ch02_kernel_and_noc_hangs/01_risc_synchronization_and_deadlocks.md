# RISC Synchronization and Deadlocks

The coordination between the RISC-V processors inside a single Tensix core is orchestrated by firmware protocols implemented in `brisc.cc`, `ncrisc.cc`, and the TRISC startup code. These protocols use shared mailbox memory in L1 and follow a strict state-machine model. When any step in these protocols fails -- because a subordinate core crashes, a signal is missed, or a race condition corrupts mailbox state -- the result is a firmware-level hang that precedes any user kernel execution.

This section covers every synchronization point in the BRISC firmware main loop and the ERISC active firmware, documenting each as a potential hang site. It also covers Quasar's extended subordinate model.

**Prerequisites:** [Chapter 1, `01_what_is_a_hang.md`](../ch01_anatomy_of_a_hang/01_what_is_a_hang.md) (RISC-V wait-loop model, kernel lifecycle), [Chapter 1, `02_blocking_primitives_taxonomy.md`](../ch01_anatomy_of_a_hang/02_blocking_primitives_taxonomy.md) (GW, NTW, W waypoints), [Chapter 1, `04_hang_causes_across_architectures.md`](../ch01_anatomy_of_a_hang/04_hang_causes_across_architectures.md) (WH NCRISC IRAM, BH dynamic NOC, Quasar DM architecture).

Reference files: `tt_metal/hw/firmware/src/tt-1xx/brisc.cc`, `tt_metal/hw/firmware/src/tt-1xx/ncrisc.cc`, `tt_metal/hw/firmware/src/tt-1xx/active_erisc.cc`, `tt_metal/hw/inc/hostdev/dev_msgs.h`, `tt_metal/hw/firmware/src/tt-2xx/dm.cc`

---

## The Subordinate Synchronization Model

BRISC is the orchestrator of each Tensix core. It is the first processor to start after device initialization and the last to signal completion. NCRISC and the three TRISC cores (TRISC0/TRISC1/TRISC2) are **subordinates** -- they wait for BRISC to tell them what to do, execute their assigned kernel, and then signal back to BRISC that they are done.

The synchronization mailbox is a shared L1 data structure:

```c++
// tt_metal/hw/firmware/src/tt-1xx/brisc.cc
tt_l1_ptr subordinate_map_t* const subordinate_sync =
    (subordinate_map_t*)mailboxes->subordinate_sync.map;
```

The `subordinate_sync` structure is a union where each byte represents one subordinate core's state:

| Byte Index | Processor | Field Name |
|---|---|---|
| 0 | NCRISC (DM1) | `dm1` |
| 1 | TRISC0 (Unpack) | `trisc0` |
| 2 | TRISC1 (Math) | `trisc1` |
| 3 | TRISC2 (Pack) | `trisc2` |

The full 32-bit word `subordinate_sync->all` can be compared in a single instruction against `RUN_SYNC_MSG_ALL_SUBORDINATES_DONE` (value `0x00000000`). When all four bytes are zero, every subordinate has finished.

### Synchronization Signal Values

| Signal | Value | Meaning |
|--------|-------|---------|
| `RUN_SYNC_MSG_DONE` | `0x00` | Core has finished its current task |
| `RUN_SYNC_MSG_LOAD` | `0x01` | Core should load CBs (and IRAM on WH) |
| `RUN_SYNC_MSG_WAITING_FOR_RESET` | `0x02` | WH NCRISC: ready to be reset to IRAM address |
| `RUN_SYNC_MSG_INIT_SYNC_REGISTERS` | `0x03` | Core should initialize hardware sync registers |
| `RUN_SYNC_MSG_INIT` | `0x40` | Core is initializing |
| `RUN_SYNC_MSG_GO` | `0x80` | Core should begin kernel execution |

### The Kernel Launch Sequence

The following sequence occurs for every kernel invocation on a Tensix core. Each numbered step is a potential hang point:

```
Step 1:  BRISC receives RUN_MSG_GO from dispatcher        [GW -> GD]
Step 2:  BRISC sets NCRISC subordinate_sync to RUN_SYNC_MSG_LOAD
Step 3:  NCRISC wakes from its W loop, begins CB load + IRAM copy (WH only)
Step 4:  BRISC sets up local CB interfaces
Step 5:  BRISC sets TRISCs to RUN_SYNC_MSG_GO via run_triscs()
Step 6:  [Non-WH] BRISC sets NCRISC to RUN_SYNC_MSG_GO
         [WH] BRISC waits for NCRISC to signal WAITING_FOR_RESET, then resets it
Step 7:  BRISC sets up remote CB interfaces + barrier (BH: NABW)
Step 8:  BRISC executes its own kernel                    [R -> D]
Step 9:  BRISC calls wait_ncrisc_trisc()                  [NTW -> NTD]
Step 10: BRISC signals RUN_MSG_DONE to dispatcher
```

> **Danger:** Steps 2 and 5 contain spin loops that **lack WAYPOINT markers** -- the NCRISC IRAM handshake (`while (subordinate_sync->dm1 != RUN_SYNC_MSG_WAITING_FOR_RESET)`) and the TRISC0 init wait (`while (subordinate_sync->trisc0 != RUN_SYNC_MSG_DONE)`) in `run_triscs()`. If these loops hang, the watcher will show a stale waypoint on BRISC from a preceding phase. Always check `subordinate_sync` byte values directly when BRISC appears stuck between waypoints.

---

## Hang Cause 2.1.1: BRISC Waiting for Subordinates (`NTW`)

### Symptom

BRISC is stuck at waypoint `NTW`. One or more subordinate cores (NCRISC, TRISC0, TRISC1, TRISC2) have not signaled `RUN_SYNC_MSG_DONE`. The subordinate core(s) will show their own wait waypoint (`CRBW`, `CWFW`, `NRBW`, `NSW`, etc.) or may still be at waypoint `R` (running).

### Root Cause

The `wait_ncrisc_trisc()` function in `brisc.cc` spins until every subordinate byte in the `subordinate_sync` word equals zero:

```c++
// tt_metal/hw/firmware/src/tt-1xx/brisc.cc
inline void wait_ncrisc_trisc() {
    WAYPOINT("NTW");
    while (subordinate_sync->all != RUN_SYNC_MSG_ALL_SUBORDINATES_DONE) {
#if defined(ARCH_WORMHOLE)
        // Avoid hammering L1 while other cores are trying to work.
        asm volatile("nop; nop; nop; nop; nop");
#endif
        invalidate_l1_cache();
    }
    WAYPOINT("NTD");
}
```

This hang is always a **derived hang** -- BRISC itself is not the root cause. The root cause is in whichever subordinate has not yet signaled done. That subordinate is either:

- Stuck in a user kernel blocking call (CB wait, NOC barrier, semaphore wait)
- Stuck in its own firmware synchronization (e.g., NCRISC waiting for IRAM load)
- Crashed or stuck in `assert_and_hang`

### Diagnosis Steps

1. Read the watcher waypoints for all five RISC-V cores on the affected Tensix.
2. BRISC will show `NTW`. Identify which subordinate(s) are NOT at waypoint `D` (done).
3. For each non-done subordinate, examine its waypoint to determine which blocking primitive it is stuck at.
4. Classify the subordinate's hang using the appropriate section of this chapter or Chapter 1's blocking primitives taxonomy.

### Fix

Fix the root cause in the subordinate core. The `NTW` waypoint on BRISC will automatically resolve once all subordinates complete.

### Prevention

- Always verify that every kernel launched on subordinate cores will terminate. Every `cb_wait_front` must have a matching `cb_push_back` from the producer. Every `noc_async_read_barrier` must have all reads properly issued.
- If only certain RISC-V cores are enabled for a given kernel launch (via the `enables` bitmask in the launch message), the disabled cores are set to `RUN_SYNC_MSG_DONE` by default and will not block BRISC.

---

## Hang Cause 2.1.2: Wormhole NCRISC IRAM Halt-Reset Hang

### Symptom

BRISC is spinning in a tight loop *without a waypoint marker*, waiting for `subordinate_sync->dm1 == RUN_SYNC_MSG_WAITING_FOR_RESET`. The watcher cannot directly identify this spin because there is no `WAYPOINT` call in the waiting code. BRISC will appear stuck between waypoints, potentially showing a stale waypoint from an earlier phase. NCRISC may show waypoint `W` (its own wait state) or no waypoint if it crashed before reaching the sync point.

### Root Cause

On Wormhole, NCRISC executes kernels from a dedicated Instruction RAM (IRAM) rather than directly from L1. The startup sequence requires a handshake between BRISC and NCRISC:

```c++
// tt_metal/hw/firmware/src/tt-1xx/brisc.cc
inline void start_ncrisc_kernel_run(uint32_t enables) {
#if defined(ARCH_WORMHOLE)
    if (enables & ...DM1...) {
        // The NCRISC behaves badly if it jumps from L1 to IRAM,
        // so instead halt it and then reset it to the IRAM address it provides.
        while (subordinate_sync->dm1 != RUN_SYNC_MSG_WAITING_FOR_RESET);
        subordinate_sync->dm1 = RUN_SYNC_MSG_GO;
        volatile tt_reg_ptr uint32_t* cfg_regs = core.cfg_regs_base(0);
        cfg_regs[NCRISC_RESET_PC_PC_ADDR32] = mailboxes->ncrisc_halt.resume_addr;
        assert_just_ncrisc_reset();
        // Wait a bit to ensure NCRISC has time to actually reset
        // (otherwise it may just continue where it left off).
        // This wait value was chosen empirically.
        riscv_wait(5);
        deassert_all_reset();
    }
#endif
}
```

The protocol has three distinct failure modes:

**Failure Mode A: DMA stall.** NCRISC copies its kernel from L1 to IRAM via DMA. If the DMA copy never completes (L1 corruption, bad kernel image, DMA engine hang), NCRISC never writes `mailboxes->ncrisc_halt.resume_addr` and never sets `subordinate_sync->dm1 = RUN_SYNC_MSG_WAITING_FOR_RESET`.

**Failure Mode B: Reset delay insufficient.** After asserting the NCRISC reset line, BRISC waits `riscv_wait(5)` before deasserting. The comment is explicit: "This wait value was chosen empirically." If the delay is insufficient (silicon variation, temperature), NCRISC may "continue where it left off" instead of resetting properly, executing stale IRAM content.

**Failure Mode C: Branch predictor contamination.** On WH, the NCRISC branch predictor may retain state from the L1-based firmware code. After reset to the IRAM address, the branch predictor can mispredict based on stale entries, causing the kernel entry sequence to jump to incorrect addresses. The NCRISC reset sequence includes 13 repetitions of mispredict-inducing branches (~54 instructions, ~110 cycles) specifically to flush the branch predictor. If this flush is interrupted or incomplete, kernel entry can hang.

### Diagnosis Steps

1. Identify a Wormhole system where BRISC appears stuck between `GD` and `R` waypoints.
2. Check `subordinate_sync->dm1` -- if it is not `0x02` (`RUN_SYNC_MSG_WAITING_FOR_RESET`), BRISC is stuck in the IRAM handshake.
3. Check NCRISC waypoint: if it shows `W` (waiting for BRISC notification), NCRISC has not yet been told to load. If it shows no waypoint, it may have crashed during DMA.
4. Inspect `mailboxes->ncrisc_halt.resume_addr`: if it is `0`, NCRISC never wrote its resume address.

### Fix

- If the kernel image is corrupted: rebuild and re-flash. Ensure `ncrisc_kernel_size16` in the launch message correctly represents the kernel size.
- If the DMA copy failed: investigate L1 memory integrity; the source data for the IRAM copy may be corrupted.
- If the reset timing is suspect (non-deterministic failures): the hang may correlate with chip temperature or clock frequency.

### Prevention

- This pattern is Wormhole-only. On GS and BH, NCRISC executes from L1 directly, and `start_ncrisc_kernel_run_early()` simply sets `subordinate_sync->dm1 = RUN_SYNC_MSG_GO` without the reset dance.
- When developing WH-specific kernels, validate that the kernel binary fits in IRAM and that the `ncrisc_kernel_size16` field is set correctly.

---

## Hang Cause 2.1.3: Wormhole NCRISC Reset Timing Failure

### Symptom

NCRISC appears to execute stale or corrupted code after being reset. It may produce incorrect NOC transactions, write to wrong addresses, or hang at unexpected waypoints. This manifests non-deterministically and is more common under high L1 contention.

### Root Cause

After asserting the NCRISC reset line, BRISC waits a fixed delay before deasserting:

```c++
// tt_metal/hw/firmware/src/tt-1xx/brisc.cc
assert_just_ncrisc_reset();
// Wait a bit to ensure NCRISC has time to actually reset (otherwise it
// may just continue where it left off). This wait value was chosen
// empirically.
riscv_wait(5);
deassert_all_reset();
```

If the delay is insufficient due to silicon variation, temperature, or clock frequency, NCRISC may "continue where it left off" instead of resetting properly, executing whatever was previously in IRAM or executing from an incorrect PC.

### Diagnosis Steps

1. Look for NCRISC executing unexpected code paths or producing incorrect NOC transactions.
2. Confirm the system is Wormhole. This failure mode does not exist on other architectures.
3. If `riscv_wait(5)` is suspect, the hang may be non-deterministic and correlate with chip temperature or clock frequency.

### Fix

This is a known hardware timing concern. The firmware authors selected `riscv_wait(5)` as sufficient for current silicon. If a new silicon revision changes timing, this value may need adjustment.

### Prevention

- On non-WH architectures, this entire code path does not exist.
- If porting to new WH silicon revisions, validate that the reset timing is still adequate.

---

## Hang Cause 2.1.4: TRISC Initialization Deadlock

### Symptom

BRISC is stuck in `run_triscs()`, waiting for TRISC0 to finish its initialization synchronization registers setup. The watcher may show BRISC at a waypoint between `GD` and `R`, or may show a stale waypoint. This spin loop has **no WAYPOINT marker**.

### Root Cause

Before launching TRISC kernels, BRISC waits for TRISC0 to complete its hardware synchronization register initialization:

```c++
// tt_metal/hw/firmware/src/tt-1xx/brisc.cc
inline void run_triscs(uint32_t enables) {
    // Wait for init_sync_registers to complete.
    // Should always be done by the time we get here.
    while (subordinate_sync->trisc0 != RUN_SYNC_MSG_DONE) {
        invalidate_l1_cache();
    }

    if (enables & ...) {
        subordinate_sync->trisc0 = RUN_SYNC_MSG_GO;
        subordinate_sync->trisc1 = RUN_SYNC_MSG_GO;
        subordinate_sync->trisc2 = RUN_SYNC_MSG_GO;
    }
}
```

The synchronization register initialization is triggered at the end of the previous kernel iteration by `trigger_sync_register_init()`. TRISC0 must execute the initialization and write `RUN_SYNC_MSG_DONE` before BRISC reaches `run_triscs()` on the next iteration. If TRISC0 is slow or crashed during the previous kernel, it will never signal done.

### Diagnosis Steps

1. Check `subordinate_sync->trisc0` value. If it is `0x03` (`RUN_SYNC_MSG_INIT_SYNC_REGISTERS`), TRISC0 has not yet started or completed the initialization.
2. This spin loop has no WAYPOINT, so the watcher will show a stale waypoint on BRISC.
3. Check if TRISC0 has an assert in its mailbox from the previous kernel iteration.

### Fix

Address whatever caused TRISC0 to crash or fail to complete sync register initialization. This is typically a compute kernel bug from the previous iteration.

### Prevention

Ensure that compute kernels on TRISC0 do not corrupt the firmware-managed synchronization register initialization code path. Avoid writing to reserved L1 regions from user kernels.

---

## Hang Cause 2.1.5: Go-Signal Hang (`GW`)

### Symptom

BRISC is stuck at waypoint `GW`. All subordinate cores may be at `W` (their own wait states) or at `D` (done from the previous kernel). The watcher shows the core has been at `GW` for an extended period. No kernel execution has begun.

### Root Cause

The BRISC main loop waits for the dispatcher to send a go signal:

```c++
// tt_metal/hw/firmware/src/tt-1xx/brisc.cc
while (
    ((go_message_signal = mailboxes->go_messages[mailboxes->go_message_index].signal)
        != RUN_MSG_GO) &&
    !(mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config.preload
        & DISPATCH_ENABLE_FLAG_PRELOAD)) {
    invalidate_l1_cache();
    // Handle RUN_MSG_RESET_READ_PTR, RUN_MSG_REPLAY_TRACE ...
}
```

This loop exits when the go signal equals `RUN_MSG_GO` (value `0x80`) or the preload flag is set. If neither condition is satisfied:
- The dispatch core is itself hung (unable to send the go signal via NOC multicast)
- The go signal NOC write was lost or corrupted
- The `go_message_index` is out of sync between dispatcher and worker
- The launch message ring buffer state is corrupted

### Diagnosis Steps

1. Confirm BRISC is at `GW`.
2. Read `mailboxes->go_messages[go_message_index].signal`. If it is `RUN_MSG_DONE` (0x00) or `RUN_MSG_INIT` (0x40), the go signal was never received.
3. Check the dispatch core's waypoint. If the dispatch core is also hung (at `NWBW`, `NSW`, etc.), the root cause is in the dispatch path (see Chapter 4).
4. Check `go_message_index` on both the dispatcher and the worker to verify they agree.

### Fix

- If the dispatch core is hung, fix the dispatch hang first.
- If the NOC write delivering the go signal failed, investigate the NOC path between the dispatch core and the worker.
- If `go_message_index` is corrupted, investigate L1 memory integrity.

### Prevention

- The firmware includes a comment noting that "we also have a barrier before mcasting the launch message (as a hang workaround)" -- there is already a known workaround in the dispatch path to prevent go-signal delivery failures.
- Ensure the dispatch system is functioning correctly before diagnosing worker-side `GW` hangs.

---

## Hang Cause 2.1.6: Launch Message Ring Buffer Corruption

### Symptom

BRISC exits the `GW` loop but reads corrupted or stale launch message data. This may cause incorrect `enables` values (launching wrong subordinates), incorrect `noc_index`, or wrong kernel text offsets. The subsequent kernel execution may hang at any blocking primitive.

### Root Cause

The launch message system uses an 8-entry ring buffer:

```c++
// tt_metal/hw/inc/hostdev/dev_msgs.h
constexpr uint32_t launch_msg_buffer_num_entries = 8;

struct mailboxes_t {
    ...
    volatile uint32_t launch_msg_rd_ptr;
    struct launch_msg_t launch[launch_msg_buffer_num_entries];
    volatile struct go_msg_t go_messages[go_message_num_entries];
    ...
};
```

The read pointer advances after each kernel completes:

```c++
mailboxes->launch_msg_rd_ptr =
    (launch_msg_rd_ptr + 1) & (launch_msg_buffer_num_entries - 1);
```

Corruption can occur if:
- The `launch_msg_rd_ptr` is reset at the wrong time (via `RUN_MSG_RESET_READ_PTR`)
- The dispatcher writes a new launch message to a slot that BRISC has not yet consumed
- L1 corruption (from a stray NOC write) overwrites part of the launch message

The firmware uses `DISPATCH_ENABLE_FLAG_PRELOAD` in the `preload` field as a data-ready indicator. The comment states: "kernel_configs.preload is last in the launch message. so other data is valid by the time it's set."

### Diagnosis Steps

1. Dump the launch message ring buffer from L1.
2. Compare `launch_msg_rd_ptr` with the dispatcher's write pointer.
3. Inspect the `enables` field: does it match the intended kernel configuration?
4. Check if `kernel_text_offset` values point to valid kernel code.

### Fix

Ring buffer corruption typically indicates a dispatch-side bug or a stray NOC write. Check the `RUN_MSG_RESET_READ_PTR` handling path, or identify the stray NOC write source.

### Prevention

- The `preload` field is intentionally placed last in the `kernel_config_msg_t` structure to serve as a write-completion sentinel. Do not rearrange the launch message structure.
- Enable watcher sanitization to catch stray NOC writes that could corrupt the mailbox region.

---

## Hang Cause 2.1.7: Mismatched Kernel Configuration (Enabled Processors vs. Assigned Kernels)

### Symptom

BRISC hangs at `NTW` waiting for a subordinate that was told to run but has no kernel assigned. Alternatively, a subordinate with no kernel is never signaled and its sync byte remains at `RUN_SYNC_MSG_INIT`, preventing the all-done check from succeeding.

### Root Cause

The `enables` field in `kernel_config_msg_t` is a bitmask indicating which processors should execute kernels:

```c++
// tt_metal/hw/firmware/src/tt-1xx/brisc.cc
if (enables & (1u << ...TensixProcessorTypes::MATH0...)) {
    subordinate_sync->trisc0 = RUN_SYNC_MSG_GO;
    subordinate_sync->trisc1 = RUN_SYNC_MSG_GO;
    subordinate_sync->trisc2 = RUN_SYNC_MSG_GO;
}
```

If the `enables` mask says a processor is active but no kernel was loaded for it, the processor may enter its kernel entry point with corrupted or stale code, leading to unpredictable behavior including infinite loops. Conversely, if a processor is not enabled but its sync byte is not properly handled, BRISC may wait for a "done" signal that never arrives.

### Diagnosis Steps

1. Check the `enables` field in the launch message for the hung core.
2. Verify that each enabled processor has a valid kernel loaded (check `watcher_kernel_ids` in the launch message).
3. Check each subordinate's sync byte value against what it should be.

### Fix

Ensure that the program object at the host level correctly assigns kernels to the intended processor types. If using `CreateKernel` at the Metal API level, verify the `CoreType` and processor assignment.

### Prevention

- Use the Metal API's program construction utilities, which handle the `enables` mask automatically.
- When writing custom dispatch code, always verify that the `enables` mask matches the kernel assignments.

---

## Hang Cause 2.1.8: ERISC Context Switching Failure

### Symptom

An active Ethernet RISC (ERISC) core stops making progress. The ethernet link may appear healthy (routing firmware keeps running), but data movement operations through this ERISC core are blocked. Other cores waiting for semaphore signals from the ERISC core hang at `NSW` or `NSMW`.

### Root Cause

Active ERISC cores must periodically call `internal_::risc_context_switch()` to yield to the base firmware, which handles ethernet routing. The active ERISC firmware calls this in multiple locations:

```c++
// tt_metal/hw/firmware/src/tt-1xx/active_erisc.cc
inline void wait_subordinate_eriscs() {
    WAYPOINT("SEW");
    do {
        invalidate_l1_cache();
        internal_::risc_context_switch(kg_noc_mode == DM_DYNAMIC_NOC);
    } while (subordinate_sync->all != RUN_SYNC_MSG_ALL_SUBORDINATES_DONE);
    WAYPOINT("SED");
}
```

If a user kernel running on an ERISC core enters a tight spin loop without calling `risc_context_switch()`, the routing firmware is starved. This does not immediately hang the ERISC core itself, but:

1. Other ERISC cores expecting routing service from this core will time out
2. Cross-chip data movement that routes through this ERISC core will stall
3. Semaphore increments from remote chips may be delayed or lost

### Diagnosis Steps

1. Identify an ERISC core that is at waypoint `R` (running user kernel) for an extended period.
2. Check if the ERISC core's `routing_info_t` shows stalled routing state.
3. Verify that the user kernel running on the ERISC core includes context switch calls.
4. Use `check_eth_status.py` from tt-triage to verify Ethernet link health.

### Fix

Insert `internal_::risc_context_switch()` calls in any ERISC kernel spin loop.

### Prevention

- Always use the provided `eth_noc_semaphore_wait` function (which includes `run_routing()` calls) instead of a bare `noc_semaphore_wait` on ERISC cores:

```c++
// Pattern from ethernet/dataflow_api.h
void eth_noc_semaphore_wait(volatile tt_l1_ptr uint32_t* sem_addr,
                            uint32_t val, uint32_t wait_min = 0) {
    uint32_t count = 0;
    while ((*sem_addr) != val) {
        invalidate_l1_cache();
        if (count == wait_min) {
            run_routing();
            count = 0;
        } else {
            count++;
        }
    }
}
```

- The ERISC firmware uses a different exit mechanism than Tensix firmware: `erisc_exit()` returns to base firmware rather than entering `while(1)`. If an ERISC core hangs permanently, the ethernet link through that core is lost until chip reset.

---

## Hang Cause 2.1.9: ERISC Core State Corruption (Wormhole Errata)

### Symptom

An active ERISC core hangs intermittently when jumping into a user kernel. The hang occurs at kernel entry, not during kernel execution. It may reproduce non-deterministically.

### Root Cause

The `active_erisc.cc` firmware contains an explicit workaround for a hardware errata:

```c++
// tt_metal/hw/firmware/src/tt-1xx/active_erisc.cc
// After running the base firmware, some core state (for erisc0) seems
// broken, so jumps into the kernel may occasionally hang. Resetting the
// core fixes the issue.
```

The workaround involves a full GPR and local memory save/restore cycle:

1. `enter_reset()`: Saves all callee-saved registers (s0-s11, ra, gp) to the stack, copies all local memory (8 KiB) to a reserved L1 area (`MEM_ERISC_L1_TEMP_STORAGE`), saves the stack pointer to the halt mailbox, then enters an infinite loop.
2. The companion ERISC1 core triggers a hardware reset of ERISC0.
3. `resume_from_reset()`: Copies local memory back from L1, restores registers from the stack, and returns to the saved program counter.

If this reset workaround is not applied, ERISC0's microarchitectural state remains corrupted from the base firmware execution, causing intermittent hangs on kernel entry.

### Diagnosis Steps

1. Confirm the system is Wormhole with active ethernet cores.
2. Check if the hang occurs at the very beginning of kernel execution (waypoint transitions from firmware to `R`).
3. Verify that the `enter_reset` / `resume_from_reset` mechanism is being invoked.

### Fix

This is a hardware errata with a firmware workaround already in place. If the hang persists:
- Ensure the firmware version includes the `enter_reset` / `resume_from_reset` workaround.
- Verify that `MEM_ERISC_L1_TEMP_STORAGE` is not being corrupted by other code.

### Prevention

This is handled automatically by the active ERISC firmware. No user action is needed unless custom ERISC firmware is being developed.

---

## Hang Cause 2.1.10: Inter-RISC Semaphore Deadlock

### Symptom

Two or more RISC-V cores on the same Tensix are stuck at semaphore waits (`NSW` or `NSMW`), each waiting for a signal from the other. This is a classic deadlock: Core A waits for Core B, and Core B waits for Core A.

### Root Cause

User kernels can use L1 semaphores to coordinate between BRISC, NCRISC, and TRISC cores. A deadlock occurs when the dependency graph forms a cycle:

```
BRISC waits at NSW for semaphore X (to be set by NCRISC)
NCRISC waits at NSW for semaphore Y (to be set by BRISC)
```

Neither core can proceed because each is waiting for the other. This can also involve more than two cores:

```
BRISC waits for NCRISC (semaphore X)
NCRISC waits for TRISC2 (semaphore Y)
TRISC2 waits for BRISC (semaphore Z, via CB back-pressure)
```

### Diagnosis Steps

1. Read all five RISC-V core waypoints on the affected Tensix.
2. For each core at `NSW`/`NSMW`, identify the semaphore address and target value.
3. For each semaphore, determine which core is responsible for setting it.
4. Trace the dependency graph. If it forms a cycle, you have found the deadlock.

### Fix

Restructure the kernel synchronization protocol to eliminate the cyclic dependency:
- Establish a total ordering of semaphore acquisitions (always acquire in the same order on all cores)
- Use a single producer-consumer direction rather than bidirectional signaling
- Replace fine-grained semaphore coordination with CB-based producer/consumer patterns

### Prevention

- Keep the synchronization protocol simple. One-directional signaling (producer signals consumer, never the reverse) is deadlock-free by construction.
- If bidirectional coordination is needed between BRISC and NCRISC, consider using different iterations: BRISC signals NCRISC for iteration N, NCRISC signals BRISC for iteration N-1. This temporal ordering breaks the cycle.
- Document the expected semaphore protocol for each kernel configuration and review it for cycles before deployment.

---

## Hang Cause 2.1.11: Blackhole Atomic Barrier During CB Setup (`NABW`)

### Symptom

BRISC is stuck at waypoint `NABW` during firmware execution (not during user kernel execution). This occurs during the `barrier_remote_cb_interface_setup` function, which is called before the user kernel starts.

### Root Cause

On Blackhole, after setting up remote circular buffer interfaces (which involves NOC atomic writes), BRISC must wait for those atomics to be acknowledged:

```c++
// tt_metal/hw/firmware/src/tt-1xx/brisc.cc
inline void barrier_remote_cb_interface_setup(
    uint8_t noc_index, uint32_t noc_mode, uint32_t end_cb_index) {
#if defined(ARCH_BLACKHOLE)
    // cq_dispatch does not update noc transaction counts
    // so skip this barrier on the dispatch core
    if (end_cb_index != NUM_CIRCULAR_BUFFERS) {
        WAYPOINT("NABW");
        if (noc_mode == DM_DYNAMIC_NOC) {
            do {
                invalidate_l1_cache();
            } while (!ncrisc_dynamic_noc_nonposted_atomics_flushed(noc_index));
        } else {
            while (!ncrisc_noc_nonposted_atomics_flushed(noc_index));
        }
        invalidate_l1_cache();
        WAYPOINT("NABD");
    }
#endif
}
```

If the NOC atomic operations cannot complete (target core unreachable, NOC congestion, incorrect NOC coordinates for the remote CB), BRISC hangs at `NABW`. Note: the dispatch core is explicitly exempted because "cq_dispatch does not update noc transaction counts."

### Diagnosis Steps

1. Confirm BRISC is at `NABW` and the architecture is Blackhole.
2. This is a firmware-phase hang (occurs before waypoint `R`). Check whether any remote CB interfaces were configured by examining `min_remote_cb_start_index` in the launch message.
3. Check NOC atomic counters: are there outstanding atomics that have not been acknowledged?
4. Verify that the target cores for the remote CB setup are valid and reachable.

### Fix

- If the target coordinates are wrong, fix the program configuration that specifies remote CB interfaces.
- If the NOC is congested, this may be a transient condition that resolves with a retry.

### Prevention

- This is a Blackhole-only firmware mechanism. It does not exist on GS or WH.
- When using remote circular buffers on Blackhole, ensure that the remote cores are configured and running before the kernel that references them is launched.

---

## Hang Cause 2.1.12: Dynamic NOC Mode Stale State Contamination (Blackhole)

### Symptom

On Blackhole with dynamic NOC mode enabled. A kernel hangs at a NOC barrier (`NRBW`, `NWBW`, `NABW`) immediately or shortly after starting, despite not having issued any NOC transactions of the type the barrier is waiting for. Alternatively, the BRISC firmware triggers an `NKFW` assertion at the end of a kernel indicating unbalanced NOC counters.

### Root Cause

In dynamic NOC mode, both BRISC and NCRISC share the NOC instances. If a kernel running on one RISC exits without barriering all its outstanding NOC transactions, those transactions remain in the hardware counters. The next kernel to run on the other RISC inherits the stale counter state.

After kernel execution, the BRISC firmware verifies that all NOC transactions have been cleaned up:

```c++
// tt_metal/hw/firmware/src/tt-1xx/brisc.cc
if (noc_mode == DM_DYNAMIC_NOC) {
    WAYPOINT("NKFW");
    invalidate_l1_cache();
    for (int noc = 0; noc < NUM_NOCS; noc++) {
        ASSERT(ncrisc_dynamic_noc_reads_flushed(noc));
        ASSERT(ncrisc_dynamic_noc_nonposted_writes_sent(noc));
        ASSERT(ncrisc_dynamic_noc_nonposted_writes_flushed(noc));
        ASSERT(ncrisc_dynamic_noc_nonposted_atomics_flushed(noc));
        ASSERT(ncrisc_dynamic_noc_posted_writes_sent(noc));
    }
    WAYPOINT("NKFD");
}
```

The NKFW firmware check in `brisc.cc` uses the generic `ASSERT(condition)` macro (no second argument), so all five assertions produce `DebugAssertTripped` (value 3) in the watcher assert mailbox. The watcher log will show which core tripped, but not which specific counter was unbalanced.

For finer-grained diagnostics, the kernel wrapper files (`brisck.cc`, `ncrisck.cc`, `active_erisck.cc`, `idle_erisck.cc`) perform similar post-kernel NOC checks with specialized assert types:
- `DebugAssertNCriscNOCReadsFlushedTripped` (value 4)
- `DebugAssertNCriscNOCNonpostedWritesSentTripped` (value 5)
- `DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped` (value 6)
- `DebugAssertNCriscNOCPostedWritesSentTripped` (value 7)

These kernel-wrapper checks fire at kernel exit (before the firmware NKFW check), so if both are enabled, the more specific kernel-wrapper assert will typically trip first.

Without assertions enabled, this "phantom hang" is one of the most difficult-to-diagnose failure modes on Blackhole.

### Diagnosis Steps

1. Check both BRISC and NCRISC NOC counters for both NOC instances.
2. Look for an `NKFW` assertion in the watcher log from the previous kernel invocation.
3. Enable `TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1` to catch unbalanced counters at kernel exit.

### Fix

**Buggy pattern:**
```c++
// WRONG: kernel exits without barriering NOC reads
void kernel_main() {
    noc_async_read(src_addr, dst_addr, size);
    // ... use data (which may not have arrived yet) ...
    // Missing: noc_async_read_barrier();
}
```

**Corrected pattern:**
```c++
// CORRECT: always barrier before kernel exit
void kernel_main() {
    noc_async_read(src_addr, dst_addr, size);
    noc_async_read_barrier();
    // ... use data (safe, barrier ensures completion) ...
}
```

### Prevention

- In dynamic NOC mode, always pair every `noc_async_read` with a `noc_async_read_barrier` and every `noc_async_write` with a `noc_async_write_barrier` within the same kernel execution.
- Enable assertions during development to catch unbalanced counters via the `NKFW` check.

---

## Hang Cause 2.1.13: Quasar Extended Subordinate Wait

### Symptom

On Quasar (tt-2xx). The DM0 core is at waypoint `NTW`, waiting for completion. The wait check involves significantly more cores than on tt-1xx architectures.

### Root Cause

On Quasar, the DM core must wait for all DM subordinates AND all four Neo engines, each containing four TRISC cores:

```c++
// tt_metal/hw/firmware/src/tt-2xx/dm.cc
inline void wait_subordinates() {
    WAYPOINT("NTW");
    while (subordinate_sync->allDMs != RUN_SYNC_MSG_ALL_SUBORDINATES_DMS_DONE ||
           subordinate_sync->allNeo0 != RUN_SYNC_MSG_ALL_SUBORDINATES_DONE ||
           subordinate_sync->allNeo1 != RUN_SYNC_MSG_ALL_SUBORDINATES_DONE ||
           subordinate_sync->allNeo2 != RUN_SYNC_MSG_ALL_SUBORDINATES_DONE ||
           subordinate_sync->allNeo3 != RUN_SYNC_MSG_ALL_SUBORDINATES_DONE);
    WAYPOINT("NTD");
}
```

With 16+ TRISCs plus multiple DM cores, the number of potential subordinate hang sources is much larger. A bug in a kernel on Neo2's TRISC1 will hang Neo2 but not Neo0/1/3 -- though DM0 will still hang at `NTW` because it waits for ALL.

### Diagnosis Steps

1. Read each `allNeo*` and `allDMs` field separately to identify which Neo engine or DM has not completed.
2. Within the hung Neo engine, identify which of the four TRISC cores has not signaled done.
3. Trace the hung TRISC's waypoint to find the root cause.

### Fix

Fix the underlying kernel bug on the specific TRISC/Neo that is hung. The triage process is the same as on tt-1xx, just with more subordinates to check.

### Prevention

- Test kernels on individual Neo engines before scaling to all four.
- Use watcher to monitor per-TRISC waypoints during development.

---

## Summary: RISC Synchronization Hang Signatures

| Scenario | Waypoint Pattern | Architecture | Root Cause Category |
|----------|-----------------|--------------|---------------------|
| 2.1.1 NTW | BRISC at `NTW`, subordinate at blocking waypoint | All | Derived: subordinate kernel hang |
| 2.1.2 IRAM | BRISC stuck (no waypoint), past `GW` | WH only | NCRISC halt-reset failure |
| 2.1.3 Reset Timing | NCRISC at unexpected waypoint | WH only | Insufficient reset delay |
| 2.1.4 TRISC Init | BRISC stuck (no waypoint), in `run_triscs()` | All | TRISC0 sync register stall |
| 2.1.5 Go-Signal | BRISC at `GW` | All | Dispatch failure or NOC error |
| 2.1.6 Ring Buffer | Varied -- incorrect kernel execution | All | L1 corruption or dispatch bug |
| 2.1.7 Mismatch | BRISC at `NTW`, subordinate at `R` or stuck | All | `enables` mask vs. kernel mismatch |
| 2.1.8 ERISC Context | ERISC at `R` for extended period | WH, BH | Missing `risc_context_switch()` |
| 2.1.9 ERISC Errata | ERISC hangs at kernel entry | WH only | Hardware errata (erisc0 state) |
| 2.1.10 Semaphore | Multiple cores at `NSW`/`NSMW` | All | Cyclic dependency in semaphore protocol |
| 2.1.11 NABW | BRISC at `NABW` | BH only | Remote CB setup atomics incomplete |
| 2.1.12 Stale NOC | Barrier hang or `NKFW` assertion | BH only | Unbalanced NOC counters from prior kernel |
| 2.1.13 Quasar | DM0 at `NTW`, any Neo engine hung | Quasar | Any subordinate in any Neo engine |

> **Tip:** The two spin loops in BRISC firmware that **lack waypoint markers** are the NCRISC IRAM handshake (2.1.2) and the TRISC0 init wait (2.1.4). When BRISC appears stuck between waypoints, always check the raw `subordinate_sync` byte values to distinguish these from other hang types.

---

**Previous:** [`index.md`](./index.md) | **Next:** [`02_circular_buffer_deadlocks.md`](./02_circular_buffer_deadlocks.md)
