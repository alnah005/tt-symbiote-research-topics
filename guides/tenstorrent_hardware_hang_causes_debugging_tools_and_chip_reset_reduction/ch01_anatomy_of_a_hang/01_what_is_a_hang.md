# What Is a Hang

## Defining the Problem

A **hang** on Tenstorrent hardware is a state in which one or more RISC-V processor cores on the device enter a spin loop whose exit condition can never be satisfied under the current system state. The host observes this as an operation that fails to complete within any reasonable time. No error is reported, no exception is raised, and no data corruption signal fires -- the system simply stops making forward progress.

This definition must be distinguished from several superficially similar failure modes:

| Failure Mode | Forward Progress | Error Signal | Data Integrity | Recovery |
|---|---|---|---|---|
| **Hang** | None -- RISC-V cores spin indefinitely | None (silent) | Preserved but inaccessible | Requires kill or chip reset |
| **Crash** | Halted by fault | RISC-V trap, segfault, or firmware assert | Potentially corrupted | Process restart; chip may be fine |
| **Error/Assert** | Halted at assertion (`assert_and_hang` writes diagnostic data, then enters `while(1)`) | Explicit error message from watcher or assert macro | Preserved | Fix bug, re-run |
| **Data Corruption** | Continues (incorrect results) | None until validation | Corrupted | Re-run with corrected program |
| **Slowness** | Ongoing but degraded | None (performance issue) | Preserved | Profiling, optimization |

The critical distinction is that a hang produces **no signal whatsoever** from the device. The RISC-V cores are alive and executing instructions -- they are simply executing the same tight loop forever. From the host's perspective, a dispatched program never returns. From the device's perspective, every core involved is "busy" spinning on a condition variable, a semaphore, a NOC status register, or a circular buffer counter that will never reach the expected value.

## Why Tenstorrent Hardware Is Susceptible to Hangs

The Tenstorrent architecture is built around a producer-consumer model with explicit synchronization. Each Tensix core contains five RISC-V "Baby" processors:

- **BRISC** (Data Movement 0): The primary orchestrator. It runs the main firmware loop, dispatches work to other cores, and executes data-movement kernels. It is the first core to start and the last to finish each kernel invocation.
- **NCRISC** (Data Movement 1): Handles a second independent data-movement kernel, typically responsible for reading data from DRAM or remote cores. On Wormhole, it executes from IRAM rather than L1.
- **TRISC0** (Unpack): Runs the unpacker portion of the compute kernel.
- **TRISC1** (Math): Runs the math/FPU portion of the compute kernel.
- **TRISC2** (Pack): Runs the packer portion of the compute kernel.
- **ERISC** (Ethernet RISC): Present on Ethernet-connected cores (Wormhole, Blackhole, Quasar), handles inter-chip communication. Active ethernet cores run a separate firmware loop with its own hang-susceptible patterns.

On Quasar (tt-2xx architecture), the processor organization changes: a single **DM** (Data Mover) core replaces BRISC and NCRISC, and there are four TRISC cores (TRISC0-TRISC3) per Neo engine, with four Neo engines per tile.

These processors coordinate through **circular buffers** in local SRAM (L1) and through **semaphores** in L1 that are modified by remote NOC transactions. The coordination is implemented as **spin loops** -- each blocking API call compiles down to a tight `do { ... } while(condition)` loop that polls a memory location or hardware register.

This is a deliberate design choice. The Baby RISC-V cores are intentionally minimal processors (closer to a MIPS R3000 than a modern superscalar core). They have no interrupt-driven scheduler, no OS, and no preemption. When a core calls `cb_wait_front(cb_out, 1)`, it literally spins in a tight loop reading the `pages_received` register until tiles become available. If the producer never pushes those tiles -- because it is itself waiting on something, or because its NOC transfer hit an error -- the consumer spins forever.

**The critical insight is: every hang is a spin-wait loop that never terminates.** There are no blocking OS primitives, no sleep queues, no interrupt-driven wakeups. If a condition variable is never set, the core spins forever.

## The RISC-V Wait-Loop Model

Every blocking operation in the Metalium dataflow API follows the same pattern:

```
WAYPOINT("XXYW");      // Write "waiting" waypoint to debug mailbox
do {
    // possibly invalidate L1 cache
    // read a hardware register or memory location
} while (condition_not_met);
WAYPOINT("XXYD");      // Write "done" waypoint to debug mailbox
```

The waypoint mechanism (defined in `tt_metal/hw/inc/api/debug/waypoint.h`) writes a 4-character ASCII tag into a per-processor debug mailbox in L1. When the watcher thread on the host periodically reads these mailboxes, it can determine *which* blocking call each processor is stuck on. The "W" suffix means "waiting"; the "D" suffix means "done." If the watcher sees a core stuck at `CRBW` across multiple polling intervals, it knows that core is blocked inside `cb_reserve_back`.

Here is the actual waypoint implementation:

```c
// From waypoint.h
template <uint32_t x>
inline void write_debug_waypoint(volatile tt_l1_ptr uint32_t* debug_waypoint) {
    debug_waypoint[internal_::get_hw_thread_idx()] = x;
}

#define WAYPOINT(x) write_debug_waypoint<helper(x)>(WATCHER_WAYPOINT_MAILBOX)
```

The `helper` function at compile time packs up to 4 ASCII characters into a single `uint32_t` for efficient storage. Each of the five RISC-V processors gets its own slot in the `debug_waypoint` array, so the host can read the state of all five processors on any core simultaneously.

A representative example is `cb_reserve_back`, which spins waiting for a circular buffer consumer to free space. It sets waypoint `CRBW` before the loop and `CRBD` after. If the consumer never acknowledges tiles, `free_space_pages` never reaches the required threshold and the loop spins forever. See [02_blocking_primitives_taxonomy.md](./02_blocking_primitives_taxonomy.md#crbw-cb_reserve_back----circular-buffer-reserve-back-wait) for the full code listing and detailed failure mode analysis.

## Lifecycle of a Kernel Execution

The following describes the BRISC main loop (from `brisc.cc`), which is the most important lifecycle to understand because BRISC orchestrates all other cores:

### Phase 1: Go-Wait (Waypoint `GW`)

```
while (go_message_signal != RUN_MSG_GO && !preload_flag) {
    invalidate_l1_cache();
    // handle RUN_MSG_RESET_READ_PTR, RUN_MSG_REPLAY_TRACE
}
```

BRISC polls the `go_messages` mailbox, waiting for the dispatcher to send `RUN_MSG_GO` (value `0x80`). During this phase, it also handles administrative signals like `RUN_MSG_RESET_READ_PTR` (value `0xc0`) and `RUN_MSG_REPLAY_TRACE` (value `0xf0`).

**Hang risk:** If the dispatcher never sends the go signal -- due to a host-side bug, a dispatch NOC write that fails, or a multi-chip routing failure -- BRISC will spin at waypoint `GW` forever.

### Phase 2: Configuration and Subordinate Launch (Waypoint transition to `R`)

Once `GW` exits, BRISC reads the launch message, initializes circular buffer interfaces, configures the NOC, and signals subordinate cores:

- NCRISC is told to load (`RUN_SYNC_MSG_LOAD`, value `0x1`)
- TRISCs are told to run (`RUN_SYNC_MSG_GO`, value `0x80`)
- Circular buffer interfaces are set up (local and remote)
- On Blackhole, an atomic barrier (`NABW`/`NABD`) ensures remote CB interfaces are established before kernel execution begins

### Phase 3: Kernel Execution (Waypoint `R`)

The actual user kernel runs. During this phase, the kernel may call any of the blocking primitives documented in Section 2. Each of these can individually hang.

### Phase 4: Done (Waypoint `D`)

The kernel returns. BRISC sets waypoint `D`.

### Phase 5: Wait for Subordinates (Waypoint `NTW` / `NTD`)

```
while (subordinate_sync->all != RUN_SYNC_MSG_ALL_SUBORDINATES_DONE) {
    invalidate_l1_cache();
}
```

BRISC waits for NCRISC and all TRISCs to signal completion. This is a 32-bit comparison where each byte in the `subordinate_sync` union represents one core's status. All must be `RUN_SYNC_MSG_DONE` (value `0x0`).

**Hang risk:** If any subordinate core hangs in its own kernel, BRISC will remain at waypoint `NTW` indefinitely. This is a **derived hang** -- the root cause is in the subordinate, but the observable symptom propagates to BRISC.

On Quasar (tt-2xx), the same pattern extends to four Neo engines:

```c++
// dm.cc (tt-2xx architecture)
WAYPOINT("NTW");
while (subordinate_sync->allDMs != RUN_SYNC_MSG_ALL_SUBORDINATES_DMS_DONE ||
       subordinate_sync->allNeo0 != RUN_SYNC_MSG_ALL_SUBORDINATES_DONE ||
       subordinate_sync->allNeo1 != RUN_SYNC_MSG_ALL_SUBORDINATES_DONE ||
       subordinate_sync->allNeo2 != RUN_SYNC_MSG_ALL_SUBORDINATES_DONE ||
       subordinate_sync->allNeo3 != RUN_SYNC_MSG_ALL_SUBORDINATES_DONE);
WAYPOINT("NTD");
```

### Phase 6: Signal Completion

BRISC writes `RUN_MSG_DONE` (value `0x0`) to its go-message slot and, in `DISPATCH_MODE_DEV`, sends an atomic increment to the dispatcher core via NOC to notify completion.

## The Go-Signal Protocol

The dispatch system uses a specific set of signal values to communicate between host and device:

| Signal | Value | Meaning |
|--------|-------|---------|
| `RUN_MSG_INIT` | `0x40` | Core is initializing |
| `RUN_MSG_GO` | `0x80` | Kernel should begin execution |
| `RUN_MSG_DONE` | `0x00` | Kernel execution complete |
| `RUN_MSG_RESET_READ_PTR` | `0xc0` | Reset the launch message read pointer |
| `RUN_MSG_RESET_READ_PTR_FROM_HOST` | `0xe0` | Same, but from host (no NOC ack needed) |
| `RUN_MSG_REPLAY_TRACE` | `0xf0` | Replay a captured trace |

The subordinate synchronization uses a parallel protocol:

| Signal | Value | Meaning |
|--------|-------|---------|
| `RUN_SYNC_MSG_INIT` | `0x40` | Subordinate is initializing |
| `RUN_SYNC_MSG_GO` | `0x80` | Subordinate should run kernel |
| `RUN_SYNC_MSG_LOAD` | `0x01` | Subordinate should load CBs/IRAM |
| `RUN_SYNC_MSG_WAITING_FOR_RESET` | `0x02` | WH NCRISC: ready for IRAM reset |
| `RUN_SYNC_MSG_INIT_SYNC_REGISTERS` | `0x03` | Initialize sync registers |
| `RUN_SYNC_MSG_DONE` | `0x00` | Subordinate execution complete |

## Observable Symptoms

When a hang occurs, the following symptoms are observable from different vantage points:

### From the Host

1. **Timeout**: The `EnqueueProgram` or `Finish` call on the host side never returns. The host-side timeout (if configured) eventually fires.
2. **Process unresponsive**: The user's application appears frozen. CPU utilization may be near zero (the host thread is blocked waiting for device completion).
3. **No error output**: Unlike crashes or asserts, no error message is printed. The watcher (if enabled) may eventually report stale waypoints.
4. **Need for `tt-smi` reset**: In cases where the hang has corrupted NOC state or left the dispatch system in an inconsistent state, simply killing the host process is insufficient. A chip-level reset via `tt-smi -r` (or the equivalent UMD warm reset API) is required.

### From the Watcher

The watcher is a host-side monitoring thread that periodically reads the WAYPOINT mailboxes from device L1 memory. When enabled (via `WATCHER_ENABLED`), during a hang the watcher will observe:

- One or more cores stuck at a **wait waypoint** (suffix `W`): `CRBW`, `CWFW`, `NRBW`, `NWBW`, `NSW`, `NSMW`, `NTW`, `GW`
- The corresponding **done waypoint** (suffix `D`) is never reached
- Other cores may show `NTW` (waiting for the hung core to finish)

### From NOC Status Registers

Each NOC maintains hardware counters for transactions issued, sent, and acknowledged. During a hang:

- `NIU_MST_RD_RESP_RECEIVED` may be less than `noc_reads_num_issued[noc]` (read responses missing)
- `NIU_MST_WR_ACK_RECEIVED` may be less than `noc_nonposted_writes_acked[noc]` (write acknowledgments missing)
- Command buffer status (`NOC_CMD_CTRL`) may not equal `NOC_CTRL_STATUS_READY` (command buffer backlogged)

## The Hang Lifecycle

A hang progresses through distinct phases, each offering different diagnostic opportunities:

### Phase 1: Root Cause Occurs (t=0)

Something goes wrong. Common root causes include:
- A circular buffer push/pop mismatch causes a deadlock between producer and consumer
- A NOC transfer targets an invalid address and the response never arrives
- A semaphore increment is lost due to an inline-write back-pressure hang on Blackhole
- The dispatch system fails to deliver a go-signal
- An alignment violation causes a NOC transaction to silently fail

### Phase 2: Spin Loop Entry (t=0 to t+microseconds)

The affected RISC-V processor enters a blocking call and begins spinning. The waypoint is written (e.g., `CWFW` for `cb_wait_front` waiting). At this point, the core is consuming power but making no progress. In many cases, the core that first enters a spin loop is not the root cause -- it may be a downstream consumer waiting for data that will never arrive.

### Phase 3: Cascade (t+microseconds to t+milliseconds)

The initial hang often cascades. If a writer kernel hangs, the circular buffer it reads from fills up. Then the compute kernel's `cb_reserve_back` for its output buffer starts spinning because space is never freed. Then the compute kernel stops consuming from its input buffers, so the reader kernel's `cb_reserve_back` for input buffers also starts spinning. Within milliseconds, all five RISC-V processors on a core can be blocked. If multi-core operations are in progress, the hang propagates across cores via semaphore dependencies.

### Phase 4: Host Detection (t+seconds)

The host detects the hang through one of several mechanisms:
- **Timeout**: The operation exceeds its configured timeout (typically seconds)
- **Watcher**: The watcher thread observes unchanging waypoints across multiple polling intervals and reports the stalled cores
- **Heartbeat failure** (Wormhole): The `RISC_POST_HEARTBEAT` mechanism writes an incrementing counter to address `0x1C`; if the watcher sees this counter stop incrementing, the core is stuck
- **Never**: If the host program does not have a timeout and the watcher is not enabled, the hang is never detected. The program simply stops producing output and remains stuck until the user manually intervenes.

### Phase 5: Resolution Decision

The host must decide between two paths:

**Kill (targeted recovery)**: If the hang is isolated to specific kernels, the host can attempt to terminate just those kernels and reclaim the affected cores. This is faster and less disruptive, but only works when the NOC fabric itself is not in a stuck state.

**Full chip reset**: If the NOC itself is wedged (outstanding transactions that will never complete, corrupted routing state, or hardware-level back-pressure stalls), a full chip reset is the only option. This is the expensive path -- it requires re-initializing the device, reloading firmware, and restarting all workloads.

### The Recovery Hierarchy

1. **Retry the operation.** Some transient conditions (NOC congestion, semaphore race) may resolve if the kernel is simply re-launched.
2. **Reset the affected core.** If the RISC-V core can be individually reset without affecting others, this preserves the state of neighboring cores. The ERISC firmware on Wormhole/Blackhole demonstrates this pattern: the `enter_reset` function in `active_erisc.cc` saves all register state and local memory to L1, then enters a tight infinite loop. A companion core triggers the actual hardware reset, and `resume_from_reset` restores state from L1.
3. **Reset the chip.** If NOC or memory subsystem state is corrupted, only a full chip reset will recover.
4. **Reset the cluster.** In multi-chip configurations, certain failure modes (ethernet link hangs, fabric routing corruption) may require resetting all connected chips.

The entire purpose of the debugging techniques in this guide is to:
1. Move detection earlier (Phase 4 closer to Phase 1)
2. Diagnose the root cause accurately (distinguish Phase 1 from Phase 3 symptoms)
3. Make kill viable more often (reduce the need for full reset)
4. Prevent hangs altogether (fix the root cause)

## When Is a Chip Reset Required?

| Scenario | Chip Reset Required? | Reason |
|---|---|---|
| CB deadlock (producer/consumer mismatch) | No | Kernel logic error; firmware can be restarted. |
| NOC address violation (write to harvested row) | Usually yes | NOC hardware may have outstanding transactions that block future use. |
| L1 corruption affecting NOC registers | Yes | NOC control state is corrupted; no software recovery path. |
| Ethernet link hang in multi-chip | Yes | Ethernet core firmware state is inconsistent across chips. |
| Dispatch/CQ stall due to missing completion signal | Sometimes | Depends on whether the stall is in software (recoverable) or hardware (reset needed). |
| Watcher assert-and-hang | No | The core is deliberately halted; firmware restart will clear it. |
| Blackhole inline-write back-pressure | Yes | Hardware-level stall with no software recovery. |

The general rule: **if the NOC hardware itself is in a bad state, a chip reset is required.** If the hang is purely in software spin-loops with the NOC hardware otherwise healthy, a process kill and re-launch is sufficient.

## The `assert_and_hang` Pattern: A Hang by Design

Not all infinite loops are unintentional. The Metalium debug infrastructure includes a deliberate hang mechanism defined in `tt_metal/hw/inc/api/debug/assert.h`:

```c
inline void assert_and_hang(uint32_t line_num,
                            debug_assert_type_t assert_type = DebugAssertTripped) {
    debug_assert_msg_t tt_l1_ptr* v = GET_MAILBOX_ADDRESS_DEV(watcher.assert_status);
    if (v->tripped == DebugAssertOK) {
        v->line_num = line_num;
        v->tripped = assert_type;
        v->which = internal_::get_hw_thread_idx();
    }

    while (1) { ; }
}
```

When an assertion fails (and watcher is enabled), the firmware:
1. Records the source line number in the assert mailbox
2. Records which RISC-V processor triggered the assertion
3. Records the type of assertion (general assertion, out-of-bounds runtime argument access, etc.)
4. Enters an infinite loop

This is distinguishable from a true hang because the assert mailbox contains diagnostic data. The watcher thread on the host can read this mailbox and report exactly which assertion failed and on which line.

**ERISC special handling:** Ethernet RISC-V cores (ERISC) get special treatment -- instead of hanging forever, they record the assertion then exit back to base firmware, since ERISC cores are not restarted between kernel launches. If ERISC entered `while(1)`, the ethernet link would be permanently lost until chip reset.

The `ASSERT` macro wraps this:

```c
#define ASSERT(condition, ...)                        \
    do {                                              \
        if (not(condition))                           \
            assert_and_hang(__LINE__, ##__VA_ARGS__); \
    } while (0)
```

Without watcher enabled, there are two fallback behaviors:
- **Lightweight asserts** (`LIGHTWEIGHT_KERNEL_ASSERTS`): Executes the RISC-V `ebreak` instruction, which produces a detectable exception but no mailbox data
- **No asserts**: The `ASSERT` macro compiles to nothing, and assertion failures become silent data corruption or undiagnosed hangs

## The `debug_sanitize_post_addr_and_hang` Pattern: NOC Address Validation

A second deliberate hang mechanism exists in `tt_metal/hw/inc/internal/debug/sanitize.h` for catching NOC address violations before they cause silent failures:

```c
void debug_sanitize_post_addr_and_hang(
    uint8_t noc_id, uint64_t noc_addr, uint32_t l1_addr, uint32_t len,
    debug_sanitize_noc_cast_t multicast, debug_sanitize_noc_dir_t dir,
    debug_sanitize_noc_which_core_t which_core, uint16_t return_code) {

    if (return_code == DebugSanitizeOK) return;

    // Record the violation details in the sanitize mailbox
    v[noc_id].noc_addr = noc_addr;
    v[noc_id].l1_addr = l1_addr;
    v[noc_id].len = len;
    v[noc_id].which_risc = internal_::get_hw_thread_idx();
    v[noc_id].is_multicast = (multicast == DEBUG_SANITIZE_NOC_MULTICAST);
    v[noc_id].is_write = (dir == DEBUG_SANITIZE_NOC_WRITE);
    v[noc_id].is_target = (which_core == DEBUG_SANITIZE_NOC_TARGET);
    v[noc_id].return_code = return_code;

    while (1) { ; }
}
```

This function is called before every NOC transaction when watcher is enabled. It validates:
- **Address bounds**: L1 address within valid range (`debug_valid_worker_addr`), DRAM address within bank bounds (`debug_valid_dram_addr`), PCIe address within mapped region (`debug_valid_pcie_addr`)
- **Alignment**: L1 and NOC addresses have matching alignment per the target core type
- **Multicast validity**: Start and end coordinates form a valid rectangle of Tensix cores
- **Coordinate validity**: NOC XY coordinates map to a known core type
- **CB bounds**: L1 address within a circular buffer stays inside that buffer's allocated region
- **Linked transaction safety**: No unicast transaction is issued while a linked multicast is pending

Each validation failure produces a specific return code (e.g., `DebugSanitizeNocAddrOverflow`, `DebugSanitizeNocTargetInvalidXY`, `DebugSanitizeNocMulticastInvalidRange`) that the host can read from the sanitize mailbox to diagnose the exact problem.

Like `assert_and_hang`, the ERISC variant exits to base firmware instead of spinning forever. And like assertions, the entire sanitization layer compiles to no-ops when watcher is disabled, meaning that in production builds, these address violations cause silent NOC failures that often manifest as hangs in the data-dependent spin loops.

## The 5-Part Diagnostic Format

Throughout this guide, each hang scenario is documented using a consistent 5-part format:

1. **Symptom:** What the watcher waypoint, NOC counters, or host-side timeout looks like
2. **Root Cause:** The underlying programming error, hardware limitation, or race condition
3. **Diagnosis Steps:** How to identify this specific hang from the available debugging data
4. **Fix:** What to change in the code, configuration, or hardware setup to resolve the hang
5. **Prevention:** How to prevent the hang from recurring in future development

This structure is designed to be useful both as a reference during active debugging (start from Symptom, trace through Diagnosis Steps to the Fix) and as a learning resource (start from Root Cause, understand the Prevention strategy).

---

**Next:** [`02_blocking_primitives_taxonomy.md`](./02_blocking_primitives_taxonomy.md)
