# Hang Taxonomy

This section presents a six-category classification system for hardware hangs on Tenstorrent devices. Each category is defined by the subsystem where the root cause originates, even though the observable symptoms (frozen waypoints, host timeout) may appear similar across categories. The taxonomy provides the structured vocabulary needed to triage hangs efficiently and route them to the correct debugging workflow.

## The Six Categories

1. **Kernel-Level Hangs** -- Bugs in user-written data movement or compute kernels
2. **NOC Hangs** -- Failures in the Network-on-Chip transaction layer
3. **Memory Hangs** -- Incorrect memory addressing, alignment, or resource conflicts
4. **Dispatch Hangs** -- Failures in the host-to-device command dispatch pipeline
5. **Multi-Chip Hangs** -- Failures involving Ethernet links, remote chips, or fabric routing
6. **Host-Device Hangs** -- Failures in the PCIe interface or host-side synchronization

---

## Category 1: Kernel-Level Hangs

### Definition

A kernel-level hang is caused by a logical error in user-written kernel code that results in a blocking primitive's exit condition becoming unsatisfiable. The hardware and firmware are functioning correctly; the bug is in the program running on the device.

### Symptoms

Kernel-level hangs exhibit the general hang characteristics described in [01_what_is_a_hang.md](./01_what_is_a_hang.md#observable-symptoms). The distinguishing feature is that NOC status registers typically show no outstanding transactions -- the hang is in CB or semaphore logic (waypoints `CRBW`, `CWFW`, `NSW`, `NSMW`), not in the NOC subsystem.

### RISC-V Core Involvement

| Core | Typical Waypoint | Role |
|---|---|---|
| BRISC (DM0) | `CRBW`, `CWFW`, `NSW`, `NTW` | Reader kernel or subordinate wait |
| NCRISC (DM1) | `CRBW`, `CWFW`, `NSW` | Writer kernel |
| TRISC0 (Unpack) | `CWFW` | Waiting for input from reader |
| TRISC1 (Math) | (indirect -- waits on TRISC0) | Stalled waiting for unpack |
| TRISC2 (Pack) | `CRBW` | Waiting for output CB space |

### Common Root Causes

1. **Mismatched push/pop counts** (1a): Producer pushes N tiles but consumer pops M != N tiles per iteration. Over time, the CB fills (if M < N) or the consumer waits for tiles that never arrive (if M > N).

2. **Incorrect `cb_wait_front` cumulative semantics** (1b): Issuing `cb_wait_front(8)` four times instead of `cb_wait_front(8)`, `cb_wait_front(16)`, `cb_wait_front(24)`, `cb_wait_front(32)` as required by the API contract.

3. **CB size not evenly divisible by tile count** (1c): The API requires that the CB total size be an even multiple of the argument passed to `cb_reserve_back` or `cb_wait_front`. Violating this leads to wrap-around arithmetic errors.

4. **Semaphore protocol errors** (1d): Waiting on a semaphore value that no other core is programmed to write, or using `noc_semaphore_wait` (exact equality) when the signaling pattern can skip values.

5. **Loop bound mismatches** (1e): Reader kernel iterates N times but compute kernel expects M > N iterations. After the reader finishes, the compute kernel hangs waiting for tiles that will never be produced.

6. **Compute pipeline stall** (1f): The three TRISC cores coordinate through hardware semaphores and the `tile_regs_acquire` / `tile_regs_commit` / `tile_regs_wait` / `tile_regs_release` protocol. If this protocol is violated, the pipeline stalls and subsequent acquire calls block forever.

### Chapter References

Detailed analysis of kernel-level hangs: Chapter 2 (Kernel-Level and NOC Hang Mechanisms).

---

## Category 2: NOC Hangs

### Definition

A NOC hang occurs when a Network-on-Chip transaction fails to complete, causing the RISC-V core that issued it to spin indefinitely at a NOC barrier. The root cause is in the NOC subsystem: invalid addresses, routing errors, congestion-induced packet loss, or hardware-level command buffer stalls.

### Symptoms

- Core frozen at `NRBW` (read barrier) or `NWBW` (write barrier)
- NOC status registers show a mismatch: `NIU_MST_RD_RESP_RECEIVED < noc_reads_num_issued[noc]` or `NIU_MST_WR_ACK_RECEIVED < noc_nonposted_writes_acked[noc]`
- If the command buffer is stalled: core frozen at `RP2W` (waiting for `NOC_CMD_CTRL == NOC_CTRL_STATUS_READY`)
- If watcher/sanitize is enabled, the sanitize mailbox may contain a pre-hang violation report
- May affect multiple cores if the NOC congestion is systemic

### RISC-V Core Involvement

| Core | Typical Waypoint | Role |
|---|---|---|
| BRISC (DM0) | `NRBW`, `NWBW`, `RP2W` | Issuing reads/writes as reader kernel |
| NCRISC (DM1) | `NRBW`, `NWBW`, `RP2W` | Issuing reads/writes as writer kernel |

TRISC cores do not directly issue NOC transactions in the standard programming model, so they do not directly experience NOC hangs. However, they will hang indirectly if a data-movement core that feeds them is stuck at a NOC barrier.

### Common Root Causes

1. **Invalid NOC coordinates** (2a): Writing to or reading from a NOC address with coordinates that do not map to any physical endpoint (wrong coordinates, harvested row/column).

2. **Address out of range** (2b): Reading from an L1 address beyond the valid range on the target core. The behavior is architecture-dependent.

3. **Multicast with wrong `num_dests`** (2c): The software counter for expected write acknowledgments is incremented by `num_dests`. If this does not match the actual multicast rectangle, the barrier hangs waiting for acks that will never arrive.

4. **NOC command buffer hardware stall** (2d): If the NOC hardware enters a state where a command buffer never becomes ready, all subsequent transactions on that NOC are blocked. This is a hardware-level issue typically requiring chip reset.

5. **Blackhole inline-write back-pressure** (2e): Inline writes to L1 on Blackhole use all four memory ports and can hang the NOC pipeline under back-pressure. Known hardware limitation with a software workaround.

6. **NOC congestion / virtual channel deadlock** (2f): Under extreme traffic patterns, circular dependency in VC allocation across multiple NOC transactions. Rare but possible.

### Chapter References

NOC hang mechanisms: Chapter 2. NOC debugging tools: Chapter 6.

---

## Category 3: Memory Hangs

### Definition

A memory hang is caused by incorrect memory addressing, alignment violations, or resource conflicts that prevent a NOC transaction from completing or a local memory access from succeeding.

### Symptoms

- Core frozen at `NRBW` or `NWBW` (the NOC transaction targeting the bad address never completes)
- May manifest as `RP2W` if the bad transaction occupies a command buffer slot indefinitely
- Watcher sanitize checks (if enabled) may have fired an assert before the hang, providing early warning
- On Blackhole, specific patterns around inline writes to L1

### Common Root Causes

1. **L1 buffer overflow / overlap** (3a): Two buffers in L1 overlap due to incorrect allocation or a CB that is too large. Critical cases include CB data region overlapping CB metadata (corrupting `pages_received`/`pages_acked` counters) or overlapping NOC register space.

2. **L1 address alignment violations** (3b): NOC reads and writes have alignment requirements that differ per architecture (16-byte for L1, 32/64-byte for DRAM/PCIe). Misaligned addresses can cause transactions to silently fail.

3. **Bank conflict causing back-pressure** (3c): Multiple simultaneous accesses to the same L1 bank create back-pressure, especially on Blackhole where inline writes use all four memory ports.

4. **Buffer overflow into reserved regions** (3d): Writing beyond an allocated buffer into firmware mailbox space, CB metadata, or reserved memory regions corrupts state that other cores depend on.

5. **DRAM address via unsupported API** (3e): Some APIs (e.g., `noc_inline_dw_write`) do not support DRAM addresses. Using them with DRAM targets produces undefined behavior.

6. **Stale address after buffer reallocation** (3f): If a kernel caches a buffer address and the buffer is reallocated, the stale address may point to invalid or repurposed memory.

### Chapter References

Memory-related hang debugging: Chapter 3. Address sanitization: Chapter 6.

---

## Category 4: Dispatch Hangs

### Definition

A dispatch hang occurs when the host-to-device command dispatch pipeline fails to deliver a go-signal, launch message, or completion notification. The device firmware is functioning correctly -- it is waiting for commands that never arrive.

### Symptoms

- Worker cores frozen at `GW` (waiting for go-signal)
- The dispatch core itself may be frozen at a NOC barrier or semaphore wait
- Host `EnqueueProgram` or `Finish` call never returns
- No kernel execution occurs (waypoint never reaches `"R"`)

### RISC-V Core Involvement

| Core | Typical Waypoint | Role |
|---|---|---|
| BRISC (DM0) on worker cores | `GW` | Waiting for go-signal from dispatch |
| BRISC (DM0) on dispatch core | `NWBW`, `NSW`, or other | Stuck trying to deliver messages |

### Common Root Causes

1. **Dispatch core hung on NOC write** (4a): The dispatch core multicasts launch messages and go-signals to worker cores. If the NOC write fails, worker cores never receive the go-signal. The source code in `brisc.cc` explicitly mentions a "hang workaround" related to barriers before multicasting the launch message.

2. **Launch message ring buffer full** (4b): If the worker has not consumed previous launch messages, the dispatch core may spin waiting for ring buffer space.

3. **Completion notification not received** (4c): After kernel execution, each worker sends a completion notification back to the dispatch core via `notify_dispatch_core_done()`. If this NOC write fails, the dispatch core waits for a completion that never arrives.

4. **Read pointer reset failure** (4d): The `RUN_MSG_RESET_READ_PTR` signal is used to reset the launch message read pointer. If this mechanism fails, the worker and dispatch core can enter a deadlock.

5. **Trace replay failure** (4e): When replaying a traced program, if the trace was captured with different CB or semaphore initial state than what exists during replay, the replayed kernels may hang.

### Chapter References

Dispatch hang debugging: Chapter 4.

---

## Category 5: Multi-Chip Hangs

### Definition

A multi-chip hang involves failures across chip boundaries, typically through Ethernet links. These hangs occur when a NOC transaction must traverse an Ethernet link to reach a remote chip, and the remote chip or the link itself is in a failed state.

### Symptoms

- Ethernet data-mover cores frozen at `eth_noc_semaphore_wait` or ethernet-specific barriers
- Worker cores on one chip frozen at `NSW` or `NWBW` waiting for signals from a remote chip
- One chip in a multi-chip system may be entirely functional while another is hung, creating an asymmetric failure
- Fabric router hangs on Blackhole (known issue #28758 related to inline write counters)
- Hangs that only reproduce at multi-chip scales (N300, T3K, Galaxy) but not on single-chip

### RISC-V Core Involvement

| Core | Context | Typical Waypoint |
|---|---|---|
| Ethernet core RISC-V (ERISC) | Data mover | Stuck in `eth_noc_semaphore_wait` |
| BRISC/NCRISC on worker | Cross-chip semaphore | `NSW`, `NSMW` |
| Fabric router core | Routing firmware | Various internal loops |

### Common Root Causes

1. **Remote chip hung or in error state** (5a): If the remote chip has experienced its own hang (of any category), it cannot service requests from the local chip.

2. **Ethernet link degradation** (5b): Physical link issues can cause packet loss on the Ethernet link, resulting in missing semaphore increments or NOC responses.

3. **EDM (Ethernet Data Mover) hang** (5c): The EDM firmware on Ethernet cores manages data movement between chips. If an EDM kernel hangs, all cross-chip traffic through that Ethernet core is blocked.

4. **Fabric router inline-write counter mismatch (Blackhole)** (5d): The known issue where inline write counters in the fabric router must always be updated. If not, the fabric router can enter a state where it spins checking counters that will never match.

5. **Cross-chip semaphore desynchronization** (5e): A semaphore increment from chip A arrives before chip B's core is ready to receive it, causing an overshoot in `noc_semaphore_wait`.

### Chapter References

Multi-chip hang debugging: Chapter 5.

---

## Category 6: Host-Device Hangs

### Definition

A host-device hang occurs when the interface between the host CPU and the Tenstorrent device fails. This includes PCIe communication failures, host-side synchronization errors, and timeout handling issues.

### Symptoms

- Host `Finish()` call never returns, but all device cores show `GW` (idle) -- the device has finished but the host does not know
- PCIe read/write operations fail or return stale data
- Host-side completion polling never sees the expected value
- Device may be fully functional but unreachable from the host

### RISC-V Core Involvement

Typically none directly -- the hang is on the host side. However, if the host-device communication is blocked, the host cannot send new programs, and the device sits idle.

### Common Root Causes

1. **PCIe link degradation** (6a): Hardware-level PCIe issues can cause reads and writes to the device to fail or return incorrect data.

2. **Completion notification via PCIe lost** (6b): When the dispatch mode is `DISPATCH_MODE_DEV`, the completion notification travels from the device core via NOC to the PCIe controller, then to the host.

3. **Hugepage / MMIO failures** (6c): The UMD maps device memory into the host process's address space using hugepages. If hugepage allocation fails or the mapping is corrupted, host-side accesses hang.

4. **Host-side synchronization deadlocks** (6d): The host program may have its own synchronization bugs (e.g., calling `Finish()` on a command queue that was never submitted to).

### Chapter References

Host-device debugging: Chapter 4.

---

## Symptoms Cross-Reference Matrix

This matrix provides a quick lookup between observable symptoms and likely hang categories. When triaging a hang, start with the observable symptom and check which categories could produce it.

| Observable Symptom | Cat. 1 Kernel | Cat. 2 NOC | Cat. 3 Memory | Cat. 4 Dispatch | Cat. 5 Multi-Chip | Cat. 6 Host-Device |
|---|---|---|---|---|---|---|
| Watcher shows `CRBW`/`CWFW` | **Primary** | Secondary (cascade) | Secondary (cascade) | No | No | No |
| Watcher shows `NRBW`/`NWBW` | No | **Primary** | **Primary** | No | Possible | No |
| Watcher shows `NSW`/`NSMW` | **Primary** | No | No | No | **Primary** | No |
| Watcher shows `RP2W`/`NWPW` | No | **Primary** | Possible | No | No | No |
| All cores at `GW` | No | No | No | **Primary** | No | **Primary** |
| No waypoint (core never entered kernel) | No | No | No | Possible | No | **Primary** |
| Heartbeat stopped | Yes (all cores stuck) | Yes | Yes | Yes | Yes | No |
| Assert mailbox populated | Possible | Possible | Possible | No | No | No |
| Sanitize mailbox populated | No | **Yes** (if watcher on) | **Yes** (if watcher on) | No | No | No |
| Survives kernel kill, needs chip reset | No | Sometimes | Sometimes | Sometimes | Usually | Sometimes |
| Non-deterministic | Rare | Rare | Rare | Rare | Possible | Rare |

---

## Compounding: How Hangs Cascade

Real-world hangs rarely involve a single core or a single category. The most common pattern is a **cascade**, where one hang causes dependent operations to stall, which in turn causes further stalls.

### Cascade Pattern 1: Intra-Tensix CB Pipeline (Vertical Cascading)

A hang in any subordinate core cascades through the CB chain to BRISC:

```
TRISC0 (Unpack) hangs at CWFW waiting for reader input
    -> TRISC1 (Math) stalls (no input from TRISC0)
       -> TRISC2 (Pack) stalls (no input from TRISC1)
          -> NCRISC (Writer) hangs at CWFW waiting for pack output
             -> BRISC hangs at NTW waiting for all subordinates
```

Root cause: Reader kernel (BRISC) never pushed tiles to the input CB.

### Cascade Pattern 2: Cross-Core Semaphore Chain (Horizontal Cascading)

```
Core (3,4) BRISC hangs at NSW waiting for semaphore from Core (3,5)
    -> Core (3,5) NCRISC hangs at NWBW (write to Core (3,4) never acks)
       -> NOC transaction to Core (3,4) is stuck because Core (3,4) has
          a pending inline write that is back-pressuring the L1 ports (BH)
```

Root cause: Blackhole inline-write back-pressure. Categories involved: Kernel (semaphore protocol), NOC (back-pressure), Memory (L1 port contention).

### Cascade Pattern 3: Multi-Chip Propagation (Cross-Chip Cascading)

```
Chip 0 Core (5,5) hangs at CWFW (kernel bug -- producer exits early)
    -> Chip 0 never sends completion semaphore to Chip 1
       -> Chip 1 Core (2,3) hangs at NSW waiting for Chip 0's semaphore
          -> Chip 1 Ethernet data mover hangs in eth_noc_semaphore_wait
             -> Chip 2 (connected to Chip 1 via Ethernet) hangs similarly
```

Root cause: Kernel bug on Chip 0. Categories involved: Kernel, Multi-Chip.

### Cascade Pattern 4: Cascading to Dispatch

When enough cores hang, the dispatch system itself can be affected:

```
Many worker cores hang (any category)
    -> Dispatcher waits for completion notifications that never arrive
       -> Dispatcher's completion counter never reaches expected value
          -> Dispatcher never sends go signals for the next batch
             -> Remaining healthy cores hang at GW
                -> Full device hang
```

### The Compounding Principle

Every hang can potentially compound across categories. The practical implication is that during triage, the **first** frozen waypoint in the causal chain is the most important. Later waypoints (e.g., `NTW` on BRISC) are symptoms of the root cause, not the root cause itself.

**Diagnostic principle**: When multiple cores are hung, always trace the dependency chain backward to find the core that hung *first*. This is the core whose waypoint shows a direct interaction with the external world (NOC barrier, hardware register wait) rather than a local synchronization primitive (CB wait, subordinate wait).

---

## Decision Tree for Rapid Triage

When a hang is detected (host timeout, watcher alert, or manual observation), use this decision tree to classify it:

```
1. Are any worker cores past "R" (Running)?
   |
   +-- NO: All cores at "GW"
   |       -> DISPATCH HANG (Category 4) or HOST-DEVICE HANG (Category 6)
   |          Check: Is the dispatch core itself hung?
   |            YES --> Check dispatch core waypoint (compound with Category 2)
   |            NO  --> Category 6: Go message routing issue or host-side failure
   |
   +-- YES: At least one core past "R"
       |
       2. Is any ERISC core stuck in an ethernet spin loop?
          |
          +-- YES: -> MULTI-CHIP HANG (Category 5), likely root cause
          |         Verify ethernet link status.
          |
          +-- NO: Continue to step 3
               |
               3. What waypoint is the first hung core showing?
                  |
                  +-- "CRBW" or "CWFW"
                  |   -> KERNEL HANG (Category 1, CB protocol)
                  |      Check: Are push/pop counts matched?
                  |      Is the partner core also stuck?
                  |        YES at a CB waypoint --> CB deadlock (1a)
                  |        YES at a NOC barrier --> compound (Category 1 + 2)
                  |        NO (producer exited) --> tile count mismatch (1e)
                  |
                  +-- "NRBW" or "NWBW"
                  |   -> NOC HANG (Category 2) or MEMORY HANG (Category 3)
                  |      Check: NOC status registers for outstanding txns
                  |      Check: Were the target addresses valid? (sanitize mailbox)
                  |      If target is DRAM --> likely Category 3
                  |      If target is another core --> Category 2
                  |      If target is on another chip --> Category 5
                  |      Is this Blackhole with inline writes to L1? --> BH-specific (2e)
                  |
                  +-- "NSW" or "NSMW"
                  |   -> KERNEL HANG (Category 1, semaphore) or MULTI-CHIP HANG (Category 5)
                  |      Check: Is the signaling core on the same chip?
                  |      If same chip: Check signaling core's waypoint
                  |      If different chip: -> MULTI-CHIP HANG (Category 5)
                  |
                  +-- "RP2W" or "NWPW"
                  |   -> NOC HANG (Category 2, hardware command buffer stall)
                  |      Check: NOC CMD_CTRL register state
                  |      Likely requires chip reset
                  |
                  +-- "NTW"
                  |   -> DERIVED HANG: BRISC waiting for subordinate
                  |      Check: Which subordinate is NOT at waypoint "D"?
                  |      Classify based on subordinate's waypoint (recurse)
                  |
                  +-- "NABW"
                  |   -> BLACKHOLE-SPECIFIC: Atomics flush barrier
                  |      Check: Remote CB interface setup path
                  |      Check: NOC atomics outstanding count
                  |
                  +-- "R" (Running kernel)
                  |   -> Kernel is in user code, not at a known blocking primitive
                  |      Possible user spin-loop or infinite loop
                  |
                  +-- No waypoint / stale waypoint
                      -> Watcher may not be enabled, or core is stuck before
                         first waypoint. Check assert mailbox.
                         If assert mailbox populated --> assert_and_hang (not a true hang)
```

### Using the Decision Tree

1. Start at the top: determine whether any cores reached kernel execution (`"R"` or beyond).
2. Find the **first** hung core in the causal chain. This is the core whose waypoint indicates a direct blocking primitive, not a derived wait like `NTW`.
3. The waypoint directly tells you which blocking primitive is stuck.
4. Use the blocking primitive's exit condition (from Section 1.2) to determine what event is missing.
5. Trace backwards to find why that event cannot occur -- this leads to the root cause and the correct category.

---

## Hang Frequency by Category

Based on patterns observed in the codebase (workarounds, comments, and issue references), the approximate frequency ranking of hang categories is:

1. **Kernel-level (Category 1):** Most frequent. CB mismatches and tile-count errors are the most common user-facing hang causes.
2. **NOC (Category 2):** Second most frequent. Often interacts with Category 1 when data-movement kernels stall.
3. **Dispatch (Category 4):** Moderate frequency. More common during development and testing, less common in production because dispatch code paths are well-tested.
4. **Multi-chip (Category 5):** Frequency increases with scale. Rare on single-chip, common on Galaxy-class systems.
5. **Memory (Category 3):** Relatively rare. Memory subsystem hangs are usually permanent (hardware failure) rather than transient.
6. **Host-device (Category 6):** Rare. Usually indicates a system-level issue (PCIe, driver, OS) rather than a Tenstorrent-specific bug.

---

**Next:** [`04_hang_causes_across_architectures.md`](./04_hang_causes_across_architectures.md)
