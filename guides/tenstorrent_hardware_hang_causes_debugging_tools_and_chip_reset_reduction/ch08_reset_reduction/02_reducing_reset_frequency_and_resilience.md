# 02 -- Reducing Reset Frequency and Building Resilience

## Summary

This section presents actionable strategies for reducing the frequency of hangs that require chip resets, organized into four areas: prevention practices that eliminate hang root causes at development time, multi-chip resilience patterns that tolerate partial failures, graceful recovery mechanisms that avoid resets when hangs do occur, and test infrastructure that catches hang-prone code before deployment. Each recommendation is tied to specific hang categories from Chapters 2-5, with quantitative impact estimates. The section concludes with a 20-item prevention checklist suitable for code review.

## Prerequisites

- Chapter 2 ([`02_circular_buffer_deadlocks.md`](../ch02_kernel_and_noc_hangs/02_circular_buffer_deadlocks.md), [`03_noc_address_sanitization_and_violations.md`](../ch02_kernel_and_noc_hangs/03_noc_address_sanitization_and_violations.md), [`04_noc_barrier_and_semaphore_hangs.md`](../ch02_kernel_and_noc_hangs/04_noc_barrier_and_semaphore_hangs.md)): CB deadlocks, NOC barrier violations, semaphore misuse.
- Chapter 3 ([`01_l1_memory_corruption_and_overflow.md`](../ch03_memory_related_hangs/01_l1_memory_corruption_and_overflow.md)): L1 overflow, DRAM violations.
- Chapter 4 ([`02_host_synchronization_and_timeout_detection.md`](../ch04_dispatch_and_host_device_hangs/02_host_synchronization_and_timeout_detection.md)): Dispatch timeouts, trace replay.
- Chapter 5 ([`02_ccl_collective_operation_hangs.md`](../ch05_multi_chip_and_ccl_hangs/02_ccl_collective_operation_hangs.md)): CCL deadlocks, Ethernet link failures.
- Chapter 6 ([`01_watcher_system.md`](../ch06_debugging_tools/01_watcher_system.md), [`05_profiler_tracy_and_noc_debug.md`](../ch06_debugging_tools/05_profiler_tracy_and_noc_debug.md)): Watcher, NOC debug dump, debug delay.
- Section 01 of this chapter (reset hierarchy).

---

## 1. Prevention Practices That Reduce Hang Frequency

These practices target the root causes documented in Chapters 2-5. Each practice includes the hang category it addresses, the specific failure mode it prevents, and an estimate of what fraction of hangs in that category it would catch.

### 1.1 Always Enable Watcher NOC Sanitization During Development

**Addresses:** Chapter 2, NOC address violations ([`03_noc_address_sanitization_and_violations.md`](../ch02_kernel_and_noc_hangs/03_noc_address_sanitization_and_violations.md)); Chapter 3, L1 overflow corruption leading to bad NOC addresses.

**Practice:** Set `TT_METAL_WATCHER=120` (120ms polling interval) during all development and testing. The watcher NOC sanitizer validates every NOC transaction on the device side, catching malformed addresses, alignment violations, and out-of-range accesses *before* they become hardware-level NOC deadlocks.

**Impact estimate:** Approximately 60-70% of NOC-related hangs are caused by bad addresses (wrong coordinates, misaligned addresses, out-of-range L1 accesses). All of these are caught by watcher sanitization *before* they reach the NOC hardware. The remaining 30-40% (multicast path conflicts, flow control deadlocks, dependency cycles) are not address-related and require other mechanisms.

**Performance cost:** Watcher disables DMA operations and adds sanitization checks to every NOC transaction. Typical overhead is 2-5x on NOC-heavy workloads. Not suitable for production, but acceptable for development and CI testing.

**Configuration detail:** Individual watcher features can be disabled to reduce overhead while keeping the most valuable checks:

| Feature | Env Var to Disable | What It Catches |
|---|---|---|
| NOC sanitization | `TT_METAL_WATCHER_DISABLE_SANITIZE_NOC` | Address violations, range errors |
| Read-only L1 protection | `TT_METAL_WATCHER_DISABLE_SANITIZE_READ_ONLY_L1` | Writes to read-only L1 regions |
| Write-only L1 protection | `TT_METAL_WATCHER_DISABLE_SANITIZE_WRITE_ONLY_L1` | Reads from write-only regions |
| CB sanitization | `TT_METAL_WATCHER_DISABLE_CB_SANITIZE` | CB out-of-bounds access |
| Waypoints | `TT_METAL_WATCHER_DISABLE_WAYPOINT` | Per-core execution progress tracking |
| Stack usage | `TT_METAL_WATCHER_DISABLE_STACK_USAGE` | Stack overflow detection |
| Assertions | `TT_METAL_WATCHER_DISABLE_ASSERT` | Firmware ASSERT() macro |
| Ethernet link status | `TT_METAL_WATCHER_DISABLE_ETH` | Link-down detection |

The minimum recommended configuration for hang prevention is NOC sanitization + CB sanitization + waypoints.

**Environment configuration patterns:**

```bash
# Development environment (maximum diagnostics)
export TT_METAL_WATCHER=120
export TT_METAL_WATCHER_DUMP_ALL=1

# CI environment (faster polling, less log volume)
export TT_METAL_WATCHER=50

# Production (watcher disabled, lightweight asserts enabled)
unset TT_METAL_WATCHER
export TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1
```

### 1.2 Use Lightweight Kernel Asserts in Production

**Addresses:** Chapter 2 (all kernel-level hangs where preconditions are checkable), Chapter 3 (L1/DRAM range violations).

**Practice:** Enable `TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1` for production builds. Unlike watcher asserts (which require the full watcher infrastructure), lightweight kernel asserts use the `ebreak` instruction on failure:

```c
// When LIGHTWEIGHT_KERNEL_ASSERTS is enabled:
// ASSERT(condition) compiles to:
//   if (!(condition)) asm volatile("ebreak");
// This triggers a RISC-V exception detectable by tt-triage's
// dump_lightweight_asserts.py script.
```

**Impact estimate:** Lightweight asserts catch precondition violations that would otherwise manifest as hangs seconds or minutes later -- verifying that a CB has been properly configured, that a semaphore address is within valid L1 range, etc. These "early exit" conditions convert approximately 15-20% of what would become undiagnosable hangs into immediate, identifiable assertion failures with source location information.

### 1.3 Follow Circular Buffer API Constraints

**Addresses:** Chapter 2, CB deadlock patterns ([`02_circular_buffer_deadlocks.md`](../ch02_kernel_and_noc_hangs/02_circular_buffer_deadlocks.md)).

The CB API has strict constraints that, when violated, produce hangs that are difficult to diagnose without watcher:

**Rule 1: ntiles must evenly divide CB size.**

```c
// WRONG -- will silently deadlock if CB has 8 tile slots
// and ntiles=3 (8 is not divisible by 3)
cb_reserve_back(cb_out, 3);
cb_push_back(cb_out, 3);

// CORRECT -- use ntiles that evenly divides CB capacity
cb_reserve_back(cb_out, 2);  // 8 / 2 = 4, clean division
cb_push_back(cb_out, 2);
```

**Rule 2: Consistent ntiles across calls.** Using different `ntiles` values for successive `cb_reserve_back` or `cb_wait_front` calls on the same CB can desynchronize producer and consumer pointers.

**Rule 3: Single-thread CB pointer updates.** Only one RISC processor should call `cb_push_back` for a given CB, and only one should call `cb_pop_front`. Concurrent updates from multiple RISCs corrupt the pointers.

**Rule 4: Producer-consumer pairing.** Every CB must have exactly one producer (writer) and one consumer (reader). Missing consumer calls cause the producer to stall permanently once the buffer fills.

**Impact estimate:** CB deadlocks account for an estimated 30-40% of kernel-level hangs. Following these constraints eliminates essentially all of them.

**Danger:** The CB API does NOT validate these constraints at runtime (even with watcher enabled, though CB sanitization catches some out-of-bounds cases). Violations are silent until the hang occurs. This is why the static analysis proposal (Section 03, Proposal 8) is important.

### 1.4 Validate NOC Addresses Before Issuing Transactions

**Addresses:** Chapter 2, NOC address violations; Chapter 3, DRAM address range violations.

**Practice:** Before issuing any NOC read or write, validate:
- The target coordinates `(x, y)` are within the valid grid for the current architecture (accounting for harvested rows)
- The L1 address is within `[0, L1_SIZE)` for the target core
- The DRAM address is within the valid bank range
- Alignment requirements are met (e.g., 16-byte alignment for NOC transactions, 32-byte alignment for DRAM)
- Multicast targets form a valid rectangular grid with no harvested cores

For dynamically computed addresses (from runtime arguments, computed buffer offsets):

```c
// DEFENSIVE -- validate before issuing
uint32_t dram_addr = get_noc_addr_dram(bank_id, offset);
ASSERT(offset + transfer_size <= dram_bank_size);
ASSERT(l1_addr >= MEM_L1_BASE && l1_addr + transfer_size <= MEM_L1_SIZE);
noc_async_read(dram_addr, l1_addr, transfer_size);
```

This is especially important when buffer addresses come from runtime arguments (`get_arg_val<uint32_t>(arg_idx)`), because a host-side allocation failure can result in a zero or garbage address being passed to the kernel (see Chapter 3, [`04_allocation_failures_and_silent_oom.md`](../ch03_memory_related_hangs/04_allocation_failures_and_silent_oom.md)).

### 1.5 Proper Barrier Placement

**Addresses:** Chapter 2, NOC barrier hangs ([`04_noc_barrier_and_semaphore_hangs.md`](../ch02_kernel_and_noc_hangs/04_noc_barrier_and_semaphore_hangs.md)); specifically `WRITE_FLUSH_BARRIER` and `READ_BARRIER` issues detected by the NOC debug dump feature.

**Rule:** Every `noc_async_write` must have a corresponding `noc_async_write_barrier` before any code that depends on the write having completed. Every `noc_async_read` must have a corresponding `noc_async_read_barrier` before the L1 destination is read. Missing barriers cause data races and counter overflow leading to infinite barrier waits.

A common mistake is omitting the write barrier before a semaphore increment:

```c
// WRONG -- semaphore may arrive before data
noc_async_write(data_addr, remote_data_addr, size);
noc_semaphore_inc(remote_sem_addr, 1);  // Race!

// CORRECT -- barrier ensures data is written before signal
noc_async_write(data_addr, remote_data_addr, size);
noc_async_write_barrier();
noc_semaphore_inc(remote_sem_addr, 1);  // Safe: data is committed
```

On Wormhole, a write barrier is also required before multicast operations that are not linked to a previous multicast, to prevent mcast path reservation hangs.

**Impact estimate:** Missing barriers are a contributing factor in approximately 20% of NOC-related hangs. The `TT_METAL_NOC_DEBUG_DUMP=1` feature catches all of these statically after a test run, reporting `NOCDebugState` issue types: `WRITE_FLUSH_BARRIER`, `READ_BARRIER`, `UNFLUSHED_WRITE_AT_END`, `WRITE_TO_LOCKED_CORE_LOCAL_MEM`, `WRITE_TO_LOCKED_CB`.

### 1.6 Semaphore Initialization and Protocol Discipline

**Addresses:** Chapter 2, [`04_noc_barrier_and_semaphore_hangs.md`](../ch02_kernel_and_noc_hangs/04_noc_barrier_and_semaphore_hangs.md).

**Requirements:**
1. **Initialize before use:** All semaphores must be initialized (typically to 0) before any core begins execution. The host should write initial values during program setup.
2. **Unique addresses:** Each coordination pair must use a distinct semaphore address. Sharing semaphore addresses between independent coordination groups causes cross-talk.
3. **Reset between iterations:** When running multiple iterations of a kernel, semaphores must be reset to their initial state between iterations. A common bug: semaphore incremented in iteration N but not decremented, so iteration N+1 starts with a non-zero value.
4. **Matching increment/wait counts:** If core A increments a semaphore N times, the waiting core must expect exactly N increments.
5. **Prefer `noc_semaphore_wait_min` over `noc_semaphore_wait`:** The `wait_min` variant checks `>= val` instead of `== val`, making it resilient to double-increments or out-of-order arrivals.

### 1.7 NOC Transaction Ordering to Avoid Circular Dependencies

**Addresses:** Chapter 2, linked transaction deadlocks.

**Rule:** When multiple cores exchange data via NOC, the transaction ordering must be acyclic. A classic circular dependency:

```
Core A writes to Core B's L1 (waiting for B to acknowledge)
Core B writes to Core A's L1 (waiting for A to acknowledge)
```

If both cores issue their writes simultaneously and the NOC paths share resources, neither write can complete -- a hardware-level deadlock requiring Level 2 warm reset.

**Prevention:** Establish a total ordering on cores. Core with the lower coordinate always writes first; the other core waits for the write to land (via semaphore), then issues its own write.

### 1.8 Timeout Wrappers for Semaphore Waits

**Addresses:** All categories that result in infinite semaphore waits.

**Practice:** Wrap `noc_semaphore_wait` with a bounded retry count:

```cpp
// Device-side timeout wrapper pattern
inline void noc_semaphore_wait_with_timeout(
    volatile tt_l1_ptr uint32_t* sem_addr,
    uint32_t target_val,
    uint32_t max_cycles) {
    uint32_t start = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    while (*sem_addr != target_val) {
        uint32_t now = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        if ((now - start) > max_cycles) {
            // Timeout: write diagnostic info to ring buffer, then assert
            WATCHER_RING_BUFFER_PUSH(*sem_addr);
            ASSERT(false);  // Triggers controlled hang with diagnostic info
        }
    }
}
```

This converts an undiagnosable infinite hang into a controlled assertion with timing information, reducing diagnostic time from hours to seconds.

---

## 2. Multi-Chip Resilience Patterns

Multi-chip hangs (Chapter 5) are among the most expensive because they require coordinated resets across multiple devices. The following patterns reduce multi-chip hang frequency.

### 2.1 CCL Operation Ordering to Avoid Cross-Rank Deadlocks

**Addresses:** Chapter 5, [`02_ccl_collective_operation_hangs.md`](../ch05_multi_chip_and_ccl_hangs/02_ccl_collective_operation_hangs.md).

**Rule:** All devices participating in a collective operation must call the same collective in the same order. A common deadlock:

```
Rank 0: all_gather(tensor_A), all_gather(tensor_B)
Rank 1: all_gather(tensor_B), all_gather(tensor_A)    // WRONG ORDER
```

**Prevention:** Use a centralized operation schedule that all ranks follow identically. In TTNN, the model definition implicitly defines the operation order -- the risk arises when different ranks take different code paths (e.g., conditional logic based on rank ID that changes operation ordering).

### 2.2 Ethernet Link Health Monitoring

**Addresses:** Chapter 5, [`01_ethernet_and_fabric_fundamentals.md`](../ch05_multi_chip_and_ccl_hangs/01_ethernet_and_fabric_fundamentals.md).

The erisc firmware monitors link status via `WATCHER_CHECK_ETH_LINK_STATUS()` (defined in `tt_metal/hw/inc/api/debug/eth_link_status.h`). When a link goes down:

```cpp
// tt_metal/hw/inc/api/debug/eth_link_status.h
void hang_on_down_link() {
    v->link_down = 1;
    go_message_ptr->signal = RUN_MSG_DONE;
    internal_::disable_erisc_app();
    erisc_exit();
}
```

This is the most resilient failure handling in the current codebase: instead of hanging, the erisc core exits gracefully. The watcher device reader tracks link-down events and link retraining counts per core.

**Gap:** While the erisc core itself exits gracefully, the *workload* depending on that link still hangs -- the CCL or fabric operation waiting for data from the downed link will spin indefinitely. The resilient CCL proposal (Section 03, Proposal 10) addresses this gap.

The `skip_eth_cores_with_retrain` runtime option allows the system to automatically exclude unstable Ethernet links from the usable topology, degrading performance but preventing hangs.

### 2.3 Fabric Topology Validation Before Launch

**Addresses:** Chapter 5, [`03_topology_and_mesh_configuration_hangs.md`](../ch05_multi_chip_and_ccl_hangs/03_topology_and_mesh_configuration_hangs.md).

**Practice:** Before launching multi-chip workloads, validate:

1. **Device count:** Check that the number of active devices matches the mesh configuration. The `DeviceManager::add_devices_to_pool` function enforces this with a `TT_FATAL` when fabric is enabled but not all devices are active.

2. **Link status:** Check that all expected Ethernet links are up.

3. **Topology:** For operations that assume a specific topology (ring, mesh, torus), verify that the actual topology matches. The fabric `FabricReliabilityMode` enum in `tt_metal/api/tt-metalium/experimental/fabric/fabric_types.hpp` provides three modes:
   - `STRICT_SYSTEM_HEALTH_SETUP_MODE` (0): Requires all links/devices to match the mesh graph descriptor. Any discrepancy is an error.
   - `RELAXED_SYSTEM_HEALTH_SETUP_MODE` (1): Allows initialization with fewer routing planes based on live link count.
   - `DYNAMIC_RECONFIGURATION_SETUP_MODE` (2): Placeholder for runtime reconfiguration (not yet implemented).

`STRICT_SYSTEM_HEALTH_SETUP_MODE` is currently the default and is the safest option for preventing topology-related hangs.

### 2.4 Verifying All Ranks Participate in Collectives

**Addresses:** Chapter 5, CCL deadlocks from missing participants.

**Rule:** Before launching any CCL operation, verify that all expected ranks are ready. A single rank that fails to participate will cause all other ranks to wait indefinitely. The fabric initialization code sets `FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE` during device manager setup, ensuring that if any device fails during fabric setup, the initialization fails early.

---

## 3. Graceful Recovery Mechanisms

These mechanisms allow the system to detect and respond to hangs without requiring a Level 2+ reset.

### 3.1 The ERISC Graceful Exit Pattern

**Addresses:** Chapter 5, Ethernet link failures.

The erisc exit pattern (`hang_on_down_link()` in `eth_link_status.h`) is the gold standard for resilient failure handling:

1. Record diagnostic information (link_down flag in watcher mailbox)
2. Signal completion to prevent cascading waits (`go_message_ptr->signal = RUN_MSG_DONE`)
3. Disable the application layer
4. Exit to base firmware via `erisc_exit()`

This pattern should be extended to other failure modes. Today it is only used for Ethernet link-down events and assertion failures on erisc cores. Tensix cores lack an equivalent -- they hang in `while(1){}` loops when assertions fire, which requires a Level 2 reset to clear.

**Why Tensix cores cannot simply adopt this pattern:** BRISC, NCRISC, and TRISC share state through circular buffers and L1 semaphores. If one RISC on a Tensix core encounters an error and exits, the other RISCs may be stuck in spin-loops waiting for the errored RISC. Only BRISC can safely terminate all processors on a core via the subordinate sync protocol.

**Quantitative impact:** If all watcher assertions on Tensix cores used an erisc-style graceful exit (signal completion, write diagnostics, halt cleanly), the pattern would convert approximately 25-30% of Level 2 resets into Level 0 graceful terminations.

### 3.2 Inspector Auto-Serialization on Dispatch Timeout

**Addresses:** Chapter 4, dispatch timeout scenarios.

When a dispatch timeout is detected, `MetalContext::on_dispatch_timeout_detected()` automatically:

1. Serializes all Inspector data (program states, workload states, device states) to disk
2. Optionally executes `TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE` (e.g., `./tools/tt-triage.py`)
3. Throws an exception that unwinds the host stack

**Value:** This captures diagnostic state *before* the reset destroys it. Without this, the developer must reproduce the hang to gather diagnostics. With auto-serialization, the first occurrence provides actionable data.

**Configuration for maximum diagnostic capture:**

```bash
export TT_METAL_OPERATION_TIMEOUT_SECONDS=120
export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="./tools/tt-triage.py --verbosity=4"
export TT_METAL_DISPATCH_PROGRESS_UPDATE_MS=1000
```

### 3.3 Dispatch Progress Heartbeats

**Addresses:** Chapter 4, dispatch hang detection.

`TT_METAL_DISPATCH_PROGRESS_UPDATE_MS` configures periodic progress heartbeats from dispatch kernels. The host monitors these heartbeats and distinguishes "slow" from "stuck" -- the timeout clock resets with each progress update, preventing false-positive timeouts on legitimately long-running workloads. A 100ms heartbeat interval allows detecting a dispatch stall within 200ms (two missed heartbeats), compared to the default operation timeout of many seconds.

### 3.4 Firmware Watchdog Timer Registers

The RISC-V debug registers include `RISCV_DEBUG_REG_WATCHDOG_TIMER` entries that, in principle, allow per-core watchdog timers. Today, these are not actively used by the runtime -- they are referenced in HAL definitions but not configured during normal operation. However, the M3-level watchdog is active: if firmware becomes completely unresponsive, the M3 watchdog automatically triggers a chip reset after the configured timeout (default 10 seconds, configurable via the `auto_reset_timeout` kernel module parameter). This is explored in detail in Proposal 7 of Section 03.

### 3.5 Watcher Auto-Unpause Mode

**Addresses:** Development-time debugging with watcher `PAUSE()` breakpoints.

The `auto_unpause` watcher setting automatically clears the pause flag after the watcher has read the core's state. This allows `PAUSE()` to function as a non-blocking breakpoint: the core pauses, the watcher captures its state, the watcher clears the pause, and the core resumes. Without auto-unpause, a PAUSE() that is not manually cleared causes a permanent hang.

---

## 4. Test Infrastructure for Hang Detection

### 4.1 Systematic Hang Reproduction Tests

The codebase includes dedicated tests for hang-related functionality:

| Test File | What It Validates |
|---|---|
| `test_assert.cpp` | Watcher assertion detection and reporting |
| `test_link_training.cpp` | Ethernet link failure detection and recovery |
| `test_stack_usage.cpp` | Stack overflow detection via watcher |
| `test_pause.cpp` | Watcher pause/resume mechanism |
| `test_noc_sanitize_delays.cpp` | NOC sanitization with debug delay timing |
| `test_reads_writes.cpp` | NOC debugging issue detection for reads/writes |
| `test_scoped_lock.cpp` | Scoped lock debugging for CB and memory regions |
| `test_fabric_deadlock_stability_*.yaml` | Multi-chip fabric deadlock avoidance validation |

### 4.2 The `hang_device` Test Operation

The `hang_device` operation (`ttnn/cpp/ttnn/operations/experimental/test/hang_device/`) deliberately induces a hang on a specified device, allowing tooling validation:

- Verify that watcher detects the hang and reports correct waypoint information
- Verify that tt-triage can extract callstacks from the hung core
- Verify that dispatch timeout detection fires within the expected window
- Verify that Inspector auto-serialization captures the expected state
- Verify that warm reset successfully recovers the device after the hang

This is invaluable for CI: a test that deliberately hangs a device, validates that all diagnostic tools capture the expected information, resets the device, and verifies recovery.

### 4.3 Debug Delay for Race Condition Testing

**Addresses:** Chapter 2, timing-dependent hangs.

The debug delay feature (`TT_METAL_READ_DEBUG_DELAY_CORES`, `TT_METAL_WRITE_DEBUG_DELAY_CORES`, `TT_METAL_ATOMIC_DEBUG_DELAY_CORES`) artificially slows NOC operations on specified cores:

- A hang that occurs "sometimes" can often be made 100% reproducible by adding delays to the right cores
- The `WATCHER_DEBUG_DELAY` compile-time constant controls the number of delay cycles per transaction
- The `debug_insert_delays_msg_t.feedback` field confirms that delays are actually being applied

**Practice for CI:** Include debug-delay runs in the test matrix for NOC-heavy kernels. A kernel that passes without delays but hangs with delays has a latent race condition that will eventually manifest in production.

### 4.4 NOC Debug Dump for Regression Testing

**Addresses:** Chapter 2, missing barrier issues.

Running tests with `TT_METAL_NOC_DEBUG_DUMP=1` and checking for zero reported issues provides a regression gate for NOC correctness.

**CI integration pattern:**
1. Run the test with `TT_METAL_NOC_DEBUG_DUMP=1`
2. Parse the output for NOC debug issues
3. Fail the test if any `UNFLUSHED_WRITE_AT_END` or `WRITE_TO_LOCKED_*` issues are found
4. Alert (but don't fail) on `WRITE_FLUSH_BARRIER` or `READ_BARRIER` issues (these may be intentional in some patterns)

---

## 5. Summary: Estimated Aggregate Impact

Combining all prevention practices, resilience patterns, and recovery mechanisms:

| Strategy | Hang Categories Addressed | Estimated Reduction in Resets |
|---|---|---|
| Watcher NOC sanitization in dev | Ch2 NOC address errors | ~60-70% of NOC hangs caught before deployment |
| CB API constraint adherence | Ch2 CB deadlocks | ~90% of CB deadlocks eliminated |
| Lightweight kernel asserts | Ch2-3 precondition violations | ~15-20% of hangs converted to assertions |
| Missing barrier detection (NOC debug dump) | Ch2 barrier violations | ~20% of NOC hangs caught in CI |
| Semaphore discipline | Ch2 semaphore misuse | ~80% of semaphore hangs eliminated |
| CCL ordering enforcement | Ch5 cross-rank deadlocks | ~90% of CCL ordering hangs eliminated |
| Fabric topology validation | Ch5 topology misconfiguration | ~70% of topology hangs prevented |
| ERISC graceful exit (existing) | Ch5 link failures | Already converts link-down from hang to clean exit |
| Inspector auto-serialize | Ch4 dispatch timeouts | Does not prevent hang but preserves diagnostics |
| Dispatch heartbeats | Ch4 dispatch stalls | Does not prevent hang but detects 5-10x faster |

**Net estimate:** If all prevention practices were followed rigorously, approximately 50-60% of hangs that currently require Level 2+ resets could be either prevented entirely (caught at dev/CI time) or converted to Level 0 graceful recoveries. The remaining 40-50% require the future tooling improvements proposed in Section 03.

---

## 6. Prevention Checklist

The following checklist summarizes all prevention practices from this section. It can be used as a code review checklist for new kernels and ops:

| # | Check | Hang Category Prevented | Chapter Reference |
|---|-------|------------------------|-------------------|
| 1 | Watcher enabled in dev/CI environments | All NOC, memory, CB violations | Ch2, Ch3 |
| 2 | Lightweight kernel asserts in production | Silent corruption | Ch6, [`01_watcher_system.md`](../ch06_debugging_tools/01_watcher_system.md) |
| 3 | CB ntiles evenly divides CB capacity | CB deadlock | Ch2, [`02_circular_buffer_deadlocks.md`](../ch02_kernel_and_noc_hangs/02_circular_buffer_deadlocks.md) |
| 4 | Consistent ntiles across all CB calls | CB deadlock | Ch2, [`02_circular_buffer_deadlocks.md`](../ch02_kernel_and_noc_hangs/02_circular_buffer_deadlocks.md) |
| 5 | Single-thread CB pointer updates | CB state corruption | Ch2, [`02_circular_buffer_deadlocks.md`](../ch02_kernel_and_noc_hangs/02_circular_buffer_deadlocks.md) |
| 6 | NOC address bounds checked before use | NOC address violation | Ch2, [`03_noc_address_sanitization_and_violations.md`](../ch02_kernel_and_noc_hangs/03_noc_address_sanitization_and_violations.md) |
| 7 | Write barrier before semaphore signal | Data-before-signal race | Ch2, [`04_noc_barrier_and_semaphore_hangs.md`](../ch02_kernel_and_noc_hangs/04_noc_barrier_and_semaphore_hangs.md) |
| 8 | Read barrier before reading DMA destination | Stale-data corruption | Ch2, [`04_noc_barrier_and_semaphore_hangs.md`](../ch02_kernel_and_noc_hangs/04_noc_barrier_and_semaphore_hangs.md) |
| 9 | Write barrier before multicast (Wormhole) | Mcast path reservation hang | Ch2, [`04_noc_barrier_and_semaphore_hangs.md`](../ch02_kernel_and_noc_hangs/04_noc_barrier_and_semaphore_hangs.md) |
| 10 | Semaphores initialized before kernel launch | Semaphore hang | Ch2, [`04_noc_barrier_and_semaphore_hangs.md`](../ch02_kernel_and_noc_hangs/04_noc_barrier_and_semaphore_hangs.md) |
| 11 | Unique semaphore addresses per coordination pair | Semaphore aliasing | Ch2, [`04_noc_barrier_and_semaphore_hangs.md`](../ch02_kernel_and_noc_hangs/04_noc_barrier_and_semaphore_hangs.md) |
| 12 | Semaphores reset between loop iterations | Stale semaphore value | Ch2, [`04_noc_barrier_and_semaphore_hangs.md`](../ch02_kernel_and_noc_hangs/04_noc_barrier_and_semaphore_hangs.md) |
| 13 | `noc_semaphore_wait_min` preferred over `wait` | Semaphore overshoot | Ch2, [`04_noc_barrier_and_semaphore_hangs.md`](../ch02_kernel_and_noc_hangs/04_noc_barrier_and_semaphore_hangs.md) |
| 14 | Host-side allocation checked before kernel launch | Garbage address hang | Ch3, [`04_allocation_failures_and_silent_oom.md`](../ch03_memory_related_hangs/04_allocation_failures_and_silent_oom.md) |
| 15 | NOC alignment matches architecture requirements | Alignment stall | Ch3, [`03_alignment_and_tile_size_mismatches.md`](../ch03_memory_related_hangs/03_alignment_and_tile_size_mismatches.md) |
| 16 | Tile sizes consistent across reader/compute/writer | Size mismatch corruption | Ch3, [`03_alignment_and_tile_size_mismatches.md`](../ch03_memory_related_hangs/03_alignment_and_tile_size_mismatches.md) |
| 17 | All CCL ranks execute same collective in same order | Cross-rank deadlock | Ch5, [`02_ccl_collective_operation_hangs.md`](../ch05_multi_chip_and_ccl_hangs/02_ccl_collective_operation_hangs.md) |
| 18 | Ethernet link health verified before multi-chip ops | Link-down hang | Ch5, [`01_ethernet_and_fabric_fundamentals.md`](../ch05_multi_chip_and_ccl_hangs/01_ethernet_and_fabric_fundamentals.md) |
| 19 | Dispatch timeout configured with auto-triage | Undetected hang | Ch4, [`02_host_synchronization_and_timeout_detection.md`](../ch04_dispatch_and_host_device_hangs/02_host_synchronization_and_timeout_detection.md) |
| 20 | NOC debug dump enabled in CI | Missing barrier bugs | Ch6, [`05_profiler_tracy_and_noc_debug.md`](../ch06_debugging_tools/05_profiler_tracy_and_noc_debug.md) |

> Items 1-2, 7-9, and 19-20 are the highest-impact practices: they catch the broadest categories of bugs with the least effort. Prioritize these for immediate adoption.

---

**Previous:** [`01_current_reset_mechanisms.md`](./01_current_reset_mechanisms.md) | **Next:** [`03_future_tooling_proposals.md`](./03_future_tooling_proposals.md)
