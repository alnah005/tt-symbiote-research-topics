# 03 -- Future Tooling Proposals

## Summary

This section presents twelve concrete proposals for future tools and infrastructure improvements that would significantly reduce the frequency and cost of chip resets. Each proposal includes a priority ranking (P0-P3), effort estimate, dependency tracking, the specific codebase gap it addresses, a quantitative impact estimate, and an implementation outline grounded in existing code paths. The proposals are followed by a dependency graph and a phased implementation roadmap showing how reset outcomes improve at each phase.

## Prerequisites

All prior chapters (1-7) and Sections 01-02 of this chapter. Each proposal references specific gaps identified in the debugging tools (Chapter 6) and workflows (Chapter 7).

---

## Priority Summary

| Priority | Proposal | Effort | Key Benefit |
|----------|----------|--------|-------------|
| **P0** | 1. Automatic Hang Classification | Small (1-2 weeks) | Eliminates manual log reading for ~80% of hangs |
| **P0** | 2. Pre-Reset State Snapshots | Small (1-2 weeks) | Preserves diagnostic evidence that warm reset destroys |
| **P0** | 3. Structured Error Propagation | Medium (3-4 weeks) | Replaces `while(1){}` with push-based error reporting |
| **P1** | 4. Device-Side Heartbeat Monitoring | Medium (3-4 weeks) | Detects hangs in ~300ms instead of 5-30 seconds |
| **P1** | 5. Partial Device Reset | Medium (4-5 weeks) | Recovers individual cores without full chip reset |
| **P1** | 6. Enhanced NOC Debug Infrastructure | Medium (3-4 weeks) | Makes NOC debug dump production-ready |
| **P2** | 7. Firmware Watchdog with Automatic Recovery | Medium (4-5 weeks) | Sub-millisecond hang detection at firmware level |
| **P2** | 8. Static Analysis and Pre-Flight Validation | Large (8-12 weeks) | Catches hang-inducing bugs at compile time |
| **P2** | 9. Deterministic Command Stream Replay | Medium (4-5 weeks) | Exact reproduction of hang-inducing workloads |
| **P2** | 10. Resilient CCL Operations | Large (8-12 weeks) | Fault-tolerant multi-chip collectives |
| **P3** | 11. Workload Checkpoint/Restart | Large (10-16 weeks) | Recovery without losing training/inference progress |
| **P3** | 12. Unified Diagnostic Dashboard | Medium (4-6 weeks) | Single pane of glass for multi-chip debugging |

---

## Proposal 1: Automatic Hang Detection with Root Cause Classification

**Priority: P0** | **Effort: Small (1-2 weeks)** | **Dependencies: None (builds on existing watcher)**

### Current Gap

When a hang occurs, the developer must manually read watcher logs, correlate waypoint codes across cores, inspect NOC sanitization results, and cross-reference with dispatch state to determine the hang category. This manual process takes 15-60 minutes per incident and requires expert knowledge of the hang taxonomy from Chapter 1. Meanwhile, the data needed for automatic classification already exists in structured form: watcher waypoints identify which blocking primitive each core is stuck in (e.g., `CRBW` = CB reserve back wait, `CWFW` = CB wait front wait, `NSW` = NOC semaphore wait, `NWBW` = NOC write barrier wait), NOC sanitization results identify address violations, and dispatch state identifies pipeline stalls.

### Hang Categories Addressed

All categories from Chapters 2-5. This proposal does not prevent hangs but dramatically reduces time-to-diagnosis.

### Impact Estimate

**Time savings:** Converting 30-minute manual triage sessions to 5-second automated classification for the ~80% of hangs that match known patterns. Estimated engineering time savings: 100+ hours per team per quarter for active development teams.

### Implementation Outline

Extend the watcher server's `dump()` method (in `watcher_server.cpp`) to run a classification function after reading all core states:

**Classification rules (ordered by specificity):**

| Pattern | Classification | Confidence |
|---------|---------------|------------|
| Core A: `CRBW` on NCRISC, Core A: `CWFW` on TRISC | CB deadlock (intra-core) | High |
| Core A: `CRBW`, Core B: `CRBW` (mutual producer/consumer) | CB deadlock (inter-core) | High |
| Core stuck at `NSW` with no other core writing to target semaphore | Semaphore hang (missing increment) | High |
| Core stuck at `NRBW` or `NWBW` with no sanitize violation | NOC backpressure/deadlock | Medium |
| Sanitization mailbox populated with return code | NOC address violation | High |
| Assert mailbox populated | Deliberate assert-and-hang | High |
| Prefetch core at `HQW` (waiting for host) | Prefetch stall (host not feeding commands) | High |
| Dispatch core at `PWW`/`WCW` (waiting for workers) | Worker hang (dispatch is a victim) | Medium |
| ERISC core reporting link-down | Ethernet link failure | High |
| Multiple chips with ERISC cores at `NSW` | Multi-chip collective deadlock | Medium |

**Output:** Classification string in watcher log, structured JSON for automated tooling, suggested diagnostic steps and chapter reference, and recommended reset level.

### Effort Breakdown

- Decision tree implementation: 2-3 days
- Integration with `WatcherDeviceReader`: 1-2 days
- Testing with `hang_device` and known hang patterns: 2-3 days
- Documentation and output formatting: 1 day

**Builds on:** Watcher server (`watcher_server.cpp`), `NOCDebugState` (`noc_debugging.hpp`), Inspector data model.

---

## Proposal 2: Automatic State Snapshots Before Reset

**Priority: P0** | **Effort: Small (1-2 weeks)** | **Dependencies: None**

### Current Gap

A warm reset destroys ALL device state: L1 contents, NOC transaction queues, CB pointers, semaphore values, dispatch kernel state. This is exactly the state needed to diagnose the hang that triggered the reset. The `WarmResetCommunication` IPC mechanism sends a `PRE_RESET` notification 2 seconds before the reset executes -- this window is currently used for process cleanup, not for diagnostic capture. An estimated 40-50% of first-time hang investigations fail because the developer resets the device before capturing sufficient diagnostic state.

### Hang Categories Addressed

All categories. Every hang benefits from preserved diagnostic state.

### Impact Estimate

Eliminates the "reproduce the hang with watcher enabled" cycle that often requires 3-5 additional runs (each potentially requiring their own reset).

### Implementation Outline

1. **Register a diagnostic capture callback** with `WarmResetCommunication::Monitor::start_monitoring()` to capture state during the 2-second pre-reset window.

2. **Snapshot contents** (prioritized by diagnostic value within the 2-second window):
   - Watcher mailbox data for all cores (waypoints, sanitize results) -- ~1ms
   - CB read/write pointers for all CBs -- ~10ms
   - Semaphore values at known addresses -- ~10ms
   - Dispatch kernel issue/completion queue pointers -- ~5ms
   - L1 contents for flagged cores (cores with stale waypoints) -- ~100ms per core
   - NOC debug registers -- ~50ms
   - ERISC state: link status, fabric telemetry counters -- ~10ms

3. **Storage:** Write to `generated/pre_reset_snapshot_<timestamp>.bin` with a structured binary format parsable by tt-triage scripts. Add `parse_pre_reset_snapshot.py` to the tt-triage suite.

4. **Configuration:** Enabled by default, controllable via `TT_METAL_PRE_RESET_SNAPSHOT=1`.

### Effort Breakdown

- Snapshot capture function: 2-3 days
- Binary format definition and serialization: 1-2 days
- Parser script: 1-2 days
- Integration with UMD warm reset path: 1 day
- Testing: 2-3 days

**Builds on:** `WarmResetCommunication` IPC system, watcher device reader, Inspector serialization.

---

## Proposal 3: Structured Firmware-to-Host Error Propagation

**Priority: P0** | **Effort: Medium (3-4 weeks)** | **Dependencies: None (synergizes with Proposals 1 and 2)**

### Current Gap

When firmware detects an error (NOC sanitize violation, assertion failure, CB overflow), the response is:
- **Tensix cores:** Enter `while(1){}` and wait for the host to notice (via watcher polling or dispatch timeout)
- **ERISC cores:** Call `erisc_exit()` -- but this only sets `RUN_MSG_DONE`, it does not propagate the *error type*

There is no structured error channel from firmware to host. The completion queue is boolean: either the operation completed or it did not. An estimated 60-70% of all hangs are detected by firmware *before* the actual hardware deadlock occurs -- these are the candidates for error propagation.

### Hang Categories Addressed

All categories where firmware detects the error before the hardware locks up: NOC sanitize violations, CB violations, assertion failures.

### Impact Estimate

If firmware could propagate errors to the host immediately (instead of hanging), the host could abort the workload and restart without a reset. Estimated conversion: 40-50% of current Level 2 resets into Level 0 graceful recoveries.

### Implementation Outline

1. **Define an error mailbox** in L1 at a well-known address:

```
struct firmware_error_t {
    uint32_t error_code;      // Enumerated error type
    uint32_t core_x, core_y;  // Which core detected the error
    uint32_t risc_id;         // Which RISC (BRISC, NCRISC, TRISC0/1/2)
    uint32_t context[4];      // Error-specific context (address, counter, etc.)
};
```

2. **On error detection:** Instead of `while(1){}`, write the error to the mailbox, set `RUN_MSG_DONE` with an error flag, and halt cleanly (extending the erisc pattern to Tensix cores).

3. **Dispatch forwarding:** The dispatch kernel periodically checks the error mailbox for all worker cores and forwards any errors to the host via a reserved completion queue event type (`CQ_COMPLETION_EVENT_ERROR`).

4. **Host side:** The `SystemMemoryManager::completion_queue_wait_front` processes error events and raises structured exceptions with full core/error context.

5. **Backward compatibility:** The error mailbox is at a new L1 address that old firmware does not write to. Old firmware continues to hang; new firmware propagates errors. The host checks the error mailbox only if the firmware version supports it.

### Effort Breakdown

- Error record definition and L1 memory map changes: 2-3 days
- Firmware-side error mailbox writes: 3-4 days
- Dispatch kernel error forwarding: 3-4 days
- Host-side error event processing: 3-4 days
- Testing and edge cases: 4-5 days

**Builds on:** Watcher mailbox protocol, erisc `hang_on_down_link()` pattern, dispatch completion queue.

---

## Proposal 4: Device-Side Heartbeat Monitoring

**Priority: P1** | **Effort: Medium (3-4 weeks)** | **Dependencies: Benefits from Proposal 3 (error propagation) for reporting, but can be implemented independently**

### Current Gap

The current hang detection model is host-initiated: the host must either poll (watcher, at configurable intervals) or hit a timeout (dispatch operation timeout). Neither detects hangs proactively. The fabric infrastructure already has heartbeat monitoring (`FabricTelemetrySettings` includes `HEARTBEAT_TX` and `HEARTBEAT_RX` tracking for erisc cores), but this is limited to Ethernet cores running fabric firmware and is not available for Tensix cores.

### Hang Categories Addressed

All categories. A universal heartbeat detects any failure to make progress, regardless of root cause.

### Impact Estimate

**Detection latency improvement:** From seconds-to-minutes (dispatch timeout) to ~300ms (three missed heartbeat polls at 100ms intervals). Earlier detection preserves more diagnostic state and reduces the window during which cascading failures corrupt state across the cluster. Estimated 10-15% improvement in first-time diagnosis success rate.

### Implementation Outline

1. **Device side:** Each RISC processor writes a monotonically increasing counter to a dedicated L1 mailbox location on each iteration of its main loop. This is a single 4-byte write -- negligible overhead.

2. **Host side:** Integrate into the existing watcher polling loop. If a counter has not advanced between two consecutive reads, the core is flagged as stalled.

3. **Graduated response:**
   - First missed heartbeat: log warning, increase polling frequency
   - Second missed heartbeat: trigger automatic triage (`tt-triage.py`)
   - Third missed heartbeat: trigger Inspector serialization and prepare for recovery

4. **Configuration:** `TT_METAL_HEARTBEAT_ENABLED=1`, `TT_METAL_HEARTBEAT_POLL_MS=100`.

### Effort Breakdown

- Firmware heartbeat instrumentation: 3-4 days
- Host-side polling and stall detection: 3-4 days
- Integration with watcher and classification: 2-3 days
- Testing across architectures: 3-4 days

**Builds on:** Watcher server polling infrastructure, fabric telemetry heartbeat model.

---

## Proposal 5: Partial Device Reset (Per-Core Reset from Host API)

**Priority: P1** | **Effort: Medium (4-5 weeks)** | **Dependencies: Proposal 4 (heartbeat) for identifying which core to reset**

### Current Gap

As documented in Section 01, Level 1 (per-core soft reset) exists at the hardware register level (`RISCV_DEBUG_REG_SOFT_RESET_0`) and the UMD API level (`Cluster::assert_risc_reset_at_core`), but lacks the software infrastructure for safe use: no dependency tracking, no state restoration, no NOC transaction cleanup.

### Hang Categories Addressed

Chapter 2 kernel-level hangs (CB deadlocks, semaphore timeouts) where a single core is identified as the root cause. Estimated 30-40% of all hangs.

### Impact Estimate

Converting Level 2 warm resets (2-20 seconds, resets all cores) into Level 1 per-core resets (microseconds, resets one core) for the ~35% of hangs involving isolated core failures.

### Implementation Outline

1. **Runtime dependency graph:** Maintain a per-program map of which cores communicate with which other cores (via CB, semaphore, or NOC). This information is partially available in the `Program` object's kernel configurations.

2. **Safe reset determination:** Given a hung core C:
   - If no other core is waiting on C: safe to reset C alone
   - If cores {D, E} are waiting on C: must reset {C, D, E} together
   - If the dependency set includes dispatch cores: fall back to Level 2

3. **Reset sequence:**
   a. Capture the core's state to host memory (leveraging Proposal 2's snapshot infrastructure)
   b. Assert soft reset on the target core(s) via `RISCV_DEBUG_REG_SOFT_RESET_0`
   c. Wait for NOC drain: poll outstanding NOC transaction counters until they reach zero, or timeout
   d. Clear L1 state for the reset core(s)
   e. Reload firmware to the reset core
   f. Deassert soft reset with staggered start

4. **NOC limitation:** This proposal does not solve NOC-level deadlocks. The automatic hang classifier (Proposal 1) can distinguish NOC-level deadlocks (which need Level 2) from software spin-loop hangs (which are safe for Level 1).

5. **API:** Expose `Device::reset_cores(std::vector<CoreCoord> cores)`.

### Effort Breakdown

- API definition and host-side implementation: 4-5 days
- UMD register access for soft reset: 2-3 days
- NOC drain and timeout handling: 3-4 days
- Single-core firmware reload: 3-4 days
- Dispatch pipeline recovery: 3-4 days
- Testing: 5-7 days

**Builds on:** `Cluster::assert_risc_reset_at_core` / `deassert_risc_reset_at_core`, `RISCV_DEBUG_REG_SOFT_RESET_0`, Program kernel configuration metadata.

---

## Proposal 6: Enhanced NOC Debug Infrastructure

**Priority: P1** | **Effort: Medium (3-4 weeks)** | **Dependencies: None**

### Current Gap

The `TT_METAL_NOC_DEBUG_DUMP` feature (`noc_debugging.hpp`) is a powerful missing-barrier detector, but has limitations: significant overhead (not production-ready), post-mortem only (not real-time), no NOC utilization metrics, and host-side-only tracking that cannot monitor remote (non-MMIO) devices.

### Hang Categories Addressed

Chapter 2, all NOC-related hangs -- the second most common category after CB deadlocks and the most difficult to diagnose.

### Impact Estimate

Stabilizing NOC debug dump as a production-ready tool would allow it to run in CI for all NOC-heavy tests. Estimated to catch an additional 15-20% of NOC-related hangs before deployment.

### Implementation Outline

Three sub-proposals:

**6A: Stabilize NOC debug dump for production-CI use (5-7 days).** Fix known false positives in `NOCDebugState` tracking for non-standard NOC access patterns. Add CI integration.

**6B: NOC transaction replay log (7-10 days).** Add an optional, bounded-size circular log of recent NOC transactions (source core, target, type, size, timestamp) stored in L1 via the watcher ring buffer infrastructure. On hang detection, the host reads the buffer to reconstruct the deadlock cycle.

**6C: Real-time NOC utilization metrics (4-5 days).** Add per-NOC (NOC0/NOC1) transaction counters to the watcher polling loop: transactions issued, completed, bytes transferred. High utilization combined with stale waypoints is a strong signal of congestion approaching a deadlock.

**Builds on:** `NOCDebugState` (`noc_debugging.hpp`), watcher ring buffer, profiler NOC event collection.

---

## Proposal 7: Firmware Watchdog with Automatic Recovery

**Priority: P2** | **Effort: Medium (4-5 weeks)** | **Dependencies: Proposal 3 (error propagation) for reporting; Proposal 5 (partial reset) for recovery**

### Current Gap

The `RISCV_DEBUG_REG_WATCHDOG_TIMER` register exists in hardware but is not used by the runtime. All hang detection is host-initiated (watcher polling or dispatch timeout), meaning a hung core sits idle for seconds to minutes before anyone notices. The M3-level watchdog provides last-resort reset, but it resets the entire chip.

### Hang Categories Addressed

All categories where a kernel exceeds its expected execution time. Most valuable for Chapter 2 kernel deadlocks and Chapter 5 CCL operations where one stalled rank blocks all others.

### Impact Estimate

Combined with Proposal 3 (error propagation) and Proposal 5 (partial reset), a firmware watchdog could trigger a controlled exit, propagate the timeout error to the host, and enable a per-core reset -- all without a Level 2 warm reset. Estimated to convert 20-25% of Level 2 resets into Level 0/1 recoveries.

### Implementation Outline

1. **Configuration:** During kernel launch, the host writes a timeout value (in clock cycles) to the watchdog timer register. The timeout is derived from expected execution time plus a safety margin (e.g., 10x).

2. **Watchdog kick:** The firmware main loop and all blocking primitives in `dataflow_api.h` periodically reset the watchdog counter. When the watchdog is disabled (compile-time flag), this is zero cost.

3. **Expiration handler:** On timeout, the handler adopts the erisc exit pattern for Tensix cores: save error state to the error mailbox (Proposal 3), set `RUN_MSG_DONE`, halt cleanly.

4. **Key challenge:** Kicking the watchdog from within legitimate spin-waits requires modifying every blocking primitive. The modification must be very low cost (one register write per N loop iterations).

### Effort Breakdown

- Watchdog configuration and firmware instrumentation: 5-7 days
- Expiration handler: 4-5 days
- Blocking primitive modifications: 3-4 days
- Host-side configuration: 2-3 days
- Testing and edge case handling: 5-7 days

**Builds on:** `RISCV_DEBUG_REG_WATCHDOG_TIMER` hardware capability, erisc exit pattern, Proposal 3.

---

## Proposal 8: Static Analysis and Pre-Flight Validation

**Priority: P2** | **Effort: Large (8-12 weeks)** | **Dependencies: None (independent tooling)**

### Current Gap

Many hangs are caused by configuration errors that are deterministic and could be detected before the kernel reaches the device: CB size not divisible by ntiles, missing CB consumer, missing NOC barrier, NOC write to invalid address, semaphore address conflicts. Currently, these are caught (if at all) at runtime by watcher.

### Hang Categories Addressed

Chapter 2 (CB deadlocks, NOC violations), Chapter 3 (memory address errors). Estimated 25-30% of all hangs are caused by static misconfigurations detectable before launch.

### Impact Estimate

Catch 100% of static misconfigurations before they reach the device. This directly eliminates ~25-30% of resets with zero runtime overhead.

### Implementation Outline

Three sub-proposals:

**8A: CB configuration validator (Small, 1-2 weeks).** At `CreateCircularBuffer` time, verify that every CB has a producer and consumer, page size is consistent across kernels, total size is a multiple of page size, and ntiles arguments are compatible.

**8B: Tile size consistency checker (Small, 1-2 weeks).** Cross-check tile sizes used by reader, compute, and writer kernels. Flag inconsistencies at compile time.

**8C: NOC barrier linter (Large, 6-10 weeks).** Statically analyze kernel source or compiled binaries to verify that every `noc_async_write` has a corresponding barrier before dependent operations. This is fundamentally a data-flow analysis problem requiring either AST walking or RISC-V disassembly.

**Implementation vehicle:** Checks 8A and 8B can be added to the `Program` validation phase in `tt_metal/impl/program/program.cpp`, running at program creation time (host side, before any device interaction).

**Builds on:** `Program` compilation and validation infrastructure, `CreateCircularBuffer` API.

---

## Proposal 9: Deterministic Command Stream Replay

**Priority: P2** | **Effort: Medium (4-5 weeks)** | **Dependencies: Builds on existing LightMetal infrastructure**

### Current Gap

LightMetal capture/replay (`lightmetal_capture.cpp`, `lightmetal_replay_impl.cpp`) records and replays Metal API call sequences. However, it operates at the API level, not at the byte level of the command stream written to system memory. Non-deterministic behavior within the runtime (timing, thread scheduling, memory allocation) may differ between capture and replay. An estimated 30-40% of hangs reported in CI are non-reproducible on developer machines due to timing differences.

### Hang Categories Addressed

Chapter 4 (dispatch and command queue hangs), Chapter 2 (timing-dependent kernel deadlocks).

### Impact Estimate

Each non-reproducible hang typically requires 5-20 reproduction attempts (each requiring a reset). Byte-level command stream capture would make these 100% reproducible.

### Implementation Outline

1. **Capture:** Instrument `SystemMemoryManager` to log the exact bytes written to the system memory hugepage, with timestamps and command boundaries.
2. **Replay:** Feed the captured byte stream directly into the system memory region, bypassing the Metal API.
3. **Differential mode:** Compare API-level replay output against recorded byte stream to identify non-determinism.
4. **Address remapping:** The recorded stream contains absolute DRAM addresses that may differ after reset/reallocation; the player must apply remapping.

### Effort Breakdown

- Byte-stream recorder: 3-4 days
- Byte-stream player: 4-5 days
- Address remapping: 3-4 days
- Differential comparison mode: 2-3 days
- Testing: 4-5 days

**Builds on:** LightMetal capture/replay infrastructure, `SystemMemoryManager`.

---

## Proposal 10: Resilient CCL Operations

**Priority: P2** | **Effort: Large (8-12 weeks)** | **Dependencies: Proposal 4 (heartbeat) for detecting failed participants, Proposal 5 (partial reset) for recovering failed chips**

### Current Gap

Current CCL operations are all-or-nothing: if any participating rank fails, all other ranks wait indefinitely. The erisc graceful exit pattern handles the link-down case at the firmware level, but the workload-level impact is not handled. Multi-chip hangs account for approximately 10% of all hangs but represent a disproportionate share of reset time (each requires resetting 2-32 devices).

### Hang Categories Addressed

Chapter 5, all CCL-related hangs.

### Impact Estimate

Resilient CCL would convert approximately 60-70% of multi-chip hangs from "all devices reset" to "one device skipped/rerouted." For long-running inference on Galaxy, this avoids resetting all 32+ devices for a single link failure.

### Implementation Outline

**10A: Timeout-based fallback (3-4 weeks).** Replace infinite `noc_semaphore_wait` in CCL operations with timeout-bounded waits. On timeout, the rank reports an error and exits cleanly.

**10B: Automatic rerouting on fabric link failure (3-4 weeks).** When an Ethernet link fails, dynamically update routing tables via the `ControlPlane` and `FabricSwitchManager` to exclude the failed link. The `FabricReliabilityMode::DYNAMIC_RECONFIGURATION_SETUP_MODE` (value 2) in `fabric_types.hpp` is the intended home for this functionality.

**10C: CCL operation replay (2-3 weeks).** After partial failure and recovery, automatically replay the failed collective with remaining healthy devices.

**Builds on:** Fabric `FabricReliabilityMode` enum, erisc graceful exit pattern, `ControlPlane` routing table infrastructure.

---

## Proposal 11: Workload-Level Resilience (Checkpoint/Restart)

**Priority: P3** | **Effort: Large (10-16 weeks)** | **Dependencies: Proposal 5 (partial reset), Proposal 10 (resilient CCL)**

### Current Gap

When a hang occurs during a long-running workload, the entire workload must be restarted from the beginning after the reset. For Galaxy-scale training where hangs occur every 2-4 hours on average, this represents massive lost compute.

### Hang Categories Addressed

All categories. This does not prevent hangs but dramatically reduces their cost.

### Impact Estimate

With periodic checkpointing (e.g., every 10 minutes), a hang during a 24-hour training run loses at most 10 minutes of work instead of hours. Estimated 90%+ reduction in lost compute for long-running workloads.

### Implementation Outline

1. **Automatic periodic checkpoints:** Snapshot model state (weights, optimizer state, data loader position) to host memory or disk at configurable intervals.
2. **Automatic retry:** After warm reset, reload the last checkpoint and resume.
3. **Per-op retry:** For individual operations that timeout, attempt retry before escalating to full workload restart (requires idempotent operations).
4. **Multi-chip coordination:** Checkpointing must be synchronized across all ranks via a `barrier_with_checkpoint` primitive.

### Effort Breakdown

- Checkpoint infrastructure: 4-5 weeks
- Incremental checkpoint tracking: 3-4 weeks
- Training loop integration: 2-3 weeks
- Inference serving integration: 2-3 weeks
- Testing: 2-3 weeks

**Builds on:** Model checkpointing facilities, dispatch retry infrastructure, Proposal 10.

---

## Proposal 12: Unified Diagnostic Dashboard

**Priority: P3** | **Effort: Medium (4-6 weeks)** | **Dependencies: Proposals 1, 2, 3, 4, 6 (all are data sources)**

### Current Gap

Diagnostic data is scattered across watcher log files, tt-triage terminal output, Inspector RPC responses, Tracy profiler traces, NOC debug dumps, and per-core register reads. Correlating data across these sources requires switching between multiple tools. For multi-chip debugging, the complexity multiplies by the number of chips.

### Implementation Outline

1. **Chip topology view:** Visual mesh/cluster with per-core status indicators (healthy/stuck/errored). Click a core to see watcher state, heartbeat status, and last waypoint.
2. **Timeline view:** Aggregated event timeline across all devices showing operation starts/completes, hang detections, error events, and resets.
3. **Hang investigation view:** Automatic classification (Proposal 1), pre-reset snapshot data (Proposal 2), and recommended remediation.
4. **CCL operation view:** Collective progress across all devices, highlighting stragglers.
5. **Historical analysis:** Store diagnostic data from multiple runs to identify patterns.

**Implementation:** Python backend consuming Inspector RPC, watcher logs, and tt-triage output. Lightweight web UI with WebSocket updates. SQLite for historical data.

### Effort Breakdown

- Data aggregation backend: 2-3 weeks
- Web UI: 2-3 weeks
- Historical storage and analysis: 1-2 weeks

---

## Dependency Graph

```
                                    Independent
                                        |
            +---------------------------+---------------------------+
            |                           |                           |
    [P0] Proposal 1             [P0] Proposal 2             [P0] Proposal 3
    Auto Classification         Pre-Reset Snapshots          Error Propagation
            |                           |                           |
            |           +---------------+                           |
            |           |                                           |
            v           v                                           v
    [P1] Proposal 4                                     [P1] Proposal 6
    Heartbeat Monitoring                               Enhanced NOC Debug
            |
            +---------------------------+
            |                           |
            v                           v
    [P1] Proposal 5             [P2] Proposal 7
    Partial Device Reset        Firmware Watchdog
            |                           |
            |                           |
            v                           v
    [P2] Proposal 10           [P2] Proposal 9
    Resilient CCL              Deterministic Replay
            |
            v
    [P3] Proposal 11
    Checkpoint/Restart


    Independent (no inter-proposal dependencies):
    [P2] Proposal 8  -- Static Analysis
    [P3] Proposal 12 -- Dashboard (consumes data from 1,2,3,4,6)
```

---

## Implementation Roadmap

**Phase 1 (Immediate, 1-2 months):** Proposals 1, 2, 3 (P0). These three proposals are independent and can be developed in parallel by separate engineers. They collectively transform the diagnostic experience: automatic hang classification replaces manual log reading, pre-reset snapshots preserve evidence, and structured error propagation replaces `while(1){}` detection-by-absence.

**Phase 2 (Next quarter, 2-4 months):** Proposals 4, 5, 6 (P1). These build on Phase 1 and collectively enable faster hang detection (heartbeats: ~300ms vs 30s), finer-grained recovery (per-core reset vs full chip reset), and better NOC diagnostics. Proposal 5 is the most impactful for reset reduction because it enables Level 1 resets from the host.

**Phase 3 (6-12 months):** Proposals 7, 8A/8B, 9, 10A (P2). These are more complex and benefit from Phase 1-2 infrastructure. The firmware watchdog (7) provides ultimate low-latency detection. Static analysis (8A/8B) prevents bugs at compile time. Deterministic replay (9) enables exact hang reproduction. Resilient CCL (10A) prevents single-chip failures from cascading.

**Phase 4 (12+ months):** Proposals 8C, 10B/10C, 11, 12 (P3). These are the largest investments and the most dependent on prior work. The NOC barrier linter (8C) is a research-level static analysis problem. Automatic fabric rerouting (10B) requires deep fabric firmware changes. Workload checkpoint/restart (11) is a full application-framework feature. The dashboard (12) is most useful after all other data sources are stable.

---

## Expected Impact on Reset Frequency

| Scenario | Current Outcome | After Phase 1 | After Phase 2 | After Phase 3 |
|----------|----------------|---------------|---------------|---------------|
| CB deadlock | Full chip reset | Same (but diagnosed in seconds, not minutes) | Level 1 (per-core reset) | Level 0 (prevented by static analysis) |
| NOC address violation | Full chip reset | Same (but evidence preserved by snapshot) | Full chip reset (NOC state still needs clearing) | Prevented by firmware watchdog + error propagation |
| Semaphore hang | Full chip reset | Same (but auto-classified) | Level 1 (per-core reset if NOC is clean) | Prevented by static analysis |
| Multi-chip collective deadlock | Full coordinated reset | Same (but auto-classified, evidence preserved) | Same (but detected in ~300ms, not 30s) | Timeout fallback (Proposal 10A), no reset needed |
| Dispatch stall | Full chip reset | Same (but auto-classified) | Level 1 (partial reset of dispatch core) | Prevented by better error propagation |
| Intermittent timing-dependent hang | Unpredictable | Evidence preserved by snapshot | Better diagnosis via NOC replay log | Deterministic reproduction via command stream replay |

The cumulative effect: Phase 1 makes hangs diagnosable without expert knowledge. Phase 2 reduces the blast radius of most hangs from full-chip to per-core. Phase 3 prevents many hang categories from occurring at all. Phase 4 provides full workload resilience.

---

## Summary: Proposal Impact Matrix

| # | Proposal | Prevents Hangs? | Reduces Reset Level? | Reduces Diagnosis Time? | Estimated Reset Reduction |
|---|---|---|---|---|---|
| 1 | Auto hang classification | No | No | Yes (30min -> 5s) | Indirect (faster recovery) |
| 2 | Pre-reset state snapshots | No | No | Yes (eliminates repro cycles) | Indirect (40-50% fewer repro attempts) |
| 3 | Error propagation | No | Yes (L2 -> L0) | Yes (immediate error info) | 40-50% of L2 -> L0 |
| 4 | Device-side heartbeat | No | No | Yes (seconds -> ~300ms) | Indirect (earlier detection) |
| 5 | Partial reset | No | Yes (L2 -> L1) | No | 30-35% of L2 -> L1 |
| 6 | Enhanced NOC debug | Partially | No | Yes (better diagnostics) | 15-20% in CI |
| 7 | Firmware watchdog | No | Yes (L2 -> L0/L1) | Yes (us-level detection) | 20-25% of L2 -> L0/L1 |
| 8 | Static analysis | Yes | N/A | N/A (prevents entirely) | 25-30% eliminated |
| 9 | Deterministic replay | No | No | Yes (100% repro rate) | Indirect (5-20x fewer repro attempts) |
| 10 | Resilient CCL | Partially | Yes (coordinated L2 -> single L2) | No | 60-70% of multi-chip hangs |
| 11 | Checkpoint/restart | No | No | No | Reduces cost, not count |
| 12 | Unified dashboard | No | No | Yes (single-pane view) | Indirect |

**Priority recommendation:** Proposals 3 (error propagation) and 8A/8B (static analysis) offer the highest direct reset reduction with moderate implementation effort. Proposal 3 requires firmware changes but builds on the well-proven erisc exit pattern. Proposal 8A/8B is purely host-side and can be implemented incrementally.

Proposals 1 (auto classification) and 2 (pre-reset snapshots) offer the highest indirect impact by dramatically reducing diagnosis time, which reduces the total number of reset cycles needed to resolve each hang.

For multi-chip deployments, Proposal 10 (resilient CCL) is the highest priority because multi-chip hangs have the highest per-incident cost (resetting an entire Galaxy takes 30+ seconds and disrupts all 32+ devices).

---

**Previous:** [`02_reducing_reset_frequency_and_resilience.md`](./02_reducing_reset_frequency_and_resilience.md)
**Chapter index:** [`index.md`](./index.md)
