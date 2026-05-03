# Chapter 4: Dispatch, Command Queue, and Host-Device Interaction Hangs

The dispatch pipeline is the host-device communication backbone: every program launch, buffer transfer, and synchronization event flows through this infrastructure. When the dispatch pipeline hangs, the observable symptom is almost always "the device stopped responding" -- but the root cause can lie in any of a dozen different subsystems, from the host-side `SystemMemoryManager` running out of fetch queue entries, to the on-device prefetch kernel spinning for data the host never wrote, to the dispatch kernel waiting for workers that will never signal completion.

This chapter systematically catalogs every known hang mechanism in the fast dispatch pipeline, host synchronization layer, and trace replay infrastructure. Unlike the kernel-level deadlocks in Chapter 2 or the memory corruption scenarios in Chapter 3, dispatch hangs often involve **cross-boundary coordination failures** -- the root cause is on the host but manifests on the device, or vice versa. The 5-part diagnostic format (Symptom / Root Cause / Diagnosis Steps / Fix / Prevention) introduced in [Chapter 1](../ch01_anatomy_of_a_hang/01_what_is_a_hang.md) applies to every scenario.

## Prerequisites

- **Chapter 1, [`01_what_is_a_hang.md`](../ch01_anatomy_of_a_hang/01_what_is_a_hang.md)**: The RISC-V spin-loop hang model, the 5-part diagnostic format, and the distinction between deliberate and incidental hangs.
- **Chapter 1, [`02_blocking_primitives_taxonomy.md`](../ch01_anatomy_of_a_hang/02_blocking_primitives_taxonomy.md)**: Familiarity with `noc_semaphore_wait`, `noc_async_write_barrier`, and the waypoint (`WAYPOINT`) tracing mechanism. This chapter introduces dispatch-specific waypoints (`QRBW`/`QRBD`, `PWW`/`PWD`, `WCW`/`WCD`, `HQW`, `DAPW`/`DAPD`, `CBRW`/`CBRD`).
- **Chapter 1, [`03_hang_taxonomy.md`](../ch01_anatomy_of_a_hang/03_hang_taxonomy.md)**: The six-category classification system, particularly category 4 (dispatch/command queue stalls) and category 6 (host-device interaction hangs).
- **Chapter 2, [`04_noc_barrier_and_semaphore_hangs.md`](../ch02_kernel_and_noc_hangs/04_noc_barrier_and_semaphore_hangs.md)**: NOC semaphore increment/wait mechanics. Dispatch-to-worker go signals and worker completion signals use the same semaphore infrastructure. The mcast path reservation workaround is used inside `process_write_packed` in the dispatch kernel.
- **Chapter 3, [`01_l1_memory_corruption_and_overflow.md`](../ch03_memory_related_hangs/01_l1_memory_corruption_and_overflow.md)**: L1 memory layout awareness. Dispatch kernels write to mailbox regions and config buffer areas; corruption of these regions can cause secondary dispatch hangs.

## How to Use This Chapter

- **If the host throws `TIMEOUT: device timeout in fetch queue wait, potential hang detected`**: start with [Section 01, Scenario 4.1.4](./01_dispatch_architecture_and_hang_points.md#414-fetch-queue-full-host-blocked-in-fetch_queue_reserve_back) (fetch queue backpressure) and [Section 02, Scenario 4.2.2](./02_host_synchronization_and_timeout_detection.md#422-fetch-queue-wait-timeout) (fetch queue wait timeout).
- **If the host throws `TIMEOUT: device timeout, potential hang detected, the device is unrecoverable`**: start with [Section 02, Scenario 4.2.1](./02_host_synchronization_and_timeout_detection.md#421-completion-queue-wait-timeout) (completion queue timeout).
- **If the watcher shows dispatch_d stuck at waypoint `PWW`/`PWD` or `WCW`/`WCD`**: start with [Section 01, Scenario 4.1.6](./01_dispatch_architecture_and_hang_points.md#416-dispatch-kernel-waiting-for-workers-process_wait--go-signal) (dispatch waiting for workers).
- **If the watcher shows prefetch stuck at waypoint `HQW`**: start with [Section 01, Scenario 4.1.1](./01_dispatch_architecture_and_hang_points.md#411-prefetch-kernel-stall-on-host-empty-fetch-queue) (prefetch waiting for host).
- **If the watcher shows dispatch at `CBRW`**: start with [Section 01, Scenario 4.1.12](./01_dispatch_architecture_and_hang_points.md#4112-dispatch-cb-page-release-stall-cbrw) (CB page release stall).
- **If the watcher shows dispatch_s stuck at waypoint `DCW`**: start with [Section 01, Scenario 4.1.7](./01_dispatch_architecture_and_hang_points.md#417-dispatch-subordinate-waiting-for-dispatch-master-notification) (dispatch subordinate sync).
- **If the watcher shows `!CMD` on any dispatch core**: start with [Section 01, Scenario 4.1.11](./01_dispatch_architecture_and_hang_points.md#4111-invalid-dispatch-command-cmd-waypoint) (invalid command).
- **If a hang occurs only during trace replay**: start with [Section 03](./03_trace_replay_and_lightmetal.md).
- **If a hang occurs after switching sub-device configurations**: start with [Section 02, Scenario 4.2.7](./02_host_synchronization_and_timeout_detection.md#427-sub-device-manager-state-inconsistencies) (sub-device state).
- **If the hang is reproducible only intermittently** and you need deterministic reproduction: jump to the LightMetal section in [Section 03](./03_trace_replay_and_lightmetal.md#lightmetal-deterministic-capture-and-replay).

## The Fast Dispatch Pipeline at a Glance

Before diving into individual hang scenarios, the following diagram summarizes the data flow. Each arrow is a potential hang point:

```
 HOST                           DEVICE
 ====                           ======

 +-------------------+
 | Application Code  |
 | (EnqueueProgram,  |
 |  EnqueueReadBuf,  |
 |  Finish, etc.)    |
 +--------+----------+
          |
          v
 +-------------------+     hugepage (PCIe)     +-------------------+
 | SystemMemoryMgr   | ----------------------> | Prefetch Kernel   |
 | - issue_queue      |  fetch_queue entries    | (cq_prefetch.cpp) |
 | - completion_queue  |  command data          | - reads from host |
 | - fetch_queue       |                        | - STALL mechanism |
 +--------+----------+                         +--------+----------+
          |                                              |
          |  completion_queue_wait_front                  | dispatch CB (ring buffer)
          |  (host polls for device writes)               | semaphore handshake
          |                                              v
          |                                     +-------------------+
          |                                     | Dispatch Kernel   |
          |      <--completion queue writes---  | (cq_dispatch.cpp) |
          |                                     | - writes to workers|
          |                                     | - writes to host  |
          |                                     +--------+----------+
          |                                              |
          |                                     go_signal| worker writes
          |                                     (mcast)  | (config, binaries)
          |                                              v
          |                                     +-------------------+
          |                                     | Worker Cores      |
          |                                     | (BRISC,NCRISC,    |
          |                                     |  TRISC0-2)        |
          |                                     +--------+----------+
          |                                              |
          |                                     stream_reg completion signal
          |                                              |
          |                                              v
          |                                     +-------------------+
          |                                     | Dispatch_S Kernel |
          |                                     | (subordinate)     |
          |                                     | - sends go signals|
          |                                     |   asynchronously  |
          |                                     +-------------------+
```

Every arrow in this diagram represents a synchronization point that can hang. The following sections analyze each one systematically.

## Chapter Contents

| # | File | Focus | Key Waypoints / Error Messages |
|---|------|-------|------|
| 1 | [`01_dispatch_architecture_and_hang_points.md`](./01_dispatch_architecture_and_hang_points.md) | Fast dispatch pipeline (prefetch, dispatch, dispatch_s), `SystemMemoryManager`, command types with hang risk ratings, prefetch kernel hang points, dispatch kernel hang points, dispatch subordinate, worker config buffer exhaustion, CB page release stalls, relay_mux topology (14 scenarios) | `HQW`, `UAPW`, `CNSW`, `QRBW`/`QRBD`, `PWW`/`PWD`, `WCW`/`WCD`, `DCW`, `DAPW`/`DAPD`, `CBRW`/`CBRD`, `!CMD` |
| 2 | [`02_host_synchronization_and_timeout_detection.md`](./02_host_synchronization_and_timeout_detection.md) | Synchronize/Finish semantics, completion queue mechanism, timeout detection (`TT_METAL_OPERATION_TIMEOUT_SECONDS`), `loop_and_wait_with_timeout`, `on_dispatch_timeout_detected` with Inspector integration, `DeviceManager::close_devices` with `skip_synchronize`, async ordering violations, multi-queue dependencies, sub-device manager state, event ID wrap-around, host polling starvation (10 scenarios) | `TIMEOUT: device timeout`, `on_dispatch_timeout_detected`, Inspector |
| 3 | [`03_trace_replay_and_lightmetal.md`](./03_trace_replay_and_lightmetal.md) | Trace capture/replay (`TraceBuffer`, `TraceDescriptor`), `RUN_MSG_REPLAY_TRACE`, stale state assumptions, program cache interactions, config buffer sync drift, repeated replay drift, LightMetal capture/replay as reproduction tool, key invariants checklists (10 scenarios) | `RUN_MSG_REPLAY_TRACE`, `CQ_PREFETCH_CMD_EXEC_BUF`, `StallState::STALLED` |

**Covers research questions:** Q3 (all host-device interaction hang causes), Q1 (dispatch command queue stalls).

---

**Next:** [`01_dispatch_architecture_and_hang_points.md`](./01_dispatch_architecture_and_hang_points.md)
