# 4.1 Dispatch Architecture and Hang Points

[Previous: Chapter Index](./index.md) | [Next: Host Synchronization and Timeout Detection](./02_host_synchronization_and_timeout_detection.md)

---

This section documents the fast dispatch pipeline -- the path from host command submission to worker core execution -- and every point in that pipeline where a hang can originate. The fast dispatch architecture is a multi-stage producer-consumer chain: the **host** writes commands to hugepage-backed system memory, the **prefetch kernel** reads those commands and relays them to the **dispatch kernel**, the dispatch kernel writes configuration data and go signals to **worker cores**, and the optional **dispatch subordinate** (`dispatch_s`) asynchronously sends go signals to overlap dispatch with worker execution.

Each stage communicates via semaphores and ring buffers. A stall at any stage propagates upstream, and if no forward progress can be made, the pipeline hangs.

Reference files: `tt_metal/impl/dispatch/kernels/cq_prefetch.cpp`, `tt_metal/impl/dispatch/kernels/cq_dispatch.cpp`, `tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate.cpp`, `tt_metal/impl/dispatch/kernels/cq_commands.hpp`, `tt_metal/impl/dispatch/hardware_command_queue.hpp`, `tt_metal/impl/dispatch/system_memory_manager.cpp`, `tt_metal/impl/dispatch/worker_config_buffer.hpp`

## 4.1.0 Fast Dispatch Pipeline Overview

The fast dispatch pipeline consists of these stages:

```
Host (CPU)
  |  writes commands to hugepage (issue queue)
  |  writes fetch queue entries (prefetch_q) via TLB
  v
Prefetch Kernel (cq_prefetch.cpp)
  |  reads commands from PCIe via NOC async reads
  |  relays inline data / reads DRAM out-of-band data
  |  writes pages to dispatch buffer (downstream_cb)
  v
Dispatch Kernel (cq_dispatch.cpp)    --->  Dispatch Subordinate (cq_dispatch_subordinate.cpp)
  |  processes commands from dispatch buffer           |  receives go signal / wait commands
  |  writes config data to worker L1                   |  asynchronously sends go signals to workers
  |  writes completion signals to host                 |  updates worker completion counts
  v
Worker Cores
  |  receive launch messages, run kernels
  |  signal completion via stream register increments
```

### The H and D Variants

The prefetch and dispatch kernels each have "H" (host-side) and "D" (device-side) variants, selected by compile-time flags `IS_H_VARIANT` and `IS_D_VARIANT`:

- **`prefetch_hd`**: Combined host+DRAM prefetcher (common for MMIO-connected devices). Fetches from host via PCIe and from device DRAM.
- **`prefetch_h`**: Host-only prefetcher for the MMIO side of a remote (tunneled) device.
- **`prefetch_d`**: DRAM-only prefetcher on the remote device, receiving forwarded commands from prefetch_h.
- **`dispatch_hd`**: Combined host+device dispatcher. Handles both host-bound writes (completion queue) and device-bound writes (worker cores).
- **`dispatch_h`**: Host-only dispatcher that handles completion queue writes and relays device-bound commands.
- **`dispatch_d`**: Device-only dispatcher that receives relayed commands and writes to worker cores.

For remote (non-MMIO) devices, the pipeline becomes: `prefetch_h` --> `prefetch_d` --> `dispatch_d` --> `dispatch_h`, with the Ethernet tunnel connecting the h and d sides.

### Key Data Structures

| Structure | Location | Purpose |
|-----------|----------|---------|
| **Issue Queue** | Hugepage (host DRAM) | Ring buffer where host writes command data |
| **Fetch Queue** (`prefetch_q`) | Prefetcher L1 | Array of 16-bit entries; each encodes a fetch size and optional stall flag |
| **CmdDat Queue** (`cmddat_q`) | Prefetcher L1 | Ring buffer holding fetched commands from PCIe |
| **Dispatch Buffer** (`dispatch_cb`) | Dispatch kernel L1 | Ring buffer (pages, blocks) where prefetcher writes command data for dispatch |
| **Completion Queue** | Hugepage (host DRAM) | Ring buffer where dispatch writes completion signals for host polling |
| **WorkerConfigBuffer** | Host-side state | Ring buffer manager tracking where kernel config/binaries are placed in worker L1 |

### Command Types

Commands are defined in `cq_commands.hpp`. The prefetch kernel processes `CQPrefetchCmdId` commands; the dispatch kernel processes `CQDispatchCmdId` commands.

**Prefetch commands** (host to prefetcher):

| Command | ID | Purpose | Hang Risk |
|---------|----|---------|-----------|
| `CQ_PREFETCH_CMD_RELAY_INLINE` | 5 | Relay inline data to dispatch | Low -- bounded by dispatch buffer space |
| `CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH` | 6 | Relay inline header without flushing page | Medium -- stateful, incorrect use corrupts |
| `CQ_PREFETCH_CMD_RELAY_PAGED` | 3 | Read paged data from DRAM and relay | Medium -- DRAM read latency + dispatch backpressure |
| `CQ_PREFETCH_CMD_EXEC_BUF` | 7 | Execute commands from a DRAM buffer (trace replay) | High -- stalls prefetcher, hijacks cmddat_q |
| `CQ_PREFETCH_CMD_STALL` | 9 | Drain pipe through dispatcher | High -- blocks until dispatch signals back |
| `CQ_PREFETCH_CMD_TERMINATE` | 11 | Shut down prefetcher | Low |

**Dispatch commands** (prefetcher to dispatch):

| Command | ID | Purpose | Hang Risk |
|---------|----|---------|-----------|
| `CQ_DISPATCH_CMD_WRITE_LINEAR` | 1 | Unicast/multicast write to worker L1 | Medium -- NOC backpressure |
| `CQ_DISPATCH_CMD_WRITE_PACKED` | 5 | Packed writes to multiple cores | High -- mcast path reservation bug workaround |
| `CQ_DISPATCH_CMD_WRITE_PACKED_LARGE` | 6 | Large packed writes (kernel binaries) | High -- linked NOC transactions |
| `CQ_DISPATCH_CMD_WAIT` | 7 | Wait for workers to complete | High -- spins on stream register |
| `CQ_DISPATCH_CMD_SEND_GO_SIGNAL` | 14 | Multicast/unicast go signal to workers | High -- waits for worker completion first |
| `CQ_DISPATCH_NOTIFY_SUBORDINATE_GO_SIGNAL` | 15 | Notify dispatch_s it is safe to send go signal | Medium |
| `CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST` | 3 | Write completion data back to host | Medium -- completion queue backpressure |
| `CQ_DISPATCH_CMD_SET_WRITE_OFFSET` | 12 | Set relocation offset for subsequent writes | Low |
| `CQ_DISPATCH_CMD_TERMINATE` | 13 | Shut down dispatcher | Low |

---

## 4.1.1 Prefetch Kernel Stall on Host (Empty Fetch Queue)

**Symptom:** The prefetch kernel is stuck at waypoint `HQW`. No commands are being processed. The host may or may not have written commands to the issue queue.

**Root Cause:** The prefetch kernel's `fetch_q_get_cmds` function has exhausted all in-flight reads and the committed command fence equals the command pointer (no commands available). It enters a spin loop polling the fetch queue for new work from the host:

```cpp
// In cq_prefetch.cpp, fetch_q_get_cmds()
WAYPOINT("HQW");
uint32_t heartbeat = 0U;
while ((fetch_size = *prefetch_q_rd_ptr) == 0U) {
    invalidate_l1_cache();
    IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
}
```

This is the normal idle state -- but it becomes a hang when:
1. The host has enqueued commands but the fetch queue write was not made visible (missing `sfence` before the TLB write).
2. The host thread is blocked waiting for a completion event that requires more commands to be dispatched first (circular dependency).
3. A bug causes the host to skip writing the fetch queue entry after writing the issue queue data.
4. The PCIe TLB write from the host does not reach the device L1 (hardware fault, PCIe link degradation).

**Diagnosis Steps:**
1. Check watcher output for the prefetch core: waypoint should show `HQW`. The core coordinates and device ID identify which command queue is stalled.
2. Read the prefetch queue entries from L1 on the prefetcher core. If the entry at the read pointer is zero, the host did not write work.
3. Check the host-side `SystemMemoryManager` state: compare `prefetch_q_dev_ptrs[cq_id]` (host write pointer) with the device-side read pointer (`prefetch_q_rd_ptr_addr`). If they match, the host has not enqueued anything; if they differ, the write may not have been visible.
4. Check if the host thread is blocked (e.g., in `completion_queue_wait_front`) -- this indicates a circular dependency.
5. Check PCIe link health via `tt-smi`. TLP errors or link retraining can cause writes to be lost.

**Fix:**
```cpp
// BUGGY: missing sfence before fetch queue write
void enqueue_command(SystemMemoryManager& mgr, uint8_t cq_id, uint32_t size) {
    void* region = mgr.issue_queue_reserve(size, cq_id);
    write_command_data(region, size);
    mgr.issue_queue_push_back(size, cq_id);
    // BUG: fetch_queue_write may see stale issue queue data
    mgr.fetch_queue_write(size, cq_id);
}

// CORRECTED: issue_queue_push_back already writes to hugepage;
// fetch_queue_write calls tt_driver_atomics::sfence() internally
// Ensure issue_queue_push_back is called BEFORE fetch_queue_write
void enqueue_command(SystemMemoryManager& mgr, uint8_t cq_id, uint32_t size) {
    void* region = mgr.issue_queue_reserve(size, cq_id);
    write_command_data(region, size);
    mgr.issue_queue_push_back(size, cq_id);  // updates host write pointer in hugepage
    mgr.fetch_queue_reserve_back(cq_id);      // waits for fetch queue space
    mgr.fetch_queue_write(size, cq_id);        // sfence + TLB write to prefetcher L1
}
```

**Prevention:**
- Always call `issue_queue_push_back` before `fetch_queue_write`.
- For host-side circular dependencies, use separate command queues or break the dependency with explicit synchronization points.
- Always wrap device operations in RAII guards that call `Finish()` or `close_device()` in the destructor.
- Set `TT_METAL_OPERATION_TIMEOUT_SECONDS` to a reasonable value so that host-side timeouts detect the hang rather than spinning forever.

---

## 4.1.2 Prefetch Kernel Stall Waiting for Dispatch Buffer Space

**Symptom:** The prefetch kernel is stuck at waypoint `UAPW` (inside `CBReader::acquire_pages`) or `CNSW`/`CNIW` (inside `CBWriter::acquire_pages`). The prefetcher has data to relay but the dispatch buffer is full -- the dispatch kernel has not consumed pages fast enough.

**Root Cause:** The prefetch kernel writes data into the dispatch buffer (`downstream_cb`), which is a ring buffer managed by page-granularity semaphores. Before writing a page, the prefetcher calls `cb_writer.acquire_pages(n)`, which spins until the dispatch kernel increments its semaphore, signaling that pages have been consumed:

```cpp
// In cq_common.hpp, CBWriter::acquire_pages
WAYPOINT("CNSW");  // or CNIW for inline
while (wrap_gt(num_pages_acquired + n, *sem_addr)) {
    invalidate_l1_cache();
    IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
}
WAYPOINT("CNSD");
```

The dispatch kernel only releases pages after processing a block of commands and after waiting for outstanding NOC writes from that block to complete (see `release_block_pages` in `CBReaderWithReleasePolicy`). If the dispatch kernel is itself stalled -- waiting for workers, waiting for completion queue space, or waiting on a NOC barrier -- the prefetcher will back up.

**Diagnosis Steps:**
1. Identify the prefetch core waypoint (`UAPW` or `CNSW`/`CNIW`).
2. Read the dispatch kernel's semaphore value and compare with the prefetcher's expected count.
3. Find the dispatch kernel's waypoint -- this reveals why dispatch is stalled (e.g., `PWW` for waiting on workers, `QRBW` for completion queue backpressure).
4. If dispatch shows `NWBW` (NOC write barrier), investigate NOC-level backpressure per [Chapter 2, Section 04](../ch02_kernel_and_noc_hangs/04_noc_barrier_and_semaphore_hangs.md).

**Fix:** The fix depends on the downstream stall. Common resolutions:
- If workers are slow and dispatch is blocked at `PWW`/`WCW`: optimize kernel execution time or reduce the number of programs dispatched without synchronization.
- If the completion queue is full (`QRBW`): ensure the host is polling and consuming completion events promptly.
- If NOC backpressure is the root cause: see [Chapter 3, Section 02](../ch03_memory_related_hangs/02_dram_and_noc_backpressure.md).

**Prevention:**
- Profile the dispatch pipeline to ensure the bottleneck is not in the dispatch or worker stages.
- Increase the dispatch buffer size if the pipeline is frequently stalled due to buffer exhaustion.

---

## 4.1.3 Prefetch StallState: STALLED for ExecBuf (Trace Replay)

**Symptom:** The prefetch kernel stops fetching new commands from the host. Waypoint may show `HQW` (idle) but with `stall_state == STALLED`. Commands enqueued by the host after the trace command are not being processed.

**Root Cause:** When the host enqueues a trace replay via `CQ_PREFETCH_CMD_EXEC_BUF`, the fetch queue entry is marked with a stall flag (MSB set). The prefetcher processes this entry, transitions to `StallState::STALLED`, and stops reading new fetch queue entries:

```cpp
// In cq_prefetch.cpp, fetch_q_get_cmds()
if (stall_state == StallState::STALLED) {
    ASSERT(inflight_count == 0U);
    ASSERT(issue_fence == fence);
    return;  // Do not process any more fetch queue entries
}
```

The stall is intentional: trace replay (`CQ_PREFETCH_CMD_EXEC_BUF`) hijacks the `cmddat_q` to stream commands from a DRAM buffer. If the prefetcher continued fetching from the host during this time, the newly fetched data would corrupt the trace data.

The stall is lifted when the trace completes and the dispatch kernel sends `CQ_DISPATCH_CMD_EXEC_BUF_END`, which the prefetch kernel processes in `process_exec_buf_end`:

```cpp
// process_exec_buf_end resets stall_state to NOT_STALLED
stall_state = StallState::NOT_STALLED;
```

A hang occurs if the trace replay itself hangs (see [Section 03](./03_trace_replay_and_lightmetal.md)), or if the `EXEC_BUF_END` command is lost or corrupted.

**Diagnosis Steps:**
1. Read the prefetch kernel's `stall_state` variable from L1 -- if it equals `1` (`STALLED`), trace replay is in progress.
2. Check if the dispatch kernel is processing exec_buf commands (look for trace buffer DRAM reads).
3. If the dispatch kernel is stalled, diagnose its waypoint to find the root cause within the trace command sequence.
4. Verify exec buf integrity: read the DRAM buffer being executed and check for valid command sequences ending with `CQ_DISPATCH_CMD_EXEC_BUF_END`.

**Fix:** Fix the underlying trace replay hang (see [Section 03](./03_trace_replay_and_lightmetal.md) for trace-specific scenarios).

**Prevention:**
- Validate trace buffers after capture using `TraceBuffer::validate()`.
- Ensure trace captures do not include operations that depend on dynamic host state that may change between capture and replay.
- Do not modify device state (buffer allocations, sub-device configuration) between trace capture and replay without recapturing.

---

## 4.1.4 Fetch Queue Full: Host Blocked in fetch_queue_reserve_back

**Symptom:** The host thread hangs inside `SystemMemoryManager::fetch_queue_reserve_back`. Eventually, the `loop_and_wait_with_timeout` fires with `TIMEOUT: device timeout in fetch queue wait, potential hang detected`.

**Root Cause:** The host writes fetch queue entries to the prefetcher's L1 via TLB. The fetch queue is a fixed-size ring buffer (typically 256 entries). Before writing a new entry, the host must ensure there is space by reading the prefetcher's read pointer:

```cpp
// In system_memory_manager.cpp, fetch_queue_reserve_back()
auto fetch_operation_body = [&]() {
    tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        &fence, sizeof(uint32_t), this->prefetcher_cores[cq_id], prefetch_q_rd_ptr);
    this->prefetch_q_dev_fences[cq_id] = fence;
};
auto fetch_wait_condition = [&]() -> bool {
    return this->prefetch_q_dev_ptrs[cq_id] == this->prefetch_q_dev_fences[cq_id];
};
auto fetch_on_timeout = []() {
    MetalContext::instance().on_dispatch_timeout_detected();
    TT_THROW("TIMEOUT: device timeout in fetch queue wait, potential hang detected");
};
```

If the prefetcher is not consuming entries (because it is stalled waiting on the dispatch buffer, or because it is in `STALLED` state for a trace), the fetch queue fills up and the host blocks.

The timeout mechanism in `loop_and_wait_with_timeout` monitors dispatch progress by reading the dispatch kernel's progress counter from L1. If the counter does not advance within `TT_METAL_OPERATION_TIMEOUT_SECONDS`, `MetalContext::on_dispatch_timeout_detected()` is called, which can serialize Inspector data and execute a configurable triage command.

**Diagnosis Steps:**
1. Check the host stack trace -- it should show `fetch_queue_reserve_back` -> `loop_and_wait_with_timeout`.
2. Read the prefetcher's `prefetch_q_rd_ptr_addr` from device L1 and compare with the host's `prefetch_q_dev_ptrs[cq_id]`.
3. If the read pointer has not advanced, the prefetcher is stalled -- diagnose its waypoint.
4. Check the dispatch progress counter: if it is advancing, the timeout will not fire (the system is making progress, just slowly).

**Fix:** Resolve the downstream stall that is preventing the prefetcher from consuming fetch queue entries. If the root cause is a worker hang, fix the kernel; if the root cause is completion queue backpressure, ensure the host drains completions.

**Prevention:**
- Set `TT_METAL_OPERATION_TIMEOUT_SECONDS` to a reasonable value for the workload to detect hangs early.
- Monitor dispatch progress with the Inspector RPC interface for long-running workloads.
- Do not set `TT_METAL_CQ_SIZE_OVERRIDE` to a value smaller than the minimum required by the fetch queue sizing constraint.

---

## 4.1.5 Dispatch Kernel Waiting for Prefetcher Data

**Symptom:** The dispatch kernel is stuck at waypoint `UAPW` (inside `CBReader::acquire_pages`). It has processed all available commands and is waiting for the prefetcher to write more data into the dispatch buffer.

**Root Cause:** This is the mirror of Scenario 4.1.2: the dispatch kernel's `dispatch_cb_reader` calls `acquire_pages()`, which spins on a semaphore waiting for the prefetcher to signal that new pages are available:

```cpp
// In cq_common.hpp, CBReader::acquire_pages()
WAYPOINT("UAPW");
uint32_t heartbeat = 0;
do {
    invalidate_l1_cache();
    IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat, 0);
} while ((upstream_count_ = *sem_addr) == local_count_);
WAYPOINT("UAPD");
```

Normally this is the dispatch kernel's idle state. It becomes a hang when:
1. The prefetcher is stalled (see Scenarios 4.1.1 through 4.1.4).
2. The prefetcher has crashed (assertion failure, deliberate hang from `!CMD`).
3. An NOC write from the prefetcher to the dispatch buffer failed silently, so the semaphore was never incremented.

**Diagnosis Steps:**
1. Confirm the dispatch kernel waypoint is `UAPW`.
2. Read the prefetcher's waypoint to determine if it is stalled or has crashed.
3. If the prefetcher shows `!CMD`, it received an invalid command -- check the command data in `cmddat_q` for corruption.
4. Check the semaphore values on both the prefetcher and dispatch cores to verify they are consistent. If valid command data is present in the dispatch CB beyond `cmd_ptr` but the semaphore count has not advanced, the semaphore tracking is out of sync.

**Fix:** Resolve the upstream stall in the prefetcher. If the prefetcher has crashed due to an invalid command, fix the host-side code that generated the malformed command. If semaphore tracking is out of sync, this is a firmware bug -- file a bug report with the semaphore values and dispatch CB memory dump.

**Prevention:**
- Enable the watcher to detect prefetcher crashes early.
- Use the `CQ_PREFETCH_CMD_DEBUG` command to inject waypoint checkpoints into the command stream for debugging.

---

## 4.1.6 Dispatch Kernel Waiting for Workers (process_wait / go signal)

**Symptom:** The dispatch kernel is stuck at waypoint `PWW` (inside `process_wait`) or `WCW` (inside `process_go_signal_mcast_cmd` or `wait_for_workers` in dispatch_s). Workers have not completed their kernels.

**Root Cause:** The dispatch kernel issues a `CQ_DISPATCH_CMD_WAIT` or `CQ_DISPATCH_CMD_SEND_GO_SIGNAL` command that requires waiting for worker cores to signal completion. Workers signal completion by incrementing a stream register on the dispatch core. The dispatch kernel spins until the stream register reaches the expected count:

```cpp
// In cq_dispatch.cpp, process_wait()
WAYPOINT("PWW");
if (wait_memory) {
    volatile tt_l1_ptr uint32_t* sem_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    do {
        invalidate_l1_cache();
        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
    } while (!wrap_ge(*sem_addr, count));
}
if (wait_stream) {
    last_wait_count = count;
    last_wait_stream = stream;
    volatile uint32_t* sem_addr = reinterpret_cast<volatile uint32_t*>(
        STREAM_REG_ADDR(stream, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX));
    do {
        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
    } while (!stream_wrap_ge(*sem_addr, count));
}
WAYPOINT("PWD");
```

The triage tool can read `last_wait_count` and `last_wait_stream` from dispatch kernel L1 to determine exactly what the dispatcher is waiting for. These are declared `volatile` and available via `extern "C"` linkage for easy extraction.

A hang occurs when:
1. A worker kernel hangs (see [Chapter 2](../ch02_kernel_and_noc_hangs/index.md) for kernel-level hangs).
2. The worker count is wrong -- the dispatch kernel expects more completions than workers will produce (e.g., a sub-device configuration mismatch).
3. Workers are signaling a different stream than the dispatcher expects.
4. The worker cores were never launched because the go signal was lost or sent to the wrong cores.

**Diagnosis Steps:**
1. Read `last_wait_count` and `last_wait_stream` from dispatch kernel L1.
2. Read the corresponding stream register value: `STREAM_REG_ADDR(stream, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX)`. The delta between the current value and the expected count reveals how many workers have not signaled.
3. Use the watcher to check all worker core waypoints -- find which cores are stuck and what they are waiting for.
4. Verify that the number of workers in the sub-device configuration matches the expected completion count.
5. Check whether the go signal was actually sent (inspect the go signal address in worker core L1). If the go signal value is 0 or stale, the workers were never launched.

**Fix:** Fix the worker kernel hang, or correct the worker count mismatch. If the issue is a sub-device configuration error, ensure `create_sub_device_manager` is called with the correct core sets. If the go signal was lost, check for NOC errors on the dispatch-to-worker write path.

**Prevention:**
- Validate sub-device configurations before dispatch.
- Always validate that the number of cores in the program grid matches the expected completion count.
- Use smaller programs or intermediate synchronization points to limit the blast radius of a worker hang.

---

## 4.1.7 Dispatch Subordinate Waiting for Dispatch Master Notification

**Symptom:** The dispatch subordinate (`dispatch_s`) is stuck at waypoint `DCW` -- it is waiting for the dispatch master (`dispatch_d`) to signal that it is safe to send a go signal to workers.

**Root Cause:** When `dispatch_s` is enabled (distributed dispatcher mode), program dispatch is split: `dispatch_d` writes kernel configuration data to workers, then notifies `dispatch_s` via the `CQ_DISPATCH_NOTIFY_SUBORDINATE_GO_SIGNAL` command. `dispatch_s` waits for this notification before sending the go signal:

```cpp
// In cq_dispatch_subordinate.cpp, process_go_signal_mcast_cmd()
WAYPOINT("DCW");
uint32_t& mcasts_sent = num_mcasts_sent[sync_index];
while (wrap_ge(mcasts_sent, *sync_sem_addr)) {
    invalidate_l1_cache();
    update_worker_completion_count_on_dispatch_d();
}
mcasts_sent++;
```

Note that `dispatch_s` also feeds worker completion counts back to `dispatch_d` via `update_worker_completion_count_on_dispatch_d()`. If this write fails, `dispatch_d` may be stuck waiting for completions that `dispatch_s` has already observed but failed to relay.

A hang occurs when:
1. `dispatch_d` is itself stalled before reaching the `NOTIFY_SUBORDINATE_GO_SIGNAL` command.
2. The notification write from `dispatch_d` to `dispatch_s` fails (NOC error, wrong address).
3. When `distributed_dispatcher` is true and `dispatch_d` and `dispatch_s` are on separate cores, the inline write used for notification (`noc_inline_dw_write`) may fail due to Blackhole inline write backpressure (see [Chapter 3, Section 01](../ch03_memory_related_hangs/01_l1_memory_corruption_and_overflow.md)).
4. The sync semaphore addresses are misconfigured (wrong `dispatch_s_sync_sem_base_addr`).

**Diagnosis Steps:**
1. Check `dispatch_s` waypoint: `DCW`.
2. Check `dispatch_d` waypoint to determine if it is stalled before the notification.
3. Read the sync semaphore address on `dispatch_s` L1 and the counter on `dispatch_d` -- if `dispatch_d`'s counter is ahead, the notification write did not arrive.
4. If `dispatch_d` is at `NWBW` (write barrier), investigate NOC-level issues.
5. In distributed mode, check whether the inline dword writes from `dispatch_s` to `dispatch_d` (for worker completion count updates) are reaching `dispatch_d`. Read the stream registers on both cores.

**Fix:** Resolve the `dispatch_d` stall, or fix the NOC write failure that prevented the notification from arriving.

**Prevention:**
- On Blackhole, ensure the inline write workaround is active for the notification path.
- In distributed dispatcher mode, both `dispatch_d` and `dispatch_s` must be configured with matching stream indices and semaphore addresses. Validate the dispatch topology configuration.

---

## 4.1.8 Dispatch Subordinate Page Acquisition Stall

**Symptom:** The dispatch subordinate (`dispatch_s`) is stuck at waypoint `DAPW` -- it cannot acquire a command page from the prefetcher.

**Root Cause:** `dispatch_s` receives its commands through a separate circular buffer written by the prefetcher (`dispatch_s_buffer`). The page acquisition uses the same semaphore-based mechanism as the main dispatch buffer:

```cpp
// In cq_dispatch_subordinate.cpp, cb_acquire_pages_dispatch_s()
WAYPOINT("DAPW");
uint32_t heartbeat = 0;
while (wrap_gt(num_pages_acquired + n, *sem_addr)) {
    invalidate_l1_cache();
    update_worker_completion_count_on_dispatch_d();
    IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
}
WAYPOINT("DAPD");
```

This stalls when the prefetcher has not written the command for `dispatch_s` yet. The prefetcher writes to both the main dispatch buffer and the `dispatch_s` buffer in the same relay path. If the prefetcher is stalled on the main dispatch buffer (Scenario 4.1.2), `dispatch_s` commands are also delayed.

**Diagnosis Steps:**
1. Confirm `dispatch_s` waypoint is `DAPW`.
2. Check the prefetcher waypoint -- if it is stalled, `dispatch_s` is a secondary victim.
3. Read the `dispatch_s` buffer semaphore and the prefetcher's write state for the `dispatch_s` path.

**Fix:** Resolve the prefetcher stall (see Scenarios 4.1.1-4.1.4).

**Prevention:** Same as Scenario 4.1.2 -- ensure the main dispatch pipeline is not bottlenecked.

---

## 4.1.9 Worker Config Buffer Exhaustion

**Symptom:** The dispatch pipeline stalls because the `WorkerConfigBufferMgr` cannot allocate space for a new program's kernel configuration. The host-side `reserve` call triggers a `need_sync` condition, which inserts a `CQ_DISPATCH_CMD_WAIT` into the command stream, but workers never complete to satisfy the wait.

**Root Cause:** The `WorkerConfigBufferMgr` manages a ring buffer of kernel configuration slots in worker L1. Each program occupies a slot; the slot is freed only after the program completes execution. If programs are dispatched faster than workers execute them, all slots fill up:

```cpp
// In worker_config_buffer.cpp, reserve()
struct ConfigBufferEntry {
    uint32_t addr;
    uint32_t size;
    uint32_t sync_count;
};
// If addr + size > end, need_sync = true
```

The ring buffer has only `kernel_config_entry_count = 8` entries. When all 8 entries are occupied by in-flight programs, the host must insert a wait command to drain at least one program before dispatching the next. If the oldest program is hung (worker kernel hang), this creates a deadlock: the dispatch pipeline is waiting for a completion that will never come.

After a trace replay, the situation is worse: `mark_completely_full` is called, which marks the entire config buffer as occupied with a sync count matching the post-trace expected completion count. Any subsequent non-trace dispatch must wait for the trace to fully complete.

**Diagnosis Steps:**
1. Host-side: check if `WorkerConfigBufferMgr::reserve` is returning `need_sync = true` for every program.
2. Check the dispatch kernel waypoint -- if it is at `PWW` or `WCW`, it is waiting for the sync that will free config buffer space.
3. Read `last_wait_count` and `last_wait_stream` from dispatch kernel L1. This indicates the sync count the config buffer manager requested.
4. If the sync never completes, one or more worker kernels are hung -- use watcher to identify them.

**Fix:**
```cpp
// BUGGY: dispatching too many programs without synchronization
for (int i = 0; i < 100; i++) {
    EnqueueProgram(cq, program[i], false);
    // No synchronization -- config buffer fills after ~8 programs
}
// If program[0] hangs, all subsequent programs are blocked

// CORRECTED: add periodic synchronization to prevent config buffer overflow
for (int i = 0; i < 100; i++) {
    EnqueueProgram(cq, program[i], false);
    if (i % 4 == 3) {
        Finish(cq);  // Drain completed programs, freeing config buffer slots
    }
}
```

**Prevention:**
- Insert periodic `Finish` or `Synchronize` calls in long dispatch loops.
- Monitor config buffer utilization during development.
- Ensure worker kernels complete within a bounded time.
- Use trace replay for repetitive workloads to avoid per-program config buffer allocation overhead.

---

## 4.1.10 Completion Queue Backpressure (QRBW)

**Symptom:** The dispatch kernel is stuck at waypoint `QRBW` (inside `completion_queue_reserve_back`). It has data to write to the host completion queue but the queue is full.

**Root Cause:** The completion queue is a ring buffer in hugepage memory. The dispatch kernel writes completion events; the host reads them. Before writing, the dispatch kernel checks for available space:

```cpp
// In cq_dispatch.cpp, completion_queue_reserve_back()
WAYPOINT("QRBW");
do {
    invalidate_l1_cache();
    completion_rd_ptr_and_toggle = *get_cq_completion_read_ptr();
    // ... calculate available_space ...
} while (data_size_16B > available_space);
WAYPOINT("QRBD");
```

If the host is not calling `completion_queue_pop_front` to advance the read pointer, the completion queue fills up and the dispatch kernel stalls.

This can happen when:
1. The host thread is blocked in a long computation and not polling for completions.
2. The host is waiting for a different event (e.g., on a different CQ) while the completion queue for this CQ fills up.
3. The host crashed or exited without draining the completion queue.
4. The host has consumed the data but failed to update the read pointer on the device (TLB write failure).

**Diagnosis Steps:**
1. Confirm dispatch waypoint is `QRBW`.
2. Read the completion queue read pointer from dispatch kernel L1 (`dev_completion_q_rd_ptr`) and compare with the write pointer (`dev_completion_q_wr_ptr`). If the write pointer has caught up to the read pointer (accounting for toggle bits), the queue is full.
3. Check the host-side completion queue read pointer (`SystemMemoryCQInterface::completion_fifo_rd_ptr`) -- if it matches the device-side read pointer, the host has not consumed any completions.
4. Check for PCIe errors: the read pointer update from host to device goes through a TLB write. PCIe errors can prevent delivery.

**Fix:** Ensure the host thread is regularly polling for and consuming completion events. If using asynchronous dispatch, use a dedicated thread for completion polling.

**Prevention:**
- Use `Finish` or `Synchronize` periodically to drain completions.
- Do not block the host thread on long computations between dispatch calls.
- Use `TT_METAL_OPERATION_TIMEOUT_SECONDS` to detect when the completion queue is stuck.

---

## 4.1.11 Invalid Dispatch Command (!CMD Waypoint)

**Symptom:** The dispatch kernel prints "dispatcher_d invalid command" or "dispatcher_h invalid command" via DPRINT and then hits waypoint `!CMD` followed by `ASSERT(0)`, causing a deliberate hang.

**Root Cause:** The dispatch kernel's command switch statement encounters a `cmd_id` value that does not match any known `CQDispatchCmdId`. This is always a bug -- it means the command data in the dispatch buffer is corrupted:

```cpp
// In cq_dispatch.cpp, process_cmd_d()
switch (cmd->base.cmd_id) {
    case CQ_DISPATCH_CMD_WRITE_LINEAR: ...
    case CQ_DISPATCH_CMD_WRITE_PACKED: ...
    // ...
    default:
        DPRINT << "dispatcher_d invalid command:" << cmd_ptr << " " << ...;
        WAYPOINT("!CMD");
        ASSERT(0);
}
```

Common causes:
1. The prefetcher relayed corrupted data from the issue queue (host bug).
2. An NOC write from the prefetcher to the dispatch buffer was silently corrupted.
3. The `cmd_ptr` calculation drifted due to a page size mismatch or alignment error in the page-rounding logic (`cmd_ptr = round_up_pow2(cmd_ptr, dispatch_cb_page_size)`).
4. A previous command's length field was wrong, causing the dispatch kernel to interpret data as a command header.
5. Ring buffer overflow: the prefetch kernel wrote past the end of the dispatch CB, overwriting commands that the dispatch kernel had not yet consumed.

**Diagnosis Steps:**
1. Read the DPRINT output (if enabled) for the invalid command bytes -- includes `cmd_ptr`, available bytes, and the first 4 dwords at the command pointer.
2. Read the dispatch buffer around `cmd_ptr` from L1 to examine the corrupted data.
3. Compare the data with the expected command stream in the host issue queue.
4. Check for L1 corruption per [Chapter 3, Section 01](../ch03_memory_related_hangs/01_l1_memory_corruption_and_overflow.md).
5. Check semaphore counts: if the dispatch CB semaphore counts are inconsistent (dispatch thinks more data is available than actually written), the ring buffer may have overflowed.

**Fix:** Fix the host-side code that generated the malformed command. If the corruption is in-transit, investigate NOC-level issues. If L1 corruption is the cause, identify and fix the source of corruption.

**Prevention:**
- Enable the `CQ_PREFETCH_CMD_DEBUG` / `CQ_DISPATCH_CMD_DEBUG` commands for checksum validation of the command stream.
- Use address sanitization to catch NOC errors early.
- Enable the watcher's L1 sanitization to detect NOC writes that could corrupt the dispatch CB.

---

## 4.1.12 Dispatch CB Page Release Stall (CBRW)

**Symptom:** The dispatch kernel (or prefetcher) is stuck at waypoint `CBRW` -- it is waiting for outstanding NOC writes from a previous block to be sent before releasing the block's pages back to the producer.

**Root Cause:** The `CBReaderWithReleasePolicy::release_block_pages` method delays page release by one block: when a block is completed, it records the current NOC write count and waits for those writes to be sent before releasing the *previous* block. This ensures that the producer does not overwrite data that is still being transferred:

```cpp
// In cq_common.hpp, release_block_pages()
WAYPOINT("CBRW");
while (!noc_nonposted_writes_sent_at_count(noc_index, this->block_noc_writes_to_clear_));
ReleasePolicy::template release<noc_idx, noc_xy, sem_id>(cb_pages_per_block);
WAYPOINT("CBRD");
```

This becomes a hang when NOC writes from the block never complete -- for example, due to:
1. A NOC write targeting an invalid address that stalls the NOC.
2. DRAM backpressure preventing write completion (see [Chapter 3, Section 02](../ch03_memory_related_hangs/02_dram_and_noc_backpressure.md)).
3. The mcast path reservation hang (see [Chapter 2, Section 04](../ch02_kernel_and_noc_hangs/04_noc_barrier_and_semaphore_hangs.md)).

**Diagnosis Steps:**
1. Check the waypoint: `CBRW`.
2. Read `noc_nonposted_writes_num_issued` and `noc_nonposted_writes_acked` to determine how many writes are outstanding.
3. Check NOC status registers for stalled transactions.

**Fix:** Resolve the underlying NOC issue that is preventing writes from completing.

**Prevention:**
- Enable NOC address sanitization to catch invalid write targets early.
- Profile DRAM bandwidth to detect saturation before it causes stalls.

---

## 4.1.13 Relay Mux Topology Hangs (Multi-Device Dispatch)

**Symptom:** In multi-device configurations (T3K, Galaxy), the dispatch kernel stalls during `relay_to_next_cb` -- it cannot write to the downstream device via the fabric relay client.

**Root Cause:** When dispatch spans multiple devices, the `relay_client` connects through a **fabric mux** kernel. The dispatch kernel writes data to the downstream dispatch buffer on a remote device via the fabric relay path. Flow control is managed through semaphores and a buffer index scheme:

```cpp
// In cq_dispatch.cpp, relay_to_next_cb()
relay_client.init_write_state_only<my_noc_index, NCRISC_WR_CMD_BUF>(
    get_noc_addr_helper(downstream_noc_xy, 0));
```

In the multi-device dispatch topology, the pipeline extends beyond a single chip. For remote (non-MMIO) devices, commands flow: `prefetch_h` (MMIO device) --> Ethernet tunnel --> `prefetch_d` (remote device) --> `dispatch_d` (remote device) --> `dispatch_h` (MMIO device, for completion events). Each link in this chain is an additional hang point.

Hangs can occur when:
1. The remote device's dispatch buffer is full and not being consumed.
2. The fabric mux connection handshake did not complete (connection setup failure).
3. The Ethernet link between devices is down or experiencing errors.
4. The remote device was reset or closed, leaving the fabric mux in an inconsistent state.
5. Message ordering violations across the mux channels cause state corruption on the receiving end.

**Diagnosis Steps:**
1. Check the dispatch kernel waypoint on the source device.
2. Read the fabric mux status and connection handshake addresses on both devices.
3. Check Ethernet link status via UMD or the watcher's Ethernet link status feature.
4. Verify the remote device's dispatch kernel is running and consuming data.

**Fix:** Ensure all devices in the multi-device topology are properly initialized before dispatching. If an Ethernet link is down, the fabric layer must handle the error or the workload must be repartitioned.

**Prevention:**
- Validate fabric connections before running multi-device workloads.
- Implement timeouts in the fabric relay path (currently, stalls are unbounded).
- See [Chapter 5](../ch05_multi_chip_and_ccl_hangs/index.md) for comprehensive multi-chip hang analysis.

---

## 4.1.14 Mcast Path Reservation Hang Workaround in Dispatch

**Symptom:** The dispatch kernel hangs during `process_write_packed` when issuing multicast NOC writes. The hang is caused by a known hardware issue with multicast path reservations that affects all architectures supporting multicast (WH, BH, Quasar).

**Root Cause:** Issuing a multicast NOC write when a previous multicast from a different source has not completed can cause a NOC-level deadlock due to path reservation conflicts. The dispatch kernel works around this by using a `wait_for_barrier` lambda that calls `noc_async_write_barrier()` before every multicast write in the `process_write_packed` loop. The lambda skips the barrier for unicast writes (see [Chapter 2, Section 04](../ch02_kernel_and_noc_hangs/04_noc_barrier_and_semaphore_hangs.md) for the full code).

If the write barrier itself hangs (because the writes it is waiting for are stalled due to the same path reservation issue or due to DRAM backpressure), the dispatch kernel stalls at `NWBW`.

This is distinct from the general NOC barrier hangs described in Chapter 2 because it specifically occurs in the dispatch kernel's write-packed path and is triggered by the hardware workaround itself.

**Diagnosis Steps:**
1. Check the dispatch kernel waypoint: `NWBW` during `process_write_packed`.
2. Read NOC transaction counters (`noc_nonposted_writes_num_issued` vs. `noc_nonposted_writes_acked`) to determine how many writes are outstanding.
3. Check if the outstanding writes target cores that are themselves experiencing NOC backpressure.

**Fix:** Resolve the underlying NOC backpressure or stall that prevents the write barrier from completing. If the issue is DRAM saturation, reduce concurrent write traffic.

**Prevention:**
- On Wormhole, be aware that multicast-heavy dispatch patterns (many packed writes) may trigger this workaround path more frequently.
- Profile NOC utilization during dispatch to detect saturation.
- See [Chapter 2, Section 04](../ch02_kernel_and_noc_hangs/04_noc_barrier_and_semaphore_hangs.md) for the general mcast path reservation workaround.

---

## Summary Table

| Scenario | Waypoint | Stalled Component | Root Cause Category | Severity |
|----------|----------|-------------------|---------------------|----------|
| 4.1.1 Empty fetch queue | `HQW` | Prefetcher | Host not sending work / circular dependency | Medium |
| 4.1.2 Dispatch buffer full | `UAPW`/`CNSW` | Prefetcher | Dispatch backpressure | High |
| 4.1.3 ExecBuf stall | `HQW` (STALLED) | Prefetcher | Trace replay in progress | Medium |
| 4.1.4 Fetch queue full | Host blocked | Host | Prefetcher backpressure | High |
| 4.1.5 No prefetcher data | `UAPW` | Dispatch | Prefetcher stall or crash | High |
| 4.1.6 Worker wait | `PWW`/`WCW` | Dispatch | Worker hang or count mismatch | Critical |
| 4.1.7 Subordinate notification | `DCW` | Dispatch_s | Dispatch_d stall or NOC failure | High |
| 4.1.8 Subordinate page stall | `DAPW` | Dispatch_s | Prefetcher backpressure | Medium |
| 4.1.9 Config buffer exhaustion | `PWW` (indirect) | Host/Dispatch | Too many in-flight programs | High |
| 4.1.10 Completion queue full | `QRBW` | Dispatch | Host not consuming completions | High |
| 4.1.11 Invalid command | `!CMD` | Dispatch | Command corruption | Critical |
| 4.1.12 Page release stall | `CBRW` | Dispatch | NOC write failure | High |
| 4.1.13 Relay mux stall | Various | Dispatch (relay) | Fabric/remote device issue | High |
| 4.1.14 Mcast path reservation | `NWBW` | Dispatch | WH hardware workaround stall | High |

---

[Previous: Chapter Index](./index.md) | [Next: Host Synchronization and Timeout Detection](./02_host_synchronization_and_timeout_detection.md)
