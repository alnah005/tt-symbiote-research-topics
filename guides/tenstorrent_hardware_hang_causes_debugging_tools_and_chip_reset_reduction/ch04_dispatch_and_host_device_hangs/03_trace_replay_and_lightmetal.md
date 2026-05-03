# 4.3 Trace Replay and LightMetal

[Previous: Host Synchronization and Timeout Detection](./02_host_synchronization_and_timeout_detection.md) | [Next: Chapter 5 -- Multi-Chip, CCL, and Fabric Hang Causes](../ch05_multi_chip_and_ccl_hangs/index.md)

---

This section covers hang causes specific to the **trace capture/replay** mechanism and the **LightMetal capture/replay** system. Trace replay is a performance optimization that pre-records a sequence of dispatch commands and replays them from a DRAM buffer, bypassing the normal host-to-prefetcher path. LightMetal is a higher-level capture/replay system that records entire Metal API call sequences for deterministic reproduction. Both introduce unique hang scenarios related to stale state, device configuration drift, and state synchronization.

Reference files: `tt_metal/impl/trace/trace_buffer.hpp`, `tt_metal/impl/trace/dispatch.cpp`, `tt_metal/impl/dispatch/system_memory_manager.cpp`, `tt_metal/llrt/hal/generated/dev_msgs.hpp` (`RUN_MSG_REPLAY_TRACE`), `tt_metal/impl/lightmetal/lightmetal_capture.hpp`, `tt_metal/impl/lightmetal/lightmetal_replay_impl.hpp`, `tt_metal/impl/dispatch/worker_config_buffer.hpp`

## 4.3.0 Trace Capture/Replay Architecture

### Capture Phase: Bypass Mode

During trace capture, the host dispatches programs normally but additionally records the raw command bytes that flow through the dispatch pipeline. The `SystemMemoryManager` enters **bypass mode**, which redirects command data to an in-memory buffer instead of writing to hugepages:

```cpp
// system_memory_manager.cpp
void SystemMemoryManager::set_bypass_mode(const bool enable, const bool clear) {
    this->bypass_enable = enable;
    if (clear) {
        this->bypass_buffer.clear();
        this->bypass_buffer_write_offset = 0;
    }
}
```

In bypass mode, `issue_queue_reserve()` returns a pointer into the bypass buffer instead of hugepage memory, and `fetch_queue_write()` becomes a no-op. The captured commands are stored in a `TraceDescriptor`:

```cpp
// trace_buffer.hpp
struct TraceDescriptor {
    std::unordered_map<SubDeviceId, TraceWorkerDescriptor> descriptors;
    std::vector<SubDeviceId> sub_device_ids;
    std::vector<uint32_t> data;  // Raw command bytes
};

struct TraceWorkerDescriptor {
    uint32_t num_completion_worker_cores = 0;
    uint32_t num_traced_programs_needing_go_signal_multicast = 0;
    uint32_t num_traced_programs_needing_go_signal_unicast = 0;
};
```

The `TraceWorkerDescriptor` records per-sub-device metadata critical for replay synchronization: the expected completion signal count and the number of programs requiring multicast vs. unicast go signals.

```
Normal Dispatch:
  Host -> SystemMemoryManager -> hugepage -> prefetch kernel -> dispatch kernel -> workers

Trace Capture:
  Host -> SystemMemoryManager (bypass) -> TraceDescriptor.data (vector<uint32_t>)
  (same host API calls, but data is captured, not sent to device)
```

### Replay Phase: RUN_MSG_REPLAY_TRACE and ExecBuf

When the host replays a trace, the captured command data is loaded into a DRAM buffer (`TraceBuffer::buffer`), interleaved across DRAM banks with a carefully chosen page size (between 1 KB and 8 KB) to optimize prefetcher read bandwidth. The replay sequence (in `trace_dispatch::issue_trace_commands`) then executes:

1. **Send `RUN_MSG_REPLAY_TRACE` go signal** to all worker cores. This signal is defined as a special firmware-level value:

```cpp
// dev_msgs.hpp
constexpr uint32_t RUN_MSG_REPLAY_TRACE = 0xf0;
```

When a worker core's BRISC firmware receives this value, it does not start kernel execution. Instead, it resets the launch message ring buffer read pointer to the beginning, preparing workers to re-execute the captured command sequence from position 0.

2. **Wait for workers to acknowledge the reset** by issuing `CQ_DISPATCH_CMD_WAIT` commands with `CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM | CQ_DISPATCH_CMD_WAIT_FLAG_CLEAR_STREAM` flags. This ensures all workers are ready before trace commands arrive.

3. **Issue `CQ_PREFETCH_CMD_EXEC_BUF`** to the prefetch kernel, pointing it to the DRAM trace buffer. The fetch queue entry is marked with the stall flag (MSB set), which causes the prefetcher to transition to `StallState::STALLED` after processing the entry. In this mode, the prefetcher reads commands from the DRAM buffer instead of from the host issue queue.

4. **Dispatch processes commands** from the trace buffer identically to live dispatch. Go signals, wait commands, and worker writes all execute as captured.

5. **ExecBuf ends**: The last command in the trace buffer is `CQ_DISPATCH_CMD_EXEC_BUF_END`, which signals the prefetcher to exit stall mode (`StallState::NOT_STALLED`) and resume reading from the host issue queue.

### Host-Side State Reset

A critical part of trace replay is resetting host-side dispatch state to match the trace's expectations. `trace_dispatch::reset_host_dispatch_state_for_trace()` saves and resets three categories of state:

```cpp
// trace/dispatch.cpp, reset_host_dispatch_state_for_trace()
// 1. Save and reset expected_num_workers_completed to 0
std::copy(expected_num_workers_completed.begin(), ...);
std::fill(expected_num_workers_completed.begin(), ..., 0);

// 2. Save and reset launch message buffer write pointer to 0
std::copy(worker_launch_message_buffer_state.begin(), ...);
for (uint32_t i = 0; i < num_sub_devices; ++i) {
    worker_launch_message_buffer_state[i].reset();
}

// 3. Save and reset WorkerConfigBufferMgr
std::copy(config_buffer_mgr.begin(), ...);
for (uint32_t i = 0; i < num_sub_devices; ++i) {
    config_buffer_mgr[i].mark_completely_full(expected_num_workers_completed[i]);
}
```

After the trace replay completes, `load_host_dispatch_state()` restores the saved values and `update_worker_state_post_trace_execution` adjusts state to account for the trace's effects:

```
Before Trace Replay:
  expected_num_workers_completed = [N0, N1, ...]  (from prior programs)
  worker launch msg wptr = [W0, W1, ...]

During reset_host_dispatch_state_for_trace:
  Save: expected_num_workers_completed_reset = [N0, N1, ...]
  Reset: expected_num_workers_completed = [0, 0, ...]
  Save: worker_launch_message_buffer_state_reset = [W0, W1, ...]
  Reset: worker launch msg wptr = [0, 0, ...]
  Config buffer: mark_completely_full(0) -- all space marked as in-use

After Trace Replay (update_worker_state_post_trace_execution):
  expected_num_workers_completed[i] = desc.num_completion_worker_cores
  worker launch msg wptr = num_traced_programs (per type)
  Config buffer: mark_completely_full(expected_num_workers_completed[i])
```

This save/restore mechanism is fragile: if the device state drifts from what the host expects, the restored state will be wrong, causing subsequent operations to hang. The following scenarios document every known failure mode.

---

## 4.3.1 Stale Buffer Addresses in Trace Data After Reallocation

**Symptom:** A trace that worked correctly when first captured causes a hang or corruption when replayed after buffer reallocation. The dispatch kernel may write to a NOC address that no longer corresponds to a valid buffer. Workers may execute garbage instructions, read wrong data, or the watcher may report a NOC address violation (`DebugSanitizeNocAddrOverflow` or `DebugSanitizeNocAddrUnderflow`).

**Root Cause:** Trace capture records the exact dispatch commands, including all NOC addresses (buffer base addresses + write offsets). These addresses are absolute device addresses computed from the buffer allocation at capture time and embedded in `CQ_DISPATCH_CMD_WRITE_LINEAR`, `CQ_DISPATCH_CMD_WRITE_PACKED`, and `CQ_DISPATCH_CMD_WRITE_PACKED_LARGE` commands.

If a buffer is deallocated and reallocated between capture and replay, its address may change. The recorded commands will write to the old address. The `CQ_DISPATCH_CMD_SET_WRITE_OFFSET` command provides a relocation mechanism -- the dispatch kernel maintains `write_offset[]` values that are added to all non-host destination addresses -- but the buffer base addresses embedded in write commands are absolute and cannot be relocated without recapturing the trace.

This is distinct from L1 corruption scenarios in [Chapter 3, Section 01](../ch03_memory_related_hangs/01_l1_memory_corruption_and_overflow.md) because the corruption source is the trace replay itself, not a kernel bug.

**Diagnosis Steps:**
1. Check whether buffers were reallocated between capture and replay. Compare buffer addresses using `Buffer::address()` against the addresses embedded in the trace data.
2. Inspect the NOC address causing the violation (if the watcher reports one). Check whether the address matches a buffer's old allocation.
3. Dump the trace buffer contents from DRAM and decode the commands to find embedded addresses. Compare them with current allocations.
4. If workers execute garbage instructions after trace replay, dump L1 memory at the expected kernel binary locations and verify the contents match the trace data.

**Fix:**
```cpp
// BUGGY: reallocating buffer between capture and replay
auto buf = CreateBuffer(config);
auto trace_id = BeginTraceCapture(device, cq_id);
EnqueueProgram(cq, program_using_buf, false);
EndTraceCapture(device, cq_id, trace_id);

DeallocateBuffer(buf);
buf = CreateBuffer(config);  // May get a different address!

ReplayTrace(device, cq_id, trace_id, false);  // Uses stale address

// CORRECTED: keep buffers alive during replay
auto buf = CreateBuffer(config);
auto trace_id = BeginTraceCapture(device, cq_id);
EnqueueProgram(cq, program_using_buf, false);
EndTraceCapture(device, cq_id, trace_id);

// Do NOT reallocate buf
ReplayTrace(device, cq_id, trace_id, false);  // Same address as capture
```

**Prevention:**
- Keep all buffers used in a trace alive for the entire lifetime of the trace.
- If buffer reallocation is unavoidable, recapture the trace after reallocation.
- Treat the L1 and DRAM allocation state as frozen between trace capture and replay.
- Use `TraceBuffer::validate()` before replay to check trace buffer integrity (though this validates the trace DRAM buffer contents, not embedded target addresses).

---

## 4.3.2 Worker Launch Message Read Pointer Mismatch

**Symptom:** After trace replay issues the `RUN_MSG_REPLAY_TRACE` go signal, workers do not execute the expected kernels. Some workers may execute the wrong kernel, hang waiting for a go signal that already passed, or miss the `RUN_MSG_REPLAY_TRACE` signal entirely. The dispatch kernel stalls at `PWW` or `WCW` waiting for worker completions that never arrive.

**Root Cause:** The `RUN_MSG_REPLAY_TRACE` go signal (value `0xf0`) instructs workers to reset their launch message read pointer to the beginning of the ring buffer. The trace was captured with launch messages starting at position 0. When the trace is replayed, the dispatch kernel writes the same launch messages at the same positions.

The hang occurs when:
1. **A worker misses the replay go signal** because it is still executing a kernel from a previous operation. The go signal write may be overwritten by the next go signal before the worker reads it.
2. **The launch message ring buffer contains stale data** from previous (non-trace) dispatches. If programs were dispatched between capture and replay, old launch messages remain in the ring buffer. The worker resets its read pointer and reads the stale message before the trace's first launch message is written.
3. **The reset/restore state machine leaves pointers inconsistent.** After replay, `reset_host_dispatch_state_for_trace()` resets the host-side write pointer, but if the replay fails partway through, the host-side and device-side pointers diverge.

The trace replay sequence sends a `CQ_DISPATCH_CMD_WAIT` with `CLEAR_STREAM` to ensure workers have acknowledged the reset before proceeding. If this wait is satisfied prematurely (due to a worker count mismatch from a sub-device configuration change), the dispatch kernel may start writing before workers are ready.

**Diagnosis Steps:**
1. Check the dispatch kernel waypoint: if `PWW` immediately after trace replay starts, the wait is for the reset acknowledgment.
2. Read the go signal value in worker L1 (`mcast_go_signal_addr` or `unicast_go_signal_addr`) on each worker core. Verify it contains `RUN_MSG_REPLAY_TRACE` or the subsequent `RUN_MSG_GO` from the trace commands.
3. Read `launch_msg_rd_ptr` from each worker's mailbox. It should point to the first entry in the ring buffer after receiving `RUN_MSG_REPLAY_TRACE`.
4. Compare `last_wait_count` on the dispatch core with the actual stream register value. The delta reveals how many workers have not acknowledged.
5. Verify the `expected_num_workers_completed` value used in the `CQ_DISPATCH_CMD_WAIT` matches the actual number of workers that will signal completion.

**Fix:** Ensure all prior programs complete before starting trace replay. Use `Finish()` on the command queue before replaying:

```cpp
// BUGGY: trace captured with one sub-device config, replayed with another
auto trace_id = BeginTraceCapture(device, cq);
EnqueueProgram(cq, program, false);  // targets sub_device_0 with 64 cores
EndTraceCapture(device, cq, trace_id);

device->load_sub_device_manager(different_config);  // changes core set
ReplayTrace(device, cq, trace_id, false);  // BUG: worker count mismatch

// CORRECTED: ensure sub-device config matches trace expectations
device->load_sub_device_manager(original_config);  // restore original config
ReplayTrace(device, cq, trace_id, false);
```

**Prevention:**
- Do not change sub-device configurations between trace capture and replay.
- Validate that the active sub-device configuration matches the trace's `TraceWorkerDescriptor` before replay.
- Use trace replay only for deterministic, repeating workloads where the device state at the start of each replay is identical to the state at the start of capture.
- Do not interleave trace replay with non-traced operations on the same command queue without a full `Finish()` between them.

---

## 4.3.3 Config Buffer Sync Count Drift During Trace Replay

**Symptom:** After one or more trace replays, the dispatch kernel hangs at a `CQ_DISPATCH_CMD_WAIT` that is part of the config buffer synchronization. The wait count does not match the stream register value, and the difference grows with each replay iteration.

**Root Cause:** The `WorkerConfigBufferMgr` tracks a `sync_count` that correlates with the number of worker completions. During trace capture, the sync counts embedded in `CQ_DISPATCH_CMD_WAIT` commands reflect the state at capture time. During replay, the host resets the expected completion counts to 0 (via `reset_host_dispatch_state_for_trace`) and reconfigures the config buffer manager:

```cpp
// dispatch.cpp, reset_host_dispatch_state_for_trace()
for (uint32_t i = 0; i < num_sub_devices; ++i) {
    config_buffer_mgr[i].mark_completely_full(expected_num_workers_completed[i]);
}
```

The `mark_completely_full()` call marks the entire config buffer as occupied, with a sync count equal to the post-trace expected completion count. The next `reserve` call will always require a sync, because the ring buffer appears full:

```cpp
// In worker_config_buffer.cpp, mark_completely_full()
auto& free_entry = this->entries_[kNewFreeIndex][idx];
free_entry.addr = this->base_addrs_[idx];
free_entry.size = this->end_addrs_[idx] - this->base_addrs_[idx];  // entire buffer
free_entry.sync_count = sync;

auto& alloc_entry = this->entries_[kNewAllocIndex][idx];
alloc_entry.addr = this->end_addrs_[idx];  // forces wrap and allocation failure
```

The drift occurs when:
1. The trace was captured with a different starting sync count than the replay uses.
2. After multiple replays, the accumulated completions on the device do not match the host's expectation because `update_worker_state_post_trace_execution()` does not correctly account for all completions.
3. The stream register clear (`CQ_DISPATCH_CMD_WAIT_FLAG_CLEAR_STREAM`) in the replay setup does not fully reset the counter due to a race with workers completing.

**Diagnosis Steps:**
1. Read the stream register value and compare against the wait count in the `CQ_DISPATCH_CMD_WAIT` command. The difference shows how many "phantom" completions or missing completions exist.
2. Track sync counts across replay iterations: log `expected_num_workers_completed` before and after each replay. If the values drift, identify where the accounting goes wrong.
3. Check `update_worker_state_post_trace_execution()`: verify its logic matches the actual replay behavior.
4. Verify that the `CLEAR_STREAM` flag is present in the trace replay's wait commands.

**Fix:** If sync counts drift, this is typically a bug in the host-side state management. Ensure that `reset_host_dispatch_state_for_trace()` and `load_host_dispatch_state()` are exactly inverse operations. As a workaround, call `Synchronize()` between replay iterations to force a clean state.

**Prevention:**
- After implementing trace replay for a new feature, test with at least 100 consecutive replays to verify that sync counts do not drift.
- Always synchronize (`Finish`) between consecutive trace replays.
- Add assertions in `update_worker_state_post_trace_execution()` that verify the post-replay state matches expectations.
- After trace replay, use `Finish` to drain the pipeline before dispatching non-trace programs.

---

## 4.3.4 ExecBuf Stall-After Flag Stuck

**Symptom:** The prefetcher processes a trace replay command but never exits stall mode. Commands enqueued by the host after the trace are never processed. The device appears completely unresponsive. Eventually the host times out in `fetch_queue_reserve_back` or `completion_queue_wait_front`.

**Root Cause:** The `CQ_PREFETCH_CMD_EXEC_BUF` fetch queue entry is marked with the stall-after flag (MSB set). The prefetcher transitions to `StallState::STALLED` only after the tagged read for this entry is retired:

```cpp
// In cq_prefetch.cpp, fetch_q_get_cmds()
if (inflight[idx].flags == InflightFlags::STALL_AFTER) {
    ASSERT(inflight_count == 0U);
    ASSERT(issue_fence == fence);
    stall_state = StallState::STALLED;
    return;
}
```

The stall is lifted when `CQ_DISPATCH_CMD_EXEC_BUF_END` is processed by the prefetcher:

```cpp
// process_exec_buf_end resets stall_state
stall_state = StallState::NOT_STALLED;
```

A permanent stall occurs when:
1. The `EXEC_BUF_END` command is missing from the trace buffer (incomplete trace capture).
2. The dispatch kernel hangs while processing trace commands (e.g., a worker hang within the trace causes `CQ_DISPATCH_CMD_WAIT` to spin at `PWW`), so `EXEC_BUF_END` is never reached.
3. The `EXEC_BUF_END` command is corrupted in the trace buffer.

```
Host: EnqueueTrace --> writes EXEC_BUF to fetch queue with stall flag set
Prefetch: reads EXEC_BUF, issues DRAM read, enters STALLED
Dispatch: processes exec buf commands...
  --> CQ_DISPATCH_CMD_SEND_GO_SIGNAL (succeeds)
  --> Workers start, one hangs at CB deadlock
  --> CQ_DISPATCH_CMD_WAIT (spins forever at PWW)
  --> EXEC_BUF_END never reached
  --> Prefetch remains STALLED
  --> Host times out in fetch_queue_reserve_back or completion_queue_wait_front
```

There is no graceful way to clear the stall state without a chip reset. The dispatch pipeline is designed with the assumption that exec_buf commands will always complete.

**Diagnosis Steps:**
1. Read `stall_state` from the prefetcher's L1. If it equals `1` (`STALLED`), trace execution is in progress.
2. Check the dispatch kernel's progress to determine if trace commands are being processed. If it is at `PWW` or `WCW`, a worker within the trace is hung.
3. Inspect the end of the trace buffer data for the `CQ_DISPATCH_CMD_EXEC_BUF_END` command.
4. If the dispatch kernel is hung, diagnose the worker hang using [Section 01, Scenario 4.1.6](./01_dispatch_architecture_and_hang_points.md#416-dispatch-kernel-waiting-for-workers-process_wait--go-signal) and Chapters 2-3.

**Fix:** Ensure trace capture always includes the `EXEC_BUF_END` command. If the trace contains a program that hangs, fix the underlying kernel bug and recapture the trace. If recovery is needed:
1. Detect via timeout (`TT_METAL_OPERATION_TIMEOUT_SECONDS`).
2. Close the device (which may itself hang -- see [Scenario 4.2.5](./02_host_synchronization_and_timeout_detection.md#425-devicemanagerclose_devices-with-skip_synchronize)).
3. Perform a chip reset.

**Prevention:**
- Validate trace buffers after capture by checking for the `EXEC_BUF_END` terminator.
- Test all programs individually before capturing them into a trace.
- Use `TT_METAL_OPERATION_TIMEOUT_SECONDS` to detect trace-replay hangs.

---

## 4.3.5 Program Cache Stale Entries After Trace Replay

**Symptom:** A program that was previously cached and ran successfully now causes a hang when re-executed. Workers execute a stale kernel binary from L1 that references memory locations that now belong to a different program. Symptoms vary: incorrect computation, NOC address violations, spin-loop hangs in the stale kernel, or L1 corruption causing secondary hangs.

**Root Cause:** The program cache stores compiled kernel binaries and their L1 addresses so they do not need to be recompiled and re-uploaded for each execution. However, cached L1 addresses become stale when:

1. The L1 region was overwritten by a trace replay (which writes to the same config buffer space).
2. The L1 region was reclaimed by the `WorkerConfigBufferMgr` due to a sync that freed old entries.
3. A `clear_loaded_sub_device_manager` call invalidated the dispatch state without updating the cache.
4. Another program's config buffer slot was allocated at overlapping L1 addresses.

After trace replay, `mark_completely_full` is called, which effectively marks all config buffer space as reclaimable. The next non-trace program dispatch may allocate config buffer space that overlaps with the cached program's binaries. When the cached program is dispatched, the dispatch commands reference the stale L1 addresses from the original compilation.

**Diagnosis Steps:**
1. Check if the program was dispatched from cache (no binary writes in the command stream; the program cache hit counter should be incremented).
2. Read the L1 address where the program binary should reside and verify it contains the expected data. If the data is wrong, check if a trace replay or config buffer reallocation overwrote it.
3. Disable the program cache and verify the hang disappears: call `disable_and_clear_program_cache()` or set the appropriate flag.
4. Check the program's binary status via the Inspector -- `program_set_binary_status` tracks whether binaries are valid.

**Fix:**
```cpp
// BUGGY: relying on cached program after state change
EnqueueProgram(cq, program, false);
Finish(device);

// Sub-device reconfiguration invalidates L1 layout
device->load_sub_device_manager(new_manager);

// Re-execute: cache hit, but L1 addresses are stale
EnqueueProgram(cq, program, false);  // May hang

// CORRECTED: invalidate cache after reconfiguration
EnqueueProgram(cq, program, false);
Finish(device);

device->load_sub_device_manager(new_manager);

// Force recompilation by clearing the program cache or using a new Program object
device->clear_program_cache();
EnqueueProgram(cq, program, false);
```

**Prevention:**
- The program cache should be invalidated whenever sub-device configuration changes or the L1 allocation layout is modified.
- When debugging unexpected hangs on re-execution, always try disabling the program cache first to rule out stale entries.
- Be aware that any operation that changes the L1 allocation layout (buffer creation/deletion, sub-device changes, trace replay) can invalidate cached programs.
- In debug builds, validate that cached program L1 addresses do not overlap with current allocations.

---

## 4.3.6 Trace Replay with Mismatched Device Configuration

**Symptom:** A trace captured on one device configuration fails when replayed on a different configuration. The failure may be a NOC address violation (if core coordinates differ), a missing completion (if the number of workers differs), or a dispatch hang (if dispatch core locations differ). Workers may be stuck waiting for go signals addressed to a different sub-device index, or the dispatch kernel's expected worker count does not match the actual configuration.

**Root Cause:** Trace replay bypasses the normal program dispatch path and directly replays the raw dispatch commands. These commands contain device-specific information encoded at capture time:
- **Absolute NOC coordinates** for worker cores (in `CQ_DISPATCH_CMD_WRITE_PACKED` sub-commands)
- **Core counts** embedded in go signal commands (`num_unicast_txns`, `num_worker_cores_to_mcast`)
- **Buffer addresses** that depend on the DRAM bank count and L1 size
- **Dispatch core coordinates** in relay and synchronization commands

The `issue_trace_commands` function computes wait counts based on the trace descriptor:

```cpp
// dispatch.cpp, issue_trace_commands()
uint32_t expected_num_workers = expected_num_workers_completed[index];
if (desc.num_traced_programs_needing_go_signal_multicast) {
    expected_num_workers += device->num_worker_cores(HalProgrammableCoreType::TENSIX, id);
}
if (desc.num_traced_programs_needing_go_signal_unicast) {
    expected_num_workers += device->num_virtual_eth_cores(id);
}
```

If `device->num_worker_cores()` returns a different count than what was present during capture, the wait will either under-count (hang waiting for signals that will never arrive) or over-count (hang waiting for non-existent signals).

**Diagnosis Steps:**
1. Compare the device configuration at capture time vs. replay time: core grid, DRAM banks, dispatch core type, sub-device layout, harvested rows.
2. Compare the `TraceWorkerDescriptor` values with the current device's core counts. Check `num_worker_cores` and `num_virtual_eth_cores` for each sub-device.
3. Decode the trace buffer commands and check NOC coordinates and core counts against the current configuration.
4. Check if `SET_WRITE_OFFSET` commands in the trace data match the current device's write offsets.

**Fix:** Re-capture the trace on the target device configuration. Do not attempt to transfer traces between different device types or sub-device configurations.

```cpp
// BUGGY: capture and replay with different sub-device configs
device->load_sub_device_manager(config_a);
auto trace_id = BeginTraceCapture(device, cq_id);
// ... capture programs ...
EndTraceCapture(device, cq_id, trace_id);

device->load_sub_device_manager(config_b);  // Different config!
ReplayTrace(device, cq_id, trace_id);  // Mismatch!

// CORRECTED: replay with same config as capture
device->load_sub_device_manager(config_a);
ReplayTrace(device, cq_id, trace_id);
```

**Prevention:**
- Associate trace buffers with the device configuration used during capture.
- Validate device configuration consistency before replay.
- Store the sub-device manager ID and device descriptor metadata alongside the trace for validation at replay time.
- Consider using `write_offset` relocation for address-independent traces, though this only handles a subset of address changes.

---

## 4.3.7 Repeated Trace Replay Drift

**Symptom:** A trace works correctly on the first replay but hangs on subsequent replays. The hang typically manifests as a worker count mismatch or config buffer exhaustion. The dispatch kernel is at `PWW` or `WCW`, waiting for a completion count that does not match the stream register value.

**Root Cause:** Each trace replay updates the host-side state via `update_worker_state_post_trace_execution`:

```cpp
// In dispatch.cpp
expected_num_workers_completed[index] = desc.num_completion_worker_cores;
worker_launch_message_buffer_state[index].set_mcast_wptr(
    desc.num_traced_programs_needing_go_signal_multicast);
config_buffer_mgr[index].mark_completely_full(expected_num_workers_completed[index]);
```

If the trace is replayed multiple times without proper state reset, the cumulative state can drift:
1. `expected_num_workers_completed` is set to a fixed value after each replay, but the dispatch kernel's stream register continues to accumulate. After N replays, the stream register value is N * completions, but the host expects only 1 * completions.
2. The `CLEAR_STREAM` flag in the `CQ_DISPATCH_CMD_WAIT` should reset the stream register, but if the clear does not execute (e.g., because the wait was satisfied with an old value), the register accumulates.
3. Mixing trace replay with non-trace program dispatch between replays compounds the issue, as non-trace completions add to the stream register count without the trace's reset logic accounting for them.

**Diagnosis Steps:**
1. Read the dispatch stream register value and compare with the expected completion count.
2. Check if the `CLEAR_STREAM` flag is present in the trace replay's wait commands.
3. Verify that `reset_host_dispatch_state_for_trace` is called before each replay to save and restore state.
4. Track `expected_num_workers_completed` values across replay iterations -- if they drift, the accounting is wrong.

**Fix:** Ensure `reset_host_dispatch_state_for_trace` and `load_host_dispatch_state` are properly paired around each trace replay. Call `Finish` between trace replays to fully drain the pipeline:

```cpp
// BUGGY: repeated replay without synchronization
for (int i = 0; i < 100; i++) {
    ReplayTrace(device, cq_id, trace_id, false);
    // No sync: stream registers accumulate across replays
}

// CORRECTED: synchronize between replays
for (int i = 0; i < 100; i++) {
    ReplayTrace(device, cq_id, trace_id, false);
    Finish(cq);  // Drain pipeline, ensure consistent state
}
```

**Prevention:**
- Always synchronize (`Finish`) between consecutive trace replays.
- Monitor stream register values across replays for drift.
- The framework should handle this automatically; if it does not, file a bug.
- After implementing trace replay for a new feature, test with at least 100 consecutive replays to verify state consistency.

---

## 4.3.8 Trace Buffer DRAM Read Failure

**Symptom:** The prefetcher hangs during ExecBuf mode, stuck at a NOC read barrier waiting for trace data from DRAM. The trace buffer in DRAM may have been corrupted, deallocated, or the DRAM read address is invalid.

**Root Cause:** During trace replay, the prefetcher reads pages from the DRAM trace buffer using `noc_async_read`:

```cpp
// In cq_prefetch.cpp, exec_buf path
noc_async_read(noc_addr, scratch_read_addr, amt_read);
noc_async_read_barrier();
```

The DRAM address is computed based on the trace buffer's base address, page size, and bank interleaving. The `TraceBuffer` holds a `std::shared_ptr<Buffer>` that points to the DRAM allocation. If the trace buffer is released (explicitly or through the Buffer destructor) and the DRAM region is reallocated for another purpose, the prefetcher reads stale or corrupted data from the new occupant. The DRAM read uses transaction ID 1 (`trid = 1`), separate from the host PCIe reads (which use TRIDs 2+). If the DRAM read stalls (e.g., due to an invalid address or DRAM controller error), the prefetcher blocks at the read barrier.

**Diagnosis Steps:**
1. Check the prefetcher waypoint -- if it is at a NOC read barrier (`NRBW`), it is waiting for a DRAM read.
2. Read the trace buffer address from the `CQ_PREFETCH_CMD_EXEC_BUF` command parameters.
3. Verify that the trace buffer is still allocated at the expected DRAM address.
4. Use `TraceBuffer::validate()` to compare the DRAM buffer contents against the original `TraceDescriptor::data`:
   ```cpp
   void TraceBuffer::validate() {
       std::vector<uint32_t> backdoor_data;
       detail::ReadFromBuffer(this->buffer, backdoor_data);
       if (backdoor_data != this->desc->data) {
           log_error(LogMetalTrace, "Trace buffer expected: ...");
           log_error(LogMetalTrace, "Trace buffer observed: ...");
       }
   }
   ```
5. Check DRAM controller status for errors.

**Fix:** Ensure the trace buffer remains allocated and valid throughout the replay. Do not deallocate `TraceBuffer::buffer` while replays may still reference it:

```cpp
// BUGGY: Trace buffer freed before replay
auto trace_id = BeginTraceCapture(device, cq_id);
// ... enqueue programs ...
EndTraceCapture(device, cq_id, trace_id);

some_buffer_cleanup();  // May free trace DRAM

ReplayTrace(device, cq_id, trace_id);  // HANG: reads garbage from DRAM

// CORRECTED: Ensure trace buffer persists until replay is complete
ReplayTrace(device, cq_id, trace_id);
Finish(device, cq_id);  // Wait for replay to complete
ReleaseTrace(device, trace_id);  // Now safe to free
```

**Prevention:**
- Hold a shared pointer to the `TraceBuffer` for the duration of all replays.
- Never release a trace buffer while replays are pending.
- Use `TraceBuffer::validate()` in debug builds to verify DRAM buffer integrity before replay.

---

## 4.3.9 LightMetal Capture Missing Operations Leading to Replay Hang

LightMetal is a higher-level capture/replay system that records entire Metal API call sequences (not dispatch commands) using FlatBuffers serialization. `LightMetalCaptureContext` serializes each API call (`CreateBuffer`, `EnqueueProgram`, `Finish`, etc.) into a FlatBuffer `Command`, preserving object identity via `global_id` maps (Buffer, Program, Kernel, CBHandle each get a unique ID). `LightMetalReplayImpl` reconstructs the object graph from the resulting `LightMetalBinary` and re-executes the captured commands in order.

LightMetal replay enables **deterministic reproduction** of hang-causing sequences: capture the API call sequence leading to a hang, replay it on a different device (same type) with additional instrumentation (watcher, DPRINT, address sanitization), and bisect the sequence to find the minimum reproduction case. Enable capture via `LightMetalCaptureContext::get().set_tracing(true)` before any device interaction, save with `create_light_metal_binary().save_to_file()`, and replay with `LightMetalReplayImpl::run()`. If the hang occurs before `Finish()` returns, use `TT_METAL_OPERATION_TIMEOUT_SECONDS` to force termination; the partial binary may still be useful.

### Hang Scenario

**Symptom:** A LightMetal replay hangs at a point that corresponds to a valid execution in the original capture. The replay device is in a different state than expected because an API call was not captured.

**Root Cause:** The LightMetal capture system relies on instrumentation at each Metal API entry point. If a new API is added or an existing API path is not instrumented, the corresponding operation is missing from the capture. During replay, the device state diverges from what the subsequent captured commands expect.

For example, if a `SetRuntimeArgs` call is not captured, the replay will dispatch a program with stale runtime arguments. If those arguments include buffer addresses, the program may access incorrect memory and hang. Similarly, if a `CreateBuffer` call is missed (e.g., because capture was enabled mid-execution), the replay's `get_buffer_from_map(global_id)` will fail with a missing key, or the buffer will have incorrect properties.

Note: The trace-related APIs (`ReplayTraceCommand`, etc.) are deprecated (Issue #24955). LightMetal replay currently logs these commands but does not execute them:

```cpp
// In lightmetal_replay_impl.cpp
void LightMetalReplayImpl::execute(const tt::tt_metal::flatbuffer::ReplayTraceCommand* cmd) {
    log_debug(LogLightMetal, "LightMetalReplay(ReplayTrace) cq_id: {} tid: {} blocking: {}",
              cmd->cq_id(), cmd->tid(), cmd->blocking());
    // ReplayTrace(this->device_, cmd->cq_id(), cmd->tid(), cmd->blocking());  // DISABLED
}
```

**Diagnosis Steps:**
1. Compare the LightMetal binary's command list with the expected API call sequence. Look for gaps where a buffer creation, kernel configuration, or runtime arg update is missing.
2. Check if the replay device's buffer allocations match the capture's expectations (global ID to address mapping).
3. Add `LightMetalCompareCommand` checkpoints to validate buffer contents at specific points. If outputs diverge, there is likely a capture gap.
4. Verify that capture was enabled before any device API calls.

**Fix:** Ensure all relevant API calls are instrumented in the capture path. Report missing instrumentation as a LightMetal infrastructure bug. Enable capture from the very beginning of the application, before any device interaction.

**Prevention:**
- Enable LightMetal capture before creating any objects that will be referenced later.
- Validate LightMetal binaries by comparing replay output with original execution output using `LightMetalCompareCommand`.
- Test LightMetal replay as part of CI to ensure binaries remain valid across code changes.
- Design LightMetal captures to be self-contained: all buffers created, initialized, and used within the capture.

---

## 4.3.10 LightMetal Replay Device Configuration Mismatch

**Symptom:** A LightMetal binary captured on one device fails to replay correctly on another device (or the same device after a reset). The replay may hang during buffer creation (address mismatch), program execution (wrong kernel binaries), or synchronization (event ID mismatch). On a different device type (e.g., captured on Wormhole, replayed on Blackhole), the failure may be a NOC address violation, wrong L1 sizes, or missing completion signals.

**Root Cause:** LightMetal replay reconstructs all API calls, but the underlying device may differ from the capture device in critical ways:

1. **Buffer allocation nondeterminism:** The allocator may assign different addresses on replay, causing programs that reference specific addresses (embedded in RTAs) to use wrong locations. LightMetal captures create fresh device-side objects, so L1 addresses may differ from the original.
2. **Different core grid:** Harvested rows differ between devices, so core coordinates in the captured programs may be invalid.
3. **Different L1/DRAM sizes:** Buffer allocations that succeeded on the original device may fail on the replay device.
4. **Different dispatch core mapping:** The prefetch and dispatch cores may be at different locations.
5. **Kernel compilation nondeterminism:** If kernel compilation depends on runtime state that differs between capture and replay, the kernel binaries may differ.
6. **Device state drift:** If the replay device has different firmware versions, different clock settings, or different initial memory contents, operations may behave differently.

**Diagnosis Steps:**
1. Compare device descriptors: verify that the replay device matches the capture device's architecture, harvesting, and configuration.
2. Compare buffer addresses: during capture, log buffer `global_id` to address mappings. During replay, log the same. Check for divergence.
3. Check for architecture-specific command parameters (e.g., Blackhole inline write workarounds, Wormhole NCRISC IRAM paths).
4. Replay on the same device without reset: if replay works in the same session but not after a reset, the issue is likely initial state (firmware, memory contents).

**Fix:** Replay LightMetal binaries on identical (or compatible) device configurations. Ensure binaries include all necessary state initialization rather than relying on device state from a previous session.

**Prevention:**
- Include device metadata in the LightMetal binary and validate it at replay time before executing any commands.
- Test LightMetal replay on a freshly opened device to ensure the binary is self-contained.
- Note the deprecation of trace APIs (Issue #24955) -- plan migration to alternative replay mechanisms.
- Use the `LightMetalCompareCommand` to add assertions that catch divergence early during replay.

---

## Key Invariants for Safe Trace Replay

1. All buffers referenced by the trace must remain at their capture-time addresses.
2. The sub-device configuration must be identical to the capture-time configuration.
3. The config buffer manager must be reset to empty before trace execution begins.
4. Worker launch message read pointers must be reset to zero via `RUN_MSG_REPLAY_TRACE`.
5. Worker completion stream registers must be cleared before the traced commands execute.
6. The prefetch stall flag must only be set when the trace will execute to completion.
7. The trace DRAM buffer must remain allocated and uncorrupted for the trace's entire lifetime.
8. The program cache must be consistent with the L1 layout at replay time.

## Key Properties of LightMetal for Hang Debugging

1. LightMetal captures the API call sequence, not the dispatch command stream -- it is higher-level and includes buffer contents.
2. Replay creates fresh device-side objects (buffers, programs) from the binary, so L1 addresses may differ from the original. This means LightMetal captures are *not* subject to stale-address issues (unlike trace replay), but they may surface different allocation-dependent bugs.
3. The `LightMetalCompareCommand` can validate that replay produces the same buffer contents as the original, serving as a correctness oracle.

---

## Summary Table

| Scenario | Root Cause | Stalled Component | Severity |
|----------|-----------|-------------------|----------|
| 4.3.1 Stale buffer addresses | L1/DRAM allocation changed between capture/replay | Workers (corrupt) / Dispatch | Critical |
| 4.3.2 Launch msg rptr mismatch | Worker reset race / sub-device mismatch | Dispatch (`PWW`) / Workers | High |
| 4.3.3 Config buffer sync drift | Stream register accumulation across replays | Dispatch (`PWW`/`WCW`) | High |
| 4.3.4 ExecBuf stall stuck | Missing `EXEC_BUF_END` or dispatch hang in trace | Prefetcher (permanently stalled) | Critical |
| 4.3.5 Program cache stale entry | Config buffer reallocation after trace / L1 layout change | Workers (corrupt) | High |
| 4.3.6 Device config mismatch | NOC coords / bank mapping / worker count changed | Various | High |
| 4.3.7 Repeated replay drift | Host state accounting bug across iterations | Dispatch (wait) | Medium |
| 4.3.8 Trace buffer DRAM read failure | Buffer deallocated or corrupted | Prefetcher (read barrier) | Critical |
| 4.3.9 LightMetal missing operations | Uninstrumented API call / capture gap | Replay device (state divergence) | Medium |
| 4.3.10 LightMetal device mismatch | Different device type / harvesting / config | Replay device (various) | Medium |

---

## Cross-Reference to Other Chapters

| This Chapter | Related Section | Connection |
|-------------|----------------|------------|
| 4.3.1 Stale addresses | Ch3 01 L1 corruption | Stale trace addresses cause L1 corruption |
| 4.3.1 Stale addresses | Ch3 04 allocation failures | Allocator state changes invalidate embedded addresses |
| 4.3.2 Launch msg mismatch | Ch2 all sections | Worker kernel hangs cause dispatch wait failures |
| 4.3.4 ExecBuf stall | 4.1.3 Prefetch stall | Same `StallState::STALLED` mechanism |
| 4.3.4 ExecBuf stall | 4.1.6 Worker wait | Worker hang within trace prevents ExecBuf completion |
| 4.3.5 Program cache | Ch3 01 L1 corruption | Stale cached addresses cause L1 writes to wrong locations |
| 4.3.6 Device config | 4.2.8 Sub-device state | Sub-device switch without sync causes trace mismatch |
| 4.3.8 DRAM read failure | Ch3 02 DRAM backpressure | DRAM issues affect trace buffer reads |

---

[Previous: Host Synchronization and Timeout Detection](./02_host_synchronization_and_timeout_detection.md) | [Next: Chapter 5 -- Multi-Chip, CCL, and Fabric Hang Causes](../ch05_multi_chip_and_ccl_hangs/index.md)
