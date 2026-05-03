# 4.2 Host Synchronization and Timeout Detection

[Previous: Dispatch Architecture and Hang Points](./01_dispatch_architecture_and_hang_points.md) | [Next: Trace Replay and LightMetal](./03_trace_replay_and_lightmetal.md)

---

This section covers the host-side mechanisms for synchronizing with device execution, how timeouts detect and report hangs, and the hang scenarios that arise from incorrect host-device synchronization. The host communicates with the device through two primary paths: the **issue queue** (host to device) and the **completion queue** (device to host). Hangs in this layer typically manifest as a host thread blocking indefinitely, eventually triggering a timeout exception.

The host-side synchronization layer is where software meets hardware: the host waits for the device to signal completion, tracks events, manages device lifecycle, and detects timeouts. Hangs in this layer are distinctive because the device may or may not be truly hung -- the host's *perception* of a hang depends on its timeout configuration, its polling strategy, and whether its own bookkeeping has drifted from the device's actual state.

Reference files: `tt_metal/impl/dispatch/system_memory_manager.cpp`, `tt_metal/impl/device/device_manager.hpp`, `tt_metal/impl/context/metal_context.cpp`, `tt_metal/impl/dispatch/hardware_command_queue.cpp`

## 4.2.0 Synchronization Model Overview

### Synchronize / Finish Semantics

The host uses **event-based synchronization** to wait for device operations to complete:

- **`Finish(command_queue)`**: Blocks until all previously enqueued operations on the specified command queue have completed. Internally, this waits for the last enqueued event to appear in the completion queue.
- **`Synchronize(device)`**: Blocks until all operations across all command queues on the device have completed.

Both ultimately call `SystemMemoryManager::completion_queue_wait_front`, which polls the completion queue write pointer until the dispatch kernel signals that the operation is done.

### Completion Queue Mechanism

The completion queue is a ring buffer in hugepage memory. The dispatch kernel writes completion events via NOC writes to the PCIe endpoint:

```
dispatch kernel                                  host
     |                                             |
     |  1. Process CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST
     |     with is_event == true
     |  2. completion_queue_reserve_back (QRBW)     |
     |  3. Write event data to completion queue     |
     |  4. completion_queue_push_back               |
     |     notify_host_of_completion_queue_write_pointer()
     |  -----------------------------------------> |
     |                                             |  reads completion_wr_ptr
     |                                             |  compares with completion_rd_ptr
     |                                             |  if match: wait (poll)
     |                                             |  if differ: event is available
     |                                             |
     |                   <------------------------ |  sends updated completion_rd_ptr
     |  (frees completion queue space)             |     (TLB write to device L1)
```

The write pointer includes a toggle bit (bit 31) for wrap-around detection. The host reads the write pointer from hugepage memory (backed by a host-visible copy stored at `HOST_COMPLETION_Q_WR_PTR` offset in the CQ region).

### Event Tracking

The `SystemMemoryManager` maintains per-CQ event counters:

- `cq_to_event[cq_id]`: next event ID to assign (monotonically increasing).
- `cq_to_last_completed_event[cq_id]`: last event ID confirmed by the completion queue.

Events use 32-bit wrap-around semantics (`wrap_ge` comparison). The `set_last_completed_event` method includes an assertion that events only increase:

```cpp
TT_ASSERT(wrap_ge(event_id, this->cq_to_last_completed_event[cq_id]),
    "Event ID is expected to increase...");
```

---

## 4.2.1 Completion Queue Wait Timeout

**Symptom:** The host throws `TIMEOUT: device timeout, potential hang detected, the device is unrecoverable`. The host was waiting for a completion event that the device never wrote.

**Root Cause:** `SystemMemoryManager::completion_queue_wait_front` polls the completion queue write pointer. If the write pointer does not advance beyond the read pointer within the configured timeout, the timeout fires:

```cpp
// In system_memory_manager.cpp, completion_queue_wait_front()
auto wait_condition = [&cq_interface, &write_ptr, &write_toggle]() -> bool {
    return cq_interface.completion_fifo_rd_ptr == write_ptr and
           cq_interface.completion_fifo_rd_toggle == write_toggle;
};

auto on_timeout = [&exit_condition]() {
    exit_condition.store(true);
    MetalContext::instance().on_dispatch_timeout_detected();
    TT_THROW("TIMEOUT: device timeout, potential hang detected, the device is unrecoverable");
};
```

The timeout duration is configured by `TT_METAL_OPERATION_TIMEOUT_SECONDS` (default: 0, meaning no timeout). When set to a positive value, the `loop_and_wait_with_timeout` function monitors both elapsed time and dispatch progress.

**Critical detail:** The timeout is progress-aware. It tracks the dispatch kernel's progress counter (number of commands processed). If the dispatch kernel is making progress (processing commands, even if slowly), the timeout resets. The timeout only fires when no progress is detected for the full duration. (See Scenario 4.2.3 for cases where this masking behavior itself becomes a problem.)

**Diagnosis Steps:**
1. Note the timeout message. Check if `on_dispatch_timeout_detected` produced Inspector data or ran a triage command.
2. If Inspector data was serialized, use the Inspector RPC interface to examine the dispatch state, program states, and core waypoints.
3. Read the dispatch progress counter from device L1 -- if it advanced since the last check, the dispatch kernel was making progress (but not completing the specific operation being waited on).
4. Read the completion queue write and read pointers to determine if the dispatch kernel ever wrote the expected completion event.
5. Use the watcher to check all dispatch and worker core waypoints. Walk the dispatch pipeline: the first stalled kernel identifies the root cause level.

**Fix:** The fix depends on the root cause. This timeout is a symptom of any device-side hang. Use the diagnostic information to identify the specific hang scenario from this chapter or Chapters 2-3. If event tracking has drifted, compare `SystemMemoryManager::cq_to_event[cq_id]` against `cq_to_last_completed_event[cq_id]`.

**Prevention:**
- Set `TT_METAL_OPERATION_TIMEOUT_SECONDS` to a value that is generous enough for your workload but will catch true hangs (e.g., 30-120 seconds for most workloads).
- Configure `TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE` to automatically run `tt-triage.py` when a timeout occurs, capturing a full diagnostic snapshot.
- Ensure every code path that enqueues operations has a matching `Finish()` or close path.

---

## 4.2.2 Fetch Queue Wait Timeout

**Symptom:** The host throws `TIMEOUT: device timeout in fetch queue wait, potential hang detected`. The host was trying to enqueue a new command but the prefetcher has not consumed existing fetch queue entries.

**Root Cause:** `SystemMemoryManager::fetch_queue_reserve_back` polls the prefetcher's fetch queue read pointer. If the read pointer does not advance (the prefetcher is stalled), the host blocks:

```cpp
// In system_memory_manager.cpp, fetch_queue_reserve_back()
auto fetch_wait_condition = [&]() -> bool {
    return this->prefetch_q_dev_ptrs[cq_id] == this->prefetch_q_dev_fences[cq_id];
};
auto fetch_on_timeout = []() {
    MetalContext::instance().on_dispatch_timeout_detected();
    TT_THROW("TIMEOUT: device timeout in fetch queue wait, potential hang detected");
};

auto get_dispatch_progress = [&]() -> uint32_t {
    return get_cq_dispatch_progress(this->device_id, cq_id);
};
loop_and_wait_with_timeout(
    fetch_operation_body, fetch_wait_condition, fetch_on_timeout,
    timeout_duration, get_dispatch_progress);
```

Like the completion queue timeout, this also monitors dispatch progress. The fetch queue has a fixed number of entries (`prefetch_q_entries`), and the prefetcher advances the read pointer as it consumes entries. If the entire pipeline is stalled, the prefetcher cannot consume entries and the host cannot enqueue more.

**Diagnosis Steps:**
1. Compare host-side `prefetch_q_dev_ptrs[cq_id]` with device-side `prefetch_q_dev_fences[cq_id]` (the read pointer read from device).
2. Read the prefetcher's actual read pointer from L1 -- if it matches the host's cached value, the prefetcher is truly stalled.
3. Check the prefetcher waypoint to determine why it is stalled (see [Section 01](./01_dispatch_architecture_and_hang_points.md) scenarios 4.1.1-4.1.3).
4. If the dispatch progress counter is advancing but the fetch queue is not being consumed, the prefetch kernel (not dispatch) is the bottleneck.

**Fix:** Resolve the device-side stall that is preventing the prefetcher from consuming commands.

**Prevention:** Same as Scenario 4.2.1 -- set appropriate timeout values and configure automatic triage.

---

## 4.2.3 Progress-Based Timeout Bypass (Slow Progress Masking a Hang)

**Symptom:** A hang occurs but no timeout is triggered, despite `TT_METAL_OPERATION_TIMEOUT_SECONDS` being set. The host appears stuck indefinitely.

**Root Cause:** The `loop_and_wait_with_timeout` function uses a **progress-based timeout** rather than a simple wall-clock timeout. The timeout only fires if the `get_progress()` callback returns the same value for the entire `timeout_duration`:

```cpp
// In system_memory_manager.cpp
auto progress_update_interval = std::chrono::milliseconds(
    tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_progress_update_ms());

while (true) {
    func_body();
    if (!wait_condition()) break;

    if (std::chrono::high_resolution_clock::now() - last_progress_update_time >= progress_update_interval) {
        uint32_t current_progress = get_progress();
        last_progress_update_time = std::chrono::high_resolution_clock::now();
        if (current_progress != last_progress_value) {
            last_progress_value = current_progress;
            last_progress_time = std::chrono::high_resolution_clock::now();  // Reset timeout
        }
    }

    auto elapsed = std::chrono::duration<float>(current_time - last_progress_time).count();
    if (elapsed >= timeout_duration.count()) {
        on_timeout();
        break;
    }
    std::this_thread::yield();
}
```

The `get_progress()` callback reads the dispatch kernel's progress counter (`dev_dispatch_progress_ptr`) from device L1. This counter is incremented every time dispatch processes a command:

```cpp
// In cq_dispatch.cpp, kernel_main()
done = is_d_variant ? process_cmd_d(cmd_ptr, l1_cache) : process_cmd_h(cmd_ptr);
dispatch_progress++;
*get_dispatch_progress_ptr() = dispatch_progress;
```

The hang goes undetected when:
1. The dispatch kernel is processing commands that take longer than the timeout but the progress counter advances between progress checks (each command advances it once, resetting the timer).
2. The progress counter memory location is corrupted and reads as a changing value.
3. The `progress_update_interval` is too large, causing the host to check progress too infrequently.

**Diagnosis Steps:**
1. Lower the progress update interval: temporarily reduce `TT_METAL_DISPATCH_PROGRESS_UPDATE_MS` to detect whether progress is truly stalled.
2. Read the dispatch progress counter manually: compare the value over several seconds. If it changes, dispatch is alive; if constant, dispatch is hung.
3. Check for L1 corruption at the progress counter address: the progress counter is stored at `dev_dispatch_progress_ptr`. If another NOC write has corrupted this location, it may read as a changing value even when dispatch is hung.

**Fix:**
```cpp
// BUGGY: setting timeout=0 to disable, masking real hangs
std::setenv("TT_METAL_OPERATION_TIMEOUT_SECONDS", "0", 1);
// BAD: disables timeout entirely, test will hang forever

// CORRECTED: use a finite timeout appropriate for the workload
std::setenv("TT_METAL_OPERATION_TIMEOUT_SECONDS", "300", 1);
// GOOD: set a generous but finite timeout
```

**Prevention:**
- Never set `TT_METAL_OPERATION_TIMEOUT_SECONDS` to 0 in CI environments.
- If a workload has known long-running commands, set the timeout to at least 2x the expected maximum command duration.
- In addition to the global timeout, implement application-level progress monitoring.

---

## 4.2.4 on_dispatch_timeout_detected: Inspector Integration and Triage

**Symptom:** After a timeout, the `on_dispatch_timeout_detected` callback fires, potentially serializing diagnostic data and running an external command.

**Root Cause:** This is not a hang scenario itself but the timeout response mechanism. When either the completion queue or fetch queue timeout fires, `MetalContext::on_dispatch_timeout_detected()` is called:

```cpp
// In metal_context.cpp
void MetalContext::on_dispatch_timeout_detected() {
    std::lock_guard<std::mutex> lock(dispatch_timeout_detection_mutex_);
    if (!dispatch_timeout_detection_processed_) {
        dispatch_timeout_detection_processed_ = true;
        log_error(tt::LogMetal, "Timeout detected");

        if (rtoptions_.get_serialize_inspector_on_dispatch_timeout()) {
            Inspector::serialize_rpc();
        }

        std::string command = rtoptions_.get_dispatch_timeout_command_to_execute();
        if (!command.empty()) {
            int result = std::system(command.c_str());
        }
    }
}
```

Key behaviors:
1. **Thread-safe:** Uses a mutex and a processed flag to ensure the handler runs at most once, even if multiple timeouts fire simultaneously across different CQs or threads.
2. **Inspector serialization:** If `TT_METAL_SERIALIZE_INSPECTOR_ON_DISPATCH_TIMEOUT` is set, the Inspector RPC data is serialized to disk, preserving the state of all programs, devices, and dispatch cores.
3. **External command:** If `TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE` is set (e.g., `./tools/tt-triage.py`), the command is executed synchronously. This can capture watcher data, NOC state, and other diagnostic artifacts.

**Fix:**

| Environment Variable | Default | Purpose |
|---------------------|---------|---------|
| `TT_METAL_OPERATION_TIMEOUT_SECONDS` | `0` (disabled) | Timeout duration for host-side waits |
| `TT_METAL_DISPATCH_PROGRESS_UPDATE_MS` | `100` | Interval between dispatch progress reads |
| `TT_METAL_SERIALIZE_INSPECTOR_ON_DISPATCH_TIMEOUT` | `false` | Serialize Inspector state on timeout |
| `TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE` | empty | Shell command to execute on timeout |

**Diagnosis Steps:** When a timeout fires:
1. Check the log output for "Timeout detected" and any subsequent Inspector or triage output.
2. If Inspector data was serialized, analyze it for dispatch core states, program binary statuses, and active workloads.
3. If a triage command was executed, examine its output for watcher snapshots, NOC status, and core dumps.

**Prevention:**
```bash
# Recommended environment setup for debugging hangs
export TT_METAL_OPERATION_TIMEOUT_SECONDS=60
export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="./tools/tt-triage.py --save-artifacts"
export TT_METAL_WATCHER=120  # enable watcher with 120ms polling
```

---

## 4.2.5 DeviceManager::close_devices with skip_synchronize

**Symptom:** The application hangs during device teardown because `close_devices` is waiting for the dispatch pipeline to drain, but the device is in a hung state.

**Root Cause:** `DeviceManager::close_devices` orchestrates the orderly shutdown of devices, including terminating dispatch kernels, tearing down fabric firmware, and closing profiler and command queue infrastructure. The `skip_synchronize` parameter is declared but may not be fully effective in all code paths:

```cpp
// In device_manager.cpp
bool DeviceManager::close_devices(const std::vector<IDevice*>& devices,
                                  bool skip_synchronize = false);
```

The teardown process sends `CQ_DISPATCH_CMD_TERMINATE` / `CQ_PREFETCH_CMD_TERMINATE` to the dispatch kernels and waits for them to exit. If the dispatch pipeline is already hung, the terminate commands cannot be delivered or processed.

In multi-device (Galaxy) configurations, devices must be closed in reverse tunnel order -- from the farthest tunneled device to the closest MMIO device. If a remote device is hung, closing closer devices may also hang because the tunnel is shared.

**Diagnosis Steps:**
1. If the application hangs during teardown, check which device is being closed.
2. Check if the dispatch kernels on that device are still running (watcher data).
3. If a device is known to be hung, the only recovery is a board-level reset.

**Fix:**
```cpp
// BUGGY: trying to close a hung device normally
try {
    // ... device operations that hang
} catch (const std::exception& e) {
    // BAD: this will hang again
    device_manager.close_device(device_id);
}

// CORRECTED: use skip_synchronize for hung devices
try {
    // ... device operations that hang
} catch (const std::exception& e) {
    // GOOD: skip synchronization on hung device
    device_manager.close_devices({device}, /*skip_synchronize=*/true);
}
```

**Prevention:**
- Set `TT_METAL_OPERATION_TIMEOUT_SECONDS` so that hangs are detected *before* teardown.
- If a hang is detected, reset the board before attempting `close_devices`.
- In error handlers that catch timeout or device-hang exceptions, always use `skip_synchronize=true`.
- Use RAII wrappers that detect previous exceptions and pass `skip_synchronize=true` to the close path.
- In Galaxy configurations, implement health checks on remote devices before starting workloads.

---

## 4.2.6 Async Operation Ordering Violations

**Symptom:** A hang occurs sporadically on the device after a sequence of asynchronous operations. The hang manifests as workers waiting for data that was never written, or dispatch commands arriving in an unexpected order. The device typically hangs with the dispatch kernel waiting for workers (`PWW`) or with workers stuck at a semaphore/barrier.

**Root Cause:** When using asynchronous dispatch (operations are enqueued without blocking), the host assumes that operations on the same command queue will be processed in FIFO order. However, certain sequences can create implicit dependencies that violate this assumption:

1. **Write-then-read dependency:** A buffer write followed by a program that reads the same buffer. If the write and program are on the same CQ, ordering is guaranteed. If they are on different CQs, the program may execute before the write completes.

2. **Cross-program dependency:** Program A writes a result to a buffer; Program B reads it. If both are on the same CQ, ordering is guaranteed. If on different CQs, explicit synchronization is needed.

3. **Buffer reallocation race:** A buffer is deallocated and reallocated between two programs. If the second program uses the same L1 address, it may read stale data from the first program.

4. **Circular dependency from wrong dispatch order:** Program A depends on Program B's output, but A is dispatched first within the same command queue. Dispatch waits for A's workers, which will never complete because they need B's output.

**Diagnosis Steps:**
1. Check if the workload uses multiple command queues.
2. Identify any cross-CQ data dependencies (buffer reads/writes that span CQs).
3. Check if buffer addresses are being reused across programs without synchronization.
4. Reproduce with a single command queue -- if the hang disappears, the issue is a cross-queue dependency.
5. Add `Finish()` between suspected conflicting operations -- if the hang disappears, identify which pair has the implicit dependency.
6. Enable the watcher and look for workers stuck at data barriers (`NRBW`, `NWBW`).

**Fix:**
```cpp
// BUGGY: cross-CQ dependency without synchronization
EnqueueWriteBuffer(cq0, buffer, data, false);  // Write on CQ0
EnqueueProgram(cq1, program, false);             // Program on CQ1 reads buffer
// BUG: program may execute before write completes

// CORRECTED: synchronize between CQs
EnqueueWriteBuffer(cq0, buffer, data, false);
Finish(cq0);                                      // Wait for write to complete
EnqueueProgram(cq1, program, false);               // Now safe to read
```

For cross-queue dependencies using events (from V3):
```cpp
// BUGGY: No synchronization between queues
EnqueueProgram(cq0, programB, false);  // Produces data
EnqueueProgram(cq1, programA, false);  // Consumes data, may start before B completes

// CORRECTED: Explicit event synchronization
EnqueueProgram(cq0, programB, false);
auto event = EnqueueRecordEvent(cq0);
EnqueueWaitForEvent(cq1, event);
EnqueueProgram(cq1, programA, false);
```

**Prevention:**
- Use a single command queue unless performance requirements demand multiple CQs.
- When using multiple CQs, insert explicit `Finish` or event-based synchronization at dependency boundaries.
- Always enqueue programs in dependency order within a single command queue.
- Design workloads to minimize cross-CQ data sharing.

---

## 4.2.7 Multi-Queue Implicit Dependencies and Deadlocks

**Symptom:** Operations on one command queue appear to block operations on another command queue, even though they should be independent. In the worst case, both CQs are stuck at `PWW` or `WCW`.

**Root Cause:** Although each hardware command queue has its own prefetcher and dispatch kernel pair, there are shared resources that create implicit coupling:

1. **Shared worker cores:** If both CQs dispatch programs to overlapping sets of worker cores, the second program cannot be configured until the first completes. The `CQ_DISPATCH_CMD_WAIT` in one CQ's stream waits for worker completion, but the workers may be running a program from the other CQ.

2. **Mutual dependency:** Queue 0's program waits for a semaphore that queue 1's program will increment, and queue 1's program waits for a semaphore that queue 0's program will increment. Neither can make progress.

3. **Shared DRAM bandwidth:** Both CQs read from the same DRAM banks. Heavy DRAM traffic on one CQ can cause backpressure that stalls the other (see [Chapter 3, Section 02](../ch03_memory_related_hangs/02_dram_and_noc_backpressure.md)).

4. **Shared NOC bandwidth:** Both CQs' dispatch kernels use the same NOC for writing to worker cores. NOC congestion or backpressure can affect both.

**Diagnosis Steps:**
1. Identify which CQ is stalled and which is making progress (or both stalled).
2. Check if worker cores are shared between the two CQs' sub-device configurations.
3. For mutual deadlocks, identify the worker programs on each queue and their data/semaphore dependencies. Determine whether the dependencies form a cycle.
4. Profile DRAM bandwidth usage across both CQs.

**Fix:** Partition worker cores between CQs using sub-device managers:
```cpp
// BUGGY: both CQs target the same workers
EnqueueProgram(cq0, program_a, false);  // Uses all Tensix cores
EnqueueProgram(cq1, program_b, false);  // Also uses all Tensix cores
// BUG: CQ1 cannot configure workers until CQ0's program completes

// CORRECTED: use sub-device managers to partition cores
auto sub_device_0 = SubDevice({CoreRangeSet(CoreRange({0,0},{3,7}))});
auto sub_device_1 = SubDevice({CoreRangeSet(CoreRange({4,0},{7,7}))});
auto mgr_id = device->create_sub_device_manager({sub_device_0, sub_device_1});
device->load_sub_device_manager(mgr_id);
// Now CQ0 dispatches to sub_device_0 and CQ1 dispatches to sub_device_1
```

**Prevention:**
- Use sub-device managers to explicitly partition worker cores between CQs.
- Prefer single-queue dispatch unless the workload genuinely benefits from multi-queue parallelism.
- When using multiple queues, establish a clear dependency hierarchy (e.g., queue 0 is always the "producer" queue). Avoid mutual cross-queue dependencies.

---

## 4.2.8 Sub-Device Manager State Inconsistencies

**Symptom:** A hang occurs after switching sub-device configurations mid-execution. Workers may be stuck waiting for go signals addressed to a different sub-device index, or the dispatch kernel's expected worker count does not match the actual configuration.

**Root Cause:** The `SubDeviceManagerTracker` manages named sub-device configurations that determine which workers are targeted by each dispatch stream:

```cpp
// sub_device_manager_tracker.hpp
SubDeviceManagerId create_sub_device_manager(
    tt::stl::Span<const SubDevice> sub_devices,
    DeviceAddr local_l1_size);
void load_sub_device_manager(SubDeviceManagerId sub_device_manager_id);
void clear_loaded_sub_device_manager();
```

When `load_sub_device_manager` is called, it changes the active sub-device configuration, which affects:
1. The dispatch core's stream-to-worker mapping.
2. The expected worker completion counts.
3. The go signal multicast/unicast destinations.

If a sub-device manager is loaded while programs from the previous configuration are still in flight, the dispatch kernel's state becomes inconsistent with the actual worker state. This can cause:
- Go signals sent to cores not running the expected program (dispatch waits forever for completion).
- Kernel config written to cores not included in the new sub-device (potential L1 corruption).

The dispatch kernel tracks worker completion via stream registers indexed by sub-device:
```cpp
// Stream index for worker completion per sub-device
constexpr uint32_t first_stream_used = FIRST_STREAM_USED;
uint32_t dispatch_s_sync_sem_addr = dispatch_s_sync_sem_base_addr + sync_index * L1_ALIGNMENT;
```

**Diagnosis Steps:**
1. Check if `load_sub_device_manager` was called without first calling `Finish` or `Synchronize`.
2. Read the dispatch stream register values and compare them with the expected worker counts for the active sub-device configuration.
3. Check if workers from the previous sub-device configuration are still running.
4. Compare the active sub-device manager's core set with the program's expected core set.

**Fix:**
```cpp
// BUGGY: switching sub-device config without synchronization
device->load_sub_device_manager(config_a);
EnqueueProgram(cq, program_for_config_a, false);
device->load_sub_device_manager(config_b);  // BUG: config_a programs may still be running
EnqueueProgram(cq, program_for_config_b, false);

// CORRECTED: synchronize before switching
device->load_sub_device_manager(config_a);
EnqueueProgram(cq, program_for_config_a, false);
Finish(cq);  // Ensure all config_a programs complete
device->load_sub_device_manager(config_b);
EnqueueProgram(cq, program_for_config_b, false);
```

Also ensure programs are dispatched under the correct sub-device:
```cpp
// BUGGY: Program compiled for sub-device A, dispatched under sub-device B
auto sdm_a = device->create_sub_device_manager({sub_device_a}, l1_size);
device->load_sub_device_manager(sdm_a);
auto program = create_program_for_sub_device_a();

auto sdm_b = device->create_sub_device_manager({sub_device_b}, l1_size);
device->load_sub_device_manager(sdm_b);
EnqueueProgram(cq, program, false);  // HANG: program targets sub_device_a cores

// CORRECTED: Ensure correct sub-device is loaded before dispatch
device->load_sub_device_manager(sdm_a);
EnqueueProgram(cq, program, false);
```

**Prevention:**
- Always call `Finish` or `Synchronize` before switching sub-device configurations.
- Minimize sub-device configuration changes during execution.
- If possible, create all needed sub-device configurations upfront and only switch between them at natural synchronization points.
- Always verify the active sub-device manager before dispatching a program.

---

## 4.2.9 Event ID Wrap-Around and Stale Events

**Symptom:** The host reports "Event ID is expected to increase" assertion failure, or a `Synchronize` call returns immediately despite programs still being in flight.

**Root Cause:** Event IDs are 32-bit integers that wrap around. The `wrap_ge` comparison function uses signed difference to handle wrap-around:

```cpp
bool wrap_ge(uint32_t a, uint32_t b) {
    int32_t diff = a - b;
    return diff >= 0;
}
```

This works correctly as long as `a` and `b` are within 2^31 of each other. The assertion in `set_last_completed_event` catches violations:

```cpp
TT_ASSERT(wrap_ge(event_id, this->cq_to_last_completed_event[cq_id]),
    "Event ID is expected to increase. Wrapping not supported for sync. "
    "Completed event {} but last recorded completed event is {}",
    event_id, this->cq_to_last_completed_event[cq_id]);
```

Scenarios that trigger this:
1. An event is reported complete out of order (should not happen with a single CQ, but possible with bugs or completion queue corruption).
2. The event counter was reset (via `reset_event_id`) while the device still had pending events with the old counter values.
3. `set_current_and_last_completed_event` is called with inconsistent values (this method is intended for state restoration, e.g., after trace replay, and both CQs must be idle when it is called).
4. A bug in the completion processing reads an old completion queue entry and reports a stale event ID.

**Diagnosis Steps:**
1. Read the current event ID and last completed event ID from the `SystemMemoryManager`.
2. Compare with the completion queue data to determine if events were received out of order.
3. Check if `reset_event_id` was called at an inappropriate time.
4. Check for memory corruption in the completion queue region -- an out-of-bounds NOC write from a worker core may be overwriting completion data (see [Chapter 3, Section 01](../ch03_memory_related_hangs/01_l1_memory_corruption_and_overflow.md)).

**Fix:** Ensure event ID management is consistent with the device state. Do not reset event counters while operations are in flight.

**Prevention:**
- Do not call `reset_event_id` or `set_current_and_last_completed_event` unless all CQs on the device are fully drained.
- Use `Finish` on all CQs before any event counter manipulation.
- Enable NOC address sanitization to catch writes to the completion queue region.
- For very long-running workloads, periodically call `Synchronize()` and reset event counters.

---

## 4.2.10 Host Thread Starvation in Completion Polling

**Symptom:** The host appears to make no progress even though the device is completing operations. Timeouts may fire despite device-side progress. In multi-device configurations, host CPU usage is at 100%.

**Root Cause:** The `loop_and_wait_with_timeout` function calls `std::this_thread::yield()` between polls to avoid busy-waiting:

```cpp
// In system_memory_manager.cpp
while (true) {
    func_body();
    if (!wait_condition()) break;
    // ...
    std::this_thread::yield();
}
```

If the host system is under heavy load (many threads, CPU oversubscription), the polling thread may not be scheduled frequently enough to detect completion events promptly. In extreme cases, the thread may not run for longer than the timeout duration, causing a false timeout.

Additionally, the dispatch progress check reads from device L1 over PCIe, which adds latency. The progress update interval (`dispatch_progress_update_ms`, default 100ms) limits how often these reads occur.

On systems with many devices (e.g., a Galaxy configuration with 32 chips), each device has a polling thread that busy-waits. This can exhaust CPU resources, starving threads responsible for posting new commands -- preventing device progress.

**Diagnosis Steps:**
1. Check host CPU utilization -- if all cores are at 100%, thread scheduling is the bottleneck.
2. Check if the timeout fires but the device-side state shows normal operation.
3. Increase `TT_METAL_OPERATION_TIMEOUT_SECONDS` and verify if the workload eventually completes.
4. Check if the completion queue is not being consumed despite data being available -- the consumer thread may be starved.

**Fix:** Reduce host CPU contention, or increase the timeout duration. Use CPU affinity settings to ensure polling threads and worker threads are on separate cores:
- For Galaxy/multi-device configurations, tune the `worker_thread_to_cpu_core_map_` and `completion_queue_reader_to_cpu_core_map_` settings in `DeviceManager`.

**Prevention:**
- Do not oversubscribe the host CPU when running TT-Metal workloads.
- Set the timeout duration conservatively (at least 2x the expected worst-case completion time).
- Use `TT_METAL_DISPATCH_PROGRESS_UPDATE_MS` to balance between responsiveness and PCIe read overhead.
- In multi-device configurations, explicitly map threads to CPU cores using the provided configuration options.

---

## Summary Table

| Scenario | Error / Behavior | Stalled Component | Root Cause Category | Severity |
|----------|-----------------|-------------------|---------------------|----------|
| 4.2.1 Completion timeout | `TIMEOUT: device timeout, potential hang detected` | Host (waiting) | Device-side hang | Critical |
| 4.2.2 Fetch queue timeout | `TIMEOUT: device timeout in fetch queue wait` | Host (enqueueing) | Prefetcher stall | Critical |
| 4.2.3 Progress-based timeout bypass | No timeout despite hang | Host (stuck) | Slow command or corrupted progress counter | High |
| 4.2.4 Timeout callback | Inspector/triage output | N/A (diagnostic) | N/A | Informational |
| 4.2.5 Device close hang | Teardown blocks | Host (closing) | Dispatch pipeline hung | High |
| 4.2.6 Async ordering violation | Data-dependent hang | Workers | Cross-CQ dependency | High |
| 4.2.7 Multi-queue deadlock | Both CQs stuck at `PWW` | Dispatch/Workers | Mutual cross-queue dependency | High |
| 4.2.8 Sub-device state mismatch | Wrong worker count | Dispatch | Config switch without sync | High |
| 4.2.9 Event wrap-around | Assertion failure | Host | Event counter inconsistency | Medium |
| 4.2.10 Host thread starvation | False timeout | Host | CPU oversubscription | Low |

---

[Previous: Dispatch Architecture and Hang Points](./01_dispatch_architecture_and_hang_points.md) | [Next: Trace Replay and LightMetal](./03_trace_replay_and_lightmetal.md)
