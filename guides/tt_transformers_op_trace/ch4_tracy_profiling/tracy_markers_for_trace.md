# Tracy Markers for Trace Boundaries

This file explains the C++ macros in `tt_metal/tools/profiler/tt_metal_tracy.hpp` that stamp trace lifecycle events into the Tracy event stream, and how `import_tracy_op_logs` in `tools/tracy/process_ops_logs.py` reads those events to annotate every op with its trace membership.

## C++ macros in `tt_metal_tracy.hpp`

All four macros are conditionally compiled under `#if defined(TRACY_ENABLE)` and are no-ops in non-Tracy builds. They all check `getDeviceProfilerState()` before doing anything, so they are also silent when the device profiler is disabled.

### `TracyTTMetalBeginMeshTrace(device_ids, trace_id)`

Called when the host begins recording a trace. For each device ID in `device_ids`:

1. Calls `profiler_state_manager()->mark_trace_begin(device_id, trace_id)` to update internal state.
2. If `TracyTTMetalTraceTrackingEnabled()` (i.e., `TT_METAL_PROFILER_TRACE_TRACKING=1`), emits the Tracy message:
   ```
   `TT_METAL_TRACE_BEGIN: {device_id}, {trace_id}`
   ```

### `TracyTTMetalEndMeshTrace(device_ids, trace_id)`

Called when trace recording finishes. For each device ID:

1. Calls `mark_trace_end(device_id, trace_id)`.
2. Emits (gated only on `getDeviceProfilerState()`, **not** on `TracyTTMetalTraceTrackingEnabled()`):
   ```
   `TT_METAL_TRACE_END: {device_id}, {trace_id}`
   ```

> **Note:** `TracyTTMetalEndMeshTrace` is confirmed to emit its Tracy message without a `TracyTTMetalTraceTrackingEnabled()` check (verified in `tt_metal/tools/profiler/tt_metal_tracy.hpp`). Unlike `TracyTTMetalBeginMeshTrace` and `TracyTTMetalReplayMeshTrace`, which both wrap their `TracyMessage` call inside `if (TracyTTMetalTraceTrackingEnabled())`, the END macro emits whenever the device profiler is active (`getDeviceProfilerState()` returns true). This means the END message appears in `tracy_ops_data.csv` regardless of the `TT_METAL_PROFILER_TRACE_TRACKING` setting, as long as `TT_METAL_DEVICE_PROFILER=1` is set.

### `TracyTTMetalReplayMeshTrace(device_ids, trace_id)`

Called each time a previously captured trace is replayed. For each device ID:

1. Calls `mark_trace_replay(device_id, trace_id)`.
2. If trace tracking is enabled, emits:
   ```
   `TT_METAL_TRACE_REPLAY: {device_id}, {trace_id}`
   ```

The Tracy timestamp at which this message is emitted becomes the **wall-clock anchor** for all device ops that belong to this replay session. `import_tracy_op_logs` appends this timestamp to `traceReplays[device_id][trace_id]`, and `lookup_trace_replay_timestamp` uses the session index to retrieve it.

### `TracyTTMetalEnqueueMeshWorkloadTrace(mesh_device, mesh_workload, trace_id)`

Called when a program is enqueued into an active trace. For each device in the workload's device range, if `trace_id` has a value and trace tracking is enabled, emits:

```
`TT_METAL_TRACE_ENQUEUE_PROGRAM: {device_id}, {trace_id}, {runtime_id}`
```

This message records which `runtime_id` (program hash) was enqueued into which trace. It is used by `profiler_state_manager()->add_runtime_id_to_trace(...)` on the C++ side to build the mapping from trace IDs to their constituent programs.

## How `import_tracy_op_logs` parses these messages

`import_tracy_op_logs` in `tools/tracy/process_ops_logs.py` reads `tracy_ops_data.csv` row by row. Each row has a `MessageName` field containing the raw Tracy message string and a `total_ns` field containing the Tracy wall-clock timestamp in nanoseconds.

The function maintains two data structures:

```python
traceIDs = {}      # dict[device_id -> current open trace_id | None]
traceReplays = {}  # dict[device_id -> dict[trace_id -> list[replay_timestamps]]]
```

For each row that contains `"TT_METAL"` and `"TRACE"` (but not `TT_METAL_TRACE_ENQUEUE_PROGRAM`, which is handled separately):

- **`TT_METAL_TRACE_BEGIN`**: sets `traceIDs[device_id] = trace_id`.
- **`TT_METAL_TRACE_END`**: asserts that the currently open trace matches, then sets `traceIDs[device_id] = None`.
- **`TT_METAL_TRACE_REPLAY`**: appends `opDataTime` to `traceReplays[device_id][trace_id]`. Each append corresponds to one replay session; the position in the list is the session index (1-based after `lookup_trace_replay_timestamp` adjusts by `index = session_id - 1`).

For every op row (lines containing `"TT_DNN"` or `"TT_METAL"` and `"OP"`), whether uncached or cached, the function sets:

```python
opData["metal_trace_id"] = traceIDs.get(deviceID)  # None if no trace is open
```

This means every op emitted while `traceIDs[device_id]` is non-None gets the current trace ID stamped on it.

> **Key insight:** The `traceIDs` dict tracks open traces in document order as the CSV is scanned. An op is "inside a trace" if and only if the most recent `TT_METAL_TRACE_BEGIN` for that device came before the op row in the CSV, with no intervening `TT_METAL_TRACE_END`.

## `METAL TRACE ID` and `METAL TRACE REPLAY SESSION ID` columns

After `import_tracy_op_logs` returns, downstream processing in `process_ops_logs.py` writes two columns to `ops_perf_results_<date>.csv`:

- **`METAL TRACE ID`**: the `metal_trace_id` value from `opData`; `null`/empty for ops dispatched outside any trace.
- **`METAL TRACE REPLAY SESSION ID`**: populated only from `TT_METAL_TRACE_REPLAY` messages. Session ID 1 corresponds to the **first replay call** (not the capture pass). The capture pass has a non-null `METAL TRACE ID` (set while the `TT_METAL_TRACE_BEGIN` / `TT_METAL_TRACE_END` window is open) but an **empty** `METAL TRACE REPLAY SESSION ID`, because no `TT_METAL_TRACE_REPLAY` message is emitted during capture. Each subsequent replay increments the counter: session 1 = first replay, session 2 = second replay, and so on.

Both columns are defined in the `OPS_CSV_HEADER` list at the top of `process_ops_logs.py` (lines 89–90).

## `TraceReplayDict` and `lookup_trace_replay_timestamp`

```python
TraceReplayDict = Dict[int, Dict[int, List[int]]]
#                      ^          ^        ^
#                 device_id   trace_id  list of replay timestamps (ns)
```

`lookup_trace_replay_timestamp` takes a `(device_id, trace_id, session_id)` triple and returns the Tracy wall-clock timestamp for that replay session:

```python
def lookup_trace_replay_timestamp(
    traceReplays: Optional[TraceReplayDict],
    device_id: int,
    trace_id: Optional[int],
    session_id: Optional[int],
) -> Optional[int]:
    timestamps = traceReplays[device_id][trace_id]
    index = session_id - 1          # session IDs are 1-based
    return timestamps[index]
```

This timestamp is used when correlating device-side op timing with Tracy wall-clock time. During trace replay the device executes ops without issuing individual host notifications; the only host-visible event is the `TT_METAL_TRACE_REPLAY` message. By anchoring all device ops from a given replay session to the timestamp of that message, the post-processor can place them correctly on the Tracy GUI timeline and compute accurate host-to-device latency offsets.

---

**Next:** [`reading_profiling_output.md`](./reading_profiling_output.md)
