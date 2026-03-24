# Differentiating Trace Ops from Normal Ops

This file explains how to tell apart normally dispatched ops from trace-captured or trace-replayed ops in the `ops_perf_results_<date>.csv` output, and what the practical differences in timing semantics are between the two dispatch modes.

## `METAL TRACE ID` as the primary signal

The `METAL TRACE ID` column is the definitive indicator:

- **Null / empty**: the op was dispatched through the normal host command queue. Each op was individually submitted, executed, and profiled by the device firmware.
- **Non-null integer**: the op was dispatched as part of a trace, either during capture (`METAL TRACE REPLAY SESSION ID` is **empty**) or during replay (`METAL TRACE REPLAY SESSION ID` >= 1). The first replay iteration has session ID = 1; subsequent replays have session ID >= 2.

The value in `METAL TRACE ID` is the integer trace ID passed to `BeginTrace` / `EndTrace` / `ReplayTrace`. If a model uses multiple traces (for example, one for prefill and one for decode), each trace gets a distinct ID and you can filter the CSV to a specific trace.

## Normal dispatch vs. trace replay: timing semantics

### Normal dispatch

Each normally dispatched op has:

- An individual `DEVICE KERNEL DURATION [ns]` measured by reading the device firmware's start and end timestamp counters from DRAM after the op completes.
- An `OP TO OP LATENCY [ns]` that includes host-side dispatch overhead for the next op: kernel compilation check, CQ write, command buffer management.
- A `HOST DURATION [ns]` that reflects the full round-trip including synchronization.

Normal dispatch timing is useful for identifying which individual ops are slow, but the aggregate sum overestimates production latency because it includes dispatch overhead between every op.

### Trace replay

Replayed ops (session ID >= 1) have a fundamentally different timing model:

- The device executes the entire sequence without waiting for per-op host commands. The firmware timestamps for each kernel are still recorded in DRAM and uploaded after the replay completes.
- `DEVICE KERNEL DURATION [ns]` values are valid and accurate per-op measurements.
- However, the **wall-clock start time** of each replayed op is anchored to the `TT_METAL_TRACE_REPLAY` Tracy message timestamp, not to an individual op dispatch event. `lookup_trace_replay_timestamp` supplies this anchor.
- `OP TO OP LATENCY [ns]` during replay reflects the gap between consecutive kernels on the device, not host dispatch time. This is typically much smaller than in normal dispatch.

> **Key insight:** Per-op timing within a replay session is available directly from the `DEVICE KERNEL DURATION [ns]` column. `lookup_trace_replay_timestamp` in `process_ops_logs.py` anchors each op row's wall-clock start time to the corresponding `TT_METAL_TRACE_REPLAY` Tracy message, so timestamp data is correctly attached to every row without any manual "elapsed time between messages" arithmetic. Use the `DEVICE KERNEL DURATION [ns]` values directly for per-op analysis within a session. For end-to-end replay duration (including scheduling overhead not captured in per-op rows), use `--device-trace-profiler` mode rather than attempting to measure intervals between `TT_METAL_TRACE_REPLAY` messages — that interval-based approach fails for the final replay session, which has no subsequent message to bound it.

## `TT_METAL_TRACE_PROFILER=1` — total replay duration mode

Set via `--device-trace-profiler` (see `running_with_tracy.md` for the full flag reference). In this mode the device profiler skips per-op kernel timestamps during replay and instead measures each `ReplayTrace` call as a single total duration — the most accurate measurement of decode-step latency because it eliminates per-op DRAM write overhead.

Use this mode for end-to-end throughput measurement. Use the default mode when you need per-op breakdown within a replay iteration.

## Using `GLOBAL CALL COUNT` to reconstruct replay iterations

When per-op rows are present for a replayed trace (default mode without `--device-trace-profiler`), the `GLOBAL CALL COUNT` column increments monotonically across all ops in the CSV. To reconstruct which ops belonged to which replay iteration:

1. Filter to rows where `METAL TRACE ID` matches the trace of interest.
2. Group by `METAL TRACE REPLAY SESSION ID`. Each unique session ID corresponds to one replay iteration.
3. Within each session group, sort by `GLOBAL CALL COUNT` to recover the execution order within that iteration.

The combination of `(METAL TRACE ID, METAL TRACE REPLAY SESSION ID, GLOBAL CALL COUNT)` uniquely identifies every op instance across all iterations.

> **Note:** Ops from different replay sessions have different `METAL TRACE REPLAY SESSION ID` values but the same `OP CODE` sequence. When computing per-layer averages across replay iterations, group by `OP CODE` and session ID, then aggregate over sessions.

## `TT_METAL_PROFILER_NO_CACHE_OP_INFO` and `--no-op-info-cache`

By default, when `python3 -m tracy -r` is running, op metadata (kernel source, attributes, input/output shapes) for identical ops is cached: the first occurrence of an op emits the full JSON metadata, and subsequent occurrences with the same hash emit a compact reference (`TT_METAL OP CACHED: <hash>, <device>, <op_id>`). `import_tracy_op_logs` reconstructs the full metadata for cached ops by looking up the hash in `cached_ops`.

Passing `--no-op-info-cache` sets `TT_METAL_PROFILER_NO_CACHE_OP_INFO=1`, which disables this caching. Every op — including identical ops in every replay iteration — emits its full metadata. The effect in the CSV:

- All rows have populated `COMPUTE KERNEL SOURCE`, `ATTRIBUTES`, `INPUTS`, `OUTPUTS`, and related columns.
- The `tracy_ops_data.csv` intermediate file is much larger.
- Post-processing is slower.

Use `--no-op-info-cache` when you need to verify that op metadata is consistent across replay iterations or when debugging issues where cached metadata appears incorrect. For routine performance measurement, the default cached mode is sufficient.

```bash
# With caching disabled (full metadata for every op, every session):
python3 -m tracy -r --no-op-info-cache -- pytest tests/tt_transformers/test_llama_model.py
```

---

**Next:** [Chapter 5 — Worked Example: Llama Decode with Trace and Tracy](../ch5_worked_example/index.md)
