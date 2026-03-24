# Running Tests with Tracy

This file explains how to launch a profiling run using `python3 -m tracy -r`, what environment variables the launcher sets on the test subprocess, and what output artifacts to expect. It also covers `run_device_profiler` — the programmatic wrapper used by model performance tests.

## Full invocation

```bash
python3 -m tracy -r [options] -- pytest <test_file> [pytest_args...]
```

The `-r` flag activates **report mode**. Without `-r`, `python3 -m tracy` runs the target script inside the current process using Tracy's Python instrumentation hooks but does not spawn a capture process and does not generate a CSV report. With `-r`, the launcher:

1. Starts a `tracy-capture` subprocess listening on a free TCP port.
2. Re-launches itself **without** `-r` as the test subprocess, passing the port via `TRACY_PORT`.
3. Waits for the test process to exit.
4. Calls `generate_report` to run `csvexport` and `process_ops`.

Key flags for trace-capture workflows:

| Flag | Effect |
|---|---|
| `-r` | Report mode: spawn capture process and generate CSV |
| `--no-device` | Omit device-side data from the report |
| `--device-trace-profiler` | Set `TT_METAL_TRACE_PROFILER=1` (see below) |
| `--no-op-info-cache` | Set `TT_METAL_PROFILER_NO_CACHE_OP_INFO=1` |
| `-o <dir>` | Override the profiler artifacts folder |
| `-n <suffix>` | Append a suffix to the report file name |

## Environment variables set by `-r` mode

When `-r` is active, `tools/tracy/__main__.py` sets the following variables on the test subprocess environment before launching it:

### `TTNN_OP_PROFILER=1`

Enables the TTNN-level op profiler. This causes TTNN to emit Tracy zone events for every dispatched op, providing the op-level timing rows that appear in `tracy_ops_times.csv`.

### `TT_METAL_DEVICE_PROFILER=1`

Enables the device-side profiler. Without this variable the device firmware does not write profiling timestamps to DRAM and no device kernel durations appear in the final CSV. This variable controls the `profiler_enabled` flag in `rtoptions`.

### `TT_METAL_PROFILER_TRACE_TRACKING=1`

Activates the `profiler_trace_tracking` field in `rtoptions` (set in `tt_metal/llrt/rtoptions.cpp`). When this is set, the `TracyTTMetalTraceTrackingEnabled()` predicate in `tt_metal/tools/profiler/tt_metal_tracy.hpp` returns `true`, which causes the following C++ macros to emit Tracy messages:

- `TracyTTMetalBeginMeshTrace` emits `` `TT_METAL_TRACE_BEGIN: {device_id}, {trace_id}` ``
- `TracyTTMetalReplayMeshTrace` emits `` `TT_METAL_TRACE_REPLAY: {device_id}, {trace_id}` ``
- `TracyTTMetalEnqueueMeshWorkloadTrace` emits `` `TT_METAL_TRACE_ENQUEUE_PROGRAM: {device_id}, {trace_id}, {runtime_id}` ``

> **Note:** `TracyTTMetalEndMeshTrace` emits its `TT_METAL_TRACE_END` message unconditionally — it does **not** require `TT_METAL_PROFILER_TRACE_TRACKING=1`; it is gated only on `TT_METAL_DEVICE_PROFILER=1`. See `tracy_markers_for_trace.md` for the full C++ macro details.

These messages are what `import_tracy_op_logs` in `process_ops_logs.py` reads to stamp `metal_trace_id` on every op that was dispatched while a trace was open. Without `TT_METAL_PROFILER_TRACE_TRACKING=1`, the `METAL TRACE ID` column in the final CSV will be empty for all rows.

> **Note:** `TT_METAL_PROFILER_TRACE_TRACKING=1` is only effective when `TT_METAL_DEVICE_PROFILER=1` is also set. The `rtoptions` parser checks `this->profiler_enabled` before activating trace tracking (see `rtoptions.cpp` line ~784).

## `TT_METAL_TRACE_PROFILER=1` — device-trace profiler mode

Set via the `--device-trace-profiler` command-line flag:

```bash
python3 -m tracy -r --device-trace-profiler -- pytest <test_file>
```

This sets `TT_METAL_TRACE_PROFILER=1` in the environment, which activates `profiler_trace_profiler` (exposed as `get_profiler_trace_only()`) in `rtoptions`. In this mode the device profiler does not collect per-op kernel durations during trace replay; instead it measures the **total duration of each trace replay as a single unit**. This is the most accurate way to measure real decode-step latency because it avoids the per-op DRAM write overhead that would otherwise perturb timing.

> **Warning:** `TT_METAL_TRACE_PROFILER=1` requires `TT_METAL_DEVICE_PROFILER=1` to be set first; the `rtoptions` parser only enables it if `profiler_enabled` is already true.

## Output artifacts

After a successful `-r` run the following files appear under the profiler artifacts directory (default: `generated/profiler/`):

| File | Location | Contents |
|---|---|---|
| `tracy_profile_log_host.tracy` | `logs/` subdirectory | Raw Tracy binary capture; viewable in the Tracy GUI |
| `tracy_ops_times.csv` | `logs/` subdirectory | Per-op host-side timing from `csvexport -u -p TT_` |
| `tracy_ops_data.csv` | `logs/` subdirectory | Tracy message log from `csvexport -m -s ";"` |
| `ops_perf_results_<date>.csv` | `reports/<date>/` subdirectory | Final merged per-op performance table |

The date-stamped report path is constructed in `process_model_log.py`:

```python
filename = output_report_dir / runDate / f"ops_perf_results_{runDate}.csv"
```

## `run_device_profiler` — programmatic equivalent

Model performance tests (such as `test_device_perf.py` in `models/tt_transformers/tests/`) do not call `python3 -m tracy -r` directly. They use the `run_device_profiler` helper defined in `models/perf/device_perf_utils.py` (imported from `tools/tracy/process_model_log.py`):

```python
def run_device_profiler(
    command,
    output_logs_subdir,
    check_test_return_code=True,
    device_analysis_types=[],
    python_post_process=True,
    ...
):
    ...
    python_post_process_opt = ""
    if python_post_process:
        python_post_process_opt = "-r"
    profiler_cmd = (
        f"python3 -m tracy -p {python_post_process_opt} "
        f"-o {output_profiler_dir} {check_return_code} "
        f"{device_analysis_opt} ... -t 5000 -m {command}"
    )
    subprocess.run([profiler_cmd], shell=True, check=True)
```

`run_device_profiler` assembles the `python3 -m tracy` invocation from its arguments and calls it as a shell command. The command **always** passes `-p` (partial-profiling mode, which restricts Tracy to enabled zones only). When `python_post_process=True` (the default), it additionally passes `-r` (report mode), making the full prefix `python3 -m tracy -p -r`. These are two separate flags: `-p` activates partial-zone profiling; `-r` spawns the capture process and generates the CSV report. Tests call `post_process_ops_log` afterwards to read the resulting CSV and assert on kernel durations.

---

**Next:** [`tracy_markers_for_trace.md`](./tracy_markers_for_trace.md)
