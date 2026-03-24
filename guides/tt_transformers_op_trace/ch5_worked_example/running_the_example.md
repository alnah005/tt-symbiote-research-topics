# Running the Example

This file walks through everything needed to launch a Llama model under Tracy profiling with trace capture enabled: the environment variables to set, the `run_device_profiler` invocation, the sequence of phases you will observe in stdout, and where to find the output files after the run completes.

## 1. Prerequisites

Three environment variables must be in place before the test process starts. `python3 -m tracy -r` sets them automatically on the test subprocess; if you are running the profiler programmatically via `run_device_profiler`, they are set for you by the same mechanism. This section explains what each one does so that you can diagnose missing-column problems when they occur.

| Variable | What it enables |
|---|---|
| `TT_METAL_PROFILER_TRACE_TRACKING=1` | Causes the C++ macros `TracyTTMetalBeginMeshTrace`, `TracyTTMetalReplayMeshTrace`, and `TracyTTMetalEnqueueMeshWorkloadTrace` to emit Tracy messages; without this, the `METAL TRACE ID` column is empty for every row in the CSV. |
| `TTNN_OP_PROFILER=1` | Enables per-op profiling hooks in TTNN so that every dispatched op produces a timing row in `tracy_ops_times.csv`; without this there are no per-op rows to merge. |
| `TT_METAL_DEVICE_PROFILER=1` | Activates device-side firmware timestamp collection; without this, `DEVICE KERNEL DURATION [ns]` is blank for all rows. |

> **Note:** `TT_METAL_PROFILER_TRACE_TRACKING=1` has no effect unless `TT_METAL_DEVICE_PROFILER=1` is also set. The `rtoptions` parser checks `profiler_enabled` before activating trace tracking. See [running_with_tracy.md](../ch4_tracy_profiling/running_with_tracy.md) for the full environment variable reference.

Two device parameters must also be set correctly before the mesh device is opened:

- `num_command_queues: 1` — trace capture requires a single hardware command queue; two-queue mode is incompatible with `ttnn.begin_trace_capture`.
- `trace_region_size` — a DRAM reservation for the command buffer. Use `get_supported_trace_region_size` from `models/tt_transformers/tt/trace_region_config.py` to look up the correct size for your (model, device) pair. If this is too small the run will fail at capture time with an out-of-memory error.

## 2. Launching the Model

The standard entry point for model performance tests is `run_device_profiler` in `models/perf/device_perf_utils.py`. Below is a representative call that profiles a Llama-3.1-8B decode run on an N300 board with trace enabled:

```python
from models.perf.device_perf_utils import run_device_profiler

command = (
    "pytest models/tt_transformers/tests/test_device_perf.py"
    " -k test_llama_decode_perf"
    " --timeout=600"
)

run_device_profiler(
    command=command,
    output_logs_subdir="llama_decode_trace",
    python_post_process=True,   # passes -r; generates the CSV report
)
```

`run_device_profiler` assembles the following shell command internally:

```bash
python3 -m tracy -p -r \
    -o generated/profiler \
    -t 5000 \
    -m pytest models/tt_transformers/tests/test_device_perf.py \
       -k test_llama_decode_perf --timeout=600
```

The `-p` flag restricts Tracy instrumentation to explicitly enabled zones (partial-profiling mode). The `-r` flag spawns the capture subprocess and triggers CSV generation after the test exits. The `--tracy` flag, when present in a test's pytest parametrize list, causes the test to run with `enable_trace=True`; check the specific test's parameters to confirm.

If you prefer to invoke Tracy directly from the command line for a quick interactive run:

```bash
TT_METAL_PROFILER_TRACE_TRACKING=1 \
TTNN_OP_PROFILER=1 \
TT_METAL_DEVICE_PROFILER=1 \
python3 -m tracy -r -- pytest \
    models/tt_transformers/tests/test_llama_model.py \
    -k "test_llama_model_inference and trace" \
    --timeout=600
```

## 3. What Happens During the Run

The run produces four consecutive phases. The table below summarises what you will observe in stdout and in the resulting CSV for each phase.

| Phase | Stdout markers | `METAL TRACE ID` | `METAL TRACE REPLAY SESSION ID` |
|---|---|---|---|
| Compile run (warm-up, non-traced) | `Starting decode warmup` / `Warming up prefill for sequence length: …` | null | null |
| Trace-capture pass | `Done Capturing Decode Trace` (or prefill equivalent) | non-null integer | empty string (not null — distinct from the null in warm-up rows) |
| Production replay passes | First real output token logged | non-null integer (same as capture) | 1, 2, 3, … |
| Framework teardown | Device close, deallocation messages | null | null |

### Phase 1 — Compile run (warm-up)

`WarmupForwardMixin.warmup_model_prefill` sweeps every supported sequence length, issuing one ordinary forward pass per length to compile kernels and warm the program cache; its log lines appear first. Then `warmup_model_decode` issues one forward pass per sampling-config variant for the same purpose. You will see log lines such as:

```
Warming up prefill for sequence length: 128
Warming up prefill for sequence length: 512
...
Starting decode warmup
Decode warmup completed
```

During this phase every op is dispatched normally (no trace bracket is open), so every row in the CSV has `METAL TRACE ID` = null. The `TT_METAL_TRACE_BEGIN` Tracy message has not yet been emitted.

### Phase 2 — Trace-capture pass

After warm-up completes, the `Generator` calls `_capture_decode_trace_text` (for decode) or `_capture_trace_prefill` (for prefill). The sequence is:

1. A compile run (identical to warm-up, but called here to ensure any remaining compilation is done before the bracket opens).
2. `ttnn.begin_trace_capture` — the C++ macro `TracyTTMetalBeginMeshTrace` emits `` `TT_METAL_TRACE_BEGIN: {device_id}, {trace_id}` `` into the Tracy stream. From this point on, `import_tracy_op_logs` will stamp every subsequent op row with the active `trace_id` in `METAL TRACE ID`.
3. A full forward pass. All ops dispatched here are recorded into the command buffer.
4. `ttnn.end_trace_capture` — the `TT_METAL_TRACE_END` message is emitted.

Stdout shows:
```
Done Compiling Model
Done Capturing Decode Trace
```

In the CSV, these rows have `METAL TRACE ID` = (e.g.) `1` and `METAL TRACE REPLAY SESSION ID` = empty string (not null — the column is present but empty because no replay has occurred yet). No `TT_METAL_TRACE_REPLAY` message has been emitted yet, so no session ID is assigned.

> **Key insight:** The trace-capture pass dispatches each op individually to the hardware command queue — so it can record the sequence — not as a replayed batch. Device kernel durations in these rows are valid but include single-dispatch overhead. For production latency measurement, use replay rows only.

### Phase 3 — Production replay passes

Each subsequent call to `decode_forward` (or `prefill_forward_text` for a traceable sequence length) calls `ttnn.execute_trace`. At this point:

1. `TracyTTMetalReplayMeshTrace` emits `` `TT_METAL_TRACE_REPLAY: {device_id}, {trace_id}` `` before the replay is dispatched.
2. The device executes the pre-encoded command buffer without any host re-encoding.
3. `import_tracy_op_logs` reads the `TT_METAL_TRACE_REPLAY` message and increments the session counter for this `(device_id, trace_id)` pair. The first replay produces `METAL TRACE REPLAY SESSION ID = 1`, the second `= 2`, and so on.

Each production token therefore adds one full block of rows to the CSV, all with the same non-null `METAL TRACE ID` and a monotonically increasing `METAL TRACE REPLAY SESSION ID`.

> **See [reading_profiling_output.md](../ch4_tracy_profiling/reading_profiling_output.md) for details on how `lookup_trace_replay_timestamp` anchors each row's wall-clock start time to the correct `TT_METAL_TRACE_REPLAY` message.**

## 4. Locating the Output Files

After the run, the profiler artifacts directory has the following structure (using the default `generated/profiler/` root):

```
generated/profiler/
├── logs/
│   ├── tracy_profile_log_host.tracy    # Raw Tracy binary; open in Tracy GUI
│   ├── tracy_ops_times.csv             # Per-op host-side timing (csvexport -u -p TT_)
│   └── tracy_ops_data.csv              # Tracy message log (csvexport -m -s ";")
└── reports/
    └── <YYYY-MM-DD>/
        └── ops_perf_results_<YYYY-MM-DD>.csv   # Final merged per-op performance table
```

The file you will work with for analysis is:

```
generated/profiler/reports/<date>/ops_perf_results_<date>.csv
```

If you passed `-o <dir>` or supplied `output_logs_subdir` to `run_device_profiler`, the root changes accordingly. The `reports/<date>/` structure inside that root is always the same.

> **Note:** If `OpsPerCore*.csv` files are mentioned in older tooling or documentation, they refer to an earlier per-device CSV produced before the final `ops_perf_results_<date>.csv` merge step. The `ops_perf_results_<date>.csv` in `reports/` is the authoritative output for analysis.

---

**Previous:** [Chapter 5 index](./index.md)

**Next:** [annotated_ops_report.md](./annotated_ops_report.md)
