# Annotated Ops Report

This file walks through `ops_perf_results_<date>.csv` phase by phase after a Llama decode run with trace and Tracy enabled. By the end you will know how to programmatically isolate warm-up rows, the trace-capture block, and the production-replay block, and how to compute per-decode-step device time from the replay rows.

## 1. Overview of the CSV Structure

`ops_perf_results_<date>.csv` is a merged per-op performance table produced by `process_ops_logs.py`. Each row represents one op dispatch. The columns most relevant to trace analysis are:

| Column | Meaning |
|---|---|
| `OP CODE` | Op name (e.g., `Matmul`, `LayerNorm`, `RMSNorm`) |
| `OP TYPE` | Dispatch category (`tt_dnn_device`, `signpost`, etc.) |
| `METAL TRACE ID` | Integer trace ID if the op was dispatched inside a `TT_METAL_TRACE_BEGIN/END` window; null otherwise |
| `METAL TRACE REPLAY SESSION ID` | 1-based replay counter per `(device_id, trace_id)` pair; null for warm-up/compile rows (no trace open); empty string for capture-phase rows (trace open but no replay yet); ≥ 1 for replay-phase rows |
| `DEVICE KERNEL DURATION [ns]` | Time the kernel occupied the device cores, in nanoseconds |
| `GLOBAL CALL COUNT` | Monotonically increasing integer across all ops in the file; useful for ordering within a replay session |

> See [reading_profiling_output.md](../ch4_tracy_profiling/reading_profiling_output.md) for the full column-by-column description and the replay session anchoring mechanism.

## 2. Step-by-Step: Isolating Warm-Up Rows

Warm-up rows occupy the non-repeating prefix of the op-code list — the part that `find_repeated_runs` strips before it finds the first identical repeating block.

```python
import pandas as pd
from models.tt_transformers.tests.test_utils import find_repeated_runs

df = pd.read_csv("generated/profiler/reports/2025-01-15/ops_perf_results_2025-01-15.csv")

left = find_repeated_runs(df["OP CODE"].tolist(), num_runs=2)
if left == -1:
    raise ValueError(
        "find_repeated_runs could not locate 2 identical blocks. "
        "Check that the CSV contains both a capture pass and at least one replay pass."
    )

df_warmup = df.iloc[:left]   # everything before the first repeating block
```

`left` is the first row index at which the remaining op-code list divides evenly into two identical consecutive blocks. All rows before `left` are warm-up rows; they have `METAL TRACE ID` = null because no trace bracket was open when they were dispatched. See Section 6 for the `left == -1` failure mode.

## 3. Step-by-Step: Identifying the Trace-Capture Block

After stripping the warm-up prefix, `split_compile_and_trace` assigns the **first** of the two identical repeating blocks to `df_model_compilation`. Despite the name, this is not the JIT-compilation phase — kernels are already compiled by the time this block runs. It is the **trace-capture phase**: the forward pass executed inside the `ttnn.begin_trace_capture` / `ttnn.end_trace_capture` bracket.

```python
from models.tt_transformers.tests.test_utils import split_compile_and_trace

(
    df_model_compilation,  # trace-capture phase (first repeating block)
    df_model_trace,        # final production-replay phase (last repeating block)
    df_first_layer_compilation,
    df_first_layer_trace,
    df_mid_layers_compilation,
    df_mid_layers_trace,
    df_model_tail_compilation,
    df_model_tail_trace,
) = split_compile_and_trace(df, mode="decode", num_runs=2)
```

> **Key insight:** `df_model_compilation` is the trace-capture block, not the warm-up or JIT-compile block. The name follows the convention in `test_device_perf.py` where "compilation" means "the first of the two repeated passes", which in the trace workflow happens to be the capture pass. The actual JIT-compile-phase ops are in `df_warmup` above, not in `df_model_compilation`.

To confirm that a block is the capture phase rather than a replay phase, check that `METAL TRACE REPLAY SESSION ID` is empty (empty string or NaN) across those rows — not a numeric session ID:

```python
col = df_model_compilation["METAL TRACE REPLAY SESSION ID"]
assert (col.isna() | (col == "")).all(), (
    "Expected capture-phase rows to have empty-string METAL TRACE REPLAY SESSION ID (not a replay session number)"
)
```

If you need to slice the capture block using signposts instead of the heuristic, you can bracket the capture call in your test code:

```python
from tracy import signpost

signpost("capture_start")
generator._capture_decode_trace_text(...)
signpost("capture_stop")
```

Then filter by label in post-processing:

```python
markers = df[df["OP TYPE"] == "signpost"]["OP CODE"]
capture_start = markers[markers == "capture_start"].index[0]
capture_stop  = markers[markers == "capture_stop"].index[0]
df_capture = df.loc[capture_start + 1 : capture_stop - 1]
```

Note the use of `.loc` with `stop - 1` to exclude the signpost row itself; `.loc` with integer labels is end-inclusive, so omitting the `- 1` would include the `capture_stop` signpost row in the slice.

> See [differentiating_warmup_from_production.md](../ch3_warmup/differentiating_warmup_from_production.md) for the full `.iloc` vs. `.loc` safety discussion.

## 4. Step-by-Step: Identifying the Production-Replay Block

`df_model_trace` is the last of the two identical repeating blocks. Its rows were dispatched during `ttnn.execute_trace` calls, so `METAL TRACE REPLAY SESSION ID` is non-null (>= 1) for every row.

```python
# Verify that all replay rows carry a non-null session ID
assert df_model_trace["METAL TRACE REPLAY SESSION ID"].notna().all(), (
    "Expected all replay rows to have a non-null METAL TRACE REPLAY SESSION ID"
)

# Compute total device time for one decode step (one full replay pass)
total_device_ns = df_model_trace["DEVICE KERNEL DURATION [ns]"].sum()
total_device_us = total_device_ns / 1_000
print(f"Decode step device time: {total_device_us:.1f} us")
```

If the run produced multiple replay iterations (e.g., `num_runs=3` for two replays), `df_model_trace` contains only the final replay because `split_compile_and_trace` always takes the last block. For per-iteration analysis across all replays, group by session ID directly on the full DataFrame:

```python
trace_id = df["METAL TRACE ID"].dropna().iloc[0]   # the single trace ID used for decode
df_replays = df[
    (df["METAL TRACE ID"] == trace_id) &
    (df["METAL TRACE REPLAY SESSION ID"].notna())
]

per_session = (
    df_replays
    .groupby("METAL TRACE REPLAY SESSION ID")["DEVICE KERNEL DURATION [ns]"]
    .sum()
    .rename("total_kernel_ns")
)
print(per_session)
```

> See [differentiating_trace_ops_from_normal_ops.md](../ch4_tracy_profiling/differentiating_trace_ops_from_normal_ops.md) for details on `GLOBAL CALL COUNT`-based ordering within a replay session.

## 5. Annotated Example Table

The table below shows representative rows from a hypothetical Llama-3.1-8B decode profiling run (two-layer excerpt). Row groups are annotated with their phase.

| Row | METAL TRACE ID | METAL TRACE REPLAY SESSION ID | OP CODE | DEVICE KERNEL DURATION [ns] | Phase |
|---|---|---|---|---|---|
| 1 | _(null)_ | _(null)_ | `Embedding` | 48200 | warm-up (null trace ID) |
| 2 | _(null)_ | _(null)_ | `RMSNorm` | 12400 | warm-up (null trace ID) |
| 3 | _(null)_ | _(null)_ | `Matmul` | 95300 | warm-up (null trace ID) |
| 4 | _(null)_ | _(null)_ | `ScaledDotProductAttentionDecode` | 180500 | warm-up (null trace ID) |
| 5 | _(null)_ | _(null)_ | `Matmul` | 91800 | warm-up (null trace ID) |
| 6 | _(null)_ | _(null)_ | `RMSNorm` | 12100 | warm-up (null trace ID) |
| — | — | — | _(more warm-up rows: all 32 layers)_ | — | warm-up |
| 200 | `1` | _(empty)_ | `RMSNorm` | 12350 | capture (trace ID set, session ID empty string — not null) |
| 201 | `1` | _(empty)_ | `Matmul` | 94200 | capture (trace ID set, session ID empty string — not null) |
| 202 | `1` | _(empty)_ | `ScaledDotProductAttentionDecode` | 179100 | capture (trace ID set, session ID empty string — not null) |
| 203 | `1` | _(empty)_ | `Matmul` | 90500 | capture (trace ID set, session ID empty string — not null) |
| 204 | `1` | _(empty)_ | `RMSNorm` | 12200 | capture (trace ID set, session ID empty string — not null) |
| — | — | — | _(more capture rows: all 32 layers)_ | — | capture |

> **Note:** _(empty)_ means the `METAL TRACE REPLAY SESSION ID` column is present but contains an empty string — it is **not** null/NaN. This distinguishes capture-phase rows (trace open, no replay yet) from warm-up rows (which carry a true null because no trace was open at all).
| 400 | `1` | `1` | `RMSNorm` | 11800 | replay session 1 (first production token) |
| 401 | `1` | `1` | `Matmul` | 88600 | replay session 1 |
| 402 | `1` | `1` | `ScaledDotProductAttentionDecode` | 171200 | replay session 1 |
| 403 | `1` | `1` | `Matmul` | 87900 | replay session 1 |
| 404 | `1` | `1` | `RMSNorm` | 11700 | replay session 1 |

Key observations from the table:

- Warm-up rows (rows 1–199): `METAL TRACE ID` is null throughout. Kernel durations include compilation overhead and are higher than steady-state.
- Capture rows (rows 200–399): `METAL TRACE ID` = `1`, `METAL TRACE REPLAY SESSION ID` = empty string (not null — the column is present but empty because no replay has occurred yet). Kernel durations are valid but reflect single-dispatch overhead from the recording path, not peak replay throughput.
- Replay rows (rows 400+): `METAL TRACE ID` = `1`, `METAL TRACE REPLAY SESSION ID` = `1` (incrementing with each token). Kernel durations are consistently lower than capture rows for the same op because the device reads directly from the pre-encoded command buffer.

> **Key insight:** The same `Matmul` op shows 94 200 ns in the capture row and 88 600 ns in the replay row — a roughly 6% difference. This gap is caused by the per-op dispatch overhead during the capture pass that replay eliminates. For latency-critical comparisons, always use replay rows.

## 6. Common Pitfalls

### `num_runs=1` returns two identical full-DataFrame copies silently

Passing `num_runs=1` to `split_compile_and_trace` or `find_repeated_runs` causes the function to return `left=0` immediately (any single full block trivially satisfies "1 identical block"). Both `df_model_compilation` and `df_model_trace` are then assigned the entire DataFrame. No error or warning is raised. Any per-phase latency comparison will give identical results for "compile" and "trace", making it appear as if there is no performance difference. Always pass `num_runs=2` for the standard one-capture-plus-one-replay case.

### `.loc` end-inclusivity

`.loc` with integer labels is end-inclusive. If you use `.loc[start + 1 : stop]` to exclude signpost rows, the `stop` row (the `"stop"` signpost itself) is included in the slice. Use `.loc[start + 1 : stop - 1]` to exclude it. `.iloc` is end-exclusive and does not have this issue, but is only safe on a DataFrame with an unbroken default `RangeIndex`. See [reading_profiling_output.md](../ch4_tracy_profiling/reading_profiling_output.md) for the full safety note.

### Not checking `left != -1` before slicing

`find_repeated_runs` returns `-1` when it cannot locate `num_runs` identical consecutive blocks. If you use the return value directly as a slice bound without checking:

```python
# Dangerous: if left == -1, this silently drops the final row
df_warmup = df.iloc[:left]
```

The slice `df.iloc[:-1]` drops the last row of the DataFrame without any indication that something went wrong. Always guard:

```python
left = find_repeated_runs(df["OP CODE"].tolist(), num_runs=2)
if left == -1:
    raise ValueError("CSV structure does not match num_runs=2; verify the run produced both a capture pass and at least one replay.")
df_warmup = df.iloc[:left]
```

---

**Previous:** [running_the_example.md](./running_the_example.md)

**Next:** [Guide index](../index.md)
