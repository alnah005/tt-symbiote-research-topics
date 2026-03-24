# Differentiating Warm-Up from Production Calls

This file explains how to tell warm-up calls apart from real inference calls, both at the Python source level and in profiling output. By the end you will know how to inspect the `already_warmed_up_prefill` guard to reason about call sequencing, how to read a Tracy ops CSV to identify the compile phase and the trace-replay phase, how `split_compile_and_trace` in `test_utils.py` automates that split, and how Tracy signposts let you filter warm-up rows entirely.

## Python-Level Differentiation

### Zero-Filled Inputs

The most direct signal is the input tensors. Warm-up calls always supply zero-filled token tensors and zero-filled page tables constructed inside `_create_decode_warmup_inputs` or the `warmup_model_prefill` loop. Production calls supply tensors filled with real token IDs and valid page-table block indices. If you add logging or assertions inside `decode_forward` or `prefill_forward_text`, checking `(tokens == 0).all()` is a reliable heuristic for detecting a warm-up call in a controlled test environment.

### The `already_warmed_up_prefill` Boolean

`Generator.__init__` sets `self.already_warmed_up_prefill = False`. The first call to `warmup_model_prefill` sets it to `True` before dispatching any forward pass. You can inspect this attribute at any point to determine whether warm-up has completed:

```python
if not generator.already_warmed_up_prefill:
    # warm-up has not yet run; any prefill call will trigger it
    ...
else:
    # warm-up is done; subsequent calls go straight to inference
    ...
```

> **Note:** There is no `already_warmed_up_decode` counterpart on `WarmupForwardMixin`. Decode warm-up relies on the caller (vLLM engine setup, demo harness) to invoke `warmup_model_decode` exactly once. If you need to guard decode warm-up in new calling code, implement a similar boolean on the subclass.

### Call Ordering

Warm-up always completes before any request is dequeued from the scheduler. In a vLLM deployment `warmup_model_prefill` and `warmup_model_decode` are called during engine initialization, in that order: prefill warm-up runs first to compile and capture traces for every sequence length, then decode warm-up runs to compile and capture traces for every sampling-config variant. The first real `prefill_forward_text` call arrives only after initialization returns. In demo scripts the pattern is explicit: warm-up calls appear in the `Warming up model...` block and inference calls appear inside the `Starting inference...` loop.

## Profiling-Level Differentiation

### The Tracy Ops CSV Structure

When a model is run under the Tracy profiler (`TT_METAL_ENABLE_TRACY=1`), the profiler writes an ops CSV whose filename follows the pattern `ops_perf_results_<date>.csv`. Each row represents one op dispatch. The key columns for warm-up analysis are:

| Column | Warm-up compile rows | Warm-up capture rows | Production replay rows |
|---|---|---|---|
| `OP CODE` | kernel op names | same kernel op names | same kernel op names |
| `OP TYPE` | `tt_dnn_device`, etc. | same | same |
| `METAL TRACE ID` | empty / null | non-null (trace being built) | non-null (trace being replayed) |
| `METAL TRACE REPLAY SESSION ID` | null | null | non-null (replay session index) |

During the compile phase (the first forward pass at each sequence length or sampling config), ops are dispatched directly to the device command queue and `METAL TRACE ID` is null. During the trace-capture phase (between `TT_METAL_TRACE_BEGIN` and `TT_METAL_TRACE_END` in the host log), `process_ops_logs.py` assigns the active trace ID to every op it parses, so capture-phase rows carry a **non-null** `METAL TRACE ID`. During production replay (`TT_METAL_TRACE_REPLAY`), each row also carries a non-null `METAL TRACE ID` — populated from the C++ device perf CSV — and additionally receives a non-null `METAL TRACE REPLAY SESSION ID` that identifies which replay iteration the row belongs to.

> **Key insight:** Both capture-phase rows and replay-phase rows carry a non-null `METAL TRACE ID`. To distinguish them, check `METAL TRACE REPLAY SESSION ID`: null means capture; non-null means replay. Compile-phase rows are everything before the first non-null `METAL TRACE ID` row.

### `split_compile_and_trace` in `test_utils.py`

`split_compile_and_trace` in `models/tt_transformers/tests/test_utils.py` automates the extraction of compile-phase and trace-phase DataFrames from the ops CSV.

```python
def split_compile_and_trace(
    df: pd.DataFrame,
    mode: str = "prefill",
    num_runs: int = 1,
    num_layers: int = None,
) -> tuple:
    ...
```

> **Warning:** The default `num_runs=1` is unsafe for typical use. With `num_runs=1`, `find_repeated_runs` returns `left=0` immediately (any single block trivially satisfies "1 identical block"), so `adjusted_len = len(df)` and both `df_model_compilation` and `df_model_trace` are assigned the entire DataFrame. No error is raised; the caller silently receives two identical full-DataFrame slices and any per-phase comparison produces meaningless results. Always pass `num_runs` explicitly — use `num_runs=2` for the standard one-capture-plus-one-replay case.

`num_runs` is the number of **identical consecutive op-code blocks** that `find_repeated_runs` must locate; see the algorithm section below for the full implementation. For the standard tt-transformers case, always use `num_runs=2`. If your CSV has a different block structure, use `find_repeated_runs` directly and interpret the returned index based on your known CSV layout.

The function works by calling `find_repeated_runs(df["OP CODE"].tolist(), num_runs)` to locate the first row index `first_run_start` such that the suffix of the op-code list starting at that index can be split evenly into `num_runs` identical blocks.

Once `first_run_start` is known:

```python
adjusted_len   = (len(df) - first_run_start) // num_runs
first_run_end  = first_run_start + adjusted_len
last_run_start = len(df) - adjusted_len

df_model_compilation = df[first_run_start : first_run_end]  # first repeating block = trace-capture phase (NOT JIT compile)
df_model_trace       = df[last_run_start :]
```

`df_model_compilation` contains the **first** of the `num_runs` identical repeating blocks — this is the **trace-capture phase**, not the JIT-compile phase. The actual JIT-compile-phase ops occupy the non-repeating prefix that `find_repeated_runs` stripped, and `split_compile_and_trace` does not return that slice. `first_run_start` is a local variable inside the function and is not part of the return tuple.

> **Warning:** If the CSV structure does not match `num_runs` (e.g., the CSV has fewer than `num_runs` repeated blocks), `find_repeated_runs` returns `-1` inside `split_compile_and_trace`, and the index arithmetic produces incorrect slice boundaries silently — the function will not raise. Always verify `num_runs` matches the actual CSV structure before calling.

If you need to measure JIT compilation cost, call `find_repeated_runs` directly on the original DataFrame outside the function:

```python
from models.tt_transformers.tests.test_utils import find_repeated_runs

first_run_start = find_repeated_runs(df["OP CODE"].tolist(), num_runs)
if first_run_start == -1:
    raise ValueError(
        "Could not find repeating blocks; check num_runs matches the CSV structure"
    )
df_jit_compile = df[:first_run_start]   # non-repeating prefix = JIT-compile phase
```

Pass the same `num_runs` value you would pass to `split_compile_and_trace`. The returned `first_run_start` is the boundary between the non-repeating compile prefix and the start of the first identical repeating block (the capture phase). Always check that `first_run_start != -1` before slicing: a `-1` result means `find_repeated_runs` could not locate `num_runs` identical consecutive blocks, which indicates the CSV structure does not match the expected `num_runs` value. Without the guard, `df[:-1]` silently drops the final row of the DataFrame instead of surfacing the mismatch.

`df_model_trace` contains the **last** identical block, which corresponds to the final trace-replay run. Any intermediate identical blocks (e.g., earlier replay iterations when `num_runs > 2`) are discarded because only the first (capture) and last (final replay) blocks are needed for capture-vs-replay comparison.

The function then calls `find_repeated_block` on `df_model_compilation` (the capture-phase DataFrame) to identify the repeating per-layer op pattern, and uses that to split each of `df_model_compilation` and `df_model_trace` into first-layer, mid-layer, and model-tail sub-DataFrames.

`split_compile_and_trace` is called from `models/tt_transformers/tests/test_device_perf.py` to compute per-phase latency breakdowns in automated performance tests.

> **Key insight:** `find_repeated_runs` identifies the split boundary purely from the structure of the op-code sequence. It does not depend on `METAL TRACE ID` being present in the CSV. This makes it usable even with profiler outputs that omit that column.

### `find_repeated_runs` Algorithm

`find_repeated_runs(ops, num_runs)` scans left to right looking for the smallest `left` such that `ops[left:]` divides evenly into `num_runs` identical contiguous blocks:

```python
def find_repeated_runs(ops, num_runs):
    left = 0
    while left < len(ops):
        n = len(ops) - left
        if n % num_runs == 0:
            run_length = n // num_runs
            first = ops[left : left + run_length]
            if all(ops[left + i*run_length : left + (i+1)*run_length] == first
                   for i in range(1, num_runs)):
                return left
        left += 1
    return -1
```

## Tracy Signposts for Filtering Warm-Up Rows

Tracy signposts let you annotate the profiling stream with named markers. When signposts are present in the ops CSV, `post_process_ops_log` in `tools/tracy/process_model_log.py` can filter the DataFrame to only the rows between `start` and `stop` markers:

```python
def post_process_ops_log(output_logs_subdir, columns=None, sum_vals=True, op_name="", has_signposts=False):
    ...
    if has_signposts:
        markers = df[df["OP TYPE"] == "signpost"]["OP CODE"]
        start = markers[markers == "start"].index[0]
        stop  = markers[markers == "stop"].index[0]
        df = df.iloc[start + 1 : stop]
```

`markers[markers == "start"].index[0]` returns the **index label** of the matching row, not a positional integer. `.iloc` takes positional integers. These coincide here because `df` is created immediately above via `pd.read_csv(filename)` with no prior filtering, so it retains a default 0-based `RangeIndex` — index labels equal their positions. If you adapt this pattern in code that pre-filters `df` (e.g., after dropping header rows, merging device CSVs, or applying a boolean mask), the RangeIndex will be disrupted and `.iloc[start + 1 : stop]` will silently slice the wrong rows. In that case replace `.iloc` with `.loc` to perform label-based slicing, taking care to account for their different end-boundary semantics:

- `.iloc` is **end-exclusive**: `df.iloc[start + 1 : stop]` returns rows at positions `start + 1` through `stop - 1`, excluding the row at position `stop` (the `"stop"` signpost row).
- `.loc` with integer labels is **end-inclusive**: `df.loc[start + 1 : stop]` returns all rows whose label falls in the closed interval `[start + 1, stop]`, which **includes** the row whose label equals `stop` — i.e., the `"stop"` signpost row itself.

To exclude the signpost row when using `.loc`, use `df.loc[start + 1 : stop - 1]`. This is the correct label-based equivalent of the original `.iloc[start + 1 : stop]` slice.

Rows with `OP TYPE == "signpost"` carry the marker name in `OP CODE`. Everything before the first `start` marker — which includes all warm-up compile and capture rows — is excluded from the analysis.

### Inserting Signposts

In test or demo code, import `signpost` from `tracy` and bracket the production inference loop:

```python
from tracy import signpost

# ... warm-up completes here ...

signpost("start")   # marks beginning of production measurement window
for step in range(num_decode_steps):
    decode_forward(...)
signpost("stop")    # marks end of production measurement window
```

A representative usage pattern appears in `models/demos/falcon7b_common/tests/run_falcon_end_to_end.py`:

```python
if device_perf:
    from tracy import signpost

...
if device_perf:
    signpost("start")   # start device perf measurement

# ... model forward call(s) ...

if device_perf:
    signpost("stop")    # stop device perf measurement
```

The guard `if device_perf:` (or `if not is_ci_env:` in other tests) keeps signposts out of normal functional runs so that the overhead of the Tracy instrumentation is only incurred when profiling is explicitly requested.

### Passing `has_signposts=True`

Once signposts are present in the profiling artifact, pass `has_signposts=True` to `run_device_perf` or `post_process_ops_log_detailed`:

```python
results = run_device_perf(
    command, subdir, num_iterations, cols, batch_size,
    has_signposts=True,
)
```

This propagates the flag to the underlying `post_process_ops_log` call, which trims the DataFrame to the region between `start` and `stop` before aggregating durations. Without this flag the aggregation includes warm-up rows and overstates the per-step latency.

> **Warning:** Signposts only filter the DataFrame at the Python post-processing layer. They do not change what the Tracy profiler captures; all ops — warm-up and production — are still recorded to the binary trace file. If you need to exclude warm-up from a Tracy timeline visualization you must set the time range manually in the Tracy GUI.

---

**Next:** [Chapter 4 — Tracy Profiling with Trace Capture](../ch4_tracy_profiling/index.md)
