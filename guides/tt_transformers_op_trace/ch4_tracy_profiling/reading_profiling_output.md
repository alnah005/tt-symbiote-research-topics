# Reading Profiling Output

This file explains how to interpret `ops_perf_results_<date>.csv` when a trace-capture workflow was profiled, covering the meaning of null vs. non-null `METAL TRACE ID`, the replay session numbering, and the helper functions that automate the compile/trace split.

## Structure of `ops_perf_results_<date>.csv` with trace active

A typical Llama decode profiling run produces three consecutive phases of ops in the CSV:

### Phase 1 — compile (warm-up) run

Rows where `METAL TRACE ID` is empty or null.

These rows represent ops dispatched during the warm-up pass before any trace was captured. At this point, kernels are being JIT-compiled and dispatch overhead is high. Device kernel durations reflect compilation overhead rather than steady-state performance, so these rows are excluded from any production latency measurement.

> **Note:** Some ops near the top of the CSV may also be null-trace rows from framework initialization (device open, tensor allocations, etc.) that run before the model's first forward pass.

### Phase 2 — trace capture run

Rows where `METAL TRACE ID` is non-null and `METAL TRACE REPLAY SESSION ID` is **empty/null**.

These rows represent ops dispatched during the single pass in which the trace was recorded (`BeginTrace` → model forward → `EndTrace`). The `METAL TRACE ID` column is non-null because these ops were dispatched while a `TT_METAL_TRACE_BEGIN` / `TT_METAL_TRACE_END` window was open. However, no `TT_METAL_TRACE_REPLAY` message is emitted during capture, so `METAL TRACE REPLAY SESSION ID` is left empty for all capture-pass rows. Kernels are already compiled at this point, so device durations are valid. However, the capture pass is still a single-dispatch run: each op is dispatched individually to the hardware command queue so that the trace can record the sequence. This means dispatch overhead is still present and the durations are slightly higher than replay.

### Phase 3 — production replay iterations

Rows where `METAL TRACE ID` is non-null and `METAL TRACE REPLAY SESSION ID` >= 1.

These rows represent ops that ran as part of a `ReplayTrace` call. Session ID 1 is the first production replay, session 2 the second, and so on. Each session ID maps to one `TT_METAL_TRACE_REPLAY` message recorded in `tracy_ops_data.csv`; `import_tracy_op_logs` appends that message's timestamp to `traceReplays[device_id][trace_id]`, and the list index (`session_id - 1`) selects the correct entry. Replayed ops are the gold standard for measuring decode-step latency: the entire sequence of kernels executes from a pre-compiled, pre-dispatched command buffer with minimal host involvement.

> **Key insight:** When comparing compile vs. trace performance, always use rows with `METAL TRACE REPLAY SESSION ID` >= 1 (replay rows). Capture-pass rows have `METAL TRACE ID` non-null but `METAL TRACE REPLAY SESSION ID` empty; they give valid kernel durations but include trace-recording overhead.

## `split_compile_and_trace` in `test_utils.py`

`models/tt_transformers/tests/test_utils.py` provides `split_compile_and_trace` to automate the phase split:

```python
def split_compile_and_trace(
    df: pd.DataFrame,
    mode: str = "prefill",
    num_runs: int = 1,
    num_layers: int = None,
):
```

Internally it calls `find_repeated_runs(df["OP CODE"].tolist(), num_runs)` to locate the first index in the op-code list where the remaining rows can be evenly divided into `num_runs` identical blocks. This heuristic works because each run of the model emits the same sequence of op codes; the function finds the boundary where repetition begins.

Once the boundary is found:

```python
first_run_end = first_run_start + adjusted_len
last_run_start = len(df) - adjusted_len
df_model_compilation = df[first_run_start:first_run_end]
df_model_trace = df[last_run_start:]
```

`df_model_compilation` is the **first** repeated block and `df_model_trace` is the **last** repeated block (the final production replay). The function also calls `find_repeated_block` to further partition the blocks into per-layer regions if `num_layers` is provided.

In the standard tt-transformers decode workflow, the warm-up (compile) phase includes a sampling-kernel compile pass at the end that emits a distinct op sequence not present in the capture or replay phases. Because the warm-up phase's op-code sequence differs from the post-compile phases, it is the **non-repeating prefix** that `find_repeated_runs` strips. The two post-compile phases — trace capture and trace replay — produce identical op-code sequences and are identified as the two repeated blocks. Accordingly, with `num_runs=2`:

- `df_model_compilation` — the **first** repeated block — contains the **trace-capture phase** rows (not warm-up rows). These are the ops dispatched while the trace was being recorded (`BeginTrace` → model forward → `EndTrace`). Kernels are already compiled at this point; durations are valid but include single-dispatch overhead from the trace-recording path.
- `df_model_trace` — the **last** repeated block — contains the final **trace replay** rows, which are the gold standard for decode-step latency.

The warm-up rows (null `METAL TRACE ID`) occupy the non-repeating prefix before `first_run_start` and are not returned by `split_compile_and_trace`. To recover them, call `find_repeated_runs` directly on the original DataFrame and slice `df[:first_run_start]` as documented in Chapter 3's `differentiating_warmup_from_production.md`.

Example call from `test_device_perf.py` (standard decode case — one capture pass plus one replay):

```python
(
    df_model_compilation, df_model_trace,
    ...
) = split_compile_and_trace(df, mode="decode", num_runs=2)
```

## `post_process_ops_log` with `has_signposts=True`

For workflows that need precise start/stop points rather than relying on the `find_repeated_runs` heuristic, `post_process_ops_log` in `process_model_log.py` supports signpost-based filtering:

```python
def post_process_ops_log(output_logs_subdir, columns=None, sum_vals=True, op_name="", has_signposts=False):
    ...
    if has_signposts:
        markers = df[df["OP TYPE"] == "signpost"]["OP CODE"]
        start = markers[markers == "start"].index[0]
        stop = markers[markers == "stop"].index[0]
        df = df.iloc[start + 1 : stop]
```

When `has_signposts=True`, the function finds rows whose `OP TYPE` is `"signpost"` and `OP CODE` is `"start"` or `"stop"`, then slices the DataFrame to include only the rows between them.

> **`.iloc` vs. `.loc` safety note:** `.iloc` requires **positional** integers, while `.index[0]` returns the row's **label** (DataFrame index value). These are identical only when `df` has an unbroken default `RangeIndex` (`0, 1, 2, …`). `post_process_ops_log` operates on a DataFrame read fresh from `pd.read_csv` — before any filtering — so its RangeIndex is always unbroken and `.iloc` is safe here. If you copy this pattern for a **pre-filtered** DataFrame (one that has been subset before reaching this code), the label and position will differ. In that case use `.loc[start + 1 : stop - 1]` instead: `.loc` is label-based and end-inclusive, so `stop - 1` is needed to exclude the signpost row itself. This is the same pattern documented in Chapter 3's `differentiating_warmup_from_production.md`.

To insert these markers in your test, call `ttnn.tracy_message` before and after the production loop:

```python
ttnn.tracy_message("TT_SIGNPOST: start")
for _ in range(num_iterations):
    output = model.decode_forward(...)
ttnn.tracy_message("TT_SIGNPOST: stop")
```

`import_tracy_op_logs` recognises any message containing `"TT_SIGNPOST"` and records it as a signpost entry in the `signposts` dict (keyed by `sp_1`, `sp_2`, …). The post-processor later writes these as rows with `OP TYPE = "signpost"` in the CSV.

## `ttnn.start_tracy_zone` and `ttnn.stop_tracy_zone`

Defined in `ttnn/ttnn/profiler.py`:

```python
def start_tracy_zone(source: str, functName: str, lineNum: int, color: int = 0):
    ttnn._ttnn.profiler.start_tracy_zone(source, functName, lineNum, color)

def stop_tracy_zone(name: str = "", color: int = 0):
    return ttnn._ttnn.profiler.stop_tracy_zone(name, color)
```

These functions create named Tracy zones that appear as colored spans in the Tracy GUI timeline. Use them to bracket Python regions you want to annotate — for example, the tokenizer call or KV-cache update — without affecting the op-level profiling data in the CSV.

```python
ttnn.start_tracy_zone(__file__, "kv_cache_update", 42)
update_kv_cache(...)
ttnn.stop_tracy_zone("kv_cache_update")
```

## `tracy_frame()` — frame delimiters for decode steps

```python
def tracy_frame():
    ttnn._ttnn.profiler.tracy_frame()
```

`tracy_frame()` emits a Tracy frame marker. In the Tracy GUI, frame markers are shown as vertical lines that divide the timeline into discrete steps. For autoregressive decode, call `tracy_frame()` once per token:

```python
for token_idx in range(max_tokens):
    output = model.decode_forward(tokens, ...)
    ttnn.tracy_frame()
```

This makes it easy to click on any individual decode step in the GUI and inspect the ops and durations for that token position without manually hunting for the boundaries in the timeline.

---

**Next:** [`differentiating_trace_ops_from_normal_ops.md`](./differentiating_trace_ops_from_normal_ops.md)
