# Plan: TT Transformers Trace Capture Guide

## Audience

**Target reader:** ML engineers who work with the `tt_transformers` model stack on top of `ttnn` and Tenstorrent hardware. They already write and run inference loops using the `Generator` class, understand the distinction between prefill and decode phases, and have seen `enable_trace=True` in demo test parameters without a clear picture of what it does internally. They are comfortable with Python and have at least skimmed TTNN tensor and op APIs.

**What they already know:**
- The `Generator` class in `models/tt_transformers/tt/generator.py` and its `prefill_forward_text` / `decode_forward` entry points
- The PREFILL / DECODE mode split and the role of KV cache
- That `simple_text_demo.py` passes `trace_region_size` as a device parameter and `enable_trace` as a test parameter
- Basic `ttnn` tensor operations and the concept of on-device tensors
- That Tracy is a profiling tool that can produce per-op timing reports

**What they do not yet know:**
- What `ttnn.begin_trace_capture` / `ttnn.end_trace_capture` / `ttnn.execute_trace` actually record and how replay works
- Why a warm-up compile run must precede every trace capture, and what "compile" means in this context
- How the model differentiates warm-up calls from production calls at runtime
- How `WarmupForwardMixin.warmup_model_decode` and `warmup_model_prefill` orchestrate warm-up across sequence lengths and batch sizes
- What Tracy markers appear during trace capture vs. replay, and how to tell the phases apart in a Tracy timeline
- Which environment variables and `python3 -m tracy` flags are required to see trace-related events in a profiling report
- How `metal_trace_id` is attached to ops in the ops CSV and how `split_compile_and_trace` uses that to partition perf data

---

## Chapter List

### Chapter 1 — Trace Capture in TTNN: The Core API
**Description:** Introduces the three-function TTNN trace API (`begin_trace_capture`, `end_trace_capture`, `execute_trace`), explains what is recorded and why replay bypasses host dispatch overhead, and establishes the compile-capture-replay lifecycle that all subsequent chapters build on.

**Directory:** `ch1_trace_capture_api/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Diagram: three-phase lifecycle — compile run, capture run, replay runs
  - Glossary of terms introduced in this chapter (trace, command buffer, buffer aliasing, `cq_id`, `trace_id`)
  - "What's next" section listing files in reading order

- `trace_api_overview.md`
  - Introduce `ttnn.begin_trace_capture(mesh_device, cq_id=0)` and what it returns (`trace_id`)
  - Introduce `ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)` and what it seals into the command buffer
  - Introduce `ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)` and the non-blocking dispatch model
  - Explain what is recorded: the command sequence, kernel arguments, and device buffer addresses as they exist at capture time
  - Explain what is NOT recorded: Python control flow, tensor shape recomputation, host-side sampling logic, or any op called before `begin_trace_capture`
  - Minimal standalone code pattern showing the three calls in sequence

- `compile_run_requirement.md`
  - Explain why a "compile run" (an ordinary, non-traced forward pass) must execute before every trace capture
  - Describe what the compile run does: TTNN op compilation, program cache warming, kernel binary upload to device
  - Explain that if the compile run is skipped, the capture run triggers compilation during recording, causing the kernel build artifacts to be embedded in the trace rather than being reusable
  - Show the pattern used in `_capture_decode_trace_text` and `_capture_trace_prefill` in `generator.py`: compile run first, then capture bracket
  - Cover `trace_region_size` in `device_params`: what it reserves and why OOM occurs if it is too small (reference `trace_region_config.py` values per model/device)

- `replay_mechanics.md`
  - Explain why replay is faster than live dispatch: no host-side op re-encoding, no kernel re-selection, device reads a pre-encoded command buffer directly
  - Describe buffer aliasing: trace replay reuses the exact device memory addresses recorded at capture time, so input tensors must stay at those addresses
  - Explain how `copy_host_to_device` with `device_tensors=` reuses existing device buffers rather than allocating new ones, which is the mechanism that makes input updates between replays possible
  - Describe `blocking=False` in `execute_trace` and what that means for host/device overlap
  - Cover when `ttnn.synchronize_device()` or a readback is needed after `execute_trace`

---

### Chapter 2 — How tt-transformers Uses Trace Capture
**Description:** Walks through the actual trace capture and execution code paths in the `Generator` class, covering both the decode and prefill trace flows, the keying strategy that maps (seq_len, model_id) to stored trace IDs, and the split-sampling trace variant.

**Directory:** `ch2_generator_trace_flows/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Diagram: state machine of `Generator` instance — first call captures, subsequent calls replay
  - Recap of Chapter 1 prerequisites
  - "What's next" section listing files in reading order

- `decode_trace_flow.md`
  - Walk through `decode_forward` in `generator.py`: how `enable_trace=True` routes to `_decode_forward_trace_text`
  - Explain `_capture_decode_trace_text`: compile run via `_decode_forward_no_trace_text`, then `begin_trace_capture` / `ttnn_decode_forward` / `end_trace_capture` for each data-parallel shard
  - Explain `_decode_forward_trace_text`: on first call captures and stores `trace_ids_decode[sampling_on_device]`; on subsequent calls updates device inputs and calls `execute_trace`
  - Show how `reset_batch` / `reset_inputs` determines whether `copy_host_to_device` re-copies page tables and token tensors before each replay
  - Cover the `sampling_on_device` boolean as a second key in `trace_ids_decode`: why two separate traces may be captured (greedy vs. sampled)
  - Cover the split-sampling trace variant: when `enable_split_sampling=True` and a `sampling` module is present, `capture_trace` on the sampling module captures a second trace for the sampling step separately

- `prefill_trace_flow.md`
  - Walk through `prefill_forward_text` routing: `can_enable_trace` on `model_args` determines whether the prefill is traceable for a given sequence length
  - Explain `_capture_trace_prefill`: compile run, then capture bracket around `transform_and_embed_prefill_inputs_device` and `ttnn_prefill_forward`
  - Explain `_easy_trace_prefill`: the `trace_key = f"{prefill_seq_len}_{model_id}"` keying scheme that stores one trace per (seq_len, model_id) pair in `trace_id_prefill`
  - Show `_prefill_forward_trace`: how it calls `prepare_prefill_inputs_trace`, copies to device buffers, then `execute_trace`
  - Explain `can_enable_trace` in `model_config.py`: the allowed sequence lengths per device type (`get_trace_prefill_supported_seq_lens`), the constraint that `num_cached_tokens == 0`, and why chunked prefill and prefix caching are currently excluded
  - Note that prefill trace is only available when paged attention is enabled (page table required)

- `model_config_trace_settings.md`
  - Describe the `trace_region_size` dict in `trace_region_config.py`: how it maps (model, device) to byte sizes and how `get_supported_trace_region_size` is used in conftest to override the pytest fixture
  - Explain `get_trace_prefill_supported_seq_lens` in `model_config.py`: the per-device default lists and the per-model override dict (e.g., Llama-3.1-8B on N300 supports up to 8192 while N150 is limited to 128)
  - Describe the `num_command_queues: 1` device parameter in `simple_text_demo.py` and why trace requires a single command queue
  - Show where `trace_region_size` is passed through the device fixture and what happens at runtime if it is exhausted

---

### Chapter 3 — Model Warm-Up and Its Relationship to Trace Capture
**Description:** Explains the warm-up phase that vLLM and the demo driver perform before inference begins — what it compiles, how it generates the captured traces for supported sequence lengths, and how to distinguish warm-up calls from production calls at runtime and in profiling output.

**Directory:** `ch3_warmup/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Diagram: timeline from device open → warm-up → first real inference request
  - Recap of how compile runs from Chapter 1 are embedded in warm-up
  - "What's next" section listing files in reading order

- `warmup_decode.md`
  - Describe `WarmupForwardMixin.warmup_model_decode` in `models/common/warmup/warmup_utils.py`: it calls `decode_forward` once with zeroed tokens and page table at max batch size to compile all ops and capture the decode trace
  - Explain `_create_decode_warmup_inputs`: `torch.zeros(max_batch_size, 1)` tokens, `torch.zeros(max_batch_size)` start pos, and a zeroed page table
  - Describe how `_create_sampling_params` produces the set of sampling configurations to sweep (greedy, non-greedy with/without penalties, log_probs) so that all trace variants are captured during warm-up
  - Show how the `already_warmed_up_prefill` guard and the analogous decode call prevent redundant warm-up on repeated calls
  - Explain the log line `"Starting decode warmup"` / `"Decode warmup completed"` as the observable boundary in stdout

- `warmup_prefill.md`
  - Describe `warmup_model_prefill` in `generator.py`: it sweeps `get_warmup_prefill_supported_seq_lens()` — the power-of-two sequence lengths up to `capped_warmup_seq_len` — and calls `prefill_forward_text` for each
  - Explain `get_warmup_prefill_supported_seq_lens` in `model_config.py` and `calculate_prefill_warmup_seq_lens`: how it merges the trace-supported lengths with the full warmup range
  - Explain the `model_id` loop: for `model_id == 0` all sequence lengths are compiled; for `model_id > 0` only trace-supported lengths are re-run (because compilation artifacts are shared, only trace capture is per-mesh)
  - Describe the warm-up token tensors: `torch.zeros(batch_size, supported_length, dtype=torch.long)` and the corresponding page table
  - Show the `warmup_prefill=True` default in `prefill_forward_text` and how passing `warmup_prefill=False` skips the gate (used in test code)
  - Note the skip condition for chunked prefill: if `page_table_warmup is None` and the seq_len exceeds `max_prefill_chunk_size`, warm-up is skipped with a warning

- `differentiating_warmup_from_production.md`
  - Explain how to tell warm-up calls from production calls at the Python level: warm-up uses zero-filled tensors and happens before any user request; the `already_warmed_up_prefill` boolean prevents re-entry
  - Explain how to tell them apart in profiling output: warm-up calls appear in the ops CSV before any `TT_METAL_TRACE_REPLAY` markers; the `metal_trace_id` column is null for warm-up compile runs and non-null for capture/replay runs
  - Describe the `split_compile_and_trace` utility in `test_utils.py`: how it uses `find_repeated_runs` to detect the boundary between the compile phase and the trace phase in a perf CSV, and how `df_model_compilation` vs `df_model_trace` splits the data
  - Cover signposts: how `tracy.signpost("start")` / `("stop")` can be inserted around the production loop to filter warm-up rows in `post_process_ops_log` via `has_signposts=True`

---

### Chapter 4 — Tracy Profiling with Trace Capture
**Description:** Covers how to run tt-transformers under Tracy, which environment variables and CLI flags activate trace-aware profiling, what Tracy markers are emitted during capture and replay, and how to read a profiling report to separate warm-up from production and trace-captured ops from non-traced ops.

**Directory:** `ch4_tracy_profiling/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Diagram: `python3 -m tracy -r` subprocess flow — capture process, test process, csvexport, `process_ops`
  - Recap of Tracy's role relative to Chapters 1–3
  - "What's next" section listing files in reading order

- `running_with_tracy.md`
  - Show the full invocation: `python3 -m tracy -r [options] -- pytest <test_file> ...` and the role of `-r` (report mode) vs. running without it
  - Explain the environment variables set by `python3 -m tracy -r`: `TTNN_OP_PROFILER=1`, `TT_METAL_DEVICE_PROFILER=1`, `TT_METAL_PROFILER_TRACE_TRACKING=1`
  - Describe `TT_METAL_PROFILER_TRACE_TRACKING=1`: this causes `TracyTTMetalBeginMeshTrace`, `TracyTTMetalEndMeshTrace`, and `TracyTTMetalReplayMeshTrace` macros in C++ to emit `TT_METAL_TRACE_BEGIN`, `TT_METAL_TRACE_END`, and `TT_METAL_TRACE_REPLAY` Tracy messages
  - Explain `TT_METAL_TRACE_PROFILER=1` (set via `--device-trace-profiler`): activates `profiler_trace_profiler` in rtoptions, which profiles device-side trace execution durations rather than individual op durations
  - Describe output artifacts: `tracy_profile_log_host.tracy`, `tracy_ops_times.csv`, `tracy_ops_data.csv`, and the final `ops_perf_results_<date>.csv` in the reports folder
  - Show the `run_device_profiler` helper in `models/perf/device_perf_utils.py` as the programmatic equivalent used in `test_device_perf.py`

- `tracy_markers_for_trace.md`
  - Explain the C++ macros in `tt_metal_tracy.hpp` that emit Tracy messages:
    - `TracyTTMetalBeginMeshTrace` emits `` `TT_METAL_TRACE_BEGIN: {device_id}, {trace_id}` ``
    - `TracyTTMetalEndMeshTrace` emits `` `TT_METAL_TRACE_END: {device_id}, {trace_id}` ``
    - `TracyTTMetalReplayMeshTrace` emits `` `TT_METAL_TRACE_REPLAY: {device_id}, {trace_id}` ``
    - `TracyTTMetalEnqueueMeshWorkloadTrace` emits `` `TT_METAL_TRACE_ENQUEUE_PROGRAM: {device_id}, {trace_id}, {runtime_id}` `` (only when `TT_METAL_PROFILER_TRACE_TRACKING=1`)
  - Explain how `import_tracy_op_logs` in `process_ops_logs.py` parses these messages: it maintains a `traceIDs` dict keyed by `device_id` and sets `opData["metal_trace_id"]` for every op emitted while a trace is open
  - Show how the `METAL TRACE ID` and `METAL TRACE REPLAY SESSION ID` columns appear in the final ops CSV
  - Describe the `TraceReplayDict` and `lookup_trace_replay_timestamp` function: how replay session IDs index into per-trace replay timestamps to correctly anchor device ops to the right Tracy wall-clock window

- `reading_profiling_output.md`
  - Walk through interpreting `ops_perf_results_<date>.csv` when trace is active:
    - Rows with `METAL TRACE ID` null: warm-up compile run or non-traced ops
    - Rows with `METAL TRACE ID` set and `METAL TRACE REPLAY SESSION ID` = 1: the trace capture run (single recording)
    - Rows with `METAL TRACE ID` set and `METAL TRACE REPLAY SESSION ID` >= 2: production replay iterations
  - Explain how `split_compile_and_trace` in `test_utils.py` automates this split using the `find_repeated_runs` heuristic on op names: it finds where the same op sequence repeats `num_runs` times and slices the DataFrame accordingly
  - Describe `post_process_ops_log` with `has_signposts=True`: how inserting `ttnn.tracy_message("`TT_SIGNPOST: start`")` before the production loop and `"stop"` after enables filtering to exclude warm-up rows without needing the heuristic
  - Show how to use `ttnn.start_tracy_zone` / `ttnn.stop_tracy_zone` from `ttnn/ttnn/profiler.py` (exposed via `ttnn.start_tracy_zone` in `ttnn/__init__.py`) to bracket custom Python regions for annotation in the Tracy GUI timeline
  - Cover `tracy_frame()` as a frame marker to delimit individual decode steps in the Tracy GUI

- `differentiating_trace_ops_from_normal_ops.md`
  - Explain the `metal_trace_id` column as the primary signal: ops with a non-null value were dispatched as part of a trace capture or replay; ops with null were dispatched normally
  - Describe what "normal dispatch" looks like in the report vs. "trace replay": normal ops have individual per-op `DEVICE KERNEL DURATION [ns]` values measured independently; trace-replayed ops share the replay session timestamp anchored by `TracyTTMetalReplayMeshTrace`
  - Explain the `TT_METAL_TRACE_PROFILER=1` / `--device-trace-profiler` mode: instead of measuring individual op durations, the profiler measures the total duration of each trace replay as a unit, which is more accurate for understanding real decode step latency
  - Show how to cross-reference the `GLOBAL CALL COUNT` column with the `metal_trace_id` and session ID to reconstruct which ops belonged to which replay iteration
  - Cover the `TT_METAL_PROFILER_NO_CACHE_OP_INFO` env var (`--no-op-info-cache` flag): by default, op metadata for identical ops is cached to save space; disabling the cache shows full op info for every cached trace replay call

---

### Chapter 5 — Worked Example: Llama Decode with Trace and Tracy
**Description:** Combines all prior chapters into an end-to-end walkthrough of running a Llama model through warm-up, trace capture, and production decode under Tracy profiling, annotated with expected log output and a guide to interpreting the resulting ops report.

**Directory:** `ch5_worked_example/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Prerequisites: Chapters 1–4 must be read first; a working tt-metal dev environment is assumed
  - "What's next" section pointing to the two topic files and then to the end

- `running_the_example.md`
  - Show the complete pytest invocation for `simple_text_demo.py` under Tracy: environment variables to set, device fixture parameters (`trace_region_size`, `num_command_queues`), and the `-r` flag
  - Walk through the expected stdout sequence: device open → warm-up decode ("`Starting decode warmup`") → warm-up prefill ("`Warming up prefill for sequence length: ...`") → "`Done Compiling Model`" → "`Done Capturing Decode Trace`" → first production decode token
  - Explain how `enable_trace=False` changes the flow: warm-up still runs (compile is still needed), but no `begin_trace_capture` / `execute_trace` calls are made; every decode step is a live-dispatch call
  - Show the effect of `token_accuracy=True` (which forces `enable_trace=False`): why teacher-forcing cannot be used with trace (the input token changes each step based on a reference, which requires re-preparation outside the trace)
  - Describe how to set `trace_region_size` per model and device using `get_supported_trace_region_size` in `trace_region_config.py` and the conftest override pattern

- `annotated_ops_report.md`
  - Show an illustrative ops CSV excerpt (not an actual run, but a representative structure) covering the three phases: warm-up compile rows, trace capture rows, and production replay rows
  - Annotate each section: what `METAL TRACE ID` values to expect, how `METAL TRACE REPLAY SESSION ID` increments, and which rows `split_compile_and_trace` would assign to `df_model_compilation` vs. `df_model_trace`
  - Walk through using `post_process_ops_log` programmatically: how to load the CSV, filter to the trace-replay rows, and sum `DEVICE KERNEL DURATION [ns]` to get the per-decode-step device time
  - Explain the latency difference visible between compile-phase rows and trace-replay rows for the same op: the compile row includes kernel build time; the trace replay row shows only execution time
  - Cover the `split_compile_and_trace` utility output tuple: `(df_model_compilation, df_model_trace, df_first_layer_compilation, df_first_layer_trace, df_mid_layers_compilation, df_mid_layers_trace, df_model_tail_compilation, df_model_tail_trace)` and how each is used in `test_device_perf.py` to verify per-layer and tail-op performance

---

## Conventions

**Terminology:**

| Term | Meaning in this guide |
|---|---|
| trace | A pre-encoded command buffer captured by `ttnn.begin_trace_capture` / `ttnn.end_trace_capture` that can be replayed without host re-dispatch |
| compile run | The first, non-traced forward pass that triggers TTNN kernel compilation and program cache warming; always precedes trace capture |
| capture run | The second forward pass executed inside the `begin_trace_capture` / `end_trace_capture` bracket; its commands are recorded into the trace buffer |
| replay | Any subsequent call to `ttnn.execute_trace`; replays the captured command buffer without re-invoking the Python or host dispatch path |
| warm-up | The pre-inference phase (implemented by `WarmupForwardMixin`) that performs one or more compile runs and capture runs for each supported configuration before real requests arrive |
| `cq_id` | Command queue identifier; `cq_id=0` is the primary compute queue used for all trace operations in tt-transformers |
| `trace_id` | An integer handle returned by `begin_trace_capture` that identifies the captured buffer; stored in `trace_id_prefill` or `trace_ids_decode` on the `Generator` instance |
| trace key | The string `f"{prefill_seq_len}_{model_id}"` used to index `trace_id_prefill` for prefill traces; decode traces are keyed by `sampling_on_device` boolean |
| `metal_trace_id` | The integer trace ID attached to each op row in the profiling CSV; null for non-traced ops |
| replay session ID | A monotonically increasing integer per trace ID that distinguishes the first (capture) replay from subsequent (production) replays in the ops CSV |
| `TTNN_OP_PROFILER` | Environment variable that enables per-op profiling hooks in TTNN; required for op-level data in the Tracy report |
| `TT_METAL_PROFILER_TRACE_TRACKING` | Environment variable that enables emission of `TT_METAL_TRACE_BEGIN/END/REPLAY` Tracy messages; required to see `metal_trace_id` in the ops CSV |
| `TT_METAL_TRACE_PROFILER` | Environment variable that switches device profiling to measure total trace replay duration instead of per-op duration |
| signpost | A `ttnn.tracy_message("`TT_SIGNPOST: <label>`")` call that appears as a marker in the ops CSV and can be used to filter rows between start/stop labels |

**Notation:**

- All TTNN Python API symbols are formatted as inline code: `ttnn.begin_trace_capture`, `ttnn.execute_trace`, etc.
- All C++ macro names are formatted as inline code: `TracyTTMetalBeginMeshTrace`, `TracyTTMetalReplayMeshTrace`, etc.
- Environment variable names are formatted as inline code: `TT_METAL_PROFILER_TRACE_TRACKING`, `TTNN_OP_PROFILER`, etc.
- File paths relative to the tt-metal repository root are formatted as inline code paths: `models/tt_transformers/tt/generator.py`.
- Timing values are always in microseconds (us); if a value exceeds 1000 us, milliseconds (ms) are used with a parenthetical us equivalent.
- Tracy message strings are shown in backtick-delimited form matching the actual format in the source: `` `TT_METAL_TRACE_BEGIN: {device_id}, {trace_id}` ``.
- Diagrams use swimlane format with Python host on top, TTNN runtime in the middle, and device at the bottom.

**Formatting rules:**

- Each `.md` file begins with an H1 title matching the file's topic, followed by a one-paragraph orientation that states what the reader will know by the end of the file.
- Code examples are complete and runnable where possible; if a snippet requires surrounding context, that context is shown in a collapsed `<details>` block.
- Every chapter's `index.md` ends with a "What's next" section listing the files in that chapter in reading order.
- Callout blocks use blockquote syntax with a bold label: `> **Note:**`, `> **Warning:**`, `> **Key insight:**`.
- No emoji in any file.
- File references to actual source files in the codebase always include the path from the tt-metal repo root so readers can locate them directly.

---

## Cross-Chapter Dependencies

```
Chapter 1 (Trace Capture in TTNN: The Core API)
  - Introduces: trace_id, begin/end/execute_trace, compile run, capture run, replay,
                buffer aliasing, trace_region_size, cq_id, blocking=False
  - Required by: all subsequent chapters

Chapter 2 (How tt-transformers Uses Trace Capture)
  - Depends on: Chapter 1 (trace API, compile run requirement, buffer aliasing)
  - Introduces: Generator trace state (trace_id_prefill, trace_ids_decode), decode trace flow,
                prefill trace flow with can_enable_trace, trace keying strategy,
                split-sampling trace variant, model_config trace settings,
                copy_host_to_device with device_tensors reuse
  - Required by: Chapters 3 (warm-up drives the compile+capture calls), 4 (profiling interprets
                 the flows documented here), 5 (worked example runs this code)

Chapter 3 (Model Warm-Up and Its Relationship to Trace Capture)
  - Depends on: Chapter 1 (compile run concept), Chapter 2 (Generator trace flows, prefill/decode
                capture paths that warm-up invokes)
  - Introduces: WarmupForwardMixin, warmup_model_decode, warmup_model_prefill,
                get_warmup_prefill_supported_seq_lens, already_warmed_up_prefill guard,
                sampling config sweep during warm-up, zero-filled warmup tensors,
                distinguishing warm-up from production at the Python level
  - Required by: Chapter 4 (Tracy markers appear during warm-up; understanding the phases is
                 required to correctly interpret profiling output), Chapter 5 (worked example
                 describes expected warm-up log output)

Chapter 4 (Tracy Profiling with Trace Capture)
  - Depends on: Chapter 1 (what trace is), Chapter 2 (trace_id, decode/prefill flows),
                Chapter 3 (warm-up phase boundary, compile vs. capture vs. replay phases)
  - Introduces: python3 -m tracy -r invocation, TTNN_OP_PROFILER, TT_METAL_DEVICE_PROFILER,
                TT_METAL_PROFILER_TRACE_TRACKING, TT_METAL_TRACE_PROFILER,
                TracyTTMetalBeginMeshTrace / EndMeshTrace / ReplayMeshTrace C++ macros,
                metal_trace_id column, METAL TRACE REPLAY SESSION ID,
                split_compile_and_trace utility, signpost mechanism,
                ttnn.start_tracy_zone / stop_tracy_zone / tracy_frame
  - Required by: Chapter 5 (worked example uses all the profiling tools and output formats
                 described here)

Chapter 5 (Worked Example: Llama Decode with Trace and Tracy)
  - Depends on: all prior chapters
  - Synthesizes: trace API (Ch1), Generator trace state and flows (Ch2), warm-up sequence
                 and boundaries (Ch3), Tracy invocation and output interpretation (Ch4)
  - Introduces no new concepts; provides an integrated, annotated end-to-end reference
```
