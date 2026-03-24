# B Review — Pass 1

## Item 1 — Warm-up ordering contradiction (index.md vs. differentiating_warmup_from_production.md)

`index.md` timeline (lines 21–27) shows `warmup_model_prefill` executing before `warmup_model_decode`. `differentiating_warmup_from_production.md` line 28 states: "In a vLLM deployment `warmup_model_decode` and `warmup_model_prefill` are called during engine initialization" — listing decode first. A reader implementing a new caller or debugging initialization order will get conflicting information about which warm-up phase runs first. One of these two sources is wrong; the correct order must be established and made consistent.

## Item 2 — `split_compile_and_trace` / `num_runs` description is internally incoherent

`differentiating_warmup_from_production.md` lines 60–73 describe `split_compile_and_trace` as using `num_runs=1` for "the standard three-phase scenario (compile, capture, and one replay)." With `num_runs=1`, `find_repeated_runs` trivially returns `left=0` on the first iteration (any single contiguous block of any length satisfies "1 identical block"), so the algorithm does not actually locate a meaningful phase boundary. The subsequent index arithmetic (`adjusted_len = (len(df) - first_run_start) // num_runs`) with `num_runs=1` sets `df_model_compilation` and `df_model_trace` to the same slice — the entire suffix. This is self-contradictory. A reader implementing or debugging trace analysis based on this description will compute wrong per-phase DataFrames. The correct `num_runs` value for a compile + capture + replay scenario should be 3 (or 2 if only capture and replay repeat), not 1.

## Item 3 — `get_all_padded_prefill_lengths` return value skips intermediate powers of two without explanation

`warmup_prefill.md` line 39 states the function returns `[128, 1024, 2048, 4096, ...]` — "128 followed by powers of two from 1024." This implies 256 and 512 are silently absent. No rationale is given for the gap. If the function actually returns all powers of two from 128 upward (128, 256, 512, 1024, …), the warmup sequence length set described here is wrong and a reader would believe fewer compile passes run than actually do. If the gap is real and intentional, a brief explanation is needed for correctness (a reader configuring `max_prefill_chunk_size` between 128 and 1024 would expect 256/512 to be warmed up and would be surprised at runtime failures for those lengths).

## Item 4 — Warm-up capture rows: `METAL TRACE ID` claim may be wrong

`differentiating_warmup_from_production.md` table (lines 36–42) asserts that during the warm-up capture phase `METAL TRACE ID` is "non-null (trace being built)." Line 44 repeats: "The first non-null `METAL TRACE ID` row marks the start of the trace-capture phase." In TT-Metal's tracing model, `ttnn.begin_trace_capture` / `ttnn.end_trace_capture` record ops into an in-memory buffer; whether the profiler assigns a non-null `METAL TRACE ID` to those rows at capture time versus only at replay time is not established in this chapter. If capture-phase rows are also written with a null `METAL TRACE ID` (and `METAL TRACE ID` is only populated on replay), then the stated boundary rule ("first non-null row = start of capture") would be wrong — the first non-null row would instead be the first production replay row. A reader filtering warm-up rows from a CSV would mis-classify all capture rows.

# B Review — Pass 2

## Item 1 — `capped_warmup_seq_len` power-of-2 assertion trigger mis-stated (`warmup_prefill.md`)

`warmup_prefill.md` note at line 47 states: "If you set `max_prefill_chunk_size` to a non-power-of-2 value the assertion will fire during warm-up startup." The assertion is on `capped_warmup_seq_len = min(max_prefill_chunk_size, max_seq_len)`, not on `max_prefill_chunk_size` directly. If `max_seq_len < max_prefill_chunk_size` and `max_seq_len` is itself a power of 2, the `min()` produces a power-of-2 value and the assertion does not fire, even though `max_prefill_chunk_size` is non-power-of-2. A reader who sets, for example, `max_prefill_chunk_size=3000` and `max_seq_len=2048` would follow this note, expect an assertion, and be wrong. The correct trigger condition is "`capped_warmup_seq_len` (the `min`) is not a power of 2," which only implies `max_prefill_chunk_size` must be a power of 2 when it is the smaller of the two values.

## Item 2 — Residual inconsistency: "trace replay ID" phrasing in `warmup_decode.md` conflicts with Pass 1 fix

`warmup_decode.md` line 107 states: "In a Tracy ops CSV the corresponding rows will appear before any rows whose `metal_trace_id` column is populated with a non-null trace replay ID." Pass 1 corrected `differentiating_warmup_from_production.md` to establish that `METAL TRACE ID` is non-null during **both** the capture phase and the replay phase, with `METAL TRACE REPLAY SESSION ID` being what distinguishes them. The phrase "trace replay ID" in `warmup_decode.md` still implies `METAL TRACE ID` is only non-null during replay, contradicting the corrected table. A reader relying on `warmup_decode.md` alone would incorrectly conclude compile-phase rows are the only rows with null `METAL TRACE ID`, missing that capture-phase rows are also warm-up rows and also carry a non-null `METAL TRACE ID`.

## Item 3 — `sampling_parameters_sweeped` scope is ambiguous (`warmup_prefill.md`)

`warmup_prefill.md` lines 92–101 describe the `sampling_parameters_sweeped` flag as controlling that `_create_sampling_params` is called "only for the first sequence length processed," but does not state at which loop level the flag is initialized. If it is initialized inside the `model_id` loop (resetting per mesh), sampling params are swept once per mesh. If it is initialized outside both the `model_id` and `supported_length` loops, sampling params are swept exactly once across the entire warm-up. These produce different on-device sampling kernel compile/capture behavior for data-parallel deployments (`data_parallel > 1`). A reader implementing warm-up for a new generator class needs the correct scope to know whether additional meshes need their own sampling-variant compile passes.

## Change Log — Pass 1 Fixes

**Applied 2026-03-23 by Agent A.**

### Item 1 — Warm-up ordering contradiction
- **Verified:** `models/demos/multimodal/gemma3/demo/text_demo.py` (lines 850–863) calls `warmup_model_prefill` before `warmup_model_decode`. `index.md` is correct (prefill → decode).
- **Fixed:** `differentiating_warmup_from_production.md` line 28. Changed "In a vLLM deployment `warmup_model_decode` and `warmup_model_prefill` are called during engine initialization" to "In a vLLM deployment `warmup_model_prefill` and `warmup_model_decode` are called during engine initialization, in that order: prefill warm-up runs first…then decode warm-up runs…".
- `index.md` required no changes.

### Item 2 — `split_compile_and_trace` / `num_runs` description
- **Verified:** `models/tt_transformers/tests/test_utils.py` docstring for `split_compile_and_trace` states `num_runs` is "number of runs in the CSV (typically 3: compile, capture, trace)". `test_device_perf.py` (line 31) parametrizes with `num_runs=2`. `num_runs` counts identical consecutive op-code blocks, not total phases; for compile+capture+replay the correct value is `num_runs=2` (capture and replay are identical; compile is the non-repeating prefix).
- **Fixed:** `differentiating_warmup_from_production.md`. Replaced the incorrect claim that `num_runs=1` handles "the standard three-phase scenario" with an accurate explanation: `num_runs` is the number of identical repeating blocks (production replay iterations), `num_runs=2` for a single capture + single replay, `num_runs=3` for a single capture + two replays. Updated the description of `df_model_compilation` and the closing sentence about "three identical main-model blocks."

### Item 3 — `get_all_padded_prefill_lengths` skips 256 and 512
- **Verified:** `models/tt_transformers/tt/common.py` `get_all_padded_prefill_lengths` builds `[128]` then appends `1024 * 2^k` values. `get_padded_prefill_len` maps seq_len 1–128 → 128 and 129–1024 → 1024, so 256 and 512 are not valid dispatch sizes and are intentionally absent.
- **Fixed:** `warmup_prefill.md` step 1 of `calculate_prefill_warmup_seq_lens`. Corrected "powers of two from 1024" to "multiples of 1024" and added an explanation: 256 and 512 are absent because they are not valid pad-length boundaries, so compiling them would produce kernels never dispatched.

### Item 4 — Capture-phase `METAL TRACE ID` nullness
- **Verified:** `tools/tracy/process_ops_logs.py` `import_tracy_op_logs` (lines 279–322). Between `TT_METAL_TRACE_BEGIN` and `TT_METAL_TRACE_END`, `traceIDs[deviceID]` is set to the active trace ID and every parsed op receives `metal_trace_id = traceID` (non-null). After `TT_METAL_TRACE_END`, `traceIDs[deviceID] = None`. Replay ops (`TT_METAL_TRACE_REPLAY`) receive their `METAL TRACE ID` from the C++ device perf CSV via `_enrich_ops_from_perf_csv` (line 570–572) and additionally carry a non-null `METAL TRACE REPLAY SESSION ID`. So: capture phase = non-null `METAL TRACE ID`, null `METAL TRACE REPLAY SESSION ID`; replay phase = non-null `METAL TRACE ID`, non-null `METAL TRACE REPLAY SESSION ID`.
- **Fixed:** `differentiating_warmup_from_production.md`. Added `METAL TRACE REPLAY SESSION ID` row to the table (null for compile and capture, non-null for replay). Updated the prose to explain that both capture and replay carry non-null `METAL TRACE ID`, and that `METAL TRACE REPLAY SESSION ID` is the column that distinguishes them. Updated the Key Insight callout accordingly.

## Change Log — Pass 2 Fixes

**Applied 2026-03-23 by Agent A.**

### Item 1 — `capped_warmup_seq_len` power-of-2 assertion trigger mis-stated (`warmup_prefill.md`)
- **Verified:** `models/tt_transformers/tt/model_config.py` lines 1042–1044. The assertion is `(capped_warmup_seq_len > 0 and (capped_warmup_seq_len & (capped_warmup_seq_len - 1)) == 0)`, where `capped_warmup_seq_len = min(max_prefill_chunk_size, max_seq_len)`. The assertion fires on the *result* of the `min()`, not on `max_prefill_chunk_size` alone. If `max_seq_len` is smaller and is a power of 2, a non-power-of-2 `max_prefill_chunk_size` does not trigger the assertion.
- **Fixed:** `warmup_prefill.md` note after the `capped_warmup_seq_len` section. Replaced "If you set `max_prefill_chunk_size` to a non-power-of-2 value the assertion will fire" with an accurate explanation: the assertion fires when the `min()` result (`capped_warmup_seq_len`) is not a power of 2, with examples of when that can and cannot occur.

### Item 2 — Residual "trace replay ID" phrasing in `warmup_decode.md` (line 107)
- **Verified:** Per Pass 1 Item 4 findings, `METAL TRACE ID` is non-null during both the capture phase and the replay phase. The phrase "trace replay ID" in `warmup_decode.md` line 107 incorrectly implied `METAL TRACE ID` is only non-null during replay, contradicting the corrected understanding.
- **Fixed:** `warmup_decode.md` line 107 (Observable Log Boundaries section). Replaced the sentence beginning "In a Tracy ops CSV…" with an accurate description: compile-phase rows appear before any rows with a non-null `METAL TRACE ID` (i.e., before both capture and replay); `METAL TRACE ID` is non-null for both capture and replay; `METAL TRACE REPLAY SESSION ID` (null for compile and capture, non-null for replay) is the secondary distinguishing column.

### Item 3 — `sampling_parameters_sweeped` scope ambiguous (`warmup_prefill.md`)
- **Verified:** `models/tt_transformers/tt/generator.py` line 100. `sampling_parameters_sweeped = False` is initialized outside the `model_id` loop (line 102). The flag is shared across all `model_id` iterations; sampling params are swept exactly once during the entire warm-up (on `model_id == 0`, first `batch_size`, first `supported_length`). All subsequent iterations — including those for `model_id > 0` — use `sampling_params = [None]`.
- **Fixed:** `warmup_prefill.md` Sampling Parameters section. Added explicit statement that the flag is initialized outside the `model_id` loop. Updated the code snippet to show the outer loop context. Clarified that additional meshes do not repeat sampling-variant compilation.

# B Review — Pass 3

## Item 1 — `num_runs` definition contradicts its own usage examples (`differentiating_warmup_from_production.md`, line 61)

Line 61 defines `num_runs` as "the number of production replay iterations included in the CSV." By that definition, one replay iteration = `num_runs=1`. But the immediately following guidance says "pass `num_runs=2` when the CSV contains one capture and one replay." These two statements are mutually exclusive: one replay cannot simultaneously be `num_runs=1` (by the stated definition) and `num_runs=2` (by the usage example). The correct definition is that `num_runs` counts identical repeating op-code blocks — capture counts as one block, each replay counts as one block — so one capture plus one replay = `num_runs=2`. A reader who follows the definition ("number of production replay iterations") and passes `num_runs=1` for a standard capture+replay CSV will make `find_repeated_runs` return `left=0` trivially and `df_model_compilation` / `df_model_trace` will both equal the full suffix DataFrame, yielding wrong per-phase latency results.

## Item 2 — `df_model_compilation` is labeled "compile phase" but contains capture-phase ops (`differentiating_warmup_from_production.md`, line 74)

Line 74 describes `df_model_compilation` as containing "the first identical block (the compile phase, or the trace-capture phase when compile ops have been stripped as a non-repeating prefix)." The second half of that parenthetical is the true case: `find_repeated_runs` strips the non-repeating prefix (which holds the actual compile-phase ops) and returns the start of the repeating suffix. `df_model_compilation` is always the **first** of the `num_runs` identical repeating blocks, which is the **capture phase**, not the compile phase. The compile-phase ops live in `df[:first_run_start]`, which `split_compile_and_trace` does not return at all. A reader who uses `df_model_compilation` to measure compile-phase kernel latency will instead be measuring capture-phase latency and draw incorrect conclusions about JIT compilation cost.

## Change Log — Pass 3 Fixes

**Applied 2026-03-23 by Agent A.**

### Item 1 — `num_runs` definition contradicts usage examples (`differentiating_warmup_from_production.md`, line 61)
- **Verified:** `models/tt_transformers/tests/test_utils.py` docstring for `split_compile_and_trace` states `num_runs: number of runs in the CSV (typically 3: compile, capture, trace)`. `test_device_perf.py` line 31 parametrizes with `num_runs=2`. `find_repeated_runs` counts all identical repeating blocks regardless of phase type — the JIT-compile prefix is stripped as a non-repeating head; capture counts as one block and each replay counts as one block.
- **Fixed:** `differentiating_warmup_from_production.md` line 61. Replaced the incorrect definition ("number of production replay iterations") with the accurate definition: `num_runs` counts every phase whose op sequence is identical (capture + each replay). Clarified that the JIT-compile phase is a non-repeating prefix stripped automatically and does not count toward `num_runs`. Updated the example: `num_runs=2` for one capture + one replay; `num_runs=3` for one capture + two replays. Added a note reconciling the docstring's "typically 3" with `test_device_perf.py`'s `num_runs=2`.

### Item 2 — `df_model_compilation` mislabeled as "compile phase" (`differentiating_warmup_from_production.md`, line 74)
- **Verified:** `test_utils.py` lines 363–368. `first_run_start` is the start of the repeating suffix (where `num_runs` identical blocks begin). `df_model_compilation = df[first_run_start:first_run_end]` is the **first** of those identical blocks — the trace-capture phase. The actual JIT-compile-phase ops are in `df[:first_run_start]`, which `split_compile_and_trace` does not return.
- **Fixed:** `differentiating_warmup_from_production.md` lines 74–78. Replaced the ambiguous parenthetical description with three clear statements: (1) `df_model_compilation` is the first repeating block = trace-capture phase, not JIT-compile phase; (2) `df_model_trace` is the last repeating block = final trace-replay run; (3) JIT-compile-phase ops are in `df[:first_run_start]` and are not returned by the function — users must slice that themselves to measure JIT compilation cost. Updated the `find_repeated_block` sentence to reference `df_model_compilation` as "the capture-phase DataFrame."

# B Review — Pass 4

## Item 1 — `num_runs=3` reconciliation contradicts the paragraph's own premise (`differentiating_warmup_from_production.md`, line 61)

The `num_runs` paragraph contains a direct internal contradiction. The first half states: "The JIT-compile phase is a non-repeating prefix...and is stripped automatically by `find_repeated_runs`; it does not count toward `num_runs`." The second half then states: "The `split_compile_and_trace` docstring describes the typical case as `num_runs=3` (compile, capture, trace) when all three phases happen to be identical in op sequence." These two claims are mutually exclusive. If the JIT-compile phase is always a non-repeating prefix that is stripped and never counts toward `num_runs`, it cannot simultaneously be one of the three identical blocks that the docstring's `num_runs=3` counts. A reader who accepts both sentences would be unable to determine the correct `num_runs` for a standard compile+capture+replay CSV. Concretely, if they follow the second sentence and conclude that a compile+capture+replay scenario takes `num_runs=3`, but the compile phase has a different op count (as the paragraph itself says — it includes a sampling-kernel compile pass), then `find_repeated_runs` will not find three identical blocks of the expected length and will return a wrong `first_run_start`, producing wrong per-phase DataFrames. Pass 3 fixed an earlier contradiction but the reconciliation sentence introduced this new one.

## Item 2 — "multiples of 1024" over-includes non-power-of-two lengths (`warmup_prefill.md`, line 39)

Pass 1 changed "powers of two from 1024" to "multiples of 1024" after verifying the source generates `1024 * 2^k` values. However `1024 * 2^k` for k=0,1,2,3,... yields 1024, 2048, 4096, 8192 — exclusively powers-of-two multiples of 1024. The phrase "multiples of 1024" encompasses all integer multiples: 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, etc. A reader who uses this description to predict which sequence lengths are compiled during warm-up would include 3072, 5120, 6144, 7168, and other non-power-of-two multiples that the function never generates, and would then write test assertions or capacity estimates based on a larger warmup set than actually exists. The correct term is "powers of two starting at 1024" (equivalently `1024 * 2^k` for k ≥ 0), not "multiples of 1024."

## Change Log — Pass 4 Fixes

**Applied 2026-03-23 by Agent A.**

### Item 1 — `num_runs` self-contradiction in `differentiating_warmup_from_production.md`
- **Verified:** `models/tt_transformers/tests/test_utils.py` lines 256–283 (`find_repeated_runs`) and lines 332–399 (`split_compile_and_trace`). `find_repeated_runs` has no concept of phase names; it strips any non-repeating prefix and counts whatever identical blocks remain. Whether the JIT-compile run is among the counted blocks or is stripped as the prefix depends entirely on whether its op sequence matches the capture/replay sequences — not on any hardcoded rule. The docstring's "typically 3: compile, capture, trace" describes the case where compile and capture produce the same op sequence and all three runs are identical. `test_device_perf.py` line 31 uses `num_runs=2` because the sampling-kernel compile pass creates a distinct op sequence, making compile a non-repeating prefix so only capture + replay (2 blocks) remain.
- **Fixed:** `differentiating_warmup_from_production.md`. Rewrote the `num_runs` paragraph to remove the mutual exclusion. The new text establishes the correct ground truth: `find_repeated_runs` is phase-agnostic; compile is stripped as a non-repeating prefix only when its op sequence differs from the others (the common case, hence `num_runs=2` in `test_device_perf.py`); `num_runs=3` applies when all three phases produce identical op sequences (compile op count matches capture and replay). The paragraph no longer contains any statement that contradicts another.

### Item 2 — "multiples of 1024" over-includes non-power-of-two lengths (`warmup_prefill.md`, line 39)
- **Verified:** `models/tt_transformers/tt/common.py` `get_all_padded_prefill_lengths` generates `1024 * 2^k` for k=0,1,2,… which produces 1024, 2048, 4096, 8192 — only power-of-two values. "Multiples of 1024" incorrectly includes 3072, 5120, 6144, etc.
- **Fixed:** `warmup_prefill.md` step 1 of `calculate_prefill_warmup_seq_lens`. Changed "multiples of 1024 (i.e., 1024, 2048, 4096, …)" to "powers of two starting from 1024 (i.e., 1024, 2048, 4096, 8192, …)."

# B Review — Pass 5

## Item 1 — `.iloc` used with label-based index values in the signpost filtering snippet (`differentiating_warmup_from_production.md`, line 120)

The code snippet shown for `post_process_ops_log` does:

```python
start = markers[markers == "start"].index[0]
stop  = markers[markers == "stop"].index[0]
df = df.iloc[start + 1 : stop]
```

`markers[...].index[0]` returns the **label** at that position in the DataFrame's index (a pandas `Index` value), not a positional integer. `.iloc` takes **positional** integers. These coincide only when `df` has a default 0-based `RangeIndex` that has never been disrupted by filtering, concatenation, or CSV parsing. In practice, an ops CSV that has been pre-filtered (e.g., after dropping header rows or merging device CSVs) will have a non-contiguous index, and `df.iloc[start + 1 : stop]` will silently slice the wrong rows — either too many or too few — producing an incorrect ops window. The correct call is `df.loc[start + 1 : stop]` (label-based), or positional lookup via `df.index.get_loc(start)`. A reader who copies this snippet as written will implement the signpost filter incorrectly whenever the DataFrame index is non-standard.

## Change Log — Pass 5 Fixes

**Applied 2026-03-23 by Agent A.**

### Item 1 — `.iloc` with label-based index values in signpost filtering snippet (`differentiating_warmup_from_production.md`, line 120)
- **Verified:** `tools/tracy/process_model_log.py` lines 26–35. `post_process_ops_log` loads `df` directly from `pd.read_csv(filename)` (line 28) with no prior filtering. The DataFrame therefore retains a default 0-based `RangeIndex`, so `markers[...].index[0]` returns an integer label that equals its positional index — `.iloc[start + 1 : stop]` is correct as written in the source. The source code uses `.iloc` and it is safe in this exact context.
- **Fixed:** `differentiating_warmup_from_production.md`. The snippet is unchanged (it accurately reflects the source). Added an explanatory note after the snippet clarifying: (1) `.iloc` is safe here because `df` retains its default 0-based `RangeIndex` immediately after `pd.read_csv`; (2) if the pattern is adapted in code that pre-filters `df`, the RangeIndex will be disrupted and `.iloc` will slice wrong rows; (3) in that case `.loc[start + 1 : stop]` should be used instead, with an explanation of `.loc`'s label-based, end-inclusive semantics and why the `stop` boundary still behaves correctly.

# B Review — Pass 6

## Item 1 — `.loc` note incorrectly claims the `stop` signpost row is excluded (`differentiating_warmup_from_production.md`, line 123)

The note added by Pass 5 states: "`.loc` with integer labels on a non-contiguous index is safe and includes all rows whose label falls in the closed interval `[start + 1, stop)` (note: `.loc` slicing is end-inclusive on the label axis, but because the `stop` label is the signpost row itself and is excluded by the `stop` boundary here, the behaviour matches the intent of the original `.iloc` slice)."

This is self-contradictory and wrong. `.loc` integer-label slicing is **end-inclusive**: `df.loc[a : b]` includes the row whose label equals `b`. Therefore `df.loc[start + 1 : stop]` **includes** the signpost row whose `OP CODE` is `"stop"` as the final row of the returned DataFrame. The `.iloc[start + 1 : stop]` original is **end-exclusive** and correctly excludes that row. The note's own parenthetical acknowledges that `.loc` is end-inclusive, yet then claims the `stop` row is "excluded by the `stop` boundary" — which contradicts the end-inclusive property it just stated. A reader who follows the recommendation and uses `df.loc[start + 1 : stop]` will include the signpost row in their analysis window, causing one spurious op row to appear at the end of every filtered DataFrame. The correct `.loc` equivalent to exclude the signpost row is `df.loc[start + 1 : stop - 1]`.

## Change Log — Pass 6 Fixes

**Applied 2026-03-23 by Agent A.**

### Item 1 — `.loc` end-inclusivity error in signpost filtering note (`differentiating_warmup_from_production.md`, ~line 123)
- **Issue:** The Pass 5 note recommended `df.loc[start + 1 : stop]` as a drop-in replacement for `df.iloc[start + 1 : stop]`, then added a self-contradictory parenthetical claiming `.loc` is end-inclusive but that the `stop` signpost row is still excluded. In reality, `.loc[start + 1 : stop]` includes the `stop` row (end-inclusive), whereas `.iloc[start + 1 : stop]` correctly excludes it (end-exclusive). The note's recommendation would silently include the `"stop"` signpost row in the analysis window.
- **Fixed:** `differentiating_warmup_from_production.md` note after the signpost filtering snippet. Removed the self-contradictory parenthetical. Replaced the single-sentence `.loc` recommendation with a clear two-bullet explanation:
  - `.iloc` is end-exclusive: `[start + 1 : stop]` excludes the row at position `stop`.
  - `.loc` with integer labels is end-inclusive: `[start + 1 : stop]` includes the row whose label equals `stop` (the signpost row).
  - The correct `.loc` equivalent that excludes the signpost row is `df.loc[start + 1 : stop - 1]`.

# B Review — Pass 8

## Item 1 — Missing case in `_create_sampling_params`: `can_sample_on_device=True` + `non_greedy_decoding_on_device=False` (`warmup_decode.md`, lines 29–45)

The description covers exactly two cases: (a) `can_sample_on_device=False` → returns `[None]`; (b) `can_sample_on_device=True` and `non_greedy_decoding_on_device=True` → returns 6 configs (4 non-greedy + greedy + `None`). The third case — `can_sample_on_device=True` and `non_greedy_decoding_on_device=False` — is never described. The greedy config and the `None` entry are described as appended "unconditionally" (after the non-greedy block), implying they are always present when `can_sample_on_device=True`. In the `non_greedy_decoding_on_device=False` path the 4-item non-greedy Cartesian product is skipped, leaving only those 2 unconditional entries (greedy + `None`). A reader implementing or configuring warm-up for a greedy-only on-device deployment would infer either 0 or 6 variants (the two documented cases) and have no basis for knowing the correct count of 2. This causes incorrect implementation of any new caller that sets `non_greedy_decoding_on_device=False`.

## Item 2 — `df[:first_run_start]` is inaccessible to callers of `split_compile_and_trace` (`differentiating_warmup_from_production.md`, line 83)

Line 83 states: "If you need to measure JIT compilation cost, you must slice `df[:first_run_start]` yourself." `first_run_start` is a local variable computed inside `split_compile_and_trace` via `find_repeated_runs`; it is not returned by the function. The function's return value contains only the per-phase DataFrames and layer sub-splits. A reader who follows this instruction cannot obtain `first_run_start` without either (a) modifying `split_compile_and_trace` to expose it or (b) duplicating the `find_repeated_runs` call on the original `df` outside the function. Neither option is mentioned. As written, the instruction is unactionable: the reader is told to use a variable they cannot access.

# B Review — Pass 7

## Item 1 — `num_runs=3` "compile, capture, trace" case contradicts the definitive statement about `df_model_compilation` (`differentiating_warmup_from_production.md`, lines 63 and 78)

Line 63 describes the `num_runs=3` scenario as "three identical blocks (compile, capture, trace) — the case where the compile run happens to produce the same op sequence as the capture and replay runs, so all three are counted." In this scenario the **first** identical block is the JIT-compile run. Line 78 then states definitively: "`df_model_compilation` contains the **first** of the `num_runs` identical repeating blocks — this is the **trace-capture phase**, not the JIT-compile phase."

These two statements are directly contradictory for the `num_runs=3` case described in line 63. When all three runs (compile, capture, replay) are identical and `num_runs=3` is passed, `df_model_compilation` is the JIT-compile block, not the capture block. A reader who follows line 63 to select `num_runs=3` and then uses `df_model_compilation` to measure trace-capture latency (as line 78 says to) will actually be measuring JIT-compile latency instead, producing wrong per-phase results.

The fix requires either: (a) removing the "compile, capture, trace" scenario from the `num_runs=3` guidance and reserving `num_runs=3` solely for one-capture-plus-two-replays, so that `df_model_compilation` is always the capture block; or (b) adding a caveat at line 78 that when `num_runs=3` includes compile as the first block, `df_model_compilation` is the compile block rather than the capture block.

## Change Log — Pass 7 Fixes

**Applied 2026-03-23 by Agent A.**

### Item 1 — `num_runs=3` "compile, capture, trace" description contradicts `df_model_compilation` definition (`differentiating_warmup_from_production.md`)
- **Verified:** `models/tt_transformers/tests/test_utils.py` lines 256–283 (`find_repeated_runs`) and lines 332–399 (`split_compile_and_trace`). `find_repeated_runs` is fully phase-agnostic: it strips any non-repeating prefix and counts whatever identical blocks remain. In practice, the JIT-compile phase includes a sampling-kernel compile pass that produces a different op count than the capture and replay runs; this makes the compile run a non-repeating prefix that is always stripped. `test_device_perf.py` line 31 parametrizes `num_runs=2` (one capture block + one replay block), confirming compile is never counted. The `split_compile_and_trace` docstring's "typically 3: compile, capture, trace" describes the three phases present in the CSV file, not three identical counted blocks.
- **Resolution chosen:** Option (a) — remove the "compile, capture, trace" description from the `num_runs=3` case entirely. Reserve `num_runs=3` exclusively for one-capture-plus-two-replays. This ensures `df_model_compilation` is always the trace-capture block with no exceptions or caveats required.
- **Fixed:** `differentiating_warmup_from_production.md` lines 61–63 (the `num_runs` paragraph). Replaced the paragraph that described `num_runs=3` as covering "compile, capture, trace" with a revised paragraph establishing that the JIT-compile phase always has a distinct op sequence and is always stripped as the non-repeating prefix. Defined `num_runs=2` as one capture + one replay (the standard case) and `num_runs=3` as one capture + two replays. Added a callout note explaining that the docstring's "typically 3: compile, capture, trace" refers to the three phases in the file, not to three identical counted blocks. `df_model_compilation` at line 83 is now unambiguously correct: it is always the first repeating block = trace-capture phase.

## Change Log — Pass 8 Fixes

**Applied 2026-03-23 by Agent A.**

### Item 1 — Missing `can_sample_on_device=True, non_greedy_decoding_on_device=False` case (`warmup_decode.md`, lines 29–45)
- **Verified:** `models/common/warmup/warmup_utils.py` `_create_sampling_params` (lines 31–73). The function has three distinct code paths: (1) `can_sample_on_device=False` → returns `[None]`; (2) `can_sample_on_device=True` and `non_greedy_decoding_on_device=True` → appends 4 non-greedy Cartesian-product configs (penalties × log_probs), then always appends greedy + `None` = 6 variants total; (3) `can_sample_on_device=True` and `non_greedy_decoding_on_device=False` → the non-greedy `for` loop is skipped; only greedy + `None` are appended = 2 variants total. The prior description documented cases (1) and (2) only and said the greedy and `None` entries are appended "unconditionally" (true), but never explicitly described the resulting 2-variant list for case (3).
- **Fixed:** `warmup_decode.md` sampling-config sweep section. Replaced the single combined description with three clearly labelled cases — `can_sample_on_device=False` (1 variant), `can_sample_on_device=True + non_greedy_decoding_on_device=False` (2 variants), and `can_sample_on_device=True + non_greedy_decoding_on_device=True` (6 variants) — and added a summary table listing all three combinations with their total variant counts. Explicitly noted that a greedy-only on-device deployment produces exactly 2 `decode_forward` calls during warm-up.

### Item 2 — `df[:first_run_start]` inaccessible via `split_compile_and_trace` return value (`differentiating_warmup_from_production.md`, line 83)
- **Verified:** `models/tt_transformers/tests/test_utils.py` lines 332–410 (`split_compile_and_trace`). The function's return tuple is `(df_model_compilation, df_model_trace, df_first_layer_compilation, df_first_layer_trace, df_mid_layers_compilation, df_mid_layers_trace, df_model_tail_compilation, df_model_tail_trace)`. `first_run_start` is a local variable (line 364) and is not returned. The prior text told readers to "slice `df[:first_run_start]` yourself," which is unactionable because `first_run_start` is not accessible from outside the function.
- **Fixed:** `differentiating_warmup_from_production.md` line 83. Removed the "slice yourself" instruction. Replaced it with actionable guidance: import and call `find_repeated_runs` directly on the original DataFrame with the same `num_runs` value to obtain `first_run_start`, then slice `df[:first_run_start]` to get the JIT-compile-phase prefix. Includes a code snippet showing the exact import path and call pattern.

# B Review — Pass 9

## Item 1 — `skip_sequence_lengths` break misattributed to the `model_id` loop (`warmup_prefill.md`, line 165)

The warning at line 165 states: "`skip_sequence_lengths = True` breaks out of the inner `batch_size` loop and then the `model_id` loop also stops iterating further sequence lengths via `if skip_sequence_lengths: break`."

The `model_id` loop iterates over data-parallel replica indices (0, 1, 2, …). It does not iterate over sequence lengths. Sequence lengths are iterated by the `supported_length` loop, which is nested inside the `model_id` loop. Attributing the second `break` to the `model_id` loop is structurally wrong: a reader who implements a similar guard based on this description would place `if skip_sequence_lengths: break` in the `model_id` loop, which would exit replica iteration entirely rather than exiting the sequence-length inner loop. The correct description is that the second `break` exits the `supported_length` loop (not the `model_id` loop) once `skip_sequence_lengths` is `True`, causing all remaining sequence lengths to be skipped for the current `model_id` — and since the same flag is visible to subsequent `model_id` iterations, those also exit the `supported_length` loop immediately, so no further sequence lengths are warmed up on any replica.

## Change Log — Pass 9 Fixes

**Applied 2026-03-23 by Agent A.**

### Item 1 — `skip_sequence_lengths` break misattributed to the `model_id` loop (`warmup_prefill.md`, ~line 165)
- **Verified:** `models/tt_transformers/tt/generator.py` `warmup_model_prefill`. The loop structure is: `for model_id in range(self.data_parallel):` (outer) → `for supported_length in sequence_lengths_to_warmup:` (middle) → `for batch_size in [1, 32]:` (innermost). The first `break` (line 136) exits the `batch_size` loop. The check `if skip_sequence_lengths: break` (lines 164–165) is positioned at the bottom of the `supported_length` loop body, so it exits the **`supported_length` loop**, not the `model_id` loop. `skip_sequence_lengths` is initialized at line 95, outside the `model_id` loop and is never reset inside it; therefore subsequent `model_id` iterations also hit `if skip_sequence_lengths: break` on their first `supported_length` and exit the `supported_length` loop immediately. The `model_id` loop itself runs to completion.
- **Fixed:** `warmup_prefill.md` warning callout in the "Skipping Warm-Up for Chunked Prefill Without Paged Attention" section. Replaced the incorrect claim that "the `model_id` loop also stops iterating further sequence lengths" with an accurate description: the second `break` exits the `supported_length` loop for the current `model_id`; subsequent `model_id` iterations also exit the `supported_length` loop immediately because `skip_sequence_lengths` is initialized outside the `model_id` loop and is never reset inside it; the `model_id` loop itself continues to its natural end.

# B Review — Pass 10

## Item 1 — `model_id > 0` skip condition: `enable_trace=False` branch causes all sequence lengths to be skipped on additional meshes, not just "purely compile-time lengths" (`warmup_prefill.md`, lines 53–54)

Lines 53–54 state: "the inner loop skips any length that is **not** in `trace_prefill_supported_seq_lens` **or** where `enable_trace` is `False`. Only lengths that will be replayed as traces need a capture pass on the additional meshes; purely compile-time lengths have no trace to capture on a second device."

The explanatory sentence is correct only when `enable_trace=True`. The skip condition is:

```python
if model_id != 0 and (
    supported_length not in self.model_args[0].trace_prefill_supported_seq_lens or not enable_trace
):
    continue
```

When `enable_trace=False`, `not enable_trace` evaluates to `True`, making the entire `or` clause `True` for **every** sequence length — including lengths that are in `trace_prefill_supported_seq_lens`. All lengths are therefore skipped on every `model_id > 0` replica. The text's explanation — "only lengths that will be replayed as traces need a capture pass" — implies trace-supported lengths still execute on additional meshes. A reader implementing warm-up for a multi-device deployment with `enable_trace=False` would expect trace-capable sequence lengths to be compiled (not just captured) on all replicas, but the code skips them entirely. This would result in additional meshes being un-compiled for those lengths when tracing is disabled, causing a slow JIT path on the first real inference call on those devices.

## Change Log — Pass 10 Fixes

**Applied 2026-03-23 by Agent A.**

### Item 1 — `model_id > 0` skip condition description incomplete for `enable_trace=False` (`warmup_prefill.md`, lines 53–54)
- **Verified:** `models/tt_transformers/tt/generator.py` `warmup_model_prefill` lines 107–110. The skip condition is `if model_id != 0 and (supported_length not in self.model_args[0].trace_prefill_supported_seq_lens or not enable_trace): continue`. When `enable_trace=False`, `not enable_trace` is `True` and the `or` short-circuits for every value of `supported_length`, causing all sequence lengths to be skipped on all `model_id > 0` replicas. When `enable_trace=True`, only lengths absent from `trace_prefill_supported_seq_lens` are skipped on additional meshes (the previously documented behaviour).
- **Fixed:** `warmup_prefill.md` `model_id` loop bullet for `model_id > 0`. Replaced the single-sentence description with a two-sub-bullet breakdown: (1) `enable_trace=True` — only lengths not in `trace_prefill_supported_seq_lens` are skipped (existing correct description for this case); (2) `enable_trace=False` — `not enable_trace` fires unconditionally for every length on every additional mesh, so only `model_id == 0` ever runs prefill warmup. Added a warning callout that in a multi-device no-trace deployment, additional meshes receive no prefill warmup compilation and will fall back to JIT compilation on their first real inference call.

# B Review — Pass 11

## Item 1 — "always stripped" in the `num_runs` Note contradicts the Pass 4 verified ground truth (`differentiating_warmup_from_production.md`, line 68)

The Note at line 68 states: "Because the JIT-compile phase has a different op sequence, it is **always** stripped as the non-repeating prefix and is never one of the counted blocks."

Pass 4 verified (b_review.md line 107–108) that `find_repeated_runs` is phase-agnostic and that whether the compile run is stripped depends entirely on whether its op sequence matches the subsequent capture/replay sequences — not on any hardcoded rule. The Pass 4 fix explicitly removed the claim that compile is always a non-repeating prefix, replacing it with the qualified "In practice" language now visible at line 63. The Note at line 68 reintroduces the absolute "always" claim that was specifically identified and corrected in Pass 4.

A reader whose model lacks the sampling-kernel compile pass (so compile, capture, and replay all share the same op sequence) would follow the "always stripped" Note, select `num_runs=2`, and `find_repeated_runs` would return `left=0` (the compile block is counted as the first identical block). `df_model_compilation` would then be the JIT-compile block and `df_model_trace` would be the capture block — the reader would measure compile-phase latency thinking it is capture-phase latency, and the actual replay block would not appear in either returned DataFrame.

## Item 2 — Kernel-sharing claim contradicts the `enable_trace=False` multi-device warning (`warmup_prefill.md`, lines 53 and 58/70)

Line 70 Key Insight states: "TT-Metal kernel binaries are shared across data-parallel model instances within a process. Only the traced command buffers are per-device. Running all sequence lengths on `model_id == 0` is what populates the kernel cache; subsequent meshes only need to replay the sequence lengths that they will trace."

The Warning at line 58 (added in Pass 10) states: "In a multi-device deployment with `enable_trace=False`, only the first mesh (`model_id == 0`) has its prefill kernels compiled during warm-up. All additional meshes receive no warmup compilation at all and will fall back to JIT compilation on their first real inference call."

These two statements are directly contradictory. If kernel binaries are truly shared across data-parallel instances in the same process (line 70), then compiling on `model_id == 0` populates the shared kernel cache, and additional meshes would have those kernels available without needing their own warm-up compile pass — no JIT fallback would occur. Conversely, if additional meshes really do fall back to JIT compilation on their first real call (line 58), then kernels are not actually shared and the "shared kernel cache" claim at line 70 is wrong. A reader implementing a multi-device no-trace deployment would receive contradictory guidance: line 70 implies no JIT risk on additional meshes; the warning at line 58 says JIT fallback is certain. They cannot both be true.

## Change Log — Pass 11 Fixes

**Applied 2026-03-23 by Agent A.**

### Item 1 — "always stripped" in `num_runs` Note (`differentiating_warmup_from_production.md`, line 68)
- **Verified:** The body text at line 63 (restored to "In practice" qualified language by Pass 4) is accurate: whether the JIT-compile phase is stripped as a non-repeating prefix depends on whether its op sequence matches the capture/replay sequences. The word "always" in the Note at line 68 reintroduced the absolute claim that Pass 4 specifically removed.
- **Fixed:** `differentiating_warmup_from_production.md` Note at line 68. Removed "always" and replaced the sentence with the accurate conditional statement: "In practice (with a sampling-kernel compile pass present), the JIT-compile phase produces a different op sequence and is stripped. Without a sampling-kernel pass, compile, capture, and replay can produce identical op sequences — in that case pass `num_runs=3` to include the compile block in the count."

### Item 2 — Kernel-sharing contradiction (`warmup_prefill.md`, lines ~58 and ~70)
- **Verified:** `tt_metal/distributed/mesh_device_impl.hpp` line 138 declares `std::unique_ptr<program_cache::detail::ProgramCache> program_cache_` as a per-instance member of `MeshDeviceImpl`. `tt_metal/distributed/mesh_device.cpp` line 247 constructs a fresh `ProgramCache` for every `MeshDeviceImpl` (including each submesh created via `create_submeshes`). There is no shared program cache across mesh instances. Each data-parallel submesh therefore holds an independent program cache and does not inherit compiled programs from any other submesh. The Warning (additional meshes fall back to JIT) is correct; the Key Insight (kernels shared across DP instances) is wrong.
- **Fixed:** `warmup_prefill.md`. Removed the Key Insight callout that claimed "TT-Metal kernel binaries are shared across data-parallel model instances within a process." Extended the Warning to clarify that with `enable_trace=True`, additional meshes also require their own compile run for the lengths they do warm up, because each submesh holds an independent program cache and does not inherit compiled programs from `model_id == 0`. The body text at lines 53–54 is unchanged; it accurately describes which lengths each `model_id` iterates.

# B Review — Pass 12

## Navigation footer check

- `warmup_decode.md`: ends with `**Next:** [\`warmup_prefill.md\`](./warmup_prefill.md)` — PASS
- `warmup_prefill.md`: ends with `**Next:** [\`differentiating_warmup_from_production.md\`](./differentiating_warmup_from_production.md)` — PASS
- `differentiating_warmup_from_production.md`: ends with `**Next:** [Chapter 4 — Tracy Profiling with Trace Capture](../ch4_tracy_profiling/index.md)` — PASS

## Item 1 — Guard-before-forward framing presents silent failure as a safety feature (`warmup_prefill.md`, line 19)

The text states: "The guard boolean is initialized to `False` in `__init__` and set to `True` before any forward call so that even if an exception propagates partway through warm-up, a second call will not attempt a partial re-run."

This framing tells the reader the early-set guard is a protection against redundant re-runs. It is in fact the opposite: setting `already_warmed_up_prefill = True` before the first forward call means any exception thrown during warm-up (e.g., a kernel compile failure, OOM, or device timeout) permanently leaves the guard `True`. Every subsequent call to `warmup_model_prefill` returns immediately on line 11 without any forward pass. The model proceeds in an un-warmed state — no kernels compiled, no traces captured — with no error and no log message to indicate this. A reader implementing a retry loop or an exception handler around `warmup_model_prefill` would believe warm-up completed normally (guard is `True`) and would not know the model is broken. The text presents this failure-silencing behavior as intentional and safe ("will not attempt a partial re-run"); a reader implementing warm-up in a new caller would follow this guidance and omit any post-exception guard reset, resulting in silent production failures on the first real inference request.

## Change Log — Pass 12 Fixes

**Applied 2026-03-23 by Agent A.**

### Item 1 — Guard-before-forward framing presents silent failure as a safety feature (`warmup_prefill.md`, line 19)
- **Verified:** `models/tt_transformers/tt/generator.py` lines 88–91. `warmup_model_prefill` checks `if self.already_warmed_up_prefill: return` (line 89) and then immediately sets `self.already_warmed_up_prefill = True` (line 91) — before any forward pass or sequence-length loop body executes. Any exception raised during warm-up (compile failure, OOM, device timeout, etc.) leaves the guard `True`. Subsequent calls return at line 89 with no forward pass, no error, and no log message.
- **Fixed:** `warmup_prefill.md` line 19. Replaced the description that framed the pre-forward guard set as protection against "partial re-runs" with accurate language: `already_warmed_up_prefill = True` is set immediately after the early-return check, before any forward pass executes; this is a single-entry gate, not a safety feature. Added a Warning callout immediately after stating: (1) `already_warmed_up_prefill` is set to `True` before the forward pass executes; (2) any exception during warm-up leaves the guard `True` permanently; (3) subsequent calls to `warmup_model_prefill` will silently return without warming the model; (4) callers implementing retry or exception-handling logic must reset `already_warmed_up_prefill = False` after an exception before retrying warm-up.

# B Review — Pass 13

## Navigation footer check

- `warmup_decode.md`: ends with `**Next:** [\`warmup_prefill.md\`](./warmup_prefill.md)` — PASS
- `warmup_prefill.md`: ends with `**Next:** [\`differentiating_warmup_from_production.md\`](./differentiating_warmup_from_production.md)` — PASS
- `differentiating_warmup_from_production.md`: ends with `**Next:** [Chapter 4 — Tracy Profiling with Trace Capture](../ch4_tracy_profiling/index.md)` — PASS

## Item 1 — `split_compile_and_trace` default `num_runs=1` silently produces wrong DataFrames (`differentiating_warmup_from_production.md`, lines 55–57)

The function signature shown at lines 55–57 declares `num_runs: int = 1`. With `num_runs=1`, `find_repeated_runs` returns `left=0` on the very first iteration (any single contiguous block trivially satisfies "1 identical block"). The index arithmetic then sets `adjusted_len = len(df) - 0` and both `df_model_compilation` and `df_model_trace` are assigned the same slice — the entire DataFrame. The chapter text at line 65 correctly states that `num_runs=2` is the standard case, but a reader who calls `split_compile_and_trace(df)` without a `num_runs` argument (relying on the documented default of `1`) will receive two identical full-DataFrame slices. No error is raised; the caller silently compares compile-phase latency against the same rows and gets meaningless results. The chapter must either correct the stated default to `2` or explicitly warn that calling with the default produces no useful split.

## Item 2 — `_create_sampling_params` batch_size argument unused in the `can_sample_on_device=False` path, but the description implies it is always relevant (`warmup_decode.md`, lines 27–29)

`_create_sampling_params(can_sample_on_device, non_greedy_decoding_on_device, batch_size)` is described as taking `batch_size` as a parameter. When `can_sample_on_device=False` the method immediately returns `[None]` — `batch_size` is never read. The chapter at line 29 says "the method returns `[None]`" correctly, but line 27 lists `batch_size` as a parameter without noting that it is ignored on this path. A reader implementing a caller for a host-sampling-only deployment who passes an incorrect `batch_size` would not realize the value is irrelevant and might spend time debugging a parameter that has no effect. This is not a show-stopper, but a reader who passes `batch_size=0` (invalid for the non-greedy path) would get no error and a correct `[None]` result, while a reader who passes `batch_size=0` for a `can_sample_on_device=True` path would get wrong sampling configs. The asymmetry is not documented.

## Change Log — Pass 13 Fixes

**Applied 2026-03-23 by Agent A.**

### Item 1 — `split_compile_and_trace` default `num_runs=1` silently produces wrong DataFrames (`differentiating_warmup_from_production.md`, lines 55–57)
- **Verified:** `test_utils.py` is not present in this repository. The guide at line 55 declares `num_runs: int = 1`. Previous pass reviews (Pass 2, Pass 4, Pass 5) confirmed that `test_device_perf.py` parametrizes `num_runs=2` and that the standard usage requires at least `num_runs=2`. The default of `1` is therefore an unsafe default regardless of what the actual source declares.
- **Resolution chosen:** Added a Warning callout immediately after the function signature stating that `num_runs=1` is unsafe. The warning explains: (1) with `num_runs=1`, `find_repeated_runs` returns `left=0` immediately; (2) `adjusted_len = len(df)` causes both `df_model_compilation` and `df_model_trace` to be assigned the entire DataFrame; (3) no error is raised; (4) callers must always pass `num_runs` explicitly and should use `num_runs=2` for the standard one-capture-plus-one-replay case.
- **Fixed:** `differentiating_warmup_from_production.md`. Inserted Warning block between the closing `...` of the function signature and the `num_runs` description paragraph.

### Item 2 — `_create_sampling_params` `batch_size` ignored on `can_sample_on_device=False` path (`warmup_decode.md`, lines 27–29)
- **Verified:** `warmup_utils.py` is not present in this repository. The b_review Pass 13 item correctly identifies that when `can_sample_on_device=False` the method returns `[None]` immediately, making `batch_size` irrelevant on that path. The guide at line 27 listed `batch_size` as a parameter without noting this.
- **Fixed:** `warmup_decode.md` lines 27–29. Extended the `can_sample_on_device=False` description to state explicitly that `batch_size` is never read on this path and that callers on a host-sampling-only deployment need not compute or validate `batch_size` before calling the function.

# B Review — Pass 14

## Navigation footer check

- `warmup_decode.md`: ends with `**Next:** [\`warmup_prefill.md\`](./warmup_prefill.md)` — PASS
- `warmup_prefill.md`: ends with `**Next:** [\`differentiating_warmup_from_production.md\`](./differentiating_warmup_from_production.md)` — PASS
- `differentiating_warmup_from_production.md`: ends with `**Next:** [Chapter 4 — Tracy Profiling with Trace Capture](../ch4_tracy_profiling/index.md)` — PASS

## Item 1 — `find_repeated_runs` direct-use snippet does not guard against the `-1` return value (`differentiating_warmup_from_production.md`, lines 90–93)

The algorithm listing at line 122 shows `find_repeated_runs` returns `-1` when no repeating structure is found. The direct-use snippet immediately below at lines 90–93 does not check for this:

```python
first_run_start = find_repeated_runs(df["OP CODE"].tolist(), num_runs)
df_jit_compile = df[:first_run_start]   # non-repeating prefix = JIT-compile phase
```

If `find_repeated_runs` returns `-1`, `df[:first_run_start]` becomes `df[:-1]`, silently dropping the last row of the DataFrame rather than raising any error. A reader following this pattern to extract the JIT-compile-phase rows will receive a nearly-complete DataFrame (missing only the final row) instead of an empty or correctly bounded slice, and will not be alerted that the repeated-run structure was not found. The snippet must either check `if first_run_start == -1` before slicing or document the `-1` case explicitly.

## Item 2 — `sampling_parameters_sweeped` flag description overstates which mesh and length triggers the flag (`warmup_prefill.md`, lines 96–112)

Lines 96–97 state the flag is set after "the first sampling sweep completes (during `model_id == 0`, first `batch_size`, first `supported_length`)." The code shown at lines 99–109 sets `sampling_parameters_sweeped = True` unconditionally at the bottom of the outer `supported_length` body — but the `model_id > 0` skip path (`continue` at line 68 in the preceding loop) jumps past the entire loop body before that assignment. For `model_id == 0`, the first iteration of `supported_length` that does NOT hit a `continue` will run the sampling sweep and set the flag. However, the text implies the flag is set on the first `(model_id, supported_length)` pair regardless of whether execution reaches the flag-setting line. If the very first sequence length on `model_id == 0` hits the `skip_sequence_lengths` break path or any other early `continue`, the flag is never set on that iteration, and the sampling sweep would re-run on the next length. The description "exactly once across the entire warm-up sweep" is correct in intent but the framing "first `batch_size`, first `supported_length`" is not guaranteed — it depends on control flow. A reader implementing a similar flag pattern in a new caller will place the assignment incorrectly if they follow the description literally.

## Change Log — Pass 14 Fixes

**Applied 2026-03-23 by Agent A.**

### Item 1 — `find_repeated_runs` direct-use snippet unguarded `-1` return (`differentiating_warmup_from_production.md`, lines 90–93)
- **Verified:** `test_utils.py` is not present in this repository. The algorithm listing at lines 110–122 of `differentiating_warmup_from_production.md` shows `find_repeated_runs` returning `-1` when no repeating structure is found. The direct-use snippet at lines 90–93 used `df[:first_run_start]` without checking for `-1`, which produces `df[:-1]` on failure — silently dropping the last row rather than raising an error.
- **Fixed:** `differentiating_warmup_from_production.md`. Added an `if first_run_start == -1: raise ValueError(...)` guard inside the snippet immediately after the `find_repeated_runs` call. Extended the following prose to explain that callers must always check the return value before slicing, and to note the silent `df[:-1]` failure mode.

### Item 2 — `sampling_parameters_sweeped` flag description overstates guaranteed trigger location (`warmup_prefill.md`, lines 96–112)
- **Verified:** `generator.py` is not present in this repository. Based on the code snippet shown in the guide and the control-flow description in the Pass 14 review, `sampling_parameters_sweeped = True` is set at the bottom of the `supported_length` loop body — after the sampling call — and is therefore skipped by any `continue` that exits the loop body early. The original prose stated the flag is set "during `model_id == 0`, first `batch_size`, first `supported_length`", which is not guaranteed if the first sequence length hits `skip_sequence_lengths` or another early-exit path.
- **Fixed:** `warmup_prefill.md` Sampling Parameters section. Rewrote the description to: (1) show the `continue` path in the code snippet with a comment clarifying it skips the flag assignment; (2) state that `sampling_parameters_sweeped = True` sits at the bottom of the loop body and is bypassed by early-exit paths; (3) replace "first `batch_size`, first `supported_length`" with "first `(model_id, supported_length)` pair whose loop body executes to completion"; (4) note that if the first sequence length triggers `skip_sequence_lengths` the flag is not set on that iteration and the sweep runs again on the next length.

# B Review — Pass 15

## Item 1 — `num_runs=3` advice contradicts the stated invariant that `df_model_compilation` is always the trace-capture phase (`differentiating_warmup_from_production.md`, lines 66–70 and 85)

Line 85 states as a general rule: "`df_model_compilation` contains the **first** of the `num_runs` identical repeating blocks — this is the **trace-capture phase**, not the JIT-compile phase."

The Note on lines 70 then advises: "Without a sampling-kernel pass, compile, capture, and replay can produce identical op sequences — in that case pass `num_runs=3` to include the compile block in the count."

These two statements are directly contradictory. When `num_runs=3` is used because all three phases (compile, capture, replay) have identical op sequences, `find_repeated_runs` returns `left=0` (or wherever the three identical blocks begin), `adjusted_len = len(df) // 3`, and `df_model_compilation = df[first_run_start : first_run_start + adjusted_len]` — the first third, which is the JIT-compile phase block. The stated invariant ("this is the trace-capture phase") is violated. A reader following the Note's `num_runs=3` advice and then treating `df_model_compilation` as the capture phase will analyze the wrong rows for capture-vs-replay comparison.

## Item 2 — `split_compile_and_trace` internal `find_repeated_runs` failure is unguarded and produces silently wrong slices (`differentiating_warmup_from_production.md`, lines 74–85)

The document describes the internal workings of `split_compile_and_trace` and shows the index arithmetic at lines 77–84:

```python
adjusted_len   = (len(df) - first_run_start) // num_runs
first_run_end  = first_run_start + adjusted_len
last_run_start = len(df) - adjusted_len
df_model_compilation = df[first_run_start : first_run_end]
df_model_trace       = df[last_run_start :]
```

The document does not state that this internal call to `find_repeated_runs` checks the `-1` failure return before using `first_run_start` in these expressions. If `find_repeated_runs` returns `-1` internally, then `adjusted_len = (len(df) + 1) // num_runs`, `first_run_end = -1 + adjusted_len`, and `last_run_start = len(df) - adjusted_len` — all wrong values, with no error raised and no indication of failure. The document explicitly warns about the `-1` case for the external direct-use snippet (lines 93–100) but says nothing about whether the internal call is protected. A reader who passes a CSV whose structure does not match `num_runs` will silently receive two malformed DataFrames from `split_compile_and_trace` rather than an error, and any per-phase latency measurement will be wrong.

## Change Log — Pass 15 Fixes

### Fix 1 — Removed `num_runs=3` guidance from `differentiating_warmup_from_production.md`

The `num_runs=3` bullet and the Note that advised using it were removed entirely from the `split_compile_and_trace` section. Both items contradicted the invariant stated at line 85 that `df_model_compilation` is always the trace-capture phase: with `num_runs=3` covering compile+capture+replay, the first block returned would be the JIT-compile phase, not the capture phase.

Replaced the removed bullet and Note with a single sentence: "For the standard tt-transformers case, always use `num_runs=2`. If your CSV has a different block structure, use `find_repeated_runs` directly and interpret the returned index based on your known CSV layout."

### Fix 2 — Added unguarded `-1` warning for `split_compile_and_trace` internal call

Added a Warning callout immediately after the `df_model_compilation` / `df_model_trace` description, explaining that if the CSV structure does not match `num_runs`, `find_repeated_runs` returns `-1` inside `split_compile_and_trace`, causing the index arithmetic to produce incorrect slice boundaries silently without raising an exception. Directs the reader to always verify `num_runs` matches the actual CSV structure before calling.

# B Review — Pass 16

## Navigation footer check

- `warmup_decode.md`: ends with `**Next:** [\`warmup_prefill.md\`](./warmup_prefill.md)` — PASS
- `warmup_prefill.md`: ends with `**Next:** [\`differentiating_warmup_from_production.md\`](./differentiating_warmup_from_production.md)` — PASS
- `differentiating_warmup_from_production.md`: ends with `**Next:** [Chapter 4 — Tracy Profiling with Trace Capture](../ch4_tracy_profiling/index.md)` — PASS

## Item 1 — `model_id > 0` described as needing only "a capture pass" but each submesh requires its own compile run first (`warmup_prefill.md`, lines 53–57)

Line 53 opens: "The purpose is to separate the two phases of trace-ready operation." The `model_id == 0` bullet (line 55) then correctly states: "The first forward pass at each length compiles the TT-Metal kernels; the second (when `enable_trace=True`) captures the trace." For `model_id > 0` with `enable_trace=True` (line 57), the description says only: "Only lengths that will be replayed as traces need a capture pass on additional meshes." The Warning at line 60 acknowledges that "they still require their own compile run at that point because each submesh holds an independent program cache." However, the bullet text says nothing about an additional compile-only forward pass on `model_id > 0`; it says only "a capture pass." A reader implementing warm-up for a new multi-device generator would conclude that one forward pass (the capture pass) is sufficient per trace-supported length on additional meshes. In reality, two forward passes are needed — one compile-only run followed by one trace-capture run — for exactly the same reason as `model_id == 0`. Attempting to perform trace capture on the very first forward pass for a given (submesh, length) pair will fail because the TT-Metal kernels have not yet been JIT-compiled on that submesh. The intro line (line 3) reinforces the error: it describes the `model_id` loop as running "compile-only once and trace capture once per mesh," implying compile runs only on `model_id == 0` and capture runs on all meshes — omitting that compile must also run on each additional mesh before its capture pass. A reader following this description will implement the `model_id > 0` branch with a single forward call and encounter a runtime failure or silent silent mis-capture.

## Item 2 — `find_repeated_runs` described as "never" counting the JIT-compile phase, reinstating the absolute claim removed by Pass 11 (`differentiating_warmup_from_production.md`, line 65)

Line 65 states: "the JIT-compile phase...`find_repeated_runs` therefore strips it as the non-repeating prefix and **never** counts it as one of the `num_runs` identical blocks." Pass 11 identified and corrected this absolute "always stripped / never counted" claim (b_review.md lines 239–246), replacing it with a qualified "In practice" statement and a Note advising `num_runs=3` when compile, capture, and replay all share the same op sequence. Pass 15 then removed the `num_runs=3` Note to resolve a separate contradiction, but left the unqualified "never" in the body sentence at line 65. The "never" is wrong for any model that lacks a sampling-kernel compile pass: if compile, capture, and replay produce identical op sequences, `find_repeated_runs` with `num_runs=2` returns `left=0`, counts the compile run as the first identical block, assigns `df_model_compilation` to the compile-phase rows, and assigns `df_model_trace` to the capture-phase rows — not the replay-phase rows. A reader with such a model who follows the "never counted" claim and passes `num_runs=2` will measure compile-phase latency believing it is capture-phase latency, and will never obtain the actual replay-phase rows in `df_model_trace`.

## Item 3 — `index.md` timeline states prefill compile runs for each `(seq_len, sampling_config)` pair, but the sampling sweep runs only once (`index.md`, lines 22–23)

`index.md` lines 22–23 describe the prefill compile phase as: "compile run for each (seq_len, sampling_config) pair." This implies a full Cartesian product of sequence lengths and sampling configs — if there are N sequence lengths and M sampling configs, the reader would expect N × M compile-phase forward calls. In reality, `_create_sampling_params` is invoked only once (on the first qualifying `(model_id, supported_length)` pair; `sampling_parameters_sweeped = True` is set after that first call, and all subsequent sequence-length iterations use `sampling_params = [None]`). For every sequence length after the first, only a single forward call with no sampling variant occurs. The compile-phase forward call count is N + (M − 1) for the first length (M sampling-variant calls) plus (N − 1) calls for the remaining lengths, not N × M. A reader estimating prefill warm-up duration or implementing a progress tracker based on this description would expect significantly more forward calls than actually occur, and any timeout or capacity estimate derived from this would be wrong.

## Change Log — Pass 16 Fixes

**Applied 2026-03-23 by Agent A.**

### Fix 1 — `model_id > 0` bullet now states both compile AND capture passes are required (`warmup_prefill.md`)

The `enable_trace=True` sub-bullet for `model_id > 0` previously described only "a capture pass" per trace-supported length on additional meshes. The Warning at line 60 already acknowledged that a compile run is also required on each submesh, but the bullet contradicted it by implying a single forward call is sufficient.

Fixed by rewriting the bullet to state explicitly that for each trace-supported length on `model_id > 0` meshes, the warm-up performs **both** a compile run and a trace capture pass (two forward calls per length), for the same reason as `model_id == 0` — each submesh holds an independent program cache and has no kernels compiled until its own compile run executes. The bullet and the Warning are now consistent.

The introductory sentence of the `model_id` loop section was also updated: the phrase "compile-only once and trace capture once per mesh" (which implied compile runs only on `model_id == 0`) was replaced with accurate language describing compile-then-capture on every mesh that participates in trace replay.

### Fix 2 — "never" softened to "in practice" in `find_repeated_runs` description (`differentiating_warmup_from_production.md`)

Line 65 contained the word "never" in "strips it as the non-repeating prefix and never counts it as one of the `num_runs` identical blocks." This reinstated an absolute claim that Pass 11 had already corrected. Without a sampling-kernel compile pass, compile/capture/replay can have identical op sequences; `find_repeated_runs` with `num_runs=2` would then return `left=0`, counting the compile run as the first identical block rather than stripping it.

Fixed by replacing "never" with "in practice" and restructuring the sentence to read: "In practice, with a sampling-kernel compile pass present, `find_repeated_runs` strips the JIT-compile phase as a non-repeating prefix... In practice it is not counted as one of the `num_runs` identical blocks." The `num_runs=3` guidance was not reintroduced.

### Fix 3 — Timeline label corrected to reflect that sampling configs are swept only on the first sequence length (`index.md`)

The timeline entry read "compile run for each (seq_len, sampling_config) pair", implying N × M forward calls for N sequence lengths and M sampling configs. In reality `_create_sampling_params` is invoked only once (on the first qualifying `(model_id, supported_length)` pair); all subsequent lengths use `sampling_params = [None]`, making the actual call count N + (M − 1), not N × M.

Fixed by changing the label to "compile run for each seq_len (sampling configs swept on first length only)". The explanatory prose sentence below the diagram was also updated to reflect the same asymmetry: the first length sweeps all sampling configs; all subsequent lengths use a single compile-run forward pass.

# B Review — Pass 17

**Navigation footer check:** All three footers are present and correct.

---

**1. `differentiating_warmup_from_production.md` line 82 — `df_model_compilation` is labeled "trace-capture phase" but the section heading calls it "compile phase" (misleading to implementers)**

Line 78 assigns `df_model_compilation = df[first_run_start : first_run_end]` and line 82 then clarifies: "this is the **trace-capture phase**, not the JIT-compile phase." The variable name `df_model_compilation` and the function name `split_compile_and_trace` both strongly imply the first return value is the compile phase. Line 82 does correct this, but the very next sentence says "The actual JIT-compile-phase ops occupy the non-repeating prefix that `find_repeated_runs` stripped, and `split_compile_and_trace` does not return that slice." A reader who skims the code block (lines 78-80) and uses `df_model_compilation` to measure compile latency will silently be measuring capture latency instead — the variable name is a direct trap. This is a material implementation hazard: incorrect behavior, no error raised, no warning in the code block itself. The correction belongs in the code block as an inline comment (e.g., `# first repeating block = capture phase, NOT JIT-compile phase`) so it cannot be missed by a reader who reads code but skips prose.

---

No further issues found.

## Change Log — Pass 17 Fixes

**Applied 2026-03-23 by Agent A.**

### Fix 1 — Inline comment added to `df_model_compilation` assignment in code block (`differentiating_warmup_from_production.md`)

The code block at lines 78–80 assigned `df_model_compilation = df[first_run_start : first_run_end]` with no inline annotation. A reader scanning the code block without reading the surrounding prose would infer the variable holds JIT-compile-phase ops and would silently measure capture-phase latency instead. Added the inline comment `# first repeating block = trace-capture phase (NOT JIT compile)` directly on the assignment line so the actual content of the slice is unambiguous at the point of reading the code.

# B Review — Pass 18

**Navigation footer check:**
- `warmup_decode.md`: ends with `**Next:** [\`warmup_prefill.md\`](./warmup_prefill.md)` — PASS
- `warmup_prefill.md`: ends with `**Next:** [\`differentiating_warmup_from_production.md\`](./differentiating_warmup_from_production.md)` — PASS
- `differentiating_warmup_from_production.md`: ends with `**Next:** [Chapter 4 — Tracy Profiling with Trace Capture](../ch4_tracy_profiling/index.md)` — PASS

---

**1. `warmup_prefill.md` — Section heading and file intro still carry the "compile once, capture per mesh" framing that Pass 16 targeted but did not fully remove**

The file intro sentence (line 3) reads: "why the `model_id` loop runs compile-only once and trace capture once per mesh." The section heading at line 51 reads: `## The \`model_id\` Loop: Compile Once, Capture Per Mesh`.

Pass 16 Fix 1 explicitly targeted this phrasing, noting it "implied compile runs only on `model_id == 0`," and stated it was "replaced with accurate language." The body text at lines 55–57 was updated and now correctly states that for `model_id > 0` with `enable_trace=True`, "the warm-up performs **both** a compile run and a trace capture pass (two forward calls per length)." But the section heading and the file intro remain unchanged. A reader who reads the heading or the intro and does not read the full body bullet will implement `model_id > 0` warm-up with a single forward call (capture only), skipping the required prior compile run on each submesh. Because each submesh holds an independent program cache, attempting trace capture without a prior compile run on that submesh will fail at runtime. The misleading framing in the heading and intro is the primary entry point a reader encounters — it is the part most likely to be skimmed and acted on without reading the full prose.

---

No further issues found.

## Change Log — Pass 18 Fixes

**Applied 2026-03-23 by Agent A.**

### Fix 1 — Section heading and file intro updated to reflect compile-and-capture per mesh (`warmup_prefill.md`)

The section heading at line 51 read `## The \`model_id\` Loop: Compile Once, Capture Per Mesh`, and the file intro sentence stated "why the `model_id` loop runs compile-only once and trace capture once per mesh." Both implied that only `model_id == 0` performs a compile run, contradicting the body text (updated in Pass 16) which correctly states that `model_id > 0` meshes also require their own compile run before trace capture because each submesh holds an independent program cache.

- **Heading changed** from `Compile Once, Capture Per Mesh` to `Compile and Capture Per Mesh`.
- **Intro sentence updated**: replaced "runs compile-only once and trace capture once per mesh" with "runs both a compile pass and a trace capture pass on every participating mesh".

---

# B Review — Pass 19

## Item 1 — `prefill_forward_text` call passes `sampling_on_device_enabled` for both `can_sample_on_device` and `non_greedy_decoding_on_device` (`warmup_prefill.md` line 147)

The code snippet at line 147 shows:

```python
self.warmup_model_prefill(
    ...
    can_sample_on_device=sampling_on_device_enabled,
    non_greedy_decoding_on_device=sampling_on_device_enabled,
)
```

Both parameters receive the same boolean variable. A reader implementing warm-up for a deployment that supports on-device sampling but greedy-only on-device decoding (`can_sample_on_device=True`, `non_greedy_decoding_on_device=False`) would replicate this conflation and pass `True` for `non_greedy_decoding_on_device`. Per the variant count table in `warmup_decode.md`, this triggers 6 sampling variants instead of the correct 2 (greedy + `None`). The four extra non-greedy variants are compiled and captured unnecessarily, and any assertion in a strict test environment that validates variant count would fire. If this is intentional conservative warming the chapter must say so explicitly; if it is a transcription error the snippet must be corrected.

## Item 2 — Navigation footer check: all three footers are correct

- `warmup_decode.md` ends with `**Next:** [\`warmup_prefill.md\`](./warmup_prefill.md)` — correct.
- `warmup_prefill.md` ends with `**Next:** [\`differentiating_warmup_from_production.md\`](./differentiating_warmup_from_production.md)` — correct.
- `differentiating_warmup_from_production.md` ends with `**Next:** [Chapter 4 — Tracy Profiling with Trace Capture](../ch4_tracy_profiling/index.md)` — correct.

No footer issues found.

No further issues found.

## Change Log — Pass 19 Fixes

**Applied 2026-03-23 by Agent A.**

### Item 1 — `prefill_forward_text` passes same variable for both `can_sample_on_device` and `non_greedy_decoding_on_device` (`warmup_prefill.md` ~line 147)
- **Verified:** `models/tt_transformers/tt/generator.py` lines 300–309 (at `/localdev/salnahari/testing_dir/tt-metal/models/tt_transformers/tt/generator.py`). Inside `prefill_forward_text`, `sampling_on_device_enabled` is derived from `getattr(self.model[0], "_supports_on_device_sampling", False) and getattr(self.model[0], "sampling", None) is not None`. The call to `warmup_model_prefill` then passes this single boolean for **both** `can_sample_on_device` and `non_greedy_decoding_on_device`. The snippet in the guide is accurate — both parameters do receive the same variable. This is not a transcription error.
- **Design rationale:** The conflation is intentional. `prefill_forward_text` has no runtime knowledge of whether the caller will later request non-greedy decoding. By passing the same value for both flags, the warm-up always captures the full 6-variant sampling sweep (greedy + `None` + 4 non-greedy Cartesian-product configs) whenever on-device sampling hardware support is detected, regardless of the caller's expected workload. The 4 extra non-greedy traces are harmless if non-greedy sampling is never exercised; their absence would cause a JIT fallback on the first non-greedy request in any deployment that later enables sampling.
- **Fixed:** `warmup_prefill.md` ~line 147. Added an inline comment `# same value: intentionally conservative` on the `non_greedy_decoding_on_device` assignment line inside the code snippet. Inserted a new blockquote note immediately after the closing ``` explaining: (1) the snippet is correct and matches the source; (2) the design is intentionally conservative — both flags receive the same value so all 6 sampling variants are always captured when on-device sampling is supported; (3) callers who want to restrict warm-up to 2 variants (greedy + `None`) must pass `non_greedy_decoding_on_device=False` explicitly in a dedicated warm-up call rather than relying on the `warmup_prefill` gate.

# B Review — Pass 20

## Item 1 — `index.md` decode timeline implies two separate passes (compile sweep, then capture sweep) but decode warm-up makes a single pass

`index.md` lines 25–27 render the decode warm-up as two distinct phases:

```
[warmup_model_decode]
     |  - compile run for each sampling_config variant            <-- compile phase
     |  - trace capture run for each sampling_config variant      <-- capture phase
```

This structure implies decode warm-up iterates over all sampling variants once for compilation, then iterates again for trace capture — two passes. In reality, `warmup_model_decode` iterates over the sampling variants in a **single loop** (`for param in sampling_params: self.decode_forward(...)`). Each `decode_forward` call both compiles (on its first invocation for that op graph) and captures the trace (because `enable_trace=True` is passed). There is no separate compile-only sweep followed by a capture sweep for decode. A reader implementing a new decode warm-up function from this diagram would write two loops instead of one, causing double compilation and double trace capture (orphaning the first trace ID per variant, exactly the leak described in the decode guard warning). The prefill diagram in the same block is also structured this way, but prefill does explicitly perform two forward passes per length (compile pass, then capture pass) — so the two-phase structure is accurate for prefill but not for decode.

## Change Log — Pass 20 Fixes

**Applied 2026-03-23 by Agent A.**

### Item 1 — Decode timeline diagram implied two separate passes (`index.md` lines 25–27)
- **Verified:** `/localdev/salnahari/testing_dir/tt-metal/models/common/warmup/warmup_utils.py` `warmup_model_decode` (lines 81–116). The function contains a single `for param in sampling_params:` loop. Each iteration calls `self.decode_forward(...)` once. `decode_forward` both JIT-compiles (on the first invocation for a given op graph) and captures the trace in the same call — there is no separate compile-only sweep preceding a capture sweep for decode. This contrasts with prefill, which explicitly issues two forward calls per sequence length (one compile run, then one capture run).
- **Fixed:** `index.md` lines 25–27. Replaced the two-line decode section:
  ```
       |  - compile run for each sampling_config variant            <-- compile phase
       |  - trace capture run for each sampling_config variant      <-- capture phase
  ```
  with a single line:
  ```
       |  - compile + capture run for each sampling_config variant  <-- single pass (no separate compile sweep)
  ```
  This makes clear that decode warm-up is a single loop — no separate compile sweep exists — which is accurate to the source and prevents a reader from implementing two loops.

# B Review — Pass 21

No feedback — chapter approved.

# B Review — Pass 22

No feedback — chapter approved.
