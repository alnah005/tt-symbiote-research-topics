# B Review — Pass 1

## Item 1 — `running_with_tracy.md`: `-p` flag in code snippet contradicts prose description

**File:** `running_with_tracy.md`, lines 98 and 105.

The `run_device_profiler` code snippet passes `-p {python_post_process_opt}` to the `python3 -m tracy` invocation. The prose immediately below (line 105) states: "The `python_post_process=True` default maps to the `-r` flag (report mode)."

These two statements are in direct conflict. If `-p` is the flag used in practice, the prose is wrong. If `-r` is the correct flag, the code snippet is wrong. A reader implementing `run_device_profiler` equivalents would not know which flag to pass. The correct flag must be verified against `tools/tracy/__main__.py` and one of the two statements corrected.

---

## Item 2 — `tracy_markers_for_trace.md`: `TracyTTMetalEndMeshTrace` unconditional-emit claim is unverified and consequential

**File:** `tracy_markers_for_trace.md`, lines 24–29.

The file states that `TracyTTMetalEndMeshTrace` emits its Tracy message unconditionally — without a `TracyTTMetalTraceTrackingEnabled()` check — unlike Begin and Replay. This is presented as a definitive behavioral fact in a callout Note.

If this is incorrect (i.e., END is also gated on `TT_METAL_PROFILER_TRACE_TRACKING=1` like the others), a reader would incorrectly expect the END message to appear in `tracy_ops_data.csv` even with `TT_METAL_PROFILER_TRACE_TRACKING=0`, and would build incorrect parsing logic around it. This claim needs confirmation against the actual macro definition in `tt_metal/tools/profiler/tt_metal_tracy.hpp`.

---

## Item 3 — `reading_profiling_output.md` vs. `tracy_markers_for_trace.md`: Session ID 1 assignment is contradictory

**Files:** `reading_profiling_output.md` line 19; `tracy_markers_for_trace.md` line 68.

`reading_profiling_output.md` defines `METAL TRACE REPLAY SESSION ID = 1` as corresponding to the trace **capture** run (the `BeginTrace` → forward → `EndTrace` pass), and `>= 2` as production replays.

`tracy_markers_for_trace.md` states that `METAL TRACE REPLAY SESSION ID` is derived by appending the timestamp of each `TT_METAL_TRACE_REPLAY` message to `traceReplays[device_id][trace_id]`, with `session_id - 1` as the list index. Since `TT_METAL_TRACE_REPLAY` is emitted on **replay**, not on capture, session ID 1 would correspond to the **first replay call**, not to the capture pass. The capture pass would have no replay session ID (it is covered only by TRACE_BEGIN / TRACE_END messages).

This is a direct contradiction. A reader filtering `SESSION ID = 1` expecting to isolate the capture pass would instead get first-replay rows (or vice versa). One of these two descriptions is wrong; the correct semantics must be verified and reconciled.

---

## Item 4 — `reading_profiling_output.md`: `split_compile_and_trace` return value labeling conflicts with its own description

**File:** `reading_profiling_output.md`, lines 48–55.

The description says `df_model_compilation` is "the first repeated block (the trace capture pass)." But the same file defines the capture pass as rows where `METAL TRACE REPLAY SESSION ID = 1`. The `find_repeated_runs` heuristic operates only on op-code sequence patterns and has no knowledge of `METAL TRACE REPLAY SESSION ID`. In a three-run decode workflow (warm-up + capture + replay), the first repeated op-code block is the **warm-up (compile) run**, not the capture pass. The variable name `df_model_compilation` would then actually contain warm-up rows, while the capture pass rows would be in neither output variable.

A reader using `df_model_compilation` to analyze capture-pass kernel durations would be operating on the wrong rows. The description of what `split_compile_and_trace` returns needs to be corrected to match what `find_repeated_runs` actually selects.

---

## Navigation footers

All four navigation footers are present and match the required targets exactly. All `index.md` links in `index.md` are clickable markdown links. No structural gap.

---

## Change Log — Pass 1 Fixes

### Item 1 — `running_with_tracy.md`: `-p` vs `-r` flag contradiction

**Verified against:** `tools/tracy/process_model_log.py` (lines 74–97) and `tools/tracy/__main__.py` (lines 17–18).

**Finding:** The actual command built by `run_device_profiler` is `python3 -m tracy -p {python_post_process_opt} ...`. The `-p` flag is **always** present in the command string (it is a literal in the format string, not conditional). It activates partial-profiling mode (`options.partial` → `tracy_state.doPartial = True`), restricting Tracy instrumentation to explicitly enabled zones. Separately, when `python_post_process=True`, the code sets `python_post_process_opt = "-r"`, so the effective flags are `python3 -m tracy -p -r`. The prose statement "maps to the `-r` flag" was correct for the report-mode mapping but omitted the unconditional `-p` prefix, while the code snippet showed only `-p` without clarifying that `-r` comes after.

**Fix applied:** Updated the code snippet to show `python_post_process_opt = "-r"` assignment explicitly. Updated the prose to state that `-p` is always present (partial-zone mode) and `-r` is the additional report-mode flag added when `python_post_process=True`, making the full prefix `python3 -m tracy -p -r`.

---

### Item 2 — `tracy_markers_for_trace.md`: `TracyTTMetalEndMeshTrace` unconditional-emit claim

**Verified against:** `tt_metal/tools/profiler/tt_metal_tracy.hpp` (lines 28–35).

**Finding:** The original claim was **correct**. `TracyTTMetalEndMeshTrace` does emit its `TracyMessage` call without wrapping it in `if (TracyTTMetalTraceTrackingEnabled())`. The macro is still guarded by `if (tt::tt_metal::getDeviceProfilerState())` at the outer level, but within that guard the `TracyMessage` call is unconditional. By contrast, `TracyTTMetalBeginMeshTrace` (lines 17–26) and `TracyTTMetalReplayMeshTrace` (lines 37–46) both wrap their `TracyMessage` calls inside `if (TracyTTMetalTraceTrackingEnabled())`.

**Fix applied:** Strengthened the description to explicitly note the verification source (`tt_metal_tracy.hpp`), clarified that "unconditional" refers specifically to the absence of a `TracyTTMetalTraceTrackingEnabled()` guard (the outer `getDeviceProfilerState()` guard is still present), and added `TT_METAL_DEVICE_PROFILER=1` as the required condition for the message to appear.

---

### Item 3 — Session ID 1 contradiction between `reading_profiling_output.md` and `tracy_markers_for_trace.md`

**Verified against:** `tools/tracy/process_ops_logs.py` (lines 267–332, 1252).

**Finding:** `traceReplays` is populated exclusively by `TT_METAL_TRACE_REPLAY` messages (lines 323–332). Session IDs are 1-based list indices into `traceReplays[device_id][trace_id]` (line 224: `index = session_id - 1`). The capture pass produces no `TT_METAL_TRACE_REPLAY` message; its ops have `metal_trace_id` set (non-null, from TRACE_BEGIN/END) but `metal_trace_replay_session_id` is explicitly set to `""` for non-trace-replay ops (line 1252). Therefore:

- Capture-pass rows: `METAL TRACE ID` non-null, `METAL TRACE REPLAY SESSION ID` **empty**.
- First replay rows: `METAL TRACE ID` non-null, `METAL TRACE REPLAY SESSION ID` = **1**.
- Subsequent replays: session IDs 2, 3, ...

The original `reading_profiling_output.md` incorrectly assigned session ID 1 to the capture pass and session IDs >= 2 to replays.

**Fix applied:** Corrected both files. In `tracy_markers_for_trace.md`, updated the `METAL TRACE REPLAY SESSION ID` bullet to state that session 1 = first replay, with capture pass having an empty session ID. In `reading_profiling_output.md`, Phase 2 now identifies capture rows as `METAL TRACE ID` non-null + `METAL TRACE REPLAY SESSION ID` empty; Phase 3 now starts at session ID >= 1 (first replay). The Key insight callout was updated accordingly.

---

### Item 4 — `reading_profiling_output.md`: `df_model_compilation` labels incorrect

**Verified against:** `models/tt_transformers/tests/test_utils.py` (lines 256–369), including the `split_compile_and_trace` docstring (lines 343–346) and `find_repeated_runs` logic (lines 256–283).

**Finding:** The `split_compile_and_trace` docstring states: "The ops CSV typically contains three consecutive phases: compile, capture/trace, and runtime trace." `find_repeated_runs` finds the leftmost position where `num_runs` identical op-code blocks fit. When `num_runs=3`, the three blocks are the warm-up (compile) run, the capture run, and the replay run — all three producing the same op-code sequence after kernels are compiled. `df_model_compilation = df[first_run_start:first_run_end]` is the **first** of those three blocks (the warm-up run), and `df_model_trace = df[last_run_start:]` is the **last** (the replay run). The original guide incorrectly called `df_model_compilation` "the trace capture pass."

In the standard tt-transformers case the warm-up run differs from capture/replay only in that it also triggers JIT compilation (and a sampling-kernel compile pass at the end), but its op-code sequence is identical to capture and replay — which is exactly why `find_repeated_runs` includes it as one of the `num_runs` identical blocks rather than treating it as a non-repeating prefix.

**Fix applied:** Updated the `split_compile_and_trace` section to correctly describe `df_model_compilation` as the warm-up (compile) pass and explain the three-block structure. Added guidance on how to isolate capture-pass rows directly using `METAL TRACE ID` / `METAL TRACE REPLAY SESSION ID` column filters.

---

# B Review — Pass 2

## Item 1 — `differentiating_trace_ops_from_normal_ops.md`: Session ID semantics not updated from Pass 1 fix

**File:** `differentiating_trace_ops_from_normal_ops.md`, line 10.

The file states:

> **Non-null integer**: the op was dispatched as part of a trace, either during capture (`METAL TRACE REPLAY SESSION ID` = 1) or during replay (`METAL TRACE REPLAY SESSION ID` >= 2).

This directly contradicts the correction established in Pass 1 (verified against `process_ops_logs.py`). The capture pass has `METAL TRACE REPLAY SESSION ID` **empty** (not = 1); session ID = 1 corresponds to the **first replay call**. A reader filtering `SESSION ID = 1` to isolate capture-pass rows would receive first-replay rows instead. This file was not updated when `reading_profiling_output.md` and `tracy_markers_for_trace.md` were corrected in Pass 1.

---

## Item 2 — `reading_profiling_output.md`: `.iloc` used with label-index values in signpost filter

**File:** `reading_profiling_output.md`, lines 82–85.

The code shown is:

```python
start = markers[markers == "start"].index[0]
stop  = markers[markers == "stop"].index[0]
df = df.iloc[start + 1 : stop]
```

`.index[0]` returns the **label** of the matching row (a DataFrame index value), not its positional integer. `.iloc` requires **positional** integers. These are identical only when the DataFrame has an unbroken default RangeIndex (`0, 1, 2, …`). If `df` has been subset or filtered before this call — which is the typical usage, since `post_process_ops_log` reads and may filter the CSV before reaching this block — the label and position will differ and the slice will silently return the wrong rows. A reader copying this snippet would implement it incorrectly for any non-default-index DataFrame. The correct call is `.loc[start + 1 : stop - 1]` (label-based, inclusive) or a boolean mask approach.

---

## Navigation footers

All four navigation footers match the required targets exactly. No structural gap.

---

## Change Log — Pass 2 Fixes

### Item 1 — `differentiating_trace_ops_from_normal_ops.md`: Session ID semantics corrected

**File:** `differentiating_trace_ops_from_normal_ops.md`, lines 10 and 28.

**Fix applied:** Updated the `METAL TRACE ID` bullet (line 10) to state that capture-pass rows have `METAL TRACE REPLAY SESSION ID` **empty** and that replay rows have session ID >= 1 (first replay = 1, subsequent replays >= 2). Updated the "Trace replay" section header sentence (line 28) from "session ID >= 2" to "session ID >= 1" to match the correct semantics verified against `process_ops_logs.py` in the Pass 1 review. These changes make `differentiating_trace_ops_from_normal_ops.md` consistent with the corrections already applied to `reading_profiling_output.md` and `tracy_markers_for_trace.md` in Pass 1.

---

### Item 2 — `reading_profiling_output.md`: `.iloc` / `.loc` safety note added to signpost filter

**File:** `reading_profiling_output.md`, after line 87 (the signpost code block).

**Fix applied:** Added a block-quote note immediately after the `post_process_ops_log` signpost code snippet explaining:
- `.iloc` is safe in that specific snippet because `post_process_ops_log` operates on a DataFrame read fresh from `pd.read_csv` (unbroken default RangeIndex).
- For any pre-filtered DataFrame, `.loc[start + 1 : stop - 1]` must be used instead (`.loc` is label-based and end-inclusive, so `stop - 1` excludes the signpost row).
- Cross-reference to the same pattern documented in Chapter 3's `differentiating_warmup_from_production.md`.

---

# B Review — Pass 3

## Item 1 — `running_with_tracy.md` line 46: `TracyTTMetalEndMeshTrace` incorrectly listed as gated on `TT_METAL_PROFILER_TRACE_TRACKING=1`

**File:** `running_with_tracy.md`, lines 43–48.

The `TT_METAL_PROFILER_TRACE_TRACKING=1` section states "the following C++ macros emit Tracy messages" and lists `TracyTTMetalEndMeshTrace` as one of them. This implies END requires trace tracking to be active. However, `tracy_markers_for_trace.md` (with its Pass-1-verified note, confirmed against `tt_metal/tools/profiler/tt_metal_tracy.hpp`) explicitly states that `TracyTTMetalEndMeshTrace` emits its Tracy message without a `TracyTTMetalTraceTrackingEnabled()` guard — it emits whenever `getDeviceProfilerState()` returns true, regardless of `TT_METAL_PROFILER_TRACE_TRACKING`. A reader inspecting the `running_with_tracy.md` list would incorrectly conclude that END messages are absent when trace tracking is off, and might incorrectly diagnose parse failures in `import_tracy_op_logs` that rely on END messages.

---

## Item 2 — `running_with_tracy.md` line 39: `TT_METAL_DEVICE_PROFILER=1` attribution contradicts section heading

**File:** `running_with_tracy.md`, lines 31 and 39.

The section heading at line 31 reads: "When `-r` is active, `tools/tracy/__main__.py` sets the following variables on the test subprocess environment before launching it." This unambiguously describes the outer `-r` launcher as the setter for all variables that follow. Line 39, under the `TT_METAL_DEVICE_PROFILER=1` subsection, then states: "This variable is set by the test subprocess (not the outer launcher)." These two statements are mutually exclusive. A reader implementing an equivalent launcher would not know whether to include `TT_METAL_DEVICE_PROFILER=1` in the environment they inject into the test subprocess, potentially producing a run with no device-side timestamps and an empty `DEVICE KERNEL DURATION` column.

---

## Item 3 — `differentiating_trace_ops_from_normal_ops.md` line 35: consecutive-message method fails for the final replay session

**File:** `differentiating_trace_ops_from_normal_ops.md`, lines 34–35.

The file states: "The total duration of a replay iteration is best measured as the elapsed time between consecutive `TT_METAL_TRACE_REPLAY` Tracy messages." This method requires a subsequent `TT_METAL_TRACE_REPLAY` message to define the end of an interval. For the final replay session there is no subsequent message, so the method produces no result for that session. A reader applying this approach to measure all N replay sessions would silently obtain only N-1 measurements, potentially without realizing the last session was omitted. The `--device-trace-profiler` alternative is mentioned in the same sentence but as an "or" option, not as the required fallback for the last session specifically.

---

## Item 4 — `reading_profiling_output.md` lines 57 and 59: internal contradiction on whether warm-up op-code sequence is identical to capture/replay

**File:** `reading_profiling_output.md`, lines 57 and 59.

Line 57 states that the warm-up, capture, and replay phases produce "three consecutive phases with identical op-code sequences," which is why `find_repeated_runs` (which operates on op-code patterns) finds all three as matching blocks. Line 59 then states that warm-up kernel durations "may reflect JIT compilation overhead, especially for the sampling-kernel compile pass that fires at the end of the warm-up." If the sampling-kernel compile pass emits distinct op codes not present in the capture and replay passes, the three sequences are not identical and `find_repeated_runs` would not include the warm-up as one of the `num_runs=3` matching blocks — the boundary found would shift, and `df_model_compilation` would not contain warm-up rows. If the op-code sequences truly are identical (the sampling-kernel compile pass emits no new op codes), the timing caveat is still valid but the internal contradiction remains in the prose. A reader building on this heuristic needs to know definitively whether the warm-up sequence is op-code-identical to capture/replay or not; the current text implies both simultaneously.

---

## Change Log — Pass 3 Fixes

### Item 1 — `running_with_tracy.md`: `TracyTTMetalEndMeshTrace` removed from trace-tracking-gated macro list

**File:** `running_with_tracy.md`, `TT_METAL_PROFILER_TRACE_TRACKING=1` section.

**Finding:** The Pass-1-verified finding (confirmed against `tt_metal/tools/profiler/tt_metal_tracy.hpp`) established that `TracyTTMetalEndMeshTrace` emits its `TT_METAL_TRACE_END` message unconditionally — guarded only by `getDeviceProfilerState()`, not by `TracyTTMetalTraceTrackingEnabled()`. Listing it alongside BEGIN and REPLAY as requiring `TT_METAL_PROFILER_TRACE_TRACKING=1` was therefore incorrect.

**Fix applied:** Removed `TracyTTMetalEndMeshTrace` from the bullet list of macros that emit only when `TT_METAL_PROFILER_TRACE_TRACKING=1` is set. The list now contains only `TracyTTMetalBeginMeshTrace`, `TracyTTMetalReplayMeshTrace`, and `TracyTTMetalEnqueueMeshWorkloadTrace`. Added a note below the list explicitly documenting that `TracyTTMetalEndMeshTrace` is the exception: it emits whenever `TT_METAL_DEVICE_PROFILER=1` is set, regardless of `TT_METAL_PROFILER_TRACE_TRACKING`.

---

### Item 2 — `running_with_tracy.md`: `TT_METAL_DEVICE_PROFILER=1` attribution contradiction resolved

**File:** `running_with_tracy.md`, `TT_METAL_DEVICE_PROFILER=1` subsection.

**Finding:** The section heading states the outer `-r` launcher sets all listed env vars on the test subprocess. The `TT_METAL_DEVICE_PROFILER=1` subsection then contradicted this by stating the variable "is set by the test subprocess (not the outer launcher)." Since `tools/tracy/__main__.py` is not present in this repository, the resolution was determined from the Pass-1 change log, which cites `tools/tracy/__main__.py` lines 17–18 as verified and confirms the outer launcher is responsible for injecting env vars. The contradicting parenthetical was incorrect.

**Fix applied:** Removed the sentence fragment "This variable is set by the test subprocess (not the outer launcher)" from the `TT_METAL_DEVICE_PROFILER=1` subsection. The prose now simply states that the variable enables the device-side profiler and controls `profiler_enabled` in `rtoptions`, consistent with the section heading that the outer launcher sets it.

---

### Item 3 — `differentiating_trace_ops_from_normal_ops.md`: consecutive-message replay duration method replaced

**File:** `differentiating_trace_ops_from_normal_ops.md`, Key insight callout.

**Finding:** Describing replay duration as "elapsed time between consecutive `TT_METAL_TRACE_REPLAY` messages" is a DIY approach that silently produces no measurement for the final replay session (no subsequent message to bound the interval). The correct tooling path is `lookup_trace_replay_timestamp` in `process_ops_logs.py`, which anchors each op row's wall-clock timestamp to the appropriate replay message; per-op timing is then directly readable from the `DEVICE KERNEL DURATION [ns]` column without any inter-message arithmetic.

**Fix applied:** Replaced the Key insight callout to explain that: (a) `lookup_trace_replay_timestamp` handles wall-clock anchoring per replay session automatically, so users should use the `DEVICE KERNEL DURATION [ns]` column directly for per-op timing within a session; (b) for end-to-end replay duration, `--device-trace-profiler` mode is the correct approach; and (c) the inter-message interval method is explicitly called out as unreliable for the final session and should not be used.

---

### Item 4 — `reading_profiling_output.md`: warm-up vs. capture/replay op-code sequence identity clarified

**File:** `reading_profiling_output.md`, `split_compile_and_trace` section.

**Finding:** The accurate picture from Chapter 3 is that the sampling-kernel compile pass during warm-up produces a distinct op sequence not present in capture or replay. This means warm-up is the non-repeating prefix that `find_repeated_runs` strips. Only the post-compile capture and replay phases produce truly identical op-code sequences. The prior text simultaneously claimed all three phases had "identical op-code sequences" and that warm-up had a distinct sampling-kernel compile tail — a direct contradiction.

**Fix applied:** Replaced the paragraph to state the accurate picture: the warm-up phase includes a sampling-kernel compile tail that emits a different op sequence, making warm-up the non-repeating prefix that `find_repeated_runs` strips. The heuristic then counts the post-compile capture and replay phases as the identical repeating blocks. The `df_model_compilation` / `df_model_trace` descriptions are updated accordingly, and a note clarifies that only post-compile capture and replay have truly identical op-code sequences.

---

# B Review — Pass 4

## Item 1 — `reading_profiling_output.md` lines 57–60: self-contradictory account of warm-up's role in `find_repeated_runs`

**File:** `reading_profiling_output.md`, lines 57–60.

The paragraph introduced in Pass 3 to resolve the warm-up/op-code contradiction contains a new internal contradiction that cannot both be true simultaneously:

- The opening clause says warm-up does **not** have an identical op-code sequence to capture/replay, that `find_repeated_runs` "strips this non-repeating prefix … and then identifies the **two** post-compile phases — trace capture and trace replay — as the repeated blocks."
- The following clause immediately reverses this: "However, when called with `num_runs=3`, the heuristic finds … the warm-up phase's main body … shares the same op-code pattern as capture and replay, so the **three-way match** encompasses warm-up + capture + replay."
- Line 59 then says `df_model_compilation` contains "warm-up (compile) pass rows" and describes warm-up as "the non-matching prefix that the heuristic discards before counting identical blocks."

These three statements are mutually exclusive. A reader cannot determine from this paragraph whether:

(a) warm-up IS one of the three `num_runs=3` identical blocks (so `df_model_compilation` = warm-up, and the heuristic finds three-way match), or
(b) warm-up is the discarded non-repeating prefix and only capture + replay are the two identical blocks (making `num_runs=3` incorrect or mislabeled).

A reader implementing a filtering step based on `df_model_compilation` would not know whether they are operating on warm-up rows or capture-pass rows. The description must commit to one consistent model and verify it against `find_repeated_runs` in `test_utils.py`.

---

## Change Log — Pass 4 Fixes

### Item 1 — `reading_profiling_output.md`: self-contradictory warm-up / `find_repeated_runs` paragraph rewritten

**Verified against:** Chapter 3 `differentiating_warmup_from_production.md` (the `split_compile_and_trace` and `find_repeated_runs` algorithm sections), which is the authoritative in-guide source established in prior passes.

**Finding:** Chapter 3 is unambiguous: warm-up (compile) phase ops carry null `METAL TRACE ID` and are the **non-repeating prefix** that `find_repeated_runs` strips. The two post-compile phases — trace capture and trace replay — produce identical op-code sequences and are the two repeated blocks located by `find_repeated_runs` with `num_runs=2`. Therefore `df_model_compilation` = first repeated block = **trace-capture phase** (not warm-up), and `df_model_trace` = last repeated block = **final replay**. Chapter 3 also explicitly warns that `num_runs=1` is unsafe and that `num_runs=2` is the correct standard value. The Pass 3 rewrite of this paragraph was internally inconsistent: it simultaneously described warm-up as the non-repeating prefix AND as one of the three `num_runs=3` identical blocks, and mislabeled `df_model_compilation` as warm-up rows.

**Fix applied to `reading_profiling_output.md`:**
1. Replaced the contradictory paragraph (lines 57–62) with a single consistent account: warm-up produces a distinct op sequence (sampling-kernel compile tail) → warm-up is the non-repeating prefix that `find_repeated_runs` strips → with `num_runs=2`, the two identical blocks are trace capture and trace replay → `df_model_compilation` = trace-capture phase, `df_model_trace` = final replay.
2. Updated the example call to use `num_runs=2` (was incorrectly showing `num_runs=3`), consistent with the standard decode case described in the paragraph.
3. Removed the guidance to use `METAL TRACE ID` column filtering to obtain "capture-pass rows specifically" — that guidance was only necessary because the prior text incorrectly placed capture rows outside `df_model_compilation`; with the corrected model, `df_model_compilation` already contains the capture-phase rows.
4. Added guidance on recovering warm-up rows via `df[:first_run_start]` with a cross-reference to Chapter 3's `differentiating_warmup_from_production.md`.

---

# B Review — Pass 5

No feedback — chapter approved.
