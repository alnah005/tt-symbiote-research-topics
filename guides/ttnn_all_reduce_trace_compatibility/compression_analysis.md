# Change Log — Final Pass B Review Pass 1 Fixes

- Item 1: Fixed `FABRIC_1D` → `FABRIC_1D_RING` in Ch3 `q3_requirements_and_limitations.md` Requirement 3 and Requirement 5. Verified against codebase (`fabric_types.hpp`, `fabric_host_utils.cpp`, and T3K CI configs) that `FABRIC_1D_RING` (value 2, with deadlock avoidance via datelines) is the correct enum for T3K ring topology; added a Warning callout distinguishing the two constants. Ch3 and Ch4 now agree on `FABRIC_1D_RING`.
- Item 2: Fixed composite path description in Ch2 `index.md` Key Finding callout and reading-order entry. Replaced the shorthand "all-gather + `moreh_sum`" with the accurate label `composite_all_gather` (= `ttnn::prim::all_broadcast` + `ttnn::concat`) + local reduce (`local_sum` / `local_sum_float32`), matching the terminology used in `composite_path.md` and Ch3.
- Item 3: Fixed wrong chapter cross-reference in Ch4 `integration_checklist.md` line "See Chapter 2 (`q2_semaphore_state.md`)" → "See Chapter 3 (`q2_semaphore_state.md`)", since `q2_semaphore_state.md` lives under `ch3_verdict/`.
- Item 4: Added "Capture-window buffer vs. warm-up buffer" subsection to Ch1 `buffer_address_stability.md` (Persistent Output Buffers section) defining both terms, explaining that `TraceEntry.trace_output` (warm-up buffer) is never written by `ttnn.execute_trace` and that callers must read from `capture_output` (capture-window buffer) to observe replay results. Includes a Warning callout and a forward reference to Ch4 for the full treatment.

# Change Log — Final Pass B Review Pass 2 Fixes

- Fixed `FABRIC_1D` → `FABRIC_1D_RING` in Ch3 index.md Q3 answer row.

# Cross-Chapter Compression Analysis — Pass 1

## Summary
- Total files analyzed: 16
- Estimated current total line count: ~2287 lines
- Estimated post-compression line count: ~2120 lines
- Estimated reduction: ~7%

## CRUCIAL Suggestions

### [ch2_all_reduce_internals/index.md and ch3_verdict/q1_trace_compatibility.md] ~lines 10–38 (ch2 index) / ~lines 50–103 (q1)
**Issue:** The Ch2 index "Summary answer" block (lines 10–38) and the Ch3 Q1 "Path selection: composite vs non-composite" section (q1_trace_compatibility.md lines 50–103) both reproduce the full four-predicate path-selection logic in prose and code. The Ch2 index block describes all four predicates (`composite_all_gather`, `composite_reduce_scatter`, `composite_for_2d_mesh`, `dim != composite_dim`) and the T3K "Key finding" with essentially the same content as q1's §"T3K path assessment" table plus the composite-vs-non-composite decision tree. The composite path's `all_broadcast` + `concat` + `moreh_sum`/`local_sum_float32` chain is described in both locations.
**Suggestion:** Shrink the Ch2 index "Summary answer" block to a one-paragraph high-level statement of the verdict (non-composite path taken on T3K decode shapes, no global semaphores, `scattered_tensor` pre-allocation required) and a forward pointer to `call_chain.md` for the predicate details. Delete the duplicate prose paragraph on composite path internals from the Ch2 index; retain the full treatment only in `composite_path.md` (Ch2) and `q1_trace_compatibility.md` (Ch3). Estimated saving: ~18 lines from ch2/index.md.

### [ch2_all_reduce_internals/call_chain.md and ch3_verdict/q1_trace_compatibility.md] ~lines 109–207 (call_chain) / ~lines 50–103 (q1)
**Issue:** `call_chain.md` §4 "Path-selection predicates" (lines 132–207) and `q1_trace_compatibility.md` §"Path selection: composite vs non-composite" (lines 50–103) both explain the same four predicates (`use_composite_all_gather`, `use_composite_reduce_scatter`, `is_true_2d_mesh`, `composite_for_2d_mesh`) at nearly equal depth, including the three-step sequential-guard logic for `use_composite_reduce_scatter` and the three conditions for `use_composite_all_gather`. Both cite the same source file (`composite_common.cpp`) and produce the same T3K table showing all four predicates as `false`.
**Suggestion:** Keep the detailed predicate breakdown exclusively in `call_chain.md` (Ch2). In `q1_trace_compatibility.md`, replace the "Path selection" section with a prose summary paragraph (two to three sentences) stating the conditions under which the non-composite path is taken on T3K, followed by a "See `call_chain.md` §4 for full predicate definitions" cross-reference. Estimated saving: ~45 lines from q1_trace_compatibility.md.

### [ch3_verdict/q3_requirements_and_limitations.md and ch4_integration/integration_checklist.md] ~lines 55–92 (q3) / ~lines 87–129 (integration_checklist)
**Issue:** Requirement 2a in `q3_requirements_and_limitations.md` (lines 55–92) and the `scattered_tensor` checklist item in `integration_checklist.md` (lines 87–129) both contain an identical 15-line Python code block showing the `ttnn.reduce_scatter` + `ttnn.all_gather` split with `output_tensor=scattered_tensor`. The surrounding prose in both files also covers the same ground: pre-allocate `scattered_tensor` before capture, pass it as `output_tensor` to `ttnn.reduce_scatter`, treat the unmodified `ttnn.all_reduce` as trace-unsafe until the split is done.
**Suggestion:** Keep the full code block and surrounding prose in `integration_checklist.md` (Ch4), which is the actionable implementation guide. In `q3_requirements_and_limitations.md`, replace Requirement 2a's code block with a one-sentence statement describing the remediation approach, followed by "See `integration_checklist.md` for the implementation pattern." Estimated saving: ~20 lines from q3_requirements_and_limitations.md.

### [ch1_trace_mechanics/semaphore_initialization_and_replay.md and ch3_verdict/q2_semaphore_state.md] ~lines 91–113 (ch1 semaphore file) / ~lines 26–46 (q2)
**Issue:** `semaphore_initialization_and_replay.md` §"First Observation About `ttnn.all_reduce`" (lines 91–113) reproduces the exact same C++ code block — `::ttnn::experimental::all_reduce_async(... std::nullopt, std::nullopt, std::nullopt ...)` — that also appears in `q2_semaphore_state.md` §"How `ttnn.all_reduce` passes semaphore arguments" (lines 26–46). Both sections state the same conclusion: all three semaphore arguments are `std::nullopt`, no `GlobalSemaphore` is created at the caller level, and the downstream handling is determined by the path inside `all_reduce_async`. The Ch1 section also carries the same caveat that absence of caller-visible semaphore args does not prove absence of internal semaphore state.
**Suggestion:** In `semaphore_initialization_and_replay.md`, replace the full "First Observation" section with a brief two-to-three sentence observation (no code block): "`ttnn.all_reduce` passes `std::nullopt` for all three semaphore argument groups; the caller holds no `GlobalSemaphore` handle to reset between replays. Whether the downstream path uses internal global semaphores is determined in Chapter 2." Delete the duplicated `std::nullopt` code block from this file; the definitive treatment lives in `q2_semaphore_state.md`. Estimated saving: ~15 lines from ch1/semaphore_initialization_and_replay.md.

## MINOR Suggestions

### [ch1_trace_mechanics/index.md] ~lines 5–14
**Issue:** The "Context" section in Ch1's index introduces `TTNNLinearIColShardedWAllReduced`, its `@trace_enabled` inheritance, and the two research sub-questions. This exact framing — including the module name, its topology/cluster_axis values, and the two sub-questions about buffer stability and global semaphore state — is already stated in the guide-level index (index.md lines 2–3, and the chapter description column in the Chapter Index table). There is meaningful overlap between the Ch1 index "Context" block and the guide-level index.
**Suggestion:** Trim the Ch1 "Context" section to two to three sentences that directly state what the chapter defines (trace buffer recording model, buffer stability constraint, local vs global semaphore distinction) rather than re-framing the whole guide's central question. The full framing is already at the guide level and need not be repeated in each chapter's index.

### [ch3_verdict/index.md and ch2_all_reduce_internals/index.md] ~lines 6–8 (ch3 index) / ~lines 3–8 (ch2 index)
**Issue:** Both chapter indexes open with a sentence describing what the chapter "synthesises" or "traces" from/to the next chapter. These introductory sentences partially restate the guide-level Chapter Index table descriptions (guide index lines 23–26), which already capture the same information in a denser format.
**Suggestion:** Each chapter index's opening paragraph can be cut to one sentence that names what is new in that chapter (not what prior chapters established), since the guide-level Chapter Index table already cross-maps chapter scope. Minor impact (~3–4 lines each).

### [ch4_integration/integration_checklist.md] ~lines 143–159
**Issue:** The bias buffer checklist item (lines 140–159) restates Limitation 3 from `q3_requirements_and_limitations.md` (lines 218–224) at approximately double the length. Both describe the same concern: `self.tt_bias` must be at a fixed device address, and the in-place `+=` bias-add may or may not reuse `tt_output`'s buffer. The checklist version adds implementation detail (two numbered sub-conditions), but the core concern is restated rather than extended with new information.
**Suggestion:** This is an acceptable depth increase in an actionable checklist context (Ch4 is the integration guide), so no cut is required. Consider adding a one-line back-reference: "See `q3_requirements_and_limitations.md` Limitation 3 for background." This avoids a future edit-drift problem without requiring text deletion.

## Load-Bearing Evidence

- `ch1_trace_mechanics/buffer_address_stability.md` line ~92: "**Capture-window buffer** (`capture_output`): The output tensor produced by `module.forward` when called *inside* the `begin_trace_capture` / `end_trace_capture` window. `ttnn.execute_trace` writes replay results into this buffer." — load-bearing because this is the only place in Ch1 that defines the capture-window vs warm-up buffer distinction, which is referenced by both Ch4 content files; removing or merging this would break the forward reference in `buffer_address_stability.md`'s own warning callout and the `integration_checklist.md` annotation at line 131.

- `ch2_all_reduce_internals/reduce_scatter_all_gather_path.md` line ~127: "`scattered_tensor` is a dynamically-allocated DRAM intermediate and therefore faces the same trace incompatibility risk as the composite-path intermediates." — load-bearing because this is the only file that proves the non-composite path is not unconditionally safe, which is the key nuance distinguishing Ch2's verdict from a naive "no global semaphores = safe" conclusion; Ch3 Q1 and Ch4 both depend on this finding.

- `ch3_verdict/q3_requirements_and_limitations.md` line ~169: "The `is_trace_enabled` predicate in `run_config.py` is: `return (isinstance(module, tuple(_TRACE_ENABLED_CLASSES)) and not isinstance(module, tuple(_TRACE_DISABLED_CLASSES)))`" — load-bearing because this is the only location across the guide that quotes the actual predicate implementation and confirms `TTNNLinearIColShardedWAllReduced` satisfies it by inheritance; removing this loses the grounding evidence for the `@trace_enabled` status claim made in Ch1 index and Ch4 checklist.

- `ch4_integration/minimal_test_pattern.md` lines ~283–297 (Mapping table): "Step 6 — reading replay output from `capture_output` | Low-level test reads `capture_output` directly; `module_run` line 1064 instead returns `entry.trace_output` (warm-up buffer — stale)" — load-bearing because the mapping table is the only location that explicitly cross-maps each manual test step to the exact `run_config.py` line number; collapsing or abbreviating it would lose the only machine-checkable link between the test skeleton and the internal implementation.

- `ch2_all_reduce_internals/composite_path.md` line ~169: "`gather_tensor` — the output of `composite_all_gather`. It is sized as `[num_devices, initial_shape[0] * initial_shape[1], H, W]`... `all_broadcast` to replicate the reshaped tensor... then `ttnn::concat` along `composite_dim = 0`, so the first dimension becomes `num_devices`, not the second." — load-bearing because this is the only place that corrects the intuitive but wrong assumption that `gather_tensor`'s first dimension is 1; this shape detail explains why the allocation is dynamic and not trivially pre-allocatable, making the incompatibility claim concrete rather than asserted.

## VERDICT
- Crucial updates: yes

# Agent A Change Log — Final Pass B Pass 3 + C Cross-Chapter Pass 1 Compression

- B Pass 3 Item 1: Fixed `FABRIC_1D` → `FABRIC_1D_RING` in `q1_trace_compatibility.md` — two occurrences: the T3K path-assessment table (Fabric config row) and the Key Finding callout block (lines ~128 and ~140).
- C Item 1: Collapsed Ch2 `index.md` "Summary answer" block from a two-paragraph + Key Finding callout (~29 lines) to a single focused paragraph (~9 lines) stating the non-composite verdict, the `scattered_tensor` pre-allocation precondition, and forward pointers to `call_chain.md` and Ch3 `q1_trace_compatibility.md`. Saved ~20 lines.
- C Item 2: Replaced the full "Path selection: composite vs non-composite" section in `ch3_verdict/q1_trace_compatibility.md` (~53 lines including the C++ predicate code block and prose breakdown of all four guards) with a 5-line prose summary naming the four predicates and the T3K conditions, plus a cross-reference link to `call_chain.md` §4 for the full treatment. Saved ~48 lines.
- C Item 3: Replaced Requirement 2a's 15-line Python code block in `ch3_verdict/q3_requirements_and_limitations.md` with a 6-line prose summary describing the remediation approach (split `ttnn.all_reduce` into explicit `ttnn.reduce_scatter` + `ttnn.all_gather` with a pre-allocated `output_tensor`) and a cross-reference to `ch4_integration/integration_checklist.md` for the full annotated implementation pattern. Saved ~20 lines.
- C Item 4: Replaced the "First Observation About `ttnn.all_reduce`" section in `ch1_trace_mechanics/semaphore_initialization_and_replay.md` — removed the 14-line `std::nullopt` C++ code block and the two follow-on prose paragraphs — with a 5-line prose-only summary noting that all three semaphore args are `std::nullopt` and deferring to Ch2 `index.md` and Ch3 `q2_semaphore_state.md` for the full analysis. Saved ~16 lines.

# Change Log — Final Pass B Review Pass 4 Fix

- Reframed "Evidence from the existing codebase" section in q1_trace_compatibility.md to correctly distinguish async-variant test evidence from direct synchronous-path validation; added explicit open test gap note.

# Cross-Chapter Compression Analysis — Pass 2

## Summary
- Total files analyzed: 4
- Estimated current total line count: ~2183 lines (Pass 1 baseline ~2287 minus ~104 lines saved by Pass 1 fixes)
- Estimated post-compression line count: ~2158 lines
- Estimated reduction: ~1%

## CRUCIAL Suggestions
(re-check of Pass 1 CRUCIAL items only)

All four CRUCIAL items from Pass 1 have been adequately addressed:

- **CRUCIAL 1** (`ch2_all_reduce_internals/index.md` "Summary answer" bloat): The block is now ~11 lines with a one-paragraph verdict and forward pointers to `call_chain.md` and Ch3 Q1. The duplicate composite-path prose and Key Finding callout have been removed. Addressed.
- **CRUCIAL 2** (`ch3_verdict/q1_trace_compatibility.md` path-selection duplication): The "Path selection: composite vs non-composite" section is now 8 lines of prose plus a cross-reference to `call_chain.md` §4; the full C++ predicate breakdown and T3K table are gone. Addressed.
- **CRUCIAL 3** (`ch3_verdict/q3_requirements_and_limitations.md` Requirement 2a code block): The 15-line Python code block has been replaced with 8 lines of prose description and a cross-reference to `integration_checklist.md`. Addressed.
- **CRUCIAL 4** (`ch1_trace_mechanics/semaphore_initialization_and_replay.md` "First Observation"): The C++ `std::nullopt` code block and follow-on paragraphs have been replaced with a 7-line prose-only summary deferring to Ch2 and Ch3 Q2. Addressed.

## MINOR Suggestions

### [ch3_verdict/q1_trace_compatibility.md] ~lines 28–47
**Issue:** The "Conclusion" section's bullet on the composite path (lines 28–47) contains a full internal blow-by-blow: `composite_all_gather` → `all_broadcast` + `concat` → `local_sum` (calling `ttnn::moreh_sum`) / `local_sum_float32` (calling `ttnn::transpose` + `ttnn::sum`), the two named intermediates (`reshaped_tensor`, `gather_tensor`), the conditional third buffer from `sharded_to_interleaved`, and the `change_mem_config` / `ttnn.DRAM_MEMORY_CONFIG` caveat spanning 19 lines. The same mechanism and the same intermediate names are covered in `composite_path.md` (Ch2), which is the authoritative location for composite-path internals.
**Suggestion:** Trim the composite-path bullet in the Conclusion to three sentences: state that the composite branch allocates two (or three) transient intermediates that cannot be pre-allocated through the public API, name those intermediates, and add a forward reference to `composite_path.md` for the full breakdown. Estimated saving: ~14 lines from `q1_trace_compatibility.md`.

## Load-Bearing Evidence

- `ch3_verdict/q1_trace_compatibility.md` line ~97: "> **Key finding:** On T3K with `FABRIC_1D_RING` and standard hidden dimensions, `TTNNLinearIColShardedWAllReduced` routes `ttnn.all_reduce` to the non-composite reduce-scatter + all-gather path. The composite path is not reached during normal decode operation." — load-bearing because this is the only location in the four checked files that states the primary T3K routing verdict as a named Key Finding callout; removing it would eliminate the single most scannable summary a reader jumping directly to Q1 would rely on.

- `ch3_verdict/q3_requirements_and_limitations.md` line ~139: "`def is_trace_enabled(module) -> bool: return (isinstance(module, tuple(_TRACE_ENABLED_CLASSES)) and not isinstance(module, tuple(_TRACE_DISABLED_CLASSES)))`" — load-bearing because this is the only place in all checked files that quotes the actual predicate implementation confirming `TTNNLinearIColShardedWAllReduced` satisfies `is_trace_enabled` by inheritance; collapsing it removes the grounding evidence for the trace-enabled status claim used by Ch4.

- `ch1_trace_mechanics/semaphore_initialization_and_replay.md` line ~91: "`ttnn.all_reduce` passes `std::nullopt` for all three semaphore argument groups (`barrier_semaphores`, `rs_global_semaphores`, `ag_global_semaphores`) when it calls `ttnn::experimental::all_reduce_async`. No `GlobalSemaphore` object is created or consumed at the caller level, so the caller holds nothing to reset between replays." — load-bearing because this is the only sentence in Ch1 that names all three semaphore argument groups by their actual parameter names and states the caller-level nullopt conclusion; it grounds the deferral to Ch2 with enough specificity to be independently verifiable.

- `ch2_all_reduce_internals/index.md` line ~14: "The one precondition is that the intermediate `scattered_tensor` — a dynamic DRAM allocation created by `ttnn::reduce_scatter` — must be pre-allocated as a persistent buffer before `ttnn.begin_trace_capture` is called" — load-bearing because this is the only sentence in the Ch2 index that names `scattered_tensor` explicitly as a DRAM allocation and states the pre-allocation precondition; it is the first entry point a reader encounters when traversing the chapter, and its removal would leave the summary without its critical caveat.

## VERDICT
- Crucial updates: no

# Change Log — Final Pass B Review Pass 5 Fix

- Unified keyword name for reduce_scatter pre-allocation argument: verified correct name is `output_tensor`; updated all occurrences across ch2, ch3, ch4.

# Change Log — Final Pass B Review Pass 6 Fix

- Corrected scattered_tensor shape in reduce_scatter_all_gather_path.md §3: [1,1,32,...] → [1,1,1,...] for decode case; added note for prefill (seq_len=S).

# Change Log — Final Pass B Review Pass 7 Fix

- Clarified across ch2/ch3/ch4 that satisfying the scattered_tensor pre-allocation requirement requires refactoring TTNNLinearIColShardedWAllReduced.forward to use direct ttnn.reduce_scatter + ttnn.all_gather calls; ttnn.all_reduce hardcodes std::nullopt internally and cannot expose output_tensor.
