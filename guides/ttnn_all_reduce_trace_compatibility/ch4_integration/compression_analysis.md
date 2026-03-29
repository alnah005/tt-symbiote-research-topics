# Change Log — B Review Pass 1 Fixes

- Item 1: Fixed `comp_pcc` return type documentation in both `integration_checklist.md` and `minimal_test_pattern.md`. The second return value was incorrectly named `pcc_message` and described as a string. Verified against `tt-metal/models/common/utility_functions.py` line 573, which returns `(cal_pcc >= pcc, cal_pcc)` where `cal_pcc` is a NumPy float scalar. Renamed `pcc_message` to `pcc_value` in both code snippets, updated assert format strings to use `{pcc_value:.6f}` (numeric formatting), and corrected the prose note in `minimal_test_pattern.md` Step 7 from "`(bool, str)` pair" to "`(bool, float)` pair" with an accurate description of the second element.
- Item 2: Not applied — back-links are not required by guide navigation footer rules.

# Change Log — B Review Pass 2 Fixes

- Item 1: Not applied — forward reference to ../index.md is correct per guide rules; index.md written in Final Pass.
- Item 2: Corrected the `trace_output` attribution in the "warm-up and buffer pre-allocation" checklist item of `integration_checklist.md`. Verified against `tt-metal/models/experimental/tt_symbiote/core/run_config.py` lines 989–1000: `trace_output` (line 990) is assigned from the warm-up call outside the capture window, and the capture-window call's return value is discarded (`_ = module.forward(*trace_func_args, ...)`). The prior description incorrectly stated that replays write into the warm-up output buffer. The corrected text accurately states that `TraceEntry.trace_output` holds the warm-up output tensor returned to callers, while `ttnn.execute_trace` during replays writes into the output buffer that was allocated inside the `begin_trace_capture` / `end_trace_capture` window — a distinct allocation from the warm-up output buffer. The caller responsibility to keep `entry.trace_output` alive is preserved.
- Item 3: Added a new pre-condition checklist item in `integration_checklist.md` covering bias buffer stability, derived from Limitation 3 in `ch3_verdict/q3_requirements_and_limitations.md`. The item requires that `self.tt_bias` be pre-loaded at a fixed device address before trace capture and that the in-place `tt_output += self.tt_bias` add's output buffer stability be verified or remediated before the trace is captured, treating the bias step as a source of trace instability independent of the `ttnn.all_reduce` intermediate buffers.

# Change Log — B Review Pass 3 Fixes

- Item 1: Replaced the incorrect "routes the result through `entry.trace_output`" claim in `integration_checklist.md` (formerly lines 146–149) with an accurate description verified against `run_config.py` lines 989–1064. The accurate description states: `ttnn.execute_trace` writes replay results into the capture-window output buffer (the buffer live during `begin_trace_capture`/`end_trace_capture`); `entry.trace_output` (the warm-up buffer) is never updated by replays. `TracedRun.module_run` returns `entry.trace_output` as the result handle on every call (line 1064), so callers using `TracedRun` receive the warm-up buffer — a distinct allocation from the capture-window buffer — and will read stale warm-up data rather than replay results. Callers invoking `ttnn.execute_trace` directly must read from the capture-window buffer to observe replay outputs. A clarifying comment was also added to the aliasing-detection code snippet to warn that reading `entry.trace_output` after `execute_trace` reads the warm-up buffer, not the capture-window buffer.
- Item 2: Restructured `minimal_test_pattern.md` Step 5 to match the actual two-phase sequence in `_capture_trace` (lines 989–994 of `run_config.py`). The prior skeleton incorrectly assigned `trace_output = layer.forward(trace_input)` inside the capture window. The corrected pattern introduces Step 4b (warm-up before `begin_trace_capture`: `warm_up_output = layer.forward(persistent_input)`) and Step 5 (capture window: `_ = layer.forward(persistent_input)` with the return value intentionally discarded). Prose and warning blocks make explicit that `execute_trace` writes into the capture-window output buffer, not into `warm_up_output`. Step 6 was updated to use `persistent_input` (renamed from `trace_input`) and `warm_up_output` with a comment explaining TracedRun semantics versus direct capture-window buffer access. The mapping table was updated to include Step 4b (warm-up) as a distinct row mapped to `_capture_trace` line 990, and to accurately describe the Step 5 capture-window row (mapped to lines 992–994) and the Step 6 replay-output row (mapped to line 1064).

# Change Log — B Review Pass 4 Fixes

- Item 1: Fixed `NameError` in `minimal_test_pattern.md` caused by `persistent_input` being referenced in Step 4b before being defined in Step 5. Moved `persistent_input` allocation (and `cq_id` assignment) into Step 4, merged it with the `scattered_tensor` pre-allocation step (renamed "Step 4 — Pre-allocate `persistent_input` and `scattered_tensor` before capture"). Step 4b now correctly uses `persistent_input` (allocated in Step 4), and Step 5 references the same buffer inside the capture window. Verified against `run_config.py` lines 969–990: `_capture_trace` allocates `trace_inputs` before the warm-up call at line 990, so allocating `persistent_input` before the warm-up is mechanistically correct.
- Item 2: Fixed Step 6 aliasing check in `minimal_test_pattern.md` to read from `capture_output` (the capture-window output buffer) instead of `warm_up_output` after each `ttnn.execute_trace` call. Verified against `run_config.py` lines 992–994 and 1063–1064: `execute_trace` writes into the buffer allocated inside the `begin_trace_capture`/`end_trace_capture` window; `warm_up_output` (stored as `entry.trace_output` at line 1000) is never written by any replay call and always holds stale warm-up data. The check now also retains the capture-window return value as `capture_output` in Step 5 (instead of discarding it with `_ = ...`) to make the buffer handle available. Updated prose after the code block to explain why `warm_up_output` is not used.
- Item 3: Fixed the aliasing detection code block in `integration_checklist.md` to read from `capture_output` instead of `entry.trace_output` after each `ttnn.execute_trace` call. Replaced the prior NOTE comment (which acknowledged the limitation but did not fix it) with a corrected comment explaining that `execute_trace` writes into the capture-window output buffer (verified against `run_config.py` lines 992–994 and 1063), that `entry.trace_output` is the warm-up buffer and is never written by replays (line 1000), and that `capture_output` must be retained from the `layer.forward` call inside `begin_trace_capture`/`end_trace_capture` to observe replay results.
- Item 4: Fixed Step 7 PCC comparison in `minimal_test_pattern.md` so it compares the actual replay output against a non-traced reference. `output_1` (from Step 6) is now read from `capture_output` after `ttnn.execute_trace`, representing the traced path output. It is compared against `reference_output` (from Step 3), which was produced by a direct non-traced `layer.forward` call. Added a comment block explaining why comparing `output_1` against `warm_up_output` would be non-functional (both are from non-traced forward passes over the same input, giving trivially high PCC regardless of trace correctness). Updated the prose after the code block to explain why the PCC is meaningful with the corrected tensor pair.

# Change Log — B Review Pass 5 Fixes

- Item 1: Added a prose sentence before the aliasing detection code block in `integration_checklist.md` defining `capture_output` as the output tensor returned by `layer.forward` inside the `begin_trace_capture` / `end_trace_capture` window, with a note that it must be retained (not discarded with `_ = ...`) and a cross-reference to `minimal_test_pattern.md` Step 5 for the full definition and allocation context. This resolves the undefined-variable issue where a reader implementing from the checklist alone would have no concrete variable to bind `capture_output` to.
- Item 2: Not applied — back-links not required by guide navigation footer rules.

# Compression Analysis: Chapter 4 — Integration Checklist and Test Strategy — Pass 1

## Summary
- Total files analyzed: 3
- Estimated current line count: ~633 lines
- Estimated post-compression line count: ~470 lines
- Estimated reduction: ~26%

## CRUCIAL Suggestions

### [minimal_test_pattern.md] ~lines 164–189 (Step 4b prose block)
**Issue:** The block comment inside Step 4b (lines 167–178) and the following callout note (lines 183–188) both re-explain the warm-up / two-phase sequence. The block comment restates what `integration_checklist.md` "Warm-up and buffer pre-allocation" already covers in full, and the note re-explains `persistent_input` allocation order — already established in Step 4's comment. The two together add ~22 lines of commentary around a 2-line code snippet (`warm_up_output = layer.forward(persistent_input)` + `synchronize_device`).
**Suggestion:** Replace the entire block comment with a single-line reference comment (`# Mirrors _capture_trace line 990 — warm-up before begin_trace_capture; see integration_checklist.md`) and delete the callout note entirely. The step heading and the single comment are sufficient.

### [minimal_test_pattern.md] ~lines 237–279 (Step 6 prose and comment block)
**Issue:** The block comment in Step 6 (lines 242–248) and the closing prose paragraph (lines 272–279) both re-explain that `execute_trace` writes into `capture_output`, not `warm_up_output`. This identical point is made in Step 5's two warning callouts (lines 222–233) and again in Step 7's block comment (lines 287–299). Across Steps 5, 6, and 7, the `capture_output` vs `warm_up_output` distinction is stated five times in `minimal_test_pattern.md` alone; it appears a further two times in `integration_checklist.md` (lines 131–157 and 200–211).
**Suggestion:** Delete the Step 6 block comment entirely (it directly duplicates Step 5's second warning). Cut the closing prose paragraph after Step 6 to one sentence: "Both reads use `capture_output` — the capture-window buffer — because `execute_trace` never writes into `warm_up_output`." Saves ~14 lines.

### [integration_checklist.md] ~lines 131–157 (output buffer semantics checklist item)
**Issue:** The checklist item beginning "The output tensor of the forward pass is captured as `trace_output`…" (lines 131–157) is a 27-line prose block that explains the capture-window vs warm-up buffer distinction in minute detail — two `TraceEntry` allocations, the warm-up call, the capture call, the return-value discard, and the known `TracedRun.module_run` limitation. Every one of these points is also covered by `minimal_test_pattern.md` Steps 4b, 5, 6, and the mapping table. The checklist item is written at tutorial depth, not checklist depth.
**Suggestion:** Collapse to a 4–6 line checklist entry: assert that `trace_output` (warm-up buffer, stored as `TraceEntry.trace_output`) is distinct from the capture-window output buffer that `execute_trace` writes into, and cross-reference `minimal_test_pattern.md` Step 5 for the full explanation. Remove the internal "Buffer semantics callers must understand:" sub-paragraph. Saves ~20 lines.

### [minimal_test_pattern.md] ~lines 287–315 (Step 7 block comment and closing prose)
**Issue:** The Step 7 block comment (lines 287–299) re-explains `capture_output` vs `warm_up_output` for the fifth time in this file alone. The closing prose (lines 308–315) then re-explains why PCC of ~1.0 is meaningful — a point already made in `integration_checklist.md` "Numerical accuracy" (lines 242–244).
**Suggestion:** Trim the block comment to 3 lines covering only the comparison pairing (`output_1` from `execute_trace` vs `reference_output` from Step 3) and why `warm_up_output` must not be used (one sentence). Cut the closing prose to the single sentence about the 0.999 threshold and bfloat16 arithmetic. Saves ~10 lines.

## MINOR Suggestions

### [minimal_test_pattern.md] ~lines 214–220 (Step 5 closing prose)
**Issue:** The paragraph after the Step 5 code block (lines 214–220) re-explains the two-phase warm-up/capture sequence and the `_capture_trace` discard-vs-retain difference. This same contrast is already stated in the two warning callouts that follow (lines 222–233). The prose paragraph is therefore sandwiched between the code and a pair of callouts that say the same thing more precisely.
**Suggestion:** Delete lines 214–220 entirely and let the two callouts stand on their own. Saves ~7 lines.

### [integration_checklist.md] ~lines 201–211 (aliasing detection code comment block)
**Issue:** The 11-line block comment above the aliasing detection code re-states the `capture_output` / `entry.trace_output` distinction yet again. The defining sentence immediately before the code block (lines 193–199) plus the in-line comment on the `ttnn.execute_trace` line are sufficient context.
**Suggestion:** Reduce the block comment to 3 lines: (1) `execute_trace` writes into the capture-window buffer (`capture_output`), (2) retain `capture_output` from the forward call inside `begin/end_trace_capture`, (3) see `minimal_test_pattern.md` Step 5. Delete the `entry.trace_output` / warm-up buffer explanation — it duplicates the checklist item above. Saves ~8 lines.

### [index.md] ~lines 3–8 (introductory paragraph)
**Issue:** The first paragraph (lines 3–8) says the chapter "translates the requirements established in Chapter 3 into two concrete deliverables" and then names both deliverables. The reading-order section immediately below (lines 24–34) names the same two deliverables with more specific descriptions, making the introductory paragraph partially redundant.
**Suggestion:** Shorten the intro to two sentences: the purpose statement and the bridge claim. Remove the phrase "a sequential checklist of pre-conditions and post-capture validation steps … and an annotated test skeleton that exercises the module end-to-end under trace capture and replay" — this is restated more usefully in the reading-order bullets. Saves ~3 lines.

### [minimal_test_pattern.md] ~lines 333–348 (mapping table rows for Steps 4–6)
**Issue:** Several mapping table cells (Step 4b, Step 5, Step 6) contain embedded prose explanations inside the table cells — parenthetical asides and bold "not" qualifiers — that re-explain the capture-window vs warm-up buffer distinction. Table cells should be telegraphic; this prose belongs in the step prose, not the table.
**Suggestion:** Strip parenthetical explanations from table cells and leave only the function/line reference and the action verb. Example: the Step 5 cell (line 341) can be reduced from 3 lines to 1. Saves ~6 lines.

## Load-Bearing Evidence
- `integration_checklist.md` line ~87: `"- [ ] **\`scattered_tensor\` is pre-allocated and passed to \`ttnn.reduce_scatter\` as \`output_tensor=\` before trace capture.**"` — load-bearing because this is the primary unresolved blocking action item: the required refactor from `ttnn.all_reduce` to explicit `reduce_scatter` + `all_gather` with a pre-allocated intermediate. The code snippet showing the split is the only concrete implementation guidance and cannot be cut.
- `integration_checklist.md` line ~159: `"- [ ] **\`self.tt_bias\` is pre-loaded at a fixed device address before trace capture (bias buffer stability).**"` — load-bearing because this checklist item covers a distinct instability source (Limitation 3) separate from the all_reduce intermediate buffers; removing it would silently drop a required pre-condition.
- `minimal_test_pattern.md` line ~335: the mapping table header row and its first entry — load-bearing because the full table is the only cross-reference linking each numbered test step to specific `run_config.py` line numbers; it cannot be cut, only trimmed of redundant prose within cells.

## VERDICT
- Crucial updates: yes

# Agent A Change Log — C Pass 1 Compression

- Item 1: `minimal_test_pattern.md` Step 4b — replaced the 12-line block comment (lines 167–178) and the following 6-line callout note (lines 183–188) with a single reference comment (`# Mirrors _capture_trace line 990 — warm-up before begin_trace_capture; see integration_checklist.md pre-conditions.`). The 2-line code snippet (`warm_up_output = layer.forward(persistent_input)` + `synchronize_device`) is preserved unchanged.
- Item 2: `minimal_test_pattern.md` Step 6 — deleted the 9-line block comment (lines 242–248) that duplicated Step 5's second warning callout. Replaced the 8-line closing prose paragraph (lines 272–279) with a single sentence: "Both reads use `capture_output` — the capture-window buffer — because `execute_trace` never writes into `warm_up_output`." The code block and assertion are unchanged.
- Item 3: `integration_checklist.md` ~lines 131–157 — collapsed the 27-line tutorial prose block into a 6-line checklist-depth entry asserting that `TraceEntry.trace_output` (warm-up buffer) is distinct from the capture-window output buffer, noting the `TracedRun.module_run` known limitation, and cross-referencing `minimal_test_pattern.md` Steps 4b, 5, and 6 for the full treatment.
- Item 4: `minimal_test_pattern.md` Step 7 — trimmed the 13-line block comment to 3 lines covering the comparison pairing (`output_1` from `capture_output` vs `reference_output` from Step 3) and the prohibition on comparing against `warm_up_output`. Cut the 8-line closing prose to one sentence about the 0.999 threshold and bfloat16 arithmetic.

# Compression Analysis: Chapter 4 — Pass 2

## Summary
- Total files analyzed: 3
- Estimated current line count: ~564 lines (index.md 34, integration_checklist.md 229, minimal_test_pattern.md 301)
- Estimated post-compression line count: ~540 lines
- Estimated reduction: ~4%

## CRUCIAL Suggestions
(re-check of Pass 1 CRUCIAL items only)

All four Pass 1 CRUCIAL items are adequately addressed per the Agent A Change Log:
- CRUCIAL 1 (Step 4b prose block): Reduced to single reference comment — confirmed at `minimal_test_pattern.md` line 167.
- CRUCIAL 2 (Step 6 block comment + closing prose): Block comment deleted; closing prose collapsed to one sentence at line 247 — confirmed.
- CRUCIAL 3 (`integration_checklist.md` ~lines 131–157): Collapsed to 8-line checklist entry with cross-reference at lines 131–138 — confirmed.
- CRUCIAL 4 (Step 7 block comment + closing prose): Block comment trimmed to 3 lines; closing prose is one sentence at line 265 — confirmed.

No Pass 1 CRUCIAL items remain unaddressed.

## MINOR Suggestions

### [minimal_test_pattern.md] ~lines 197–203 (Step 5 closing prose paragraph)
**Issue:** The prose paragraph after the Step 5 code block (lines 197–203) re-explains the two-phase warm-up/capture sequence and the `_capture_trace` discard-vs-retain contrast. The two warning callouts immediately following (lines 205–216) make the same points more precisely. The prose is sandwiched between the code and callouts that supersede it.
**Suggestion:** Delete lines 197–203 entirely and let the two callouts stand on their own. Saves ~7 lines.

### [integration_checklist.md] ~lines 174–203 (aliasing detection code comment block)
**Issue:** The 11-line block comment above the aliasing detection code (`# NOTE: ttnn.execute_trace writes…`) re-states the `capture_output` / `entry.trace_output` distinction. The defining sentence immediately before the code block (lines 174–177) and the inline comment on the `ttnn.execute_trace` line together already provide sufficient context. The block comment's `entry.trace_output` explanation duplicates the now-collapsed checklist item above.
**Suggestion:** Reduce the block comment to 3 lines: (1) `execute_trace` writes into the capture-window buffer (`capture_output`), (2) retain `capture_output` from the forward call inside `begin/end_trace_capture`, (3) see `minimal_test_pattern.md` Step 5. Saves ~8 lines.

### [index.md] ~lines 3–8 (introductory paragraph)
**Issue:** The first paragraph names both deliverables (checklist + test skeleton) in a long subordinate clause. The reading-order bullets below (lines 26–34) describe the same two deliverables with more specific and useful detail, making the intro clause partially redundant.
**Suggestion:** Shorten the intro to two sentences: the purpose statement and the bridge claim. Remove the long subordinate clause beginning "a sequential checklist of pre-conditions…" since the reading-order bullets cover this more usefully. Saves ~3 lines.

### [minimal_test_pattern.md] ~lines 285–297 (mapping table verbose cells)
**Issue:** Several mapping table cells (Step 4b, Step 5, Step 6 rows) contain parenthetical prose that re-explains the capture-window vs warm-up buffer distinction inside table cells. The Step 5 cell (line 291) spans 3 lines of embedded explanation. Table cells should be telegraphic; explanatory prose belongs in the step body.
**Suggestion:** Strip parenthetical explanations from table cells and leave only the function/line reference and action verb. The Step 5 row can be reduced from 3 lines to 1. Saves ~6 lines.

## Load-Bearing Evidence
- `integration_checklist.md` line ~87: `"- [ ] **\`scattered_tensor\` is pre-allocated and passed to \`ttnn.reduce_scatter\` as \`output_tensor=\` before trace capture.**"` — load-bearing because the code snippet showing the `ttnn.all_reduce` → `reduce_scatter` + `all_gather` split is the only concrete implementation guidance for the primary unresolved blocking action item and cannot be cut.
- `integration_checklist.md` line ~140: `"- [ ] **\`self.tt_bias\` is pre-loaded at a fixed device address before trace capture (bias buffer stability).**"` — load-bearing because this item covers a distinct instability source (Limitation 3) not addressed elsewhere; removing it would silently drop a required pre-condition.
- `minimal_test_pattern.md` line ~285: mapping table header and entries — load-bearing because the table is the only cross-reference linking each numbered test step to specific `run_config.py` line numbers; it can only be trimmed, not removed.

## VERDICT
- Crucial updates: no
