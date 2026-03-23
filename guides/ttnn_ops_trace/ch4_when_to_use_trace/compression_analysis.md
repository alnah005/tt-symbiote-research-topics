# Compression Analysis: Chapter 4 — When to Use Trace — Pass 1

## Summary
- Total files analyzed: 3
- Estimated current line count: ~415 lines
- Estimated post-compression line count: ~330 lines
- Estimated reduction: ~20%

---

## CRUCIAL Suggestions

### [`when_not_to_trace.md`] ~lines 50–60
**Issue:** Disqualifying Condition 3 ("Python Control Flow on Device Outputs") opens by explicitly admitting it is "a generalization of Condition 2." The first two paragraphs restate the readback-causes-wrong-branch problem already explained in Condition 2 (lines 22–46). The only genuinely new content is the structure-invariant vs. structure-variant distinction (lines 54–59) and the `config.use_gated_ffn` vs. `gate_value.item()` contrast.
**Suggestion:** Collapse Condition 3 into a short addendum at the end of Condition 2. Keep only the structure-invariant/structure-variant framing and the `config` flag contrast. Remove the repeated framing paragraph and the "dispatch sequence" restatement. Saves ~10 lines.

### [`when_not_to_trace.md`] ~lines 195–250
**Issue:** The `<details>` block contains the full capture-and-replay boilerplate (`open_device`, pre-allocation, `begin_trace_capture`, `synchronize_device`, decode loop, `release_trace`). The surrounding text (line 129) explicitly states this pattern is "introduced in detail in Chapter 3's `trace_constraints.md`." Reproducing it in full here is a cross-chapter duplicate. The trailing `Note` (lines 252–253) about event barriers vs. `synchronize_device` is also present in Chapter 3.
**Suggestion:** Replace the `<details>` block and its trailing `Note` with a single cross-reference sentence pointing to Chapter 3's `trace_constraints.md` Steps 1–3 for the full setup. Keep only the two focused code snippets (the `decode_step` outer wrapper and `decode_core` inner loop, lines 139–193) that are unique to this chapter's decision-making context. Saves ~55 lines.

### [`latency_sensitive_workloads.md`] ~lines 95–101
**Issue:** The prefill vs. decode comparison table (lines 95–101) is explicitly attributed to "The table from Chapter 3's `trace_constraints.md`" — it is a verbatim cross-chapter duplicate imported into this file.
**Suggestion:** Remove the table. The three bullet points immediately above it (lines 86–91) already cover the same three disqualifying reasons for prefill in prose. A cross-reference to Chapter 3's table is sufficient. Saves ~9 lines.

---

## MINOR Suggestions

### [`index.md`] ~lines 3 and 77–84
**Issue:** The opening paragraph (line 3) summarizes Chapters 1–3 in one sentence each. The "Relationship to Chapters 1, 2, and 3" section (lines 77–84) then re-summarizes the same chapters individually with slightly more prose. The two sections cover the same ground without adding distinct information at either location.
**Suggestion:** Trim the "Relationship" section to a single sentence per chapter that is not already captured in the opening paragraph, or remove the section entirely and fold any net-new detail (e.g., the async-vs-trace regime distinction in line 81) into the opening paragraph. Saves ~5–7 lines.

### [`latency_sensitive_workloads.md`] ~lines 13–15
**Issue:** Lines 13–14 state the conclusion ("Trace helps when the host is on the critical path — when the device would be idle if it weren't for trace"). The immediately following callout note (lines 15–16) restates this in different words ("The regime where the host is on the critical path is exactly the regime described in Chapter 1… if the device is running at close to 100% utilization already, trace will not help"). The note adds only the cross-reference; the conceptual content is duplicated.
**Suggestion:** Fold the cross-reference link into lines 13–14 and delete the note block. Saves ~3 lines.

### [`latency_sensitive_workloads.md`] ~lines 136–138
**Issue:** Two consecutive `> Note` / `> Warning` callouts both end with the same instruction: measure your specific workload. The second callout (lines 138) also restates the maintenance-cost caveat that is covered extensively in `when_not_to_trace.md`.
**Suggestion:** Merge the two callouts into one. Remove the maintenance-cost sentence (it belongs in `when_not_to_trace.md`, not here) and keep only the single reminder that the table numbers are illustrative and that measurement is required. Saves ~3 lines.

### [`when_not_to_trace.md`] ~lines 139–172 (code comments)
**Issue:** The `decode_step` outer wrapper code (lines 139–172) contains over-explained inline comments. Examples: `# This is a host readback, which is why it lives in the outer wrapper.` (line 163), `# EOS detection: also host-side, also in the outer wrapper.` (line 167), and `# Non-blocking: returns before device finishes.` (line 154) — these restate properties of the pattern that the surrounding prose (lines 127–136) already explains fully.
**Suggestion:** Reduce these comments to standard docstring-style identifiers (e.g., `# token sampling` and `# EOS check`) and remove the redundant rationale clauses. Saves ~4 lines and reduces noise-to-signal ratio in the code block.

### [`when_not_to_trace.md`] ~lines 256–269 (summary table)
**Issue:** The summary table repeats the four disqualifying conditions already covered in full sections above and also restates prefill and batch-size content from `latency_sensitive_workloads.md`. The table's "Guidance" column is a one-line paraphrase of each section.
**Suggestion:** Keep the table but remove rows whose guidance is already the title of a section immediately above (dynamic shapes, readback, Python control flow, self-configuring ops). Retain only the cross-cutting rows (loop count, batch size, active development, debugging) that add decision-making guidance not covered by a dedicated section heading. Saves ~4–5 lines and eliminates the sense of rereading the same chapter twice.

---

## Load-Bearing Evidence

- `index.md` line ~23: `"if a decode step takes 5 ms (5,000 us) end-to-end and contains 64 ops, total dispatch overhead is roughly 64 x (17–63 us) = 1.1–4.0 ms"` — load-bearing because this is the only location in the chapter that works through the dispatch-overhead-fraction calculation with concrete numbers that anchor all subsequent threshold reasoning (the flowchart's ">5%" rule, the batch-size table).
- `latency_sensitive_workloads.md` lines ~43–77: the async-dispatch vs. trace-replay ASCII timelines — load-bearing because this is the only visual in the chapter that shows the device-idle gap closing from 135 us to ~55 us; it is the primary evidence for the claimed 2.4x improvement and makes the latency argument concrete rather than asserted.
- `when_not_to_trace.md` lines ~33–44: the `DISQUALIFIED` Python code example showing `ttnn.to_torch()` inside a capture region — load-bearing because it is the only executable illustration of Condition 2's silent failure mode and the comment `# Replay always executes softmax, regardless of current logits.max()` captures the exact hazard that cannot be adequately conveyed in prose alone.
- `when_not_to_trace.md` lines ~54–59: the structure-invariant vs. structure-variant dispatch distinction (`if config.use_gated_ffn` vs. `if ttnn.to_torch(gate_value).item() > 0.5`) — load-bearing because this contrast is the conceptual core of Condition 3 and does not appear elsewhere in the chapter; it is what differentiates benign Python control flow from disqualifying data-dependent dispatch.

---

## VERDICT
- Crucial updates: yes

## Change Log — Pass 1 CRUCIAL Fixes

- `when_not_to_trace.md`: Collapsed Condition 3 into addendum of Condition 2; kept structure-invariant/variant distinction. Replaced <details> capture boilerplate with cross-reference to Ch3 trace_constraints.md.
- `latency_sensitive_workloads.md`: Removed duplicated prefill/decode table; replaced with cross-reference to Ch3.

---

# Compression Analysis: Chapter 4 — When to Use Trace — Pass 2

## Summary
- Total files analyzed: 3
- Estimated current line count: ~433 lines
- Estimated post-compression line count: ~405 lines
- Estimated reduction: ~6%

## CRUCIAL Suggestions
None — all Pass 1 CRUCIAL items resolved.

## MINOR Suggestions

### [`when_not_to_trace.md`] ~lines 62–70 and 72–82
**Issue:** The third bullet of "Captured buffers are pinned" ("If you need to resize any of the traced tensors... you must release the existing trace... and re-capture") duplicates the subject of the immediately following subsection "Shape changes require re-capture." The bullet is a one-sentence preview of the next section's full content.
**Suggestion:** Remove the third bullet. The "Shape changes require re-capture" section (lines 72–82) already covers resizing as a trigger for re-capture. Saves ~3 lines and removes the sense of reading the same point twice in adjacent paragraphs.

### [`when_not_to_trace.md`] ~lines 96–106
**Issue:** Both "Replay errors are silent" and "Per-op debugging is not possible during replay" end with the same conclusion about maintaining a parallel live dispatch path. Line 100–101: "requires you to maintain a live dispatch code path in parallel with the trace code path during testing." Line 106: "The operational cost of this dual-path maintenance grows with model complexity." The dual-path maintenance point is made twice in consecutive subsections.
**Suggestion:** Remove the sentence at line 106 ("The operational cost of this dual-path maintenance grows with model complexity.") since line 100–101 already establishes the requirement. The sentence at line 106 adds no new information; it just re-emphasizes what was just said. Saves ~1 line and tightens the paragraph.

### [`when_not_to_trace.md`] ~lines 143–150 (code comments)
**Issue:** Pass 1 minor item partially unresolved. The code comment at line 143 ("# Replay the traced inner loop. Non-blocking: returns before device finishes.") and line 146 ("# Synchronize to read the output. This is the one mandatory sync per step.") restate properties of `execute_trace` and `synchronize_device` that the surrounding prose at lines 118–125 already explains. The "Non-blocking" and "one mandatory sync per step" details are present in the explanatory text above the code block.
**Suggestion:** Shorten to `# replay traced sequence` and `# mandatory sync before reading output`. Remove the rationale clauses. The line 147–149 comment about event barriers does contain unique guidance and should be kept. Saves ~2 lines of comment text and reduces noise within the code block.

### [`latency_sensitive_workloads.md`] ~lines 13–15
**Issue:** Pass 1 minor item unresolved. Lines 13–14 conclude "Trace helps when the host is on the critical path — when the device would be idle if it weren't for trace." The callout note at line 15 opens "The regime where the host is on the critical path is exactly the regime described in Chapter 1's `host_dispatch_path.md`" — restating the same phrase before adding the cross-reference.
**Suggestion:** Fold the cross-reference into the prose sentence ("...the device would be idle; see Chapter 1's `host_dispatch_path.md`") and delete the separate callout block. Saves ~3 lines.

### [`latency_sensitive_workloads.md`] ~lines 128–130
**Issue:** Pass 1 minor item unresolved. Two consecutive callouts (`> Note` at lines 128–129 and `> Warning` at line 130) both close with an instruction to measure the specific workload. The Note ends: "Measure your specific workload." The Warning ends: "weigh it carefully against the maintenance cost described in `when_not_to_trace.md`" — but its preceding sentence says "Even at batch sizes where trace provides only a moderate speedup, the structural constraints trace imposes... remain in full force" which is a restatement of the table rows above it.
**Suggestion:** Merge the two callouts. Remove the maintenance-cost sentence from the Warning (it belongs in `when_not_to_trace.md` and is covered there); keep only the caveat that the table numbers are illustrative and that measurement is required. Saves ~3 lines.

### [`index.md`] ~lines 3 and 77–84
**Issue:** Pass 1 minor item unresolved. The opening paragraph (line 3) already states "Chapter 1 showed... Chapter 2 showed... Chapter 3 showed..." The "Relationship to Chapters 1, 2, and 3" section (lines 77–84) repeats this per-chapter summary with additional prose. The only content in the Relationship section not covered by the opening paragraph is the async-vs-trace regime distinction in the Chapter 2 entry (line 81) and the phrase "This chapter synthesizes those constraints into a decision, not a re-explanation" (line 83).
**Suggestion:** Remove the Relationship section entirely. Fold the async-vs-trace regime distinction from line 81 into the opening paragraph (line 3) as a clause after the Chapter 2 summary. The "not a re-explanation" note is editorial commentary that can be dropped. Saves ~7 lines.

## Load-Bearing Evidence
- `index.md` line ~23: `"if a decode step takes 5 ms (5,000 us) end-to-end and contains 64 ops, total dispatch overhead is roughly 64 x (17–63 us) = 1.1–4.0 ms"` — load-bearing because this is the only location in the chapter that works through the dispatch-overhead-fraction calculation with concrete numbers, anchoring the flowchart's ">5%" threshold and the batch-size table.
- `latency_sensitive_workloads.md` lines ~43–77: the async-dispatch vs. trace-replay ASCII timelines — load-bearing because this is the only visual showing the device-idle gap closing from 135 us to ~55 us; it is the primary evidence for the 2.4x claim and cannot be conveyed in prose alone.
- `when_not_to_trace.md` lines ~33–44: the `DISQUALIFIED` Python code example with `ttnn.to_torch()` inside a capture region — load-bearing because it is the only executable illustration of Condition 2's silent failure mode; the comment `# Replay always executes softmax, regardless of current logits.max()` captures the exact hazard.
- `when_not_to_trace.md` line ~48: `"An if config.use_gated_ffn branch whose condition is a Python configuration flag resolved at import time is structure-invariant... A if ttnn.to_torch(gate_value).item() > 0.5 branch is structure-variant"` — load-bearing because this contrast is the conceptual core of Condition 2's addendum and does not appear elsewhere in the chapter.
- `when_not_to_trace.md` lines ~128–183: the `decode_step` / `decode_core` code pair — load-bearing because these are the only concrete implementation examples of the traced inner loop / untraced outer wrapper pattern; they show in-place tensor update, non-blocking execute_trace, and host-side token sampling as distinct structural elements.

## VERDICT
- Crucial updates: no
