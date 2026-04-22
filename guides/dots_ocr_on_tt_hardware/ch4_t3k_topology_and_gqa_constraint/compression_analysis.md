# Compression Analysis: Chapter 4 — T3K Topology and GQA Constraint — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~284 lines (index.md: 29, gqa_tp_constraint.md: 87, t3k_submesh_and_env_vars.md: 82, chunked_prefill.md: 86)
- Estimated post-compression line count: ~248 lines
- Estimated reduction: ~13%

---

## CRUCIAL Suggestions

**CRUCIAL-1: `DOTS_MAX_SEQ_LEN` and `DOTS_MAX_SEQ_LEN_WH_LB` defined in two separate env var tables**

- Files: `t3k_submesh_and_env_vars.md` lines 34–35 (env var reference table) and `chunked_prefill.md` lines 43–46 (chunked prefill env vars table)
- What is duplicated: Both files define `DOTS_MAX_SEQ_LEN` and `DOTS_MAX_SEQ_LEN_WH_LB` in their own env var tables. The `t3k_submesh_and_env_vars.md` table is the declared reference for all env vars (its section heading is "Env Var Reference"), yet `chunked_prefill.md` re-defines the same two variables with overlapping prose.
- `DOTS_MAX_SEQ_LEN` is nearly verbatim across both files. The chunked_prefill.md version adds one detail (silent truncation of long OCR documents) that is genuinely new, but the base definition is duplicated.
- `DOTS_MAX_SEQ_LEN_WH_LB` is defined in both tables. The chunked_prefill.md version extends the definition with the lower-bound clarification ("lower bound, not a target"), which adds value, but the opening definition sentence restates what is already in t3k_submesh_and_env_vars.md.
- Fix: Remove the full env var table from `chunked_prefill.md`. Inline the two unique clarifications (silent truncation risk for `DOTS_MAX_SEQ_LEN`; lower-bound-not-target warning for `DOTS_MAX_SEQ_LEN_WH_LB`) as notes within the narrative prose of chunked_prefill.md, immediately following the first mention of each variable. Add a cross-reference sentence: "See [t3k_submesh_and_env_vars.md](t3k_submesh_and_env_vars.md) for the full env var reference." This eliminates ~8 duplicate lines while preserving the one genuinely new detail in each entry.

**CRUCIAL-2: TP>2 shape assertion failure described in full in two files**

- Files: `gqa_tp_constraint.md` lines 58–69 (Failure Modes at TP > 2 section, 12 lines) and `t3k_submesh_and_env_vars.md` line 37 (warning callout within the env var table)
- What is duplicated: `gqa_tp_constraint.md` is the authoritative source for the failure mechanism — it explains the integer-division truncation, the assertion path, and distinguishes this from OOM errors. `t3k_submesh_and_env_vars.md` line 37 repeats the failure outcome ("will not produce degraded performance — it will produce an immediate shape assertion failure") and already redirects to gqa_tp_constraint.md for the derivation.
- The warning callout in t3k_submesh_and_env_vars.md is not wrong, but it re-narrates the failure rather than simply pointing to the source of truth. A reader who has followed the reading order already encountered the full derivation.
- Fix: Shorten the warning in `t3k_submesh_and_env_vars.md` line 37 to a single cross-reference sentence: "> **Warning:** Values above 2 cause an immediate shape assertion failure at model initialization. See [gqa_tp_constraint.md](gqa_tp_constraint.md)." This removes ~1–2 redundant lines from the env var file without touching the authoritative derivation.

**CRUCIAL-3: GQA TP≤2 cause restated in index.md note after full derivation exists in gqa_tp_constraint.md**

- Files: `index.md` lines 19–26 (Quick Reference table and the bold Note callout) and `gqa_tp_constraint.md` lines 1–54 (entire derivation)
- What is duplicated: The index.md Quick Reference table row `num_key_value_heads | 2 | TP ∈ {1, 2} only` and the bold Note callout ("The single biggest deployment constraint for dots.ocr on T3K is GQA head count (`num_key_value_heads=2`)...") re-summarize the core conclusion of the derivation. The Quick Reference table itself is appropriate as a navigation aid; the Note callout below it, however, re-explains the causal chain ("All topology decisions flow from this one architectural choice") that gqa_tp_constraint.md already derives in full.
- Fix: Remove the Note callout from index.md (lines 26–27). The Quick Reference table rows already give readers the key numbers (`num_key_value_heads=2`, `TP ∈ {1, 2}`) before they click through to the derivation. The callout adds no new fact and duplicates the conclusion of a section the reader is about to read. Estimated reduction: 2 lines.

---

## MINOR Suggestions

**MINOR-1: chunked_prefill.md overview paragraph re-states the L1 constraint before the dedicated section**

- File: `chunked_prefill.md` lines 3–5
- The overview says "processing a long prefill sequence in a single pass would exceed the L1 SRAM capacity of Wormhole N300 devices." The "L1 SRAM Constraint" section (lines 13–18) then explains this at length. The overview sentence is a forward-reference that is accurate but adds little beyond what the section heading already signals.
- Fix: Shorten the overview's L1 reference to a clause rather than a full causal sentence. E.g., replace "but processing a long prefill sequence in a single pass would exceed the L1 SRAM capacity of Wormhole N300 devices. The `Generator` class solves this by splitting..." with "but long prefill sequences exceed L1 SRAM on Wormhole N300 devices. The `Generator` class addresses this by splitting...". Saves 1 line.

**MINOR-2: gqa_tp_constraint.md Comparison section closes with two self-evident sentences**

- File: `gqa_tp_constraint.md` lines 83–84
- "This is not a topology decision that can be reversed at inference time. Changing the KV head count would require retraining the model." After the mathematical derivation (which roots the constraint in `num_key_value_heads`), these sentences state the obvious. The derivation already makes clear that the constraint is architectural, not configurational.
- Fix: Delete lines 83–84. The preceding sentence ("dots.ocr achieves a comparable compression ratio with far fewer parameters total (~3B), but the lower absolute KV head count is the side effect that constrains TP to 2 on T3K.") is a sufficient closing observation.

**MINOR-3: t3k_submesh_and_env_vars.md "Why open the full mesh first?" paragraph is a re-explanation of steps already shown**

- File: `t3k_submesh_and_env_vars.md` lines 25–26
- The three-step submesh open protocol (lines 17–24) makes the full-mesh-first requirement implicit. The "Why open the full mesh first?" paragraph then narrates the same reasoning in prose. The race condition detail is load-bearing (see Load-Bearing Evidence below), but the opening clause ("Why open the full mesh first? The Galaxy interconnect requires the host to register all 8 devices as a single logical unit before any sub-allocation is possible.") merely restates step 1's justification.
- Fix: Trim to the load-bearing race condition detail only. Remove the opening restatement of step 1 and keep only: "A partial open leaves the remaining devices in an undefined ownership state that can cause initialization failures for any subsequent process attempting to claim them. Opening the full mesh atomically prevents this race condition." Saves ~1 line.

**MINOR-4: chunked_prefill.md "Why Long Sequences" section over-explains OCR token production before reaching the constraint**

- File: `chunked_prefill.md` lines 8–11
- Lines 8–10 describe what OCR inputs look like (image resolution, vision tokens, multi-page scans) before the actual engineering point (prefill sequence length exceeds 4,096 tokens). The context is useful but two sentences could do the work of four.
- Fix: Collapse lines 8–11 to: "A 896×1344 OCR image produces 1536 vision tokens; a multi-page or dense-text scan can push total prefill length well past 4,096 tokens. At `max_position_embeddings=131072`, all input tokens must pass through all 28 transformer layers before decode begins — the prefill phase is where this cost is paid." Saves ~2 lines.

---

## Load-Bearing Evidence

The following content must not be cut. Each item is unique technical fact, causal chain, or concrete data not replicated elsewhere in Chapter 4.

1. **GQA divisibility derivation** — `gqa_tp_constraint.md` lines 27–54. The full step-by-step derivation (GCD computation, divisor enumeration, per-TP-candidate table) is the mathematical foundation of the chapter. The index.md Quick Reference gives conclusions only; this is the only place the reasoning is shown.

2. **Integer-division truncation path for TP=4** — `gqa_tp_constraint.md` lines 64–68. The specific failure path (`2 // 4 = 0`, shape assertion on KV head count) distinguishes this error from OOM errors. This causal chain is not reproduced elsewhere.

3. **Warning: KV head shape failures vs. OOM errors** — `gqa_tp_constraint.md` lines 69–71 (two Warning/Note callouts). The distinction "structural error, not numerical or memory error; caught deterministically at startup before device memory is allocated for activations" is load-bearing for operators debugging startup failures. It appears only here.

4. **Qwen 2.5 VL 7B comparison table** — `gqa_tp_constraint.md` lines 77–83. The side-by-side comparison (head counts, GCD, max TP) and the GQA ratio discussion (6:1 for dots.ocr, 7:1 for Qwen2.5-VL-7B at 7B parameters) appear nowhere else and provide the context for why dots.ocr's architecture choice was reasonable despite the TP limit.

5. **Galaxy interconnect ownership and race condition** — `t3k_submesh_and_env_vars.md` lines 25–26. The explanation that a partial open leaves devices in an undefined state causing initialization failures for other processes is operationally critical and unique to this file.

6. **Mesh teardown ordering requirement** — `t3k_submesh_and_env_vars.md` lines 40–46. The requirement to release the submesh before the parent mesh, and the consequence of violating this order (dangling device references, Galaxy fabric corruption), is a correctness constraint. It does not appear in any other file.

7. **Scheduling implications: 8-device accounting** — `t3k_submesh_and_env_vars.md` lines 49–56. The three concrete scheduling consequences (Llama-3 70B blocked, two dots.ocr instances blocked, 8-device accounting rule) are unique to this file and operationally critical for shared-server deployments.

8. **LM head memory budget calculation** — `t3k_submesh_and_env_vars.md` lines 62–79. The derivation of op counts (75 ops at TP=1, 38 ops at TP=2) from `vocab_size=151936` and `DOTS_LM_HEAD_MAX_COLUMNS_PER_DEVICE=2048` is concrete numerical data not repeated elsewhere.

9. **L1 SRAM scaling analysis** — `chunked_prefill.md` lines 13–18. The O(S × hidden_size) activation scaling and O(S²) attention score scaling characterization is the engineering justification for chunked prefill. It does not appear in the other files.

10. **TP=2 synchronization constraint on chunk boundaries** — `chunked_prefill.md` lines 50–60. The requirement that chunk boundaries be identical on both devices due to Galaxy collective ops (all-reduce, all-gather) is unique to this file and directly constrains how `DOTS_MAX_SEQ_LEN_WH_LB` can be used.

11. **TTFT formula and concrete example** — `chunked_prefill.md` lines 64–79. The TTFT formula, the 1536-token / 512-chunk-size example (3 chunks × 80 ms = 240 ms minimum), and the benchmark guidance are quantitative data not repeated elsewhere.

12. **`DOTS_MAX_SEQ_LEN_WH_LB` lower-bound clarification** — `chunked_prefill.md` lines 47–48 (Warning callout). The distinction that this variable is a lower bound, not a target chunk size, and that actual chunks may be larger is a correctness note. If the env var table in chunked_prefill.md is removed (per CRUCIAL-1), this warning must be preserved inline.

---

## VERDICT
- Crucial updates: yes

## Agent A Change Log — Pass 1

- CRUCIAL-1 applied: Removed full env var table from chunked_prefill.md; replaced with cross-reference to t3k_submesh_and_env_vars.md and two inline notes preserving the unique clarifications (silent truncation risk for DOTS_MAX_SEQ_LEN; lower-bound-not-target warning for DOTS_MAX_SEQ_LEN_WH_LB).
- CRUCIAL-2 applied: Shortened TP>2 warning in t3k_submesh_and_env_vars.md to a single cross-reference sentence pointing to gqa_tp_constraint.md.
- CRUCIAL-3 applied: Removed Note callout from index.md; Quick Reference table rows are sufficient navigation aids.

---

# Compression Analysis: Chapter 4 — T3K Topology and GQA Constraint — Pass 2

## Summary

| File | Current lines | Est. post-compression | Reduction |
|------|--------------|-----------------------|-----------|
| index.md | 26 | 26 | 0 lines (0%) |
| gqa_tp_constraint.md | 86 | 84 | 2 lines (2%) |
| t3k_submesh_and_env_vars.md | 81 | 80 | 1 line (1%) |
| chunked_prefill.md | 85 | 81 | 4 lines (5%) |
| **Total** | **278** | **271** | **7 lines (~3%)** |

Pass 1 reduced the chapter from ~284 lines to 278 lines (6 lines removed). Pass 2 minor-only suggestions would yield an additional ~7 lines, bringing the estimated total reduction across both passes to ~13 lines (~5% of the original 284).

---

## Pass 1 Fix Verification

**CRUCIAL-1 — Env var table removed from chunked_prefill.md:** RESOLVED. `chunked_prefill.md` lines 43–48 now contain a cross-reference sentence ("See [t3k_submesh_and_env_vars.md](t3k_submesh_and_env_vars.md) for the full env var reference.") followed by two inline bullet points naming only the two chunked-prefill-relevant variables, with the lower-bound Warning callout preserved immediately after. No table is present; no variable is re-defined.

**CRUCIAL-2 — TP>2 warning shortened in t3k_submesh_and_env_vars.md:** RESOLVED. Line 37 now reads: `> **Warning:** Values above 2 cause an immediate shape assertion failure at model initialization. See [gqa_tp_constraint.md](gqa_tp_constraint.md).` — a single cross-reference sentence. The prior re-narration of the failure mechanism is gone.

**CRUCIAL-3 — Note callout removed from index.md:** RESOLVED. `index.md` ends at line 26 with the "**Next:** [GQA TP Constraint](...)" link. No Note callout restating the GQA causal chain is present. The Quick Reference table rows (`num_key_value_heads | 2 | TP ∈ {1, 2} only`, etc.) remain as appropriate navigation aids.

---

## CRUCIAL Suggestions

None identified. All three Pass 1 CRUCIAL items are resolved. No new duplicate-content blocks spanning multiple files were introduced by the Pass 1 edits.

---

## MINOR Suggestions

The four MINOR items from Pass 1 were not addressed in Pass 1 and remain valid. They are restated here with current line references for Pass 2 action.

**MINOR-1 (carry-forward): chunked_prefill.md overview re-states the L1 constraint before the dedicated section**

- File: `chunked_prefill.md` line 5
- The overview sentence "but processing a long prefill sequence in a single pass would exceed the L1 SRAM capacity of Wormhole N300 devices" pre-empts the "L1 SRAM Constraint" section (lines 13–17) without adding information. A reader scanning section headings already knows what that section covers.
- Fix: Shorten to a forward clause. Replace "but processing a long prefill sequence in a single pass would exceed the L1 SRAM capacity of Wormhole N300 devices. The `Generator` class solves this by splitting the prefill phase into fixed-size chunks, each of which fits within the device's L1 budget." with "but long prefill sequences exceed L1 SRAM on Wormhole N300 devices, so the `Generator` class splits the prefill phase into fixed-size chunks." Saves 1 line.

**MINOR-2 (carry-forward): gqa_tp_constraint.md Comparison section closes with two self-evident sentences**

- File: `gqa_tp_constraint.md` lines 83–84
- "This is not a topology decision that can be reversed at inference time. Changing the KV head count would require retraining the model." The mathematical derivation already roots the constraint in `num_key_value_heads` as an architectural constant. These sentences state what the preceding derivation makes obvious.
- Fix: Delete lines 83–84. The sentence ending "...but the lower absolute KV head count is the side effect that constrains TP to 2 on T3K." is a sufficient closing observation for the comparison section. Saves 2 lines.

**MINOR-3 (carry-forward): t3k_submesh_and_env_vars.md "Why open the full mesh first?" opening clause re-states step 1**

- File: `t3k_submesh_and_env_vars.md` line 25
- The paragraph opens: "Why open the full mesh first? The Galaxy interconnect requires the host to register all 8 devices as a single logical unit before any sub-allocation is possible." Step 1 of the protocol (line 17) already says: "the function calls the TTNN mesh device API to claim all 8 devices." The rhetorical question and its first sentence repeat the step header. The load-bearing content — the race-condition consequence of a partial open — begins in the second sentence.
- Fix: Remove the opening question and first sentence, keeping only: "A partial open — claiming only 2 of the 8 N300 cards — leaves the remaining 6 in an undefined ownership state that can cause initialization failures for any subsequent process attempting to claim them. Opening the full mesh atomically prevents this race condition." Saves ~1 line.

**MINOR-4 (carry-forward): chunked_prefill.md "Why Long Sequences" section over-explains OCR token production**

- File: `chunked_prefill.md` lines 9–11
- Lines 9–10 describe what OCR inputs look like (image resolution, vision tokens, multi-page scans, dense text) before reaching the engineering point (prefill length exceeds 4,096 tokens). The context is useful but uses four sentences where two would suffice.
- Fix: Collapse to: "A 896×1344 OCR image produces 1536 vision tokens; a multi-page or dense-text scan can push total prefill length well past 4,096 tokens. At `max_position_embeddings=131072`, all input tokens must pass through all 28 transformer layers before decode begins — the prefill phase is where this cost is paid." Saves ~2 lines.

---

## Load-Bearing Evidence

The following content remains load-bearing and must not be cut in Pass 2 or any subsequent pass. Items 1–12 from Pass 1 are reproduced with updated line numbers where Pass 1 edits shifted content.

1. **GQA divisibility derivation** — `gqa_tp_constraint.md` lines 27–54. Step-by-step GCD computation, divisor enumeration, and per-TP-candidate validity table. Conclusions only are shown in index.md; reasoning is unique to this file.

2. **Integer-division truncation path for TP=4** — `gqa_tp_constraint.md` lines 64–68. The specific failure path (`2 // 4 = 0`, shape assertion on KV head count) distinguishing this error from OOM errors. Not reproduced elsewhere.

3. **KV head shape failure vs. OOM error distinction** — `gqa_tp_constraint.md` lines 69–71 (Note + Warning callouts). "Structural error, not numerical or memory error; caught deterministically at startup before device memory is allocated for activations." Critical for operators debugging startup failures; unique to this file.

4. **Qwen 2.5 VL 7B comparison table** — `gqa_tp_constraint.md` lines 77–82. Side-by-side head-count, GCD, and max-TP comparison plus GQA ratio discussion (6:1 for dots.ocr, 7:1 for Qwen2.5-VL-7B). Appears nowhere else; provides architectural context for the TP limit.

5. **Galaxy interconnect race condition** — `t3k_submesh_and_env_vars.md` line 25 (second sentence onward). The undefined-ownership-state consequence of a partial open, and the atomic full-mesh open as the fix. Operationally critical; unique to this file.

6. **Mesh teardown ordering requirement** — `t3k_submesh_and_env_vars.md` lines 40–46. Submesh must be released before parent mesh; violating order causes dangling device references and Galaxy fabric corruption. Correctness constraint not present in any other file.

7. **Scheduling implications: 8-device accounting** — `t3k_submesh_and_env_vars.md` lines 49–56. Three concrete consequences: Llama-3 70B blocked, two dots.ocr instances cannot co-run, 8-device accounting required. Operationally critical for shared-server deployments; unique to this file.

8. **LM head memory budget calculation** — `t3k_submesh_and_env_vars.md` lines 62–79. Derivation of op counts (75 at TP=1, 38 at TP=2) from `vocab_size=151936` and `DOTS_LM_HEAD_MAX_COLUMNS_PER_DEVICE=2048`. Concrete numerical data not reproduced elsewhere.

9. **L1 SRAM scaling analysis** — `chunked_prefill.md` lines 13–17. O(S × hidden_size) activation scaling and O(S²) attention score scaling as the engineering basis for chunked prefill. Not present in other files.

10. **TP=2 synchronization constraint on chunk boundaries** — `chunked_prefill.md` lines 50–60. Chunk boundaries must be identical on both submesh devices due to Galaxy collective ops (all-reduce, all-gather). Directly constrains use of `DOTS_MAX_SEQ_LEN_WH_LB`; unique to this file.

11. **TTFT formula and concrete example** — `chunked_prefill.md` lines 64–79. TTFT formula, 1536-token / 512-chunk-size worked example (3 chunks × 80 ms = 240 ms minimum), and benchmark guidance. Quantitative data not repeated elsewhere.

12. **`DOTS_MAX_SEQ_LEN_WH_LB` lower-bound Warning callout** — `chunked_prefill.md` line 48. The distinction that this variable is a lower bound, not a target chunk size, and that actual chunks may be larger. Preserved inline per CRUCIAL-1 resolution; must not be removed.

---

## VERDICT

Crucial updates: no
