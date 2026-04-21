# Compression Analysis: Chapter 1 — Config Diff — Pass 1

## Summary
- Total files analyzed: 3
- Estimated current line count: ~619 lines
- Estimated post-compression line count: ~480 lines
- Estimated reduction: ~22%

## CRUCIAL Suggestions

### [structural_fields.md] ~lines 53–107
**Issue:** The "Shape Derivations" LaTeX block restates in equation form the exact numeric shapes already spelled out in the `Governed weight dimensions` column of the table directly above it (e.g., the table already says `q_proj` shape = `[8192, 7168]`; the LaTeX simply re-derives that same number). The block adds ~55 lines of notation for zero new information.
**Suggestion:** Delete the entire "Shape Derivations" subsection (lines 53–107). The closing sentence "Because every hyperparameter entering these expressions is identical in both configs, the shapes are provably identical without inspecting the safetensors files." can be promoted as a one-line closing remark directly after the table if the point must be preserved.

### [structural_fields.md] ~lines 148–169
**Issue:** The four-item numbered list under "Why Identical Structural Fields Mean No TTNN Changes Are Needed" uses one long paragraph per point, but each point resolves to the same underlying claim: identical hyperparameters → identical shapes → no TTNN changes. Items 1–3 are paraphrases of each other (matmul config, weight preprocessing, tensor allocation all reduce to "same shapes, nothing changes").
**Suggestion:** Collapse to a single short paragraph (3–4 sentences) or a compact 4-item bulleted list with no sub-prose. Estimated saving: ~12 lines.

### [index.md] ~lines 55–75
**Issue:** The "Notes on the Diff Table" section (lines 55–75) re-explains each of the three diff-table rows in prose. The table rows themselves already name the change type and the downstream chapter link. The notes duplicate this and preview content from `new_and_modified_fields.md`, creating a three-way repetition (table → notes → new_and_modified_fields.md).
**Suggestion:** Remove the "Notes on the Diff Table" section entirely. The table's `TTNN impact analysed in` column already points readers to the right chapter. If a one-line risk summary is wanted, it can be folded into a `Notes` column in the table itself.

### [new_and_modified_fields.md] ~lines 179–187
**Issue:** The subsection "What Happens Without `bos_token_id` (Qwen3.5 Behaviour)" (lines 179–187) repeats information already established in the preceding paragraph (lines 121–126) of the same section: that Qwen3.5 has no `bos_token_id`, that the HuggingFace default is `None`, and that users always supply `input_ids`. The subsection adds only the detail about `ValueError` in recent transformers — which is a minor edge case that doesn't change the TTNN conclusion.
**Suggestion:** Delete the subsection heading and merge the one non-redundant sentence ("Adding `bos_token_id: 248044` to Qwen3.6 enables the `input_ids=None` code path without crashing") into the preceding paragraph. Estimated saving: ~8 lines.

## MINOR Suggestions

### [new_and_modified_fields.md] ~lines 341–347
**Issue:** The paragraph below the "Other Fields: No Additional Changes" table repeats, in prose, the point already made in the `Notes` column for `temperature` and `top_p` — that these are `generation_config.json` fields, not `config.json` fields. The table column already says "Not a `config.json` field; belongs in `generation_config.json`" for both rows.
**Suggestion:** Delete the closing paragraph (lines 345–347). The table notes are sufficient.

### [new_and_modified_fields.md] ~lines 272–295
**Issue:** The "Weight Keys Introduced by the MTP Head" subsection restates the attention projection shapes (`[8192, 7168]`, `[512, 7168]`, etc.) that `structural_fields.md` already derives and tabulates. The parenthetical `# same shape as backbone` comments on lines 279–282 explicitly flag that the information is not new.
**Suggestion:** Remove the per-line shape comments from the code block (they are already in `structural_fields.md`) and trim the explanatory sentence that follows to remove the phrase "All shapes are derived from the same hyperparameters as the backbone (`hidden_size = 7168`, `num_attention_heads = 64`, `head_dim = 128`, `num_key_value_heads = 4`)." — readers can cross-reference.

### [structural_fields.md] ~lines 186–188
**Issue:** The closing sentence "The values in this table are consistent between the two versions. They are documented here to confirm there are no hidden numerical changes that could alter TTNN kernel selection or numerical behaviour at inference time." restates the section's purpose already stated in the opening sentence of "Purpose of This File."
**Suggestion:** Delete those two sentences; the table title "Fields Confirmed Identical" already conveys this.

### [index.md] ~lines 19–28
**Issue:** The "After reading this chapter you will know:" bullet list previews the same four topics already covered by the Quick-Reference Diff Table plus the Reading Order section immediately below. It is a third pass over the same outline in three successive paragraphs.
**Suggestion:** Remove the four-bullet "you will know" list. The Reading Order section (which describes each file's content) plus the Quick-Reference Diff Table together serve the same navigation function more concisely.

### [new_and_modified_fields.md] ~lines 258–268
**Issue:** The paragraph "Even if a future version of `Qwen3_5MoeForConditionalGeneration` did instantiate the MTP head..." (lines 258–268) is hedging language. The section already establishes that the current class does not invoke MTP during standard decode; this paragraph speculates about a future version to make the same point a second time.
**Suggestion:** Delete this paragraph. The conclusion (MTP is not called during standard decode) is already established in the preceding paragraph.

## Load-Bearing Evidence

- `index.md` line ~49: the Quick-Reference Diff Table (header + 3 data rows) — load-bearing because it is the only place in the chapter that presents all three config changes side-by-side in a scannable format with change-type classification and chapter pointers; removing it would eliminate the primary navigation artifact for the chapter.
- `structural_fields.md` line ~39: the "Structural Hyperparameters Governing Weight Shapes" table — load-bearing because it is the authoritative single-location record linking each config field to its concrete governed weight dimension; the shape derivation math block restates it, but the table itself must be kept.
- `new_and_modified_fields.md` lines ~36–68: the two JSON code blocks showing where `partial_rotary_factor` lives in each config — load-bearing because the distinction between the nested `rope_parameters` location (Qwen3.5) and the additional top-level location (Qwen3.6) is the entire substance of that section; prose alone cannot replace the visual diff.

## VERDICT
- Crucial updates: yes

---

## B Feedback Application Log — Pass 1

- Item 1: Changed "≈ 205M parameters" to "≈ 169M parameters" in `new_and_modified_fields.md`
- Item 2: Corrected MTP FFN shapes from moe_intermediate_size=2048 to intermediate_size=14336 (dense MTP head); updated all weight key shapes and recalculated parameter count
- Item 3: Renamed section heading in `structural_fields.md` from "Fields That Differ Numerically..." to "Fields Confirmed Identical — Listed for Completeness"

## B Feedback Application Log — Pass 2

- Item 1: Updated MTP parameter count from "≈ 169M" to "≈ 433M" and verified FFN equation term uses intermediate_size=14336 in `new_and_modified_fields.md`

## B Feedback Application Log — Pass 3

- Item 1: Removed incorrect `<|im_start|>` identification for `bos_token_id = 248044` in `new_and_modified_fields.md`; replaced with a note to inspect the Qwen3.6 tiktoken vocabulary file directly

## C Compression Application Log — Pass 1

- C1: Deleted "Shape Derivations" LaTeX subsection from `structural_fields.md`; promoted closing remark after table
- C2: Collapsed four-item numbered list to compact bullet list in `structural_fields.md`
- C3: Removed "Notes on the Diff Table" section from `index.md`
- C4: Merged "What Happens Without bos_token_id" subsection into preceding paragraph in `new_and_modified_fields.md`

---

# Compression Analysis: Chapter 1 — Config Diff — Pass 2

## Summary

- Files re-analyzed: 3
- Current line count: `index.md` ~54 lines, `structural_fields.md` ~126 lines, `new_and_modified_fields.md` ~345 lines; total ~525 lines
- Estimated post-compression line count: ~490 lines
- Estimated reduction this pass: ~35 lines (~7%)

## CRUCIAL Suggestions

### [structural_fields.md + new_and_modified_fields.md] Duplicate `d_rot` LaTeX equation

**Issue:** The rotary-dimension derivation equation

```
d_rot = floor(128 * 0.25) = 32
```

appears twice verbatim: in `structural_fields.md` lines 83–91 (inside "RoPE and Rotary Embedding Fields") and in `new_and_modified_fields.md` lines 22–29 (inside the `partial_rotary_factor` section). Both instances produce the same number, state the same conclusion ("32 of each head's 128 dimensions are rotated"), and reference the same values. Neither adds anything the other does not already say.

**Suggestion:** Delete the equation block and its surrounding two sentences from `structural_fields.md` (the instance inside the RoPE table section). The `new_and_modified_fields.md` instance is the correct home for this derivation because that file is where the `partial_rotary_factor` change is analysed. The `structural_fields.md` table already records `partial_rotary_factor: 0.25` in its `Notes` column; the numeric consequence (`d_rot = 32`) is available by cross-reference. Estimated saving: ~9 lines.

## MINOR Suggestions

### [new_and_modified_fields.md] ~lines 248–262 — Speculative "future version" paragraph (carried from Pass 1, still present)

**Issue:** The paragraph beginning "Even if a future version of `Qwen3_5MoeForConditionalGeneration` did instantiate the MTP head..." re-argues the conclusion already established in the paragraph immediately above it (the current class does not invoke MTP during standard decode). The paragraph is conditional speculation that does not add a new fact.

**Suggestion:** Delete the paragraph (~15 lines). The conclusion is already stated; the speculation dilutes the factual clarity of the section.

### [index.md] ~lines 19–28 — "After reading this chapter you will know" list (carried from Pass 1, still present)

**Issue:** The four-item "you will know" list previews the same four topics already covered in the Quick-Reference Diff Table and the Reading Order section that follow it immediately. This is a third pass over the same outline.

**Suggestion:** Remove the four-bullet list. The Reading Order section plus the Diff Table together serve the same navigation purpose more concisely. Estimated saving: ~8 lines.

### [structural_fields.md] ~lines 119–121 — Closing two sentences in "Fields Confirmed Identical" (carried from Pass 1, still present)

**Issue:** "The values in this table are consistent between the two versions. They are documented here to confirm there are no hidden numerical changes..." restates the section purpose already declared in the "Purpose of This File" opening and in the table section heading itself.

**Suggestion:** Delete those two sentences. Estimated saving: ~3 lines.

### [new_and_modified_fields.md] ~lines 339–344 — Closing paragraph after "Other Fields: No Additional Changes" table (carried from Pass 1, still present)

**Issue:** The paragraph re-explains in prose the point already made in the `Notes` column for `temperature` and `top_p` rows — that these belong in `generation_config.json`, not `config.json`. The table notes already say "Not a `config.json` field; belongs in `generation_config.json`" for both rows.

**Suggestion:** Delete the closing paragraph (~3 lines). The table notes are sufficient.

### [new_and_modified_fields.md] ~lines 270–280 — Per-line shape comments in MTP weight-key code block (carried from Pass 1, still present)

**Issue:** Comments such as `# [8192, 7168]  same shape as backbone` and `# [7168]  RMSNorm over hidden_size` restate shape information that is already tabulated in `structural_fields.md`. The phrase "same shape as backbone" explicitly acknowledges the redundancy.

**Suggestion:** Remove the inline shape comments from the code block. The surrounding prose already establishes that MTP shapes derive from the same backbone hyperparameters. Estimated saving: ~5 lines of comment noise within the block.

## Load-Bearing Evidence

- `index.md` line 49–54: The Quick-Reference Diff Table (3 data rows with change-type and chapter-pointer columns) — load-bearing as the only place all three config changes appear side-by-side with downstream chapter links; it cannot be cut.
- `structural_fields.md` lines 39–53: The "Structural Hyperparameters Governing Weight Shapes" table — load-bearing as the authoritative single-location record linking each field to its concrete governed weight dimension; the table itself must be kept even as surrounding prose is trimmed.
- `new_and_modified_fields.md` lines 34–69: The two JSON code blocks showing where `partial_rotary_factor` lives in each config (nested-only for Qwen3.5 vs. also top-level for Qwen3.6) — load-bearing because the structural distinction is the entire substance of that section.

## VERDICT

Crucial updates: yes

(One new CRUCIAL issue found: duplicate `d_rot` LaTeX equation across `structural_fields.md` and `new_and_modified_fields.md`. All four Pass-1 CRUCIAL items are confirmed resolved.)

## C Compression Application Log — Pass 2

- C1: Deleted duplicate d_rot equation block from `structural_fields.md` (kept only in `new_and_modified_fields.md`)

## B Feedback Application Log — Pass 6

- Item 1: Added "Prerequisites" section to `index.md` with three prerequisite rows (HuggingFace config format, Qwen3.5-35B-A3B architecture, TTNN weight loading)

---

# Compression Analysis: Chapter 1 — Config Diff — Pass 3

## Summary
- Files re-analyzed: 3
- Current line count: `index.md` ~64 lines, `structural_fields.md` ~116 lines, `new_and_modified_fields.md` ~344 lines; total ~524 lines
- Estimated post-compression line count: ~505 lines
- Estimated reduction this pass: ~19 lines (~4%)

## CRUCIAL Suggestions

### [new_and_modified_fields.md] lines 247–260 — Speculative "MTP During Standard Autoregressive Decoding" section (carried from Pass 1 and Pass 2 MINOR; re-classified CRUCIAL because ≥5 lines remain uncut)

**Issue:** The entire subsection "### MTP During Standard Autoregressive Decoding" (heading + 14 body lines) argues from a hypothetical: "Even if a future version of `Qwen3_5MoeForConditionalGeneration` did instantiate the MTP head, the MTP module would not be called…". The preceding subsection (lines 228–245) already establishes that the current class does not instantiate the MTP head and that MTP weights are silently skipped by `load_state_dict`. This subsection adds no new fact — it speculates about a version that does not exist in order to re-confirm a conclusion already reached. Both Pass 1 and Pass 2 MINOR suggestions flagged it; it has not been cut.

**Suggestion:** Delete the entire "### MTP During Standard Autoregressive Decoding" subsection (lines 247–260, ~14 lines). The conclusion — MTP is inference-inactive — is fully established in the preceding subsection. If the speculative decoding / training distinction is considered load-bearing, it can be reduced to a single parenthetical sentence appended to line 245.

## MINOR Suggestions

### [new_and_modified_fields.md] lines 270–279 — Inline shape comments in MTP weight-key code block (carried from Pass 1 and Pass 2)

**Issue:** Every line of the MTP key listing carries an inline `# [shape]` comment (e.g., `# [8192, 7168]  same shape as backbone`, `# [7168]  RMSNorm over hidden_size`). The comment "same shape as backbone" explicitly acknowledges the shapes are not new information — they are already tabulated in `structural_fields.md`. The surrounding prose (lines 282–288) repeats the same shapes in sentence form.

**Suggestion:** Remove all inline `# [...]` shape comments from the code block (10 comment strings across the 10 key lines). Estimated saving: ~3–5 lines of horizontal noise; preserves the key-name listing, which is load-bearing.

### [new_and_modified_fields.md] lines 337–340 — Closing paragraph after "Other Fields: No Additional Changes" table (carried from Pass 1 and Pass 2)

**Issue:** The paragraph "Generation defaults such as `temperature` and `top_p` are stored in a separate `generation_config.json` file…" restates what is already in the `Notes` column of the two table rows for `temperature` and `top_p` ("Not a `config.json` field; belongs in `generation_config.json`"). The table notes are unambiguous; the closing paragraph adds only the word "sampling" and no new information.

**Suggestion:** Delete lines 337–340 (~4 lines). The table notes are sufficient.

### [index.md] lines 19–28 — "After reading this chapter you will know" bullet list (carried from Pass 1 and Pass 2)

**Issue:** The four-bullet "you will know" list previews the same four topics already covered immediately below it in the "Quick-Reference Diff Table" (line 59–63) and the "Reading Order" section (lines 42–50). This is the third consecutive pass-through of the same four topics in the same file.

**Suggestion:** Remove the four-bullet list (~8 lines). The Reading Order section and the Diff Table together cover the same navigation function. If a brief forward-pointer is wanted, the opening paragraph (lines 11–16) already summarises the three key changes.

### [structural_fields.md] lines 109–111 — Closing two sentences in "Fields Confirmed Identical" section (carried from Pass 1 and Pass 2)

**Issue:** "The values in this table are consistent between the two versions. They are documented here to confirm there are no hidden numerical changes that could alter TTNN kernel selection or numerical behaviour at inference time." This restates the section's declared purpose in the "Purpose of This File" opening (lines 5–11) and is already implied by the section heading "Fields Confirmed Identical — Listed for Completeness."

**Suggestion:** Delete lines 109–111 (~3 lines). The heading and the "Purpose" section already communicate this.

## Load-Bearing Evidence
- `index.md` line ~59: The Quick-Reference Diff Table (3 data rows) — the only place in the chapter that presents all three config changes side-by-side with change-type classification and chapter pointers; must not be cut.
- `structural_fields.md` lines ~39–53: The "Structural Hyperparameters Governing Weight Shapes" table — the authoritative single-location record linking each config field to its concrete governed weight dimension; the table itself must be kept.
- `new_and_modified_fields.md` lines ~34–69: The two JSON code blocks showing `partial_rotary_factor` nested-only in Qwen3.5 vs. also top-level in Qwen3.6 — the structural distinction is the entire substance of that section and cannot be replaced by prose alone.

## VERDICT
- Crucial updates: yes

## C Compression Application Log — Pass 3

- C1: Deleted entire "MTP During Standard Autoregressive Decoding" subsection from `new_and_modified_fields.md`

---

# Compression Analysis: Chapter 1 — Config Diff — Pass 4

## Summary
- Files re-analyzed: 3
- Current line count: `index.md` ~64 lines, `structural_fields.md` ~116 lines, `new_and_modified_fields.md` ~330 lines; total ~510 lines
- Estimated post-compression: ~492 lines
- Estimated reduction this pass: ~18 lines (~4%)

## CRUCIAL Suggestions

None.

## MINOR Suggestions

### [index.md] lines 19–28 — "After reading this chapter you will know" bullet list (carried from Pass 1, Pass 2, and Pass 3; never applied)

**Issue:** The four-bullet "you will know" list at lines 19–28 previews the same four topics covered immediately below it in the "Reading Order" section (lines 42–50) and the "Quick-Reference Diff Table" (lines 59–63). Every bullet maps one-to-one onto a diff-table row plus a reading-order entry; there is no information here that does not appear in those two constructs. This is the fourth consecutive pass in which this item has been flagged and not cut.

**Suggestion:** Delete the four-bullet "you will know" list (~10 lines including the heading line). The opening paragraph (lines 11–16) already summarises the three key changes; the Reading Order section and Diff Table together handle navigation. Estimated saving: ~10 lines.

### [structural_fields.md] lines 109–111 — Closing two sentences in "Fields Confirmed Identical" section (carried from Pass 1, Pass 2, and Pass 3; never applied)

**Issue:** "The values in this table are consistent between the two versions. They are documented here to confirm there are no hidden numerical changes that could alter TTNN kernel selection or numerical behaviour at inference time." This is a restatement of the section heading ("Fields Confirmed Identical — Listed for Completeness") and of the "Purpose of This File" paragraph already at the top of the file. The section heading is unambiguous; no closing restatement is needed.

**Suggestion:** Delete lines 109–111 (~3 lines). Estimated saving: ~3 lines.

### [new_and_modified_fields.md] lines 270–279 — Inline `# [shape]` comments in MTP weight-key code block (carried from Pass 1, Pass 2, and Pass 3; never applied)

**Issue:** All ten key lines in the MTP weight-key listing carry inline `# [shape]` comments (e.g., `# [8192, 7168]  same shape as backbone`, `# [7168]  RMSNorm over hidden_size`). The comment text "same shape as backbone" explicitly acknowledges the information is redundant with `structural_fields.md`. The prose immediately following the code block (lines 282–288) repeats the same shapes in sentence form. The key names themselves are the load-bearing content; the comments add only horizontal noise.

**Suggestion:** Strip all inline `# [...]` shape comments from the ten key lines. Estimated saving: ~2–4 lines of horizontal width per line (no line deletions, but the code block becomes significantly more scannable and self-contained).

### [new_and_modified_fields.md] lines 337–340 — Closing paragraph after "Other Fields: No Additional Changes" table (carried from Pass 1, Pass 2, and Pass 3; never applied)

**Issue:** The paragraph "Generation defaults such as `temperature` and `top_p` are stored in a separate `generation_config.json` file alongside the checkpoint, not in `config.json` itself. Changes to those files affect sampling behaviour but have no TTNN module impact." re-explains what the `Notes` column already says for both the `temperature` and `top_p` rows: "Not a `config.json` field; belongs in `generation_config.json`." The only new word the paragraph contributes is "sampling," which adds nothing actionable.

**Suggestion:** Delete lines 337–340 (~4 lines). The table notes are sufficient and self-explanatory.

## Load-Bearing Evidence
- `index.md` line ~59: The Quick-Reference Diff Table (3 data rows with `Field`, `Qwen3.5 value`, `Qwen3.6 value`, `Change type`, and `TTNN impact analysed in` columns) — the only place in the chapter that presents all three config changes side-by-side in a scannable format with downstream chapter links; must not be cut.
- `structural_fields.md` lines ~39–53: The "Structural Hyperparameters Governing Weight Shapes" table — the authoritative single-location record linking each config field (`hidden_size`, `num_hidden_layers`, `head_dim`, etc.) to its concrete governed weight dimension; the surrounding explanatory prose can be trimmed but the table itself must be kept.
- `new_and_modified_fields.md` lines ~34–69: The two JSON code blocks showing `partial_rotary_factor` nested-only inside `rope_parameters` for Qwen3.5 vs. also present at the root level for Qwen3.6 — the structural distinction is the entire substance of the `partial_rotary_factor` section and cannot be replaced by prose alone.

## VERDICT
- Crucial updates: no
