# Compression Analysis: Chapter 3 GQA Tensor Layout

## Summary

- **Files analyzed**: 4 (`index.md`, `head_axis_conventions.md`, `gqa_grouping_in_kernel.md`, `paged_layout_for_gqa.md`)
- **Estimated current line count**: 375 lines (index.md: 64, head_axis_conventions.md: 97, gqa_grouping_in_kernel.md: 111, paged_layout_for_gqa.md: 103)
- **Estimated post-compression line count**: ~310 lines
- **Estimated reduction**: ~17%

---

## CRUCIAL Suggestions

### 1. GQA Head-Grouping Diagram Duplicated Between `index.md` and `gqa_grouping_in_kernel.md`

**Files**: `index.md` (lines 30–47) and `gqa_grouping_in_kernel.md` (lines 19–30)

**What duplicates what**: `index.md` contains a full ASCII grouping diagram showing 4 KV heads each serving 4 Q heads, along with the formula `kv_head_idx = q_head_idx // group_size` (line 47), and the concrete values `nkv=4`, `nh=16`, `group_size=4`. `gqa_grouping_in_kernel.md` restates the same example (`nh=16`, `nkv=4`, `group_size=4`) with a detailed lookup table of q_head_idx → kv_head_idx mappings (lines 21–30), which covers identical numerical content.

**Which file should keep it**: `gqa_grouping_in_kernel.md` should keep the full derivation table — that is its dedicated topic. `index.md` should reduce its diagram to a one-sentence conceptual pointer plus a single inline formula reference, removing the full ASCII art block and the specific numeric values (which are restated in depth in the child file).

**Estimated savings**: ~12 lines from `index.md`.

---

### 2. The 32-Padding Silent Bug Explained in Both `head_axis_conventions.md` and `gqa_grouping_in_kernel.md`

**Files**: `head_axis_conventions.md` (lines 46–53) and `gqa_grouping_in_kernel.md` (lines 51–96)

**What duplicates what**: `head_axis_conventions.md` introduces the padding behavior of `nlp_create_qkv_heads_decode` — that `nh` is padded to `pnh` (multiple of 32) and `nkv` is padded to `pnh / original_group_size` — and explicitly flags the silent failure if `nkv` were padded independently. It then defers the reader to `gqa_grouping_in_kernel.md` for "the full analysis." `gqa_grouping_in_kernel.md` contains that full analysis: the failure scenario walkthrough, the correct padding formula with code, and the three-row comparison table. The substantive content in `head_axis_conventions.md` lines 46–53 — specifically the two bullet-point rules and the warning about changed group ratios — is a near-verbatim subset of the explanation in `gqa_grouping_in_kernel.md`.

**Which file should keep it**: `gqa_grouping_in_kernel.md` should keep the full treatment. `head_axis_conventions.md` should be reduced to a single sentence noting that `nlp_create_qkv_heads_decode` applies hardware-tile padding and forwarding readers to `gqa_grouping_in_kernel.md`, removing the two bullet points and the cross-reference warning that pre-explain the child file's content.

**Estimated savings**: ~7 lines from `head_axis_conventions.md`.

---

### 3. Summary Code Block in `head_axis_conventions.md` Restates the Shape Tables Above It

**File**: `head_axis_conventions.md` (lines 76–91, the `## Summary` code block)

**What duplicates what**: The code block (lines 77–90) re-lists:
- `q` shape `[1 x b x nh x dh]` — already in the table at lines 13–21
- `k`, `v` shape `[b x nkv x s x dh]` — already in the table at lines 24–32
- The `nlp_create_qkv_heads_decode` padding rule — already explained at lines 48–52

The code block adds no new information; it is a pure restatement of the two preceding sections. This is a table-restates-text pattern.

**Which file should keep it**: The shape tables (lines 9–32) are the canonical form and should stay. The Summary code block (lines 75–91) should be removed, or collapsed to at most two lines referencing the tables above.

**Estimated savings**: ~16 lines from `head_axis_conventions.md`.

---

## MINOR Suggestions

### 1. "Next Steps" Sections Duplicate `index.md` Navigation

All three child files (`head_axis_conventions.md` line 94–96, `gqa_grouping_in_kernel.md` lines 108–110, `paged_layout_for_gqa.md` lines 100–102) end with a "Next Steps" section that mirrors the reading order already laid out in `index.md`'s Contents table (lines 51–58) and its own "Next Steps" line (lines 61–63). These sections are not wrong, but they restate the sequential order that `index.md` already establishes. They could be collapsed to a single italicized line (e.g., "_Next: `paged_layout_for_gqa.md`_") rather than a full paragraph, saving roughly 2–4 lines per file.

### 2. Warning Callouts Repeat the Same Core Claim in Adjacent Prose

Three warning callouts in two files address silent failures with no runtime error:
- `head_axis_conventions.md` lines 71–72: warns about silent wrong attention when `nkv == b` in the legacy layout
- `gqa_grouping_in_kernel.md` lines 70–71: warns about silent MHA collapse from independent `nkv` padding
- `paged_layout_for_gqa.md` lines 57–58: warns about silent wrong KV data from expanded-head-count writes

Each warning ends with a near-identical phrase: "No error is raised" / "no runtime error" / "silently, with no runtime error." The warnings themselves cover distinct failure modes and must stay, but the "no error is raised" tail clause is repeated verbatim in all three. A single chapter-level callout in `index.md` noting that all silent failures in this chapter produce no runtime error would allow the per-file warnings to drop their closing "no error" sentence, saving ~3 lines total without losing any content.

### 3. Opening Paragraphs of Child Files Restate `index.md` Learning Objectives

Each child file's `## Overview` paragraph restates information already encoded in `index.md`:
- `head_axis_conventions.md` lines 4–5 restates LO 1 and the silent-failure theme from index lines 5–6 and 13.
- `gqa_grouping_in_kernel.md` lines 4–5 restates LO 3 and the silent MHA collapse theme from index lines 16 and 47.
- `paged_layout_for_gqa.md` lines 4–5 restates LO 5 and the expanded-head-count topic from index lines 17 and 57.

These overviews are one-sentence summaries that add no new information for a reader arriving sequentially from `index.md`. They could be removed entirely, or kept only as a single-sentence "scope" line, saving ~2 lines per file (~6 total).

---

## Load-Bearing Evidence

The following specific facts must NOT be removed from any file:

1. **Q tensor shape is `[1 x b x nh x dh]`, not `[b x nh x dh]` or `[b x 1 x nh x dh]`** — `head_axis_conventions.md`, lines 13–21 (table). The leading `1` is axis 0, and its meaning (single decode step, not a placeholder) is explained in lines 37–41. Both the shape and the explanation are load-bearing.

2. **K/V tensor shape is `[b x nkv x s x dh]`, with batch first** — `head_axis_conventions.md`, lines 24–32 (table). The old layout `[nkv x b x S x dh]` and the swap introduced by issue #12330 are documented at lines 58–71. The specific condition `nkv == b` under which the swap is invisible to shape validation (line 69) is load-bearing.

3. **`kv_head_idx = q_head_idx // group_size` is the sole GQA mechanism in the kernel** — `gqa_grouping_in_kernel.md`, lines 11–17. The phrase "There is no other GQA-specific logic in the kernel" (line 17) is a critical architectural fact.

4. **Correct padding formula: `nkv_padded = pnh // original_group_size`** — `gqa_grouping_in_kernel.md`, lines 74–87 (code block). The three-row comparison table at lines 90–95 (showing nh=16/nkv=4 padded correctly vs. incorrectly) is the clearest illustration of the silent MHA collapse and must stay.

5. **Paged KV cache shape is `[max_num_blocks x nkv x block_size x dh]`** — `paged_layout_for_gqa.md`, lines 11–23 (table + prose). The explanation that `max_num_blocks` is the paged dimension and that a block table maps logical positions to physical block indices (lines 20–22) establishes the meaning of "paged" and is load-bearing.

6. **`paged_update_cache` input shape must be `[b x nkv x 1 x dh]`, not `[b x nh x 1 x dh]`** — `paged_layout_for_gqa.md`, lines 29–57. The three-item numbered list at lines 52–55 explaining why writing `nh` heads is wrong (memory bloat, misaligned block table, silent wrong attention) must be preserved in full.

7. **The `nh % nkv == 0` precondition is required but not checked by the kernel** — `gqa_grouping_in_kernel.md`, lines 34–47. The code snippet at lines 41–47 illustrating the silent failure when `nh=17, nkv=4` (producing a reference to non-existent `kv_head_idx=4`) is load-bearing.

---

## VERDICT: Crucial updates: yes

## Agent A Change Log — C Feedback Pass 1
- index.md: Replaced head-grouping diagram and formula with pointer to gqa_grouping_in_kernel.md
- head_axis_conventions.md: Replaced 32-padding collapse explanation with one forwarding sentence to gqa_grouping_in_kernel.md
- head_axis_conventions.md: Removed Summary code block that restated the shape tables above it

---

# Compression Analysis: Chapter 3 — GQA Tensor Layout Requirements — Pass 2

## Summary

- **Files analyzed**: 4 (`index.md`, `head_axis_conventions.md`, `gqa_grouping_in_kernel.md`, `paged_layout_for_gqa.md`)
- **Line counts after Pass 1 fixes**: `index.md`: 49, `head_axis_conventions.md`: 73, `gqa_grouping_in_kernel.md`: 111, `paged_layout_for_gqa.md`: 103 — **total: 336 lines**
- **Pass 1 reduction**: 375 → 336 lines (~39 lines removed, ~10%)
- **Estimated post-Pass-2 compression**: ~330 lines (minor; no crucial removals identified)

---

## Pass 1 Fix Verification

### Fix 1 — `index.md`: Head-grouping diagram and kv_head_idx formula replaced

**Status: CONFIRMED.**

`index.md` lines 30–34 now read:

> "The head-grouping formula and padding collision table are covered in full in `gqa_grouping_in_kernel.md`."

The full ASCII grouping diagram, the concrete numeric values (`nh=16`, `nkv=4`, `group_size=4`), and the inline formula `kv_head_idx = q_head_idx // group_size` are all gone from `index.md`. The section heading "Conceptual Diagram: Head Grouping" is preserved as a navigation anchor, now containing only the pointer sentence. Approximately 15 lines removed.

### Fix 2 — `head_axis_conventions.md`: 32-padding collapse explanation replaced with single forwarding sentence

**Status: CONFIRMED.**

`head_axis_conventions.md` lines 45–48 now read:

> "When TTNN pads nh and nkv independently, the effective group_size can silently change; see `gqa_grouping_in_kernel.md` for the full analysis."

The original two bullet-point rules (the `pnh = multiple of 32` rule and the `nkv_padded = pnh / original_group_size` rule) and the cross-reference warning that pre-explained the child file's content have been removed. The single sentence preserves the concept while eliminating ~7 lines of near-verbatim content that duplicated `gqa_grouping_in_kernel.md`.

### Fix 3 — `head_axis_conventions.md`: Summary code block removed

**Status: CONFIRMED.**

The `## Summary` section and its code block are entirely absent from `head_axis_conventions.md`. The file now ends at line 72 with the "Next Steps" section. The shape tables at lines 13–21 (Q tensor) and lines 24–32 (K/V tensors) are intact and remain the canonical form. Approximately 17 lines removed.

---

## CRUCIAL Suggestions

None. All three Pass 1 crucial duplications have been resolved. A scan of the current file states found no remaining cross-file duplication that rises to the crucial threshold:

- `gqa_grouping_in_kernel.md` lines 99–104 contain a 4-bullet `## Summary` section that distills the body. This is a condensed recap, not verbatim duplication — the bullets do not re-present any code blocks or tables from the body, only short restatements of conclusions. This does not meet the crucial threshold.
- `paged_layout_for_gqa.md` lines 90–96 contain a 3-row `## Summary` table that restates the correct cache shapes from lines 11 and 30. The table format is new (not present earlier in the file) and the "Common Mistake" column consolidates information that otherwise requires reading two separate sections. This does not meet the crucial threshold.

---

## MINOR Suggestions

The following are carried forward from Pass 1 (unresolved) and one new observation:

### 1. "Next Steps" Sections Duplicate `index.md` Navigation (carried forward)

`head_axis_conventions.md` lines 70–72, `gqa_grouping_in_kernel.md` lines 108–110, and `paged_layout_for_gqa.md` lines 100–102 each contain a "Next Steps" paragraph restating the reading order already encoded in `index.md`'s Contents table and its own Next Steps line. These could each be reduced to a single italicized line (e.g., `_Next: paged_layout_for_gqa.md_`), saving ~2 lines per file (~6 total).

### 2. "No Error Is Raised" Tail Clause Repeated in All Three Warning Callouts (carried forward)

`head_axis_conventions.md` line 66 ("There is no runtime error"), `gqa_grouping_in_kernel.md` line 70 ("No error is raised"), and `paged_layout_for_gqa.md` line 57 ("silently, with no runtime error") each end their warning callout with a near-identical phrase. The three warnings cover distinct failure modes and must stay; only the closing "no error" clause is redundant across files. A single chapter-level note in `index.md` line 5 (which already says "produce wrong outputs silently, with no error raised") could absorb this repetition, allowing the per-file closing clauses to be trimmed. Estimated saving: ~3 lines total.

### 3. Opening `## Overview` Paragraphs Restate `index.md` Learning Objectives (carried forward)

`head_axis_conventions.md` lines 3–5, `gqa_grouping_in_kernel.md` lines 3–5, and `paged_layout_for_gqa.md` lines 3–5 each open with a one-sentence overview that restates the corresponding learning objective from `index.md` lines 13–17. For a reader arriving sequentially from `index.md`, these add no new information. Each could be removed or collapsed to a single "scope" line, saving ~2 lines per file (~6 total).

### 4. `gqa_grouping_in_kernel.md` Summary Bullets Restate Section Conclusions (new)

`gqa_grouping_in_kernel.md` lines 99–104 contain a `## Summary` section with 4 bullets that restate the conclusions of the three preceding sections (kernel formula, `nh % nkv` requirement, padding failure, correct padding strategy). The bullets are concise and add navigation value, but they are strictly restatements. This is the same pattern as the removed code block in `head_axis_conventions.md`, though less egregious (bullets vs. full code re-listing). Estimated saving if removed: ~8 lines. Recommend keeping unless line budget is critical, as the summaries are brief.

---

## Load-Bearing Evidence

The following specific facts were confirmed present and intact across all four files after Pass 1 changes. None were disturbed by the edits.

1. **Q tensor shape `[1 x b x nh x dh]` with the leading `1` explained as the decode contract** — `head_axis_conventions.md` lines 13–21 (shape table) and lines 37–41 (prose explanation). Both the shape and the rationale for the `1` are present and unmodified.

2. **K/V tensor shape `[b x nkv x s x dh]` with the axis-swap silent failure condition `nkv == b`** — `head_axis_conventions.md` lines 24–32 (shape table), lines 51–67 (layout change section). The specific condition under which the old and new layouts are numerically indistinguishable is present at line 64.

3. **`kv_head_idx = q_head_idx // group_size` is the sole GQA mechanism** — `gqa_grouping_in_kernel.md` lines 11–17. The phrase "There is no other GQA-specific logic in the kernel" is present at line 17. The lookup table at lines 23–28 illustrating the `nh=16, nkv=4` mapping is intact.

4. **`nh % nkv == 0` precondition is required but not checked** — `gqa_grouping_in_kernel.md` lines 34–47. The code snippet at lines 40–47 showing the silent failure when `nh=17, nkv=4` (Q head 16 maps to non-existent KV head 4) is present and unmodified.

5. **Correct padding formula `nkv_padded = pnh // original_group_size`** — `gqa_grouping_in_kernel.md` lines 72–87 (code block) and lines 89–95 (three-row comparison table). Both the code and the table showing correct vs. incorrect padding for `nh=16, nkv=4` are present and unmodified.

6. **Paged KV cache shape `[max_num_blocks x nkv x block_size x dh]`** — `paged_layout_for_gqa.md` lines 11–22. The explanation that `max_num_blocks` is the paged dimension and that a block table maps logical positions to physical block indices is present at lines 20–22.

7. **`paged_update_cache` input must be `[b x nkv x 1 x dh]`, not `[b x nh x 1 x dh]`** — `paged_layout_for_gqa.md` lines 30–70. The three-item numbered list at lines 52–55 explaining memory bloat, misaligned block table, and silent wrong attention is present and unmodified.

---

## VERDICT: Crucial updates: no
