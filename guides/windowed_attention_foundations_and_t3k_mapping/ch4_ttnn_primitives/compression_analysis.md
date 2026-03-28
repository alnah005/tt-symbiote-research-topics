# Compression Analysis: Chapter 4 — TTNN Primitive Operations and Tensor Shapes — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~1,358 lines
- Estimated post-compression line count: ~1,170 lines
- Estimated reduction: ~14%

## CRUCIAL Suggestions

### C1 — Remove "Reading Order" previews from `index.md` (lines 18–42, ~25 lines)

The three bullet points in the "Reading Order" section of `index.md` each summarize the sub-file in 4–6 lines. Every sub-file opens with its own scope paragraph that covers the same ground. The previews duplicate those openers word-for-word in several places (e.g., the index's description of `decode_primitives.md` mentions `pos_offset`, `ttnn.update_cache`, `compute_with_storage_grid_size`, `q_chunk_size`, `k_chunk_size`, window enforcement, and memory configs — all of which the file itself introduces in its own first paragraph and section headings). Replacing the three bullets with one-line links (`1. decode_primitives.md`, `2. prefill_primitives.md`, `3. kernel_or_op_gap_analysis.md`) loses no navigational value.

**Estimated saving: ~22 lines.**

---

### C2 — Collapse duplicate symbol-notation blocks in `decode_primitives.md` and `prefill_primitives.md`

`decode_primitives.md` lines 13–24 define `B`, `H_q`, `H_kv`, `w`, `d`, `T` as a prose bulleted list. `prefill_primitives.md` lines 14–23 re-define the identical six symbols (with `T` redefined as prompt length rather than generation step) in a table. The two blocks cover the same concept set. The prose version in `decode_primitives.md` also contains three lines (22–24) solely to enumerate GQA/MHA/MQA as special cases — information that is already common knowledge for the target audience and is re-stated again at line 120 of the same file.

**Recommended action:** Keep the table form in `prefill_primitives.md` as the canonical notation reference. In `decode_primitives.md` replace the prose block (lines 13–24) with a one-line cross-reference ("Symbol conventions follow the table in `prefill_primitives.md`; `T` here denotes absolute token position, not prompt length.") and delete the GQA/MHA/MQA enumeration at lines 22–24. The GQA rule (`H_q / H_kv` is an integer) is re-stated clearly at lines 143–144 and 165–167 where it is actually relevant.

**Estimated saving: ~14 lines.**

---

### C3 — Cut the O(T²) waste explanation from `prefill_primitives.md` Option A (lines 197–209)

`prefill_primitives.md` lines 197–209 explain that the kernel does not skip masked tiles and therefore performs O(T²) work. This exact finding appears again in `kernel_or_op_gap_analysis.md` as Gap G1 (line 367 in the consolidated table) and is expanded with the recommended fix in the "Closing G1" subsection (lines 379–406). The prefill file's version is a preview of the gap-analysis conclusion, not an addition to it. In `prefill_primitives.md` the two important caveats (shape restriction and O(T²) behavior) can be combined into a single bullet list of ≤4 lines, with a forward-reference to `kernel_or_op_gap_analysis.md` gap G1 for detail. The current 13-line block repeats G1 almost verbatim.

**Estimated saving: ~9 lines.**

---

### C4 — Remove redundant prose re-statement after the Strategy 2 code block in `decode_primitives.md` (lines 283–290)

Lines 283–290 of `decode_primitives.md` are a paragraph that says: in steady state the mask is all-zeros, pre-allocate as a zero tensor, reuse across steps, only changes during the fill phase. The code block immediately above (lines 264–279) already shows `position_mask[:, :, :, n_valid:] = float('-inf')` with an explicit `if n_valid < w` guard, and the inline comment says "In steady state: all w slots are valid → mask is all-zeros." The prose paragraph adds nothing beyond what the code and its comments already state.

**Estimated saving: ~8 lines.**

---

### C5 — Collapse the GQA "Inherited" subsections for Ops 3 and 4 in `kernel_or_op_gap_analysis.md`

`kernel_or_op_gap_analysis.md` has a "GQA Support" heading for each of the four ops. For Op 1 (`ttnn.scaled_dot_product_attention`, lines 77–78) and Op 2 (`ttnn.scaled_dot_product_attention_decode`, lines 165–167) these subsections contain one substantive sentence each. For Op 3 (`paged_sdpa`, lines 253–255) and Op 4 (`paged_sdpa_decode`, lines 333–334) the sections contain only "Inherited from [parent op]; supported." This is a heading + one line that restates what "Inherited" already implies. Move the inheritance note into the interface description or the gap-summary table row for each paged op and remove the standalone heading.

**Estimated saving: ~8 lines (4 headings + 4 single-sentence bodies).**

---

### C6 — Remove the closing prose summary paragraph from `kernel_or_op_gap_analysis.md` (lines 469–475)

The final paragraph ("All identified gaps can be closed without writing a new TTNN kernel...") re-states in prose the conclusion already encoded in the "Op Extension vs New Kernel vs New Program Config" table directly above it (lines 459–467). The table's last row and column make the same point cell by cell. The paragraph is a verbatim narrative of the table. Delete it; the table is self-contained.

**Estimated saving: ~7 lines.**

---

## MINOR Suggestions

### M1 — Trim the `triu(-2)` walk-through paragraph in `prefill_primitives.md` (lines 113–122)

Lines 113–122 are a paragraph that narrates the worked example from the table at lines 100–111 step by step ("For row 3 col 0 is zeroed (0 < 3-2=1); for row 4 cols 0–1 are zeroed..."). The table is visually self-explanatory for any reader familiar with matrix masking. The paragraph adds at most marginal clarity. Trimming it to two sentences ("The `tril()` call enforces causality; `triu(-(w-1))` enforces the lower window bound. Their intersection is the band-diagonal mask.") would save ~8 lines and eliminates step-by-step narration that restates the table.

---

### M2 — Condense the L1 working-set note in `decode_primitives.md` (lines 215–221)

The parenthetical "Note:" inside the code block at lines 215–221 runs four lines to clarify that the 260 KiB figure assumes one head at a time and that K/V tiles are shared across heads in the same group. This note hedges a figure that is already marked "illustrative." A single parenthetical sentence appended to the "Total" line achieves the same result: "(K/V tiles shared across heads in a group; Q and score buffer scale with heads_per_core.)"

---

### M3 — Merge the two `valid_seq_len` explanations in `kernel_or_op_gap_analysis.md` Op 2 (lines 142–146)

Lines 142–146 explain `valid_seq_len` as an alternative to `attn_mask` for the fill phase, then note that in steady state `valid_seq_len = w` and no masking is needed. Line 155 in the Window Expressibility subsection repeats "The op with `S_kv = w` and no masking is the correct steady-state decode implementation." These two sentences make the same point in adjacent subsections. One can be dropped.

---

### M4 — Remove the hedging opener in `decode_primitives.md` Step 3 (lines 82–83)

Lines 82–83 read "Two implementation paths are available: the explicit three-op sequence using `ttnn.matmul` and `ttnn.softmax`, and the preferred fused op `ttnn.scaled_dot_product_attention_decode`." This sentence is immediately followed by "#### Path A" and "#### Path B" headings that make the two paths structurally explicit. The sentence is redundant with its own section headings.

---

## Load-Bearing Evidence

- **`decode_primitives.md` lines 71–76** ("Write-before-read ordering: `ttnn.update_cache` must complete before...") — this is the only place in Chapter 4 that addresses TTNN's asynchronous command-queue ordering guarantee for the cache-write-before-attention constraint. Cutting it would remove the only explanation of why no explicit sync primitive is needed.

- **`decode_primitives.md` lines 235–257** (Strategy 1, window-limited slice) — the analysis that Strategy 1 is only correct during the fill phase and degenerates to a full pass in steady state is load-bearing: it is the specific technical argument that motivates preferring Strategy 2. This cannot be shortened without losing the reasoning.

- **`prefill_primitives.md` lines 270–317** (Prefill Side-Effect section with code block and shape diagram) — the `pos_offset` derivation, the `(T - w + i) % w` slot-assignment formula, and the T=12, w=8 worked example are the only concrete specification in Chapter 4 of the prefill-to-decode handoff. None of this material appears in any other file.

- **`kernel_or_op_gap_analysis.md` lines 363–374** (Consolidated Gap Table) — the seven-row severity table is the primary artifact of the gap analysis and the only place where all gaps are rated and cross-referenced together. It cannot be shortened.

- **`kernel_or_op_gap_analysis.md` lines 170–184** (Circular-Buffer Input, Op 2 — the RoPE-at-write-time explanation) — the reasoning that slot order is irrelevant to softmax when RoPE is baked into stored K values is load-bearing. It is the core technical justification for why the circular buffer layout is compatible with `ttnn.scaled_dot_product_attention_decode` without a reorder step.

## VERDICT
- Crucial updates: yes

## Change Log (Pass 1)

Applied 2026-03-27 by Agent A.

| ID | File | Change |
|----|------|--------|
| C1 | `index.md` | Replaced three verbose Reading Order bullet summaries (~22 lines) with three bare one-line clickable links. |
| C2 | `prefill_primitives.md` | Removed the six-symbol notation table and replaced with a one-line cross-reference to the canonical definition in `decode_primitives.md`. |
| C3 | `prefill_primitives.md` | Removed the 12-line O(T²) compute-waste explanation under Option A (point 2 + follow-on paragraph) and replaced with a one-sentence cross-reference to `kernel_or_op_gap_analysis.md`. |
| C4 | `decode_primitives.md` | Deleted the 8-line prose paragraph after the Strategy 2 code block that restated the code and its inline comments. |
| C5 | `kernel_or_op_gap_analysis.md` | Merged the GQA "Inherited" heading + tautological body sentence into a single compact heading for both Op 3 and Op 4. |
| C6 | `kernel_or_op_gap_analysis.md` | Deleted the 7-line closing prose paragraph that restated the "Op Extension vs New Kernel" table in narrative form. |

# Compression Analysis: Chapter 4 — TTNN Primitive Operations and Tensor Shapes — Pass 2

## Summary
- Total files analyzed: 4
- Estimated current line count: ~1,297 lines
- Estimated post-compression line count: ~1,275 lines
- Estimated reduction: ~2%

## CRUCIAL Suggestions
None — all resolved.

**C1 — RESOLVED.** `index.md` Reading Order section (lines 19–22) now contains three bare one-line links with no descriptive prose.

**C2 — RESOLVED.** `prefill_primitives.md` notation section (lines 13–15) is now a single cross-reference sentence; the duplicate six-symbol table is gone.

**C3 — RESOLVED.** `prefill_primitives.md` Option A constraint 2 (line 191) is now one sentence forwarding to `kernel_or_op_gap_analysis.md`; the 12-line O(T²) restatement is gone.

**C4 — RESOLVED.** The 8-line prose paragraph that re-stated the Strategy 2 code and its inline comments has been removed from `decode_primitives.md`. A short 4-line paragraph (lines 300–303) remains after the ascii diagram; it makes a distinct point (the windowed constraint is implicit in steady state, not mask-encoded) that is not present in the code block and is not the target of C4.

**C5 — RESOLVED.** `kernel_or_op_gap_analysis.md` Op 3 and Op 4 GQA subsections are now single-line headings (`### GQA Support — Inherited from ...`) with no separate body sentence.

**C6 — RESOLVED.** `kernel_or_op_gap_analysis.md` ends at the "Next" link (line 467); the closing prose paragraph that restated the table is gone.

## MINOR Suggestions

### M1 (carry-forward) — Trim the `triu(-2)` walk-through paragraph in `prefill_primitives.md` (lines 106–112)

The paragraph beginning "`triu(-2)` keeps entry `[row, col]` when `col >= row - 2`..." narrates the worked mask example row-by-row. The ascii table at lines 94–104 is self-explanatory for the target audience. Collapsing to two sentences ("The `tril()` call enforces causality; `triu(-(w-1))` enforces the lower window bound. Their intersection is the band-diagonal mask from `prefill_access_patterns.md`.") saves ~7 lines.

### M2 (carry-forward) — Condense the L1 working-set note in `decode_primitives.md` (lines 215–221)

The "Note:" block inside the L1 working-set code block runs four lines to hedge the 260 KiB figure with shared-heads caveats. A single inline parenthetical appended to the "Total" line achieves the same result, saving ~3 lines.

### M3 (carry-forward) — Drop the duplicated `valid_seq_len` steady-state note in `kernel_or_op_gap_analysis.md` Op 2

Lines 142–146 explain that `valid_seq_len = w` in steady state; line 155 makes the identical point in the adjacent Window Expressibility subsection. One occurrence can be removed (~2 lines).

### M4 (new) — Remove the introductory sentence for Step 3 in `decode_primitives.md` (lines 82–83)

"Two implementation paths are available: the explicit three-op sequence using `ttnn.matmul` and `ttnn.softmax`, and the preferred fused op `ttnn.scaled_dot_product_attention_decode`." The two `####`-level headings that follow ("Path A" and "Path B") already signal this structure; the sentence is a verbal table-of-contents for a two-item section. Deleting it saves 2 lines with no information loss.

## Load-Bearing Evidence

- **`index.md` line 8:** `"TTNN operations, specifying the exact tensor shapes at every op boundary"` — confirms the chapter's scope statement; removing or paraphrasing this would lose the precise contract that motivates every tensor-shape table below it.
- **`decode_primitives.md` lines 71–76:** `"Write-before-read ordering: ttnn.update_cache must complete before the attention call..."` — the only explanation in Chapter 4 of why no explicit synchronisation primitive is required under TTNN's async command-queue model.
- **`prefill_primitives.md` lines 270–284:** `"slot = (T - n_keep + i) % w"` — the only concrete specification of the prefill-to-decode circular-buffer handoff formula; not repeated in any other file.
- **`kernel_or_op_gap_analysis.md` lines 360–368:** The consolidated gap table with the seven-row severity column — the primary artifact of the gap analysis; cannot be shortened without losing cross-reference integrity.

## VERDICT
- Crucial updates: no
