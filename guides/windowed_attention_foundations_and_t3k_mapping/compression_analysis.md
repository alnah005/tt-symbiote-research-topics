# Compression Analysis: Cross-Chapter — Pass 1

## Summary

- Total files analyzed: 24 (guide index + 7 chapter index files + 16 content files; meta-files b_review.md, plan.md, per-chapter compression_analysis.md excluded)
- Estimated current line count: ~5,745 lines (across all guide content files)
- Estimated post-compression line count: ~5,350 lines
- Estimated reduction: ~7%

Note: the cross-chapter redundancy budget is modest. The guide is well-structured with explicit forward/backward references between chapters; most repeated material is intentional pedagogical bridging. The genuine redundancy is concentrated in two areas: the gap table duplication between Ch4 and Ch7, and the op-survey prose duplication between the same pair of files.

---

## CRUCIAL Suggestions

### CRUCIAL-1: Gap Table Duplicated Verbatim Between Ch4 and Ch7

**Files:** `ch4_ttnn_primitives/kernel_or_op_gap_analysis.md` (lines 360–368) and `ch7_roofline_and_kernels/existing_kernel_survey.md` (lines 263–271)

Both files contain an identical seven-row gap table (G1–G7) listing the same gap ID, description, ops affected, and severity. The Ch4 version is the authoritative derivation; the Ch7 version is explicitly labelled a "high-level synthesis" but then re-derives the same table in full rather than referencing it.

The Ch7 `existing_kernel_survey.md` opens with: *"For the detailed interface specifications of each op, the per-op gap tables, and the full reasoning behind each gap classification, see `../ch4_ttnn_primitives/kernel_or_op_gap_analysis.md`."* That sentence already commits Ch7 to being a summary — but the seven-row gap table that follows is a verbatim copy, not a summary.

**Action:** In `existing_kernel_survey.md`, replace the full G1–G7 table (and its header row) with a one-sentence pointer to the Ch4 table plus a prose summary of the severity verdict (e.g., "No gap blocks a correct initial implementation; G1 and G4 are the only high-severity performance gaps"). Keep the "Recommended Implementation Path" section in Ch7 intact — it adds value by synthesising the caller-side conventions into a concrete decision tree, which is distinct from the per-gap analysis in Ch4.

**Estimated saving:** ~30 lines (the 7-row table + column headers + surrounding blank lines duplicated in Ch7).

---

### CRUCIAL-2: Full Per-Op Survey Prose Duplicated Between Ch4 and Ch7

**Files:** `ch4_ttnn_primitives/kernel_or_op_gap_analysis.md` (the four Op sections, ~300 lines of interface + per-dimension analysis) and `ch7_roofline_and_kernels/existing_kernel_survey.md` (the four Op sections, ~180 lines)

The Ch7 file re-surveys all four ops (`ttnn.sdpa`, `ttnn.sdpa_decode`, `paged_sdpa`, `paged_sdpa_decode`) across the same five dimensions (mask support, window expressibility, GQA support, circular-buffer compatibility, shape restrictions). The prose differs in granularity — Ch7 omits the Python interface blocks and per-dimension gap summaries — but covers the same conceptual ground in the same order with the same conclusions.

Specific near-verbatim overlaps:

- Ch7 `existing_kernel_survey.md` lines 40–48 vs Ch4 `kernel_or_op_gap_analysis.md` lines 52–60: both describe `is_causal=True` as mutually exclusive with `attn_mask` and not encoding the windowed lower bound.
- Ch7 lines 61–63 vs Ch4 lines 68–73: both state that for `w << T` the kernel computes O(T²) scores and wastes FLOPs on masked-out tiles.
- Ch7 lines 77–80 vs Ch4 lines 82–94: both explain the circular-buffer slot-order irrelevance when RoPE is applied at write time.
- Ch7 lines 100–115 vs Ch4 lines 136–155: both describe `valid_seq_len` as the fill-phase mechanism and explain steady-state as requiring no mask.
- Ch7 lines 124–129 vs Ch4 lines 170–184: both state that `ttnn.sdpa_decode` is correct regardless of physical slot order if RoPE is applied at write time, with near-identical phrasing.
- Ch7 lines 145–163 vs Ch4 lines 210–262: both describe `paged_sdpa` forwarding `attn_mask`, the two strategies for window enforcement, and the circular-buffer-as-pages incompatibility.
- Ch7 lines 184–220 vs Ch4 lines 286–352: both cover `paged_sdpa_decode` mask support, `valid_seq_len` forwarding, and the same incompatibility with circular buffers.

**Action:** Reduce the Ch7 per-op sections to a summary table (one row per op, one column per dimension, using the Yes/Partial/Via caller/No vocabulary already used in the existing Ch7 gap summary table at lines 233–239) plus a brief prose paragraph per op noting only the single most important caveat. Remove the dimension-by-dimension sub-sections. The Ch7 file already contains the correct summary table at lines 233–239; the per-op prose above it that leads to the same conclusions is what is redundant. Estimated saving: ~120 lines from Ch7.

---

## MINOR Suggestions

### MINOR-1: AI ≈ 1 FLOP/byte Decode Result Stated Four Times

**Files and locations:**
- `ch3_data_dependencies/decode_access_patterns.md` lines 265–269 — full derivation (authoritative)
- `ch6_t3k_sharding/index.md` line 29 — cited in prerequisites as "establishing that windowed decode is bandwidth-bound (AI ≈ 1 FLOP/byte)"
- `ch7_roofline_and_kernels/index.md` lines 22–23 — cited in prerequisites as "The arithmetic intensity derivation (AI ≈ 1 FLOP/byte for decode)"
- `ch7_roofline_and_kernels/roofline_analysis.md` — re-derives the same AI calculation (not read directly but referenced in the Ch7 index)

The Ch6 and Ch7 index mentions are appropriate one-line prerequisites citations; those are not redundant. The concern is whether `roofline_analysis.md` re-derives the AI formula from scratch. If it does, that derivation should be replaced with a single-sentence reference back to Ch3. No action needed in the index files.

**Action:** Verify `roofline_analysis.md` and, if it reproduces the AI derivation table, reduce it to a one-line reference plus the final result. (Could not confirm without reading that file, but the pattern is consistent with Ch7's stated role as synthesis.)

**Estimated saving:** ~10–20 lines if the derivation is indeed reproduced.

---

### MINOR-2: "Wormhole Roofline Crossover ~111 FLOPs/byte" Constant Cited in Three Places

**Files:**
- `ch1_math_foundations/full_vs_windowed_attention.md` line 147 — parenthetical "(see Chapter 7 roofline analysis)"
- `guide index.md` line 53 — Quick Reference table: "~111 FLOPs/byte on Wormhole"
- `ch3_data_dependencies/decode_access_patterns.md` line 272 — "Wormhole's roofline crossover is approximately 111 FLOPs/byte"

The guide-level index table entry is load-bearing as a quick reference. The Ch1 mention is a cross-reference pointer with no numeric value stated. The Ch3 mention is the only place the 111 number appears in a content file outside Ch7. This is a minor issue: the 111 value is stated as a result from Ch7, but is used in Ch3's conclusion section to motivate micro-architectural guidance. The Ch3 use is legitimate since it contextualises the AI = 1 FLOP/byte result. No change required; flagged for awareness.

---

### MINOR-3: Circular-Buffer-as-Pages Strategy Described Twice

**Files:**
- `ch5_paged_kv_cache/paged_sdpa_and_windowing.md` — defines "Strategy B: Circular-buffer-as-pages" (authoritative)
- `ch4_ttnn_primitives/kernel_or_op_gap_analysis.md` lines 443–449 (G5 closure section) — describes the same strategy as the recommended path for closing G5

The Ch4 mention of "circular-buffer-as-pages strategy (Chapter 5)" is framed as a forward reference. It gives one-paragraph detail on the mechanism (allocating `ceil(w / block_size)` pages, O(1) per-step update) which partially duplicates what Ch5 establishes in depth. The detail level in Ch4 is appropriate for a gap-closure recommendation; it does not re-derive the full strategy. This is an acceptable overlap given that Ch4's gap analysis must stand independently for readers following the "Understand TTNN op gaps" path who may skip Ch5.

**Action:** None strictly required. Optionally, the Ch4 G5 closure paragraph (lines 443–449) could be trimmed by one sentence (removing the O(1) per step quantification, leaving it to Ch5) to make the forward reference cleaner without losing correctness.

**Estimated saving:** 2–3 lines.

---

### MINOR-4: Chapter Index Files Repeat Chapter Scope Already in Content Files

**Files:** `ch4_ttnn_primitives/index.md` (lines 26–52), `ch5_paged_kv_cache/index.md` (lines 38–71), `ch6_t3k_sharding/index.md` (lines 52–97)

Each of these index files contains a "Chapter Scope" section that describes what each content file covers in 3–8 sentences. This prose is largely a paraphrase of the content files' own opening sections. The overlap is not verbatim — the index summaries are at a higher level — but they occupy 20–45 lines per chapter and a reader arriving at the index then immediately opening the content file reads the same scope statement twice.

This is a structural pattern rather than a single duplicated passage. The index files serve as navigation aids so some scope description is appropriate. The issue is that Ch4, Ch5, and Ch6 indexes are more verbose than Ch1–Ch3 and Ch7, whose indexes stay at 2–3 sentences per file. Trimming the Ch4/Ch5/Ch6 scope sections to match the leaner Ch1–Ch3 style would save ~60 lines without losing navigability.

**Action:** Reduce "Chapter Scope" / "Reading Order" description sections in `ch4_ttnn_primitives/index.md`, `ch5_paged_kv_cache/index.md`, and `ch6_t3k_sharding/index.md` to 2–3 sentences per file entry rather than full paragraphs. The full scope belongs in the content files.

**Estimated saving:** ~55 lines.

---

## Load-Bearing Evidence

- **Ch3 `decode_access_patterns.md` AI derivation** — the full table and formula deriving AI ≈ 1 FLOP/byte must remain in Ch3; it is the only place the derivation is shown. All downstream references in Ch6 and Ch7 are one-line citations pointing back to it. Cutting or moving this derivation would break the prerequisite chain for Chapters 6 and 7.

- **Ch4 `kernel_or_op_gap_analysis.md` per-op interface blocks** — the Python interface signatures with named arguments (`attn_mask`, `valid_seq_len`, `program_config`, etc.) are load-bearing for engineers implementing windowed attention. They must remain in Ch4, which is the authoritative TTNN primitive reference. Removing them from Ch4 in favour of Ch7 would invert the intended dependency direction.

- **Ch7 `existing_kernel_survey.md` "Recommended Implementation Path" section** — this section (the caller-side conventions for prefill, non-paged decode, and paged decode) synthesises findings from Ch3, Ch4, and Ch5 into a concrete decision checklist. Even though individual elements appear elsewhere, the assembled decision path appears only in Ch7 and is the primary output of the guide for readers on the "Integrate windowed attention" path. It must not be cut.

- **Ch1 complexity tables (prefill and decode)** — the two complexity tables in `full_vs_windowed_attention.md` (O(T·w) vs O(T²) comparison) are the definitional source for every "saving factor" claim in subsequent chapters. They must remain exactly where they are; Ch2, Ch3, and Ch7 all cite these tables by chapter reference.

- **Ch5 paged strategy analysis** — the two-strategy analysis (page-aware windowing vs circular-buffer-as-pages) and the correctness invariants for page eviction are not reproduced anywhere else. The Ch4 and Ch7 forward references depend on this analysis existing in Ch5. Cutting any part of it would leave those references unsupported.

---

## VERDICT

- Crucial updates: yes

The two crucial issues (CRUCIAL-1 and CRUCIAL-2) together account for the dominant cross-chapter redundancy: Ch7 `existing_kernel_survey.md` re-presents the Ch4 gap table verbatim and re-surveys all four ops in prose that Ch4 already covers more thoroughly. The Ch7 file's own opening sentence acknowledges Ch4 as the authoritative source but then duplicates rather than summarises it. Resolving these two issues accounts for ~150 of the ~395 estimated saveable lines; the minor issues add another ~80–95 lines. Total estimated reduction across the full guide: ~230 lines (~7% of the 5,745-line guide content baseline), concentrated almost entirely in `ch7_roofline_and_kernels/existing_kernel_survey.md` and the three verbose chapter index files.

---

## Change Log (Pass 1)

Applied 2026-03-28 by Agent A.

| ID | File | Change |
|----|------|--------|
| CRUCIAL-1 | `ch7_roofline_and_kernels/existing_kernel_survey.md` | Replaced verbatim G1–G7 gap table with pointer sentence + severity verdict |
| CRUCIAL-2 | `ch7_roofline_and_kernels/existing_kernel_survey.md` | Collapsed four per-op survey sections to one paragraph each; removed dimension subheadings |

---

# Compression Analysis: Cross-Chapter — Pass 2

## Summary

- File examined: `ch7_roofline_and_kernels/existing_kernel_survey.md` (225 lines)
- CRUCIAL-1 resolved: yes
- CRUCIAL-2 resolved: yes
- New cross-chapter CRUCIAL issues found: none
- MINOR suggestions carried forward or newly identified: 3

## CRUCIAL-1 Verification

**Verdict: resolved.**

Pass 1 required the verbatim G1–G7 seven-row table at lines 263–271 to be replaced with a pointer sentence plus a prose severity verdict. The updated file contains no gap table in the former location. The entire "Identified Gaps" subsection (lines 90–92) now reads:

> "The full G1–G7 gap inventory with per-op attribution and interface details is in [`kernel_or_op_gap_analysis.md`](../ch4_ttnn_primitives/kernel_or_op_gap_analysis.md). No gap blocks a correct initial implementation; G1 (O(T²) wasted prefill FLOPs) and G4 (chunked prefill shape restriction) are the only high-severity performance gaps."

This is exactly what Pass 1 prescribed: one pointer sentence pointing to Ch4 as the authoritative source, followed by a prose severity verdict identifying the two high-severity gaps by name. The verbatim duplication has been eliminated.

## CRUCIAL-2 Verification

**Verdict: resolved.**

Pass 1 required the four per-op sections to be collapsed from multi-subsection dimension-by-dimension prose to one paragraph each, with dimension subheadings removed. The updated file's four op sections (lines 37–51) each consist of a single paragraph with no sub-headings. Spot-checking against the specific near-verbatim overlaps identified in Pass 1:

- The former lines 40–48 (mutual exclusivity of `is_causal=True` and `attn_mask`) have been compressed into a single clause: "The key caveat for windowed attention is that `is_causal=True` does not encode the window lower bound; a full band-diagonal `attn_mask` of shape `[1, 1, T, T]` with 0 in the band `[t - w + 1, t]` and `-inf` outside must be passed explicitly." One sentence replacing approximately eight.
- The former lines 61–63 (O(T² − T·w) wasted FLOPs) are compressed into a clause at the end of the Op 1 paragraph: "the kernel applies the mask correctly but does not skip compute on fully-masked tiles, so for `w << T` approximately O(T² − T·w) FLOPs are wasted — this is the G1 performance gap."
- The former lines 77–80 (RoPE-at-write-time, circular buffer slot order irrelevance) no longer appear in Op 2's paragraph; the point is handled by reference to the Recommended Implementation Path section.
- The former lines 100–115 (`valid_seq_len` fill-phase mechanism) are compressed into one clause in Op 2: "`valid_seq_len` parameter handles fill-phase masking correctly by implicitly masking positions beyond the specified valid count."
- The former long prose sections for Op 3 and Op 4 are each a single paragraph, with page-table detail reduced to the essential pointer toward Chapter 5 Strategy B.

The dimension-by-dimension sub-sections (Mask support, Window expressibility, GQA support, Circular-buffer compatibility, Shape restrictions) no longer appear in the per-op prose. All five dimensions are now addressed exclusively in the summary table at lines 62–68, which is the correct authoritative location in Ch7.

## New Cross-Chapter CRUCIAL Issues

**None found.**

A full cross-chapter check was performed against the remaining content of the updated `existing_kernel_survey.md`:

1. **"Recommended Implementation Path" section (lines 94–131):** The three caller-side convention checklists (prefill, non-paged decode, paged decode) are unique to Ch7. Ch4's "Recommended Paths Forward" section describes how to close individual gaps but does not assemble a phase-by-phase implementation checklist. The Ch7 section adds synthesis value not present elsewhere in the guide.

2. **Per-gap approach table (lines 204–212):** Ch4 contains a structurally similar table at lines 454–462 (`kernel_or_op_gap_analysis.md`). However, the two tables have different column schemas — Ch4's table has `Gap | Approach | Justification` while Ch7's has `Gap | Recommended Approach | Scope | Kernel Change?` — and the cell content differs in granularity. Ch7's table is a caller-oriented decision aid; Ch4's is a gap-closure rationale table. These are complementary, not duplicative. No CRUCIAL action required.

3. **Chunked prefill Python code block (lines 150–185):** This is present only in Ch7. Ch4's G1 closure section (lines 376–401) contains a truncated loop stub (lines 390–397) but explicitly omits the mask-construction logic with a comment `...`. Ch7's block is the only place the complete mask construction (including the `is_causal=True` warning and the correct `causal_ok & window_ok` formulation) appears. It is not a duplication; it is the authoritative implementation reference.

4. **"Overall Conclusion" section (lines 214–220):** States the same seven-gap summary verdict as the Ch4 gap table caption. This is a two-sentence synthesis paragraph, not a structural duplication. Acceptable.

## MINOR Suggestions

### MINOR-A: Per-Gap Approach Table Partially Overlaps Ch4 "Op Extension vs New Kernel" Table

**Files:** `ch7_roofline_and_kernels/existing_kernel_survey.md` lines 204–212 and `ch4_ttnn_primitives/kernel_or_op_gap_analysis.md` lines 454–462.

Both tables enumerate G1–G7 with a recommended approach. A reader who reads both chapters will see two tables covering the same seven gaps. As noted above, the column schemas differ and the tables serve different purposes (decision aid vs closure rationale), so this is MINOR rather than CRUCIAL. To make the relationship explicit and reduce potential reader confusion, Ch7's table could gain a one-line caption noting: "For the gap-closure rationale behind each approach, see the 'Op Extension vs New Kernel' table in Ch4." This adds one line but prevents the reader from treating the two tables as independent authorities.

**Estimated saving:** 0 lines (addition, not removal), but reduces cognitive redundancy.

### MINOR-B: "No gap blocks a correct initial implementation" Stated Twice in the Same File

**File:** `ch7_roofline_and_kernels/existing_kernel_survey.md`

The phrase (or its close equivalent) appears at line 92 ("No gap blocks a correct initial implementation") and again at line 98 ("The gap severity column shows that no gap blocks a correct initial implementation"). Both are in the same file, seven lines apart. The second occurrence at line 98 is the more appropriate location (opening the implementation path section); the first at line 92 is the trailing sentence of the gap summary subsection. Consider removing the sentence from line 92, leaving the severity verdict there as "G1 (O(T²) wasted prefill FLOPs) and G4 (chunked prefill shape restriction) are the only high-severity performance gaps" and letting the "no gap blocks" statement live solely in the Recommended Implementation Path section.

**Estimated saving:** 1 line; clarity improvement.

### MINOR-C: Pass 1 MINOR-1 (AI Derivation in `roofline_analysis.md`) Remains Unverified

Pass 1 flagged a conditional action: if `roofline_analysis.md` re-derives AI ≈ 1 FLOP/byte from scratch, that derivation should be replaced with a reference back to Ch3. That file was not read in Pass 1 and has not been changed. The concern remains open.

**Action:** In a subsequent pass, read `ch7_roofline_and_kernels/roofline_analysis.md` and verify whether the AI derivation table is reproduced. If so, replace with a single-sentence reference to `ch3_data_dependencies/decode_access_patterns.md` lines 265–269.

**Estimated saving:** 10–20 lines if the derivation is indeed reproduced.

## Load-Bearing Evidence

- `ch7_roofline_and_kernels/existing_kernel_survey.md` line 92: `"The full G1–G7 gap inventory with per-op attribution and interface details is in [kernel_or_op_gap_analysis.md]"` — confirms CRUCIAL-1 resolution; the table is gone and a pointer stands in its place.
- `ch7_roofline_and_kernels/existing_kernel_survey.md` line 38: `"The key caveat for windowed attention is that is_causal=True does not encode the window lower bound; a full band-diagonal attn_mask of shape [1, 1, T, T] ... must be passed explicitly."` — confirms CRUCIAL-2 resolution for Op 1; a single-sentence compression of the former multi-sentence sub-section.
- `ch7_roofline_and_kernels/existing_kernel_survey.md` line 46: `"Strategy B (circular-buffer-as-pages from [paged_sdpa_and_windowing.md]) is recommended for circular-buffer paging because neither window enforcement mechanism is built into the op itself."` — confirms CRUCIAL-2 resolution for Op 3; the former ~18 lines of two-strategy analysis reduced to a single clause with a cross-reference.
- `ch7_roofline_and_kernels/existing_kernel_survey.md` lines 62–68 (Gap Summary Table): the five-dimension, four-op capability matrix remains intact and unmodified — confirming that the authoritative Ch7 summary data was preserved while redundant prose was removed.
- `ch4_ttnn_primitives/kernel_or_op_gap_analysis.md` lines 360–368 (Consolidated Gap Table): the G1–G7 table with Ops Affected, Severity, and Description columns remains in Ch4 — confirming that the authoritative derivation was not affected by the Ch7 edit.

## VERDICT

- Crucial updates: no

Both CRUCIAL issues from Pass 1 have been fully resolved in `ch7_roofline_and_kernels/existing_kernel_survey.md`. No new cross-chapter CRUCIAL duplications were introduced by the Pass 1 edits. Three MINOR suggestions are noted above (MINOR-A, MINOR-B, MINOR-C), of which MINOR-C is a carry-forward of the unverified Pass 1 MINOR-1 item. The guide is in a sound state with respect to cross-chapter verbatim redundancy. The remaining MINOR items are low-priority clarity improvements, not correctness or redundancy blockers.
