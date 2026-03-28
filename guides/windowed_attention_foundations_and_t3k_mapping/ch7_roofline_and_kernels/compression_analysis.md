# Compression Analysis: Chapter 7 — Roofline Analysis and Existing Kernel Survey — Pass 1

## Summary
- Total files analyzed: 3
- Estimated current line count: ~840 lines
- Estimated post-compression line count: ~690 lines
- Estimated reduction: ~18%

## CRUCIAL Suggestions

### C1 — `roofline_analysis.md` lines 51–53: Remove hedge sentence about "smaller contributions retained for completeness"
The sentence "For the AI derivation the analysis focuses on the dominant terms; smaller contributions such as the output write and query read are retained for completeness but do not change the conclusion" (lines 51–53) pre-explains what the derivation is about to show. The reader will see that Q and output bytes are annotated `(negligible for w >> 1)` directly in the math blocks that follow. Delete this sentence; the inline annotations carry the same information.

### C2 — `roofline_analysis.md` lines 139–151: Collapse the "Key Insight" subsection
The "Key Insight: AI Does Not Depend on w or T" section (lines 137–151) uses 13 lines to state three bullet points plus a closing sentence. The first two bullets say the same thing twice (increasing w leaves AI unchanged; decreasing w leaves AI unchanged). Merge them into a single sentence. The closing sentence ("windowed attention does not change the bandwidth-bound characterisation...") is the load-bearing conclusion and should be kept, but the whole section can be trimmed to ~5 lines without any information loss.

### C3 — `roofline_analysis.md` lines 233–247: Remove the "Performance Scaling" subsection
The subsection "Performance Scaling with w vs T" (lines 233–247) restates in prose what the table immediately above it already shows numerically. The two bullet points ("Full attention: DRAM bandwidth grows linearly with T" and "Windowed attention: DRAM bandwidth is constant once T ≥ w") are directly readable from the table rows. This subsection adds no new information; delete it and rely on the table's Notes section, which already states the same conclusion at line 225–231.

### C4 — `index.md` lines 54–74: Remove "Chapter Scope" section entirely
The "Chapter Scope" section (lines 52–74) repeats content already present in the "Reading Order" section (lines 37–50) of the same file. "Reading Order" already summarises what each sub-file covers — roofline position, bandwidth-bound regime, throughput advantage table for `roofline_analysis.md`, and the four-op survey, gap summary table, and implementation path conclusion for `existing_kernel_survey.md`. "Chapter Scope" then restates every point in slightly different words (e.g., "32 TFLOPS peak FP16 and 288 GB/s peak DRAM bandwidth" and "crossover arithmetic intensity is approximately 111 FLOPs/byte" all appear again). Delete "Chapter Scope" (lines 52–74) in its entirety; the "Reading Order" section is sufficient.

### C5 — `existing_kernel_survey.md` lines 394–414: Shorten "Overall Conclusion" by removing the decode bandwidth restatement
The final two sentences of the "Overall Conclusion" section (lines 405–410, "For decode, no additional optimisation is available at the kernel level — the operation is bandwidth-bound at all practical batch sizes, and the bandwidth cost is already minimised by the fixed-size circular buffer") reproduce the conclusion from `roofline_analysis.md`'s "Implications for Implementation" section (lines 307–346). The conclusion in `existing_kernel_survey.md` should state only what this file has established (kernel gaps and paths); the bandwidth-bound decode characterisation is the roofline file's conclusion. Remove those two sentences.

## MINOR Suggestions

### M1 — `roofline_analysis.md` lines 75–77: Delete the GQA-FLOPs clarification paragraph
Lines 79–82 ("When GQA is active (H_q > H_kv), the same K and V head serves H_q / H_kv query heads. The FLOPs above use H_q throughout because each of the H_q query heads still performs a full inner product...") state an obvious consequence of the FLOP formula. The formulas already show `H_q` explicitly; the clarification is not needed. Two lines can be cut.

### M2 — `existing_kernel_survey.md` lines 224–231: Trim legend block
The legend block beneath the gap summary table (lines 241–259) uses a fenced code block to define `sdpa` and `sdpa_decode` abbreviations and re-explain "Partial" and "Via caller" semantics that are already defined in the cell-marking legend above the table (lines 226–231). The only non-redundant content is the clarification that `inh.` means "inherited from underlying TTNN op". Collapse the legend block to a single note line beneath the table rather than a full code block.

### M3 — `roofline_analysis.md` lines 161–168: Cut the compute-utilisation commentary
The sentence "The compute units are idle more than 99% of the time" (line 168) and its context ("For MHA (G=1): 1 × 288 GB/s = 288 GFLOP/s — versus a peak of 32 TFLOPS") restate in words what the roofline diagram (lines 170–190) already makes visually clear. Delete the prose sentence at line 168 and retain the diagram.

### M4 — `existing_kernel_survey.md` lines 384–393: Merge G3/G4 rows note into gap table
The "Op Extension vs New Program Config vs New Kernel" table (lines 382–393) notes for G4: "Same as G1 (external loop IS chunked mode)". This is the only new information in that row relative to what was already established in the G1 discussion. Rather than maintaining a separate seven-row table, annotate G4 in the Gaps table with "(same path as G1)" and remove the separate table, saving ~12 lines.

## Load-Bearing Evidence

- `index.md` lines 8–17: The Q7/Q8 framing and the single-sentence conclusion ("Together these questions determine the implementation path: whether a new kernel is required...") establish the chapter's purpose and must not be cut. All other prose in `index.md` is duplicative of sub-file content.
- `roofline_analysis.md` lines 113–135: The AI formula derivation (`AI = H_q / H_kv`) and the MHA/GQA sub-cases are the quantitative core of the chapter. Every line is load-bearing; none can be collapsed further without losing the mathematical argument.
- `roofline_analysis.md` lines 265–279: The `B_crossover = AI_crossover / G` derivation and the concrete values (B_crossover ≈ 28 for G=4, ≈ 111 for MHA) are unique quantitative results not repeated elsewhere in the guide.
- `roofline_analysis.md` lines 307–346: The "Implications for Implementation" section contains four numbered conclusions that are cross-referenced by later chapters and by `existing_kernel_survey.md`. The prefill AI formula clarification in point 4 (lines 329–341) is notably dense but load-bearing — it disambiguates two conflicting AI formulas (`T·w/(T+w)` vs `G·w`) and is not restated anywhere else.
- `existing_kernel_survey.md` lines 263–271: The seven-row gap table (G1–G7) is the single authoritative gap inventory for the guide. Every row is load-bearing.
- `existing_kernel_survey.md` lines 325–365: The Approach A chunked-loop Python code block and its inline comments (particularly the `is_causal=True` correctness warning at lines 342–350) are load-bearing. The warning about incorrect masking when `k_start < t0` is not stated elsewhere and removing it would create a latent correctness trap for implementers.

## VERDICT
- Crucial updates: yes

---

# Compression Analysis: Chapter 7 — Roofline Analysis and Existing Kernel Survey — Pass 3

## Summary
- Files analyzed: `index.md`, `roofline_analysis.md`, `existing_kernel_survey.md`
- Pass 1 identified 5 crucial and 4 minor issues; Pass 2 identified 2 additional crucial and 2 minor issues
- Pass 3 performs a full cross-file sweep for any remaining CRUCIAL redundancy not already catalogued
- Pass 3 finds no new CRUCIAL issues

## Scope and Method

Pass 3 checks every candidate pairing of formulas, facts, and tables across the three files for CRUCIAL-level duplication: two occurrences where one adds zero information beyond the other. Minor issues M5 and M6 from Pass 2 are explicitly below the CRUCIAL threshold and are not re-examined.

## Candidates Examined

**Hardware numbers table (`roofline_analysis.md` lines 12–18) vs `index.md`.**
`index.md` contains no hardware numbers; it only cross-references the roofline file by pointer. No duplication.

**AI crossover value (111 FLOPs/byte) across files.**
Derived and used solely within `roofline_analysis.md`. Neither `existing_kernel_survey.md` nor `index.md` states this number. The within-paragraph duplicate at `roofline_analysis.md` lines 314–315 was already flagged as M6 (minor, below threshold). No new CRUCIAL issue.

**Gap capability matrix (`existing_kernel_survey.md` lines 233–239) vs Identified Gaps table (lines 263–271).**
These are complementary tables, not duplicates. The capability matrix is a 4-op × 5-dimension Yes/Partial/Via-caller/No grid; the gap table is a 7-row inventory with description, affected ops, and severity. Neither is a subset of the other.

**"Op Extension vs New Program Config vs New Kernel" table (`existing_kernel_survey.md` lines 383–391) vs the Identified Gaps table.**
The third table adds "Recommended Approach", "Scope", and "Kernel Change?" columns not present in either earlier table. The G4 note ("Same as G1") was flagged as M4 in Pass 1 (minor). No new CRUCIAL issue.

**`roofline_analysis.md` "Implications" point 3 (lines 293–297) vs `index.md` lines 34–35.**
`index.md` is a cross-reference pointer only; it does not reproduce the bandwidth-scaling rationale. No duplication.

**FLOP count formulas (`roofline_analysis.md` lines 57–69) across files.**
These formulas appear only in `roofline_analysis.md`. No cross-file duplication.

**B_crossover formula (`roofline_analysis.md` lines 244–246) across files.**
Appears only in `roofline_analysis.md`. Not repeated elsewhere. No duplication.

**Throughput comparison table (`roofline_analysis.md` lines 204–211) across files.**
`index.md` references its existence in one clause; it does not reproduce data rows. No duplication.

**Approach A Python code block comments (`existing_kernel_survey.md` lines 342–360) vs surrounding prose.**
The prose motivates the code; the inline comments clarify the implementation. The information is complementary (code + explanation), not identical restatement. No CRUCIAL issue.

**"No new kernel required" conclusion — `existing_kernel_survey.md` lines 275–308 vs lines 395–403.**
This pairing was identified as C7 in Pass 2 and remains an open prior-pass finding. Pass 3 does not re-flag it as a new issue.

**87.5% wasted-FLOPs fact — `existing_kernel_survey.md` lines 60–64 vs lines 314–316.**
This pairing was identified as C6 in Pass 2 and remains an open prior-pass finding. Pass 3 does not re-flag it as a new issue.

## New CRUCIAL Issues Found

None.

## VERDICT
- Crucial updates: no

---

# Compression Analysis: Chapter 7 — Roofline Analysis and Existing Kernel Survey — Pass 2

## Summary
- Files analyzed: `index.md`, `roofline_analysis.md`, `existing_kernel_survey.md`
- Pass 1 identified 5 crucial and 4 minor issues; Pass 2 checks what remains after those are applied
- Pass 2 finds 2 new crucial issues (both within `existing_kernel_survey.md`) and 2 minor issues
- Estimated additional lines saved: ~10

## CRUCIAL Issues

### C6 — `existing_kernel_survey.md` lines 60–64 vs lines 314–316: Duplicated 87.5% wasted-FLOPs fact with identical example

**Occurrence A** (lines 60–64, Op 1 Window Expressibility section):
> "The kernel does not skip compute on fully-masked tiles. For `w << T` the score matrix is `T × T` but only `T × w` entries are in-band. The out-of-band scores are computed, then suppressed by the `-inf` mask before softmax. The wasted FLOPs are O(T² − T·w). For T = 32768 and w = 4096 this is approximately 87.5% of the FLOPs wasted on masked-out tiles."

**Occurrence B** (lines 314–316, Performance-Optimised Implementation section):
> "The only gap that materially limits performance at scale is **G1** (wasted O(T²) − O(T·w) FLOPs during prefill for large T and small w). For T = 32768 and w = 4096, the wasted fraction is ~87.5%."

Both occurrences state the identical fact: G1 gap causes ~87.5% wasted FLOPs at T=32768, w=4096. Occurrence A is the first and more fully explained instance (it also explains the mechanism — scores computed then suppressed). Occurrence B in the Performance-Optimised Implementation section only needs to invoke G1 by reference; the 87.5% figure and the T/w example are already established. The redundant portion is the quantification `For T = 32768 and w = 4096, the wasted fraction is ~87.5%` at lines 315–316 — dropping this phrase (but keeping `The only gap that materially limits performance at scale is **G1**...`) loses zero information.

**Redundant:** lines 315b–316 of occurrence B (the `For T = 32768 and w = 4096, the wasted fraction is ~87.5%.` clause).
**Lines saved:** ~1 (partial line deletion or sentence merge).

### C7 — `existing_kernel_survey.md` lines 396–403 vs lines 275–308: Overall Conclusion restates the "Correct Initial Implementation" section body

**Occurrence A** (lines 275–308, "Correct Initial Implementation — No New Kernels Required" section):
- States "no gap blocks a correct initial implementation" (line 277)
- Lists caller-side conventions for prefill non-paged, decode non-paged, and decode paged paths (lines 281–306)
- Concludes "These conventions require no TTNN op modifications and no new kernel code" (line 308)

**Occurrence B** (lines 396–403, "Overall Conclusion" section):
> "A correct initial implementation of windowed attention on TTNN and tt-transformers requires no new kernels. The four existing ops... collectively cover the full windowed attention compute graph. The window constraint is enforced through caller-side conventions: a fixed-shape circular buffer for the KV cache, a band-diagonal additive mask for prefill, a fill-phase validity mask for the early decode steps, and page-table management for the paged path."

Occurrence B lists the same four caller-side conventions already spelled out in occurrence A (circular buffer, band-diagonal mask, fill-phase validity mask, page-table management). The Overall Conclusion adds zero new facts — it is a word-for-word abstraction of the section body immediately above it. The section header "Correct Initial Implementation — No New Kernels Required" at line 275 already serves as the summary. The Overall Conclusion paragraph (lines 396–403) can be reduced to a single forwarding sentence such as "As established above, a correct initial implementation requires no new kernels" followed only by the unique information in lines 405–407 (the performance-optimised enhancement note).

**Redundant:** `existing_kernel_survey.md` lines 396–403 (the prose paragraph listing the four caller-side conventions again).
**Lines saved:** ~6 (collapse to 1–2 sentences).

## MINOR Issues

### M5 — `existing_kernel_survey.md` lines 396–407: "Overall Conclusion" subsection title is redundant with the section heading

The `## Overall Conclusion` heading at line 394 duplicates the role of the "Correct Initial Implementation — No New Kernels Required" subsection heading at line 275. After applying C7 above, the remaining content of the Overall Conclusion is only the performance-optimised note (lines 405–407). That single sentence does not need its own top-level `##` heading. The heading can be dropped and the sentence appended to the performance-optimised subsection or to the gap summary.

**File:** `existing_kernel_survey.md`, line 394.
**Lines saved:** ~1.

### M6 — `roofline_analysis.md` lines 314–315: Duplicate statement of the crossover value within a single paragraph

Lines 314–315 read: "...128 FLOPs/byte exceeds the roofline crossover of 111 FLOPs/byte, confirming the compute-bound regime" — this sentence is the second time within the same paragraph (point 4, lines 300–320) that the value 111 FLOPs/byte is invoked as the crossover threshold. The first invocation is at line 311: "well above the roofline crossover of 111 FLOPs/byte". The second invocation at line 314–315 repeats both the number and the conclusion ("confirming the compute-bound regime") without adding new information. The clause "— 128 FLOPs/byte exceeds the roofline crossover of 111 FLOPs/byte, confirming the compute-bound regime for T >> w" at lines 314–315 can be deleted, since line 311 already makes the same point and the reader has already accepted the conclusion.

**File:** `roofline_analysis.md`, lines 314–315.
**Lines saved:** ~1.

## Load-Bearing Evidence

The following content was verified in Pass 2 as unique and non-removable:

- `existing_kernel_survey.md` lines 60–64: The causal mechanism explanation ("scores computed then suppressed by -inf mask before softmax") is the unique, load-bearing instance of the 87.5% wasted-FLOPs fact. This must be retained; only the duplicate quantification in occurrence B (lines 315–316) is redundant.
- `existing_kernel_survey.md` lines 281–307: The three-path caller-convention enumeration (prefill, decode non-paged, decode paged) is the primary load-bearing statement of the implementation path. The Overall Conclusion's restatement of these conventions (lines 396–403) is what is redundant.
- `existing_kernel_survey.md` lines 405–407: "A performance-optimised implementation adds one enhancement: a chunked windowed prefill loop (or, as a longer-term item, a band-mask-aware Flash Attention kernel) to reduce prefill FLOPs from O(T²) to O(T·w)." This sentence is the only content in the Overall Conclusion that is not already stated elsewhere; it must be preserved.
- `roofline_analysis.md` lines 300–313: The prefill AI formula disambiguation passage (T·w/(T+w) vs G·w/d) remains load-bearing and was already identified in Pass 1. The duplicate 111 FLOPs/byte invocation at lines 314–315 can be trimmed without harming it.

## VERDICT
- Crucial updates: yes

---

# Compression Analysis: Chapter 7 — Roofline Analysis and Existing Kernel Survey — Pass 3

## Summary
- Files analyzed: `index.md`, `roofline_analysis.md`, `existing_kernel_survey.md`
- Pass 1 identified 5 crucial and 4 minor issues; Pass 2 identified 2 additional crucial and 2 minor issues
- Pass 3 performs a full cross-file sweep for any remaining CRUCIAL redundancy not already catalogued
- Pass 3 finds no new CRUCIAL issues

## Scope and Method

Pass 3 checks every candidate pairing of formulas, facts, and tables across the three files for CRUCIAL-level duplication: two occurrences where one adds zero information beyond the other. Minor issues M5 and M6 from Pass 2 are explicitly below the CRUCIAL threshold and are not re-examined.

## Candidates Examined

**Hardware numbers table (`roofline_analysis.md` lines 12–18) vs `index.md`.**
`index.md` contains no hardware numbers; it only cross-references the roofline file by pointer. No duplication.

**AI crossover value (111 FLOPs/byte) across files.**
Derived and used solely within `roofline_analysis.md`. Neither `existing_kernel_survey.md` nor `index.md` states this number. The within-paragraph duplicate at `roofline_analysis.md` lines 314–315 was already flagged as M6 (minor, below threshold). No new CRUCIAL issue.

**Gap capability matrix (`existing_kernel_survey.md` lines 233–239) vs Identified Gaps table (lines 263–271).**
These are complementary tables, not duplicates. The capability matrix is a 4-op × 5-dimension Yes/Partial/Via-caller/No grid; the gap table is a 7-row inventory with description, affected ops, and severity. Neither is a subset of the other.

**"Op Extension vs New Program Config vs New Kernel" table (`existing_kernel_survey.md` lines 383–391) vs the Identified Gaps table.**
The third table adds "Recommended Approach", "Scope", and "Kernel Change?" columns not present in either earlier table. The G4 note ("Same as G1") was flagged as M4 in Pass 1 (minor). No new CRUCIAL issue.

**`roofline_analysis.md` "Implications" point 3 (lines 293–297) vs `index.md` lines 34–35.**
`index.md` is a cross-reference pointer only; it does not reproduce the bandwidth-scaling rationale. No duplication.

**FLOP count formulas (`roofline_analysis.md` lines 57–69) across files.**
These formulas appear only in `roofline_analysis.md`. No cross-file duplication.

**B_crossover formula (`roofline_analysis.md` lines 244–246) across files.**
Appears only in `roofline_analysis.md`. Not repeated elsewhere. No duplication.

**Throughput comparison table (`roofline_analysis.md` lines 204–211) across files.**
`index.md` references its existence in one clause; it does not reproduce data rows. No duplication.

**Approach A Python code block comments (`existing_kernel_survey.md` lines 342–360) vs surrounding prose.**
The prose motivates the code; the inline comments clarify the implementation. The information is complementary (code + explanation), not identical restatement. No CRUCIAL issue.

**"No new kernel required" conclusion — `existing_kernel_survey.md` lines 275–308 vs lines 395–403.**
This pairing was identified as C7 in Pass 2 and remains an open prior-pass finding. Pass 3 does not re-flag it as a new issue.

**87.5% wasted-FLOPs fact — `existing_kernel_survey.md` lines 60–64 vs lines 314–316.**
This pairing was identified as C6 in Pass 2 and remains an open prior-pass finding. Pass 3 does not re-flag it as a new issue.

## New CRUCIAL Issues Found

None.

## VERDICT
- Crucial updates: no
