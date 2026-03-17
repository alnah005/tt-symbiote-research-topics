# Compression Analysis: Chapter 2 — Math Fidelity Levels

## Files Analyzed

| File | Line Count |
|---|---|
| `index.md` | 65 |
| `fidelity_precision_model.md` | 81 |
| `fidelity_and_moe_accuracy.md` | 109 |
| `fidelity_selection_workflow.md` | 137 |
| **Total** | **392** |

---

## Load-Bearing Evidence

Facts that must NOT be removed because they are unique, technically precise, or directly actionable:

1. **Mantissa truncation is pre-multiply, not post-multiply.** `fidelity_precision_model.md` lines 8-9 establish that `math_fidelity` truncates mantissa bits *before* the product is formed, introducing a systematic per-element rounding error into each partial product that accumulates across the K loop. This distinction from post-multiply rounding is not stated elsewhere and is essential to understanding why errors compound.

2. **`math_fidelity` and `fp32_dest_acc_en` are independent, additive levers.** `fidelity_precision_model.md` lines 10-11 (Tip callout) explicitly states the two fields are independent. `fidelity_precision_model.md` lines 70-73 ("What Math Fidelity Does NOT Affect") further defines the exact scope boundary: fidelity affects FPU multiply-accumulate only, not SFPU ops, accumulator precision, or packer path. This scoping is unique to that section.

3. **HiFi4 failure is a non-fidelity diagnostic signal.** `fidelity_selection_workflow.md` lines 26-27 states that if PCC is not > 0.9999 at HiFi4, the cause is a non-fidelity issue (dtype mismatch, layout, weight init, or API bug). This diagnostic logic exists only in the workflow file and must not be removed.

4. **Gate/up K_t=224 vs down K_t=64, counterintuitive sensitivity ordering.** `fidelity_and_moe_accuracy.md` lines 57-64 (K-loop table + explanation) establishes the unintuitive result: gate/up projections have deeper K loops (K_t=224 vs K_t=64) yet are less fidelity-sensitive than down projections, because the SiLU nonlinearity absorbs accumulated error. This specific comparison and the sqrt(K_t) heuristic caveat are load-bearing for reader understanding.

5. **PCC > 0.999 per-projection is necessary but not sufficient for production quality.** `fidelity_selection_workflow.md` lines 110-112 (Tip callout) specifies that end-to-end generation quality checks (perplexity, BLEU, HumanEval, GSM8K) must also pass. This production-readiness caveat does not appear in any other file.

6. **Accumulator precision (`fp32_dest_acc_en=True`) should be tested for down projections in addition to fidelity.** `fidelity_selection_workflow.md` lines 90 (Warning callout) specifically notes that down projections need combined fidelity + accumulator sweep, with a forward reference to Chapter 3. This is the only location where the interaction between `math_fidelity` and `fp32_dest_acc_en` for the down projection is made actionable.

7. **PCC Python function using `torch.corrcoef` in `fidelity_and_moe_accuracy.md` lines 79-82.** The standalone minimal `pcc(a, b)` function is the simplest and most reusable form. The fuller template in `fidelity_selection_workflow.md` wraps TTNN setup, which is load-bearing in its own right. Both serve distinct purposes (one is a utility function, the other is a complete sweep harness), so both must be retained.

---

## CRUCIAL Suggestions

### CRUCIAL-1: Near-Verbatim Duplicate — PCC Small-Sample Warning

**Location A:** `fidelity_and_moe_accuracy.md` lines 94–102 ("Note on PCC Test Reliability")
**Location B:** `fidelity_selection_workflow.md` lines 94–98 ("Warning About Small Batch Sizes")

The two sections make identical claims: M=1 produces high-variance PCC due to small element count; use M >= 128 for reliable results; the numerical properties of the kernel do not change, only the statistical reliability. Location B adds one clarifying sentence ("The numerical properties of the matmul kernel do not change between M=1 and M=128; only the statistical reliability of the PCC estimate changes") that is worth keeping.

**Recommendation:** Remove the "Note on PCC Test Reliability" section from `fidelity_and_moe_accuracy.md` entirely (approximately 10 lines including heading and warning callout). Retain the version in `fidelity_selection_workflow.md` and add a forward reference sentence in `fidelity_and_moe_accuracy.md` (e.g., "See the PCC reliability note in `fidelity_selection_workflow.md` before finalizing any measurement."). **Estimated saving: ~10 lines.**

---

### CRUCIAL-2: Fidelity Decision Table Duplicated Across `index.md` and `fidelity_selection_workflow.md`

**Location A:** `index.md` lines 33–38 ("Fidelity Selection Decision Table") — 3-column table: Projection, Recommended Fidelity, Reasoning.
**Location B:** `fidelity_selection_workflow.md` lines 116–124 ("Summary: Per-Projection Fidelity Map for DeepSeek-V3 Expert Shapes") — 4-column table adding K_t values and Typical PCC.

The core claim is the same in both: gate/up → LoFi, down → HiFi2. Location B's table is strictly more informative (adds K_t and PCC columns), making Location A's table a subset.

**Recommendation:** Remove the 3-column table from `index.md` (lines 33–38, approximately 6 lines of table body) and replace it with a one-sentence reference pointing to the summary table in `fidelity_selection_workflow.md`. The Tip callout below the table in `index.md` (lines 39) should also be removed as it is subsumed by the workflow file's guidance. **Estimated saving: ~8 lines.**

---

### CRUCIAL-3: SiLU Absorption Rationale Restated Within `fidelity_and_moe_accuracy.md`

**Location A:** `fidelity_and_moe_accuracy.md` lines 9–19 ("Why Gate Projections Tolerate LoFi") — establishes that SiLU compresses large-magnitude values, its output sensitivity approaches zero for large |z|, and LoFi errors are absorbed.
**Location B:** `fidelity_and_moe_accuracy.md` lines 42–45 ("Why Down Projections Need HiFi2") — restates: "The SiLU acts as a noise filter: it compresses the distribution and absorbs small rounding errors, so LoFi mantissa errors in gate/up projections are largely neutralized before they can affect the residual stream."

The restatement in Location B is not wrong, but it re-explains what was already proven two sections earlier. The down-projection section needs only the contrast ("unlike gate/up, down has no such absorber") without re-deriving why the absorber works.

**Recommendation:** Trim lines 44–45 in the down-projection section to a single back-reference sentence (e.g., "Gate and up projections benefit from SiLU absorption as established above; down projections have no equivalent absorber."). **Estimated saving: ~3 lines.**

---

## MINOR Suggestions

### MINOR-1: "Next Steps" Sections Duplicate `index.md` Reading Order

`fidelity_precision_model.md` lines 78–80, `fidelity_and_moe_accuracy.md` lines 106–108, and `fidelity_selection_workflow.md` lines 128–136 all include "Next Steps" sections that direct the reader to the next file in the reading order, which is already defined in `index.md` lines 54–58. The workflow file's "Next Steps" also echoes the chapter-completion bullet points.

These sections add no information that `index.md` does not already provide. A reader who loses track can return to `index.md`; restating navigation in every file inflates line count and creates maintenance burden if the reading order ever changes.

**Recommendation:** Remove or shorten "Next Steps" in the two intermediate files (`fidelity_precision_model.md`, `fidelity_and_moe_accuracy.md`) to a single-line pointer ("Continue with the next file in the reading order from `index.md`."). Retain the workflow file's "Next Steps" because it also introduces Chapter 3 and `fp32_dest_acc_en`, which is cross-chapter navigation not in `index.md`. **Estimated saving: ~8 lines across two files.**

---

### MINOR-2: LoFi ~2x Throughput Claim Repeated in Two Files

`fidelity_precision_model.md` line 21 states "the hardware completes tile FMAs in approximately 50% of the cycles required by HiFi4" (i.e., ~2× faster). `index.md` learning objective line 22 restates "LoFi offers higher throughput (~2× over HiFi4)". Since `index.md` is the entry point and `fidelity_precision_model.md` is the authoritative characterization file, the learning objective in `index.md` can reference the source without restating the number.

**Recommendation:** Shorten `index.md` learning objective #2 to remove the numeric claim ("Explain why LoFi offers higher throughput for matmul-intensive workloads by completing each tile FMA in fewer cycles."), leaving the quantification in `fidelity_precision_model.md` where it is properly qualified. **Estimated saving: ~1 line (in-place edit, not removal).**

---

### MINOR-3: Overview Paragraph in `fidelity_and_moe_accuracy.md` Restates `index.md` Central Result

`fidelity_and_moe_accuracy.md` lines 3–5 (Overview) restates that gate/up tolerate LoFi and down needs HiFi2, mirroring `index.md` lines 12–13 almost exactly. The opening overview of a sub-file repeating the chapter overview adds no orientation value since readers arrive from `index.md`.

**Recommendation:** Trim the Overview section of `fidelity_and_moe_accuracy.md` to a single sentence stating the chapter's question ("This section explains why the three MoE projection types map to different fidelity requirements."), removing the restatement of the conclusion that the reader is about to derive. **Estimated saving: ~3 lines.**

---

### MINOR-4: Warning Callout About 256 Experts is Isolated and Unsupported

`fidelity_and_moe_accuracy.md` line 49 (Warning callout) asserts that across 256 experts per forward pass, accumulated residual stream drift "can meaningfully degrade end-to-end model quality." This claim is stated but not quantified or sourced. It is also not echoed elsewhere — which means it is not a duplicate — but it is an unvalidated claim that overstates certainty. However, the structural problem (a warning that implies but does not support causation) is a quality issue, not a duplication issue; flagged here for completeness.

**Recommendation:** Qualify the warning with "may" language and note it is an empirical claim requiring model-level validation: change "can meaningfully degrade" to "may degrade" and add "validate at the model output level to confirm." This is an in-place edit, not a line reduction.

---

## Summary

| Metric | Value |
|---|---|
| Files analyzed | 4 |
| Current total line count | 392 |
| Lines removable (CRUCIAL duplicates) | ~21 |
| Lines removable (MINOR duplicates) | ~12 |
| Total lines after compression | ~359 |
| Reduction | ~8.4% |

The chapter is tightly written overall. The majority of content is unique and load-bearing. The most impactful single change is CRUCIAL-1 (removing the duplicate M=1 PCC warning from `fidelity_and_moe_accuracy.md`), followed by CRUCIAL-2 (removing the redundant 3-column decision table from `index.md`). MINOR-1 yields the largest line reduction after the CRUCIAL changes.

---

## VERDICT

Crucial updates: yes

## Agent A Change Log — C Feedback Pass 1
- fidelity_and_moe_accuracy.md: Replaced duplicate M=1/small-batch PCC warning with cross-reference to fidelity_selection_workflow.md
- index.md: Removed duplicate 3-column fidelity table; replaced with pointer to fidelity_selection_workflow.md's complete 4-column version
- fidelity_and_moe_accuracy.md: Removed internal restatement of SiLU absorption rationale; added back-reference to earlier section

---

# Compression Analysis: Chapter 2 — Math Fidelity Levels — Pass 2

## Summary

| File | Line Count (Pass 1) | Line Count (Pass 2) | Delta |
|---|---|---|---|
| `index.md` | 65 | 56 | −9 |
| `fidelity_precision_model.md` | 81 | 80 | −1 |
| `fidelity_and_moe_accuracy.md` | 109 | 100 | −9 |
| `fidelity_selection_workflow.md` | 137 | 136 | −1 |
| **Total** | **392** | **372** | **−20** |

Pass 1 achieved a net reduction of 20 lines (from 392 to 372), approximately 5.1% compression — slightly below the projected ~21 CRUCIAL-line reduction, consistent with the fixes being applied correctly at tight scope.

---

## Verification of Pass 1 Fixes

### Fix 1 — CRUCIAL-1: M=1/small-batch PCC warning in `fidelity_and_moe_accuracy.md`

**Status: CONFIRMED APPLIED.**

`fidelity_and_moe_accuracy.md` lines 92–95 now read:

```
## Note on PCC Test Reliability

See the PCC reliability note in `fidelity_selection_workflow.md` before finalizing any measurement.
```

The original ~10-line "Note on PCC Test Reliability" block (including the Warning callout restating M=1 variance, the M >= 128 guidance, and the "numerical properties do not change" sentence) has been replaced with a 3-line section: heading + one cross-reference sentence + blank line. The authoritative version remains intact in `fidelity_selection_workflow.md` lines 94–99. Fix correctly applied.

### Fix 2 — CRUCIAL-2: 3-column fidelity table in `index.md`

**Status: CONFIRMED APPLIED.**

`index.md` lines 29–32 now read:

```
## Fidelity Selection Decision Table

For the full fidelity comparison with K_t and PCC data, see `fidelity_selection_workflow.md`.
```

The original 3-column table (Projection / Recommended Fidelity / Reasoning, approximately 6 body lines) and the Tip callout below it have been removed. The section heading was retained and now contains a single pointer sentence. The 4-column authoritative table remains intact in `fidelity_selection_workflow.md` lines 116–124. Fix correctly applied.

### Fix 3 — CRUCIAL-3: SiLU absorption rationale restatement in `fidelity_and_moe_accuracy.md`

**Status: CONFIRMED APPLIED.**

`fidelity_and_moe_accuracy.md` lines 41–45 now read:

```
The primary factor making the down projection accuracy-sensitive is direct residual stream injection:

**Direct residual stream injection.** The down projection output is added directly to the residual stream — it is not followed by any nonlinearity before the next layer's normalization and attention. There is no SiLU, ReLU, or other compressive function to absorb accumulated rounding error. Any per-tile LoFi rounding error flows into every subsequent layer without a reset.

Gate and up projections benefit from SiLU absorption as established above; down projections have no equivalent absorber.
```

The earlier version contained approximately 2–3 lines restating why SiLU absorption works (compresses distribution, output sensitivity approaches zero for large |z|). This has been trimmed to a single back-reference sentence: "Gate and up projections benefit from SiLU absorption as established above; down projections have no equivalent absorber." The original SiLU derivation is fully preserved in the "Why Gate Projections Tolerate LoFi" section (lines 9–19). Fix correctly applied.

---

## CRUCIAL Suggestions

None — all Pass 1 CRUCIAL items resolved.

All three CRUCIAL duplications identified in Pass 1 have been addressed:
- CRUCIAL-1: Duplicate M=1/small-batch PCC warning removed from `fidelity_and_moe_accuracy.md`; cross-reference to `fidelity_selection_workflow.md` substituted.
- CRUCIAL-2: Redundant 3-column fidelity decision table removed from `index.md`; pointer to the 4-column version in `fidelity_selection_workflow.md` substituted.
- CRUCIAL-3: Internal SiLU absorption rationale restatement removed from the down-projection section of `fidelity_and_moe_accuracy.md`; single back-reference sentence substituted.

No new CRUCIAL duplications were identified in Pass 2. Each of the four files now contains content that is either unique to that file or properly cross-referenced rather than restated. The `fidelity_precision_model.md` file remains entirely non-duplicative with the other three files across all its sections.

---

## MINOR Suggestions

The following MINOR suggestions from Pass 1 remain unresolved. None was addressed by the Pass 1 edits.

### MINOR-1 (Carried Forward): "Next Steps" Sections Duplicate `index.md` Reading Order

`fidelity_precision_model.md` line 78–80 and `fidelity_and_moe_accuracy.md` lines 98–100 both contain "Next Steps" sections that direct the reader to the next file in sequence, information already encoded in `index.md` lines 46–51 (Reading Order). Both Next Steps sections in the intermediate files have not been shortened to a single-line pointer.

**Recommendation (unchanged):** Trim "Next Steps" in `fidelity_precision_model.md` and `fidelity_and_moe_accuracy.md` to a single line each (e.g., "Continue with the next file in the reading order from `index.md`."). Retain `fidelity_selection_workflow.md`'s "Next Steps" in full because it introduces Chapter 3. Estimated saving: ~8 lines across two files.

### MINOR-2 (Carried Forward): LoFi ~2x Throughput Claim in Two Files

`index.md` line 22 retains "LoFi offers higher throughput (~2× over HiFi4)" in Learning Objective #2. `fidelity_precision_model.md` line 21 is the authoritative source with proper qualification ("approximately 50% of the cycles"). The numeric claim in the learning objective is a minor restatement.

**Recommendation (unchanged):** Shorten `index.md` learning objective #2 to remove the numeric claim, leaving it in `fidelity_precision_model.md` where it is properly qualified. Estimated saving: ~1 line (in-place edit).

### MINOR-3 (Carried Forward): Overview Paragraph in `fidelity_and_moe_accuracy.md` Restates `index.md` Central Result

`fidelity_and_moe_accuracy.md` lines 3–5 (Overview) restates that gate/up tolerate LoFi and down needs HiFi2. This mirrors `index.md` lines 12–13. The overview has not been trimmed to a single orienting question sentence.

**Recommendation (unchanged):** Trim the Overview to a single sentence stating the chapter's question, removing the restatement of the conclusion. Estimated saving: ~3 lines.

### MINOR-4 (Carried Forward): Unqualified Warning About 256-Expert Residual Drift

`fidelity_and_moe_accuracy.md` line 49 (Warning callout) asserts accumulated residual stream drift across 256 experts "can meaningfully degrade end-to-end model quality" without quantification or source citation. The "may" qualification recommended in Pass 1 was not applied.

**Recommendation (unchanged):** Change "can meaningfully degrade" to "may degrade" and add "validate at the model output level to confirm." In-place edit, no line reduction.

---

## Load-Bearing Evidence

1. **Mantissa truncation is pre-multiply, not post-multiply.** `fidelity_precision_model.md` lines 8–9 establish that `math_fidelity` truncates mantissa bits before the product is formed, introducing systematic per-element rounding error into each partial product that accumulates across the K loop. This distinction from post-multiply rounding is not stated anywhere else across the four files and is foundational to understanding why errors compound differently from, e.g., output quantization.

2. **`math_fidelity` and `fp32_dest_acc_en` are independent, additive levers.** `fidelity_precision_model.md` lines 10–11 (Tip callout) and lines 70–73 ("What Math Fidelity Does NOT Affect") together define the exact scope boundary: fidelity affects FPU multiply inputs only; accumulator precision, SFPU ops, and packer path are all outside its scope. This scoping is unique to `fidelity_precision_model.md` and must not be merged or removed.

3. **HiFi4 failure is a non-fidelity diagnostic signal.** `fidelity_selection_workflow.md` lines 26–27 states that PCC not > 0.9999 at HiFi4 indicates a non-fidelity cause (dtype mismatch, layout error, weight init bug, or API bug). This diagnostic decision rule exists only in the workflow file. No other file encodes what a HiFi4 failure means or what to do when it occurs.

4. **Gate/up K_t=224 vs down K_t=64, counterintuitive sensitivity ordering.** `fidelity_and_moe_accuracy.md` lines 57–64 establishes that gate/up have deeper K loops (K_t=224) yet lower fidelity sensitivity than down (K_t=64), because SiLU absorption dominates over K-loop depth. The sqrt(K_t) heuristic caveat ("this heuristic applies only when K-loop depth is actually the dominant sensitivity driver") is stated only here and is essential for a reader who might otherwise misapply the heuristic.

5. **PCC > 0.999 per-projection is necessary but not sufficient for production quality.** `fidelity_selection_workflow.md` lines 110–112 (Tip callout) specifies that end-to-end generation quality checks (perplexity, BLEU, HumanEval, GSM8K) must also pass. This production-readiness caveat does not appear in any other file and is the only place in the chapter that names specific downstream evaluation metrics.

6. **`fp32_dest_acc_en=True` must be co-tested with fidelity for down projections.** `fidelity_selection_workflow.md` line 90 (Warning callout) specifies that the PCC sweep template sets `fp32_dest_acc_en=False` by default and that down projections require a combined fidelity + accumulator sweep, with a forward reference to Chapter 3. This is the only location in the chapter where the interaction of `math_fidelity` and `fp32_dest_acc_en` for down projections is made explicitly actionable with a cross-chapter pointer.

7. **The standalone `pcc(a, b)` utility function in `fidelity_and_moe_accuracy.md` lines 79–82 and the full sweep harness in `fidelity_selection_workflow.md` lines 57–85 serve distinct purposes.** The utility function is minimal and reusable outside the TTNN context. The harness wraps full TTNN device setup, config construction, and tensor conversion. Both must be retained because a reader may need the utility function in isolation (e.g., when integrating into an existing test harness) or the full harness (e.g., when running a fresh sweep from scratch). They are not duplicates of each other.

---

## VERDICT

Crucial updates: no
