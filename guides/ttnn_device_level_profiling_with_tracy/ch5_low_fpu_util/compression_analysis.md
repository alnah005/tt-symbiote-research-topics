## Agent A Change Log — B Review Pass 1
- csv_signatures.md: Corrected Cause 1 detection threshold to CORE COUNT > (M_t × N_t) / 4.
- causes_of_low_fpu_util.md + csv_signatures.md: Removed incorrect FPU UTIL as Cause 6 signal; Cause 6 signal is DEVICE KERNEL DURATION only.
- csv_signatures.md: Fixed self-contradicting M=16 Cause 7 example; replaced with general principle statement.

## Agent A Change Log — B Review Pass 2
- causes_of_low_fpu_util.md: Removed remaining Cause 6 FPU UTIL claim; signal is DEVICE KERNEL DURATION only.
- causes_of_low_fpu_util.md: Fixed Cause 7 mechanism direction — kernel over-iterates (not under-iterates) on padded tiles.

## Agent A Change Log — B Review Pass 3
- remediation_levers.md: Fixed misleading "borderline" label for 8-core/16-tile example (ratio 2.0 < 4 threshold); labeled as tile-starved below guideline.

## Agent A Change Log — B Review Pass 4
- causes_of_low_fpu_util.md: Corrected FP32 throughput penalty explanation from incorrect "512-bit SIMD" to accurate tile-processing cost (more internal passes per 32×32 tile).

## Agent A Change Log — B Review Pass 5
- remediation_levers.md: Fixed in0_block_w L1 cost formula — replaced double-counted K_t_per_block with per_core_M × in0_block_w.

## Agent A Change Log — B Review Pass 6
- remediation_levers.md: Clarified BF8 format benefit — reduced bandwidth/TRISC0 load, not increased TRISC1 FMA rate; FPU throughput column updated accordingly.

## Agent A Change Log — B Review Pass 7
- csv_signatures.md: Unified Cause 5 NOC BW UTIL thresholds — changed warning threshold from > 0.7 to > 0.8 to match detection condition.

## Agent A Change Log — B Review Pass 8
- csv_signatures.md: Fixed Cause 7 exclusion — Cause 3 now ruled out for both HiFi2 and HiFi4 (only proceed to Cause 7 if MATH FIDELITY is LoFi).

---

# Compression Analysis: Chapter 5 — Low FPU Utilization: Causes and Remediation — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~697 lines
- Estimated post-compression line count: ~600 lines
- Estimated reduction: ~14%

## CRUCIAL Suggestions

### causes_of_low_fpu_util.md ~lines 82–86 / remediation_levers.md ~lines 62–65
**Issue:** The math fidelity throughput table (columns: fidelity level, FMA iterations per tile, relative throughput) appears in full in both files. The `causes_of_low_fpu_util.md` version is under "Cause 3 — Mechanism"; the `remediation_levers.md` version is under "`math_fidelity` Parameter". They are character-for-character equivalent in content; neither adds a column or row the other lacks.
**Suggestion:** Remove the table from `causes_of_low_fpu_util.md` and replace it with a one-sentence prose statement of the throughput ratios (e.g., "HiFi4 performs 4× more FMA iterations per tile than LoFi, cutting throughput to ~0.25×; HiFi2 performs 2×, cutting to ~0.5×."). Keep the authoritative table in `remediation_levers.md` where it sits alongside the usage example and the accuracy warning.

### causes_of_low_fpu_util.md ~lines 7–11 / csv_signatures.md ~lines 7–9 / index.md ~line 5
**Issue:** The FPU UTIL classification thresholds (>0.7 compute-bound, <0.3 bandwidth-bound) and the Wormhole B0 ridge point (8.0 FLOPs/byte) are restated verbatim as standalone prose or bullet lists in all three files. Chapter 4 already owns these definitions; Chapter 5 cross-references Chapter 4 in its prerequisites.
**Suggestion:** In `causes_of_low_fpu_util.md` (lines 7–11) and `csv_signatures.md` (lines 7–9), replace the full restatement with a single parenthetical back-reference: "(thresholds established in Chapter 4: >0.7 compute-bound, <0.3 bandwidth/overhead-bound; ridge point 8.0 FLOPs/byte)." The `index.md` usage is embedded in a narrative sentence and can stay as-is.

### causes_of_low_fpu_util.md ~line 225 / remediation_levers.md ~lines 199–203
**Issue:** The boundary between Cause 7 and Chapter 6 dispatch overhead is explained twice. `causes_of_low_fpu_util.md` line 225 has a full Note callout distinguishing device-side (Cause 7) from host-side (Chapter 6) dispatch latency, with a sentence defining mesh trace. `remediation_levers.md` lines 199–203 contains an entire "Dispatch Overhead Elimination via Mesh Trace" section that re-explains the same boundary, adds no new diagnostic information, and is only a pointer to Chapter 6.
**Suggestion:** Delete the `remediation_levers.md` "Dispatch Overhead Elimination via Mesh Trace" section (lines 197–203, including its Note callout). The Note in `causes_of_low_fpu_util.md` already establishes the boundary and points to Chapter 6. `remediation_levers.md` should cover only the seven FPU-utilization levers; the Chapter 6 pointer is out of scope for that file.

## MINOR Suggestions

### csv_signatures.md ~line 25
**Issue:** After the 7-item Diagnostic Checklist, line 25 adds a prose sentence restating the ordering rationale: "This order is chosen to progress from causes that require only a single column check … to causes that require kernel-level investigation." The checklist items already encode this rationale in their parenthetical annotations ("rule out first; takes seconds to verify", "reached only after all above are ruled out"). The sentence is redundant with its own list.
**Suggestion:** Delete line 25. The checklist is self-explanatory.

### causes_of_low_fpu_util.md ~line 40 / remediation_levers.md ~line 42
**Issue:** The guideline `M_t × N_t / core_count ≥ 4` appears three times: `causes_of_low_fpu_util.md` line 35, `csv_signatures.md` line 54, and `remediation_levers.md` line 31. The `csv_signatures.md` occurrence is necessary (it is the detection condition). The `remediation_levers.md` occurrence is necessary (it drives the config choice). The `causes_of_low_fpu_util.md` occurrence in the Fix subsection (line 35) duplicates the remediation_levers version.
**Suggestion:** In `causes_of_low_fpu_util.md` Cause 1 Fix subsection (line 35), replace the formula with a forward reference: "See `remediation_levers.md` for the tile-per-core guideline and worked example." Keep the formula in the other two files.

### csv_signatures.md ~lines 173–174
**Issue:** The "Quantitative signal" paragraph for Cause 7 in `csv_signatures.md` (lines 173–174) re-explains the padding mechanism (kernel iterates over padding zeros, depressing FPU UTIL in proportion to padding fraction). This mechanism is already fully described in `causes_of_low_fpu_util.md` lines 219–223.
**Suggestion:** Shorten to one sentence: "The FPU UTIL deficit scales with the padding fraction in the dominant dimension." Remove the mechanism re-explanation; keep the cross-reference to `causes_of_low_fpu_util.md`.

### csv_signatures.md ~lines 149–150 / causes_of_low_fpu_util.md ~lines 155–156
**Issue:** The warning that Cause 5 should not be confused with a legitimately bandwidth-bound op appears as a full Warning callout in `csv_signatures.md` (lines 149–150) and is also stated in `causes_of_low_fpu_util.md` lines 155–156 ("This is distinct from Cause 4 in that the bottleneck is not a single core's unpacker latency but rather system-wide bandwidth saturation."). The distinction between Cause 4 and Cause 5 belongs in `causes_of_low_fpu_util.md`; the Warning in `csv_signatures.md` separately conflates two different distinctions (Cause 5 vs. Cause 4, and Cause 5 vs. legitimately bandwidth-bound) in a way that muddies both.
**Suggestion:** Trim the `csv_signatures.md` Warning callout to one sentence: "Cause 5 applies only when AI > 8.0 FLOPs/byte; an elementwise op with high NOC BW UTIL and low FPU UTIL is legitimately bandwidth-bound and needs no fix." Delete the rest of the warning paragraph (already covered in `causes_of_low_fpu_util.md`).

## Load-Bearing Evidence
- **index.md line 44:** `"> **Note:** The diagnostic checklist in csv_signatures.md specifies the recommended order for ruling out causes"` — this cross-reference note is the only place in index.md that explains *why* the checklist order matters; removing it would leave the summary table without a pointer to the ordering rationale, so it must be preserved.
- **causes_of_low_fpu_util.md lines 219–223:** `"If the kernel's inner loop is generated using padded tile counts rather than logical tile counts, the math engine performs more FMA iterations than the op's logical shape requires"` — this is the primary (and only) mechanistic explanation of Cause 7 in the chapter; it cannot be cut without losing the root cause description entirely.
- **csv_signatures.md lines 162–169:** The six-condition exclusion checklist for Cause 7 (ruling out Causes 1–6 before concluding Cause 7) — this structured list is load-bearing diagnostic logic not replicated in either `causes_of_low_fpu_util.md` or `remediation_levers.md`.
- **remediation_levers.md lines 187–193:** The `in0_block_w` L1 cost formula and per-value buffering guideline table — this is the only place in the chapter where the L1 footprint tradeoff is quantified with a formula; it must be kept intact.

## VERDICT
- Crucial updates: yes

## Agent A Change Log — B Review Pass 10
- causes_of_low_fpu_util.md: Fixed inverted fidelity hierarchy sentence — HiFi4 is 4× more iterations than LoFi (not the reverse).

## Agent A Change Log — C Compression Pass 1
- causes_of_low_fpu_util.md: Removed duplicate math fidelity table; replaced with prose sentence referencing multipliers.
- causes_of_low_fpu_util.md + csv_signatures.md: Collapsed verbose FPU UTIL threshold preambles to parenthetical back-references to Chapter 4.
- remediation_levers.md: Removed duplicate Chapter 6 dispatch overhead section; single cross-reference in causes_of_low_fpu_util.md is sufficient.

---

# Compression Analysis: Chapter 5 — Low FPU Utilization: Causes and Remediation — Pass 2

## Summary

Pass 1 crucial fixes confirmed resolved (all three items). No new crucial issues found. Four carry-over minor items from Pass 1 remain unaddressed, plus one new minor redundancy identified in `causes_of_low_fpu_util.md`.

## CRUCIAL Suggestions

None. All three Pass 1 crucial items are resolved:
- Duplicate math fidelity table in `causes_of_low_fpu_util.md` — replaced with prose sentence.
- Verbose FPU UTIL threshold preambles in `causes_of_low_fpu_util.md` and `csv_signatures.md` — collapsed to parenthetical back-references.
- Duplicate Chapter 6 dispatch overhead section in `remediation_levers.md` — deleted.

## MINOR Suggestions

### csv_signatures.md line 21 (carry-over from Pass 1 Minor #1)
**Issue:** The sentence "This order is chosen to progress from causes that require only a single column check (Causes 6, 3, 2) to causes that require computed ratios (Cause 4), to causes that require combined conditions (Cause 5), to causes that require kernel-level investigation (Cause 7)." restates the ordering rationale already encoded in the checklist items' own parenthetical annotations ("rule out first; takes seconds to verify", "reached only after all above are ruled out").
**Suggestion:** Delete the sentence. The checklist is self-explanatory.

### causes_of_low_fpu_util.md Cause 1 Fix subsection line 29 (carry-over from Pass 1 Minor #2)
**Issue:** The formula `M_t × N_t / core_count ≥ 4` is stated three times: once in the Cause 1 Fix subsection of `causes_of_low_fpu_util.md`, once in `csv_signatures.md` (detection condition), and once in `remediation_levers.md` (config guideline). The detection and remediation occurrences are each necessary in their file; the third occurrence in `causes_of_low_fpu_util.md`'s Fix subsection duplicates the `remediation_levers.md` version.
**Suggestion:** In `causes_of_low_fpu_util.md` Cause 1 Fix subsection, replace the formula block with a forward reference: "See `remediation_levers.md` for the tile-per-core guideline and worked example."

### csv_signatures.md Cause 7 "Quantitative signal" paragraph lines 169–170 (carry-over from Pass 1 Minor #3)
**Issue:** The "Quantitative signal" paragraph re-explains the padding mechanism ("kernel's inner loop is generated from padded tile counts rather than logical tile counts, a portion of FPU cycles are spent on padding zeros rather than real data, depressing FPU UTIL in proportion to the padding fraction"). The full mechanism is already described in `causes_of_low_fpu_util.md` lines 207–213.
**Suggestion:** Trim to one sentence: "The FPU UTIL deficit scales with the padding fraction in the dominant dimension." Keep the existing cross-reference to `causes_of_low_fpu_util.md` for the investigation steps.

### csv_signatures.md Cause 5 Warning callout lines 140–145 (carry-over from Pass 1 Minor #4)
**Issue:** The Warning callout for Cause 5 conflates two separate distinctions in a single paragraph: (a) Cause 5 vs. legitimately bandwidth-bound ops, and (b) Cause 5 vs. Cause 4. The Cause 4/5 distinction is already handled under "Distinguishing from Cause 5" in the Cause 4 section. The Warning duplicates both distinctions in one callout.
**Suggestion:** Trim the Warning to one sentence covering only the bandwidth-bound confusion: "Cause 5 applies only when AI > 8.0 FLOPs/byte; an elementwise op with high NOC BW UTIL and low FPU UTIL is legitimately bandwidth-bound and needs no fix."

### causes_of_low_fpu_util.md Cause 3 Mechanism lines 74–77 (new)
**Issue:** After stating "HiFi4 runs ~4× more FMA iterations per tile than LoFi" (line 73) and pointing to the authoritative table in `remediation_levers.md`, the next paragraph opens with "When HiFi4 is the active fidelity, the math engine performs four times as many passes per tile as LoFi." This is a verbatim restatement of the preceding sentence — same multiplier, same comparison pair, same direction.
**Suggestion:** Delete the opening clause "When `HiFi4` is the active fidelity, the math engine performs four times as many passes per tile as `LoFi`." from the second paragraph, starting the paragraph instead with "If the application does not require that precision…".

## Load-Bearing Evidence

- **causes_of_low_fpu_util.md line 73:** `"HiFi4 runs ~4× more FMA iterations per tile than LoFi, and HiFi2 runs ~2× more than LoFi (authoritative throughput table in remediation_levers.md)"` — confirms the Pass 1 table-removal fix is in place and the cross-reference to the table is preserved; the prose is the only in-file quantification of fidelity cost now that the table is gone.
- **csv_signatures.md lines 162–167:** The six-condition exclusion checklist for Cause 7 — load-bearing diagnostic logic not replicated elsewhere; must be kept intact.
- **remediation_levers.md lines 187–193:** The `in0_block_w` L1 cost formula (`extra_L1_bytes = per_core_M × in0_block_w × 32 × 32 × bytes_per_element`) and per-value buffering guideline table — the only place in the chapter where the L1 footprint tradeoff is quantified; must be preserved.
- **causes_of_low_fpu_util.md lines 207–213:** The Cause 7 mechanism (kernel over-iterates on padded tiles) — primary and only mechanistic explanation of Cause 7; cannot be cut.

## VERDICT
- Crucial updates: no
