## Agent A Change Log — B Review Pass 1
- roofline_model_primer.md: Fixed tile throughput claim from "one tile per cycle" to "one tile every 16 cycles" (2048 FLOPs ÷ 128 FLOPs/cycle).
- roofline_model_primer.md: Changed AI formula from ≈ to = (exact for BF16 square matrices).
- classification_method.md + index.md: Unified NOC BW UTIL threshold to < 0.4 everywhere (matching classification table).

## Agent A Change Log — B Review Pass 2
- roofline_model_primer.md: Fixed FPU ceiling units — 128 FMA ops/cycle = 256 FLOPs/cycle; ridge point corrected from 4.0 to 8.0 FLOPs/byte.
- roofline_model_primer.md: Corrected L1 size from 192 KB to ~1.5 MB per Tensix core (Wormhole B0).

## Agent A Change Log — B Review Pass 3
- classification_method.md: Updated ridge point from 4.0 to 8.0 FLOPs/byte throughout.
- worked_examples.md: Updated ridge point from 4.0 to 8.0 FLOPs/byte throughout.

## Agent A Change Log — B Review Pass 4
- worked_examples.md: Fixed FPU consumption rate from 128 to 256 FLOPs/cycle (128 FMA ops × 2 FLOPs/FMA).

---

# Compression Analysis: Chapter 4 — Compute-Bound vs. Bandwidth-Bound Analysis — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~645 lines
- Estimated post-compression line count: ~555 lines
- Estimated reduction: ~14%

## CRUCIAL Suggestions

### classification_method.md ~lines 172–181
**Issue:** The "Classification Result Table" (lines 172–181) is an exact copy of the Quick-Reference Decision Table already present in `index.md` lines 37–44, with identical columns, identical threshold values, and only a slightly longer trailing note. Having two byte-for-byte identical four-row tables across two files in the same chapter is pure duplication.
**Suggestion:** Delete the table body from `classification_method.md` and replace it with a single cross-reference sentence: "See the Quick-Reference Decision Table in [`index.md`](./index.md) for threshold values; the note on engineering judgment below still applies." Keep only the note that follows the table (line 181), as it adds the useful caveat about 0.65/0.15 edge cases that the index table lacks.

### roofline_model_primer.md ~lines 91–99
**Issue:** Lines 91–97 calculate the matmul tile throughput in narrative prose (SIMD width → 32 elements/cycle → 64 FLOPs/cycle for elementwise → separate path for matrix ops → 1024 MACs/tile → 8 cycles/tile → 256 FLOPs/cycle). Line 99 is then a blockquote that immediately re-states the same final conclusion ("128 FP16 FMA operations per cycle … 256 FLOPs/cycle per core") in full. The blockquote restates the derivation's endpoint word-for-word.
**Suggestion:** Delete the `> **Note:**` blockquote at line 99. The derivation in lines 91–97 already establishes the 256 FLOPs/cycle figure; repeating it as a callout adds no information.

### roofline_model_primer.md ~lines 128–130
**Issue:** Lines 128 and 130 both restate the ridge-point conclusion ("8 FLOPs/byte") immediately after the code block that computed it. Line 128 adds a sentence; line 130 is a `> **Tip:**` blockquote restating the identical rule ("If you compute AI for an op and it exceeds 8, your first hypothesis should be compute-bound. If it is below 8, your first hypothesis should be bandwidth-bound."). This exact heuristic is already encoded in the decision table in `index.md` and the flowchart in `classification_method.md`.
**Suggestion:** Delete the `> **Tip:**` blockquote (lines 130–131). The sentence on line 128 is sufficient to deliver the ridge-point takeaway.

## MINOR Suggestions

### index.md ~lines 5–11
**Issue:** The Overview paragraph (lines 5–11) contains two hedging constructions back-to-back: "Understanding which limit applies … is the prerequisite for any meaningful optimization effort" followed by the restatement "an op that is compute-bound will not benefit from reducing memory traffic, and an op that is bandwidth-bound will not benefit from algorithmic tricks that reduce FLOP count." The restatement just expands the meaning of "prerequisite" without adding new information. The blockquote on line 9 then asks the same question a third way.
**Suggestion:** Cut the parenthetical restatement sentence ("an op that is compute-bound … FLOP count") and keep only the opening sentence plus the blockquoted question. Saves ~2 lines and removes the repetition without losing any meaning.

### worked_examples.md ~lines 1–3
**Issue:** The preamble sentence on line 3 explicitly lists all five steps ("(1) theoretical AI, (2) FPU UTIL, (3) NOC BW UTIL, (4) DEVICE KERNEL DURATION / PM IDEAL, (5) TRISC breakdown") which are already named as section headings inside every example that follows. The list previews content visible two lines later.
**Suggestion:** Replace the preamble with one sentence: "Each example applies the five-step procedure from [`classification_method.md`](./classification_method.md)." Drop the enumerated step list.

### worked_examples.md ~lines 31–32
**Issue:** The `> **Note:**` on lines 31–32 in Example 1 offers a "quick ratio" shortcut (M×K×N / (K×N) ≈ M) for approximating AI when K and N dominate. This is a narrow edge-case shortcut applicable only when the specific shape inequality holds, and it is not referenced again anywhere. It adds cognitive load for minimal payoff.
**Suggestion:** Delete the note. Readers who want a fast approximation already have the formula on lines 13–25.

### classification_method.md ~lines 71–75
**Issue:** The cross-reference bullet list at lines 71–75 ("FPU UTIL high + NOC BW UTIL low → compute-bound; FPU UTIL low + NOC BW UTIL high → bandwidth-bound; FPU UTIL low + NOC BW UTIL low → overhead-bound") restates three of the four rows of the table that immediately precedes it (lines 65–69), just reformatted from a table into bullets.
**Suggestion:** Delete the three bullet cross-reference lines. The table already conveys all three cases. Keep only the introductory sentence "Cross-reference FPU UTIL and NOC BW UTIL together:" if a lead-in to Step 4 is needed, or merge it into Step 4's opening.

## Load-Bearing Evidence
- `index.md` line ~44: `"> **Note:** These thresholds are starting points. Always cross-check against \`DEVICE KERNEL DURATION / PM IDEAL\` and the TRISC duration breakdown before drawing a final conclusion."` — load-bearing because it is the only place in the chapter index that explicitly warns readers not to rely on the quick-reference table alone; removing it would leave the table without a critical caveat.
- `roofline_model_primer.md` line ~97: `"The matmul engine sustains one tile every 8 cycles when fully pipelined (2048 FLOPs ÷ 256 FLOPs/cycle = 8 cycles per tile), yielding a practical throughput ceiling of 128 FP16/BF16 FMA ops per cycle per core."` — load-bearing because it is the only place the 8-cycles-per-tile pipeline depth is derived; this figure underlies the ridge-point calculation and cannot be cut without losing the derivation.
- `classification_method.md` line ~91: `"Warning: High \`FPU UTIL\` does not automatically mean the kernel is fast. It means the FPU is doing useful work for most of TRISC1's active time. If \`DEVICE KERNEL DURATION\` is still much larger than \`PM IDEAL\`, there may be stalls outside TRISC1 (e.g., data-movement waiting in TRISC0 or TRISC2)."` — load-bearing because it is the only prose in the chapter that disambiguates high FPU UTIL from high overall efficiency; this prevents a common misreading of the metric.
- `worked_examples.md` line ~218: `"The small matmul is the most instructive: its theoretical AI suggests compute-bound, but the actual CSV metrics reveal overhead-bound behavior because the tile count is too small to keep the pipeline fed. Always let the measured \`FPU UTIL\` and \`overhead_ratio\` override the theoretical prediction."` — load-bearing because it is the chapter's central pedagogical lesson, stated explicitly only once, in the summary comparison table's caption.

## VERDICT
- Crucial updates: yes

## Agent A Change Log — C Compression Pass 1
- classification_method.md: Removed duplicate Classification Result Table; replaced with cross-reference to index.md.
- roofline_model_primer.md: Removed > Note: blockquote that restated the 256 FLOPs/cycle derivation conclusion.
- roofline_model_primer.md: Removed > Tip: blockquote that restated the AI > 8 / < 8 ridge point heuristic.

## Agent A Change Log — B Review Pass 6
- worked_examples.md: Removed misleading AI ≈ M shortcut; kept exact AI ≈ 683 FLOPs/byte value with qualification.

---

# Compression Analysis: Chapter 4 — Compute-Bound vs. Bandwidth-Bound Analysis — Pass 2

## Summary
- Pass 1 CRUCIAL items: all three confirmed resolved.
- No new CRUCIAL issues found.
- Two MINOR issues from Pass 1 remain unaddressed (classification_method.md bullet restatement; index.md parenthetical sentence); one Pass 1 MINOR item is now resolved (worked_examples.md AI shortcut Note, removed in B Review Pass 6). One additional MINOR item identified in worked_examples.md preamble.

## CRUCIAL Suggestions
None. All three Pass 1 CRUCIAL items are confirmed fixed:
1. `classification_method.md` duplicate Classification Result Table — replaced with cross-reference to `index.md` (line 172–174 now reads "See the Quick-Reference Decision Table in `index.md`...").
2. `roofline_model_primer.md` restating Note blockquote for 256 FLOPs/cycle — absent from current file.
3. `roofline_model_primer.md` restating Tip blockquote for AI > 8 / < 8 heuristic — absent from current file.

## MINOR Suggestions

### classification_method.md lines 71–75 (unresolved from Pass 1)
**Issue:** The three bullet lines following the `NOC BW UTIL` table ("FPU UTIL high + NOC BW UTIL low (< 0.4) → compute-bound", "FPU UTIL low + NOC BW UTIL high → bandwidth-bound (NoC)", "FPU UTIL low + NOC BW UTIL low (< 0.4) → overhead-bound (see Step 5)") restate three of the four rows of the table immediately above them in prose bullet form. Same information, two formats, two lines apart.
**Suggestion:** Delete lines 71–75. The table already covers all three cases. If a transition sentence is needed, use: "These combined readings feed directly into Step 4."

### index.md lines 5–6 (unresolved from Pass 1)
**Issue:** The Overview paragraph closes with "an op that is compute-bound will not benefit from reducing memory traffic, and an op that is bandwidth-bound will not benefit from algorithmic tricks that reduce FLOP count." This expands the word "prerequisite" from the preceding sentence without adding new information. The blockquote on line 9 then restates the core question a third time.
**Suggestion:** Delete the parenthetical expansion sentence. Keep the opening sentence and the blockquoted question; they are load-bearing.

### worked_examples.md line 3 (unresolved from Pass 1)
**Issue:** The preamble enumerates all five steps in order — "(1) theoretical AI, (2) `FPU UTIL`, (3) `NOC BW UTIL`, (4) `DEVICE KERNEL DURATION / PM IDEAL`, (5) TRISC breakdown" — which are also the section headings inside every example. The list previews content visible within the next three lines of each example.
**Suggestion:** Replace with: "Each example applies the five-step procedure from [`classification_method.md`](./classification_method.md)." Saves one line and removes redundant enumeration.

## Load-Bearing Evidence
- `roofline_model_primer.md` line 97: `"The matmul engine sustains one tile every 8 cycles when fully pipelined (2048 FLOPs ÷ 256 FLOPs/cycle = 8 cycles per tile), yielding a practical throughput ceiling of 128 FP16/BF16 FMA ops per cycle per core."` — the only place the 8-cycles-per-tile pipeline depth is derived; removing it would orphan the ridge-point calculation.
- `classification_method.md` line 57: `"Warning: High \`FPU UTIL\` does not automatically mean the kernel is fast."` — the only prose that disambiguates high FPU UTIL from high overall efficiency; cutting it would leave a common misreading unaddressed.
- `worked_examples.md` lines 216–217: `"The small matmul is the most instructive: its theoretical AI suggests compute-bound, but the actual CSV metrics reveal overhead-bound behavior because the tile count is too small to keep the pipeline fed. Always let the measured \`FPU UTIL\` and \`overhead_ratio\` override the theoretical prediction."` — the chapter's central pedagogical lesson, stated explicitly only in this summary caption.
- `index.md` line 44: `"> **Note:** These thresholds are starting points. Always cross-check against \`DEVICE KERNEL DURATION / PM IDEAL\` and the TRISC duration breakdown before drawing a final conclusion."` — the only caveat in the chapter index warning readers not to rely on the quick-reference table alone.

## VERDICT
- Crucial updates: no
