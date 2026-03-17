# Compression Analysis: Chapter 3 Accuracy Analysis

## Summary
- Files analyzed: `index.md`, `accuracy_metrics_for_moe.md`, `projection_sensitivity.md`, `bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md`, `qwen_vs_deepseek_accuracy_comparison.md`
- Estimated current line count: 471
- Estimated post-compression line count: ~370
- Estimated reduction: ~21%

---

## CRUCIAL Suggestions

### C-1: PCC Thresholds Table — Three Near-Verbatim Copies

The per-dtype, per-projection PCC ranges appear as a table in three files:

- `index.md` lines 38–44 (Summary Table: Quantization Level vs. Expected Accuracy)
- `accuracy_metrics_for_moe.md` lines 47–53 (PCC Thresholds in Practice)
- `bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md` lines 20–28 (PCC Ranges by Dtype and Projection)

All three encode the same core facts: bfloat8_b gate/up 0.98–0.99, bfloat8_b down 0.975–0.985, bfloat4_b gate/up 0.96–0.98, bfloat4_b down 0.94–0.97. The `index.md` version also folds in perplexity delta and recommended use columns, making it the richest version.

**Canonical copy:** `accuracy_metrics_for_moe.md` — it is the dedicated metrics file and already carries the most complete explanation.
**Action:** Expand the `accuracy_metrics_for_moe.md` table to include the Perplexity Delta and Recommended Use columns from `index.md`. Remove the table from `bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md` and replace with a cross-reference: `See accuracy_metrics_for_moe.md § PCC Thresholds`. Reduce `index.md` table to a concise two-sentence reference summary with a link to `accuracy_metrics_for_moe.md`.
**Estimated savings:** ~18 lines.

---

### C-2: `torch.corrcoef` PCC Code Pattern — Three Near-Verbatim Copies

The same `torch.corrcoef(torch.stack([a_flat, b_flat]))[0, 1]` pattern appears inside full function bodies in three files:

- `accuracy_metrics_for_moe.md` lines 30–43 — `compute_pcc()` standalone helper, with example usage.
- `projection_sensitivity.md` lines 101–126 — `measure_projection_pcc()` inlines the corrcoef call at lines 123–125.
- `bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md` lines 62–95 — `pcc_for_dtype()` inlines the corrcoef call at lines 82–84.

The standalone `compute_pcc()` in `accuracy_metrics_for_moe.md` is the cleanest and most reusable definition. The other two files duplicate the pattern rather than calling the helper.

**Canonical copy:** `accuracy_metrics_for_moe.md` `compute_pcc()` (lines 30–43).
**Action:** In `projection_sensitivity.md` and `bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md`, replace the inline `torch.corrcoef` block with a call to `compute_pcc()` and add a one-line import comment: `# compute_pcc defined in accuracy_metrics_for_moe.md`. The surrounding function bodies (which test different dtypes/fidelity combos) are not duplicates and should be retained.
**Estimated savings:** ~10 lines.

---

### C-3: Cumulative Residual-Stream Error Explanation — Two Near-Verbatim Copies

Both of the following passages explain why per-layer PCC of 0.97 compounds and why residual additions accumulate additive (not multiplicative) error:

- `accuracy_metrics_for_moe.md` lines 76–83 (section "Why PCC Alone Is Insufficient")
- `bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md` lines 31–38 (section "Cumulative Error in Deep Models")

Both invoke the same 28-MoE-layer scenario and reach the same conclusion (ΔPPL > 2 PPL if per-layer PCC < 0.97 for down projection).

**Canonical copy:** `accuracy_metrics_for_moe.md` lines 76–83 — it contains the more rigorous explanation (residual additions accumulate additively, attention amplification caveat, dual validation protocol).
**Action:** In `bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md`, replace the "Cumulative Error in Deep Models" paragraph (lines 31–38) with a one-line cross-reference: `For the theory of cumulative residual-stream error, see accuracy_metrics_for_moe.md § Why PCC Alone Is Insufficient.` Keep the rule-of-thumb sentence ("if per-layer PCC drops below 0.97 for the down projection, expect ΔPPL > 2") as a single stand-alone callout because it is used as a practical decision anchor in that file.
**Estimated savings:** ~6 lines.

---

### C-4: DeepSeek-V3 Mixed-Precision Settings — Two Near-Verbatim Copies

The specific mixed-precision configuration (w1/w3 = bfloat4_b + LoFi, w2 = bfloat8_b + HiFi2, validated at PCC ~0.97) appears in:

- `projection_sensitivity.md` lines 88–97 (section "Empirical Evidence from DeepSeek-V3")
- `qwen_vs_deepseek_accuracy_comparison.md` lines 16–27 (section "DeepSeek-V3 Quantization Strategy", table rows and bullet list)

Both state the same three projections with the same dtypes and MathFidelity values and the same PCC 0.97 validation figure.

**Canonical copy:** `qwen_vs_deepseek_accuracy_comparison.md` — it is the file specifically dedicated to model-by-model detail and already provides the fuller context (QAT training rationale, K_t=64, architectural explanation).
**Action:** In `projection_sensitivity.md`, replace lines 88–97 with a two-sentence summary: the sensitivity ordering is empirically confirmed by DeepSeek-V3 (w2=bfloat8_b+HiFi2, w1/w3=bfloat4_b+LoFi), and point to `qwen_vs_deepseek_accuracy_comparison.md` for per-model detail.
**Estimated savings:** ~8 lines.

---

### C-5: MathFidelity Is Accumulation Passes, Not Mantissa Truncation — Two Near-Verbatim Copies

The clarification that MathFidelity controls accumulation passes (not dtype or mantissa truncation) and that dequantization always outputs bfloat16 appears in:

- `projection_sensitivity.md` lines 80–86 (Recommended Fidelity Settings table + prose block)
- `bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md` lines 14–16 (Dtype Encoding Summary, last two sentences)

The `projection_sensitivity.md` version is the authoritative, detailed explanation.

**Canonical copy:** `projection_sensitivity.md` lines 80–86.
**Action:** In `bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md`, shorten lines 14–16 to a single sentence with a cross-reference: `Compute accumulation precision is a separate concern controlled by MathFidelity; see projection_sensitivity.md § Recommended Fidelity Settings.`
**Estimated savings:** ~2 lines.

---

## MINOR Suggestions

### M-1: `index.md` Key Constants Block vs. Thresholds in `accuracy_metrics_for_moe.md`

`index.md` lines 59–64 define Python constants (`PCC_BFLOAT16_BASELINE`, `PCC_BFLOAT8B_MIN`, etc.) that shadow the prose thresholds in `accuracy_metrics_for_moe.md`. These are not verbatim duplicates (constants vs. table rows) but they encode the same numerical facts. Consider moving the constants block into `accuracy_metrics_for_moe.md` immediately after the PCC Thresholds table, and referencing them from `index.md` with a one-line note.

### M-2: Sensitivity Ordering Statement Repeated in Multiple Files

The sentence "down projection is most accuracy-sensitive; gate/up tolerate lower fidelity because SiLU acts as an error filter" appears as the core claim in `projection_sensitivity.md` (dedicated section) but is also restated in `index.md` Learning Objectives (line 17–18) and in `qwen_vs_deepseek_accuracy_comparison.md` (lines 34–36). The restatements in the other two files are brief and serve as orientation/summary context; they do not need removal but could be condensed to a single sentence with a link.

### M-3: Practical Decision Rules Overlap

`bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md` lines 98–103 (Practical Decision Rule) and `qwen_vs_deepseek_accuracy_comparison.md` lines 72–88 (Recommended Evaluation Procedure) both describe the same three-stage evaluation progression (all bfloat8_b → mixed → all bfloat4_b). The latter is more complete (includes pseudocode). Consider consolidating or cross-linking.

### M-4: SwiGLU Structure Diagram Mentioned in Two Files

`projection_sensitivity.md` lines 5–15 provides the SwiGLU code block with variable assignments. `qwen_vs_deepseek_accuracy_comparison.md` references gate/up/down without re-printing the code, which is the correct pattern. No action required beyond noting it is handled well.

---

## Load-Bearing Evidence

The following facts must not be removed from any consolidation pass. File and line references are to the canonical locations.

| Fact | Canonical Location |
|---|---|
| PCC formula: `torch.corrcoef(torch.stack([a_flat, b_flat]))[0, 1]` | `accuracy_metrics_for_moe.md` line 35 |
| bfloat8_b gate/up PCC range 0.98–0.99 | `accuracy_metrics_for_moe.md` line 51; `index.md` line 41 |
| bfloat8_b down PCC range 0.975–0.985 | `accuracy_metrics_for_moe.md` line 52; `index.md` line 42 |
| bfloat4_b gate/up PCC range 0.96–0.98 | `accuracy_metrics_for_moe.md` line 53; `index.md` line 43 |
| bfloat4_b down PCC range 0.94–0.97 | `accuracy_metrics_for_moe.md` line 54; `index.md` line 44 |
| Perplexity delta: bfloat8_b < 1 PPL | `accuracy_metrics_for_moe.md` line 63 |
| Perplexity delta: bfloat4_b (all projections) 1–3 PPL | `accuracy_metrics_for_moe.md` line 65 |
| Sensitivity ordering: down > gate > up | `projection_sensitivity.md` line 21 |
| SiLU as error filter for gate/up projections | `projection_sensitivity.md` lines 56–64 |
| MathFidelity = accumulation passes, not mantissa truncation | `projection_sensitivity.md` lines 80–86 |
| DeepSeek-V3 expert FFN down projection K_t = 64 | `qwen_vs_deepseek_accuracy_comparison.md` line 31 |

---

## VERDICT: Crucial updates: yes

---

## Agent A Change Log — C Feedback Pass 1

**C-1 (PCC Thresholds Table, 3 copies → 1 canonical):**
- Expand `accuracy_metrics_for_moe.md` PCC Thresholds table to include Perplexity Delta and Recommended Use columns (merge from `index.md` lines 38–44).
- Remove the 9-row table from `bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md` (lines 20–28); replace with cross-reference sentence.
- Condense `index.md` summary table to a 2-sentence overview referencing `accuracy_metrics_for_moe.md`.
- Estimated savings: ~18 lines.

**C-2 (corrcoef pattern, 3 copies → 1 canonical):**
- Keep `compute_pcc()` in `accuracy_metrics_for_moe.md` (lines 30–43) as the single definition.
- In `projection_sensitivity.md` lines 123–125 and `bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md` lines 82–84, replace inline `torch.corrcoef` block with `return compute_pcc(ref_out, out)` and an import comment.
- Estimated savings: ~10 lines.

**C-3 (cumulative residual error, 2 copies → 1 canonical):**
- Keep `accuracy_metrics_for_moe.md` lines 76–83 (full explanation) as canonical.
- In `bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md` lines 31–38, replace with cross-reference + one stand-alone rule-of-thumb sentence.
- Estimated savings: ~6 lines.

**C-4 (DeepSeek-V3 mixed-precision settings, 2 copies → 1 canonical):**
- Keep `qwen_vs_deepseek_accuracy_comparison.md` lines 16–27 as canonical.
- In `projection_sensitivity.md` lines 88–97, replace with 2-sentence summary + cross-reference.
- Estimated savings: ~8 lines.

**C-5 (MathFidelity clarification, 2 copies → 1 canonical):**
- Keep `projection_sensitivity.md` lines 80–86 as canonical.
- In `bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md` lines 14–16, replace with one cross-reference sentence.
- Estimated savings: ~2 lines.

**Total estimated savings from CRUCIAL items: ~44 lines (~9% of 471).**
(Post-compression estimate of ~370 lines also accounts for M-1 and M-3 minor condensations of ~12 lines.)

## Agent A Change Log — C Feedback Pass 1
- index.md: Replaced PCC thresholds table with cross-reference to accuracy_metrics_for_moe.md
- bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md: Replaced PCC thresholds table with cross-reference
- projection_sensitivity.md + bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md: Replaced compute_pcc() re-definitions with cross-references to accuracy_metrics_for_moe.md
- bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md: Replaced cumulative error explanation with cross-reference to accuracy_metrics_for_moe.md
- projection_sensitivity.md: Replaced DeepSeek mixed-precision block with cross-reference to qwen_vs_deepseek_accuracy_comparison.md
- bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md: Replaced MathFidelity passes clarification with cross-reference to projection_sensitivity.md

---

# Compression Analysis: Chapter 3 Accuracy Analysis — Pass 2

## Summary
- Pass 1 fixes: all 5 verified correctly applied
- Current line count after Pass 1: 455 (index.md 62 + accuracy_metrics_for_moe.md 83 + projection_sensitivity.md 121 + bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md 94 + qwen_vs_deepseek_accuracy_comparison.md 95)
- New crucial duplications: none

## CRUCIAL Suggestions
None

## MINOR Suggestions

### M-3 (carry-forward): Practical Decision Rule vs. Recommended Evaluation Procedure

`bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md` lines 88–94 (Practical Decision Rule, 3-bullet list) and `qwen_vs_deepseek_accuracy_comparison.md` lines 70–88 (Recommended Evaluation Procedure, 3-stage list + pseudocode) both encode the same progression: all bfloat8_b → bfloat4_b gate/up + bfloat8_b down → all bfloat4_b. The `qwen_vs_deepseek_accuracy_comparison.md` version is more complete (includes pseudocode and explicit per-model verdict). Consider replacing the 3-bullet block in `bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md` with a single cross-reference sentence pointing to `qwen_vs_deepseek_accuracy_comparison.md § Recommended Evaluation Procedure`. Estimated savings: ~5 lines.

## Load-Bearing Evidence

All load-bearing items confirmed present at canonical locations:

| Fact | Canonical Location | Status |
|---|---|---|
| `compute_pcc()` definition with `torch.corrcoef` | `accuracy_metrics_for_moe.md` lines 30–43 | PRESENT |
| PCC threshold ranges (bfloat8_b gate/up 0.98–0.99, down 0.975–0.985; bfloat4_b gate/up 0.96–0.98, down 0.94–0.97) | `accuracy_metrics_for_moe.md` lines 47–53 | PRESENT |
| Perplexity delta bfloat8_b < 1 PPL | `accuracy_metrics_for_moe.md` line 63 | PRESENT |
| Perplexity delta bfloat4_b all projections 1–3 PPL | `accuracy_metrics_for_moe.md` line 65 | PRESENT |
| Sensitivity ordering: down > gate > up | `projection_sensitivity.md` line 20 | PRESENT |
| SiLU as error filter for gate/up | `projection_sensitivity.md` lines 56–64 | PRESENT |
| DeepSeek-V3 full strategy (w2=bfloat8_b+HiFi2, w1/w3=bfloat4_b+LoFi, K_t=64, QAT rationale) | `qwen_vs_deepseek_accuracy_comparison.md` lines 16–36 | PRESENT |

### Pass 1 Fix Verification Detail

- **C-1**: `index.md` lines 38–42 — 4-sentence cross-reference to `accuracy_metrics_for_moe.md § PCC Thresholds in Practice`; no table. `bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md` lines 19–22 — single cross-reference sentence; no table. Canonical table intact in `accuracy_metrics_for_moe.md`. VERIFIED.
- **C-2**: `projection_sensitivity.md` lines 119–120 — `# compute_pcc defined in accuracy_metrics_for_moe.md` + `return compute_pcc(ref_out, out)`; no inline corrcoef. `bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md` lines 74–75 — same comment + `return compute_pcc(ref, out)`; no inline corrcoef. Canonical `compute_pcc()` intact in `accuracy_metrics_for_moe.md`. VERIFIED.
- **C-3**: `bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md` lines 24–30 — cross-reference sentence to `accuracy_metrics_for_moe.md § Why PCC Alone Is Insufficient` plus retained stand-alone rule-of-thumb (PCC < 0.97 → ΔPPL > 2). Canonical explanation intact in `accuracy_metrics_for_moe.md` lines 76–83. VERIFIED.
- **C-4**: `projection_sensitivity.md` lines 88–93 — 2-sentence summary (ordering confirmed by DeepSeek-V3, w2=bfloat8_b+HiFi2, w1/w3=bfloat4_b+LoFi, PCC ~0.97) with explicit cross-reference to `qwen_vs_deepseek_accuracy_comparison.md § DeepSeek-V3 Quantization Strategy`. Full canonical block intact in `qwen_vs_deepseek_accuracy_comparison.md`. VERIFIED.
- **C-5**: `bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md` lines 15–17 — single cross-reference sentence to `projection_sensitivity.md § Recommended Fidelity Settings`. Canonical MathFidelity explanation intact in `projection_sensitivity.md` lines 80–86. VERIFIED.

## VERDICT: Crucial updates: no
