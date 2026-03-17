# Compression Analysis: Chapter 4 math_approx_mode

## Summary
- Files analyzed: `index.md`, `sfpu_approx_operations.md`, `approx_mode_accuracy_risks.md`, `approx_mode_for_moe.md`
- Estimated current line count: 64 + 92 + 97 + 105 = 358
- Estimated post-compression line count: ~260
- Estimated reduction: ~27%

---

## CRUCIAL Suggestions

### C-1: `COMPUTE_KERNEL_CONFIG_LOFI` code block duplicated in 3 files

**Files and approximate lines:**
- `index.md` lines 39–57 (both LOFI and HIFI2 blocks together)
- `sfpu_approx_operations.md` lines 71–78 (LOFI block) and lines 84–91 (HIFI2 block)
- `approx_mode_for_moe.md` lines 18–27 (as `COMPUTE_KERNEL_CONFIG_GATE`), lines 33–42 (as `COMPUTE_KERNEL_CONFIG_UP`), lines 49–58 (as `COMPUTE_KERNEL_CONFIG_DOWN`)

**Description:** The `COMPUTE_KERNEL_CONFIG_LOFI` and `COMPUTE_KERNEL_CONFIG_HIFI2` Python blocks are reproduced verbatim or with only comment/name changes across all four files. `index.md` already presents them together as a "Key Config Reference" section. `sfpu_approx_operations.md` re-declares them in full to explain why each flag is set. `approx_mode_for_moe.md` re-declares them a third time under projection-specific names (`COMPUTE_KERNEL_CONFIG_GATE`, `COMPUTE_KERNEL_CONFIG_UP`, `COMPUTE_KERNEL_CONFIG_DOWN`) — which are structurally identical to LOFI and HIFI2.

**Canonical location:** `index.md` lines 39–57. This is the designated "Key Config Reference" section and the natural single source of truth for these two config objects.

**Action:**
- `sfpu_approx_operations.md` sections "Why COMPUTE_KERNEL_CONFIG_LOFI Sets math_approx_mode=False" (lines 67–78) and "Why COMPUTE_KERNEL_CONFIG_HIFI2 Sets math_approx_mode=True" (lines 80–91): remove the full code blocks; replace with a prose sentence and a cross-reference to `index.md` Key Config Reference. The explanatory rationale prose in these two sections is non-redundant and should be kept.
- `approx_mode_for_moe.md` projection subsections (lines 18–27, 33–42, 49–58): the per-projection named configs (`COMPUTE_KERNEL_CONFIG_GATE`, `COMPUTE_KERNEL_CONFIG_UP`, `COMPUTE_KERNEL_CONFIG_DOWN`) duplicate LOFI and HIFI2 exactly except for variable names. Consolidate to a single inline example or reference the canonical names from `index.md`. The summary table (lines 63–69) and rationale prose are non-redundant and must be kept.

**Estimated savings:** ~35 lines of code blocks removed from `sfpu_approx_operations.md`; ~35 lines from `approx_mode_for_moe.md` (retaining rationale prose). Total: ~70 lines.

---

### C-2: SFPU operations table duplicated in `index.md` and `sfpu_approx_operations.md`

**Files and approximate lines:**
- `index.md` lines 24–36 ("Affected vs. Unaffected Operations" table)
- `sfpu_approx_operations.md` lines 16–26 ("Operations Routed Through the SFPU" table)

**Description:** Both tables list the same six SFPU ops (`exp`, `reciprocal`, `sqrt`, `sigmoid`, `gelu`, `silu`) and the same three FPU ops (`matmul`, `linear`, `dot_product`), with the same affected/not-affected split. `sfpu_approx_operations.md` adds an "Approx Error" column (~0.1–0.2% figures), but the core structure and all op names are identical. `index.md` is the chapter overview; it does not need a full op table when `sfpu_approx_operations.md` already provides the authoritative detailed version.

**Canonical location:** `sfpu_approx_operations.md` lines 16–26 (the detailed table with error column is more informative).

**Action:** In `index.md`, replace the full "Affected vs. Unaffected Operations" table (lines 24–36) with a short prose summary (two sentences stating SFPU ops are affected, FPU matmul is not) and a forward reference to `sfpu_approx_operations.md`. The learning objectives and chapter structure sections already give the reader a roadmap; the full table is redundant at the index level.

**Estimated savings:** ~14 lines from `index.md`.

---

### C-3: "math_approx_mode has zero effect on matmul" explanation repeated in 3 files

**Files and approximate lines:**
- `index.md` line 5 (overview paragraph)
- `sfpu_approx_operations.md` lines 50–65 ("Operations NOT Affected by math_approx_mode" section with verification code block)
- `approx_mode_accuracy_risks.md` lines 56–81 ("When Approximation Mode Is Irrelevant" section with full verification code block)

**Description:** The claim that setting `math_approx_mode=True/False` on a pure matmul produces bit-identical outputs is stated in the overview and then demonstrated twice with nearly identical verification code blocks — once in `sfpu_approx_operations.md` and once in `approx_mode_accuracy_risks.md`. Both verification blocks define `cfg_approx` and `cfg_exact` with identical fields and assert bit-identity.

**Canonical location:** `sfpu_approx_operations.md` lines 50–65. This is the architecture file; explaining what the flag does (and does not do) to the FPU fits there. The accuracy file should reference that explanation rather than repeat it.

**Action:** In `approx_mode_accuracy_risks.md`, remove the "When Approximation Mode Is Irrelevant" section (lines 56–81) entirely, or reduce it to a single sentence with a cross-reference to `sfpu_approx_operations.md`. The verification code block adds no new information relative to the one already in `sfpu_approx_operations.md`.

**Estimated savings:** ~25 lines from `approx_mode_accuracy_risks.md`.

---

## MINOR Suggestions

### M-1: "math_approx_mode affects SFPU only, not FPU" restated in preambles

`index.md` line 5, `sfpu_approx_operations.md` lines 7–10, and `approx_mode_for_moe.md` lines 29 and 44 each restate that `math_approx_mode` touches SFPU only and has no effect on matmul. After CRUCIAL consolidations, this will naturally thin out. No separate action needed beyond the C-3 fix; ensure at least one full statement survives in `sfpu_approx_operations.md`.

### M-2: SiLU-in-MoE-FFN safety repeated across two files

`approx_mode_accuracy_risks.md` lines 35–54 (prose + code showing SiLU is safe, bounded per-element error) and `approx_mode_for_moe.md` lines 44, 56, 60 (rationale prose for up and down projections) both explain that SiLU approximation error does not accumulate across sequence length. The accuracy file has the full derivation; the MoE file should reference it rather than re-derive. Mild overlap — the MoE file's phrasing is distinct enough to be tolerable, but consider reducing the MoE file's rationale prose for gate/up to one sentence plus a cross-reference.

### M-3: "Practical Guidance / when in doubt" section in `approx_mode_for_moe.md`

`approx_mode_for_moe.md` lines 84–104 introduces a `COMPUTE_KERNEL_CONFIG_SAFE` object and re-ranks `math_approx_mode` as the least important accuracy lever. This ranking information is useful but is partly redundant with the PCC table in `approx_mode_accuracy_risks.md` lines 83–96. Consider moving the ranking prose into `approx_mode_accuracy_risks.md` and keeping only the `COMPUTE_KERNEL_CONFIG_SAFE` snippet in `approx_mode_for_moe.md` (or vice versa). Low-priority consolidation.

---

## Load-Bearing Evidence

The following facts are confirmed present in the chapter and must not be removed during consolidation:

- `math_approx_mode` affects SFPU only; it has no effect on FPU matmul or dot-product operations.
- SFPU ops subject to `math_approx_mode`: `exp`, `reciprocal`, `sqrt`, `sigmoid`, `gelu`, `silu`.
- Approximation mechanism: piecewise polynomial lookup (degree 2–3); error ~0.1–0.3% per element depending on op.
- `COMPUTE_KERNEL_CONFIG_LOFI`: `math_fidelity=LoFi`, `math_approx_mode=False`.
- `COMPUTE_KERNEL_CONFIG_HIFI2`: `math_fidelity=HiFi2`, `math_approx_mode=True`.
- Gate (w1) and Up (w3) projections: `math_approx_mode=False` (SFPU not exercised; conservative default).
- Down (w2) projection: `math_approx_mode=True` with HiFi2 fidelity.
- Softmax at K >= 16K: use `math_approx_mode=False` to avoid exp accumulation error in denominator.
- SiLU in MoE FFN: `math_approx_mode=True` is safe; error is bounded per-element with no reduction accumulation.
- PCC delta < 0.0001 with HiFi2 at typical input scales for MoE FFN path.

---

## VERDICT: Crucial updates: yes

---

## Agent A Change Log — C Feedback Pass 1

| ID | File | Action | Lines affected (approx) | Priority |
|---|---|---|---|---|
| C-1a | `sfpu_approx_operations.md` | Remove full code blocks from "Why COMPUTE_KERNEL_CONFIG_LOFI" (lines 71–78) and "Why COMPUTE_KERNEL_CONFIG_HIFI2" (lines 84–91) sections; replace each with one cross-reference sentence to `index.md` Key Config Reference. Keep rationale prose. | −20 lines | High |
| C-1b | `approx_mode_for_moe.md` | Consolidate per-projection named config blocks (`COMPUTE_KERNEL_CONFIG_GATE`, `COMPUTE_KERNEL_CONFIG_UP`, `COMPUTE_KERNEL_CONFIG_DOWN` at lines 18–27, 33–42, 49–58) to references to the canonical `COMPUTE_KERNEL_CONFIG_LOFI` / `COMPUTE_KERNEL_CONFIG_HIFI2` names from `index.md`. Keep all rationale prose and the summary table. | −35 lines | High |
| C-2 | `index.md` | Replace "Affected vs. Unaffected Operations" table (lines 24–36) with a 2-sentence prose summary plus forward reference to `sfpu_approx_operations.md`. | −14 lines | High |
| C-3 | `approx_mode_accuracy_risks.md` | Remove "When Approximation Mode Is Irrelevant" section (lines 56–81) or reduce to 1 sentence + cross-reference to `sfpu_approx_operations.md`. | −25 lines | High |
| M-2 | `approx_mode_for_moe.md` | Trim gate/up rationale prose for SiLU safety to 1 sentence + cross-reference to `approx_mode_accuracy_risks.md` "When Approximation Mode Is Safe" section. | −6 lines | Low |
| M-3 | `approx_mode_for_moe.md` or `approx_mode_accuracy_risks.md` | Move accuracy-lever ranking prose to accuracy file; keep only `COMPUTE_KERNEL_CONFIG_SAFE` snippet in MoE file. | −5 lines | Low |

---

# Compression Analysis: Chapter 4 math_approx_mode — Pass 2

## Summary
- Pass 1 fixes: all 3 CRUCIAL fixes (C-1, C-2, C-3) verified correctly applied
- Current line count after Pass 1: index.md (56) + sfpu_approx_operations.md (78) + approx_mode_accuracy_risks.md (76) + approx_mode_for_moe.md (77) = 287 lines (vs. estimated 358 pre-compression; actual reduction ~20%)
- New crucial duplications: none

## CRUCIAL Suggestions
None

## MINOR Suggestions

### M-2 (carry-forward, not yet applied)
`approx_mode_for_moe.md` lines 26 and 32 still carry full rationale sentences for why SiLU approximation error is bounded. `approx_mode_accuracy_risks.md` § When Approximation Mode Is Safe already provides the authoritative derivation (lines 33–54). Consider trimming each of the two gate/up rationale sentences in `approx_mode_for_moe.md` to a single clause plus a cross-reference. Low impact (~4 lines).

### M-3 (carry-forward, not yet applied)
`approx_mode_for_moe.md` § Practical Guidance (lines 56–76) contains the accuracy-lever ranking prose and `COMPUTE_KERNEL_CONFIG_SAFE`. This ranking partially overlaps the PCC table in `approx_mode_accuracy_risks.md` lines 62–75. No action required for correctness; consolidation optional.

### M-4 (new, minor)
`approx_mode_for_moe.md` lines 46–53 (Qwen MoE forward pseudocode) and `approx_mode_accuracy_risks.md` lines 39–52 (SiLU-in-FFN safety pseudocode) both illustrate the SwiGLU pattern. The overlap is conceptual only — the former shows architecture, the latter proves per-element error bounding. No action required; note for future consolidation if a dedicated "MoE FFN structure" reference section is added.

## Load-Bearing Evidence

All load-bearing items confirmed present:

- SFPU-only scope (not FPU matmul): confirmed — `index.md` line 5, `sfpu_approx_operations.md` line 10
- The 6 SFPU ops (exp, reciprocal, sqrt, sigmoid, gelu, silu): confirmed — `sfpu_approx_operations.md` lines 18–26 table; also summarized in prose at `index.md` line 25
- ~0.1–0.3% per-evaluation error: confirmed — `sfpu_approx_operations.md` table Approx Error column; `index.md` line 25 prose
- `COMPUTE_KERNEL_CONFIG_LOFI`: `math_approx_mode=False`: confirmed canonical in `index.md` lines 35–40
- `COMPUTE_KERNEL_CONFIG_HIFI2`: `math_approx_mode=True`: confirmed canonical in `index.md` lines 43–48
- Gate/Up=False, Down=True with HiFi2: confirmed — `approx_mode_for_moe.md` summary table lines 36–40 and prose sections lines 16–32
- K>=16K softmax threshold: confirmed — `approx_mode_accuracy_risks.md` line 5 (section heading) and line 17 (recommendation)
- SiLU in MoE FFN is safe: confirmed — `approx_mode_accuracy_risks.md` § When Approximation Mode Is Safe, lines 33–54
- PCC delta < 0.0001 with HiFi2: confirmed — `approx_mode_accuracy_risks.md` line 68 (PCC table, "Random inputs, typical scale" row)

## VERDICT: Crucial updates: no
