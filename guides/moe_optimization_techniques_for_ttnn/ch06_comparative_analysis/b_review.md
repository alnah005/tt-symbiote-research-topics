# B Review — MoE Optimization Chapter 6: Comparative Analysis (Post-Fix Re-Review)

## Verdict: APPROVED with one residual minor note

No critical issues remain. The two targeted fixes were applied correctly and verified. One pre-existing minor inconsistency in the Scenario 3 narrative is noted below but does not affect correctness of results.

---

## Fix Verification

### Fix 1: `performance_comparison_matrix.md` — Scenario table capacity expressions

All four table expressions were corrected from denominator 256 (= E) to denominator 32 (= E/k):

| Scenario | Expression in file | Arithmetic check | Result |
|----------|--------------------|-----------------|--------|
| 1: Prefill large (B=32, S=2048) | `⌈65,536/32⌉ = 2,048` | 65,536 / 32 = 2,048 | Correct |
| 2: Prefill small (B=1, S=2048)  | `⌈2,048/32⌉ = 64`     | 2,048 / 32 = 64      | Correct |
| 3: Decode large (B=32, S=1)     | `⌈32/32⌉ = 1`         | 32 / 32 = 1          | Correct |
| 4: Decode small (B=1, S=1)      | `⌈1/32⌉ = 1`          | ceil(0.03125) = 1    | Correct |

These now agree with the ground truth formula C = ceil(k×B×S / E) = ceil(B×S / 32) (using k=8, E=256 → denominator E/k = 32). The table expressions are internally consistent with the formula displayed at the top of the section.

### Fix 2: `index.md` line 98 — Decode large batch utilization

The summary table entry for Decode, Large Batch (B ≥ 32) was changed from "Moderate" to "High ($\rho \approx 1.0$)". Verified correct: at B=32, S=1 there are 32 × 8 = 256 expert assignments distributed over 256 experts, so under uniform routing nearly every expert receives exactly 1 token and ρ ≈ 1.0. "High" is the correct characterization and is consistent with the detailed Scenario 3 analysis in `performance_comparison_matrix.md`.

---

## Residual Minor Note (pre-existing, not part of applied fixes)

- **`performance_comparison_matrix.md` Scenario 3 narrative (line 44):** The prose states "expert-grouped capacity is $C = \lceil 32/256 \rceil = 1$ token/expert." The expression `⌈32/256⌉` evaluates to ceil(0.125) = 1, which gives the numerically correct answer of 1 but uses the wrong denominator (256 = E instead of 32 = E/k). The correct simplified expression is `⌈32/32⌉ = 1`. This is a cosmetic inconsistency — the final result is correct and the surrounding analysis is unaffected — but it is inconsistent with the corrected table and the formula derivation at the top of the file. This was not part of the four applied fixes and does not affect any conclusions.

---

## Confirmed Correct (all files)

### `index.md`

- Model parameters table (E=256, k=8, H=7168, N=8, E_d=32): all correct.
- ρ = k/E = 8/256 ≈ 3.1% at decode B=1: correctly stated and explained.
- Decision flowchart thresholds (ρ < 0.1 → sparse; 0.1 ≤ ρ ≤ 0.5 → profile; ρ > 0.5 → batched): match ground truth exactly.
- Summary table (line 98) decode large batch utilization now reads "High (ρ ≈ 1.0)": correct after fix.

### `performance_comparison_matrix.md`

- Expert capacity formula derivation (lines 13–17): C = ceil(k×B×S/E) = ceil(B×S/32) correctly shown.
- Scenario table capacity expressions (lines 29–32): all four corrected and verified (see Fix 1 above).
- Scenario 1 narrative: C = 2,048; average assignments = 524,288/256 = 2,048; ρ ≈ 1.0; batched matmul recommended. Correct.
- Scenario 2 narrative: C = 64 tokens/expert; all 256 experts active; ρ ≈ 1.0; batched matmul preferred. Correct.
- Scenario 3 narrative: 32×8 = 256 assignments over 256 experts; ρ ≈ 1.0; batched matmul preferred; borderline note included. Result correct (see residual note above for the cosmetic expression issue).
- Scenario 4 narrative: C = 1; 8 of 256 experts active; 248 experts (96.9%) idle; ρ = 8/256 = 3.1%; sparse matmul strongly preferred. Correct.
- Tiles along H = ceil(7168/32) = 224. Correct.
- 248 skipped experts × 224 tiles = 55,552 skipped tile operations. Correct.
- Non-monotonic latency analysis: crossover at ρ ≈ 0.5, not ρ = 1.0; metadata floor is non-zero; all qualitative and quantitative claims correct.

### `memory_and_bandwidth_tradeoffs.md`

- Decode B=1, S=1: C = ceil(8×1×1/256) = 1; tensor [256,1,7168]; total elements = 256×7168 = 1,835,008; active = 8×7168 = 57,344; useful fraction = 57,344/1,835,008 = 3.1%; DRAM waste = 96.9%. All arithmetic correct.
- Sparse matmul at decode B=1: 8 active experts × 224 tiles = 1,792 active tiles; 248×224 = 55,552 skipped tiles; activation reads = 57,344 elements. Correct.
- "32× fewer activation bytes" claim: ratio = 256/8 = 32. Correct.
- Sparsity tensor at decode B=1: M_t = ceil(1/32) = 1; shape [32,224]; size = 7,168 bytes ≈ 7 KB. Correct.
- Sparsity tensor at prefill B=32, S=2048: M_t = ceil(2048/32) = 64; shape [2048,224]; size = 458,752 bytes ≈ 448 KB. Correct.
- Per-device T3K decode B=1: gather layout [32,1,7168]; 31 of 32 rows zero-padded = 96.9% waste. Correct.
- DRAM bandwidth figure "≈300 GB/s" is flagged [UNVERIFIED] in the document; this is an honest acknowledgment and not a hidden error.

### `decision_guide.md`

- `measure_sparsity_ratio`: rho = active_experts / E. For B=1, k=8: returns 8/256 = 0.03125. Logic correct.
- `get_matmul_strategy`: rho > 0.5 → 'batched'; not is_decode → 'batched'; else → 'sparse'. Matches ground truth decision rules exactly.
- Rule 2 sparsity ratio thresholds (ρ < 0.1, 0.1–0.5, ρ > 0.5): match ground truth.
- Anti-patterns 1–4: all correctly described and consistent with quantitative analysis.
- Minor pre-existing note: the formula `ρ ≤ k × 8 / (E × C)` in Rule 3 (line 54) is non-standard and the arithmetic does not tightly support the stated "rough alignment with ρ < 0.1." This is a qualitative remark, not a critical error, and was not in scope for the applied fixes.
