# Chapter 3 Change Log

## Fix 1: Corrected inverse frequency and wavelength table (multidimensional_rope_theory.md)

The inv_freq/wavelength table at lines 149-155 had wrong values for i>0. The entries appeared to use an incorrect denominator (~33.6) instead of spatial_dim=36. Recomputed all entries using `omega_i = 1 / 100^(2i/36)`:

| Index | Old omega | New omega | Old lambda | New lambda |
|-------|-----------|-----------|------------|------------|
| 0     | 1.000     | 1.000     | 6.3        | 6.3        |
| 4     | 0.278     | 0.359     | 22.6       | 17.5       |
| 8     | 0.077     | 0.129     | 81.3       | 48.7       |
| 12    | 0.021     | 0.046     | 293        | 135.4      |
| 17    | 0.004     | 0.013     | 1635       | 486.5      |

Also updated the prose reference to wavelength of i=17 from ~1635 to ~486.5.

## Fix 2: Corrected numerical example inv_freq values (reference_implementation.md)

The numerical example for patch position (5, 12) at lines 175-187 had wrong inv_freq values for i=1,2,3 and consequently wrong freqs_x and freqs_y. Recomputed using `inv_freq[i] = 1 / 100^(2i/36)`:

| Index | Old inv_freq | New inv_freq |
|-------|-------------|-------------|
| 1     | 0.760       | 0.774       |
| 2     | 0.578       | 0.599       |
| 3     | 0.439       | 0.464       |

Updated derived freqs_x (5 * omega) and freqs_y (12 * omega) accordingly.

## Fix 3: Corrected mistaken-theta lambda_max value (multidimensional_rope_theory.md)

The calculation of lambda_max with theta=10000 stated `2*pi * 10000^(34/36) ~ 48,700`. The correct value is `2*pi * 10000^(34/36) ~ 37,667`. Updated the number from 48,700 to 37,667.

## Fix 4: Corrected phi vector layout from interleaved to concatenated (multidimensional_rope_theory.md)

The phi vector at line 114 used an interleaved frequency-doubling pattern `[f0, f0, f1, f1, ...]` where each frequency appeared twice in adjacent positions. The actual `rotate_half` implementation constructs this vector via `torch.cat((freqs, freqs), dim=-1)`, which produces a concatenated layout `[f0, f1, ..., f_{d_s/2-1}, f0, f1, ..., f_{d_s/2-1}]`. Updated the formula and the accompanying prose explanation to reflect the correct concatenated pattern.

---

## Compression Analysis (Agent C)

### Verdict: Crucial updates: no

### File-by-File Assessment

#### 1. index.md

**Load-Bearing Evidence:** The Learning Objectives, Chapter Contents table, and Overview section all independently describe the same three-stage structure (theory, reference implementation, gap analysis) and the same core idea (split head_dim in half, apply RoPE per axis, concatenate). Lines 29-31 restate lines 7-9; lines 35-39 restate lines 21-25 in prose form.

**MINOR suggestion:** Remove the Overview section (lines 27-41) or collapse it to a single short paragraph. The Learning Objectives already tell the reader what they will learn, and the Chapter Contents table already maps the three stages. The Overview adds ~200 words that repeat both without new information. The forward-references to Chapter 6 and Chapter 7 (line 41) could move to a one-line note after the Chapter Contents table.

#### 2. multidimensional_rope_theory.md

**Load-Bearing Evidence:** The Concatenation of cos/sin section (lines 191-204) restates the 2D factored formula already given in "The Full 2D RoPE Formula" (lines 95-117). Lines 192-202 express the same concat(cos_x, cos_y) structure with concrete tensor shapes that are already derivable from the Concrete Numbers table (lines 131-139). The final sentence of that section ("This split-apply-concat pattern is the 'factored' aspect of the approach") repeats the explanation at lines 76-86.

**MINOR suggestion:** Fold the Concatenation section into the Concrete Numbers section as a short "Tensor shapes" sub-block (3-4 lines of shape annotations). The cos/sin shape information is useful, but the surrounding prose duplicates earlier derivations. This would save ~15 lines.

#### 3. reference_implementation.md

**Load-Bearing Evidence:** The Summary section (lines 334-341) is a three-bullet restatement of the Overview section (lines 7-11) from the same file. Both describe the identical three-stage pipeline (position IDs, cos/sin tables, application) in nearly the same words. The Dimensional Breakdown (lines 253-274) also visually recaps the split-apply-concat pattern that was already demonstrated line-by-line in the `apply_multidimensional_rope` code walkthrough immediately above it (lines 202-236).

**MINOR suggestion:** Remove the Summary section entirely -- readers reaching the end of the file have just read the detailed walkthrough and gain nothing from a three-sentence recap. The Dimensional Breakdown ASCII diagram (lines 257-274) could be trimmed to just the diagram itself (removing the prose preamble "For a single patch with head_dim=72:") since the code block above already sets that context.

#### 4. ttnn_rope_gap_analysis.md

**Load-Bearing Evidence:** The "Assumptions Baked Into Current Kernels" section (lines 17-25) restates information from the Capabilities table immediately above it (lines 9-15). For example, "1D position: Each token has a single integer position m" (line 19) restates "Accepts a scalar position index per sequence element (1D position)" from the table (line 14). Three of the four assumptions are already captured in the table.

**MINOR suggestion:** Merge the Assumptions list into the Capabilities table as an "Assumptions / Limitations" column, or remove it and let Gap 1/2/3 explain the mismatches (which they already do). This would eliminate ~10 lines of restatement. Additionally, the Recommendation prose (lines 273-301) could be shortened -- the Strategy Comparison table (lines 261-269) already conveys the ranking, and the recommendation text restates the table's data points (transfer overhead, op count, effort) before reaching the decision tree.

### Cross-File Redundancy

The "split head_dim into two halves, apply RoPE independently, concatenate" concept is explained from scratch in all four files:
- index.md lines 29-31
- multidimensional_rope_theory.md lines 76-117, 191-204
- reference_implementation.md lines 213-274, 334-339
- ttnn_rope_gap_analysis.md lines 37-45

The theory file owns this concept. The other three files could reference it with a one-line back-pointer rather than re-deriving or re-explaining it each time. Similarly, `rope_theta=100.0` rationale appears in both multidimensional_rope_theory.md (lines 169-189) and ttnn_rope_gap_analysis.md (lines 47-55); the gap analysis could simply reference the theory file's explanation.

### Estimated Savings

| File | Current lines | Saveable lines | Reduction |
|------|--------------|----------------|-----------|
| index.md | 42 | ~15 | ~36% |
| multidimensional_rope_theory.md | 220 | ~15 | ~7% |
| reference_implementation.md | 345 | ~20 | ~6% |
| ttnn_rope_gap_analysis.md | 319 | ~20 | ~6% |
| **Total** | **926** | **~70** | **~8%** |

Overall the chapter is well-structured and the redundancy is moderate. No content is incorrect or misleading -- the bloat is limited to repeated explanations of the same core concepts across files.
