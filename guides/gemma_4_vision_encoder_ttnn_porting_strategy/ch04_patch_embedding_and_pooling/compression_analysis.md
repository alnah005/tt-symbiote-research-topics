# Change Log for adaptive_pooling_port.md

## 2026-04-03

### Fix 1: Gemma 3 pooling kernel size in comparison table
- **Location:** Comparison table, "Kernel size" row
- **Was:** "Derived from fixed grid (8x8 blocks)"
- **Now:** "Derived from fixed grid (4x4 blocks)"
- **Rationale:** Gemma 3 uses a 64x64 patch grid pooled to a 16x16 output, giving kernel_size = 64/16 = 4, not 8.

### Fix 2: sqrt(1152) scaling rationale corrected
- **Location:** Step 4 (Scaling) prose, and "What Changed and Why" item 3
- **Was:** Incorrectly attributed sqrt(1152) scaling to compensating for the magnitude reduction caused by averaging 9 patches (claiming ~1/sqrt(9) magnitude loss).
- **Now:** Explains sqrt(1152) as standard hidden-dimension scaling (the same sqrt(d) convention used throughout the model, e.g., in attention layers), keeping magnitudes consistent for downstream RMSNorm and linear projection.
- **Rationale:** sqrt(1152) equals sqrt(hidden_size), not a function of the patch count (9). The scaling follows the same convention applied elsewhere in the model.

### Fix 3: Corrected claim that aspect ratio changes output token count
- **Location:** "What Changed and Why" item 2 (around line 151)
- **Was:** Claimed "the actual number of valid output tokens depends on the image's aspect ratio" and that different aspect ratios "produce different numbers of valid tokens."
- **Now:** Explains that the divisibility-by-48 constraint ensures (h/48)*(w/48) always equals the token budget exactly, so different aspect ratios change the grid shape (e.g., wider vs. taller) but NOT the total output token count.
- **Rationale:** The variable-resolution design enforces that resized dimensions are divisible by 48, guaranteeing the product of grid dimensions matches the token budget. The pooler must handle variable grid geometries, but the token count itself is fixed.

### Fix 4: Removed invalid near-square patch grid entries from token budget table
- **Location:** `patch_embedding_port.md`, token budget table (lines 205-209)
- **Was:** Each row included a third "near-square" example grid (27x24, 36x36, 51x51, 72x72, 102x102) that did not yield the correct token budget after 3x3 pooling (e.g., 27x24 gives 9x8=72 tokens, not 70; 36x36 gives 12x12=144, not 140).
- **Now:** Removed the invalid near-square entries. Each row retains only the two valid landscape/portrait grid shapes (e.g., 21x30 and 30x21 for the 70-token budget) whose pooled dimensions correctly multiply to the token budget.
- **Rationale:** Patch grid dimensions must be divisible by 3 (since image dims are divisible by 48 and patch size is 16, giving 48/16=3). The near-square entries either did not satisfy this constraint or, when they did, produced pooled token counts that did not match the budget (e.g., 72 instead of 70). No near-square factorization of these token budgets into two multiples of 3 exists for 70, so the entries were removed rather than replaced with incorrect values.

---

# Compression Analysis — Agent C

## Date: 2026-04-03

## Verdict

**Crucial updates: no**

All three files are well-structured and contain load-bearing technical content. The redundancy found is minor and does not warrant urgent revision.

---

## File-by-File Analysis

### 1. `index.md`

**Load-Bearing Evidence:** The "Why These Are Critical Path Items" section (three numbered reasons, lines 32-38) provides the strategic justification for prioritizing these modules. This rationale does not appear in the subchapter files and is the primary content unique to the index page.

**Minor suggestion — trim "Why These Are Critical Path Items" prose overlap with Overview paragraph.**
Lines 28-29 (Overview paragraph) already state that the embedder and pooler "diverge most from the Gemma 3 SigLIP architecture" and "must handle variable spatial dimensions." Reason 1 ("No direct Gemma 3 equivalent to copy") and Reason 2 ("Variable spatial dimensions affect the entire pipeline") restate these same two points with only slightly more detail. Consider condensing the Overview paragraph to a single sentence and letting the numbered list carry the full argument, or vice versa, removing the numbered reasons whose content is already covered by the Overview and the subchapter files.

Estimated savings: ~4 lines.

---

### 2. `patch_embedding_port.md`

**Load-Bearing Evidence:** The "TTNN Implementation Plan" section (lines 106-131) provides the concrete `ttnn.embedding` lookup strategy with padding-index clamping and masking. This is the actionable porting guidance that cannot be removed.

**Minor suggestion — remove duplicate value-scaling fusion tip.**
The idea of fusing the `2x - 1` scaling into the linear weight is mentioned twice:
- As an inline code comment in the full forward pass (lines 171-172): `"# Can be fused with the linear projection by pre-scaling the weight matrix: ..."`
- As a standalone Tip block (line 193): `"The value scaling 2 * x - 1 can be fused into the linear projection weight to save two element-wise ops..."`

The standalone Tip is the more detailed version. The inline comment should be shortened to a brief reference, e.g., `# Value scaling (see fusion Tip below)`, to avoid repeating the same optimization rationale.

Estimated savings: ~3 lines.

---

### 3. `adaptive_pooling_port.md`

**Load-Bearing Evidence:** The three TTNN implementation options (lines 159-293) with their trade-off analysis (advantages/challenges per option) are the core porting guidance. This content is unique and cannot be cut.

**Minor suggestion — condense "What Changed and Why" subsection.**
The "What Changed and Why" subsection (lines 145-153) has three items that largely narrate the comparison table directly above (lines 133-143). Specifically:
- Item 1 ("Variable grid shapes require explicit cell assignment") restates the "Spatial awareness" and "Input patches" rows.
- Item 3 ("The sqrt(d) scaling follows the standard hidden-dimension convention") restates the "Post-pooling scaling" row.

These could be collapsed into a single brief paragraph noting that the comparison table above captures the differences, with only Item 2 (the aspect-ratio/token-budget interaction) retained as a standalone callout since it adds analytical depth beyond what the table conveys.

Estimated savings: ~6 lines.

---

## Cross-File Redundancy

**Minor observation:** The token-budget-to-patch-count table appears in both `patch_embedding_port.md` (lines 203-209) and `adaptive_pooling_port.md` (lines 103-109). Both tables serve their local context (patch count variation vs. patches-per-cell uniformity), so this is justified duplication. No action needed.

---

## Summary

| File | Redundancy Level | Suggestion |
|------|-----------------|------------|
| `index.md` | Low | Deduplicate Overview paragraph vs. numbered reasons |
| `patch_embedding_port.md` | Low | Remove duplicate value-scaling fusion tip |
| `adaptive_pooling_port.md` | Low | Condense "What Changed and Why" narration of comparison table |

Total estimated savings across all three files: ~13 lines. None of the suggestions affect technical correctness or remove load-bearing content.
