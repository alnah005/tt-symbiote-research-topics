# B Review — Chapters 4 and 5: GDN Fused Kernel and Scan Primitives Survey

## Pass 1

### Issues found: 3

---

**Issue 1 — `wormhole_t3k_adaptation.md`, Section 2.1 (Available Core Grid), wrong Tensix core count per chip**

**Error:** "Wormhole has a (8, 4) = 32 Tensix core grid available per chip."

The T3K mesh consists of 8 Wormhole B0 devices, each with an 8×8 grid of Tensix cores (64 cores per chip). The value `(8, 4) = 32` is wrong on both counts: the grid dimensions are 8×8, not 8×4, and the product is 64, not 32.

The error does not affect any downstream conclusion in the file — the kernel uses only 4 cores per device and nothing else in the file depends on the stated 32-core figure — but the stated hardware spec is factually incorrect.

**Correction:** Replace "Wormhole has a (8, 4) = 32 Tensix core grid available per chip." with "Wormhole has a (8, 8) = 64 Tensix core grid available per chip."

---

**Issue 2 — `gdn_full_fused_inplace_analysis.md`, Section 3.3 (CB Layout), wrong total CB usage figure**

**Error:** "Total CB usage per core: approximately 38 KB."

The five CBs listed in the same table sum to: 32,768 + 2,048 + 2,048 + 2,048 + 2,048 = 40,960 bytes = **40 KB**, not 38 KB. This is confirmed by the explicit arithmetic in `wormhole_t3k_adaptation.md` Section 1.2, which computes the identical CB set and correctly arrives at 40,960 bytes (40 KB). The "~38 KB" figure in the analysis file is an arithmetic error — it is 2,048 bytes (one 2 KB CB) short of the correct total.

The error does not affect the safety conclusion (40 KB is still well within 1.5 MB), but it creates a contradiction between the two files in the same chapter and is incorrect on its own terms.

**Correction:** Replace "Total CB usage per core: approximately 38 KB." with "Total CB usage per core: approximately 40 KB (40,960 bytes)."

---

**Issue 3 — `mamba_ssm_kernel_review.md`, Section 2 (What Mamba SSM Computes), wrong dimension annotation on the outer product**

**Error:** "`B_t ⊗ x_t` is the outer product write: `[d_model] ⊗ [d_state] → [d_model, d_state]`"

`B_t` is defined two lines earlier in the same section as "a per-step, input-dependent write vector `[d_state]`", and `x_t` is defined as "the input `[d_model]`". The annotation `[d_model] ⊗ [d_state]` on the expression `B_t ⊗ x_t` reverses the dimensions: it assigns `[d_model]` to `B_t` and `[d_state]` to `x_t`, the opposite of their definitions. The correct annotation for `B_t ⊗ x_t` is `[d_state] ⊗ [d_model] → [d_model, d_state]` (or equivalently, the expression should be written as `x_t ⊗ B_t: [d_model] ⊗ [d_state] → [d_model, d_state]` if the conventional row-first outer product convention is preferred). As written, the type annotation contradicts the variable definitions in the same paragraph.

**Correction:** Replace "`B_t ⊗ x_t` is the outer product write: `[d_model] ⊗ [d_state] → [d_model, d_state]`" with "`B_t ⊗ x_t` is the outer product write: `[d_state] ⊗ [d_model] → [d_model, d_state]`" (keeping the expression `B_t ⊗ x_t` to match the formula above, but correcting the type annotation to match the defined shapes of `B_t` and `x_t`).

---

## Pass 1 Change Log

Changes applied in response to Pass 1 issues:

No changes required. Per the review instructions, fixes are not applied to source files by the reviewer.

---

## Pass 2

### Verification of Pass 1 fixes

**Fix 1 (wormhole_t3k_adaptation.md — core grid, KB figures, idle cores):** APPLIED

Section 2.1 (line 72): "Wormhole has a (8, 8) = 64 Tensix core grid available per chip."
Key Finding (line 7): "total CB usage per core is approximately 40 KB — well within Wormhole's 1.5 MB L1."
Section 2.2 (line 86): "which is acceptable: the decode step is bandwidth-bound on the DRAM state read/write, not compute-bound." and the preceding text reads "Either layout leaves 60 cores idle per DeltaNet layer dispatch (64 available − 4 used)".

All three sub-fixes are present and correct.

**Fix 2 (gdn_full_fused_inplace_analysis.md — 38 KB → 40 KB):** PARTIALLY APPLIED

Line 67 (Section 3.3): "Total CB usage per core: approximately 40 KB (32 KB + 2 KB + 2 KB + 2 KB + 2 KB)." — CORRECTED.
Line 89 (Section 4.1): "The expected total CB usage for the DeltaNet fused kernel is approximately 40 KB (see Section 3.3)." — CORRECTED.
Line 118 (Section 5, Reuse Classification table): "| CB total size | Expected ~38 KB — well within 1.5 MB Wormhole L1; no fundamental obstacle |" — NOT CORRECTED. See Issue 1 below.

**Fix 3 (mamba_ssm_kernel_review.md — dimension annotation corrected):** APPLIED

Line 41: "`B_t ⊗ x_t` is the outer product write: `[d_state] ⊗ [d_model] → [d_model, d_state]` (following the convention that the result rows are indexed by `d_model`)"

The annotation is corrected and the parenthetical note about the result-row convention is present.

### Issues found: 1

---

**Issue 1 — `gdn_full_fused_inplace_analysis.md`, Section 5 (Reuse Classification table), surviving "~38 KB" instance**

**Error:** Line 118, Section 5 Reuse Classification table:

> `| CB total size | Expected ~38 KB — well within 1.5 MB Wormhole L1; no fundamental obstacle |`

The Pass 1 fix corrected the two occurrences in Section 3.3 (line 67) and Section 4.1 (line 89), but this third occurrence in Section 5 was not addressed. The correct figure, established by the arithmetic in Section 3.3 itself (32 KB + 2 KB + 2 KB + 2 KB + 2 KB = 40 KB) and confirmed by `wormhole_t3k_adaptation.md` Section 1.2, is 40 KB. The "~38 KB" value in the summary table is the same arithmetic error as the one that was fixed in the other two locations.

The safety conclusion in the table row ("well within 1.5 MB Wormhole L1; no fundamental obstacle") is correct regardless, but the stated figure is wrong and inconsistent with the corrected figures in Sections 3.3 and 4.1 of the same file.

**Correction:** Replace `| CB total size | Expected ~38 KB — well within 1.5 MB Wormhole L1; no fundamental obstacle |` with `| CB total size | Expected ~40 KB — well within 1.5 MB Wormhole L1; no fundamental obstacle |`.

---

## Pass 2 Change Log

Changes applied in response to Pass 2 issues:
- `gdn_full_fused_inplace_analysis.md`, Section 5, line 118: "Expected ~38 KB" → "Expected ~40 KB" (surviving instance of the Pass 1 Issue 2 arithmetic error, missed in the initial fix pass).

---

## Pass 3

### Verification of Pass 2 fix

**Fix (gdn_full_fused_inplace_analysis.md line 118 — CB total size "~38 KB" → "~40 KB"):** APPLIED

Current text (line 118):

> `| CB total size | Expected ~40 KB — well within 1.5 MB Wormhole L1; no fundamental obstacle |`

### Issues found: 0

---

No issues found. Chapters 4 and 5 approved.
