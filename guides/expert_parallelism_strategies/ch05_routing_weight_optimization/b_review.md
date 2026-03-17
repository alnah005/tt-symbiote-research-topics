# B Review — Expert Parallelism Chapter 5: Routing Weight Optimization

## Verdict: APPROVED

No issues found — chapter approved.

---

## Verification Summary

All three changed files were read in full and checked against ground truth and the compression change specifications.

---

## Change Verification

### C1+C2 — `router_forward_pass.md` Section 3

**Requirement:** Section 3 must still correctly state partial selection is O(E + k log k), full sort is O(E log E), ~7.3× faster, and must point to `topk_selection_efficiency.md` for details.

**Findings (line 117):**

> "Full sort is O(E log E); partial selection (heap-based) is O(E + k log k). For E=256, k=8, partial selection is approximately 7.3× faster. Full complexity analysis, the min-heap algorithm step-by-step, and batched comparison counts at B=32 are in `topk_selection_efficiency.md`."

All three required facts are present and correct:

| Claim | Value in file | Ground truth / Expected | Correct? |
|---|---|---|---|
| Partial selection complexity | O(E + k log k) | O(E + k log k) = 280 for E=256, k=8 | Yes |
| Full sort complexity | O(E log E) | O(E log E) = 2,048 for E=256 | Yes |
| Speedup ratio | ~7.3× | 2,048 / 280 ≈ 7.3× | Yes |
| Pointer to detail file | `topk_selection_efficiency.md` | `topk_selection_efficiency.md` | Yes |

The removed table and step-by-step algorithm description remain fully intact in `topk_selection_efficiency.md` (Section 1), so no content has been lost — only relocated. The cross-reference accurately describes what the linked file contains.

---

### C3 — `topk_selection_efficiency.md` Section 4

**Requirement:** Section 4 must now end with a cross-reference to `router_kernel_fusion.md` Section 2 instead of the Warning block.

**Findings (line 139):**

> "Implementation caveats and TTNN support requirements for this fusion are covered in `router_kernel_fusion.md` Section 2, which is the authoritative home for kernel composition implementation guidance."

The Warning block has been removed. The cross-reference correctly identifies both the target file and section, and accurately describes the type of content found there. No substantive information about implementation caveats has been lost — it is present in the authoritative location (see below).

---

### Authoritative Warning Block — `router_kernel_fusion.md` Section 2

**Requirement:** The Warning about TTNN kernel composition support must still be present in Section 2 as the authoritative copy.

**Findings (lines 121–122):**

> **Warning:** Tile-level fusion of matmul and top-k requires the kernel compiler to support partial output accumulation into the top-k buffer as each output tile completes. Verify TTNN kernel composition capabilities before relying on this optimization. If not available, a two-pass approach (full matmul, then fused sigmoid + topk) still eliminates one kernel boundary.

The Warning is intact and unmodified. Readers following the cross-reference from `topk_selection_efficiency.md` will reach this warning without any content gap.

---

## Ground Truth Spot Checks

The following ground-truth values were verified across all three files:

| Claim | Ground truth | File(s) checked | Result |
|---|---|---|---|
| BF16 mantissa bits | 7 bits | `router_forward_pass.md` line 145 | Correct |
| W_r BF16 size | 3.67 MB | `router_forward_pass.md` line 25; `router_kernel_fusion.md` line 277 | Correct |
| W_r INT8 size | 1.84 MB | `router_kernel_fusion.md` lines 281, 283, 307, 366 | Correct (all 4 instances) |
| O(E + k log k) for E=256, k=8 | 256 + 8×3 = 280 | `topk_selection_efficiency.md` line 30 | Correct |
| Full sort comparisons for E=256 | 256×8 = 2,048 | `topk_selection_efficiency.md` line 23 | Correct |
| Speedup ratio | 2,048 / 280 ≈ 7.3× | `topk_selection_efficiency.md` line 44; `router_forward_pass.md` line 117 | Correct |

---

## Correctness Assessment of Removals

Neither removal introduced a correctness gap:

1. The top-k complexity table and min-heap algorithm description removed from `router_forward_pass.md` Section 3 are retained verbatim in `topk_selection_efficiency.md` Section 1 and Section 5. The cross-reference in Section 3 accurately points readers to that content.

2. The TTNN kernel composition Warning removed from `topk_selection_efficiency.md` Section 4 is retained verbatim in `router_kernel_fusion.md` Section 2. The new cross-reference explicitly names that file and section, and describes its role as the authoritative source.

No numerical values, algorithmic descriptions, or implementation warnings were deleted from the chapter — all content was relocated with accurate cross-references. The chapter is internally consistent.
