# Agent B Review: Chapter 4

## Pass 1

1. **Incorrect Gemma 3 pooling kernel size in comparison table (`adaptive_pooling_port.md`, line 138).** The table states Gemma 3 uses "8x8 blocks" as the kernel size. This is wrong. Gemma 3 SigLIP has a 64x64 patch grid (4096 patches) pooled to 256 output tokens, which is a 16x16 output grid. The pooling kernel is therefore 64/16 = 4, i.e., 4x4 blocks with stride 4, not 8x8. Fix: change "8x8 blocks" to "4x4 blocks".

2. **Misleading rationale for sqrt(hidden_size) scaling (`adaptive_pooling_port.md`, lines 68-74).** The text claims the sqrt(1152) scaling "compensates for the magnitude reduction caused by averaging" and connects it to the 1/sqrt(9) RMS reduction from averaging 9 patches. However, sqrt(1152) ~ 33.94 bears no mathematical relationship to the number of averaged patches (9). If the goal were to compensate for averaging, the correct factor would be sqrt(9) = 3 or simply 9. The sqrt(d) scaling is a standard transformer convention (used in embedding layers, attention, etc.) that is unrelated to the pooling kernel size. The stated rationale is misleading even if the scaling factor itself is correct. Recommend rewriting to note this is a standard hidden-dimension scaling convention, separating it from the averaging explanation.

3. **No other factual errors found.** All numerical parameters (patch_size=16, hidden_size=1152, pooling_kernel_size=3, position_embedding_size=10240, token budgets 70/140/280/560/1120), tensor shapes, the 0.36% sparsity calculation, the divisibility-by-48 constraint, and the linear projection dimension (1152 -> 5376) are consistent with the Chapter 1 key facts and internally self-consistent.

## Pass 2

Pass 1 issues verified as fixed:
- Issue 1 (Gemma 3 kernel size): Now correctly reads "4x4 blocks" in the comparison table (line 138).
- Issue 2 (sqrt(1152) rationale): Now correctly describes the scaling as "standard hidden-dimension scaling (the same sqrt(d) convention used throughout the model)" (lines 71-74), with no misleading connection to averaging compensation.

New issues found:

1. **Incorrect claim that valid output token count varies with aspect ratio (`adaptive_pooling_port.md`, lines 151-152).** The text states: "the actual number of valid output tokens depends on the image's aspect ratio." This is wrong. The image processor selects dimensions (h, w) both divisible by 48 such that (h/48) * (w/48) equals the token budget exactly. Different aspect ratios produce different grid shapes (e.g., 20x14 vs 14x20 for budget 280) but the total output token count is always equal to the budget. The chapter's own tables confirm this: every example in the token budget tables shows output tokens equal to the budget regardless of grid shape. Fix: rewrite to say that different aspect ratios produce different grid *shapes* but the same total output token count, since the divisibility-by-48 constraint ensures exact factorization.

## Pass 3

Pass 2 issue verified as fixed:
- Issue 1 (variable output token count): The text at `adaptive_pooling_port.md` lines 149-151 now correctly states "Aspect-ratio preservation changes the grid shape, not the token count" and explicitly notes the total count is fixed by the divisibility-by-48 constraint. Correct.

New issues found:

1. **Incorrect "near-square" grid shape examples in variable input shape table (`patch_embedding_port.md`, lines 205-209).** Every row in the "Example Grid Shapes" column includes a third near-square example that does not produce the correct patch count for its token budget. Specifically: budget 70 lists "27x24" (= 648 patches, not 630); budget 140 lists "36x36" (= 1296, not 1260); budget 280 lists "51x51" (= 2601, not 2520); budget 560 lists "72x72" (= 5184, not 5040); budget 1120 lists "102x102" (= 10404, not 10080). Furthermore, 27, 51, and 102 are not divisible by 3, violating the constraint that patch grid dimensions must be multiples of 3 (since image pixel dimensions must be multiples of 48 = 16 x 3). The landscape/portrait examples in each row are correct (e.g., 21x30, 42x60, etc.). Fix: replace each near-square example with a valid factorization of the required patch count where both dimensions are multiples of 3. For example, budget 70: 630 = 18x35 (but 35 is not divisible by 3) -- in fact 70 has no factorization into two factors both divisible by 3 that are near-square. The valid (h_cells x w_cells) factorizations for 70 are limited (e.g., 7x10, 5x14, etc.), meaning (h_patches x w_patches) options are 21x30, 15x42, etc. Either remove the near-square column entries or replace them with valid non-square factorizations.

## Pass 4

Pass 3 issue verified as fixed:
- Issue 1 (invalid near-square grid entries): The token budget table in `patch_embedding_port.md` (lines 203-209) now contains only the valid landscape/portrait pair per row (e.g., 21x30 and 30x21 for budget 70). All near-square entries have been removed. The remaining examples are verified correct: each pair multiplies to the stated approximate total patches (e.g., 21x30 = 630), each value divides evenly by 9 to yield the token budget, and every grid dimension is divisible by 3.

**No feedback — chapter approved.**
