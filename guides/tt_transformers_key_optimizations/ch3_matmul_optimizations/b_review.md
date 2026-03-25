# B Review — Chapter 3: MatMul Optimizations — Pass 1

## Files Reviewed

- `index.md`
- `matmul_program_configs.md`
- `weight_layout_and_quantization.md`
- `l1_sharding_for_matmul.md`

---

## Issues

### Issue 1 — `weight_layout_and_quantization.md`, line 76

**Wrong text:**
> "LoFi is approximately 4× faster than HiFi4 in throughput."

**Why it's wrong:**
The confirmed math fidelity pass counts are: LoFi=1 pass, HiFi2=2 passes, HiFi3=3 passes, HiFi4=4 passes. Characterizing LoFi as "4× faster than HiFi4" is a simplification that conflates pass count with wall-clock throughput and implies a linear relationship that is not confirmed. The correct factual statement is that LoFi performs 1 accumulation pass versus HiFi4's 4 passes — i.e., LoFi uses 1/4 the passes of HiFi4.

**Fix:**
Replace:
> "LoFi is approximately 4× faster than HiFi4 in throughput."

With:
> "LoFi uses 1 accumulation pass versus HiFi4's 4 passes — one quarter the passes of HiFi4."

---

## Summary

1 factual error identified. All other claims checked against confirmed facts:

- Dst register constraint (`out_subblock_h * out_subblock_w <= 8` in fp16, `<= 4` in fp32): **correct**
- BFP4_B bit layout (~4.5 bits/element, 3-bit mantissa + 1 sign + shared 8-bit exponent per 16-value block): **correct**
- BFP8_B bit layout (~8.5 bits/element, 7-bit mantissa): **correct**
- Math fidelity pairings (BFP4+LoFi, BFP8+HiFi2, BF16+HiFi4): **correct**
- Llama 3.1 8B N150 throughput numbers (~28 t/s/u performance mode, ~23 t/s/u accuracy mode, ~22% / +5 t/s/u difference): **correct**
- Matrix FPU primitive (16×16 sub-tiles): **correct**
- `packer_l1_acc`: not mentioned in these files; no issue

---

VERDICT: CHANGES NEEDED

---

## Agent A Change Log — Pass 1 Fix

- Fixed factual error on line 76 of `weight_layout_and_quantization.md`: replaced "LoFi is approximately 4× faster than HiFi4 in throughput" with "LoFi uses 1 accumulation pass versus HiFi4's 4 passes — one quarter the passes of HiFi4." The original phrasing incorrectly implied a confirmed 4× wall-clock throughput multiplier; the corrected text accurately describes the pass count relationship without overstating the performance claim.

---

# B Review — Chapter 3: MatMul Optimizations — Pass 2

## Files Reviewed

- `index.md`
- `matmul_program_configs.md`
- `weight_layout_and_quantization.md`
- `l1_sharding_for_matmul.md`

---

## Issues

### Issue 1 — Internal inconsistency: shape constraint for DRAM-sharded

**Location A:** `matmul_program_configs.md`, line 89
> "This constrains valid layer sizes but is satisfied by all standard transformer hidden dimensions (multiples of 256 or 512)."

**Location B:** `l1_sharding_for_matmul.md`, line 92
> "Standard transformer hidden dimensions (multiples of 128 or 256) satisfy this constraint for the core grids used on N150 and N300."

**Why it's wrong:**
Both sentences describe the same constraint (DRAM-sharded shape divisibility for standard transformer hidden dimensions) but give different multiples: "256 or 512" in one file versus "128 or 256" in the other. These two claims are mutually inconsistent. At most one can be correct; the discrepancy means at least one file contains a factual error. The two files should agree on which multiples characterize standard transformer hidden dimensions in this context.

**Fix:**
Reconcile both statements to use the same, accurate set of multiples. The more conservative bound ("multiples of 128 or 256") is a strictly weaker claim than "multiples of 256 or 512," meaning the `l1_sharding_for_matmul.md` version includes more valid sizes. The correct value should be verified against actual N150/N300 DRAM bank counts and the core grid sizes used in tt-transformers; whichever is accurate, the two files must match.

---

## Summary

1 factual error identified (internal inconsistency in DRAM-sharded shape constraint between two files). All other claims re-checked against confirmed facts:

- Pass 1 fix applied correctly — LoFi/HiFi4 pass count language now accurate: **confirmed**
- Dst register constraint (`out_subblock_h * out_subblock_w <= 8` in fp16, `<= 4` in fp32): **correct**
- BFP4_B bit layout (3-bit mantissa + 1 sign + shared 8-bit exponent per 16-value block): **correct**
- BFP8_B bit layout (7-bit mantissa + 1 sign + shared 8-bit exponent per 16-value block): **correct**
- Math fidelity pairings (BFP4+LoFi, BFP8+HiFi2, BF16+HiFi4): **correct**
- Math fidelity pass counts (LoFi=1, HiFi2=2, HiFi3=3, HiFi4=4): **correct**
- Llama 3.1 8B N150 throughput (~28 t/s/u performance mode BFP4 MLP, ~23 t/s/u accuracy mode BFP8 MLP): **correct**
- BF16 mantissa width (7-bit explicit mantissa field): **correct**
- `packer_l1_acc`: not mentioned in these files; no issue

---

VERDICT: CHANGES NEEDED

---

## Agent A Change Log — Pass 2 Fix

1. Fixed internal inconsistency in `matmul_program_configs.md` (line 89, Shape Constraint section): removed the specific multiples "multiples of 256 or 512" and replaced with the general constraint: "N must be divisible by the product of DRAM bank count and core grid column count; standard transformer hidden dimensions (4096, 8192, etc.) satisfy this constraint in practice."
2. Fixed internal inconsistency in `l1_sharding_for_matmul.md` (line 92, Shape Constraint section): removed the specific multiples "multiples of 128 or 256" and replaced with the same general constraint: "the weight N dimension must be divisible by the product of DRAM bank count and core grid column count; standard transformer hidden dimensions (4096, 8192, etc.) satisfy this constraint in practice." Also removed the device-specific qualifier referencing N150 and N300, which is no longer needed given the generalized statement.

---

# B Review — Chapter 3: MatMul Optimizations — Pass 3

## Files Reviewed

- `index.md`
- `matmul_program_configs.md`
- `weight_layout_and_quantization.md`
- `l1_sharding_for_matmul.md`

---

## Issues

### Issue 1 — `weight_layout_and_quantization.md`, line 76

**Wrong text:**
> "BFP4+LoFi is the fastest possible matmul configuration, approximately 8× lower bandwidth and 4× fewer FPU passes than BF16+HiFi4."

**Why it's wrong:**
The "4× fewer FPU passes" is correct (LoFi=1 pass vs HiFi4=4 passes). However, the "approximately 8× lower bandwidth" claim is not. BFP4_B stores each element in approximately 4.5 bits (3-bit mantissa + 1 sign + 0.5-bit share of the 8-bit block exponent, per confirmed facts). BF16 stores each element in 16 bits. The bandwidth reduction factor is 16 / 4.5 ≈ 3.56×, not 8×. An 8× reduction would require approximately 2 bits per element, which does not correspond to BFP4_B's confirmed format.

**Fix:**
Replace "approximately 8× lower bandwidth" with "approximately 3.5× lower bandwidth" (or equivalently, "less than half the bandwidth"), which correctly reflects the 16-bit vs ~4.5-bit per-element storage ratio.

---

## Summary

1 factual error identified. All other claims re-checked against confirmed facts:

- Pass 1 and Pass 2 fixes applied correctly: **confirmed**
- Dst register constraint (`out_subblock_h * out_subblock_w <= 8` in fp16, `<= 4` in fp32): **correct**
- BFP4_B bit layout (3-bit mantissa + 1 sign + shared 8-bit exponent per 16-value block, ~4.5 bits/element): **correct**
- BFP8_B bit layout (7-bit mantissa + 1 sign + shared 8-bit exponent per 16-value block, ~8.5 bits/element): **correct**
- Math fidelity pairings (BFP4+LoFi, BFP8+HiFi2, BF16+HiFi4): **correct**
- Math fidelity pass counts (LoFi=1, HiFi2=2, HiFi3=3, HiFi4=4): **correct**
- LoFi/HiFi4 pass count language (1 pass vs 4 passes): **correct**
- DRAM-sharded shape constraint (general statement, no specific multiples): **correct**
- Llama 3.1 8B N150 throughput (~28 t/s/u BFP4 MLP, ~23 t/s/u BFP8 MLP): **correct**
- LoFi's 5×7-bit multiplier → 12-bit intermediate product (5+7=12 bits maximum product width): **mathematically correct**
- `packer_l1_acc`: not mentioned in these files; no issue

---

VERDICT: CHANGES NEEDED

---

## Agent A Change Log — Pass 3 Fix

- Fixed `weight_layout_and_quantization.md` line 76: changed "approximately 8× lower bandwidth" to "approximately 3.5× lower bandwidth (BFP4_B at ~4.5 bits/element versus BF16 at 16 bits/element)". The original claim of 8× was incorrect — BFP4_B stores ~4.5 bits per element (3-bit mantissa + 1 sign + 0.5-bit amortized share of the 8-bit block exponent), and BF16 stores 16 bits per element, giving a bandwidth reduction factor of 16 / 4.5 ≈ 3.56×. An 8× reduction would require ~2 bits per element, which does not match BFP4_B's confirmed format.

---

# B Review — Chapter 3: MatMul Optimizations — Pass 4

## Issues Found

### Issue 1 — `index.md`, line 17 (program config summary table)

**Wrong text:**
> `MatmulMultiCoreReuseProgramConfig` | Standard 2D tiled matmul | **Weight reuse across core grid**

**Why it's wrong:**
`matmul_program_configs.md` explicitly states that in this config "each core fetches its required activation and weight tiles from DRAM independently" — there is no weight reuse across the core grid. Weight reuse across the core grid (via NoC multicast) is the defining characteristic of `MatmulMultiCoreReuseMultiCastProgramConfig`, not `MatmulMultiCoreReuseProgramConfig`. Describing the standard 2D config as having "weight reuse across core grid" misattributes the multicast config's key property to the baseline config.

**Fix:**
Replace the key characteristic for `MatmulMultiCoreReuseProgramConfig` with an accurate description of what this config does: independent per-core DRAM fetch, no cross-core weight sharing.

---

## Verification of Prior Fixes

- Pass 1 fix: LoFi/HiFi4 described as "1 accumulation pass versus HiFi4's 4 passes — one quarter the passes of HiFi4" — **confirmed present and correct** (`weight_layout_and_quantization.md` line 76)
- Pass 2 fix: DRAM-sharded shape constraint generalized to "product of DRAM bank count and core grid column count" in both `matmul_program_configs.md` (line 89) and `l1_sharding_for_matmul.md` (line 92) — **confirmed present and consistent**
- Pass 3 fix: "approximately 3.5× lower bandwidth (BFP4_B at ~4.5 bits/element versus BF16 at 16 bits/element)" — **confirmed present and correct** (`weight_layout_and_quantization.md` line 76)

VERDICT: CHANGES NEEDED

---

## Agent A Change Log — Pass 4 Fix

- Fixed `index.md` program config summary table: changed the key characteristic for `MatmulMultiCoreReuseProgramConfig` from "Weight reuse across core grid" to "Independent per-core DRAM fetch; no cross-core weight sharing". The original description incorrectly attributed the multicast config's defining property (weight reuse across the core grid via NoC broadcast) to the baseline config. Per `matmul_program_configs.md`, each core in `MatmulMultiCoreReuseProgramConfig` fetches its weight tiles from DRAM independently — there is no cross-core weight reuse.

---

# B Review — Chapter 3: MatMul Optimizations — Pass 5

## Issues Found

### Issue 1 — `weight_layout_and_quantization.md`, lines 82–86 (Transposed Weight Layout section)

**Wrong text:**
> "TTNN matmul follows the convention `output = activation @ weight`, where activation has shape `[M, K]` and weight has shape `[K, N]`. This is the standard convention for linear layers when weights are stored as `[in_features, out_features]`.
>
> For a weight matrix W of shape `[in_features, out_features]`, there are two options:
>
> - **Pre-transpose on host**: Store W transposed as W^T of shape `[out_features, in_features]` and call `ttnn.matmul(activation, W_T)` where `W_T` has shape `[K, N]` with K=in_features and N=out_features."

**Why it's wrong:**
The text contains a self-contradictory shape claim. It states that TTNN expects weight of shape `[K, N]` = `[in_features, out_features]` (line 82), and then sets up W as already having shape `[in_features, out_features]` (line 84). Transposing W gives W^T of shape `[out_features, in_features]` — yet the text then asserts W_T has "shape `[K, N]` with K=in_features and N=out_features." A matrix of shape `[out_features, in_features]` has its first dimension equal to out_features, not in_features, so calling it `[K, N]` with K=in_features is incorrect.

The actual situation is: PyTorch checkpoints store weight as `[out_features, in_features]` (the transpose of TTNN's expected `[K, N]` = `[in_features, out_features]`). The pre-transpose operation converts from the checkpoint's `[out_features, in_features]` layout to `[in_features, out_features]` = `[K, N]`. The text had the starting shape of W wrong, which made the subsequent shape arithmetic internally inconsistent.

**Fix:**
Correct the starting shape of W to `[out_features, in_features]` (the checkpoint convention) so that transposing it yields W^T of shape `[in_features, out_features]` = `[K, N]`, consistent with TTNN's expected layout.

---

## Verification of Prior Fixes

- Pass 1 fix: LoFi/HiFi4 described as "LoFi uses 1 accumulation pass versus HiFi4's 4 passes — one quarter the passes of HiFi4" — **confirmed present and correct** (`weight_layout_and_quantization.md` line 76)
- Pass 2 fix: DRAM-sharded shape constraint generalized to "product of DRAM bank count and core grid column count" in both `matmul_program_configs.md` (line 89) and `l1_sharding_for_matmul.md` (line 92) — **confirmed present and consistent**
- Pass 3 fix: "approximately 3.5× lower bandwidth (BFP4_B at ~4.5 bits/element versus BF16 at 16 bits/element)" — **confirmed present and correct** (`weight_layout_and_quantization.md` line 76)
- Pass 4 fix: `index.md` program config table entry for `MatmulMultiCoreReuseProgramConfig` reads "Independent per-core DRAM fetch; no cross-core weight sharing" — **confirmed present and correct**

VERDICT: CHANGES NEEDED

---

## Agent A Change Log — Pass 5 Fix

- Fixed `weight_layout_and_quantization.md` Transposed Weight Layout section (lines 82–86): corrected the starting shape of the checkpoint weight matrix from `[in_features, out_features]` to `[out_features, in_features]` (the actual PyTorch checkpoint convention), and updated the surrounding prose to match. The original text described W as already having shape `[in_features, out_features]` (which is the layout TTNN expects, requiring no transpose), then claimed that transposing it produces W^T of shape `[out_features, in_features]` with K=in_features — an internally inconsistent statement. The corrected text correctly identifies that the checkpoint stores weight as `[out_features, in_features]`, and that pre-transposing yields W^T of shape `[in_features, out_features]` = `[K, N]`, which TTNN matmul accepts directly without a `transpose_b` flag.

---

# B Review — Chapter 3: MatMul Optimizations — Pass 6

## Issues Found

### Issue 1 — `matmul_program_configs.md`, lines 24, 27, and 99: Dst constraint described as "fp16 mode" — omits BF16

**Wrong text (line 24):**
> `out_subblock_h * out_subblock_w <= 8` in fp16 mode (Dst holds 8 tiles)

**Wrong text (line 27):**
> Valid combinations for fp16 mode include (1×8), (2×4), (4×2), (8×1), (2×2), (1×4), (4×1), (1×1), and others satisfying the constraint.

**Wrong text (line 99):**
> The Dst register constraint — `out_subblock_h * out_subblock_w <= 8` in fp16 mode, `<= 4` in fp32 mode — is a hard limit.

**Why it's wrong:**
The confirmed fact is: `out_subblock_h * out_subblock_w ≤ 8` for **fp16/bf16**; `≤ 4` for **fp32**. The text uses "fp16 mode" in all three occurrences, omitting BF16. This is a factual error for this guide in particular: every activation tensor in the workloads described (BFP4×BF16, BFP8×BF16, BF16×BF16) uses BF16 as the output accumulation type. A practitioner using BF16 who reads "fp16 mode" might incorrectly conclude the ≤8 constraint does not apply to their workload. BF16 must be named alongside fp16.

**Fix:**
Replace "fp16 mode" with "fp16/bf16 mode" in all three occurrences.

---

## Verification of Prior Fixes

- Pass 1 fix: LoFi/HiFi4 described as "LoFi uses 1 accumulation pass versus HiFi4's 4 passes — one quarter the passes of HiFi4" — **confirmed present and correct** (`weight_layout_and_quantization.md` line 76)
- Pass 2 fix: DRAM-sharded shape constraint generalized to "product of DRAM bank count and core grid column count" in both `matmul_program_configs.md` (line 89) and `l1_sharding_for_matmul.md` (line 92) — **confirmed present and consistent**
- Pass 3 fix: "approximately 3.5× lower bandwidth (BFP4_B at ~4.5 bits/element versus BF16 at 16 bits/element)" — **confirmed present and correct** (`weight_layout_and_quantization.md` line 76)
- Pass 4 fix: `index.md` program config table entry for `MatmulMultiCoreReuseProgramConfig` reads "Independent per-core DRAM fetch; no cross-core weight sharing" — **confirmed present and correct**
- Pass 5 fix: Transposed Weight Layout section starts with W of shape `[out_features, in_features]` (PyTorch checkpoint convention) — **confirmed present and correct** (`weight_layout_and_quantization.md` lines 82–86)

VERDICT: CHANGES NEEDED

---

## Agent A Change Log — Pass 6 Fix

- Fixed `matmul_program_configs.md`: replaced "fp16 mode" with "fp16/bf16 mode" in all three occurrences (lines 24, 27, and 99). The Dst register constraint `out_subblock_h * out_subblock_w ≤ 8` applies to both fp16 and BF16 output types, not fp16 alone. Since every matmul in this guide produces BF16 output (BFP4×BF16, BFP8×BF16, BF16×BF16), omitting BF16 from the constraint description was a factual error that could mislead practitioners applying the constraint to their BF16 workloads.

---

# B Review — Chapter 3: MatMul Optimizations — Pass 7

## Issues Found

### Issue 1 — `matmul_program_configs.md`, "Why This Matters for Decode" section

**Wrong text:**
> "a 4096×16384 BFP8 weight matrix is 128 MB"

**Why it's wrong:**
A 4096×16384 matrix contains 4096 × 16384 = 67,108,864 elements. BFP8_B stores each element at approximately 8.5 bits/element (7-bit mantissa + 1 sign + shared 8-bit exponent per 16-value block). At 8.5 bits/element: 67,108,864 × 8.5 / 8 ≈ 71,240,192 bytes ≈ 68 MB. The stated value of 128 MB corresponds to BF16 storage (2 bytes/element × 67,108,864 = 134,217,728 bytes = 128 MB), not BFP8. The text incorrectly assigns the BF16 size to a BFP8 matrix.

**Fix:**
Replace "128 MB" with "approximately 68 MB".

---

## Verification of Prior Fixes

- Pass 1 fix: LoFi/HiFi4 described as "LoFi uses 1 accumulation pass versus HiFi4's 4 passes — one quarter the passes of HiFi4" — **confirmed present and correct** (`weight_layout_and_quantization.md` line 76)
- Pass 2 fix: DRAM-sharded shape constraint generalized to "product of DRAM bank count and core grid column count" in both `matmul_program_configs.md` and `l1_sharding_for_matmul.md` — **confirmed present and consistent**
- Pass 3 fix: "approximately 3.5× lower bandwidth (BFP4_B at ~4.5 bits/element versus BF16 at 16 bits/element)" — **confirmed present and correct** (`weight_layout_and_quantization.md` line 76)
- Pass 4 fix: `index.md` program config table entry for `MatmulMultiCoreReuseProgramConfig` reads "Independent per-core DRAM fetch; no cross-core weight sharing" — **confirmed present and correct**
- Pass 5 fix: Transposed Weight Layout section starts with W of shape `[out_features, in_features]` (PyTorch checkpoint convention) and correctly describes pre-transposition to `[in_features, out_features]` = `[K, N]` — **confirmed present and correct** (`weight_layout_and_quantization.md`)
- Pass 6 fix: All three occurrences of "fp16 mode" in `matmul_program_configs.md` now read "fp16/bf16 mode" — **confirmed present and correct**

VERDICT: CHANGES NEEDED

---

## Agent A Change Log — Pass 7 Fix

- Fixed `matmul_program_configs.md`, "Why This Matters for Decode" section: changed "a 4096×16384 BFP8 weight matrix is 128 MB" to "a 4096×16384 BFP8 weight matrix is approximately 68 MB". The original value of 128 MB is the BF16 size for that shape (67,108,864 elements × 2 bytes = 128 MB). BFP8_B stores ~8.5 bits/element (7-bit mantissa + 1 sign + shared 8-bit exponent per 16-value block), giving 67,108,864 × 8.5 / 8 ≈ 68 MB. Using the BF16 figure for a BFP8 matrix understates the compression benefit and would mislead practitioners reasoning about on-device memory budgets.

---

# B Review — Chapter 3: MatMul Optimizations — Pass 8

## Issues Found

### Issue 1 — `weight_layout_and_quantization.md`, line 99 (Key Takeaways): "half the bandwidth of BF16" for BFP8

**Wrong text:**
> "BFP8 provides sufficient precision at half the bandwidth of BF16."

**Why it's wrong:**
BFP8_B stores approximately 8.5 bits/element (7-bit mantissa + 1 sign + shared 8-bit exponent per 16-value block). BF16 stores 16 bits/element. The bandwidth reduction factor is 16 / 8.5 ≈ 1.88×, not 2× ("half"). Stating "half the bandwidth" overstates BFP8's compression by approximately 6%. The confirmed fact is explicitly: BFP8 vs BF16 bandwidth = 16 / 8.5 ≈ 1.88× (NOT 2×). This is the same category of error corrected in Pass 3 for BFP4 vs BF16. The corrected text should use "approximately 1.88× lower bandwidth" (or equivalently "approximately 47% less bandwidth") rather than "half."

**Fix:**
Replace "half the bandwidth of BF16" with "approximately 1.88× lower bandwidth than BF16".

---

## Verification of Prior Fixes

- Pass 1 fix: LoFi/HiFi4 described as "LoFi uses 1 accumulation pass versus HiFi4's 4 passes — one quarter the passes of HiFi4" — **confirmed present and correct** (`weight_layout_and_quantization.md` line 76)
- Pass 2 fix: DRAM-sharded shape constraint generalized to "product of DRAM bank count and core grid column count" in both `matmul_program_configs.md` (line 89) and `l1_sharding_for_matmul.md` (line 92) — **confirmed present and consistent**
- Pass 3 fix: "approximately 3.5× lower bandwidth (BFP4_B at ~4.5 bits/element versus BF16 at 16 bits/element)" — **confirmed present and correct** (`weight_layout_and_quantization.md` line 76)
- Pass 4 fix: `index.md` program config table entry for `MatmulMultiCoreReuseProgramConfig` reads "Independent per-core DRAM fetch; no cross-core weight sharing" — **confirmed present and correct**
- Pass 5 fix: Transposed Weight Layout section starts with W of shape `[out_features, in_features]` (PyTorch checkpoint convention) and correctly describes pre-transposition to `[in_features, out_features]` = `[K, N]` — **confirmed present and correct** (`weight_layout_and_quantization.md`)
- Pass 6 fix: All three occurrences of "fp16 mode" in `matmul_program_configs.md` now read "fp16/bf16 mode" — **confirmed present and correct**
- Pass 7 fix: "a 4096×16384 BFP8 weight matrix is approximately 68 MB" — **confirmed present and correct** (`matmul_program_configs.md` line 85)

VERDICT: CHANGES NEEDED

---

## Agent A Change Log — Pass 8 Fix

- Fixed `weight_layout_and_quantization.md`, Key Takeaways section (line 99): changed "BFP8 provides sufficient precision at half the bandwidth of BF16" to "BFP8 provides sufficient precision at approximately 1.88× lower bandwidth than BF16". The original "half the bandwidth" implies a 2× reduction. BFP8_B stores ~8.5 bits/element (7-bit mantissa + 1 sign + shared 8-bit exponent per 16-value block) versus BF16's 16 bits/element, giving a bandwidth ratio of 16 / 8.5 ≈ 1.88×, not 2×. This is the same category of bandwidth ratio error corrected for BFP4 vs BF16 in Pass 3.

---

# B Review — Chapter 3: MatMul Optimizations — Pass 9

## Issues Found

### Issue 1 — `weight_layout_and_quantization.md`, line 34 (Why BFP4 Helps): "roughly a 2× reduction" for BFP4 vs BFP8 bandwidth

**Wrong text:**
> "BFP4 stores each weight element in approximately 4.5 bits... compared to approximately 8.5 bits for BFP8. This is roughly a 2× reduction in weight storage and bandwidth."

**Why it's wrong:**
The text itself supplies the operands: ~4.5 bits (BFP4) vs ~8.5 bits (BFP8). The ratio is 8.5 / 4.5 ≈ 1.89×, not 2×. Stating "roughly 2×" overstates the compression by approximately 6%, the same category of error as Pass 3 (BFP4 vs BF16 bandwidth) and Pass 8 (BFP8 vs BF16 bandwidth). The following sentence then compounds this by saying "halving the weight bandwidth requirement," which also implies 2× when the actual reduction is ~1.89×.

**Fix:**
Replace "roughly a 2× reduction in weight storage and bandwidth" with "approximately a 1.9× reduction in weight storage and bandwidth (8.5 / 4.5 ≈ 1.89×)"; replace "Halving the weight bandwidth requirement" with "Reducing the weight bandwidth requirement by approximately 1.9×".

---

### Issue 2 — `weight_layout_and_quantization.md`, line 54 (BFP8 for Attention Weights): "halving memory vs BF16" for BFP8

**Wrong text:**
> "it retains sufficient mantissa precision (7 bits) while halving memory vs BF16."

**Why it's wrong:**
Pass 8 established and corrected the identical claim in the Key Takeaways section: BFP8_B stores ~8.5 bits/element vs BF16's 16 bits/element, giving a reduction factor of 16 / 8.5 ≈ 1.88×, not 2× ("halving"). The BFP8 Attention Weights section contains the same error in a different location that Pass 8 did not address.

**Fix:**
Replace "while halving memory vs BF16" with "while reducing memory to approximately 53% of BF16 (approximately 1.88× lower, since BFP8_B at ~8.5 bits/element vs BF16 at 16 bits/element gives 16 / 8.5 ≈ 1.88×)".

---

## Verification of Prior Fixes

- Pass 1 fix: LoFi/HiFi4 described as "LoFi uses 1 accumulation pass versus HiFi4's 4 passes — one quarter the passes of HiFi4" — **confirmed present and correct** (`weight_layout_and_quantization.md` line 76)
- Pass 2 fix: DRAM-sharded shape constraint generalized to "product of DRAM bank count and core grid column count" in both `matmul_program_configs.md` and `l1_sharding_for_matmul.md` — **confirmed present and consistent**
- Pass 3 fix: "approximately 3.5× lower bandwidth (BFP4_B at ~4.5 bits/element versus BF16 at 16 bits/element)" — **confirmed present and correct** (`weight_layout_and_quantization.md` line 76)
- Pass 4 fix: `index.md` program config table entry for `MatmulMultiCoreReuseProgramConfig` reads "Independent per-core DRAM fetch; no cross-core weight sharing" — **confirmed present and correct**
- Pass 5 fix: Transposed Weight Layout section starts with W of shape `[out_features, in_features]` (PyTorch checkpoint convention) and correctly describes pre-transposition to `[in_features, out_features]` = `[K, N]` — **confirmed present and correct** (`weight_layout_and_quantization.md`)
- Pass 6 fix: All three occurrences of "fp16 mode" in `matmul_program_configs.md` now read "fp16/bf16 mode" — **confirmed present and correct**
- Pass 7 fix: "a 4096×16384 BFP8 weight matrix is approximately 68 MB" — **confirmed present and correct** (`matmul_program_configs.md` line 85)
- Pass 8 fix: "BFP8 provides sufficient precision at approximately 1.88× lower bandwidth than BF16" in Key Takeaways — **confirmed present and correct** (`weight_layout_and_quantization.md` line 99)

VERDICT: CHANGES NEEDED

---

## Agent A Change Log — Pass 9 Fix

- Fixed `weight_layout_and_quantization.md`, "Why BFP4 Helps" section (line 34): changed "This is roughly a 2× reduction in weight storage and bandwidth" to "This is approximately a 1.9× reduction in weight storage and bandwidth (8.5 / 4.5 ≈ 1.89×)". BFP4 is ~4.5 bits/element and BFP8 is ~8.5 bits/element; the ratio is 8.5/4.5 ≈ 1.89×, not 2×. Claiming "roughly 2×" overstates the compression by ~6%, the same category of error corrected for BFP4 vs BF16 (Pass 3) and BFP8 vs BF16 (Pass 8). Also changed the following sentence's "Halving the weight bandwidth requirement" to "Reducing the weight bandwidth requirement by approximately 1.9×" for consistency with the corrected ratio.

- Fixed `weight_layout_and_quantization.md`, "BFP8 for Attention Weights" section (line 54): changed "while halving memory vs BF16" to "while reducing memory to approximately 53% of BF16 (approximately 1.88× lower, since BFP8_B at ~8.5 bits/element vs BF16 at 16 bits/element gives 16 / 8.5 ≈ 1.88×)". Pass 8 corrected the identical "half the bandwidth" claim in the Key Takeaways section but did not address this earlier occurrence in the BFP8 Attention Weights section. The bandwidth reduction for BFP8 vs BF16 is ~1.88×, not 2×.

---

# B Review — Chapter 3: MatMul Optimizations — Pass 10

## Issues Found

### Issue 1 — `weight_layout_and_quantization.md`, line 36 ("Why BFP4 Helps"): "roughly doubles" contradicts the stated ~1.89× ratio

**Wrong text:**
> "Reducing the weight bandwidth requirement by approximately 1.9× roughly doubles the effective matmul throughput for that layer, assuming the bottleneck was the weight DRAM read and not something else (e.g., activation memory or compute)."

**Why it's wrong:**
The sentence correctly states the bandwidth reduction is approximately 1.9× (i.e., ~1.89×), but then says this "roughly doubles" throughput. "Roughly doubles" implies ~2×. However, in a purely bandwidth-bound regime, a 1.89× reduction in bandwidth translates to a 1.89× improvement in throughput — not 2×. The two figures in the same sentence are mutually inconsistent: the bandwidth ratio stated is ~1.89×, and the throughput conclusion should follow the same ratio. This is a residual error from the Pass 9 fix, which correctly updated "Halving the weight bandwidth requirement" to "Reducing the weight bandwidth requirement by approximately 1.9×" but left the consequent "roughly doubles" in place.

**Fix:**
Replace "roughly doubles the effective matmul throughput" with "roughly improves the effective matmul throughput by approximately 1.9×".

---

## Verification of Prior Fixes

- Pass 1 fix: LoFi/HiFi4 described as "LoFi uses 1 accumulation pass versus HiFi4's 4 passes — one quarter the passes of HiFi4" — **confirmed present and correct** (`weight_layout_and_quantization.md` line 76)
- Pass 2 fix: DRAM-sharded shape constraint generalized to "product of DRAM bank count and core grid column count" in both `matmul_program_configs.md` (line 89) and `l1_sharding_for_matmul.md` (line 92) — **confirmed present and consistent**
- Pass 3 fix: "approximately 3.5× lower bandwidth (BFP4_B at ~4.5 bits/element versus BF16 at 16 bits/element)" — **confirmed present and correct** (`weight_layout_and_quantization.md` line 76)
- Pass 4 fix: `index.md` program config table entry for `MatmulMultiCoreReuseProgramConfig` reads "Independent per-core DRAM fetch; no cross-core weight sharing" — **confirmed present and correct**
- Pass 5 fix: Transposed Weight Layout section starts with W of shape `[out_features, in_features]` (PyTorch checkpoint convention) and correctly describes pre-transposition to `[in_features, out_features]` = `[K, N]` — **confirmed present and correct** (`weight_layout_and_quantization.md`)
- Pass 6 fix: All three occurrences of "fp16 mode" in `matmul_program_configs.md` now read "fp16/bf16 mode" — **confirmed present and correct**
- Pass 7 fix: "a 4096×16384 BFP8 weight matrix is approximately 68 MB" — **confirmed present and correct** (`matmul_program_configs.md` line 85)
- Pass 8 fix: "BFP8 provides sufficient precision at approximately 1.88× lower bandwidth than BF16" in Key Takeaways — **confirmed present and correct** (`weight_layout_and_quantization.md` line 99)
- Pass 9 fix (Issue 1): "This is approximately a 1.9× reduction in weight storage and bandwidth (8.5 / 4.5 ≈ 1.89×)" and "Reducing the weight bandwidth requirement by approximately 1.9×" — **confirmed present and correct** (`weight_layout_and_quantization.md` line 34/36); however the consequent "roughly doubles" on line 36 is the residual error flagged in Issue 1 above
- Pass 9 fix (Issue 2): "reducing memory to approximately 53% of BF16 (approximately 1.88× lower, since BFP8_B at ~8.5 bits/element vs BF16 at 16 bits/element gives 16 / 8.5 ≈ 1.88×)" — **confirmed present and correct** (`weight_layout_and_quantization.md` line 54)

VERDICT: CHANGES NEEDED

---

## Agent A Change Log — Pass 10 Fix

- Fixed `weight_layout_and_quantization.md`, "Why BFP4 Helps" section (line 36): changed "roughly doubles the effective matmul throughput" to "roughly improves the effective matmul throughput by approximately 1.9×". The Pass 9 fix correctly updated "Halving the weight bandwidth requirement" to "Reducing the weight bandwidth requirement by approximately 1.9×" but left "roughly doubles" in the same sentence, which still implies 2×. In a bandwidth-bound regime, a 1.89× reduction in bandwidth yields approximately 1.89× higher throughput, not 2×. "Roughly doubles" is internally inconsistent with the ~1.89× ratio stated earlier in the same sentence.

---

# B Review — Chapter 3: MatMul Optimizations — Pass 11

## Issues Found

### Issue 1 — `weight_layout_and_quantization.md`, line 76: "approximately 3.5×" should be "approximately 3.56×"

**Wrong text:**
> "approximately 3.5× lower bandwidth (BFP4_B at ~4.5 bits/element versus BF16 at 16 bits/element)"

**Why it's wrong:**
The confirmed bandwidth ratio for BFP4_B (~4.5 bits) vs BF16 (16 bits) is 16/4.5 ≈ 3.56×. The Pass 3 fix correctly replaced the erroneous "8×" but landed on "3.5×", which is still imprecise. 3.56 rounds to 3.6×, not 3.5×. The confirmed fact is 3.56×; "3.5×" understates the ratio and is inconsistent with the confirmed value.

**Fix:**
Replace "approximately 3.5× lower bandwidth" with "approximately 3.56× lower bandwidth".

---

## Verification of Prior Fixes

- Pass 1 fix: LoFi/HiFi4 pass count — "LoFi uses 1 accumulation pass versus HiFi4's 4 passes — one quarter the passes of HiFi4" — **confirmed present and correct**
- Pass 2 fix: DRAM-sharded shape constraint generalized in both `matmul_program_configs.md` and `l1_sharding_for_matmul.md` to "product of DRAM bank count and core grid column count" — **confirmed present and consistent**
- Pass 3 fix: "approximately 3.5× lower bandwidth" — **present but imprecise** (see Issue 1 above; 3.56× is the confirmed value)
- Pass 4 fix: `index.md` table entry for `MatmulMultiCoreReuseProgramConfig` reads "Independent per-core DRAM fetch; no cross-core weight sharing" — **confirmed present and correct**
- Pass 5 fix: Transposed Weight Layout section begins with W of shape `[out_features, in_features]` — **confirmed present and correct**
- Pass 6 fix: All Dst register mode references read "fp16/bf16 mode" — **confirmed present and correct**
- Pass 7 fix: "a 4096×16384 BFP8 weight matrix is approximately 68 MB" — **confirmed present and correct**
- Pass 8 fix: "approximately 1.88× lower bandwidth than BF16" for BFP8 in Key Takeaways — **confirmed present and correct**
- Pass 9 fix (Issue 1): "approximately a 1.9× reduction in weight storage and bandwidth (8.5 / 4.5 ≈ 1.89×)" — **confirmed present and correct**
- Pass 9 fix (Issue 2): "approximately 1.88× lower, since BFP8_B at ~8.5 bits/element vs BF16 at 16 bits/element gives 16 / 8.5 ≈ 1.88×" — **confirmed present and correct**
- Pass 10 fix: "roughly improves the effective matmul throughput by approximately 1.9×" — **confirmed present and correct**

VERDICT: CHANGES NEEDED

---

## Agent A Change Log — Pass 11 Fix

- Fixed `weight_layout_and_quantization.md`, line 76: changed "approximately 3.5× lower bandwidth" to "approximately 3.56× lower bandwidth". The Pass 3 fix correctly removed the erroneous "8×" but replaced it with "3.5×", which is still imprecise. The confirmed ratio is 16/4.5 ≈ 3.56×; the value 3.56 rounds to 3.6×, not 3.5×. Updated to "approximately 3.56×" to match the confirmed bandwidth ratio.

---

# B Review — Chapter 3: MatMul Optimizations — Pass 12

## Issues Found

No issues found.

## Verification of Prior Fixes

- Pass 1 fix: "LoFi uses 1 accumulation pass versus HiFi4's 4 passes — one quarter the passes of HiFi4" — **confirmed present and correct** (`weight_layout_and_quantization.md` line 76)
- Pass 2 fix: DRAM-sharded shape constraint generalized to "product of DRAM bank count and core grid column count" in both `matmul_program_configs.md` (line 89) and `l1_sharding_for_matmul.md` (line 92) — **confirmed present and consistent**
- Pass 3 fix: "approximately 3.5× lower bandwidth" (original replacement value) — superseded by Pass 11 fix; see below
- Pass 4 fix: `index.md` table entry for `MatmulMultiCoreReuseProgramConfig` reads "Independent per-core DRAM fetch; no cross-core weight sharing" — **confirmed present and correct**
- Pass 5 fix: Transposed Weight Layout section begins with W of shape `[out_features, in_features]` (PyTorch checkpoint convention) and pre-transposing yields W^T of shape `[in_features, out_features]` = `[K, N]` — **confirmed present and correct** (`weight_layout_and_quantization.md` lines 82–91)
- Pass 6 fix: All three Dst register references in `matmul_program_configs.md` read "fp16/bf16 mode" — **confirmed present and correct**
- Pass 7 fix: "a 4096×16384 BFP8 weight matrix is approximately 68 MB" — **confirmed present and correct** (`matmul_program_configs.md` line 85)
- Pass 8 fix: "approximately 1.88× lower bandwidth than BF16" for BFP8 in Key Takeaways — **confirmed present and correct** (`weight_layout_and_quantization.md` line 99)
- Pass 9 fix (Issue 1): "approximately a 1.9× reduction in weight storage and bandwidth (8.5 / 4.5 ≈ 1.89×)" — **confirmed present and correct** (`weight_layout_and_quantization.md` line 34)
- Pass 9 fix (Issue 2): "reducing memory to approximately 53% of BF16 (approximately 1.88× lower, since BFP8_B at ~8.5 bits/element vs BF16 at 16 bits/element gives 16 / 8.5 ≈ 1.88×)" — **confirmed present and correct** (`weight_layout_and_quantization.md` line 54)
- Pass 10 fix: "roughly improves the effective matmul throughput by approximately 1.9×" — **confirmed present and correct** (`weight_layout_and_quantization.md` line 36)
- Pass 11 fix: "approximately 3.56× lower bandwidth (BFP4_B at ~4.5 bits/element versus BF16 at 16 bits/element)" — **confirmed present and correct** (`weight_layout_and_quantization.md` line 76)

VERDICT: APPROVED

---

# B Review — Chapter 3: MatMul Optimizations — Pass 13

## Issues Found

### Issue 1 — `matmul_program_configs.md`, line 77: Activation size arithmetic wrong — "512 KB" should be "256 KB"

**Wrong text:**
> "while the activation batch is tiny (32 tokens × 4096 hidden = 512 KB in BF16)"

**Why it's wrong:**
32 tokens × 4096 hidden = 131,072 elements. BF16 stores 2 bytes per element. 131,072 × 2 = 262,144 bytes = 256 KB. The stated value of 512 KB is exactly 2× the correct answer. 512 KB would require 262,144 elements, corresponding to either 64 tokens × 4096 hidden or 32 tokens × 8192 hidden — neither matches the example as written. This is a straightforward arithmetic error in the concrete sizing example. The error was present before C's edit and was not introduced by the compression; C's edit retained this line verbatim (correct per the compression analysis, which flagged this as load-bearing), but the underlying figure is wrong.

**Fix:**
Replace "32 tokens × 4096 hidden = 512 KB in BF16" with "32 tokens × 4096 hidden = 256 KB in BF16".

---

## Verification of C's Edit

C's compression of the DRAM-sharded section (matmul_program_configs.md lines 75–83) is structurally sound:

- The decode-rationale framing ("In decode, M is small (1–32 tokens)...") is retained — correct per LBE item 1.
- The concrete size example ("approximately 68 MB ... activation batch is tiny") is retained — correct per LBE item 2.
- The per-bank mechanism detail is removed and replaced with a cross-reference to `l1_sharding_for_matmul.md#dram-sharded-matmul-decode` — correct per C2 and C3.
- The Shape Constraint subsection is replaced with a cross-reference — consistent with the C1 change already applied in Pass 1 (which removed specific multiples) and the Pass 2 fix (which generalized the constraint). The authoritative constraint text remains intact in `l1_sharding_for_matmul.md` line 92.
- The throughput sentence is removed (C3 applied) — the authoritative throughput table remains in `l1_sharding_for_matmul.md` lines 82–86.
- The cross-reference anchor `#dram-sharded-matmul-decode` matches the heading "## DRAM-Sharded Matmul (Decode)" in `l1_sharding_for_matmul.md`.

No factual errors introduced by C's edit. The one issue found (512 KB) was pre-existing.

## Verification of All Prior Fixes

- Pass 1 fix: "LoFi uses 1 accumulation pass versus HiFi4's 4 passes — one quarter the passes of HiFi4" — **confirmed present and correct** (`weight_layout_and_quantization.md` line 76)
- Pass 2 fix: DRAM-sharded shape constraint generalized to "product of DRAM bank count and core grid column count" in both files — **confirmed; `matmul_program_configs.md` now cross-references `l1_sharding_for_matmul.md` where the authoritative constraint text is present and correct**
- Pass 3/11 fix: "approximately 3.56× lower bandwidth (BFP4_B at ~4.5 bits/element versus BF16 at 16 bits/element)" — **confirmed present and correct** (`weight_layout_and_quantization.md` line 76)
- Pass 4 fix: `index.md` table entry for `MatmulMultiCoreReuseProgramConfig` reads "Independent per-core DRAM fetch; no cross-core weight sharing" — **confirmed present and correct**
- Pass 5 fix: Transposed Weight Layout section begins with W of shape `[out_features, in_features]` (PyTorch checkpoint convention) — **confirmed present and correct** (`weight_layout_and_quantization.md` lines 82–91)
- Pass 6 fix: All three Dst register references in `matmul_program_configs.md` read "fp16/bf16 mode" — **confirmed present and correct** (lines 24, 25, 87)
- Pass 7 fix: "a 4096×16384 BFP8 weight matrix is approximately 68 MB" — **confirmed present and correct** (`matmul_program_configs.md` line 77)
- Pass 8 fix: "approximately 1.88× lower bandwidth than BF16" for BFP8 in Key Takeaways — **confirmed present and correct** (`weight_layout_and_quantization.md` line 99)
- Pass 9 fix (Issue 1): "approximately a 1.9× reduction in weight storage and bandwidth (8.5 / 4.5 ≈ 1.89×)" — **confirmed present and correct** (`weight_layout_and_quantization.md` line 34)
- Pass 9 fix (Issue 2): "reducing memory to approximately 53% of BF16 (approximately 1.88× lower, since BFP8_B at ~8.5 bits/element vs BF16 at 16 bits/element gives 16 / 8.5 ≈ 1.88×)" — **confirmed present and correct** (`weight_layout_and_quantization.md` line 54)
- Pass 10 fix: "roughly improves the effective matmul throughput by approximately 1.9×" — **confirmed present and correct** (`weight_layout_and_quantization.md` line 36)

VERDICT: CHANGES NEEDED

---

## Agent A Change Log — Pass 13 Fix

- Fixed `matmul_program_configs.md`, line 77: changed "32 tokens × 4096 hidden = 512 KB in BF16" to "32 tokens × 4096 hidden = 256 KB in BF16". The arithmetic is: 32 × 4096 = 131,072 elements; at 2 bytes/element (BF16), that is 262,144 bytes = 256 KB. The original value of 512 KB is exactly 2× the correct answer and does not correspond to the stated dimensions. This pre-existing error was retained verbatim by C's compression (correct behavior — C preserved the load-bearing concrete sizing example) but the underlying figure was wrong.

---

# B Review — Chapter 3: MatMul Optimizations — Pass 14

## Issues Found

No issues found.

## Verification of Prior Fixes

- **Pass 11 fix** (BFP4 vs BF16 ratio "3.56×"): Confirmed present in `weight_layout_and_quantization.md` line 76 — "approximately 3.56× lower bandwidth (BFP4_B at ~4.5 bits/element versus BF16 at 16 bits/element)". Correct.
- **Pass 13 fix** (activation size "256 KB"): Confirmed present in `matmul_program_configs.md` line 77 — "32 tokens × 4096 hidden = 256 KB in BF16". Arithmetic verified: 32 × 4096 × 2 = 262,144 bytes = 256 KB. Correct.
- **Pass 13 fix** (C agent condensed DRAM-sharded section): Confirmed. `l1_sharding_for_matmul.md` DRAM-Sharded section is appropriately concise with no inflated claims.

All remaining arithmetic-sensitive claims verified:
- BFP8_B weight size (4096×16384×1.0625 ≈ 68 MB): correct
- BFP4_B bit layout (~4.5 bits: 3 mantissa + 1 sign + 0.5 shared exponent): correct
- BFP8_B vs BFP4_B ratio (8.5/4.5 ≈ 1.89×): correct
- BFP8_B vs BF16 ratio (16/8.5 ≈ 1.88×): correct
- Dst register constraint (≤8 fp16/bf16, ≤4 fp32): correct
- Math fidelity pass counts (LoFi=1, HiFi2=2, HiFi4=4) and pairings: correct
- LoFi vs HiFi4 pass count (4× fewer passes): correct
- Llama 3.1 8B N150 throughput (~28 t/s/u BFP4, ~23 t/s/u BFP8): correct
- Height sharding: 32 users / 32 cores = 1 row per core: correct

VERDICT: APPROVED
