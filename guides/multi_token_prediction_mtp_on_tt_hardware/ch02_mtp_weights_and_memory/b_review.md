## B Feedback — Pass 1

1. [mtp_vs_backbone_compute_cost.md, ~line 49 and ~line 56] The Dense FFN table row reports a subtotal of `~88,082,432` FLOPs (which includes the ~2,048 SwiGLU elementwise FLOPs), but the math block immediately below uses `88,080,384` (the projection-only total, explicitly excluding SwiGLU). A reader who sums the table rows gets `231,211,008 + 88,082,432 = 319,293,440`, not the `319,291,392` stated in the math block. The two numbers must agree. — Fix: Change the `Dense FFN total` table cell from `~88,082,432` to `88,080,384` (consistent with the explicit statement that SwiGLU elementwise FLOPs are excluded from totals), or update the math block sum to `319,293,440` and propagate that value to the arithmetic intensity calculation and the index.md summary table.

## B Feedback Application Log — Pass 1

- Fix 1: Corrected Dense FFN FLOP table entry from ~88,082,432 to 88,080,384 (projections-only, consistent with math block); updated total MTP FLOPs from 319,293,440 to 319,291,392 in mtp_vs_backbone_compute_cost.md

---

## B Feedback — Pass 2

1. [mtp_weight_inventory.md, ~line 145] "FFN active params per token" for one backbone MoE block is given as `8 × 44.0M / 128 experts = 2.75M active`. This is wrong: activating 8 experts each with 44.04M parameters gives `8 × 44,040,192 = 352,321,536 ≈ 352.3M` active params per token, not 2.75M. The division by 128 (total experts) produces a meaningless fractional weight count, not an active parameter count. — Fix: Change the cell to `8 × 44.0M = 352.3M active` and remove the erroneous `/ 128 experts` division.

2. [mtp_vs_backbone_compute_cost.md, ~lines 14–15] The FLOP table lists `k_proj` and `v_proj` weight shapes as `[7168, 896]`, but `mtp_weight_inventory.md` (line 45) gives the actual stored shapes as `[896, 7168]` (PyTorch convention: [out_features, in_features]). The stated convention for the FLOP table is input `[1, M]` × weight `[M, N]`, which transposes the stored weight. A reader who copies the shape from the FLOP table to load or initialize the weight matrix would use the wrong shape. — Fix: Add a note that the FLOP table uses computation-order shapes `[in, out]`, which are the transpose of the stored PyTorch weight shapes `[out, in]` given in `mtp_weight_inventory.md`; or change the column to reflect stored shapes and adjust the convention note accordingly.

## B Feedback Application Log — Pass 2

- Fix 1: Corrected active backbone FFN parameter count from wrong "8 × 44.0M / 128 = 2.75M" to correct "8 × 44.0M = 352M" in mtp_weight_inventory.md comparison table
- Fix 2: Changed k_proj/v_proj FLOP table shape entries from [7168, 896] to [896, 7168] in mtp_vs_backbone_compute_cost.md to match stored-weight convention

---

## B Feedback — Pass 3

1. [mtp_vs_backbone_compute_cost.md, ~lines 45–46] `gate_proj` and `up_proj` weight shapes are listed as `[7168, 2048]` in the Dense FFN FLOP table, but the actual stored checkpoint shapes (per ground truth and `mtp_weight_inventory.md` line 78–79) are `[2048, 7168]` (PyTorch [out_features, in_features] convention). This is the identical stored-vs-computation-order discrepancy that was fixed for k_proj/v_proj in Pass 2 but was not applied to the FFN projections. A reader loading or initializing these weight tensors from the FLOP table would use the wrong shape. — Fix: Change gate_proj and up_proj "Weight shape" entries from `[7168, 2048]` to `[2048, 7168]`, consistent with the stored-weight convention now used for k_proj/v_proj in the same file.

2. [mtp_vs_backbone_compute_cost.md, ~lines 9–15] After the Pass 2 fix, the FLOP convention note (line 9) states weight shape as `[M, N]` where M is the input dimension (7168 for k_proj), but the k_proj and v_proj rows now show `[896, 7168]` (output-dimension first = stored shape). A reader applying the stated formula `2 × M × N` to `[896, 7168]` reads M=896 and N=7168, computing `2 × 896 × 7168` — the result is numerically identical due to commutativity, but M no longer means "input dim" as the convention states. This breaks the convention for any non-square projection and would mislead a reader trying to derive FLOPs for an analogous projection. — Fix: Update the convention note to state that the "Weight shape" column reflects stored PyTorch shapes `[out_features, in_features]` and that FLOPs = `2 × in_features × out_features`, so the formula generalizes correctly to all rows in the table.

## B Feedback Application Log — Pass 3

- Fix 1: Corrected gate_proj/up_proj FLOP table shapes from [7168, 2048] to [2048, 7168] in mtp_vs_backbone_compute_cost.md (stored-shape convention: [out, in])
- Fix 2: Updated FLOP convention note to describe stored-shape [out_features, in_features] representation, consistent with all weight entries in the table

---

## B Feedback — Pass 4

**No feedback — chapter approved.**

---

## B Feedback — Pass 5

1. [mtp_vs_backbone_compute_cost.md, lines 96–104] The backbone is stated to have **94** transformer layers, but the ground truth specifies `num_hidden_layers = 80`. This wrong layer count propagates to two derived values in the same file: (a) `FLOPs_backbone_total = 94 × 936M ≈ 87,984M ≈ 88B` should be `80 × 936M = 74,880M ≈ 75B FLOPs`; (b) the MTP FLOP fraction `0.36%` should be `319M / 74,880M ≈ 0.43%`; (c) the cross-check `0.34/94 ≈ 0.36%` should be `0.34/80 ≈ 0.43%`; (d) the dense-FFN counterfactual `1/94 ≈ 1.1%` should be `1/80 ≈ 1.25%`. Fix: Replace 94 with 80 throughout this section and recompute the four derived percentages accordingly.

2. [mtp_memory_footprint.md, line 98] The same wrong layer count appears here: `1.1 / (94 × 39.9) ms ≈ 0.03%` should use 80 layers: `1.1 / (80 × 39.9) ≈ 0.034%` (still negligible, but the stated figure is wrong). Fix: Replace 94 with 80 and recalculate.

3. [mtp_weight_inventory.md, line 135] The comparison table introduction refers to "one of the **94** MoE backbone layers." Fix: Replace 94 with 80.

---

## B Feedback — Pass 6

**No feedback — chapter approved.**
