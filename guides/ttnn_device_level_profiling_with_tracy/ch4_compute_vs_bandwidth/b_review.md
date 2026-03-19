## Pass 1

**File:** `roofline_model_primer.md`, line 97
**Error:** Self-contradictory claim about FPU throughput. The text states "A full outer product of two 32×32 BF16 tiles involves 2048 FLOPs per tile. The matmul engine can sustain **one tile multiply per cycle** when fully pipelined, yielding a practical throughput ceiling of **128 FP16/BF16 FMA ops per cycle per core**." One tile per cycle at 2048 FLOPs/tile equals 2048 FLOPs/cycle, not 128. The following note (line 99) then compounds this by writing "128 FMA ops/cycle figure (= 64 FLOPs/cycle × 2 for FMA pipelining)", which uses "FMA ops" and "FLOPs" interchangeably with the same number, conflicting with the earlier definition that 1 FMA = 2 FLOPs. The ridge-point calculation on line 122 correctly uses T_peak_FPU = 128 FLOPs/cycle and produces the right ridge of 4.0 FLOPs/byte, but the derivation path is internally broken: a reader who follows lines 97–99 literally will compute 2048 FLOPs/cycle and arrive at a ridge point of 64 FLOPs/byte instead of 4.
**Fix:** Change "one tile multiply per cycle" to "one tile multiply every 16 cycles" (since 2048 FLOPs ÷ 128 FLOPs/cycle = 16 cycles per tile). Remove or rewrite the parenthetical on line 99 to say clearly: "128 FLOPs/cycle is the sustained figure; equivalently, each 32×32 tile multiply (2048 FLOPs) takes approximately 16 cycles to complete through the fully pipelined FMA unit."

---

**File:** `classification_method.md`, line 65–69 (Step 3 NOC BW UTIL table) vs. line 176 (Classification Result Table) and `index.md` line 39
**Error:** Conflicting low-end threshold for `NOC BW UTIL`. Step 3 defines "lightly loaded / not the bottleneck" as `NOC BW UTIL < 0.3`, but the Classification Result Table (and the identical table in `index.md`) classify the compute-bound row as `NOC BW UTIL < 0.4`. A reader following Step 3 with a measured `NOC BW UTIL = 0.35` would categorize the NoC as "moderately loaded" (0.3–0.7 bucket) and withhold a compute-bound classification, while the result table would classify the same op as compute-bound. Chapter 5 depends on these thresholds to identify low-FPU-utilization candidates, so the ambiguity will propagate downstream.
**Fix:** Unify the threshold. Either change the Step 3 table's lower bin boundary from `< 0.3` to `< 0.4`, or change the result table's compute-bound row from `< 0.4` to `< 0.3`. The step-by-step procedure and the summary table must use the same number.

---

**File:** `roofline_model_primer.md`, line 50
**Error:** The square-matrix simplification is presented as an approximation when it is exact for BF16. The formula `AI = MKN / (MK + KN + MN)` is already the simplified form after the 2-byte BF16 factors cancel (numerator `2MKN`, denominator `2(MK + KN + MN)`). Setting M = K = N = D gives `D³ / 3D² = D/3`, which is exact, not approximate. The file correctly computes `1024/3 ≈ 341` but the prose says "simplifies to `AI ≈ D/3`" with an approximation sign. Minor in isolation, but a student checking the formula for correctness will introduce needless doubt.
**Fix:** Change `AI ≈ D³ / (3D²) = D / 3` to `AI = D³ / (3D²) = D / 3` (exact equality for BF16 square matrices).

---

**File:** `index.md`, line 35 and 53 — links clickable: verified. All three content-file links (`roofline_model_primer.md`, `classification_method.md`, `worked_examples.md`) use relative paths and are present on disk. Navigation footer (lines 59–61) is present and links both previous and next chapters. No structural gap: all three planned files exist and contain the content Chapter 5 depends on (FPU UTIL thresholds, ridge point, hardware ceilings).

No additional issues found. Items 1 and 2 above would cause a downstream reader to compute an incorrect ridge point or misclassify an op; item 3 is a precision error that undermines formula verification.

## Pass 2

**File:** `roofline_model_primer.md`, line 99
**Error:** The parenthetical "(= 64 FLOPs/cycle × 2 for FMA pipelining)" attached to "128 FMA ops/cycle" is dimensionally incoherent and will cause a reader to compute the wrong ridge point. The note conflates two different unit systems in a single equation: "64 FLOPs/cycle × 2 = 128 FLOPs/cycle" is a valid FLOPs identity, but labeling the result "128 FMA ops/cycle" then implies 1 FMA op = 1 FLOP. However, the file's own definition (lines 94–95) states each FMA counts as 2 FLOPs, which means 128 FMA ops/cycle = 256 FLOPs/cycle. A reader who takes the label "128 FMA ops/cycle" and correctly applies the 2 FLOPs/FMA conversion will compute T_peak_FPU = 256 FLOPs/cycle and arrive at a ridge point of 256 / 32 = 8.0 FLOPs/byte instead of the correct 4.0 FLOPs/byte. The ridge-point calculation at line 122 uses 128 as FLOPs/cycle (correct), but the derivation note makes the unit inconsistent.
**Fix:** Replace the parenthetical with a unit-consistent statement, e.g.: "128 FLOPs/cycle (= 64 elements/cycle × 2 FLOPs per FMA, with the matmul engine sustaining one 32×32 tile every 16 cycles)." Remove the phrase "128 FMA ops/cycle" or replace it with "128 FLOPs/cycle" throughout, since the ridge-point formula requires FLOPs/cycle.

---

**File:** `roofline_model_primer.md`, line 148
**Error:** L1 size is stated as "192 KB of L1 SRAM" per Tensix core. Wormhole B0 architecture documentation specifies approximately 1.5 MB of L1 SRAM per Tensix core. A reader using 192 KB to evaluate whether a kernel's working set fits in L1 will conclude DRAM spill for many workloads that actually reside entirely in L1. For example, a 32×32 BF16 matmul tile triple (A, B, C tiles) occupies 3 × 32 × 32 × 2 = 6,144 bytes — well within either figure — but a moderate matmul working set (e.g., several K-dimension tiles) that fits in 1.5 MB but not 192 KB would be misclassified as DRAM-bound, leading to incorrect optimization choices downstream.
**Fix:** Change "192 KB" to "1.5 MB" (or the correct figure from Tenstorrent's Wormhole B0 documentation) and update any downstream reasoning in the same paragraph that depends on this number.

---

No further issues found. The arithmetic intensity computations in all three worked examples are numerically correct. Threshold values in the classification tables are internally consistent across `index.md` and `classification_method.md` (both use > 0.7 / < 0.4 boundaries as specified). The ridge point of 4.0 FLOPs/byte is correctly derived and used consistently throughout all four files.

## Pass 3

**File:** `classification_method.md`, lines 32–35, 37, and flowchart lines 138–139; `worked_examples.md`, lines 29, 89, and 165
**Error:** The ridge point is stated as **4.0 FLOPs/byte** throughout `classification_method.md` and all three worked examples. However, `roofline_model_primer.md` correctly derives the ridge point as **8.0 FLOPs/byte** (256 FLOPs/cycle ÷ 32 bytes/cycle = 8.0), consistent with the canonical Wormhole B0 values (128 FMA ops/cycle × 2 FLOPs/FMA = 256 FLOPs/cycle; 32 bytes/cycle NoC read link). Using 4.0 FLOPs/byte causes a direct wrong-classification outcome: any op with AI between 4.0 and 8.0 FLOPs/byte (e.g., a small BF16 reduction or a moderate-K matmul) would be called bandwidth-bound by `classification_method.md` while `roofline_model_primer.md` correctly places it in the compute-bound region. The small matmul in Example 2 has AI ≈ 25.6 FLOPs/byte, so its classification is unaffected, but the boundary region between the two primers is actively inconsistent.
**Fix:** In `classification_method.md`, change every occurrence of "4.0 FLOPs/byte" to "8.0 FLOPs/byte" (lines 32, 35, 37, and the flowchart condition `AI < 4.0` / `AI >= 4.0` at lines 138–139). In `worked_examples.md`, change the three occurrences of "ridge point of 4.0 FLOPs/byte" (lines 29, 89, 165) to "ridge point of 8.0 FLOPs/byte". No other numerical changes are required; the AI computations and FLOP/byte counts in all examples are correct.

## Pass 4

**File:** `worked_examples.md`, line 43
**Error:** The prose explanation for Example 1 (large matmul) states "the FPU is consuming it at **128 FLOPs/cycle**." The correct figure for the Wormhole B0 matmul engine is **256 FLOPs/cycle** (128 FMA ops/cycle × 2 FLOPs/FMA), as established in `roofline_model_primer.md` lines 97–99 and the ridge-point derivation at line 122. A reader using 128 FLOPs/cycle to estimate how quickly the FPU drains NoC-delivered data (e.g., to compute required bandwidth or verify that the NoC is lightly loaded) will be off by a factor of 2, producing an incorrect bandwidth-requirement estimate.
**Fix:** Change "the FPU is consuming it at 128 FLOPs/cycle" to "the FPU is consuming it at 256 FLOPs/cycle (128 FMA ops/cycle × 2 FLOPs/FMA)".

---

No further issues found. All ridge-point values now consistently read 8.0 FLOPs/byte across all four files. L1 is correctly stated as ~1.5 MB. Threshold values (FPU UTIL > 0.7 / < 0.3; NOC BW UTIL < 0.4 for compute-bound) are consistent between `index.md`, `classification_method.md`, and `worked_examples.md`. All arithmetic intensity calculations in the three worked examples are numerically correct.

## Pass 5

**No feedback — chapter approved.**

All previously flagged issues have been resolved in the current file text:

- `roofline_model_primer.md` line 97: cycles-per-tile derivation now reads "2048 FLOPs ÷ 256 FLOPs/cycle = 8 cycles per tile" — numerically correct.
- `roofline_model_primer.md` line 99 Note: states "128 FP16 FMA operations per cycle per core ... peak FPU throughput is 256 FLOPs/cycle per core" — consistent with 1 FMA = 2 FLOPs.
- `roofline_model_primer.md` line 125: ridge point = 256 / 32 = 8.0 FLOPs/byte — correct.
- `roofline_model_primer.md` line 148: L1 stated as "~1.5 MB of L1 SRAM (1,536 KB)" — correct.
- `classification_method.md` line 32: ridge point stated as 8.0 FLOPs/byte — consistent with primer.
- `classification_method.md` lines 73–75 and `index.md` line 39: NOC BW UTIL threshold for compute-bound is `< 0.4` in all locations — consistent.
- `worked_examples.md` line 43: FPU consumption stated as 256 FLOPs/cycle — correct.
- All three worked example AI values (≈683, ≈25.6, ≈1.25 FLOPs/byte) verified arithmetically correct.
- FPU UTIL thresholds (> 0.7 compute-bound, < 0.3 bandwidth-bound) consistent across all four files.

## Pass 6

**File:** `worked_examples.md`, line 31 (Note block after Step 1 of Example 1)
**Error:** The "quick mental shortcut" claims "The ratio M×K×N / (K×N) ≈ M = 1024, which also gives the same rough AI estimate quickly." This implies AI ≈ 1024 FLOPs/byte, but the correctly computed AI for this shape (M=1024, K=4096, N=4096) is ≈ 683 FLOPs/byte — a 50% overestimate. The approximation assumes KN dominates the denominator entirely, but here MK + MN = 8,388,608 is half of KN = 16,777,216, so the remaining terms are not negligible. A reader using this shortcut to estimate AI for similar near-square (large K, large N, moderate M) shapes will consistently overshoot by a factor up to 3× as M approaches K and N.
**Fix:** Correct the shortcut formula. The accurate approximation when B dominates is AI ≈ MKN / (KN) = M only when KN >> MK + MN, i.e., K >> M (not the case here where K = 4M). A more accurate quick estimate is AI ≈ MKN / (MK + KN + MN). For this specific shape the Note should either state the correct value (≈683) or caveat that the shortcut is only valid when K and N are both much larger than M (e.g., K, N ≥ 8M). Because both 683 and 1024 are far above the ridge point of 8.0 FLOPs/byte, the classification conclusion is unaffected, but the formula will mislead readers who apply the shortcut to shapes closer to the ridge.

## Pass 7

**No feedback — chapter approved.**

The Pass 6 flagged note block (a quick-shortcut formula at `worked_examples.md` line 31) is no longer present in the current file. All prior issues from Passes 1–6 are resolved in the current text:

- `roofline_model_primer.md`: FPU throughput derivation (8 cycles/tile at 256 FLOPs/cycle), ridge point (256 / 32 = 8.0 FLOPs/byte), and L1 size (1.5 MB / 1,536 KB) are all correct and consistent.
- `classification_method.md`: Ridge point threshold uses 8.0 FLOPs/byte; NOC BW UTIL low-end threshold for compute-bound (`< 0.4`) is consistent with `index.md`; FPU UTIL definition and interpretation table are internally consistent.
- `worked_examples.md`: All three AI values (≈683, ≈25.6, ≈1.25 FLOPs/byte) verified arithmetically correct; tile counts correct; FPU consumption stated as 256 FLOPs/cycle in Example 1.
- `index.md`: Threshold table consistent with `classification_method.md`.

No issue was found that would cause a reader to obtain a wrong numerical answer, implement something incorrectly, or be materially misled into a wrong conceptual understanding.
