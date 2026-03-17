# B Review — Chapter 2: SiLU on Wormhole Hardware — Pass 1

1. **`cycles_vs_matmul.md`, lines 91–92 — Incorrect intermediate multiplication results in the gate_proj byte-count calculation.**

   The document states:
   ```
   4096 × 10922 = 44,795,904
   32   × 10922 =    350,336
   ```
   The correct values are:
   ```
   4096 × 10922 = 44,736,512   (off by 59,392)
   32   × 10922 =    349,504   (off by 832)
   ```
   The final total byte count (~90.5 MB) and arithmetic intensity (~31.6 FLOP/byte) are approximately correct because the error is small relative to the dominant K×N term. However, a reader who verifies the intermediate arithmetic step-by-step will find the numbers do not add up. Fix: replace `44,795,904` with `44,736,512` and `350,336` with `349,504`. The corrected sum is `(131,072 + 44,736,512 + 349,504) × 2 = 90,434,176 bytes ≈ 90.4 MB`, which is consistent with the stated ~90.5 MB conclusion.

2. **`tensix_compute_engine.md`, lines 80–83 — The no-overlap claim is stated more broadly than the hardware constraint justifies.**

   The document claims: "There is no hardware mechanism to pipeline the SFPU activation for tile T with the FPU matmul for tile T+1 **within the same core**." The verified constraint is narrower: FPU and SFPU cannot both operate on the *same* output tile simultaneously because they share the destination register. Whether FPU work on tile T+1 can overlap with SFPU work on tile T (double-buffering across tiles) is a separate question that the document asserts as impossible without qualification. If this stronger claim is incorrect, it will cause readers to overestimate sequential latency for multi-tile workloads. Fix: restrict the claim to the verified constraint — "FPU and SFPU cannot both operate on the same output tile simultaneously" — and either cite evidence for the T/T+1 no-overlap claim or remove it.

## Agent A Change Log — B Feedback Pass 1
- cycles_vs_matmul.md: Fixed intermediate arithmetic: 4096×10922=44,736,512 and 32×10922=349,504
- tensix_compute_engine.md: Softened FPU/SFPU sequential constraint to the verified per-tile claim; removed cross-tile pipeline restriction assertion

# B Review — Chapter 2: SiLU on Wormhole Hardware — Pass 2

**Pass 1 fix verification:**

- Fix 1 confirmed (`cycles_vs_matmul.md`, line 91): Intermediate values now read `44,736,512` and `349,504`. The corrected sum `(131,072 + 44,736,512 + 349,504) × 2 = 90,434,176 bytes` is correctly reported as `~90.4 MB`. Fix applied correctly.
- Fix 2 confirmed (`tensix_compute_engine.md`, line 80): The cross-tile T/T+1 pipeline restriction has been removed. The text now states only the verified per-tile constraint: "Within a single tile, the FPU matmul pass and the SFPU activation pass are sequential." Fix applied correctly.

**Remaining issue:**

1. **`cycles_vs_matmul.md`, line 93 — Minor rounding inconsistency between stated byte total and value used in intensity calculation.** Line 91 derives and reports the byte total as `~90.4 MB` (90,434,176 bytes / 10^6). Line 93 then divides by `90.5e6` to compute arithmetic intensity. The two values are inconsistent: using the stated 90.4 MB gives ≈ 31.6 FLOP/byte (shown), while using 90.5 MB gives ≈ 31.6 FLOP/byte as well (the difference is negligible at this precision). The inconsistency is cosmetic but a reader verifying the chain will notice the mismatch. Fix: change `90.5e6` to `90.4e6` on line 93 to match the stated total, or note that both round to ~31.6 and are equivalent.

No other errors found. All hardware facts, SiLU decomposition steps, 32-pass constraint, ridge point (437 FLOP/byte), SiLU arithmetic intensity (0.5 FLOP/byte), and ReLU < SiLU ≈ GELU cost ordering are correct across all four files.

## Agent A Change Log — B Feedback Pass 2
- cycles_vs_matmul.md: Fixed byte-count denominator inconsistency — changed from 90.5e6 to 90.4e6 (consistent with stated ~90.4 MB)

# B Review — Chapter 2: SiLU on Wormhole Hardware — Pass 3

Pass 2 fix verified. No feedback — chapter approved.
