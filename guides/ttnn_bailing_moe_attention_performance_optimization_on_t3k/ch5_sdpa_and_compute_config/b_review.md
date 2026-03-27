# Agent B Review — Chapter 5 — Pass 1

## Issue 1 — Factual arithmetic error: accumulator tile size in `paged_sdpa_chunk_sizes.md`

**File:** `paged_sdpa_chunk_sizes.md`, lines 64–65

**Claim:**
```
Total accumulator tiles for all N_q heads: 16 × 4 = 64 tiles (128B each in FP32)
Total accumulator size: 64 × 128 B = 8 KB  [ESTIMATE]
```

**Error:** A TTNN/Wormhole tile is 32×32 elements. In FP32 (4 bytes/element) that is 32 × 32 × 4 = 4,096 B (4 KB) per tile — not 128 B. The figure "128B each in FP32" is wrong by a factor of 32. The total 8 KB figure happens to be numerically consistent with the stated per-tile size (64 × 128 = 8,192 B ≈ 8 KB) but is derived from a wrong premise. The correct total for 64 full FP32 tiles would be 64 × 4,096 B = 256 KB. (The actual footprint is smaller because the Q sequence dimension at decode batch=1 is 1 token, not 32, so each "accumulator tile" is a partial tile; but the document does not make that distinction — it states the accumulator in units of full tiles and then assigns those tiles the wrong byte size.)

**Impact:** The size reasoning underpinning the conclusion that `q_chunk_size=0` is safe for `N_q=16` rests on this figure. While the qualitative conclusion is likely correct, the quantitative support is factually wrong and will mislead readers reasoning about L1 budget.

---

## Issue 2 — Factual arithmetic error: S_crossover calculation in `math_fidelity_tradeoff.md`

**File:** `math_fidelity_tradeoff.md`, lines 189–195

**Claim:**
```
S_crossover = DRAM_bandwidth / (KV_bytes_per_token * N_kv)
            = 192e9 / (512 * 4)
            = 192e9 / 2048
            ≈ 93,750 tokens  [ESTIMATE]
```

**Error:** 192 × 10^9 / 2,048 = 93,750,000 — not 93,750. The result is off by a factor of 1,000 (three decimal places dropped). The correct crossover is approximately 93.75 million tokens, not 93,750 tokens.

**Impact:** The qualitative conclusion ("Ling's typical decode context S ≤ 8192 is far below this crossover") remains correct either way — 8,192 is below both 93,750 and 93,750,000. However, the stated numeric result is wrong and any reader checking the arithmetic will lose confidence in the document. It should be corrected.

---

# Agent B Review — Chapter 5 — Pass 2

## Issue 1 — Factual error: BF16 machine epsilon stated as 2^{-8} in `math_fidelity_tradeoff.md`

**File:** `math_fidelity_tradeoff.md`, lines 90–92

**Claim:**
```
A BF16 number has 8 mantissa bits (+ 1 implicit leading 1), giving approximately 2–3 decimal
digits of precision.
...
ε_BF16 ≈ 2^{-8} ≈ 0.0039
```

**Error:** BFloat16 has 7 stored (explicit) mantissa bits plus 1 implicit leading bit, giving 8 significant bits total. Machine epsilon is defined as 2^{-(p-1)} where p is the number of significant bits, which gives 2^{-(8-1)} = 2^{-7} ≈ 0.0078 — not 2^{-8} ≈ 0.0039. The document conflates "8 mantissa bits" (the total significant-bit count) with the exponent in machine epsilon, producing a value that is off by a factor of 2.

**Impact:** This error propagates directly into the HiFi4 per-multiply error bound used throughout the precision analysis (line 92: "~0.004 at HiFi4"), the HiFi2 bound derived from it ("~0.016 at HiFi2, roughly 4× larger"), and all downstream error estimates in the precision table (lines 154–161) and the output accumulation estimates (lines 139–148). All quantitative error bounds are understated by a factor of 2 relative to the correct BF16 machine epsilon. The qualitative conclusions (HiFi2 is within BF16 noise) may still hold, but they rest on a wrong numerical foundation that a careful reader will notice.

---

# Agent B Review — Chapter 5 — Pass 3

No feedback — chapter approved.
