## Pass 1

### Item 1

**File:** `prefill_access_patterns.md`
**Approximate line:** 212–218 (ideal-reuse AI formula)

**Error:** The arithmetic intensity formula for the ideal-reuse case is wrong in two independent ways.

First, the denominator expression `2·(T·d + T·d + T·d)·2` contains only three T·d terms (Q read, K read, O write), omitting the V read (also T·d·2 bytes). The denominator should contain four terms.

Second, even with three terms the arithmetic is wrong: `4·T·w·d / (2·(3·T·d)·2) = 4Twd / 12Td = w/3`, not `w/6` as stated.

With the correct four-term denominator — Q, K, V, O each costing T·d·2 bytes → total 4·T·d·2 = 8·T·d bytes — the result is:

```
AI_ideal = 4·T·w·d / (8·T·d) = w/2
```

The stated formula and result `w/6`, and the downstream value `≈ 683 FLOPs/byte` for w = 4096 (which should be 2048 FLOPs/byte), are both incorrect. The summary table entry `w/6 ≈ 683` is also wrong and must be updated to `w/2 ≈ 2048`.

**Fix:** Replace the formula denominator with `(T·d + T·d + T·d + T·d)·2` (four T·d terms) and propagate the corrected result `w/2`. Update the numerical example to `4096/2 = 2048 FLOPs/byte` and the summary table accordingly.

---

### Item 2

**File:** `prefill_access_patterns.md`
**Approximate line:** 240–244 (no-reuse AI formula approximation step)

**Error:** The approximation step writes the denominator as `4·w·d·2`, yielding 0.5 FLOPs/byte. This is wrong. The full denominator `(T·d + T·w·d + T·w·d + T·d)·2` approximates for large w to `(2·T·w·d)·2 = 4·T·w·d` (the dtype factor of 2 is already absorbed in the `·2` at the end). Dividing: `4·T·w·d / 4·T·w·d = 1 FLOP/byte`, not 0.5.

The spurious extra `·2` in the approximated denominator introduces a factor-of-2 error that will mislead anyone using this figure to assess bandwidth-boundedness.

**Fix:** Replace the approximation step denominator from `4·w·d·2` to `4·w·d`, giving `AI_no_reuse ≈ 1 FLOP/byte`. Update the surrounding text and summary table row ("~0.5" → "~1").

---

## Change Log (Pass 1)

**Date:** 2026-03-27
**File edited:** `prefill_access_patterns.md`

### Fix 1 — Ideal-reuse arithmetic intensity (lines ~207–251)

Replaced the incorrect formula `AI = w/6` with the Flash-Attention style tiling derivation:

- FLOPs = 4·T·w·d
- Bytes = 2·d·(2w + T) (K and V each loaded once as w·d·2 bytes; Q loaded as T·d·2 bytes)
- AI = 2·T·w / (2w + T)
- Asymptotically: AI ≈ 2w for T ≫ 2w; AI = 2w/3 when T = w

Updated the numerical example from `w/6 ≈ 683 FLOPs/byte` to `2w ≈ 8192 FLOPs/byte` (w = 4096). The compute-bound conclusion is preserved (8192 >> 111 FLOPs/byte roofline crossover).

### Fix 2 — No-reuse arithmetic intensity (lines ~253–277)

Removed the spurious extra factor of 2 in the approximated denominator. Corrected derivation:

- Bytes (no reuse, large w) ≈ 4·T·w·d
- AI = 4·T·w·d / 4·T·w·d = 1 FLOPs/byte (not 0.5)

Updated surrounding prose and summary table row from `~0.5` to `~1`. The bandwidth-bound conclusion is preserved (1 << 111 FLOPs/byte roofline crossover).

### Summary Table

Updated both changed rows:
- Ideal K/V reuse: `w/6 ≈ 683` → `≈ 2w ≈ 8192 (T ≫ 2w)`
- No reuse: `~0.5` → `~1`
- Ratio between ideal and no-reuse updated from `w/3` to `approximately 2w`.

---

## Pass 2

### Item 1

**File:** `prefill_access_patterns.md`
**Lines:** 215–217 (ideal-reuse bytes formula) vs 257–258 (no-reuse bytes formula)

**Error:** The ideal-reuse bytes formula omits the output tensor O write, while the no-reuse bytes formula includes it. The ideal-reuse denominator is `2·d·(2w + T)` (K + V + Q only). The no-reuse denominator expands to `(T·d + T·w·d + T·w·d + T·d)·2`, where the trailing T·d term is the O write. Because the two AI figures are computed on different bases (one includes the write, one does not), the factor-of-~2w gap quoted in line 277 ("for w = 4096 is over 8000×") is inflated by roughly 2×. A reader cross-checking the ratio ideal/no-reuse as `2w / 1 = 8192` will get a different number if they independently include the O write in both denominators (the ratio would then be `≈ w / 1 = 4096`).

**Fix:** Either include `T·d·2` for the O write in the ideal-reuse bytes expression — giving `Bytes = 2·d·(2w + 2T)` and `AI = T·w / (w + T)`, which approaches `w` (not `2w`) for T ≫ w — or explicitly state that the O write is excluded from both calculations and remove the O write term from the no-reuse denominator. Whichever convention is chosen must be applied uniformly.

---

### Item 2

**File:** `decode_access_patterns.md`
**Lines:** 82 and 151–152

**Error:** The byte values are labeled as MiB but computed using base-10 megabytes (MB). The calculation `1 × 32 × 2 × 4096 × 128 × 2 = 67,108,864 bytes` equals **64 MiB** (÷ 1,048,576) or **67.1 MB** (÷ 1,000,000). The text states 67.1 MiB, which overstates the true MiB figure by ~4.9%. The same error appears at line 151: `536,870,912 bytes = 512 MiB`, not 536.9 MiB. A reader sizing on-device DRAM budgets in binary units (as is standard for hardware-level work) will compute incorrect headroom if they use these figures directly.

**Fix:** Change "67.1 MiB" to "64 MiB" and "536.9 MiB" to "512 MiB", or relabel both as "MB" if base-10 units are intended throughout.

---

## Change Log (Pass 2)

**Date:** 2026-03-27

### Fix 1 — Ideal-reuse bytes formula missing O write (`prefill_access_patterns.md`)

**Problem:** The ideal-reuse bytes expression included only K + V + Q (`2·d·(2w + T)`), omitting the output tensor O write (`T·d·2`). The no-reuse expression already included the Q read and O write as two T·d terms. This asymmetry made the two AI values non-comparable and inflated the ideal/no-reuse ratio by roughly 2×.

**Fix applied (option a — include O writes in both):**

Updated the ideal-reuse bytes formula to include the O write:

- Bytes = `2·d·(2w + T) + 2·T·d` = `2·d·(2w + 2T)` = `4·d·(w + T)`
- AI = `4·T·w·d / (4·d·(w+T))` = `T·w / (w+T)`
- When T ≫ w: AI ≈ w (was 2w — corrected)
- When T = w: AI = w/2 (was 2w/3 — corrected)
- Numerical example updated: `AI_ideal ≈ 4096 FLOPs/byte` for w = 4096 (was 8192)

The no-reuse bytes expression already included Q and O write terms; updated surrounding prose to make this explicit. The no-reuse AI result of ~1 FLOPs/byte is unchanged.

Updated the prose noting the ideal/no-reuse ratio: changed "approximately 2w, which for w = 4096 is over 8000×" to "approximately w, which for w = 4096 is over 4000×".

**Summary table updated:**

- Ideal K/V reuse: `≈ 2w ≈ 8192 (T ≫ 2w)` → `≈ w ≈ 4096 (T ≫ w)`
- Partial reuse: `Between ~1 and ~2w` → `Between ~1 and ~w`
- No reuse: unchanged at ~1

**Conclusion check:** Ideal-reuse case remains compute-bound (4096 >> 111 FLOPs/byte roofline crossover). No-reuse case remains bandwidth-bound (1 << 111). Both conclusions are preserved.

---

### Fix 2 — MiB/MB unit mislabeling (`decode_access_patterns.md`, lines 82 and 151)

**Problem:** Values labeled as MiB were computed with base-10 division. The true base-2 values are:

- `1 × 32 × 2 × 4096 × 128 × 2 = 67,108,864 bytes = 64 MiB` (not 67.1 MiB)
- `1 × 32 × 2 × 32768 × 128 × 2 = 536,870,912 bytes = 512 MiB` (not 536.9 MiB)

**Fix applied:** Used base-2 (MiB) throughout with clean binary values:

- Line 82: `67.1 MiB` → `64 MiB`
- Line 151: `536.9 MiB` → `512 MiB`
- Line 154: `versus 67.1 MiB` → `versus 64 MiB`

The 8× reduction factor at T = 32768, w = 4096 is preserved: 512 MiB / 64 MiB = 8×.

---

## Pass 3

### Item 1

**File:** `prefill_access_patterns.md`
**Approximate line:** 39

**Error:** The stated "approximately 134M valid entries" is the crude approximation T·w = 32768·4096 = 134,217,728, but the exact formula given two lines above yields a materially different number. Using the exact expression `w(w-1)/2 + w·(T-w+1)`:

```
4096 × 4095 / 2           =   8,386,560   (fill-phase contribution)
4096 × (32768 - 4096 + 1) = 117,424,128   (steady-state contribution)
Total                     = 125,810,688  ≈ 126M
```

The text presents 134M as the result of applying the exact formula, but 134M is the approximation T·w. The discrepancy is ~6.5%. A reader who verifies the arithmetic directly from the formula will compute 126M and conclude either the formula or the number is wrong. The downstream 4× reduction claim (126M vs 537M full triangle) is approximately correct either way, but the stated figure itself is wrong.

**Fix:** Change "approximately 134M" to "approximately 126M" (or equivalently note that the approximation T·w ≈ 134M overestimates the exact value by the fill-phase correction w(w-1)/2 ≈ 8M).

---

No further issues found.

Pass 2 items verified:
- `prefill_access_patterns.md`: ideal-reuse AI formula correctly shows `AI = T·w / (w+T)`, approaches `w` for T >> w; no-reuse AI is ~1 FLOP/byte; both use consistent O write accounting. Confirmed.
- `decode_access_patterns.md` lines 82 and 151: values correctly read 64 MiB and 512 MiB. Confirmed.

---

## Change Log (Pass 3)

**Date:** 2026-03-27

### Fix 1 — Incorrect valid-entries count (`prefill_access_patterns.md`, line 39)

**Problem:** The text stated "approximately 134M valid entries" for T = 32768, w = 4096. The value 134M is the crude approximation T·w = 32768 × 4096 = 134,217,728. The exact formula given two lines above, `w(w-1)/2 + w·(T-w+1)`, yields:

- Fill phase: 4096 × 4095 / 2 = 8,386,560
- Steady-state: 4096 × (32768 − 4096 + 1) = 4096 × 28673 = 117,424,128
- Total = 125,810,688 ≈ 126M

The stated 134M overstated the exact result by ~6.5%.

**Fix applied:** Changed "approximately 134M" to "approximately 126M" on line 39. The downstream 4× reduction claim (126M vs 536M full lower triangle) remains approximately correct and was not altered.

---

## Pass 4

Pass 3 item verified: `prefill_access_patterns.md` line 39 now reads "approximately 126M". Exact value is 125,831,168 (= 4096×4095/2 + 4096×28673), which rounds correctly to 126M. Confirmed.

All numerical claims in both files were re-derived independently. No issues found that would cause a downstream reader to get a wrong numerical answer, implement something incorrectly, or be materially misled.

No feedback — chapter approved.

## Pass 5

All numerical claims were re-derived independently:

- Valid entries (prefill line 39): exact formula gives 125,831,168 ≈ 126M; full triangle T(T+1)/2 ≈ 537M; stated 4× reduction is correct. ✓
- Ideal-reuse AI (prefill lines 205–239): FLOPs = 4Twd; bytes = 4d(w+T); AI = Tw/(w+T); limits w/2 (T=w) and w (T≫w); numerical 4096 FLOPs/byte for w=4096. All correct. ✓
- No-reuse AI (prefill line 251): ~1 FLOP/byte. Verified: bytes ≈ 4Twd; AI = 1. ✓
- Wormhole roofline (prefill lines 245–246): 32e12/288e9 ≈ 111 FLOPs/byte. ✓
- Prefill FLOPs (prefill line 172): 4×32768×4096×128 = 68.7 GFLOPs. ✓
- Decode 64 MiB per layer (decode line 82): 1×32×2×4096×128×2 = 67,108,864 bytes = 64 MiB. ✓
- Full-attention 512 MiB (decode lines 151–154): 1×32×2×32768×128×2 = 536,870,912 bytes = 512 MiB; 8× reduction. ✓
- Decode AI (decode lines 284–288): FLOPs = 4wd; bytes = (2w+2)d×2; AI = w/(w+1) ≈ 1. ✓
- Wrap boundary diagram (decode lines 109–121): T=11, w=8, wp=(11+1) mod 8 = 4; slot assignments verify correctly. ✓
- Write-hazard analysis (decode line 243): slot (T+1) mod w holds position T-w+1. ✓
- Band-diagonal mask diagram (prefill lines 48–58): checked rows 0–3 and 7 against formula max(0,t-w+1) ≤ s ≤ t with w=3. ✓

No feedback — chapter approved.
