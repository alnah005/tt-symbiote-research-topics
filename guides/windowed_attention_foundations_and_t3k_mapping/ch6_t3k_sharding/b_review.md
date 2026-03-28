## Pass 1

### Item 1 — Sequence-parallel CCL volume wrong by 8x (sharding_strategies.md, lines 276–280 and 317–319)

**File:** `sharding_strategies.md`
**Lines:** ~276–280 (numeric substitution), ~317–319 (decision matrix), ~325–326 (ratio)

**Error:** The K+V all-gather formula is correct — `2 × B × H_kv × (w/N) × d × 2 × (N-1)` — but the worked example substitutes `w = 4096` where `w/N = 512` should go. The stated result is:

> `2 × 32 × 8 × 512 × 128 × 2 × 7 = 3,758,096,384 bytes ≈ 3.5 GiB`

Actual: `2 × 32 × 8 × 512 × 128 × 2 × 7 = 469,762,048 bytes = 448 MiB`. The claimed `3,758,096,384` equals `2 × 32 × 8 × 4096 × 128 × 2 × 7` (using full `w` instead of `w/N`). Every downstream number is therefore 8× too large:

| Quantity | Stated | Correct |
|---|---|---|
| K+V CCL volume | ≈ 3.5 GiB | ≈ 448 MiB |
| CCL latency at 12.5 GB/s | ≈ 286 ms | ≈ 38 ms |
| Ratio (seq-par CCL / compute) | ≈ 159× | ≈ 20× |

The decision matrix (line ~300) and the comparative text block (line ~317) repeat the same wrong figures. The qualitative conclusion (seq-parallel is badly communication-bound) remains correct, but a reader implementing anything from these numbers — bandwidth budgets, latency projections, pipeline sizing — will be off by 8×.

**Fix:** In the worked example replace `512` with `4096/8 = 512` (label it clearly as `w/N`) and recompute: volume = 448 MiB, latency ≈ 38 ms, ratio ≈ 20×. Update the decision matrix and comparative text block accordingly.

---

### Item 2 — Same 8× CCL error propagated throughout ccl_operations.md (lines ~205–209 and ~298–303)

**File:** `ccl_operations.md`
**Lines:** ~205–209 (K+V volume), ~298–303 (w-scaling table)

**Error:** `ccl_operations.md` repeats the identical wrong substitution: `2 × 32 × 8 × 512 × 128 × 2 × 7 ≈ 3,758 MiB ≈ 3.5 GiB`. The w-scaling table at lines ~298–303 is uniformly 8× too large:

| w | Stated | Correct |
|---|---|---|
| 1024 | 896 MiB | 112 MiB |
| 4096 | 3,584 MiB | 448 MiB |
| 8192 | 7,168 MiB | 896 MiB |
| 32768 | 28,672 MiB | 3,584 MiB |

The latency column (72 ms / 286 ms / 573 ms / 2.3 s) is similarly 8× too large. The bandwidth-analysis table at line ~244 repeats `≈ 1.75 GiB` for sequence-parallel K only (correct is ~224 MiB) and `≈ 3.5 GiB` for K+V (correct is ~448 MiB). These are the numbers Chapter 7's end-to-end latency projections will consume directly.

**Fix:** Recompute each row with `w/N` as the shard depth. Sequence-parallel CCL for `w = 4096` is 448 MiB at ≈ 38 ms — still clearly prohibitive versus the 222 µs attention compute, but the numbers must be accurate.

---

### Item 3 — "Boundary device" claim incorrect for clean-boundary decode steps (per_device_window_application.md, lines ~343–346)

**File:** `per_device_window_application.md`
**Lines:** ~343–346

**Error:** The text states:

> "the boundary between valid and invalid slots falls on exactly one device (device `floor(T / (w/N))`), and that device handles the partial validity by applying a local mask covering slots `T+1` through `(floor(T / (w/N)) + 1) * (w/N) - 1` of its local shard."

This is wrong whenever `(T+1)` is an exact multiple of `w/N` (i.e., the valid/invalid boundary falls cleanly between two devices). For example, `T = 3`, `w = 16`, `N = 8` (so `w/N = 2`): `floor(3/2) = 1`, so the formula points at device 1 (slots 2–3). Both slots are valid (`s ≤ T = 3`). The "mask range" is slots `T+1 = 4` through `(1+1)*2-1 = 3`, which is empty — no masking on device 1 is needed or applied. The actual first device with invalid slots is device 2 (`floor((T+1)/(w/N)) = floor(4/2) = 2`), which holds only invalid slots and needs a full `-inf` mask. An implementer following the stated text will believe device 1 is "the boundary device" and look there for partial-masking logic, when in reality device 1 needs no special handling and device 2 needs a full mask.

**Fix:** Qualify the claim: "When `(T+1)` is not a multiple of `w/N`, the boundary falls within the shard of device `floor(T / (w/N))`; that device applies a partial mask. When `(T+1)` is an exact multiple of `w/N`, the boundary falls between two adjacent devices with no partial shard: devices `0` through `(T+1)/(w/N) - 1` are fully valid and device `(T+1)/(w/N)` onwards are fully invalid."

---

### Item 4 — Worst-case padding analysis misidentifies the worst case (per_device_window_application.md, lines ~281–283)

**File:** `per_device_window_application.md`
**Lines:** ~281–283

**Error:** The text states:

> "In the worst case (`w mod N = N/2`), up to `N/2` wasted slots per device are introduced."

This is wrong. Padding slots equal `N - (w mod N)`. The maximum number of padding slots occurs when `w mod N = 1` (one extra-slot device, seven padding-slot devices = `N - 1 = 7` total wasted slots). When `w mod N = N/2 = 4`, only `N - 4 = 4` padding slots are introduced — fewer than the `r = 1` case. The worst case is `w mod N = 1`, not `w mod N = N/2`. This would cause an implementer to underestimate buffer overhead when `w mod N = 1`.

**Fix:** Change "In the worst case (`w mod N = N/2`)" to "In the worst case (`w mod N = 1`), `N - 1 = 7` padding slots are introduced across the seven smaller shards."

---

### Item 5 — index.md "Next" link is broken (index.md, implicit in "What Comes Next" section)

**File:** `ch6_t3k_sharding/index.md`
**Lines:** ~117–127 ("What Comes Next" section, no explicit Next link)

**Error:** `per_device_window_application.md` (the last content file of Chapter 6) ends with:

```
**Next:** [Chapter 7 — Roofline Analysis and Existing Kernel Survey](../ch7_roofline_and_kernels/index.md)
```

That file does not exist. This is a content file (not index.md footer) "Next:" navigation link, so it falls within scope. A reader following the reading order will hit a dead link at the end of the chapter. Additionally, `index.md` has no "Next:" footer at all, while the three content files each have one — the index is the entry point and is the one place a reader would first look for the forward link.

**Fix for the broken link:** Because `ch7_roofline_and_kernels/` does not yet exist, replace the clickable link with plain text: `**Next:** Chapter 7 — Roofline Analysis and Existing Kernel Survey (not yet written)`. This prevents a broken-link error without implying the chapter exists.

## Change Log (Pass 1)

**Date:** 2026-03-28
**Agent:** Agent A

### Fix 1 — `sharding_strategies.md`: Sequence-parallel CCL volume corrected (8× error)

The worked example substituted the full window size `w = 4096` where the per-device shard `w/N = 512` was required. Corrected all downstream quantities:

| Quantity | Before (wrong) | After (correct) |
|---|---|---|
| K+V CCL volume | ≈ 3.5 GiB | ≈ 448 MiB |
| CCL latency at 12.5 GB/s | ≈ 286 ms | ≈ 38 ms |
| Ratio (seq-par CCL / compute) | ≈ 159× | ≈ 20× |

Updated locations: numeric substitution block (~lines 276–286), decision matrix CCL volume and latency rows (~line 300–301), comparative text block (~lines 312–326), and the recommendation body (~line 341).

### Fix 2 — `ccl_operations.md`: Same 8× error corrected throughout

Corrected the K+V all-gather numeric result, the bandwidth-analysis table, the transfer-time comparison block, the w-scaling table, and the CCL summary table to use `w/N` as the per-device shard depth:

| w | Before (wrong) | After (correct) |
|---|---|---|
| 1024 | ≈ 896 MiB / 72 ms | ≈ 112 MiB / 9 ms |
| 4096 | ≈ 3,584 MiB / 286 ms | ≈ 448 MiB / 38 ms |
| 8192 | ≈ 7,168 MiB / 573 ms | ≈ 896 MiB / 72 ms |
| 32768 | ≈ 28,672 MiB / 2.3 s | ≈ 3,584 MiB / 287 ms |

Also corrected the bandwidth-analysis table: sequence-parallel K volume changed from ≈ 1.75 GiB to ≈ 224 MiB; K+V from ≈ 3.5 GiB to ≈ 448 MiB. CCL summary block updated to show ~448 MiB / ~38 ms for w=4k.

### Fix 3 — `per_device_window_application.md` ~lines 343–346: Boundary-device formula guarded for exact multiples

The original text claimed "the boundary falls on exactly one device (device `floor(T / (w/N))`)" unconditionally. This is incorrect when `(T+1)` is an exact multiple of `w/N`, in which case the boundary falls cleanly between two devices and no partial-masking is required on any shard.

Added a two-case guard:
- If `(T+1) mod (w/N) == 0`: boundary is between devices; no partial mask needed.
- Otherwise: boundary falls within device `floor(T / (w/N))`; that device applies a partial mask as before.

### Fix 4 — `per_device_window_application.md` ~lines 281–283: Worst-case padding corrected

The text claimed "In the worst case (`w mod N = N/2`), up to `N/2` wasted slots per device are introduced." This is incorrect. The number of padding slots is `N - (w mod N)`, which is maximised when `w mod N = 1` (yielding `N - 1 = 7` wasted slots), not when `w mod N = N/2 = 4` (which yields only 4 wasted slots).

Changed to: "Worst-case padding occurs when `w mod N = 1`, which wastes `N - 1 = 7` slots on the last device."

### Issue 5 (skipped) — Dead link to Ch7

The footer link `../ch7_roofline_and_kernels/index.md` in `per_device_window_application.md` was left unchanged per instructions: Ch7 has not been written yet, and this link is correct per the guide plan.

## Pass 2

### Pass 1 Verification

All four corrections from Pass 1 are confirmed present in the current files:

1. `sharding_strategies.md` — CCL volume now reads `≈ 448 MiB`, latency `≈ 38 ms`, ratio `≈ 20×`. Decision matrix and comparative block updated consistently. **Verified.**
2. `ccl_operations.md` — K+V volume (`448 MiB`), bandwidth table (`≈ 224 MiB` for K, `≈ 448 MiB` for K+V), w-scaling table, and CCL summary block all correct. **Verified.**
3. `per_device_window_application.md` ~lines 347–351 — Two-case guard present and correct: exact-multiple case names the right boundary devices; otherwise-case names device `floor(T / (w/N))`. **Verified.**
4. `per_device_window_application.md` ~lines 281–282 — Worst-case padding now reads `w mod N = 1` → `N - 1 = 7` wasted slots. **Verified.**

### New Issue

**Item 1 — `T_write(s)` formula is wrong (`per_device_window_application.md`, line ~173)**

**File:** `per_device_window_application.md`
**Line:** ~173

**Error:** The document defines the slot-to-step mapping as:

```
T_write(s) = T - ((w - 1 - s + (T % w)) % w)
```

This formula is incorrect. A concrete counterexample: `T = 10`, `w = 4`. The write pointer is `T % w = 2`, so slot 2 was just written at step 10, slot 1 at step 9, slot 0 at step 8, slot 3 at step 7. Expected `T_write` values: `{0→8, 1→9, 2→10, 3→7}`.

The formula produces:

| s | formula result | correct |
|---|---|---|
| 0 | `10 - ((3 - 0 + 2) % 4)` = `10 - 1` = **9** | **8** |
| 1 | `10 - ((3 - 1 + 2) % 4)` = `10 - 0` = **10** | **9** |
| 2 | `10 - ((3 - 2 + 2) % 4)` = `10 - 3` = **7** | **10** |
| 3 | `10 - ((3 - 3 + 2) % 4)` = `10 - 2` = **8** | **7** |

Every slot's age is mis-attributed to the adjacent slot. The correct formula is:

```
T_write(s) = T - ((T % w - s + w) % w)
```

The document's claim that "the window predicate `T - w + 1 ≤ T_write(s) ≤ T` is satisfied by all `w` slots at steady state" happens to remain true (the range of wrong values still spans `{T-w+1, ..., T}`), so the high-level conclusion is not affected. However, any implementer who uses the stated formula to compute the age of a specific slot — for custom masking logic, debugging, or cache eviction decisions — will get the wrong answer for every slot except when `T % w == w - 1`.

**Fix:** Replace the formula on line ~173 with `T_write(s) = T - ((T % w - s + w) % w)`.

---

**Date:** 2026-03-28
**Agent:** Agent B (Pass 2)

## Change Log (Pass 2)

**Date:** 2026-03-28
**Agent:** Agent A (Pass 2)

### Fix 1 — `per_device_window_application.md` line ~173: `T_write(s)` formula corrected

Replaced the incorrect formula:

```
T_write(s) = T - ((w - 1 - s + (T % w)) % w)
```

with the correct formula:

```
T_write(s) = T - ((T % w - s + w) % w)
```

Verification at T=10, w=4 (write pointer at slot T%w=2):

| s | T_write(s) | meaning |
|---|---|---|
| 0 | 10 - ((2-0+4)%4) = 10-2 = **8** | slot 0 was written at step 8 |
| 1 | 10 - ((2-1+4)%4) = 10-1 = **9** | slot 1 was written at step 9 |
| 2 | 10 - ((2-2+4)%4) = 10-0 = **10** | slot 2 was just written (current step) |
| 3 | 10 - ((2-3+4)%4) = 10-3 = **7** | slot 3 was written at step 7 |

All four values are correct and consistent with the circular-buffer write order. The surrounding text (window predicate and fill-phase description) required no changes.

## Pass 3

**Date:** 2026-03-28
**Agent:** Agent B (Pass 3)

### Pass 2 Correction Verified

`per_device_window_application.md` line 173: `T_write(s) = T - ((T \% w - s + w) \% w)` — confirmed present and correct. Spot-checked at T=10, w=4: slot 0 → step 8, slot 1 → step 9, slot 2 → step 10, slot 3 → step 7. All correct.

### Result

No feedback — chapter approved.

## Pass 4

**Date:** 2026-03-28
**Agent:** Agent B (Pass 4)

### Prior Fixes Verified

All four Pass 1 corrections and the Pass 2 `T_write(s)` formula fix are confirmed present and correct in the current files. Spot-checked key numbers:

- `sharding_strategies.md`: K+V CCL volume = 448 MiB, latency ≈ 38 ms, ratio ≈ 20×. Decision matrix consistent.
- `ccl_operations.md`: bandwidth table shows seq-parallel K ≈ 224 MiB, K+V ≈ 448 MiB; w-scaling table correct (w=4096 → 448 MiB / 38 ms); CCL summary consistent.
- `per_device_window_application.md`: `T_write(s) = T - ((T % w - s + w) % w)` present; two-case boundary guard present; worst-case padding correctly states `w mod N = 1` → `N - 1 = 7` wasted slots.

### Result

No feedback — chapter approved.
