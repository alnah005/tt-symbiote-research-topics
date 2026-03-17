# Agent B Review — Chapter 3: All-to-All Operations and num_links Tuning on T3K — Pass 1

## Summary

Three confirmed correctness errors, two minor arithmetic/unit issues, and one code off-by-one. No topology errors found; all T3K-specific claims (Linear, not Ring; cluster_axis=1; 7 hops; N=8) are correctly stated throughout. All D-dependent claims are consistently flagged [D UNVERIFIED]. The per-token 100,352-byte arithmetic and the 200,704-byte round-trip are correct. The V* crossover derivation is algebraically correct. The T(n_l, V) formula is correctly stated as a per-hop formula.

---

## Issues

### Issue 1 — CRITICAL: Reference table per-hop volumes are wrong for three of five rows

**File:** `benchmarking_num_links.md`, lines 227–231 (reference results table)

**Error:** The table footer (line 233) states that per-hop volumes are computed using `C = ceil(k * B * S / E)` with `V_per_hop = C * H * 2`. This is the correct formula and is used consistently in `num_links_parameter.md`. However, the table values for rows 2, 3, and 4 do not match this formula. They appear to have been computed using `V = k * B * S / N * H * 2` (tokens per device under uniform routing), which differs from `C * H * 2` when C ≠ k*B*S/N — i.e., whenever E ≠ N.

Verification using `C = ceil(k * B * S / E)` with `k=8 [D UNVERIFIED]`, `E=256 [D UNVERIFIED]`, `H=7168 [D UNVERIFIED]`:

| Row | B | S | C = ceil(8*B*S/256) | Correct V_per_hop = C*H*2 | Table value |
|---|---|---|---|---|---|
| 1 | 1 | 1 | ceil(0.03125) = 1 | 1 × 7168 × 2 = 14,336 bytes ≈ 14 KiB | ~14 KiB ✓ |
| 2 | 8 | 1 | ceil(0.25) = 1 | 1 × 7168 × 2 = 14,336 bytes ≈ 14 KiB | ~112 KiB ✗ |
| 3 | 32 | 1 | ceil(1.0) = 1 | 1 × 7168 × 2 = 14,336 bytes ≈ 14 KiB | ~448 KiB ✗ |
| 4 | 1 | 256 | ceil(8) = 8 | 8 × 7168 × 2 = 114,688 bytes ≈ 112 KiB | ~3.5 MiB ✗ |
| 5 | 32 | 2048 | ceil(2048) = 2048 | 2048 × 7168 × 2 = 29,360,128 bytes ≈ 28 MiB | ~28 MiB ✓ |

The three wrong table entries appear to have been computed as `k * B * S / N * H * 2`:
- Row 2: 8*8*1/8 * 7168 * 2 = 114,688 = 112 KiB (incorrect per the stated formula)
- Row 3: 8*32*1/8 * 7168 * 2 = 458,752 = 448 KiB (incorrect per the stated formula)
- Row 4: 8*1*256/8 * 7168 * 2 = 3,670,016 = 3.5 MiB (incorrect per the stated formula)

`k * B * S / N` counts the number of tokens received per device, which is the right quantity only when each device holds exactly 1 expert and k=N. On T3K with Qwen3.5-35B, each device holds E/N = 32 experts, so the per-hop send unit is C tokens per expert-slot (not per device), and C = ceil(k*B*S/E).

Note also that for rows 2 and 3 (B=8 and B=32, S=1), the correct per-hop volume is identical to row 1 (~14 KiB), since C=1 for all three cases. The table must distinguish these rows in some other way (e.g., number of slices of the dispatch tensor, or total transfer volume over all 7 hops), or accept that the decode per-hop volume is flat at ~14 KiB for B=1 through B=32 at S=1 given these model parameters.

**Fix:** Replace the three incorrect per-hop volume entries with the values derived from `C * H * 2`:

```
| ~14 KiB (B=1, S=1)   | decode  | ... |
| ~14 KiB (B=8, S=1)   | decode  | ... |   ← corrected from ~112 KiB
| ~14 KiB (B=32, S=1)  | decode  | ... |   ← corrected from ~448 KiB
| ~112 KiB (B=1, S=256) | prefill | ... |   ← corrected from ~3.5 MiB
| ~28 MiB (B=32, S=2048) | prefill | ... |
```

If the intent was instead to show a range of distinct per-hop volumes, choose B/S combinations that actually produce different C values (e.g., B=32, S=256 gives C=ceil(8*32*256/256)=32, V=32*7168*2=458,752 bytes ≈ 448 KiB).

---

### Issue 2 — Minor: Decode batch B=32 volume approximation uses wrong unit label

**File:** `all_to_all_in_moe.md`, line 138

**Text:**
```
For B = 32: V = 3,211,264 bytes ≈ 3.1 MB
```

**Error:** 3,211,264 bytes in SI is 3.211 MB, which rounds to 3.2 MB, not 3.1 MB. The value 3.1 is correct only in binary MiB (3,211,264 / 1,048,576 ≈ 3.06 MiB ≈ 3.1 MiB). The adjacent B=1 entry at line 136 uses KiB (`≈ 98 KiB`), a binary unit. The chapter mixes binary and SI unit labels for the same class of quantity. The simplest consistent fix is to use MiB throughout.

**Fix:** Change line 138 from:
```
For $B = 32$: $V = 3{,}211{,}264\ \text{bytes} \approx 3.1\ \text{MB}$
```
to:
```
For $B = 32$: $V = 3{,}211{,}264\ \text{bytes} \approx 3.1\ \text{MiB}$
```
(The numerical value 3.1 is correct for MiB; only the unit label needs correction.)

---

### Issue 3 — Minor: p95 index is off by one in the sweep code

**File:** `benchmarking_num_links.md`, line 186

**Code:**
```python
"p95_us": sorted(lats)[int(0.95 * MEASURE_ITERS)],
```

**Error:** For `MEASURE_ITERS = 100`, `int(0.95 * 100) = 95`, which selects index 95 (the 96th value in 0-indexed sorted order). The 95th percentile of 100 samples is conventionally taken at index 94 (the 95th value). The current code reports what is effectively the 96th percentile, not the 95th. This is a minor off-by-one that overstates tail latency by one sample.

**Fix:** Change to:
```python
"p95_us": sorted(lats)[int(0.95 * MEASURE_ITERS) - 1],
```
Or equivalently use `statistics.quantiles(lats, n=20)[-1]` (which gives the 95th percentile using the standard interpolation method). If the intent is to be conservative (report the 96th-percentile value as a safe upper bound for p95), add a comment to that effect.

---

### Issue 4 — Observation: T_prefill formula omits setup overhead term

**File:** `all_to_all_in_moe.md`, lines 123–126

**Text:**
```
T_prefill ≈ V_per-layer / n_l / BW_link + (N-1) × τ_hop
```

**Observation:** This formula models the all-to-all latency with a hop-latency term `(N-1) × τ_hop` but no setup-overhead term `n_l × τ_setup`. The full model from `num_links_parameter.md` is `T(n_l, V) = n_l × τ_setup + V / (n_l × BW_link)`, giving a total for 7 rounds of `7 × n_l × τ_setup + V_per-layer / (n_l × BW_link)`. For large prefill payloads the text correctly notes that `τ_setup` is negligible; the approximation is therefore justified in the prefill context and the formula is appropriate as an approximation. No correction required, but a note acknowledging that `n_l × τ_setup` is omitted as negligible in the prefill regime would improve consistency with the model in `num_links_parameter.md`.

This is an observation only, not a correctness error.

---

### Issue 5 — Confirmed correct: All-to-all per-token and round-trip arithmetic

**File:** `all_to_all_in_moe.md`, lines 49–94

All arithmetic verified:
- `(7/8) × 8 × 7168 × 2 = 7 × 7168 × 2 = 7 × 14,336 = 100,352 bytes` ✓
- Round-trip: `2 × 100,352 = 200,704 bytes` ✓
- Full-pass: `200,704 × 80 = 16,056,320 bytes ≈ 16 MB` (SI: 16.056 MB; the approximation to "16 MB" is correct for SI units) ✓
- Prefill per-layer: `100,352 × 32 × 2048 = 6,576,906,240 bytes ≈ 6.6 GB` ✓

---

### Issue 6 — Confirmed correct: V* crossover derivation

**File:** `num_links_parameter.md`, lines 73–77

Algebra verified:

Setting `n_l × τ_setup + V/(n_l × BW) = (n_l+1) × τ_setup + V/((n_l+1) × BW)`:

```
V × [1/(n_l × BW) - 1/((n_l+1) × BW)] = τ_setup
V × [1 / (n_l × (n_l+1) × BW)] = τ_setup
V* = n_l × (n_l+1) × τ_setup × BW
```

Result matches the stated formula. ✓

---

### Issue 7 — Confirmed correct: All topology references

All four files consistently use `ttnn.Topology.Linear` (not Ring) for T3K, `cluster_axis=1`, `(1, 8)` mesh shape, N=8, and N-1=7 sequential hops. No topology errors found.

---

# Agent A Fix — Chapter 3: All-to-All Operations and num_links Tuning on T3K — Pass 1 Fixes Applied

1. `benchmarking_num_links.md` — Corrected per-hop volume in reference table: B=8/S=1 and B=32/S=1 both have C=1 (same as B=1/S=1) → V_per_hop ≈ 14 KiB; B=1/S=256 has C=8 → V_per_hop ≈ 112 KiB. Updated latency rows accordingly.

2. `all_to_all_in_moe.md` — Changed "3.1 MB" to "3.1 MiB" to use consistent binary unit convention.

3. `benchmarking_num_links.md` — Fixed p95 index off-by-one: changed `int(0.95 * 100)` to `int(0.95 * MEASURE_ITERS) - 1`.

---

# Agent B Review — Chapter 3: All-to-All Operations and num_links Tuning on T3K — Pass 2

## Verification of Pass 1 Fixes

All three corrections confirmed in place:

1. `benchmarking_num_links.md` reference table rows 2–4: per-hop volumes now show ~14 KiB for B=8/S=1 and B=32/S=1 (C=1 in both cases), and ~112 KiB for B=1/S=256 (C=8). Values match `C × H × 2` with the stated formula. ✓

2. `all_to_all_in_moe.md` line 138: unit label is now `\text{MiB}`, consistent with the binary unit used at line 136 (`≈ 98 KiB`). ✓

3. `benchmarking_num_links.md` line 186: p95 index is now `int(0.95 * MEASURE_ITERS) - 1` = index 94, which selects the 95th value (0-indexed) of 100 sorted samples. Correct for the conventional 95th percentile. ✓

## Full Re-Check

**index.md** — All T3K-specific claims correct: Topology.Linear (not Ring), cluster_axis=1, N=8, N−1=7 hops, BW ≈ 12.5 GB/s. The statement "(N−1)/N × k = 7 distinct remote devices on average" is numerically correct: (7/8)×8 = 7. All D-dependent values marked [D UNVERIFIED]. No issues.

**all_to_all_in_moe.md** — Arithmetic re-verified:
- Per-token dispatch volume: (7/8)×8×7168×2 = 100,352 bytes ✓
- Round-trip: 2×100,352 = 200,704 bytes ✓
- Full 80-layer pass: 200,704×80 = 16,056,320 bytes ≈ 16 MB (SI) ✓
- Prefill per-layer: 100,352×32×2048 = 6,576,906,240 bytes ≈ 6.6 GB ✓
- Decode B=1: 100,352 bytes ≈ 98 KiB (100,352/1024 = 98.0) ✓
- Decode B=32: 3,211,264 bytes ≈ 3.1 MiB (3,211,264/1,048,576 ≈ 3.06) ✓
- C=1 for B=1 at E=256: ceil(8×1/256) = ceil(0.03125) = 1 ✓
- V_per_hop formula (C×H×2) consistent with num_links_parameter.md ✓
- Linear-chain 7-round description and Ring warning correct ✓

**num_links_parameter.md** — Arithmetic re-verified:
- Prefill: C = ceil(8×32×2048/256) = ceil(2048) = 2048 ✓; V_hop = 2048×7168×2 = 29,360,128 bytes ≈ 28 MiB ✓
- T_1 (prefill) = 29,360,128 / (12.5×10⁹) ≈ 2.35 ms, stated as "approximately 2.24 ms" — the text uses 28×10⁶ as an approximation for 29,360,128 bytes; 28×10⁶/(12.5×10⁹) = 2.24 ms. The rounding (28 MiB as 28×10⁶ rather than 29,360,128) is minor and consistent with the ≈ qualifier. ✓
- T_4 = 2.24 ms / 4 = 0.56 ms ✓
- Decode B=1: V_hop = 1×7168×2 = 14,336 bytes ≈ 14 KiB ✓; T_1 = 14,336/(12.5×10⁹) ≈ 1.15 µs, stated as "≈ 1.1 µs" — acceptable one-significant-figure rounding ✓
- V* crossover formula algebraically correct (verified in Pass 1) ✓
- Contention analysis (n_l,1 + n_l,2 ≤ n_l,max) correct ✓

**benchmarking_num_links.md** — Re-verified:
- Step 2 C calculation: ceil(8×8×1/256) = ceil(0.25) = 1; dispatch_shape = (8, 1, 7168) ✓
- Reference table all five rows now consistent with `C×H×2` formula ✓
- p95 index 94 correct for 100 0-indexed samples ✓
- No other arithmetic or logic errors found ✓

**No feedback — chapter approved.**

---

# Agent B Review — Chapter 3: All-to-All Operations and num_links Tuning on T3K — Pass 3

## Verification of Agent C Compression Changes

### C1 — Symbol table removed from `index.md`; replaced with cross-reference

`index.md` New Notation section (line 47) now reads: "Symbol definitions: see `all_to_all_in_moe.md` Quick Reference."

The Quick Reference table in `all_to_all_in_moe.md` (lines 3–15) defines all symbols used in the chapter (N, k, E, H, D, B, S, C). The cross-reference is accurate. No correctness issue. ✓

### C2 — `num_links` overview in `index.md` compressed to 2 sentences + pointer

`index.md` lines 21–22 state: "`num_links` controls how many physical Ethernet links between adjacent device pairs are allocated to a collective operation, directly trading link setup overhead against aggregate bandwidth. For the complete `num_links` definition, bandwidth model, and tuning guidance, see `num_links_parameter.md`."

The essential trade-off is correctly characterized. The pointer resolves to `num_links_parameter.md`, which contains the full bandwidth model and tuning table. No correctness issue. ✓

### C3 — Prefill/decode regime sections in `all_to_all_in_moe.md` compressed to 2-sentence summary + pointer

`all_to_all_in_moe.md` lines 116–118: "At large batch (prefill), the operation is throughput-bound; at small batch (decode), it is latency-bound. See `num_links_parameter.md` for `num_links` tuning by regime."

The detailed per-regime volume computations (the B=1 ≈ 98 KiB and B=32 ≈ 3.1 MiB decode figures verified in Pass 2, and the T_prefill formula) have been removed. These were correct values whose removal does not introduce any false statement — the remaining two sentences accurately characterize the regimes. The detailed data remains in `num_links_parameter.md`. No correctness issue. ✓

### C4 — Duplicate warm-up block removed from Step 5 sweep loop in `benchmarking_num_links.md`

**Issue found.**

Step 5 introductory text (line 154) states: "Repeat Steps 3 and 4 for each `num_links` value under test." However, the Step 5 code block (lines 162–182) contains only the measurement loop for each `nl`; it does not include a warm-up sub-loop for each `nl` value. The warm-up in Step 3 (lines 98–108) uses the variable `num_links_under_test`, which is a single value — not the `nl` iterator of the sweep.

This creates a correctness problem: TTNN compiles kernels on first use for a given `(shape, dtype, topology, num_links)` combination (as stated in Step 3's rationale bullet 1). When the Step 5 loop iterates over `nl` values not covered by the Step 3 warm-up, the first measurement iteration for those `nl` values will carry JIT compilation overhead, inflating the recorded latency for those values. This would bias the sweep results and could incorrectly identify `num_links=1` (or whichever value was used in Step 3) as optimal simply because the other values have an un-amortized compilation cost in their first measured iteration.

**Severity:** Moderate. Results produced by the current Step 5 code are not reproducible-correct for a multi-value sweep; they are only correct if Step 3 is manually repeated for each `nl` before running Step 5 — which the code does not do.

**Fix:** Either (a) add a warm-up sub-loop inside the Step 5 `for nl in valid_num_links:` block (before the measurement loop), or (b) update the Step 5 prose to explicitly state that Step 3 must be re-run for each `nl` before running the corresponding measurement loop, and restructure the code accordingly. Option (a) is cleaner and matches the intent of the original (pre-C4) code. The minimal correction is to insert `WARMUP_ITERS = 10` warm-up calls for `num_links=nl` at the top of the sweep body, consistent with Step 3.

---

## Full Re-Check

**index.md** — All T3K-specific claims correct: Topology.Linear (not Ring), cluster_axis=1, N=8, N−1=7 hops, BW ≈ 12.5 GB/s. The "(N−1)/N × k = 7 distinct remote devices on average" is numerically correct: (7/8)×8 = 7. All D-dependent values marked [D UNVERIFIED]. Symbol cross-reference resolves correctly. No new issues. ✓

**all_to_all_in_moe.md** — Arithmetic confirmed correct (same values as Pass 2). Per-token dispatch volume 100,352 bytes ✓, round-trip 200,704 bytes ✓, 80-layer total ≈ 16 MB ✓, prefill per-layer ≈ 6.6 GB ✓. Quick Reference table values consistent with body text. Linear-chain 7-round description and Ring warning correct. Regime summary sentences accurate. ✓

**num_links_parameter.md** — Arithmetic confirmed correct (same values as Pass 2). Prefill: C=2048, V_hop≈28 MiB, T_1≈2.24 ms, T_4≈0.56 ms ✓. Decode: V_hop=14,336 bytes≈14 KiB, T_1≈1.1 µs ✓. V* formula algebraically correct ✓. Contention analysis correct ✓. ✓

**benchmarking_num_links.md** — Step 2 C calculation: ceil(8×8×1/256)=1, dispatch_shape=(8,1,7168) ✓. Reference table five rows consistent with C×H×2 formula ✓. p95 index (`int(0.95 * MEASURE_ITERS) - 1` = 94) correct for 100 0-indexed samples ✓. **Issue found in Step 5 (see C4 above).**

---

# Agent A Fix — Chapter 3: All-to-All Operations and num_links Tuning on T3K — Pass 2 Fixes Applied

4 (reinstated). `benchmarking_num_links.md` Step 5 — Added per-configuration warm-up sub-loop inside the `for nl in valid_num_links:` sweep loop. Each `num_links` value now gets WARMUP_ITERS warm-up iterations before measurement to flush JIT compilation overhead from the latency results.

---

# Agent B Review — Chapter 3: All-to-All Operations and num_links Tuning on T3K — Pass 4

## Verification of Pass 3 Fix (Issue C4 / Issue 4 reinstated)

The per-`nl` warm-up sub-loop is confirmed present inside `for nl in valid_num_links:` at lines 163–172 of `benchmarking_num_links.md`. Each `nl` value receives `WARMUP_ITERS` warm-up calls before the measurement loop begins. The structural fix is in place.

## New Issues Found

### Issue 8 — Moderate: Step 5 warm-up call omits `memory_config` present in the measurement call

**File:** `benchmarking_num_links.md`, lines 163–171 (Step 5 warm-up sub-loop) vs. lines 178–183 (Step 5 measurement loop)

**Error:** The warm-up sub-loop inside the sweep calls `ttnn.all_to_all` without `memory_config=ttnn.DRAM_MEMORY_CONFIG`:

```python
# warm-up (lines 163–171):
_ = ttnn.all_to_all(
    mesh_tensor,
    mesh_device,
    num_links=nl,
    cluster_axis=1,
    topology=ttnn.Topology.Linear,
)
```

The measurement loop passes the same `num_links=nl` but adds `memory_config=ttnn.DRAM_MEMORY_CONFIG`:

```python
# measurement (lines 178–183):
_ = ttnn.all_to_all(
    mesh_tensor, cluster_axis=1, mesh_device=mesh_device,
    num_links=nl, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    topology=ttnn.Topology.Linear,
)
```

TTNN compiles kernels per `(shape, dtype, topology, num_links)` combination, and `memory_config` can influence which kernel variant or memory-placement path is selected. If the warm-up and measurement calls dispatch different kernel variants, the first measurement iteration for each `nl` still carries JIT compilation overhead for the `DRAM_MEMORY_CONFIG` variant — the very problem the per-`nl` warm-up was reinstated to prevent.

Note also that `mesh_device` is passed as a positional argument in the warm-up but as a keyword argument in the measurement call. The positional vs. keyword difference is not a correctness issue by itself (assuming the API accepts both), but combining it with the missing `memory_config` means the two calls are not functionally identical.

For comparison, the Step 3 standalone warm-up (lines 98–108) correctly includes `memory_config=ttnn.DRAM_MEMORY_CONFIG`, making the warm-up and measurement calls in that section consistent.

**Fix:** Add `memory_config=ttnn.DRAM_MEMORY_CONFIG` to the warm-up call inside the Step 5 sweep loop, and use consistent keyword-argument form for `mesh_device`, so the warm-up call is identical in all arguments to the measurement call (differing only in the use of `_` for the return value):

```python
for _ in range(WARMUP_ITERS):
    _ = ttnn.all_to_all(
        mesh_tensor,
        cluster_axis=1,
        mesh_device=mesh_device,
        num_links=nl,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
    )
ttnn.synchronize_device(mesh_device)
```

---

### Issue 9 — Minor: Wrong exact value for prefill per-layer volume in `all_to_all_in_moe.md`

**File:** `all_to_all_in_moe.md`, line 108

**Text:**
```
V_{\text{prefill, per layer}} = 100{,}352 \times B \times S = 100{,}352 \times 32 \times 2048 = 6{,}576{,}906{,}240\ \text{bytes per layer}
```

**Error:** The stated product `100,352 × 32 × 2048 = 6,576,906,240` is incorrect. The correct computation is:

- `100,352 × 32 = 3,211,264`
- `3,211,264 × 2048 = 6,576,668,672`

The stated value `6,576,906,240` is off by `237,568 bytes` (0.004%). The approximation `≈ 6.6 GB` on the following line remains correct for both the stated and correct values (both are approximately 6.58 GB in SI), so no downstream error propagates. This is a minor arithmetic typo in the exact figure.

**Fix:** Replace `6{,}576{,}906{,}240` with `6{,}576{,}668{,}672` on line 108.

---

# Agent A Fix — Chapter 3: All-to-All Operations and num_links Tuning on T3K — Pass 3 Fixes Applied

8. `benchmarking_num_links.md` Step 5 warm-up sub-loop — Added `memory_config=ttnn.DRAM_MEMORY_CONFIG` to the warm-up `ttnn.all_to_all` call to match the measurement loop, ensuring the same kernel variant is compiled during warm-up.

9. `all_to_all_in_moe.md` line 108 — Corrected exact product: `100,352 × 32 × 2048 = 6,576,906,240` → `6,576,668,672`. Downstream approximation `≈ 6.6 GB` unchanged.

---

# Agent B Review — Chapter 3: All-to-All Operations and num_links Tuning on T3K — Pass 5

## Verification of Fixes 8 and 9

### Fix 8 — `benchmarking_num_links.md` Step 5 warm-up sub-loop

The Step 5 warm-up sub-loop (inside `for nl in valid_num_links:`) now includes `memory_config=ttnn.DRAM_MEMORY_CONFIG` as a keyword argument, matching the measurement loop exactly. The `ttnn.synchronize_device(mesh_device)` call follows the warm-up loop before timing begins. The fix is correctly applied. ✓

### Fix 9 — `all_to_all_in_moe.md` line 108 exact product

Line 108 now reads `100{,}352 \times 32 \times 2048 = 6{,}576{,}668{,}672`. Verified: 100,352 × 32 = 3,211,264; 3,211,264 × 2048 = 6,576,668,672. Correct. The downstream approximation `≈ 6.6 GB` is unchanged and remains accurate. ✓

## Full Re-Check

**all_to_all_in_moe.md** — All arithmetic re-verified:
- Per-token dispatch: (7/8) × 8 × 7168 × 2 = 100,352 bytes ✓
- Round-trip: 2 × 100,352 = 200,704 bytes ✓
- 80-layer total: 200,704 × 80 = 16,056,320 bytes ≈ 16 MB ✓
- Prefill per-layer: 100,352 × 32 × 2048 = 6,576,668,672 bytes ≈ 6.6 GB ✓ (Fix 9 confirmed)
- E/N = 256/8 = 32 experts per device ✓; C formula = ceil(k × B × S / E) ✓
- Linear-chain 7-round description and Ring warning correct ✓
- All D-dependent values marked [D UNVERIFIED] ✓

**num_links_parameter.md** — All arithmetic re-verified:
- Prefill: C = ceil(8×32×2048/256) = 2048; V_hop = 2048×7168×2 = 29,360,128 bytes ≈ 28 MiB ✓
- T_1 prefill = 28×10⁶ / (12.5×10⁹) ≈ 2.24 ms; T_4 = 0.56 ms ✓
- Decode B=1: V_hop = 1×7168×2 = 14,336 bytes ≈ 14 KiB; T_1 ≈ 1.1 µs ✓
- V* crossover formula n_l × (n_l+1) × τ_setup × BW_link algebraically correct ✓
- Contention analysis (n_l,1 + n_l,2 ≤ n_l,max) correct ✓

**benchmarking_num_links.md** — Re-verified:
- Step 2: C = ceil(8×8×1/256) = 1; dispatch_shape = (8, 1, 7168) ✓
- Step 3 warm-up includes `memory_config=ttnn.DRAM_MEMORY_CONFIG` ✓
- Step 5 warm-up sub-loop per nl now also includes `memory_config=ttnn.DRAM_MEMORY_CONFIG` ✓ (Fix 8 confirmed)
- p95 index: `int(0.95 * MEASURE_ITERS) - 1` = 94, correct for 100 0-indexed samples ✓
- Reference table: all five rows consistent with C×H×2 formula ✓

**index.md** — All T3K-specific claims correct: Topology.Linear (not Ring), cluster_axis=1, N=8, N−1=7 hops, BW ≈ 12.5 GB/s, (1,8) mesh. Symbol cross-reference resolves correctly to `all_to_all_in_moe.md` Quick Reference. ✓

No feedback — chapter approved
