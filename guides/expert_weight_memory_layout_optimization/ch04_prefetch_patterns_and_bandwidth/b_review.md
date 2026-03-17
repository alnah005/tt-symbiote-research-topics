# B Review — Chapter 4: Prefetch Patterns and Bandwidth — Pass 1

## Finding 1 — ROW_MAJOR and COL_MAJOR traversal descriptions are swapped (`sharded_access_pattern.md`)

In `sharded_access_pattern.md` the traversal orders for the two orientations are inverted relative to their names.

The ROW_MAJOR section (line 39) describes: core (0,0) → (0,1) → ... → (0,7) → (1,0). In ttnn `CoreCoord(x, y)` notation this increments y (the row) while holding x constant, which is column-major scanning (down a column first).

The COL_MAJOR section (line 53) describes: core (0,0) → (1,0) → ... → (7,0) → (0,1). This increments x (the column) while holding y constant, which is row-major scanning (across a row first).

The two descriptions are reversed. ROW_MAJOR should scan across x first: (0,0), (1,0), ..., (7,0), (0,1), ...; COL_MAJOR should scan down y first: (0,0), (0,1), ..., (0,9), (1,0), ...

## Finding 2 — Prefill arithmetic intensity is miscalculated (`bandwidth_estimation.md`)

At line 110-111, the chapter applies its own formula `M*N / (N+M)` with M=512, K=4096, N=14336:

```
512 * 14336 / (14336 + 512) = 7,340,032 / 14,848 ≈ 494 FLOP/byte
```

The chapter states ≈ 489 FLOP/byte. The correct result using the formula shown is approximately 494 FLOP/byte, not 489. The conclusion (marginally compute-bound, since 494 > 437) is unaffected, but the numerical value is wrong by about 1%.

## Finding 3 — Double-buffer transfer time is inconsistent with stated shard size (`sharded_access_pattern.md`)

Line 64 states: "At ~300 GB/s, transferring 768 KB takes ~2.56 µs."

768 KB = 786,432 bytes. At 300 GB/s: 786,432 / 300,000,000,000 ≈ 2.62 µs, not 2.56 µs. The value 2.56 µs would correspond to transferring approximately 750 KB (768,000 bytes), not 768 KB. The correct figure for 768 KB at 300 GB/s is ~2.62 µs.

## Agent A Change Log — B Feedback Pass 1
- sharded_access_pattern.md: Fixed ROW_MAJOR/COL_MAJOR descriptions — ROW_MAJOR scans x-first (same row), COL_MAJOR scans y-first (same column)
- bandwidth_estimation.md: Fixed prefill AI from ~489 to ~494 FLOP/byte (M=512, N=14336 in formula M*N/(N+M))
- sharded_access_pattern.md: Fixed double-buffer transfer time from ~2.56 µs to ~2.62 µs (768 KB / 300 GB/s = 2.62 µs)

---

# B Review — Chapter 4: Prefetch Patterns and Bandwidth — Pass 2

## Pass 1 Fix Verification

**Fix 1 — ROW_MAJOR/COL_MAJOR descriptions (`sharded_access_pattern.md` line 39, 53):** Verified correct. ROW_MAJOR now reads x-first: (0,0)→(1,0)→...→(7,0)→(0,1)→... COL_MAJOR now reads y-first: (0,0)→(0,1)→...→(0,7)→(1,0)→... Both match the canonical definitions.

**Fix 2 — Prefill AI (`bandwidth_estimation.md` line 111):** Verified correct. 512×14336/(14336+512) = 7,340,032/14,848 ≈ 494.3 FLOP/byte. Value now reads ~494.

**Fix 3 — Double-buffer transfer time (`sharded_access_pattern.md` line 64):** Verified correct. 768 KB = 786,432 bytes; 786,432/300,000,000,000 ≈ 2.621 µs → ~2.62 µs. Value now reads ~2.62 µs.

## New Finding — COL_MAJOR column-height example is inconsistent with the 8×10 grid (`sharded_access_pattern.md` line 53)

The COL_MAJOR description reads: "core (0,0) gets shard 0, core (0,1) gets shard 1, ..., core (0,7) gets shard 7, then core (1,0) gets shard 8, etc."

The Wormhole B0 Tensix grid is 8 columns × 10 rows (x=0..7, y=0..9). Each column therefore contains 10 cores (y=0..9). Under column-major scanning, all 10 cores in column 0 are assigned shards 0–9 before moving to column 1. The example implies the column is only 8 rows tall (stopping at y=7 before wrapping to (1,0)), which is wrong for a 10-row grid.

The correct sequence for a full 8×10 grid is: core (0,0)→shard 0, (0,1)→shard 1, ..., (0,9)→shard 9, then (1,0)→shard 10, etc. The example should show (0,9) as the last core before the column wraps to (1,0), not (0,7).

## Agent A Change Log — B Feedback Pass 2
- sharded_access_pattern.md: Fixed COL_MAJOR column-height example — column wraps after core (0,9)→shard 9 to (1,0)→shard 10 (Wormhole B0 grid has 10 rows y=0..9, not 8)

---

# B Review — Chapter 4: Prefetch Patterns and Bandwidth — Pass 3

## Pass 2 Fix Verification

**Fix — COL_MAJOR column-height example (`sharded_access_pattern.md` line 53):** Verified correct. The text now reads: "core (0,0) gets shard 0, core (0,1) gets shard 1, ..., core (0,9) gets shard 9, then core (1,0) gets shard 10, etc." The column wraps after y=9 (all 10 rows exhausted) before advancing to x=1. This is consistent with the Wormhole B0 8×10 grid (x=0..7, y=0..9) and matches the authoritative COL_MAJOR definition.

## No feedback — chapter approved.
