# B Review — Chapter 6: Performance Analysis and Trade-offs — Pass 1

## Issue 1 — Wrong Qwen crossover `effective_M` value (`bandwidth_gain_analysis.md`, crossover table)

The table lists Qwen 235B-A22B's crossover `effective_M` as **~448**, but the chapter's own formula gives a different result.

The formula used for Mixtral is:

```
effective_M_crossover = ridge × N / (N − ridge)
                      = 437 × 14336 / (14336 − 437)
                      = 6,264,832 / 13,899
                      ≈ 451   ✓  (matches table)
```

Applying the same formula to Qwen (`d_ff = 2048`, used as N):

```
effective_M_crossover = 437 × 2048 / (2048 − 437)
                      = 895,376 / 1611
                      ≈ 556   ✗  (table shows ~448)
```

~448 is wrong; the self-consistent value from the chapter's own formula is ~556. The table entry for Qwen's crossover `effective_M` must be corrected.

---

## Issue 2 — L1 weight footprint formula incorrectly includes `M_t` (`tradeoff_matrix.md`, "Interaction with L1 Memory Pressure")

The chapter states:

```
L1_weight_footprint = 2 × in0_block_w × M_t × tile_size_bytes
```

with `in0_block_w = 8 tiles`, `M_t = 128 tile rows for d_model = 4096`, BF16 tile = 2048 bytes, yielding 4 MB.

`M_t` is the number of output (or activation) tile rows; it is not a component of the weight double-buffer. The weight double-buffer holds two blocks of K-direction × N-direction weight tiles. For streaming weight tiles from DRAM, the correct formula is:

```
L1_weight_double_buffer = 2 × in0_block_w × in1_block_w × tile_size_bytes
```

Including `M_t = 128` inflates the computed value by 128×, from the correct order of ~32 KB (with `in1_block_w = 1`) up to 4 MB, and makes it appear that direct DRAM-sharded access always exceeds the 1.5 MB L1 limit. This conclusion is materially wrong given the actual formula.

---

## Issue 3 — Narrative says "24 weight tensors" when the total across all layers is 768 (`shard_setup_overhead.md`, "Reshard Latency Estimate")

The text reads:

> "For Mixtral 8x7B with 8 experts × 3 projections = 24 weight tensors across 32 layers: `total_reshard_time ≈ 24 × 32 × 3 ms ≈ 2.3 seconds`"

The phrase "24 weight tensors across 32 layers" is factually wrong. 24 is the per-layer count (8 experts × 3 projections). The total number of weight tensors that must be resharded is 8 × 3 × 32 = **768**. The formula `24 × 32 × 3 ms = 2,304 ms ≈ 2.3 s` is arithmetically correct, but the narrative description — "24 weight tensors" — misrepresents the total and will mislead a reader who is not computing it manually. The narrative should state "768 weight tensors total (24 per layer × 32 layers)."

---

## Agent A Change Log — B Feedback Pass 1
- bandwidth_gain_analysis.md: Fixed Qwen crossover effective_M from ~448 to ~556 (437 × 2048 / 1611 ≈ 556)
- tradeoff_matrix.md: Removed M_t from L1 weight footprint formula; updated computed examples
- shard_setup_overhead.md: Clarified "24 weight tensors per layer (768 total across 32 layers)"

---

# B Review — Chapter 6: Performance Analysis and Trade-offs — Pass 2

## Pass 1 Fix Verification

All three Pass 1 fixes are confirmed applied correctly:

1. `bandwidth_gain_analysis.md` crossover table: Qwen effective_M now reads `~556`. Formula check: 437 × 2048 / (2048 − 437) = 895,376 / 1,611 ≈ 556.0. Correct.
2. `tradeoff_matrix.md` L1 formula: now reads `2 × in0_block_w × per_core_N_t × tile_size_bytes`; example computes 2 × 8 × 1 × 2048 = 32,768 bytes = 32 KB. Correct.
3. `shard_setup_overhead.md` line 169: now reads "24 weight tensors per layer (8 experts × 3 projections each), totaling 768 tensors across 32 layers." Correct.

---

## Issue 1 — `index.md` decision table: "Prefill, small batch" boundary definition contradicts `tradeoff_matrix.md`

**File:** `index.md`, line 33 (decision table row for "Prefill, small batch")

**The error:** The decision table in `index.md` defines the prefill-small-batch regime as `≤ 16 × top_k active tokens`. For Mixtral (top_k=2) this gives a boundary of 32; for Qwen (top_k=8) it gives 128. These values do not match the regime boundaries established in `tradeoff_matrix.md`, which defines "Prefill, small batch" as `64 < effective_M ≤ 256` — a boundary that is independent of top_k and consistent across both model-specific tables in that file.

Mixtral example: `index.md` boundary implies `effective_M ≤ 32` for the prefill-small regime, but `tradeoff_matrix.md` starts that regime at `effective_M > 64`. A Mixtral run with effective_M=50 falls in the `index.md` prefill-small bucket but in the decode-large bucket of `tradeoff_matrix.md`. The two files give contradictory layout recommendations for that case.

**Correct definition:** The "Prefill, small batch" boundary should read `64 < batch_size × top_k ≤ 256` (consistent with `tradeoff_matrix.md`).

---

## Issue 2 — `index.md` decision table: latency delta for "Prefill, small batch" contradicts `tradeoff_matrix.md`

**File:** `index.md`, line 33; `tradeoff_matrix.md`, line 13

**The error:** `index.md` states the indicative latency delta for the "Prefill, small batch" regime as `−5 to −15%`. `tradeoff_matrix.md` states the same regime (c) as `−5 to −10%`. The upper bound differs by 5 percentage points. The two files are inconsistent on the same quantity; a reader comparing them gets different numbers with no explanation. `tradeoff_matrix.md` is the authoritative trade-off reference for this chapter; `index.md` should match it.

**Correct value:** `−5 to −10%` (matching `tradeoff_matrix.md` regime (c)).

---

No further errors found in `bandwidth_gain_analysis.md`, `shard_setup_overhead.md`, or the remaining content of `tradeoff_matrix.md`. All numerical derivations checked (Mixtral weight bytes, total decode weight bytes, decode latency estimates at 200 and 270 GB/s, reshard latency bounds, T3K per-chip token-pair estimate) are internally consistent with the stated authoritative facts.

## Agent A Change Log — B Feedback Pass 2
- index.md: Fixed "Prefill, small batch" boundary from `≤ 16 × top_k active tokens` to `64 < batch_size × top_k ≤ 256`
- index.md: Fixed latency delta for "Prefill, small batch" from `-5 to -15%` to `-5 to -10%`

---

# B Review — Chapter 6: Performance Analysis and Trade-offs — Pass 3

## Pass 2 Fix Verification

Both Pass 2 fixes are confirmed applied correctly in `index.md`:

1. **Boundary fix (`index.md`, line 33):** The "Prefill, small batch" row now reads `64 < batch_size × top_k ≤ 256`. This matches the authoritative definition and is consistent with `tradeoff_matrix.md` regime (c). Fix is correct.
2. **Latency delta fix (`index.md`, line 33):** The "Prefill, small batch" latency delta now reads `−5 to −10%`. This matches `tradeoff_matrix.md` regime (c) and the authoritative fact. Fix is correct.

---

## Errors Found

### Error 1 — `index.md` decision table: "Prefill, large batch" boundary contradicts `tradeoff_matrix.md`

**File:** `index.md`, line 34
**Stated:** `> 512 active tokens` (boundary for "Prefill, large batch")
**Correct:** `effective_M > 256` — `tradeoff_matrix.md` regime (d) defines the large-batch prefill threshold as `effective_M > 256`, not `> 512`. The two files are inconsistent on the same boundary: at `effective_M = 300`, `index.md` places the workload in "Prefill, small batch" while `tradeoff_matrix.md` places it in "Prefill, large batch" with the recommendation to use interleaved layout. The gap between 256 and 512 is factually inconsistent.

---

### Error 2 — `index.md` vs `tradeoff_matrix.md`: "Decode, large batch" latency delta upper bound differs

**File:** `index.md`, line 32 and `tradeoff_matrix.md`, line 12
**Stated in `index.md`:** `−10 to −20%` for "Decode, large batch"
**Stated in `tradeoff_matrix.md`:** `−10 to −25%` for regime (b) "Decode, large batch"
**Correct:** The two files are inconsistent on the upper bound (20% vs 25%) for the same regime. `tradeoff_matrix.md` is the authoritative trade-off reference for this chapter (as established in the Pass 2 review); `index.md` should match it. The value in `index.md` understates the upper bound by 5 percentage points.

---

## Verdict

Two factual errors remain. Both are in `index.md` and both are cross-file inconsistencies where `index.md` diverges from the authoritative `tradeoff_matrix.md` on regime boundaries and latency figures.

## Agent A Change Log — B Feedback Pass 3
- index.md: Fixed "Prefill, large batch" boundary from `> 512 active tokens` to `effective_M > 256`
- index.md: Fixed "Decode, large batch" latency delta from `-10 to -20%` to `-10 to -25%`

# B Review — Chapter 6: Performance Analysis and Trade-offs — Pass 4

## Pass 3 Fix Verification

1. **"Prefill, large batch" boundary (`index.md`, decision table):** Now reads `effective_M > 256`. Matches `tradeoff_matrix.md` regime (d): `effective_M > 256`. Fix confirmed correct.
2. **"Decode, large batch" latency delta (`index.md`, decision table):** Now reads `−10 to −25%`. Matches `tradeoff_matrix.md` regime (b): `−10 to −25%`. Fix confirmed correct.

## Verdict

No feedback — chapter approved.

All four content files are internally consistent and consistent with the authoritative facts. Full checks performed:
- Ridge point ~437 FLOP/byte: confirmed in `bandwidth_gain_analysis.md` and `index.md` Key Constants table.
- Qwen crossover effective_M ~556 (437 × 2048 / 1611): confirmed in `bandwidth_gain_analysis.md` crossover table.
- Mixtral crossover effective_M ~451: confirmed correct.
- L1 weight footprint formula `2 × in0_block_w × per_core_N_t × tile_size_bytes` (no M_t): confirmed in `tradeoff_matrix.md`.
- Tensor count 24 per layer, 768 total across 32 layers: confirmed in `shard_setup_overhead.md` line 169.
- All four `index.md` decision table rows now match the corresponding `tradeoff_matrix.md` regime definitions and latency figures exactly.
- Decode latency arithmetic (672 MB at 200 GB/s ≈ 3.4 ms; at 270 GB/s ≈ 2.5 ms) and reshard latency arithmetic (24 × 32 × 3 ms ≈ 2.3 s) are both correct.
