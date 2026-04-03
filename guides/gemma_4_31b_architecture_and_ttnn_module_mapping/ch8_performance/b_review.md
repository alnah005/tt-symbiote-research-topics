# Agent B Review: Chapter 8

## Pass 1

### Issue 1: Weight memory totals inconsistent between index.md and memory_budget.md

`index.md` line 30 states "Weight memory (BFP8, per device) | ~4,143 MB", matching Ch6's figure. However, `memory_budget.md` lines 49-56 compute the total as ~4,156 MB (50 sliding at 3,005 MB + 10 global at 765 MB + 336 MB embedding + 50 MB misc = 4,156 MB). The DRAM budget table in `memory_budget.md` (lines 170-178) also uses 4,156 MB as the weight column. The 13 MB discrepancy arises because `memory_budget.md` rounds each per-layer BFP8 total up (60.1 MB sliding, 76.5 MB global) before multiplying by layer count, while Ch6 (`kv_cache_sharding.md` line 207) sums from exact per-projection bytes and arrives at 4,143 MB. One file must be corrected to match the other. Ch6's bottom-up calculation is more precise, so `memory_budget.md` should adopt 4,143 MB (and adjust the per-layer BFP8 totals or note rounding).

**Files:** `index.md` line 30, `memory_budget.md` lines 51-56 and 170-178.

### Issue 2: "BFP8 weights + BFP8 KV at ~131K" headroom stated as ~2.7 GB in index.md, but memory_budget.md computes ~2.9 GB

`index.md` line 36 says "Max sequence length (BFP8 weights + BFP8 KV) | ~131K | ~2.7 GB headroom". The corresponding entry in `memory_budget.md` line 185 shows total = 9,328 MB at S=131,072, giving headroom = 12,288 - 9,328 = 2,960 MB = ~2.89 GB, not ~2.7 GB. The index.md value should be corrected to ~2.9 GB.

**File:** `index.md` line 36.

### Issue 3: "BF16 weights + BF16 KV | ~8K | ~4.4 GB headroom" is arithmetically wrong

`memory_budget.md` line 192 claims the BF16+BF16 configuration supports ~8K context with ~4.4 GB headroom. Using the file's own numbers: weights = 7,899 MB, KV at S=8,192 = 740 MB, activations ~2 MB, total ~8,641 MB. Headroom = 12,288 - 8,641 = 3,647 MB = ~3.6 GB, not ~4.4 GB. Even at S=2,048 (weights 7,899 + KV 260 + 2 = 8,161 MB), headroom is only ~4.0 GB. No sequence length produces ~4.4 GB headroom under BF16 weights + BF16 KV. The headroom figure should be corrected to ~3.6 GB (at S=8K) or the max context revised downward.

**File:** `memory_budget.md` line 192.

### Issue 4: Global SDPA statement in index.md says "256 MB per layer per device" at S=32,768 but this is the KV cache read, not the KV cache size per layer

`index.md` lines 47-48 state: "At S=32,768, each global SDPA reads a KV cache of 256 MB per layer per device (BF16). This is 2,560 MB total across 10 global layers." Per Ch6, the KV cache size per global layer per device at S=32,768 is 256 MB (K+V combined). This number is correct. However, the phrasing "reads a KV cache of 256 MB" is accurate only if the entire cache is read during SDPA, which is indeed the case at decode (full causal attention). No correction needed on the number, but flagging for completeness -- the statement is technically correct.

**Verdict:** Not an error. Withdrawn.

### Issue 4 (replacement): Fused gate+up latency listed as ~48 us in sliding layer summary, but two projections fused should total ~96 us bandwidth, saving only launch overhead

`decode_latency_analysis.md` line 162 lists "Gate+Up projection (fused) | ~48 us" for the sliding layer. Each of gate and up is 14.5 MB at BFP8, totaling 29.0 MB fused. At 300 GB/s, 29.0 MB / 300 GB/s = ~96.7 us. The unfused table (lines 35-36) correctly lists each as ~48 us. A fused gate+up should still read ~29 MB and take ~96 us of bandwidth time (minus one kernel launch, ~5 us savings). The per-layer summary appears to count only one of the two projections (48 us instead of ~96 us), understating the fused gate+up cost by ~48 us. This would make the sliding layer total ~226 us, not 178 us, and propagate into all total-decode-latency figures.

**File:** `decode_latency_analysis.md` lines 162-167, and the total decode latency table at lines 193-199.

### Summary

Three arithmetic/consistency errors found (Issues 1, 2, 3), plus one potentially significant latency undercount (Issue 4 replacement) that would affect all downstream decode throughput projections. Issue 4 is the most consequential: if the fused gate+up bandwidth cost is ~96 us rather than ~48 us, the baseline decode latency at S=8K rises from ~13.3 ms to ~15.7 ms, and all optimization projections shift accordingly.

## Pass 2

All four Pass 1 issues have been resolved:

1. **Weight memory inconsistency (Issue 1):** `memory_budget.md` now uses 4,143 MB consistently throughout, with a rounding note (lines 58-62) explaining the discrepancy between rounded per-layer totals and Ch6's authoritative bottom-up figure.
2. **BFP8+BFP8 headroom misquote (Issue 2):** `index.md` line 36 now correctly states ~2.9 GB headroom (12,288 - 9,315 = 2,973 MB).
3. **BF16 headroom (Issue 3):** `memory_budget.md` line 198 now correctly states ~3.6 GB headroom (12,288 - 8,641 = 3,647 MB).
4. **Fused gate+up latency (Issue 4):** `decode_latency_analysis.md` line 164 now lists ~96 us for fused gate+up. The sliding layer total is ~226 us (line 169), and all total decode latency figures are consistent: 50 x 226 us + 10 x T_global gives 16.2 ms at S=8K and 22.6 ms at S=32K. Tokens/s figures also check out.

Verified arithmetic across all four files. Per-layer component latencies sum correctly to per-layer totals, per-layer totals multiply correctly to full-model totals, DRAM headroom figures match (total subtracted from 12,288 MB), and the index.md summary table is consistent with the detail files.

**No feedback --- chapter approved.**
