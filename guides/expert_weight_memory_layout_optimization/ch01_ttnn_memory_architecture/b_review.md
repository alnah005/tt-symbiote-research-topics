# B Review — Chapter 1: TTNN Memory Architecture — Pass 1

1. **`wormhole_memory_hierarchy.md`, line 46 — Wrong Tensix core grid (64 vs 80 cores)**
   The file states "The full Wormhole B0 Tensix grid is 8 columns × 8 rows = 64 Tensix cores" and derives "64 cores × 1.5 MB = 96 MB nominal". Every other document in this repository (t3k guide, expert_parallelism_strategies, moe_optimization_techniques) consistently describes Wormhole B0 as an **8×10 = 80 core grid** with **120 MB aggregate L1**. A reader using the 64-core figure would underestimate the available compute grid and compute wrong per-core shard dimensions (e.g., for an 80-core HEIGHT_SHARDED layout they would compute ceil(rows/64) instead of ceil(rows/80)). Fix: change "8 columns × 8 rows = 64 Tensix cores" to "8 columns × 10 rows = 80 Tensix cores"; update the aggregate calculation to "80 cores × 1.5 MB = 120 MB nominal"; update the harvested-row range to "a production card typically exposes a 7×10, 8×9, or similar usable grid".

2. **`wormhole_memory_hierarchy.md`, line 64 — "12 controllers total" contradicts the 6-controller architecture**
   The bandwidth table note reads "~32 GB/s | 12 controllers total; aggregate theoretical ~384 GB/s". However line 5 of the same file establishes that Wormhole B0 has **six** DRAM controllers (not twelve). The "12 controllers" label conflates banks with controllers. This note would cause a reader to calculate the wrong aggregate bandwidth (384 GB/s vs. the 300-320 GB/s stated in the adjacent row) and to misunderstand the memory topology. Fix: change "12 controllers total; aggregate theoretical ~384 GB/s" to "6 controllers total, 2 banks each; aggregate theoretical ~300–320 GB/s".

3. **`wormhole_memory_hierarchy.md`, line 64 — Per-controller bandwidth figure (~32 GB/s) is inconsistent with the aggregate**
   With 6 controllers and a stated aggregate of ~300–320 GB/s (line 65), per-controller bandwidth is ~50–53 GB/s. The ~32 GB/s figure in the table implicitly assumes 12 controllers (12 × 32 = 384 GB/s), which is wrong as noted in item 2. A reader building a performance model would use 32 GB/s per controller and compute incorrect bandwidth estimates for kernels that read from a single DRAM bank. Fix: change "~32 GB/s" to "~50 GB/s" (consistent with 6 controllers × 50 GB/s ≈ 300 GB/s aggregate).

---

# B Review — Chapter 1: TTNN Memory Architecture — Pass 2

**Pass 1 fixes verified:** All three items from Pass 1 are correctly applied in the current files.
- Tensix grid is now "8 columns × 10 rows = 80 Tensix cores" with "80 cores × 1.5 MB = 120 MB nominal" and a corrected harvested-row range of "7×10, 8×9, or similar".
- Bandwidth table now reads "6 controllers total, 2 banks each; aggregate theoretical ~300–320 GB/s".
- Per-controller figure is now "~50 GB/s".

**Remaining correctness issue:**

1. **`memory_config_api.md`, WIDTH_SHARDED example — CoreRange comment is transposed (1 row, 8 columns vs. 1 column, 8 rows)**

   The code constructs `CoreRange(CoreCoord(0, 0), CoreCoord(0, 7))` and the inline comment says `# 1 row, 8 columns → 8 shards`. In TTNN's `CoreCoord(x, y)` convention, `x` is the column index and `y` is the row index. A range from `CoreCoord(0, 0)` to `CoreCoord(0, 7)` spans column 0 to column 0 and row 0 to row 7 — that is **1 column, 8 rows**, not "1 row, 8 columns." The shard count of 8 is numerically correct, so the downstream `shard_spec.shape` math is unaffected. However, a reader who copies this pattern and tries to reason about or extend the core grid (e.g., converting to a 2D BLOCK_SHARDED grid, or adding a second column of cores) will build a wrong mental model of which axis they are varying and may construct incorrect `CoreRange` bounds for grids that are not 1×N. Fix: change the comment to `# 1 column, 8 rows → 8 shards`.

**No further correctness issues found.** The physical topology, bandwidth figures, API signatures, shard shape arithmetic, and the reshard code pattern are all internally consistent and correct.

# B Review — Chapter 1: TTNN Memory Architecture — Pass 3

**Pass 2 fix verified:** `memory_config_api.md` line 158 now reads `# 1 column, 8 rows → 8 shards`, correctly describing `CoreRange(CoreCoord(0, 0), CoreCoord(0, 7))` as 1 column spanning 8 rows. The fix is correctly applied.

No feedback — chapter approved.
