# Chapter 6 Corrections Change Log

## Fix 1: FLOPs comparison in sharding_strategy_analysis.md (~line 249-251)

**Issue:** "176M FLOPs on a single device" was misleading for the Q projection comparison. With TP=8, Q projection FLOPs are also split across devices, so the per-device figure is 22M (same as K), not 176M.

**Change:** Clarified that 176M FLOPs is the total across 8 devices (22M per device), and noted that the Q projection comparison is also 176M total / 22M per device after TP=8 sharding.

## Fix 2: All-reduce payload description in weight_sharding.md (~line 152)

**Issue:** "well under 1 KB per device contribution" was incorrect. The total payload is B x 1 x 5376 x 2 = 10,752 bytes (~10.5 KB), giving ~1.3 KB per device.

**Change:** Replaced with "~10.5 KB total (~1.3 KB per device contribution)".

## Fix 3: Inconsistent CCL topology in weight_sharding.md (~lines 157 vs 220-221)

**Issue:** The all-reduce correctly used `ttnn.Topology.Linear` (line 157/168), but the reduce-scatter in `TTNNLinearIColShardedWRowSharded` claimed `ttnn.Topology.Ring` (line 221). On T3K 1x8 without a wrap-around link, both should use Linear topology.

**Change:** Changed reduce-scatter topology from `ttnn.Topology.Ring` to `ttnn.Topology.Linear` with an explanatory note about the T3K 1x8 mesh lacking wrap-around.

## Fix 4: Global layer weight total in weight_sharding.md (~line 131)

**Issue:** Global layer per-device weight total was listed as ~153.6 MB. Summing the components (22.0 + 22.0 + 22.0 + 28.9 + 28.9 + 28.9 + 0.04) gives ~152.74 MB, which rounds to ~152.8 MB.

**Change:** Corrected total from ~153.6 MB to ~152.8 MB. Updated derived values: 10-layer total from ~1,536 MB to ~1,528 MB (BF16), and BFP8 from ~768 MB to ~764 MB.

## Fix 5: FFN FLOPs mislabeled as per-device in sharding_strategy_analysis.md (~line 251)

**Issue:** The FFN projection FLOPs figure "231M FLOPs per projection per device" was incorrect. The 231M figure (2 x 5376 x 21504) is the total FLOPs before TP sharding. With TP=8, the per-device figure is ~29M FLOPs.

**Change:** Replaced "231M FLOPs per projection per device" with "231M FLOPs total per projection, ~29M per device with TP=8".

## Fix 6: Global layer BFP8 weight total in kv_cache_sharding.md (DRAM budget table)

**Issue:** The global layer BFP8 weight total was listed as ~768 MB, inconsistent with the corrected value of ~764 MB in weight_sharding.md (see Fix 4). This caused the total weight budget (4,147 MB) and all derived totals and headroom figures in the combined DRAM budget table to be slightly overstated.

**Change:** Corrected global layer weights from ~768 MB to ~764 MB. Updated total weights from ~4,147 MB to ~4,143 MB. Recalculated all combined budget totals and headroom values (e.g., S=8,192 total changed from 4,887 MB to 4,883 MB, headroom from 7,401 MB to 7,405 MB). Updated 131K BFP8 KV cache total from 9,317 MB to 9,313 MB and headroom from 2.7 GB to ~3.0 GB.

---

# Compression Analysis --- Chapter 6: TP Sharding

**Analyst:** Agent C (Compressor)
**Date:** 2026-04-03
**Scope:** Redundancy and bloat only. No factual corrections.

---

## File 1: `index.md`

**Crucial updates: no**

**Load-Bearing Evidence:** The "Central Challenge" section (lines 26--44) restates the same fractional-split problem already introduced in the Overview paragraph (lines 5--12). The table on line 30 and the prose below it are a second explanation of the identical point --- 4 global KV heads / 8 devices = 0.5, which cannot be split. This is the chapter index; readers will immediately encounter the full treatment in `sharding_strategy_analysis.md`.

**Minor suggestion:** Merge the "Central Challenge" section into the Overview paragraph. The table from lines 30--33 can stay (it is useful at a glance), but the two paragraphs of explanatory prose (lines 35--44) duplicate information that the Overview already conveys in condensed form. Removing the redundant prose would cut ~10 lines without losing any information the reader does not already have from the Overview or will not get in the next file.

---

## File 2: `sharding_strategy_analysis.md`

**Crucial updates: no**

**Load-Bearing Evidence:** Option D (lines 214--305) is described as "operationally identical to Option A" (line 232), yet it devotes ~90 lines to re-presenting per-device shapes, memory costs, and CCL operations that are already fully enumerated under Option A (lines 22--97). The per-device summary table (lines 284--292) and the "Why the Costs Are Acceptable" subsection are the only new content; the mechanism description, activation shapes, and CCL pattern are verbatim repetitions of Option A.

**Minor suggestion:** Collapse Option D's mechanism and shape sections into a single sentence referencing Option A (e.g., "Option D uses the same replication mechanism and per-device shapes as Option A; the distinction is the justification for accepting those costs"). Keep only the "Why the Costs Are Acceptable" analysis and the per-device summary table, which are genuinely new. This would eliminate ~30 lines of duplicated content. Additionally, the comparison matrix (lines 310--317) groups A and D into a single column, confirming that they do not need separate per-device-shape writeups.

---

## File 3: `weight_sharding.md`

**Crucial updates: no**

**Load-Bearing Evidence:** The "All-Reduce After Row-Parallel Matmuls" section (lines 136--174) re-explains the row-parallel / all-reduce pattern that was already defined in the "Sharding Principles" section (lines 4--37). The principle section states: "partial output that must be summed across devices via ttnn.all_reduce" (line 16), and the ASCII diagram on lines 20--33 shows the full column-parallel-then-row-parallel-then-all-reduce flow. Lines 136--174 then spell out the identical pattern again with a concrete numeric example, the all-reduce code snippet, and the "2 all-reduce operations per layer" count. The code snippet and the per-layer count are load-bearing, but the surrounding re-explanation of row-parallel semantics is not.

**Minor suggestion:** Trim the "All-Reduce After Row-Parallel Matmuls" section to retain only (a) the payload size calculation (lines 151--154), (b) the code snippet (lines 161--170), and (c) the "2 all-reduces per layer, 120 total" count (lines 172--174). Remove the re-explanation of what row-parallel means and how partial sums work, since that is already covered in the principles section. This would save ~10--15 lines.

---

## File 4: `kv_cache_sharding.md`

**Crucial updates: no**

**Load-Bearing Evidence:** The per-layer-per-device global KV cache table (lines 82--92) lists 9 sequence-length entries. Of these, the values at S=1024, S=2048, S=4096, S=16384, and S=65536 are never referenced again in any downstream calculation or in any other file in this chapter. The "Total Global KV Across 10 Layers" table (lines 105--114) and the "Combined KV Cache Budget" table (lines 125--133) already provide the aggregated numbers at the sequence lengths that matter. The per-layer table could show 4 representative entries (e.g., 8K, 32K, 131K, 262K) and still convey the linear growth pattern without bloating the page.

**Minor suggestion:** Reduce the per-layer global KV cache table from 9 rows to 4--5 rows covering the sequence lengths that reappear in downstream tables (8,192; 32,768; 131,072; 262,144 and optionally 2,048 as a small-context reference). Similarly, the BFP8 per-layer table (lines 94--101) can be trimmed to match. The same applies to the "Total Global KV" table (lines 105--114), which has 8 rows but only 4--5 are used in the combined budget table. This would remove ~10--12 rows of intermediate data that serve no unique analytical purpose.

---

## Cross-File Redundancy

The fractional-split problem (4 KV heads / 8 devices = 0.5) is stated three times across the chapter:
1. `index.md` Overview (lines 7--12)
2. `index.md` Central Challenge (lines 26--44)
3. `sharding_strategy_analysis.md` The Problem (lines 1--18)

Occurrences 1 and 2 are in the same file. Merging them (as suggested above) would leave the problem stated once in the index and once in the analysis file, which is an appropriate level of repetition for a chapter-index-then-detail structure.

Additionally, the Option A memory table at S=8,192 in `sharding_strategy_analysis.md` (line 59: 64.0 MB per layer) and the per-layer global KV table in `kv_cache_sharding.md` (line 87: 64.0 MB) present the same computed value in the same format. This is acceptable cross-file repetition (different analytical contexts), but worth noting.

---

## Summary

| File | Estimated Removable Lines | Nature of Bloat |
|------|--------------------------|----------------|
| `index.md` | ~10 | Duplicate problem statement within same file |
| `sharding_strategy_analysis.md` | ~30 | Option D re-presents Option A verbatim |
| `weight_sharding.md` | ~10--15 | Row-parallel semantics explained twice |
| `kv_cache_sharding.md` | ~10--12 rows | Intermediate table entries unused downstream |
| **Total** | **~60--67 lines** | |

No crucial updates are needed. All files are factually consistent and structurally sound; the suggestions target only redundant exposition and over-enumerated tables.
