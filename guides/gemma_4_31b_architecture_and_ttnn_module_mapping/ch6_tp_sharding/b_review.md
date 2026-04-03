# Agent B Review: Chapter 6

## Pass 1

1. **Q projection FLOPs incorrectly labeled "on a single device"** (`sharding_strategy_analysis.md`, line 250). The text states: "the Q projection (2 x 5376 x 16384 = 176M FLOPs on a single device)." With TP=8, each device computes only its column shard `[5376, 2048]`, giving 2 x 5376 x 2048 = 22M FLOPs per device -- identical to the K projection. The 176M figure is the unsharded total across all 8 devices, not a per-device number. This undermines the argument that the redundant K projection is small relative to Q, since they are actually the same size per device. Fix: change to "176M FLOPs total across all 8 devices (22M per device)" or compare against the FFN projections instead.

2. **All-reduce payload described as "well under 1 KB per device contribution"** (`weight_sharding.md`, line 152). The total payload is B x 1 x 5376 x 2 = 10,752 bytes = ~10.5 KB at B=1. Per-device contribution is 10,752 / 8 = 1,344 bytes = ~1.3 KB. Neither figure is "well under 1 KB." Fix: say "~10.5 KB total" or "~1.3 KB per device contribution."

3. **Inconsistent CCL topology for row-parallel projections** (`weight_sharding.md`, lines 157-158 vs. 220-221). The all-reduce code example correctly uses `ttnn.Topology.Linear`, justified by the absence of a wrap-around link on T3K. However, the description of `TTNNLinearIColShardedWRowSharded` states it uses `ttnn.Topology.Ring` for reduce-scatter. If T3K has no wrap-around link between device 0 and device 7, Ring topology should not be available. Either the topology claim for all-reduce is wrong, the topology claim for reduce-scatter is wrong, or an explanation is needed for why they differ. Clarify which topology is actually used for each collective on T3K 1x8.

4. **Minor rounding: global layer weight total stated as ~153.6 MB, actual is ~152.8 MB** (`weight_sharding.md`, line 131). Computed: 3 x 22.02 + 3 x 28.90 + 0.04 = 152.8 MB. The ~0.8 MB discrepancy is small but unnecessary given that exact values are available. Consider correcting to ~152.8 MB or ~153 MB.

No other factual issues found. All KV cache arithmetic, per-device shapes, memory budget tables, and DRAM headroom calculations are correct.

## Pass 2

All four Pass 1 issues have been corrected. Two new factual issues found:

1. **FFN FLOPs mislabeled as per-device** (`sharding_strategy_analysis.md`, line 251). The text states: "the FFN projections (2 x 5376 x 21504 = 231M FLOPs per projection per device)." With TP=8 column-parallel sharding, each device computes 2 x 5376 x 2688 = ~29M FLOPs per projection per device. The 231M figure is the unsharded total across all 8 devices. This is the same category of error as the original Pass 1 Issue 1 (which was fixed for Q but not for FFN). Fix: change to "231M FLOPs per projection total (~29M per device)."

2. **Global layer BFP8 weight total inconsistent across files** (`kv_cache_sharding.md`, line 204 vs. `weight_sharding.md`, line 134). `weight_sharding.md` correctly states BFP8 global layer weights are ~764 MB per device (152.8 / 2 x 10 = 764). However, `kv_cache_sharding.md` lists them as ~768 MB in the weight budget table, a 4 MB discrepancy. This propagates to the total weights row (~4,147 MB should be ~4,143 MB) and to every headroom figure in the combined budget table. Fix: change 768 to 764 in kv_cache_sharding.md and recompute the total weights and all downstream headroom values.

## Pass 3

No feedback — chapter approved.
