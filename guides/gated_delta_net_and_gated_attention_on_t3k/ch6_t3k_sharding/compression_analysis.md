# Compression Analysis: Chapter 6 — T3K Sharding Strategy — Pass 1

## Summary
- Total files analyzed: 5
- Estimated current line count: ~447 lines
- Estimated post-compression line count: ~377 lines
- Estimated reduction: ~16%

## CRUCIAL Suggestions

None.

## MINOR Suggestions

### [head_parallel_state_sharding.md] ~lines 123–138
**Issue:** Section 4 ("Per-Device Arithmetic Intensity After Sharding") explains that sharding does not change arithmetic intensity because FLOPs and bytes scale by the same factor. This restates Chapter 5 roofline material and is not essential to the sharding decision itself.
**Suggestion:** Condense to a single sentence: "Sharding does not change arithmetic intensity (FLOPs and bytes scale equally), so each device remains memory-bandwidth-bound with L1-resident state equally feasible at 128 KiB per device." Or remove entirely.

### [head_parallel_state_sharding.md] ~lines 101–119 & [kv_cache_sharding_for_gated_attention.md] ~lines 76–94
**Issue:** Both sections describe the identical 4 KiB all-gather cost and 0.16 µs time estimate in near-identical prose and calculation blocks.
**Suggestion:** In `kv_cache_sharding_for_gated_attention.md`, replace the repeated calculation block with: "CCL cost is identical to Gated Delta Net: 4 KiB all-gather, ~0.16 µs at 25 GB/s." Keep the full calculation only in `head_parallel_state_sharding.md`.

### [alternative_sharding_strategies.md] ~lines 74–82
**Issue:** The comparison table is followed by a prose summary that restates the table's conclusion verbatim.
**Suggestion:** Remove the prose summary; the table is self-documenting.

### [alternative_sharding_strategies.md] ~lines 1–32
**Issue:** Section 1 (Replicated State) is verbose for a rejected alternative; the narrative duplicates information from the comparison table and index summary.
**Suggestion:** Condense to 3–4 sentences covering: memory per device, CCL cost for input all-gather, and the B>1 synchronization fragility. Keep the numbers; remove the extended narrative.

### [kv_cache_sharding_for_gated_attention.md] ~lines 46–49 & line 105
**Issue:** The value "5,120 MiB ≈ 5.12 GiB per device" appears three times across the chapter (kv_cache_sharding_for_gated_attention.md derivation, summary table, and index.md).
**Suggestion:** Keep the authoritative appearance in the memory budget table and the final summary table. Remove the intermediate prose restatement; reference the table instead.

## Load-Bearing Evidence

- `head_parallel_state_sharding.md` line 3: "Each device holds a contiguous block of 4 value heads and computes the full recurrent step for those heads independently." — Core definition of the recommended strategy.
- `head_parallel_state_sharding.md` lines 104–105: "All-gather payload (per layer, B=1, T=1): 8 devices × [1, 1, 256] × 2 bytes = 4,096 bytes = 4 KiB" — The quantitative basis for the "negligible overhead" claim; must be preserved.
- `kv_cache_sharding_for_gated_attention.md` line ~64: "The KV cache for Gated Attention exceeds the per-device DRAM budget at T = 262,144." — The critical insight identifying the actual memory bottleneck.
- `alternative_sharding_strategies.md` line ~30: "Head-parallel sharding avoids this by design: each device owns and updates only its 4 heads." — Essential justification for head-parallel over replicated state for B>1.
- `alternative_sharding_strategies.md` line ~67: "blocked sharding produces contiguous memory access patterns in the state tensor" — Load-bearing because it determines the blocked vs. interleaved implementation recommendation.

## VERDICT
- Crucial updates: no
