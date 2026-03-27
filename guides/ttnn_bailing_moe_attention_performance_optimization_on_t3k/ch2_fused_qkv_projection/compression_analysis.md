# Compression Analysis: Chapter 2 — Fused QKV Projection — Pass 1

## Summary
- Total files analyzed: 3
- Estimated current line count: ~530 lines
- Estimated post-compression line count: ~430 lines
- Estimated reduction: ~19%

## CRUCIAL Suggestions

### [fusion_mechanics.md] ~lines 107–113
**Issue:** The three-bullet definition of `num_links=1/2/4` bandwidth doublings (e.g., "num_links=1: The all-reduce is serialized over a single 100 Gb/s link. Effective usable bandwidth for this all-reduce: ~12.5 GB/s…") is repeated almost verbatim in `num_links_tuning.md` lines 7–22, which is the file dedicated to this topic. One copy is fully redundant.
**Suggestion:** In `fusion_mechanics.md`, collapse lines 107–113 to a single sentence ("num_links controls how many of the 16 available 100 Gb/s Ethernet ports per chip are allocated to the CCL operation; see `num_links_tuning.md` for the quantitative implications.") and remove the three-bullet repetition.

### [fusion_mechanics.md] ~lines 119–143
**Issue:** The "Matmul Compute Time" sub-section opens with a compute-bound model (lines 130–143: peak_compute = 80 cores × 0.5 TFLOP/s, t_matmul ≈ 0.079 µs) and then immediately discards it ("in practice the matmul is latency-bound, not compute-bound"). The entire compute-model block exists only to be invalidated, making it ~14 lines of vestigial prose. The DRAM-bandwidth model on lines 145–160 is the load-bearing analysis.
**Suggestion:** Delete lines 130–143 entirely (the compute-throughput model and its conclusion). Keep the memory-bandwidth model (lines 145–160) as the sole analysis path. Add one sentence of motivation: "Because batch=1 matrix-vector products are memory-bandwidth-bound rather than compute-bound, the relevant bottleneck is DRAM read speed."

### [latency_savings_analysis.md] ~lines 82–87
**Issue:** The tile-count arithmetic block ("TTNN's tile layout uses 32×32 bfloat16 tiles internally. The fused (4096, 384) weight fits (128 × 12) tiles = 1536 tiles; the unfused shapes fit 1024 + 256 + 256 = 1536 tiles.") demonstrates that both paths produce exactly the same tile count (1536 = 1536). Because the tile counts are identical, the arithmetic supports no differential conclusion; the only load-bearing result — "Both require 3.0 MB of DRAM reads" — is stated immediately after and stands without the tile arithmetic.
**Suggestion:** Delete the tile-count arithmetic block (lines 82–87). Retain only the sentence "Both require 3.0 MB of DRAM reads, but the fused case allows the TTNN compiler to schedule a single contiguous DRAM read burst, potentially improving DRAM bandwidth utilization and eliminating 2 kernel dispatch overheads."

### [num_links_tuning.md] ~lines 117–123
**Issue:** The "At Larger Hidden Sizes" sub-section performs a payload doubling calculation for H=8192 and concludes in two sentences: "Still negligibly small. At hidden_size=8192, the conclusion is the same: `num_links=1` is optimal for batch=1." The sub-section adds no new mechanism or data; it merely confirms the batch=1 result holds at a hypothetical hidden size.
**Suggestion:** Remove the sub-section entirely. The "Conditions Under Which This Should Be Revisited" section on line 142 already covers the H≥8192 case with the appropriate action ("revisit the crossover batch size calculation").

## MINOR Suggestions

### [latency_savings_analysis.md] ~lines 3–7
**Issue:** The "Purpose" section is four sentences of meta-description ("This file quantifies… It establishes the baseline… then estimates… A brief preview…") that restates what section headers already make self-evident.
**Suggestion:** Reduce to one sentence or remove entirely. The file title and section headers convey the same information.

### [latency_savings_analysis.md] ~lines 130–135
**Issue:** The code block `t_fused_total ≈ t_matmul + t_all_reduce ≈ (10–35 µs) + (3–8 µs)` restates numbers already shown in the summary table directly above (lines 107–113).
**Suggestion:** Delete the code block; the prose sentence following it ("The fused matmul accounts for roughly 75–80%…") can stand alone.

### [num_links_tuning.md] ~lines 148
**Issue:** The final "Conditions" bullet — "Empirical measurement disagrees with this analysis. This entire analysis rests on estimated synchronization overheads… The first action after any deployment should be to profile…" — is a generic hedge applicable to all estimates across all three files, not specific to `num_links` conditions.
**Suggestion:** Delete this bullet. The cross-reference to TTNN op timers is already present in `latency_savings_analysis.md` lines 147–173 and `num_links_tuning.md` line 148 itself adds no unique condition.

### [fusion_mechanics.md] ~lines 172–183
**Issue:** The "Model Summary" table (lines 176–181) and "Key insight" paragraph (lines 183) restate conclusions reached in the preceding prose sub-sections. The table values are already derived in the text; the "Key insight" sentence repeats the closing sentence of both sub-sections.
**Suggestion:** Retain the summary table (it is useful for scanning) but delete the "Key insight" paragraph (lines 183) as it duplicates the table's final row annotation and the earlier prose conclusions.

## Load-Bearing Evidence
- `fusion_mechanics.md` line ~34: "In practice the allocation is rounded to head boundaries where possible, but for analysis purposes the 384-column-per-chip figure is exact." — load-bearing because this is the only place the 384-columns-per-chip figure is grounded with a caveat about head-boundary rounding; removing it would leave the shard calculation unqualified.
- `latency_savings_analysis.md` line ~126: "This is a meaningful result: CCL overhead, not compute, is the dominant cost being eliminated by fusion." — load-bearing because this explicit attribution of savings source (CCL vs. compute) is the primary analytical conclusion of the file and is not restated with equal clarity elsewhere.
- `num_links_tuning.md` lines ~92–101: The `B_crossover` derivation (`B_crossover = (3 µs × 12.5 GB/s) / (384 × 2 bytes) ≈ 49`) — load-bearing because this is the only quantitative threshold that defines when `num_links > 1` becomes beneficial; it underpins the batch-size table and the recommendation section.

## VERDICT
- Crucial updates: yes

---

## Agent A Change Log — B Feedback Pass 3

- Fix 1 (num_links_tuning.md ~line 23): Corrected per-chip Ethernet port budget fraction from "1/16" to "1/8" — `num_links=1` consumes 1 port per direction = 2 ports per chip out of 16 total, so the correct fraction is 2/16 = 1/8.

## Agent A Change Log — B Feedback Pass 2

- Fix 1 (num_links_tuning.md ~line 23): Corrected mislabeled bandwidth figure. Replaced "T3K aggregate Ethernet CCL bandwidth across all 8 chips is ~200 GB/s" with accurate figures: T3K total raw Ethernet capacity is ~1,600 GB/s (8 chips × 16 ports × 12.5 GB/s) [ESTIMATE]; a single num_links=1 CCL ring operation uses ~200 GB/s bidirectional (8 chips × 1 link × 12.5 GB/s × 2 directions) [ESTIMATE].
- Fix 2 (latency_savings_analysis.md ~line 124): Corrected speedup range from "~1.6–1.8×" to "~1.8–2.2×" to match the table values (unfused 29–78 µs, fused 13–43 µs; best case 29/13 ≈ 2.2×, worst case 78/43 ≈ 1.8×); added inline derivation sentence for clarity.

## Agent A Change Log — B Feedback Pass 1

- Fix 1 (latency_savings_analysis.md ~line 87): Corrected unfused tile count for K and V weight shards from 128 each to 256 each (shape `(4096, 64)` yields 128 row-tiles × 2 col-tiles = 256 tiles), changing the unfused total from 1024 + 128 + 128 = 1280 to 1024 + 256 + 256 = 1536 tiles.
- Fix 2 (latency_savings_analysis.md ~line 137): Corrected arithmetic intensity for `(1, 4096) × (4096, 384)` matmul from `0.125 FLOPs/byte` to `1.0 FLOPs/byte` (FLOPs = 2 × 4096 × 384 = 3,145,728; weight bytes = 4096 × 384 × 2 = 3,145,728; intensity = 1.0).

# Compression Analysis: Chapter 2 — Fused QKV Projection — Pass 2

## Summary
- Total files analyzed: 3
- Estimated current line count: ~496 lines (fusion_mechanics.md ~160, latency_savings_analysis.md ~178, num_links_tuning.md ~158)
- Estimated post-compression line count: ~482 lines
- Estimated reduction: ~3%

## CRUCIAL Suggestions

None. All four Pass 1 CRUCIAL items were confirmed applied and clean. Fresh scan found no new CRUCIAL redundancy across the three files.

Re-check results:
- `fusion_mechanics.md` num_links cross-ref (Pass 1 item 1): Applied — line 107 is a single cross-reference sentence. No regression.
- `fusion_mechanics.md` vestigial compute-throughput model (Pass 1 item 2): Applied — "Theoretical Bandwidth Model" opens directly with the DRAM-bandwidth model; the invalidated peak-compute block is gone. No regression.
- `latency_savings_analysis.md` tile-count equality block (Pass 1 item 3): Applied — the 1536 = 1536 arithmetic is gone; only the load-bearing DRAM read burst sentence remains. No regression.
- `num_links_tuning.md` H≥8192 sub-section (Pass 1 item 4): Applied — collapsed to a single redirect sentence at line 117. No regression.

## MINOR Suggestions

### [fusion_mechanics.md] line 155 — "Key insight" paragraph
**Issue:** The "Key insight" paragraph at the end of the Theoretical Bandwidth Model section ("At Ling's hidden size and batch=1, the fused QKV projection is memory-bandwidth-bound during the matmul phase and latency-bound during the all-reduce phase. Increasing `num_links` beyond 1 is unlikely to help…") duplicates the table row annotation in the summary table directly above (lines 146–153) and repeats conclusions already stated in `latency_savings_analysis.md` lines 138–139 and `num_links_tuning.md` lines 76 and 124–128. This item was flagged as Pass 1 MINOR #4 and was not yet applied.
**Suggestion:** Delete the "Key insight" paragraph (~3 lines). The summary table's final row and the dedicated `num_links_tuning.md` file carry this conclusion with more precision.

### [latency_savings_analysis.md] lines 3–7 — "Purpose" section
**Issue:** Four sentences of meta-description ("This file quantifies… It establishes the baseline… then estimates… A brief preview…") that repeat what the section headers already convey. Flagged in Pass 1 MINOR #1, not yet applied.
**Suggestion:** Reduce to one sentence or delete entirely. The file title and section headers are sufficient orientation.

### [latency_savings_analysis.md] lines 130–135 — redundant code block
**Issue:** The code block `t_fused_total ≈ t_matmul + t_all_reduce ≈ (10–35 µs) + (3–8 µs)` restates figures already shown in the summary table directly above (lines 107–113). Flagged in Pass 1 MINOR #2, not yet applied.
**Suggestion:** Delete the code block. The prose sentence that follows it ("The fused matmul accounts for roughly 75–80%…") can stand alone.

### [num_links_tuning.md] lines 142–143 — generic empirical-measurement hedge bullet
**Issue:** The final "Conditions" bullet ("Empirical measurement disagrees with this analysis. This entire analysis rests on estimated synchronization overheads… The first action after any deployment should be to profile…") is a generic caveat applicable to all estimates across all three files. It adds no condition specific to `num_links`. Flagged in Pass 1 MINOR #3, not yet applied.
**Suggestion:** Delete this bullet. The cross-reference to TTNN op timers is already present in `latency_savings_analysis.md` lines 147–173, and the generic hedge adds no actionable `num_links`-specific guidance.

## Load-Bearing Evidence

- `fusion_mechanics.md` line 34: "In practice the allocation is rounded to head boundaries where possible, but for analysis purposes the 384-column-per-chip figure is exact." — the only place the 384-columns-per-chip figure is qualified with the head-boundary rounding caveat; removing it would leave the shard calculation unqualified.
- `latency_savings_analysis.md` line 126: "This is a meaningful result: CCL overhead, not compute, is the dominant cost being eliminated by fusion." — the primary analytical conclusion of the file, not restated with equal clarity elsewhere.
- `num_links_tuning.md` lines 92–101: The `B_crossover` derivation (`B_crossover ≈ 49`) — the only quantitative threshold defining when `num_links > 1` becomes beneficial; it underpins both the batch-size table and the recommendation section.

## VERDICT
- Crucial updates: no

---

## Agent A Change Log — C Compression Pass 1

- Change 1 (fusion_mechanics.md lines 107–113): Collapsed three-bullet `num_links=1/2/4` bandwidth definitions to a single cross-reference sentence pointing to `num_links_tuning.md`; removed ~9 lines of duplicated content.
- Change 2 (fusion_mechanics.md lines 119–143): Replaced ~14-line vestigial compute-throughput model (peak_compute derivation and t_matmul ≈ 0.079 µs calculation that was immediately invalidated) with one motivating sentence establishing the memory-bandwidth-bound nature of batch=1 matrix-vector products; the DRAM-bandwidth model is now the sole analysis path.
- Change 3 (latency_savings_analysis.md lines 82–87): Removed the tile-count arithmetic block (1536 = 1536 equality demonstration) that produced no differential insight; retained only the load-bearing conclusion sentence about the single contiguous DRAM read burst and dispatch overhead elimination.
- Change 4 (num_links_tuning.md lines 117–123): Collapsed the "At Larger Hidden Sizes" sub-section (which merely confirmed the batch=1 result for H=8192 with no new mechanism) to a single sentence redirecting to the revisit-condition table.
