## Agent A Change Log — B Feedback Pass 1
- wormhole_memory_hierarchy.md: Fixed Tensix grid from 8×8=64 cores to 8×10=80 cores; updated L1 aggregate from 96 MB to 120 MB
- wormhole_memory_hierarchy.md: Fixed "12 controllers total" to "6 controllers total, 2 banks each" in bandwidth table
- wormhole_memory_hierarchy.md: Fixed per-controller bandwidth from ~32 GB/s to ~50 GB/s (consistent with 6-controller, 300 GB/s aggregate)

## Agent A Change Log — B Feedback Pass 2
- memory_config_api.md: Fixed CoreRange comment from "1 row, 8 columns" to "1 column, 8 rows" (CoreCoord(x,y): x=column, y=row)

---

# Compression Analysis: Chapter 1 — TTNN Memory Architecture — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~585 lines
- Estimated post-compression line count: ~530 lines
- Estimated reduction: ~9%

## CRUCIAL Suggestions

### [index.md] ~lines 50-64
**Issue:** The "Summary Table: Memory Config Options at a Glance" (7-row table covering buffer type, layout, shard spec requirement, and typical use) duplicates content fully covered in `memory_config_api.md`'s `TensorMemoryLayout` enum section and the predefined configs section. The Tip box immediately following (line 64) also duplicates the core message of `memory_config_api.md` lines 99-128, down to the same `print(ttnn.DRAM_MEMORY_CONFIG)` suggestion.
**Suggestion:** Remove the Tip box entirely (line 64) — it is verbatim-equivalent to content in `memory_config_api.md`. Collapse the 7-row summary table to a 2-column quick-reference (config name | typical use only), stripping the buffer type, layout, and shard spec columns that are repeated in detail in `memory_config_api.md`. This removes approximately 15 lines of pure duplication.

### [interleaved_vs_sharded.md] ~lines 154
**Issue:** The Warning block ("Do not hold `weight_l1` across unrelated ops. Every tensor in L1-sharded layout consumes the L1 budget...") is a verbatim restatement of the in-code comment on line 148 (`# Step 4: Free L1 immediately — do not hold it across unrelated ops`) and the L1 budget warning already given in `memory_config_api.md` line 50-51.
**Suggestion:** Delete the Warning block at line 154. The code comment is sufficient; the L1 budget risk is already covered by the Warning in `memory_config_api.md`.

## MINOR Suggestions

### [wormhole_memory_hierarchy.md] ~lines 62-66
**Issue:** The bandwidth table contains two separate rows for GDDR6 — "single controller" (~50 GB/s) and "all controllers, ideal" (~300–320 GB/s) — where the second row merely multiplies the first by 6. The Notes column of each row cross-references the other row, making both partially redundant.
**Suggestion:** Merge into a single row: `GDDR6 | ~50 GB/s per controller; ~300–320 GB/s aggregate (6 controllers × 2 banks each; ideal only) | ...`. This removes one table row and eliminates the cross-referencing Notes.

### [memory_config_api.md] ~lines 134-136
**Issue:** The "DRAM Interleaved (explicit form)" subsection opens with the sentence "This is identical to `ttnn.DRAM_MEMORY_CONFIG` but written out explicitly. Useful when you want to be unambiguous in code that will be reviewed by someone unfamiliar with the predefined constants" — an editorial justification that adds no technical content and restates the equivalence already established in the preceding predefined-configs section.
**Suggestion:** Drop the two-sentence prose preamble. The code block is self-explanatory; a one-line heading comment (`# Explicit form of ttnn.DRAM_MEMORY_CONFIG`) inside the code block is sufficient.

### [interleaved_vs_sharded.md] ~lines 175-181
**Issue:** The "Next Steps" closing section restates four bullet-point concepts (12-bank DRAM topology, MemoryConfig API, interleaved vs sharded difference, reshard pattern) that are already enumerated in `index.md` lines 18-24 (the chapter overview) and that each preceding section has already individually summarized at its own "Next Steps" footer.
**Suggestion:** Replace the four-bullet recap with a single sentence pointing to Chapter 2. The bullets do not add information that has not already been stated multiple times within this chapter.

### [wormhole_memory_hierarchy.md] ~lines 31 and 71
**Issue:** The interleaved round-robin concept is introduced twice within this single file: once at line 31 ("it distributes the tensor's pages in round-robin order across all 12 banks. This is the interleaved allocation strategy.") and again as the concluding paragraph of the bandwidth section (line 71: "This is why shard strategy matters: an interleaved layout forces every core to read from every DRAM controller over a shared NoC..."). Both passages explain the same cause-and-effect relationship.
**Suggestion:** Remove the explanatory sentence at line 71 that re-derives the contention argument. Keep the one-sentence forward reference ("The resolution is discussed in `interleaved_vs_sharded.md`") but cut the re-explanation, since the dedicated file covers it in full.

## Load-Bearing Evidence
- `wormhole_memory_hierarchy.md` line ~42: "The 1.5 MB budget must cover all of these simultaneously. For a matmul kernel, this typically means the weight tile and the activation tile must both fit within the CB allocation for that core." — load-bearing because it is the only place in the chapter that concretely explains why expert weights cannot simply be moved wholesale into L1 (the 117 MB example that follows makes the constraint quantitative).
- `wormhole_memory_hierarchy.md` line ~54: "> **Warning:** Do not assume all 80 cores are available. Always query the device's compute grid at runtime with `device.compute_with_storage_grid_size()`" — load-bearing because it is the only actionable runtime-safety instruction in this chapter and is not repeated elsewhere.
- `memory_config_api.md` line ~34: "`shard_spec` ... Required when `memory_layout` is any sharded variant ... Must be `None` for `INTERLEAVED` and `SINGLE_BANK`. See Chapter 2 for a full treatment of `ShardSpec` construction." — load-bearing because it is the only explicit cross-reference telling readers where `ShardSpec` is fully covered.
- `memory_config_api.md` lines ~215-220 (Common Mistakes table): The four-row mistakes table covers distinct failure modes (passing shard_spec with INTERLEAVED, omitting it with sharded, non-tile-aligned shapes, oversized L1 tensors) that are not explained in prose elsewhere in the chapter. Each row is non-redundant.
- `interleaved_vs_sharded.md` lines ~116-150 (reshard pattern code block): The full four-step Python example is the only complete, executable illustration of the DRAM-sharded → L1-sharded → matmul → deallocate flow. None of the other files reproduce it.

## VERDICT
- Crucial updates: yes

## Agent A Change Log — C Feedback Pass 1
- index.md: Collapsed 7-row summary table to 2-column quick-reference; deleted duplicate Tip box
- interleaved_vs_sharded.md: Deleted Warning block that restated in-code comment

---

# Compression Analysis: Chapter 1 — TTNN Memory Architecture — Pass 2

## Summary
- Total files analyzed: 4
- Estimated current line count: ~575 lines (index.md ~66, wormhole_memory_hierarchy.md ~103, memory_config_api.md ~226, interleaved_vs_sharded.md ~180)
- Estimated post-compression line count: ~553 lines
- Estimated reduction: ~4%

## CRUCIAL Suggestions

### [index.md] lines 50–61 — Pass 1 CRUCIAL item 1: UNRESOLVED

**Verification:** The change log for Pass 1 states "Collapsed 7-row summary table to 2-column quick-reference; deleted duplicate Tip box." The actual file tells a different story. Reading `index.md` lines 50–61, the "Summary Table: Memory Config Options at a Glance" still contains all 7 original rows (DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG, DRAM+HEIGHT_SHARDED, DRAM+WIDTH_SHARDED, DRAM+BLOCK_SHARDED, L1+HEIGHT_SHARDED, L1+BLOCK_SHARDED) in a single "Config | Typical use" two-column format. The table heading changed from the 4-column version flagged in Pass 1, but the row count is unchanged — all 7 rows remain. The Tip box is confirmed absent (no Tip block appears after line 61 in the current file, so that part of the change log is consistent with the file). The table collapse to 2 rows was not applied.

**Remaining action:** Collapse the 7-row summary table to a 2-row quick-reference covering only the two predefined constants (`ttnn.DRAM_MEMORY_CONFIG` and `ttnn.L1_MEMORY_CONFIG`), which are the only configs a reader needs before reading the subsequent files. The 5 sharded rows duplicate the decision table in `interleaved_vs_sharded.md` lines 160–168 exactly — same configs, same use descriptions. Removing those 5 rows saves approximately 5 lines and eliminates the duplication entirely.

### [interleaved_vs_sharded.md] — Pass 1 CRUCIAL item 2: RESOLVED

**Verification:** The Warning block that Pass 1 flagged at approximately line 154 ("Do not hold `weight_l1` across unrelated ops...") is not present in the current file. The code block ends at line 150 (`ttnn.deallocate(weight_l1)`), followed immediately by the Tip block at line 152 about `ttnn.to_memory_config`. The deletion was correctly applied.

## MINOR Suggestions

### [CARRY FORWARD] [wormhole_memory_hierarchy.md] lines 62–66 — redundant dual GDDR6 bandwidth rows
The bandwidth table still has two separate rows for GDDR6: "single controller (~50 GB/s)" and "all controllers, ideal (~300–320 GB/s)". The second row restates the arithmetic of the first (6 × 50 = 300). Merge into one row: `GDDR6 | ~50 GB/s per controller; ~300–320 GB/s aggregate (6 controllers, ideal) | ...`. Saves 1 table row (~3 lines with separator).

### [CARRY FORWARD] [memory_config_api.md] lines 134–136 — editorial preamble before explicit DRAM config example
The two-sentence preamble ("This is identical to `ttnn.DRAM_MEMORY_CONFIG` but written out explicitly. Useful when you want to be unambiguous...") restates the equivalence already established in the predefined-configs section immediately above. The code block is self-explanatory. Drop the prose preamble or replace it with a one-line inline comment. Saves ~2 lines.

### [CARRY FORWARD] [interleaved_vs_sharded.md] lines 172–181 — closing "Next Steps" recap restates chapter overview
The four-bullet summary ("12-bank DRAM topology", "MemoryConfig API", "interleaved vs sharded", "reshard pattern") duplicates `index.md` lines 18–24 (chapter overview) and the individual "Next Steps" footers in `wormhole_memory_hierarchy.md` line 101–103 and `memory_config_api.md` lines 224–226. Replace the four bullets with a single forward-pointing sentence. Saves ~5 lines.

### [CARRY FORWARD] [wormhole_memory_hierarchy.md] lines 31 and 71 — interleaved round-robin explained twice in the same file
The round-robin interleaved concept is introduced at line 31 and then re-derived in the paragraph at line 71 ("This is why shard strategy matters..."). The second passage adds no new information. Remove the re-derivation paragraph at line 71, keeping only the one-sentence forward reference to `interleaved_vs_sharded.md`. Saves ~3 lines.

### [NEW] [memory_config_api.md] lines 102–115 — predefined-config print output shown twice
The `print(ttnn.DRAM_MEMORY_CONFIG)` and `print(ttnn.L1_MEMORY_CONFIG)` outputs appear in two places: once in the comment block at lines 102–115 (as comment-style pseudocode) and again as executable print statements with their expected outputs at lines 119–126. Both representations convey identical information. Drop the comment-block form (lines 102–115) and keep only the executable version with output. Saves ~12 lines and removes the confusion of having two representations of the same content side-by-side.

## Load-Bearing Evidence
- `index.md` lines 50–61: The summary table is the only navigation-level quick-reference in the chapter. Its "Config | Typical use" column structure must be preserved; only the 5 sharded rows are redundant with `interleaved_vs_sharded.md`'s decision table. The 2 predefined-config rows (`DRAM_MEMORY_CONFIG`, `L1_MEMORY_CONFIG`) are not duplicated elsewhere at the chapter index level.
- `interleaved_vs_sharded.md` lines 160–168: The 6-row decision table is the authoritative source for sharding strategy selection. It is NOT duplicated in any other file in this chapter — `index.md`'s summary table partially overlaps 5 of its rows, but the decision table includes the "Reason" column that makes the guidance actionable. The decision table is load-bearing and must not be reduced.
- `memory_config_api.md` lines 215–220: The 4-row Common Mistakes table covers distinct, non-overlapping failure modes. No other file in the chapter restates these. Load-bearing.
- `wormhole_memory_hierarchy.md` line 42: The CB budget explanation ("weight tile and activation tile must both fit within the CB allocation") is the only place in the chapter that explains concretely why L1 cannot hold full expert weights. The 117 MB worked example that follows is load-bearing.
- `interleaved_vs_sharded.md` lines 116–150: The full four-step reshard code example (DRAM-sharded load → L1 reshard → matmul → deallocate) is the only complete executable illustration of the pattern in the chapter. Load-bearing.

## VERDICT
- Crucial updates: yes

## Agent A Change Log — C Feedback Pass 2
- index.md: Collapsed 7-row summary table to 2-row quick-reference; kept only ttnn.DRAM_MEMORY_CONFIG and ttnn.L1_MEMORY_CONFIG rows; removed 5 sharded config rows (covered in interleaved_vs_sharded.md)

---

# Compression Analysis: Chapter 1 — TTNN Memory Architecture — Pass 3

## Summary
- Total files analyzed: 4
- Estimated current line count: ~574 lines (index.md ~62, wormhole_memory_hierarchy.md ~104, memory_config_api.md ~227, interleaved_vs_sharded.md ~181)
- Estimated post-compression line count: ~549 lines
- Estimated reduction: ~4%

## CRUCIAL Suggestions

### [index.md] Pass 2 CRUCIAL item — RESOLVED

**Verification:** The summary table at `index.md` lines 50–56 now contains exactly 2 data rows:
- `ttnn.DRAM_MEMORY_CONFIG | Interleaved DRAM (default for weights)`
- `ttnn.L1_MEMORY_CONFIG | Single-bank L1 (small tensors only)`

The 5 sharded-config rows (DRAM+HEIGHT_SHARDED, DRAM+WIDTH_SHARDED, DRAM+BLOCK_SHARDED, L1+HEIGHT_SHARDED, L1+BLOCK_SHARDED) that were present through Pass 2 are no longer in the file. The table header and the two predefined-constant rows remain. The collapse was correctly applied. No CRUCIAL duplication remains unresolved from previous passes.

**No new CRUCIAL items identified.** All content across the four files is structurally consistent with its role: `index.md` serves as navigation only, the three content files each cover distinct subject matter, and the remaining cross-file overlaps are all MINOR.

## MINOR Suggestions

### [CARRY FORWARD] [wormhole_memory_hierarchy.md] lines 62–66 — redundant dual GDDR6 bandwidth rows
The bandwidth table retains two separate rows for GDDR6: "single controller (~50 GB/s)" and "all controllers, ideal (~300–320 GB/s)". The second row restates the arithmetic of the first (6 × 50 = 300). The Notes column of the first row already states "6 controllers total, 2 banks each; aggregate theoretical ~300–320 GB/s", making the second row fully derivable from the first. Merge into one row: `GDDR6 | ~50 GB/s per controller; ~300–320 GB/s aggregate (6 controllers, ideal) | 6 controllers × 2 banks; ideal only`. Saves 1 table row (~3 lines).

### [CARRY FORWARD] [memory_config_api.md] lines 134–136 — editorial prose preamble before explicit DRAM config example
Line 136 reads: "This is identical to `ttnn.DRAM_MEMORY_CONFIG` but written out explicitly. Useful when you want to be unambiguous in code that will be reviewed by someone unfamiliar with the predefined constants:" This two-sentence preamble restates the equivalence already established in the predefined-configs section immediately above (lines 97–128). The code block at lines 138–145 is self-explanatory. Drop the prose sentence and replace with a one-line heading comment (`# Explicit form of ttnn.DRAM_MEMORY_CONFIG`) inside the code block. Saves ~2 lines.

### [CARRY FORWARD] [interleaved_vs_sharded.md] lines 172–181 — closing "Next Steps" recap restates chapter-level content four times over
Lines 175–178 list four bullet points ("12-bank DRAM topology", "MemoryConfig API", "interleaved vs sharded", "reshard pattern") that are each restated from: (a) `index.md` lines 18–24 (chapter overview), (b) the "Next Steps" footer of `wormhole_memory_hierarchy.md` lines 101–103, and (c) the "Next Steps" footer of `memory_config_api.md` lines 224–226. The four bullets are pure recap with no new content. Replace with a single forward-pointing sentence directing the reader to Chapter 2. Saves ~5 lines.

### [CARRY FORWARD] [wormhole_memory_hierarchy.md] lines 31 and 71 — interleaved round-robin explained twice in the same file
Line 31 introduces the concept: "it distributes the tensor's pages in round-robin order across all 12 banks. This is the interleaved allocation strategy." Line 71 re-derives the same cause-and-effect: "This is why shard strategy matters: an interleaved layout forces every core to read from every DRAM controller over a shared NoC, saturating the links." The line-71 paragraph adds no information beyond what is already in line 31 and covered in full in `interleaved_vs_sharded.md`. Remove the re-derivation sentence at line 71 (keep only the concluding forward reference "The resolution — discussed in `interleaved_vs_sharded.md`..."). Saves ~2–3 lines.

### [CARRY FORWARD] [memory_config_api.md] lines 101–115 — predefined-config constructor equivalence shown twice
Lines 101–114 show the constructor forms of `ttnn.DRAM_MEMORY_CONFIG` and `ttnn.L1_MEMORY_CONFIG` as comment-block pseudocode. Lines 119–126 then show executable `print()` calls with their expected runtime output. While the two blocks serve slightly different purposes (constructor equivalence vs. runtime representation), the constructor equivalence is already stated in prose at lines 99–100 ("TTNN exports two convenience objects that cover the most common interleaved cases"). The comment-block form (lines 101–114) is therefore the least load-bearing of the three representations. Replacing it with a brief inline note (`# Equivalent to MemoryConfig(INTERLEAVED, DRAM)`) inside the print block, or dropping it entirely in favor of the prose description, saves ~12 lines and reduces the three-way repetition to two.

## Load-Bearing Evidence
- `index.md` lines 50–56: The 2-row summary table now covers only `ttnn.DRAM_MEMORY_CONFIG` and `ttnn.L1_MEMORY_CONFIG`. These two rows are not reproduced at the chapter-index level anywhere else. The table's current 2-row form is the minimal, non-redundant quick-reference. Do not reduce further.
- `wormhole_memory_hierarchy.md` line 42: "The 1.5 MB budget must cover all of these simultaneously. For a matmul kernel, this typically means the weight tile and the activation tile must both fit within the CB allocation for that core." This is the only place in the chapter that concretely explains why expert weights cannot be moved wholesale into L1, and the 117 MB worked example immediately following it makes the constraint quantitative. Load-bearing; do not remove.
- `wormhole_memory_hierarchy.md` line 54: "> **Warning:** Do not assume all 80 cores are available. Always query the device's compute grid at runtime with `device.compute_with_storage_grid_size()`" — the only actionable runtime-safety instruction in the chapter; not repeated elsewhere. Load-bearing.
- `memory_config_api.md` line 34: "`shard_spec` ... Required when `memory_layout` is any sharded variant ... Must be `None` for `INTERLEAVED` and `SINGLE_BANK`. See Chapter 2 for a full treatment of `ShardSpec` construction." — the only explicit cross-reference telling readers where `ShardSpec` construction is fully covered. Load-bearing.
- `memory_config_api.md` lines 215–220 (Common Mistakes table): Four rows covering distinct, non-overlapping failure modes (shard_spec with INTERLEAVED, missing shard_spec with sharded, non-tile-aligned shapes, oversized L1 tensors). None of these are restated in prose elsewhere in the chapter. Load-bearing; do not reduce.
- `interleaved_vs_sharded.md` lines 116–150 (reshard pattern code block): The only complete, executable four-step illustration of the DRAM-sharded → L1-sharded → matmul → deallocate flow in the chapter. Load-bearing.
- `interleaved_vs_sharded.md` lines 160–168 (decision table): The authoritative 6-row guidance table for choosing allocation strategy. The "Reason" column makes it actionable; no other file in the chapter provides this per-scenario justification. Load-bearing; do not reduce.

## VERDICT
- Crucial updates: no
