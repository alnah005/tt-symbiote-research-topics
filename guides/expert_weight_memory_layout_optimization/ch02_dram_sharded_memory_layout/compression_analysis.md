# Compression Analysis — Chapter 2: DRAM-Sharded Memory Layout

## Summary

| Metric | Value |
|---|---|
| Files analyzed | 4 |
| Estimated current line count | ~410 lines total (index.md: 41, shard_spec_deep_dive.md: 117, sharding_strategies.md: 109, constructing_dram_sharded_config.md: 176) |
| Estimated post-compression line count | ~325 lines |
| Estimated reduction | ~21% |

---

## CRUCIAL Suggestions

### CRUCIAL-1: ShardSpec three-field summary duplicated between `index.md` and `shard_spec_deep_dive.md`

- **`index.md` lines 23–28** (ShardSpec Quick Reference table) restates the `grid`, `shape`, and `orientation` fields — including the ROW_MAJOR/COL_MAJOR description — that are already given in full prose in **`shard_spec_deep_dive.md` lines 7–26** (The Three Fields section).
- The table in `index.md` is not a mere signpost; it duplicates the substance: field names, types, and behavioral descriptions. The only difference is condensation.
- **Which file should keep it:** `shard_spec_deep_dive.md` already contains the authoritative, detailed treatment. The table in `index.md` (lines 23–28) should be removed or replaced with a single sentence such as "See `shard_spec_deep_dive.md` for the full field reference." The quick-reference table adds minimal navigation value because the file is short and the reader is directed there immediately anyway.
- **Estimated saving:** ~6 lines from `index.md`.

### CRUCIAL-2: WIDTH_SHARDED worked example duplicated between `shard_spec_deep_dive.md` and `constructing_dram_sharded_config.md`

- **`shard_spec_deep_dive.md` lines 81–110** present a full worked example: tensor `gate_proj [4096, 14336]`, WIDTH_SHARDED across 8 DRAM banks, shard shape `[4096, 1792]`, grid `CoreRange(CoreCoord(0,0), CoreCoord(7,0))`, ROW_MAJOR orientation, plus a verification checklist.
- **`constructing_dram_sharded_config.md` lines 11–48** (Step 1 and Step 2) reproduce the identical tensor shape, identical shard dimensions, identical grid, and identical orientation. The Step 1 code block is nearly verbatim with only a comment change; Step 2 adds `MemoryConfig` construction and a `to_memory_config` call.
- The overlap is not just thematic: both use `[4096, 14336]` with `shape=[4096, 1792]` and `CoreRange(CoreCoord(0,0), CoreCoord(7,0))`. A reader following the recommended reading order encounters the same setup twice.
- **Which file should keep it:** `constructing_dram_sharded_config.md` is the dedicated construction guide and owns the complete three-step pattern. `shard_spec_deep_dive.md` should keep a shorter, more abstract example (e.g., just the `ShardSpec` constructor with placeholder values) and remove the full verification checklist and arithmetic derivation for this specific tensor (lines 81–110 can be trimmed to ~10 lines by removing the full derivation and retaining only the `ShardSpec` code snippet without duplicating the verification checklist).
- **Estimated saving:** ~20 lines from `shard_spec_deep_dive.md`.

### CRUCIAL-3: Shard shape arithmetic table in `shard_spec_deep_dive.md` partially duplicated by Strategy Comparison table in `sharding_strategies.md`

- **`shard_spec_deep_dive.md` lines 70–77** contain the "Shard Shape Arithmetic" table listing HEIGHT_SHARDED, WIDTH_SHARDED, and BLOCK_SHARDED constraints (e.g., `shard_shape[0] * num_shards = tensor_height`).
- **`sharding_strategies.md` lines 96–103** contain the "Strategy Comparison" table with the same three strategies and largely the same conceptual information (which dimension is partitioned, use case, NoC pattern).
- While the two tables differ in columns (arithmetic vs. use-case framing), the HEIGHT/WIDTH/BLOCK enumeration and the partitioned-dimension column effectively restate what the arithmetic table already established. A reader following both files learns the dimension-partitioning rule twice.
- **Which file should keep it:** The arithmetic table belongs in `shard_spec_deep_dive.md` as it is needed for calculating shard sizes. The Strategy Comparison table in `sharding_strategies.md` should drop its "Tensor dimension partitioned" column (since it repeats the arithmetic table) and retain only the "Typical expert weight use case" and "NoC access pattern" columns, which are genuinely additive.
- **Estimated saving:** ~3 lines from `sharding_strategies.md` (column removal or row consolidation).

### CRUCIAL-4: Wormhole B0 DRAM controller specification repeated verbatim in two files

- **`sharding_strategies.md` line 45**: "Wormhole B0 has 6 DRAM controllers (12 GDDR6 banks, 2 per controller)."
- **`constructing_dram_sharded_config.md` line 103**: "Note: Wormhole B0 has 6 DRAM controllers (12 GDDR6 banks, 2 per controller), not 8; the choice of 8 shards is justified by alignment with the 8-column Tensix grid width and the resulting tile-aligned shard size (14336/8=1792 elements)."
- Both files give this hardware fact, but `constructing_dram_sharded_config.md` provides additional justification (why 8 shards despite 6 controllers). The mention in `sharding_strategies.md` is a parenthetical aside that does not add to the tip in which it appears.
- **Which file should keep it:** `constructing_dram_sharded_config.md` should retain the full explanation. `sharding_strategies.md` line 45 should remove the parenthetical "(12 GDDR6 banks, 2 per controller)" and instead cross-reference the construction guide for the hardware detail.
- **Estimated saving:** ~1 line from `sharding_strategies.md`.

### CRUCIAL-5: `create_sharded_memory_config` warning stated twice in `constructing_dram_sharded_config.md`

- **`constructing_dram_sharded_config.md` line 38** (Step 2 prose): "The helper `ttnn.create_sharded_memory_config` is designed for L1 sharding; using it for DRAM placement can produce the wrong `buffer_type` silently."
- **`constructing_dram_sharded_config.md` lines 107–117** (Common Mistake #4): a full code example repeating the same warning with a wrong-code block and the same explanation.
- The Step 2 prose mention is a complete, actionable warning. The Common Mistakes entry re-explains the same thing with a code block that adds detail but the prose explanation is near-verbatim.
- **Which file should keep it:** Common Mistake #4 (lines 107–117) should be kept as the authoritative treatment since it includes the code example. The inline prose in Step 2 (line 38) should be shortened to a one-clause caution with a cross-reference ("see Common Mistakes #4 below").
- **Estimated saving:** ~1–2 lines from `constructing_dram_sharded_config.md`.

---

## MINOR Suggestions

### MINOR-1: "Next Steps" navigation duplicated across child files and index

- `shard_spec_deep_dive.md` lines 114–116 and `sharding_strategies.md` lines 106–108 each contain a "Next Steps" section pointing to the next file in the reading order. This reading order is already explicitly listed in `index.md` line 40 ("Recommended reading order: ...").
- These sections are not harmful but add ~5 lines of redundancy. Consider removing the "Next Steps" sections from the child files entirely, or collapsing each to a single sentence without a header (saving ~4 lines total).

### MINOR-2: Summary/overview sentences restating the index opening paragraph

- `index.md` lines 2–3 introduce the chapter as covering ShardSpec geometry, the three sharding strategies, and MemoryConfig construction.
- `shard_spec_deep_dive.md` line 3, `sharding_strategies.md` lines 1–3, and `constructing_dram_sharded_config.md` lines 1–3 each open with a paragraph scoping their respective topic — which maps directly to the three-part description in `index.md`. While per-file introductions are expected, the phrasing is closely parallel. These are MINOR because per-file openers serve navigational purposes, but trimming each from 2–3 sentences to 1 sentence would save ~6 lines collectively.

### MINOR-3: Tile-alignment warning repeated in multiple locations

- The warning that shard dimensions must be multiples of 32 for tile-layout tensors appears in: `shard_spec_deep_dive.md` line 17 (Warning callout), `shard_spec_deep_dive.md` lines 105–107 (verification checklist), `constructing_dram_sharded_config.md` lines 94–105 (Common Mistake #3), and `constructing_dram_sharded_config.md` line 105 (Warning callout). The rule is stated four times across two files. Common Mistake #3 is the most complete treatment; the standalone Warning callout in `shard_spec_deep_dive.md` (line 17) and the verification checklist entries could cross-reference it instead of restating the rule.

### MINOR-4: Code comments restating adjacent prose

- `constructing_dram_sharded_config.md` line 23 comment `# shard_height=4096, shard_width=14336/8=1792` restates the derivation already worked out in the preceding paragraph (and again in the End-to-End Example at line 146).
- `shard_spec_deep_dive.md` lines 87–89 comments `# integer, tile-aligned (1792 = 56 * 32) ✓` and `# full height per shard (WIDTH_SHARDED does not split height)` restate the Shard Shape Arithmetic table immediately above. These comments are helpful in isolation but redundant given the prose. Minor cleanup opportunity only.

### MINOR-5: DRAM vs L1 distinction explained in both `sharding_strategies.md` and `shard_spec_deep_dive.md`

- `sharding_strategies.md` lines 79–93 dedicate a full subsection ("DRAM-Sharded vs L1-Sharded: Key Distinction") to explaining that DRAM-sharded CoreRangeSets refer to DRAM controller positions, not Tensix cores.
- `shard_spec_deep_dive.md` lines 9–11 state the same fact: "For DRAM-sharded tensors, these 'cores' refer to DRAM controller positions in the Wormhole NoC grid, not Tensix compute cores."
- The `sharding_strategies.md` section is more complete and contextualizes both sides. The one-sentence mention in `shard_spec_deep_dive.md` is acceptable as a definitional aside, but the subsection in `sharding_strategies.md` could cross-reference back rather than re-deriving the point from scratch.

---

## Load-Bearing Evidence

Items that must NOT be removed or paraphrased away under any compression:

1. **`constructing_dram_sharded_config.md` line 38** — "Note: for DRAM-sharded layouts, always use the `ttnn.MemoryConfig` constructor directly with `buffer_type=BufferType.DRAM`. The helper `ttnn.create_sharded_memory_config` is designed for L1 sharding; using it for DRAM placement can produce the wrong `buffer_type` silently." This is the single most actionable correctness rule in the chapter and is not obvious from the API signature.

2. **`sharding_strategies.md` lines 83–87** — The explanation that for DRAM-sharded buffers the CoreRangeSet refers to DRAM controller positions in the NoC grid (not Tensix L1 cores), and that sharding reduces NoC contention compared to interleaved DRAM because each bank serves a dedicated non-overlapping set of shards. This is the core motivation for DRAM sharding as a technique.

3. **`shard_spec_deep_dive.md` lines 72–77** — The shard shape arithmetic constraints table: `shard_shape[0] * num_shards = tensor_height` for HEIGHT_SHARDED, `shard_shape[1] * num_shards = tensor_width` for WIDTH_SHARDED, and the 2D constraint for BLOCK_SHARDED. These formulas are the mechanical foundation for all shard size calculations.

4. **`constructing_dram_sharded_config.md` lines 97–103** — The explanation that choosing 6 shards for width 14336 yields a non-integer (14336/6 = 2389.33), and the clarification that Wormhole B0 has 6 DRAM controllers (not 8) with 8 shards justified by alignment with the 8-column Tensix grid. This fact prevents a common misconception about matching shard count to DRAM controller count.

5. **`constructing_dram_sharded_config.md` line 105** — Warning: "The error from a non-tile-aligned shard shape surfaces at buffer allocation, not at `ShardSpec` construction." This timing detail (error deferred to `to_memory_config`, not caught at spec construction) is critical for debugging.

6. **`sharding_strategies.md` lines 89–92** — The L1-sharded contrast: "Each shard lives in the L1 SRAM of the corresponding Tensix compute core (1.5 MB per core on Wormhole B0)... Used for activations during compute — the canonical pattern is to move DRAM-sharded weights into L1-sharded activations immediately before a matmul, then deallocate." This defines the runtime usage pattern that motivates the entire chapter.

7. **`constructing_dram_sharded_config.md` lines 62–64** — Verification tip: "Always verify `memory_config()` and `shard_spec()` after a layout transition, especially during initial integration. Silent mismatches (e.g., landing in INTERLEAVED because a helper defaulted `buffer_type`) can cause hard-to-diagnose performance regressions rather than immediate errors." This guards against the most dangerous failure mode (silent wrong buffer type).

---

## VERDICT

Crucial updates: yes

## Agent A Change Log — C Feedback Pass 1
- index.md: Removed ShardSpec Quick Reference table; replaced with pointer to shard_spec_deep_dive.md
- shard_spec_deep_dive.md: Replaced concrete WIDTH_SHARDED worked example with abstract [M, N] example + cross-reference to constructing_dram_sharded_config.md
- sharding_strategies.md: Removed redundant shard-shape-arithmetic column from Strategy Comparison table
- constructing_dram_sharded_config.md: Replaced repeated DRAM controller hardware fact with cross-reference to sharding_strategies.md
- constructing_dram_sharded_config.md: Removed prose duplicate of create_sharded_memory_config warning; added forward reference to Common Mistake #4

---

# Compression Analysis: Chapter 2 — DRAM-Sharded Memory Layout — Pass 2

## Summary

| Metric | Value |
|---|---|
| Files analyzed | 4 |
| Actual line count after Pass 1 | ~434 lines total (index.md: 37, shard_spec_deep_dive.md: 112, sharding_strategies.md: 109, constructing_dram_sharded_config.md: 176) |
| Pass 1 reduction vs. original ~410 estimate | Pass 1 fixes were successfully applied; actual post-fix count is ~434, slightly above the original ~410 estimate due to the original being an undercount. Net content was reduced by ~20 lines across the 5 fixes relative to what the files would have contained without the changes. |
| Remaining estimated reduction potential | ~30 lines (from MINOR issues; no CRUCIAL cross-file duplications remain) |

---

## Pass 1 Fix Verification

All 5 Pass 1 fixes were confirmed applied correctly:

1. **CONFIRMED — index.md ShardSpec Quick Reference table removed.** Lines 22–25 of `index.md` now contain only the `## ShardSpec Quick Reference` heading and a single pointer sentence: "For the full ShardSpec field reference, see `shard_spec_deep_dive.md`." The three-field table (grid/shape/orientation with ROW_MAJOR/COL_MAJOR descriptions) that duplicated `shard_spec_deep_dive.md` lines 7–26 is gone.

2. **CONFIRMED — shard_spec_deep_dive.md concrete example replaced with abstract [M, N] example.** Lines 81–106 now show a `[M, N]` / `num_cores` placeholder example with no concrete tensor dimensions. Line 105 cross-references `constructing_dram_sharded_config.md` for the concrete expert weight example. The full `[4096, 14336]` derivation and verification checklist that duplicated `constructing_dram_sharded_config.md` are gone.

3. **CONFIRMED — sharding_strategies.md redundant column dropped from Strategy Comparison table.** Lines 96–102 show the table with exactly two columns: "Typical expert weight use case" and "NoC access pattern". The third column restating dimension-partitioning arithmetic (already covered in `shard_spec_deep_dive.md`) has been removed.

4. **CONFIRMED — constructing_dram_sharded_config.md DRAM controller hardware fact replaced with cross-reference.** Line 103 now reads "Note: Wormhole B0 has 6 DRAM controllers (see `sharding_strategies.md` for hardware context), not 8…" — the verbatim parenthetical "(12 GDDR6 banks, 2 per controller)" that duplicated `sharding_strategies.md` line 45 has been removed.

5. **CONFIRMED — constructing_dram_sharded_config.md prose duplicate of create_sharded_memory_config warning replaced.** Line 38 now reads "…always use the `ttnn.MemoryConfig` constructor directly with `buffer_type=BufferType.DRAM` (see Common Mistake #4 below for details on why `ttnn.create_sharded_memory_config` must not be used here)." The full duplicate prose warning has been replaced with a forward reference.

---

## CRUCIAL Suggestions

None.

All five Pass 1 crucial duplications have been resolved. A scan of the current file contents reveals no new cross-file CRUCIAL duplications:

- The concrete `[4096, 14336]` / `[4096, 1792]` values now appear only in `constructing_dram_sharded_config.md`, which is the designated construction guide. Their repetition within that single file (Step 1, Verification, End-to-End Example) serves the step-by-step instructional pattern and is not a cross-file duplication.
- The 8-shard justification rationale ("alignment with the 8-column Tensix grid width", "14336/8=1792 elements") still appears in both `sharding_strategies.md` line 45 (WIDTH_SHARDED Tip) and `constructing_dram_sharded_config.md` line 103 (Common Mistake #3). However, `sharding_strategies.md` is now the authoritative source for the hardware specification, and `constructing_dram_sharded_config.md` cross-references it. The rationale in Common Mistake #3 provides essential context for why 8 was chosen over 6 — different enough in purpose that this does not rise to CRUCIAL.
- No other cross-file fact repetitions were found that meet the CRUCIAL threshold (substantive, actionable, and duplicated verbatim or near-verbatim with no additive content in either location).

---

## MINOR Suggestions

Carried forward from Pass 1 (all unresolved):

### MINOR-1: "Next Steps" navigation sections duplicated across child files and index

- `shard_spec_deep_dive.md` lines 109–111 contain a "Next Steps" section pointing to `sharding_strategies.md`.
- `sharding_strategies.md` lines 106–108 contain a "Next Steps" section pointing to `constructing_dram_sharded_config.md`.
- `constructing_dram_sharded_config.md` lines 173–175 contain a "Next Steps" section pointing to Chapter 3.
- The reading order is already established in `index.md` line 36. Each per-file "Next Steps" section adds ~3 lines of redundant navigation. Estimated saving: ~6–9 lines.

### MINOR-2: Summary/overview opening sentences in child files mirror index introduction

- `index.md` lines 2–3 describe the chapter as covering ShardSpec geometry, three sharding strategies, and MemoryConfig construction.
- Each child file opens with a 1–2 sentence scope statement that maps directly to the same three topics. Per-file openers serve standalone navigation but the phrasing is closely parallel. Trimming each opener from 2–3 sentences to 1 sentence would save ~6 lines collectively.

### MINOR-3: Tile-alignment warning repeated four times across two files

- `shard_spec_deep_dive.md` line 17: Warning callout ("TTNN raises an error at buffer allocation time if either shard dimension is not a multiple of 32").
- `shard_spec_deep_dive.md` line 103: Tip in worked example ("Choose `num_cores` to be a divisor of both the target dimension and 32 simultaneously").
- `constructing_dram_sharded_config.md` lines 94–101: Common Mistake #3 code block and prose.
- `constructing_dram_sharded_config.md` line 105: Warning callout (unique — "surfaces at buffer allocation, not at ShardSpec construction").
- Common Mistake #3 and the Warning at line 105 together form the most complete and actionable treatment. The standalone Warning in `shard_spec_deep_dive.md` line 17 and the Tip in the worked example could be shortened to cross-references. Note: the timing detail at line 105 is load-bearing and must be preserved (see Load-Bearing Evidence item 5).

### MINOR-4: Code comments restating adjacent prose

- `constructing_dram_sharded_config.md` line 23 comment `# shard_height=4096, shard_width=14336/8=1792` restates the derivation in the End-to-End Example already present at line 146.
- `shard_spec_deep_dive.md` lines 87–88 pseudocode comments (`# must be an integer and divisible by 32`, `# full height per shard…`) restate the Shard Shape Arithmetic table immediately above. Minor cleanup opportunity; no content loss.

### MINOR-5: DRAM vs L1 distinction explained in both `sharding_strategies.md` and `shard_spec_deep_dive.md`

- `sharding_strategies.md` lines 79–93: Full subsection "DRAM-Sharded vs L1-Sharded: Key Distinction" with both sides described in detail.
- `shard_spec_deep_dive.md` lines 9–11: One-sentence aside that DRAM-sharded CoreRangeSets refer to DRAM controller positions, not Tensix cores.
- The one-sentence mention in `shard_spec_deep_dive.md` is definitionally necessary at the point where `CoreRangeSet` is introduced. The subsection in `sharding_strategies.md` is the full treatment. No change needed, but the `sharding_strategies.md` subsection could add a back-reference to `shard_spec_deep_dive.md` rather than re-deriving the DRAM-controller-position fact from scratch.

---

## Load-Bearing Evidence

Items confirmed present and intact after Pass 1 (must NOT be removed or paraphrased away):

1. **`constructing_dram_sharded_config.md` line 38** — Forward reference to Common Mistake #4 for the `ttnn.create_sharded_memory_config` / `buffer_type=BufferType.DRAM` correctness rule. The full explanation remains at Common Mistake #4 (lines 107–117) with code example. The rule itself ("using it for DRAM placement can produce the wrong `buffer_type` silently") is the single most actionable correctness fact in the chapter.

2. **`sharding_strategies.md` lines 83–87** — Explanation that DRAM-sharded CoreRangeSets refer to DRAM controller positions in the NoC grid (not Tensix L1 cores), and that sharding reduces NoC contention compared to interleaved DRAM. Both points are present and unmodified.

3. **`shard_spec_deep_dive.md` lines 71–77** — Shard shape arithmetic constraints table (HEIGHT/WIDTH/BLOCK formulas). Present and unmodified; the abstract worked example below it now references rather than duplicates these formulas.

4. **`constructing_dram_sharded_config.md` lines 97–103** — Explanation that 14336/6 = 2389.33 (non-integer), and the clarification that 8 shards — not 6 — is the correct choice, justified by Tensix grid alignment. The cross-reference to `sharding_strategies.md` for the hardware spec is present. The actionable arithmetic demonstration is intact.

5. **`constructing_dram_sharded_config.md` line 105** — Warning: "The error from a non-tile-aligned shard shape surfaces at buffer allocation, not at `ShardSpec` construction." Timing detail is unique to this location and present.

6. **`sharding_strategies.md` lines 89–92** — L1-sharded contrast: shard lives in Tensix L1 SRAM (1.5 MB per core on Wormhole B0); canonical pattern is DRAM-sharded weights moved to L1-sharded activations immediately before matmul. Present and unmodified.

7. **`constructing_dram_sharded_config.md` lines 62–64** — Verification tip: "Always verify `memory_config()` and `shard_spec()` after a layout transition … Silent mismatches … can cause hard-to-diagnose performance regressions rather than immediate errors." Present and unmodified.

8. **`shard_spec_deep_dive.md` lines 103–105** — New tip (introduced by Pass 1 abstract example): "Always work out shard_width and shard_height on paper before constructing `ShardSpec`. Choose `num_cores` to be a divisor of both the target dimension and 32 simultaneously." This is additive guidance not present in the original concrete example and is now load-bearing in the abstract example context.

---

## VERDICT

Crucial updates: no
