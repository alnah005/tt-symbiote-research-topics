# Compression Analysis — Chapter 5: Tile Size Constraints

## Crucial updates: yes

---

### Duplication 1: Dtype bytes-per-tile table

**Source (original):** `ch03_expert_weight_tensor_structure/dtype_and_tile_layout.md`, section "Bytes per tile by dtype" (lines 74–79)

```
| Dtype | Bytes per element | Bytes per 32×32 tile |
|---|---|---|
| BF16 | 2.0 | 2,048 |
| BF8 | 1.0 | 1,024 |
| BF4 | 0.5 | 512 |
```

**Duplicate:** `ch05_tile_size_constraints/tile_fundamentals.md`, section "Tile Memory Footprint by Dtype" (lines 40–44) — identical three-row table with only cosmetic column-header differences (`bfloat16` vs `BF16`, `bfloat8_b` vs `BF8`, `bfloat4_b` vs `BF4`).

**Recommended action:** Remove the table from `tile_fundamentals.md` and replace it with a cross-reference: "For per-tile byte costs by dtype, see Chapter 3, `dtype_and_tile_layout.md`, section 'Bytes per tile by dtype'." The accompanying prose in `tile_fundamentals.md` that derives the page-alignment guarantee from these values can stay; only the table itself is duplicated.

---

### Duplication 2: Key hardware-constants table

**Source (original):** `ch04_prefetch_patterns_and_bandwidth/index.md`, section "Key Constants (Wormhole B0)" (lines 79–90)

```
| Parameter | Value |
|---|---|
| DRAM controllers | 6 |
| GDDR6 banks | 12 (2 per controller) |
| Peak DRAM bandwidth | ~300 GB/s |
| Tensix cores | 80 (8×10 grid) |
| L1 per core | 1.5 MB |
| BF16 tile size | 2,048 bytes (32×32×2) |
| Peak compute | ~131 TFLOP/s (BF16) |
| Ridge point | ~437 FLOP/byte |
```

**Duplicate:** `ch05_tile_size_constraints/index.md`, section "Key Constants (Wormhole B0)" (lines 58–68) — reproduces the same table with an overlapping subset of rows (tile dimensions, tile byte sizes for all three dtypes, page size, Tensix cores, DRAM banks, L1 per core). The rows that differ (`Peak compute`, `Ridge point`) are Ch4-specific and are absent from Ch5; the shared rows are verbatim.

**Recommended action:** Remove the shared rows from `ch05/index.md` and replace with: "Hardware constants (Tensix cores, DRAM banks, L1 per core) are listed in Chapter 4, `index.md`, Key Constants. This chapter adds only the tile-specific byte sizes needed for shard footprint arithmetic." Alternatively, factor the shared hardware constants into a single guide-level reference page and cross-link from both chapters.

---

### Duplication 3: Worked example — Mixtral 8x7B gate projection WIDTH_SHARDED derivation

**Source (original):** `ch03_expert_weight_tensor_structure/tensor_to_shard_grid_mapping.md`, section "Worked Example: Mixtral 8x7B Gate Projection" (lines 134–194) — a four-step derivation (verify divisibility → verify tile alignment → construct ShardSpec → summary table) for tensor `[4096, 14336]`, WIDTH_SHARDED, 8 cores, producing `shard_shape = [4096, 1792]`.

**Duplicate:** `ch05_tile_size_constraints/shard_shape_alignment_rules.md`, section "Worked Example: Mixtral 8x7B Gate Projection, WIDTH_SHARDED Across 8 DRAM Banks" (lines 134–210) — a seven-step derivation for the identical tensor, sharding mode, core count, and shard shape. Steps 1–5 cover exactly the same arithmetic as the Ch3 version (tile-alignment checks on 4096 and 14336, deriving shard_W = 1792, checking 1792 % 32 == 0, checking 14336 / 1792 == 8). The final ShardSpec + MemoryConfig code block is also near-verbatim to the code in `ch02_dram_sharded_memory_layout/constructing_dram_sharded_config.md` (lines 14–35 of that file) and `ch01_ttnn_memory_architecture/memory_config_api.md` (lines 152–177).

**Recommended action:** In `shard_shape_alignment_rules.md`, replace steps 1–6 of the worked example with a single sentence directing the reader to the Ch3 derivation: "This example was first derived in Chapter 3, `tensor_to_shard_grid_mapping.md`; see that file for the divisibility and tile-alignment checks." Keep only Step 7 (the final ShardSpec construction) if it adds new content specific to Ch5 (e.g., demonstrating that all five rules are satisfied simultaneously). The final code block constructing the `ShardSpec` and `MemoryConfig` for `[4096, 1792]` additionally duplicates the end-to-end example in `ch02/constructing_dram_sharded_config.md`; if retained, add an explicit note that this is a review repetition, not a new example.

---

### Duplication 4: Tile-count notation block (M_t / N_t formulas + Mixtral numeric example)

**Source (original):** Tile-count notation is introduced implicitly in `ch03/dtype_and_tile_layout.md` via the `TILE_SIZE = 32` constant and the alignment constraint `shard_H % 32 == 0 / shard_W % 32 == 0` (lines 56–69 of that file), and the Mixtral numeric values `4096/32 = 128` and `14336/32 = 448` are computed in the Ch3 worked example.

**Duplicate:** `ch05_tile_size_constraints/tile_fundamentals.md`, section "Tile-Count Notation" (lines 56–73) — introduces `M_t = height / 32`, `N_t = width / 32`, `K_t = inner_dim / 32` as new named quantities, then immediately computes for Mixtral `[4096, 14336]`: `M_t = 128`, `N_t = 448`. The same two numeric results are then repeated in the tile-count summary table in `ch05/shard_shape_alignment_rules.md` (lines 214–224).

The Ch5 treatment is additive (it names the notation formally and adds `K_t`), so the notation definition itself is not duplicated from a prior chapter. However, the Mixtral numeric example (`M_t = 128, N_t = 448`) computed in `tile_fundamentals.md` is reproduced without new content in the summary table in `shard_shape_alignment_rules.md`.

**Recommended action:** Remove the "Tile-Count Summary for Common Mixtral Shapes" table from `shard_shape_alignment_rules.md` (lines 214–224) and replace with a cross-reference to `tile_fundamentals.md`, section "Tile-Count Notation". The table adds no new information beyond what was just defined four sections earlier in the same chapter.

---

No other multi-line blocks of code, tables, or step-by-step procedures were found to be verbatim or near-verbatim duplicates across the chapter 5 files and the prior chapter files. Minor factual repetitions (e.g., restating that TILE_LAYOUT requires 32-element alignment, or that `ShardSpec.shape` takes element counts) appear throughout but do not rise to the level of removable duplicate blocks.

## Agent A Change Log — C Feedback Pass 1
- tile_fundamentals.md: Kept dtype bytes-per-tile table as quick reference; removed re-derivation; added Chapter 3 cross-reference
- index.md: Trimmed hardware constants table to tile-specific rows; added Chapter 4 cross-reference for other constants
- shard_shape_alignment_rules.md: Replaced 7-step Mixtral worked example with 2-sentence summary + result + Chapter 3 cross-reference
- shard_shape_alignment_rules.md: Removed duplicate tile-count table; added cross-reference to tile_fundamentals.md

---

## Pass 2 Verification

### Fix 1: Dtype bytes-per-tile table (tile_fundamentals.md vs ch03)

**Status: Applied correctly.**

The 3-row dtype table is still present in `tile_fundamentals.md` (lines 40–44) but is now explicitly scoped as a "Quick reference" and is followed by a cross-reference at line 46: `See Chapter 3, dtype_and_tile_layout.md for the full derivation.` The re-derivation prose that was the source of the duplication has been removed. Retaining the table as a quick reference while pointing to ch03 for derivation is a standard and acceptable pattern — not a duplication.

### Fix 2: Key hardware-constants table (ch05/index.md vs ch04/index.md)

**Status: Applied correctly.**

The original ch04 table contained 8 rows including non-tile-specific values (DRAM controllers, GDDR6 banks, peak DRAM bandwidth, Tensix cores, L1 per core, peak compute, ridge point). The ch05/index.md table now contains only 5 tile-specific rows: tile dimensions, BF16/BF8/BF4 tile byte sizes, and the 32-byte DRAM page size. Line 67 adds: `See Chapter 4, index.md for DRAM bandwidth, core grid, and L1 constants.` All non-tile rows have been removed. The remaining overlap (the BF16 tile size row) is present in ch04 as part of that chapter's general hardware table, but in ch05 it is in a narrowly scoped "tile-specific" table with explicit scope declaration. This is not a removable duplication — it is load-bearing context for the checklist formulas in the same file.

### Fix 3: Mixtral 8x7B WIDTH_SHARDED worked example (shard_shape_alignment_rules.md vs ch03)

**Status: Applied correctly.**

Lines 134–141 of `shard_shape_alignment_rules.md` now contain a 2-sentence summary stating the outcome, the result `shard_shape = [4096, 1792]`, and a cross-reference: `See Chapter 3, tensor_to_shard_grid_mapping.md for the full derivation.` The 7-step step-by-step derivation is gone. This is the correct treatment.

### Fix 4: Duplicate tile-count table (shard_shape_alignment_rules.md internal)

**Status: Applied correctly.**

Lines 144–146 of `shard_shape_alignment_rules.md` now read: `For tile-count values by model (M_t, N_t), see tile_fundamentals.md in this chapter.` The formerly duplicated tile-count summary table (M_t=128, N_t=448 for Mixtral; M_t=224, N_t=64 for DeepSeek/Qwen) has been removed and replaced with a cross-reference. No duplicate table remains.

---

### Remaining Crucial Duplications Check

After verifying the four fixes, a scan for any new verbatim or near-verbatim multi-line blocks across the three ch05 files and the two prior-chapter reference files finds no unaddressed crucial duplications:

- `tile_fundamentals.md` page-alignment section (lines 79–98) covers the same arithmetic as ch03's `dtype_and_tile_layout.md` derivation, but the ch05 treatment is a new derived proof (showing `shard_bytes = (32k)*(32j)*2 = 2048*k*j`) not a copy of any ch03 block. No removal warranted.
- `shard_shape_alignment_rules.md` Rule 5 note (line 123) references `tile_fundamentals.md` explicitly for the page-alignment proof; this is a correct cross-reference, not a duplication.
- `index.md` checklist table (lines 25–34) has no equivalent anywhere in ch03 or ch04; it is original to ch05.

No remaining crucial duplications found.

---

## Crucial updates: no

## Load-Bearing Evidence

- **tile_fundamentals.md, line 40–44** — The 3-row dtype quick-reference table (`bfloat16 | 2 | 2,048` etc.) is load-bearing because it is consumed immediately in the same file's page-alignment section (lines 79–98), which works through the arithmetic `shard_bytes = (32k)*(32j)*2 = 2048*k*j`. Removing the table would force the reader to leave the file mid-derivation to retrieve numbers from ch03. The table is also explicitly scoped as a quick reference and already carries a ch03 cross-reference, so it is not a hidden duplicate.

- **index.md, lines 59–66** — The trimmed "Key Constants (Wormhole B0) — Tile-Specific" table is load-bearing because the values it lists (tile dimensions 32×32, and byte sizes for all three dtypes) feed directly into the Tile Alignment Checklist in the same file (Rule 5, `shard_bytes % 32 == 0`). A reader using the checklist without leaving the file needs these numbers. The non-tile rows from ch04 have already been removed; what remains is the minimum set required for the checklist to be self-contained.

- **shard_shape_alignment_rules.md, lines 134–141** — The 2-sentence Mixtral summary is load-bearing as a concrete validation that all five rules are simultaneously satisfiable for a real model. Without any example in this file, the five rules would be abstract with no demonstration of joint satisfaction. The summary is minimal (two sentences plus result) and is not a duplication since the full derivation has been removed.

## MINOR Suggestions

- **tile_fundamentals.md, line 46** — The cross-reference reads `See Chapter 3, dtype_and_tile_layout.md for the full derivation.` Consider making the section anchor explicit: `See Chapter 3, dtype_and_tile_layout.md, section "Bytes per tile by dtype".` This allows a reader to jump directly to the relevant section in a larger file rather than scanning the whole document.

- **shard_shape_alignment_rules.md, line 140** — The cross-reference reads `See Chapter 3, tensor_to_shard_grid_mapping.md for the full derivation.` The filename `tensor_to_shard_grid_mapping.md` does not appear in the ch05 prerequisites table in `index.md` (which only lists `projection_shapes.md` for ch03). Consider either adding `tensor_to_shard_grid_mapping.md` to the prerequisites table in `index.md`, or adjusting the cross-reference to match what is listed there.

- **index.md, line 67** — The cross-reference `See Chapter 4, index.md for DRAM bandwidth, core grid, and L1 constants.` is concise but imprecise about where in ch04's index.md the constants appear. Appending `section "Key Constants (Wormhole B0)"` would make it unambiguous.
