# Compression Analysis: L1 State Management — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~324 lines
- Estimated post-compression line count: ~300 lines
- Estimated reduction: ~7%

## CRUCIAL Suggestions

None.

## MINOR Suggestions

### 1. [index.md / l1_state_design.md] ~lines 5-6 / 3
**Issue:** The "12 MB per layer" figure and "48 GDN layers" count appear in both the index and the first section of l1_state_design.md.
**Suggestion:** Since the index already establishes these numbers, l1_state_design.md could open more concisely: "The total state footprint (48 layers x 12 MB) of 576 MB far exceeds the ~1.5 MB of usable L1 per core."

### 2. [height_sharded_kernel.md / sdpa_l1_conflict.md] ~lines 97-103 / 69-74
**Issue:** Validation status table duplicated between height_sharded_kernel.md and sdpa_l1_conflict.md. The sdpa_l1_conflict.md table is a strict superset.
**Suggestion:** Replace the height_sharded_kernel.md "Validation Status" section (~6 lines) with a single sentence and forward reference to sdpa_l1_conflict.md.

### 3. [Multiple files] "SDPA circular buffer conflict" phrasing
**Issue:** The exact phrase appears in five occurrences across four files (index.md x2, height_sharded_kernel.md, sdpa_l1_conflict.md x2).
**Suggestion:** The index and sdpa_l1_conflict.md title are appropriate anchors; other instances could use lighter phrasing like "the SDPA conflict" where context is established. Cosmetic, noted for completeness.

## Load-Bearing Evidence
- `l1_state_design.md` line ~22: "_swap_l1_state() save/load phase with output_tensor=gdn._dram_state and explicit ttnn.deallocate()" — load-bearing as the core zero-allocation mechanism preventing memory fragmentation
- `height_sharded_kernel.md` line ~32: "volatile tt_l1_ptr pointer-based memcpy" — load-bearing as the defining artifact of the HEIGHT_SHARDED approach eliminating all NOC transactions
- `sdpa_l1_conflict.md` line ~14: "1,264 KB per core SDPA watermark" and "~240 KB remains on a 1,504 KB Blackhole core" — load-bearing as the quantitative foundation for why HEIGHT_SHARDED cannot scale beyond 1-2 layers
- `index.md` line ~3: "GDN layers consume 85% of total decode time" — load-bearing as the motivational anchor for the entire chapter

## VERDICT
- Crucial updates: no
