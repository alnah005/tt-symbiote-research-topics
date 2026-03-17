# Compression Analysis — Chapter 4: Prefetch Patterns and Bandwidth

## Crucial updates: no

No crucial cross-file duplications found.

---

### Analysis Notes

All four files in Chapter 4 (`index.md`, `noc_and_dram_access.md`, `sharded_access_pattern.md`, `bandwidth_estimation.md`) were compared line-by-line against all files in Chapters 1, 2, and 3.

**What was checked and why it was not flagged:**

1. **DRAM controller topology** (`noc_and_dram_access.md` lines 21–26, NoC column index diagram) vs. `ch01/wormhole_memory_hierarchy.md` lines 7–29. The ch01 file shows a logical bank/controller hierarchy diagram; ch04 shows a horizontal NoC-column placement diagram. Different visual structures, different information (column positions vs. bank-count hierarchy). Not verbatim.

2. **Interleaved vs. sharded access diagram** (`ch04/index.md` lines 38–64, `Core(N,0) ──► D0–D5` diagram) vs. `ch01/interleaved_vs_sharded.md` lines 75–104 (`Core A/B/C/D ──→ NoC link → DRAM Bank` diagram). Both illustrate the same concept but with different structures, different numbers of cores, and different detail levels. Not verbatim or near-verbatim.

3. **Key Constants table** (`ch04/index.md` lines 79–89) vs. `ch01/wormhole_memory_hierarchy.md` bandwidth table (lines 62–67). The ch04 table lists compute-focused constants (peak TFLOP/s, ridge point, tile bytes); the ch01 table lists bandwidth tiers (per-controller GB/s, L1 TB/s, NoC link GB/s). Different rows, different columns, different purpose. Not duplicated.

4. **WIDTH_SHARDED ShardSpec API block** (`ch04/sharded_access_pattern.md` lines 72–85) vs. `ch02/constructing_dram_sharded_config.md` lines 13–44. Similar pattern (ShardSpec + MemoryConfig + to_memory_config), but ch04 uses generic placeholder variable names (`weight_rows`, `weight_cols // 8`) while ch02 uses concrete Mixtral numbers (`[4096, 1792]`) as part of a step-by-step tutorial. The surrounding context, variable names, and narrative purpose are distinct enough that neither block is redundant — the ch04 block is a quick-reference snippet and the ch02 block is a full pedagogical walkthrough. Not flagged.

5. **Roofline model derivation** (`ch04/bandwidth_estimation.md` lines 78–114). The ridge-point formula, arithmetic intensity derivation, and decode/prefill regime analysis appear nowhere in Chapters 1, 2, or 3. Unique to ch04.

6. **Bandwidth efficiency figures** (20–40% loss under interleaved access) appear in both `ch04/noc_and_dram_access.md` line 59 (prose) and `ch04/bandwidth_estimation.md` lines 49–53 (table). This is a within-ch04 restatement of a single number, not a multi-line block duplication against a prior chapter, and is therefore out of scope for this review.
