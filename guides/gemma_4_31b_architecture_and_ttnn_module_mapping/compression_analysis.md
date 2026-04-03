# Compression Analysis: Gemma 4 31B Architecture and TTNN Module Mapping

**Agent:** C (Compressor)
**Scope:** Cross-chapter redundancy in index.md files only
**Date:** 2026-04-03

## Crucial updates: no

## Summary

The guide has moderate cross-chapter repetition of architectural constants and tensor shapes, but the duplications serve a legitimate quick-reference purpose for readers entering at different chapters. No verbatim paragraph-level duplication was found. The redundancy is limited to parameter tables that re-state values already canonically defined in earlier chapters.

## Findings

### Finding 1: Weight shape table duplicated between Ch2 and Ch5

**Ch2 (Master Shape Table, lines 41-47)** defines Q/K/V/O weight shapes for both layer types as the canonical reference. **Ch5 (Key Parameters Quick Reference, lines 84-88)** reproduces identical weight shape values in a 16-row table that also includes head counts, head dims, RoPE type, window size, K=V sharing, V-norm, K-norm, and GQA ratio.

The Ch5 table is the largest single instance of cross-chapter duplication. While it adds a few derived values (fused QKV shape, GQA ratio) not present in Ch2, the majority of its rows are verbatim repetitions of Ch1 and Ch2 content.

**Suggestion (MINOR):** Replace the Ch5 weight shape rows (lines 84-88) with a brief note such as "See Ch2 Master Shape Table for all projection weight shapes" and retain only the Ch5-specific rows (fused QKV shape, GQA ratio, window, RoPE type) that are directly relevant to the attention module design discussion.

### Finding 2: Key constants table in Ch6 overlaps with Ch1

**Ch6 (Key Constants, lines 71-84)** lists 14 parameters including `hidden_size`, `intermediate_size`, all head counts, head dims, `sliding_window`, and `max_position_embeddings`. Every one of these values already appears in **Ch1 (Quick-Reference Table, lines 34-62)**.

The Ch6 table adds T3K-specific values (device count, DRAM per device, TP degree) that are not in Ch1, but these are only 3 of the 14 rows.

**Suggestion (MINOR):** Trim the Ch6 Key Constants table to only the values that are new or directly load-bearing for the sharding analysis: `num_key_value_heads`, `num_global_key_value_heads`, `head_dim`, `global_head_dim`, T3K device count, DRAM per device, and TP degree. Add a cross-reference to Ch1 for the full parameter list.

### Finding 3: Head-count-per-device table appears in both Ch6 overview and Ch6 Central Challenge

Within Ch6 itself, the overview paragraph (lines 7-9) states "32 Q heads / 8 devices = 4 per device" and "16 KV heads / 8 devices = 2 per device" and "4 KV heads / 8 devices = 0.5 per device." The Central Challenge table (lines 30-33) then presents the exact same arithmetic in tabular form. This is intra-chapter redundancy but noted here since both appear in the same index.md.

**Suggestion (MINOR):** Remove the arithmetic from the overview paragraph and let the Central Challenge table be the single presentation. The overview can simply state that the fractional KV split for global layers is the defining challenge.

### Finding 4: "50 sliding + 10 global layers" stated in 5 of 6 files

The phrase "50 sliding-window + 10 global layers" (or close variants) appears in:
- Top-level index.md (line 3)
- Ch1 index.md (line 7, implied by "60 decoder layers" + heterogeneous design)
- Ch5 index.md (lines 8-9)
- Ch6 index.md (lines 8-9)
- Ch8 index.md (line 28)

This is not actionable -- it is a fundamental architectural fact that naturally appears wherever context is needed. No change recommended.

### Finding 5: Total parameter count (~30.7B) stated in multiple locations

The ~30.7B parameter count appears in the top-level index (line 3), Ch2 (line 114), and Ch8 (line 28). This is minor and acceptable as a key summary statistic.

**No change recommended.**

## Load-Bearing Evidence

The primary evidence for Findings 1 and 2 is the side-by-side comparison of the Ch5 Key Parameters Quick Reference table (16 rows) against the Ch1 Quick-Reference Table (30 rows) and Ch2 Master Shape Table (15 rows). Of the 16 rows in the Ch5 table, 12 contain values that are exact copies of entries in Ch1 or Ch2. Similarly, 11 of 14 rows in the Ch6 Key Constants table are exact copies of Ch1 entries. The remaining rows in each table contain chapter-specific derived values that justify the table's existence but not its full size.

## Conclusion

Cross-chapter redundancy is present but moderate. The duplicated content is confined to quick-reference parameter tables, not prose or analysis. Three MINOR suggestions are offered to reduce table sizes in Ch5 and Ch6 by cross-referencing canonical definitions in Ch1 and Ch2. No crucial updates are needed.
