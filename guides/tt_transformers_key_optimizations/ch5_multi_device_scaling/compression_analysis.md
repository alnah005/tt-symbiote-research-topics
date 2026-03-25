# Compression Analysis — Chapter 5: Multi-Device Scaling
Reviewer: Agent C (compression reviewer)
Date: 2026-03-24
Files reviewed:
- index.md
- tensor_parallelism.md
- ccl_and_ethernet.md

## Summary

Four instances of verbatim or near-verbatim cross-file redundancy were found and resolved. The most significant was a full duplication of N300/T3K hardware specifications between `index.md` and `tensor_parallelism.md`, with a partial echo in `ccl_and_ethernet.md`. A near-identical "Summary of Collectives" table appeared in both `tensor_parallelism.md` and `ccl_and_ethernet.md` (the latter version was more complete). Two inline rationale passages — why `reduce_scatter` is used instead of `all_reduce` for row-parallel, and why `all_reduce` is appropriate only for the vocab-parallel LM head — were duplicated across both TP files. All crucial redundancies have been resolved by designating one file as the authority for each piece of content and replacing duplicates with explicit cross-references.

## CRUCIAL Suggestions

1. **N300/T3K hardware specs: `index.md` duplicated `tensor_parallelism.md`**
   - Location: `index.md` § Hardware Context (lines 36–41 original)
   - Duplicate of: `tensor_parallelism.md` § Hardware Configurations (lines 13–26)
   - Action taken: Replaced `index.md` Hardware Context body with a two-sentence pointer to `tensor_parallelism.md` and `ccl_and_ethernet.md`. `tensor_parallelism.md` retains the authoritative N300/T3K spec table and confirmed annotations.

2. **"Summary of Collectives per Pattern" table: `tensor_parallelism.md` duplicated `ccl_and_ethernet.md`**
   - Location: `tensor_parallelism.md` § Summary of Collectives per Pattern (lines 116–122 original)
   - Duplicate of: `ccl_and_ethernet.md` § Summary table (lines 53–58) and § Per-Layer CCL Summary (lines 150–162)
   - Action taken: Removed the table from `tensor_parallelism.md` and replaced it with a one-line cross-reference to the two authoritative sections in `ccl_and_ethernet.md`.

3. **`reduce_scatter` vs `all_reduce` rationale: `tensor_parallelism.md` duplicated `ccl_and_ethernet.md`**
   - Location: `tensor_parallelism.md` § Row-Parallel Linear, last paragraph of "Pattern" subsection (lines 83–84 original)
   - Duplicate of: `ccl_and_ethernet.md` § ttnn.reduce_scatter, paragraph 3 (lines 36–37)
   - Action taken: Trimmed `tensor_parallelism.md` to one sentence of attribution and replaced the rationale with a cross-reference to `ccl_and_ethernet.md`.

4. **`all_reduce` "only place in transformer" justification: `tensor_parallelism.md` duplicated `ccl_and_ethernet.md`**
   - Location: `tensor_parallelism.md` § Vocab-Parallel LM Head, "Pattern" subsection, last sentence (line 114 original)
   - Duplicate of: `ccl_and_ethernet.md` § ttnn.all_reduce, last two sentences (lines 50–51)
   - Action taken: Trimmed `tensor_parallelism.md` to a single cross-reference sentence; `ccl_and_ethernet.md` retains the full explanation.

## MINOR Suggestions

1. **Quantization bandwidth figures (BFP4: 3.56x, BFP8: 1.88x) repeated across all three files**
   - Locations: `index.md` § Relationship to Prior Chapters line 46; `tensor_parallelism.md` § Column-Parallel Linear, Weight Layout and Quantization (lines 58); `ccl_and_ethernet.md` § Interaction with Weight Quantization (lines 80–81).
   - The figures appear in different analytical contexts each time (index: cross-chapter orientation; tensor_parallelism.md: per-device storage; ccl_and_ethernet.md: CCL vs weight bandwidth trade-off), so the repetition is defensible. Consider removing the figures from `index.md` and adding a pointer to Ch3, since `index.md` already notes "see Ch3" for quantization context.
   - Not applied; flagged for author review.

2. **N300/T3K Ethernet descriptions partially repeated in `ccl_and_ethernet.md`**
   - Location: `ccl_and_ethernet.md` § Wormhole Ethernet Interconnect (lines 66–67)
   - This is a brief, bandwidth-context-specific statement (not a full spec repeat) and serves the local narrative about CCL bandwidth constraints. Acceptable as-is; could add "(see `tensor_parallelism.md` § Hardware Configurations)" if further tightening is desired.
   - Not applied; flagged for author review.

## Load-Bearing Evidence

The following content must NOT be cut regardless of future compression passes:

- `tensor_parallelism.md` § Hardware Configurations: The complete N300/T3K spec table with `[confirmed]` annotations and the Llama 3.1 70B production configuration note. This is the single authoritative hardware reference for the chapter.
- `ccl_and_ethernet.md` § ttnn.reduce_scatter: The explanation of why `reduce_scatter` rather than `all_reduce` is used after row-parallel layers. This is the authoritative rationale that `tensor_parallelism.md` now points to.
- `ccl_and_ethernet.md` § ttnn.all_reduce: The explanation of why `all_reduce` is the correct and only appropriate collective for the vocab-parallel LM head. Same reasoning.
- `ccl_and_ethernet.md` § Per-Layer CCL Summary: The full transformer block collective map. This is the most complete cross-reference table in the chapter.
- `tensor_parallelism.md` § GQA with Tensor Parallelism: The KV head sharding rule (TP <= n_kv_heads for sharding, TP > n_kv_heads forces replication) and the Llama 3.1 70B boundary-case example. Unique to this chapter; no duplication found.
- `ccl_and_ethernet.md` § When Tensor Parallelism Helps vs Hurts: The decision framework table and the CCL-dominates-at-small-batch analysis. Unique content; not echoed elsewhere.

## VERDICT
- Crucial updates: yes

---

# Compression Pass 2 — Chapter 5: Multi-Device Scaling
Date: 2026-03-24

## CRUCIAL Suggestions

None found.

## MINOR Suggestions

1. **BFP4/BFP8 bandwidth figures repeated in index.md "Relationship to Prior Chapters"**
   - Location: `index.md` § Relationship to Prior Chapters, line 41: "BFP8: 1.88x lower than BF16; BFP4: 3.56x lower than BF16"
   - Also present in: `tensor_parallelism.md` § Column-Parallel Linear, Weight Layout and Quantization (line 58); `ccl_and_ethernet.md` § Interaction with Weight Quantization (lines 80–81).
   - This was flagged as Minor in Pass 1 and remains unresolved. The figures serve different analytical purposes in tensor_parallelism.md (per-device storage) and ccl_and_ethernet.md (CCL vs weight bandwidth trade-off), so those two instances are justified. The index.md instance adds no new context and could be replaced with a pointer to Ch3 and the sub-files without information loss.
   - Recommendation: remove the inline figures from `index.md` line 41 and replace with "see Ch3 and `ccl_and_ethernet.md` § Interaction with Weight Quantization for how these factors interact with CCL cost."

2. **"Not PCIe" detail repeated in ccl_and_ethernet.md**
   - Location: `ccl_and_ethernet.md` § Wormhole Ethernet Interconnect (line 66): "Neither configuration uses PCIe for chip-to-chip data transfer."
   - Also present in: `tensor_parallelism.md` § N300 (line 15): "there is no PCIe path between them."
   - The authoritative hardware spec lives in `tensor_parallelism.md`. The ccl_and_ethernet.md mention is brief and contextually grounded in the bandwidth discussion, which makes it defensible. A pointer such as "(confirmed in `tensor_parallelism.md` § Hardware Configurations)" would suffice if further tightening is desired.
   - This was flagged as Minor in Pass 1 and remains unresolved. Still acceptable as-is.

## VERDICT
- Crucial updates: no
