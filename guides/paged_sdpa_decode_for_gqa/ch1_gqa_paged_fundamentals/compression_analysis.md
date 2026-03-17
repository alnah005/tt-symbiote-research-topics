# Compression Analysis: Chapter 1 — GQA and Paged Attention Fundamentals — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~524 lines
- Estimated post-compression line count: ~430 lines
- Estimated reduction: ~18%

---

## CRUCIAL Suggestions

### [gqa_concept.md + gqa_plus_paging_interaction.md] ~lines 115–121 (gqa_concept.md) and ~lines 108–122 (gqa_plus_paging_interaction.md)
**Issue:** The "performance bug, not a correctness bug" explanation for storing `nh` heads instead of `nkv` is written out in full in both files. `gqa_concept.md` section 5 ("Broadcast vs. Native") ends with a long Note (line 121) explaining exactly this. `gqa_plus_paging_interaction.md` section 5 Scenario B (lines 121) repeats the same reasoning nearly verbatim: `group_size=1`, Q head `i` attends to copy of original KV head `i // group_size`, output is "numerically correct," it is a "performance and memory waste, not a correctness failure."
**Suggestion:** In `gqa_concept.md`, shorten the Note to one sentence: "If you apply `repeat_interleave` before calling the native-GQA kernel, output is numerically correct but you waste `group_size×` memory and bandwidth; see `gqa_plus_paging_interaction.md` Scenario B for full details." The detailed chain-of-reasoning (group_size=1, kv_head_idx = q_head_idx // 1, copy of original head) then lives only in `gqa_plus_paging_interaction.md`. Saves ~6 lines.

### [paged_kv_cache_concept.md + gqa_plus_paging_interaction.md] ~lines 82–88 (paged_kv_cache_concept.md) and ~lines 51–71 (gqa_plus_paging_interaction.md)
**Issue:** The physical block address computation (`logical_block = pos // block_size`, `offset_in_block = pos % block_size`, `physical_block = page_table_tensor[i, logical_block]`) is shown as a code block in `paged_kv_cache_concept.md` section 4 and again, extended by two steps, in `gqa_plus_paging_interaction.md` section 3. The first three lines are identical; only the GQA head-selection lines (step 3 extension) are new.
**Suggestion:** In `paged_kv_cache_concept.md` section 4, remove the standalone code block showing the three-line address computation (lines 84–88) since it is fully subsumed by the more complete version in `gqa_plus_paging_interaction.md`. Replace it with a prose sentence: "The kernel computes the physical address from `pos`, `block_size`, and the page table entry; the full derivation is in `gqa_plus_paging_interaction.md` section 3." Saves ~7 lines.

---

## MINOR Suggestions

### [paged_kv_cache_concept.md] ~lines 53–59
**Issue:** The "Supported Block Sizes" table has two columns: `block_size` and "Token slots per physical block." The second column is identical to the first — it just repeats the numbers 32, 64, 128.
**Suggestion:** Drop the table entirely; replace with one sentence: "TTNN supports `block_size` values of 32, 64, or 128 tokens per block." The Tip immediately below the table (lines 63–64) provides the substantive trade-off guidance and should be kept. Saves ~6 lines.

### [paged_kv_cache_concept.md] ~lines 104–110 and ~lines 120–128
**Issue:** Each K/V shape (contiguous and paged) is followed by a "Dimension breakdown" prose list that restates every axis label already visible in the shape expression. The summary comparison table at lines 132–138 then compresses this same information again.
**Suggestion:** Remove the "Dimension breakdown" bullet lists under each shape. The shapes themselves plus the summary table are sufficient. Saves ~10 lines.

### [gqa_concept.md] ~lines 156–158 and [paged_kv_cache_concept.md] ~lines 199–201
**Issue:** Both files end with a "Next Steps" section that is a single sentence pointing to the next file. `index.md` already provides an explicit ordered reading list (lines 76–82) that covers the same navigation.
**Suggestion:** Remove the "Next Steps" sections from both `gqa_concept.md` and `paged_kv_cache_concept.md`. Keep the "Next Steps" section only in `gqa_plus_paging_interaction.md` (line 179–180) since it transitions out of the chapter entirely. Saves ~6 lines.

### [paged_kv_cache_concept.md] ~lines 8–16
**Issue:** Section 1 ends with the sentence "For long-context models (128K token context windows), even a single sequence occupies gigabytes per layer in a full contiguous allocation. Maintaining a pool of such allocations for a dynamic batch is impractical." This restates the two bullet points above it (static waste, fragmentation) without adding new information.
**Suggestion:** Delete the final two sentences of section 1 (starting "For long-context models..."). The quantitative detail is already covered in section 7's table. Saves ~2 lines.

---

## Load-Bearing Evidence

- `gqa_concept.md` line ~51: `"nh = nkv * group_size"` and the Flash-Decode kernel line `kv_head_idx = i // group_size` — load-bearing because this is the primary invariant the chapter is built around; the exact formula and its integer-division consequence must be stated exactly here.
- `gqa_concept.md` line ~57: `"If nh % nkv != 0, this integer division produces an incorrect mapping for some query heads, causing wrong attention scores with no error raised."` — load-bearing because it is the sole silent-failure warning for the head divisibility invariant; removing it leaves engineers unaware of the failure mode.
- `paged_kv_cache_concept.md` line ~90: `"page_table_tensor must be row-major int32 on device. Using the wrong dtype (e.g., int64 or float32) or the wrong layout (e.g., tile layout) causes the kernel to read incorrect physical block indices. This is a silent failure: no error is raised and the output is numerically wrong."` — load-bearing because it names a specific, silent dtype/layout constraint that has no other mention in the chapter.
- `gqa_plus_paging_interaction.md` line ~121: `"Scenario B: Cache was (wrongly) initialized with nkv=16 ... The attention output is numerically correct. However, this is a performance and memory waste, not a correctness failure"` — load-bearing as the full explanation of the nuanced Scenario B; the version in `gqa_concept.md` should be condensed to a cross-reference pointing here.
- `gqa_plus_paging_interaction.md` lines ~156–162: The "Head Count Invariants Summary" table listing all five invariants with their violation consequences — load-bearing because it is the only place all five invariants and their failure modes are co-located.
- `gqa_plus_paging_interaction.md` lines ~167–174: The padding forward reference explaining `nkv=4` padded to 32, `nh=16` padded to 32, effective `group_size=1` — load-bearing because it flags a non-obvious silent correctness failure that is deferred to Chapter 3.

---

## VERDICT
- Crucial updates: yes

---

## Agent A Change Log — C Feedback Pass 1
- gqa_concept.md: Reduced duplicate "performance bug" explanation to one-sentence cross-reference to gqa_plus_paging_interaction.md Scenario B
- paged_kv_cache_concept.md: Removed duplicate standalone block address code block; replaced with prose cross-reference to gqa_plus_paging_interaction.md

## Agent A Change Log — B Feedback Pass 2
- gqa_plus_paging_interaction.md: Fixed group_size=1 label from "MQA" to "MHA behavior" (group_size=1 means 1:1 Q-to-KV mapping = MHA, not MQA)

## Agent A Change Log — B Feedback Pass 1
- gqa_concept.md: Fixed repeat_interleave warning — clarified this is a performance bug not correctness bug
- gqa_concept.md: Fixed MHA baseline reduction factor from nkv/nh=1.0 to nh/nkv=1
- gqa_plus_paging_interaction.md: Fixed Scenario B — repeat_interleave produces correct results, not wrong ones
- paged_kv_cache_concept.md: Added clarification that `1` in paged_update_cache shape is single-token decode dimension

---

# Compression Analysis: Chapter 1 — GQA and Paged Attention Fundamentals — Pass 2

## Summary
- Total files analyzed: 4
- Estimated current line count: ~618 lines (index.md 83, gqa_concept.md 159, paged_kv_cache_concept.md 195, gqa_plus_paging_interaction.md 181)
- Estimated post-compression line count: ~594 lines
- Estimated reduction: ~4%

## CRUCIAL Suggestions
None — all Pass 1 CRUCIAL items resolved.

- **Item 1 verified:** `gqa_concept.md` line 121 now reads as a single-sentence cross-reference ("See Scenario B in `gqa_plus_paging_interaction.md` for the full analysis of why this wastes memory without affecting correctness."). The detailed Scenario B chain-of-reasoning lives only in `gqa_plus_paging_interaction.md` line ~121. Duplication eliminated.
- **Item 2 verified:** `paged_kv_cache_concept.md` Section 4 no longer contains a standalone block-address code block. Line ~81 now carries only a prose cross-reference ("For a token at position `pos` in batch element `i`, see `gqa_plus_paging_interaction.md` for a worked example of physical block address computation including GQA head selection."). Duplication eliminated.

No new CRUCIAL redundancy was found. The warning at `paged_kv_cache_concept.md` lines 148–149 about expanding GQA before writing to the cache correctly defers full detail to `gqa_plus_paging_interaction.md` without repeating Scenario A/B logic.

## MINOR Suggestions

### [paged_kv_cache_concept.md] ~lines 53–59
**Issue:** The "Supported Block Sizes" table has two columns (`block_size` and "Token slots per physical block") whose values are identical — both repeat 32, 64, 128. The second column adds no information.
**Suggestion:** Replace the table with one sentence: "TTNN supports `block_size` values of 32, 64, or 128 tokens per block." Keep the Tip below it (block size trade-off guidance). Saves ~6 lines.

### [paged_kv_cache_concept.md] ~lines 98–103 and ~lines 115–119
**Issue:** Each K/V shape (contiguous and paged) is followed by a "Dimension breakdown" bullet list that restates every axis label already visible in the shape expression itself. The summary comparison table at lines ~127–130 compresses the same information a third time.
**Suggestion:** Remove the "Dimension breakdown" bullet lists under both shapes. The shape expressions and the summary table are sufficient. Saves ~10 lines.

### [gqa_concept.md] ~lines 156–158 and [paged_kv_cache_concept.md] ~lines 192–194
**Issue:** Both files end with a "Next Steps" section that is a single pointer sentence to the next file. `index.md` already provides an explicit ordered reading list (lines 76–82) covering the same navigation path.
**Suggestion:** Remove the "Next Steps" sections from `gqa_concept.md` and `paged_kv_cache_concept.md`. Retain the "Next Steps" section in `gqa_plus_paging_interaction.md` (lines 178–180) only, since it transitions out of the chapter entirely. Saves ~6 lines.

### [paged_kv_cache_concept.md] ~lines 13–14
**Issue:** Section 1 closes with "For long-context models (128K token context windows), even a single sequence occupies gigabytes per layer in a full contiguous allocation. Maintaining a pool of such allocations for a dynamic batch is impractical." These sentences restate the two bullet points above without adding new information; the quantitative detail is already given in Section 7's table.
**Suggestion:** Delete those two trailing sentences. Saves ~2 lines.

## Load-Bearing Evidence
- `gqa_concept.md` line ~51: `"nh = nkv * group_size"` and `kv_head_idx = i // group_size` — load-bearing as the primary invariant the chapter is built around; must remain exact.
- `gqa_concept.md` line ~57: `"If nh % nkv != 0, this integer division produces an incorrect mapping for some query heads, causing wrong attention scores with no error raised."` — load-bearing as the sole silent-failure warning for head divisibility; removing it leaves the failure mode undocumented.
- `gqa_concept.md` line ~121: `"> **Note:** Do not mix these two approaches carelessly. See Scenario B in gqa_plus_paging_interaction.md for the full analysis of why this wastes memory without affecting correctness."` — load-bearing as the single remaining cross-reference anchor confirming Pass 1 Item 1 was applied; must not be further shortened.
- `paged_kv_cache_concept.md` line ~81: `"For a token at position pos in batch element i, see gqa_plus_paging_interaction.md for a worked example of physical block address computation including GQA head selection."` — load-bearing as confirmation that Pass 1 Item 2 was applied; the prose cross-reference that replaced the duplicate code block.
- `paged_kv_cache_concept.md` line ~83: `"page_table_tensor must be row-major int32 on device. Using the wrong dtype (e.g., int64 or float32) or the wrong layout (e.g., tile layout) causes the kernel to read incorrect physical block indices. This is a silent failure: no error is raised and the output is numerically wrong."` — load-bearing: names a specific silent dtype/layout constraint with no other mention in the chapter.
- `gqa_plus_paging_interaction.md` lines ~156–162: Head Count Invariants Summary table listing all five invariants with violation consequences — load-bearing as the only place all five invariants and failure modes are co-located.
- `gqa_plus_paging_interaction.md` lines ~167–174: Padding forward reference explaining `nkv=4` padded to 32, `nh=16` padded to 32, effective `group_size=1` — load-bearing as it flags a non-obvious silent correctness failure deferred to Chapter 3.

## VERDICT
- Crucial updates: no
