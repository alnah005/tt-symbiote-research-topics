# Compression Analysis: Chapter 1 — Hardware and TTNN Foundations — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~650 lines
- Estimated post-compression line count: ~530 lines
- Estimated reduction: ~18%

---

## CRUCIAL Suggestions

### [math_fidelity_and_data_formats.md] ~lines 24–36
**Issue:** The HiFi2 cell in the Math Fidelity Levels table (line 24) is massively bloated — it is a full paragraph rather than a table cell. It explains BFP8_B's shared exponent, the BF16 × BF16 case, and the rationale for HiFi4, all inline inside a table. Then lines 32–36 restate almost all of the same content in the "What 'Passes' Mean for Accuracy" subsection immediately below ("HiFi2 covers approximately the top half of the mantissa bits… For weights stored in BFP8 format, HiFi2 is empirically validated as sufficient… The reason HiFi4's extra passes provide diminishing returns for BFP8 × BF16 is that BFP8_B's shared block exponent…"). The identical reasoning (shared exponent → constrained dynamic range → HiFi4 wasteful for BFP8) appears verbatim in both the table and the prose section, and then a third time in the Key Pairing Rule note at line 175.
**Suggestion:** Trim the HiFi2 table cell to a single short phrase (e.g., "Upper mantissa range; empirically sufficient for BFP8 × BF16 (see Key Pairing Rule below)"). Move the BFP8 shared-exponent explanation to one canonical location — the Key Pairing Rule block — and delete the duplicate paragraph in "What 'Passes' Mean for Accuracy." This removes roughly 15–20 lines of repeated material.

### [math_fidelity_and_data_formats.md] ~lines 175–176 vs lines 32–36
**Issue:** The Key Pairing Rule block (lines 172–178) re-explains the BFP8_B shared-exponent argument for a third time ("BFP8_B has the same per-value mantissa width as BF16 (7 bits each) — the distinction is that BFP8_B uses a shared block exponent across 16 values. The shared exponent means the effective dynamic range per block is already constrained…"). This is word-for-word the same explanation already given in lines 32–36 and implied in the table cell at line 24.
**Suggestion:** In the Key Pairing Rule, replace the long BFP8 parenthetical with a single cross-reference sentence: "BFP8_B's shared block exponent constrains effective dynamic range per block, making HiFi4's extra passes redundant — see the 'What Passes Mean for Accuracy' section above." This cuts ~5 lines from the note without losing the reasoning.

### [ttnn_tensor_model.md] ~lines 229 vs lines 246–253
**Issue:** The BFP8_B description at line 229 includes a detailed inline derivation of the per-element storage cost ("The shared exponent adds a separate 8-bit field per 16-value block, equivalent to ~0.5 bits/value overhead… giving ~8.5 bits/element total, consistent with the memory footprint table below"). The memory footprint table at lines 246–253 then independently reconfirms the same 8.5-bit figure with a table note: "The shared exponent is 8 bits for every 16 values, adding 0.5 bits per element overhead to both BFP8 and BFP4." The derivation is given twice in two consecutive sections.
**Suggestion:** Remove the inline derivation from the BFP8_B prose (line 229) and keep only the result ("~1.06 bytes per element"). Let the table note at line 253 remain as the single explanation. This removes ~1.5 lines of redundant arithmetic.

### [index.md] ~lines 45–49 vs lines 35–39
**Issue:** The "Files in This Chapter" table (lines 35–39) already lists each file with a brief description. The "Reading Order" section (lines 45–49) then repeats the same three files with nearly identical descriptions appended to the numbered list entries. The added value of the reading order section is the sequencing rationale, not the per-file descriptions, which duplicate the table.
**Suggestion:** Shorten each numbered item in the Reading Order section to the file link plus a one-clause dependency note only (e.g., "1. [tensix_architecture.md] — start here; all TTNN abstractions build on constraints described here."). Drop the redundant paraphrase of the table descriptions. Saves ~4 lines.

---

## MINOR Suggestions

### [tensix_architecture.md] ~lines 96
**Issue:** The NoC aggregate bandwidth sentence is hedged to the point of providing no actionable information: "Aggregate intra-chip NoC bandwidth… is significantly higher than DRAM bandwidth, making it practical to saturate DRAM without becoming NoC-bound on most workloads; a precise aggregate figure is architecture-version-dependent and should be taken from Tenstorrent's official documentation." This occupies a full table/bullet entry but conveys only "NoC > DRAM bandwidth, look it up." The DRAM bandwidth figure (256–300 GB/s) is given concretely two sections later; the NoC figure is left as a gesture toward external docs.
**Suggestion:** Collapse to: "NoC aggregate bandwidth significantly exceeds DRAM bandwidth (see Tenstorrent official docs for exact figures), so most workloads are DRAM-bound, not NoC-bound." One sentence, same information.

### [tensix_architecture.md] ~lines 168–174 (Key Takeaways) vs body
**Issue:** The Key Takeaways section is thorough but re-states several points nearly verbatim from the body — notably the Dst tile counts (lines 63–69 vs line 171), the 120 KB L1 figure (line 122 vs line 172), and the LoFi/HiFi4 throughput ratio (lines 160–161 vs line 174). This is expected for a takeaways section, but the current phrasing is lengthy enough that it reads as a sixth section rather than a concise summary.
**Suggestion:** Trim each takeaway bullet to one tight sentence by removing re-stated parenthetical details. For example, line 171 ("The Dst register file holds 8 tiles in fp16 accumulation mode and 4 tiles in fp32 accumulation mode; this hard constraint drives all output subblock dimension choices in matmul program configs") can drop "this hard constraint drives…" since that implication was already stated in the body.

### [math_fidelity_and_data_formats.md] ~lines 161–163 (Decision Guide table)
**Issue:** Rows for "QKV projection (decode)" and "QKV projection (prefill)" are identical in every column (weight dtype, fidelity, fp32_dest_acc_en, packer_l1_acc) and the Rationale column of the prefill row says "Same as decode." The "Output projection wo (decode)" row similarly has "Same reasoning as QKV" as its rationale.
**Suggestion:** Merge the QKV decode and prefill rows into one ("QKV projection (decode + prefill)") and fold the `wo` row's rationale into the QKV row or make it a brief note. This removes two largely duplicate rows from the table and saves ~3 lines.

### [ttnn_tensor_model.md] ~lines 60–66 (Layout Conversion code comment)
**Issue:** The inline comment `# Row-major is also available (used for certain element-wise ops or host-side processing)` restates what has already been explained in the RowMajorLayout prose section three paragraphs above. The comment adds nothing the code itself does not already show via `ttnn.ROW_MAJOR_LAYOUT`.
**Suggestion:** Remove the inline comment or shorten to `# RowMajorLayout — less common, mainly for host-side ops`.

### [math_fidelity_and_data_formats.md] ~lines 88–101 (`packer_l1_acc` numbered-list illustration)
**Issue:** The two numbered lists (default mode: steps 1–4, then packer_l1_acc=True mode: steps 1–5) are prose restatements of the single-paragraph explanation of `packer_l1_acc` given immediately above them on lines 77–79. The concept is not complex enough to require both a paragraph definition and a step-by-step walkthrough.
**Suggestion:** Remove the "default mode" numbered list (steps 1–4 for the broken case) entirely — the broken behavior is already described in the paragraph. Keep the `packer_l1_acc=True` numbered list but trim it to 3 steps (fold steps 3–4 into one: "Repeat for each K-block"). Saves ~6 lines.

---

## Load-Bearing Evidence

- `tensix_architecture.md` line ~28: "A full 32×32 TTNN tile is composed of four 16×16 faces arranged in a 2×2 grid. The matrix engine processes a 32×32 × 32×32 matmul by internally iterating over the 16×16 face sub-tiles in the required sequence." — load-bearing because it is the only place in Chapter 1 where the face-level execution sequence inside the matrix engine is explained; this is referenced by ttnn_tensor_model.md's face layout diagram.
- `tensix_architecture.md` line ~66–69: table of `out_subblock_h × out_subblock_w ≤ 8` (fp16) and `≤ 4` (fp32) constraints — load-bearing because these exact numeric limits are the constraints math_fidelity_and_data_formats.md references when explaining the fp32_dest_acc_en trade-off, and Chapter 3 builds program configs around them.
- `ttnn_tensor_model.md` line ~240: "Halving the weight bandwidth vs BFP8 was measured to give +22% decode throughput on Llama 3.1 8B (23 → 28 tokens/s/user on N150)" — load-bearing because this is a concrete measured benchmark datum; it is not derivable from first principles in the chapter and is cited again in math_fidelity_and_data_formats.md line ~192.
- `math_fidelity_and_data_formats.md` line ~11: "LoFi = 1 pass, HiFi2 = 2 passes, HiFi3 = 3 passes, HiFi4 = 4 passes. The higher the pass count, the more of the mantissa significance range is covered… Fewer passes = fewer mantissa bits covered = lower precision, but proportionally higher throughput." — load-bearing because this is the foundational definition of the math fidelity system; the decision table and all throughput ratios in the chapter derive from it.
- `math_fidelity_and_data_formats.md` line ~77–79: "`packer_l1_acc=True`, the Packer instead adds the current Dst tile onto whatever value is already at the output L1 address: `L1[output_addr] += Dst[tile]`" — load-bearing because this is the precise semantic definition of the flag; the surrounding sections assume it.

---

## VERDICT
- Crucial updates: yes

---

## Agent A Compression Change Log — Pass 1

### CRUCIAL Item 1 — `math_fidelity_and_data_formats.md`: HiFi2 table cell trimmed

**Applied.** The HiFi2 row in the Math Fidelity Levels table (formerly line 24) was a multi-sentence paragraph re-explaining BFP8_B's shared exponent, the BF16 × BF16 case, and HiFi4 rationale inline inside the table cell. Replaced with a single brief phrase: "Upper mantissa range; empirically sufficient for BFP8 × BF16 (see Key Pairing Rule below)." Removes ~5 lines of redundant content from the table.

### CRUCIAL Item 2 — `math_fidelity_and_data_formats.md`: "What Passes Mean for Accuracy" HiFi2 paragraph condensed

**Applied.** The four-sentence HiFi2 paragraph in the "What Passes Mean for Accuracy" section (formerly line 36) duplicated the shared-exponent rationale already present in the Key Pairing Rule. Replaced with a single sentence acknowledging empirical sufficiency and directing the reader to the Key Pairing Rule section for the full rationale. The Key Pairing Rule block (lines ~175–176) was left intact as the canonical location. Removes ~3 lines of duplicate reasoning.

### CRUCIAL Item 3 — `ttnn_tensor_model.md`: 0.5 bits/element derivation de-duplicated

**Applied.** The BFP8_B prose (formerly line 229) contained a full inline derivation of the shared-exponent overhead ("8-bit field per 16-value block, equivalent to ~0.5 bits/value overhead on top of the per-value 8 bits — giving ~8.5 bits/element total"). This same derivation appeared again in the memory footprint table note (line 253). The prose sentence was shortened to state only the result (~8.5 bits/element) with a pointer to the table note ("see memory footprint table below for derivation"), leaving the table note as the single derivation location. Removes ~1 line of redundant arithmetic.

### CRUCIAL Item 4 — `index.md`: Reading Order section de-duplicated

**Applied.** The three numbered items in the Reading Order section each repeated content (file scope, topics covered) already present in the "Files in This Chapter" table immediately above. Each numbered item was trimmed to the file link plus a single clause explaining only the sequencing dependency (why this file before the next). Per-file content descriptions were removed. Saves ~3 lines.

---

# Compression Analysis: Chapter 1 — Hardware and TTNN Foundations — Pass 2

## Summary
- Total files analyzed: 4
- Estimated current line count: ~752 lines
- Estimated post-compression line count: ~728 lines
- Estimated reduction: ~3%

## CRUCIAL Suggestions

### [math_fidelity_and_data_formats.md] lines 83–102 — `packer_l1_acc` in Depth section duplicates the field description at lines 77–79

**Issue:** The `packer_l1_acc` field description at lines 77–79 already delivers the complete mechanism: (a) default mode performs a read-modify-write on the L1 output buffer at each K-block write event, (b) `packer_l1_acc=True` instead does `L1[output_addr] += Dst[tile]` eliminating that extra read, (c) this is a throughput optimization not a correctness requirement. The "in Depth" section that immediately follows (lines 83–102, ~20 lines) adds nothing that is not already in those three sentences. Specifically:
- The "default mode" numbered list (lines 87–92) restates the broken-case behavior already described in the prose above it. The note at line 92 explicitly says "accumulation across K-blocks is not broken without `packer_l1_acc`" — which was already said at line 79.
- The `packer_l1_acc=True` numbered list (lines 96–100) is a step-by-step restatement of `L1 += Dst` per K-block, already expressed in one sentence at line 77–78.
- The closing paragraph at line 102 re-explains "eliminates the extra L1 read from the read-modify-write cycle" in 3 sentences that were already covered in one sentence at line 77–79.

The only genuinely new content in the entire "in Depth" section is: (1) the `in0_block_w` / Dst capacity clarification note at the end of line 102, and (2) the "When `packer_l1_acc` helps most" bullet list at lines 104–108.

**Suggestion:** Delete the two numbered lists (lines 85–100) and the three-sentence closing paragraph (line 102, up to the `in0_block_w` note). Keep only: (a) the section heading, (b) the `in0_block_w`/Dst constraints note (currently appended to line 102 — move it to stand alone), and (c) the "When `packer_l1_acc` helps most" bullet list. This removes approximately 12–14 lines of pure restatement while preserving the two pieces of genuinely new content in the section.

---

## MINOR Suggestions

### [math_fidelity_and_data_formats.md] lines 161–163 (Decision Guide table) — QKV decode/prefill rows and `wo` row are near-duplicate (flagged Pass 1 MINOR, not yet applied)

**Issue:** The "QKV projection (decode)" and "QKV projection (prefill)" rows are identical in every column (weight dtype BFP8_B, fidelity HiFi2, fp32_dest_acc_en False, packer_l1_acc True). The prefill row's Rationale cell says "Same as decode." The "Output projection `wo` (decode)" row has "Same reasoning as QKV" in its Rationale. Three table rows with two of them explicitly marked as duplicates of the first.
**Suggestion:** Merge QKV decode and prefill into a single "QKV projection (decode + prefill)" row and consolidate the `wo` row's rationale into that row or add a brief note. Saves ~2–3 lines.

### [math_fidelity_and_data_formats.md] line 201 (Key Takeaways — `packer_l1_acc` bullet) — disproportionately long

**Issue:** The `packer_l1_acc` Key Takeaways bullet (line 201) runs ~3 lines and re-explains the read-modify-write mechanism in detail ("eliminating the read-modify-write overhead (read L1 partial sum, add Dst, write back) that the Packer otherwise performs at each K-block write event"). Every other takeaway bullet in the same section is one concise sentence. This bullet essentially re-summarizes both the field description section and the "in Depth" section.
**Suggestion:** Trim to one sentence: "`packer_l1_acc=True` enables in-place L1 accumulation (`L1 += Dst`), eliminating per-K-block read-modify-write overhead; use it for large-K matmuls with L1 output buffers." Saves ~2 lines.

### [tensix_architecture.md] lines 159–162 (throughput "Key observations") vs line 174 (Key Takeaways)

**Issue:** The "Key observations" bullet list following the throughput table (lines 159–162) states the 3.5× and 2× multipliers relative to HiFi4. The Key Takeaways section at line 174 restates: "Wormhole peak throughput scales from ~74 TOPS (HiFi4) to ~262 TOPS (LoFi) — a 3.5× range." The 3.5× figure appears twice in the same file within ~15 lines.
**Suggestion:** In the Key Takeaways, drop the parenthetical "(a 3.5× range)" since the takeaway is already quantified by citing the absolute TOPS values. Saves 1 line.

---

## Load-Bearing Evidence

- `math_fidelity_and_data_formats.md` lines 104–108: "**When `packer_l1_acc` helps most:** Large K-dimension matmuls (MLP FF1/FF2 with hidden dim 14336 in Llama 3 70B); `in0_block_w` set to a larger value than Dst can hold in a single pass; The output buffer is in L1 (not DRAM)." — Load-bearing: this is the only place in the chapter that gives concrete conditions under which the flag is beneficial, including the Llama 3 70B hidden dim 14336 as a calibrating example. Nothing in the field description or the In Depth section repeats these three conditions.

- `math_fidelity_and_data_formats.md` line 102 (final clause): "`in0_block_w` (the K-dimension tile block size in the program config) is constrained by L1 storage capacity for input tiles in the SrcA/SrcB circular buffers, not by Dst capacity; Dst capacity separately constrains `out_subblock_h × out_subblock_w`. The `packer_l1_acc` flag has no effect on either of those constraints." — Load-bearing: this is the only location in Chapter 1 that explicitly clarifies the orthogonality of the `in0_block_w` constraint vs the Dst subblock constraint vs `packer_l1_acc`. Cutting it would leave a conceptual gap.

- `tensix_architecture.md` lines 103–104: "In a well-optimized kernel, the Reader kernel on a core's RISC-V processor issues NoC DMA requests to pre-fetch the next tile from DRAM into L1 while the Compute kernel is still processing the current tile. This overlap is the basis for double-buffering (Chapter 4)." — Load-bearing: sole location in Chapter 1 where double-buffering is introduced and forward-referenced to Chapter 4; removing it breaks the conceptual bridge.

- `ttnn_tensor_model.md` lines 293–294 (Further Reading): The `bfloat8.hpp` / `bfloat4.hpp` path note with the explicit caveat "Note that `bfloat16.hpp` in the same directory is for standard BF16 (Brain Float 16) and does not contain the block floating-point format definitions. File names should be verified against the current repository, as paths may change across versions." — Load-bearing: the disambiguation between `bfloat16.hpp` and `bfloat8.hpp`/`bfloat4.hpp` is unique to this location; a reader following the reference without this note would open the wrong file.

- `math_fidelity_and_data_formats.md` lines 175–176 (Key Pairing Rule, BF16 × BF16 bullet): "BF16 × BF16 (7-bit × 7-bit mantissa) → HiFi4 (4 passes) for full precision; here **both** operands carry full 7-bit mantissas, and the complete cross-product requires more passes than HiFi2 can provide — this is why HiFi4 is required for attention score QK^T in prefill where both Q and K are BF16." — Load-bearing: the explicit statement that HiFi2 is insufficient for BF16 × BF16 (not just a positive recommendation for HiFi4) is critical for the decision guide; it appears only here in a form that makes the negative case clear.

---

## VERDICT
- Crucial updates: yes

---

## Agent A Compression Change Log — Pass 2

### CRUCIAL Item 1 — `math_fidelity_and_data_formats.md`: `packer_l1_acc in Depth` section de-duplicated

**Applied.** Removed the two numbered step lists (default mode: 4 steps, lines 87–92; `packer_l1_acc=True` mode: 5 steps, lines 96–100) and the three-sentence closing re-explanation paragraph (line 102, up to the `in0_block_w` note) that all restated the mechanism already fully defined in the field description at lines 77–79. Kept: (a) the section heading (`packer_l1_acc in Depth: Multi-Tile Accumulation`), (b) the `in0_block_w`/Dst capacity orthogonality note (formerly the final clause of line 102, now a standalone sentence), and (c) the "When `packer_l1_acc` helps most" bullet list (lines 104–108). Result is approximately 13 lines shorter.

---

# Compression Analysis: Chapter 1 — Hardware and TTNN Foundations — Pass 3

## Summary
- Total files analyzed: 4
- Estimated current line count: ~735 lines
- Estimated post-compression line count: ~724 lines
- Estimated reduction: ~2%

## CRUCIAL Suggestions

No new CRUCIAL items identified.

After two passes the files are substantively tight. No remaining block of ≥10 lines of unnecessary repetition was found where cutting loses zero information. The candidates examined and ruled out:
- The `index.md` "What You Will Be Able to Do" section (lines 22–29) overlaps with the per-file Key Takeaways sections, but serves a distinct navigational purpose as chapter-level learning objectives; cutting it would remove the only pre-read orientation point.
- The `WormholeComputeKernelConfig` code examples section (lines 95–134 in `math_fidelity_and_data_formats.md`) contains inline comments that partially restate field descriptions, but the code itself is load-bearing and the comments are brief context notes, not full re-explanations.
- The BFP4 +22% benchmark datum (23 → 28 t/s/u) appears in both `ttnn_tensor_model.md` line 240 and `math_fidelity_and_data_formats.md` line 175, but this is a single sentence in each location — well under the 10-line threshold.

## MINOR Suggestions

### [math_fidelity_and_data_formats.md] lines 143–146 (Decision Guide table) — QKV decode/prefill and `wo` rows near-duplicate (carried over from Pass 1 and Pass 2 MINOR; still unapplied)

**Issue:** The "QKV projection (decode)" and "QKV projection (prefill)" rows are identical across all five columns; the prefill row's Rationale cell says "Same as decode." The "Output projection `wo` (decode)" row has "Same reasoning as QKV" as its Rationale. This is explicit inline acknowledgment of duplication.
**Suggestion:** Merge the two QKV rows into "QKV projection (decode + prefill)" and fold the `wo` row's rationale into it or add a one-line note below the merged row. Saves ~2 table rows (~3 lines).

### [math_fidelity_and_data_formats.md] line 184 (Key Takeaways — `packer_l1_acc` bullet) — disproportionately long (carried over from Pass 2 MINOR; still unapplied)

**Issue:** The `packer_l1_acc` Key Takeaways bullet runs ~3 lines and re-explains the read-modify-write mechanism in detail ("eliminating the read-modify-write overhead (read L1 partial sum, add Dst, write back) that the Packer otherwise performs at each K-block write event"). Every other takeaway bullet in the section is one tight sentence.
**Suggestion:** Trim to: "`packer_l1_acc=True` enables in-place L1 accumulation (`L1 += Dst`), eliminating per-K-block read-modify-write overhead; use it for large-K matmuls with L1 output buffers." Saves ~2 lines.

### [tensix_architecture.md] lines 159–162 vs line 174 — 3.5× throughput ratio stated twice (carried over from Pass 2 MINOR; still unapplied)

**Issue:** The "Key observations" bullet at line 161 states "LoFi is approximately 3.5× faster than HiFi4." The Key Takeaways bullet at line 174 adds "(a 3.5× range)" parenthetically while also citing the absolute TOPS figures, making the ratio redundant at that location.
**Suggestion:** Drop "(a 3.5× range)" from the Key Takeaways bullet; the absolute values (74 TOPS vs 262 TOPS) already convey the ratio to any reader who needs it. Saves ~1 line.

### [tensix_architecture.md] line 96 — NoC aggregate bandwidth sentence is non-informative (carried over from Pass 1 MINOR; still unapplied)

**Issue:** The NoC per-link figure (32 bytes/cycle) is given concretely, but the aggregate bandwidth entry hedges to "significantly higher than DRAM bandwidth… a precise aggregate figure is architecture-version-dependent and should be taken from Tenstorrent's official documentation." The sentence occupies a full bullet but adds no actionable number.
**Suggestion:** Collapse to one clause appended to the per-link bullet: "Aggregate NoC bandwidth substantially exceeds DRAM bandwidth; see Tenstorrent official docs for chip-specific totals." Saves ~1 line.

## Load-Bearing Evidence

- `tensix_architecture.md` lines 66–69: table with constraints `out_subblock_h × out_subblock_w ≤ 8` (fp16) and `≤ 4` (fp32) — load-bearing because these exact limits are the hard bounds that drive all matmul program config choices in Chapter 3, and are referenced by `math_fidelity_and_data_formats.md`'s explanation of `fp32_dest_acc_en`.

- `math_fidelity_and_data_formats.md` lines 155–159 (Key Pairing Rule): "BFP8_B has the same per-value mantissa width as BF16 (7 bits each) — the distinction is that BFP8_B uses a shared block exponent across 16 values. The shared exponent means the effective dynamic range per block is already constrained… BF16 × BF16 (7-bit × 7-bit mantissa) → HiFi4 (4 passes) for full precision; here both operands carry full 7-bit mantissas, and the complete cross-product requires more passes than HiFi2 can provide." — load-bearing because this is the canonical explanation of why HiFi2 suffices for BFP8 × BF16 but is insufficient for BF16 × BF16; the decision table depends on it and it is the only location where the negative case (HiFi2 insufficient) is made explicit.

- `math_fidelity_and_data_formats.md` lines 83–85 (`packer_l1_acc in Depth` — `in0_block_w` note): "`in0_block_w` (the K-dimension tile block size in the program config) is constrained by L1 storage capacity for input tiles in the SrcA/SrcB circular buffers, not by Dst capacity; Dst capacity separately constrains `out_subblock_h × out_subblock_w`. The `packer_l1_acc` flag has no effect on either of those constraints." — load-bearing because this is the only location in Chapter 1 that explicitly states the orthogonality of these three constraints; omitting it leaves a conceptual gap that misleads readers configuring program configs in Chapter 3.

- `ttnn_tensor_model.md` lines 293–294 (Further Reading): the `bfloat8.hpp`/`bfloat4.hpp` file path note with the caveat that `bfloat16.hpp` in the same directory is for standard BF16 and does not contain block floating-point format definitions — load-bearing because readers navigating the tt-metal source to verify BFP formats would otherwise open the wrong file.

- `tensix_architecture.md` lines 103–104: "In a well-optimized kernel, the Reader kernel on a core's RISC-V processor issues NoC DMA requests to pre-fetch the next tile from DRAM into L1 while the Compute kernel is still processing the current tile. This overlap is the basis for double-buffering (Chapter 4)." — load-bearing as the sole location introducing double-buffering and its Chapter 4 forward reference.

## VERDICT
- Crucial updates: no
