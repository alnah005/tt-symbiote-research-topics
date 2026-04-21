# Compression Analysis: Chapter 1 — Complete Architecture Overview — Pass 1

## Summary
- Total files analyzed: 2
- Estimated current line count: ~789 lines (356 + 433)
- Estimated post-compression line count: ~640 lines
- Estimated reduction: ~19%

## CRUCIAL Suggestions

### [forward_pass_dataflow.md] ~lines 253-300 vs [architecture_and_hyperparams.md] ~lines 128-174
**Issue:** The state management section in `forward_pass_dataflow.md` (Section 3) restates nearly all of the recurrent state sizing and KV cache sizing already established in `architecture_and_hyperparams.md` Sections 4 and 5. Specific duplications:
- GDN state is `[B, 32, 128, 128]` in float32 -- stated in both files with the same derivation.
- "Must be in float32 for numerical stability" -- paraphrased in both files (architecture line 133: "requires higher precision than bfloat16 can provide"; dataflow line 259: "Must be in float32 for numerical stability across long sequences").
- Conv1d ring buffer is "4 slots of shape `[B, 8192]` in bfloat16" -- near-identical in both files (architecture line 135, dataflow line 261).
- GDN memory per layer at B=1 is ~2 MB -- computed in both files (architecture line 131: "B x 2 MB"; dataflow line 266: "~2.1 MB").
- Total GDN state for 30 layers at B=1 is ~60-63 MB -- in both files (architecture line 131: "60 MB"; dataflow line 268: "63 MB").
- KV cache per layer per token is 2,048 bytes -- derived identically in both files (architecture line 171, dataflow line 279).
- Total KV cache at 262K is ~5.1 GB -- stated in both files (architecture line 172, dataflow line 288).
- The dataflow file even includes a full KV cache table (lines 283-288) that adds only two intermediate data points (4K and 32K) beyond what the architecture file already provides.
**Suggestion:** In `forward_pass_dataflow.md` Section 3, replace the duplicated sizing derivations with a brief cross-reference: "State sizes are detailed in [architecture_and_hyperparams.md, Sections 4-5](./architecture_and_hyperparams.md#4-gated-deltanet-configuration)." Retain only what is unique to the dataflow file: the State Comparison table (lines 292-300) and the narrative about the hybrid tradeoff (lines 301). Keep the KV cache scaling table if the intermediate data points (4K, 32K) are considered useful for implementation planning, but remove the re-derived per-layer/per-token formulas.

### [forward_pass_dataflow.md] ~lines 82-99 vs ~lines 26-40
**Issue:** The Gated Attention sublayer pseudocode block (lines 86-98) is structurally identical to the Gated DeltaNet sublayer pseudocode block (lines 29-40), differing only in the attention call on one line. The two blocks share 8 identical lines: the `x_in` assignment, both RMSNorm calls, both residual adds, and the MoE sublayer. Line 95 even includes the comment `# Sublayer 2: MoE FFN (identical to DeltaNet layers)`, explicitly acknowledging the duplication.
**Suggestion:** Show the full pseudocode once for the generic decoder layer, then describe only the differing attention call for each layer type. For example, present the two-sublayer skeleton once, then say "Sublayer 1 dispatches to `GatedDeltaNet(x_norm)` or `GatedAttention(x_norm, ...)` depending on layer type." This eliminates ~12 lines of repeated pseudocode.

### [architecture_and_hyperparams.md] ~lines 42-58 vs ~lines 32-42
**Issue:** The hybrid layer layout is described three times in quick succession: (1) the pattern string on line 35, (2) the bullet-point breakdown on lines 39-40 with the ratio explanation on line 42, and (3) the full Layer Layout Table on lines 44-57. The table simply expands the pattern that was already shown twice, adding no new information. Line 58 ("Every layer, regardless of attention type, uses the same MoE FFN block") is also restated in the table's FFN Type column which is uniformly "MoE" for every row.
**Suggestion:** Keep the pattern string and bullet-point breakdown. Remove the Layer Layout Table entirely -- it adds 14 lines to spell out what the pattern already communicates. If the table is retained for quick-reference value, at minimum remove the FFN Type column (it is constant) and collapse the ellipsis rows.

### [forward_pass_dataflow.md] ~lines 375-396
**Issue:** The "Routing Decision Visualization" ASCII diagram (lines 375-396) restates the same flow just described in Steps 1-5 (lines 309-370) immediately above it. Every element in the diagram -- the router matmul, top-8 selection, per-expert SwiGLU, shared expert, weighted sum -- was already shown with code blocks and exact tensor shapes in the step-by-step walkthrough. The diagram adds no new information.
**Suggestion:** Remove the "Routing Decision Visualization" subsection (saves ~22 lines). The step-by-step walkthrough with code blocks already serves as both explanation and visual reference. Alternatively, if a compact diagram is desired, replace the entire Steps 1-5 prose with a single annotated diagram, but do not keep both.

### [architecture_and_hyperparams.md] ~lines 345-351 vs ~lines 168-174
**Issue:** Section 10 "Context Length" (lines 347-351) restates information already given in Sections 1, 4, and 5. Specifically: "262,144 tokens (262K)" is from the Top-Level Hyperparameters table (line 18); "approximately 1M tokens with appropriate RoPE scaling" is a restatement of line 26; "Only the 10 Gated Attention layers maintain KV caches" is from Sections 4 and 5; and the 5.1 GB KV cache figure and comparison to a hypothetical 40-layer model are restated from Section 5 (lines 172-174, nearly word for word).
**Suggestion:** Remove the "Context Length" subsection from Section 10 entirely. All of its content exists verbatim in earlier sections. If a summary is desired, a single sentence referencing the earlier sections suffices.

### [architecture_and_hyperparams.md] ~lines 157-165
**Issue:** The "Key Extensions" in Section 5 re-explain Partial RoPE and GQA in prose that overlaps with the table immediately above (lines 148-155) and with Section 7 (lines 237-259). Partial RoPE is explained in Section 5 lines 159, then again in Section 7 lines 243-247 with the identical formula. GQA with "16 query heads and 2 KV heads, each KV head is shared by 8 query heads" is stated in the table (line 152) and then restated in prose (line 165).
**Suggestion:** In Section 5, reduce the Partial RoPE paragraph to a forward reference: "See [Section 7](#7-rope-configuration) for the full RoPE configuration." Keep the brief mention of which 64 dims are rotary, but remove the formula and theta value (they appear in Section 7). For GQA, remove the restated ratio from the prose since it is in the table two lines above.

## MINOR Suggestions

### [architecture_and_hyperparams.md] ~line 5
**Issue:** "This is the same architecture class used by Qwen3.5-35B-A3B" is a parenthetical that may be useful context but is stated again implicitly by the architecture class name appearing in the table on line 13. Mild redundancy.
**Suggestion:** Merge into the first sentence: "Qwen3.6-35B-A3B uses the `Qwen3_5MoeForConditionalGeneration` architecture class (shared with Qwen3.5-35B-A3B), model type `qwen3_5_moe`." Saves one sentence.

### [architecture_and_hyperparams.md] ~line 26
**Issue:** "The 262K context window is the native training length. With appropriate RoPE scaling or NTK-aware interpolation, the model can be extended to approximately 1M tokens at inference time, though quality may degrade beyond the native length." The phrase "though quality may degrade beyond the native length" is hedging that is already implied by "approximately" and "extended."
**Suggestion:** Shorten to: "The 262K context window is the native training length; with RoPE scaling, inference can extend to ~1M tokens."

### [architecture_and_hyperparams.md] ~lines 84-88
**Issue:** The RMSNorm formula and its explanation ("where $w$ is the learned per-dimension weight initialized to zero, ensuring the initial effective scale is 1.0") appear once here, then the concept of zero-centered RMSNorm with `add_unit_offset=True` is referenced again at line 112 and line 161-162 ("zero-centered RMSNorm with separate learned weights"). Minor cross-file duplication also occurs in `forward_pass_dataflow.md` lines 32, 37, 96, 135.
**Suggestion:** Define the zero-centered RMSNorm formula once in Section 3 (as done), and in all subsequent mentions simply say "zero-centered RMSNorm" without re-explaining the formulation.

### [forward_pass_dataflow.md] ~lines 1-3
**Issue:** "This file traces the complete forward pass of Qwen3.6-35B-A3B for both text-only and multimodal inputs. All tensor shapes and operations reference the hyperparameters established in architecture_and_hyperparams.md." This preamble is serviceable but could be tighter.
**Suggestion:** "Traces the complete forward pass for text-only and multimodal inputs. Tensor shapes reference [architecture_and_hyperparams.md](./architecture_and_hyperparams.md)." Saves one line.

### [forward_pass_dataflow.md] ~lines 149-174
**Issue:** The "Complete Single-Token Decode Summary" ASCII diagram (lines 149-174) restates the flow already described in Stages 1-4 (lines 7-147). While summary diagrams have value, this one is 25 lines and every element in it was already shown in the preceding pseudocode blocks.
**Suggestion:** Trim the summary diagram to show only the layer pattern and omit the embedding/LM-head stages that were just described. Alternatively, collapse to ~10 lines by removing the per-layer expansion (lines 158-164 spell out layers 0-3 then ellipsis, which the reader already understands from the pattern).

### [forward_pass_dataflow.md] ~lines 319-327
**Issue:** The comment block in Step 2 includes: "This sync to host is mandatory: the host must know which expert indices to dispatch. By running top-k and softmax on CPU, no custom device kernel is needed, and the data volume (512 bytes) makes the DMA transfer negligible." The phrase "the host must know which expert indices to dispatch" is obvious from context (the code just showed top-k on CPU). The justification for why CPU is fine (512 bytes, no custom kernel needed) spans 3 lines where 1 would suffice.
**Suggestion:** Shorten to: "The 512-byte transfer to host for top-k selection is negligible; no custom device kernel is needed."

### [forward_pass_dataflow.md] ~lines 398-403
**Issue:** The "Routing Characteristics" bullet list (lines 400-403) restates numbers from the preceding walkthrough: "256 routed + 1 shared = 257 total" (from Step 4 context), "9/257 ~ 3.5%" (from architecture file line 343), "512 bytes" (from Step 2), and expert matmul sizes (from Step 4 code). This is a summary-after-the-summary.
**Suggestion:** Remove entirely, or retain only the final bullet about DRAM bandwidth being the likely bottleneck (line 403), which is the only new insight not stated elsewhere.

### [architecture_and_hyperparams.md] ~lines 193-197
**Issue:** The SwiGLU formula and variable definitions (lines 194-197) are standard and repeated nearly verbatim in `forward_pass_dataflow.md` Step 4 code (lines 355-358) and the routing visualization (lines 387-391). The formula itself is not bloat in the architecture file, but having the identical computation shown three times across the two files is redundant.
**Suggestion:** Define the SwiGLU expert formula once in the architecture file and reference it from the dataflow file rather than re-showing the full gate/up/down sequence in both the step-by-step and the diagram.

## Load-Bearing Evidence
N/A -- Crucial updates are present.

## VERDICT
- Crucial updates: yes

---

## Change Log

**2026-04-21 -- All 6 CRUCIAL suggestions applied:**

1. **State management duplication (forward_pass_dataflow.md Section 3):** Replaced ~35 lines of re-derived GDN state sizes, conv buffer sizes, KV cache sizes, and memory totals with a single-sentence cross-reference to architecture_and_hyperparams.md Sections 4--5. Retained the State Comparison table and hybrid tradeoff narrative.
2. **Repeated sublayer pseudocode (forward_pass_dataflow.md):** Collapsed the two identical pseudocode blocks into a single "Common Decoder Layer Structure" showing the shared skeleton once, with the differing attention call parameterized. Removed the duplicated block from the Gated Attention section.
3. **Triple-stated layer layout (architecture_and_hyperparams.md Section 2):** Removed the bullet breakdown (indices listing and ratio explanation). Kept the pattern string and table, with a single-sentence summary of the 75%/25% split.
4. **Routing visualization (forward_pass_dataflow.md Section 4):** Removed the 22-line ASCII diagram. The step-by-step walkthrough (Steps 1--5) remains as the sole description.
5. **Context Length subsection (architecture_and_hyperparams.md Section 10):** Removed the entire Context Length sub-section. Replaced with a single-sentence cross-reference to Sections 1 and 5.
6. **Partial RoPE duplication (architecture_and_hyperparams.md Section 5):** Replaced the Section 5 Partial RoPE paragraph with a forward reference to Section 7. Also trimmed the restated GQA ratio from the Key Extensions prose since it appears in the table above.

---

# Compression Analysis: Chapter 1 -- Pass 2 (Re-check of CRUCIAL items)

## Summary
- Files re-analyzed: 2 (`architecture_and_hyperparams.md`, `forward_pass_dataflow.md`)
- Pass 1 flagged 6 CRUCIAL items; all 6 were marked as applied in the Change Log
- Pass 2 finding: all 6 CRUCIAL fixes are confirmed present in the files; no regressions detected
- Estimated current line count: ~700 lines (345 + 353)
- Estimated further compressible lines: ~20 lines
- Estimated reduction from further changes: ~3%

## Re-check of Each CRUCIAL Item

### CRUCIAL #1: State management duplication (forward_pass_dataflow.md Section 3)
**Status: CONFIRMED FIXED.** Lines 232--248 of `forward_pass_dataflow.md` now contain a single cross-reference sentence ("State sizes and per-layer memory derivations are detailed in architecture_and_hyperparams.md, Sections 4--5") followed by only the State Comparison table and the hybrid tradeoff paragraph. No re-derived GDN state sizes, conv buffer sizes, KV cache per-token formulas, or memory totals remain.

### CRUCIAL #2: Repeated sublayer pseudocode (forward_pass_dataflow.md)
**Status: CONFIRMED FIXED.** Lines 24--40 present a single "Common Decoder Layer Structure" pseudocode block. The Gated Attention section (line 84) begins directly with its internal steps without repeating the RMSNorm/residual/MoE skeleton.

### CRUCIAL #3: Triple-stated layer layout (architecture_and_hyperparams.md Section 2)
**Status: CONFIRMED FIXED.** The bullet breakdown listing layer indices and separate ratio explanation is gone. Lines 32--54 now contain: the pattern string (line 35), a single sentence with the 75%/25% ratio (line 38), and the Layer Layout Table (lines 40--53).

### CRUCIAL #4: Routing visualization (forward_pass_dataflow.md Section 4)
**Status: CONFIRMED FIXED.** No separate ASCII routing diagram follows Step 5. The section proceeds directly from Step 5 (line 310) to "Routing Characteristics" (line 318).

### CRUCIAL #5: Context Length subsection (architecture_and_hyperparams.md Section 10)
**Status: CONFIRMED FIXED.** Section 10 is titled "Parameter Count Analysis" and ends at line 341 with a cross-reference: "Context length and KV cache sizing are covered in Section 1 and Section 5." No standalone Context Length sub-section exists.

### CRUCIAL #6: Partial RoPE duplication (architecture_and_hyperparams.md Section 5)
**Status: CONFIRMED FIXED.** Line 155 reads: "Only the first 64 of 256 head dimensions receive rotary encoding. See Section 7 for the full RoPE configuration including frequency spectrum and M-RoPE details." No frequency formula or theta value is restated. The GQA prose (line 161) uses "see table above" without restating the 8:1 ratio.

## Load-Bearing Evidence

- **architecture_and_hyperparams.md, lines 40--53 (Layer Layout Table):** This table provides a concrete visual mapping of which specific layer indices correspond to which attention type. While the pattern string on line 35 conveys the same information abstractly, implementers indexing into `config.json["layer_types"]` benefit from seeing the explicit layer-number-to-type mapping (e.g., confirming layer 39 is `full_attention`). The table cannot be cut without forcing the reader to mentally expand the pattern.

- **forward_pass_dataflow.md, lines 136--159 (Complete Single-Token Decode Summary diagram):** This ASCII flow diagram is the only place in the two files that shows the full end-to-end path -- from embedding lookup through all 40 layers to argmax -- in a single visual. The preceding Stages 1--4 describe each piece in isolation; this diagram stitches them together. Removing it would leave no compact visual overview of the entire decode step.

- **architecture_and_hyperparams.md, lines 165--170 (KV Cache Size derivation):** This derivation (2,048 bytes per layer per token, 5.1 GB at 262K) is the canonical source that forward_pass_dataflow.md Section 3 now cross-references. It cannot be cut because doing so would orphan the cross-reference and leave no KV cache sizing in either file.

- **forward_pass_dataflow.md, lines 318--323 (Routing Characteristics):** The final bullet ("These are small matmuls that may not fully saturate accelerator compute units, making DRAM bandwidth the likely bottleneck for loading expert weights") is the only statement in either file that identifies the performance bottleneck for expert dispatch. The preceding steps show what happens but not what limits throughput.

## MINOR Suggestions

### [architecture_and_hyperparams.md] lines 42--53: Layer Layout Table FFN Type column
**Issue:** The "FFN Type" column contains "MoE" in every single row, and line 54 additionally states "Every layer, regardless of attention type, uses the same MoE FFN block." A uniform column conveys no distinguishing information.
**Suggestion:** Remove the FFN Type column from the table and keep line 54 as the sole statement that all layers use MoE. Saves ~1 character per row but more importantly removes visual clutter from a reference table.

### [forward_pass_dataflow.md] lines 265--272: Host-side top-k justification
**Issue:** The paragraph after the top-k code block (lines 270--272) reads: "This sync to host is mandatory: the host must know which expert indices to dispatch. By running top-k and softmax on CPU, no custom device kernel is needed, and the data volume (512 bytes) makes the DMA transfer negligible." Three sentences to justify a 512-byte transfer.
**Suggestion:** Shorten to: "The 512-byte host sync for top-k is negligible; no custom device kernel is needed." One sentence conveys the same two points (small data volume, no kernel required).

### [architecture_and_hyperparams.md] line 170: Hypothetical comparison
**Issue:** "This is significantly smaller than what a 40-layer full-attention model would require, because only 10 of 40 layers need KV caches and each layer has only 2 KV heads." The comparison to a hypothetical 40-layer model restates what the reader already knows from the 75/25 split and the GQA ratio in the table above.
**Suggestion:** Shorten to: "Only 10 of 40 layers maintain KV caches, each with just 2 heads." The implication that this is smaller than full attention is self-evident.

### [forward_pass_dataflow.md] lines 320--322: Restated numbers in Routing Characteristics
**Issue:** The first three bullets of Routing Characteristics restate numbers from the immediately preceding Steps 1--5: expert count (257), activation ratio (3.5%), and router data volume (512 bytes). These exact figures appear in the step-by-step walkthrough and in architecture_and_hyperparams.md line 339.
**Suggestion:** Remove the first three bullets and retain only the fourth (DRAM bandwidth bottleneck insight), which is the only new content. The summary numbers are available in both the walkthrough above and the architecture file.

## VERDICT
- Crucial updates: no

---

# Compression Analysis: Chapter 1 -- Pass 2 (Agent C Independent Re-check)

## Summary
- Files re-analyzed: 2 (`architecture_and_hyperparams.md`, `forward_pass_dataflow.md`)
- Current line count: 346 + 354 = 700 lines
- All 6 CRUCIAL items from Pass 1 independently confirmed fixed in file content
- All 4 MINOR items from the prior Pass 2 section remain unapplied in file content
- One new cross-file duplication identified (token IDs)
- Estimated further compressible lines: ~18 lines
- Estimated reduction from further changes: ~2.5%

## Re-check of Each CRUCIAL Item (Independent Verification)

### CRUCIAL #1: State management duplication (forward_pass_dataflow.md Section 3)
**Status: CONFIRMED FIXED.** `forward_pass_dataflow.md` line 234 reads: "State sizes and per-layer memory derivations are detailed in [architecture_and_hyperparams.md, Sections 4--5](./architecture_and_hyperparams.md#4-gated-deltanet-configuration)." Only the State Comparison table (lines 238--248) and the hybrid tradeoff paragraph (lines 247--248) follow. No GDN state size derivation, conv buffer formula, KV cache bytes-per-token calculation, or memory total appears in this section.

### CRUCIAL #2: Repeated sublayer pseudocode (forward_pass_dataflow.md)
**Status: CONFIRMED FIXED.** Lines 24--40 present a single "Common Decoder Layer Structure" pseudocode block with a parameterized dispatch note on line 26. The Gated Attention section beginning at line 84 goes directly to its internal step list without reproducing the RMSNorm/residual/MoE skeleton.

### CRUCIAL #3: Triple-stated layer layout (architecture_and_hyperparams.md Section 2)
**Status: CONFIRMED FIXED.** The section now contains: the pattern string (line 35), a single combined sentence with the 75%/25% ratio and mechanism names (line 38), and the Layer Layout Table (lines 40--53). The previous separate bullet-by-bullet index listing is absent.

### CRUCIAL #4: Routing visualization (forward_pass_dataflow.md Section 4)
**Status: CONFIRMED FIXED.** Step 5 ends at line 313. The section moves directly to "Routing Characteristics" (line 318) with no intervening ASCII diagram.

### CRUCIAL #5: Context Length subsection (architecture_and_hyperparams.md Section 10)
**Status: CONFIRMED FIXED.** Section 10 is "Parameter Count Analysis." Line 341 reads: "Context length and KV cache sizing are covered in [Section 1](#1-top-level-hyperparameters) and [Section 5](#5-gated-attention-configuration)." No standalone Context Length subsection exists.

### CRUCIAL #6: Partial RoPE duplication (architecture_and_hyperparams.md Section 5)
**Status: CONFIRMED FIXED.** Section 5 lines 155--156 contain only the forward reference: "Only the first 64 of 256 head dimensions receive rotary encoding. See [Section 7](#7-rope-configuration) for the full RoPE configuration including frequency spectrum and M-RoPE details." The frequency formula and theta value appear only in Section 7. The GQA prose on line 161 reads "The high GQA ratio (see table above)" without restating the 8:1 ratio numerically.

## Re-check of Prior Pass 2 MINOR Items (Pending Application)

All four MINOR items from the prior Pass 2 section remain unapplied in the current file content:

- **MINOR #1** (FFN Type column in Layer Layout Table): The column still exists with "MoE" in every row, and line 54 still states the same fact in prose.
- **MINOR #2** (Host-side top-k justification, forward_pass_dataflow.md lines 270--272): The three-sentence explanation still reads verbatim as flagged.
- **MINOR #3** (Hypothetical comparison, architecture_and_hyperparams.md line 170): "This is significantly smaller than what a 40-layer full-attention model would require..." still present.
- **MINOR #4** (Routing Characteristics first three bullets): All four bullets remain; the first three restate figures already visible in the preceding Steps 1--5.

## New Finding: Token ID Cross-File Duplication

### [forward_pass_dataflow.md] lines 212--216 vs [architecture_and_hyperparams.md] lines 280--287
**Issue:** `architecture_and_hyperparams.md` contains a four-row Special Tokens table (Section 8, lines 280--287) listing all four vision token IDs. `forward_pass_dataflow.md` lines 212--216 reproduce the identical four IDs as a bullet list immediately after the interleaving sequence diagram. The two presentations differ only in format (table vs. bullets); the token ID values and names are identical in both.
**Assessment:** The forward_pass_dataflow.md bullet list provides inline context for the interleaving diagram on line 209, so it is not without purpose. However, a brief inline comment referencing the architecture file table would serve the same purpose in fewer lines. This is a MINOR issue, not CRUCIAL, because the duplication is compact (5 lines) and the inline placement provides genuine readability value for the interleaving diagram.

## Load-Bearing Evidence

- **architecture_and_hyperparams.md, lines 40--53 (Layer Layout Table):** The pattern string on line 35 expresses the repeating unit abstractly; the table makes specific layer indices unambiguous for implementers (e.g., confirming layers 3, 7, 11 are `full_attention` and layer 39 is the final `full_attention`). An implementer building the layer dispatch loop benefits from the concrete index-to-type mapping without needing to mentally expand 10 repetitions of the pattern.

- **forward_pass_dataflow.md, lines 136--159 (Complete Single-Token Decode Summary diagram):** This is the only location in either file presenting the full decode path -- embedding lookup through 40 layers to argmax -- as a single continuous visual. Each preceding stage (1--4) describes its own piece in isolation. The summary diagram provides the integrating view that allows a reader to confirm they understand the full chain without re-reading all four stages. It is not redundant with the stages; it is a synthesis of them.

- **architecture_and_hyperparams.md, lines 165--170 (KV Cache Size derivation):** The derivation (2,048 bytes/layer/token, 5.1 GB at 262K context) is the authoritative source cross-referenced by `forward_pass_dataflow.md` Section 3. Removing it would leave the cross-reference dangling and eliminate the only per-token KV cache sizing from both files.

- **forward_pass_dataflow.md, lines 276--288 (Shared expert overlap explanation, lines 292--312 routed expert loop):** The overlap between shared expert computation and host-side top-k is described in the narrative surrounding Step 3 (line 275: "While the router logits are being transferred and processed on the host, the device queue has already begun computing the shared expert forward pass"). This is the only place in either file that explains the latency-hiding mechanism; it is implementation-critical and cannot be cut.

## MINOR Suggestions

### [architecture_and_hyperparams.md] lines 44--53: FFN Type column (carries over from prior Pass 2 MINOR #1)
**Issue:** Every row of the Layer Layout Table has "MoE" in the FFN Type column. Line 54 restates this as prose. A uniform column adds visual bulk without conveying variation.
**Suggestion:** Remove the FFN Type column; keep line 54 ("Every layer, regardless of attention type, uses the same MoE FFN block") as the sole statement.

### [forward_pass_dataflow.md] lines 270--272: Verbose host-sync justification (carries over from prior Pass 2 MINOR #2)
**Issue:** Three sentences justify a 512-byte CPU transfer. The mandatory nature of the sync is established by the code block above it; the volume and no-custom-kernel points can be merged.
**Suggestion:** Replace with: "The 512-byte host sync for top-k is negligible; no custom device kernel is needed."

### [forward_pass_dataflow.md] lines 319--322: Routing Characteristics first three bullets (carries over from prior Pass 2 MINOR #4)
**Issue:** "Expert count: 256 routed + 1 shared = 257 total," "Activation ratio: 9/257 ~3.5%," and "Router data volume per sync: 256 logits × 2 bytes = 512 bytes" all restate numbers that appear verbatim in the preceding steps and in `architecture_and_hyperparams.md` line 339.
**Suggestion:** Remove the first three bullets; retain only the fourth (DRAM bandwidth bottleneck), which is the only non-restated insight in the list.

### [forward_pass_dataflow.md] lines 212--216: Token ID bullets (new)
**Issue:** The four vision token IDs listed here appear identically in the Special Tokens table in `architecture_and_hyperparams.md` Section 8.
**Suggestion:** Replace the four bullet lines with a single inline reference: "(token IDs listed in [architecture_and_hyperparams.md Section 8](./architecture_and_hyperparams.md#8-vision-encoder-configuration))." The interleaving diagram on line 209 communicates the structure; the exact IDs are a lookup detail better kept in one place.

## VERDICT
- Crucial updates: no
