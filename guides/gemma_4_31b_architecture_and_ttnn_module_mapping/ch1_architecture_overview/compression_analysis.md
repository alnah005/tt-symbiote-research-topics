# Change Log

## Agent B Pass 1 Fixes
- Fixed layer count error across all four chapter files: replaced "48 sliding" with "50 sliding" and "12 global" with "10 global" (actual config.json has 50 sliding_attention + 10 full_attention layers at indices 5, 11, 17, 23, 29, 35, 41, 47, 53, 59).
- Updated K=V memory savings in novel_components.md from "264 MB" to "~220 MB" (10 layers x ~22 MB instead of 12 x 22 MB).
- Fixed arithmetic error in heterogeneous_attention_configs.md: replaced incorrect global attention parameter count 186,646,528 with correct value 187,170,816.

---

# Compression Analysis: Chapter 1 — Architecture Overview — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~566 lines
- Estimated post-compression line count: ~430 lines
- Estimated reduction: ~24%

## CRUCIAL Suggestions
### [novel_components.md] ~lines 8-33
**Issue:** The K=V sharing section duplicates most of the explanation already given in `heterogeneous_attention_configs.md` lines 77-117, including the same dataflow (shared projection, K path with scaled RMSNorm + partial RoPE, V path with unscaled RMSNorm and no RoPE), the same weight shape [5376, 2048], and the same conceptual justification. The ASCII diagram in `heterogeneous_attention_configs.md` also covers the same flow.
**Suggestion:** Consolidate K=V sharing into a single canonical location in `heterogeneous_attention_configs.md` (which already has the diagram). In `novel_components.md`, reduce the K=V sharing section to a 2-3 line summary with a cross-reference link, retaining only the parameter savings calculation (~220 MB) which is unique to that file.

### [novel_components.md] ~lines 195-208
**Issue:** The "Post-Attention and Post-FFN Norms" section repeats the four-RMSNorm-per-layer explanation already given in `layer_organization.md` lines 54-67 (table of submodules) and lines 64-67 (explicit note about four norms vs. two in LLaMA). The numbered list of the four norms is identical information.
**Suggestion:** Remove the "Post-Attention and Post-FFN Norms" section from `novel_components.md` entirely. Add a one-line entry to the summary table at the bottom referencing `layer_organization.md` for details. The layer_organization file already provides the authoritative description.

### [heterogeneous_attention_configs.md] ~lines 65-75
**Issue:** The "GQA Group Sizes" section restates values already present in the side-by-side comparison table at line 24 (GQA group size row: "32 / 16 = 2" and "32 / 4 = 8"). The follow-up paragraph about KV cache memory for global layers is the only new content.
**Suggestion:** Delete the "GQA Group Sizes" heading and the bullet restatement of 32/16=2 and 32/4=8. Move the KV cache memory observation (lines 73-75) into the "Implications for TTNN Implementation" section as item 2, where KV cache is already discussed.

### [layer_organization.md] ~lines 78-159
**Issue:** The block diagram (lines 78-138) and the numbered "Forward Pass Data Flow" list (lines 142-154) convey the same information in two formats. The numbered list adds only the detail that post-norm happens before residual add, which is already shown in the diagram's box ordering.
**Suggestion:** Keep the block diagram and remove the numbered forward pass list. Move the one unique observation (lines 156-159, about post-norm-then-residual differing from standard pre-norm-only) into a short note below the diagram.

## MINOR Suggestions
### [index.md] ~lines 66-69
**Issue:** The RoPE parameters table duplicates information from the side-by-side comparison table in `heterogeneous_attention_configs.md` lines 27-31. Both list rope_type, rope_theta, and partial_rotary_factor per layer type.
**Suggestion:** Replace the RoPE table in `index.md` with a one-line note: "See the side-by-side table in `heterogeneous_attention_configs.md` for per-type RoPE parameters."

### [novel_components.md] ~lines 78-133
**Issue:** The PLE section is 55 lines documenting a feature that is disabled in the 31B model. While useful for the architecture family, the detail level (multimodal handling, injection submodules, math formulas) is disproportionate for a disabled feature in a guide specifically about the 31B variant.
**Suggestion:** Condense to ~15 lines: keep the status note, a brief conceptual description, and mention it exists for other variants. Move the full PLE specification to an appendix or a separate reference file if coverage of the broader Gemma 4 family is desired.

### [heterogeneous_attention_configs.md] ~lines 1-14
**Issue:** The opening paragraph contains hedging and verbose phrasing: "Unlike models that simply vary the attention mask (e.g., alternating causal and sliding)" is a parenthetical comparison to unnamed models that adds little. The sentence "This means the weight tensors have different shapes depending on the layer type" restates what lines 39-49 will show with a table.
**Suggestion:** Tighten the opening to two sentences: state that Gemma 4 31B uses two structurally different attention configurations (varying head count, head dim, RoPE, and KV strategy) and that both use the same class configured at construction time.

### [novel_components.md] ~lines 26-33
**Issue:** "Value vectors carry semantic content that should not be position-dependent" is a speculative justification. The sentence "The divergent post-processing (different norms and selective RoPE) ensures that the K and V representations remain functionally distinct despite originating from the same weights" restates the dataflow already described in the previous lines.
**Suggestion:** Cut to one sentence: "Sharing the projection saves one [5376, 2048] matrix per global layer (~220 MB total at BF16 across 10 layers)."

### [novel_components.md] ~lines 146-154
**Issue:** The logit softcapping implementation is described both as a formula (line 143) and as a three-step numbered list (lines 148-150). The numbered list restates the formula in words.
**Suggestion:** Remove the three-step numbered list; the formula is self-explanatory.

### [layer_organization.md] ~lines 39-47
**Issue:** Lines 39-43 state two "invariants" that are obvious from the pattern shown in lines 22-35 (every 6th layer at position 6k+5 is global; the final layer is global). Lines 44-47 then re-explain how the pattern is stored in config.json, which is already stated in `index.md` lines 73-76.
**Suggestion:** Remove the two named invariants; they are visually apparent from the pattern table. Keep the config.json reference but shorten to one line.

## VERDICT
- Crucial updates: yes

---

## Agent A — Compression Applied (2026-04-03)

All four CRUCIAL suggestions have been applied:

1. **novel_components.md — K=V sharing:** Reduced from ~26 lines to a 5-line summary with cross-reference to `heterogeneous_attention_configs.md`. Retained the unique ~220 MB parameter savings calculation.
2. **novel_components.md — Four-RMSNorm:** Removed the "Post-Attention and Post-FFN Norms" section entirely (~14 lines). Updated the summary table entry for "4x RMSNorm per layer" to reference `layer_organization.md`.
3. **heterogeneous_attention_configs.md — GQA Group Sizes:** Deleted the "GQA Group Sizes" heading and bullet restatement (~11 lines). Moved the KV cache memory observation into TTNN Implications item 2.
4. **layer_organization.md — Forward pass list:** Removed the numbered forward pass list (~18 lines). Preserved the block diagram and added a short note below it about the post-norm-then-residual pattern differing from standard pre-norm-only designs.

---

# Compression Analysis: Chapter 1 — Architecture Overview — Pass 2

## Summary
- Total files analyzed: 4
- Estimated current line count: ~590 lines
- Estimated post-compression line count: ~530 lines
- Estimated reduction: ~10%

## CRUCIAL Suggestions
None

All four CRUCIAL items from Pass 1 have been verified as fixed:
1. **K=V duplication (novel_components.md):** Section is now 5 lines (lines 8-17) with a cross-reference link to `heterogeneous_attention_configs.md`. No duplicated dataflow or diagram. CONFIRMED FIXED.
2. **4-RMSNorm duplication (novel_components.md):** The "Post-Attention and Post-FFN Norms" section has been removed entirely. The summary table (line 218) references `layer_organization.md`. CONFIRMED FIXED.
3. **GQA restatement (heterogeneous_attention_configs.md):** No separate "GQA Group Sizes" section exists. The KV cache memory observation has been integrated into TTNN Implications item 2 (lines 116-121). CONFIRMED FIXED.
4. **Block diagram + list redundancy (layer_organization.md):** The numbered forward pass list has been removed. A concise note about the post-norm-then-residual pattern follows the diagram (lines 140-143). CONFIRMED FIXED.

## MINOR Suggestions
### [index.md] lines 64-69 — RoPE table still duplicates heterogeneous_attention_configs.md
The RoPE parameters table (rope_type, rope_theta, partial_rotary_factor per layer type) is still present in both `index.md` and the side-by-side table in `heterogeneous_attention_configs.md` (lines 27-31). Replace the index.md table with a one-line cross-reference. Saves ~7 lines.

### [novel_components.md] lines 62-117 — PLE section remains disproportionate for a disabled feature
At ~56 lines, the PLE section documents multimodal handling, injection submodule details, and math formulas for a feature that is disabled in 31B (`hidden_size_per_layer_input=0`). Could be condensed to ~15 lines with a pointer to an appendix.

### [novel_components.md] lines 130-134 — Logit softcapping formula and step list still coexist
The formula at line 127 and the three-step numbered list at lines 131-133 convey identical information. Removing the numbered list saves 5 lines with no information loss.

### [layer_organization.md] lines 39-43 — "Two invariants" are self-evident from the pattern table
The two named invariants (every 6th layer at 6k+5 is global; final layer is global) are visually obvious from the 10-group pattern table at lines 22-33. These 5 lines could be removed.

### [heterogeneous_attention_configs.md] lines 5-10 — Opening paragraph remains verbose
The comparison to unnamed models ("Unlike models that simply vary the attention mask") and the sentence "This means the weight tensors have different shapes depending on the layer type" could be tightened from 6 lines to 2-3 lines.

## Load-Bearing Evidence
- **index.md** (line 50): `| attention_k_eq_v | true | K=V sharing enabled in global layers |` — This parameter table is the single consolidated reference for all text_config parameters; every row maps directly to a config.json field and cannot be cut without losing the quick-lookup purpose of the file.
- **layer_organization.md** (lines 78-138): The block diagram is the only visual representation of the decoder layer dataflow in the entire chapter. It shows the ordering of all four RMSNorm layers, residual connections, and the MLP structure. Removing it would force readers to reconstruct the flow from prose alone.
- **heterogeneous_attention_configs.md** (lines 65-85): The K=V sharing section with its ASCII diagram is now the single canonical location for the shared-projection dataflow. After Pass 1 consolidated this content here from novel_components.md, removing it would leave no detailed description of K=V mechanics anywhere in the chapter.
- **novel_components.md** (lines 19-61): The V-norm section (definition, contrast with standard RMSNorm, TTNN implementation strategies) is unique content not duplicated elsewhere. The three TTNN implementation options (all-ones weight, with_scale=False path, manual ops) are actionable engineering guidance that cannot be cut.

## VERDICT
- Crucial updates: no
