# Change Log

## Agent B Pass 1 Fixes
- config_parameters.md: Corrected MLP parameter count per layer from 14,860,416 to 14,874,624 (the correct result of 3 * 1152 * 4304).
- config_parameters.md: Updated per-layer total from 20,173,584 to 20,187,792.
- config_parameters.md: Updated 27-layer total from 544,686,768 to 545,070,384.
- module_hierarchy.md: Added explicit transpose step (`freqs = freqs.transpose(1, 2)` giving `[batch, num_patches, 18]`) between the frequency computation (shape `[batch, 18, num_patches]`) and the concatenation step in the RoPE forward pass, fixing the undocumented dimension order change.

## Agent B Pass 2 Fixes
- index.md: Changed "approximately 550M parameters" to "approximately 570M parameters" to match the detailed ~569.6M calculation in config_parameters.md.
- config_parameters.md: Changed MLP expansion ratio from 3.73 to 3.74 (4304/1152 = 3.736... rounds to 3.74, not 3.73).

---

# Compression Analysis: Chapter 1 — Gemma 4 Vision Architecture — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~592 lines
- Estimated post-compression line count: ~530 lines
- Estimated reduction: ~10%

## CRUCIAL Suggestions
### [variable_resolution_processing.md] ~lines 15-28 and [index.md] ~line 32
**Issue:** The divisibility-by-48 constraint is explained in full three times across the chapter: once in index.md (line 32), once in the config_parameters.md derived-dimensions table (line 58), and at length in variable_resolution_processing.md (lines 15-28). The variable_resolution_processing.md version is the most detailed, but the other two restate the same formula and rationale independently.
**Suggestion:** Keep the full explanation only in variable_resolution_processing.md. In index.md line 32, trim to: "Both height and width must be divisible by 48 (see [variable_resolution_processing.md](./variable_resolution_processing.md) for derivation)." The config_parameters.md derived-dimensions table row on line 58 is fine as a formula-only reference and can stay.

### [variable_resolution_processing.md] ~lines 44-52 and [module_hierarchy.md] ~line 45
**Issue:** The "no ImageNet normalization" fact and the `2 * (pixel_values - 0.5)` rescaling code are stated in both module_hierarchy.md (line 45, describing the patch embedder forward pass) and variable_resolution_processing.md (lines 44-57, as a standalone section with code block). The variable_resolution_processing.md version adds the three-step processor summary, but the core fact and code snippet are duplicated verbatim.
**Suggestion:** In variable_resolution_processing.md lines 44-57, replace the code block and surrounding prose with a one-line cross-reference: "The patch embedder rescales pixels from [0,1] to [-1,1] internally (see [module_hierarchy.md, Patch Embedder](./module_hierarchy.md#gemma4visionpatchembedder)) — no ImageNet mean/std normalization is applied." Keep the three-step processor summary (lines 54-57) as it adds new information about the processor pipeline.

### [config_parameters.md] ~lines 37-44 and [module_hierarchy.md] ~lines 82-92
**Issue:** The RoPE inverse-frequency formula is stated twice with nearly identical LaTeX. module_hierarchy.md gives it as part of the RoPE module description (lines 88-92), and config_parameters.md restates it under "RoPE Frequency Derivation" (lines 37-44) with the same formula and the same "18 frequency pairs" derivation.
**Suggestion:** Keep the formula in module_hierarchy.md (the module-level description is the natural home). In config_parameters.md, replace lines 37-44 with a cross-reference: "See [module_hierarchy.md, RoPE Parameter Computation](./module_hierarchy.md#rope-parameter-computation) for the full frequency derivation." Retain the `rope_theta` and `rope_type` table rows.

## MINOR Suggestions
### [index.md] ~lines 30-38
**Issue:** The overview paragraph restates three defining characteristics that are then expanded in detail across the other three files. The phrasing "represents a significant architectural departure" is hedging/marketing language.
**Suggestion:** Shorten to: "Gemma 4 replaces Gemma 3's frozen SigLIP encoder (fixed 224x224 or 896x896 square inputs) with a custom vision encoder featuring: (1) variable-resolution input preserving native aspect ratio, (2) 2D Rotary Position Embeddings with explicit (x, y) grid coordinates, and (3) configurable token budgets (70 to 1120 soft tokens). The encoder is ~570M parameters with hidden dimension 1152, projected to 5376 for the 31B language model."

### [variable_resolution_processing.md] ~line 3
**Issue:** "Unlike the Gemma 3 SigLIP encoder, which resized all images to a fixed square (224x224 or 896x896)" duplicates the same Gemma 3 comparison already made in index.md line 30.
**Suggestion:** Trim to: "Unlike Gemma 3's fixed-square SigLIP encoder, Gemma 4 preserves the original aspect ratio of each image."

### [module_hierarchy.md] ~line 191
**Issue:** The `Gemma4ClippableLinear` explanation ("wraps `nn.Linear(bias=False)` with optional input/output clamping. For the 31B model, `use_clipped_linears=False`, so these are plain bias-free linears") appears in the attention TTNN porting notes, and the same fact is captured in config_parameters.md line 26 (`use_clipped_linears=False`). The explanation is implicitly restated in the MLP section (line 220) with "Three bias-free linears."
**Suggestion:** State the `Gemma4ClippableLinear` explanation once in the attention section (line 191), and in the MLP TTNN porting notes (line 220) add "(see attention notes above for `Gemma4ClippableLinear` details)" instead of re-deriving "bias-free."

### [config_parameters.md] ~lines 100-106
**Issue:** The pixel-count derivation formula ("And since each patch covers 16 * 16 = 256 pixels... roughly equivalent to a 804x804 image") overlaps with variable_resolution_processing.md's Example Resolutions table (lines 32-41), which covers the same ground with concrete per-image examples.
**Suggestion:** Replace config_parameters.md lines 100-106 with: "See [variable_resolution_processing.md](./variable_resolution_processing.md) for worked examples of how pixel budgets map to specific resolutions."

### [module_hierarchy.md] ~line 160
**Issue:** The Tip block explaining the "sandwich norm" pattern is verbose: "This differs from the more common pre-norm-only pattern (where normalization is applied only before the sub-layer). The post-norms act as additional stabilizers on the residual stream."
**Suggestion:** Shorten to: "This is a 'sandwich norm' pattern — normalization both before and after the sub-layer. The post-norms stabilize the residual stream."

## Load-Bearing Evidence
- `index.md` line ~38: "The encoder produces approximately 570M parameters and outputs hidden states of dimension 1152" — load-bearing because it anchors the parameter count that must match config_parameters.md's detailed breakdown
- `module_hierarchy.md` line ~184: "scaling=1.0 (no 1/sqrt(d) — the QK norms serve as the scaling mechanism)" — load-bearing because this non-standard scaling is a critical TTNN porting detail that must not be lost
- `config_parameters.md` line ~34: "rope_theta | 100.0" — load-bearing because this value differs from the language model's theta and directly affects RoPE frequency computation
- `variable_resolution_processing.md` line ~42: "The actual number of pooled tokens varies per image depending on its aspect ratio" — load-bearing because it warns that token counts are approximate, which is essential for TTNN memory planning

## VERDICT
- Crucial updates: yes

---

## Agent A Compression Pass — Applied CRUCIAL Suggestions

1. **index.md line 32 (divisibility-by-48):** Trimmed the inline explanation to a cross-reference pointing to `variable_resolution_processing.md` for the full derivation. The `config_parameters.md` derived-dimensions table row was left as-is.
2. **variable_resolution_processing.md lines 44-52 (no ImageNet normalization):** Replaced the duplicated code block and prose with a one-line cross-reference to `module_hierarchy.md`. Retained the three-step processor summary.
3. **config_parameters.md lines 37-44 (RoPE formula):** Replaced the duplicated frequency derivation formula with a cross-reference to `module_hierarchy.md`. Retained the `rope_theta` and `rope_type` table rows and the Tip about theta scaling.

---

# Compression Analysis: Chapter 1 — Gemma 4 Vision Architecture — Pass 2

## Summary
- Total files analyzed: 4
- Estimated current line count: ~582 lines
- Estimated post-compression line count: ~555 lines
- Estimated reduction: ~5%

## CRUCIAL Suggestions
None

All three CRUCIAL items from Pass 1 have been verified as fixed:
1. **Divisibility-by-48 triple explanation:** index.md line 32 now contains a concise cross-reference to variable_resolution_processing.md. The config_parameters.md derived-dimensions table retains only the formula row. No remaining duplication.
2. **No-ImageNet duplication:** variable_resolution_processing.md lines 44-46 now cross-reference module_hierarchy.md instead of restating the rescaling code. The three-step processor summary is preserved as unique content.
3. **RoPE formula duplication:** config_parameters.md lines 37-39 now cross-reference module_hierarchy.md for the full derivation. The rope_theta/rope_type table rows and the Tip about theta scaling are retained as non-duplicative.

## MINOR Suggestions
### [module_hierarchy.md] ~lines 299-305: End-to-end data flow summary partially restates the complete forward pass
**Issue:** The "Complete End-to-End Data Flow Summary" LaTeX equation on line 302 recaps the same pipeline already documented step-by-step in lines 246-277 ("Gemma4VisionModel -- Complete Forward Pass"). The equation adds a compact visual but the two sections are back-to-back.
**Suggestion:** Remove the "Complete End-to-End Data Flow Summary" heading and fold the single-line LaTeX equation into the end of the "Complete Forward Pass" section as a closing summary line, eliminating 5 lines of structural overhead.

### [variable_resolution_processing.md] ~line 3: Gemma 3 comparison still slightly verbose
**Issue:** The opening sentence "Unlike the Gemma 3 SigLIP encoder, which resized all images to a fixed square (224x224 or 896x896)" still restates the Gemma 3 resolution details already provided in index.md line 30.
**Suggestion:** Trim to: "Unlike Gemma 3's fixed-square SigLIP encoder, Gemma 4 preserves each image's native aspect ratio." (This was also flagged as MINOR in Pass 1 and has not yet been applied.)

### [config_parameters.md] ~lines 96-102: Pixel count derivation overlaps with variable_resolution_processing.md examples
**Issue:** The formula block deriving total pixels from token budget ($T \times 2304$) and the "roughly equivalent to a 804x804 image" note restate information that variable_resolution_processing.md covers with concrete examples in its resolution table. (Also flagged as MINOR in Pass 1, not yet applied.)
**Suggestion:** Shorten to a cross-reference: "See [variable_resolution_processing.md](./variable_resolution_processing.md#example-resolutions) for worked resolution examples."

## Load-Bearing Evidence
- **index.md** line 32: "Both height and width must be divisible by 48 (see variable_resolution_processing.md for derivation)" -- load-bearing cross-reference that anchors the constraint to its single canonical explanation
- **module_hierarchy.md** lines 88-92: The RoPE inverse-frequency formula ($f_i = 1/\theta^{2i/d}$ with $\theta=100.0$) -- now the single canonical location for this derivation after Pass 1 de-duplication
- **config_parameters.md** lines 37-39: "See module_hierarchy.md for the full frequency derivation" -- load-bearing cross-reference ensuring the RoPE formula is not orphaned from the parameter table
- **variable_resolution_processing.md** lines 44-46: The no-ImageNet normalization cross-reference to module_hierarchy.md -- load-bearing because it preserves the critical preprocessing fact while avoiding duplication

## VERDICT
- Crucial updates: no
