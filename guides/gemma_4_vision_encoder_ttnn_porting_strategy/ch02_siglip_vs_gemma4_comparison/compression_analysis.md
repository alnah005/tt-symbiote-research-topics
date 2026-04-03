# Chapter 2 Change Log

## Fix 1: Tile-alignment padding value in config_diff.md (line 111)

- **File:** `config_diff.md`
- **Error:** Padding needed for Gemma 3's 588-dim input was listed as "4 elements to reach 608".
- **Correction:** Changed to "20 elements to reach 608" (608 - 588 = 20; next multiple of 32 after 588 is 608).

## Fix 2: Gemma 4 patch embedding weight shape in config_diff.md (line 68)

- **File:** `config_diff.md`
- **Error:** Weight tensor shape for Gemma 4 listed as `[768, 1152]`.
- **Correction:** Changed to `[1152, 768]`. PyTorch `nn.Linear(in_features, out_features)` stores weights with shape `[out_features, in_features]`, so `nn.Linear(768, 1152)` produces a `[1152, 768]` weight matrix.

## Fix 3: Longest RoPE wavelength in positional_encoding_shift.md (line 126)

- **File:** `positional_encoding_shift.md`
- **Error:** Longest wavelength stated as approximately 527 positions.
- **Correction:** Changed to approximately 487 positions. The correct calculation of $2\pi \cdot 100^{34/36}$ yields ~487, not ~527.

---

# Compression Analysis — Agent C

## Summary

Chapter 2 comprises four files totaling ~500 lines of content. The writing is generally efficient given the reference-guide nature of the material. However, there is meaningful cross-file redundancy: the same facts (shared config values, patch embedding details, positional encoding descriptions, reuse percentages) are restated in multiple files. Within files, some tables repeat information already stated in prose, and a few explanatory passages belabor points that the target audience (TTNN engineers) would grasp from the table alone.

## CRUCIAL Suggestions

**Crucial updates: no.**

There are no crucial compression changes needed. The redundancy is a deliberate consequence of making each file self-contained as a reference document. Removing it would require readers to flip between files for basic context, which harms usability for a technical guide. The bloat level is low-to-moderate and does not obscure the core content.

## MINOR Suggestions

### MINOR-1: Deduplicate the Gemma 3 vs. Gemma 4 patch embedding comparison table

The exact same Conv2d-vs-Linear comparison appears in three places:
- `config_diff.md` lines 64-71 (Patch Embedding Weights table)
- `module_mapping.md` lines 150-156 (patch embedder comparison table)
- `config_diff.md` lines 27-38 (Changed Parameters table, row for "Patch embedding type")

The `module_mapping.md` table at lines 150-156 could replace its full side-by-side table with a cross-reference: "See [config_diff.md - Patch Embedding Weights](./config_diff.md#patch-embedding-weights) for the detailed shape comparison." This saves ~8 lines.

### MINOR-2: Deduplicate the reuse estimate tables between index.md and module_mapping.md

`index.md` lines 38-43 contains a "Reuse Estimate at a Glance" table with percentage breakdowns. `module_mapping.md` lines 249-254 contains a nearly identical "Reuse Summary" table with the same percentages plus effort estimates. The `index.md` table could be shortened to a single sentence referencing the detailed table: "See [`module_mapping.md` - Reuse Summary](./module_mapping.md#reuse-summary) for per-category effort estimates (~40-50% direct reuse, ~30% modify, ~20% new)."

### MINOR-3: Condense the Gemma 3 positional encoding explanation in positional_encoding_shift.md

Lines 6-36 of `positional_encoding_shift.md` spend 30 lines explaining Gemma 3's simple 1D embedding (table, code sample, three limitations). The target audience has completed Chapter 1 and is assumed familiar with the Gemma 3 codebase (per prerequisites in `index.md`). The code sample (lines 21-27) and the "How It Works" property table (lines 12-17) together convey the same information; one could be cut. Removing the code block saves 8 lines without losing information for the stated audience.

### MINOR-4: Remove low-value "Tip" box in module_mapping.md about MLP reusability

The tip at lines 49 (`module_mapping.md`) stating that MLP accounts for ~70% of per-layer parameters is useful context, but the preceding table already makes the "direct reuse" classification clear. The tip restates what the table shows. Consider removing or shortening to a single line.

### MINOR-5: Trim the "Why Two Mechanisms?" section in positional_encoding_shift.md

Lines 84-95 explain why Gemma 4 uses both learned embeddings and RoPE. The table at lines 88-92 is clear and sufficient. The two paragraphs after the table (lines 93-95) paraphrase what the table already says. Removing the post-table prose saves ~4 lines.

### MINOR-6: Consolidate the three-row RoPE implementation strategy tables

The same three RoPE strategies (CPU precompute, compose from ops, custom kernel) appear in:
- `module_mapping.md` lines 197-201 (numbered list)
- `positional_encoding_shift.md` lines 182-187 (table with effort/performance columns)

The `module_mapping.md` version could be replaced with a cross-reference to the more detailed table in `positional_encoding_shift.md`.

## Load-Bearing Evidence

These are specific content samples from each file that demonstrate the files contain substantive, non-redundant technical content and should be preserved:

1. **index.md, line 59:** "existing TTNN sharding strategies, memory configurations, and matmul decompositions are likely to transfer with adjustments rather than full rewrites" -- this actionable conclusion ties the architectural comparison to concrete engineering decisions.

2. **config_diff.md, lines 106-113:** The TTNN Tile Alignment section showing that Gemma 4's 768 patch dimension is perfectly 32-aligned (768 = 24 * 32) while Gemma 3's 588 is not -- this is a hardware-specific insight not derivable from the architecture alone.

3. **module_mapping.md, lines 259-272:** The dependency-ordered implementation sequence -- this is unique planning content that synthesizes the module analysis into an actionable build order.

4. **positional_encoding_shift.md, lines 111-130:** The `rope_theta=100.0` analysis comparing vision vs. language model theta values with wavelength calculations -- this is specialized content explaining a non-obvious design choice.

## VERDICT

**No crucial changes.** Six minor suggestions that would collectively save ~30-40 lines (~6-8% reduction) primarily by deduplicating cross-file repetitions of patch embedding details, reuse estimates, and RoPE implementation strategies. The redundancy is modest and partially justified by the self-contained reference design of each file.
