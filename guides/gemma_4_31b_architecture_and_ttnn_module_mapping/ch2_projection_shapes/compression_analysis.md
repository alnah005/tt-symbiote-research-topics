# Change Log

## Agent B Pass 1 Fixes
- Fixed global layer total parameter count on line 109: replaced incorrect value 539,067,352 with correct sum 534,009,856 (sum of Q=88,080,384 + K=11,010,048 + V=0 + O=88,080,384 + Gate=115,605,504 + Up=115,605,504 + Down=115,605,504 + Norms=22,528).
- Fixed full-model formula on line 114: replaced incorrect global layer count 539,067,104 with 534,009,856, and corrected the total from 30,747,047,944 to 30,697,339,904 (the ~30.7B approximation remains unchanged).

# Compression Analysis: Chapter 2 — Projection Shapes — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~724 lines
- Estimated post-compression line count: ~590 lines
- Estimated reduction: ~19%

## CRUCIAL Suggestions

(none)

## MINOR Suggestions

### [index.md] ~lines 39-79
**Issue:** The "Master Shape Table" (Attention Projections, Attention Norm Parameters, FFN Projections, Layer-Level Norms, PLE Projections) duplicates nearly every shape that is derived in detail in `qkv_projections.md`, `ffn_projections.md`, and `ple_shapes.md`. The tables in those subfiles include fuller context (parameter counts, byte sizes, derivation formulas).
**Suggestion:** Replace the five Master Shape Table subsections with a single compact reference table covering only the weight name and shape columns (no Notes column), and add a note directing readers to the subfiles for derivation and details. This would reduce ~40 lines to ~20.

### [qkv_projections.md] ~lines 196-223
**Issue:** The "Decode Activation Shape Summary" tables for sliding and global layers repeat shapes already stated inline under each projection's section (Q reshape at line 60-61, K reshape at 84-86, V at 115, O at 184-185). The per-projection sections are the authoritative source; the summary is a convenience duplicate.
**Suggestion:** Keep the summary tables (they serve as a quick-reference), but remove the inline decode-shape bullets from the Q, K, V, and O sections (lines 60-61, 84-86, 115, 184-185), since the summary tables are more complete. This saves ~12 lines and removes the dual-maintenance burden.

### [ffn_projections.md] ~lines 117-127
**Issue:** The "Uniformity Across Layers" section restates the opening sentence of the file (lines 5-6: "the FFN projections are identical across all 60 layers") with three elaborated bullet points. The same uniformity point also appears in `index.md` line 56 heading.
**Suggestion:** Merge the three bullets into the opening paragraph as a single sentence (e.g., "This uniformity means a single set of TTNN program configs, sharding strategy, and weight-loading code path covers all layers.") and delete the standalone section. Saves ~11 lines.

### [ffn_projections.md] ~lines 3-6
**Issue:** The opening sentence "Unlike the attention projections, which vary between sliding and global layers, the FFN projections are identical across all 60 layers" is a cross-reference contrast that belongs in the index, not in a file whose readers have already navigated to the FFN-specific page.
**Suggestion:** Shorten to "The FFN projections are identical across all 60 layers." The contrast with attention is already established by the chapter index. Saves minor verbosity.

### [index.md] ~lines 83-109
**Issue:** The "Per-Layer Parameter Counts" tables for sliding and global layers duplicate the parameter and byte columns that already appear in the subfiles' weight-shape tables (`qkv_projections.md` line 43 ff., `ffn_projections.md` line 52 ff.). The only unique value-add is the per-layer total and the full-model budget formula (lines 111-119).
**Suggestion:** Remove the per-component rows from both parameter-count tables and keep only the total row for each layer type, plus the full-model formula. Readers can derive component counts from the subfiles. Saves ~25 lines.

## Load-Bearing Evidence
- `index.md` line ~114: "$50 \times 478{,}959{,}104 + 10 \times 534{,}009{,}856 + 262{,}144 \times 5{,}376 = 30{,}697{,}339{,}904 \approx 30.7\text{B params}$" — load-bearing because this is the only place in the chapter that computes the full-model parameter count and validates the "31B" name
- `qkv_projections.md` line ~148: "This saves one `[5376, 2048]` weight matrix per global layer (~22 MB at BF16, ~220 MB total across 10 global layers)." — load-bearing because it quantifies the concrete memory saving from K=V sharing, not repeated elsewhere
- `ffn_projections.md` line ~108: "GELU_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))" — load-bearing because this is the exact activation formula needed for TTNN numerical fidelity
- `ple_shapes.md` line ~102: "per_layer_input = (per_layer_proj + per_layer_embed) * 2^{-0.5}" — load-bearing because this normalization constant is the only place the PLE combination scaling factor is documented

## VERDICT
- Crucial updates: no
