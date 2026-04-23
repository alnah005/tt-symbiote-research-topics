# Compression Analysis -- Chapter 3: Custom Layers and Ops

## Summary

These four files total roughly 1,900 lines. The content is well-structured and technically precise. However, the index.md file duplicates substantial portions of its sub-files (normalization summaries, experimental ops summaries, conv layer summaries), and the sub-files themselves repeat "no TT-Symbiote equivalent" verdicts and porting requirements multiple times. The Key Takeaways sections across files restate points already established in the comparison tables and porting notes directly above them. Estimated compressible content: 10-15%.

---

## CRUCIAL Suggestions

None.

---

## MINOR Suggestions

### 1. index.md duplicates sub-file content extensively

The "Layer Classification Summary" section (lines 33-81) reproduces detailed information that is the primary content of the three sub-files. For example:

- **Normalization (lines 33-44)**: Describes the two-phase pattern for DistributedRMSNorm and DistributedLayerNorm, names all experimental ops, and lists all TT-Symbiote equivalents. This is the core content of normalization_layers.md.
- **Convolution (lines 65-70)**: Describes Conv2d TP support, Conv3d context parallelism, and TTNNConv2dNHWC -- all repeated in convolution_layers.md.
- **Embeddings (lines 72-81)**: A full taxonomy with four sub-categories; this section has no sub-file, so it belongs here, but the others could be trimmed to one-line forward references.

**Suggestion**: Reduce the Normalization, Linear, Feedforward, and Convolution subsections in index.md to 1-2 sentences each plus the "See [sub-file]" link. The detailed descriptions belong in the sub-files where they already exist.

### 2. "No TT-Symbiote Conv3d" stated five times across files

The absence of Conv3d in TT-Symbiote is stated in:
- index.md line 70: "There is no TT-Symbiote Conv3d."
- index.md Key Takeaway #4 (line 91): "TT-Symbiote has no Conv3d support"
- ttnn_experimental_ops.md line 231: "TT-Symbiote has no Conv3d support"
- convolution_layers.md line 18: "There is no TT-Symbiote Conv3d."
- convolution_layers.md line 360: "There is no Conv3d in TT-Symbiote."
- convolution_layers.md Key Takeaway #2 (line 437): "There is no TT-Symbiote Conv3d module."

**Suggestion**: State once per file at most. In convolution_layers.md, the section header "TT-Symbiote Conv3d" (line 358) plus one statement suffices -- remove the restatements in the overview (line 18) and Key Takeaway #2 body (line 437) can reference rather than restate.

### 3. normalization_layers.md Key Takeaways restate comparison tables

All five Key Takeaways (lines 342-351) recapitulate information from the comparison tables and porting notes already present in the same file:
- Takeaway #1 ("Single-device norms are directly portable") restates the RMSNorm and LayerNorm comparison tables.
- Takeaway #2 ("Distributed norms use different TTNN APIs") restates the DistributedRMSNorm comparison table row for "API namespace" and "Pre-gather op."
- Takeaway #4 ("Compute precision differs") restates the LayerNorm comparison table row "Compute kernel config."
- Takeaway #5 ("Weight preparation patterns diverge") restates information from every comparison section's "Weight management" row.

**Suggestion**: Condense the five takeaways to three, focusing only on cross-cutting insights not obvious from the per-layer sections. Remove #1 (obvious from the tables) and #5 (a framework-level pattern already explained in Ch1 prerequisites).

### 4. ttnn_experimental_ops.md repeats normalization_layers.md descriptions

The four normalization experimental ops (sections 2.1-2.4, lines 92-188) restate information already covered in normalization_layers.md:
- The wan_fused_rmsnorm_pre/post_allgather code snippets, parameter lists, TT-Symbiote equivalents, and "Difference" paragraphs largely duplicate the DistributedRMSNorm section of normalization_layers.md.
- The dit_layernorm_pre/post_allgather entries duplicate the DistributedLayerNorm section.

**Suggestion**: In the experimental ops catalog, reduce the four norm entries to brief cross-references: state the op name, one-line purpose, and "See normalization_layers.md > DistributedRMSNorm for full comparison." Keep the parameter signatures (unique to this file) but remove the "Difference" paragraphs that duplicate the other file.

### 5. convolution_layers.md fused variant descriptions are partially redundant

The fused Conv+BN variants (TTNNConv2dBNNHWC, TTNNConv2dBNActivationNHWC) are described twice:
- Lines 199-208 in the TTNNConv2dNHWC section's "Fused Variants" subsection.
- Lines 377-389 in the "Additional Convolution Modules in TT-Symbiote" section with expanded detail.

Both sections also note these are "irrelevant for diffusion models" (line 208 and line 385/441).

**Suggestion**: Remove the "Fused Variants" subsection (lines 197-208) from the TTNNConv2dNHWC section. The later "Additional Convolution Modules" section covers them more thoroughly.

### 6. Hedging language and over-qualification

Several passages use unnecessary hedging:
- normalization_layers.md line 81: "The weight shape difference (`[1, dim]` vs `[32, dim]`) stems from different broadcasting strategies." -- The explanation that follows ("TT-Symbiote expands the weight to 32 rows to match tile dimensions; TT-DiT relies on TTNN's internal broadcasting") is the same point restated.
- convolution_layers.md line 233: "The builder abstraction adds indirection" paragraph explains the indirection, then repeats "may obscure performance tuning opportunities that TT-DiT's direct `ttnn.conv2d` call exposes" which is the same point as the opening sentence.

**Suggestion**: Trim these to single statements without the restatement.

---

## Load-Bearing Evidence

- **index.md** line 38: `"DistributedRMSNorm uses a two-phase pattern: ttnn.experimental.wan_fused_rmsnorm_pre_allgather to compute local statistics, then all-gather, then ttnn.experimental.wan_fused_rmsnorm_post_allgather to apply the norm."` -- This sentence is the core content of normalization_layers.md lines 156-184 compressed into one line, demonstrating the duplication between index and sub-file.
- **normalization_layers.md** line 343: `"Single-device norms are directly portable: RMSNorm and LayerNorm both map to ttnn.rms_norm / ttnn.layer_norm in both frameworks."` -- Restates the conclusion already visible in the Summary Table (line 332) where both RMSNorm and LayerNorm show "Compatible."
- **ttnn_experimental_ops.md** line 115: `"The wan_fused_* variant computes statistics in float32 by default; the stable version in TT-Symbiote uses bfloat16."` -- Same point made in normalization_layers.md line 215 (comparison table: "Statistics dtype: float32 vs bfloat16") and line 226 ("TT-DiT computes statistics in float32; TT-Symbiote uses bfloat16").
- **convolution_layers.md** line 360: `"There is no Conv3d in TT-Symbiote."` -- Same statement as line 18 ("There is no TT-Symbiote Conv3d"), index.md line 70, and ttnn_experimental_ops.md line 231.

---

## VERDICT

**Crucial updates: no**

The redundancy is real but moderate (10-15% of total content). It consists of cross-file duplication between the index and sub-files, within-file restatement of comparison table findings in Key Takeaways, and repeated "no equivalent" declarations. The MINOR suggestions would tighten the chapter without losing any information.
