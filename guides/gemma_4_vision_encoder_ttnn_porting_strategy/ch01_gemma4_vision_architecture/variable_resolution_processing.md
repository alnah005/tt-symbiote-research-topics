# Variable-Resolution Image Processing

Gemma 4's vision encoder is designed around variable-resolution input. Unlike the Gemma 3 SigLIP encoder, which resized all images to a fixed square (224x224 or 896x896), Gemma 4 preserves the original aspect ratio of each image. This section explains how the variable-resolution pipeline works and what it means for TTNN porting.

## Aspect-Ratio Preservation

The Gemma 4 image processor resizes each image so that:

1. The original aspect ratio is preserved (no squashing or stretching).
2. The total number of pixels fits within the pixel budget implied by the token budget.
3. Both the height and width of the resized image are divisible by 48.

This means a 1920x1080 landscape photo and a 1080x1920 portrait photo produce different patch grid shapes, even when targeting the same token budget.

### The Divisibility-by-48 Constraint

The number 48 arises from the interaction of two config parameters:

$$
\text{min divisor} = \text{patch\_size} \times \text{pooling\_kernel\_size} = 16 \times 3 = 48
$$

**Why both factors matter:**

- **patch\_size = 16**: the image is divided into non-overlapping 16x16 pixel patches. The height and width must be divisible by 16 for this tiling to work without remainder.
- **pooling\_kernel\_size = 3**: the pooler groups patches into 3x3 spatial blocks for average pooling. The number of patches along each axis must therefore be divisible by 3.

Combining both requirements: pixels along each axis must be divisible by $16 \times 3 = 48$.

### Example Resolutions

For a token budget of 280 (2520 pre-pooling patches, 645K pixels):

| Original Image | Resized To | Patches (H x W) | Total Patches | Pooled Tokens |
|---------------|-----------|-----------------|---------------|---------------|
| 1920 x 1080 (16:9) | 912 x 528 | 57 x 33 | 1881 | 209 |
| 1080 x 1080 (1:1) | 816 x 816 | 51 x 51 | 2601 | 289 |
| 1080 x 1920 (9:16) | 528 x 912 | 33 x 57 | 1881 | 209 |
| 4032 x 3024 (4:3) | 864 x 624 | 54 x 39 | 2106 | 234 |
| 640 x 480 (4:3) | 864 x 624 | 54 x 39 | 2106 | 234 |

> **Warning:** The actual number of pooled tokens varies per image depending on its aspect ratio. The `default_output_length=280` is a target budget, not a guaranteed exact count. The processor chooses dimensions that get close to but do not exceed the pixel budget, while satisfying the divisibility constraint.

## No ImageNet Normalization

The patch embedder rescales pixels from [0,1] to [-1,1] internally (see [`module_hierarchy.md`](./module_hierarchy.md)) — no ImageNet mean/std normalization is applied.

This simplification means the image processor only needs to:
1. Resize to a valid resolution (divisible by 48, within pixel budget)
2. Convert pixel values to `[0, 1]` float range
3. Extract non-overlapping 16x16 patches and flatten to `[batch, num_patches, 768]`

## Position ID Construction

After patching, each patch receives a 2D position ID `(x, y)` based on its grid location:

- For an image resized to $H \times W$ pixels, the patch grid has shape $(H/16) \times (W/16)$.
- The patch at row $r$, column $c$ receives position ID $(c, r)$ — note the (x, y) convention where x is the column index.
- Padding patches (used to fill batches to equal length) receive the sentinel value $(-1, -1)$.

The position IDs are passed alongside pixel values as a `[batch, max_num_patches, 2]` integer tensor.

### Position ID Range

With `position_embedding_size=10240`, valid position IDs range from 0 to 10239 per axis. The maximum supported image dimension is therefore:

$$
\text{max pixels per axis} = 10240 \times 16 = 163{,}840 \text{ pixels}
$$

In practice, the pixel budget constrains images to much smaller sizes than this theoretical maximum.

## Implications for TTNN Porting

Variable-resolution input creates several challenges for TTNN implementation on Tenstorrent hardware.

### Challenge 1: Variable Sequence Lengths Across a Batch

Different images in the same batch produce different numbers of patches. The reference implementation handles this by padding to the maximum patch count and using a boolean mask (`padding_positions`) throughout the pipeline.

For TTNN, the options are:

- **Pad to maximum and mask**: mirrors the reference implementation. Wastes compute on padding tokens but keeps batch processing uniform. Attention masking must correctly exclude padding positions.
- **Fixed token budget with padding**: resize all images to produce exactly the same number of patches (by adjusting the pixel budget enforcement). This loses some aspect-ratio fidelity but simplifies the implementation.
- **Per-image processing**: process each image independently without batching. Simple but loses batch parallelism.

> **Tip:** For an initial TTNN port, padding to the maximum patch count within a batch is the most straightforward approach. The attention mask infrastructure already exists in the reference code and can be mapped to TTNN's masking support.

### Challenge 2: Variable Attention Sequence Lengths

The vision encoder's self-attention operates over the full patch sequence. For the default 280-token budget, the pre-pooling patch count is approximately 2520, but the exact number varies. This means:

- Attention matrices have shape `[batch, 16, seq_len, seq_len]` where `seq_len` varies.
- The memory footprint for attention scales quadratically with `seq_len`.
- TTNN tile sizes and memory allocation must handle a range of sequence lengths or be configured per-batch.

### Challenge 3: Dynamic Pooling Grid

The pooler computes grid assignments dynamically based on position IDs and the target output length. The grid-cell assignment depends on the actual spatial layout of patches, which varies per image. This makes the pooler harder to express as a static TTNN operation graph.

### Challenge 4: Ragged Output After Padding Removal

After pooling, the reference implementation removes padding tokens via boolean indexing (`hidden_states[mask]`), producing a flat tensor of shape `[total_valid_tokens, 1152]`. This ragged output is then inserted into the language model's token sequence.

For TTNN, this gather-and-scatter pattern may need to be replaced with a padded representation that carries a validity mask through the multimodal embedder projection.

### Recommended Strategy

| Aspect | Recommendation |
|--------|---------------|
| Batch padding | Pad all images in a batch to the same patch count |
| Attention | Use padded bidirectional attention with masking |
| Pooling | Implement as a custom kernel or decompose into gather + reduce |
| Output | Keep padded representation through projection; remove padding at the language model integration point |
| Token budget | Start with a fixed budget (e.g., 280) for initial bring-up |

---

**Next:** [Chapter 2 — Gemma 3 SigLIP vs. Gemma 4 Vision Encoder Comparison](../ch02_siglip_vs_gemma4_comparison/index.md)
