# Vision Encoder Specifications

## Architecture

The Qwen3.6-35B-A3B vision encoder is a Vision Transformer (ViT) with the following configuration:

| Parameter | Value |
|-----------|-------|
| `num_hidden_layers` | 27 |
| `hidden_size` | 1152 |
| `patch_size` | 16 |
| `num_attention_heads` | 16 |
| `head_dim` | 72 (= 1152 / 16) |
| `spatial_merge_size` | 2 |
| `temporal_patch_size` | 2 |

Each attention head has dimension $\text{head dim} = 1152 / 16 = 72$.

The total parameter count of the vision encoder is approximately **300M** (27 layers × roughly 11M parameters per ViT layer, plus the projection from vision hidden size to decoder hidden size).

## Image Processing Pipeline

Given an input image of height $H$ and width $W$ pixels, processing proceeds as follows:

1. **Resize and pad.** The image is resized and padded so that both $H$ and $W$ are divisible by the patch size (16).

2. **Patch extraction.** The image is split into non-overlapping $16 \times 16$ pixel patches. This produces

$$N_{\text{patches}} = \left\lceil \frac{H}{16} \right\rceil \times \left\lceil \frac{W}{16} \right\rceil$$

patches.

3. **Linear projection.** Each patch (a flattened $16 \times 16 \times C$ pixel block) is linearly projected to a vector of dimension 1152 (the ViT hidden size).

4. **Position embeddings.** 2-D sinusoidal or learned position embeddings are added to each patch token.

5. **27 ViT layers.** The sequence of $N_{\text{patches}}$ tokens of dimension 1152 passes through 27 transformer layers (multi-head self-attention + MLP). Each attention layer operates with 16 heads of dimension 72.

6. **Spatial 2×2 merge (pooling).** After the final ViT layer, a spatial merge reduces the token count by a factor of 4. Specifically, each non-overlapping $2 \times 2$ group of adjacent patch tokens is pooled (average-pooled) into a single token. The number of vision tokens after this step is

$$N_{\text{vision}} = \frac{\left\lceil H/16 \right\rceil \times \left\lceil W/16 \right\rceil}{4}$$

7. **Projection to decoder space.** A learned linear layer maps each merged vision token from dimension 1152 to dimension 2048 (the decoder hidden size). These $N_{\text{vision}}$ embeddings of dimension 2048 are then injected directly into the text token sequence.

### Token Count Example

For a 448×448 image:

$$N_{\text{patches}} = \frac{448}{16} \times \frac{448}{16} = 28 \times 28 = 784$$

After spatial merge:

$$N_{\text{vision}} = \frac{784}{4} = 196 \text{ vision tokens}$$

For a 224×224 image:

$$N_{\text{vision}} = \frac{14 \times 14}{4} = \frac{196}{4} = 49 \text{ vision tokens}$$

## Video Processing Pipeline

Video adds a temporal dimension on top of the image pipeline:

1. **Frame sampling.** Frames are sampled from the video at a target frame rate. Let $T$ be the number of sampled frames.

2. **Per-frame encoding.** Each frame is processed independently through steps 1–6 of the image pipeline above, producing $N_{\text{vision}}$ tokens per frame after spatial merge.

3. **Temporal merge.** The `temporal_patch_size=2` parameter groups consecutive pairs of frames together, halving the temporal token count:

$$N_{\text{temporal}} = \frac{T}{2} \times N_{\text{vision per frame}}$$

4. **Projection.** The same learned linear (1152 → 2048) is applied to all temporal tokens, and the resulting embeddings are injected into the text sequence.

### Shape Summary

At each stage of the image pipeline, the tensor shape is:

```
Input image:          [H, W, C]
After patch split:    [N_patches, 16*16*C]
After projection:     [N_patches, 1152]
After 27 ViT layers:  [N_patches, 1152]
After spatial merge:  [N_patches/4, 1152]
After linear proj:    [N_patches/4, 2048]   → injected into text sequence
```

For video with $T$ frames:

```
After per-frame ViT + spatial merge:  [T, N_patches/4, 1152]
After temporal merge (size=2):        [T/2, N_patches/4, 1152]
After linear proj:                    [T/2 * N_patches/4, 2048]  → injected into text sequence
```

---

**Next:** [Vision Encoder Comparison](./vision_encoder_comparison.md)
