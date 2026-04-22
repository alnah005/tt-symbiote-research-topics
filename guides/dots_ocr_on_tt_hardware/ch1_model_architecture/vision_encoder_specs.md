# Vision Encoder Specs

This file walks through the `vision_config` block from `config.json` for dots.ocr, derives the output token count formula for a given input image, and calculates the vision encoder parameter count from first principles.

## Full vision_config listing

```json
{
    "vision_config": {
        "hidden_size": 1536,
        "intermediate_size": 4224,
        "num_hidden_layers": 42,
        "num_attention_heads": 12,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "temporal_patch_size": 1,
        "post_norm": true,
        "rms_norm_eps": 1e-05,
        "num_channels": 3,
        "attn_implementation": "flash_attention_2",
        "use_bias": false
    }
}
```

## Dimensions and depth

`hidden_size: 1536` is the width of the vision transformer's residual stream. It is identical to the text decoder's `hidden_size`. This means vision tokens can be fed directly into the text decoder without a cross-modal projection layer.

The patch embedding output dimension equals `hidden_size=1536`. This is a derived conclusion — `embed_dim` is not a separate field in the actual `vision_config`; it follows directly from `hidden_size`. No dimension change occurs between the patch embedding and the first ViT block.

`num_hidden_layers: 42` is the total number of vision transformer encoder blocks. This is substantially deeper than Qwen2.5-VL-7B's 32-layer vision encoder. 42 layers at `hidden_size: 1536` makes this the larger of the two model components by parameter count.

`intermediate_size: 4224` is the inner dimension of each vision MLP. The ratio to hidden size is:

$$\frac{4224}{1536} = 2.75$$

This is consistent with a standard ViT feed-forward ratio of approximately $4\times$ (with $4 \times 1536 = 6144$ being the exact $4\times$ value); the 4224 is a smaller-than-4x ratio, indicating a compact MLP design in the vision encoder.

`num_attention_heads: 12` gives a per-head dimension of:

$$d_{head} = \frac{1536}{12} = 128$$

The vision encoder uses standard full (non-grouped) multi-head attention — there is no `num_key_value_heads` field in the vision_config, and no GQA compression. All 12 heads are both query and key/value heads.

## Patch embedding

`patch_size: 14` divides each spatial dimension of the input image into non-overlapping 14×14 pixel patches. `num_channels: 3` specifies RGB input.

Each patch is a tensor of shape $(14, 14, 3)$, flattened to a vector of length $14 \times 14 \times 3 = 588$ and then linearly projected to `hidden_size: 1536` by the patch embedding layer.

## Temporal patch size and static image design

`temporal_patch_size: 1` means the temporal (video frame) dimension is collapsed to a stride of 1, leaving it unchanged. For static images, there is exactly one "frame," so this setting is a no-op in terms of computation. It signals that dots.ocr was designed exclusively for static document images and does not process video sequences.

Qwen2.5-VL-7B uses `temporal_patch_size: 2`, which merges pairs of frames temporally. This is one of the key structural differences between the two models.

## Spatial merge (PatchMerger)

`spatial_merge_size: 2` controls the PatchMerger module that follows the ViT encoder. The PatchMerger takes a 2×2 grid of adjacent patch tokens and merges them into a single token using a linear projection:

$$\text{tokens\_out} = \text{Linear}(4 \times d_{model}, d_{model})\left(\text{concat}(t_1, t_2, t_3, t_4)\right)$$

This reduces the token count by a factor of $2 \times 2 = 4$ in the spatial dimensions.

## Token count formula

For an input image of height $H$ and width $W$ (in pixels):

1. **Patch grid**: the image is divided into a grid of $\frac{H}{14} \times \frac{W}{14}$ patches (assuming $H$ and $W$ are multiples of 14).
2. **After ViT**: the encoder produces $\frac{H}{14} \times \frac{W}{14}$ tokens.
3. **After spatial merge**: the PatchMerger reduces by $2 \times 2 = 4$, yielding:

$$N_{\text{vision tokens}} = \frac{H}{14 \times 2} \times \frac{W}{14 \times 2} = \frac{H \times W}{784}$$

For a standard A4-sized document image resized to $896 \times 1344$ pixels (a common resolution for document OCR):

$$N = \frac{896 \times 1344}{784} = \frac{1{,}204{,}224}{784} = 1{,}536 \text{ tokens}$$

For a $1120 \times 1120$ square image:

$$N = \frac{1{,}254{,}400}{784} = 1{,}600 \text{ tokens}$$

These token counts are large relative to the text portions of typical input sequences, which is why `max_position_embeddings: 131072` is required.

## Post-norm arrangement

`post_norm: true` means RMSNorm is applied **after** the attention sublayer output and **after** the MLP sublayer output, rather than before them (pre-norm). This is the reverse of the text decoder's arrangement.

The post-norm residual block computes:

$$x \leftarrow \text{RMSNorm}(x + \text{Attention}(x))$$
$$x \leftarrow \text{RMSNorm}(x + \text{MLP}(x))$$

whereas a pre-norm block computes:

$$x \leftarrow x + \text{Attention}(\text{RMSNorm}(x))$$
$$x \leftarrow x + \text{MLP}(\text{RMSNorm}(x))$$

The post-norm arrangement affects the placement of normalization operations in the TTNN kernel graph. Each block has two RMSNorm operations regardless of arrangement, but they appear at different points in the data flow.

`rms_norm_eps: 1e-05` for the vision encoder (versus `1e-06` for the text decoder). This difference must be respected when configuring normalization kernels for the two submodels.

## Attention implementation

`attn_implementation: "flash_attention_2"` specifies FlashAttention-2 as the reference implementation for the vision encoder's attention. This is the HuggingFace field that controls kernel selection during training. On TT hardware, the TTNN attention kernel replaces this; the field's presence indicates that the original training assumed fused attention kernels.

`use_bias: false` means the vision encoder's attention and MLP layers use no bias terms. This is the opposite of the text decoder (`attention_bias: true`). The implication is that all attention projection matrices in the vision encoder have no bias vectors — simplifying weight loading and reducing parameter count slightly.

## Parameter count derivation

### Per vision transformer block

The vision encoder has 12 query/key/value heads, no GQA, and uses `use_bias: false`.

**Attention projections** (Q, K, V, O, no bias):

| Projection | Shape | Parameters |
|---|---|---|
| Q | $1536 \times 1536$ | 2,359,296 |
| K | $1536 \times 1536$ | 2,359,296 |
| V | $1536 \times 1536$ | 2,359,296 |
| O | $1536 \times 1536$ | 2,359,296 |
| **Attention total** | | **9,437,184** |

**MLP** (SwiGLU with 3 matrices: gate_proj, up_proj, down_proj — confirmed from `tt/vision_mlp.py` which implements `y = down_proj(silu(gate_proj(x)) * up_proj(x))`):

| Matrix | Shape | Parameters |
|---|---|---|
| gate_proj | $1536 \times 4224$ | 6,488,064 |
| up_proj | $1536 \times 4224$ | 6,488,064 |
| down_proj | $4224 \times 1536$ | 6,488,064 |
| **MLP total** | | **19,464,192** |

$$3 \times 1536 \times 4224 = 19{,}464{,}192$$

**2x RMSNorm scale vectors** ($2 \times 1536$):

$$3{,}072$$

**Per-block total**:

$$9{,}437{,}184 + 19{,}464{,}192 + 3{,}072 = 28{,}904{,}448 \approx 28.9\text{M}$$

### Across 42 blocks

$$42 \times 28{,}904{,}448 = 1{,}213{,}986{,}816 \approx 1{,}214\text{M}$$

### Non-block vision components

**Patch embedding linear layer** (projects flattened patch to hidden_size):

$$14 \times 14 \times 3 \times 1536 = 588 \times 1536 = 903{,}168$$

**PatchMerger** (merges 4 adjacent tokens into 1 via a linear layer):

$$4 \times 1536 \times 1536 = 9{,}437{,}184$$

**Vision total**:

$$1{,}213{,}986{,}816 + 903{,}168 + 9{,}437{,}184 = 1{,}224{,}327{,}168 \approx 1{,}224\text{M} \approx 1.22\text{B}$$

### Full model parameter breakdown

The text decoder parameter count derived in [`text_decoder_hyperparameters.md`](./text_decoder_hyperparameters.md) is approximately 1,777M (~1.78B). The model card's "1.7B LLM foundation" is a rounded figure referring to this text decoder component alone.

The vision encoder (~1.22B) is a separate component on top of that 1.7B LLM. The full model is:

| Component | Parameters |
|---|---|
| Vision encoder (ViT blocks + patch embed + PatchMerger) | ~1,224M |
| Text decoder transformer blocks (28 layers) | ~1,310M |
| Embedding tables (input embed + lm head) | ~467M |
| RMSNorm final | 1,536 |
| **Total** | **~3,001M ≈ 3.0B** |

The model card's "1.7B" refers only to the text decoder (LLM) portion. Adding the vision encoder (~1.2B) gives the ~2.7–3.0B full model total. The precise parameter count from the checkpoint should be verified with `sum(p.numel() for p in model.parameters())` during the TTNN port work.

---

**Next:** [`relationship_to_qwen25vl.md`](./relationship_to_qwen25vl.md)
