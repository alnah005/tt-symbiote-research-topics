# Configuration Parameters

This file documents every parameter in `Gemma4VisionConfig` for the 31B model, sourced from the model's `config.json` on HuggingFace and the HuggingFace Transformers `configuration_gemma4.py` source.

## Full Gemma4VisionConfig Parameter Table

| Parameter | Value (31B) | Description |
|-----------|-------------|-------------|
| `hidden_size` | 1152 | Hidden dimension throughout the vision encoder |
| `num_hidden_layers` | 27 | Number of transformer encoder layers |
| `num_attention_heads` | 16 | Number of query attention heads |
| `num_key_value_heads` | 16 | Number of key/value heads (MHA, not GQA) |
| `head_dim` | 72 | Dimension per attention head |
| `intermediate_size` | 4304 | MLP intermediate dimension (gate and up projections) |
| `patch_size` | 16 | Side length of each image patch in pixels |
| `pooling_kernel_size` | 3 | Spatial pooling kernel size (3x3 grid averaging) |
| `position_embedding_size` | 10240 | Maximum distinct positions per spatial axis |
| `hidden_activation` | `"gelu_pytorch_tanh"` | Activation function in the MLP (tanh-approximated GeLU) |
| `rms_norm_eps` | 1e-6 | Epsilon for all RMSNorm layers |
| `max_position_embeddings` | 131072 | Maximum sequence length for position embeddings |
| `attention_bias` | `False` | Whether Q/K/V/O projections use bias terms |
| `attention_dropout` | 0.0 | Dropout rate in attention (disabled) |
| `default_output_length` | 280 | Default number of soft tokens per image |
| `dtype` | `"bfloat16"` | Default computation dtype |
| `standardize` | `True` | Whether to apply output standardization (zero-mean, unit-variance) |
| `use_clipped_linears` | `False` | Whether to clamp linear layer inputs/outputs |

## RoPE Parameters

The vision encoder's RoPE configuration is nested under `rope_parameters`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `rope_theta` | 100.0 | Base frequency for rotary embeddings |
| `rope_type` | `"default"` | RoPE variant (standard inverse-frequency computation) |

### RoPE Frequency Derivation

See [`module_hierarchy.md`](./module_hierarchy.md) for the full frequency derivation.

> **Tip:** The vision encoder's `rope_theta=100.0` is much smaller than the language model's base frequency (typically 10000.0 or higher). This reflects the fact that spatial positions in images span a much smaller range than sequence positions in text, so lower-frequency rotations are not needed.

## Derived Dimensions

Several important dimensions are computed from the config rather than stored directly:

| Derived Quantity | Formula | Value |
|-----------------|---------|-------|
| Patch input dim | `3 * patch_size^2` | 768 |
| Total Q/K/V width | `num_attention_heads * head_dim` | 1152 |
| MLP expansion ratio | `intermediate_size / hidden_size` | 3.74 |
| Spatial dim per axis (RoPE) | `head_dim / 2` | 36 |
| Frequency pairs per axis | `spatial_dim / 2` | 18 |
| Minimum image divisor | `patch_size * pooling_kernel_size` | 48 |

Note that `num_attention_heads * head_dim = 16 * 72 = 1152 = hidden_size`. This means the vision encoder uses standard multi-head attention (MHA), not grouped-query attention (GQA). Every head has its own unique K and V projections.

## Parameter Count Estimate

| Component | Parameters | Calculation |
|-----------|-----------|-------------|
| `input_proj` | 884,736 | 768 * 1152 |
| `position_embedding_table` | 23,592,960 | 2 * 10240 * 1152 |
| Per-layer attention (Q+K+V+O) | 5,308,416 | 4 * 1152 * 1152 |
| Per-layer Q/K norms (learnable) | 144 | 2 * 72 |
| Per-layer MLP (gate+up+down) | 14,874,624 | 2 * 1152 * 4304 + 4304 * 1152 |
| Per-layer RMSNorm (4 norms) | 4,608 | 4 * 1152 |
| **Per layer total** | **20,187,792** | |
| **27 layers total** | **545,070,384** | |
| Pooler | 0 | No learnable parameters |
| Standardization buffers | 2,304 | 2 * 1152 (not trainable) |
| **Vision encoder total** | **~569M** | |
| Multimodal projection | 6,193,152 | 1152 * 5376 |
| **Full vision pipeline total** | **~575M** | |

> **Tip:** The position embedding table accounts for ~24M parameters (about 4% of the encoder). This is a substantial allocation that enables the encoder to handle images up to 10240 patches along either axis.

## Token Budget Configuration

The `default_output_length` parameter controls how many soft tokens each image produces. Five standard token budgets are supported:

| Token Budget | Patches Before Pooling | Approximate Pixel Count | Use Case |
|-------------|----------------------|------------------------|----------|
| 70 | 630 | ~161K pixels | Fast inference, low-detail tasks |
| 140 | 1260 | ~323K pixels | Balanced speed/quality |
| **280** (default) | **2520** | **~645K pixels** | General-purpose default |
| 560 | 5040 | ~1.29M pixels | High-detail tasks |
| 1120 | 10080 | ~2.58M pixels | Maximum detail, OCR, fine-grained understanding |

The relationship between token budget $T$, pooling kernel size $k$, and number of pre-pooling patches $N$ is:

$$
N = T \times k^2 = T \times 9
$$

And since each patch covers `16 * 16 = 256` pixels:

$$
\text{total pixels} = N \times 256 = T \times 9 \times 256 = T \times 2304
$$

For the default budget of 280: $280 \times 2304 = 645{,}120$ pixels, which is roughly equivalent to a 804x804 image.

### Setting the Token Budget at Inference

The token budget is set at the image processor level. When calling the model, the processor resizes images so that the total patch count divided by $k^2$ equals the desired output length. The `default_output_length=280` is used when no explicit budget is provided.

> **Warning:** Changing the token budget changes the number of patches fed to the encoder, which in turn changes the attention sequence length. For TTNN, this means the attention tile sizes and memory allocation must accommodate the range of possible sequence lengths.

---

**Next:** [`variable_resolution_processing.md`](./variable_resolution_processing.md)
