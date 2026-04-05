# Chapter 2: Configuration Hierarchy

This chapter documents the four configuration classes that parameterize every component of the Gemma 4 multimodal model. All defaults cited below are taken directly from [`configuration_gemma4.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/configuration_gemma4.py). Understanding these configs is essential before reading any modeling code, because every architectural decision -- number of layers, head dimensions, RoPE frequencies, MoE routing -- is driven by values set here.

## Configuration Class Map

```
Gemma4Config  (top-level, model_type="gemma4")
  |
  +-- text_config   --> Gemma4TextConfig   (model_type="gemma4_text")
  +-- vision_config --> Gemma4VisionConfig  (model_type="gemma4_vision")
  +-- audio_config  --> Gemma4AudioConfig   (model_type="gemma4_audio")
```

All four classes inherit from `PreTrainedConfig` and use the `@strict` decorator from `huggingface_hub.dataclasses`, which rejects unknown keyword arguments at construction time.

---

## 2.1 Gemma4Config (Top-Level Multimodal Config)

`Gemma4Config` is the entry point. It holds special token IDs for modality boundaries and composes three sub-configs.

### Sub-Config Registry

```python
sub_configs = {
    "text_config": Gemma4TextConfig,
    "vision_config": Gemma4VisionConfig,
    "audio_config": Gemma4AudioConfig,
}
```

This `sub_configs` dict tells the HuggingFace serialization machinery which class to instantiate when loading each sub-config from a JSON dict.

### Special Token IDs

| Parameter | Default | Purpose |
|---|---|---|
| `Gemma4Config.boi_token_id` | `255999` | Begin-of-image sentinel |
| `Gemma4Config.eoi_token_id` | `258882` | End-of-image sentinel |
| `Gemma4Config.image_token_id` | `258880` | Placeholder token for image soft-token slots |
| `Gemma4Config.video_token_id` | `258884` | Placeholder token for video soft-token slots |
| `Gemma4Config.boa_token_id` | `256000` | Begin-of-audio sentinel |
| `Gemma4Config.eoa_token_index` | `258883` | End-of-audio sentinel |
| `Gemma4Config.audio_token_id` | `258881` | Placeholder token for audio soft-token slots |

Note that the field name `Gemma4Config.eoa_token_index` is asymmetric with `Gemma4Config.eoi_token_id` -- this is how it appears in the source, not a typo.

### Other Top-Level Parameters

| Parameter | Default | Notes |
|---|---|---|
| `Gemma4Config.initializer_range` | `0.02` | Std-dev for weight initialization |
| `Gemma4Config.tie_word_embeddings` | `True` | LM head shares weights with input embeddings |

### `__post_init__` Behavior

The `__post_init__` method handles three cases for each sub-config (`text_config`, `vision_config`, `audio_config`):

1. **`None`** -- For `text_config`, a default `Gemma4TextConfig()` is created automatically. For `vision_config` and `audio_config`, the value stays `None` and the corresponding tower will not be initialized in the model.
2. **`dict`** -- Instantiated into the corresponding config class (e.g., `Gemma4TextConfig(**self.text_config)`).
3. **Already a config object** -- Used as-is.

This means `text_config` is mandatory (always created), while the vision and audio towers are optional.

---

## 2.2 Gemma4TextConfig

`Gemma4TextConfig` is the largest and most complex config. It controls the 30-layer decoder, including sliding/global attention patterns, dual RoPE parameterizations, per-layer input embeddings, and MoE blocks.

### Core Architecture Parameters

| Parameter | Default | Description |
|---|---|---|
| `Gemma4TextConfig.vocab_size` | `262144` | Vocabulary size (256K tokens) |
| `Gemma4TextConfig.hidden_size` | `2304` | Model hidden dimension |
| `Gemma4TextConfig.intermediate_size` | `9216` | MLP intermediate dimension (4x hidden) |
| `Gemma4TextConfig.num_hidden_layers` | `30` | Total decoder layers |
| `Gemma4TextConfig.num_attention_heads` | `8` | Query heads per layer |
| `Gemma4TextConfig.num_key_value_heads` | `4` | KV heads per layer (GQA ratio 2:1) |
| `Gemma4TextConfig.head_dim` | `256` | Per-head dimension for sliding attention layers |
| `Gemma4TextConfig.hidden_activation` | `"gelu_pytorch_tanh"` | Activation function |
| `Gemma4TextConfig.max_position_embeddings` | `131072` | Maximum sequence length (128K) |
| `Gemma4TextConfig.rms_norm_eps` | `1e-6` | RMSNorm epsilon |
| `Gemma4TextConfig.attention_bias` | `False` | No bias in attention projections |
| `Gemma4TextConfig.attention_dropout` | `0.0` | No attention dropout |

### Embedding and Token Parameters

| Parameter | Default | Description |
|---|---|---|
| `Gemma4TextConfig.pad_token_id` | `0` | Padding token |
| `Gemma4TextConfig.eos_token_id` | `1` | End-of-sequence token |
| `Gemma4TextConfig.bos_token_id` | `2` | Beginning-of-sequence token |
| `Gemma4TextConfig.tie_word_embeddings` | `True` | Share input/output embeddings |
| `Gemma4TextConfig.use_cache` | `True` | Enable KV cache by default |

### Sliding Window and Layer Types

`Gemma4TextConfig.sliding_window` defaults to `512`. The `Gemma4TextConfig.layer_types` list controls whether each layer uses sliding or global (full) attention. When `Gemma4TextConfig.layer_types` is `None` (the default), `__post_init__` generates the list automatically using a **5:1 sliding-to-global pattern**:

```python
sliding_window_pattern = 6  # by default 5:1
self.layer_types = [
    "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
    for i in range(self.num_hidden_layers)
]
```

For 30 layers, this produces:

| Layer Index | 0 | 1 | 2 | 3 | 4 | **5** | 6 | 7 | 8 | 9 | 10 | **11** | 12 | 13 | 14 | 15 | 16 | **17** | 18 | 19 | 20 | 21 | 22 | **23** | 24 | 25 | 26 | 27 | 28 | **29** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Type | S | S | S | S | S | **G** | S | S | S | S | S | **G** | S | S | S | S | S | **G** | S | S | S | S | S | **G** | S | S | S | S | S | **G** |

(S = sliding attention layer, G = global attention layer)

There is an additional enforcement: the **last layer must be `full_attention`**. If the generated or user-provided pattern has a different type for the last layer, `__post_init__` forces it to `"full_attention"` with a warning.

When `Gemma4TextConfig.use_bidirectional_attention` is set to `"all"`, the sliding window size is modified: `sliding_window = (sliding_window // 2) + 1`, giving `257` from the default `512`. This adjustment accounts for FlashAttention's exclusive-bound semantics in bidirectional mode.

### Global Attention Layer Parameters

Global attention layers use different head dimensions and optionally different KV head counts:

| Parameter | Default | Description |
|---|---|---|
| `Gemma4TextConfig.global_head_dim` | `512` | Head dimension for global attention layers (vs `256` for sliding) |
| `Gemma4TextConfig.num_global_key_value_heads` | `None` | KV heads for global layers; if `None`, falls back to `Gemma4TextConfig.num_key_value_heads` (`4`) |

This means global attention layers have **2x the head dimension** of sliding attention layers, increasing their capacity for long-range information.

### Dual RoPE Parameters

Gemma 4 uses **different RoPE configurations for sliding vs. global attention layers**. When `Gemma4TextConfig.rope_parameters` is `None`, `__post_init__` sets the following defaults:

```python
{
    "sliding_attention": {
        "rope_type": "default",
        "rope_theta": 10_000.0
    },
    "full_attention": {
        "rope_type": "proportional",
        "partial_rotary_factor": 0.25,
        "rope_theta": 1_000_000.0
    }
}
```

Key differences:

| Property | Sliding Attention Layers | Global Attention Layers |
|---|---|---|
| RoPE type | `"default"` (standard sinusoidal) | `"proportional"` (NTK-aware scaling) |
| Theta | `10,000` | `1,000,000` |
| Partial rotary factor | `1.0` (implicit, full rotation) | `0.25` (only 25% of head_dim gets RoPE) |

The global layers apply RoPE to only the first quarter of each head's dimensions (`0.25 * 512 = 128` dimensions), leaving the remaining 384 dimensions as position-independent. This is a deliberate design to preserve semantic content in most of the head while still encoding position.

### Per-Layer Input Embeddings

| Parameter | Default | Description |
|---|---|---|
| `Gemma4TextConfig.vocab_size_per_layer_input` | `262144` | Vocabulary size for per-layer embeddings |
| `Gemma4TextConfig.hidden_size_per_layer_input` | `256` | Hidden dimension for per-layer embeddings |

These parameters support a per-layer residual stream architecture where a small embedding (`[262144, 256]`) is looked up and added at each decoder layer.

### Mixture-of-Experts (MoE) Parameters

| Parameter | Default | Description |
|---|---|---|
| `Gemma4TextConfig.enable_moe_block` | `False` | Whether MoE is active |
| `Gemma4TextConfig.num_experts` | `None` | Total expert count |
| `Gemma4TextConfig.top_k_experts` | `None` | Experts activated per token |
| `Gemma4TextConfig.moe_intermediate_size` | `None` | FFN intermediate size per expert |
| `Gemma4TextConfig.use_double_wide_mlp` | `False` | Whether to fuse gate + up projections into a double-width linear |

All MoE parameters default to disabled/`None` in the base config. The actual values come from the model checkpoint's `config.json` (e.g., `google/gemma-4-e2b-it` enables MoE).

### Key-Value Sharing

| Parameter | Default | Description |
|---|---|---|
| `Gemma4TextConfig.attention_k_eq_v` | `False` | Whether K and V share projection weights |
| `Gemma4TextConfig.num_kv_shared_layers` | `0` | Number of consecutive layers sharing KV projections (0 = no sharing) |

### Tensor Parallelism Plan

`Gemma4TextConfig` defines `base_model_tp_plan` for distributed training/inference:

| Weight Pattern | Strategy |
|---|---|
| `layers.*.self_attn.q_proj` | `colwise` |
| `layers.*.self_attn.k_proj` | `colwise` |
| `layers.*.self_attn.v_proj` | `colwise` |
| `layers.*.self_attn.q_norm` | `replicated_with_grad_allreduce` |
| `layers.*.self_attn.k_norm` | `replicated_with_grad_allreduce` |
| `layers.*.self_attn.o_proj` | `rowwise` |
| `layers.*.mlp.gate_proj` | `colwise` |
| `layers.*.mlp.up_proj` | `colwise` |
| `layers.*.mlp.down_proj` | `rowwise` |

---

## 2.3 Gemma4VisionConfig

`Gemma4VisionConfig` controls the SigLIP-based vision encoder (a ViT variant).

### Core Parameters

| Parameter | Default | Description |
|---|---|---|
| `Gemma4VisionConfig.hidden_size` | `768` | Encoder hidden dimension |
| `Gemma4VisionConfig.intermediate_size` | `3072` | MLP intermediate dimension (4x hidden) |
| `Gemma4VisionConfig.num_hidden_layers` | `16` | Transformer encoder layers |
| `Gemma4VisionConfig.num_attention_heads` | `12` | Self-attention heads |
| `Gemma4VisionConfig.num_key_value_heads` | `12` | KV heads (MHA, not GQA) |
| `Gemma4VisionConfig.head_dim` | `64` | Per-head dimension |
| `Gemma4VisionConfig.hidden_activation` | `"gelu_pytorch_tanh"` | Activation function |
| `Gemma4VisionConfig.rms_norm_eps` | `1e-6` | RMSNorm epsilon |
| `Gemma4VisionConfig.max_position_embeddings` | `131072` | Maximum position embeddings |
| `Gemma4VisionConfig.attention_bias` | `False` | No bias in attention projections |
| `Gemma4VisionConfig.attention_dropout` | `0.0` | No attention dropout |
| `Gemma4VisionConfig.initializer_range` | `0.02` | Weight init std-dev |

### Vision-Specific Parameters

| Parameter | Default | Description |
|---|---|---|
| `Gemma4VisionConfig.patch_size` | `16` | Pixels per patch side (16x16 patches) |
| `Gemma4VisionConfig.pooling_kernel_size` | `3` | Spatial pooling kernel after patchification |
| `Gemma4VisionConfig.position_embedding_size` | `10240` | Size of learned 2D position embedding table (`10 * 1024`) |
| `Gemma4VisionConfig.use_clipped_linears` | `False` | Whether to clamp linear layer weights |
| `Gemma4VisionConfig.standardize` | `False` | Whether to apply bias and scale to soft tokens from pooler |

### 2D RoPE

When `Gemma4VisionConfig.rope_parameters` is `None`, `__post_init__` sets:

```python
{"rope_type": "default", "rope_theta": 100.0}
```

The vision encoder uses a theta of `100.0` (also stored as the class attribute `Gemma4VisionConfig.default_theta = 100.0`), which is far smaller than the text model's theta values. This is appropriate because the vision encoder operates over a 2D grid of patches with much shorter effective sequence lengths than text.

### Vision Tensor Parallelism Plan

`Gemma4VisionConfig` defines its own `base_model_tp_plan` with the same colwise/rowwise strategy as the text model, but scoped under `encoder.layers.*`.

---

## 2.4 Gemma4AudioConfig

`Gemma4AudioConfig` controls the Conformer-based audio encoder.

### Core Parameters

| Parameter | Default | Description |
|---|---|---|
| `Gemma4AudioConfig.hidden_size` | `1024` | Conformer hidden dimension |
| `Gemma4AudioConfig.num_hidden_layers` | `12` | Conformer blocks |
| `Gemma4AudioConfig.num_attention_heads` | `8` | Self-attention heads |
| `Gemma4AudioConfig.hidden_act` | `"silu"` | Activation function (note: `hidden_act`, not `hidden_activation`) |
| `Gemma4AudioConfig.rms_norm_eps` | `1e-6` | RMSNorm epsilon |
| `Gemma4AudioConfig.initializer_range` | `0.02` | Weight init std-dev (validated: must be in `[0.0, 1.0]`) |

### Subsampling Parameters

| Parameter | Default | Description |
|---|---|---|
| `Gemma4AudioConfig.subsampling_conv_channels` | `[128, 32]` | Channel sizes for the two convolutional layers in the sub-sample convolution projection |

Note: The default is defined as a tuple `(128, 32)` but `__post_init__` converts tuples to lists for JSON serialization compatibility.

### Conformer Parameters

| Parameter | Default | Description |
|---|---|---|
| `Gemma4AudioConfig.conv_kernel_size` | `5` | Depthwise convolution kernel in Conformer blocks |
| `Gemma4AudioConfig.residual_weight` | `0.5` | Scale factor for hidden states before adding residual in feedforward |
| `Gemma4AudioConfig.attention_chunk_size` | `12` | Sub-sequence size for chunked attention |
| `Gemma4AudioConfig.attention_context_left` | `13` | Left context window for attention chunks |
| `Gemma4AudioConfig.attention_context_right` | `0` | Right context window (causal: no right context) |
| `Gemma4AudioConfig.attention_logit_cap` | `50.0` | Cap applied to attention logits |
| `Gemma4AudioConfig.attention_invalid_logits_value` | `-1.0e9` | Fill value for invalid/masked attention positions |

### Output and Stability Parameters

| Parameter | Default | Description |
|---|---|---|
| `Gemma4AudioConfig.output_proj_dims` | `1536` | Dimension of the final linear projection from `hidden_size` to model output space |
| `Gemma4AudioConfig.use_clipped_linears` | `True` | Apply weight clipping to linear layers (note: `True` by default, unlike vision) |
| `Gemma4AudioConfig.gradient_clipping` | `1e10` | Gradient clipping bound for stability |

---

## 2.5 Key Cross-Modal Contrasts

The per-config parameter tables in Sections 2.2--2.4 contain all default values. The most important differences across modalities for porting purposes are: the text decoder uses GQA (4 KV heads vs. 8 query heads) while the vision encoder uses full MHA (12 KV heads = 12 query heads); the audio encoder uses `silu` activation whereas both text and vision use `gelu_pytorch_tanh`; and `use_clipped_linears` defaults to `True` only in the audio config (it is `False` in vision and absent in text).

---

## 2.6 TTNN Porting Considerations

### Dual Head Dimensions Require Separate Attention Kernels

The sliding vs. global head dimension difference (see Section 2.2, "Global Attention Layer Parameters") means TTNN must instantiate two distinct attention configurations with different Q/K/V projection shapes: `[2304, 2048]` for sliding layers vs. `[2304, 4096]` for global layers (given 8 query heads).

### Per-Layer RoPE Tables

The dual RoPE scheme (see Section 2.2, "Dual RoPE Parameters") requires TTNN to pre-compute two separate sin/cos tables -- one for sliding layers, one for global layers -- and store them as TTNN constants to avoid recomputing per forward pass.

### Layer Type Dispatch

The `Gemma4TextConfig.layer_types` list means the decoder is not uniform. A TTNN implementation must either:
1. Instantiate two template variants (sliding and global) and dispatch per layer, or
2. Parameterize a single decoder-layer module that branches internally on layer type.

Option 1 is preferable for TTNN because it avoids runtime branching and allows each variant to have optimally tiled weight layouts.

### Per-Layer Input Embeddings

The per-layer input embedding (`[262144, 256]`) is looked up every layer and added to the residual. This is an additional embedding gather per decoder layer that must be tiled appropriately. At `hidden_size_per_layer_input = 256`, this is small enough to replicate across devices.

### KV Sharing and K=V Optimization

If `Gemma4TextConfig.attention_k_eq_v` is enabled, the K and V projections share weights. A TTNN port should detect this and avoid allocating a separate V projection weight tensor, instead aliasing the K projection output.

If `Gemma4TextConfig.num_kv_shared_layers > 0`, consecutive layers share KV projection weights. This affects weight layout: shared weights must be stored once and referenced by multiple layer instances.

### Audio Encoder: Clipped Linears and Conformer Attention

`Gemma4AudioConfig.use_clipped_linears = True` means linear layers clamp their weights at runtime. TTNN matmul ops would need a clamped-weight variant or a pre-clamping step before each forward pass.

The chunked attention with `attention_chunk_size=12`, `attention_context_left=13`, and `attention_context_right=0` creates an asymmetric local attention pattern. This is not a standard sliding window and requires custom masking logic in TTNN.

### MoE Blocks

When `Gemma4TextConfig.enable_moe_block` is active, eligible decoder layers replace their dense MLP with a sparse MoE FFN. The TTNN implementation must handle the expert routing (top-k selection), per-expert matmuls, and the combine step. The `Gemma4TextConfig.moe_intermediate_size` determines per-expert FFN width, which is typically smaller than `Gemma4TextConfig.intermediate_size`.

### Vision Encoder: Small Theta and 2D RoPE

`Gemma4VisionConfig.default_theta = 100.0` with a 2D RoPE embedding over a patch grid is architecturally distinct from the text model's 1D RoPE. The TTNN RoPE kernel must support 2D position indexing and the much smaller theta value.

---

**Next:** [Chapter 3 -- Vision Encoder](../ch3_vision_encoder/index.md)
