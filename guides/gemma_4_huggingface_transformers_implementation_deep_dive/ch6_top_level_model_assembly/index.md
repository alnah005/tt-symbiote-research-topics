# Chapter 6: Top-Level Model Assembly and Multimodal Embedding

This chapter covers how `Gemma4ForConditionalGeneration` orchestrates the vision, audio, and text subsystems into a single multimodal model, and how `Gemma4MultimodalEmbedder` projects modality-specific features into the shared language model embedding space. All classes discussed here live in [`modeling_gemma4.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py), generated from [`modular_gemma4.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modular_gemma4.py). Refer to [Chapter 3](../ch3_vision_encoder/index.md) for the vision tower, [Chapter 4](../ch4_audio_encoder/index.md) for the audio tower, and [Chapter 5](../ch5_text_decoder/index.md) for the text decoder internals.

---

## 6.1 Inheritance Hierarchy

The top-level classes inherit from their Gemma 3n counterparts, deleting and re-creating submodules as needed:

```
PreTrainedModel
  +-- Gemma4PreTrainedModel            (standalone, does NOT extend Gemma3nPreTrainedModel)
        +-- Gemma4Model                (extends Gemma3nModel)
        +-- Gemma4ForConditionalGeneration (extends Gemma3nForConditionalGeneration, GenerationMixin)
```

In the modular source, `Gemma4Model` extends `Gemma3nModel` and `Gemma4ForConditionalGeneration` extends `Gemma3nForConditionalGeneration`. During code generation into `modeling_gemma4.py`, all inherited code is flattened, so the generated file has `Gemma4Model(Gemma4PreTrainedModel)` and `Gemma4ForConditionalGeneration(Gemma4PreTrainedModel, GenerationMixin)` with no cross-model imports at runtime.

The key difference from Gemma 3n: `Gemma4MultimodalEmbedder` is drastically simplified (see Section 6.3), and the model adds video support alongside image and audio.

---

## 6.2 Gemma4PreTrainedModel

```python
class Gemma4PreTrainedModel(PreTrainedModel):
    config: Gemma4Config
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _no_split_modules = ["Gemma4TextDecoderLayer", "Gemma4VisionEncoderLayer", "Gemma4AudioLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    input_modalities = ("image", "text", "video", "audio")
```

### Attention Backend Support

All four HuggingFace attention backends are supported: eager, SDPA, Flash Attention 2, and Flex Attention. The model is marked `_can_compile_fullgraph = True`, indicating compatibility with `torch.compile` in fullgraph mode.

### No-Split Modules

The `_no_split_modules` list tells the `device_map="auto"` allocator which modules must stay on a single device:
- `Gemma4TextDecoderLayer` -- a single decoder layer with its attention + MLP/MoE
- `Gemma4VisionEncoderLayer` -- a single vision transformer block
- `Gemma4AudioLayer` -- a single audio conformer block

### `_init_weights` Dispatch

The `_init_weights` method dispatches weight initialization by module type. It calls `super()._init_weights(module)` first (which handles standard `nn.Linear` / `nn.Embedding` via `config.initializer_range`), then applies class-specific overrides:

| Module Type | Initialization |
|---|---|
| `Gemma4VisionPatchEmbedder` | `position_embedding_table` set to all ones |
| `Gemma4AudioRelPositionalEncoding` | `inv_timescales` computed from log-spaced timescales (1.0 to 10000.0) |
| `Gemma4AudioAttention` | `softcap` set to `attention_logits_soft_cap`; `per_dim_scale` set to zeros |
| `Gemma4TextRotaryEmbedding` | Per-layer-type `inv_freq` buffers computed from their respective RoPE init functions |
| `Gemma4VisionRotaryEmbedding` | `inv_freq` computed from the configured rope function |
| `Gemma4TextScaledWordEmbedding` | `embed_scale` set to `sqrt(hidden_size)` |
| `Gemma4TextRouter` | `scale` and `per_expert_scale` set to ones |
| `Gemma4TextExperts` | `gate_up_proj` and `down_proj` sampled from `N(0, initializer_range)` |
| `Gemma4TextDecoderLayer` | `layer_scalar` set to ones |
| `Gemma4ClippableLinear` (if `use_clipped_linears`) | All clip bounds set to +/-inf |
| `Gemma4VisionModel` (if `standardize`) | `std_bias` set to zeros, `std_scale` set to ones |

---

## 6.3 Gemma4MultimodalEmbedder

```python
class Gemma4MultimodalEmbedder(Gemma3nMultimodalEmbedder):  # modular inheritance
```

This class projects encoder output features into the language model's hidden dimension. Compared to its `Gemma3nMultimodalEmbedder` parent, it is dramatically simplified. The constructor deletes six inherited attributes that are not needed:

```python
del self.embedding               # hard token embedding table
del self.hard_embedding_norm      # norm for hard embeddings
del self.soft_embedding_norm      # norm for soft embeddings
del self.vocab_offset             # offset into shared vocabulary
del self.vocab_size               # modality vocabulary size
del self.embedding_post_projection_norm  # post-projection norm
```

What remains are exactly two submodules:

```
Gemma4MultimodalEmbedder
  +-- embedding_pre_projection_norm: Gemma4RMSNorm(multimodal_hidden_size, eps=rms_norm_eps, with_scale=False)
  +-- embedding_projection: nn.Linear(multimodal_hidden_size, text_hidden_size, bias=False)
```

The `multimodal_hidden_size` is read from `multimodal_config.output_proj_dims` if it exists (used by the audio config), otherwise falls back to `multimodal_config.hidden_size` (used by the vision config).

### Forward

```python
def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
    # inputs_embeds: [batch_size, seq_len, multimodal_hidden_size]
    embs_normed = self.embedding_pre_projection_norm(inputs_embeds)  # RMSNorm without learnable scale
    return self.embedding_projection(embs_normed)                    # [batch_size, seq_len, text_hidden_size]
```

The forward path is a simple two-step pipeline: scale-free RMSNorm followed by a linear projection. There are no hard/soft embedding branches, no post-projection norm, and no vocabulary offset logic -- all of which existed in the Gemma 3n version.

Two instances of `Gemma4MultimodalEmbedder` are created in `Gemma4Model`:
- `embed_vision` -- initialized with `Gemma4VisionConfig` and `Gemma4TextConfig`
- `embed_audio` -- initialized with `Gemma4AudioConfig` and `Gemma4TextConfig`

---

## 6.4 Gemma4Model -- Full Module Tree

```
Gemma4Model (extends Gemma3nModel via modular / extends Gemma4PreTrainedModel in generated code)
  |
  +-- language_model: Gemma4TextModel              (see Chapter 5)
  |     +-- embed_tokens, layers, rotary_emb, norm
  |     +-- [optional] embed_tokens_per_layer, per_layer_model_projection, ...
  |
  +-- vision_tower: Gemma4VisionModel | None       (see Chapter 3)
  |     Created via AutoModel.from_config(config.vision_config)
  |
  +-- embed_vision: Gemma4MultimodalEmbedder | None
  |     embedding_pre_projection_norm + embedding_projection
  |     Projects vision features -> text_hidden_size
  |
  +-- audio_tower: Gemma4AudioModel | None         (see Chapter 4)
  |     Created via AutoModel.from_config(config.audio_config)
  |
  +-- embed_audio: Gemma4MultimodalEmbedder | None
  |     embedding_pre_projection_norm + embedding_projection
  |     Projects audio features -> text_hidden_size
```

Key attributes:
- `vocab_size` = `Gemma4TextConfig.vocab_size`
- `vocab_size_per_layer_input` = `Gemma4TextConfig.vocab_size_per_layer_input` (used for per-layer embedding if enabled)

The vision/audio towers and their embedders are `None` when the respective config section is absent, making the model gracefully degrade to text-only.

---

## 6.5 Gemma4Model.forward -- Data Flow

The `forward` method of `Gemma4Model` is the central orchestration point. It accepts raw token IDs alongside preprocessed multimodal tensors and produces fused hidden states. The flow has nine distinct stages:

### Stage 1: Compute Placeholder Masks

```python
image_mask, video_mask, audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds)
multimodal_mask = image_mask | video_mask | audio_mask
```

`get_placeholder_mask` compares `input_ids` against three sentinel token IDs from `Gemma4Config`:
- `Gemma4Config.image_token_id` (258880) -- marks image placeholder positions
- `Gemma4Config.video_token_id` (258884) -- marks video placeholder positions
- `Gemma4Config.audio_token_id` (258881) -- marks audio placeholder positions

Each mask is a `BoolTensor` of shape `[batch_size, seq_len]`. When `input_ids` is not available (pre-embedded input), the method compares embedding vectors directly against the embedding of each sentinel token.

### Stage 2: Replace Multimodal IDs with PAD

```python
llm_input_ids = input_ids.clone()
llm_input_ids[multimodal_mask] = config.text_config.pad_token_id
inputs_embeds = self.get_input_embeddings()(llm_input_ids)
```

Multimodal sentinel tokens would be out-of-vocabulary for the text embedding table. They are replaced with `pad_token_id` before embedding lookup, producing valid (but placeholder) embeddings that will be overwritten in later stages.

### Stage 3: Per-Layer Inputs (Conditional)

```python
if self.config.get_text_config().hidden_size_per_layer_input:
    pad_embedding = self.language_model.embed_tokens.weight[pad_token_id, :]
    llm_inputs_embeds = torch.where(multimodal_mask[..., None], pad_embedding, inputs_embeds)
    per_layer_inputs = self.language_model.get_per_layer_inputs(llm_input_ids, llm_inputs_embeds)
```

When `Gemma4TextConfig.hidden_size_per_layer_input` is set (256 by default), the model computes separate per-layer input embeddings. Multimodal positions are explicitly masked to the pad embedding before computing per-layer projections, so that multimodal features do not leak into the per-layer input pathway. See [Chapter 5](../ch5_text_decoder/index.md) for details on per-layer inputs.

### Stage 4: Vision Tower -- Images

```python
if pixel_values is not None:
    image_features = self.get_image_features(pixel_values, image_position_ids, return_dict=True).pooler_output
```

`get_image_features` runs:
1. `self.vision_tower(pixel_values, pixel_position_ids)` -- full vision encoder forward (see [Chapter 3](../ch3_vision_encoder/index.md))
2. `self.embed_vision(last_hidden_state)` -- projects to `text_hidden_size` and stores result in `pooler_output`

The result shape is `[num_images, num_patches, text_hidden_size]`.

### Stage 5: Vision Tower -- Videos

```python
if pixel_values_videos is not None:
    pixel_values_videos = pixel_values_videos.flatten(0, 1)  # [num_videos * num_frames, ...]
    video_position_ids = video_position_ids.flatten(0, 1)
    # Same vision tower, same embed_vision
    video_features = self.get_video_features(pixel_values_videos, video_position_ids, ...).pooler_output
```

Videos reuse the same `vision_tower` and `embed_vision` as images. The video tensor `[num_videos, num_frames, C, H, W]` is flattened to `[num_videos * num_frames, C, H, W]` so frames are processed as independent images. The position IDs `[num_videos, num_frames, max_patches, 2]` are similarly flattened.

### Stage 6: Audio Tower

```python
if input_features is not None and input_features_mask is not None:
    audio_output = self.get_audio_features(input_features, input_features_mask, return_dict=True)
    audio_features = audio_output.pooler_output       # embed_audio(last_hidden_state)
    audio_mask_from_encoder = audio_output.attention_mask  # True = valid
    audio_features = audio_features[audio_mask_from_encoder]  # strip padding
```

`get_audio_features` runs:
1. `self.audio_tower(input_features, input_features_mask)` -- full audio encoder forward (see [Chapter 4](../ch4_audio_encoder/index.md))
2. `self.embed_audio(last_hidden_state)` -- projects to `text_hidden_size` and stores in `pooler_output`

A critical detail: after projection, padding tokens are stripped using the encoder's attention mask. Only valid (non-padding) audio tokens are kept. This parallels the vision encoder's own padding stripping.

### Stage 7: masked_scatter Each Modality

For each modality (image, video, audio), the features are scattered into `inputs_embeds` at the corresponding mask positions:

```python
# For each modality:
mask_3d = mask_2d.unsqueeze(-1).expand_as(inputs_embeds)  # [batch, seq, hidden] bool
inputs_embeds = inputs_embeds.masked_scatter(mask_3d, features)
```

Before scattering, a `torch_compilable_check` verifies that the number of feature elements matches the number of placeholder positions. This catch mismatches between the preprocessor's placeholder count and the encoder's actual output length.

### Stage 8: Causal Mask Construction

```python
if not isinstance(attention_mask, dict):
    if config.get_text_config().use_bidirectional_attention == "vision":
        causal_mask_mapping = create_causal_mask_mapping(...)
    else:
        causal_mask_mapping = create_masks_for_generate(...)
```

The mask construction has two paths depending on `Gemma4TextConfig.use_bidirectional_attention`:

**Path A -- `"vision"` (larger models):** Calls `create_causal_mask_mapping`, which produces a dict with two keys:
- `"full_attention"` -- standard causal mask (via `create_causal_mask`)
- `"sliding_attention"` -- sliding window causal mask with an `or_mask_function` that enables bidirectional attention within vision token groups

The `or_mask_function` is built from `mm_token_type_ids`:
1. Tokens with type ID 1 or 2 are identified as vision tokens
2. Contiguous vision blocks are assigned group IDs via `cumsum` on boundary detection
3. The mask function allows tokens within the same vision group to attend bidirectionally (overriding the causal constraint)

**Path B -- `None` (smaller models):** Calls the standard `create_masks_for_generate`, which returns conventional causal (or causal + sliding) masks without bidirectional vision attention.

The resulting `causal_mask_mapping` dict is passed directly to the language model, which uses the appropriate mask per layer type.

### Stage 9: Run Language Model

```python
outputs = self.language_model(
    per_layer_inputs=per_layer_inputs,
    attention_mask=causal_mask_mapping,
    position_ids=position_ids,
    past_key_values=past_key_values,
    inputs_embeds=inputs_embeds,
    use_cache=use_cache,
)
```

The fused `inputs_embeds` (text + vision + audio features) and the mask mapping are passed to `Gemma4TextModel.forward` (see [Chapter 5](../ch5_text_decoder/index.md)). The output is wrapped in `Gemma4ModelOutputWithPast`, which extends `BaseModelOutputWithPast` with two additional fields:
- `image_hidden_states` -- the projected vision features (before scatter), or `None`
- `audio_hidden_states` -- the projected audio features (after padding strip), or `None`

---

## 6.6 Gemma4ForConditionalGeneration

```python
class Gemma4ForConditionalGeneration(Gemma3nForConditionalGeneration, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    base_model_prefix = "model"
```

### Module Tree

```
Gemma4ForConditionalGeneration
  +-- model: Gemma4Model              (Section 6.4)
  +-- lm_head: nn.Linear(text_hidden_size, vocab_size, bias=False)
        Weight TIED to model.language_model.embed_tokens.weight
```

The `lm_head` weight is tied to the input embedding weight via `_tied_weights_keys`. This means the output projection shares parameters with the token embedding table -- a standard practice that reduces parameter count.

### Forward

The `forward` method wraps `Gemma4Model.forward` with three additional operations:

**1. Logit Computation with Slicing:**
```python
slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
logits = self.lm_head(hidden_states[:, slice_indices, :])
# logits shape: [batch_size, logits_to_keep, vocab_size]
```
The `logits_to_keep` parameter (default 0, meaning keep all) allows computing logits only for the last N positions, which is critical for efficient generation where only the final token's logits are needed.

**2. Optional Softcapping:**
```python
if (final_logit_softcapping := config.get_text_config().final_logit_softcapping) is not None:
    logits = logits / final_logit_softcapping
    logits = torch.tanh(logits)
    logits = logits * final_logit_softcapping
```
When `Gemma4TextConfig.final_logit_softcapping` is set, logits are clamped via a tanh function that limits their magnitude to `+/- final_logit_softcapping`. This is the same softcapping used in Gemma 2/3. Currently defaults to `None` in Gemma 4 config.

**3. Loss Computation:**
When `labels` are provided, the method computes standard shifted cross-entropy loss:
- Logits are upcast to `float32` for numerical stability
- Shift: `shift_logits = logits[..., :-1, :]`, `shift_labels = labels[..., 1:]`
- If an `attention_mask` is present, padding positions are removed before loss computation
- Loss is computed via `nn.CrossEntropyLoss()` on the flattened tensors

Note: `Gemma4Model` has `accepts_loss_kwargs = False`, meaning the loss is not divided by `num_items_in_batch` -- the filtering of padding tokens already handles this.

### Output

Returns `Gemma4CausalLMOutputWithPast`, which extends `ModelOutput` with:
- `loss` -- scalar loss (if labels provided), else `None`
- `logits` -- `[batch_size, logits_to_keep, vocab_size]`
- `past_key_values` -- updated KV cache
- `hidden_states`, `attentions` -- optional layer outputs
- `image_hidden_states`, `audio_hidden_states` -- projected modality features

---

## 6.7 Generation Support

### `prepare_inputs_for_generation`

This method adapts inputs for autoregressive generation. It calls the parent class implementation and then conditionally includes multimodal inputs:

```python
if is_first_iteration or not use_cache:
    model_inputs["pixel_values"] = pixel_values
    model_inputs["pixel_values_videos"] = pixel_values_videos
    model_inputs["input_features"] = input_features
    model_inputs["input_features_mask"] = input_features_mask
```

Multimodal tensors are only passed on the first iteration (prefill). On subsequent decode steps with KV caching, the multimodal features are already encoded in the cached key-value states, so the raw pixel/audio inputs are omitted.

### `create_masks_for_generate` (staticmethod)

A static method used by the generation pipeline to create attention masks. It branches on `use_bidirectional_attention`:
- `"vision"` -- calls `create_causal_mask_mapping` (with bidirectional vision attention)
- Otherwise -- calls the standard `create_masks_for_generate`

This mirrors the logic in `Gemma4Model.forward` (Stage 8) but is exposed as a static method for use in generation contexts where the model's `forward` is not directly called for mask creation.

---

## 6.8 Output Dataclasses

### Gemma4ModelOutputWithPast

```python
class Gemma4ModelOutputWithPast(BaseModelOutputWithPast):
    image_hidden_states: torch.FloatTensor | None = None
    audio_hidden_states: torch.FloatTensor | None = None
```

Extends `BaseModelOutputWithPast` (which provides `last_hidden_state`, `past_key_values`, `hidden_states`, `attentions`) with the two modality-specific hidden state fields.

### Gemma4CausalLMOutputWithPast

```python
class Gemma4CausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple | None = None
    attentions: tuple | None = None
    image_hidden_states: torch.FloatTensor | None = None
    audio_hidden_states: torch.FloatTensor | None = None
```

In the modular source, both are trivial subclasses of their Gemma 3n counterparts (`Gemma3nModelOutputWithPast`, `Gemma3nCausalLMOutputWithPast`). In the generated file, they are fully expanded.

---

## 6.9 TTNN Porting Considerations

### Multimodal Embedder

`Gemma4MultimodalEmbedder` is straightforward to port: it is an `RMSNorm` (without learnable scale) followed by a `nn.Linear` (no bias). Both are standard TTNN operations. Two separate instances exist (vision and audio), each with different input dimensions determined by `multimodal_hidden_size`.

### masked_scatter Fusion

The `masked_scatter` calls in Stage 7 are the trickiest part of the assembly to port. In PyTorch, `masked_scatter` writes a flat source tensor into positions indicated by a boolean mask. TTNN does not have a direct `masked_scatter` equivalent, so the porter must either:

1. **Indexed write approach:** Convert the boolean mask to index tensors and use `tt_lib.tensor.scatter` or equivalent indexed assignment.
2. **Pre-allocation approach:** Build the complete `inputs_embeds` tensor by concatenating text and modality features at the correct positions during preprocessing, avoiding the need for runtime scatter entirely.
3. **Elementwise blend:** Use `where(mask, features, text_embeds)` after reshaping features to match the full sequence, though this requires careful alignment.

Option 2 is likely the most performant on TT hardware, as it moves the assembly logic to the host and produces a single contiguous tensor for device transfer.

### Placeholder Mask Computation

`get_placeholder_mask` performs element-wise comparison of `input_ids` against constant token IDs. This is a host-side operation that produces small boolean tensors. It should remain on the CPU since the results are used for control flow (conditional branching on whether modality data is present) rather than large-tensor computation.

### Causal Mask Mapping

The two-path mask construction (bidirectional vision vs. standard causal) produces a Python dict mapping attention type names to mask tensors. For TTNN, the masks should be pre-computed on the host and transferred to device once per forward pass. The `or_mask_function` used for bidirectional vision attention is a Python callable evaluated per-element -- this is only used with Flex Attention and would not apply in a TTNN port, which should directly construct the appropriate mask tensors.

### Weight Tying

The `lm_head.weight` is tied to `model.language_model.embed_tokens.weight`. When loading weights for TTNN, ensure only one copy of this weight is allocated in device memory, with both the embedding lookup and the output projection referencing the same buffer (potentially transposed for the linear projection).

### Softcapping

The optional `final_logit_softcapping` path (`x / cap -> tanh -> * cap`) is a simple elementwise sequence. If `final_logit_softcapping` is `None` (the current default), this path is skipped entirely and can be omitted from the TTNN graph.

### Per-Layer Inputs

The per-layer input computation in Stage 3 uses `torch.where` with a broadcast mask to zero out multimodal positions before computing per-layer projections. This is conceptually simple but involves a tensor of shape `[batch, seq, hidden]` and should be fused with the per-layer projection on device if possible.

### Generation Loop

For autoregressive generation, the key optimization is that multimodal inputs are only processed on the first iteration. The TTNN port should reflect this by caching the encoded multimodal features in the KV cache after prefill, and running only the text decoder path on subsequent decode steps.

---

**Next:** [Chapter 7 -- Preprocessing Pipelines](../ch7_preprocessing_pipelines/index.md)
