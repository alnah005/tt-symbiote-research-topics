# Chapter 8: Weight Conversion

This chapter covers [`convert_gemma4_weights.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/convert_gemma4_weights.py), the offline script that converts Google's Orbax (JAX) checkpoints into HuggingFace-compatible SafeTensors files. Understanding this script is critical for TTNN porting because it defines the authoritative mapping between Google's internal parameter names and the `state_dict` keys used by the HuggingFace model classes documented in earlier chapters.

---

## 8.1 Script Entry Point and CLI Flags

The script uses `absl-py` for flag parsing and entry:

```python
from absl import app, flags, logging

if __name__ == "__main__":
    app.run(main)
```

The `main()` function orchestrates the entire pipeline: load config, convert weights, instantiate model, save SafeTensors, build tokenizer, assemble processor, and write generation config.

### CLI Flags

| Flag | Type | Required | Default | Purpose |
|---|---|---|---|---|
| `--variant` | enum | Yes | -- | Model variant to convert (see Section 8.2) |
| `--checkpoint_path` | string | Yes | -- | Path to Orbax checkpoint directory |
| `--output_path` | string | Yes | -- | Destination for HF SafeTensors output |
| `--tokenizer_path` | string | Yes | -- | Path to SentencePiece `.model` file |
| `--text_dtype` | enum | No | `bfloat16` | Dtype for text decoder weights (`float32`, `bfloat16`, `float16`) |
| `--vision_dtype` | enum | No | `bfloat16` | Dtype for vision encoder weights |
| `--audio_dtype` | enum | No | `bfloat16` | Dtype for audio encoder weights |
| `--text_only` | bool | No | `False` | If `True`, saves `Gemma4ForCausalLM` instead of `Gemma4ForConditionalGeneration` |
| `--include_chat_template` | bool | No | `False` | Inject Jinja chat template into the tokenizer |
| `--include_response_schema` | bool | No | `False` | Inject structured response schema into the tokenizer |
| `--verbose` | bool | No | `False` | Log path, shape, and dtype for every converted weight |

Example invocation:

```bash
python convert_gemma4_weights.py \
    --variant='gemma-4-e2b' \
    --include_chat_template \
    --include_response_schema \
    --tokenizer_path="$HOME/tokenizers/gemma4/gemma4_cleaned_262144.model" \
    --checkpoint_path="$HOME/gemma4/checkpoints/gemma_e2b_it_orbax" \
    --output_path="$HOME/gemma4/checkpoints/gemma_e2b_it_safetensors"
```

---

## 8.2 Supported Variants

The `_VARIANTS` dictionary maps variant name strings to fully constructed `Gemma4Config` objects. Each config bundles a `Gemma4TextConfig`, `Gemma4VisionConfig`, and optionally `Gemma4AudioConfig`:

| Variant String | Text hidden_size | Layers | Heads / KV Heads | Vision Config | MoE |
|---|---|---|---|---|---|
| `gemma-4-e2b` | 1536 | 35 | 8 / 1 | On-device (768-dim, 16 layers) | No |
| `gemma-4-e4b` | 2560 | 42 | 8 / 2 | On-device (768-dim, 16 layers) | No |
| `gemma-4-31b` | 5376 | 60 | 32 / 16 | Large (1152-dim, 27 layers) | No |
| `gemma-4-26b-a4b` | 2816 | 30 | 16 / 8 | Large (1152-dim, 27 layers) | Yes (128 experts, top-8) |

### Vision Config Presets

Two vision configs are hard-coded:

**`_ON_DEVICE_VISION_CONFIG`** (for E2B and E4B):
- `hidden_size=768`, `intermediate_size=3072`, `num_hidden_layers=16`
- `num_attention_heads=12`, `head_dim=64`, `pooling_kernel_size=3`
- `use_clipped_linears=True`

**`_LARGE_MODEL_VISION_CONFIG`** (for 31B and 26B-A4B):
- `hidden_size=1152`, `intermediate_size=4304`, `num_hidden_layers=27`
- `num_attention_heads=16`, `head_dim=72`, `pooling_kernel_size=3`
- `use_clipped_linears=False`, `standardize=True`

### Layer Type Patterns

Layer types follow repeating sliding/full attention patterns:

- **E2B:** `["sliding_attention"] * 4 + ["full_attention"]` repeated 7 times (35 layers)
- **Default (E4B, 31B, 26B-A4B):** `["sliding_attention"] * 5 + ["full_attention"]` repeated N times

### RoPE Parameters

Two `RopeParameters` objects are defined per attention type:

| Attention Type | `rope_theta` | `rope_type` | `partial_rotary_factor` |
|---|---|---|---|
| `full_attention` | 1,000,000.0 | `"proportional"` | 0.25 |
| `sliding_attention` | 10,000.0 | `"default"` | (not set) |

---

## 8.3 Orbax Checkpoint Loading

The `_restore_checkpoint()` function handles loading multi-device sharded JAX checkpoints onto a single CPU device:

```python
def _restore_checkpoint(checkpoint_path: str) -> dict:
    metadata_path = os.path.join(checkpoint_path, "_METADATA")
    with open(metadata_path, "rb") as f:
        metadata = json.loads(f.read())

    tree_metadata = metadata["tree_metadata"]

    # Build a nested dict matching the checkpoint's tree structure
    target = {}
    for key_str in tree_metadata:
        keys = ast.literal_eval(key_str)
        d = target
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = np.zeros(1)  # placeholder leaf

    device = jax.devices("cpu")[0]
    sharding = SingleDeviceSharding(device)
    restore_args_tree = tree.map_structure(
        lambda _: type_handlers.ArrayRestoreArgs(sharding=sharding), target
    )
    restore = obc_args.PyTreeRestore(item=target, restore_args=restore_args_tree)

    checkpointer = obc.PyTreeCheckpointer()
    return checkpointer.restore(checkpoint_path, args=restore)
```

Key steps:
1. Read `_METADATA` JSON from the checkpoint directory to discover the parameter tree structure.
2. Build a placeholder `target` dict with `np.zeros(1)` leaves matching the tree topology.
3. Use `jax.sharding.SingleDeviceSharding` on the CPU device to consolidate all shards.
4. Call `obc.PyTreeCheckpointer().restore()` with `PyTreeRestore` args to materialize all parameters as NumPy-backed JAX arrays on CPU.

The result is a nested Python dict where leaf values are JAX arrays that can be cast to NumPy with `np.asarray()`.

---

## 8.4 Weight Name Mapping: Orbax to HuggingFace

The `convert()` function is the top-level dispatcher. It flattens the Orbax tree with `tree.flatten_with_path(ckpt)` and routes each `(path, param, value)` triple to one of three sub-converters based on the Orbax path prefix:

| Orbax Path Prefix | Dispatcher Target | HF Key Prefix |
|---|---|---|
| `transformer/...` | `convert_transformer_weights()` | `model.language_model.` (or `model.` if text-only) |
| `PatchInputVariablePoolingEncoder_0/...` | `convert_vision_encoder_weights()` | `model.vision_tower.` |
| `AudioEncoder/encoder/...` | `convert_audio_encoder_weights()` | `model.audio_tower.` |
| `...mm_input_projection` | `.transpose()` applied | `model.embed_vision.embedding_projection.weight` |
| `...audio_input_projection` | `.transpose()` applied | `model.embed_audio.embedding_projection.weight` |

After conversion, the embedding weight is shared with the LM head:

```python
hf_tree["lm_head.weight"] = hf_tree[f"{text_path_prefix}.embed_tokens.weight"]
```

### 8.4.1 Dtype Conversion Strategy

The `update_tree()` helper converts each weight through a careful memory-efficient pipeline:

1. Cast JAX array to `float32` via `np.asarray(weights, dtype=np.float32)` (single copy)
2. Create a PyTorch tensor via `torch.from_numpy()` (zero-copy, shares memory)
3. Cast to target dtype (e.g., `bfloat16`) only if different from `float32`
4. Delete intermediates to allow garbage collection

This avoids the double-copy pattern of `np.asarray()` followed by `.astype("float32")`.

---

## 8.5 Text Decoder Weight Mapping

`convert_transformer_weights()` handles two checkpoint formats:

### New Format: `transformer/layer_N/...`

Per-layer weights (not stacked). The layer index is extracted directly from the path string `transformer/layer_0/attn/q_einsum`.

### Old Format: `transformer/stacked_layers/attention_type_N/...`

Weights are stacked along dimension 0, grouped by attention type position within the 6-layer sliding window pattern. The actual layer index is computed as:

```python
layer_idx = _SLIDING_WINDOW_PATTERN * i + attention_type_index
```

where `_SLIDING_WINDOW_PATTERN = 6` and `attention_type_index` is 0-5.

### Key Mappings and Transpositions

| Orbax Path Suffix | HF Key Suffix | Transposition |
|---|---|---|
| `attn/q_einsum` | `self_attn.q_proj.weight` | `.transpose(1,0,2).reshape(hidden, heads*head_dim).transpose()` |
| `attn/kv_einsum` | `self_attn.k_proj.weight` + `self_attn.v_proj.weight` | `.transpose(0,2,1,3)` then split, reshape, transpose per proj |
| `attn/k_einsum` | `self_attn.k_proj.weight` | `.transpose(1,0,2).reshape(hidden, gkv_heads*head_dim).transpose()` |
| `attn/attn_vec_einsum` | `self_attn.o_proj.weight` | `.transpose(2,0,1).reshape(hidden, heads*head_dim)` |
| `attn/query_norm` | `self_attn.q_norm.weight` | None |
| `attn/key_norm` | `self_attn.k_norm.weight` | None |
| `mlp/gating_einsum` | `mlp.gate_proj.weight` + `mlp.up_proj.weight` | Split along dim 0 (dense) |
| `mlp/linear` | `mlp.down_proj.weight` | `.transpose()` |
| `pre_attention_norm` | `input_layernorm.weight` | None |
| `post_attention_norm` | `post_attention_layernorm.weight` | None |
| `pre_ffw_norm` | `pre_feedforward_layernorm.weight` | None |
| `post_ffw_norm` | `post_feedforward_layernorm.weight` | None |
| `per_layer_input_gate` | `per_layer_input_gate.weight` | `.transpose()` |
| `per_layer_projection` | `per_layer_projection.weight` | `.transpose()` |
| `post_per_layer_input_norm` | `post_per_layer_input_norm.weight` | None |
| (path end, param=`skip_scale`) | `layer_scalar` | None (scalar) |

### Embedder Weights

| Orbax Path | Param | HF Key | Notes |
|---|---|---|---|
| `transformer/embedder` | `input_embedding` | `embed_tokens.weight` | Direct copy |
| `transformer/embedder` | `per_layer_embeddings` | `embed_tokens_per_layer.weight` | Reshape `(vocab, layers, dim)` to `(vocab, layers*dim)` |
| `transformer/embedder/per_layer_model_projection` | -- | `per_layer_model_projection.weight` | Reshape + transpose |
| `transformer/embedder/per_layer_projection_norm` | -- | `per_layer_projection_norm.weight` | Direct copy |
| `transformer/final_norm` | -- | `norm.weight` | Direct copy |

### MoE-Specific Mappings (26B-A4B)

When `config.enable_moe_block` is `True`, the JAX checkpoint uses `mlp` for MoE experts and `mlp2` for the shared expert. The HF convention inverts this:

| Orbax Path Suffix | HF Key Suffix | Shape Transformation |
|---|---|---|
| `mlp/gating_einsum` (MoE) | `experts.gate_up_proj` | `(E, 2, moe_inter, hidden)` -> `(E, 2*moe_inter, hidden)` |
| `mlp/linear` (MoE) | `experts.down_proj` | `.transpose(0, 2, 1)` |
| `mlp/router_logits` | `router.proj.weight` | `.transpose()` |
| `mlp` (param=`router_scale`) | `router.scale` | None |
| `mlp` (param=`per_expert_scale`) | `router.per_expert_scale` | None |
| `mlp2/gating_einsum` | `mlp.gate_proj.weight` + `mlp.up_proj.weight` | Split along dim 0 |
| `mlp2/linear` | `mlp.down_proj.weight` | `.transpose()` |
| `pre_ffw_norm` (MoE) | `pre_feedforward_layernorm_2.weight` | None |
| `post_ffw1_norm` (MoE) | `post_feedforward_layernorm_2.weight` | None |
| `post_ffw2_norm` (MoE) | `post_feedforward_layernorm_1.weight` | None |

Note the `_1` / `_2` suffix swap between JAX `ffw1`/`ffw2` and HF `layernorm_1`/`layernorm_2` -- the naming inversion follows from swapping which MLP is "primary" vs "secondary."

---

## 8.6 Vision Encoder Weight Mapping

`convert_vision_encoder_weights()` handles the `PatchInputVariablePoolingEncoder_0` prefix. Vision transformer layers are stacked along dimension 0 in the Orbax checkpoint.

### Key Mappings

| Orbax Path | HF Key | Transposition |
|---|---|---|
| `entry/input_projection` (param=`w`) | `patch_embedder.input_proj.weight` | `.transpose()` |
| `entry` (param=`pos_emb`) | `patch_embedder.position_embedding_table` | `.transpose(1,0,2)` from `(10240, 2, hidden)` to `(2, 10240, hidden)` |
| `exit` (param=`scale`) | `pooler.scale` | None (shape `(1,1,d)`) |
| `standardize` (param=`bias`) | `std_bias` | None |
| `standardize` (param=`scale`) | `std_scale` | None |
| `block/.../attn/q_einsum` | `encoder.layers.{i}.self_attn.q_proj.linear.weight` | `.transpose(1,0,2).reshape(hidden, heads*head_dim).transpose()` |
| `block/.../attn/kv_einsum` | `...k_proj.linear.weight` + `...v_proj.linear.weight` | `.transpose(0,2,1,3)` then split, reshape, transpose |
| `block/.../attn/attn_vec_einsum` | `...o_proj.linear.weight` | `.transpose(2,0,1).reshape(hidden, heads*head_dim)` |
| `block/.../mlp/gating_einsum` | `...mlp.gate_proj.linear.weight` + `...mlp.up_proj.linear.weight` | Split along dim 0 |
| `block/.../mlp/linear` | `...mlp.down_proj.linear.weight` | `.transpose()` |

Note that vision encoder HF keys use `.linear.weight` (reflecting `ClippedLinear` modules), while text decoder keys use plain `.weight`.

### ClippedEinsum Parameters

For models with `use_clipped_linears=True` (on-device variants), additional clip bounds are extracted from `ClippedEinsum_0` suffixed paths. The `param` field contains names like `clip_min` or `clip_max`, and the `clip_` prefix is stripped to produce HF attribute names `min` / `max`. The same clip bounds are duplicated to both `k_proj` and `v_proj` (and `gate_proj`/`up_proj`) because the JAX implementation uses a single fused einsum for these pairs.

### Compression Einsum Parameters

Paths containing `/compression_einsum/` carry activation clipping bounds for quantized inference. These map to HF keys like `self_attn.q_proj.input_min`, `self_attn.q_proj.output_max`, etc.

---

## 8.7 Audio Encoder Weight Mapping

`convert_audio_encoder_weights()` handles the `AudioEncoder/encoder` prefix. Conformer layers are stacked along dimension 0.

### Subsample Conv Projection (SSCP)

| Orbax Path Suffix | HF Key | Transposition |
|---|---|---|
| `feature/.../input_proj` | `subsample_conv_projection.input_proj_linear.weight` | `.transpose(2,0,1).reshape(hidden, channels**2)` |
| `feature/.../subsampling_N` | `subsample_conv_projection.layerN.conv.weight` | `.transpose(3,2,0,1)` |
| `feature/.../norm_N` | `subsample_conv_projection.layerN.norm.weight` | None |
| `output_projection` (param=`kernel`) | `output_proj.weight` | `.transpose()` |
| `output_projection` (param=`bias`) | `output_proj.bias` | None |

### Conformer Layer Mappings

For each layer `i` in the stacked conformer:

| Orbax Path Suffix | HF Key Pattern | Transposition |
|---|---|---|
| `fflayer_start/.../ffn_layer1` | `layers.{i}.feed_forward1.ffw_layer_1.linear.weight` | `.transpose()` |
| `fflayer_start/.../ffn_layer2` | `layers.{i}.feed_forward1.ffw_layer_2.linear.weight` | `.transpose()` |
| `fflayer_end/.../ffn_layer1` | `layers.{i}.feed_forward2.ffw_layer_1.linear.weight` | `.transpose()` |
| `fflayer_end/.../ffn_layer2` | `layers.{i}.feed_forward2.ffw_layer_2.linear.weight` | `.transpose()` |
| `lconv/.../linear_start` | `layers.{i}.lconv1d.linear_start.linear.weight` | `.transpose()` |
| `lconv/.../linear_end` | `layers.{i}.lconv1d.linear_end.linear.weight` | `.transpose()` |
| `lconv/.../depthwise_conv1d` | `layers.{i}.lconv1d.depthwise_conv1d.weight` | `.transpose()` |
| `trans_atten/.../query_key_value_projection` | `...self_attn.{q,k,v}_proj.linear.weight` | `.transpose(1,0,2,3)` then reshape + transpose per head |
| `trans_atten/.../post` | `...self_attn.post.linear.weight` | `.transpose(2,0,1).reshape(hidden, hidden)` |
| `trans_atten/.../pos_proj` | `...self_attn.relative_k_proj.weight` | `.reshape(hidden, hidden).transpose()` |

Audio ClippedEinsum paths follow the same `clip_` prefix stripping pattern as vision.

---

## 8.8 Config, Tokenizer, and Processor Construction

After weight conversion, `main()` assembles the full HuggingFace artifact set.

### 8.8.1 Model Instantiation and Saving

```python
with accelerate.init_empty_weights():
    if _TEXT_ONLY.value:
        model = Gemma4ForCausalLM(config=config)
    else:
        model = Gemma4ForConditionalGeneration(config=config)

model.load_state_dict(state_tree, assign=True)
model.save_pretrained(output_path, state_dict=state_tree, safe_serialization=True)
```

`accelerate.init_empty_weights()` creates the model with meta tensors (zero memory), then `load_state_dict(..., assign=True)` replaces them with the converted weights in-place. This avoids the double-memory cost of allocating random-initialized parameters and then overwriting them. The `safe_serialization=True` flag ensures output is in SafeTensors format.

### 8.8.2 Tokenizer Construction

```python
sentencepiece_extractor = SentencePieceExtractor(_TOKENIZER_PATH.value)
vocab, _, merges = sentencepiece_extractor.extract()
tokenizer = GemmaTokenizer(
    vocab=vocab,
    merges=merges,
    add_bos_token=False,
    padding_side="left",
    extra_special_tokens={...},
    **chat_template_kwargs,
    **response_schema_kwargs,
)
```

The `extra_special_tokens` dict defines 18 special tokens for multimodal delimiters and tool-calling:

| Token Name | String | Purpose |
|---|---|---|
| `image_token` | `<\|image\|>` | Image soft-token placeholder |
| `boi_token` | `<\|image>` | Begin-of-image delimiter |
| `eoi_token` | `<image\|>` | End-of-image delimiter |
| `audio_token` | `<\|audio\|>` | Audio soft-token placeholder |
| `boa_token` | `<\|audio>` | Begin-of-audio delimiter |
| `eoa_token` | `<audio\|>` | End-of-audio delimiter |
| `sot_token` | `<\|turn>` | Start-of-turn |
| `eot_token` | `<turn\|>` | End-of-turn |
| `soc_token` | `<\|channel>` | Start-of-channel (thinking) |
| `eoc_token` | `<channel\|>` | End-of-channel |
| `think_token` | `<\|think\|>` | Think trigger |
| `escape_token` | `<\|"\|>` | Quote escape |
| `str_token` | `<\|tool_response>` | Start tool response |
| `etr_token` | `<tool_response\|>` | End tool response |
| `stc_token` | `<\|tool_call>` | Start tool call |
| `etc_token` | `<tool_call\|>` | End tool call |
| `std_token` | `<\|tool>` | Start tool definition |
| `etd_token` | `<tool\|>` | End tool definition |

After tokenizer creation, multimodal token IDs are extracted and written back to the config:

```python
config.image_token_id = tokenizer.image_token_id
config.boi_token_id = tokenizer.convert_tokens_to_ids(tokenizer.boi_token)
config.eoi_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eoi_token)
# ... same for audio tokens
config.save_pretrained(output_path)  # Re-save with correct IDs
```

### 8.8.3 Chat Template and Response Schema

Chat templates are downloaded from HuggingFace Hub repos at module import time:

```python
_CHAT_TEMPLATE = pathlib.Path(
    cached_file("gg-hf-gg/gemma-4-E4B-it", "chat_template.jinja")
).read_text()
_CHAT_TEMPLATE_LARGE = pathlib.Path(
    cached_file("gg-hf-gg/gemma-4-31B-it", "chat_template.jinja")
).read_text()
```

Large variants (`gemma-4-31b`, `gemma-4-26b-a4b`) use `_CHAT_TEMPLATE_LARGE`; smaller variants use `_CHAT_TEMPLATE`. When `--include_chat_template` is set, `eos_token_id` is updated to `[1, 106]` (both `<eos>` and `<end_of_turn>`).

The response schema (`_RESPONSE_SCHEMA`) is a JSON object with regex-based extraction rules for structured output parsing of assistant responses, thinking blocks, and tool calls. It is injected as the `response_schema` kwarg to `GemmaTokenizer`.

### 8.8.4 Processor Assembly (Multimodal)

For multimodal models (not `--text_only`), the script assembles a `Gemma4Processor`:

```python
feature_extractor = Gemma4AudioFeatureExtractor()
image_processor = Gemma4ImageProcessor(
    image_seq_length=vision_config.default_output_length,
    do_normalize=False,
    max_soft_tokens=vision_config.default_output_length,
    pooling_kernel_size=3,
)
video_processor = Gemma4VideoProcessor()
processor = Gemma4Processor(
    image_processor=image_processor,
    feature_extractor=feature_extractor,
    video_processor=video_processor,
    tokenizer=tokenizer,
    image_seq_length=vision_config.default_output_length,
    **chat_template_kwargs,
)
processor.save_pretrained(output_path)
```

### 8.8.5 Generation Config

```python
generation_config = GenerationConfig(
    pad_token_id=config.get_text_config().pad_token_id,
    bos_token_id=config.get_text_config().bos_token_id,
    eos_token_id=[eos_id, eot_id, str_id],  # when chat template is included
    temperature=1.0,
    do_sample=True,
    top_k=64,
    top_p=0.95,
)
generation_config.save_pretrained(output_path)
```

When `--include_chat_template` is active, the EOS token list includes the standard EOS, end-of-turn, and start-of-tool-response tokens.

---

## 8.9 Key Weight Shapes for TTNN Porting

The following table summarizes the most important weight tensors for the 31B variant, their HF `state_dict` shapes (after conversion), and the transposition applied during conversion. These are the shapes that a TTNN port will receive from `safetensors.load()`.

### Text Decoder (31B: hidden=5376, heads=32, kv_heads=16, global_kv_heads=4)

| HF Key Pattern | Shape (full attention, head_dim=256) | Notes |
|---|---|---|
| `layers.{i}.self_attn.q_proj.weight` | `(8192, 5376)` | 32 heads * 256 head_dim |
| `layers.{i}.self_attn.k_proj.weight` | `(4096, 5376)` (full) or `(1024, 5376)` (sliding, 16 kv_heads * 64 head_dim) | Full: 16 kv_heads * 256; Sliding: 16 kv_heads * 64 |
| `layers.{i}.self_attn.v_proj.weight` | Same as k_proj | |
| `layers.{i}.self_attn.o_proj.weight` | `(5376, 8192)` | |
| `layers.{i}.self_attn.q_norm.weight` | `(256,)` or `(64,)` | Per head_dim |
| `layers.{i}.self_attn.k_norm.weight` | `(256,)` or `(64,)` | Per head_dim |
| `layers.{i}.mlp.gate_proj.weight` | `(21504, 5376)` | 4 * hidden_size |
| `layers.{i}.mlp.up_proj.weight` | `(21504, 5376)` | |
| `layers.{i}.mlp.down_proj.weight` | `(5376, 21504)` | Transposed from JAX |
| `layers.{i}.input_layernorm.weight` | `(5376,)` | |
| `embed_tokens.weight` | `(262144, 5376)` | Shared with `lm_head.weight` |
| `norm.weight` | `(5376,)` | Final RMSNorm |

### MoE Expert Weights (26B-A4B: 128 experts, moe_inter=704, hidden=2816)

| HF Key Pattern | Shape | Notes |
|---|---|---|
| `layers.{i}.experts.gate_up_proj` | `(128, 1408, 2816)` | `2 * moe_intermediate_size` fused |
| `layers.{i}.experts.down_proj` | `(128, 2816, 704)` | Transposed per-expert |
| `layers.{i}.router.proj.weight` | `(128, 2816)` | |
| `layers.{i}.router.scale` | `(2816,)` | |
| `layers.{i}.router.per_expert_scale` | `(128,)` | |

### Vision Encoder (Large: hidden=1152, heads=16, head_dim=72)

| HF Key Pattern | Shape | Notes |
|---|---|---|
| `encoder.layers.{i}.self_attn.q_proj.linear.weight` | `(1152, 1152)` | |
| `encoder.layers.{i}.self_attn.k_proj.linear.weight` | `(1152, 1152)` | |
| `encoder.layers.{i}.self_attn.o_proj.linear.weight` | `(1152, 1152)` | |
| `encoder.layers.{i}.mlp.gate_proj.linear.weight` | `(4304, 1152)` | |
| `encoder.layers.{i}.mlp.down_proj.linear.weight` | `(1152, 4304)` | |
| `patch_embedder.position_embedding_table` | `(2, 10240, 1152)` | 2D sin/cos, transposed from `(10240, 2, 1152)` |
| `pooler.scale` | `(1, 1, 1152)` | Learnable output scale |

---

## 8.10 TTNN Porting Considerations

### Weight Layout and Transposition Awareness

The conversion script applies multiple transpositions to transform JAX's einsum-oriented weight layouts (e.g., `(num_heads, hidden, head_dim)` for Q) into PyTorch's `nn.Linear`-oriented layouts (e.g., `(num_heads * head_dim, hidden)`). When loading SafeTensors weights into TTNN:

- **All linear weights are already in `(out_features, in_features)` layout.** TTNN's `ttnn.linear` expects weights in this same layout, so no additional transposition is needed for standard matmuls.
- **MoE expert weights use fused `gate_up_proj` with shape `(E, 2*moe_inter, hidden)`.** TTNN must split or use strided access to separate gate and up projections.
- **MoE `down_proj` has shape `(E, hidden, moe_inter)`**, already transposed from JAX's `(E, moe_inter, hidden)`.

### Head Dimension Variability

The 31B variant uses different `head_dim` values for sliding vs. full attention layers:
- Sliding attention: `head_dim = config.head_dim` (64 for 31B)
- Full attention: `head_dim = config.global_head_dim` (256 for 31B when set, otherwise falls back to `head_dim`)

This means Q/K/V projection weight shapes vary per layer. TTNN device memory allocation must account for this non-uniform sizing when pre-allocating buffers.

### Tied Embeddings

`lm_head.weight` is a reference to `embed_tokens.weight`, not an independent tensor. When loading into TTNN, this weight should be loaded once and the same device tensor used for both the embedding lookup and the final linear projection.

### Per-Layer Embeddings

The E2B and E4B variants use `per_layer_embeddings` with shape `(vocab_size, num_layers * hidden_dim)`. This is reshaped at runtime in the modeling code via `.view(vocab_size, num_layers, hidden_dim)`. TTNN implementations should account for this reshape.

### ClippedLinear Parameters

Vision encoder weights for on-device variants carry `clip_min` and `clip_max` scalars alongside the weight matrices. These define activation clipping bounds for quantized inference. TTNN implementations targeting quantized execution should load these bounds and apply them as post-matmul clamping operations.

### Sliding Window Pattern

The 6-layer repeating pattern (`[sliding]*5 + [full]`) determines which layers use local (1024-token window for 31B) vs. global attention. This pattern affects:
- KV cache sizing per layer (sliding layers need much less cache)
- Head dimension selection (full attention layers may use larger `global_head_dim`)
- RoPE base frequency (10,000 for sliding, 1,000,000 for full)

### Memory Planning for Large Models

The `accelerate.init_empty_weights()` pattern used in the conversion script is instructive: it shows that even on the host side, the 31B model cannot be naively instantiated. For TTNN, weight loading should stream tensors to device one-at-a-time or in small batches rather than materializing the full state dict in host DRAM.

---

**End of guide.** Return to [Guide Index](../index.md)
