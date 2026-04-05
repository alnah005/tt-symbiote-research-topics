# Chapter 1: Package Overview and File Map

This chapter provides a complete inventory of every file in the `transformers/models/gemma4/` package, describes each file's role, maps the dependency relationships between them, and catalogs all 35 classes exported from the modeling file.

## 1.1 File Inventory

The Gemma 4 package consists of ten files. The table below lists each one with its purpose.

| File | Purpose |
|------|---------|
| [`__init__.py`](#__init__py) | Package entry point with lazy-loading via `_LazyModule` |
| [`configuration_gemma4.py`](#configuration_gemma4py) | Four config classes: `Gemma4AudioConfig`, `Gemma4TextConfig`, `Gemma4VisionConfig`, `Gemma4Config` |
| [`convert_gemma4_weights.py`](#convert_gemma4_weightspy) | CLI tool to convert Google Orbax checkpoints to HuggingFace safetensors format |
| [`feature_extraction_gemma4.py`](#feature_extraction_gemma4py) | `Gemma4AudioFeatureExtractor` -- mel spectrogram extraction for audio inputs |
| [`image_processing_gemma4.py`](#image_processing_gemma4py) | `Gemma4ImageProcessor` -- torchvision-based image preprocessing and patchification |
| [`image_processing_pil_gemma4.py`](#image_processing_pil_gemma4py) | `Gemma4ImageProcessorPil` -- PIL-based image preprocessing (CPU-only fallback) |
| [`modeling_gemma4.py`](#modeling_gemma4py) | All 35 model classes, auto-generated from the modular file (2564 lines) |
| [`modular_gemma4.py`](#modular_gemma4py) | Source-of-truth model definitions using cross-model inheritance (2160 lines) |
| [`processing_gemma4.py`](#processing_gemma4py) | `Gemma4Processor` -- top-level orchestrator that unifies text, image, audio, and video preprocessing |
| [`video_processing_gemma4.py`](#video_processing_gemma4py) | `Gemma4VideoProcessor` -- frame sampling and per-frame image processing for video inputs |

## 1.2 File Descriptions

### `__init__.py`

The package entry point. At runtime (when `TYPE_CHECKING` is `False`), it replaces `sys.modules[__name__]` with a `_LazyModule` instance. This means that none of the heavy modeling or processing modules are imported until a symbol from them is actually accessed. The `define_import_structure` utility scans all sibling Python files and builds a mapping from symbol names to their source modules automatically. At type-checking time (mypy, IDE autocompletion), the `if TYPE_CHECKING` branch runs unconditional star-imports so that all public symbols are visible to static analysis tools.

The lazy-loading mechanism works as follows:

1. `define_import_structure(__file__)` reads the `__all__` or public symbols from each submodule file in the package directory.
2. This produces a dictionary mapping `{"submodule_name": ["SymbolA", "SymbolB", ...]}`.
3. `_LazyModule.__getattr__` intercepts attribute access, finds which submodule owns the requested symbol, imports that submodule on demand, and caches the result.

This is critical for keeping `import transformers` fast -- the Gemma 4 modeling file alone pulls in PyTorch, torchvision, and numerous Transformers utilities, none of which are loaded until you actually instantiate a Gemma 4 class.

### `configuration_gemma4.py`

Defines the four configuration dataclasses that parameterize every component of the Gemma 4 model:

- **`Gemma4AudioConfig`** (`model_type = "gemma4_audio"`) -- Conformer-based audio encoder parameters: `Gemma4AudioConfig.hidden_size` (default 1024), `Gemma4AudioConfig.num_hidden_layers` (12), `Gemma4AudioConfig.num_attention_heads` (8), `Gemma4AudioConfig.attention_chunk_size` (12), `Gemma4AudioConfig.subsampling_conv_channels` ([128, 32]), `Gemma4AudioConfig.output_proj_dims` (1536).
- **`Gemma4TextConfig`** (`model_type = "gemma4_text"`) -- Text decoder parameters including sliding/global attention pattern, MoE configuration, and per-layer input embeddings: `Gemma4TextConfig.hidden_size` (2304), `Gemma4TextConfig.num_hidden_layers` (30), `Gemma4TextConfig.sliding_window` (512), `Gemma4TextConfig.enable_moe_block`, `Gemma4TextConfig.num_experts`, `Gemma4TextConfig.top_k_experts`, `Gemma4TextConfig.layer_types` (auto-generated 5:1 sliding-to-global pattern).
- **`Gemma4VisionConfig`** (`model_type = "gemma4_vision"`) -- SigLIP-derived vision encoder parameters: `Gemma4VisionConfig.hidden_size` (768), `Gemma4VisionConfig.num_hidden_layers` (16), `Gemma4VisionConfig.patch_size` (16), `Gemma4VisionConfig.pooling_kernel_size` (3).
- **`Gemma4Config`** (`model_type = "gemma4"`) -- Top-level composite config that nests the three sub-configs and adds special token IDs: `Gemma4Config.boi_token_id` (255999), `Gemma4Config.eoi_token_id` (258882), `Gemma4Config.image_token_id` (258880), `Gemma4Config.audio_token_id` (258881), `Gemma4Config.boa_token_id` (256000).

All four classes inherit from `PreTrainedConfig` and use the `@strict` decorator from `huggingface_hub` for validated field assignment.

### `convert_gemma4_weights.py`

A standalone CLI script that converts Google's internal Orbax (JAX-based) checkpoints into the HuggingFace safetensors format. It depends on `jax`, `orbax`, `accelerate`, and `numpy`. The script handles weight name remapping, shard consolidation, and writes the final model alongside tokenizer files, `GenerationConfig`, and processor configs. This file is not imported at runtime by any other module in the package -- it is a one-time-use conversion utility.

### `feature_extraction_gemma4.py`

Implements `Gemma4AudioFeatureExtractor`, which extends `SequenceFeatureExtractor`. It converts raw audio waveforms into 128-dimensional log-mel spectrograms using a configurable mel filter bank (default sampling rate: 16 kHz). The class includes a NumPy-based `_unfold` helper that replicates PyTorch's `Tensor.unfold` for windowed frame extraction without requiring a torch dependency at feature-extraction time.

### `image_processing_gemma4.py`

Implements `Gemma4ImageProcessor` using the `TorchvisionBackend`. This is the GPU-accelerated image processor. It imports the `get_aspect_ratio_preserving_size` function and `_SUPPORTED_SOFT_TOKENS` tuple from `image_processing_pil_gemma4.py` to share the aspect-ratio-preserving resize logic. It also provides `convert_image_to_patches`, which reshapes an image tensor of shape `[num_channels, image_height, image_width]` into a patch tensor of shape `[num_patches_height * num_patches_width, patch_size * patch_size * num_channels]`. The supported soft token counts come from the shared `_SUPPORTED_SOFT_TOKENS` tuple (defined in the PIL processor).

### `image_processing_pil_gemma4.py`

Implements `Gemma4ImageProcessorPil` using the `PilBackend`. This is the CPU-only PIL-based fallback image processor. It is the canonical home of two shared utilities:

- `get_aspect_ratio_preserving_size(height, width, patch_size, max_patches, pooling_kernel_size)` -- computes the target image dimensions that preserve aspect ratio while fitting within a patch budget, with both dimensions rounded down to the nearest multiple of `pooling_kernel_size * patch_size`.
- `_SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)` -- the fixed set of valid soft-token counts.

Both the torchvision image processor and the video processor import these from this file (directly or transitively).

### `modeling_gemma4.py`

The fully-resolved modeling file containing all 35 classes. This file is **auto-generated** from `modular_gemma4.py` by the HuggingFace modular expansion tool. The file opens with a prominent warning banner:

> *This file was automatically generated from `src/transformers/models/gemma4/modular_gemma4.py`. Do NOT edit this file manually as any edits will be overwritten by the generation of the file from the modular.*

Because all inherited code is expanded inline, this file has no cross-model imports (no references to `gemma3`, `gemma3n`, `llama`, or `mixtral`). It imports only from `configuration_gemma4.py` within the package. Every forward pass, every layer definition, and every utility function is self-contained in this single 2564-line file.

### `modular_gemma4.py`

The **source of truth** for the Gemma 4 model architecture. At 2160 lines, it is shorter than `modeling_gemma4.py` because it uses class inheritance from four other model families instead of duplicating code:

- **Gemma3** (`gemma3.modeling_gemma3`): `Gemma3Attention`, `Gemma3DecoderLayer`, `Gemma3MLP`, `Gemma3RotaryEmbedding`, `Gemma3TextModel`, `Gemma3TextScaledWordEmbedding`, `Gemma3ForCausalLM`
- **Gemma3n** (`gemma3n.modeling_gemma3n`): `Gemma3nModelOutputWithPast`, `Gemma3nCausalLMOutputWithPast`, `Gemma3nForConditionalGeneration`, `Gemma3nModel`, `Gemma3nMultimodalEmbedder`, `Gemma3nRMSNorm`, plus the helper functions `apply_rotary_pos_emb` and `eager_attention_forward`
- **Llama** (`llama.modeling_llama`): `LlamaRotaryEmbedding`
- **Mixtral** (`mixtral.modeling_mixtral`): `MixtralExperts`
- **Moonshine Streaming** (`moonshine_streaming.modeling_moonshine_streaming`): `sliding_window_mask_function`

A class that is `pass`-only (e.g., `Gemma4RMSNorm(Gemma3nRMSNorm): pass`) is identical to its parent; a class with overridden methods shows exactly where Gemma 4 diverges.

### `processing_gemma4.py`

Implements `Gemma4Processor`, which extends `ProcessorMixin` and orchestrates all preprocessing for the multimodal model. It accepts a `feature_extractor` (audio), `image_processor`, `tokenizer`, and `video_processor` in its constructor. Key parameters include `Gemma4Processor.image_seq_length` (default 280), `Gemma4Processor.audio_seq_length` (default 750), and `Gemma4Processor.audio_ms_per_token` (default 40 ms, derived from the SSCP convolution's 4x time reduction on 10 ms frames). The processor handles placeholder token expansion for images (`<start_of_image>...<end_of_image>`), audio (`<start_of_audio>...<end_of_audio>`), and video, then delegates to the appropriate sub-processors.

### `video_processing_gemma4.py`

Implements `Gemma4VideoProcessor`, extending `BaseVideoProcessor`. It handles frame sampling from video inputs and applies per-frame image processing using the same patchification and resize logic as the image processor. It imports `_SUPPORTED_SOFT_TOKENS` and `get_aspect_ratio_preserving_size` from `image_processing_gemma4.py` (which in turn imports them from the PIL image processor). It defines `Gemma4VideoProcessorKwargs` with parameters `patch_size`, `max_soft_tokens` (must be one of the shared `_SUPPORTED_SOFT_TOKENS` tuple defined in the PIL processor), and `pooling_kernel_size`.

## 1.3 Intra-Package Dependency Graph

The following diagram shows which files import from which within the `gemma4/` package. Arrows point from importer to importee.

```
configuration_gemma4.py          (no intra-package imports)
        ^
        |
        +---- modeling_gemma4.py
        |
        +---- modular_gemma4.py

image_processing_pil_gemma4.py   (no intra-package imports)
        ^
        |
        +---- image_processing_gemma4.py
        |           ^
        |           |
        |           +---- video_processing_gemma4.py
        |
        +---- processing_gemma4.py

feature_extraction_gemma4.py     (no intra-package imports)

convert_gemma4_weights.py        (no intra-package imports; uses top-level `transformers` imports)
```

## 1.4 The Modular System

HuggingFace Transformers uses a "modular" code-generation pattern for model families that share significant architecture. The relationship between the two modeling files is:

| Aspect | `modular_gemma4.py` | `modeling_gemma4.py` |
|--------|---------------------|----------------------|
| **Role** | Source of truth, human-authored | Auto-generated artifact |
| **Inheritance** | Uses cross-model class inheritance | All inheritance expanded inline |
| **Cross-model imports** | Imports from `gemma3`, `gemma3n`, `llama`, `mixtral`, `moonshine_streaming` | None -- fully self-contained |
| **Line count** | 2160 | 2564 |
| **When to read** | Understanding *what changed* vs. parent architectures | Understanding *exact runtime behavior* of forward passes |
| **Editable?** | Yes -- this is where developers make changes | No -- CI regenerates it from the modular file |

The modular file defines 35 classes. Classes that inherit without modification (e.g., `class Gemma4RMSNorm(Gemma3nRMSNorm): pass`) are expanded by the code generator into standalone classes with all parent methods copied in. Classes that override specific methods show only the overridden methods in the modular file, but the generated file contains both the overridden and inherited methods merged into a single class body.

### Complete Inheritance Tree (from `modular_gemma4.py`)

```
Gemma3nModelOutputWithPast
  └── Gemma4ModelOutputWithPast (pass)

Gemma3nCausalLMOutputWithPast
  └── Gemma4CausalLMOutputWithPast (pass)

BaseModelOutputWithPooling
  └── Gemma4AudioModelOutput (adds attention_mask field)

nn.Module
  ├── Gemma4ClippableLinear (new, no parent model)
  ├── Gemma4AudioRelPositionalEncoding (new)
  ├── Gemma4AudioAttention (new)
  ├── Gemma4AudioSubSampleConvProjectionLayer (new)
  ├── Gemma4AudioSubSampleConvProjection (new)
  ├── Gemma4AudioFeedForward (new)
  ├── Gemma4AudioLayer (new)
  ├── Gemma4VisionPatchEmbedder (new)
  ├── Gemma4VisionPooler (new)
  ├── Gemma4VisionEncoder (new)
  ├── Gemma4TextAttention (new)
  ├── Gemma4AudioLightConv1d (new)
  └── Gemma4TextRouter (new)

nn.Conv1d
  └── Gemma4AudioCausalConv1d (new)

Gemma3nRMSNorm
  └── Gemma4RMSNorm (pass)

Gemma3MLP
  ├── Gemma4VisionMLP (overrides config handling)
  └── Gemma4TextMLP (overrides config handling)

LlamaRotaryEmbedding
  └── Gemma4VisionRotaryEmbedding (overrides rope_init_fn)

Gemma3RotaryEmbedding
  └── Gemma4TextRotaryEmbedding (overrides rope_init_fn)

Gemma3Attention
  └── Gemma4VisionAttention (overrides for vision-specific projections)

Gemma3DecoderLayer
  ├── Gemma4VisionEncoderLayer (adapts for vision encoder)
  └── Gemma4TextDecoderLayer (adds MoE routing, KV sharing)

MixtralExperts
  └── Gemma4TextExperts (overrides for Gemma 4 MoE config)

Gemma3TextScaledWordEmbedding
  └── Gemma4TextScaledWordEmbedding (pass)

Gemma3TextModel
  └── Gemma4TextModel (adds per-layer input embeddings, KV sharing)

Gemma3ForCausalLM
  └── Gemma4ForCausalLM (pass)

Gemma3nMultimodalEmbedder
  └── Gemma4MultimodalEmbedder (adds audio embedding path)

Gemma3nModel
  └── Gemma4Model (adds audio tower, video support)

Gemma3nForConditionalGeneration
  └── Gemma4ForConditionalGeneration (adds audio/video generation paths)

PreTrainedModel
  └── Gemma4PreTrainedModel (abstract base for all Gemma 4 models)
        ├── Gemma4AudioModel (full audio encoder)
        └── Gemma4VisionModel (full vision encoder)
```

## 1.5 Complete Class Catalog (from `modeling_gemma4.py`)

All 35 classes exported from the auto-generated modeling file, grouped by subsystem.

### Output Dataclasses

| Class | Base | Description |
|-------|------|-------------|
| `Gemma4ModelOutputWithPast` | `BaseModelOutputWithPast` | Output for `Gemma4Model.forward()`. Adds `image_hidden_states` and `audio_hidden_states` fields. |
| `Gemma4CausalLMOutputWithPast` | `ModelOutput` | Output for `Gemma4ForConditionalGeneration.forward()`. Adds `loss`, `logits`, `past_key_values`, `image_hidden_states`, `audio_hidden_states`. |
| `Gemma4AudioModelOutput` | `BaseModelOutputWithPooling` | Output for the audio encoder. Adds an `attention_mask` field of shape `[batch_size, num_frames]`. |

### Utility Layers

| Class | Base | Description |
|-------|------|-------------|
| `Gemma4ClippableLinear` | `nn.Module` | Linear layer with optional input/output clamping via registered buffers. Used in both audio and vision encoders when `use_clipped_linears=True`. |
| `Gemma4RMSNorm` | `nn.Module` | RMS normalization. Identical to Gemma3n's implementation. |

### Audio Encoder (9 classes)

| Class | Base | Description |
|-------|------|-------------|
| `Gemma4AudioRelPositionalEncoding` | `nn.Module` | Sinusoidal relative positional encoding producing `[1, 2*context_size-1, hidden_size]` embeddings with `[sin, cos]` layout. |
| `Gemma4AudioAttention` | `nn.Module` | Chunked local attention with relative position bias, logit soft-capping at `Gemma4AudioConfig.attention_logit_cap`, and per-dim learned scaling. |
| `Gemma4AudioSubSampleConvProjectionLayer` | `nn.Module` | Single convolution-norm-ReLU block: `Conv2d(kernel=3x3, stride=2x2)` followed by `LayerNorm`. |
| `Gemma4AudioSubSampleConvProjection` | `nn.Module` | Two-layer sub-sample convolution projection (SSCP) that reduces the mel spectrogram by 4x in the time dimension, then projects to `Gemma4AudioConfig.hidden_size`. |
| `Gemma4AudioFeedForward` | `nn.Module` | Conformer-style feed-forward with residual scaling by `Gemma4AudioConfig.residual_weight` (default 0.5). |
| `Gemma4AudioCausalConv1d` | `nn.Conv1d` | Causal 1D convolution with left-padding to prevent future information leakage. |
| `Gemma4AudioLightConv1d` | `nn.Module` | Lightweight 1D convolution block: LayerNorm, pointwise expansion, GLU gating, depthwise causal conv, pointwise contraction. |
| `Gemma4AudioLayer` | `nn.Module` | Single conformer layer combining: feed-forward, self-attention with relative positional encoding, light convolution, feed-forward. |
| `Gemma4AudioModel` | `Gemma4PreTrainedModel` | Full audio encoder: SSCP front-end, stack of `Gemma4AudioLayer` conformer layers, output projection to `Gemma4AudioConfig.output_proj_dims`. |

### Vision Encoder (8 classes)

| Class | Base | Description |
|-------|------|-------------|
| `Gemma4VisionPatchEmbedder` | `nn.Module` | Patch embedding via `Conv2d(kernel=patch_size, stride=patch_size)` plus learned 2D position embeddings of size `Gemma4VisionConfig.position_embedding_size`. |
| `Gemma4VisionPooler` | `nn.Module` | Spatial average pooling with kernel `Gemma4VisionConfig.pooling_kernel_size` that reduces patch count, with optional standardization (bias + scale). |
| `Gemma4VisionMLP` | `nn.Module` | Gated MLP: `gate_proj` and `up_proj` with `gelu_pytorch_tanh` activation, followed by `down_proj`. |
| `Gemma4VisionRotaryEmbedding` | `nn.Module` | 2D rotary position embeddings for vision patches, using `rope_theta=100.0` by default. |
| `Gemma4VisionAttention` | `nn.Module` | Multi-head attention with Q/K normalization, RoPE, and support for all standard attention backends (eager, SDPA, Flash Attention). |
| `Gemma4VisionEncoderLayer` | `GradientCheckpointingLayer` | Single vision encoder layer: self-attention + MLP with pre-norm (RMSNorm). |
| `Gemma4VisionEncoder` | `nn.Module` | Stack of `Gemma4VisionEncoderLayer` layers. |
| `Gemma4VisionModel` | `Gemma4PreTrainedModel` | Full vision encoder: patch embedder, encoder stack, pooler. |

### Text Decoder (9 classes)

| Class | Base | Description |
|-------|------|-------------|
| `Gemma4TextMLP` | `nn.Module` | Text decoder gated MLP, same architecture as vision MLP but parameterized by `Gemma4TextConfig`. Supports double-wide MLP mode via `Gemma4TextConfig.use_double_wide_mlp`. |
| `Gemma4TextRotaryEmbedding` | `nn.Module` | RoPE for text, with separate configs for sliding attention layers (`rope_theta=10000`) and global attention layers (`rope_theta=1000000`, `partial_rotary_factor=0.25`). |
| `Gemma4TextAttention` | `nn.Module` | Text attention with Q/K normalization, sliding window support, optional K=V weight sharing (`Gemma4TextConfig.attention_k_eq_v`), and KV sharing across layers (`Gemma4TextConfig.num_kv_shared_layers`). |
| `Gemma4TextExperts` | `nn.Module` | Mixture-of-Experts feed-forward layer. Batched expert computation derived from Mixtral. |
| `Gemma4TextRouter` | `nn.Module` | Top-k expert routing with `Gemma4TextConfig.top_k_experts` selection and softmax-normalized gating weights. |
| `Gemma4TextDecoderLayer` | `GradientCheckpointingLayer` | Single text decoder layer: self-attention + (MLP or MoE router+experts), with sliding or global attention based on `Gemma4TextConfig.layer_types`. |
| `Gemma4TextScaledWordEmbedding` | `nn.Embedding` | Word embedding with a fixed scaling factor of `hidden_size ** 0.5` applied to the output. |
| `Gemma4TextModel` | `Gemma4PreTrainedModel` | Full text decoder: embedding layer, stack of `Gemma4TextDecoderLayer` layers, final RMSNorm. Includes per-layer input embeddings (`vocab_size_per_layer_input` x `hidden_size_per_layer_input`). |
| `Gemma4ForCausalLM` | `Gemma4PreTrainedModel` | Causal LM head on top of `Gemma4TextModel`, with `GenerationMixin` for `.generate()` support. |

### Top-Level Multimodal (4 classes)

| Class | Base | Description |
|-------|------|-------------|
| `Gemma4PreTrainedModel` | `PreTrainedModel` | Abstract base providing weight initialization and config validation for all Gemma 4 models. |
| `Gemma4MultimodalEmbedder` | `nn.Module` | Merges text embeddings with vision and audio soft tokens by replacing placeholder token positions with encoder outputs. |
| `Gemma4Model` | `Gemma4PreTrainedModel` | Composite model: vision tower (`Gemma4VisionModel`), audio tower (`Gemma4AudioModel`), multimodal embedder (`Gemma4MultimodalEmbedder`), and language model (`Gemma4TextModel`). |
| `Gemma4ForConditionalGeneration` | `Gemma4PreTrainedModel` | End-to-end model for multimodal conditional generation. Wraps `Gemma4Model` and adds a causal LM head, with `GenerationMixin` for `.generate()` support. This is the class most users instantiate. |

## 1.6 TTNN Porting Considerations

When planning a TTNN port of Gemma 4, the file map above informs several architectural decisions:

1. **Use `modeling_gemma4.py` for porting** (see the "When to read" guidance in Section 1.4 for rationale).

2. **Configuration drives architecture variation.** The `Gemma4TextConfig.layer_types` list determines which layers use sliding attention vs. global attention, `Gemma4TextConfig.enable_moe_block` toggles MoE, and `Gemma4TextConfig.num_kv_shared_layers` controls KV cache sharing. A TTNN implementation must read these config values at graph construction time to select the correct op graph for each layer.

3. **Three independent encoder subsystems.** The audio encoder (conformer-based), vision encoder (SigLIP-derived transformer), and text decoder (Gemma-family LLM with MoE) have no weight sharing and minimal architectural overlap. They can be ported and validated independently. The only coupling point is `Gemma4MultimodalEmbedder`, which performs soft-token insertion.

4. **Preprocessing stays on host.** The four preprocessing files (`feature_extraction_gemma4.py`, `image_processing_gemma4.py`, `image_processing_pil_gemma4.py`, `video_processing_gemma4.py`) and the orchestrator (`processing_gemma4.py`) run on CPU/GPU host before tensors reach the device. These do not need TTNN equivalents -- their outputs (token IDs, pixel values, mel spectrograms, attention masks) become the inputs to the TTNN model graph.

5. **`Gemma4ClippableLinear` requires special handling.** This layer uses registered buffers (`input_min`, `input_max`, `output_min`, `output_max`) loaded from the checkpoint to clamp activations. A TTNN port must either fuse these clamps into the matmul kernel or add explicit clamp ops before and after the linear operation.

6. **The 35-class catalog maps to your TTNN module hierarchy.** Each class in Section 1.5 corresponds to a unit that can be individually implemented and tested in TTNN. The grouping by subsystem (audio, vision, text, top-level) provides a natural work breakdown structure.

---

**Next:** [Chapter 2 -- Configuration Hierarchy](../ch2_configuration_hierarchy/index.md)
