# Plan: TT-DiT Framework Architecture and TT-Symbiote Porting Strategy

## Audience

This guide targets Tenstorrent engineers who:

- Are familiar with PyTorch and the fundamentals of running neural networks on TT hardware (Wormhole, MeshDevice, TTNN ops).
- Have working knowledge of **TT-Symbiote** (`models/experimental/tt_symbiote/`): its `TTNNModule` base class, dispatcher system, `from_torch` conversion pattern, `preprocess_weights_impl` / `move_weights_to_device_impl` lifecycle, `DistributedConfig`, and `TracedRun` infrastructure.
- Understand diffusion transformer (DiT) models at a conceptual level (denoising loop, text conditioning, VAE decoding) but have **not** read the TT-DiT source code.
- Want to evaluate whether and how TT-DiT's generative-model components can be ported into TT-Symbiote's unified serving infrastructure.

Prior reading: familiarity with the HuggingFace `diffusers` library and the Megatron-style parallelism concepts (tensor parallel, sequence parallel) is helpful but not required; both are explained from first principles where relevant.

---

## Chapter List

### Chapter 1: TT-DiT Architecture Overview

**Description:** Introduces the TT-DiT codebase layout, its custom `Module`/`Parameter` base classes, and the overall model lifecycle from weight loading through inference.

**Directory:** `ch1_architecture_overview`

**Files:**

- `index.md`
  - High-level map of the TT-DiT directory tree: `layers/`, `blocks/`, `models/`, `encoders/`, `pipelines/`, `parallel/`, `utils/`.
  - List of the six supported models (SD3.5, Flux1, Motif, Mochi, Wan2.2, Qwen-Image) and their pipeline entry points.
  - Relationship between layers, blocks, model transformers, and pipeline orchestration.

- `module_and_parameter.md`
  - Deep dive into `layers/module.py`: the `Module` abstract base class, `Parameter` class, `ModuleList`, and `UnregisteredModule`.
  - The `Module` lifecycle: `__init__` -> `load_torch_state_dict` -> `_prepare_torch_state` -> `forward` -> `deallocate_weights`.
  - How `Parameter` handles `total_shape`, `mesh_axes`, `local_shape` computation, dtype/layout/memory_config enforcement, and `load_torch_tensor` with `from_torch`.
  - The `save`/`load` serialization via `ttnn.dump_tensor` / `ttnn.load_tensor` in `.tensorbin` format.
  - The `set_unload_set` mechanism for swapping components in/out of device memory (used when encoders, transformer, and VAE share a device).

- `comparison_with_ttnnmodule.md`
  - Side-by-side comparison of TT-DiT `Module` vs. TT-Symbiote `TTNNModule`.
  - Key architectural differences:
    - TT-DiT `Module` is standalone (no PyTorch `nn.Module` inheritance); `TTNNModule` wraps/replaces `nn.Module` via `from_torch` and holds `_fallback_torch_layer`.
    - TT-DiT `Parameter` does conversion + placement in one step during `load_torch_state_dict`; TT-Symbiote separates `preprocess_weights_impl` (host conversion) from `move_weights_to_device_impl` (device placement).
    - TT-DiT's `_prepare_torch_state` handles weight reshaping/transposition at the module level; TT-Symbiote does this inside `preprocess_weights_impl`.
    - TT-DiT does not use a dispatcher; it calls TTNN ops directly. TT-Symbiote intercepts `torch.__dispatch__` to route ops to TTNN.
  - Table of equivalent patterns for common operations.

---

### Chapter 2: Parallelism and CCL Infrastructure

**Description:** Explains TT-DiT's 3-axis parallelism model (CFG parallel, sequence parallel, tensor parallel), the `CCLManager`, and how they compare to TT-Symbiote's `DistributedConfig`/`DistributedTensorConfig`.

**Directory:** `ch2_parallelism_and_ccl`

**Files:**

- `index.md`
  - Overview of the three parallelism axes: CFG parallel (batch duplication for classifier-free guidance), sequence parallel (spatial sequence sharding), tensor parallel (weight column/row sharding).
  - How `DiTParallelConfig` (`parallel/config.py`) composes `ParallelFactor` tuples (factor + mesh_axis) to configure each axis.
  - How pipeline code creates submeshes for CFG parallel and instantiates one `CCLManager` per submesh.

- `ccl_manager.md`
  - Detailed walkthrough of `CCLManager` (`parallel/manager.py`):
    - SubDevice setup and CCL core allocation.
    - Semaphore initialization: reduce-scatter, all-gather, neighbor-pad, slice-reshard, barrier -- all with ping-pong indexing per mesh axis.
    - Ping-pong buffer caching for persistent all-gather and reduce-scatter operations.
    - The `all_gather`, `reduce_scatter`, `all_gather_persistent_buffer`, `device_to_host` helper methods.
    - Hyperparameter tuning (`get_ag_hyperparams`, `get_rs_hyperparams`) based on tensor shape.
  - VAE-specific CCL operations in `parallel/config.py`: `vae_all_gather`, `vae_neighbor_pad`, `vae_slice_reshard`.

- `parallel_linear_layers.md`
  - How `ColParallelLinear` and `RowParallelLinear` implement Megatron-style parallelism.
  - `ColParallelLinear`: column-shards weights via `mesh_axes=[fsdp_mesh_axis, mesh_axis]`, expects replicated input, returns column-fractured output. Optional FSDP weight gathering via `all_gather_persistent_buffer`.
  - `RowParallelLinear`: row-shards weights via `mesh_axes=[mesh_axis, fsdp_mesh_axis]`, expects column-fractured input, returns reduced output via `reduce_scatter`.
  - The `_prepare_torch_state` weight reshaping for swiglu and chunked linear outputs.
  - `ttnn.experimental.minimal_matmul` and `get_matmul_config` for shape-specific blocking configurations.

- `mapping_to_symbiote.md`
  - How TT-Symbiote's `DistributedConfig` and `DistributedTensorConfig` compare: `ShardTensor2dMesh` / `ConcatMesh2dToTensor` vs. TT-DiT's per-parameter `mesh_axes`.
  - TT-Symbiote's distributed linear variants (`TTNNLinearIColShardedWRowSharded`, `TTNNLinearIReplicatedWColSharded`, `TTNNLinearIColShardedWAllReduced`) vs. TT-DiT's `ColParallelLinear`/`RowParallelLinear`.
  - Gaps: TT-Symbiote currently lacks a CCLManager-equivalent with semaphore management, ping-pong buffers, and persistent async CCL ops. Its distributed operations use simpler `ttnn.reduce_scatter`/`ttnn.all_gather` calls without ping-pong.
  - Recommendations for extending TT-Symbiote's CCL infrastructure.

---

### Chapter 3: Custom Layers and TTNN Operations

**Description:** Catalogs TT-DiT's custom layer implementations and the specific TTNN operations they depend on, identifying which have TT-Symbiote equivalents.

**Directory:** `ch3_custom_layers_and_ops`

**Files:**

- `index.md`
  - Classification of TT-DiT layers: normalization (`RMSNorm`, `LayerNorm`, `DistributedRMSNorm`, `DistributedLayerNorm`, `GroupNorm`), linear (`Linear`, `ColParallelLinear`, `RowParallelLinear`), feedforward (`FeedForward`, `ParallelFeedForward`), convolution (`Conv2d`, `Conv3d`), and embeddings (`Timesteps`, `TimestepEmbedding`, `PatchEmbed`, various combined embeddings).

- `normalization_layers.md`
  - `RMSNorm`: wraps `ttnn.rms_norm` with `_prepare_torch_state` for unsqueeze.
  - `LayerNorm`: wraps `ttnn.layer_norm` with optional row-major workaround.
  - `DistributedRMSNorm`: two-phase distributed RMSNorm using `ttnn.experimental.wan_fused_rmsnorm_pre_allgather` + all-gather + `ttnn.experimental.wan_fused_rmsnorm_post_allgather`. Supports fused RoPE and transformation matrix.
  - `DistributedLayerNorm`: uses `ttnn.experimental.dit_layernorm_pre_allgather` + all-gather + `ttnn.experimental.dit_layernorm_post_allgather`. Has a shared reciprocal tensor cache (`_recip_tensors`), dynamic weight/bias for modulation, and ROW_MAJOR_LAYOUT weights with interleaved tile-size reshaping.
  - `GroupNorm`: wraps `ttnn.group_norm` with data-parallel support and `ttnn.create_group_norm_input_mask` / `ttnn.create_group_norm_weight_bias_rm`.
  - Comparison with TT-Symbiote: `TTNNLayerNorm` (simple `ttnn.layer_norm`), `TTNNRMSNorm` (simple `ttnn.rms_norm`), `TTNNDistributedRMSNorm` (uses `ttnn.rms_norm_pre_all_gather` + `ttnn.all_gather` + `ttnn.rms_norm_post_all_gather`).

- `ttnn_experimental_ops.md`
  - Catalog of `ttnn.experimental.*` operations used by TT-DiT that are NOT used in TT-Symbiote:
    - `ttnn.experimental.minimal_matmul` and `ttnn.experimental.minimal_matmul_split` with `MinimalMatmulConfig`.
    - `ttnn.experimental.dit_layernorm_pre_allgather` and `ttnn.experimental.dit_layernorm_post_allgather`.
    - `ttnn.experimental.wan_fused_rmsnorm_pre_allgather` and `ttnn.experimental.wan_fused_rmsnorm_post_allgather` (with fused RoPE).
    - `ttnn.experimental.all_gather_async` and `ttnn.experimental.reduce_scatter_minimal_async` (async CCL with semaphores and persistent buffers).
    - `ttnn.experimental.neighbor_pad_async` and `ttnn.experimental.slice_reshard_async`.
    - `ttnn.create_layer_norm_reciprocals`.
  - For each op: purpose, required parameters, and whether TT-Symbiote has an equivalent or would need to adopt it.

- `convolution_layers.md`
  - `Conv2d` in TT-DiT: uses `ttnn.conv2d` with `WormholeComputeKernelConfig`, data-parallel support via `vae_all_gather`, and shape-specific slice parameters. Has `from_torch` class method.
  - `Conv3d` in TT-DiT: implements 3D convolution for video models (Mochi, Wan2.2) using temporal decomposition.
  - TT-Symbiote's `TTNNConv2d`: uses `TtConv2d` from `models/tt_cnn/tt/builder.py` with a different configuration API.
  - Porting considerations: Conv3d has no TT-Symbiote equivalent; Conv2d implementations diverge in their parallelization strategy.

---

### Chapter 4: Joint Attention and Transformer Blocks

**Description:** Explains TT-DiT's joint attention mechanism (the core innovation for DiT models), the transformer block structure, and how these compare to TT-Symbiote's attention modules.

**Directory:** `ch4_attention_and_transformer_blocks`

**Files:**

- `index.md`
  - Overview of how DiT attention differs from standard LLM attention: joint spatial+prompt attention with separate QKV projections, RMSNorm per head, RoPE on spatial and prompt separately, and joint SDPA.

- `joint_attention.md`
  - Walkthrough of `blocks/attention.py` `Attention` class:
    - Fused `to_qkv` projection using `ColParallelLinear` for spatial tokens, separate `add_qkv_proj` for prompt/context tokens.
    - Per-head `norm_q` / `norm_k` using `RMSNorm`.
    - Head padding via `PaddingConfig` for tile alignment.
    - `_prepare_torch_state` that merges separate Q/K/V PyTorch weights into fused QKV format with interleaved head sharding for tensor parallelism.
    - `_apply_rope` using `ttnn.alt_complex_rotate90`.
    - Two execution paths: `ttnn.transformer.ring_joint_scaled_dot_product_attention` (sequence parallel, uses CCL persistent buffers and semaphores) vs. `ttnn.transformer.joint_scaled_dot_product_attention` (single device / no SP).
    - Post-attention `to_out` / `to_add_out` projections with all-gather for TP reduction.
    - `UnregisteredModule` pattern for models that share spatial/prompt weights.
    - `context_head_factors` for context head scaling.

- `transformer_block.md`
  - Walkthrough of `blocks/transformer_block.py` `TransformerBlock` class:
    - Adaptive layer normalization with time-conditioned modulation: `norm1_linear` produces 6 chunks (shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff) via `ColParallelLinear`.
    - `DistributedLayerNorm` with dynamic `(1 + scale)` weight and `shift` bias.
    - Attention with gating: `spatial = spatial + attn_output * gate_attn`.
    - FeedForward with gating: `spatial = spatial + ff_output * gate_ff`.
    - Context (prompt) branch with optional `context_pre_only` for the final block.
  - Model-specific variants: `SD35TransformerBlock`, `WanTransformerBlock` (cross-attention added), `MochiAttention`, `QwenImageTransformer`.

- `comparison_with_symbiote_attention.md`
  - TT-Symbiote's `TTNNAttention` class: designed for LLM causal attention with `TTNNPagedAttentionKVCache`, `past_key_value` support, and causal masking.
  - Key differences from DiT attention:
    - No joint attention (spatial + prompt) -- DiT concatenates both token types before SDPA.
    - No per-head QKV normalization in TT-Symbiote.
    - TT-Symbiote uses `ttnn.transformer.scaled_dot_product_attention` (standard SDPA), not `joint_scaled_dot_product_attention` or `ring_joint_scaled_dot_product_attention`.
    - TT-Symbiote has no adaptive modulation (time-conditioned gating).
    - TT-Symbiote has paged attention / KV cache for autoregressive decoding; DiT has no KV cache (full-context attention at every denoising step).
  - Assessment of what can be reused vs. must be built new.

---

### Chapter 5: End-to-End Pipelines and Model Registration

**Description:** Explains how TT-DiT pipelines orchestrate the full inference flow (text encoding, denoising loop, VAE decoding) and how this would map to TT-Symbiote's model serving infrastructure.

**Directory:** `ch5_pipelines_and_serving`

**Files:**

- `index.md`
  - Overview of the six TT-DiT pipeline classes: `StableDiffusion3Pipeline`, `Flux1Pipeline`, `MotifPipeline`, `MochiPipeline`, `WanPipeline` (text-to-video), `QwenImagePipeline`.
  - Common pipeline lifecycle: mesh setup -> submesh creation -> CCLManager per submesh -> encoder loading -> transformer loading -> VAE loading -> denoising loop -> decode -> output.

- `pipeline_anatomy.md`
  - Detailed walkthrough of `StableDiffusion3Pipeline` as the canonical example:
    - Constructor: creates submeshes for CFG parallel, instantiates CCLManagers, configures encoder/VAE devices with potential submesh reshaping.
    - `PipelineTrace` dataclass for traced execution: holds input/output tensor handles and trace ID.
    - Text encoding: CLIP + optional T5 encoders with shared device, `unload_set` for memory management.
    - Denoising loop: scheduler step -> trace capture on first/second iteration -> trace replay on subsequent iterations.
    - VAE decoding: decoder loaded on a separate submesh, optional weight caching.
  - Memory management: the `set_unload_set` mechanism for swapping encoder/transformer/VAE on shared devices.
  - Weight caching: the `utils/cache.py` system with `config_id` for parallel-config-specific caches.

- `mapping_to_symbiote_serving.md`
  - TT-Symbiote's current model registration pattern (from test files): `module_replacement.register_module_replacement_dict_with_module_names` to replace PyTorch modules in-place within a HuggingFace model.
  - How a DiT pipeline would differ: pipeline-level orchestration (multi-component: encoder + transformer + VAE) rather than single-model acceleration.
  - Potential integration strategies:
    1. Port DiT components as TTNNModules and use TT-Symbiote's module replacement on a `diffusers` pipeline.
    2. Create a pipeline-level TTNNModule that encapsulates the full inference flow.
    3. Keep TT-DiT's pipeline orchestration but swap out leaf layers for TT-Symbiote modules.
  - Trade-offs of each approach.

---

### Chapter 6: Weight Loading and Preprocessing

**Description:** Compares TT-DiT's weight loading pipeline with TT-Symbiote's `from_torch -> preprocess_weights_impl` pattern and identifies what can be reused.

**Directory:** `ch6_weight_loading`

**Files:**

- `index.md`
  - Overview of the two weight loading paradigms.

- `tt_dit_weight_pipeline.md`
  - TT-DiT flow: HuggingFace `state_dict` -> `module.load_torch_state_dict(state_dict)` -> recursive `_load_torch_state_dict_inner` -> per-module `_prepare_torch_state` (reshape, transpose, merge QKV, pad heads, chunk for parallelism) -> `parameter.load_torch_tensor` (calls `tensor.from_torch` with `mesh_axes` for sharding).
  - The `_prepare_torch_state` pattern: each module customizes weight transformation in-place on the state dict. Examples: `Linear` transposes weights; `Attention` merges Q/K/V into fused QKV with per-device head interleaving; `TransformerBlock` renames substates and calls `prepare_chunked_linear_output`.
  - Serialization to/from `.tensorbin` files via `save`/`load` for weight caching.
  - The `utils/tensor.py` helper functions: `from_torch` with `mesh_axes` for 2D mesh sharding, `typed_tensor`, `bf16_tensor`, `unflatten`.

- `symbiote_weight_pipeline.md`
  - TT-Symbiote flow: `TTNNModule.from_torch(torch_layer)` -> stores `_fallback_torch_layer` -> `preprocess_weights_impl` (converts weights to host TTNN tensors using `preprocess_linear_weight`/`preprocess_linear_bias`) -> `move_weights_to_device_impl` (calls `ttnn.to_device`).
  - Key differences: TT-Symbiote stores the PyTorch layer reference and converts lazily; TT-DiT converts eagerly during `load_torch_state_dict`. TT-Symbiote uses `ttnn.model_preprocessing` utilities; TT-DiT uses raw `ttnn.from_torch` with `mesh_axes`.
  - Assessment: TT-DiT's `_prepare_torch_state` transformations (QKV merging, head padding, chunked linear reshaping) are model-specific and would need to be replicated in TT-Symbiote's `preprocess_weights_impl` or handled by a conversion layer.

---

### Chapter 7: Tracing and Performance

**Description:** Compares TT-DiT's tracing infrastructure with TT-Symbiote's `TracedRun`, and explains how traced DiT execution would integrate with TT-Symbiote.

**Directory:** `ch7_tracing_and_performance`

**Files:**

- `index.md`
  - Why tracing matters for DiT: the denoising loop repeats the same transformer forward pass 20-50 times with different inputs, making trace replay a massive performance win.

- `tt_dit_tracer.md`
  - TT-DiT's `Tracer` class (`utils/tracing.py`): wraps a function, captures on first call (compile + trace capture), replays on subsequent calls.
  - Two-phase first call: compile run -> `begin_trace_capture` -> capture run -> `end_trace_capture`.
  - Input update via `_update_input` with strict shape/dtype/layout checking; supports `copy_host_to_device_tensor` for host tensors.
  - `_tree_map` utility for nested structure traversal (tuples, lists, dicts).
  - Pipeline integration: `PipelineTrace` dataclass stores tensor handles for spatial, prompt, timestep inputs and latent output.

- `symbiote_traced_run.md`
  - TT-Symbiote's `TracedRun` (`core/run_config.py`): operates at the module level (not function level).
  - Three-phase lifecycle: warm-up (run 1, normal forward) -> capture (run 2, trace capture with persistent buffers) -> replay (run 3+, `execute_trace`).
  - The `@trace_enabled` / `@trace_disabled` decorators for per-class opt-in/opt-out.
  - `TTNNLayerStack`: groups multiple layers into a single trace-enabled unit.
  - Input buffer management: `_copy_inputs_to_trace_buffer`, `_copy_kwargs_to_trace_buffer`, `_capture_trace`.
  - `pre_trace_execute` / `post_trace_execute` hooks for custom buffer management.

- `integration_strategy.md`
  - Comparison: TT-DiT traces at the pipeline level (entire transformer forward), TT-Symbiote traces at the module level (individual layers or layer stacks).
  - For DiT porting, the recommended approach: use `TTNNLayerStack` to wrap the full transformer block sequence, or create a custom `@trace_enabled` module that encompasses the complete denoising step.
  - Performance considerations: DiT's ring_joint_SDPA requires semaphore state management across traces; TT-Symbiote's generic trace capture may need CCL-aware extensions.

---

### Chapter 8: Porting Strategy and Model Prioritization

**Description:** Synthesizes all findings into a concrete porting roadmap: which components to reuse, which to reimplement, and which model to port first.

**Directory:** `ch8_porting_strategy`

**Files:**

- `index.md`
  - Summary of the porting challenge: TT-DiT is a purpose-built, performance-optimized framework with tight coupling between parallelism, attention, and CCL infrastructure. TT-Symbiote is a general-purpose acceleration framework with automatic dispatch.

- `component_assessment.md`
  - Three-tier classification of TT-DiT components:
    - **Directly reusable in TT-Symbiote** (no changes needed):
      - `RMSNorm`, `LayerNorm` (simple normalization wrappers).
      - `Timesteps`, `TimestepEmbedding` (sinusoidal embeddings).
      - Activation functions (`silu`, `gelu`, `swiglu`).
      - RoPE application logic (`_apply_rope` using `ttnn.alt_complex_rotate90`).
    - **Reimplementable as TTNNModule subclasses** (need adaptation):
      - `Linear` -> can map to `TTNNLinear` with weight transpose in `preprocess_weights_impl`.
      - `ColParallelLinear` / `RowParallelLinear` -> can extend `TTNNLinearIColShardedWRowSharded` / `TTNNLinearIReplicatedWColSharded` with FSDP support and `minimal_matmul`.
      - `FeedForward` / `ParallelFeedForward` -> compose from TTNNLinear variants with activation.
      - `GroupNorm` -> needs `ttnn.group_norm` wrapper with data-parallel support.
      - `Conv2d` -> can extend TT-Symbiote's `TTNNConv2d`.
    - **Require new TT-Symbiote infrastructure**:
      - `CCLManager` with semaphore management and persistent buffer caching.
      - `DistributedLayerNorm` and `DistributedRMSNorm` with `dit_layernorm_pre/post_allgather` ops.
      - Joint attention (`joint_scaled_dot_product_attention`, `ring_joint_scaled_dot_product_attention`).
      - `DiTParallelConfig` with 3-axis parallelism.
      - `Conv3d` for video models.
      - Pipeline-level orchestration (multi-component: encoder + transformer + VAE with memory swapping).

- `model_prioritization.md`
  - Ranking of the six models by porting difficulty (easiest to hardest):
    1. **SD3.5** (recommended first candidate): 2D image generation, well-documented architecture, no Conv3d, no temporal dimension. Closest to standard LLM-style attention patterns. Established test suite.
    2. **Flux1**: similar to SD3.5 but with guidance embedding and different RoPE handling. Slightly more complex.
    3. **Motif**: similar architecture to Flux1 with context head scaling. Moderate complexity.
    4. **Qwen-Image**: adds vision encoder (Qwen2.5VL) but otherwise standard DiT. Moderate complexity due to additional encoder.
    5. **Wan2.2**: text-to-video with Conv3d layers, temporal parallelism, cross-attention blocks. Significantly more complex.
    6. **Mochi**: text-to-video with the most complex VAE (3D temporal+spatial parallelism) and unique attention patterns. Highest complexity.

- `porting_roadmap.md`
  - Phase 1: Infrastructure (CCLManager equivalent, `DiTParallelConfig` support, `minimal_matmul` integration).
  - Phase 2: Core layers (distributed normalization, parallel linear, joint attention as TTNNModules).
  - Phase 3: SD3.5 transformer as proof of concept (end-to-end correctness validation).
  - Phase 4: Pipeline integration (text encoding, denoising loop, VAE decoding within TT-Symbiote).
  - Phase 5: Additional models (Flux1, then video models).
  - Risk factors and open questions.

---

## Conventions

### Terminology

- **TT-DiT**: the codebase at `models/tt_dit/`, Tenstorrent's optimized DiT framework.
- **TT-Symbiote**: the codebase at `models/experimental/tt_symbiote/`, TT's PyTorch-to-TTNN acceleration framework.
- **Module** (capitalized, no prefix): refers to TT-DiT's `Module` class in `layers/module.py`.
- **TTNNModule** (always with TTNN prefix): refers to TT-Symbiote's base class in `core/module.py`.
- **Parameter** (capitalized, no prefix): refers to TT-DiT's `Parameter` class.
- **CCLManager**: TT-DiT's collective communication manager in `parallel/manager.py`.
- **DiTParallelConfig**: TT-DiT's 3-axis parallelism configuration.
- **DistributedConfig**: TT-Symbiote's distributed device configuration.
- **TP**: tensor parallel. **SP**: sequence parallel. **CFG-P**: classifier-free guidance parallel.
- **FSDP**: fully sharded data parallel (weight sharding across the SP mesh axis to reduce memory).
- **Joint attention**: the attention pattern where spatial tokens and prompt tokens attend to each other's keys/values jointly.
- **Ring joint SDPA**: sequence-parallel variant using ring all-gather during attention computation.

### Notation

- File paths are relative to the tt-metal repository root (e.g., `models/tt_dit/layers/module.py`).
- Class names use their fully qualified form on first mention (e.g., `blocks.attention.Attention`), then short form (e.g., `Attention`).
- TTNN operations are referenced as `ttnn.op_name` or `ttnn.experimental.op_name`.
- Tensor shapes use the convention `[batch, seq, dim]` for 3D and `[batch, heads, seq, head_dim]` for 4D unless otherwise noted.

### Formatting Rules

- Code snippets from the source use Python syntax highlighting and include the source file path as a comment.
- Comparison tables use three columns: Feature | TT-DiT | TT-Symbiote.
- Diagrams use ASCII art or Mermaid when illustrating data flow.
- Each chapter begins with a "Prerequisites" section listing which earlier chapters should be read first.
- Each chapter ends with a "Key Takeaways" bullet list (3-5 items).

---

## Cross-Chapter Dependencies

- **Chapter 2** (Parallelism) depends on **Chapter 1** (Architecture Overview) for understanding of `Module`, `Parameter`, and mesh device concepts.
- **Chapter 3** (Custom Layers) depends on **Chapter 2** (Parallelism) for understanding of `ColParallelLinear`, `RowParallelLinear`, and CCLManager used by distributed normalization layers.
- **Chapter 4** (Attention) depends on **Chapter 2** (Parallelism) for tensor parallel and sequence parallel execution, and **Chapter 3** (Custom Layers) for the normalization and linear layers used within attention.
- **Chapter 5** (Pipelines) depends on **Chapters 1-4** for understanding all components that pipelines orchestrate; also references **Chapter 2** for submesh creation and CCLManager lifecycle.
- **Chapter 6** (Weight Loading) depends on **Chapter 1** (Module/Parameter) for the loading mechanism and **Chapter 4** (Attention) for understanding the QKV merging logic in `_prepare_torch_state`.
- **Chapter 7** (Tracing) depends on **Chapter 5** (Pipelines) for understanding how traces are captured at the pipeline level, and **Chapter 2** (CCL) for understanding semaphore state across traces.
- **Chapter 8** (Porting Strategy) depends on **all preceding chapters** as the synthesis and is designed to be read last.
