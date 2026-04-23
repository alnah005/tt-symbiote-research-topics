# Component Assessment: Three-Tier Classification

## Prerequisites

- [Chapter 8 Index](./index.md): overview of the porting challenge and the structure of this chapter.
- [Chapter 1 -- Comparison with TTNNModule](../ch1_architecture_overview/comparison_with_ttnnmodule.md): architectural differences between TT-DiT `Module` and TT-Symbiote `TTNNModule`.
- [Chapter 2 -- Mapping to Symbiote](../ch2_parallelism_and_ccl/mapping_to_symbiote.md): gap analysis for parallelism and CCL infrastructure.
- [Chapter 3 -- Custom Layers and Ops](../ch3_custom_layers_and_ops/index.md): full catalog of TT-DiT layers and their TT-Symbiote equivalents.
- [Chapter 4 -- Comparison with Symbiote Attention](../ch4_attention_and_transformer_blocks/comparison_with_symbiote_attention.md): gap analysis for attention and transformer block components.

---

## Overview

This file classifies every TT-DiT component into one of three tiers based on porting effort:

| Tier | Definition | Effort per Component |
|------|-----------|---------------------|
| **Tier 1: Directly Reusable** | Component has a functional TT-Symbiote equivalent or can be used as-is with trivial wrapping. | < 1 day |
| **Tier 2: Reimplementable as TTNNModule** | Component needs a new `TTNNModule` subclass, but the underlying TTNN operations are available and the logic is straightforward. | 1--5 days |
| **Tier 3: Requires New Infrastructure** | Component depends on capabilities that TT-Symbiote fundamentally lacks. Requires framework-level changes before the component can be ported. | 1--4 weeks |

The assessment is based on findings from Chapters 1--7, with cross-references to specific comparison analyses.

---

## Tier 1: Directly Reusable

These components have working TT-Symbiote equivalents or use only basic TTNN operations that require no framework changes.

### Normalization (Single-Device)

| TT-DiT Component | TT-Symbiote Equivalent | Notes |
|---|---|---|
| `RMSNorm` | `TTNNRMSNorm` | Both wrap `ttnn.rms_norm`. Weight unsqueeze handled in `preprocess_weights_impl`. See [Ch3 normalization](../ch3_custom_layers_and_ops/normalization_layers.md). |
| `LayerNorm` | `TTNNLayerNorm` | Both wrap `ttnn.layer_norm`. TT-DiT's row-major workaround may not be needed in all contexts. See [Ch3 normalization](../ch3_custom_layers_and_ops/normalization_layers.md). |

### Activations

| TT-DiT Component | TT-Symbiote Equivalent | Notes |
|---|---|---|
| `silu` activation | `ttnn.silu` | Elementwise op, identical across frameworks. |
| `gelu` activation | `ttnn.gelu` | Elementwise op, identical across frameworks. |
| `swiglu` pattern | Composable from `ttnn.silu` + `ttnn.mul` | TT-DiT implements swiglu as fused activation in `ColParallelLinear`. TT-Symbiote's `TTNNLinearActivation` provides the same composition pattern. |

### Sinusoidal Embeddings

| TT-DiT Component | TT-Symbiote Equivalent | Notes |
|---|---|---|
| `Timesteps` | No direct equivalent, but trivial to port | Uses `ttnn.cos`, `ttnn.sin`, `ttnn.concat` -- all standard TTNN ops. ~50 lines of forward logic. |
| `TimestepEmbedding` | Composable from `TTNNLinear` + `ttnn.silu` | Two linear layers with a SiLU activation between them. |

### Basic Utilities

| TT-DiT Component | TT-Symbiote Equivalent | Notes |
|---|---|---|
| RoPE application (`_apply_rope`) | Partially reusable | TT-DiT uses `ttnn.alt_complex_rotate90`. TT-Symbiote's `TTNNRotaryPositionEmbedding` uses `ttnn.experimental.rotary_embedding_llama`. The TTNN ops differ, but the 2D spatial RoPE pattern from TT-DiT can be wrapped trivially. |
| `PaddingConfig` | No equivalent, but self-contained | Pure Python utility for computing tile-aligned head padding. Can be imported directly from TT-DiT or reimplemented in ~30 lines. |
| `utils/substate.py` helpers | Not needed | TT-Symbiote uses `from_torch` rather than state dict manipulation, so `pop_substate`/`rename_substate` are unnecessary. |

### Tier 1 Summary

**Total components:** 10
**Estimated effort:** 3--5 days for the entire tier (including writing tests).
**Dependencies:** None. These can be ported immediately.

---

## Tier 2: Reimplementable as TTNNModule Subclasses

These components require new `TTNNModule` subclasses, but the underlying TTNN operations exist and the implementation logic is well-understood from TT-DiT's source code.

### Linear Layers

| TT-DiT Component | Porting Strategy | Effort | Notes |
|---|---|---|---|
| `Linear` | Extend `TTNNLinear` | 1--2 days | TT-DiT uses `ttnn.experimental.minimal_matmul`; TT-Symbiote uses `ttnn.linear`. Both work. For performance parity, switch to `minimal_matmul` with shape-specific `MinimalMatmulConfig`. Weight transpose moves from `_prepare_torch_state` to `preprocess_weights_impl`. See [Ch3 index](../ch3_custom_layers_and_ops/index.md). |
| `ColParallelLinear` | Extend `TTNNLinearIReplicatedWColSharded` | 3--5 days | Add FSDP support (`fsdp_mesh_axis` with pre-matmul all-gather), configurable `cluster_axis`, and optional `minimal_matmul`. Weight interleaving for TP-compatible head sharding moves to `preprocess_weights_impl`. See [Ch2 parallel linears](../ch2_parallelism_and_ccl/parallel_linear_layers.md). |
| `RowParallelLinear` | Extend `TTNNLinearIColShardedWRowSharded` | 3--5 days | Add FSDP support, switch from synchronous `ttnn.reduce_scatter` to the async `tt_all_reduce()`/`tt_all_gather()` helpers that already exist in `models/tt_transformers/tt/ccl.py`. Make topology and mesh axis configurable. See [Ch2 mapping](../ch2_parallelism_and_ccl/mapping_to_symbiote.md). |

### Feedforward

| TT-DiT Component | Porting Strategy | Effort | Notes |
|---|---|---|---|
| `FeedForward` | Compose from `TTNNLinear` + activation | 1 day | Two stacked linear layers with activation. Straightforward composition. See [Ch3 index](../ch3_custom_layers_and_ops/index.md). |
| `ParallelFeedForward` | Compose from ported ColParallel + RowParallel | 2 days | `ColParallelLinear` (ff1, up-projection with activation) followed by `RowParallelLinear` (ff2, down-projection with reduce-scatter). Depends on Tier 2 parallel linear ports. |

### Convolution

| TT-DiT Component | Porting Strategy | Effort | Notes |
|---|---|---|---|
| `Conv2d` | Extend `TTNNConv2dNHWC` | 2--3 days | TT-DiT's `Conv2d` calls `ttnn.conv2d` with data-parallel support via `vae_all_gather`. TT-Symbiote's `TTNNConv2dNHWC` wraps `ttnn.conv2d` through the `tt_cnn` builder. Need to add data-parallel input gathering and shape-specific slice parameters. See [Ch3 convolution](../ch3_custom_layers_and_ops/convolution_layers.md). |

### Normalization (Distributed)

| TT-DiT Component | Porting Strategy | Effort | Notes |
|---|---|---|---|
| `DistributedRMSNorm` | Create new `TTNNDiTDistributedRMSNorm` | 3--5 days | TT-DiT uses `ttnn.experimental.wan_fused_rmsnorm_pre_allgather` + all-gather + `ttnn.experimental.wan_fused_rmsnorm_post_allgather`. TT-Symbiote's existing `TTNNDistributedRMSNorm` uses `ttnn.rms_norm_pre_all_gather` + `ttnn.rms_norm_post_all_gather` (stable API). Both patterns are valid, but TT-DiT's variant supports fused RoPE. Create a new subclass using the experimental ops. See [Ch3 normalization](../ch3_custom_layers_and_ops/normalization_layers.md). |
| `GroupNorm` | Create new `TTNNGroupNorm` | 2--3 days | Wraps `ttnn.group_norm` with `ttnn.create_group_norm_input_mask` and `ttnn.create_group_norm_weight_bias_rm`. Used exclusively in VAE decoders. No distributed variant needed initially. See [Ch3 normalization](../ch3_custom_layers_and_ops/normalization_layers.md). |

### Embeddings (DiT-Specific)

| TT-DiT Component | Porting Strategy | Effort | Notes |
|---|---|---|---|
| `PatchEmbed` | Create new `TTNNPatchEmbed` | 2 days | Implements patch embedding as an unfolded `ttnn.linear` projection (not `ttnn.conv2d`). Standard ops, model-specific shapes. |
| `SD35CombinedTimestepTextProjEmbeddings` | Create new `TTNNModule` subclass | 1--2 days | Aggregates timestep and text projection embeddings. Composed from `TTNNLinear` and `Timesteps`. Model-specific glue. See [Ch3 embeddings](../ch3_custom_layers_and_ops/index.md). |
| `CombinedTimestepGuidanceTextProjEmbeddings` | Create new `TTNNModule` subclass | 1--2 days | Similar to above with guidance embedding. Used by Flux1. |
| `Embedding` (token lookup) | Use `TTNNEmbedding` | < 1 day | Direct mapping to TT-Symbiote's existing `TTNNEmbedding`. |

### Attention Components

| TT-DiT Component | Porting Strategy | Effort | Notes |
|---|---|---|---|
| Fused QKV projection | Create `TTNNDiTFusedQKV` | 3--5 days | `ColParallelLinear` with interleaved head layout for TP. Weight preparation (merging Q/K/V into fused format, per-device head interleaving) moves to `preprocess_weights_impl`. The most complex weight transformation in TT-DiT. See [Ch4 joint attention](../ch4_attention_and_transformer_blocks/joint_attention.md) and [Ch6 weight pipeline](../ch6_weight_loading/tt_dit_weight_pipeline.md). |
| Per-head Q/K RMSNorm | Reuse `TTNNRMSNorm` with shape adaptation | 1 day | TT-Symbiote's `TTNNGR00TSelfAttention` already has optional per-head Q/K RMSNorm. The same pattern applies to 4D `[B, H, S, D]` tensors. See [Ch4 comparison](../ch4_attention_and_transformer_blocks/comparison_with_symbiote_attention.md). |

### Tier 2 Summary

**Total components:** 16
**Estimated effort:** 30--50 engineering days.
**Dependencies:** Tier 2 parallel linears depend on TT-Symbiote's existing distributed linear infrastructure and the `tt_all_reduce()`/`tt_all_gather()` helpers from `models/tt_transformers/tt/ccl.py`. Tier 2 attention components depend on the Tier 3 joint SDPA integration.

---

## Tier 3: Requires New TT-Symbiote Infrastructure

These components depend on capabilities that do not exist in TT-Symbiote and cannot be achieved by simply creating new `TTNNModule` subclasses. They require framework-level changes.

### 3.1 CCL Infrastructure Extensions

**What is needed:** TT-Symbiote's distributed linear modules currently call synchronous `ttnn.reduce_scatter` and `ttnn.all_gather` directly, bypassing the async CCL helpers (`tt_all_reduce()`/`tt_all_gather()`) that already exist in `models/tt_transformers/tt/ccl.py`. Additionally, persistent buffer caching -- essential for trace-compatible CCL -- does not exist in TT-Symbiote.

**Specific deliverables:**

| Deliverable | Description | Effort | Reference |
|---|---|---|---|
| Refactor distributed linears to use async CCL | Wire `TTNNLinearIColShardedWRowSharded` and variants to call `tt_all_reduce()`/`tt_all_gather()` instead of raw `ttnn.reduce_scatter`/`ttnn.all_gather` | 1 week | [Ch2 mapping, Gap 2](../ch2_parallelism_and_ccl/mapping_to_symbiote.md) |
| Persistent buffer cache for CCL | Add a buffer cache to `TT_CCL` (or create a new `CCLBufferManager`) that stores and reuses output buffers keyed by `(shape, dim, axis)` | 1--2 weeks | [Ch2 mapping, Gap 1](../ch2_parallelism_and_ccl/mapping_to_symbiote.md) |
| Semaphore reset for trace replay | Add `reset_global_semaphores()` to `TT_CCL` to reset semaphore state between trace captures | 2 days | [Ch2 CCL manager](../ch2_parallelism_and_ccl/ccl_manager.md) |
| Shape-based hyperparameter tuning | Add `get_ag_hyperparams()`/`get_rs_hyperparams()` to select `chunks_per_sync`, `num_workers_per_link` based on tensor shape | 3 days | [Ch2 mapping, Gap 7](../ch2_parallelism_and_ccl/mapping_to_symbiote.md) |

**Total effort:** 2--4 weeks.

### 3.2 Multi-Axis Parallelism Configuration

**What is needed:** TT-Symbiote's `DistributedConfig` provides a single tensor distribution strategy (batch+channel sharding) applied uniformly across all modules. DiT models require three independent parallelism axes (CFG, SP, TP) with per-parameter sharding directives. See [Ch2 mapping](../ch2_parallelism_and_ccl/mapping_to_symbiote.md) for the detailed comparison.

**Specific deliverables:**

| Deliverable | Description | Effort | Reference |
|---|---|---|---|
| `DiTDistributedConfig` | New config type supporting `ParallelFactor` tuples for CFG-P, SP, and TP axes | 1 week | [Ch2 mapping, Gap 3](../ch2_parallelism_and_ccl/mapping_to_symbiote.md) |
| Submesh creation and management | Add `create_submeshes()` integration to TT-Symbiote's device management | 1 week | [Ch2 mapping, Gap 4](../ch2_parallelism_and_ccl/mapping_to_symbiote.md) |
| Per-parameter sharding directives | Mechanism for `TTNNModule` subclasses to specify different distribution strategies for different weight tensors | 1 week | [Ch2 mapping](../ch2_parallelism_and_ccl/mapping_to_symbiote.md) |
| Configurable mesh axis and topology | Make `cluster_axis` and `topology` configurable on distributed linear classes | 2 days | [Ch2 mapping, Gap 6](../ch2_parallelism_and_ccl/mapping_to_symbiote.md) |

**Total effort:** 3--4 weeks.

### 3.3 Joint Attention

**What is needed:** TT-Symbiote has no attention module that handles two input sequences (spatial + prompt) through a single SDPA computation. The `ttnn.transformer.joint_scaled_dot_product_attention` kernel is a stable TTNN API, but no TT-Symbiote module calls it. See [Ch4 comparison](../ch4_attention_and_transformer_blocks/comparison_with_symbiote_attention.md).

**Specific deliverables:**

| Deliverable | Description | Effort | Reference |
|---|---|---|---|
| `TTNNDiTAttention` module | New `TTNNModule` subclass wrapping `joint_scaled_dot_product_attention` with dual-stream QKV projections, per-head RMSNorm, and separate RoPE | 2 weeks | [Ch4 joint attention](../ch4_attention_and_transformer_blocks/joint_attention.md) |
| Ring joint SDPA for sequence parallelism | Integration of `ring_joint_scaled_dot_product_attention` with persistent ping-pong buffers and CCL semaphores | 1--2 weeks | [Ch4 joint attention](../ch4_attention_and_transformer_blocks/joint_attention.md) |
| `TTNNDiTTransformerBlock` module | Block-level module combining adaptive LayerNorm, joint attention, feedforward, and gated residual connections | 1--2 weeks | [Ch4 transformer block](../ch4_attention_and_transformer_blocks/transformer_block.md) |

**Total effort:** 4--6 weeks.

### 3.4 Adaptive Layer Normalization (DistributedLayerNorm)

**What is needed:** TT-DiT's `DistributedLayerNorm` uses `ttnn.experimental.dit_layernorm_pre_allgather` and `ttnn.experimental.dit_layernorm_post_allgather` with dynamic `(1 + scale)` weight and `shift` bias -- time-conditioned modulation that TT-Symbiote's normalization layers do not support. See [Ch3 normalization](../ch3_custom_layers_and_ops/normalization_layers.md).

**Specific deliverables:**

| Deliverable | Description | Effort | Reference |
|---|---|---|---|
| `TTNNDiTDistributedLayerNorm` | New module wrapping the experimental two-phase LayerNorm with dynamic weight/bias support | 1--2 weeks | [Ch3 normalization](../ch3_custom_layers_and_ops/normalization_layers.md) |
| Reciprocal tensor cache | Port TT-DiT's `_recip_tensors` shared cache for Welford-algorithm reciprocals | 2 days | [Ch3 normalization](../ch3_custom_layers_and_ops/normalization_layers.md) |

**Total effort:** 1--2 weeks.

### 3.5 Conv3d for Video Models

**What is needed:** `ContextParallelConv3d` wraps `ttnn.experimental.conv3d` with temporal context parallelism. TT-Symbiote has no Conv3d support at all. This is required only for video models (Mochi, Wan2.2). See [Ch3 convolution](../ch3_custom_layers_and_ops/convolution_layers.md).

**Specific deliverables:**

| Deliverable | Description | Effort | Reference |
|---|---|---|---|
| `TTNNConv3d` module | New module wrapping `ttnn.experimental.conv3d` with temporal decomposition | 2 weeks | [Ch3 convolution](../ch3_custom_layers_and_ops/convolution_layers.md) |
| Context parallelism for temporal dimension | Integration with CCL for distributing temporal slices across devices | 1--2 weeks | [Ch3 convolution](../ch3_custom_layers_and_ops/convolution_layers.md) |

**Total effort:** 3--4 weeks. (Deferrable until video model porting in Phase 5.)

### 3.6 Pipeline-Level Orchestration

**What is needed:** TT-Symbiote's module replacement pattern assumes a single `nn.Module` tree. DiT pipelines orchestrate multiple independent components (encoders, transformer, VAE) with dynamic memory management (`set_unload_set`) and submesh-level device assignment. See [Ch5 mapping to serving](../ch5_pipelines_and_serving/mapping_to_symbiote_serving.md).

**Specific deliverables:**

| Deliverable | Description | Effort | Reference |
|---|---|---|---|
| Multi-component pipeline abstraction | New `TTNNPipeline` base class (or equivalent) supporting multi-model orchestration with per-component device assignment | 2 weeks | [Ch5 mapping](../ch5_pipelines_and_serving/mapping_to_symbiote_serving.md) |
| Dynamic memory management | Add `deallocate_weights()` / `unload_set` semantics to `TTNNModule` for component swapping | 1 week | [Ch1 comparison](../ch1_architecture_overview/comparison_with_ttnnmodule.md) |
| Weight caching infrastructure | Port TT-DiT's `.tensorbin` caching or implement equivalent in TT-Symbiote | 1 week | [Ch6 weight pipeline](../ch6_weight_loading/tt_dit_weight_pipeline.md) |
| Submesh-aware tracing | Extend `TracedRun` cache keys to include submesh identity; add multi-device synchronization before trace capture | 1 week | [Ch7 integration strategy](../ch7_tracing_and_performance/integration_strategy.md) |

**Total effort:** 4--6 weeks.

### Tier 3 Summary

**Total infrastructure areas:** 6
**Total estimated effort:** 17--26 weeks (for all areas).
**Critical path:** CCL extensions (3.1) and multi-axis parallelism (3.2) are prerequisites for nearly everything else. Joint attention (3.3) depends on both. Pipeline orchestration (3.6) can proceed in parallel with the other areas.

---

## Complete Classification Table

The table below lists every major TT-DiT component with its tier assignment:

| Component | Source File | Tier | TT-Symbiote Path |
|---|---|---|---|
| `RMSNorm` | `layers/normalization.py` | 1 | `TTNNRMSNorm` |
| `LayerNorm` | `layers/normalization.py` | 1 | `TTNNLayerNorm` |
| `silu`, `gelu` | (activations) | 1 | `ttnn.silu`, `ttnn.gelu` |
| `Timesteps` | `layers/embeddings.py` | 1 | New trivial `TTNNModule` |
| `TimestepEmbedding` | `layers/embeddings.py` | 1 | Compose from `TTNNLinear` |
| `PaddingConfig` | `utils/padding.py` | 1 | Import or reimplement |
| `_apply_rope` | `blocks/attention.py` | 1 | Wrap `ttnn.alt_complex_rotate90` |
| `Linear` | `layers/linear.py` | 2 | Extend `TTNNLinear` |
| `ColParallelLinear` | `layers/linear.py` | 2 | Extend `TTNNLinearIReplicatedWColSharded` |
| `RowParallelLinear` | `layers/linear.py` | 2 | Extend `TTNNLinearIColShardedWRowSharded` |
| `FeedForward` | `layers/feedforward.py` | 2 | Compose from `TTNNLinear` |
| `ParallelFeedForward` | `layers/feedforward.py` | 2 | Compose from ported parallel linears |
| `Conv2d` | `layers/conv2d.py` | 2 | Extend `TTNNConv2dNHWC` |
| `DistributedRMSNorm` | `layers/normalization.py` | 2 | New `TTNNDiTDistributedRMSNorm` |
| `GroupNorm` | `layers/normalization.py` | 2 | New `TTNNGroupNorm` |
| `PatchEmbed` | `layers/embeddings.py` | 2 | New `TTNNPatchEmbed` |
| Combined embeddings | `layers/embeddings.py` | 2 | New model-specific `TTNNModule` subclasses |
| `Embedding` (token) | `layers/embeddings.py` | 2 | `TTNNEmbedding` |
| Fused QKV projection | `blocks/attention.py` | 2 | New `TTNNDiTFusedQKV` |
| Per-head Q/K RMSNorm | `blocks/attention.py` | 2 | Reuse from `TTNNGR00TSelfAttention` pattern |
| `CCLManager` | `parallel/manager.py` | 3 | Extend `TT_CCL` + add buffer cache |
| `DiTParallelConfig` | `parallel/config.py` | 3 | New `DiTDistributedConfig` |
| Submesh management | `pipelines/*` | 3 | New device management utilities |
| `DistributedLayerNorm` | `layers/normalization.py` | 3 | New `TTNNDiTDistributedLayerNorm` |
| Joint attention | `blocks/attention.py` | 3 | New `TTNNDiTAttention` |
| Ring joint SDPA | `blocks/attention.py` | 3 | Ring SDPA + CCL integration |
| `TransformerBlock` | `blocks/transformer_block.py` | 3 | New `TTNNDiTTransformerBlock` |
| `ContextParallelConv3d` | `layers/conv3d.py` | 3 | New `TTNNConv3d` (video only) |
| Pipeline orchestration | `pipelines/*` | 3 | New `TTNNPipeline` abstraction |
| Dynamic memory management | `layers/module.py` | 3 | Add `unload_set` to `TTNNModule` |
| Weight caching | `utils/cache.py` | 3 | Port `.tensorbin` or new system |
| Submesh-aware tracing | `utils/tracing.py` | 3 | Extend `TracedRun` cache keys |

---

## Dependency Graph

The following shows the dependency relationships between Tier 3 infrastructure areas and the Tier 2 components that depend on them:

```
3.1 CCL Extensions
 |
 +---> 3.2 Multi-Axis Parallelism
 |      |
 |      +---> Tier 2: ColParallelLinear, RowParallelLinear (configurable axis)
 |      |
 |      +---> 3.3 Joint Attention (requires TP + SP infrastructure)
 |      |      |
 |      |      +---> 3.4 DistributedLayerNorm (adaptive modulation)
 |      |      |      |
 |      |      |      +---> Tier 2 TransformerBlock (composes all above)
 |      |      |
 |      |      +---> Ring Joint SDPA (requires CCL persistent buffers)
 |      |
 |      +---> 3.6 Pipeline Orchestration (submesh management)
 |
 +---> Tier 2: ParallelFeedForward (requires async CCL in reduce-scatter)

3.5 Conv3d (independent, deferrable)
 |
 +---> Video VAE porting (Mochi, Wan2.2)
```

The critical path is: **3.1 CCL Extensions -> 3.2 Multi-Axis Parallelism -> 3.3 Joint Attention -> 3.4 Adaptive LayerNorm -> TransformerBlock integration.**

---

## Key Takeaways

1. **The Tier 1 components (directly reusable) provide a foundation of basic building blocks** -- normalization, activations, sinusoidal embeddings, and RoPE -- that can be ported immediately with no infrastructure work, establishing the testing and validation patterns for subsequent tiers.

2. **Tier 2 components (new TTNNModule subclasses) represent the bulk of the model-specific code** and are individually tractable. The parallel linear layers and fused QKV projection are the most complex items, requiring careful weight transformation logic in `preprocess_weights_impl`.

3. **Tier 3 infrastructure (CCL, multi-axis parallelism, joint attention, pipeline orchestration) is the critical gating factor.** Without these framework-level extensions, the Tier 2 parallel components cannot achieve competitive performance, and the Tier 3 attention/block components cannot be implemented at all.

4. **The dependency graph has a clear critical path** from CCL extensions through multi-axis parallelism to joint attention. Work on Conv3d and pipeline orchestration can proceed in parallel once the CCL foundation is in place.

5. **Video model components (Conv3d, temporal parallelism) are entirely deferrable** until image model porting is complete. This halves the initial Tier 3 infrastructure scope.

---

**Next:** [`model_prioritization.md`](./model_prioritization.md)
