# Chapter 3: Custom Layers and TTNN Operations

## Prerequisites

- [Chapter 1 -- Module and Parameter](../ch1_architecture_overview/module_and_parameter.md): understanding of `Module`, `Parameter`, and `_prepare_torch_state`.
- [Chapter 1 -- Comparison with TTNNModule](../ch1_architecture_overview/comparison_with_ttnnmodule.md): understanding of the fundamental differences between TT-DiT's `Module` and TT-Symbiote's `TTNNModule`.
- [Chapter 2 -- Parallel Linear Layers](../ch2_parallelism_and_ccl/parallel_linear_layers.md): understanding of `Linear`, `ColParallelLinear`, and `RowParallelLinear`.

---

## Introduction

TT-DiT implements every neural network layer as a direct TTNN call -- there is no PyTorch runtime involved during inference. This chapter catalogs the custom layer implementations across five categories, identifies the specific TTNN operations each layer depends on, and maps them to their TT-Symbiote equivalents (or notes the absence thereof).

The layers fall into the following groups:

| Category | TT-DiT Source File | Layers |
|---|---|---|
| **Normalization** | `layers/normalization.py` | `RMSNorm`, `LayerNorm`, `DistributedRMSNorm`, `DistributedLayerNorm`, `GroupNorm` |
| **Linear** | `layers/linear.py` | `Linear`, `ColParallelLinear`, `RowParallelLinear` |
| **Feedforward** | `layers/feedforward.py` | `FeedForward`, `ParallelFeedForward` |
| **Convolution** | `layers/conv2d.py`, `layers/conv3d.py` | `Conv2d`, `ContextParallelConv3d` |
| **Embeddings** | `layers/embeddings.py` | `Timesteps`, `TimestepEmbedding`, `PatchEmbed`, `MochiPatchEmbed`, `WanPatchEmbed`, `WanTimeTextImageEmbedding`, `Embedding`, `PixArtAlphaTextProjection`, `SD35CombinedTimestepTextProjEmbeddings`, `CombinedTimestepGuidanceTextProjEmbeddings` |

## Chapter Files

1. [`normalization_layers.md`](./normalization_layers.md) -- Detailed comparison of all five normalization layers with their TT-Symbiote equivalents. Covers both single-device and distributed variants.
2. [`ttnn_experimental_ops.md`](./ttnn_experimental_ops.md) -- Catalog of every `ttnn.experimental.*` operation used by TT-DiT. For each: purpose, parameters, and whether TT-Symbiote has an equivalent.
3. [`convolution_layers.md`](./convolution_layers.md) -- Conv2d and Conv3d in TT-DiT versus TT-Symbiote's `TTNNConv2dNHWC`. Porting considerations for VAE workloads.

## Layer Classification Summary

### Normalization (5 layers)

TT-DiT provides both standard (single-device) and distributed (multi-device) normalization:

- **`RMSNorm`** and **`LayerNorm`** wrap `ttnn.rms_norm` and `ttnn.layer_norm` respectively. Both are direct calls with no cross-device communication.
- **`DistributedRMSNorm`** uses a two-phase pattern: `ttnn.experimental.wan_fused_rmsnorm_pre_allgather` to compute local statistics, then all-gather, then `ttnn.experimental.wan_fused_rmsnorm_post_allgather` to apply the norm. This enables normalization over activations sharded across devices.
- **`DistributedLayerNorm`** follows the same two-phase pattern using `ttnn.experimental.dit_layernorm_pre_allgather` and `ttnn.experimental.dit_layernorm_post_allgather`, with an additional Welford-algorithm reciprocal tensor.
- **`GroupNorm`** wraps `ttnn.group_norm` and is used exclusively in VAE models.

TT-Symbiote provides `TTNNLayerNorm`, `TTNNRMSNorm`, `TTNNLocalRMSNorm`, and `TTNNDistributedRMSNorm`. The distributed variant uses a different API (`ttnn.rms_norm_pre_all_gather` / `ttnn.rms_norm_post_all_gather` -- note these are stable, not experimental). There is no TT-Symbiote equivalent for `DistributedLayerNorm` or `GroupNorm`.

See [`normalization_layers.md`](./normalization_layers.md) for the full comparison.

### Linear (3 layers)

Linear layers are covered in [Chapter 2 -- Parallel Linear Layers](../ch2_parallelism_and_ccl/parallel_linear_layers.md). In brief:

- **`Linear`**: replicated weights, uses `ttnn.experimental.minimal_matmul`.
- **`ColParallelLinear`**: column-sharded weights, uses `ttnn.experimental.minimal_matmul` (and optionally `ttnn.experimental.minimal_matmul_split`).
- **`RowParallelLinear`**: row-sharded weights, uses `ttnn.experimental.minimal_matmul` with `reduce_scatter`.

All three use `ttnn.experimental.minimal_matmul` instead of `ttnn.linear`. TT-Symbiote's `TTNNLinear` uses `ttnn.linear`. This is a critical distinction covered in [`ttnn_experimental_ops.md`](./ttnn_experimental_ops.md).

### Feedforward (2 layers)

Feedforward layers are thin wrappers over linear layers:

- **`FeedForward`**: two stacked `Linear` layers with an optional activation (GELU, SiLU, SwiGLU).
- **`ParallelFeedForward`**: `ColParallelLinear` (ff1) followed by `RowParallelLinear` (ff2), implementing Megatron-style tensor parallelism. The ff1 up-projection applies the activation; ff2's `reduce_scatter` sums partial results across devices.

Both patterns have direct TT-Symbiote analogs (`TTNNLinear` composition, or the `TTNNLinearIColShardedWRowSharded`/`TTNNLinearIReplicatedWColSharded` pair).

### Convolution (2 layers)

- **`Conv2d`**: wraps `ttnn.conv2d` with support for input-channel and output-channel tensor parallelism. Used in VAE encoders/decoders.
- **`ContextParallelConv3d`**: wraps `ttnn.experimental.conv3d` with context parallelism over the temporal dimension. Used in Mochi and Wan video VAEs.

TT-Symbiote has `TTNNConv2dNHWC` which also wraps `ttnn.conv2d` but through the `tt_cnn` builder abstraction. There is no TT-Symbiote Conv3d. See [`convolution_layers.md`](./convolution_layers.md) for details.

### Embeddings (10 layers)

TT-DiT's embedding layers are model-specific and fall into four sub-categories:

1. **Sinusoidal timestep embeddings** (`Timesteps`, `TimestepEmbedding`): compute $\sin/\cos$ positional encodings from diffusion timesteps. These use basic TTNN ops (`ttnn.cos`, `ttnn.sin`, `ttnn.concat`).
2. **Patch embeddings** (`PatchEmbed`, `MochiPatchEmbed`, `WanPatchEmbed`): convert image/video patches into embedding vectors. All implement Conv2d as an unfolded `ttnn.linear` projection rather than calling `ttnn.conv2d`.
3. **Combined embeddings** (`SD35CombinedTimestepTextProjEmbeddings`, `CombinedTimestepGuidanceTextProjEmbeddings`, `WanTimeTextImageEmbedding`, `PixArtAlphaTextProjection`): aggregate timestep, text, and guidance embeddings. Composed from the other embedding and linear layers.
4. **Token embedding** (`Embedding`): wraps `ttnn.embedding` for vocabulary lookups in text encoders.

TT-Symbiote provides `TTNNEmbedding` (vocabulary lookup) but does not have equivalents for the DiT-specific sinusoidal or patch embedding layers. These would need to be implemented as new `TTNNModule` subclasses during porting.

## Key Takeaways

1. **Five normalization layers, two are distributed**: The distributed variants (`DistributedRMSNorm`, `DistributedLayerNorm`) use `ttnn.experimental.*` split-gather patterns that have no direct TT-Symbiote equivalent. Porting them requires either wrapping the experimental ops or using TT-Symbiote's existing `TTNNDistributedRMSNorm` (which uses different stable TTNN APIs).

2. **All linear layers use `ttnn.experimental.minimal_matmul`**: This is the single most impactful difference from TT-Symbiote (which uses `ttnn.linear`). The `minimal_matmul` API provides explicit control over block sizes, subblock dimensions, and core grid mapping via `ttnn.MinimalMatmulConfig`.

3. **Feedforward layers are composition, not custom ops**: `FeedForward` and `ParallelFeedForward` are pure compositions of linear layers. Porting them is a matter of wiring TT-Symbiote linear variants correctly.

4. **Conv3d is TT-DiT-exclusive**: `ttnn.experimental.conv3d` is used only by TT-DiT for video model VAEs. TT-Symbiote has no Conv3d support, making video VAE porting a non-trivial effort.

5. **Embedding layers are model-specific glue**: Most embedding layers (timestep, patch, combined) are unique to diffusion transformer workflows and have no TT-Symbiote counterpart. They are composed from standard ops and linear layers, so they can be ported as new `TTNNModule` subclasses.

---

**Next:** [`normalization_layers.md`](./normalization_layers.md)
