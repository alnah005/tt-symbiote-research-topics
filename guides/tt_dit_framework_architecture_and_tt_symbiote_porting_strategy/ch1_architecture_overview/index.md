# Chapter 1: TT-DiT Architecture Overview

## Prerequisites

None. This is the first chapter.

## Introduction

TT-DiT (Tenstorrent Diffusion Transformers) is a purpose-built framework for running diffusion transformer models on Tenstorrent Wormhole hardware. Unlike TT-Symbiote, which provides general-purpose PyTorch-to-TTNN acceleration via dispatch interception, TT-DiT is a from-scratch implementation where every layer calls TTNN operations directly. This chapter provides a high-level map of the codebase, catalogs the supported models, and explains the relationship between the framework's major abstractions.

## Directory Tree

The TT-DiT codebase lives at `models/tt_dit/` in the tt-metal repository. Below is the full directory structure with annotations:

```
tt_dit/
├── layers/                  # Core neural network layer primitives
│   ├── module.py            #   Module, Parameter, ModuleList, UnregisteredModule (base classes)
│   ├── linear.py            #   Linear, ColParallelLinear, RowParallelLinear
│   ├── normalization.py     #   RMSNorm, LayerNorm, DistributedRMSNorm, DistributedLayerNorm, GroupNorm
│   ├── feedforward.py       #   FeedForward, ParallelFeedForward
│   ├── embeddings.py        #   Timesteps, TimestepEmbedding, PatchEmbed, combined embeddings
│   ├── conv2d.py            #   Conv2d with data-parallel support
│   └── conv3d.py            #   Conv3d for video model temporal convolution
│
├── blocks/                  # Composite building blocks
│   ├── attention.py         #   Attention (joint spatial+prompt SDPA with TP and SP)
│   └── transformer_block.py #   TransformerBlock with adaptive layer normalization
│
├── models/                  # Full model architectures
│   ├── transformers/        #   Transformer models per architecture
│   │   ├── transformer_sd35.py       # SD3.5 DiT transformer
│   │   ├── attention_sd35.py         # SD3.5-specific attention variant
│   │   ├── transformer_flux1.py      # Flux1 DiT transformer
│   │   ├── transformer_motif.py      # Motif DiT transformer
│   │   ├── transformer_mochi.py      # Mochi video DiT transformer
│   │   ├── attention_mochi.py        # Mochi-specific attention variant
│   │   ├── transformer_qwenimage.py  # Qwen-Image DiT transformer
│   │   └── wan2_2/                   # Wan2.2 text-to-video
│   │       ├── transformer_wan.py    #   Wan transformer model
│   │       └── attention_wan.py      #   Wan-specific attention with cross-attention
│   ├── vae/                 #   VAE/Autoencoder decoders
│   │   ├── vae_sd35.py               # SD3.5 VAE decoder
│   │   ├── vae_qwenimage.py          # Qwen-Image VAE decoder
│   │   ├── vae_mochi.py              # Mochi 3D VAE (spatial+temporal parallelism)
│   │   ├── vae_wan2_1.py             # Wan VAE decoder
│   │   └── vae_wan2_1_encoder_host.py # Wan VAE encoder (host-side)
│   ├── StableDiffusion35.md          # Model documentation
│   ├── Flux1.md
│   ├── Motif.md
│   ├── Mochi_1.md
│   ├── Wan2_2.md
│   └── QwenImage.md                  # (referenced in README, not present as separate file)
│
├── encoders/                # Text and vision encoder implementations
│   ├── clip/                #   CLIP text encoder
│   │   ├── model_clip.py    #     CLIPEncoder (used by SD3.5, Flux1, Motif)
│   │   └── encoder_pair.py  #     Host-side CLIP + device-side CLIP pairing
│   ├── t5/                  #   T5 text encoder
│   │   ├── model_t5.py      #     T5Encoder (used by SD3.5, Flux1)
│   │   └── encoder_pair.py  #     Host-side T5 + device-side T5 pairing
│   ├── umt5/               #   UMT5 encoder (used by Wan2.2)
│   │   ├── model_umt5.py
│   │   └── encoder_pair.py
│   └── qwen25vl/           #   Qwen2.5-VL vision-language encoder (used by Qwen-Image)
│       ├── model_qwen25vl.py
│       └── encoder_pair.py
│
├── pipelines/               # End-to-end inference pipelines
│   ├── stable_diffusion_35_large/
│   │   └── pipeline_stable_diffusion_35_large.py   # StableDiffusion3Pipeline
│   ├── flux1/
│   │   └── pipeline_flux1.py                       # Flux1Pipeline
│   ├── motif/
│   │   └── pipeline_motif.py                       # MotifPipeline
│   ├── mochi/
│   │   └── pipeline_mochi.py                       # MochiPipeline
│   ├── wan/
│   │   ├── pipeline_wan.py                         # WanPipeline (text-to-video)
│   │   └── pipeline_wan_i2v.py                     # WanI2VPipeline (image-to-video)
│   └── qwenimage/
│       └── pipeline_qwenimage.py                   # QwenImagePipeline
│
├── parallel/                # Parallelism infrastructure
│   ├── config.py            #   DiTParallelConfig, ParallelFactor, EncoderParallelConfig,
│   │                        #   VAEParallelConfig, VaeHWParallelConfig, MochiVAEParallelConfig,
│   │                        #   vae_all_gather, vae_neighbor_pad, vae_slice_reshard
│   └── manager.py           #   CCLManager (semaphores, ping-pong buffers, async CCL ops)
│
├── utils/                   # Shared utilities
│   ├── tensor.py            #   from_torch, to_torch, bf16_tensor, typed_tensor, unflatten, upsample
│   ├── tracing.py           #   Tracer class for trace capture and replay
│   ├── cache.py             #   Weight caching with config_id for parallel-config-specific caches
│   ├── padding.py           #   PaddingConfig for tile-aligned head padding
│   ├── matmul.py            #   get_matmul_config, MinimalMatmulConfig helpers
│   ├── substate.py          #   pop_substate, substate, rename_substate for state_dict manipulation
│   ├── check.py             #   Validation utilities
│   ├── conv3d.py            #   Conv3d utility functions
│   ├── mochi.py             #   Mochi-specific utilities
│   └── test.py              #   Test utilities
│
├── tests/                   # Test suite
│   ├── unit/                #   Layer-level unit tests
│   ├── blocks/              #   Block-level tests
│   ├── models/              #   Per-model pipeline and component tests
│   │   ├── sd35/  flux1/  motif/  mochi/  wan2_2/  qwenimage/
│   ├── encoders/            #   Encoder tests
│   └── dataset_eval/        #   Dataset evaluation scripts
│
├── reference/               # Reference implementations
│   └── motif/               #   Reference Motif model
│
└── README.md                # Top-level documentation
```

## Supported Models

TT-DiT supports six generative models, spanning both image and video generation:

| Model | Type | Pipeline Class | Transformer | Pipeline File |
|-------|------|---------------|-------------|--------------|
| **Stable Diffusion 3.5 Large** | Text-to-image (1024x1024) | `StableDiffusion3Pipeline` | `SD35Transformer2DModel` | `pipelines/stable_diffusion_35_large/` |
| **Flux 1** (schnell & dev) | Text-to-image (1024x1024) | `Flux1Pipeline` | `Flux1Transformer` | `pipelines/flux1/` |
| **Motif** | Text-to-image (1024x1024) | `MotifPipeline` | `MotifTransformer` | `pipelines/motif/` |
| **Qwen-Image** | Text-to-image (1024x1024) | `QwenImagePipeline` | `QwenImageTransformer` | `pipelines/qwenimage/` |
| **Mochi-1** | Text-to-video (824x480, 168 frames) | `MochiPipeline` | `MochiTransformer` | `pipelines/mochi/` |
| **Wan2.2-T2V-A14B** | Text-to-video | `WanPipeline` | `WanTransformer` | `pipelines/wan/` |

Each model shares the same fundamental architecture (layers, blocks, Module system) but varies in:
- **Attention pattern**: joint spatial+prompt attention (SD3.5, Flux1, Motif), cross-attention (Wan2.2), or hybrid (Mochi).
- **Encoder stack**: CLIP + T5 (SD3.5, Flux1), CLIP + T5 (Motif), Qwen2.5-VL (Qwen-Image), UMT5 (Wan2.2), T5 (Mochi).
- **VAE architecture**: 2D spatial VAE (SD3.5, Flux1, Motif, Qwen-Image) or 3D spatial+temporal VAE (Mochi, Wan2.2).
- **Parallelism requirements**: image models use 2-axis parallelism (tensor parallel + sequence parallel); video models add temporal parallelism.

## The Four-Level Abstraction Hierarchy

TT-DiT organizes its components into four levels, from primitive to orchestrated:

### Level 1: Layers (`layers/`)

Layers are the atomic building blocks. Each layer is a subclass of `Module` (see [`module_and_parameter.md`](./module_and_parameter.md)) and wraps one or a few TTNN operations. Examples:

- `Linear` wraps `ttnn.experimental.minimal_matmul`.
- `RMSNorm` wraps `ttnn.rms_norm`.
- `Conv2d` wraps `ttnn.conv2d`.
- `ColParallelLinear` and `RowParallelLinear` add mesh-sharded weight distribution and CCL operations to linear layers.

Layers own `Parameter` instances that store weights as `ttnn.Tensor` objects. Each layer's `_prepare_torch_state` method handles weight format conversion (e.g., transposing linear weights from PyTorch's `[out, in]` to TTNN's `[in, out]` layout).

### Level 2: Blocks (`blocks/`)

Blocks compose multiple layers into functional units:

- **`Attention`** (`blocks/attention.py`): Implements joint attention with fused QKV projection, per-head RMSNorm, RoPE, and SDPA. Supports both single-device `ttnn.transformer.joint_scaled_dot_product_attention` and sequence-parallel `ttnn.transformer.ring_joint_scaled_dot_product_attention`.
- **`TransformerBlock`** (`blocks/transformer_block.py`): Combines adaptive layer normalization (time-conditioned modulation producing shift, scale, and gate tensors), an `Attention` sub-block, and a `FeedForward` sub-block. Each model variant may extend or customize this (e.g., `SD35TransformerBlock`, `WanTransformerBlock`, `MochiAttention`).

### Level 3: Models (`models/transformers/`, `models/vae/`)

Model modules compose blocks into complete neural networks:

- **Transformer models** stack `N` `TransformerBlock` instances via `ModuleList`, add patch embedding, positional encoding, and final normalization/projection. Each architecture has its own transformer file (e.g., `transformer_sd35.py`, `transformer_flux1.py`).
- **VAE decoders** implement the latent-to-pixel decoding using convolution layers, `GroupNorm`, upsampling, and model-specific parallelism. Video VAEs (Mochi, Wan2.2) use 3D convolutions with temporal decomposition.

### Level 4: Pipelines (`pipelines/`)

Pipelines orchestrate the complete inference flow:

1. **Mesh setup**: Create submeshes for CFG (classifier-free guidance) parallel execution.
2. **CCLManager initialization**: One `CCLManager` per submesh, managing semaphores and persistent buffers for all CCL operations.
3. **Encoder loading**: Load text encoders (CLIP, T5, UMT5, or Qwen2.5-VL) onto device, encode prompts, then deallocate encoder weights via `set_unload_set`.
4. **Transformer loading**: Load the DiT transformer, optionally from a weight cache.
5. **Denoising loop**: Run the scheduler's denoising steps. The first 1-2 iterations compile and capture a trace; subsequent iterations replay the trace for peak performance.
6. **VAE decoding**: Load the VAE decoder (potentially on a different submesh), decode latents to pixels, produce output images or video frames.

The pipeline's `set_unload_set` mechanism allows components to share device memory by specifying which modules must be deallocated before loading others. For example, the SD3.5 pipeline can swap between text encoders and the transformer on the same submesh.

## Data Flow: From Prompt to Image

The following shows the high-level data flow through an image generation pipeline (using SD3.5 as an example):

```
User prompt (text)
    |
    v
[CLIP Encoder] -----> pooled_projection (context embedding)
[T5 Encoder]   -----> prompt_embeds (sequence embedding)
    |
    | (encoders deallocated from device)
    v
[DiT Transformer] (loaded onto device)
    |
    |  Denoising loop (20-50 steps):
    |    1. Scheduler produces timestep + noisy latents
    |    2. Transformer forward:
    |       PatchEmbed -> [TransformerBlock x N] -> Final norm + projection
    |       Each TransformerBlock:
    |         AdaLN modulation -> Joint Attention -> Gate -> FeedForward -> Gate
    |    3. Scheduler step (update latents)
    |    4. Steps 2-3 replayed via trace after first compile
    |
    v
[VAE Decoder] (loaded, transformer may be deallocated)
    |
    v
Output image (PIL.Image)
```

For video models (Mochi, Wan2.2), the flow adds temporal dimensions: patch embedding operates in 3D, Conv3d layers handle temporal convolution, and the VAE decoder produces multiple frames.

## Files in This Chapter

- [`module_and_parameter.md`](./module_and_parameter.md) -- Deep dive into the `Module` and `Parameter` base classes that underpin every component in TT-DiT.
- [`comparison_with_ttnnmodule.md`](./comparison_with_ttnnmodule.md) -- Side-by-side comparison of TT-DiT's `Module` with TT-Symbiote's `TTNNModule`, highlighting architectural differences and equivalent patterns.

## Key Takeaways

- TT-DiT is a vertically integrated framework: every layer is a hand-written TTNN implementation, not a dispatch-intercepted PyTorch module.
- The four-level hierarchy (layers -> blocks -> models -> pipelines) provides clean separation of concerns, with `Module` and `Parameter` as the unifying abstractions.
- Six models are supported, sharing the same base infrastructure but differing in attention patterns, encoder stacks, and VAE architectures.
- Trace capture and replay in the denoising loop is critical for performance, since the same transformer forward pass runs 20-50 times per generation.

---

**Next:** [`module_and_parameter.md`](./module_and_parameter.md)
