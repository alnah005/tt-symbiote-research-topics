# Chapter 5: End-to-End Pipelines and Model Registration

## Overview

TT-DiT pipelines are the top-level orchestrators that tie together every component discussed in the preceding chapters -- text encoders, DiT transformers, VAE decoders, schedulers, and parallelism infrastructure -- into a single callable object that converts a text prompt into an image or video. This chapter examines the six pipeline classes shipped with TT-DiT, explains their common lifecycle, and then contrasts this architecture with TT-Symbiote's model-serving infrastructure to evaluate integration strategies.

## The Six Pipeline Classes

| Pipeline | File | Modality | Text Encoders | CFG Strategy | Tracing |
|---|---|---|---|---|---|
| **StableDiffusion3Pipeline** | [`stable_diffusion_35_large/`](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_dit/pipelines/stable_diffusion_35_large/) | Image | CLIP x2 + T5 | CFG-parallel via submeshes | Manual `PipelineTrace` |
| **Flux1Pipeline** | [`flux1/`](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_dit/pipelines/flux1/) | Image | CLIP + T5 | No CFG (guidance embedding) | Manual `PipelineTrace` |
| **MotifPipeline** | [`motif/`](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_dit/pipelines/motif/) | Image | CLIP + T5 | CFG-parallel via submeshes | Manual `PipelineTrace` |
| **MochiPipeline** | [`mochi/`](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_dit/pipelines/mochi/) | Video | T5 | Sequential CFG (uncond then cond) | Not yet traced |
| **WanPipeline** | [`wan/`](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_dit/pipelines/wan/) | Video | UMT5 | Two-stage denoising (boundary ratio) | Uses `cache.load_model` with dynamic loading |
| **QwenImagePipeline** | [`qwenimage/`](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_dit/pipelines/qwenimage/) | Image | Qwen-specific | -- | -- |

Despite their differences, every pipeline follows a shared lifecycle pattern described below.

## Common Pipeline Lifecycle

Every TT-DiT pipeline progresses through five stages:

```
create_pipeline()          -- static factory: picks parallel config for mesh shape
    |
    v
__init__()                 -- loads torch checkpoints, creates TT-NN models,
    |                         loads weights via cache.load_model()
    v
prepare() / warmup         -- pre-allocates buffers, may run a warmup forward pass
    |
    v
__call__() / run_single_prompt()
    |-- 1. Text encoding    (CLIP, T5, UMT5 on encoder submesh)
    |-- 2. Latent prep       (random noise + patchify/pack)
    |-- 3. Denoising loop    (N steps of transformer + scheduler step)
    |-- 4. VAE decode        (latents -> pixels, possibly on a different submesh)
    |-- 5. Post-process      (VaeImageProcessor / VideoProcessor -> PIL/tensor)
    v
output (List[Image] or video frames)
```

### Key Shared Mechanisms

1. **`create_pipeline()` factory** -- Each pipeline defines a static factory that selects parallelism configurations (TP, SP, CFG factors and axes) from a lookup table keyed by mesh shape. This removes the need for callers to understand internal parallelism details. See [Chapter 2](../ch2_parallelism_and_ccl/index.md) for the underlying `DiTParallelConfig` structure.

2. **`cache.load_model()`** -- Every pipeline uses the centralized weight-caching system (detailed in [`pipeline_anatomy.md`](./pipeline_anatomy.md) and Chapter 6) to load transformer, encoder, and VAE weights. The cache falls back to PyTorch state dicts when no cached tensors exist.

3. **`PipelineTrace` dataclass** -- Image-generation pipelines (SD3.5, Flux1, Motif) define a `PipelineTrace` that records the device-resident input/output tensor handles for the denoising step. This enables `ttnn.execute_trace()` replay on subsequent steps. Video pipelines (Mochi, Wan) do not yet use this pattern due to the complexity of their 3D-spatial inputs.

4. **Multi-submesh orchestration** -- Pipelines that use CFG parallelism (SD3.5, Motif) split the device mesh into submeshes and run conditional and unconditional predictions simultaneously. Non-CFG pipelines (Flux1) use a single submesh but may still create the submesh abstraction for future batching support.

5. **Dynamic model loading** -- When device memory is constrained, pipelines like Wan and Mochi use `set_unload_set()` and `reload_dit_model` flags to swap models in and out of device memory between pipeline stages. This is coordinated through the `Module.unload_set` mechanism.

## Chapter Contents

| File | Description |
|---|---|
| [`pipeline_anatomy.md`](./pipeline_anatomy.md) | Detailed walkthrough of `StableDiffusion3Pipeline` as the canonical example: constructor flow, `PipelineTrace`, text encoding, denoising loop, VAE decoding, memory management, and weight caching. |
| [`mapping_to_symbiote_serving.md`](./mapping_to_symbiote_serving.md) | How TT-Symbiote's module replacement pattern, run modes, and dispatch infrastructure compare to TT-DiT's pipeline architecture. Integration strategies with concrete trade-offs. |

## Prerequisites

Before reading this chapter, you should be familiar with:

- [Chapter 1 -- Architecture Overview](../ch1_architecture_overview/index.md): Framework directory structure and component taxonomy
- [Chapter 2 -- Parallelism and CCL](../ch2_parallelism_and_ccl/index.md): `DiTParallelConfig`, submeshes, `CCLManager`
- [Chapter 3 -- Custom Layers and Ops](../ch3_custom_layers_and_ops/index.md): `Module` base class, `Parameter`, weight loading
- [Chapter 4 -- Attention and Transformer Blocks](../ch4_attention_and_transformer_blocks/index.md): Transformer model structures that pipelines invoke

---

**Next:** [`pipeline_anatomy.md`](./pipeline_anatomy.md)
