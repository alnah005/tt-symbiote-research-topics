# Model Prioritization: Porting Order for Six TT-DiT Models

## Prerequisites

- [Chapter 8 Index](./index.md): overview of the porting challenge.
- [`component_assessment.md`](./component_assessment.md): three-tier classification of components and the dependency graph.
- [Chapter 1 -- Architecture Overview](../ch1_architecture_overview/index.md): the six supported models and their architectural differences.
- [Chapter 4 -- Attention and Transformer Blocks](../ch4_attention_and_transformer_blocks/index.md): attention patterns across models.
- [Chapter 5 -- Pipelines and Serving](../ch5_pipelines_and_serving/index.md): pipeline complexity per model.

---

## Overview

TT-DiT supports six generative models. This file ranks them by porting difficulty from easiest to hardest, recommends which to port first, and provides a rationale grounded in the component assessment and architectural analysis from preceding chapters.

## Ranking Criteria

Each model is evaluated on five dimensions:

| Criterion | Description | Weight |
|---|---|---|
| **Tier 3 dependencies** | Number of new infrastructure areas required beyond the shared core | High |
| **Attention complexity** | Joint, cross, hybrid, or standard attention; sequence parallelism requirements | High |
| **VAE complexity** | 2D spatial vs. 3D spatial+temporal; Conv3d requirements | Medium |
| **Encoder complexity** | Number and type of text/vision encoders | Medium |
| **Test coverage** | Existence and maturity of TT-DiT test suites for validation | Low |

## Model Rankings

### 1. Stable Diffusion 3.5 (Recommended First Candidate)

| Dimension | Assessment |
|---|---|
| **Type** | Text-to-image (1024x1024) |
| **Transformer** | `SD35Transformer2DModel` -- 38 `TransformerBlock` layers |
| **Attention** | Standard joint attention (spatial + prompt via `joint_scaled_dot_product_attention`) |
| **Encoders** | CLIP x2 + T5 (well-understood, standard HuggingFace models) |
| **VAE** | 2D spatial VAE decoder (`vae_sd35.py`) -- Conv2d + GroupNorm, no Conv3d |
| **CFG** | CFG-parallel via submeshes (standard dual-pass) |
| **Parallelism** | TP + optional SP on T3K (2x4); TP + SP on TG (4x8) |
| **Test suite** | Most comprehensive in TT-DiT: unit tests for all layers, block tests, model tests, E2E pipeline tests |
| **Unique requirements** | None beyond the shared Tier 3 infrastructure |

**Why first:**
- SD3.5 exercises the **core infrastructure** (joint attention, adaptive LayerNorm, parallel linears, CCL) without any model-specific additions beyond what every DiT model needs.
- It uses **only 2D operations** -- no Conv3d, no temporal dimension, no video-specific parallelism. This eliminates the entire Tier 3.5 (Conv3d) infrastructure area.
- Its **attention pattern is the canonical DiT joint attention** that all image models share. Porting SD3.5's attention directly enables Flux1 and Motif with minor modifications.
- The CLIP + T5 encoder stack is the **most commonly used** across TT-DiT models and across the broader HuggingFace ecosystem. TT-Symbiote integration for these encoders has the highest reuse value.
- The **comprehensive test suite** provides a reference for correctness validation at every level (layer, block, model, pipeline).
- SD3.5 is the best-documented model in TT-DiT (see `models/StableDiffusion35.md`), reducing the risk of implementation misunderstandings.

**Estimated effort after infrastructure is complete:** 4--6 weeks for full pipeline integration.

---

### 2. Flux1

| Dimension | Assessment |
|---|---|
| **Type** | Text-to-image (1024x1024) |
| **Transformer** | `Flux1Transformer` -- 19 joint blocks + 38 single blocks |
| **Attention** | Joint attention (same kernel as SD3.5) but with single-stream blocks that do not process prompt tokens |
| **Encoders** | CLIP + T5 (same as SD3.5, fewer CLIP models) |
| **VAE** | Shared VAE architecture with SD3.5 |
| **CFG** | No CFG (uses guidance embedding instead of dual-pass) |
| **Parallelism** | TP + SP (no CFG-P submeshes needed) |
| **Test suite** | Moderate coverage |
| **Unique requirements** | Guidance embedding, single-stream blocks, different RoPE handling |

**Why second:**
- Flux1 shares ~80% of its infrastructure with SD3.5 (same joint attention kernel, same encoder stack, same VAE).
- The primary difference is the **dual-block architecture**: 19 joint blocks (identical to SD3.5's `TransformerBlock`) followed by 38 single-stream blocks (which process only spatial tokens, ignoring prompt tokens after the joint blocks). The single-stream blocks are simpler than the joint blocks.
- **No CFG parallelism** simplifies the pipeline -- only one submesh is needed, with no conditional/unconditional split.
- The **guidance embedding** replaces CFG's dual pass with a learned embedding that modulates the denoising step. This is a straightforward addition to the embedding layer (a new `CombinedTimestepGuidanceTextProjEmbeddings` module).
- Different RoPE frequency computation, but the same `_apply_rope` TTNN call.

**Incremental effort over SD3.5:** 2--3 weeks.

---

### 3. Motif

| Dimension | Assessment |
|---|---|
| **Type** | Text-to-image (1024x1024) |
| **Transformer** | `MotifTransformer` -- structurally similar to SD3.5 |
| **Attention** | Joint attention with `context_head_factors` (per-head scaling for context tokens) |
| **Encoders** | CLIP + T5 (same stack) |
| **VAE** | Shared VAE architecture |
| **CFG** | CFG-parallel via submeshes |
| **Parallelism** | TP + SP (same as SD3.5) |
| **Test suite** | Moderate coverage, reference implementation available in `reference/motif/` |
| **Unique requirements** | `context_head_factors` per-head scaling |

**Why third:**
- Motif is architecturally the closest to SD3.5, differing primarily in the **`context_head_factors`** mechanism -- a per-head multiplicative scaling applied to the context (prompt) query before attention. This is a single additional `ttnn.mul` operation.
- The Motif pipeline follows the same lifecycle as SD3.5 (CFG-parallel, CLIP + T5, 2D VAE).
- The `reference/motif/` directory provides a PyTorch reference implementation for validation.

**Incremental effort over Flux1:** 1--2 weeks.

---

### 4. Qwen-Image

| Dimension | Assessment |
|---|---|
| **Type** | Text-to-image (1024x1024) |
| **Transformer** | `QwenImageTransformer` -- extends standard `TransformerBlock` with joint attention |
| **Attention** | Standard DiT joint attention (reuses `blocks/attention.py`) |
| **Encoders** | Qwen2.5-VL vision-language encoder (unique to this model) |
| **VAE** | 2D spatial VAE (similar to SD3.5) |
| **CFG** | Model-specific |
| **Parallelism** | TP-only |
| **Test suite** | Limited coverage |
| **Unique requirements** | Qwen2.5-VL encoder porting |

**Why fourth:**
- The **Qwen2.5-VL encoder** is unique to this model and would require a dedicated porting effort. It is a vision-language model with its own embedding and attention architecture, adding a full encoder porting task on top of the transformer port.
- The transformer reuses the standard `TransformerBlock` and joint attention from `blocks/transformer_block.py`, so the core DiT infrastructure from SD3.5/Flux1/Motif carries over directly.
- The 2D spatial VAE keeps this model in the "image-only" category, avoiding Conv3d requirements.
- Limited test coverage means more validation effort.

**Estimated effort:** 4--6 weeks (primarily encoder porting).

---

### 5. Wan2.2 (Text-to-Video)

| Dimension | Assessment |
|---|---|
| **Type** | Text-to-video (variable resolution) |
| **Transformer** | `WanTransformer` -- cross-attention blocks (not joint attention) |
| **Attention** | Cross-attention (`WanTransformerBlock`) with separate self-attention for spatial tokens and cross-attention to text tokens. Uses `DistributedRMSNorm` with fused RoPE. |
| **Encoders** | UMT5 encoder (unique to this model) |
| **VAE** | 3D spatial+temporal VAE with `Conv3d` layers and temporal decomposition |
| **CFG** | Two-stage denoising with boundary ratio (not standard CFG-P) |
| **Parallelism** | TP + SP + temporal parallelism |
| **Test suite** | Moderate coverage |
| **Unique requirements** | Conv3d (Tier 3.5), temporal parallelism, cross-attention blocks, UMT5 encoder, two-stage denoising, `WanTimeTextImageEmbedding` |

**Why fifth:**
- Wan2.2 introduces the **first video-specific requirements**: Conv3d in the VAE, temporal parallelism across the sequence, and 3D patch embedding.
- Its attention pattern is **cross-attention** (spatial self-attention + text cross-attention), which is architecturally distinct from the joint attention used by SD3.5/Flux1/Motif. This requires a new attention module variant.
- The **UMT5 encoder** is unique and requires a separate porting effort.
- The **two-stage denoising** with boundary ratio is a pipeline-level feature that differs from standard CFG and would need custom pipeline logic.
- The 3D VAE requires the **Conv3d infrastructure** (Tier 3.5 in the component assessment), which is deferrable for image models but mandatory here.

**Estimated effort:** 8--12 weeks.

---

### 6. Mochi (Text-to-Video, Highest Complexity)

| Dimension | Assessment |
|---|---|
| **Type** | Text-to-video (848x480, 168 frames) |
| **Transformer** | `MochiTransformer` -- hybrid attention (`MochiAttention`) with asymmetric spatial/prompt handling |
| **Attention** | Hybrid: spatial tokens use joint attention, but with a unique `MochiAttention` class that differs from standard DiT joint attention |
| **Encoders** | T5 (same as SD3.5, but different integration pattern) |
| **VAE** | Most complex VAE in TT-DiT: 3D spatial+temporal with `MochiVAEParallelConfig` (`time_parallel`, `h_parallel`, `w_parallel`) |
| **CFG** | Sequential (unconditional then conditional, not parallel) |
| **Parallelism** | TP + SP + temporal parallelism + 3-axis VAE parallelism |
| **Test suite** | Moderate coverage |
| **Unique requirements** | `MochiAttention` (unique attention variant), 3-axis VAE parallelism, `MochiPatchEmbed`, sequential CFG, `conv3d.py` utilities |

**Why last:**
- Mochi has the **most complex VAE** in TT-DiT: a 3D VAE with independent parallelism across time, height, and width dimensions. The `MochiVAEParallelConfig` introduces a third parallelism axis beyond what any other model requires.
- The `MochiAttention` class is **unique to this model** -- it implements a hybrid attention pattern that is distinct from both standard joint attention and Wan's cross-attention.
- **Sequential CFG** (running unconditional then conditional predictions in series rather than in parallel across submeshes) means the CFG-P infrastructure built for SD3.5 does not apply. A different pipeline control flow is needed.
- The 3-axis VAE parallelism requires extending the parallelism infrastructure beyond even what the transformer needs.

**Estimated effort:** 10--14 weeks.

---

## Summary Rankings

| Rank | Model | Type | Incremental Effort | Cumulative Effort | Key Blocker |
|------|-------|------|-------------------|-------------------|-------------|
| 1 | **SD3.5** | Image | 4--6 weeks | 4--6 weeks | Tier 3 core infrastructure |
| 2 | **Flux1** | Image | 2--3 weeks | 6--9 weeks | Single-stream blocks, guidance embedding |
| 3 | **Motif** | Image | 1--2 weeks | 7--11 weeks | `context_head_factors` |
| 4 | **Qwen-Image** | Image | 4--6 weeks | 11--17 weeks | Qwen2.5-VL encoder |
| 5 | **Wan2.2** | Video | 8--12 weeks | 19--29 weeks | Conv3d, temporal parallelism |
| 6 | **Mochi** | Video | 10--14 weeks | 29--43 weeks | 3-axis VAE parallelism, MochiAttention |

**Note:** The "Incremental Effort" column assumes the Tier 3 infrastructure (17--26 weeks) is already complete. The "Cumulative Effort" column reflects per-model work only, excluding shared infrastructure.

---

## Infrastructure Reuse Across Models

The following matrix shows which Tier 3 infrastructure areas are used by each model:

| Infrastructure | SD3.5 | Flux1 | Motif | Qwen-Image | Wan2.2 | Mochi |
|---|---|---|---|---|---|---|
| CCL extensions (3.1) | Required | Required | Required | Required | Required | Required |
| Multi-axis parallelism (3.2) | Required | Required | Required | Partial | Required | Required |
| Joint attention (3.3) | Required | Required | Required | Modified | No (cross-attn) | Modified |
| Adaptive LayerNorm (3.4) | Required | Required | Required | Required | Modified | Modified |
| Conv3d (3.5) | No | No | No | No | Required | Required |
| Pipeline orchestration (3.6) | Required | Required | Required | Required | Required | Required |

This confirms the strategy of building SD3.5 first: it exercises the maximum number of shared infrastructure areas while avoiding the video-specific Conv3d requirement.

---

## Recommendation

**Port SD3.5 first.** It provides the highest ratio of infrastructure validation to model-specific effort. Every Tier 3 infrastructure area except Conv3d is exercised by SD3.5, and the test suite provides the most comprehensive correctness baseline.

After SD3.5, port **Flux1 and Motif** in quick succession -- they share >80% of the infrastructure and differ primarily in pipeline-level features. This establishes TT-Symbiote as a viable platform for all three image generation models.

Defer **video models (Wan2.2, Mochi)** until the image model infrastructure is proven and Conv3d support is prioritized based on product requirements.

**Qwen-Image** can be ported at any point after SD3.5, but the Qwen2.5-VL encoder porting effort makes it less efficient as a second candidate than Flux1 or Motif.

---

## Key Takeaways

1. **SD3.5 is the optimal first candidate** because it exercises the broadest set of shared infrastructure (joint attention, adaptive LayerNorm, CCL, multi-axis parallelism) while requiring zero model-specific Tier 3 work.

2. **Image models (SD3.5, Flux1, Motif) share >80% of their infrastructure.** Porting all three is roughly 2x the effort of porting one, not 3x, because the incremental work for Flux1 and Motif is small once SD3.5 is complete.

3. **Video models (Wan2.2, Mochi) represent a qualitative step up in complexity.** Conv3d, temporal parallelism, and model-specific attention variants each require dedicated infrastructure that image models do not need. They should be deferred to a second phase.

4. **The encoder stack is a per-model cost.** CLIP + T5 (SD3.5, Flux1, Motif) can be ported once and shared. Qwen2.5-VL and UMT5 each require independent work.

5. **Sequential porting allows each model to validate the infrastructure built for its predecessor.** SD3.5 validates the core; Flux1 validates no-CFG pipelines; Motif validates per-head scaling; each subsequent model extends the validated surface area.

---

**Next:** [`porting_roadmap.md`](./porting_roadmap.md)
