# Chapter 6 — Reuse Strategy for Existing Gemma 3 TTNN Modules

## Learning Objectives

After completing this chapter, you will be able to:

- Classify every existing Gemma 3 TTNN vision encoder module into one of three reuse categories: direct reuse, modification required, or new implementation
- Identify the specific code changes needed for each module that requires modification
- Estimate the engineering effort per module and in aggregate
- Plan sprint tasks using the reuse scorecard and dependency ordering
- Justify which modules to port first based on validation dependencies and risk

## Prerequisites

- Completion of [Chapter 2 — SigLIP vs. Gemma 4 Comparison](../ch02_siglip_vs_gemma4_comparison/index.md) (module mapping, config diff)
- Completion of [Chapter 3 — 2D Factored RoPE](../ch03_2d_factored_rope/index.md) (RoPE gap analysis and implementation strategies)
- Completion of [Chapter 4 — Patch Embedding and Adaptive Pooling](../ch04_patch_embedding_and_pooling/index.md) (patch embedding and pooling implementation options)
- Familiarity with the Gemma 3 TTNN codebase at `models/demos/multimodal/gemma3/tt/`

## Chapter Contents

| File | Topic |
|------|-------|
| [`direct_reuse_modules.md`](./direct_reuse_modules.md) | Modules reusable with no or minimal changes: MLP, RMSNorm |
| [`modification_required_modules.md`](./modification_required_modules.md) | Modules needing targeted modifications: encoder block (sandwich norm, RoPE args), attention (2D RoPE), multimodal projector (adaptive pooling), patch embedding (patch_size change), model config (7 new params), checkpoint loading (key mapping rewrite) |
| [`new_implementation_modules.md`](./new_implementation_modules.md) | Modules to build from scratch: 2D RoPE, 2D position embedding, variable-resolution preprocessor, vision pooler |

## Overview

The Gemma 3 TTNN vision encoder codebase under `models/demos/multimodal/gemma3/tt/` contains approximately a dozen modules covering patch embedding, attention, MLP, normalization, projection, configuration, and checkpoint loading. Because Gemma 4 shares the same core dimensions (`hidden_size=1152`, `intermediate_size=4304`, `num_attention_heads=16`, `head_dim=72`) and the same 27-layer encoder structure, a substantial fraction of this code can be carried forward.

This chapter provides a file-by-file reuse plan. Each module is classified, the required changes are enumerated, and effort is estimated. The goal is to give engineering leads a concrete task list they can assign to sprints.

## Reuse Scorecard

The following table summarizes the reuse classification for every module involved in the Gemma 4 vision encoder port. Detailed analysis for each category is in the linked sub-pages.

| Gemma 3 TTNN File | Gemma 4 Target File | Reuse Class | Effort | Details |
|--------------------|---------------------|-------------|--------|---------|
| `gemma_image_mlp.py` | `gemma4_vision_mlp.py` | **Direct reuse** | < 1 day | [direct_reuse_modules.md](./direct_reuse_modules.md#gemma_image_mlppy) |
| `gemma_vision_rmsnorm.py` | `gemma4_vision_rmsnorm.py` | **Direct reuse** | < 1 day | [direct_reuse_modules.md](./direct_reuse_modules.md#gemma_vision_rmsnormpy) |
| `gemma_image_block.py` | `gemma4_vision_encoder_layer.py` | **Modify** | 1-2 days | [modification_required_modules.md](./modification_required_modules.md#gemma_image_blockpy) |
| `model_config.py` | `gemma4_model_config.py` | **Modify** | 1 day | [modification_required_modules.md](./modification_required_modules.md#model_configpy) |
| `load_checkpoints.py` | `gemma4_load_checkpoints.py` | **Modify** | 1-2 days | [modification_required_modules.md](./modification_required_modules.md#load_checkpointspy) |
| `gemma_image_attention.py` | `gemma4_vision_attention.py` | **Modify** | 2-3 days | [modification_required_modules.md](./modification_required_modules.md#gemma_image_attentionpy) |
| `multi_modal_projector.py` | `gemma4_multimodal_embedder.py` | **Modify** | 1-2 days | [modification_required_modules.md](./modification_required_modules.md#multi_modal_projectorpy) |
| `gemma_conv2d_patch.py` | `gemma4_vision_patch_embedder.py` | **Modify** | 2-3 days | [modification_required_modules.md](./modification_required_modules.md#gemma_conv2d_patchpy) |
| *(none)* | `gemma4_vision_rope.py` | **New** | 2-3 days | [new_implementation_modules.md](./new_implementation_modules.md#2d-rope-module) |
| *(none)* | `gemma4_vision_position_embedding.py` | **New** | 1-2 days | [new_implementation_modules.md](./new_implementation_modules.md#2d-learned-position-embedding) |
| *(none)* | `gemma4_variable_resolution.py` | **New** | 1-2 days | [new_implementation_modules.md](./new_implementation_modules.md#variable-resolution-image-preprocessor) |
| *(none)* | `gemma4_vision_pooler.py` | **New** | 2-3 days | [new_implementation_modules.md](./new_implementation_modules.md#vision-pooler-module) |

### Aggregate Effort Summary

| Category | Module Count | Estimated Effort | Share of Total Codebase |
|----------|-------------|-----------------|------------------------|
| **Direct reuse** | 2 | < 1 day | ~15% |
| **Modification required** | 6 | 8-13 days | ~50% |
| **New implementation** | 4 | 6-10 days | ~35% |
| **Total** | 12 | **14-23 days** (3-5 engineer-weeks) | 100% |

> **Tip:** The effort estimates cover initial implementation and basic PCC validation against the CPU reference. Performance optimization (sharding tuning, op fusion, tracing) is not included and may add 1-2 additional weeks. See [Chapter 7 — Implementation Roadmap](../ch07_implementation_roadmap/index.md) for the complete phased timeline.

### Dependency Order

Modules should be ported in this order to enable incremental validation at each step:

```
 1. gemma4_vision_rmsnorm.py              (no dependencies)
 2. gemma4_vision_mlp.py                  (depends on: RMSNorm for weight verification)
 3. gemma4_vision_rope.py                 (no TTNN dependencies; validates against HF)
 4. gemma4_vision_position_embedding.py   (no TTNN dependencies; validates against HF)
 5. gemma4_vision_attention.py            (depends on: RMSNorm + RoPE)
 6. gemma4_vision_encoder_layer.py        (depends on: attention + MLP + RMSNorm)
 7. gemma4_vision_patch_embedder.py       (depends on: position embedding)
 8. gemma4_vision_pooler.py               (depends on: position IDs, trivially constructed or from preprocessor)
 9. gemma4_multimodal_embedder.py         (depends on: pooler + RMSNorm + linear)
10. gemma4_variable_resolution.py         (host-only; can be validated independently)
```

Items 1-2 and 3-4 can be worked in parallel since they have no mutual dependencies.

## Reading Order

Start with [`direct_reuse_modules.md`](./direct_reuse_modules.md) to understand the baseline of code that transfers with minimal effort. Then read [`modification_required_modules.md`](./modification_required_modules.md) for the modules that need targeted changes. Finally, [`new_implementation_modules.md`](./new_implementation_modules.md) covers the modules that must be built from scratch.
