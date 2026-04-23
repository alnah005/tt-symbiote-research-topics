# TT-DiT Framework Architecture and TT-Symbiote Porting Strategy

A comprehensive guide for Tenstorrent engineers evaluating whether and how TT-DiT's generative-model components can be ported into TT-Symbiote's unified serving infrastructure.

## Chapters

1. [**Architecture Overview**](./ch1_architecture_overview/index.md) -- TT-DiT codebase layout, `Module`/`Parameter` base classes, supported models, and comparison with TT-Symbiote's `TTNNModule`.

2. [**Parallelism and CCL Infrastructure**](./ch2_parallelism_and_ccl/index.md) -- 3-axis parallelism (CFG, sequence, tensor), `CCLManager`, parallel linear layers, and mapping to TT-Symbiote's `DistributedConfig`.

3. [**Custom Layers and TTNN Operations**](./ch3_custom_layers_and_ops/index.md) -- Normalization, linear, feedforward, convolution, and embedding layers with their `ttnn.experimental.*` dependencies.

4. [**Joint Attention and Transformer Blocks**](./ch4_attention_and_transformer_blocks/index.md) -- Joint attention mechanism, adaptive layer normalization, transformer block structure, and comparison with TT-Symbiote's attention modules.

5. [**End-to-End Pipelines and Model Registration**](./ch5_pipelines_and_serving/index.md) -- Pipeline lifecycle, `PipelineTrace`, memory management, and integration strategies with TT-Symbiote's serving infrastructure.

6. [**Weight Loading and Preprocessing**](./ch6_weight_loading/index.md) -- TT-DiT's `_prepare_torch_state` pipeline vs. TT-Symbiote's three-phase `from_torch`/`preprocess`/`move` lifecycle.

7. [**Tracing and Performance**](./ch7_tracing_and_performance/index.md) -- Pipeline-level vs. module-level tracing, the TTNN trace primitive, and recommended integration strategy.

8. [**Porting Strategy and Model Prioritization**](./ch8_porting_strategy/index.md) -- Component assessment, model ranking, and a five-phase porting roadmap.

## How to Use This Guide

| Goal | Recommended Path |
|---|---|
| Understand TT-DiT's architecture | Ch 1 -> Ch 2 -> Ch 3 -> Ch 4 |
| Evaluate porting feasibility | Ch 1 -> Ch 8 (component assessment + roadmap) |
| Understand parallelism differences | Ch 2 ([`mapping_to_symbiote.md`](./ch2_parallelism_and_ccl/mapping_to_symbiote.md)) |
| Understand attention differences | Ch 4 ([`comparison_with_symbiote_attention.md`](./ch4_attention_and_transformer_blocks/comparison_with_symbiote_attention.md)) |
| Plan a specific model port | Ch 8 ([`model_prioritization.md`](./ch8_porting_strategy/model_prioritization.md)) -> Ch 5 (pipelines) |
| Understand weight loading integration | Ch 6 ([`symbiote_weight_pipeline.md`](./ch6_weight_loading/symbiote_weight_pipeline.md)) |
| Understand tracing integration | Ch 7 ([`integration_strategy.md`](./ch7_tracing_and_performance/integration_strategy.md)) |

## Quick Reference

| Concept | TT-DiT | TT-Symbiote | Where Covered |
|---|---|---|---|
| Base module class | `Module` | `TTNNModule` | [Ch 1](./ch1_architecture_overview/comparison_with_ttnnmodule.md) |
| Parallelism config | `DiTParallelConfig` | `DistributedConfig` | [Ch 2](./ch2_parallelism_and_ccl/mapping_to_symbiote.md) |
| CCL management | `CCLManager` | `TT_CCL` + helpers | [Ch 2](./ch2_parallelism_and_ccl/ccl_manager.md) |
| Matmul op | `ttnn.experimental.minimal_matmul` | `ttnn.linear` | [Ch 3](./ch3_custom_layers_and_ops/ttnn_experimental_ops.md) |
| Attention | Joint SDPA (spatial + prompt) | Causal SDPA (paged KV cache) | [Ch 4](./ch4_attention_and_transformer_blocks/comparison_with_symbiote_attention.md) |
| Weight loading | `_prepare_torch_state` -> `load_torch_tensor` | `from_torch` -> `preprocess_weights_impl` | [Ch 6](./ch6_weight_loading/index.md) |
| Tracing | Pipeline-level `Tracer` | Module-level `TracedRun` | [Ch 7](./ch7_tracing_and_performance/integration_strategy.md) |

## Prerequisites

- Familiarity with PyTorch and TTNN fundamentals (Wormhole, MeshDevice, ttnn ops)
- Working knowledge of TT-Symbiote's `TTNNModule`, dispatcher, and `DistributedConfig`
- Conceptual understanding of diffusion transformer models (denoising loop, text conditioning, VAE decoding)

## Source Code Locations

- **TT-DiT**: `models/tt_dit/` in the tt-metal repository
- **TT-Symbiote**: `models/experimental/tt_symbiote/` in the tt-metal repository

## Additional Resources

- [Plan](./plan.md) -- Original guide plan with audience, chapter descriptions, conventions, and cross-chapter dependencies.
