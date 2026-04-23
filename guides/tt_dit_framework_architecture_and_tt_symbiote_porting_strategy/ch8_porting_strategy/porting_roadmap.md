# Porting Roadmap: Phased Plan for DiT Models in TT-Symbiote

## Prerequisites

- [Chapter 8 Index](./index.md): overview of the porting challenge.
- [`component_assessment.md`](./component_assessment.md): three-tier classification and dependency graph.
- [`model_prioritization.md`](./model_prioritization.md): SD3.5 as the first candidate, model ranking.
- [Chapter 2 -- Mapping to Symbiote](../ch2_parallelism_and_ccl/mapping_to_symbiote.md): CCL and parallelism gap analysis.
- [Chapter 5 -- Mapping to Symbiote Serving](../ch5_pipelines_and_serving/mapping_to_symbiote_serving.md): integration strategies (A/B/C).
- [Chapter 7 -- Integration Strategy](../ch7_tracing_and_performance/integration_strategy.md): tracing tiers (1/2/3).

---

## Overview

This file defines a five-phase porting roadmap that takes TT-DiT's diffusion transformer capabilities from their current standalone implementation to full integration within TT-Symbiote. Each phase has concrete deliverables, success criteria, estimated duration, and dependencies on prior phases.

The roadmap follows two guiding principles:

1. **Each phase produces a testable, independently valuable deliverable.** No phase requires subsequent phases to be useful.
2. **Infrastructure before features; single-device before multi-device.** Framework-level capabilities are built first so that model-specific work can proceed on a stable foundation.

---

## Phase 1: CCL and Parallelism Infrastructure

**Goal:** Bring TT-Symbiote's distributed computing infrastructure to parity with TT-DiT's requirements for multi-device DiT execution.

**Duration:** 4--6 weeks.

### Deliverables

| # | Deliverable | Description | Source Reference |
|---|---|---|---|
| 1.1 | **Async CCL in distributed linears** | Refactor `TTNNLinearIColShardedWRowSharded`, `TTNNLinearIColShardedWAllReduced`, and `TTNNLinearIReplicatedWColSharded` to call the existing `tt_all_reduce()`/`tt_all_gather()` helpers from `models/tt_transformers/tt/ccl.py` instead of raw synchronous `ttnn.reduce_scatter`/`ttnn.all_gather`. | [Ch2 mapping, Gap 2](../ch2_parallelism_and_ccl/mapping_to_symbiote.md) |
| 1.2 | **Persistent buffer cache** | Add a `PersistentBufferCache` to `TT_CCL` (or create a companion class) that stores and reuses output buffers keyed by `(shape, dim, axis)`. Implement ping-pong double-buffering. | [Ch2 CCL manager](../ch2_parallelism_and_ccl/ccl_manager.md) |
| 1.3 | **Semaphore reset** | Add `reset_global_semaphores()` to `TT_CCL` to zero semaphore state between trace captures. | [Ch2 CCL manager](../ch2_parallelism_and_ccl/ccl_manager.md) |
| 1.4 | **Configurable mesh axis and topology** | Make `cluster_axis` and `topology` constructor parameters on all distributed linear classes, defaulting to values from `DistributedConfig`. | [Ch2 mapping, Gap 6](../ch2_parallelism_and_ccl/mapping_to_symbiote.md) |
| 1.5 | **DiTDistributedConfig** | New config dataclass supporting `ParallelFactor` tuples for CFG-P, SP, and TP axes. Includes factory methods that select defaults from a mesh-shape lookup table (mirroring TT-DiT's pipeline `default_config` dictionaries). | [Ch2 mapping, Gap 3](../ch2_parallelism_and_ccl/mapping_to_symbiote.md) |
| 1.6 | **Submesh creation** | Add `create_submeshes()` integration to TT-Symbiote's device management, with per-submesh `DistributedConfig` instances and per-submesh `TT_CCL` managers. | [Ch2 mapping, Gap 4](../ch2_parallelism_and_ccl/mapping_to_symbiote.md) |
| 1.7 | **Shape-based CCL hyperparameters** | Add `get_ag_hyperparams()`/`get_rs_hyperparams()` that select `chunks_per_sync` and `num_workers_per_link` based on tensor shape, replacing the hardcoded defaults in the async CCL helpers. | [Ch2 mapping, Gap 7](../ch2_parallelism_and_ccl/mapping_to_symbiote.md) |

### Success Criteria

- Existing TT-Symbiote distributed linear tests continue to pass (backward compatibility).
- New unit tests demonstrate:
  - Async CCL operations (all-gather, reduce-scatter) with semaphore synchronization.
  - Persistent buffer cache hit/miss/eviction behavior.
  - Semaphore reset between trace capture windows.
  - Distributed linear execution on mesh axis 0 (not just hardcoded axis 1).
  - Submesh creation from a 2x4 mesh into two 1x4 submeshes.
- Performance benchmark: distributed linear throughput on T3K matches or exceeds pre-refactor throughput.

### Risks

| Risk | Mitigation |
|---|---|
| Async CCL refactor introduces regressions in existing LLM models | Run full LLM regression suite after each deliverable. Keep synchronous CCL as a fallback flag. |
| Persistent buffer cache causes memory fragmentation | Implement eviction policy with configurable maximum cache size. Profile memory usage on T3K. |
| `TT_CCL` extensions conflict with `tt_transformers` usage | Coordinate with `tt_transformers` team. Consider creating a shared CCL utility library. |

---

## Phase 2: Core DiT Layers as TTNNModules

**Goal:** Port the computational building blocks of DiT models (layers and blocks) as `TTNNModule` subclasses, validated against TT-DiT's layer-level unit tests.

**Duration:** 6--8 weeks.

**Dependencies:** Phase 1 (CCL infrastructure, configurable mesh axis, `DiTDistributedConfig`).

### Deliverables

| # | Deliverable | Description | Tier | Source Reference |
|---|---|---|---|---|
| 2.1 | **TTNNDiTLinear** | Extended `TTNNLinear` using `ttnn.experimental.minimal_matmul` with `MinimalMatmulConfig`. Weight transpose in `preprocess_weights_impl`. | 2 | [Ch3 index](../ch3_custom_layers_and_ops/index.md) |
| 2.2 | **TTNNDiTColParallelLinear** | Extended `TTNNLinearIReplicatedWColSharded` with FSDP support, configurable mesh axis, `minimal_matmul`, and activation fusion (SwiGLU, GELU). | 2 | [Ch2 parallel linears](../ch2_parallelism_and_ccl/parallel_linear_layers.md) |
| 2.3 | **TTNNDiTRowParallelLinear** | Extended `TTNNLinearIColShardedWRowSharded` with FSDP support, async reduce-scatter via `TT_CCL`, and configurable mesh axis. | 2 | [Ch2 parallel linears](../ch2_parallelism_and_ccl/parallel_linear_layers.md) |
| 2.4 | **TTNNDiTDistributedLayerNorm** | New module wrapping `ttnn.experimental.dit_layernorm_pre_allgather` / `post_allgather` with dynamic weight/bias support and reciprocal tensor cache. | 3 | [Ch3 normalization](../ch3_custom_layers_and_ops/normalization_layers.md) |
| 2.5 | **TTNNDiTDistributedRMSNorm** | New module wrapping `ttnn.experimental.wan_fused_rmsnorm_pre_allgather` / `post_allgather` with optional fused RoPE. | 2 | [Ch3 normalization](../ch3_custom_layers_and_ops/normalization_layers.md) |
| 2.6 | **TTNNDiTAttention** | New attention module calling `ttnn.transformer.joint_scaled_dot_product_attention` with dual-stream QKV, per-head RMSNorm, RoPE, and TP all-gather on output projections. | 3 | [Ch4 joint attention](../ch4_attention_and_transformer_blocks/joint_attention.md) |
| 2.7 | **TTNNDiTRingAttention** | Sequence-parallel variant of 2.6 using `ttnn.transformer.ring_joint_scaled_dot_product_attention` with CCL persistent buffers and semaphores. | 3 | [Ch4 joint attention](../ch4_attention_and_transformer_blocks/joint_attention.md) |
| 2.8 | **TTNNDiTTransformerBlock** | Block-level module combining: adaptive `TTNNDiTDistributedLayerNorm` (time-conditioned modulation), `TTNNDiTAttention`, `ParallelFeedForward`, and gated residual connections. | 3 | [Ch4 transformer block](../ch4_attention_and_transformer_blocks/transformer_block.md) |
| 2.9 | **TTNNPatchEmbed** | Patch embedding using unfolded `ttnn.linear` projection. | 2 | [Ch3 embeddings](../ch3_custom_layers_and_ops/index.md) |
| 2.10 | **TTNNGroupNorm** | New module wrapping `ttnn.group_norm` with input mask and weight/bias preprocessing. | 2 | [Ch3 normalization](../ch3_custom_layers_and_ops/normalization_layers.md) |
| 2.11 | **DiT embedding modules** | `TTNNTimesteps`, `TTNNSD35CombinedTimestepTextProjEmbeddings`, `TTNNCombinedTimestepGuidanceTextProjEmbeddings`. | 1--2 | [Ch3 embeddings](../ch3_custom_layers_and_ops/index.md) |

### Validation Strategy

Each module is validated in isolation against TT-DiT's corresponding unit test:

1. **Weight equivalence:** Load identical PyTorch weights into both the TT-DiT module (via `load_torch_state_dict`) and the TTNNModule (via `from_torch` + `preprocess_weights` + `move_to_device`). Verify that both produce the same TTNN tensor values.

2. **Forward pass equivalence:** For identical inputs (generated from TT-DiT's test fixtures), verify that the TT-DiT module's `forward()` output matches the TTNNModule's `forward()` output within the expected numerical tolerance:
   - BF16 elementwise tolerance: $|\text{actual} - \text{expected}| < 0.01$ for normalized outputs.
   - Attention output tolerance: PCC > 0.999 against the PyTorch reference.

3. **Multi-device equivalence:** For parallel modules (2.2, 2.3, 2.6, 2.7), run on a 1x4 submesh and verify output matches the single-device reference after all-gather/reduce-scatter.

### Success Criteria

- All Tier 1 and Tier 2 modules pass weight equivalence and forward pass equivalence tests.
- `TTNNDiTTransformerBlock` produces output with PCC > 0.999 against TT-DiT's `TransformerBlock` for SD3.5 configuration.
- Multi-device tests pass on both 1x4 (TP-only) and 2x4 (TP + SP) mesh configurations.
- All modules support TT-Symbiote's `from_torch` lifecycle and are compatible with `TracedRun`.

### Risks

| Risk | Mitigation |
|---|---|
| `minimal_matmul` shape restrictions | Fall back to `ttnn.linear` for shapes that `minimal_matmul` does not support. Implement shape validation in the module. |
| Numerical divergence in distributed norms | Use TT-DiT's reciprocal tensor cache and Welford algorithm implementation. Test against the TT-DiT reference. |
| Fused QKV weight interleaving is error-prone | Port TT-DiT's `_reshape_and_merge_qkv` logic verbatim into `preprocess_weights_impl`. Test with known weight values. |

---

## Phase 3: SD3.5 End-to-End Proof of Concept

**Goal:** Demonstrate a complete SD3.5 image generation pipeline running through TT-Symbiote, producing correct images from text prompts.

**Duration:** 4--6 weeks.

**Dependencies:** Phase 2 (all core DiT TTNNModules).

### Deliverables

| # | Deliverable | Description | Source Reference |
|---|---|---|---|
| 3.1 | **SD35Transformer as TTNNModule** | Compose `TTNNDiTTransformerBlock` x 38 with patch embedding, positional encoding, and final norm/projection into a single `SD35TransformerTTNN` module. Wrap in `TTNNLayerStack` for tracing. | [Ch4 transformer block](../ch4_attention_and_transformer_blocks/transformer_block.md), [Ch7 integration](../ch7_tracing_and_performance/integration_strategy.md) |
| 3.2 | **Weight loading from HuggingFace** | Implement `from_torch` factory for `SD35TransformerTTNN` that accepts a HuggingFace `SD3Transformer2DModel` and maps all weights through `preprocess_weights_impl`. | [Ch6 weight pipeline](../ch6_weight_loading/tt_dit_weight_pipeline.md), [Ch6 symbiote weight pipeline](../ch6_weight_loading/symbiote_weight_pipeline.md) |
| 3.3 | **Encoder integration** | Port CLIP and T5 encoders using Strategy B from [Ch5 mapping](../ch5_pipelines_and_serving/mapping_to_symbiote_serving.md): TT-Symbiote module replacement for encoders, native DiT transformer for denoising. | [Ch5 mapping](../ch5_pipelines_and_serving/mapping_to_symbiote_serving.md) |
| 3.4 | **Denoising loop with TracedRun** | Implement the denoising loop using Tier 1 tracing (TTNNLayerStack wrapping the transformer) from [Ch7 integration](../ch7_tracing_and_performance/integration_strategy.md). Validate that trace capture and replay produce correct results. | [Ch7 integration](../ch7_tracing_and_performance/integration_strategy.md) |
| 3.5 | **VAE decoder** | Port SD3.5's 2D VAE decoder using the Conv2d and GroupNorm modules from Phase 2. Can initially run on a single device. | [Ch3 convolution](../ch3_custom_layers_and_ops/convolution_layers.md) |
| 3.6 | **End-to-end pipeline** | Orchestration code that ties together encoding, denoising, and VAE decoding. Initially follows Strategy B (encoder via Symbiote dispatch, transformer via native TTNNModules). | [Ch5 mapping](../ch5_pipelines_and_serving/mapping_to_symbiote_serving.md) |
| 3.7 | **Correctness validation** | Compare generated images against TT-DiT's reference outputs for a fixed set of prompts and seeds. Quantitative metric: LPIPS < 0.05 and PSNR > 30 dB against the TT-DiT reference image. | -- |

### Integration Pattern

The Phase 3 pipeline follows Strategy B from [Ch5 mapping to serving](../ch5_pipelines_and_serving/mapping_to_symbiote_serving.md):

```
HuggingFace SD3 Pipeline (host)
    |
    +-- CLIP Encoder     [TT-Symbiote module replacement, NormalRun]
    +-- T5 Encoder       [TT-Symbiote module replacement, NormalRun]
    |
    +-- SD35 Transformer [TTNNDiTTransformerBlock x 38, TracedRun via TTNNLayerStack]
    |     |
    |     +-- Denoising loop (28 steps)
    |           Step 1: Warmup (normal forward)
    |           Step 2: Trace capture
    |           Steps 3-28: Trace replay
    |
    +-- VAE Decoder      [TTNNConv2d + TTNNGroupNorm, NormalRun]
    |
    v
Output image (PIL.Image)
```

### Success Criteria

- End-to-end image generation from text prompt produces visually correct images.
- Quantitative correctness: LPIPS < 0.05 and PSNR > 30 dB vs. TT-DiT reference for at least 10 test prompts.
- Traced denoising loop achieves throughput within 80% of TT-DiT's pipeline-level trace (Tier 1 target from [Ch7 integration](../ch7_tracing_and_performance/integration_strategy.md)).
- Pipeline runs on both single-device and multi-device (1x4, 2x4) configurations.

### Risks

| Risk | Mitigation |
|---|---|
| Weight transformation errors in `preprocess_weights_impl` | Test every module individually (Phase 2 validation) before composing. Use TT-DiT's weight cache as ground truth. |
| TracedRun incompatibility with DiT's fixed-shape execution | DiT models have fixed input shapes per resolution, which is ideal for TracedRun's signature-based cache. Verify shapes are consistent across denoising steps. |
| Encoder/transformer tensor format mismatch | Use `fast_unwrap_to_device()` at the boundary to extract raw `ttnn.Tensor` from `TorchTTNNTensor`. See [Ch5 tensor bridging](../ch5_pipelines_and_serving/mapping_to_symbiote_serving.md). |

---

## Phase 4: Pipeline Integration and Production Hardening

**Goal:** Upgrade the SD3.5 proof of concept to production quality: full parallelism, optimized tracing, memory management, and serving integration.

**Duration:** 4--6 weeks.

**Dependencies:** Phase 3 (working end-to-end pipeline).

### Deliverables

| # | Deliverable | Description |
|---|---|---|
| 4.1 | **CFG-parallel pipeline** | Implement submesh-based CFG parallelism: split mesh into conditional/unconditional submeshes, run denoising in parallel, combine results with guidance scale. |
| 4.2 | **Tier 2 tracing** | Upgrade from Tier 1 (TTNNLayerStack) to Tier 2 tracing (expanded `TracedDiTForward` encompassing embeddings + transformer + output projection). Target: within 95% of TT-DiT throughput. See [Ch7 integration, Tier 2](../ch7_tracing_and_performance/integration_strategy.md). |
| 4.3 | **Dynamic memory management** | Implement `unload_set` semantics in `TTNNModule` for encoder/transformer/VAE swapping on memory-constrained configurations. |
| 4.4 | **Weight caching** | Implement `.tensorbin`-compatible weight serialization/deserialization for faster model loading after first run. |
| 4.5 | **Performance profiling** | Comprehensive profiling using TT-Symbiote's `DispatchManager.save_stats_to_file`. Compare per-op latency against TT-DiT baseline. Identify and optimize bottlenecks. |
| 4.6 | **Multi-resolution support** | Validate pipeline at multiple resolutions (512x512, 768x768, 1024x1024). Verify TracedRun re-captures traces when resolution changes. |
| 4.7 | **Error handling and fallback** | Implement graceful fallback for unsupported configurations: automatic detection of mesh shapes that do not support the target parallelism, with clear error messages. |

### Success Criteria

- SD3.5 pipeline throughput within 95% of TT-DiT on the same hardware (T3K 2x4).
- CFG-parallel pipeline produces identical results to single-submesh sequential execution.
- Memory usage on T3K matches TT-DiT (encoder/transformer/VAE fit within device DRAM via dynamic swapping).
- Weight loading from cache is >10x faster than from-scratch HuggingFace loading.

---

## Phase 5: Additional Models

**Goal:** Port remaining models in priority order, leveraging the infrastructure established in Phases 1--4.

**Duration:** Variable per model (see [`model_prioritization.md`](./model_prioritization.md)).

**Dependencies:** Phases 1--4 (complete SD3.5 pipeline).

### 5A: Flux1 (2--3 weeks)

| Deliverable | Description |
|---|---|
| Single-stream transformer blocks | Port the 38 single-stream blocks that process only spatial tokens (no prompt stream). These are simpler than joint blocks. |
| Guidance embedding | Implement `TTNNCombinedTimestepGuidanceTextProjEmbeddings` (Tier 2). |
| No-CFG pipeline | Pipeline variant without submesh splitting. Single submesh with SP + TP. |
| Flux1-specific RoPE | Adapt RoPE frequency computation for Flux1's positional encoding scheme. |

### 5B: Motif (1--2 weeks)

| Deliverable | Description |
|---|---|
| `context_head_factors` | Add per-head scaling parameter to `TTNNDiTAttention`. Single `ttnn.mul` operation. |
| Motif pipeline | Clone SD3.5 pipeline with Motif-specific parallelism defaults. |
| Reference validation | Validate against `reference/motif/` PyTorch implementation. |

### 5C: Qwen-Image (4--6 weeks)

| Deliverable | Description |
|---|---|
| Qwen2.5-VL encoder | Port the vision-language encoder as a new `TTNNModule` subclass hierarchy. |
| QwenImage attention | Adapt `TTNNDiTAttention` for the Qwen-Image attention variant. |
| QwenImage pipeline | Pipeline with Qwen-specific encoding and generation flow. |

### 5D: Wan2.2 (8--12 weeks)

| Deliverable | Description |
|---|---|
| **Conv3d infrastructure** (Tier 3.5) | Port `ContextParallelConv3d` wrapping `ttnn.experimental.conv3d` with temporal context parallelism. |
| Cross-attention blocks | New `TTNNWanTransformerBlock` with self-attention + cross-attention (distinct from joint attention). |
| UMT5 encoder | Port the UMT5 text encoder. |
| 3D VAE decoder | Port Wan's temporal+spatial VAE with Conv3d layers. |
| Two-stage denoising | Implement boundary-ratio two-stage pipeline (not standard CFG). |
| Temporal parallelism | Extend `DiTDistributedConfig` with temporal parallelism axis. |

### 5E: Mochi (10--14 weeks)

| Deliverable | Description |
|---|---|
| MochiAttention | Port the hybrid attention variant unique to Mochi. |
| 3-axis VAE parallelism | Implement `MochiVAEParallelConfig` with independent time/height/width axes. |
| MochiPatchEmbed | Port the 3D patch embedding for video. |
| Sequential CFG | Pipeline with serial unconditional-then-conditional execution. |
| Mochi 3D VAE | Port the most complex VAE in TT-DiT with full 3D parallelism. |

---

## Timeline Summary

```
Phase 1: CCL & Parallelism Infrastructure        [Weeks 1--6]
    |
    v
Phase 2: Core DiT Layers as TTNNModules          [Weeks 5--14]  (overlaps with Phase 1 tail)
    |
    v
Phase 3: SD3.5 E2E Proof of Concept              [Weeks 13--20]
    |
    v
Phase 4: Production Hardening                     [Weeks 19--26]
    |
    +---> Phase 5A: Flux1                         [Weeks 25--28]
    |
    +---> Phase 5B: Motif                         [Weeks 28--30]
    |
    +---> Phase 5C: Qwen-Image                    [Weeks 30--36]
    |
    +---> Phase 5D: Wan2.2                        [Weeks 36--48]
    |
    +---> Phase 5E: Mochi                         [Weeks 48--62]
```

**Note:** Phases 2 and 3 can overlap (start Phase 3 transformer composition while finishing Phase 2 modules). Phases 4 and 5A can also overlap. The timeline assumes a single engineer; parallel staffing can compress the schedule significantly.

---

## Open Questions

The following questions should be resolved during or before Phase 1:

| # | Question | Impact | Resolution Path |
|---|---|---|---|
| 1 | **Should TT-Symbiote adopt `ttnn.experimental.minimal_matmul` as the default for all linear layers, or only for DiT?** | If adopted globally, LLM models benefit from configurable blocking. If DiT-only, two linear code paths must be maintained. | Benchmark `minimal_matmul` vs. `ttnn.linear` on representative LLM shapes. If performance is neutral or positive, adopt globally. |
| 2 | **Should the persistent buffer cache be global (per-device) or per-module?** | Global reduces memory but requires coordination. Per-module is simpler but may waste memory. | Start with per-module (simpler), migrate to global if memory pressure is observed. |
| 3 | **How should `DiTDistributedConfig` interact with the existing `DistributedConfig`?** | Inheritance, composition, or replacement. | Composition (wrap `DistributedConfig` with DiT-specific extensions) is the least disruptive. |
| 4 | **Should the `Tracer` utility from TT-DiT be adopted as a shared primitive?** | Using TT-DiT's `Tracer` class could simplify Tier 3 tracing without going through `TracedRun`'s module-level machinery. | Evaluate during Phase 3. If `TTNNLayerStack` (Tier 1) achieves >90% throughput, the `Tracer` utility may not be needed. |
| 5 | **What is the target hardware for initial deployment: T3K (8 devices) or TG (32 devices)?** | T3K requires less parallelism configuration. TG exercises the full 3-axis parallelism. | Target T3K first. Validate TG in Phase 4. |
| 6 | **Should encoder models (CLIP, T5) be ported via TT-Symbiote module replacement (Strategy B) or as native TTNNModules?** | Strategy B is faster to implement but creates a framework boundary. Native TTNNModules are more integrated but require more work. | Use Strategy B for Phase 3 (speed), consider native TTNNModules in Phase 4 if the boundary proves problematic. |

---

## Key Takeaways

1. **The five-phase plan follows a strict dependency chain** -- infrastructure (Phase 1), building blocks (Phase 2), proof of concept (Phase 3), hardening (Phase 4), expansion (Phase 5) -- with each phase producing an independently testable deliverable.

2. **Phases 1--3 represent the minimum viable porting effort** (approximately 20--26 weeks for one engineer). At the end of Phase 3, a working SD3.5 pipeline demonstrates that TT-Symbiote can run DiT models end-to-end. Phases 4 and 5 are incremental.

3. **The critical path runs through CCL infrastructure (Phase 1) to joint attention (Phase 2.6--2.8) to SD3.5 composition (Phase 3).** Any delays in Phase 1 CCL work directly delay every subsequent phase.

4. **Risk is concentrated in two areas:** weight transformation correctness (the `_prepare_torch_state` to `preprocess_weights_impl` translation is error-prone and model-specific) and multi-device CCL integration (async semaphore management is subtle and hard to debug). Both are mitigated by the per-module validation strategy in Phase 2.

5. **The long-term payoff is significant.** Once the DiT infrastructure exists in TT-Symbiote, each subsequent model requires 30--50% less effort than a standalone TT-DiT implementation, because TT-Symbiote's module lifecycle, dispatch debugging (SEL/DPL), profiling (`DispatchManager`), and tracing (`TracedRun`) infrastructure are shared across all models.

---

**End of guide.** Return to [Guide Index](../index.md)
