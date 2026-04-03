# Gemma 4 31B Architecture and TTNN Module Mapping

This guide provides a complete architectural reference and TTNN module mapping for Gemma 4 31B (60 layers, hidden_size=5376, ~30.7B params) with heterogeneous attention (50 sliding-window + 10 global layers), targeting deployment on the T3K 1x8 Wormhole mesh with TP=8. It is written for ML systems engineers and kernel developers on the TT-NN / tt-symbiote stack who need to implement `TTNNModule` subclasses for every submodule in the Gemma 4 31B decoder.

## Prerequisites

**Required:**
- Familiarity with transformer decoder architectures (MHA, GQA, RMSNorm, FFN)
- Working knowledge of TTNN tensor operations, memory configs, and program configs
- Experience with `TTNNModule` authoring in tt-symbiote (module replacement, `forward` signatures)
- Basic understanding of T3K device topology (8 Wormhole chips, 1x8 mesh, Ethernet links)
- Exposure to paged KV cache concepts and `ttnn.transformer.scaled_dot_product_attention_decode`

**Not required:**
- Knowledge of Gemma 4 specific architectural innovations (heterogeneous attention, K=V sharing, V-norm, PLE, dual RoPE)
- Prior work with models that have structurally different layer types sharing a single decoder stack
- Experience with partial rotary position embeddings
- Understanding of how to shard tensors when KV head counts differ across layer types

## How to Use This Guide

| Goal | Recommended Path | Entry Point |
|------|-----------------|-------------|
| Understand the full Gemma 4 31B architecture from scratch | Ch1 → Ch2 → Ch3 → Ch4 → Ch5 | [Ch 1](./ch1_architecture_overview/index.md) |
| Implement the attention module (`TTNNModule` subclasses) | Ch1 → Ch2 → Ch3 → Ch4 → Ch5 | [Ch 5](./ch5_attention_module_design/index.md) |
| Understand K=V sharing and V-norm for global layers | Ch1 → Ch3 | [Ch 3](./ch3_kv_sharing_and_vnorm/index.md) |
| Configure RoPE for sliding vs global layers | Ch1 → Ch4 | [Ch 4](./ch4_dual_rope/index.md) |
| Plan tensor-parallel sharding on T3K | Ch1 → Ch2 → Ch6 | [Ch 6](./ch6_tp_sharding/index.md) |
| Assemble the full model end-to-end | Ch1 → Ch5 → Ch6 → Ch7 | [Ch 7](./ch7_model_assembly/index.md) |
| Evaluate memory budget and decode latency | Ch2 → Ch6 → Ch8 | [Ch 8](./ch8_performance/index.md) |
| Look up exact weight and activation tensor shapes | Ch2 | [Ch 2](./ch2_projection_shapes/index.md) |

## Chapter Index

| Chapter | Title | Description | Key Concepts |
|---------|-------|-------------|--------------|
| [Ch 1](./ch1_architecture_overview/index.md) | Gemma 4 31B Architecture Overview | Complete model architecture reference: 60 layers, heterogeneous attention configs, and novel components | 50 sliding + 10 global layers, hidden_size=5376, GeGLU FFN, PLE, logit soft-capping |
| [Ch 2](./ch2_projection_shapes/index.md) | Projection Weights and Tensor Shapes | Exact weight and activation shapes for every linear projection in both layer types | Q/K/V/O shapes per layer type, FFN `[5376, 21504]`, fused QKV layout, PLE shapes |
| [Ch 3](./ch3_kv_sharing_and_vnorm/index.md) | K=V Sharing and V-Norm Implementation | Deep-dive into K=V weight sharing in global layers and scale-free V-norm across all layers | Single K/V projection, divergent K/V post-processing, `RMSNorm(with_scale=False)` |
| [Ch 4](./ch4_dual_rope/index.md) | Dual RoPE and Partial Rotary Embedding | Two distinct RoPE configurations: full rotary for sliding layers and partial p-RoPE for global layers | theta=10000 vs theta=1M, `partial_rotary_factor=0.25`, cos/sin table precomputation |
| [Ch 5](./ch5_attention_module_design/index.md) | Heterogeneous Attention Module Design | TTNNModule class design for sliding and global attention, including forward pass dataflows | Single vs dual class design, `paged_sdpa_decode`, sliding window=1024, fused QKV |
| [Ch 6](./ch6_tp_sharding/index.md) | Tensor-Parallel Sharding on T3K | Sharding strategy for TP=8 with asymmetric KV head counts (16 sliding vs 4 global) | 4 Q heads/device, KV head sharding options, weight column/row parallelism, KV cache sharding |
| [Ch 7](./ch7_model_assembly/index.md) | Decoder Layer and Full Model Assembly | Full module hierarchy: decoder layer, FFN, PLE, and 60-layer model assembly | `TTNNGemma4DecoderLayer`, GeGLU FFN, PLE injection, decode loop orchestration |
| [Ch 8](./ch8_performance/index.md) | Performance Analysis and Optimization Roadmap | Memory budget, decode latency breakdown, and optimization opportunities | DRAM budget at 12 GB/chip, BFP8 KV cache, Metal Trace, multi-CQ overlap |

## Quick Reference

| Concept / API | What It Does | Where to Learn More |
|---------------|-------------|---------------------|
| Sliding layer (50 of 60) | Sliding-window attention: 32Q/16KV heads, head_dim=256, window=1024 | [Ch 1](./ch1_architecture_overview/index.md) |
| Global layer (10 of 60) | Full causal attention: 32Q/4KV heads, head_dim=512, K=V sharing, p-RoPE | [Ch 1](./ch1_architecture_overview/index.md) |
| K=V sharing (`attention_k_eq_v=True`) | Single projection produces both K and V; K gets RoPE, V does not | [Ch 3](./ch3_kv_sharing_and_vnorm/index.md) |
| V-norm (`RMSNorm` with `with_scale=False`) | Normalizes V vectors by RMS without learned scale; present in all 60 layers | [Ch 3](./ch3_kv_sharing_and_vnorm/index.md) |
| `ttnn.linear` | Matmul for Q/K/V/O and FFN projections; column-parallel or row-parallel | [Ch 2](./ch2_projection_shapes/index.md), [Ch 6](./ch6_tp_sharding/index.md) |
| `TTNNRotaryPositionEmbedding` | Applies RoPE cos/sin rotation to Q and K tensors | [Ch 4](./ch4_dual_rope/index.md) |
| Partial RoPE (`partial_rotary_factor=0.25`) | Rotates only 128 of 512 head dims in global layers; requires split-apply-concat | [Ch 4](./ch4_dual_rope/index.md) |
| `paged_sdpa_decode` | `ttnn.transformer.scaled_dot_product_attention_decode` with page table for KV cache | [Ch 5](./ch5_attention_module_design/index.md) |
| `ttnn.all_reduce` | Reduces partial sums across 8 devices after row-parallel matmuls | [Ch 6](./ch6_tp_sharding/index.md) |
| `ttnn.gelu` (tanh approx) | GeGLU gate activation in FFN layers (`gelu_pytorch_tanh`) | [Ch 7](./ch7_model_assembly/index.md) |
| Logit soft-capping (30.0) | `tanh(logits / 30) * 30` applied before final LM head output | [Ch 7](./ch7_model_assembly/index.md) |
| Metal Trace capture | Captures the decode loop as a replayable trace for reduced dispatch overhead | [Ch 8](./ch8_performance/index.md) |

## Source Code Location

This guide is self-contained and does not depend on external source code. The reference HuggingFace implementation of Gemma 4 lives in the [transformers](https://github.com/huggingface/transformers) repository under `src/transformers/models/gemma4/`. TTNN op implementations (including `scaled_dot_product_attention_decode`, `rms_norm`, and `all_reduce`) live in the [tt-metal](https://github.com/tenstorrent/tt-metal) repository. Model-level `TTNNModule` implementations are authored in the [tt-symbiote](https://github.com/tenstorrent/tt-symbiote) repository.
