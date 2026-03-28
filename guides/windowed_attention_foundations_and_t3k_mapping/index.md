# Windowed Attention: Foundations and T3K Mapping

This guide covers windowed (sliding window) attention from mathematical foundations through production deployment on the T3K 1×8 Wormhole mesh. It targets ML systems engineers and kernel developers on the TT-NN / tt-transformers stack who need to integrate windowed attention into models such as Qwen3.5 or Mistral.

## Prerequisites

**Required:**
- Familiarity with standard SDPA and causal masking
- Working knowledge of TT-NN tensor operations and program configs
- Basic understanding of T3K device topology (8 Wormhole chips, 1×8 mesh)
- Exposure to KV cache concepts for autoregressive decode

**Not required:**
- Prior windowed attention experience
- Paged KV internals
- Roofline analysis methodology

## How to Use This Guide

| Goal | Recommended Path | Entry Point |
|------|-----------------|-------------|
| Understand the math from scratch | Ch1 → Ch2 → Ch3 → Ch4 | [Ch 1](./ch1_math_foundations/index.md) |
| Integrate windowed attention into a T3K deployment | Ch1 → Ch2 → Ch4 → Ch6 | [Ch 1](./ch1_math_foundations/index.md) |
| Understand TTNN op gaps and kernel requirements | Ch4 → Ch7 | [Ch 4](./ch4_ttnn_primitives/index.md) |
| Understand paged KV cache interactions | Ch2 → Ch4 → Ch5 | [Ch 2](./ch2_kv_cache_management/index.md) |
| Check T3K memory and bandwidth budgets | Ch7 | [Ch 7](./ch7_roofline_and_kernels/index.md) |

## Chapter Index

| Chapter | Title | Description | Key Concepts |
|---------|-------|-------------|--------------|
| [Ch 1](./ch1_math_foundations/index.md) | Mathematical Foundations | Formal definitions, complexity savings | window size w, band-diagonal mask, O(T·w) vs O(T²) |
| [Ch 2](./ch2_kv_cache_management/index.md) | KV Cache Management During Decode | Circular buffer layout, steady-state memory | circular buffer, pos_offset, g_fill formula |
| [Ch 3](./ch3_data_dependencies/index.md) | Data Dependencies and Memory Access Patterns | Prefill/decode AI analysis | arithmetic intensity, bandwidth-bound vs compute-bound |
| [Ch 4](./ch4_ttnn_primitives/index.md) | TTNN Primitive Operations and Tensor Shapes | Exact op/shape mapping, gap analysis | ttnn.scaled_dot_product_attention_decode, ttnn.update_cache |
| [Ch 5](./ch5_paged_kv_cache/index.md) | Paged KV Cache Interaction | Paging + windowing strategies | paged_sdpa_decode, Strategy A/B, page table correctness |
| [Ch 6](./ch6_t3k_sharding/index.md) | T3K Mesh Sharding and CCL Implications | Head-parallel sharding, CCL volumes | ttnn.all_gather, ttnn.reduce_scatter, shard sizes |
| [Ch 7](./ch7_roofline_and_kernels/index.md) | Roofline Analysis and Existing Kernel Survey | Bandwidth-bound decode, gap closures | roofline model, G1–G7 gaps, chunked prefill |

## Quick Reference

| Concept / API | What It Does | Where to Learn More |
|---------------|-------------|---------------------|
| `w` (window size) | Number of past tokens each query attends to | [Ch 1](./ch1_math_foundations/index.md) |
| Circular buffer KV cache | Fixed `[B, H, w, d]` tensor with round-robin writes | [Ch 2](./ch2_kv_cache_management/index.md) |
| `pos_offset` | Absolute position of the oldest slot in the circular buffer | [Ch 2](./ch2_kv_cache_management/index.md) |
| Band-diagonal mask | `[T, T]` mask for prefill windowed attention | [Ch 3](./ch3_data_dependencies/index.md) |
| `ttnn.scaled_dot_product_attention_decode` | Fused decode SDPA op | [Ch 4](./ch4_ttnn_primitives/index.md) |
| `ttnn.update_cache` | Writes new K/V into circular buffer slot | [Ch 4](./ch4_ttnn_primitives/index.md) |
| `paged_sdpa_decode` | Paged variant of decode SDPA | [Ch 5](./ch5_paged_kv_cache/index.md) |
| Head-parallel sharding | Each device holds disjoint head subset | [Ch 6](./ch6_t3k_sharding/index.md) |
| `ttnn.all_gather` | Gathers KV data across 8 devices | [Ch 6](./ch6_t3k_sharding/index.md) |
| Roofline crossover | ~111 FLOPs/byte on Wormhole | [Ch 7](./ch7_roofline_and_kernels/index.md) |

## Source Code Location

This guide is self-contained and does not depend on external source code. Relevant TTNN op implementations (including `scaled_dot_product_attention_decode`, `update_cache`, and `paged_sdpa_decode`) live in the [tt-metal](https://github.com/tenstorrent/tt-metal) repository. Model-level implementations that use these ops (attention modules, KV cache management, sharding configs) live in the [tt-transformers](https://github.com/tenstorrent/tt-transformers) repository.
