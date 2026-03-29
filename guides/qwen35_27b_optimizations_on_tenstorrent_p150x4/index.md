# Guide: Qwen3.5-27B Optimizations on Tenstorrent P150x4

This guide documents the full optimization stack used to deploy Qwen3.5-27B on the P150x4 (4-chip Blackhole) platform, covering TP=4 sharding, DRAM-sharded decode, a custom fused GDN recurrence kernel, prefill TTFT optimization (5.3x speedup), and work-in-progress L1 state management. It is written for Tenstorrent engineers and advanced users who are already familiar with the tt-metal stack, tensor parallelism, and transformer fundamentals.

---

## How to Use This Guide

| Your Goal | Recommended Path | Direct Links |
|---|---|---|
| Understand the Qwen3.5-27B architecture and how it maps to 4 Blackhole chips | Start with Ch 1 | [Hybrid Architecture](ch1_architecture_and_hardware_mapping/hybrid_architecture.md), [TP Sharding Strategy](ch1_architecture_and_hardware_mapping/tp_sharding_strategy.md) |
| Optimize or debug full attention layers (partial RoPE, DRAM-sharded decode, flash SDPA) | Ch 1 then Ch 2 | [Attention Architecture](ch2_full_attention_layer_optimizations/attention_architecture.md), [DRAM-Sharded Decode](ch2_full_attention_layer_optimizations/dram_sharded_decode.md), [Flash Attention Prefill](ch2_full_attention_layer_optimizations/flash_attention_prefill.md) |
| Understand the GDN decode dataflow (conv1d, recurrence, gating) | Ch 1 then Ch 3 | [GDN Decode Flow](ch3_gdn_layer_decode_pipeline/gdn_decode_flow.md), [Recurrence Math](ch3_gdn_layer_decode_pipeline/recurrence_math.md) |
| Modify or debug the custom fused GDN kernel | Ch 3 then Ch 4 | [Kernel Dispatch](ch4_custom_fused_gdn_kernel/kernel_dispatch.md), [Compute Kernel](ch4_custom_fused_gdn_kernel/compute_kernel.md) |
| Improve prefill / TTFT performance | Ch 2 + Ch 3 then Ch 5 | [Batched Projections](ch5_prefill_ttft_optimization/batched_projections.md), [GDN Prefill Strategy](ch5_prefill_ttft_optimization/gdn_prefill_strategy.md) |
| Work on L1 state management for GDN layers | Ch 3 + Ch 4 then Ch 6 | [L1 State Design](ch6_l1_state_management/l1_state_design.md), [SDPA L1 Conflict](ch6_l1_state_management/sdpa_l1_conflict.md) |
| Assess performance and find remaining bottlenecks | Skim all prior chapters, then Ch 7 | [Performance Summary](ch7_performance_analysis/performance_summary.md), [Bottleneck Analysis](ch7_performance_analysis/bottleneck_analysis.md) |

---

## Chapter Index

| Ch | Title | Description | Key Concepts |
|---|---|---|---|
| 1 | [Architecture and Hardware Mapping](ch1_architecture_and_hardware_mapping/index.md) | Introduces the hybrid 48-GDN + 16-attention architecture and its TP=4 mapping across four Blackhole chips | Hybrid layer structure, TP sharding, column/row parallel projections, CCL ring topology |
| 2 | [Full Attention Layer Optimizations](ch2_full_attention_layer_optimizations/index.md) | Covers Qwen3.5-specific attention: partial RoPE, QK L2 norms, sigmoid gating, DRAM-sharded decode, flash SDPA prefill | Partial RoPE (64/256 dims), DRAM-sharded matmul, `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`, flash SDPA |
| 3 | [GDN Layer Decode Pipeline](ch3_gdn_layer_decode_pipeline/index.md) | Details the Gated DeltaNet decode forward pass: projections, conv1d shift register, recurrence, output gating | DeltaNet recurrence, 4-tap conv1d shift register, fused vs unfused paths, `[B*Nv_TP, Dk, Dv]` state |
| 4 | [Custom Fused GDN Kernel](ch4_custom_fused_gdn_kernel/index.md) | Deep dive into the C++ reader/compute/writer kernel that fuses L2 norm, gates, and recurrence into a single dispatch | `gdn_full_fused_inplace`, batched NOC reads, sub-tile extraction, FP32 accumulation, HEIGHT_SHARDED state paths |
| 5 | [Prefill TTFT Optimization](ch5_prefill_ttft_optimization/index.md) | Explains the 5.3x TTFT speedup (498 ms/tok to 94 ms/tok) via batched projections, flash attention, and B=1 GDN prefill | 2D matmul, batched QKVZ projection, sequential per-token recurrence, B=1 to B=32 state replication |
| 6 | [L1 State Management](ch6_l1_state_management/index.md) | Work-in-progress: moving GDN recurrence states from DRAM to L1 with a rolling window of 3 layers | L1 vs DRAM bandwidth, rolling window swap, HEIGHT_SHARDED kernel paths, SDPA CB conflict |
| 7 | [Performance Analysis](ch7_performance_analysis/index.md) | Measured performance, profiler breakdowns, and remaining optimization opportunities | 14.6 tok/s/user decode, 94 ms/tok TTFT, GDN = 85% of decode time, DRAM bandwidth bottleneck |

---

## Quick Reference

| Concept / Operation | What It Does | Where to Learn More |
|---|---|---|
| DRAM-sharded matmul | Decode projection with weight WIDTH_SHARDED across 8 DRAM cores, M=1 | [Ch 2 -- DRAM-Sharded Decode](ch2_full_attention_layer_optimizations/dram_sharded_decode.md) |
| 2D multicast matmul | Compute-bound prefill projection on 8x8 grid | [Ch 2 -- Flash Attention Prefill](ch2_full_attention_layer_optimizations/flash_attention_prefill.md), [Ch 5 -- Batched Projections](ch5_prefill_ttft_optimization/batched_projections.md) |
| DeltaNet recurrence | `state = exp(g)*state + outer(k, beta*(v - k@state))` per-token state update | [Ch 3 -- Recurrence Math](ch3_gdn_layer_decode_pipeline/recurrence_math.md) |
| Conv1d shift register | Trace-compatible 4-tap causal convolution via fixed copy chain | [Ch 3 -- Conv1d Shift Register](ch3_gdn_layer_decode_pipeline/conv1d_shift_register.md) |
| `gdn_full_fused_inplace` | Custom kernel fusing L2 norm + gates + recurrence into one dispatch | [Ch 4 -- Kernel Dispatch](ch4_custom_fused_gdn_kernel/kernel_dispatch.md) |
| Batched NOC reads | 44 reads per pair with single `noc_async_read_barrier()` | [Ch 4 -- Reader Kernel](ch4_custom_fused_gdn_kernel/reader_kernel.md) |
| B=1 to B=32 state replication | Post-prefill expansion of GDN rec_states and KV cache to full batch | [Ch 5 -- State Replication](ch5_prefill_ttft_optimization/state_replication.md) |
| L1 rolling window | Keeps 3 GDN layers' states in L1, swaps on group boundaries | [Ch 6 -- L1 State Design](ch6_l1_state_management/l1_state_design.md) |
| Partial RoPE | Applies rotary embeddings to first 64 of 256 head dims | [Ch 2 -- Attention Architecture](ch2_full_attention_layer_optimizations/attention_architecture.md) |

---

## Prerequisites

Readers should be familiar with:

- **Tensor parallelism (TP)** concepts and multi-device programming with `ttnn`
- **The tt-metal stack**: circular buffers, NOC data movement, DRAM vs L1 memory hierarchy, `TILE_LAYOUT`
- **Transformer model architecture**: attention, MLP, residual connections
- **Basic linear attention / state-space model concepts**: recurrence states vs KV caches

No prior knowledge of the Qwen3.5 architecture, DeltaNet recurrence math, or the custom kernel APIs is required -- these are covered starting in Chapter 1.

---

## Source Code Location

The Qwen3.5-27B model implementation lives in the tt-metal repository:

```
models/demos/qwen35_27b/tt/
├── model.py                 # Transformer class, layer construction
├── attention.py             # Qwen35Attention (full attention layers)
├── gdn.py                   # TtGatedDeltaNet (GDN layers, fused/unfused paths)
├── gdn_kernel_op.py         # Custom kernel dispatch (gdn_full_fused_inplace)
├── kernels/                 # C++ reader/compute/writer kernels
├── model_config.py          # Qwen35ModelArgs, TP dimensions, memory configs
└── weight_utils.py          # Weight preparation helpers (prepare_gdn_qkv, etc.)
```

Tests and profiling scripts:

```
models/demos/qwen35_27b/tests/
├── test_e2e_generate.py     # End-to-end decode throughput (14.6 tok/s/user)
├── test_ttft.py             # TTFT measurement (94 ms/tok)
└── test_profile_breakdown.py # Per-layer profiler breakdown
```
