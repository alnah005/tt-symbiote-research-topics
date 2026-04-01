# Qwen3.5 Implementation Guide — TT-Metal / Blackhole P100A

A comprehensive guide to the Qwen3.5-35B-A3B and Qwen3.5-27B implementations running
on the Tenstorrent Blackhole P100A accelerator. The guide covers every module, every
optimization, and every correctness constraint — from model architecture to measured
decode performance.

**Audience:** ML engineers and hardware-aware software engineers comfortable with transformer
architectures, PyTorch, and basic TTNN concepts. No prior knowledge of linear attention,
DeltaNet recurrence, MoE, or Blackhole-specific hardware constraints is assumed.

---

## How to Use This Guide

| Goal | Recommended path | Deep links |
|---|---|---|
| Understand what models are supported | Ch 1 | [`model_variants.md`](./ch1_model_architecture_overview/model_variants.md) |
| Learn the DeltaNet recurrence | Ch 1 → Ch 2 | [`recurrence_math.md`](./ch2_gated_deltanet_linear_attention_on_blackhole/recurrence_math.md) |
| Understand why host float32 is required | Ch 2 | [`host_recurrence.md`](./ch2_gated_deltanet_linear_attention_on_blackhole/host_recurrence.md) |
| Understand partial RoPE and GatedAttention | Ch 3 | [`partial_rope.md`](./ch3_gated_attention_full_attention_layers/partial_rope.md) |
| Trace the full forward pass | Ch 4 | [`forward_signature.md`](./ch4_decoder_block_and_uniform_dispatch/forward_signature.md) |
| Understand MoE routing and expert dispatch | Ch 5 | [`router_and_routing.md`](./ch5_mixture_of_experts/router_and_routing.md), [`expert_computation.md`](./ch5_mixture_of_experts/expert_computation.md) |
| Understand HF→meta weight conversion | Ch 6 | [`hf_to_meta_conversion.md`](./ch6_weight_precision_dram_layout_and_weight_conversion/hf_to_meta_conversion.md), [`moe_key_protection.md`](./ch6_weight_precision_dram_layout_and_weight_conversion/moe_key_protection.md) |
| Understand the 86 ms/token breakdown | Ch 7 | [`latency_breakdown.md`](./ch7_performance_analysis_and_bottlenecks/latency_breakdown.md), [`sync_overhead.md`](./ch7_performance_analysis_and_bottlenecks/sync_overhead.md) |
| Run PCC tests to verify correctness | Ch 8 | [`testing_infrastructure.md`](./ch8_optimization_roadmap_and_testing/testing_infrastructure.md), [`running_tests.md`](./ch8_optimization_roadmap_and_testing/running_tests.md) |
| Understand the optimization roadmap | Ch 7 → Ch 8 | [`bottleneck_analysis.md`](./ch7_performance_analysis_and_bottlenecks/bottleneck_analysis.md), [`optimization_roadmap.md`](./ch8_optimization_roadmap_and_testing/optimization_roadmap.md) |

---

## Prerequisites

- **Transformer fundamentals:** attention, MLP, RMSNorm, positional encoding
- **PyTorch:** tensor operations, `F.linear`, `torch.einsum`, module state dicts
- **TTNN basics:** `ttnn.linear`, `ttnn.from_torch`, `ttnn.to_torch`, device memory configs (`DRAM_MEMORY_CONFIG`, `L1_MEMORY_CONFIG`), `ttnn.TILE_LAYOUT`
- **Optional:** familiarity with Grouped Query Attention (GQA), RotaryEmbedding, or SwiGLU — the guide introduces all three from scratch

---

## Source Code Location

All source files referenced in this guide are in:

```
/localdev/salnahari/testing_dir/tt-metal_p100_qwen35/
├── models/
│   ├── demos/qwen35/               # Demo scripts, tests, reference implementations
│   │   ├── demo/demo.py            # 27B decode demo
│   │   ├── demo/demo_a3b.py        # A3B decode demo
│   │   ├── tests/test_pcc.py       # 27B PCC test suite
│   │   ├── tests/test_a3b_pcc.py   # A3B PCC test suite (incl. fused kernel)
│   │   └── reference/              # Pure-PyTorch reference scripts (no device needed)
│   └── tt_transformers/tt/         # Core TTNN module implementations
│       ├── gated_deltanet.py       # GatedDeltaNet (linear attention)
│       ├── gated_attention.py      # GatedAttention (full attention)
│       ├── qwen35_decoder.py       # DeltaNetDecoderBlock
│       ├── qwen35_moe.py           # Qwen35MoE (MoE MLP)
│       ├── qwen35_utils.py         # Weight conversion (convert_hf_to_meta_qwen35)
│       └── rope.py                 # HfRotarySetup
```

---

## Chapter Index

| Chapter | Title | Description |
|---|---|---|
| 1 | [Model Architecture Overview](./ch1_model_architecture_overview/index.md) | Two variants (27B dense, 35B-A3B MoE), layer counts, hyperparameter tables, hybrid design |
| 2 | [GatedDeltaNet: Linear Attention on Blackhole](./ch2_gated_deltanet_linear_attention_on_blackhole/index.md) | Five-step recurrence, conv1d ring buffer, SrcB TF32 constraint, fused kernel |
| 3 | [GatedAttention: Full Attention Layers](./ch3_gated_attention_full_attention_layers/index.md) | Partial RoPE (rotary_dim=64), output gate, cos/sin patching fix |
| 4 | [Decoder Block and Uniform Dispatch](./ch4_decoder_block_and_uniform_dispatch/index.md) | `DeltaNetDecoderBlock`, `attention_class`/`mlp_class` injection, forward signature |
| 5 | [Mixture of Experts](./ch5_mixture_of_experts/index.md) | 256 routed + 1 shared expert, top-8 host routing, bfp4, 15.7 GiB DRAM |
| 6 | [Weight Precision, DRAM Layout, and Weight Conversion](./ch6_weight_precision_dram_layout_and_weight_conversion/index.md) | bfp4/bfp8/bf16/fp32 per category, HF→meta pipeline, MoE key protection |
| 7 | [Performance Analysis and Bottlenecks](./ch7_performance_analysis_and_bottlenecks/index.md) | 86 ms/token breakdown, sync overhead, Python dispatch, efficiency ceiling |
| 8 | [Optimization Roadmap and Testing](./ch8_optimization_roadmap_and_testing/index.md) | Metal Trace, Multi-CQ, per-row MoE, PCC test suite, running tests |

---

## Quick Reference

| Concept / Operation | What it does | Where to learn more |
|---|---|---|
| `ttnn.experimental.gated_delta_net(...)` | Fused DeltaNet recurrence on device (fp32 state, zero host syncs) | [Ch 2 fused_kernel.md](./ch2_gated_deltanet_linear_attention_on_blackhole/fused_kernel.md) |
| `ttnn.copy(src, dst)` | In-place tensor write — preserves device address for Metal Trace | [Ch 2 fused_kernel.md](./ch2_gated_deltanet_linear_attention_on_blackhole/fused_kernel.md) |
| `HfRotarySetup.get_rot_mats()` | Returns pre-cached cos/sin matrices — no host sync | [Ch 3 partial_rope.md](./ch3_gated_attention_full_attention_layers/partial_rope.md) |
| `DeltaNetDecoderBlock` | Unified decoder block for both DeltaNet and GatedAttention layers | [Ch 4 block_structure.md](./ch4_decoder_block_and_uniform_dispatch/block_structure.md) |
| `Qwen35MoE.forward(x)` | MoE forward: router → host topk → expert matmuls → accumulate | [Ch 5 expert_computation.md](./ch5_mixture_of_experts/expert_computation.md) |
| `convert_hf_to_meta_qwen35(sd, head_dim, n_heads, n_kv_heads)` | 5-step HF→meta weight conversion with MoE key protection | [Ch 6 hf_to_meta_conversion.md](./ch6_weight_precision_dram_layout_and_weight_conversion/hf_to_meta_conversion.md) |
| `_is_moe_key(key)` | Identifies MoE weight keys for the pop-protect-reinsert pattern | [Ch 6 moe_key_protection.md](./ch6_weight_precision_dram_layout_and_weight_conversion/moe_key_protection.md) |
| `ttnn.synchronize_device(device)` | Blocks host until all device commands complete; main sync barrier | [Ch 7 sync_overhead.md](./ch7_performance_analysis_and_bottlenecks/sync_overhead.md) |
| `TestFusedKernelPCC` | Validates fused kernel without model download (synthetic data) | [Ch 8 running_tests.md](./ch8_optimization_roadmap_and_testing/running_tests.md) |
| `bfp4` (bfloat4_b) | Block floating point 4-bit — used for routed expert weights to fit 28 GB DRAM | [Ch 5 dram_budget.md](./ch5_mixture_of_experts/dram_budget.md) |

---

## Performance Summary

| Model | Precision | Throughput | Latency | Efficiency |
|---|---|---|---|---|
| Qwen3.5-35B-A3B | bfp4 experts, f32 recurrence | 11.7 tok/s | 86 ms/token | ~6.8% of 172 tok/s peak |
| Qwen3.5-27B | bfp8, f32 recurrence | 6.28 tok/s | ~159 ms/token | ~3.6% of 172 tok/s peak |

Both run on a single P100A Blackhole (28 GB DRAM). The A3B model beats the CPU baseline
of 9.05 tok/s (llama.cpp Q4_K on AmpereOne).

---

## Key Constraints

1. **Blackhole SrcB TF32 constraint:** bf16 element-wise ops on fp32 circular buffers hang the device. All DeltaNet recurrence uses the fused kernel (`ttnn.experimental.gated_delta_net`). See [Ch 2 host_recurrence.md](./ch2_gated_deltanet_linear_attention_on_blackhole/host_recurrence.md).

2. **In-place tensor writes for Metal Trace:** `ttnn.copy` preserves device tensor addresses. Required for Metal Trace replay without reallocation. See [Ch 2 fused_kernel.md](./ch2_gated_deltanet_linear_attention_on_blackhole/fused_kernel.md).

3. **MoE same-prompt batch assumption:** Current routing reads row 0 only. Mixed-prompt batching requires per-row topk (future work). See [Ch 5 architecture_overview.md](./ch5_mixture_of_experts/architecture_overview.md).

4. **Partial RoPE via cos/sin patching:** Standard `rotary_embedding_llama` cannot handle Qwen3.5's partial rotation. The fix patches cos/sin matrices at model-build time. See [Ch 3 partial_rope.md](./ch3_gated_attention_full_attention_layers/partial_rope.md).
