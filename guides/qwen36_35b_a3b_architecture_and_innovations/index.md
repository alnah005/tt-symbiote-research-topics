# Qwen3.6-35B-A3B Architecture and Innovations

This guide explains every architectural and post-training feature of Qwen3.6-35B-A3B that a Tenstorrent ML systems engineer needs to understand — what changed from Qwen3.5, what stayed the same, and what the implications are for the existing TTNN implementation. It is written for ML systems engineers and hardware-aware model developers who are familiar with transformer architectures, basic TTNN concepts, and the Qwen3.5-35B-A3B model at a high level.

---

## How to Use This Guide

| Goal | Recommended path |
|------|-----------------|
| "I just want to know what changed between Qwen3.5 and Qwen3.6" | [Chapter 3](ch3_qwen35_vs_qwen36_differences/index.md) |
| "I need to understand the full architecture before implementing" | [Chapter 1](ch1_architecture_overview/index.md) → [Chapter 2](ch2_gated_deltanet/index.md) → [Chapter 7](ch7_moe_comparison/index.md) |
| "I want to enable MTP-based speculative decoding in TTNN" | [Chapter 5](ch5_multi_token_prediction/index.md) |
| "I'm deploying multimodal (image/video) inference" | [Chapter 4](ch4_rope_and_mrope/index.md) → [Chapter 8](ch8_vision_encoder/index.md) |
| "I need to understand KV cache behavior at long contexts" | [Chapter 2](ch2_gated_deltanet/index.md) → [Chapter 6](ch6_thinking_preservation/index.md) |
| "I want to compare Qwen3.6's MoE design to other models" | [Chapter 7](ch7_moe_comparison/index.md) |

---

## Chapter Index

| Chapter | Title | Description | Key concepts |
|---------|-------|-------------|--------------|
| [Ch 1 — Architecture Overview](ch1_architecture_overview/index.md) | Architecture Overview | Full architecture: hybrid layer layout, hyperparameters, forward pass data flow | Hybrid GDN + Gated Attention layout, 3:1 pattern, MoE per layer, full hyperparameter table |
| [Ch 2 — Gated DeltaNet Deep Dive](ch2_gated_deltanet/index.md) | Gated DeltaNet Deep Dive | Delta rule formulation, gating, state dimensions, comparison to other linear attention variants | Delta rule recurrence, scalar decay gate $g_t$, head asymmetry, conv1d local mixing, GLA/Mamba2/RetNet comparison |
| [Ch 3 — Qwen3.5 vs Qwen3.6 Differences](ch3_qwen35_vs_qwen36_differences/index.md) | Qwen3.5 vs Qwen3.6 Differences | Config diff, post-training changes, benchmark comparison | Architectural identity, post-training improvements, agentic coding gains, zero TTNN code changes |
| [Ch 4 — Partial Rotary Embedding and M-RoPE](ch4_rope_and_mrope/index.md) | Partial Rotary Embedding and M-RoPE | Partial RoPE (rotary\_dim=64), M-RoPE for multimodal, mrope\_section=[11,11,10] | Partial rotary (25% of head dims), M-RoPE section assignment, text-only equivalence, multimodal position IDs |
| [Ch 5 — Multi-Token Prediction (MTP)](ch5_multi_token_prediction/index.md) | Multi-Token Prediction (MTP) | MTP training objective, speculative decoding at inference, TTNN implications | mtp\_num\_hidden\_layers=1, draft-verify decoding loop, acceptance rate, optional at inference |
| [Ch 6 — Thinking Preservation](ch6_thinking_preservation/index.md) | Thinking Preservation | Inference-time reasoning retention, KV cache impact, zero TTNN changes | Prompting-level technique, KV cache growth in Gated Attention layers, DeltaNet state independence |
| [Ch 7 — MoE Architecture and Cross-Model Comparison](ch7_moe_comparison/index.md) | MoE Architecture and Cross-Model Comparison | 256-expert MoE deep dive, DeepSeek-V3/Gemma4 comparison, Tenstorrent hardware analysis | 256 routed + 1 shared, top-8, intermediate=512, many-small-expert tradeoffs, expert parallelism |
| [Ch 8 — Vision Encoder and Multimodal Integration](ch8_vision_encoder/index.md) | Vision Encoder and Multimodal Integration | 27-layer ViT specs, image/video pipeline, Gemma4/LLaVA comparison | 27-layer ViT, hidden=1152, spatial\_merge\_size=2, temporal\_patch\_size=2, prefill-only cost |

---

## Quick Reference

| Parameter | Value | Covered in |
|-----------|-------|-----------|
| Total / active parameters | 35B total / ~3B active per token | [Ch 1](ch1_architecture_overview/index.md) |
| Layer count | 40 layers total | [Ch 1](ch1_architecture_overview/index.md) |
| Hybrid layout | 30 Gated DeltaNet + 10 Gated Attention, repeating 3:1 pattern (`full_attention_interval=4`) | [Ch 1](ch1_architecture_overview/index.md) |
| MoE config | 256 routed experts + 1 shared, top-8 routing, `moe_intermediate_size=512` | [Ch 7](ch7_moe_comparison/index.md) |
| Context window | 262,144 tokens native | [Ch 1](ch1_architecture_overview/index.md), [Ch 6](ch6_thinking_preservation/index.md) |
| MTP | `mtp_num_hidden_layers=1`; speculative decoding is optional at inference | [Ch 5](ch5_multi_token_prediction/index.md) |
| Partial RoPE | `rotary_dim=64` (25% of `head_dim=256`), `rope_theta=10,000,000` | [Ch 4](ch4_rope_and_mrope/index.md) |
| M-RoPE | `mrope_section=[11, 11, 10]`, `mrope_interleaved=true` | [Ch 4](ch4_rope_and_mrope/index.md) |
| Vision encoder | 27-layer ViT, `hidden_size=1152`, `patch_size=16`, `spatial_merge_size=2`, `temporal_patch_size=2` | [Ch 8](ch8_vision_encoder/index.md) |
| Architecture identity with Qwen3.5 | Identical at config and weight-shape level; uses `Qwen3_5MoeForConditionalGeneration` for both | [Ch 3](ch3_qwen35_vs_qwen36_differences/index.md) |
| TTNN code changes required | Zero — existing Qwen3.5 TTNN implementation runs Qwen3.6 weights without modification | [Ch 3](ch3_qwen35_vs_qwen36_differences/index.md) |

---

## Prerequisites

Before using this guide, the reader should be comfortable with:

- **Transformer architecture fundamentals**: multi-head attention, RMSNorm, residual connections, rotary position embeddings (RoPE), and autoregressive token generation.
- **Mixture of Experts (MoE)**: token routing, expert dispatch, and the distinction between routed and shared experts.
- **Qwen3.5-35B-A3B at a high level**: familiarity with its hybrid Gated DeltaNet + Gated Attention layer layout and the general TTNN implementation approach. The guide `guides/qwen35_implementation/` is a good reference.
- **Basic TTNN concepts**: tensor operations, device placement, memory configs, and the general structure of a TTNN model forward pass.

The guide does not assume prior knowledge of Multi-Token Prediction (MTP), M-RoPE, Thinking Preservation, or the specific Gated DeltaNet recurrence. Those are built from first principles in the relevant chapters.

---

## Source Code Location

Both Qwen3.5-35B-A3B and Qwen3.6-35B-A3B use the same HuggingFace architecture class:

```
architectures: ["Qwen3_5MoeForConditionalGeneration"]
model_type: "qwen3_5_moe"
```

The architecture class is unchanged between the two model versions. Any code path that instantiates or loads a `Qwen3_5MoeForConditionalGeneration` model works for Qwen3.6 weights without modification. The TTNN implementation in `guides/qwen35_implementation/` therefore applies directly to Qwen3.6.
