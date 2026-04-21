# Chapter 1 -- Complete Architecture Overview

## Overview

This chapter establishes the full architecture of Qwen3.6-35B-A3B from first principles. It covers the hybrid layer layout, all key hyperparameters, the forward pass data flow, and how the Gated DeltaNet, Gated Attention, and MoE components compose into a single decoder block.

The central finding of this chapter -- and the thesis of the entire guide -- is that **Qwen3.6-35B-A3B is architecturally identical to Qwen3.5-35B-A3B at the config and weight-shape level**. Both models use the `Qwen3_5MoeForConditionalGeneration` architecture class and the `qwen3_5_moe` model type. Every layer count, every head dimension, every expert configuration is the same. All differences between the two models are post-training: different RLHF alignment, different data mixtures, and different inference-time prompting strategies (Thinking Preservation, agentic coding improvements). The weight tensors have identical shapes and dtypes; only their values differ.

This means the existing TTNN implementation for Qwen3.5-35B-A3B can load and run Qwen3.6 weights with zero code changes to the model architecture. Subsequent chapters examine the post-training innovations and their implications in detail.

---

## Learning Objectives

After completing this chapter, readers will be able to:

1. **List the complete hyperparameter set** for Qwen3.6-35B-A3B: layer count, hidden dimension, vocabulary size, context length, and all per-component configurations for Gated DeltaNet, Gated Attention, MoE, vision encoder, MTP, and RoPE.

2. **Describe the hybrid layer layout**: 10 repetitions of (3 Gated DeltaNet + 1 Gated Attention) = 40 layers total, with every layer followed by a MoE FFN block.

3. **Trace the end-to-end forward pass** for a single text token through the decoder, identifying where the Gated DeltaNet recurrence, softmax attention, MoE routing, and residual connections occur.

4. **Explain the multimodal forward pass extension**: how the vision encoder processes image/video patches, merges them spatially, projects them into the text embedding space, and interleaves them with text tokens.

5. **Distinguish the two state management regimes**: Gated DeltaNet layers maintain a fixed-size recurrent state matrix $S \in \mathbb{R}^{d_k \times d_v}$ per head (constant memory regardless of sequence length), while Gated Attention layers maintain a KV cache that grows linearly with sequence length.

6. **State the total and active parameter counts**: 35B total parameters, approximately 3B activated per token (8 routed experts + 1 shared expert out of 256 per layer).

---

## Notation

| Symbol | Meaning | Qwen3.6 Value |
|---|---|---|
| $H$ | Model hidden dimension | 2048 |
| $T$ | Sequence length | variable |
| $B$ | Batch size | variable |
| $d_k$ | DeltaNet key/query head dimension | 128 |
| $d_v$ | DeltaNet value head dimension | 128 |
| $d_h$ | Gated Attention head dimension | 256 |
| $n_q$ | Gated Attention query heads | 16 |
| $n_{kv}$ | Gated Attention KV heads | 2 |
| $H_v$ | DeltaNet value heads | 32 |
| $H_k$ | DeltaNet key/query heads | 16 |
| $E$ | Total routed experts per MoE layer | 256 |
| $k$ | Top-k expert selection | 8 |

---

## Chapter Contents

| File | Topic |
|---|---|
| [`architecture_and_hyperparams.md`](./architecture_and_hyperparams.md) | Full hyperparameter tables, hybrid layer layout, per-component configurations (Gated DeltaNet, Gated Attention, MoE, vision encoder, MTP, RoPE), and parameter count analysis |
| [`forward_pass_dataflow.md`](./forward_pass_dataflow.md) | End-to-end forward pass for text and multimodal inputs, state management (recurrent state vs. KV cache), and MoE routing mechanics in the forward pass |

---

## Reading Order

Read the two files in order:

1. **[`architecture_and_hyperparams.md`](./architecture_and_hyperparams.md)** -- Establishes the complete set of architectural constants and the layer layout. This is the prerequisite for understanding the forward pass.

2. **[`forward_pass_dataflow.md`](./forward_pass_dataflow.md)** -- Traces data through the decoder using the hyperparameters and layout from the first file. Covers both text-only and multimodal inference paths.

---

## Relationship to Later Chapters

- **Chapter 2 (Gated DeltaNet Deep Dive)** builds on the DeltaNet hyperparameters and layer layout established here, providing the full mathematical formulation of the delta rule recurrence.
- **Chapter 3 (Qwen3.5 vs Qwen3.6 Differences)** takes this chapter's architecture description as baseline and proves that the config is identical between the two model versions.
- **Chapter 4 (Partial RoPE and M-RoPE)** expands on the rotary embedding configuration introduced here.
- **Chapter 5 (Multi-Token Prediction)** builds on the decoder forward pass described in [`forward_pass_dataflow.md`](./forward_pass_dataflow.md).
- **Chapter 7 (MoE Architecture and Cross-Model Comparison)** deepens the MoE configuration introduced here with cross-model analysis.
- **Chapter 8 (Vision Encoder)** expands the vision encoder summary into full architectural detail.

---

## Cross-References to Existing Guides

This chapter synthesizes information that is also covered (with different emphasis) in several existing guides:

- `guides/qwen35_implementation/` -- Covers the TTNN implementation of Qwen3.5-35B-A3B on Blackhole P100A, including fused kernel design. The hyperparameters documented here match those used in that guide.
- `guides/gated_delta_net_and_gated_attention_on_t3k/` -- Covers the T3K-specific implementation of Gated DeltaNet and Gated Attention, including sharding strategies.
- `guides/expert_parallelism_strategies/` -- Covers MoE expert parallelism for a different Qwen3.5 variant (the 35B-A22B dense-MoE model with 7168 hidden dimension). The MoE concepts are transferable but the specific dimensions differ.

---

## References

- [Qwen3] Qwen Team, Alibaba Group, "Qwen3 Technical Report", 2025.
- [Qwen3.6] Qwen Team, Alibaba Group, "Qwen3.6 Release Blog", 2026.
- [Yang2025] Yang, S. et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule", ICML, 2025.
- [DeepSeekV3] DeepSeek-AI, "DeepSeek-V3 Technical Report", 2024.

---

Begin reading: [`architecture_and_hyperparams.md`](./architecture_and_hyperparams.md)
