# Chapter 1 — Multi-Token Prediction: Training Objective and Architecture

## Prerequisites

None. This chapter assumes only the baseline audience knowledge described in the guide introduction: familiarity with transformer architecture fundamentals, standard autoregressive (AR) decode, HuggingFace Transformers, and general awareness of speculative decoding as a concept.

---

## What This Chapter Establishes

This chapter answers three foundational questions that all subsequent chapters depend on:

1. **Multi-Token Prediction (MTP) as a training-time loss.** MTP is an auxiliary objective added during model training. It teaches the model to predict not just the immediately next token but several future tokens simultaneously. Understanding MTP as a *training* objective — rather than a native inference mechanism — is the key framing required before reasoning about whether and how to exploit it at inference time.

2. **The architectural attachment point of the MTP head.** The MTP head is a lightweight stack of additional transformer decoder blocks appended after the main backbone. Its inputs, outputs, and weight-sharing relationships with the backbone determine how it must be handled during model loading, memory planning, and TTNN porting.

3. **The terminology used throughout this guide.** Precise definitions of terms such as "backbone," "draft token," "draft depth," "acceptance rate," and "lm_head" are established here and used consistently across all five chapters.

---

## Chapter Overview

Qwen3.6-35B-A3B exposes `mtp_num_hidden_layers: 1` in its model configuration. This single field encodes a significant architectural choice: the checkpoint contains weights for a one-block MTP head trained with a multi-token prediction auxiliary loss. Before any decision can be made about loading, porting, or exploiting this head for speculative decoding on Tenstorrent (TT) hardware, engineers need a clear mental model of what MTP is and how it is structured.

The three files in this chapter build that model bottom-up:

- **[`mtp_training_objective.md`](./mtp_training_objective.md)** covers the original MTP formulation from the research literature, its motivation as a gradient-enrichment technique, the role of `mtp_num_hidden_layers` as its key hyperparameter, and how MTP differs from related multi-step training objectives such as knowledge distillation and consistency regularization.

- **[`mtp_head_architecture.md`](./mtp_head_architecture.md)** describes the physical structure of the MTP head: what transformer components it contains, how it receives input from the backbone's final hidden state, how it shares the language model head (`lm_head`) with the backbone, and what the architectural difference is between a one-block and a multi-block MTP head. This file ends with an open question about KV cache usage during inference, which is resolved in Chapter 3.

- **[`qwen36_mtp_config.md`](./qwen36_mtp_config.md)** grounds the abstract architecture in Qwen3.6-35B-A3B's specific configuration: the concrete hyperparameter values, the checkpoint weight-key naming convention for MTP tensors, and a comparison with the Qwen3.5 lineage to establish when MTP was introduced.

---

## Reading Order

| Order | File | Core question answered |
|-------|------|------------------------|
| 1 | [`mtp_training_objective.md`](./mtp_training_objective.md) | What is MTP and why does it exist? |
| 2 | [`mtp_head_architecture.md`](./mtp_head_architecture.md) | How is the MTP head structured and wired into the backbone? |
| 3 | [`qwen36_mtp_config.md`](./qwen36_mtp_config.md) | What are the exact values for Qwen3.6-35B-A3B? |

All three files should be read in order before proceeding to Chapter 2 or Chapter 3.

---

## Key Terms Introduced in This Chapter

| Term | Brief definition |
|------|-----------------|
| MTP | Multi-Token Prediction; the auxiliary training objective and associated head |
| backbone | The main transformer stack, excluding the MTP head |
| MTP head | The lightweight transformer block(s) appended for MTP; identified by `mtp_num_hidden_layers` |
| draft token | A speculatively predicted token produced by the MTP head's auxiliary logit distribution |
| draft depth (N) | Number of future positions for which MTP produces auxiliary logits |
| lm_head | The unembedding / language model head shared by backbone and MTP head |
| AR | Autoregressive; standard token-by-token generation |

Full definitions appear in the guide conventions and are restated on first use in each file.

---

## References

- [DeepSeek-V3] DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv:2412.19437, 2024.
- [Gloeckle2024] Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction", arXiv:2404.19737, 2024.
- [Qwen3] Qwen Team, "Qwen3 Technical Report", Alibaba Cloud, 2025.
