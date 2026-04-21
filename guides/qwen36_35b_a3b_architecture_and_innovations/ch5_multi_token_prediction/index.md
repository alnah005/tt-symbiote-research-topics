# Chapter 5 — Multi-Token Prediction (MTP)

## Overview

Multi-Token Prediction (MTP) is a training-time auxiliary objective adopted by Qwen3.6-35B-A3B, following the design pioneered in the DeepSeek-V3 technical report. The core idea is simple: in addition to the standard next-token prediction head, the model trains a secondary head that predicts the token two steps ahead. The auxiliary loss encourages the main decoder to build richer internal representations, because correctly predicting a further-future token demands more structured contextual encoding.

A critical point to keep in mind throughout this chapter: **MTP is primarily a training mechanism.** Its inference-time role—enabling speculative decoding—is optional. When MTP-based speculative decoding is not enabled, the MTP module is ignored entirely and the main decoder produces correct, unmodified output. Existing TTNN inference implementations require zero changes to handle this case.

This chapter first examines the MTP module architecture and the training objective, then covers how MTP can optionally function as a draft model for speculative decoding at inference time.

---

## Learning Objectives

After completing this chapter you will be able to:

1. Describe the MTP configuration parameters (`mtp_num_hidden_layers=1`, `mtp_use_dedicated_embeddings=false`) and explain what each controls.
2. Explain why training with an MTP auxiliary loss produces better internal representations in the main decoder.
3. Trace the architecture of the MTP module from the final hidden states of the main decoder through the extra transformer layer to the secondary LM head.
4. Compare Qwen3.6's MTP design to DeepSeek-V3's MTP design and identify the shared design choices.
5. Walk through the speculative decoding accept/reject loop enabled by MTP at inference time.
6. Articulate the TTNN implications of MTP for both the inference-without-speculative-decoding case and the full speculative-decoding case.

---

## Chapter Contents

| File | Description |
|------|-------------|
| [mtp_architecture_and_training.md](./mtp_architecture_and_training.md) | MTP module architecture, configuration parameters, training objective, parameter overhead, and comparison to DeepSeek-V3 |
| [speculative_decoding_inference.md](./speculative_decoding_inference.md) | How the MTP head functions as a draft model at inference time, the accept/reject loop, throughput tradeoffs, and TTNN implications |

---

## Navigation

**Previous:** [Chapter 4 — Partial Rotary Embedding and M-RoPE](../ch4_rope_and_mrope/index.md)

**Next:** [Chapter 6 — Thinking Preservation](../ch6_thinking_preservation/index.md)
