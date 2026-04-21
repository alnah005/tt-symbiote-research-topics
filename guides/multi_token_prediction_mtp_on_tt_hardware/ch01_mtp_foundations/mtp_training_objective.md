# MTP Training Objective

Multi-Token Prediction (MTP) is an auxiliary training objective that asks a language model to predict multiple future tokens simultaneously at each training step, rather than predicting only the immediately next token. This file covers the formal objective, its motivation, its key hyperparameter, its loss-weighting scheme, and how it differs from superficially similar multi-step training techniques.

---

## The Standard Next-Token Prediction Objective

Standard autoregressive (AR) language model training minimizes the cross-entropy loss over a sequence $x_1, x_2, \ldots, x_T$:

```math
\mathcal{L}_{\text{AR}} = -\frac{1}{T} \sum_{t=1}^{T} \log P(x_{t+1} \mid x_1, \ldots, x_t)
```

At each position $t$, the model receives tokens $x_1, \ldots, x_t$ and produces a single logit distribution over the vocabulary $V$. Only the prediction of $x_{t+1}$ contributes to the loss. This is a well-understood objective, but it provides only a single scalar gradient signal per token position.

---

## The MTP Formulation

The MTP objective, introduced in the context of large-scale language model training (most prominently in DeepSeek-V3 and formalized by Gloeckle et al., 2024), extends the training signal by requiring the model to also predict $N$ additional future tokens at each position $t$:

```math
\mathcal{L}_{\text{MTP}}^{(k)}(t) = -\log P_k(x_{t+k+1} \mid x_1, \ldots, x_t), \quad k = 1, \ldots, N
```

where $P_k$ is a distribution produced by the $k$-th MTP head block applied to an intermediate hidden representation at position $t$. The full MTP auxiliary loss averages across positions and prediction depths:

```math
\mathcal{L}_{\text{aux}} = \frac{1}{N} \sum_{k=1}^{N} \frac{1}{T} \sum_{t=1}^{T} \mathcal{L}_{\text{MTP}}^{(k)}(t)
```

The combined training objective is:

```math
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{AR}} + \lambda \cdot \mathcal{L}_{\text{aux}}
```

where $\lambda$ is a scalar loss-weighting hyperparameter (discussed further below).

The key architectural implication is that each prediction depth $k$ requires its own hidden state: the MTP head takes the backbone's hidden state and a shifted token embedding as inputs, runs them through one or more additional transformer blocks, and produces auxiliary logits. For depth $k = 1$ (the common single-block case), a single MTP head block processes the backbone's final hidden state combined via element-wise addition (after independent layer normalization) with the embedding of $x_{t+1}$ to produce a distribution over $x_{t+2}$.

---

## Motivation

MTP addresses several limitations of purely next-token prediction training:

**Gradient signal enrichment.** Each training token contributes $N + 1$ loss terms instead of one. The backbone's hidden states at position $t$ receive gradient signal that depends not only on $x_{t+1}$ but on $x_{t+2}, \ldots, x_{t+N+1}$. This multi-horizon gradient encourages the backbone to encode richer contextual representations — representations that anticipate the trajectory of the sequence several steps ahead.

**Training efficiency.** More gradient signal per forward pass means that the model can reach a given perplexity in fewer training steps. DeepSeek-V3 reports measurable perplexity improvement on language modeling benchmarks when MTP is added as an auxiliary loss, without increasing the backbone parameter count.

**No inference-time cost for standard AR generation.** The MTP head weights are present in the checkpoint and the backbone is trained to produce hidden states that feed the MTP head — but a standard AR generation loop that ignores the MTP head incurs no additional compute cost. The backbone forward pass is identical regardless of whether the MTP head is subsequently applied.

**Compatibility with speculative decoding.** Because the MTP head has been trained to predict $x_{t+2}$ given the backbone's hidden state at position $t$, its output logits can serve as a draft distribution for speculative decoding at inference time. This is the central application motivation for this guide; it is developed fully in Chapter 4.

---

## Key Hyperparameter: `mtp_num_hidden_layers`

The field `mtp_num_hidden_layers` in the model configuration controls how many transformer decoder blocks the MTP head contains. Each block handles one additional prediction depth. For Qwen3.6-35B-A3B, this value is 1, meaning:

- Draft depth $N = 1$.
- One additional transformer block is appended after the backbone.
- At each training position $t$, the model predicts $x_{t+1}$ (primary loss) and $x_{t+2}$ (MTP auxiliary loss).

The chained-input structure for multi-block configurations is described in detail in [`mtp_head_architecture.md`](./mtp_head_architecture.md).

---

## Loss Weighting

The combined loss:

```math
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{AR}} + \lambda \cdot \mathcal{L}_{\text{aux}}
```

uses $\lambda$ as a scalar training hyperparameter. In DeepSeek-V3, $\lambda = 0.3$ is reported. The Qwen3 technical report does not disclose a specific value for $\lambda$.

Critically, $\lambda$ is a training hyperparameter: it governs how the gradients from the auxiliary heads are weighted against the primary next-token loss during the backward pass. It does **not** appear in the inference-time model configuration (`config.json`), and it does not affect the behavior of the model after training is complete. A practitioner loading a trained Qwen3.6 checkpoint will not find $\lambda$ in the config; only `mtp_num_hidden_layers` (which controls the head's architecture) is relevant post-training.

---

## Comparison to Related Multi-Step Training Objectives

MTP is one of several techniques that use multi-step prediction signals during training. The distinctions matter because they determine what the trained model can and cannot do at inference time.

| Technique | What is predicted | Gradient flows through | Inference-time usage |
|-----------|-------------------|------------------------|----------------------|
| MTP (this guide) | Future tokens $x_{t+2}, \ldots, x_{t+N+1}$ | Dedicated MTP head blocks + backbone | Draft logits for speculative decoding |
| Token-level knowledge distillation | Teacher's soft label distribution over $x_{t+1}$ | Backbone only | None (teacher not needed at inference) |
| Consistency regularization | Self-consistency of representations under perturbation | Backbone only | None |
| Blockwise parallel decoding | Future tokens via sequential full decoder blocks | Full transformer decoder blocks (one per future position) | Can be used directly for draft tokens |

Compared to Medusa heads (Cai et al., 2024), which are simple linear projections with no transformer blocks, MTP heads achieve higher draft acceptance rates due to their shifted-token-embedding conditioning (`hnorm`/`enorm` input combination). Compared to Blockwise Parallel Decoding (Stern et al., 2018), which also uses full transformer decoder blocks, MTP's key differentiator is the same shifted-token-embedding conditioning — the MTP head receives the ground-truth next token as an explicit input to each prediction depth (see Chapter 4, `acceptance_rate_estimation.md`).

---

## References

- [Gloeckle2024] Gloeckle, F., Idrissi, B.Y., Rozière, B., Lopez-Paz, D., and Synnaeve, G., "Better & Faster Large Language Models via Multi-token Prediction", arXiv:2404.19737, 2024.
- [DeepSeek-V3] DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv:2412.19437, 2024.
- [Qwen3] Qwen Team, "Qwen3 Technical Report", Alibaba Cloud, 2025.
- [Stern2018] Stern, M., Shazeer, N., and Uszkoreit, J., "Blockwise Parallel Decoding for Deep Autoregressive Models", NeurIPS 2018.
- [Cai2024] Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J.D., Chen, D., and Dao, T., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads", arXiv:2401.10774, 2024.

---

**Next:** [`mtp_head_architecture.md`](./mtp_head_architecture.md)
