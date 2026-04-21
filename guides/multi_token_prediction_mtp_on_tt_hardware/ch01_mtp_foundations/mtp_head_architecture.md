# MTP Head Architecture

This file describes the physical structure of the Multi-Token Prediction (MTP) head: how it is wired to the backbone, what transformer components it contains, what inputs it receives, what outputs it produces, and how its architecture scales with `mtp_num_hidden_layers`. An open question about KV cache behavior at inference time is flagged at the end and resolved in Chapter 3.

---

## Structural Overview

The MTP head is a stack of `mtp_num_hidden_layers` transformer decoder blocks appended after the main transformer backbone. Each block is a full transformer decoder block in the same sense as the backbone's layers: it includes a self-attention sub-layer, a feed-forward network (FFN) sub-layer, and the associated layer normalization and residual connections. For Qwen3.6-35B-A3B with `mtp_num_hidden_layers: 1`, the MTP head contains exactly one such block.

### Diagram

```
Input tokens x_1 ... x_t
        │
        ▼
┌─────────────────────────────────────────┐
│         Backbone Transformer Layers      │
│  (L backbone blocks, embedding in,      │
│   final hidden state h_t out)            │
└─────────────────────────────────────────┘
        │
        │  h_t  [B, S, H]
        ▼
┌─────────────────────────────────────────┐
│         Input Combination               │
│  hnorm(h_t) + enorm(embed(x_{t+1}))     │
│  → combined input c_t  [B, S, H]        │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│         MTP Head Block 1                │
│  (full transformer decoder block:       │
│   attention + FFN + layer norms)        │
└─────────────────────────────────────────┘
        │
        │  mtp_hidden  [B, S, H]
        ▼
┌─────────────────────────────────────────┐
│         lm_head (shared)                │
│  linear: H → V                          │
└─────────────────────────────────────────┘
        │
        ▼
  auxiliary logits for x_{t+2}  [B, S, V]
```

In the multi-block case (`mtp_num_hidden_layers > 1`), additional MTP head blocks are chained in series. Block $k$ receives the output hidden state of block $k-1$ as its hidden state input, combined with the embedding of $x_{t+k}$ (the embedding is shifted by one position for each successive prediction depth). Block $k$ produces auxiliary logits for position $t + k + 1$.

---

## Input to the MTP Head

The MTP head at depth $k = 1$ receives two inputs that are combined before being passed into the head block:

1. **Backbone final hidden state** $h_t$ with shape `[B, S, H]`. This is the output of the last backbone layer at sequence position $t$ — the same tensor that would normally be fed directly into the `lm_head` to produce the primary next-token logits.

2. **Shifted token embedding** $\text{embed}(x_{t+1})$ with shape `[B, S, H]`. The embedding of the token at position $t+1$ (the ground-truth next token, available during training; the predicted/sampled token during inference). This embedding is shifted by one position relative to the input sequence.

These two tensors are each independently layer-normalized (using dedicated `hnorm` and `enorm` layer norm weights, each separate from the backbone's layer norms), then added element-wise to produce the combined input $c_t$:

```math
c_t = \text{LayerNorm}_{\text{hnorm}}(h_t) + \text{LayerNorm}_{\text{enorm}}(\text{embed}(x_{t+1}))
```

The resulting combined input $c_t$ has shape `[B, S, H]` and is passed into the MTP head block's self-attention sub-layer.

This input construction is the key architectural decision that distinguishes the MTP head from a simple classifier on top of $h_t$: by injecting the embedding of the current prediction target $x_{t+1}$, the MTP head can condition its prediction of $x_{t+2}$ on what token was just accepted. This is why MTP drafts are often higher quality than purely independent multi-head predictions (cf. Medusa-style parallel heads).

---

## Output of the MTP Head

The MTP head block's final hidden state $h_t^{\text{mtp}}$ with shape `[B, S, H]` is passed through the `lm_head` — a linear projection from $H$ to $V$ (vocabulary size) — to produce auxiliary logits:

```math
\text{logits}_{\text{aux}, k=1} = h_t^{\text{mtp}} W_{\text{lm\_head}}^T, \quad \text{shape: } [B, S, V]
```

The `lm_head` weight matrix ($W_{\text{lm\_head}}$, shape `[V, H]`) is **shared** between the backbone and the MTP head. There is a single `lm_head` in the model, and both the backbone's primary predictions and each MTP head block's auxiliary predictions are produced by projecting through this same matrix.

For a model with `mtp_num_hidden_layers: 1`, there is one auxiliary logit tensor of shape `[B, S, V]` per training step, targeting position $t+2$. For `mtp_num_hidden_layers: N`, there are $N$ auxiliary logit tensors, targeting positions $t+2$ through $t+N+1$.

---

## Shared vs. Unshared Parameters

The parameter-sharing situation in the MTP head has two components that must be distinguished:

**`lm_head`: always shared.** The language model head that projects from $H$ to $V$ is shared between the backbone and all MTP head blocks. This is a deliberate design choice: sharing the `lm_head` prevents the model from learning different output distributions for primary vs. auxiliary predictions, and it reduces the total parameter count significantly (a separate `lm_head` per depth would add $N \times H \times V$ parameters).

**Transformer block weights: not shared with the backbone.** The MTP head block's self-attention projection matrices ($W_Q$, $W_K$, $W_V$, $W_O$) and FFN matrices (gate, up, down projections) are **separate** from all backbone layer weights in Qwen3.6-35B-A3B. The MTP head is not a re-use of backbone layer 0 or the last backbone layer; it is a new set of weights initialized and trained independently.

This is in contrast to some earlier MTP-like formulations that proposed reusing backbone layer weights for the auxiliary heads. Qwen3.6 and DeepSeek-V3 both use separate MTP head weights.

**Summary:**

| Component | Shared with backbone? |
|-----------|----------------------|
| `lm_head` (H → V projection) | Yes |
| Token embedding table | Yes (same embedding used for shifted token input) |
| Self-attention projections (Q, K, V, O) | No — separate MTP head weights |
| FFN projections (gate, up, down) | No — separate MTP head weights |
| Layer norm weights (pre-attn, pre-FFN) | No — separate MTP head weights |
| Input normalization (hnorm, enorm) | No — dedicated MTP head weights |

---

## One-Block vs. Multi-Block MTP Heads

For `mtp_num_hidden_layers: 1` (the Qwen3.6-35B-A3B case):

- Draft depth $N = 1$.
- One transformer decoder block is appended.
- The MTP head produces one auxiliary logit tensor targeting position $t+2$.
- At inference time, this enables one-step speculative drafting: each backbone forward pass produces a primary prediction for $t+1$ and a draft prediction for $t+2$.

For `mtp_num_hidden_layers: K` (hypothetical multi-block case):

- Draft depth $N = K$.
- $K$ transformer decoder blocks are chained in series.
- Block $k$ receives the output of block $k-1$ combined with $\text{embed}(x_{t+k})$.
- The MTP head produces $K$ auxiliary logit tensors targeting positions $t+2$ through $t+K+1$.
- At inference time, this would enable $K$-step speculative drafting from a single backbone pass.

The weight count grows linearly with `mtp_num_hidden_layers`. Each additional block adds the same number of parameters as one backbone layer (assuming matching hyperparameters). The training cost grows proportionally; each forward pass must run the full chain of $K$ MTP blocks and compute $K$ cross-entropy terms.

---

## Relationship to the Backbone's Layer Stack

The MTP head sits strictly after the backbone's final layer norm and after $h_t$ has been produced. It does not tap into intermediate backbone hidden states. This is in contrast to some other auxiliary head designs (e.g., those that attach prediction heads at multiple intermediate backbone layers). The MTP head in Qwen3.6 and DeepSeek-V3 uses only the backbone's final output representation.

Because the MTP head uses only the final backbone hidden state, it cannot begin computing until the full backbone pass completes — there is no intra-pass pipelining opportunity.

---

## Open Question: KV Cache in the MTP Head at Inference Time

At training time, the MTP head processes full-length sequences in a standard teacher-forced causal pass. The self-attention sub-layer in the MTP head operates with a causal mask over all $S$ positions, exactly as backbone layers do.

At inference time — specifically during the decode phase — the question arises: does the MTP head maintain its own KV cache, or does it recompute attention keys and values from scratch at each generation step?

This is a non-trivial question because:

1. If the MTP head uses a KV cache, it must allocate and manage a separate set of key-value tensors for its one attention layer, at the cost of additional DRAM for the KV cache entries.
2. If the MTP head does not use a KV cache, it must reprocess all $S$ tokens at every decode step, which at long sequence lengths would be prohibitively expensive.
3. The standard HuggingFace Transformers generation loop (`model.generate()`) applies KV caching to all transformer layers that participate in the decode forward pass. Whether the MTP head participates in the decode forward pass at all — and therefore whether it even needs a KV cache — is itself an open question.

**This question is resolved in Chapter 3, `huggingface_mtp_forward_pass.md`.** That file traces the HuggingFace code path for Qwen3.6's `forward()` method and determines definitively whether the MTP head is invoked during inference and whether it manages its own KV cache entries.

---

## References

- [DeepSeek-V3] DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv:2412.19437, 2024.
- [Gloeckle2024] Gloeckle, F., Idrissi, B.Y., Rozière, B., Lopez-Paz, D., and Synnaeve, G., "Better & Faster Large Language Models via Multi-token Prediction", arXiv:2404.19737, 2024.
- [Qwen3] Qwen Team, "Qwen3 Technical Report", Alibaba Cloud, 2025.
- [Cai2024] Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J.D., Chen, D., and Dao, T., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads", arXiv:2401.10774, 2024.

---

**Next:** [`qwen36_mtp_config.md`](./qwen36_mtp_config.md)
