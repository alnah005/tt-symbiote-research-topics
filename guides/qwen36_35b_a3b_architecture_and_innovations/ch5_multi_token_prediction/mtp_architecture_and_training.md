# MTP Architecture and Training

## Configuration Parameters

Qwen3.6-35B-A3B exposes two MTP-specific configuration keys:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `mtp_num_hidden_layers` | `1` | Number of extra transformer layers that constitute the MTP module |
| `mtp_use_dedicated_embeddings` | `false` | The MTP prediction head shares token embeddings with the main model; no separate embedding table is allocated |

These two parameters fully specify the MTP module's footprint. `mtp_num_hidden_layers=1` means the MTP module is a single transformer layer sitting above the final hidden states of the 40-layer main decoder. `mtp_use_dedicated_embeddings=false` means the vocabulary projection at the end of the MTP module reuses the same embedding matrix as the main LM head.

---

## Module Architecture

### Data Flow

The MTP module operates on the final hidden states emitted by layer 40 of the main decoder. The following diagram summarizes the data flow at training time:

```
Input tokens  ──►  [Main Decoder: 40 layers]  ──►  H_final  (shape: [B, T, d_model])
                                                         │
                                                         ▼
                                               [MTP Transformer Layer]
                                                         │
                                                         ▼
                                                    H_mtp  (shape: [B, T, d_model])
                                                         │
                                                         ▼
                                              [Shared LM Head / Embedding^T]
                                                         │
                                                         ▼
                                             logits_mtp  (shape: [B, T, V])
                                                         │
                                                         ▼
                                            Cross-entropy vs. tokens[t+2]
```

At position $t$, the standard LM head predicts the probability of token at position $t+1$. The MTP head predicts the probability of token at position $t+2$. The extra transformer layer is responsible for transforming the representation of "what token comes next" into "what token comes two steps later."

### The Extra Transformer Layer

The MTP transformer layer is a full transformer block:

- Multi-head attention (or grouped-query attention matching the main model's configuration)
- Feed-forward sub-layer (dense or MoE — in Qwen3.6-35B-A3B this layer is **dense**, not MoE, since MTP is a small auxiliary module)
- Pre-norm with RMSNorm (matching the main model's normalization scheme)

Because this layer takes $H_\text{final}$ as input, it has full access to the contextualized representations built by all 40 main decoder layers. Its job is to further transform those representations toward a signal useful for predicting two steps ahead.

### Shared vs. Dedicated Embeddings

With `mtp_use_dedicated_embeddings=false`, the vocabulary projection at the end of the MTP path is:

$$\text{logits\_mtp} = H_\text{mtp} \cdot E^T$$

where $E \in \mathbb{R}^{V \times d_\text{model}}$ is the same embedding matrix used by the main LM head. This design:

- Reduces parameter count (no second copy of the $V \times d_\text{model}$ embedding table)
- Ensures the MTP head's output space is anchored to the same token representations as the main model
- Matches the DeepSeek-V3 design choice

---

## Training Objective

### Standard Next-Token Loss

The main decoder is trained with the standard autoregressive cross-entropy loss over the full sequence:

$$\mathcal{L}_\text{main} = -\frac{1}{T-1} \sum_{t=1}^{T-1} \log P_\theta(x_{t+1} \mid x_{\leq t})$$

### MTP Auxiliary Loss

The MTP module adds a second cross-entropy loss computed at the offset of two positions:

$$\mathcal{L}_\text{mtp} = -\frac{1}{T-2} \sum_{t=1}^{T-2} \log P_\phi(x_{t+2} \mid x_{\leq t})$$

where $\phi$ denotes the combined parameters of the main decoder plus the MTP layer (the MTP head shares the main decoder's $\theta$ and adds its own single-layer parameters).

### Combined Objective

The total training loss is a weighted sum:

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{main} + \lambda \cdot \mathcal{L}_\text{mtp}$$

A coefficient $\lambda < 1$ (commonly 0.1 to 0.3 in practice) controls how strongly the MTP signal influences training. The gradient of $\mathcal{L}_\text{mtp}$ flows back through the MTP transformer layer and into the 40 main decoder layers, applying additional representational pressure.

### Why Does This Help?

Predicting the token two steps ahead is strictly harder than predicting the token one step ahead. To succeed at the harder task, the main decoder must encode not just "what word fits here" but "what word fits next, and what word fits after that"—a richer model of local syntactic and semantic structure. Empirically, models trained with MTP auxiliary losses tend to:

- Perplexity-improve relative to the same architecture trained without MTP, even when evaluated without the MTP head
- Acquire stronger in-context learning capabilities, likely because richer next-step representations also support richer multi-step reasoning

The key insight is that the benefit accrues to the **main decoder's representations**, not to the MTP head itself. The MTP head is a scaffold for the loss signal, not a component that must be used at inference time.

---

## Parameter Overhead

The main decoder has 40 transformer layers, almost all of which are MoE layers containing 256 routed expert FFN sub-networks (plus 1 shared expert). The MTP module adds 1 **dense** transformer layer.

A single dense transformer layer with $d_\text{model}$ hidden dimension and a 4x feed-forward ratio contains approximately:

$$N_\text{layer}^\text{dense} \approx 4 \cdot d_\text{model}^2 \quad \text{(attention)} \;+\; 8 \cdot d_\text{model}^2 \quad \text{(FFN)} \;=\; 12 \cdot d_\text{model}^2$$

For $d_\text{model}=2048$: $12 \times 2048^2 \approx 50\text{M}$ parameters.

Each main MoE layer, by contrast, holds **256** routed expert FFN sub-networks each of size $d_\text{model} \times \text{intermediate}$ (plus one shared expert and a router), totaling roughly $257 \times 2 \times 2048 \times 512 \approx 537\text{M}$ parameters per layer. Across 40 layers this is on the order of 21B parameters for the main decoder layers alone (not counting embeddings and the DeltaNet projection weights).

The MTP dense layer's contribution is therefore approximately:

$$\frac{50\text{M}}{21{,}000\text{M}} \approx 0.24\%$$

of total main-decoder parameter count—far smaller than the naive "1 layer / 40 layers = 2.5%" estimate would suggest, because each MoE layer holds ~256× more FFN parameters than a comparably sized dense layer.

The shared embedding (`mtp_use_dedicated_embeddings=false`) avoids duplicating the $V \times d_\text{model}$ embedding table, which would otherwise be the largest single additional cost.

---

## Comparison to DeepSeek-V3 MTP

DeepSeek-V3 introduced MTP as a first-class training innovation in their technical report. Qwen3.6-35B-A3B adopts a nearly identical design. The table below summarizes the shared and divergent choices:

| Design Choice | DeepSeek-V3 | Qwen3.6-35B-A3B |
|---------------|-------------|-----------------|
| Extra layers for MTP | 1 | 1 (`mtp_num_hidden_layers=1`) |
| Shared embeddings | Yes | Yes (`mtp_use_dedicated_embeddings=false`) |
| MTP layer type | Dense transformer | Dense transformer |
| Prediction offset | $t+2$ | $t+2$ |
| MTP loss weight | Auxiliary (small $\lambda$) | Auxiliary (small $\lambda$) |
| Usable for speculative decoding | Yes | Yes |
| Number of MTP steps | 1 (predicts 1 additional token, at t+2) | 1 (predicts 1 additional token, at t+2) |

The convergence on `mtp_num_hidden_layers=1` and shared embeddings across two independently developed frontier MoE models is informative. One extra layer appears to be the right tradeoff: it is cheap enough that the parameter overhead is negligible, but expressive enough to transform next-step representations into two-step representations. More layers would increase cost without proportionate benefit; zero layers would collapse MTP into a trivial linear projection that cannot perform meaningful representation transformation.

The DeepSeek-V3 paper explicitly reports that the MTP auxiliary loss improves main model perplexity. Qwen3.6 follows the same design with confidence in that empirical finding.

---

**Next:** [`speculative_decoding_inference.md`](./speculative_decoding_inference.md)
