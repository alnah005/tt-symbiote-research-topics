# MoE Architecture

## What Is a Mixture-of-Experts Layer?

A Mixture-of-Experts (MoE) layer replaces the single feed-forward network (FFN) sublayer that appears after each attention block in a standard transformer with two components: a **gating network** (also called the router) and an **expert bank**. The expert bank is a collection of $E$ independent FFN sub-networks, where $E$ is the total number of experts. For Qwen3.5-35B, $E = 256$. The gating network examines each incoming token embedding and selects a small subset of $k$ experts — the top-$k$ experts — to process that token. For Qwen3.5-35B, $k = 8$.

This design was introduced by Shazeer et al. (2017) and has become the dominant approach for scaling language models to very large parameter counts without proportionally increasing the computation per token. The key insight is that only the $k$ selected experts actually compute for a given token; the remaining $E - k$ experts are dormant for that token. Because $k \ll E$, the total number of floating-point operations (FLOPs) per token is far smaller than if all $E$ experts computed — while the total number of parameters available to the model remains large.

### Components of an MoE Layer

An MoE layer consists of three functional parts:

**1. Gating network (router).** The router weight matrix $W_r \in \mathbb{R}^{H \times E}$, where $H$ is the hidden dimension of the model, is a learned linear projection that maps each token embedding to a vector of $E$ scalar logits. A normalization function (softmax or sigmoid) converts those logits into routing probabilities. A top-$k$ selection step identifies the $k$ experts with the highest probabilities for each token.

**2. Expert bank.** A set of $E$ independently parameterized FFN sub-networks, each with its own weight matrices. In most MoE architectures, including Qwen3.5-35B, each expert has the same architecture: two linear projections with an activation function in between. The intermediate dimension of each expert is $D$.

**3. Top-$k$ selection and weighted combination.** Each token's output is the weighted sum of the outputs produced by its $k$ selected experts, where the weights are the normalized routing probabilities for those experts.

---

## Mathematical Formulation

### Router Logit Computation

Let $x \in \mathbb{R}^{H}$ be the embedding of a single token at the input to the MoE layer ($H$ is the hidden dimension). The router computes a logit vector $g \in \mathbb{R}^{E}$ via a linear projection:

$$g = x W_r$$

where $W_r \in \mathbb{R}^{H \times E}$ are the router's learned weights. There is no bias term in most implementations; Qwen3.5-35B follows this convention.

Throughout this guide, $x$ is treated as a row vector of shape $[1, H]$, so the product $x W_r$ has shape $[1, E]$; equivalently, $G = X W_r$ for a batch of $B$ tokens where $X \in \mathbb{R}^{B \times H}$. This convention matches the Python code and TTNN API, where tensors are row-major.

### Softmax Normalization

The logit vector $g$ is passed through a softmax to produce a probability distribution over all $E$ experts:

$$p_e = \frac{\exp(g_e)}{\sum_{j=0}^{E-1} \exp(g_j)}, \quad e = 0, \ldots, E-1$$

where $p_e$ is the routing probability for expert $e$. Every $p_e \in (0, 1)$ and $\sum_{e=0}^{E-1} p_e = 1$. An alternative, sigmoid routing, does not enforce this sum-to-one constraint and is discussed in Chapter 5, `router_forward_pass.md`. Regardless of whether raw probabilities come from softmax or sigmoid, the renormalization step $\hat{w}_i = p_i / \sum_{j \in I} p_j$ (applied over the selected top-$k$ index set $I$) is always performed, so the weighted combination $y = \sum_{i \in I} \hat{w}_i \cdot o_i$ is always a convex combination of expert outputs.

### Top-$k$ Selection

Given the probability vector $\{p_e\}$, the router selects the $k = 8$ experts with the highest probabilities. Let $I \subseteq \{0, \ldots, E-1\}$ be the index set of the selected experts, $|I| = k$:

$$I = \operatorname{top-}k\bigl(\{p_e \mid e = 0, \ldots, E-1\}\bigr)$$

The routing weights for the selected experts are the softmax probabilities renormalized to the selected subset:

$$\hat{w}_i = \frac{p_i}{\sum_{j \in I} p_j}, \quad i \in I$$

This renormalization ensures that $\sum_{i \in I} \hat{w}_i = 1$ (see convexity note at line 39).

### Expert Computation and Weighted Combination

For each selected expert $i \in I$, the expert FFN computes an output $o_i \in \mathbb{R}^{H}$ from the token embedding $x$:

$$o_i = \text{FFN}_i(x)$$

The final output of the MoE layer for this token is the weighted sum:

$$y = \sum_{i \in I} \hat{w}_i \cdot o_i$$

This output $y$ is then added to the residual stream, exactly as in a standard transformer FFN sublayer.

### Batched Formulation

For a batch of $B$ tokens, the router operates on the matrix $X \in \mathbb{R}^{B \times H}$ (where the first dimension is the batch of tokens and the second is the hidden dimension):

$$G = X W_r \in \mathbb{R}^{B \times E}$$

Each row $G_t \in \mathbb{R}^{E}$ is the logit vector for token $t$. Softmax is applied row-wise to produce $P \in \mathbb{R}^{B \times E}$, and top-$k$ selection is applied independently to each row. The result is a set of $B \times k$ (token, expert) pairs, each with an associated routing weight.

```python
# Pedagogical example using PyTorch for clarity. TTNN equivalents are covered in Chapter 2.
import torch

def moe_router(x: torch.Tensor, W_r: torch.Tensor, k: int):
    """
    Compute MoE routing indices and weights.

    Args:
        x:   [B, H]  — token embeddings (batch size B, hidden dim H)
        W_r: [H, E]  — router weight matrix (E = total number of experts)
        k:   int     — number of experts to select per token

    Returns:
        indices: [B, k]  — selected expert indices (0-based), dtype int64
        weights: [B, k]  — normalized routing weights, sum to 1 per row
    """
    # Router logits: [B, H] x [H, E] -> [B, E]
    g = x @ W_r                          # shape [B, E]

    # Softmax over expert dimension
    p = torch.softmax(g, dim=-1)         # shape [B, E]

    # Top-k selection: values and indices
    topk_weights, indices = torch.topk(p, k, dim=-1)   # each [B, k]

    # Renormalize selected weights to sum to 1
    weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return indices, weights
```

---

## Sparse MoE vs. Dense MoE

### Dense MoE (All Experts Active)

A dense MoE activates all $E$ experts for every token. The output is the weighted sum across all experts:

$$y_{\text{dense}} = \sum_{e=0}^{E-1} p_e \cdot o_e$$

This is equivalent to a standard ensemble and provides no computational savings; it is $E$ times more expensive than a single FFN. Dense MoE (where all experts run on every token) is rarely used at scale, as it eliminates the compute-efficiency benefit of sparse routing.

### Sparse MoE (Top-$k$ Active)

A sparse MoE activates only $k \ll E$ experts per token. The computational cost per token scales with $k$, not $E$. For Qwen3.5-35B with $E = 256$ and $k = 8$, each token activates $k/E = 8/256 = 3.125\%$ of the experts in the expert bank. Non-expert parameters — attention weights, layer norms, dense FFN layers, and embeddings — are always active for every token; see `qwen35b_config.md` for the full active-vs.-total parameter breakdown. This is the key efficiency lever: the model has a large total parameter count (enabling capacity and specialization), but the FLOPs per token remain bounded.

The trade-off is that sparse routing introduces **discrete decisions** (which $k$ experts to select) that are not differentiable in the standard sense, and it creates **communication overhead** when experts are distributed across devices — the central topic of this guide.

### Comparison Table

| Property | Dense MoE | Sparse MoE (top-$k$) |
|---|---|---|
| Experts activated per token | All $E$ | $k$ |
| FLOPs per token | $E \times \text{FFN cost}$ | $k \times \text{FFN cost}$ |
| Communication overhead | None beyond standard FFN | Dispatch + combine collectives when experts are on different devices |
| Load balance | Trivially balanced | Requires explicit balancing mechanisms |
| Parameter efficiency | Low (all params used equally) | High (total params large, active params small) |
| Qwen3.5-35B configuration | Not used | $E=256$, $k=8$ |

---

## Qwen3.5-35B Specifics

### Expert Count and Top-$k$ Routing

Qwen3.5-35B uses $E = 256$ total experts per MoE layer with $k = 8$ routing, meaning each token engages exactly 8 of the 256 experts. This is among the highest expert counts in publicly available models and creates distinctive engineering challenges for expert parallelism, as detailed in `qwen35b_config.md`.

### Active vs. Total Parameters

The distinction between active and total parameters is central to understanding why MoE models are attractive:

- **Total parameters** include all expert weights across all $E = 256$ experts in every MoE layer, plus all non-expert parameters (embedding, attention, layer norms, dense FFN layers if any).
- **Active parameters** per token include only the $k = 8$ selected expert weights for that token, plus all non-expert parameters that are always active (attention, norms, etc.).

For Qwen3.5-35B, the total parameter count is approximately 35 billion (B) parameters, while the active parameter count per token is approximately 22 billion (manufacturer designation "A22B"; see the "Active vs. Total Parameters" section of `qwen35b_config.md` for the full breakdown). This means the model's effective computation footprint per token is similar to a 22B dense model, while its total parameter capacity is roughly 35B. Note that a first-principles estimate of active parameters yields 46.3B, which exceeds the total model size — this is a known inconsistency in the architectural constants; see the verification warning in `qwen35b_config.md`. The exact breakdown depends on whether a layer is a MoE layer or a dense layer, which is detailed in `qwen35b_config.md`.

### Expert FFN Architecture

Each expert in Qwen3.5-35B is a SiLU-gated linear unit (SwiGLU) FFN [Dauphin2017][Shazeer2020] with the following structure:

$$\text{FFN}_e(x) = \bigl(\text{SiLU}(x W_{e,\text{gate}}) \odot (x W_{e,\text{up}})\bigr) W_{e,\text{down}}$$

where $\text{SiLU}(z) = z \cdot \sigma(z)$ (Sigmoid Linear Unit, equivalent to Swish with $\beta=1$), $\sigma$ is the sigmoid function, and $\odot$ denotes element-wise multiplication.

- $W_{e,\text{gate}} \in \mathbb{R}^{H \times D}$ is the gate projection,
- $W_{e,\text{up}} \in \mathbb{R}^{H \times D}$ is the up projection,
- $W_{e,\text{down}} \in \mathbb{R}^{D \times H}$ is the down projection.

This gated architecture is standard for modern MoE models and is more expressive than a plain two-layer FFN. Its effect on expert compute cost is discussed in detail in `qwen35b_config.md`.

---

## References

- [Shazeer2017] Shazeer, N. et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", ICLR, 2017.
- [Fedus2022] Fedus, W., Zoph, B., Shazeer, N., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR, 2022.
- [Lepikhin2021] Lepikhin, D. et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding", ICLR, 2021.
- [Dauphin2017] Dauphin, Y. N. et al., "Language Modeling with Gated Convolutional Networks", ICML, 2017. (GLU activation)
- [Shazeer2020] Shazeer, N., "GLU Variants Improve Transformer", arXiv:2002.05202, 2020.
- [Qwen3] Qwen Team, Alibaba Group, "Qwen3 Technical Report", 2025.
- [Ch1Index] Chapter 1, `index.md` — chapter overview, notation table, and reading order.
- [Ch5Router] Chapter 5, `ch05_routing_weight_optimization/router_forward_pass.md` — detailed treatment of softmax vs. sigmoid routing and numerical stability.

---

**Next:** [routing_problem.md](./routing_problem.md)
