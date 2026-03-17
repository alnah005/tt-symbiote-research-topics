# Weight Normalization

When Mixture-of-Experts (MoE) routing uses sigmoid activation rather than softmax, the $k$ selected routing weights do not sum to 1. This breaks the standard weighted-combination step at the output of the expert sub-layer unless an explicit renormalization is applied. This file explains why normalization is required for Qwen3.5-35B, analyzes two timing options (normalize before dispatch vs. defer to combine), and recommends the deferred approach for T3K hardware.

Throughout this file, the following constants apply:

| Symbol | Value |
|---|---|
| $E$ | 256 (number of experts) |
| $k$ | 8 (top-k selection) |
| $B$ | token batch size |
| $N$ | 8 (T3K devices) |

---

## Section 1: Why Normalization Is Needed for Sigmoid Routing

### The Sigmoid Score Distribution

Recall from `router_forward_pass.md` that Qwen3.5-35B uses sigmoid routing. For a given token, the sigmoid score for expert $e$ is:

$$s_e = \sigma(g_e) = \frac{1}{1 + e^{-g_e}}, \quad s_e \in (0, 1)$$

Each score is computed independently. The $E = 256$ scores are independent Bernoulli-like probabilities; they have no joint normalization constraint. In particular:

$$\sum_{e=1}^{E} s_e \neq 1 \quad \text{(in general)}$$

After top-$k$ selection, the $k = 8$ selected scores $\{s_{i_1}, s_{i_2}, \ldots, s_{i_8}\}$ also do not sum to 1:

$$\sum_{j=1}^{k} s_{i_j} \neq 1$$

### The Consequence for the Combine Step

The combine step computes the token's final output as a weighted sum of the outputs produced by the $k$ selected experts:

$$y = \sum_{j=1}^{k} w_{i_j} \cdot \text{ExpertOutput}(i_j, x)$$

If $w_{i_j} = s_{i_j}$ (raw sigmoid scores), the output magnitude scales with $\sum_j s_{i_j}$. For a token where all 8 selected experts have sigmoid scores of 0.8, the effective scaling is $8 \times 0.8 = 6.4$, which is far from the intended unit-magnitude combination. The output distribution would be inconsistent across tokens with different score magnitudes.

### The Normalization Formula

The correct combination weights are:

$$w_{i_j}^{\text{norm}} = \frac{s_{i_j}}{\displaystyle\sum_{l=1}^{k} s_{i_l}}, \quad j = 1, \ldots, k$$

This renormalization ensures $\sum_j w_{i_j}^{\text{norm}} = 1$ and the combine step is a proper convex combination of expert outputs.

### Comparison with Softmax Routing

With softmax routing, the $E = 256$ scores form a probability distribution over all experts. The top-$k$ selected scores are a subset of this distribution; their sum is strictly less than 1 but they share a common scale. In practice, softmax-routed models also apply the same renormalization formula, but the effect is smaller because the selected scores are already in comparable magnitude to a unit-sum constraint.

For sigmoid routing, the renormalization is functionally mandatory rather than a minor correction.

---

## Section 2: Timing of Normalization

There are two natural points at which to apply the renormalization: immediately after top-$k$ (before dispatch) or deferred to the combine step (after expert computation).

### Option A: Normalize Before Dispatch

In this approach, the router on the originating device computes the normalized weights immediately after top-$k$ selection:

```
sigmoid → top-k → normalize [B, k] weights → dispatch (tokens + normalized weights)
```

The dispatch buffer carries both token embeddings and normalized combination weights. For each of the $k = 8$ selected experts, the buffer entry includes:

- Token embedding: $H = 7168$ BF16 elements = $14{,}336$ bytes
- Normalized weight: 1 BF16 element = 2 bytes

Weight overhead per token: $k \times 2 = 16$ bytes. For $B = 32$: $32 \times 16 = 512$ bytes total, added to the dispatch payload.

**Advantage:** The combine step is simple; it receives pre-normalized weights and applies a straightforward weighted sum.

**Disadvantage:** The all-to-all dispatch payload carries weight metadata. While 512 bytes is small relative to the token embedding payload ($32 \times 14{,}336 = 458{,}752$ bytes), it increases the dispatch buffer structure complexity and requires that the weight values be packed and unpacked alongside embeddings across device boundaries.

### Option B: Defer Normalization to Combine

In this approach, raw (unnormalized) sigmoid scores are retained on the originating device while only token embeddings are transmitted in the dispatch all-to-all:

```
sigmoid → top-k → dispatch (tokens only) → expert compute → combine + normalize
```

The originating device retains the $[B, k]$ raw score tensor in local L1 during the dispatch and expert compute phases. When $k$ expert outputs return via the combine all-to-all, the originating device applies renormalization and weighted accumulation in a single fused kernel:

$$y_b = \frac{\sum_{j=1}^{k} s_{i_j} \cdot o_{i_j}}{\sum_{j=1}^{k} s_{i_j}}$$

where $o_{i_j}$ is the output from expert $i_j$ for token $b$.

**Advantage:** The dispatch payload contains only token embeddings; no weight metadata is embedded in the all-to-all buffer structure. The combine kernel handles normalization and accumulation in one pass, exposing a natural fusion opportunity.

**Disadvantage:** The raw score tensor $[B, k]$ must remain live in local L1 from the end of routing through the completion of the combine step. This spans the dispatch latency plus expert compute time.

---

## Section 3: Recommended Approach — Defer to Combine

The recommended approach for T3K is **Option B: defer normalization to the combine step**.

### Rationale

1. **Payload simplicity:** The dispatch all-to-all operates on uniform token embedding tensors. Embedding weight scalars into the dispatch payload creates a heterogeneous buffer layout that complicates both the dispatch packing kernel and the receive-side unpack kernel.

2. **L1 residency cost is trivial:** The $[B, k]$ raw score tensor that must be retained locally has a very small footprint.

   $$[B, k] \text{ in BF16} = B \times k \times 2 \text{ bytes} = 32 \times 8 \times 2 = 512 \text{ bytes}$$

   512 bytes is negligible relative to the 1.5 MB per-core L1 budget and can be kept live throughout the dispatch and expert compute phases without pressure.

3. **Fusion opportunity at combine:** The combine kernel already receives $k$ expert output tensors per token and must accumulate them. Adding a per-token scalar normalization step costs a single reduction and $k$ multiplications, naturally fused into the accumulation loop. See `router_kernel_fusion.md`, Section 3, for the fused normalization + scatter metadata kernel.

4. **No extra all-to-all for weights:** Raw scores are kept local; they do not need to be communicated across devices. Only the combine outputs (shape $[B, H] = [B, 7168]$) travel back to the originating devices via the combine all-to-all, as they would in any case.

### Implementation Sketch

```python
def combine_with_renormalization(
    expert_outputs: list,   # k tensors, each shape [B, H]
    raw_scores: list,       # k tensors, each shape [B], raw sigmoid scores (local)
) -> "ttnn.Tensor":
    """
    Renormalize sigmoid weights and compute weighted sum of expert outputs.

    Args:
        expert_outputs: List of k=8 tensors from expert computation,
                        each shape [B, H] = [B, 7168], dtype BF16.
        raw_scores:     List of k=8 score tensors, each shape [B], dtype BF16.
                        These are raw unnormalized sigmoid scores retained
                        locally since the routing step.

    Returns:
        y: Combined output, shape [B, H] = [B, 7168], dtype BF16.
    """
    import ttnn

    # Stack raw scores to shape [B, k] and compute row-wise sum for normalization
    scores_stacked = ttnn.stack(raw_scores, dim=1)       # [B, k]
    score_sum = ttnn.sum(scores_stacked, dim=1, keepdim=True)  # [B, 1]
    norm_weights = ttnn.div(scores_stacked, score_sum)   # [B, k], sums to 1 per row

    # Weighted accumulation
    outputs_stacked = ttnn.stack(expert_outputs, dim=1)  # [B, k, H]
    norm_weights_expanded = ttnn.unsqueeze(norm_weights, dim=2)  # [B, k, 1]
    y = ttnn.sum(outputs_stacked * norm_weights_expanded, dim=1)  # [B, H]
    return y
```

> **Tip:** In the actual TTNN kernel, the stack + multiply + sum sequence above should be fused into a single kernel to avoid materializing the intermediate $[B, k, H]$ tensor, which at $B=32$, $k=8$, $H=7168$ is $32 \times 8 \times 7168 \times 2 = 3{,}670{,}016$ bytes $\approx 3.67$ MB. Kernel fusion reduces this to a streaming accumulation with $O(H)$ working memory per token.

---

## Section 4: Precision Considerations

### BF16 Routing Weights

BF16 provides 7 bits of mantissa precision. For routing weight values $s_{i_j} \in (0, 1)$, the representable precision is approximately $2^{-7} \approx 0.78\%$ relative error. After renormalization, normalized weights $w_{i_j}^{\text{norm}} = s_{i_j} / \sum_l s_{i_l}$ carry the same relative precision.

### Accumulation Error with $k = 8$ Terms

The weighted sum $y = \sum_{j=1}^{k} w_j \cdot o_j$ accumulates $k = 8$ terms. In BF16 sequential accumulation, each addition introduces rounding. With $k = 8$, the accumulated rounding error is bounded by approximately:

$$\epsilon_{\text{accum}} \approx k \times \epsilon_{\text{BF16}} \approx 8 \times 2^{-7} \approx 6\%$$

In practice, the error is much smaller because not all terms are equally large and partial cancellations occur. Empirically, BF16 accumulation for $k = 8$ terms at hidden dimension $H = 7168$ produces PCC (Pearson correlation coefficient) values above 0.999 relative to FP32 reference.

### FP32 Accumulation Option

For higher precision, the combine accumulation can be performed in FP32:

- Accumulator buffer: $[B, H]$ in FP32 = $32 \times 7168 \times 4 = 917{,}504$ bytes $\approx 896$ KB
- This exceeds per-core L1 (1.5 MB) but fits with careful management
- Final conversion to BF16 before passing to the next layer

**For Qwen3.5-35B:** BF16 accumulation is the standard practice. FP32 accumulation should only be enabled if a measurable PCC degradation is observed on a representative evaluation set.

> **Warning:** Do not conflate routing precision (BF16 is sufficient for top-k selection) with combine accumulation precision (BF16 accumulation of 8 terms may introduce ~1–2 bits of rounding error in the output). These are separate concerns. The routing decision determines which experts execute; the accumulation precision determines the quality of the combined output.

### Worked Example — Normalization Effect

Consider a single token with 8 selected experts having raw sigmoid scores:

$$[0.82, 0.79, 0.75, 0.71, 0.68, 0.65, 0.61, 0.58]$$

Sum: $0.82 + 0.79 + 0.75 + 0.71 + 0.68 + 0.65 + 0.61 + 0.58 = 5.59$

Without normalization, the effective output scaling is $5.59\times$ the average expert output. After normalization:

$$w^{\text{norm}} = [0.147, 0.141, 0.134, 0.127, 0.122, 0.116, 0.109, 0.104]$$

Sum: $1.000$. The combine step now computes a proper weighted average of the expert outputs.

Had softmax been used instead, a typical top-8 selection might yield scores summing to $\approx 0.31$ (the top-8 fraction of a 256-way distribution), requiring the same renormalization formula to produce unit-sum weights.

---

## References

- Qwen Technical Report (Qwen3.5-35B), Alibaba Cloud, 2025.
- Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding," ICLR 2021.
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity," JMLR 2022.
- Chapter 2 of this guide: `ch02_all_to_all_primitives/all_to_all_combine.md`.
- Chapter 5 of this guide: `router_forward_pass.md`, `router_kernel_fusion.md`.
