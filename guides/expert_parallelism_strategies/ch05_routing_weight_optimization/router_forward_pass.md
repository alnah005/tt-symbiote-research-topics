# Router Forward Pass

This file walks through every computational step of the Mixture-of-Experts (MoE) router for Qwen3.5-35B, from the linear projection that produces per-expert logits through to the top-k selection that determines which experts receive each token. Numerical precision considerations and the handling of the auxiliary load-balancing loss are covered in the final two sections.

Throughout this file, the following constants apply:

| Symbol | Value |
|---|---|
| $H$ | 7168 (hidden dimension) |
| $E$ | 256 (number of experts) |
| $k$ | 8 (top-k selection) |
| $B$ | token batch size |

---

## Section 1: Linear Projection

The first operation in the router is a standard matrix multiplication. For an input activation tensor $x \in \mathbb{R}^{B \times H}$ and a learned router weight matrix $W_r \in \mathbb{R}^{H \times E}$, the projection produces a logit tensor:

$$g = xW_r, \quad g \in \mathbb{R}^{B \times E}$$

For Qwen3.5-35B with $H = 7168$ and $E = 256$:

- $W_r$ has shape $[7168, 256]$.
- In BF16, $W_r$ occupies $7168 \times 256 \times 2 = 3{,}670{,}016$ bytes $\approx 3.67$ MB.
- The output logit tensor $g$ has shape $[B, 256]$; at $B = 32$, this is $32 \times 256 \times 2 = 16{,}384$ bytes = 16 KB.

This is an ordinary dense matrix multiply with no sparsity structure. On Wormhole B0, the computation maps naturally to the Tensix grid: the $[7168, 256]$ weight matrix tiles into $7168/32 = 224$ row tiles and $256/32 = 8$ column tiles, each tile of size $32 \times 32$ elements. With 80 Tensix cores available, the 8 output column tiles can be distributed across cores with good occupancy.

No bias term is used in the Qwen3.5-35B router projection; the raw dot-product logits are passed directly to the activation function.

```python
import ttnn

def router_projection(x: ttnn.Tensor, W_r: ttnn.Tensor) -> ttnn.Tensor:
    """
    Compute router logits g = x @ W_r.

    Args:
        x:   Input activations, shape [B, H] = [B, 7168], dtype BF16.
        W_r: Router weight matrix, shape [H, E] = [7168, 256], dtype BF16.

    Returns:
        g: Logit tensor, shape [B, E] = [B, 256], dtype BF16.
    """
    # Standard matmul; no transpose needed if W_r is stored column-major
    # or pre-transposed for TTNN's matmul convention.
    g = ttnn.matmul(x, W_r)
    return g
```

> **Tip:** Because $W_r$ is reused for every token in the batch, it should be pinned in L1 or DRAM with high locality. At 3.67 MB, $W_r$ exceeds the per-core L1 budget of 1.5 MB but fits comfortably when distributed across multiple cores or staged from DRAM with prefetching.

---

## Section 2: Softmax vs. Sigmoid Normalization

After the linear projection, a per-expert score is derived from each raw logit $g_e$. Two common approaches exist:

### Softmax Routing

$$\text{softmax}(g)_e = \frac{\exp(g_e)}{\sum_{j=1}^{E} \exp(g_j)}$$

Properties:
- All $E = 256$ scores are non-negative and sum to exactly 1 across experts.
- Requires computing the sum of 256 exponentials; cost scales with $E$.
- For numerical stability, the **log-sum-exp trick** is applied: subtract the row maximum before exponentiating.

$$\text{softmax}(g)_e = \frac{\exp(g_e - \max_j g_j)}{\sum_{j=1}^{E} \exp(g_j - \max_j g_j)}$$

- With softmax, the top-$k$ selected scores are a subset of a probability distribution; their sum is strictly less than 1 but they are already in comparable magnitude to one another. Re-normalization of the top-$k$ subset is still applied in practice, but the effect is smaller than with sigmoid.

### Sigmoid Routing

$$\sigma(g_e) = \frac{1}{1 + \exp(-g_e)}$$

Properties:
- Each expert score is computed independently; there is no coupling across experts.
- Each $\sigma(g_e) \in (0, 1)$, but the $E = 256$ values do **not** sum to 1.
- No global reduction across all $E$ experts is required; each expert's score can be computed in parallel.
- After top-$k$ selection, the $k$ selected sigmoid scores must be explicitly renormalized before they can be used as combination weights (see `weight_normalization.md`).

**Qwen3.5-35B uses sigmoid routing.** This choice has two important implementation consequences:

1. The router projection can feed into element-wise sigmoid without requiring a global sum, reducing synchronization overhead.
2. The combine step must carry out an explicit renormalization pass; this cannot be skipped.

```python
def router_sigmoid_scores(g: ttnn.Tensor) -> ttnn.Tensor:
    """
    Apply per-expert sigmoid to raw logits.

    Args:
        g: Raw logits, shape [B, 256], dtype BF16.

    Returns:
        scores: Sigmoid-activated scores, shape [B, 256], dtype BF16.
                Values are in (0, 1) per element; rows do NOT sum to 1.
    """
    scores = ttnn.sigmoid(g)
    return scores
```

> **Warning:** Do not treat sigmoid routing scores as a probability distribution. Their row-wise sum is not 1. Passing unnormalized sigmoid scores directly to the combine weighted-sum will produce output tensors with incorrect magnitude. See `weight_normalization.md` for the correct handling.

---

## Section 3: Top-k Selection

Given the per-expert scores (shape $[B, E] = [B, 256]$), the router selects the $k = 8$ highest-scoring experts for each token. This produces two output tensors:

- **Indices tensor:** shape $[B, k] = [B, 8]$, integer dtype; the expert index for each selected slot.
- **Scores tensor:** shape $[B, k] = [B, 8]$, BF16; the raw (unnormalized) sigmoid score for each selected expert.

Each row of the logit tensor is an independent top-$k$ problem; there are no cross-token dependencies. This structure is SIMD-friendly: all $B$ rows can be processed in parallel.

**Algorithmic complexity:** Full sort is $O(E \log E)$; partial selection (heap-based) is $O(E + k \log k)$. For $E=256$, $k=8$, partial selection is approximately $7.3\times$ faster. Full complexity analysis, the min-heap algorithm step-by-step, and batched comparison counts at $B=32$ are in `topk_selection_efficiency.md`.

> **Tip:** For the tile-parallel implementation on Wormhole B0, the 256 logit elements per token map exactly to $256 / 32 = 8$ tiles of 32 elements each. These 8 tiles can be distributed across cores for a parallel local-max computation followed by a tree reduction to identify the global top-8. This is described in detail in `topk_selection_efficiency.md`.

---

## Section 4: Auxiliary Load-Balancing Loss

During training, a load-balancing auxiliary loss is added to the model's total loss to prevent expert collapse — the tendency for routing to concentrate most tokens on a small subset of experts. The auxiliary loss takes the form:

$$\mathcal{L}_{\text{aux}} = \alpha \sum_{e=1}^{E} f_e \cdot \bar{P}_e$$

where $f_e$ is the fraction of tokens routed to expert $e$ and $\bar{P}_e$ is the mean routing probability assigned to expert $e$ across the batch.

**At inference time, the auxiliary loss is not computed.** The router's inference computation graph contains exactly three operations:

1. Linear projection: $g = xW_r$
2. Sigmoid activation: $s = \sigma(g)$
3. Top-$k$ selection: $(I, S) = \text{top-k}(s, k=8)$

There is no need to compute $f_e$, $\bar{P}_e$, or $\mathcal{L}_{\text{aux}}$ during inference. These terms should be stripped from the exported inference graph. Including them would add unnecessary compute (a reduce over the batch dimension per expert) and complicate kernel fusion.

> **Warning:** When exporting Qwen3.5-35B for T3K inference, verify that the auxiliary loss nodes are not present in the traced computation graph. Frameworks that export training graphs may include auxiliary loss computations unless explicitly removed.

---

## Section 5: Numerical Precision

**BF16 representation:** BF16 uses 7 bits for the mantissa, yielding approximately $2^{-7} \approx 0.78\%$ relative precision. For the router's purposes, this precision level is acceptable in practice.

**Logit range and sigmoid behavior:**

- Near-zero logits $g_e \approx 0$ map to $\sigma(0) = 0.5$. If many experts receive near-zero logits, the top-8 selection among near-tied experts is sensitive to small rounding differences.
- In practice, trained routers develop a clear margin between selected and non-selected experts; the top-8 scores are typically well-separated from the remaining 248 scores.
- BF16 rounding at the sigmoid input can cause a relative error of ~0.4% in the output probability, which is sufficient to flip the selection order only when two experts have nearly identical scores. For $k = 8$ out of $E = 256$, this is a rare edge case.

**Full sort vs. partial selection and precision:**

The top-$k$ selection result is identical regardless of the sort algorithm used; only the number of comparisons differs. Precision of the comparison operation is the same in both cases.

**Recommendation:** BF16 precision is sufficient for Qwen3.5-35B routing at inference. FP32 routing logits are not required. If perplexity or task-accuracy regressions are observed, inspect whether the routing distribution has shifted relative to the FP32 reference before attributing the cause to BF16 precision.

**Worked example — precision margin check:**

Suppose two experts $e_a$ and $e_b$ receive BF16 logits $g_{e_a} = 1.0$ and $g_{e_b} = 0.9921875$ (the nearest BF16 value to $0.9921875$). Their sigmoid scores are approximately:

$$\sigma(1.0) \approx 0.7311, \quad \sigma(0.992) \approx 0.7296$$

The difference is $\approx 0.0015$, which is larger than the BF16 rounding unit at this magnitude ($\approx 0.0078$), so BF16 arithmetic correctly preserves the relative ordering. Only when two logits differ by less than the BF16 rounding unit ($\approx 0.78\%$ of their magnitude) does the selection become precision-dependent.

---

## References

- Qwen Technical Report (Qwen3.5-35B), Alibaba Cloud, 2025.
- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer," ICLR 2017.
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity," JMLR 2022.
- Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding," ICLR 2021.
- Micikevicius et al., "Mixed Precision Training," ICLR 2018. (BF16 precision analysis.)
- Chapter 1 of this guide: `ch01_moe_fundamentals/routing_problem.md`.
- Chapter 5 of this guide: `topk_selection_efficiency.md`, `weight_normalization.md`.

---

**Next:** [topk_selection_efficiency.md](./topk_selection_efficiency.md)
