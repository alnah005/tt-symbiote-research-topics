# Capacity Overflow Handling

## Overview

When the number of tokens routed to expert $e$ in a single forward pass exceeds the expert's capacity $C$, an **overflow** event occurs. This file defines overflow precisely, explains its impact on output quality, derives the two handling policies (hard drop and reassign), analyzes the combine-step weight renormalization required by each, and provides a worked example of the Poisson-approximate token-drop rate at $B = 32$ with $CF = 1.25$.

**Prerequisites:** `load_imbalance_detection.md` (definitions of $f_e$, $C$, $CV$, and utilization); Chapter 1, `ch01_moe_fundamentals/routing_problem.md` (dispatch/combine pattern, token dropping); Chapter 2, `ch02_all_to_all_primitives/all_to_all_combine.md` (combine buffer layout and weight accumulation).

---

## 1. Expert Capacity: Definition and Value

The expert capacity $C$ is the maximum number of tokens that expert $e$ will accept and process in one forward pass. It is set at dispatch time based on the batch size $B$, the routing top-$k$, the total expert count $E$, and the capacity factor $CF$:

$$C \;=\; \left\lceil \frac{k \cdot B \cdot CF}{E} \right\rceil \;=\; \left\lceil \frac{8 \cdot B \cdot 1.25}{256} \right\rceil \;=\; \left\lceil \frac{B}{25.6} \right\rceil$$

**Values for common batch sizes:**

| Batch size $B$ (tokens) | $k \cdot B \cdot CF / E$ | $C$ (rounded up) |
|---|---|---|
| 1 | $8 \times 1 \times 1.25 / 256 = 0.039$ | 1 |
| 8 | $8 \times 8 \times 1.25 / 256 = 0.3125$ | 1 |
| 16 | $8 \times 16 \times 1.25 / 256 = 0.625$ | 1 |
| 32 | $8 \times 32 \times 1.25 / 256 = 1.25$ | 2 |
| 64 | $8 \times 64 \times 1.25 / 256 = 2.5$ | 3 |
| 128 | $8 \times 128 \times 1.25 / 256 = 5.0$ | 5 |

The capacity factor $CF = 1.25$ provides 25% headroom above the expected average load $k \cdot B / E = B/32$ per expert. At $B = 32$, the expected average is $32/32 = 1.0$ token per expert, and $C = 2$ allows up to 2 tokens — double the mean — before overflow is triggered.

---

## 2. The Overflow Condition

**Overflow for expert $e$ at step $t$** occurs when the actual token count $n_e(t)$ satisfies:

$$n_e(t) \;>\; C$$

The number of dropped tokens for expert $e$ at step $t$ is:

$$d_e(t) \;=\; \max(0,\; n_e(t) - C)$$

The dispatch mechanism enforces this limit: tokens are accepted in the order they arrive at the dispatch buffer (see Chapter 2, `ch02_all_to_all_primitives/all_to_all_dispatch.md`). Once $C$ tokens have been accepted for expert $e$, all subsequent tokens for that expert in the same step are dropped — they are not forwarded to the expert's device and receive no computation.

---

## 3. Impact on Output Quality

**Token output for the combine step (Chapter 2, `ch02_all_to_all_primitives/all_to_all_combine.md`):**

In the standard MoE forward pass, the output for token $b$ is:

$$\mathbf{y}_b \;=\; \sum_{i \in \mathcal{S}(b)} w_{b,i}^{\text{norm}} \cdot \text{FFN}_i(\mathbf{x}_b)$$

where $\mathcal{S}(b) \subseteq \{0, \ldots, E-1\}$ is the set of experts that actually processed token $b$, $|\mathcal{S}(b)| = k$ under no overflow, and $w_{b,i}^{\text{norm}}$ are the normalized routing weights.

When one or more experts overflow and token $b$ is dropped for expert $i^* \in \{0, \ldots, k-1\}$, we have $\mathcal{S}(b) \subsetneq \text{top-}k(b)$ and $|\mathcal{S}(b)| < k$. The dropped expert $i^*$ contributes **zero** to the output: its $\text{FFN}_{i^*}(\mathbf{x}_b)$ term is absent from the sum.

This introduces a systematic error: the missing expert's contribution is replaced by zero, not by any approximation of the expert's actual output. For experts with large routing weights $w_{b,i^*}$, the error in $\mathbf{y}_b$ can be significant.

---

## 4. Overflow Handling Policies

Two policies govern what happens to an overflowed token.

### 4.1 Policy A: Hard Drop

The overflowed token's assignment to the full expert is simply discarded. The combine step uses only the experts in $\mathcal{S}(b)$ that did accept the token. The routing weights for these experts are **renormalized** so that the combine weights still sum to 1.0:

$$w_{b,i}^{\text{norm, drop}} \;=\; \frac{w_{b,i}}{\sum_{j \in \mathcal{S}(b)} w_{b,j}}, \quad i \in \mathcal{S}(b)$$

The output becomes:

$$\mathbf{y}_b^{\text{drop}} \;=\; \sum_{i \in \mathcal{S}(b)} w_{b,i}^{\text{norm, drop}} \cdot \text{FFN}_i(\mathbf{x}_b)$$

**Properties of hard drop:**
- Simple to implement: the combine buffer simply omits the missing expert slot; the renormalization is a divide by the partial weight sum.
- Weight renormalization is always applied even when $|\mathcal{S}(b)| < k$, preserving the combine output scale.
- The combine step incurs fewer terms and thus slightly less compute, but this saving is negligible.
- Output quality degrades proportionally to the routing weight $w_{b,i^*}$ of the dropped expert.

### 4.2 Policy B: Reassign (Next-Best Expert)

Instead of dropping the token entirely, the dispatch mechanism identifies the next-best expert from the router's sorted logit list (the $(k+1)$-th ranked expert) and routes the token there instead, provided that expert has spare capacity.

$$\text{If } n_{i^*}(t) > C: \quad \text{route token } b \text{ to expert } i^{*(k+1)}(b) \text{ instead of } i^*(b)$$

The routing weight for the substitute expert is taken from the router's original output. If the substitute expert is also over capacity, the process can cascade to the $(k+2)$-th best expert, up to a configurable fallback depth.

**Properties of reassignment:**
- Preserves $|\mathcal{S}(b)| = k$ in most cases (as long as a substitute expert with spare capacity exists).
- More complex dispatch: requires the router to expose the $(k+1)$-th through $(k+d)$-th expert indices as fallback candidates.
- Adds latency to the dispatch step: fallback candidates must be communicated to the dispatch kernel [VERIFY: `ttnn.moe_dispatch_with_fallback` or equivalent].
- Substitute expert may be a poor semantic match for the token, potentially degrading quality differently than hard drop.
- In practice, reassignment is beneficial when overflow is rare (low $CV$, small number of hot experts); when overflow is widespread, the fallback list is exhausted and hard drop is unavoidable anyway.

**Implementation note:** For Qwen3.5-35B on T3K, the router currently exposes only top-$k$ indices to the dispatch buffer (see Chapter 2, `ch02_all_to_all_primitives/all_to_all_dispatch.md`). Supporting reassignment requires routing $k+d$ indices per token (where $d$ is the fallback depth), increasing dispatch metadata volume by a factor of $(k+d)/k$ [VERIFY: metadata format changes in TTNN dispatch kernel].

---

## 5. Weight Renormalization Detail

Regardless of policy, the combine step always renormalizes routing weights over the set of experts that actually contributed. This prevents the output scale from collapsing when $|\mathcal{S}(b)| < k$.

For hard drop, the renormalized weight for expert $i \in \mathcal{S}(b)$ is:

$$w_{b,i}^{\text{norm}} \;=\; \frac{w_{b,i}}{Z_b}, \quad Z_b \;=\; \sum_{j \in \mathcal{S}(b)} w_{b,j}$$

For $|\mathcal{S}(b)| = k$ (no drop), $Z_b = \sum_{j=1}^{k} w_{b,j}$ is the standard normalization applied in the no-overflow case as well. For $|\mathcal{S}(b)| < k$:

$$Z_b^{\text{drop}} \;=\; \sum_{j \in \mathcal{S}(b)} w_{b,j} \;=\; Z_b - \sum_{i^* \text{ dropped}} w_{b,i^*} \;<\; Z_b$$

Dividing by the smaller $Z_b^{\text{drop}}$ inflates the surviving weights, partially compensating for the missing expert contributions.

**Numerical precision:** The renormalization division involves BF16 arithmetic with 7 mantissa bits and machine epsilon $\varepsilon_{\text{mach}} = 2^{-7} \approx 0.0078$. For $Z_b^{\text{drop}}$ near zero (all high-weight experts dropped), the division is numerically unstable. A practical guard:

$$Z_b^{\text{norm}} \;=\; \max\!\left(Z_b^{\text{drop}},\; \varepsilon_{\text{mach}}\right) \;=\; \max\!\left(Z_b^{\text{drop}},\; 2^{-7}\right)$$

This prevents NaN or Inf outputs when a token's entire set of available experts has near-zero routing weight.

---

## 6. The Role of $CF = 1.25$ in Limiting Drop Rate

The capacity factor $CF = 1.25$ is chosen to make token drops rare under near-uniform routing. The intuition is that the Poisson tail beyond $C = 2 \times f_{\text{avg,uniform}}$ is small for typical batch sizes.

Under uniform routing, the expected token count to expert $e$ per step is:

$$\lambda_e \;=\; \mathbb{E}[n_e] \;=\; f_e \;=\; \frac{k \cdot B}{E} \;=\; \frac{B}{32}$$

At $B = 32$: $\lambda_e = 1.0$ token per expert per step. The Poisson approximation gives:

$$n_e \;\approx\; \text{Poisson}(\lambda_e = 1.0)$$

The capacity is $C = 2$, so overflow occurs when $n_e > 2$, i.e., $n_e \geq 3$:

$$P(\text{overflow at expert } e \mid B=32, \text{uniform}) \;=\; P(n_e > 2) \;=\; 1 - P(n_e \leq 2)$$

$$P(n_e \leq 2) \;=\; e^{-1}\!\left(1 + 1 + \frac{1}{2}\right) \;=\; e^{-1} \times 2.5 \;\approx\; 0.3679 \times 2.5 \;=\; 0.9197$$

$$P(\text{overflow at expert } e \mid B=32, \text{uniform}) \;\approx\; 1 - 0.920 \;=\; 0.080$$

**Interpretation:** Under perfectly uniform routing and $B = 32$, each expert has an 8.0% chance of overflow in any given step. This is the irreducible Poisson tail for $CF = 1.25$. The expected number of overflowing experts per step is $0.080 \times 256 \approx 20.5$ experts. This is the baseline drop rate that $CF = 1.25$ tolerates.

---

## 7. Worked Example: $B = 32$, Uniform Routing, Poisson Approximation

**Setup:**
- $B = 32$ tokens per step
- $E = 256$ experts, $k = 8$, $CF = 1.25$
- $C = \lceil 32/25.6 \rceil = \lceil 1.25 \rceil = 2$
- $\lambda = k \cdot B / E = 8 \times 32 / 256 = 1.0$ (mean tokens per expert, uniform routing)
- $n_e \sim \text{Poisson}(1.0)$

**Step 1: Compute $P(n_e = j)$ for $j = 0, 1, 2, 3, 4$.**

$$P(n_e = j) \;=\; \frac{e^{-1} \cdot 1^j}{j!} \;=\; \frac{e^{-1}}{j!}$$

| $j$ (tokens to expert) | $P(n_e = j)$ | Cumulative $P(n_e \leq j)$ |
|---|---|---|
| 0 | $e^{-1}/1 \approx 0.3679$ | 0.3679 |
| 1 | $e^{-1}/1 \approx 0.3679$ | 0.7358 |
| 2 | $e^{-1}/2 \approx 0.1839$ | 0.9197 |
| 3 | $e^{-1}/6 \approx 0.0613$ | 0.9810 |
| 4 | $e^{-1}/24 \approx 0.0153$ | 0.9963 |
| $\geq 5$ | $\approx 0.0037$ | 1.0000 |

**Step 2: Overflow probability per expert per step.**

$$P(\text{overflow}) \;=\; P(n_e > C) \;=\; P(n_e > 2) \;=\; 1 - 0.9197 \;=\; 0.0803$$

**Step 3: Expected dropped tokens per expert per step.**

$$\mathbb{E}[d_e] \;=\; \sum_{j=3}^{\infty} (j - 2) \cdot P(n_e = j)$$

$$= 1 \times 0.0613 + 2 \times 0.0153 + 3 \times 0.0031 + \ldots$$

$$\approx 0.0613 + 0.0306 + 0.0092 + 0.0023 + \ldots \;\approx\; 0.106 \text{ tokens/expert/step}$$

**Step 4: Expected total tokens dropped per step across all $E = 256$ experts.**

$$\mathbb{E}\!\left[\sum_{e=0}^{255} d_e\right] \;=\; 256 \times 0.106 \;\approx\; 27.1 \text{ tokens dropped per step}$$

Total tokens dispatched per step: $k \times B = 8 \times 32 = 256$ token-expert assignments. Fraction dropped:

$$\text{overall drop fraction} \;=\; \frac{27.1}{256} \;\approx\; 10.6\%$$

**Step 5: Comparison with Zipf routing.**

Under Zipf routing with $\tilde{f}_{(1)} \approx 0.17$ (per-token probability $p_{(1)} \approx 0.17/k$... actually using the convention from `load_imbalance_detection.md`, $f_{(1)} = 0.17 \times B = 0.17 \times 32 = 5.44$ expected tokens per step), against $C = 2$:

$$\mathbb{E}[d_{(1)}] \;\approx\; f_{(1)} - C \;=\; 5.44 - 2 \;=\; 3.44 \text{ tokens dropped per step (top expert alone)}$$

The drop rate from the top expert alone already exceeds the total drop rate under uniform routing. This illustrates why Zipf-like distributions require mitigation strategies (Chapter 7, `dynamic_routing_strategies.md`).

---

## 8. Summary of Overflow Handling Policies

| Property | Hard drop | Reassign |
|---|---|---|
| Token count after overflow | $|\mathcal{S}(b)| < k$ | $|\mathcal{S}(b)| = k$ (if substitute available) |
| Substitute expert quality | N/A (zero contribution) | $(k+1)$-th best expert (suboptimal) |
| Weight renormalization | Required; divides by $Z_b^{\text{drop}} < Z_b$ | Standard $k$-expert normalization |
| Dispatch metadata overhead | None (standard top-$k$) | Requires $(k+d)$ indices per token |
| Implementation complexity | Low | High |
| Recommended use case | When overflow is rare (near-uniform routing) | When overflow is moderate and fallback capacity exists |

For Qwen3.5-35B on T3K with $CF = 1.25$ and near-uniform routing, **hard drop** is the recommended policy. With Zipf-like routing, expert replication should be applied first (see `dynamic_routing_strategies.md`) to reduce the overflow rate before choosing a drop policy.

---

## References

- [Ch1Routing] Chapter 1, `ch01_moe_fundamentals/routing_problem.md` — expert capacity, token dropping, dispatch/combine pattern.
- [Ch1Config] Chapter 1, `ch01_moe_fundamentals/qwen35b_config.md` — Qwen3.5-35B architectural constants.
- [Ch2Dispatch] Chapter 2, `ch02_all_to_all_primitives/all_to_all_dispatch.md` — dispatch buffer layout, capacity enforcement.
- [Ch2Combine] Chapter 2, `ch02_all_to_all_primitives/all_to_all_combine.md` — combine buffer layout, routing weight accumulation, renormalization.
- [Ch7Detection] `load_imbalance_detection.md` — definitions of $f_e$, $C$, overflow count, utilization metric.
- [Ch7Dynamic] `dynamic_routing_strategies.md` — mitigation strategies to reduce overflow before it occurs.
- [Fedus2022] Fedus, W., Zoph, B., Shazeer, N., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR, 2022.
- [Zoph2022] Zoph, B. et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models", arXiv:2202.08906, 2022.
- [Lepikhin2021] Lepikhin, D. et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding", ICLR, 2021.
- [Qwen3] Qwen Team, Alibaba Group, "Qwen3 Technical Report", 2025.

---

**Next:** [dynamic_routing_strategies.md](./dynamic_routing_strategies.md)
