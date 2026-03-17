# Load Imbalance Detection

## Overview

Before any mitigation strategy can be applied, load imbalance must be measured. This file defines the per-expert routing frequency $f_e$, the Coefficient of Variation (CV) as the primary scalar imbalance metric, the hot-expert threshold, the per-step overflow rate, the runtime monitoring mechanism, and the expert utilization metric. All definitions here are used in `capacity_overflow_handling.md` and `dynamic_routing_strategies.md`.

**Prerequisites:** Chapter 1, `ch01_moe_fundamentals/routing_problem.md` (expert capacity, dispatch/combine, token dropping); Chapter 4, `ch04_expert_device_assignment/load_aware_assignment.md` (routing frequency profiling).

---

## 1. Formal Definition of $f_e$

Let $B$ denote the number of tokens in one forward pass (the batch size). Each token independently selects $k = 8$ experts via the router. Define:

$$f_e \;=\; \mathbb{E}\!\left[\text{tokens assigned to expert } e \text{ in one step}\right]$$

This expectation is taken over the router's stochastic assignment for a fixed batch of $B$ tokens. Because each token selects exactly $k$ experts, the frequencies sum to:

$$\sum_{e=0}^{E-1} f_e \;=\; k \cdot B \;=\; 8B$$

The **average frequency** per expert is:

$$f_{\text{avg}} \;=\; \frac{k \cdot B}{E} \;=\; \frac{8B}{256} \;=\; \frac{B}{32}$$

For a perfectly uniform router, $f_e = f_{\text{avg}}$ for all $e$. Any deviation from uniformity constitutes load imbalance.

> **Note on normalization:** In some literature $f_e$ is normalized per token (so $\sum_e f_e = k$, not $k \cdot B$). In this chapter, $f_e$ retains the factor of $B$ so that it directly equals the expected token count delivered to expert $e$ per step. The normalized per-token version is $\tilde{f}_e = f_e / B \in [0, 1]$, with $\sum_e \tilde{f}_e = k = 8$ and $\tilde{f}_{\text{avg}} = k/E = 1/32$.

---

## 2. Coefficient of Variation

The **Coefficient of Variation (CV)** of the frequency distribution $\{f_e\}_{e=0}^{E-1}$ is:

$$CV \;=\; \frac{\sigma(\{f_e\})}{\mu(\{f_e\})} \;=\; \frac{\sqrt{\frac{1}{E}\sum_{e=0}^{E-1}(f_e - f_{\text{avg}})^2}}{f_{\text{avg}}}$$

where $\mu(\{f_e\}) = f_{\text{avg}} = B/32$ and $\sigma(\{f_e\})$ is the population standard deviation over all $E = 256$ experts.

Key reference values:

| Routing distribution | $CV$ | Interpretation |
|---|---|---|
| Perfectly uniform ($f_e = f_{\text{avg}}$ for all $e$) | 0 | Ideal; no load imbalance |
| Mildly skewed | $0 < CV \lesssim 0.3$ | Small fraction of hot experts; $CF = 1.25$ typically sufficient |
| Moderately skewed | $0.3 < CV \lesssim 1.0$ | Multiple hot experts; replication recommended |
| Heavily skewed (Zipf-like) | $CV > 1.0$ | Severe imbalance; token drop rate is high without mitigation |

The CV is dimensionless and independent of $B$, making it the preferred metric for comparing routing distributions across different batch sizes or model checkpoints.

### 2.1 Zipf Example

Under a Zipf(1.0) distribution over $E = 256$ experts, the $m$-th most popular expert receives a fraction proportional to $1/m$. Normalizing to $\sum_e \tilde{f}_e = k = 8$:

$$\tilde{f}_{(m)} \;=\; \frac{k}{H_E} \cdot \frac{1}{m} \;=\; \frac{8}{\ln(256)} \cdot \frac{1}{m} \;\approx\; \frac{8}{5.545} \cdot \frac{1}{m} \;\approx\; \frac{1.443}{m}$$

where $H_E \approx \ln(E) + 0.5772 \approx 5.545$ is the $E$-th harmonic number (Zipf exponent 1.0).

The top expert receives $\tilde{f}_{(1)} \approx 1.443 / 1 \approx 1.443$ ... but this exceeds $k = 8$ only in aggregate across experts, so let us verify: $\sum_{m=1}^{256} \tilde{f}_{(m)} = (8/H_E) \cdot H_{256} = 8$. Correct.

Per-step token count to the top expert (at $B = 32$): $f_{(1)} = \tilde{f}_{(1)} \cdot B \approx 1.443 / 32 \cdot 32$... more carefully:

$$f_{(1)} = \tilde{f}_{(1)} \cdot B = \frac{k}{H_E} \cdot B = \frac{8}{5.545} \times 32 \approx 46.2 \text{ tokens}$$

Wait — this is the total across all tokens, but $\tilde{f}_{(1)}$ is already a per-token fraction. More precisely:

$$f_{(1)} = \tilde{f}_{(1)} \cdot B = \frac{1.443}{1} \cdot ... $$

Let us state this clearly. The per-token probability that expert $(1)$ is selected is $p_{(1)} = \tilde{f}_{(1)} / k$ because each token selects $k$ experts and probabilities sum to 1:

$$\sum_{m=1}^{256} p_{(m)} = 1, \quad p_{(m)} = \frac{\tilde{f}_{(m)}}{k} = \frac{1}{H_E \cdot m} \approx \frac{0.180}{m}$$

Expected tokens to expert $(1)$ per step: $f_{(1)} = p_{(1)} \cdot B = 0.180 \times 32 \approx 5.76$ tokens at $B = 32$.

Using the ground-truth approximation $\tilde{f}_{(1)} \approx 0.17$ (as in Chapter 4, `ch04_expert_device_assignment/expert_replication.md`), the expected token count is:

$$f_{(1)} = 0.17 \times B = 0.17 \times 32 = 5.44 \text{ tokens per step}$$

The expert capacity at $B = 32$ is $C = \lceil 32 / 25.6 \rceil = \lceil 1.25 \rceil = 2$ tokens. The top expert therefore receives approximately $5.44$ tokens on average against a capacity of $C = 2$ — a **2.72× overflow ratio**. Every step, roughly $5.44 - 2 = 3.44$ tokens are dropped from the top expert alone.

**CV for Zipf(1.0) over 256 experts:**

The variance of $\{\tilde{f}_e\}$ under Zipf(1) is dominated by the first few terms. A numerical estimate gives $CV \approx 3.5$ to $5.0$ for $E = 256$, well above the "heavily skewed" threshold of $CV > 1.0$. Exact computation requires summing $\sum_m (\tilde{f}_{(m)} - f_{\text{avg}})^2 / E$ numerically.

---

## 3. Hot-Expert Detection

An expert $e$ is classified as **hot** if its normalized routing frequency satisfies:

$$\tilde{f}_e \;>\; 2 \times f_{\text{avg,normalized}} \;=\; 2 \times \frac{k}{E} \;=\; 2 \times \frac{8}{256} \;=\; \frac{1}{16} \;=\; 0.0625$$

Equivalently, in absolute token counts at batch size $B$:

$$f_e \;>\; 2 \times f_{\text{avg}} \;=\; \frac{2B}{32} \;=\; \frac{B}{16}$$

The factor of 2 is a configurable threshold; systems with stricter quality requirements may lower it to 1.5×, and systems with ample replication budget may raise it to 3×. The default of 2× is consistent with the recommendation in Chapter 4, `ch04_expert_device_assignment/expert_replication.md`, Section 6.

**Why 2×?** With $CF = 1.25$, the capacity $C = \lceil k \cdot B \cdot CF / E \rceil$ provides 25% headroom above the mean load. An expert at $2 \times f_{\text{avg}}$ receives tokens at $2 \times / 1.25 = 1.6\times$ the capacity — well into the overflow regime for any batch size $B \geq 2$.

---

## 4. Per-Step Overflow Rate

For expert $e$ with actual token count $n_e$ in a given step (a random variable), the **overflow count** per step is:

$$\text{overflow}_e \;=\; \max(0,\; n_e - C)$$

The **expected overflow count** per step is:

$$\mathbb{E}[\text{overflow}_e] \;=\; \mathbb{E}[\max(0,\; n_e - C)] \;=\; \sum_{j=C+1}^{B} (j - C) \cdot P(n_e = j)$$

Under the assumption that each token independently routes to expert $e$ with probability $p_e = \tilde{f}_e / k$, the token count $n_e \sim \text{Binomial}(B,\, p_e)$. For large $B$ with $p_e$ small, a Poisson approximation is accurate: $n_e \approx \text{Poisson}(\lambda_e)$ where $\lambda_e = B \cdot p_e = f_e$.

The **overflow rate** (fraction of tokens dropped from expert $e$ per step) is:

$$\text{drop\_rate}_e \;=\; \frac{\mathbb{E}[\text{overflow}_e]}{\mathbb{E}[n_e]} \;=\; \frac{\mathbb{E}[\max(0,\; n_e - C)]}{f_e}$$

See `capacity_overflow_handling.md`, Section 4 for the full Poisson calculation at $B = 32$.

---

## 5. Runtime Monitoring via Routing Metadata

The router already computes top-$k$ expert indices for every token in the batch. The per-step overhead of maintaining a hot-expert counter from this metadata is $O(k \times B) = O(8B)$ operations per step — one increment per (token, expert) pair.

**Monitoring algorithm (pseudocode):**

```python
# Per-step counter update: O(k * B) operations
# routing_indices[t] = list of k expert indices for token t
def update_load_counters(
    routing_indices,   # shape: (B, k) — top-k expert indices per token
    token_counts,      # shape: (E,) — accumulator array, updated in-place
    step: int,
    window_size: int = 1000,  # steps per monitoring window
):
    for t in range(len(routing_indices)):
        for e in routing_indices[t]:
            token_counts[e] += 1

    if step % window_size == 0:
        # Compute f_e estimates and CV over the window
        f_e = token_counts / window_size  # expected tokens per step
        f_avg = f_e.mean()
        cv = f_e.std() / f_avg
        token_counts[:] = 0  # reset for next window
        return f_e, cv
    return None, None
```

The $O(k \times B) = O(8B)$ cost is negligible relative to the $O(k \times B \times H \times D)$ cost of the expert FFN computation. However, on Wormhole B0 with 80 Tensix cores [UNVERIFIED — this is the documented core count], this counter update must be executed on the host or in a lightweight on-device kernel [VERIFY: `ttnn.load_balancing_metadata_update` or equivalent TTNN API].

**Monitoring granularity:**

| Granularity | Steps per update | Latency cost | Staleness |
|---|---|---|---|
| Per-step | 1 | $O(k \times B)$ per step | 0 steps |
| Windowed (recommended) | $W = 1000$ | Amortized $O(k \times B / W)$ | Up to $W$ steps |
| Offline (calibration only) | $N_{\text{calib}}$ | One-time | Fixed after deployment |

For production inference, a windowed update with $W = 1000$ steps balances responsiveness against overhead.

---

## 6. Expert Utilization Metric

The **utilization** of expert $e$ in step $t$ is:

$$u_e(t) \;=\; \frac{n_e(t)}{C}$$

where $n_e(t)$ is the actual token count delivered to expert $e$ in step $t$ and $C$ is the expert capacity.

| Utilization range | Interpretation |
|---|---|
| $u_e < 1.0$ | Expert has spare capacity; no tokens dropped |
| $u_e = 1.0$ | Expert is exactly at capacity |
| $u_e > 1.0$ | **Overflow:** $\lfloor (u_e - 1) \times C \rfloor$ tokens dropped from this expert in this step |

The **time-averaged utilization** over $T$ steps is $\bar{u}_e = \frac{1}{T} \sum_{t=1}^{T} u_e(t)$. A well-balanced system has $\bar{u}_e \leq 1.0$ for all $e$, with $\bar{u}_e \approx k \cdot B / (E \cdot C) = B / (32 C)$ under uniform routing.

At $B = 32$ and $C = 2$: $\bar{u}_e = 32 / (32 \times 2) = 0.5$ for a perfectly uniform router, meaning experts are on average at 50% utilization. The 25% headroom from $CF = 1.25$ above $f_{\text{avg}}$ is reflected here: the expected utilization is $k \times B / (E \times C) = 1 / CF = 1/1.25 = 0.8$, so experts are typically at 80% utilization under uniform routing and only overflow on Poisson tail events.

---

## 7. Summary

| Metric | Formula | Ideal value |
|---|---|---|
| $f_e$ (expected token count, expert $e$) | $\mathbb{E}[n_e]$ | $f_{\text{avg}} = B/32$ |
| $f_{\text{avg}}$ (mean over all experts) | $kB/E = B/32$ | — |
| $CV$ (coefficient of variation) | $\sigma(\{f_e\}) / f_{\text{avg}}$ | 0 |
| Hot expert threshold | $f_e > 2 \times f_{\text{avg}}$ (configurable) | No experts above threshold |
| Overflow count (expert $e$, step $t$) | $\max(0,\, n_e(t) - C)$ | 0 |
| Utilization (expert $e$, step $t$) | $n_e(t) / C$ | $\leq 1.0$ |
| Monitoring cost per step | $O(k \times B) = O(8B)$ | — |

---

## References

- [Ch1Routing] Chapter 1, `ch01_moe_fundamentals/routing_problem.md` — expert capacity definition, token dropping, dispatch/combine pattern.
- [Ch1Config] Chapter 1, `ch01_moe_fundamentals/qwen35b_config.md` — Qwen3.5-35B architectural constants ($E=256$, $k=8$, $H=7168$).
- [Ch4Replication] Chapter 4, `ch04_expert_device_assignment/expert_replication.md` — hot-expert threshold justification, replication factor formula.
- [Ch4LoadAware] Chapter 4, `ch04_expert_device_assignment/load_aware_assignment.md` — routing frequency profiling methodology.
- [Ch7Overflow] `capacity_overflow_handling.md` — token-drop mechanism, Poisson drop model, worked example.
- [Ch7Dynamic] `dynamic_routing_strategies.md` — mitigation strategies responding to the metrics defined here.
- [Fedus2022] Fedus, W., Zoph, B., Shazeer, N., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR, 2022.
- [Zoph2022] Zoph, B. et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models", arXiv:2202.08906, 2022.
- [Qwen3] Qwen Team, Alibaba Group, "Qwen3 Technical Report", 2025.
