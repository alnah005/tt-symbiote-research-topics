# Dynamic Routing Strategies

## Overview

This file presents the inference-time and training-time strategies for mitigating load imbalance in MoE routing. The four strategies are: (1) load-aware routing adjustment via logit modification, (2) temperature scaling of the routing softmax/sigmoid, (3) expert replication using the $r_e$ formula from Chapter 4, and (4) the auxiliary load-balancing loss applied at training time. The file closes with an analysis of online versus offline load balancing and a concrete recommendation for Qwen3.5-35B on T3K.

**Prerequisites:** `load_imbalance_detection.md` ($f_e$, $CV$, hot-expert threshold); `capacity_overflow_handling.md` (overflow model, Poisson drop rate); Chapter 4, `ch04_expert_device_assignment/expert_replication.md` (replication factor formula, dispatch interaction); Chapter 5, `ch05_routing_weight_optimization/` (routing weight computation, softmax/sigmoid temperature).

---

## 1. Load-Aware Routing Adjustment

**Concept:** Modify the router's output logits at inference time to reduce the probability of routing tokens to hot experts, redistributing load toward cooler experts without changing the router's weights.

At each step, the standard router computes a logit vector $\mathbf{z}_b \in \mathbb{R}^E$ for token $b$. The top-$k$ experts are selected by:

$$\mathcal{S}(b) \;=\; \operatorname{top-}k\!\left(\sigma(\mathbf{z}_b)\right) \quad \text{or} \quad \mathcal{S}(b) \;=\; \operatorname{top-}k\!\left(\operatorname{softmax}(\mathbf{z}_b)\right)$$

Load-aware adjustment applies an **expert bias** $\boldsymbol{\delta} \in \mathbb{R}^E$ to the logits before selection:

$$\mathbf{z}_b^{\text{adj}} \;=\; \mathbf{z}_b - \boldsymbol{\delta}, \quad \delta_e \;=\; \beta \cdot \max\!\left(0,\; \frac{f_e - f_{\text{avg}}}{f_{\text{avg}}}\right)$$

where $\beta > 0$ is a penalty coefficient (a configurable hyperparameter). For a hot expert with $f_e = 2 f_{\text{avg}}$, the bias is $\delta_e = \beta \cdot 1.0 = \beta$, reducing its effective logit by $\beta$ and making it less likely to be selected in future steps.

**Properties:**
- Operates entirely at the logit level; no change to model weights.
- Bias update cost is $O(E) = O(256)$ operations per step to compute $\boldsymbol{\delta}$ from the running $f_e$ estimates.
- The adjustment is approximate: it reduces the probability of routing to expert $e$, but the router's learned preferences still dominate for tokens with a strong semantic affinity for the hot expert.
- Requires the monitoring infrastructure from `load_imbalance_detection.md`, Section 5 to maintain current $f_e$ estimates.
- Biasing logits at inference does not align with the router's training distribution, potentially introducing a small quality degradation. The degree of degradation depends on $\beta$.

**Bias coefficient recommendation:** $\beta \in [0.1, 0.5]$ logit units. Values above 0.5 risk routing tokens to semantically inappropriate experts and degrading output quality [UNVERIFIED — optimal $\beta$ depends on the specific model and calibration data].

---

## 2. Temperature Scaling

**Concept:** Increase the temperature $\tau$ of the router's softmax or sigmoid to flatten the routing distribution, reducing the probability gap between the top expert and the remaining experts.

For a softmax router:

$$p_e(\mathbf{z}_b,\, \tau) \;=\; \frac{\exp(z_{b,e} / \tau)}{\sum_{e'=0}^{E-1} \exp(z_{b,e'} / \tau)}$$

- $\tau = 1.0$: standard softmax (no change).
- $\tau > 1.0$: flatter distribution; hot experts receive less probability mass; cold experts receive more.
- $\tau \to \infty$: uniform distribution over all experts ($CV \to 0$).

For a sigmoid router (as used in some top-$k$ gating schemes):

$$p_e(\mathbf{z}_b,\, \tau) \;=\; \sigma(z_{b,e} / \tau) \;=\; \frac{1}{1 + \exp(-z_{b,e}/\tau)}$$

Temperature scaling is applied after the router's linear projection and before the top-$k$ selection.

**Effect on $CV$:** Increasing $\tau$ monotonically decreases $CV$ of the routing distribution. At $\tau = 1$, $CV$ reflects the router's trained preferences. At $\tau \gg 1$, $CV \to 0$ but the router's expert selections become semantically meaningless (random-like).

**Trade-off:** Temperature scaling is the most computationally cheap adjustment ($O(E)$ per token for the softmax renormalization, already computed by the router), but it globally degrades routing quality by reducing the router's confidence. It is best used as a secondary measure after load-aware bias has been applied.

**Interaction with BF16 numerics:** At high temperature, the exponent arguments $z_{b,e} / \tau$ become small in magnitude. BF16 has 7 mantissa bits ($\varepsilon_{\text{mach}} = 2^{-7} \approx 0.0078$); if $z_{b,e} / \tau < \varepsilon_{\text{mach}}$ for most experts, the softmax distribution effectively becomes uniform due to numerical rounding. This can be beneficial (further flattening) or harmful (loss of router discrimination) depending on context.

---

## 3. Expert Replication

**Concept:** Replicate hot experts on multiple devices so that the total capacity available for a hot expert is $r_e \times C$ instead of $C$. The dispatch mechanism distributes tokens across replicas.

This strategy is defined in detail in Chapter 4, `ch04_expert_device_assignment/expert_replication.md`. The key formula (reproduced here for completeness) is:

$$r_e \;=\; \max\!\left(1,\; \left\lceil \frac{f_e \cdot E}{k} \right\rceil\right) \;=\; \max\!\left(1,\; \lceil 32 \cdot f_e \rceil\right)$$

where $f_e$ here is the normalized per-token frequency $\tilde{f}_e$ (expected tokens per step divided by $B$, i.e., $\tilde{f}_e = f_e / B$ in the notation of `load_imbalance_detection.md`).

**Examples for Qwen3.5-35B ($E = 256$, $k = 8$):**

| Normalized frequency $\tilde{f}_e$ | $32 \cdot \tilde{f}_e$ | $r_e$ | Replication needed |
|---|---|---|---|
| $\tilde{f}_{\text{avg}} = 1/32 \approx 0.0313$ | 1.0 | 1 | No |
| $2 \tilde{f}_{\text{avg}} \approx 0.0625$ | 2.0 | 2 | Yes — 2 devices |
| $0.17$ (Zipf top expert) | 5.44 | 6 | Yes — 6 devices |
| $4 \tilde{f}_{\text{avg}} = 0.125$ | 4.0 | 4 | Yes — 4 devices |

For the Zipf top expert with $\tilde{f}_{(1)} \approx 0.17$: without replication, at $B = 32$ the expert receives approximately $0.17 \times 32 = 5.44$ tokens per step against $C = 2$ capacity — a 2.72× overflow ratio. With $r_{(1)} = 6$ replicas, each replica receives approximately $5.44 / 6 \approx 0.91$ tokens per step, safely below $C = 2$.

**Dispatch overhead:** The replica routing table adds $O(k \times B)$ lookup operations per step in the dispatch kernel. This is the same asymptotic cost as the per-step monitoring counter update in `load_imbalance_detection.md`, Section 5 — small relative to the FFN compute cost.

**Memory cost:** Replicating expert $e$ on $r_e$ devices consumes $r_e \times W_{\text{expert}}$ total bytes across the system. On Wormhole B0 with 1.5 MB L1/core [UNVERIFIED — this is the documented L1 size per Tensix core] and 80 Tensix cores [UNVERIFIED], expert weights that must fit in L1 for fast access are constrained to $80 \times 1.5 \text{ MB} = 120 \text{ MB}$ per device. For $M$ fully-replicated experts, the per-device L1 overhead is $M \times W_{\text{expert}}$ bytes; verify against the DRAM budget for spill-over cases.

**Cross-reference:** See Chapter 4, `ch04_expert_device_assignment/expert_replication.md`, Section 5 for the dispatch metadata format with replica routing tables and the recommended round-robin replica selection policy.

---

## 4. Auxiliary Load-Balancing Loss (Training Time Only)

**Concept:** During training, add an auxiliary loss term that penalizes non-uniform expert utilization. This encourages the router to learn a more balanced routing distribution, reducing $CV$ of $\{f_e\}$ before the model is deployed.

The standard auxiliary loss (introduced in Switch Transformers [Fedus2022] and used in GShard [Lepikhin2021]) is:

$$\mathcal{L}_{\text{aux}} \;=\; \alpha \cdot E \cdot \sum_{e=0}^{E-1} \bar{f}_e \cdot \bar{P}_e$$

where:
- $\alpha$ is the loss weight coefficient; typical value $\alpha = 0.01$,
- $\bar{f}_e$ is the fraction of tokens (over the batch) routed to expert $e$: $\bar{f}_e = \frac{1}{B} \sum_{b=1}^{B} \mathbf{1}[e \in \mathcal{S}(b)]$,
- $\bar{P}_e$ is the mean routing probability for expert $e$ over the batch: $\bar{P}_e = \frac{1}{B} \sum_{b=1}^{B} p_{b,e}$.

The product $\bar{f}_e \cdot \bar{P}_e$ is minimized when $\bar{f}_e$ and $\bar{P}_e$ are uniform across experts. The prefactor $E$ ensures $\mathcal{L}_{\text{aux}}$ has the same scale regardless of the number of experts.

**CRITICAL: This loss is NOT computed at inference time.** It is a training-time regularizer only. At inference, $\mathcal{L}_{\text{aux}} = 0$ and no gradients are computed. Its sole purpose is to shape the router's learned weights so that the trained model's routing distribution is near-uniform. If a deployed model exhibits high $CV$ at inference, it means either (a) the auxiliary loss weight $\alpha$ was too small during training, or (b) the inference data distribution differs from the training distribution.

**Effect of $\alpha$:**

| $\alpha$ value | Effect on routing |
|---|---|
| $0$ | No balancing; router may collapse to routing all tokens to a few experts |
| $0.001$ | Weak regularization; some imbalance persists |
| $0.01$ | Standard value; near-uniform routing in most MoE models [Fedus2022] |
| $0.1$ | Strong regularization; routing may become too uniform, reducing expert specialization and model quality |

For Qwen3.5-35B, the training configuration uses $\alpha = 0.01$ [UNVERIFIED — value not confirmed from ground truth; $\alpha = 0.01$ is the standard literature value]. The effectiveness of the auxiliary loss in achieving low-$CV$ routing at inference depends on the training data distribution matching the deployment distribution.

---

## 5. Online vs. Offline Load Balancing

The strategies above can be applied in two modes: **offline** (based on pre-collected statistics) or **online** (adaptive at inference time).

### 5.1 Offline Load Balancing

**Process:**
1. Run a calibration workload of $N_{\text{calib}}$ steps (e.g., 5,000–10,000 token batches) through the deployed model.
2. Collect per-expert token counts; compute $f_e$ for all $e \in \{0, \ldots, E-1\}$.
3. Identify hot experts: $\{e : f_e > 2 f_{\text{avg}}\}$.
4. Compute replication factors $r_e = \max(1, \lceil 32 \cdot \tilde{f}_e \rceil)$ for hot experts.
5. Replicate hot experts; update dispatch metadata (replica routing table).
6. Optionally, pre-compute a fixed logit bias $\boldsymbol{\delta}$ from the calibration statistics.

**Properties:**
- No runtime overhead beyond a fixed dispatch table lookup.
- Assumes routing statistics are stationary (the calibration distribution represents the deployment distribution).
- Re-calibration is needed if the input distribution changes significantly (e.g., a different task or language domain).
- Replication factors $r_e$ are set once at deployment and do not change between requests.

**Monitoring cost:** The one-time calibration over $N_{\text{calib}}$ batches costs $O(k \times B \times N_{\text{calib}})$ operations total. For $k=8$, $B=32$, $N_{\text{calib}} = 10^4$: $8 \times 32 \times 10^4 = 2.56 \times 10^6$ counter increments — negligible.

### 5.2 Online Load Balancing

**Process:**
1. Maintain a running windowed estimate of $f_e$ with window size $W$ (e.g., $W = 1000$ steps; see `load_imbalance_detection.md`, Section 5).
2. After each window, recompute hot-expert flags and optionally update the logit bias $\boldsymbol{\delta}$.
3. If a previously cold expert becomes hot (or a previously hot expert cools), update the replication set and dispatch metadata.

**Properties:**
- Adapts to distribution shifts at inference time; useful for long-running deployments with varying input characteristics.
- Per-step overhead: $O(k \times B) = O(8B)$ counter increments. At $B = 32$: $256$ increments per step. On Wormhole B0 hardware, this requires a lightweight kernel or host-side execution [VERIFY: TTNN support for online load monitoring].
- Updating the replication set mid-deployment requires a coordination step across all $N = 8$ devices: the replica routing table must be broadcast before the next dispatch. This coordination adds inter-device communication overhead — approximately $O(E) = O(256)$ words broadcast over the T3K Ethernet links at 12.5 GB/s per link [UNVERIFIED — actual coordination latency depends on TTNN collective implementation; $O(E)$ words is 256 × 2 bytes = 512 bytes BF16, negligible bandwidth but non-trivial synchronization latency].
- Average hop count on the T3K 1×8 linear mesh is 3.0 hops per message; broadcast latency scales with the number of hops traversed.
- The main risk of online balancing is **oscillation**: if the bias is updated too aggressively, the router alternately over- and under-routes to experts, increasing $CV$ rather than decreasing it. A damped update ($\delta_e \leftarrow (1-\gamma) \delta_e + \gamma \cdot \Delta_e$ with momentum $\gamma \approx 0.1$) reduces oscillation.

**Comparison summary:**

| Property | Offline | Online |
|---|---|---|
| Adaptation to distribution shift | No (re-calibrate manually) | Yes (automatic within window $W$) |
| Per-step overhead | $O(1)$ (fixed table lookup) | $O(k \times B) = O(8B)$ |
| Replication changes mid-deployment | No | Yes (requires broadcast) |
| Implementation complexity | Low | High (kernel support required [VERIFY]) |
| Risk of oscillation | None | Moderate (requires damped update) |
| Recommended for Qwen3.5-35B T3K | Yes (primary recommendation) | Optional (secondary, if distribution shift expected) |

---

## 6. Recommendation for Qwen3.5-35B on T3K

Given the model parameters ($E=256$, $k=8$, $N=8$, $CF=1.25$, $E_d=32$) and the T3K hardware profile (12.5 GB/s per Ethernet link, 1×8 linear mesh, average hop count 3.0), the recommended load-balancing configuration is:

1. **Use $CF = 1.25$ (fixed).** This provides a 25% capacity buffer above the mean, tolerating Poisson tail events at a ~8% per-expert overflow probability under uniform routing (see `capacity_overflow_handling.md`, Section 6). Raising $CF$ further (e.g., $CF = 1.5$) reduces overflow but increases buffer memory allocation [UNVERIFIED — memory impact of larger dispatch buffers depends on TTNN buffer implementation].

2. **Run offline calibration.** Profile routing frequencies $\tilde{f}_e$ over a 5,000–10,000 step calibration set representative of the deployment workload. Use this to compute $r_e = \max(1, \lceil 32 \cdot \tilde{f}_e \rceil)$ for all experts.

3. **Replicate hot experts offline.** Replicate all experts with $\tilde{f}_e > 2 \tilde{f}_{\text{avg}} = 1/16$ on all $N = 8$ devices (full replication; see Chapter 4, `ch04_expert_device_assignment/expert_replication.md`, Section 6 for the dispatch metadata format). Full replication simplifies dispatch (no partial replica set tracking) at the cost of higher per-device DRAM usage: $M \times W_{\text{expert}}$ additional bytes per device.

4. **Apply hard-drop policy for residual overflow.** After replication, any remaining overflow (from rare Poisson tail events or unexpected distribution shift) is handled by hard drop with weight renormalization (see `capacity_overflow_handling.md`, Section 4.1). The renormalization guard of $\max(Z_b^{\text{drop}}, 2^{-7})$ should be applied to prevent BF16 numerical instability.

5. **Monitor $CV$ and utilization continuously (windowed, $W = 1000$ steps).** The per-step monitoring cost of $O(k \times B) = O(8 \times 32) = O(256)$ counter increments is small. If $CV$ exceeds a configured threshold (e.g., $CV > 0.5$) or average utilization $\bar{u}_e > 0.9$ for more than a configurable fraction of experts, trigger a re-calibration step.

6. **Defer online logit-bias adjustment to future work.** The $O(k \times B)$ overhead per step is small, but implementing a stable damped update requires TTNN kernel support for per-expert counter maintenance and logit modification in the router kernel [VERIFY]. This is left as a future optimization; the offline replication strategy is sufficient for typical deployment scenarios.

**Expected outcome at $B = 32$ with this configuration:**
- Under near-uniform routing: $\bar{u}_e \approx 0.5$ (50% utilization at $C=2$, $\lambda=1.0$); drop rate $\approx 8\%$ per expert per step (Poisson tail only).
- Under Zipf top-expert $\tilde{f}_{(1)} \approx 0.17$ with $r_{(1)} = 6$: per-replica load $\approx 0.17 \times 32 / 6 \approx 0.91$ tokens/step vs. $C=2$ — no overflow.
- Total system overhead from monitoring and dispatch table lookup: $O(k \times B) = O(256)$ operations per step — negligible vs. FFN compute.

---

## 7. Summary of Strategies

| Strategy | When to apply | Inference overhead | Training required | Qwen3.5-35B recommendation |
|---|---|---|---|---|
| Load-aware logit bias | Online, when $CV > 0.5$ | $O(E) = O(256)$ per step | No | Optional; defer to future work |
| Temperature scaling | Online, secondary measure | $O(E)$ per step (already in router) | No | Optional; use only if logit bias insufficient |
| Expert replication | Offline, pre-deployment | $O(k \times B)$ dispatch lookup | No | Primary recommendation; replicate top-$M$ hot experts |
| Auxiliary loss ($\alpha = 0.01$) | Training time only | $0$ at inference | Yes | Not applicable at inference; assume already applied during Qwen3.5-35B training |

---

## References

- [Ch4Replication] Chapter 4, `ch04_expert_device_assignment/expert_replication.md` — replication factor formula, dispatch metadata, replica selection policies.
- [Ch4LoadAware] Chapter 4, `ch04_expert_device_assignment/load_aware_assignment.md` — calibration methodology for routing frequencies.
- [Ch5Weights] Chapter 5, `ch05_routing_weight_optimization/` — routing weight computation, softmax/sigmoid temperature, logit normalization.
- [Ch7Detection] `load_imbalance_detection.md` — $f_e$ definition, $CV$, hot-expert threshold, monitoring cost.
- [Ch7Overflow] `capacity_overflow_handling.md` — overflow handling policies, Poisson drop model, weight renormalization.
- [Ch8Config] Chapter 8, `ch08_qwen35b_t3k_strategy/recommended_configuration.md` — final Qwen3.5-35B T3K configuration synthesizing this chapter.
- [Fedus2022] Fedus, W., Zoph, B., Shazeer, N., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR, 2022.
- [Lepikhin2021] Lepikhin, D. et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding", ICLR, 2021.
- [Zoph2022] Zoph, B. et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models", arXiv:2202.08906, 2022.
- [Rajbhandari2022] Rajbhandari, S. et al., "DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale", ICML, 2022.
- [Qwen3] Qwen Team, Alibaba Group, "Qwen3 Technical Report", 2025.
