# Hierarchical Routing

## Purpose

This file describes two-level hierarchical routing — a scheme in which a coarse router first selects a device group and a fine router then selects an expert within that group. It derives the communication reduction from group confinement, analyzes load-balance challenges specific to inference, and states the conditions under which hierarchical routing beats the all-to-all baseline.

**Prerequisite:** Chapter 1, `ch01_moe_fundamentals/moe_architecture.md` (flat routing equations); Chapter 2, `ch02_all_to_all_primitives/collective_communication_background.md` (all-to-all volume model); Chapter 3, `index.md` (notation).

---

## Concept: Two-Level Hierarchical Routing

In flat top-$k$ routing (the Chapter 2 baseline), a single linear projection plus top-$k$ selection chooses $k$ experts from all $E = 256$ experts globally. Cross-device traffic is generated whenever a selected expert lives on a different device than the token's originating device.

Hierarchical routing replaces this single-level selection with a two-stage process:

**Stage 1 — Coarse routing:** A lightweight router selects $k_c$ device groups from $G$ total groups. This is a top-$k_c$ selection over a $G$-dimensional logit vector.

**Stage 2 — Fine routing:** Within each selected group, a second router selects $k_f$ experts from the $E_d = E / G$ experts in that group.

The total number of selected experts per token is $k_\text{total} = k_c \times k_f$. For this to match Qwen3.5-35B's $k = 8$, one example configuration is $k_c = 1$, $k_f = 8$ (one group, all 8 experts from that group) or $k_c = 2$, $k_f = 4$ (two groups, 4 experts each).

---

## Group Configuration for Qwen3.5-35B

For $E = 256$ experts and $N = 8$ devices, the natural group configuration aligns device groups with physical devices:

| Parameter | Value |
|---|---|
| Number of groups $G$ | 8 (one per device) |
| Experts per group $E_d = E / G$ | 32 |
| Coarse top-$k_c$ (groups selected per token) | 1 to $N = 8$ |
| Fine top-$k_f$ (experts per group) | $k / k_c$ (varies by $k_c$) |

With $G = N = 8$, the coarse router's logit vector has dimension 8 — one logit per device group. This is a very small classification problem (compared to the 256-dimensional logit vector in flat routing).

### Configuration Examples

| $k_c$ | $k_f$ | Groups visited | Total experts ($k_c k_f$) | Cross-device sends per token (on average) |
|---|---|---|---|---|
| 1 | 8 | 1 | 8 | $1 \times 7/8 = 0.875$ remote devices |
| 2 | 4 | 2 | 8 | $2 \times 7/8 = 1.75$ remote devices |
| 4 | 2 | 4 | 8 | $4 \times 7/8 = 3.5$ remote devices |
| 8 | 1 | 8 | 8 | $8 \times 7/8 = 7$ remote devices (reduces to flat routing) |

The expected cross-device sends formula is $k_c \times (N-1)/N$: of the $k_c$ groups selected, each has a $(N-1)/N$ probability of being on a remote device (since exactly 1 of $N$ devices is local). For $N = 8$: expected remote groups $= k_c \times 7/8$.

The $k_c = 1$, $k_f = 8$ case is the best case for communication: a token selects exactly one device group, and all 8 experts live on that one device. Cross-device communication volume is **zero** for tokens where the coarse router selects the originating device's own group — which happens with probability $1/N = 1/8$. The remaining $7/8$ of tokens require one cross-device send, giving an expected 0.875 remote device sends per token. This is the key communication reduction that motivates hierarchical routing.

---

## Communication Reduction Formula

### Flat Routing (Baseline)

From Chapter 2, the per-device all-to-all dispatch volume for flat routing is:

$$V_{a2a}^\text{flat} = \frac{N-1}{N} \times B \times k \times H \times 2 = \frac{7}{8} \times B \times 8 \times 7{,}168 \times 2 = 100{,}352 \times B \text{ bytes}$$

### Hierarchical Routing (General)

With $k_c$ groups selected per token, only $k_c$ of the $N = 8$ device groups receive tokens from any given originating device. The fraction of remote groups is $(k_c - 1) / (N - 1)$ for the non-self groups, and one of the $k_c$ groups may be the originating device's own group.

Expected remote group count per token (if the coarse router assigns uniformly): $k_c \times (N-1)/N$ remote groups, plus at most one local group.

Per-token volume sent to remote groups (BF16, $H = 7{,}168$, $k_f$ experts per group, one embedding copy per visited remote group):

$$V_{a2a}^\text{hier} = \frac{k_c(N-1)}{N} \times k_f \times B \times H \times 2 = \frac{k_c(N-1)}{N} \times \frac{k}{k_c} \times B \times H \times 2 = \frac{(N-1)k}{N} \times B \times H \times 2$$

Wait — this is the same as flat routing. The reduction comes only when $k_c < N$ is not random: the key reduction occurs when the coarse router **concentrates** selections into fewer groups than flat routing would reach. In the best case ($k_c = 1$), the volume is:

$$V_{a2a}^\text{hier}(k_c=1) = \frac{1 \times (N-1)}{N} \times k_f \times B \times H \times 2 = \frac{7}{8} \times 8 \times B \times 7{,}168 \times 2 = 100{,}352 \times B \text{ bytes}$$

This again matches the flat routing volume. The **actual** reduction arises because with $k_c = 1$, there is a $(1/N) = (1/8) = 12.5\%$ probability that the single selected group is the originating device's own group — in which case no cross-device send is needed at all.

**Best-case communication volume** (when coarse router selects the local group):

$$V_{a2a}^\text{hier, local} = 0$$

**Expected communication volume** (uniform group selection, $k_c = 1$):

$$\mathbb{E}[V^\text{hier}] = \frac{N-1}{N} \times k_f \times B \times H \times 2 + \frac{1}{N} \times 0$$

This simplifies to $\frac{N-1}{N} \times k_f \times B \times H \times 2$, which equals the flat all-to-all volume when $k_f = k$ — confirming that hierarchical routing with no group locality confinement has the same expected volume as flat all-to-all. For $k_c = 1$, $k_f = 8$, $N = 8$: expected remote volume equals $\frac{7}{8} \times 8 \times B \times H \times 2 = 100{,}352 \times B$ bytes, matching flat routing in the uniform case.

**Key insight:** Hierarchical routing reduces communication only if the coarse router learns to preferentially select the local group (or a small number of nearby groups). This is a load-balance challenge — if the coarse router sends all tokens to the local group, all devices compute only local experts and communication drops to zero, but load imbalance becomes catastrophic.

---

## Load-Balance Challenge: Coarse Router Tension

The coarse router faces a fundamental tension:

- **For communication efficiency:** concentrate token-group assignments to minimize cross-device sends.
- **For load balance:** distribute token-group assignments evenly so no device's 32 experts are overwhelmed.

These objectives conflict. A coarse router that perfectly concentrates all tokens to the local group eliminates communication but saturates local experts.

During training, this tension is managed by an auxiliary load-balancing loss on the coarse router, analogous to the auxiliary loss in flat routing (Chapter 7, `ch07_load_balancing/auxiliary_loss_free_inference.md`). The loss penalizes uneven group utilization and encourages the coarse router to distribute tokens across groups.

### Inference-Time Load Skew

At inference, auxiliary losses are not applied. The coarse router produces group logits based solely on the input token embedding; there is no regularization preventing it from collapsing to a few preferred groups. Empirically, MoE routers trained with auxiliary losses exhibit moderate-to-severe load skew at inference when auxiliary losses are removed — this is documented for flat routing and is expected to be worse for coarse routing, since the coarse logit space ($G = 8$ groups) is smaller and more prone to collapse than the fine logit space ($E = 256$ experts).

---

## Training Dependency

**Critical limitation:** Qwen3.5-35B was trained with standard flat top-$k$ routing. Introducing a coarse router at inference time would require one of the following:

1. **Retraining from scratch** with hierarchical routing objectives and corresponding auxiliary losses.
2. **Fine-tuning** to introduce coarse router weights and calibrate them against the existing expert assignments. This is non-trivial because the existing experts were not organized with group structure in mind; groups defined post-hoc by device assignment may not correspond to semantically coherent expert subsets.

Neither option is available for off-the-shelf Qwen3.5-35B inference. Hierarchical routing is therefore marked as **not applicable** for Qwen3.5-35B on T3K without retraining.

Models that are candidates for hierarchical routing include those specifically designed and trained with two-level routing objectives from the outset, such as [Zhang2022] or [Zoph2022] variants with explicit group-level routing structure.

---

## Communication Reduction in the Favorable Case

For a model that *was* trained with hierarchical routing ($k_c = 1$, $G = N = 8$), the communication reduction relative to flat all-to-all is:

- **Best case (local group selected):** 0 cross-device bytes vs. $100{,}352 \times B$ bytes for flat routing. Reduction factor: $\infty$ (no communication).
- **Average case (uniform group selection):** same expected volume as flat routing, but with variance — some tokens have 0 traffic, others have up to $k_f \times H \times 2$ bytes of traffic.

The practical benefit comes from batching: tokens that select the local group are processed entirely locally (no dispatch delay), reducing average layer latency. If $p_\text{local}$ is the fraction of tokens selecting the local group:

$$\mathbb{E}[T_\text{layer}] = p_\text{local} \times T_\text{FFN} + (1 - p_\text{local}) \times (T_\text{a2a} + T_\text{FFN})$$

$$= T_\text{FFN} + (1 - p_\text{local}) \times T_\text{a2a}$$

For $p_\text{local} = 1/N = 1/8 = 0.125$: $\mathbb{E}[T_\text{layer}] = T_\text{FFN} + 0.875 \times T_\text{a2a}$. The reduction over flat routing ($T_\text{FFN} + T_\text{a2a}$) is a factor of $0.125 \times T_\text{a2a}$ — modest in the uniform case.

If the coarse router achieves $p_\text{local} = 0.5$ (50% of tokens use local experts), the average layer time is $T_\text{FFN} + 0.5 \times T_\text{a2a}$, a 50% communication reduction.

---

## When Hierarchical Routing Wins

Hierarchical routing reduces communication below the all-to-all baseline only when:

1. **The model was trained with hierarchical routing objectives.** Without this, there is no coarse router, and the scheme cannot be applied.
2. **The coarse router achieves $p_\text{local} > 1/N$** — i.e., the coarse router is biased toward local groups more than uniformly random. This requires that the training distribution and the inference distribution align, and that the coarse auxiliary loss successfully induced group locality in the routing.
3. **Cross-device communication is the dominant cost.** In T3K with all-to-all dominating by $\sim 50\times$ over expert FFN compute at decode batch sizes [D UNVERIFIED], any reduction in $T_\text{a2a}$ directly reduces overall latency.

Hierarchical routing is **not applicable** to Qwen3.5-35B on T3K in its off-the-shelf configuration.

---

## References

- [Zoph2022] Zoph, B. et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models", arXiv:2202.08906, 2022.
- [Zhang2022] Zhang, Z. et al., "MoE-I2: Compressing Mixture of Experts with Inter-Expert and Intra-Expert Interactions", arXiv, 2022.
- [Fedus2022] Fedus, W., Zoph, B., Shazeer, N., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR, 2022.
- [Hwang2023] Hwang, C. et al., "Tutel: Adaptive Mixture-of-Experts at Scale", MLSys, 2023.
- [Qwen3] Qwen Team, Alibaba Group, "Qwen3 Technical Report", 2025.
- [Ch1MoEArch] Chapter 1, `ch01_moe_fundamentals/moe_architecture.md` — flat routing equations and top-$k$ selection.
- [Ch2Background] Chapter 2, `ch02_all_to_all_primitives/collective_communication_background.md` — all-to-all volume model.
- [Ch3Index] Chapter 3, `ch03_alternative_routing_schemes/index.md` — chapter overview and notation.
- [Ch4LoadAware] Chapter 4, `ch04_expert_device_assignment/load_aware_assignment.md` — expert popularity profiling.
- [Ch7AuxLoss] Chapter 7, `ch07_load_balancing/auxiliary_loss_free_inference.md` — inference-time load skew without auxiliary losses.
