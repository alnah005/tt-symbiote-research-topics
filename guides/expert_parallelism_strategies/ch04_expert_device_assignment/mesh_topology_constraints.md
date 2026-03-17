# Mesh Topology Constraints: T3K-Specific Assignment

## Overview

The three strategies described in `uniform_partitioning.md`, `load_aware_assignment.md`, and `expert_replication.md` all minimize peak per-device token count, but none of them account for where on the physical mesh each device sits. On the T3K system, devices are connected in a **1×8 linear chain** — not a ring or torus — so communication cost depends on the distance between source and destination device. Expert pairs that are frequently co-activated (selected together by the same token) impose a double hop cost if they reside on distant devices: one hop for dispatch, one for combine, each multiplied by the inter-device distance.

This file analyzes T3K mesh topology, derives the expected hop count for all-to-all communication on a linear chain, formalizes expert co-activation and the co-activation-based placement problem, describes a greedy graph-partition approach, and characterizes when locality optimization delivers meaningful gains.

**Prerequisites:** All three prior files in this chapter (`uniform_partitioning.md`, `load_aware_assignment.md`, `expert_replication.md`). Chapter 2, `ch02_all_to_all_primitives/collective_communication_background.md` (bandwidth/latency model for all-to-all on linear mesh). Readers should be familiar with the T3K topology details established in Chapter 2.

---

## 1. T3K Linear 1×8 Mesh Topology

The T3K system connects 8 Wormhole B0 devices in a **linear (chain) topology** with device IDs $0$ through $7$:

```text
[Device 0] — [Device 1] — [Device 2] — [Device 3] — [Device 4] — [Device 5] — [Device 6] — [Device 7]
```

In TTNN this topology is identified as `ttnn.Topology.Linear` with `cluster_axis=1`. This is distinct from a ring topology (`ttnn.Topology.Ring`), which would add a wrap-around link between devices 0 and 7. T3K does **not** have a ring wrap-around.

> **Warning:** Do not assume T3K uses a ring topology. The absence of the 0–7 wrap-around link means that communication between device 0 and device 7 requires 7 hops through the intermediate devices, not 1 hop via a wrap-around. All-to-all collectives on T3K must use `ttnn.Topology.Linear`, not `ttnn.Topology.Ring`.

**Link bandwidth:** Each point-to-point Ethernet link on T3K provides approximately $\text{BW} \approx 12.5\;\text{GB/s}$. Messages traverse each hop sequentially (store-and-forward model at this abstraction level), so the latency for a message of size $S$ bytes from device $d$ to device $d'$ is approximately:

$$t(d, d') \approx |d - d'| \times \frac{S}{\text{BW}}$$

(ignoring per-hop startup latency, which is small for large messages typical in expert dispatch).

---

## 2. All-to-All Hop Cost on a Linear Chain

In the all-to-all dispatch and combine collectives (Chapter 2, `ch02_all_to_all_primitives/`), each device must send data to every other device. The number of hops from device $d$ to device $d'$ on a linear chain is $|d - d'|$.

**Average hop count:** Over all ordered pairs $(d, d')$ with $d \neq d'$, the average hop count is:

$$\bar{h} = \frac{1}{N(N-1)} \sum_{d=0}^{N-1} \sum_{d'=0, d' \neq d}^{N-1} |d - d'|$$

For $N = 8$:

$$\sum_{d=0}^{7} \sum_{d'=0, d' \neq d}^{7} |d - d'| = 2 \sum_{0 \leq d < d' \leq 7} (d' - d)$$

The sum $\sum_{0 \leq d < d' \leq 7} (d' - d)$ can be computed as:

$$\sum_{\delta=1}^{7} \delta \cdot (8 - \delta) = 1 \cdot 7 + 2 \cdot 6 + 3 \cdot 5 + 4 \cdot 4 + 5 \cdot 3 + 6 \cdot 2 + 7 \cdot 1 = 7 + 12 + 15 + 16 + 15 + 12 + 7 = 84$$

So:

$$\bar{h} = \frac{2 \times 84}{8 \times 7} = \frac{168}{56} = 3.0$$

> **Note on prior result:** An average of 3.0 hops per pair applies when considering ordered pairs $(d, d')$. Each unique unordered pair $\{d, d'\}$ is traversed in both directions (dispatch and combine), so the effective average communication distance across both collectives is $2 \times \bar{h} = 6.0$ hops per token-expert routing event. For informal estimates, the relevant figure is $\bar{h} = 3.0$ hops per single collective direction.

**Maximum hop count:** The farthest pair is devices 0 and 7, with $|0 - 7| = 7$ hops.

**Minimum hop count:** Adjacent devices have $|d - d'| = 1$ hop.

This asymmetry — up to $7\times$ difference between shortest and longest paths — means that the all-to-all bandwidth effectively utilized per link varies significantly across the chain. The middle links (devices 3–4) carry traffic from both halves of the chain and are therefore most heavily loaded during all-to-all.

---

## 3. Locality-Aware Assignment

A locality-aware assignment places experts whose token streams are **correlated in time** on the same or adjacent devices. When two experts $i$ and $j$ are co-activated by the same token, the dispatch collective sends that token to both device $\sigma(i)$ and device $\sigma(j)$. If $\sigma(i) = \sigma(j)$ (same device), one of the two dispatch hops is eliminated entirely. If $\sigma(i)$ and $\sigma(j)$ are adjacent (1 hop), communication is minimized.

**Communication cost of co-activated pair $(i, j)$:** A token routed to both experts $i$ and $j$ originates on some device $d_{\text{src}}$ and incurs two dispatch hops: $|d_{\text{src}} - \sigma(i)|$ and $|d_{\text{src}} - \sigma(j)|$. The combine costs are symmetric. The co-activation-driven contribution to total communication cost is:

$$\text{Cost}_{\text{co-act}}(i, j) = w_{ij} \times \mathbb{E}_{d_{\text{src}}}[|d_{\text{src}} - \sigma(i)| + |d_{\text{src}} - \sigma(j)|]$$

where $w_{ij}$ is the co-activation frequency between experts $i$ and $j$.

Placing $\sigma(i) = \sigma(j)$ (same device) does not reduce this expression to zero, because the token still travels from $d_{\text{src}}$ to the shared device. However, placing co-activated experts on the same device enables **local expert execution without a second dispatch hop**: if a token's $k$ selected experts all reside on the same device, no inter-device dispatch is needed at all for that token. For $k = 8$ experts per token and $E_d = 32$ experts per device, the probability that all 8 selected experts land on one device is $(32/256)^8 = (1/8)^8 \approx 5.96 \times 10^{-8}$ under uniform routing — negligible. But even placing 2 of 8 selected experts on the same device reduces dispatch volume by $1/8 = 12.5\%$ for that token.

---

## 4. Expert Co-Activation Analysis

**Definition:** Experts $i$ and $j$ are **co-activated** by a token if both appear in its top-$k$ selection. The co-activation frequency $w_{ij}$ is the probability that a randomly chosen token selects both $i$ and $j$.

**Baseline co-activation probability under uniform routing:** Under perfectly uniform routing, each expert is selected with probability $k/E = 8/256 = 1/32$ for each of the $k$ selection slots. The probability that both experts $i$ and $j$ are selected by the same token (any two distinct experts from $E$ total, chosen $k$ at a time without replacement) is:

$$w_{ij}^{\text{uniform}} = \frac{\binom{E-2}{k-2}}{\binom{E}{k}} = \frac{k(k-1)}{E(E-1)}$$

For Qwen3.5-35B with $E = 256$, $k = 8$:

$$w_{ij}^{\text{uniform}} = \frac{8 \times 7}{256 \times 255} = \frac{56}{65{,}280} \approx 0.000858$$

So under uniform routing, any given pair of experts is co-activated in about 0.086% of tokens. For a batch of $B = 1{,}000$ tokens, the expected co-activation count per pair is $\approx 0.86$ times — barely detectable.

**Non-uniform co-activation:** Learned routers develop correlations: some expert pairs specialize in related sub-tasks and are consistently co-activated. Empirically, co-activation scores for correlated pairs can be $10$–$100\times$ above the uniform baseline. For example:

$$w_{ij}^{\text{correlated}} \approx 10 \times w_{ij}^{\text{uniform}} \approx 0.0086$$

With $B = 1{,}000$ tokens, such a pair is co-activated $\approx 8.6$ times — now detectable with a calibration set of this size.

---

## 5. Co-Activation Graph and Partition Problem

**Graph construction:** Model the expert co-activation structure as a weighted graph $G = (V, E_G)$ where:

- $V = \{0, 1, \ldots, 255\}$ (one vertex per expert, $|V| = E = 256$),
- $E_G = \{\{i, j\} \mid w_{ij} > 0\}$ (one edge per co-activated pair),
- Edge weight: $w_{ij}$ (empirical co-activation frequency).

**Partition problem:** Find a partition of $V$ into $N = 8$ groups $\mathcal{G}_0, \ldots, \mathcal{G}_7$, each of size exactly $E_d = 32$, that minimizes the weighted cut between non-adjacent device groups:

$$\min_{\mathcal{G}} \sum_{d=0}^{N-1} \sum_{d'=d+1}^{N-1} \text{dist}(d, d') \times \sum_{i \in \mathcal{G}_d, j \in \mathcal{G}_{d'}} w_{ij}$$

where $\text{dist}(d, d') = |d - d'|$ is the hop distance on the linear chain. This objective penalizes co-activated pairs that are placed far apart on the chain more heavily than nearby pairs.

This is a **balanced graph partitioning** problem — NP-hard in general, but amenable to spectral methods or greedy heuristics for $|V| = 256$.

---

## 6. Practical Co-Activation Profiling

**Profiling procedure:**

```python
import torch
from collections import defaultdict
import itertools

def profile_coactivation(
    model,
    calibration_dataloader,
    num_experts: int = 256,
    top_k: int = 8,
    min_count_threshold: int = 2,  # ignore pairs seen fewer than this many times
) -> dict[tuple[int, int], float]:
    """
    Profile expert co-activation frequencies over a calibration dataset.

    Returns a dict mapping (expert_i, expert_j) -> co-activation frequency,
    where i < j and frequency = (co-activation count) / (total token count).

    Only pairs with count >= min_count_threshold are returned to reduce noise.
    """
    coactivation_counts = defaultdict(int)
    total_tokens = 0

    def make_routing_hook():
        def hook(module, input, output):
            nonlocal total_tokens
            # output[1]: routing indices of shape [batch * seq, top_k]
            # Adapt to actual model output format.
            routing_indices = output[1]  # shape: [tokens, top_k]
            batch_tokens = routing_indices.shape[0]
            total_tokens += batch_tokens
            for t in range(batch_tokens):
                experts = sorted(routing_indices[t].tolist())
                # All C(k, 2) = 28 pairs for k=8
                for i, j in itertools.combinations(experts, 2):
                    coactivation_counts[(int(i), int(j))] += 1
        return hook

    hooks = []
    for layer in model.moe_layers:
        h = layer.router.register_forward_hook(make_routing_hook())
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        for batch in calibration_dataloader:
            model(**batch)

    for h in hooks:
        h.remove()

    # Convert counts to frequencies
    coactivation_freq = {}
    for (i, j), count in coactivation_counts.items():
        if count >= min_count_threshold:
            coactivation_freq[(i, j)] = count / total_tokens

    return coactivation_freq
```

**Calibration set size for co-activation:** With $\binom{256}{2} = 32{,}640$ possible expert pairs and baseline co-activation probability $w^{\text{uniform}} \approx 0.000858$, detecting a pair that is $10\times$ above baseline ($w \approx 0.00858$) requires:

$$n \geq \frac{1}{w_{\text{target}}} \times \text{minimum count} = \frac{1}{0.00858} \times 10 \approx 1{,}170 \text{ tokens}$$

For reliable estimates of all significantly co-activated pairs, use at least 10,000 tokens in the calibration set.

---

## 7. Greedy Co-Activation Placement Algorithm

Given the co-activation graph, a practical greedy algorithm places highly co-activated expert pairs on the same device:

```python
def coactivation_aware_assignment(
    coactivation_freq: dict[tuple[int, int], float],
    num_experts: int = 256,
    num_devices: int = 8,
    experts_per_device: int = 32,
) -> list[int]:
    """
    Assign experts to devices using a greedy co-activation placement strategy.

    Algorithm:
    1. Sort expert pairs by co-activation frequency descending.
    2. For each pair (i, j) in order:
       - If both i and j are unassigned: assign both to the same device with
         the most available capacity.
       - If one is assigned (say i -> device d) and d has capacity: assign j -> d.
       - If both are assigned to different devices: no action (already fixed).
    3. Assign any remaining unassigned experts using GDF (load-aware fallback).

    Note: This greedy approach does not guarantee optimal co-activation
    minimization but is fast and produces good results in practice.
    """
    # Initialize: all experts unassigned; track device capacities
    assignment = [-1] * num_experts
    device_capacity = [experts_per_device] * num_devices

    # Sort pairs by co-activation frequency descending
    sorted_pairs = sorted(
        coactivation_freq.items(), key=lambda x: x[1], reverse=True
    )

    def find_device_with_capacity(preferred_device=None):
        """Return a device with remaining capacity, preferring preferred_device."""
        if preferred_device is not None and device_capacity[preferred_device] > 0:
            return preferred_device
        # Find any device with capacity
        for d in range(num_devices):
            if device_capacity[d] > 0:
                return d
        return None  # Should not happen if E == N * E_d

    for (i, j), freq in sorted_pairs:
        assigned_i = assignment[i]
        assigned_j = assignment[j]

        if assigned_i == -1 and assigned_j == -1:
            # Both unassigned: place together on least-loaded device with capacity
            d = find_device_with_capacity()
            if d is None:
                continue
            assignment[i] = d
            device_capacity[d] -= 1
            if device_capacity[d] > 0:
                assignment[j] = d
                device_capacity[d] -= 1

        elif assigned_i >= 0 and assigned_j == -1:
            # i is assigned; try to co-locate j with i
            d = find_device_with_capacity(preferred_device=assigned_i)
            if d is not None:
                assignment[j] = d
                device_capacity[d] -= 1

        elif assigned_i == -1 and assigned_j >= 0:
            # j is assigned; try to co-locate i with j
            d = find_device_with_capacity(preferred_device=assigned_j)
            if d is not None:
                assignment[i] = d
                device_capacity[d] -= 1

        # If both are assigned, no action needed

    # Assign remaining unassigned experts with uniform fallback
    unassigned = [e for e in range(num_experts) if assignment[e] == -1]
    for e in unassigned:
        d = find_device_with_capacity()
        if d is not None:
            assignment[e] = d
            device_capacity[d] -= 1
        else:
            raise RuntimeError(
                f"No capacity remaining to assign expert {e}. "
                f"Check that num_experts == num_devices * experts_per_device."
            )

    assert all(a >= 0 for a in assignment), "Some experts remain unassigned"
    return assignment
```

> **Tip:** In practice, combine co-activation placement with load-aware assignment by first running the co-activation greedy algorithm to handle high-frequency pairs, then running GDF (from `load_aware_assignment.md`) on the resulting assignment to equalize per-device load. The two objectives — load balance and locality — are generally compatible because co-activation constraints affect only the most popular pairs, which are a small subset of all 256 experts.

---

## 8. When Locality Optimization Helps

Locality-aware placement produces the largest gains when:

1. **Routing is highly non-uniform:** If many tokens are concentrated on a small subset of expert pairs, those pairs' placement strongly affects communication cost.

2. **Co-activation is structured:** If expert $i$ and expert $j$ are co-activated in $10\times$ more tokens than the uniform baseline, placing them on the same or adjacent devices reduces dispatch volume for those tokens.

3. **Batch size is small:** At small batch sizes ($B = 1$–$4$), latency is dominated by hop count rather than bandwidth, so reducing hops has a proportionally larger effect.

**When locality optimization does not help:**

- **Uniform routing:** If $w_{ij} \approx w^{\text{uniform}} = 0.000858$ for all pairs, there is no locality structure to exploit, and co-activation placement reduces to arbitrary assignment. GDF load-aware assignment is strictly better.
- **Large batch sizes:** At large $B$, bandwidth dominates over hop latency, and load balance is the primary lever. Locality optimization is secondary.
- **Pre-existing ring topology assumptions:** If the all-to-all collective implementation assumes a ring topology (it should not on T3K), locality placement on a linear chain may have unexpected effects.

**Summary heuristic:** Apply co-activation placement if profiling reveals at least one expert pair with $w_{ij} > 5 \times w^{\text{uniform}} \approx 0.0043$ and the routing distribution is sufficiently non-uniform ($L > 1.1$ under round-robin). Otherwise, GDF load-aware assignment (without locality) is sufficient.

---

## 9. Static Assignment Assumption

All strategies described in this chapter — uniform partitioning, load-aware bin-packing, expert replication, and co-activation placement — assume that the expert-to-device assignment is **fixed at deployment time**. For re-assignment mechanics (re-profiling, re-solving, weight migration cost, and dispatch metadata updates), see `load_aware_assignment.md`, Section 6. For most production deployments, the assignment should be treated as immutable and re-computed only during scheduled maintenance windows or when a significant workload shift is detected (e.g., the load imbalance factor $L$ sustained above 1.5 for more than 10,000 forward passes).

---

## 10. Summary: Topology-Aware vs. Topology-Agnostic Assignment

| Criterion | Topology-agnostic (GDF) | Topology-aware (co-activation + linear chain) |
|---|---|---|
| Profiling required | Routing frequencies $f_e$ only | Routing frequencies + co-activation matrix $w_{ij}$ |
| Optimization target | Peak device load $\max_d T_d$ | Peak device load + weighted cut on linear chain |
| Implementation complexity | Low (greedy sort + heap) | Moderate (co-activation profiling + graph heuristic) |
| Calibration data needed | ~1,000 tokens | ~10,000 tokens |
| Expected gain over round-robin | Large ($L$ from $\approx 1.4$ to $\approx 1.01$) | Additional 5–15% reduction in dispatch latency (estimate) |
| When to use | Always (replaces round-robin) | When profiling reveals structured co-activation |

---

## References

- [Ch1Routing] Chapter 1, `ch01_moe_fundamentals/routing_problem.md` — dispatch/combine communication pattern, token imbalance.
- [Ch2Background] Chapter 2, `ch02_all_to_all_primitives/collective_communication_background.md` — bandwidth/latency model for all-to-all on linear mesh; `ttnn.Topology.Linear` topology.
- [Ch2Dispatch] Chapter 2, `ch02_all_to_all_primitives/all_to_all_dispatch.md` — dispatch metadata format and buffer layout.
- [Ch3Matrix] Chapter 3, `ch03_alternative_routing_schemes/scheme_comparison_matrix.md` — routing scheme comparison and T3K recommendations.
- [Ch4Uniform] Chapter 4, `ch04_expert_device_assignment/uniform_partitioning.md` — round-robin assignment, load imbalance metric.
- [Ch4LoadAware] Chapter 4, `ch04_expert_device_assignment/load_aware_assignment.md` — GDF algorithm, routing frequency profiling.
- [Ch4Replication] Chapter 4, `ch04_expert_device_assignment/expert_replication.md` — expert replication and dispatch metadata with replicas.
- [Karypis1998] Karypis, G., Kumar, V., "A Fast and High Quality Multilevel Scheme for Partitioning Irregular Graphs", SIAM Journal on Scientific Computing, 20(1):359–392, 1998. (METIS graph partitioner, applicable to co-activation graph partition.)
- [Fedus2022] Fedus, W., Zoph, B., Shazeer, N., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR, 2022.
- [Rajbhandari2022] Rajbhandari, S. et al., "DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale", ICML, 2022.
- [Qwen3] Qwen Team, Alibaba Group, "Qwen3 Technical Report", 2025.
- [PlanDoc] Expert Parallelism Strategies — Research Guide Plan, `guides/expert_parallelism_strategies/plan.md`.

---

**Next:** [Chapter 5 — Routing Weight Optimization](../ch05_routing_weight_optimization/index.md)
