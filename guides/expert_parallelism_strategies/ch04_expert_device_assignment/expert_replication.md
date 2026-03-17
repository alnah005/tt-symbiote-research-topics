# Expert Replication

## Overview

Even with load-aware assignment, a single highly popular expert can saturate its host device if its routing frequency $f_e$ is large enough that the token count exceeds the device's processing capacity per forward pass. Expert replication addresses this by placing copies of popular experts on multiple devices so that incoming tokens can be distributed across replicas. This file derives the optimal replication factor, quantifies the memory cost, explains how dispatch must be modified to handle replicas, and provides practical recommendations for Qwen3.5-35B.

**Prerequisites:** `uniform_partitioning.md` (load imbalance metric and memory footprint baseline), `load_aware_assignment.md` (routing frequencies $f_e$ and bin-packing formulation), Chapter 2 `ch02_all_to_all_primitives/all_to_all_dispatch.md` (dispatch metadata and buffer layout). Expert capacity $C$ is formally defined in Chapter 7, `ch07_load_balancing/capacity_factor_mechanics.md`; this file uses the concept without repeating the full derivation.

---

## 1. Motivation: When Load-Aware Assignment Is Insufficient

Load-aware assignment (`load_aware_assignment.md`) equalizes the total load across devices by balancing the sum $\sum_{e \in \mathcal{E}_d} f_e$ across all $d$. It does not, however, eliminate the problem of a single expert with anomalously high frequency $f_e$.

**Capacity overflow condition:** Expert $e$ overflows its capacity $C$ when:

$$f_e \cdot B > C$$

where $C = \lceil k \cdot B \cdot S / E \rceil$ is the per-expert capacity (for $S = 1$: $C = \lceil k \cdot B / E \rceil$). Substituting $C = \lceil k \cdot B / E \rceil \approx k \cdot B / E$:

$$f_e \cdot B > \frac{k \cdot B}{E}$$
$$f_e > \frac{k}{E} = f_{\text{avg}} = \frac{8}{256} = 0.03125$$

Any expert with above-average frequency overflows when $CF = 1.0$. With $CF = 1.25$, the capacity is scaled to $CF \cdot k \cdot B / E$, giving a higher overflow threshold:

$$f_e > CF \cdot \frac{k}{E} = 1.25 \times 0.03125 = 0.0391$$

Experts with $f_e > 0.0391$ will still overflow at $CF = 1.25$. Empirically, the top-1% of experts in a Zipf-like distribution can have $f_e \approx 4 \times f_{\text{avg}} = 0.125$, far exceeding this threshold. Token dropping at overflow degrades model quality; expert replication prevents overflow by distributing the load across $r_e$ replicas.

---

## 2. Replication Factor Formula

If expert $e$ is replicated on $r_e$ devices, and tokens are distributed uniformly across replicas, the expected token count per replica per forward pass is:

$$T_{\text{replica}}(e) = \frac{f_e \cdot B}{r_e}$$

Setting $T_{\text{replica}}(e) \leq C \approx k \cdot B / E$ and solving for $r_e$:

$$\frac{f_e \cdot B}{r_e} \leq \frac{k \cdot B}{E}$$
$$r_e \geq \frac{f_e \cdot E}{k}$$

Therefore the **minimum replication factor** to prevent overflow is:

$$r_e = \max\left(1,\; \left\lceil \frac{f_e \cdot E}{k} \right\rceil\right)$$

For Qwen3.5-35B with $E = 256$ and $k = 8$:

$$r_e = \max\left(1,\; \left\lceil \frac{256 \cdot f_e}{8} \right\rceil\right) = \max\left(1,\; \lceil 32 \cdot f_e \rceil\right)$$

**Examples:**

| Expert frequency $f_e$ | $f_e \cdot E / k$ | Minimum $r_e$ | Replicas needed |
|---|---|---|---|
| $f_{\text{avg}} = 0.03125$ | 1 | 1 | No replication needed |
| $2 \times f_{\text{avg}} = 0.0625$ | 2 | 2 | 2 devices hold a copy |
| $4 \times f_{\text{avg}} = 0.125$ | 4 | 4 | 4 of 8 devices hold a copy |
| $0.5 \times f_{\text{avg}} = 0.01563$ | 0.5 | 1 | No replication needed |
| $0.25 \times f_{\text{avg}} = 0.00781$ | 0.25 | 1 | No replication needed |

Since $r_e$ is capped at $N = 8$ (there are only 8 devices), experts with $f_e \cdot E / k > 8$ (i.e., $f_e > 8k/E = 8 \times 8/256 = 0.25$) must be replicated on all 8 devices. For those experts, the remaining imbalance is irreducible: each replica still receives $f_e \cdot B / N$ tokens.

> **Warning:** The formula $r_e = \lceil f_e \cdot E / k \rceil$ assumes routing is stationary (frequencies do not change across batches) and that tokens are uniformly distributed across replicas. If routing burstiness is high (a single batch has $3\times$ the average traffic to one expert), a higher $r_e$ or a higher $CF$ may be needed.

---

## 3. Memory Cost of Replication

Replicating expert $e$ on $r_e$ devices costs $r_e \times W_{\text{expert}}$ bytes of total DRAM across the system, versus $W_{\text{expert}}$ bytes without replication.

**Per-device DRAM overhead:** If expert $e$ is replicated on all $N = 8$ devices, each device stores one additional copy beyond the baseline $E_d = 32$ expert allocation. The per-device DRAM overhead for replicating $M$ experts on all devices is:

$$\Delta\text{DRAM}_{\text{device}} = M \times W_{\text{expert}}$$

where $W_{\text{expert}}$ is the per-expert weight size (see `uniform_partitioning.md`, Section 2 for the full derivation: $W_{\text{expert}} = 6HD$ bytes BF16).

**Memory budget constraint:** Let $\text{DRAM}_{\text{budget}}$ be the available per-device DRAM after accounting for model weights, KV cache, activations, and system overhead. Then:

$$M \leq \frac{\text{DRAM}_{\text{budget}}}{W_{\text{expert}}}$$

> **Warning:** Replication increases DRAM memory usage proportionally. Before enabling expert replication, measure the per-device DRAM budget explicitly. On Wormhole B0 devices with limited DRAM, replicating even a small number of large experts can exhaust available memory and cause allocation failures.

---

## 4. Consistency at Inference Time

Replicated experts are **read-only** during inference: all replicas hold identical copies of the expert's weight tensors ($W_{\text{gate}}$, $W_{\text{up}}$, $W_{\text{down}}$). Because expert weights are never updated at inference time (no gradient computation, no optimizer steps), all replicas remain byte-for-byte identical after the initial weight loading step.

This means:
- No synchronization barrier is needed between replicas.
- No all-reduce is needed over replica outputs.
- Replicas are fully independent: each processes its assigned token subset and produces independent outputs. The combine collective (Chapter 2, `ch02_all_to_all_primitives/all_to_all_combine.md`) returns each token's expert output to its originating device without needing to know whether the expert was replicated.

The only consistency requirement is at weight-load time: all replicas must be initialized from the same checkpoint file. This is a one-time operation at deployment.

---

## 5. Dispatch Interaction with Replicated Experts

When expert $e$ has $r_e > 1$ replicas on devices $\mathcal{R}(e) \subseteq \{0, \ldots, N-1\}$, the dispatch operation must route each token destined for expert $e$ to exactly one of the $r_e$ replica devices (not to all of them — that would multiply computation by $r_e$).

The dispatch metadata (see Chapter 2, `ch02_all_to_all_primitives/all_to_all_dispatch.md`) must therefore include a **replica routing table** that maps each (token, expert) pair to a specific replica device:

```python
# Non-replicated case: destination device = expert_index % num_devices
destination = expert_index % num_devices

# Replicated case: destination device = one of the replica devices, chosen by policy
replica_devices = replica_map[expert_index]  # e.g., [0, 2, 5, 7] for 4 replicas
destination = select_replica(replica_devices, token_id, policy="round_robin")
```

**Replica selection policies:**

| Policy | Description | Load balance | Overhead |
|---|---|---|---|
| Round-robin | Assign token $t$ to replica $t \bmod r_e$ | Perfect (static) | Requires global token counter per expert |
| Random | Assign each token to a uniformly random replica | Expected equal | Variance per batch; no counter needed |
| Least-loaded | Route to the replica with fewest queued tokens | Optimal | Requires inter-device load query — not practical at dispatch time |
| Hash-based | Destination = hash(token_id) % $r_e$ | Perfect (static) | Requires per-token hash at dispatch; deterministic and reproducible |

For Qwen3.5-35B inference, **round-robin** or **hash-based** selection is recommended. Both are computable at dispatch time without cross-device communication and produce perfectly balanced load in expectation over a batch.

```python
def build_dispatch_metadata_with_replicas(
    routing_indices: list[list[int]],  # routing_indices[t] = list of k expert indices for token t
    replica_map: dict[int, list[int]],  # expert_e -> [device_d0, device_d1, ...]
    num_devices: int = 8,
) -> list[tuple[int, int, int]]:
    """
    Build dispatch metadata for a batch with expert replication.

    Parameters
    ----------
    routing_indices : list of list of int
        For each token t, the top-k expert indices selected by the router.
    replica_map : dict
        Maps expert index to list of device indices holding replicas.
        For non-replicated experts, maps e -> [e % num_devices].
    num_devices : int
        Total number of devices.

    Returns
    -------
    metadata : list of (token_idx, expert_idx, dest_device)
        One entry per (token, expert) routing event.
    """
    metadata = []
    # Per-expert round-robin counter (for load balancing across replicas)
    replica_counters = {e: 0 for e in replica_map}

    for t, experts in enumerate(routing_indices):
        for e in experts:
            replicas = replica_map[e]
            if len(replicas) == 1:
                dest = replicas[0]
            else:
                # Round-robin selection among replicas
                dest = replicas[replica_counters[e] % len(replicas)]
                replica_counters[e] += 1
            metadata.append((t, e, dest))

    return metadata
```

> **Tip:** The replica_map data structure must be consistent across all devices participating in the dispatch collective. Distribute it as shared configuration at deployment time, before the first forward pass.

---

## 6. Practical Recommendation for Qwen3.5-35B

Under a Zipf-like routing distribution (empirically reported for MoE models), the top-$M$ most popular experts each receive approximately:

$$f_{(1)} \approx \frac{k}{E} \cdot H_E \approx \frac{8}{256} \cdot \ln(256) \approx 0.03125 \times 5.55 \approx 0.17$$

where $H_E \approx \ln(E)$ is the $E$-th harmonic number (a rough approximation for Zipf exponent 1). Using the corrected replication formula, this implies $r_{(1)} = \lceil 0.17 \times 256 / 8 \rceil = \lceil 5.44 \rceil = 6$. The top expert needs 6 replicas — less than $N = 8$, so full replication on all devices is not strictly required (though replicating on all 8 is an acceptable conservative choice if DRAM budget permits).

More conservatively, empirical measurements on deployed MoE models report that the top-8 most popular experts typically receive $\approx 3$–$5\times$ the average frequency:

$$f_{\text{top-8}} \approx 4 \times f_{\text{avg}} = 4 \times \frac{8}{256} = 0.125$$

Expected tokens per forward pass for such an expert (at batch size $B = 32$):

$$T_{\text{expert}} = f_{\text{top}} \times B = 0.125 \times 32 = 4 \text{ tokens}$$

Average expert token count: $k \times B / E = 8 \times 32 / 256 = 1$ token per expert. A hot expert receiving 4 tokens is $4\times$ over the average. Using the replication formula: $r_e = \max(1, \lceil f_e \cdot E / k \rceil) = \max(1, \lceil 0.125 \times 256 / 8 \rceil) = 4$. Replicating it on 4 devices reduces each replica's load to $4/4 = 1$ token, matching the average load.

**Recommended configuration:**

1. Profile routing frequencies $f_e$ over a 5,000–10,000 token calibration set (see `load_aware_assignment.md`, Section 1).
2. Identify the top-$M$ experts satisfying $f_e > 2 \times f_{\text{avg}} = 2 \times k/E$.
3. Replicate each such expert on all $N = 8$ devices (full replication; partial replication adds dispatch complexity with marginal memory savings).
4. Check per-device DRAM budget: $M \times W_{\text{expert}} \leq \text{DRAM}_{\text{budget}}$.
5. Update dispatch metadata to use round-robin replica selection.

For Qwen3.5-35B, start with $M = 8$ (replicate the top-8 experts). If memory allows and profiling reveals more experts with $f_e > 2 \times f_{\text{avg}}$, increase $M$ accordingly.

```python
def select_experts_for_replication(
    frequencies: list[float],
    num_devices: int = 8,
    top_k: int = 8,
    frequency_multiplier_threshold: float = 2.0,
    max_replicated_experts: int = None,  # None = no limit (use memory budget check)
) -> list[int]:
    """
    Return the list of expert indices that should be replicated on all devices.

    An expert is replicated if its frequency exceeds
    frequency_multiplier_threshold * f_avg.

    Parameters
    ----------
    frequencies : list of float, length E=256
        Per-expert routing frequencies. sum(frequencies) == top_k.
    num_devices : int
        Number of devices (N=8).
    top_k : int
        Number of experts selected per token (k=8).
    frequency_multiplier_threshold : float
        Experts with f_e > threshold * f_avg are replicated. Default 2.0.
    max_replicated_experts : int or None
        Cap on number of experts to replicate (from memory budget). None = no cap.

    Returns
    -------
    hot_experts : list of int
        Expert indices to replicate on all num_devices devices.
    """
    num_experts = len(frequencies)
    f_avg = top_k / num_experts  # = 8/256 = 0.03125 for Qwen3.5-35B

    threshold = frequency_multiplier_threshold * f_avg
    hot_experts = [e for e, f in enumerate(frequencies) if f > threshold]

    # Sort by frequency descending so we pick the hottest experts first if capped
    hot_experts.sort(key=lambda e: frequencies[e], reverse=True)

    if max_replicated_experts is not None:
        hot_experts = hot_experts[:max_replicated_experts]

    return hot_experts


def build_replica_map(
    assignment: list[int],   # base assignment: assignment[e] = primary device
    hot_experts: list[int],  # experts to replicate on all devices
    num_devices: int = 8,
) -> dict[int, list[int]]:
    """
    Build the expert-to-replica-devices mapping.

    For hot experts: all devices hold a replica.
    For non-hot experts: only the primary device (from assignment) holds the expert.
    """
    replica_map = {}
    for e, d in enumerate(assignment):
        if e in set(hot_experts):
            replica_map[e] = list(range(num_devices))  # all devices
        else:
            replica_map[e] = [d]  # primary device only
    return replica_map
```

---

## 7. Summary: Replication Trade-offs

| Property | No replication | Full replication of top-$M$ experts |
|---|---|---|
| DRAM per device | $E_d \times W_{\text{expert}}$ | $(E_d + M) \times W_{\text{expert}}$ |
| Peak device token count | Proportional to $\max_d \sum_{e \in \mathcal{E}_d} f_e$ | Reduced for hot experts; proportional to $f_{(1)} / r_{(1)}$ |
| Dispatch metadata | Simple (`e % N`) | Requires replica_map lookup |
| Weight loading time | $E_d \times W_{\text{expert}} / \text{BW}$ per device | $(E_d + M) \times W_{\text{expert}} / \text{BW}$ per device |
| Synchronization at inference | None needed | None needed (read-only replicas) |
| Benefit | Zero overhead | Eliminates hot-expert bottleneck |
| Risk | Bottleneck at hot experts | DRAM overflow if $M$ too large |

---

## References

- [Ch1Routing] Chapter 1, `ch01_moe_fundamentals/routing_problem.md` — expert capacity concept, token dropping, dispatch/combine pattern.
- [Ch1Config] Chapter 1, `ch01_moe_fundamentals/qwen35b_config.md` — Qwen3.5-35B architectural constants.
- [Ch2Dispatch] Chapter 2, `ch02_all_to_all_primitives/all_to_all_dispatch.md` — dispatch buffer layout, send-count metadata, device addressing.
- [Ch4Uniform] Chapter 4, `ch04_expert_device_assignment/uniform_partitioning.md` — memory footprint per device, load imbalance metric.
- [Ch4LoadAware] Chapter 4, `ch04_expert_device_assignment/load_aware_assignment.md` — routing frequency profiling, bin-packing formulation.
- [Ch7Capacity] Chapter 7, `ch07_load_balancing/capacity_factor_mechanics.md` — formal definition of expert capacity $C$ and capacity factor $CF$.
- [Lepikhin2021] Lepikhin, D. et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding", ICLR, 2021.
- [Fedus2022] Fedus, W., Zoph, B., Shazeer, N., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR, 2022.
- [Zoph2022] Zoph, B. et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models", arXiv:2202.08906, 2022.
- [Rajbhandari2022] Rajbhandari, S. et al., "DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale", ICML, 2022.
- [Qwen3] Qwen Team, Alibaba Group, "Qwen3 Technical Report", 2025.
