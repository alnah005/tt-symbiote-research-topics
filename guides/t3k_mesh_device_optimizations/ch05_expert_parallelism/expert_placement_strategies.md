# Expert Placement Strategies

> **Quick Reference — No New TTNN API Symbols Introduced**
>
> This file analyzes placement strategies and their impact on load balance and communication; it does not introduce new `ttnn` API calls. Relevant API symbols (`ttnn.all_to_all`, `ttnn.Topology.Linear`) are defined in `ch02_ttnn_mesh_api/collective_primitives.md` and `ch03_all_to_all_num_links/num_links_parameter.md`.

Expert parallelism (EP) distributes the $E = 256$ experts of Qwen3.5-35B across $N = 8$ T3K devices, assigning $E_d = 32$ experts per device. The assignment rule — which experts land on which device — determines load balance, memory footprint, and all-to-all communication patterns. This file defines and analyzes four strategies: naive uniform, load-balanced, locality-aware, and expert replication.

All weight sizes use $W_{\text{expert}} = 6HD$ bytes (BF16), the canonical definition from `expert_parallelism_strategies/ch04_expert_device_assignment/uniform_partitioning.md` Section 2, where $H = 7168$ is the hidden dimension and $D$ is the expert FFN (feed-forward network) intermediate dimension [UNVERIFIED exact value].

---

## 1. Baseline — Naive Uniform Placement

### Assignment Rule

Device $d$ ($d \in \{0, 1, \ldots, 7\}$) holds experts $\{32d, 32d+1, \ldots, 32d+31\}$:

| Device | Experts |
|---|---|
| 0 | 0–31 |
| 1 | 32–63 |
| 2 | 64–95 |
| 3 | 96–127 |
| 4 | 128–159 |
| 5 | 160–191 |
| 6 | 192–223 |
| 7 | 224–255 |

The dispatch mapping is computable with integer division: `device_id = expert_index // 32`. No runtime lookup table is required.

> **Tip:** This contiguous-block rule differs from the round-robin rule (`expert_index % 8`) used in some implementations. Contiguous blocks give each device a compact index range, which can simplify expert weight tensor addressing in DRAM. Either rule partitions 256 experts into 8 groups of 32; the load-balance properties are identical under uniform routing.

### Memory per Device

Each expert's weight consists of three projection matrices (gate, up, down), giving:

$$W_{\text{expert}} = 6HD \text{ bytes (BF16)}$$

With 32 experts per device:

$$W_{\text{device}} = 32 \times W_{\text{expert}} = 32 \times 6HD = 192HD \text{ bytes (BF16)}$$

These weights reside in device DRAM as established in `ch04_memory_config/decode_memory_strategy.md`.

### Expected Token Load Under Uniform Routing

Define $f_e$ as the fraction of tokens routed to expert $e$ per forward pass (so $\sum_e f_e = k = 8$). Under uniform routing, $f_e = f_{\text{avg}} = k/E = 8/256 = 1/32$ for all $e$.

The token load on device $d$ for a batch of $B$ tokens is:

$$T_d = B \times \sum_{e \in \mathcal{E}_d} f_e$$

Under uniform routing:

$$T_d = B \times 32 \times \frac{1}{32} = B \text{ tokens per device}$$

Each device processes exactly $B$ tokens — the ideal balanced case.

### All-to-All Payload

Each device sends a padded buffer of shape $[C \times E_d, H]$ to each of the $N-1 = 7$ other devices, where $C = \lceil B \times \text{CF} / E_d \rceil$ is the per-expert capacity and $E_d = 32$ experts per device. Each element is 2 bytes (BF16). The total dispatch volume per device is:

$$V_{\text{dispatch}} = (N-1) \times C \times E_d \times H \times 2 \text{ bytes}$$

At $B=32$ ($C = 2$), $H=7168$: $V_{\text{dispatch}} = 7 \times 2 \times 32 \times 7168 \times 2 = 6{,}422{,}528$ bytes $\approx 6.4$ MB.

### Load Imbalance Under Non-Uniform Routing

Under non-uniform routing, device $d$'s load is $T_d = B \times \sum_{e \in \mathcal{E}_d} f_e$. If experts 0–31 collectively receive a higher-than-average routing share, device 0 becomes a bottleneck. All devices must wait for the slowest device before combine can proceed.

> **Warning:** Under Zipf-like routing distributions (empirically observed in MoE models), load imbalance can reach $L = \max_d T_d / B \approx 2$–$4$ with naive uniform assignment, because popular experts cluster by index. The strategies in Sections 2–4 address this.

---

## 2. Load-Balanced Placement

### Motivation

Non-uniform routing creates hotspots: devices holding high-$f_e$ experts process more tokens per step and extend the critical path for all-to-all combine. Load-balanced placement assigns experts to devices such that each device's total routing load is approximately equal, regardless of per-expert popularity.

### Profiling Step

Collect $f_e$ for all $e \in \{0, \ldots, 255\}$ by running the router over a representative calibration dataset. The resulting frequency vector $\mathbf{f} \in \mathbb{R}^{256}$ satisfies $\sum_e f_e = k = 8$ (since each token selects $k=8$ experts). The average per-expert frequency is $f_{\text{avg}} = 8/256 = 1/32$.

### Bin-Packing Assignment

The goal is to assign 256 experts into 8 bins (devices) minimizing the maximum bin load:

$$\min_{\mathcal{E}_0, \ldots, \mathcal{E}_7} \max_{d} \sum_{e \in \mathcal{E}_d} f_e \quad \text{subject to } |\mathcal{E}_d| = 32, \; \bigcup_d \mathcal{E}_d = \{0, \ldots, 255\}$$

**Greedy decreasing algorithm:**
1. Sort experts by $f_e$ descending.
2. Maintain a priority queue of devices keyed by current total load.
3. For each expert in sorted order, assign it to the least-loaded device.
4. The result typically achieves near-optimal balance for $E = 256$ experts and $N = 8$ bins.

**Target load per device:** Each device should achieve $\sum_{e \in \mathcal{E}_d} f_e \approx k/N = 8/8 = 1.0$, so each device processes approximately $B$ tokens per step — matching the uniform-routing ideal.

### Practical Impact for Qwen3.5-35B

With 256 experts and $k=8$, many experts have similar $f_e$ (close to $f_{\text{avg}} = 1/32$). Load balancing provides modest improvement unless routing is highly skewed. The improvement is largest when a small number of experts carry disproportionate load (e.g., Zipf-distributed routing with $s \geq 1.0$).

> **Tip:** If profiling reveals that the top 10 experts each have $f_e > 3 \times f_{\text{avg}}$, load-balanced placement is strongly warranted. If the distribution is near-uniform ($f_e / f_{\text{avg}} < 1.5$ for all $e$), naive uniform placement incurs minimal penalty.

### Dispatch Overhead

Load-balanced placement requires an explicit expert-to-device table built once at model initialization:

```python
expert_to_device = build_load_balanced_assignment(f_profile, num_devices=8)

def dispatch_device(expert_idx: int) -> int:
    return expert_to_device[expert_idx]
```

A single table lookup per (token, expert) pair — negligible overhead relative to the all-to-all transfer cost.

---

## 3. Locality-Aware Placement

### Co-Activation and Communication Cost

Two experts $i$ and $j$ are **co-activated** when the same token selects both in the same forward pass. On T3K's 1×8 linear mesh, the cost of sending a token to two devices is proportional to the hop distances traversed.

The T3K linear mesh has distances (see `ch01_t3k_topology/t3k_physical_layout.md`):

$$\text{dist}(i, j) = |i - j| \text{ hops}$$

The average hop count across all 28 device pairs is 3.0, derived as:

$$\bar{h} = \frac{\sum_{0 \le i < j \le 7} |i-j|}{\binom{8}{2}} = \frac{7 + 12 + 15 + 16 + 15 + 12 + 7}{28} = \frac{84}{28} = 3.0$$

If co-activated experts $i$ and $j$ are placed on **adjacent** devices (distance 1), the token traverses 1 hop to reach each expert. If they are on devices 0 and 7 (distance 7), the token traverses 7 hops. Minimizing the sum of hop counts across all co-activated expert pairs reduces total Ethernet traffic and latency.

### Placement Objective

Define the co-activation frequency $f_{ij}$ as the probability that a random token selects both expert $i$ and expert $j$. The placement objective is:

$$\min_{\sigma: \{0,\ldots,255\} \to \{0,\ldots,7\}} \sum_{i < j} f_{ij} \times |\sigma(i) - \sigma(j)|$$

where $\sigma(e)$ is the device assigned to expert $e$ and $|\sigma(i) - \sigma(j)|$ is the hop distance between devices.

This is a graph partitioning problem: experts are nodes, edge $(i,j)$ has weight $f_{ij}$, and the objective is to partition the graph into 8 balanced groups minimizing total cut weighted by device distance. Standard spectral or metis-style partitioners can be applied.

### Building the Co-Activation Matrix

```python
# co_act[i, j] = number of forward passes where token selected both expert i and j
# Shape: [256, 256]; symmetric; diagonal is zero
import numpy as np

def build_coactivation_matrix(routing_indices: np.ndarray) -> np.ndarray:
    """
    routing_indices: [num_tokens, k] array of expert indices selected per token.
    Returns co_act: [E, E] symmetric co-activation count matrix.
    """
    E = 256
    co_act = np.zeros((E, E), dtype=np.int64)
    for token_experts in routing_indices:  # token_experts: shape [k=8]
        for a in range(len(token_experts)):
            for b in range(a + 1, len(token_experts)):
                i, j = token_experts[a], token_experts[b]
                co_act[i, j] += 1
                co_act[j, i] += 1
    return co_act
```

### Practical Considerations

- Requires large-scale profiling of co-activation patterns across the calibration dataset.
- The benefit over load-balanced placement is model-dependent: in models with strongly correlated expert selection (e.g., task-specific routing), locality-aware placement reduces hop counts meaningfully; in models with near-random co-activation, the benefit is small.
- In practice, load balance is the dominant concern for throughput; locality-aware placement is added as a secondary criterion.

See `ch01_t3k_topology/topology_implications_for_collectives.md` for how hop count translates to link-level latency on T3K.

---

## 4. Expert Replication for Hot Experts

### Motivation

When a single expert has $f_e \gg f_{\text{avg}}$, even the best bin-packing assignment cannot eliminate load imbalance: the device holding that expert processes far more tokens than others. Expert replication places multiple copies of a hot expert on different devices, spreading the token load.

### Replication Factor Formula

The replication factor for expert $e$ is:

$$r_e = \max\!\left(1, \left\lceil \frac{f_e \times E}{k} \right\rceil\right) = \max(1, \lceil 32 \times f_e \rceil)$$

This formula (from `expert_parallelism_strategies/ch04_expert_device_assignment/expert_replication.md`) ensures that each replica handles approximately $f_e / r_e \approx f_{\text{avg}}$ of tokens — restoring balance to average load.

**Overflow threshold with capacity factor CF = 1.25:**

$$f_e > \frac{\text{CF} \times k}{E} = \frac{1.25 \times 8}{256} = \frac{1.25}{32} \approx 0.0391$$

Any expert with $f_e > 0.0391$ will overflow its capacity slot (capacity $C = \lceil B \times \text{CF} / 32 \rceil$) and is a candidate for replication.

### Worked Example — Zipf-Distributed Hot Expert

Consider a Qwen3.5-35B deployment with Zipf-skewed routing where the most popular expert has $f_e = 0.17$:

$$r_e = \max\!\left(1, \left\lceil \frac{0.17 \times 256}{8} \right\rceil\right) = \max(1, \lceil 5.44 \rceil) = 6$$

Six copies of this expert are placed across 6 devices (for example, devices 0–5). Each copy handles:

$$\frac{f_e}{r_e} = \frac{0.17}{6} \approx 0.028 \approx f_{\text{avg}}$$

This is approximately equal to the average per-expert load, restoring balance.

**Memory cost:** 6 copies × $W_{\text{expert}} = 6 \times 6HD = 36HD$ bytes instead of $6HD$ bytes. The 5 extra copies occupy memory that could hold 5 other experts on those devices.

**Feasibility check:** With $E_d = 32$ slots per device and 1 slot used by the replica, 31 slots remain for other experts. For a single very hot expert, replication across 6 devices is feasible without violating the 32-experts-per-device constraint (each of the 6 devices hosts the replica in addition to its normal 32 experts, requiring 33 slots; whether this is acceptable depends on total memory budget).

> **Warning:** Expert replication increases total memory consumption by $(r_e - 1) \times W_{\text{expert}}$ per replicated expert. For multiple hot experts, the cumulative memory cost may require reducing experts per device or evicting cold experts.

### Token-to-Replica Assignment

With $r_e = 6$ replicas on devices 0–5, each token routed to expert $e$ must be deterministically directed to one specific replica:

```python
def replica_device(token_id: int, expert_idx: int, replica_devices: list) -> int:
    """
    Deterministically select which replica of expert_idx receives this token.
    replica_devices: list of device ids holding replicas of expert_idx.
    """
    r = len(replica_devices)
    replica_slot = hash(token_id) % r
    return replica_devices[replica_slot]
```

Hash-based assignment distributes tokens uniformly across replicas without requiring coordination between devices.

**Dispatch metadata requirement:** The routing metadata table must carry, for each (expert, replica) pair, the destination device. The all-to-all send buffer construction must look up the replica device rather than assuming the default assignment.

### Trade-Off Summary

| Criterion | No Replication | Replication ($r_e = 6$) |
|---|---|---|
| Memory per copy | $W_{\text{expert}}$ | $6 \times W_{\text{expert}}$ |
| Load on hot device | $f_e \times B = 0.17B$ | $\approx f_{\text{avg}} \times B = B/32$ |
| Dispatch complexity | Simple modulo | Table lookup with replica index |
| Worth it when | $f_e \le 0.0391$ | $f_e > 0.0391$ (CF=1.25 threshold) |

---

## 5. Impact on All-to-All Payload and `num_links`

### Volume Is Strategy-Independent

The total dispatch all-to-all volume is determined by $B$, $H$, $k$, and $E_d$ — not by which experts are assigned to which device. All four strategies above send the same number of token embeddings across Ethernet links per step. Expert placement affects *which* device receives *which* tokens, but not the total bytes transferred.

### Load Balancing Reduces Imbalance Penalty

Unbalanced strategies (naive uniform under skewed routing) create **receive imbalance**: some devices receive far more tokens than others. In an MPI-style all-to-all, each device must send its maximum buffer size regardless of actual token count, so receiving devices that are over-loaded stall the combine step. Load-balanced and replication strategies reduce this imbalance and thus reduce effective stall time.

### Locality-Aware Placement and Latency

Locality-aware placement reduces the average hop count for co-activated expert pairs below the baseline 3.0 hops. On T3K's linear mesh with ~12.5 GB/s per Ethernet link, each saved hop reduces latency by approximately:

$$\Delta t \approx \frac{\text{message size}}{\text{12.5 GB/s}}$$

For a 6.4 MB dispatch volume at $B=32$: $\Delta t \approx 6.4 \text{ MB} / 12.5 \text{ GB/s} \approx 0.51$ ms per saved hop. Reducing average hop count from 3.0 to 2.0 (one hop saved) across all traffic saves approximately 0.51 ms per forward pass — meaningful at decode latency targets.

### `num_links` Selection

Expert placement strategy does not change the optimal `num_links` value directly. The `num_links` parameter is determined by payload size and link setup overhead as described in `ch03_all_to_all_num_links/num_links_parameter.md`. Use `num_links=1` for small decode payloads ($\lesssim 1$ MB) and `num_links=2` for larger payloads ($\gtrsim 10$ MB).

---

## References

- `ch01_t3k_topology/t3k_physical_layout.md` — T3K 1×8 mesh topology and hop-count derivation
- `ch01_t3k_topology/topology_implications_for_collectives.md` — Hop count impact on collective latency
- `ch03_all_to_all_num_links/all_to_all_in_moe.md` — Dispatch and combine volume calculations
- `ch03_all_to_all_num_links/num_links_parameter.md` — `num_links` tuning guidelines
- `ch04_memory_config/decode_memory_strategy.md` — Expert weight and buffer placement in DRAM/L1
- `expert_parallelism_strategies/ch04_expert_device_assignment/uniform_partitioning.md` — $W_{\text{expert}} = 6HD$ bytes definition (Section 2), load imbalance metric
- `expert_parallelism_strategies/ch04_expert_device_assignment/expert_replication.md` — Replication factor formula, dispatch metadata requirements
- [Fedus2022] Fedus, W., Zoph, B., Shazeer, N., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR, 2022.
- [Zoph2022] Zoph, B. et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models", arXiv:2202.08906, 2022.
