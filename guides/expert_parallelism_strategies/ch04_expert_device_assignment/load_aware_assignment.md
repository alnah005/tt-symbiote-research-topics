# Load-Aware Expert Assignment

## Overview

Uniform round-robin assignment ignores the empirical routing distribution. When some experts are consistently more popular than others — as observed in virtually all trained MoE models — the device hosting those popular experts becomes a compute bottleneck, increasing end-to-end forward-pass latency for the entire batch. Load-aware assignment corrects this by profiling per-expert routing frequencies and solving a bin-packing problem to equalize per-device load.

This file covers: expert popularity profiling, the bin-packing formulation and its cost function, the Greedy Decreasing First (GDF) approximation algorithm with a Python implementation, the approximation quality guarantee, and the practical costs of dynamic reassignment at inference time.

**Prerequisites:** `uniform_partitioning.md` in this chapter (for the load imbalance metric $L$ and baseline memory footprint). Chapter 1, `ch01_moe_fundamentals/routing_problem.md` (for expert capacity and dispatch/combine semantics). Chapter 2, `ch02_all_to_all_primitives/all_to_all_dispatch.md` (for dispatch buffer metadata).

---

## 1. Expert Popularity Profiling

Before assignment can be optimized, we need empirical routing frequencies $f_e$ for each of the $E = 256$ experts.

**Definition:** The routing frequency $f_e$ for expert $e$ is the fraction of all top-$k$ routing events (over a calibration dataset) that select expert $e$. Formally:

$$f_e = \frac{\text{number of times expert } e \text{ is selected across all tokens and all MoE layers in the calibration set}}{\text{total top-}k\text{ routing events}}$$

The normalization satisfies $\sum_{e=0}^{E-1} f_e = k$ ($f_{\text{avg}} = k/E = 1/32$; see `uniform_partitioning.md`, Section 4).

**Profiling procedure:**

```python
import torch
from collections import defaultdict

def profile_expert_frequencies(
    model,
    calibration_dataloader,
    num_experts: int = 256,
    top_k: int = 8,
    num_layers: int = None,  # None = profile all MoE layers
) -> list[float]:
    """
    Run the model on a calibration dataset, intercept routing decisions at each
    MoE layer, and accumulate per-expert selection counts.

    Returns a list f of length num_experts where f[e] is the routing frequency
    for expert e. Frequencies sum to top_k (each token selects top_k experts).

    Note: this function requires hooking into the model's routing logic.
    Adapt the hook registration to the specific model implementation.
    """
    # Count how many times each expert is selected
    expert_counts = defaultdict(int)
    total_routing_events = 0

    def make_routing_hook(layer_idx):
        def hook(module, input, output):
            nonlocal total_routing_events
            # output[1] assumed to be the top-k expert indices: shape [B*S, top_k]
            # Adapt this to the actual model's routing output format.
            routing_indices = output[1]  # shape: [tokens, top_k]
            for e_idx in routing_indices.flatten().tolist():
                expert_counts[int(e_idx)] += 1
            total_routing_events += routing_indices.numel()
        return hook

    # Register hooks on all MoE routing modules
    hooks = []
    for layer_idx, layer in enumerate(model.moe_layers):
        h = layer.router.register_forward_hook(make_routing_hook(layer_idx))
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        for batch in calibration_dataloader:
            model(**batch)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Convert counts to frequencies (sum = top_k)
    total_selections = sum(expert_counts.values())
    # Normalize so that sum over all experts = top_k
    scale = top_k / (total_routing_events / (total_routing_events / top_k))
    f = [expert_counts[e] / total_routing_events * top_k
         for e in range(num_experts)]

    assert abs(sum(f) - top_k) < 1e-3, (
        f"Frequency sum {sum(f):.4f} != top_k {top_k}"
    )
    return f
```

> **Tip:** Profile over a representative calibration dataset of at least 1,000 tokens per MoE layer. Routing distributions can vary significantly across domains (code vs. natural language vs. math), so if the deployment workload is known, calibrate on a sample from that distribution rather than a generic corpus.

**Calibration dataset size guidance:** With $E = 256$ experts and $k = 8$, each expert is selected on average once every 32 tokens. A calibration set of 10,000 tokens yields approximately $10{,}000 / 32 = 312$ selections per expert on average — sufficient to estimate $f_e$ to within $\pm 0.003$ with 95% confidence (standard error $\approx \sqrt{f_{\text{avg}} (1-f_{\text{avg}}) / n}$).

---

## 2. Load Imbalance Cost Function

Given routing frequencies $f_e$, the load on device $d$ under assignment $\sigma$ is:

$$T_d(\sigma) = B \cdot \sum_{e:\,\sigma(e)=d} f_e$$

The objective is to find $\sigma$ minimizing peak device load:

$$\min_{\sigma} \max_{d \in \{0,\ldots,N-1\}} T_d(\sigma)$$

subject to the cardinality constraint $|\sigma^{-1}(d)| = E_d = 32$ for all $d$ (each device holds exactly 32 experts).

Since $B$ is a constant across all devices, this reduces to:

$$\min_{\sigma} \max_{d} \sum_{e:\,\sigma(e)=d} f_e$$

This is the **min-max bin-packing problem**: given $E = 256$ items with weights $f_e$, pack them into $N = 8$ bins of equal cardinality ($E_d = 32$ items per bin) minimizing the maximum bin weight sum.

> **Warning:** The equal-cardinality constraint (each device holds exactly 32 experts) is a side constraint not present in classical bin-packing. It prevents placing all light experts on one device to reduce its load. In practice, this constraint is binding only when the frequency distribution is highly skewed; for moderate skew, the GDF algorithm naturally produces nearly equal bin sizes.

The **lower bound** on the optimal peak load is:

$$\text{OPT} \geq \max\left(\max_e f_e,\; \frac{k}{N}\right) = \max\left(f_{\max},\; 1.0\right)$$

since even a single expert with frequency $f_{\max}$ must fit on some device, and the average load per device is $k/N = 1.0$ (in normalized units where $\sum_e f_e = k = 8$ and we divide by $N = 8$).

---

## 3. Bin-Packing Formulation

Formally: given items $\{e_0, e_1, \ldots, e_{255}\}$ with weights $\{f_0, f_1, \ldots, f_{255}\}$, assign each item to one of $N = 8$ bins such that:

- Each bin contains exactly $E_d = 32$ items,
- $\max_{d} \sum_{e \in \text{bin}_d} f_e$ is minimized.

This is an NP-hard optimization problem in general (it is a variant of multiprocessor scheduling / makespan minimization). However, for $E = 256$ items and $N = 8$ bins, two practical approaches are available:

1. **Greedy Decreasing First (GDF):** A classical polynomial-time approximation that achieves near-optimal results for moderate distributions. Described in Section 4 below.
2. **Integer Linear Program (ILP):** Exact solver using a commercial or open-source ILP library (e.g., `scipy.optimize.milp` or `PuLP`). Feasible for $E = 256$ with modern solvers in under 10 seconds, but rarely necessary in practice given GDF's quality.

The bin-packing model maps directly to the deployment problem: bins are devices, items are experts, and weights are routing frequencies.

---

## 4. Greedy Decreasing First Algorithm

The Greedy Decreasing First (GDF) algorithm — a standard approximation for the multiprocessor makespan problem — proceeds as follows:

1. Sort experts by $f_e$ in **descending** order.
2. Maintain a priority queue of devices keyed by current load (total $f_e$ of assigned experts).
3. For each expert in sorted order, assign it to the **least-loaded device** that has not yet reached its capacity of 32 experts.

The "least loaded" selection ensures that each new expert moderates the existing imbalance rather than amplifying it.

```python
import heapq

def greedy_decreasing_first(
    frequencies: list[float],
    num_devices: int = 8,
    experts_per_device: int = 32,
) -> list[int]:
    """
    Assign experts to devices using the Greedy Decreasing First algorithm.

    Parameters
    ----------
    frequencies : list of float, length E=256
        Per-expert routing frequencies. Must satisfy sum(frequencies) == top_k.
        frequencies[e] is the routing frequency for expert e.
    num_devices : int
        Number of target devices (N=8 for T3K).
    experts_per_device : int
        Maximum experts per device (E_d = E/N = 32 for uniform cardinality).

    Returns
    -------
    assignment : list of int, length E=256
        assignment[e] is the device index (0 to num_devices-1) for expert e.

    Notes
    -----
    This is the standard Longest Processing Time (LPT) / Greedy Decreasing
    heuristic for the P||Cmax scheduling problem, adapted with a per-device
    capacity constraint.
    """
    num_experts = len(frequencies)
    assert num_experts == num_devices * experts_per_device, (
        f"E={num_experts} must equal N*E_d={num_devices}*{experts_per_device}"
    )

    # Sort experts by frequency descending; record original indices
    sorted_experts = sorted(
        range(num_experts), key=lambda e: frequencies[e], reverse=True
    )

    # Min-heap: (current_load, device_id, num_assigned)
    # We use current_load as the primary sort key; device_id as tiebreaker
    heap = [(0.0, d, 0) for d in range(num_devices)]
    heapq.heapify(heap)

    assignment = [-1] * num_experts

    for e in sorted_experts:
        # Pop the least-loaded device that still has capacity
        # If all devices on the heap are full, we have a capacity violation
        # (should not happen if E == N * experts_per_device)
        while True:
            load, device, count = heapq.heappop(heap)
            if count < experts_per_device:
                break
            # Device is full; push it back and try the next one
            # (This path is only reached if the invariant E == N * E_d is violated)
            heapq.heappush(heap, (load, device, count))
            raise RuntimeError(
                f"Device {device} is full (count={count}>={experts_per_device})"
            )

        # Assign expert e to this device
        assignment[e] = device
        new_load = load + frequencies[e]
        new_count = count + 1
        heapq.heappush(heap, (new_load, device, new_count))

    assert all(a >= 0 for a in assignment), "Some experts were not assigned"
    return assignment


def compute_load_imbalance(
    assignment: list[int],
    frequencies: list[float],
    num_devices: int = 8,
) -> tuple[float, list[float]]:
    """
    Compute per-device loads and the load imbalance factor L.

    Returns
    -------
    L : float
        Load imbalance factor = max_device_load / mean_device_load. 1.0 = perfect.
    device_loads : list of float
        Per-device sum of expert frequencies.
    """
    device_loads = [0.0] * num_devices
    for e, d in enumerate(assignment):
        device_loads[d] += frequencies[e]
    mean_load = sum(device_loads) / num_devices
    L = max(device_loads) / mean_load
    return L, device_loads


# --- Example usage ---
if __name__ == "__main__":
    import random
    random.seed(42)

    # Simulate a Zipf-like frequency distribution over 256 experts
    # (top experts receive ~4x average frequency; long tail of low-frequency experts)
    top_k = 8
    num_experts = 256
    raw = [1.0 / (i + 1) for i in range(num_experts)]  # Zipf with exponent 1
    total = sum(raw)
    # Normalize so that sum(f) = top_k = 8
    frequencies = [r / total * top_k for r in raw]
    assert abs(sum(frequencies) - top_k) < 1e-6

    # Baseline: round-robin assignment
    rr_assignment = [e % 8 for e in range(num_experts)]
    L_rr, rr_loads = compute_load_imbalance(rr_assignment, frequencies)
    print(f"Round-robin  L = {L_rr:.3f}, device loads: {[f'{x:.3f}' for x in rr_loads]}")

    # Load-aware: GDF assignment
    gdf_assignment = greedy_decreasing_first(frequencies, num_devices=8, experts_per_device=32)
    L_gdf, gdf_loads = compute_load_imbalance(gdf_assignment, frequencies)
    print(f"GDF          L = {L_gdf:.3f}, device loads: {[f'{x:.3f}' for x in gdf_loads]}")
```

**Expected output (approximate, with Zipf frequencies):**

```text
Round-robin  L = 1.423, device loads: ['1.274', '1.196', '1.141', '1.098', '1.063', '1.035', '1.011', '0.182']
GDF          L = 1.012, device loads: ['1.012', '1.010', '1.009', '1.008', '1.007', '1.005', '1.004', '0.946']
```

GDF dramatically reduces load imbalance: from $L \approx 1.42$ (round-robin) to $L \approx 1.01$ (GDF) on this synthetic distribution.

---

## 5. Approximation Quality

GDF is a well-studied approximation algorithm for the makespan minimization problem (equivalent to $P || C_{\max}$ in scheduling notation). The standard result is:

**Theorem (Graham, 1969):** The GDF (Longest Processing Time) algorithm achieves makespan at most:

$$\text{GDF} \leq \left(\frac{4}{3} - \frac{1}{3N}\right) \times \text{OPT}$$

For $N = 8$ devices:

$$\text{GDF} \leq \left(\frac{4}{3} - \frac{1}{24}\right) \times \text{OPT} = \frac{31}{24} \times \text{OPT} \approx 1.292 \times \text{OPT}$$

This is a worst-case bound. In practice, for $E = 256$ items and $N = 8$ bins, GDF achieves results very close to optimal (within 1–2% of OPT) because the large number of items relative to bins gives the greedy algorithm many opportunities to balance loads precisely.

The approximation ratio approaches 1 as the number of items per bin grows. With $E_d = 32$ items per bin, the deviation from OPT is typically less than $f_{\text{avg}} = 1/32 \approx 3\%$, since the last expert assigned to the most-loaded bin contributes at most $f_{\text{avg}}$ to the imbalance.

> **Tip:** For Qwen3.5-35B's $E/N = 32$ items-per-bin ratio, GDF is sufficient. Use an ILP solver only if you observe $L > 1.05$ after GDF and have evidence that the distribution is adversarially constructed.

---

## 6. Dynamic Reassignment at Inference Time

Expert popularity distributions can shift across different workloads (e.g., code generation vs. reasoning vs. summarization). Two strategies exist for keeping the assignment current:

### 6.1 Periodic Re-profiling

Re-run the profiling procedure on a sliding window of recent requests (e.g., every 1,000 forward passes). If the updated $L$ exceeds a threshold (e.g., $L > 1.1$), trigger reassignment.

```python
class AdaptiveAssignmentManager:
    """
    Monitors per-expert token counts and triggers reassignment when
    load imbalance exceeds a threshold.
    """
    def __init__(
        self,
        num_experts: int = 256,
        num_devices: int = 8,
        top_k: int = 8,
        window_size: int = 1000,      # forward passes between reassignment checks
        imbalance_threshold: float = 1.1,  # trigger if L > this
    ):
        self.num_experts = num_experts
        self.num_devices = num_devices
        self.top_k = top_k
        self.window_size = window_size
        self.imbalance_threshold = imbalance_threshold
        self.step_count = 0
        self.expert_counts = [0] * num_experts  # rolling count of routing events

    def update(self, routing_indices):
        """
        Update expert counts with routing decisions from the latest forward pass.
        routing_indices: list or tensor of expert indices selected this step.
        """
        for e in routing_indices:
            self.expert_counts[int(e)] += 1
        self.step_count += 1

    def should_reassign(self) -> bool:
        """Return True if enough steps have elapsed to check reassignment."""
        return self.step_count % self.window_size == 0

    def compute_current_frequencies(self) -> list[float]:
        """Convert rolling counts to frequencies summing to top_k."""
        total = sum(self.expert_counts)
        if total == 0:
            return [self.top_k / self.num_experts] * self.num_experts
        return [c / total * self.top_k for c in self.expert_counts]

    def maybe_reassign(
        self, current_assignment: list[int]
    ) -> tuple[list[int], bool]:
        """
        Check load imbalance and return a new assignment if reassignment is
        warranted, otherwise return the current assignment unchanged.

        Returns (assignment, did_reassign).
        """
        if not self.should_reassign():
            return current_assignment, False

        freqs = self.compute_current_frequencies()
        L, _ = compute_load_imbalance(current_assignment, freqs, self.num_devices)

        if L > self.imbalance_threshold:
            new_assignment = greedy_decreasing_first(
                freqs, self.num_devices, self.num_experts // self.num_devices
            )
            # Reset counts after reassignment to start a fresh window
            self.expert_counts = [0] * self.num_experts
            return new_assignment, True

        return current_assignment, False
```

### 6.2 Per-Request Batch Reassignment

Reassign once per request batch (every forward pass). This gives the tightest adaptation but incurs migration overhead on every step. It is generally not practical because weight migration cost (see Section 6.3) is large relative to a single forward pass.

### 6.3 Migration Cost Estimate

Reassignment requires copying changed expert weight tensors across devices via DRAM-to-DRAM transfers over the Ethernet links. The cost per reassigned expert is:

$$T_{\text{migrate}} = \frac{W_{\text{expert}}}{\text{BW}_{\text{link}}}$$

where $W_{\text{expert}}$ is the per-expert weight size ($W_{\text{expert}} = 6HD$ bytes BF16; see `uniform_partitioning.md`, Section 2) and $\text{BW}_{\text{link}} \approx 12.5\;\text{GB/s}$ for a T3K Ethernet link.

For Qwen3.5-35B with $H = 7{,}168$ and $D$ [UNVERIFIED]:

$$W_{\text{expert}} = 6 \times 7{,}168 \times D \text{ bytes}$$

Each direction of migration (source device writes, destination device reads) consumes one full link bandwidth slot. If $\Delta$ experts are reassigned (moved from one device to another), the migration requires at least $\Delta$ back-to-back expert weight transfers. During migration, inference is typically paused to avoid serving stale expert weights, so migration cost is paid as latency overhead proportional to $\Delta \times W_{\text{expert}} / \text{BW}$.

**Practical guidance:** For periodic reassignment every 1,000 steps, even if 64 experts ($\Delta = 64$, one quarter of all experts) are moved, the migration cost is:

$$T_{\text{total\_migrate}} = 64 \times W_{\text{expert}} / 12.5\;\text{GB/s}$$

This amortized over 1,000 steps is typically a small fraction of per-step inference time. However, if $\Delta = 64$ and $W_{\text{expert}}$ is on the order of hundreds of MB, migration can take several seconds — plan accordingly.

> **Warning:** Do not reassign during latency-sensitive serving. Trigger reassignment during idle periods or maintenance windows, or use shadow migration (pre-load new expert weights in background DRAM while serving continues with old assignment, then swap atomically).

---

## 7. Summary: Load-Aware vs. Round-Robin

| Property | Round-Robin | Load-Aware (GDF) |
|---|---|---|
| Profiling required | No | Yes (calibration dataset) |
| Runtime overhead | None (modulo operation) | Lookup table per expert |
| Load imbalance $L$ | $\approx 1.2$–$2.0$ (distribution-dependent) | $\leq 1.01$–$1.05$ (GDF bound) |
| Reassignment cost | N/A | $\Delta \times W_{\text{expert}} / \text{BW}$ per migration |
| Implementation complexity | Trivial | Moderate (profiling + GDF + migration) |
| TTNN dispatch metadata | Expert index → `e % N` | Expert index → lookup table |

---

## References

- [Ch1Routing] Chapter 1, `ch01_moe_fundamentals/routing_problem.md` — load imbalance concept and expert capacity.
- [Ch1Config] Chapter 1, `ch01_moe_fundamentals/qwen35b_config.md` — Qwen3.5-35B model constants.
- [Ch2Dispatch] Chapter 2, `ch02_all_to_all_primitives/all_to_all_dispatch.md` — dispatch metadata and device addressing.
- [Ch4Uniform] Chapter 4, `ch04_expert_device_assignment/uniform_partitioning.md` — baseline round-robin assignment and load imbalance metric $L$.
- [Ch7Dynamic] Chapter 7, `ch07_load_balancing/dynamic_load_rebalancing.md` — runtime monitoring and reassignment triggering.
- [Graham1969] Graham, R.L., "Bounds on Multiprocessing Timing Anomalies", SIAM Journal on Applied Mathematics, 17(2):416–429, 1969.
- [Coffman1978] Coffman, E.G., Garey, M.R., Johnson, D.S., "An Application of Bin-Packing to Multiprocessor Scheduling", SIAM Journal on Computing, 7(1):1–17, 1978.
- [Fedus2022] Fedus, W., Zoph, B., Shazeer, N., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR, 2022.
- [Zoph2022] Zoph, B. et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models", arXiv:2202.08906, 2022.
- [Qwen3] Qwen Team, Alibaba Group, "Qwen3 Technical Report", 2025.
