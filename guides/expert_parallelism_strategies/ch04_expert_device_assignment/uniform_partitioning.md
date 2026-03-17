# Uniform Partitioning: 32 Experts per Device

## Overview

Uniform partitioning is the baseline expert-to-device assignment strategy. It requires no profiling, no solver, and no per-deployment configuration: divide the $E = 256$ experts into $N = 8$ equal groups of $E_d = E/N = 32$ and assign group $d$ to device $d$. This file defines the assignment rule precisely, derives the memory footprint and expected token load per device, characterizes how load imbalance arises under non-uniform routing, and explains why TTNN's device addressing makes this the natural default.

**Prerequisite:** Chapter 1, `ch01_moe_fundamentals/`, and Chapter 2, `ch02_all_to_all_primitives/` in their entirety. Readers must be familiar with the MoE forward pass, the dispatch/combine communication pattern, and TTNN DRAM sharding.

---

## 1. Assignment Rule

Under round-robin assignment by expert index, device $d$ ($d \in \{0, 1, \ldots, 7\}$) holds the set of experts:

$$\mathcal{E}_d = \{e \in \{0, \ldots, 255\} \mid e \bmod N = d\} = \{d,\; d+8,\; d+16,\; \ldots,\; d+248\}$$

Each set $\mathcal{E}_d$ has exactly $|\mathcal{E}_d| = 32$ elements.

**Example mapping (first 8 experts of each device):**

| Device | Experts assigned (showing first 4 and last) |
|---|---|
| 0 | 0, 8, 16, 24, …, 248 |
| 1 | 1, 9, 17, 25, …, 249 |
| 2 | 2, 10, 18, 26, …, 250 |
| 3 | 3, 11, 19, 27, …, 251 |
| 4 | 4, 12, 20, 28, …, 252 |
| 5 | 5, 13, 21, 29, …, 253 |
| 6 | 6, 14, 22, 30, …, 254 |
| 7 | 7, 15, 23, 31, …, 255 |

The rule $\sigma(e) = e \bmod N$ is the direct correspondence between expert index and device index used by TTNN's all-to-all addressing scheme: when the dispatch kernel needs to route a token to expert $e$, it sends to device $e \bmod N$. No lookup table is required at runtime.

> **Tip:** The round-robin rule aligns with TTNN's internal device addressing, where the all-to-all collective indexes destination devices as $0$ through $N-1$. Expert $e$ goes to device $e \bmod N$, so the mapping is computable with a single modulo operation, avoiding any indirection in the dispatch metadata.

---

## 2. Memory Footprint per Device

Each expert is a two-layer feed-forward network (FFN) with the following weight tensors (using Qwen3.5-35B notation where $H = 7{,}168$ is the hidden dimension and $D$ is the expert FFN intermediate dimension [UNVERIFIED — see Chapter 1, `ch01_moe_fundamentals/qwen35b_config.md`]):

- Gate projection: $W_{\text{gate}} \in \mathbb{R}^{H \times D}$
- Up projection: $W_{\text{up}} \in \mathbb{R}^{H \times D}$
- Down projection: $W_{\text{down}} \in \mathbb{R}^{D \times H}$

Total weight bytes per expert (BF16, 2 bytes per element):

$$W_{\text{expert}} = (H \cdot D + H \cdot D + D \cdot H) \times 2 = 3 \cdot H \cdot D \cdot 2 \text{ bytes}$$

For Qwen3.5-35B with $H = 7{,}168$ and $D$ [UNVERIFIED]:

$$W_{\text{expert}} = 6 \cdot H \cdot D \text{ bytes (BF16)}$$

Under uniform partitioning, each device stores 32 expert weight tensors in DRAM:

$$W_{\text{device}} = 32 \times W_{\text{expert}} = 192 \cdot H \cdot D \text{ bytes (BF16)}$$

**Activation memory during the forward pass:** When device $d$ receives $T_d$ tokens routed to its local experts, it must allocate activation buffers of shape $[T_d, H]$ for inputs and $[T_d, D]$ for intermediate activations per expert. The peak activation footprint across all 32 local experts (if batched together) is:

$$A_{\text{device}} = T_d \times H \times 2 + T_d \times D \times 2 \text{ bytes (BF16)}$$

Under uniform routing, $T_d = k \cdot B / N = 8B/8 = B$ on average, so $A_{\text{device}} = B \times (H + D) \times 2$ bytes.

> **Warning:** Activation memory grows with batch size $B$ and can exceed DRAM capacity at large batch sizes if all 32 experts' activations are materialized simultaneously. In practice, experts are executed sequentially or in small groups, limiting peak activation memory to one expert's worth at a time. See Chapter 6, `ch06_fused_dispatch_compute_combine/expert_ffn_tiling.md` for the tiling strategy.

---

## 3. Expected Load Balance Under Uniform Token Distribution

Assume the routing is perfectly uniform: each expert $e$ has equal routing frequency $f_e = k/E = 8/256 = 1/32$.

In a forward pass with batch size $B$ tokens (and sequence length $S = 1$ for simplicity), each token selects $k = 8$ experts. The total number of (token, expert) routing events is $k \cdot B = 8B$.

Each of the $N = 8$ devices holds 32 experts. The expected number of routing events landing on device $d$ is:

$$\mathbb{E}[T_d] = \frac{32}{256} \times k \cdot B = \frac{1}{8} \times 8B = B$$

So each device receives exactly $B$ tokens on average — one token per original batch token, which matches the intuition that top-8 routing over 8 devices with 32 experts each distributes the load uniformly.

For Qwen3.5-35B with $k = N = 8$: the expected tokens per device equals the batch size $B$. This is the ideal case; deviations arise from routing non-uniformity.

---

## 4. Variance in Load Under Non-Uniform Routing

In practice, learned routers do not distribute load uniformly. Some experts are consistently preferred (high $f_e$), while others are rarely activated (low $f_e$). When high-$f_e$ experts cluster on the same device, that device becomes a bottleneck.

Let $f_e$ be the empirical routing frequency for expert $e$: the expected number of tokens (out of a batch of $B$) routed to expert $e$, normalized by $B$. Under top-$k$ routing, each token selects $k$ experts, so $\sum_{e=0}^{255} f_e = k = 8$. The average frequency is $f_{\text{avg}} = k/E = 8/256 = 1/32$.

Under this definition, the total expected token load on device $d$ is:

$$T_d = B \cdot \sum_{e \in \mathcal{E}_d} f_e$$

The mean load across devices is:

$$\bar{T} = \frac{B \cdot \sum_e f_e}{N} = \frac{B \cdot k}{N} = \frac{8B}{8} = B$$

This is consistent with the uniform-routing result in Section 3 above: under uniform routing $\sum_{e \in \mathcal{E}_d} f_e = E_d \times f_{\text{avg}} = 32 \times (1/32) = 1$, so $T_d = B \times 1 = B$.

Under non-uniform routing, the load on device $d$ is the same formula:

$$T_d = B \cdot \sum_{e \in \mathcal{E}_d} f_e$$

If the 32 experts on device $d$ collectively have above-average popularity — i.e., $\sum_{e \in \mathcal{E}_d} f_e > k/N = 1$ — then device $d$ receives more than $B$ tokens.

---

## 5. Load Imbalance Metric

The **load imbalance factor** $L$ quantifies how far the assignment deviates from ideal balance:

$$L = \frac{\max_{d} T_d}{\bar{T}} = \frac{\max_{d} \sum_{e \in \mathcal{E}_d} f_e}{\bar{\sum f}} = \frac{N \cdot \max_{d} \sum_{e \in \mathcal{E}_d} f_e}{k}$$

where $\bar{\sum f} = k/N = 1$ is the per-device average of $\sum_{e \in \mathcal{E}_d} f_e$.

A value of $L = 1.0$ indicates perfect balance. A value of $L = 2.0$ means the most-loaded device receives twice the average token count, and all other devices must wait for it to finish before the combine step can proceed.

**Worked example:** Suppose routing profiling over a calibration dataset reveals that experts $\{0, 8, 16, \ldots, 248\}$ (i.e., device 0's experts under round-robin) collectively receive 15% of all routing events, while the average is $100\%/8 = 12.5\%$. Then:

$$L = \frac{0.15}{0.125} = 1.2$$

A 20% imbalance translates directly to a 20% increase in forward-pass latency, since all-to-all combine cannot complete until the slowest device finishes its expert FFN computations.

> **Warning:** Under Zipf-like routing distributions (empirically observed in many MoE models), load imbalance can reach $L \approx 2$–$4$ with round-robin assignment, because popular experts are not evenly distributed across devices by index. Load-aware assignment (see `load_aware_assignment.md`) addresses this directly.

---

## 6. TTNN Sharding Implications

In TTNN, expert weight tensors are stored as **DRAM-interleaved shards** on each device. Under uniform partitioning with 32 local experts per device, the sharding layout is:

- Each expert's weight matrices ($W_{\text{gate}}$, $W_{\text{up}}$, $W_{\text{down}}$) are stored as separate DRAM tensors on the owning device.
- The 32 local expert weights occupy a contiguous region of device DRAM, enabling sequential or batched expert FFN execution without cross-device reads.
- During dispatch (see Chapter 2, `ch02_all_to_all_primitives/all_to_all_dispatch.md`), received token embeddings arrive in a packed buffer of shape $[T_d^{\text{recv}}, H]$, where $T_d^{\text{recv}}$ is the number of tokens dispatched to device $d$'s experts.

The dispatch buffer is padded to shape $[C \cdot E_d, H] = [C \cdot 32, H]$ where $C$ is the per-expert capacity, to allow fixed-shape all-to-all collectives. Under uniform partitioning, this padding is the same on all devices, simplifying TTNN tensor shape management.

**Core assignment for expert FFN:** On each Wormhole B0 device, Tensix cores are arranged in an $8 \times 10 = 80$-core grid with 1.5 MB L1 per core. With 32 local experts, a natural tiling assigns 2–4 experts per core group, with expert weight tiles streamed from DRAM to L1 as needed. The precise tiling is covered in Chapter 6, `ch06_fused_dispatch_compute_combine/expert_ffn_tiling.md`.

---

## 7. Why Round-Robin by Expert Index Is the Natural Default

TTNN's `all_to_all_dispatch` operation routes token embeddings to destination devices according to the routing index tensor produced by the router. Specifically, for a token assigned to expert $e$, the destination device is computed as:

```python
destination_device = expert_index % num_devices  # = e % N
```

This is the round-robin rule. It requires no external lookup table and no runtime overhead beyond a single modulo operation. The dispatch kernel encodes the destination directly from the expert index.

Under any other assignment scheme (e.g., load-aware), the dispatch operation must carry an explicit expert-to-device mapping table, adding metadata overhead and a register-file indirection per token. Round-robin eliminates this overhead entirely.

```python
# Round-robin assignment: O(1) device lookup at dispatch time
def expert_to_device_roundrobin(expert_idx: int, num_devices: int) -> int:
    """
    Returns the device index that owns expert_idx under round-robin assignment.
    Device d holds all experts e where e % num_devices == d.
    """
    return expert_idx % num_devices


# Example: verify the full assignment for E=256, N=8
num_experts = 256
num_devices = 8

# Build the assignment: device_of[e] = device that owns expert e
device_of = [expert_to_device_roundrobin(e, num_devices) for e in range(num_experts)]

# Verify: each device owns exactly 32 experts
from collections import Counter
counts = Counter(device_of)
assert all(counts[d] == 32 for d in range(num_devices)), "Imbalanced assignment"

# Print experts on device 0
device_0_experts = [e for e in range(num_experts) if device_of[e] == 0]
print(f"Device 0 experts: {device_0_experts[:5]}...{device_0_experts[-1]}")
# Output: Device 0 experts: [0, 8, 16, 24, 32]...248
```

> **Tip:** When prototyping a new assignment strategy, always implement both the forward mapping (expert → device) and the reverse mapping (device → list of experts). TTNN's dispatch metadata requires the forward map; the expert FFN kernel scheduler requires the reverse map to know which weight tensors to load.

---

## 8. Load Imbalance Under Typical Qwen3.5-35B Routing

Empirical studies of MoE routing distributions (see [Zoph2022], [Clark2022]) report that routing follows an approximately Zipf-like distribution: the most popular expert receives roughly $2$–$5\times$ the average token count. For Qwen3.5-35B with 256 experts and top-8 routing, the average per-expert frequency is:

$$f_{\text{avg}} = k / E = 8 / 256 = 0.03125$$

If the top-8 most popular experts each receive $4 \times f_{\text{avg}} = 0.125$ of all tokens, and these 8 experts are spread across 8 different devices (as they would be under round-robin, since consecutive indices map to consecutive devices), the load on each of those 8 devices increases by the difference:

$$\Delta T_d = (0.125 - 0.03125) \times B = 0.09375B$$

Adding this to the base load of $B$: $T_d^{\text{hot}} \approx B + 0.09375B = 1.09B$, giving $L \approx 1.09$.

However, if multiple hot experts happen to share the same device (as can occur by chance under round-robin), imbalance is worse. Load-aware assignment (`load_aware_assignment.md`) eliminates this chance dependency.

---

## References

- [Ch1Routing] Chapter 1, `ch01_moe_fundamentals/routing_problem.md` — dispatch/combine pattern, load imbalance concept, expert capacity.
- [Ch1Config] Chapter 1, `ch01_moe_fundamentals/qwen35b_config.md` — Qwen3.5-35B architectural constants ($H$, $D$, $E$, $k$).
- [Ch2Dispatch] Chapter 2, `ch02_all_to_all_primitives/all_to_all_dispatch.md` — dispatch buffer layout, send-count metadata, device addressing.
- [Ch3Sharding] Chapter 3, `ch03_alternative_routing_schemes/expert_sharding.md` — expert sharding vs. all-to-all volume comparison.
- [Ch6Tiling] Chapter 6, `ch06_fused_dispatch_compute_combine/expert_ffn_tiling.md` — Tensix core tiling for 32 local experts.
- [Fedus2022] Fedus, W., Zoph, B., Shazeer, N., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR, 2022.
- [Zoph2022] Zoph, B. et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models", arXiv:2202.08906, 2022.
- [Clark2022] Clark, A. et al., "Unified Scaling Laws for Routed Language Models", ICML, 2022.
- [Qwen3] Qwen Team, Alibaba Group, "Qwen3 Technical Report", 2025.

---

**Next:** [load_aware_assignment.md](./load_aware_assignment.md)
