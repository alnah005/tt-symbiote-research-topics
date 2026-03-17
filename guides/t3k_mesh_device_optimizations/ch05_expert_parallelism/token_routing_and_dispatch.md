# Token Routing and Dispatch

> **Quick Reference — TTNN API Symbols Introduced**
>
> | Symbol | Description |
> |---|---|
> | `ttnn.all_to_all` | Routes token embeddings to destination devices based on routing indices; see `ch02_ttnn_mesh_api/collective_primitives.md` for full API reference |

This file covers the path from the expert router's output through all-to-all (A2A) dispatch to each device's expert compute. The key steps are: mapping expert indices to device IDs, constructing a padded send buffer, handling variable token counts with capacity padding, and estimating dispatch latency by batch size. All values use Qwen3.5-35B constants: $E = 256$, $k = 8$, $N = 8$, $E_d = 32$, $H = 7168$.

---

## 1. On-Device Expert Assignment

### Router Output

After the router forward pass on the originating device, each of the $B$ tokens has:

- **Routing indices:** a $[B, k]$ integer tensor giving the $k = 8$ selected expert indices (values in $\{0, \ldots, 255\}$).
- **Routing scores:** a $[B, k]$ float tensor giving the unnormalized weights for each selected expert (used during combine accumulation; see `combine_and_accumulation.md`).

### Expert Index to Device Mapping

Under the naive uniform placement (Section 1 of `expert_placement_strategies.md`), device assignment is:

```python
# Naive uniform: device d holds experts [32*d, 32*d + 31]
def expert_to_device_uniform(expert_idx: int, experts_per_device: int = 32) -> int:
    """Maps expert index to device ID under contiguous uniform placement."""
    return expert_idx // experts_per_device

# Example: expert 200 → device 200 // 32 = 6
assert expert_to_device_uniform(200) == 6
# Example: expert 0 → device 0
assert expert_to_device_uniform(0) == 0
# Example: expert 255 → device 255 // 32 = 7
assert expert_to_device_uniform(255) == 7
```

Under load-balanced or locality-aware placement, an explicit lookup table replaces the integer division:

```python
# Load-balanced: expert_to_device_table built at model init from profiled assignment
def expert_to_device_table_lookup(
    expert_idx: int,
    table: list,  # length 256; table[e] = device holding expert e
) -> int:
    return table[expert_idx]
```

### Replicated Expert Assignment

For replicated experts (see `expert_placement_strategies.md` Section 4), the dispatch logic must route each token to the correct replica device:

```python
def dispatch_device_with_replication(
    token_id: int,
    expert_idx: int,
    expert_to_replicas: dict,  # expert_idx -> list of replica device ids
    experts_per_device: int = 32,
) -> int:
    """
    Returns the device that should process token_id's request for expert_idx.
    Falls back to uniform placement for non-replicated experts.
    """
    if expert_idx in expert_to_replicas:
        replicas = expert_to_replicas[expert_idx]
        return replicas[hash(token_id) % len(replicas)]
    return expert_idx // experts_per_device
```

---

## 2. Constructing the Send Buffer

### Per-Device Sorting

After computing device IDs for all $B \times k = 32 \times 8 = 256$ (token, expert) pairs in a $B=32$ forward pass, tokens must be grouped by destination device. A single token may appear in up to $\min(k, N) = 8$ destination buffers if each of its $k=8$ experts resides on a different device.

The send buffer layout is a padded tensor of shape $[C \times E_d, H]$ per destination device, where:

$$C = \left\lceil \frac{k \times B \times \text{CF}}{E} \right\rceil = \left\lceil \frac{B \times 1.25}{32} \right\rceil$$

$C$ is the **expert capacity** — the maximum number of tokens any single expert is allocated to process, with CF = 1.25 providing a 25% headroom above expected load.

**Worked examples:**

- $B = 1$: $C = \lceil 1 \times 1.25 / 32 \rceil = \lceil 0.039 \rceil = 1$; send buffer shape $[1 \times 32, 7168] = [32, 7168]$ per destination device.
- $B = 32$: $C = \lceil 32 \times 1.25 / 32 \rceil = \lceil 1.25 \rceil = 2$; send buffer shape $[2 \times 32, 7168] = [64, 7168]$ per destination device.
- $B = 256$: $C = \lceil 256 \times 1.25 / 32 \rceil = \lceil 10 \rceil = 10$; send buffer shape $[10 \times 32, 7168] = [320, 7168]$ per destination device.

### Packing Token Embeddings

```python
import ttnn
import torch

def build_send_buffer(
    token_embeddings: torch.Tensor,   # [B, H]
    routing_indices: torch.Tensor,    # [B, k]
    expert_to_device: list,           # expert_idx -> device_id
    num_devices: int = 8,
    experts_per_device: int = 32,
    capacity_factor: float = 1.25,
) -> torch.Tensor:
    """
    Builds the all-to-all dispatch send buffer.

    Returns:
        send_buf: [num_devices, C * experts_per_device, H]
        Each send_buf[d] is the packed token embeddings destined for device d,
        with rows corresponding to capacity slots for each of device d's 32 experts.
    """
    B, H = token_embeddings.shape
    k = routing_indices.shape[1]
    C = int(math.ceil(B * capacity_factor / experts_per_device))

    send_buf = torch.zeros(num_devices, C * experts_per_device, H,
                           dtype=token_embeddings.dtype)
    # expert_slot_count[d][local_expert] tracks how many tokens are packed so far
    slot_count = [[0] * experts_per_device for _ in range(num_devices)]

    for b in range(B):
        for ki in range(k):
            e = routing_indices[b, ki].item()
            d = expert_to_device[e]
            local_e = e - d * experts_per_device  # local expert index within device d
            slot = slot_count[d][local_e]
            if slot < C:
                row = local_e * C + slot
                send_buf[d, row, :] = token_embeddings[b, :]
                slot_count[d][local_e] += 1
            # else: token dropped (overflow); see Section 3

    return send_buf
```

> **Tip:** In a production TTNN implementation, `build_send_buffer` is replaced by a fused dispatch kernel that writes directly from the router's top-k output to the send buffer without materializing an intermediate index tensor. See Section 5 for the fusion opportunity.

---

## 3. Variable Token Counts and Padding

### Under Non-Uniform Routing

With $C = 2$ at $B = 32$, each expert has 2 capacity slots. The expected number of tokens per expert is $B \times f_e \approx 32 \times (1/32) = 1$ under uniform routing — well within capacity. Under non-uniform routing, popular experts may receive more than $C$ tokens.

**Overflow handling options:**

1. **Token dropping:** Tokens beyond capacity $C$ are discarded; the token's contribution from this expert is treated as zero. Fast and simple; the capacity factor CF = 1.25 makes dropping rare (expected overflow probability is low when $f_e \lesssim 0.04$).
2. **Overflow routing:** Excess tokens are re-routed to a fallback expert or the next-best expert from the router's top-$k$ ranking. Requires an extra dispatch round; used when quality is paramount.

> **Warning:** Token dropping degrades model quality proportionally to the drop rate. Monitor overflow counts during inference. If token dropping exceeds ~1% of routing events, consider increasing CF or enabling expert replication for hot experts (see `expert_placement_strategies.md` Section 4).

### Padding with Zeros

If expert $e$ on device $d$ receives fewer than $C$ tokens in a given forward pass, the remaining capacity slots in the send buffer are padded with zeros. These padded rows are processed by the expert FFN but their outputs are masked out (zero-multiplied) during the combine step, contributing nothing to the final token representations.

Fixed-shape padding is required for `ttnn.all_to_all`, which operates on static tensor shapes. Dynamic shapes are not supported in the current TTNN collective implementation.

---

## 4. Latency Breakdown

### Router Compute

The router is a linear projection $[B, H] \times [H, E] = [B, 256]$ followed by top-$k$ selection:

- FLOP count: $2 \times B \times H \times E = 2 \times B \times 7168 \times 256 \approx 3.67 \times 10^6 \times B$ FLOPs.
- At $B = 32$: ~117 MFLOPs. On Wormhole B0 with 80 Tensix cores, sub-millisecond.

### Send Buffer Construction

Send buffer packing is $O(B \times k) = O(B \times 8)$ work. At $B = 32$: 256 (token, expert) pairs to pack — dominated by memory write bandwidth to the $[C \times E_d, H]$ buffer, not compute.

### All-to-All Dispatch Communication

Each device sends $N - 1 = 7$ packed buffers (one per remote device), each of shape $[C \times E_d, H]$:

$$V_{\text{per device}} = (N-1) \times C \times E_d \times H \times 2 \text{ bytes}$$

**At $B = 1$** ($C = 1$):

$$V = 7 \times 1 \times 32 \times 7168 \times 2 = 7 \times 458{,}752 = 3{,}211{,}264 \text{ bytes} \approx 3.2 \text{ MB}$$

Transfer time at 12.5 GB/s (single link, average-hop model): $3.2 \text{ MB} / 12.5 \text{ GB/s} \approx 0.26$ ms.

**At $B = 32$** ($C = 2$):

$$V = 7 \times 2 \times 32 \times 7168 \times 2 = 7 \times 917{,}504 = 6{,}422{,}528 \text{ bytes} \approx 6.4 \text{ MB}$$

Transfer time at 12.5 GB/s (single link): $6.4 \text{ MB} / 12.5 \text{ GB/s} \approx 0.51$ ms.

With `num_links=2` (see `ch03_all_to_all_num_links/num_links_parameter.md`): approximately halved — ~0.26 ms at $B=32$.

**Summary table:**

| Batch $B$ | $C$ | Volume per device | Time @ 1 link | Time @ 2 links |
|---|---|---|---|---|
| 1 | 1 | ~3.2 MB | ~0.26 ms | ~0.13 ms |
| 32 | 2 | ~6.4 MB | ~0.51 ms | ~0.26 ms |

> **Tip:** For decode at $B \le 32$, dispatch volumes ($\lesssim 6.4$ MB) are small enough that `num_links=1` is often sufficient. For prefill at large $S$, volumes grow to hundreds of MB and `num_links=2` or higher is recommended. See `ch03_all_to_all_num_links/num_links_parameter.md` for the full bandwidth model.

### Memory Placement of Send Buffers

For decode ($B \le 32$, $S = 1$): send buffer total size is $N \times C \times E_d \times H \times 2 = 8 \times 2 \times 32 \times 7168 \times 2 = 7.34$ MB across all 8 destination buffers. With HEIGHT_SHARDED layout across 80 Tensix cores, this is $\approx 91$ KB per core — within the 1.5 MB L1 budget per core (see `ch04_memory_config/decode_memory_strategy.md`).

For prefill ($B = 32$, $S = 2048$): $C = \lceil 8 \times 32 \times 2048 \times 1.25 / 256 \rceil = \lceil 2{,}560 \rceil = 2{,}560$; total send buffer $= 8 \times 2{,}560 \times 32 \times 7168 \times 2 \approx 9.4$ GB — requires DRAM placement (see `ch04_memory_config/prefill_memory_strategy.md`).

---

## 5. Fusing Router and Dispatch Preparation

### The Fusion Opportunity

The router produces two tensors: routing indices $[B, k]$ and routing scores $[B, k]$. Immediately after top-$k$ selection, the routing indices are used solely to populate the send buffer. This creates a fusion opportunity:

1. Top-$k$ selects expert indices.
2. Expert-to-device mapping converts indices to device IDs.
3. Packing writes token embeddings directly into send buffer slots.

Steps 1–3 can be fused into a single kernel that avoids materializing an intermediate $[B, k]$ index tensor in DRAM.

```python
# Conceptual fused router + dispatch pack kernel (pseudocode)
# In TTNN, this would be a custom Tensix kernel

def fused_router_dispatch(
    hidden_states: ttnn.Tensor,     # [B, H] on device
    router_weight: ttnn.Tensor,     # [H, E] on device
    send_buffer: ttnn.Tensor,       # [N, C * E_d, H] pre-allocated in L1
    expert_to_device: list,
    k: int = 8,
    capacity_factor: float = 1.25,
) -> tuple:
    """
    Fused: router matmul -> top-k -> expert-to-device map -> send buffer pack.
    Returns (routing_scores [B, k], send_buffer [N, C*E_d, H]).
    No intermediate routing_indices tensor is materialized in DRAM.
    """
    # 1. Router projection: [B, H] x [H, E] -> [B, E]
    logits = ttnn.matmul(hidden_states, router_weight)
    # 2. Top-k selection + direct buffer packing (fused in kernel)
    routing_scores = ttnn.topk_and_pack(
        logits, k=k,
        send_buffer=send_buffer,
        hidden_states=hidden_states,
        expert_to_device=expert_to_device,
        capacity_factor=capacity_factor,
    )
    return routing_scores, send_buffer
```

For router kernel fusion implementation details, see `expert_parallelism_strategies/ch05_routing_weight_optimization/router_forward_pass.md`.

### `ttnn.all_to_all` Call

After the send buffer is populated, dispatch is a single collective call:

```python
recv_buffer = ttnn.all_to_all(
    input_tensor=send_buffer,        # [N, C * E_d, H]
    mesh_device=mesh_device,
    topology=ttnn.Topology.Linear,   # T3K 1×8 linear mesh; see ch01_t3k_topology
    cluster_axis=1,
    num_links=num_links,             # 1 or 2; see ch03_all_to_all_num_links
    memory_config=ttnn.L1_MEMORY_CONFIG,  # decode: L1; prefill: DRAM_MEMORY_CONFIG
)
# recv_buffer[d]: token embeddings from device d destined for this device's experts
```

---

## References

- `ch01_t3k_topology/t3k_physical_layout.md` — T3K topology, device IDs, `ttnn.Topology.Linear`
- `ch02_ttnn_mesh_api/collective_primitives.md` — `ttnn.all_to_all` API reference
- `ch03_all_to_all_num_links/all_to_all_in_moe.md` — Dispatch volume derivation
- `ch03_all_to_all_num_links/num_links_parameter.md` — `num_links` selection by payload size
- `ch04_memory_config/decode_memory_strategy.md` — L1 placement budget for decode send buffers
- `ch04_memory_config/prefill_memory_strategy.md` — DRAM placement for prefill send buffers
- `expert_placement_strategies.md` (this chapter) — Expert-to-device mapping strategies
- `expert_parallelism_strategies/ch05_routing_weight_optimization/router_forward_pass.md` — Router kernel fusion details
- [Fedus2022] Fedus, W., Zoph, B., Shazeer, N., "Switch Transformers", JMLR, 2022.
- [Lepikhin2021] Lepikhin, D. et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding", ICLR, 2021.

---

**Next:** [combine_and_accumulation.md](./combine_and_accumulation.md)
