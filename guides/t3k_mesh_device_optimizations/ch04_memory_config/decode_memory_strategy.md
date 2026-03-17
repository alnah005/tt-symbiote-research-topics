# Memory Placement Strategy: Decode Phase

## Overview

The decode phase generates one new token per forward pass, processing a batch of $B$ sequences each at position $S_{\text{current}} \in [1, S_{\text{max}}]$. The defining characteristics of decode are:

- **Small active tensors**: only one new token per sequence, so activation shapes are $[B, 1, H]$ or equivalently $[B, H]$ after squeezing.
- **Large persistent state**: the KV cache holds all previously computed key and value vectors for every sequence in the batch, growing with generation length.
- **Latency-critical**: decode step latency directly determines tokens-per-second. Even modest memory access improvements compound over thousands of generated tokens.
- **Memory-bound, not compute-bound**: at small batch sizes (B ≤ 32), decode ops are typically memory-bandwidth limited. Reducing DRAM round-trips improves step latency more than increasing compute throughput.

These characteristics drive a clear placement principle: **keep active tensors in L1, keep persistent state in DRAM**.

---

## Tensor-by-Tensor Analysis

### KV Cache — DRAM

**Recommendation: `ttnn.DRAM_MEMORY_CONFIG`**

The KV cache holds all previous key and value projections for the current generation. For a model with $n_{\text{layers}}$ layers, $n_{KV}$ KV heads each of dimension $H_{KV}$, and a maximum context length $S_{\text{max}}$:

$$\text{KV cache size per device} = n_{\text{layers}} \times 2 \times B \times S_{\text{max}} \times n_{KV} \times H_{KV} \times 2\,\text{bytes}$$

Even at small batch sizes and moderate context lengths, this exceeds the 120 MB aggregate L1 of one Wormhole chip by a significant margin. For example, at $n_{\text{layers}}=64$, $n_{KV}=8$, $H_{KV}=128$, $B=32$, $S_{\text{max}}=4096$:

$$64 \times 2 \times 32 \times 4096 \times 8 \times 128 \times 2 = 34.4\,\text{GB}$$

This is not even close to fitting in L1. Place the KV cache in DRAM and access it with DRAM interleaved layout to maximize read bandwidth.

```python
import ttnn

# KV cache initialization — DRAM, persistent across decode steps
kv_cache = ttnn.from_torch(
    kv_cache_torch,
    device=mesh_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,  # large, persistent — DRAM only
    dtype=ttnn.bfloat16,
)
```

> **Warning:** Never attempt to place the KV cache in L1, even at small batch sizes. The cache grows with generation length. An allocation that fits at step 1 will fail at step 512 when the sequence has grown.

---

### Current-Token Hidden States and Query Projections — L1

**Recommendation: `ttnn.L1_MEMORY_CONFIG`**

During decode, the active hidden state has shape $[B, 1, H]$ which is treated as $[B, H]$ for matrix operations. At the largest practical decode batch size $B=32$ and $H=7168$ (Qwen3.5-35B):

$$\text{Total bytes} = B \times H \times 2 = 32 \times 7168 \times 2 = 458{,}752\,\text{bytes} \approx 448\,\text{KB}$$

When sharded across 80 Tensix cores using HEIGHT_SHARDED, the 32 rows are distributed as $\lceil 32/80 \rceil = 1$ row per core:

$$\text{Per-core bytes} = 1 \times H \times 2 = 1 \times 7168 \times 2 = 14{,}336\,\text{bytes} = 14.0\,\text{KB/core}$$

This is a negligible fraction of the 1.5 MB per-core L1 budget. The query projection $Q = X W_Q$ has the same shape after projection (assuming no head dimension expansion), so it also fits comfortably.

```python
import ttnn

# Active hidden state for current token — L1 placement
# Shape: [B, H] = [32, 7168] for decode
hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)

# Query projection: output also stays in L1
q_proj = ttnn.matmul(
    hidden_states,
    wq,  # weight matrix, can stay in DRAM
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
# q_proj shape: [32, H_Q]; still small, still in L1
```

> **Tip:** Keep the hidden state in L1 across all ops in the decode step (attention, FFN, residual add, layernorm). The tensor is small enough that no op will hit L1 pressure from it alone. Write back to DRAM only once at the end of the step.

---

### Expert Routing Scores and Top-k Indices — L1

**Recommendation: `ttnn.L1_MEMORY_CONFIG`**

The router computes a score matrix of shape $[B, E]$ where $E=256$ (total experts). For $B=32$:

$$\text{Total bytes} = B \times E \times 2 = 32 \times 256 \times 2 = 16{,}384\,\text{bytes} = 16\,\text{KB}$$

This is trivially small. The top-k indices have shape $[B, k]$ = $[32, 8]$:

$$32 \times 8 \times 4\,\text{bytes (int32)} = 1{,}024\,\text{bytes} = 1\,\text{KB}$$

Both tensors should live in L1. Routing in decode is a small matmul ($[B, H] \times [H, E]$) plus softmax. Keeping the output in L1 eliminates a DRAM write between the router and the dispatch step.

```python
import ttnn

# Router forward — all outputs in L1
router_logits = ttnn.matmul(
    hidden_states,      # L1, shape [B, H]
    router_weight,      # DRAM weight matrix [H, E]
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
# router_logits: [B, E] = [32, 256] = 16 KB — fits in L1

routing_scores = ttnn.softmax(
    router_logits,
    dim=-1,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
# routing_scores: [B, E] = 16 KB — fits in L1

# top_k_indices: [B, k] = [32, 8] = 1 KB — fits in L1
top_k_indices = ttnn.topk(
    routing_scores,
    k=8,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
```

---

### All-to-All Dispatch and Combine Buffers — L1 (Decode)

**Recommendation: `ttnn.L1_MEMORY_CONFIG` when $B \leq 32$**

The all-to-all dispatch operation sends token activations from each device to the devices hosting the selected experts. The volume per device per direction is:

$$V_{\text{dispatch}} = C \times E_d \times H \times 2\,\text{bytes}$$

where $C = \lceil k B S / E \rceil$ is the expert capacity, $E_d = E / N = 256 / 8 = 32$ experts per device, and $S=1$ for decode.

At decode with $B=32$, $S=1$:

$$C = \left\lceil \frac{8 \times 32 \times 1}{256} \right\rceil = \left\lceil 1.0 \right\rceil = 1$$

$$V_{\text{dispatch}} = 1 \times 32 \times 7168 \times 2 = 458{,}752\,\text{bytes} \approx 448\,\text{KB}$$

When this buffer is sharded HEIGHT_SHARDED across 80 cores, the 32 rows distribute as $\lceil 32/80 \rceil = 1$ row per core:

$$\text{Per-core} = 1 \times H \times 2 = 1 \times 7168 \times 2 = 14{,}336\,\text{bytes} = 14.0\,\text{KB/core}$$

This fits comfortably in L1. The combine buffer (after expert computation, routing results back) has the same shape and footprint.

```python
import ttnn

# All-to-all dispatch for decode (B=32, S=1, C=1)
# dispatch_input shape: [C, E_d, H] = [1, 32, 7168]
# Total: 458,752 bytes = 448 KB — fits in L1
dispatch_output = ttnn.all_to_all(
    dispatch_input,
    cluster_axis=1,
    memory_config=ttnn.L1_MEMORY_CONFIG,  # use L1 for decode
)

# Expert FFN computation on each device — stays in L1 for decode
expert_output = expert_ffn(
    dispatch_output,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)

# All-to-all combine — return results to originating devices
combined_output = ttnn.all_to_all(
    expert_output,
    cluster_axis=1,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
```

> **Warning:** This L1 placement assumes $C=1$ (single token per expert slot at decode with $B=32$, $S=1$). If you use token dropping disabled (expert capacity > 1, $k>8$, or $E<256$), recalculate $C$ and verify the buffer still fits before using L1.

---

## Worked Example: Per-Core L1 Budget at B=32

The following estimates the per-core L1 occupancy during a decode step at $B=32$, $H=7168$, $E=256$, $k=8$, $E_d=32$ on one Wormhole B0 chip (80 cores).

All tensors below are sharded HEIGHT_SHARDED across all 80 cores unless noted.

```python
import math

# Parameters
B = 32
H = 7168
E = 256
k = 8
E_d = 32        # experts per device
S = 1           # decode: one token per step
n_cores = 80
L1_per_core = 1.5 * 1024 * 1024  # bytes
bytes_per_element = 2.0  # BF16

def sharded_per_core(total_elements: int, n_cores: int) -> int:
    """Bytes per core for HEIGHT_SHARDED, tile-aligned."""
    tile = 32
    rows_per_core = math.ceil(total_elements / n_cores / tile) * tile
    return int(rows_per_core * bytes_per_element)

# 1. Hidden state / query projection [B, H] = [32, 7168]
# HEIGHT_SHARDED puts ceil(32/80) = 1 row per core × 7168 cols × 2 bytes = 14,336 bytes = 14 KB
hidden_state_per_core_bytes = math.ceil(B / n_cores) * H * int(bytes_per_element)
print(f"Hidden state per core: {hidden_state_per_core_bytes / 1024:.1f} KB")
# Hidden state per core: 14.0 KB (for ceil(32/80)=1 row per core)

# 2. Routing scores [B, E] = [32, 256]
routing_per_core = math.ceil(B / n_cores) * E * int(bytes_per_element)
print(f"Routing scores per core: {routing_per_core / 1024:.1f} KB")
# Routing scores per core: 0.5 KB

# 3. All-to-all dispatch buffer [C, E_d, H] = [1, 32, 7168] — 448 KB total
C = math.ceil(k * B * S / E)
dispatch_total_bytes = C * E_d * H * int(bytes_per_element)
# Treated as a 2D tensor [C*E_d, H] = [32, 7168] for sharding purposes
dispatch_rows = C * E_d  # = 32
dispatch_per_core = math.ceil(dispatch_rows / n_cores) * H * int(bytes_per_element)
print(f"Dispatch buffer C={C}, total={dispatch_total_bytes/1024:.0f} KB")
print(f"Dispatch buffer per core: {dispatch_per_core / 1024:.1f} KB")
# Dispatch buffer C=1, total=448 KB
# Dispatch buffer per core: 14.0 KB (ceil(32/80)=1 row per core)

# 4. Expert FFN intermediate (per-expert activation within one device)
# After dispatch, each device runs FFN on C*E_d = 32 token slots
# FFN intermediate shape [C*E_d, D_ffn]; D_ffn is model-specific [UNVERIFIED for Qwen3.5-35B]
# Approximate with D_ffn = H = 7168 as upper bound
ffn_intermediate_per_core = math.ceil(dispatch_rows / n_cores) * H * int(bytes_per_element)
print(f"FFN intermediate per core (approx): {ffn_intermediate_per_core / 1024:.1f} KB")

# 5. Total per-core occupancy (all four tensors simultaneously in L1)
total_per_core = (
    hidden_state_per_core_bytes
    + routing_per_core
    + dispatch_per_core
    + ffn_intermediate_per_core
)
print(f"\nTotal per-core L1: {total_per_core / 1024:.1f} KB")
print(f"L1 budget:         {L1_per_core / 1024:.0f} KB")
print(f"Utilization:       {100 * total_per_core / L1_per_core:.1f}%")

# Expected output:
# Hidden state per core: 14.0 KB
# Routing scores per core: 0.5 KB
# Dispatch buffer C=1, total=448 KB
# Dispatch buffer per core: 14.0 KB
# FFN intermediate per core (approx): 14.0 KB
# Total per-core L1: 42.5 KB
# L1 budget:         1536 KB
# Utilization:       2.8%
```

At $B=32$ decode, the L1 occupancy for these four tensors is approximately **42.5 KB per core**, which is **2.8% of the 1.5 MB budget**. This leaves ample room for:
- Attention score intermediates
- Weight tile caches within kernel CBs
- Double-buffering overhead for pipelining

The L1 budget is not a concern for decode at these parameters.

---

## Trade-Off Table: L1 vs. DRAM for Decode Activations

| Aspect | L1 Placement | DRAM Placement |
|---|---|---|
| Read latency | ~1 cycle (local SRAM) | ~100+ cycles per access |
| Write latency | ~1 cycle | ~100+ cycles per access |
| Speedup potential | 10–100× faster than DRAM reads | Baseline |
| Capacity risk | Hard `MemoryAllocationError` if budget exceeded | No allocation failure (limited by device DRAM) |
| Fallback on failure | None — kernel compilation fails | N/A |
| Persistence | Lost between ops unless explicitly maintained | Persists across ops |
| Correct for KV cache? | No — too large | Yes |
| Correct for activations at B=32? | Yes — ~2.8% L1 usage | Wasteful but correct |
| Correct for dispatch buffer at B=32? | Yes — ~14 KB/core | Wasteful but correct |

The key asymmetry: L1 placement is a strict improvement in latency but carries a hard failure risk. DRAM placement is always correct (within device memory limits) but is slower. The decode activation sizes at $B \leq 32$ are so small relative to L1 capacity that the failure risk is negligible.

---

## Incremental Promotion Strategy

> **Tip:** Do not attempt to place everything in L1 from the start. Follow this incremental approach:

1. **Baseline**: run the decode step with all tensors in `ttnn.DRAM_MEMORY_CONFIG`. Verify correctness.
2. **Profile**: identify which ops have the highest DRAM bandwidth utilization. These are the prime candidates for L1 promotion.
3. **Promote activations first**: move hidden states and query projections to L1 (`ttnn.L1_MEMORY_CONFIG`). Measure step latency improvement.
4. **Promote routing tensors**: move routing scores and top-k indices to L1. Measure again.
5. **Promote dispatch buffers**: move all-to-all buffers to L1 (verify $C=1$ at the batch size you're using). Measure.
6. **Stop at ~80% L1 utilization**: leave margin for CB double-buffering and any unforeseen allocation overhead. Do not push to 100%.
7. **KV cache stays in DRAM always**: do not attempt to promote it regardless of batch size.

---

## Summary: Decode Memory Placement

| Tensor | Shape (B=32) | Total Size | Per-Core (80 cores) | Recommendation |
|---|---|---|---|---|
| KV cache | `[layers, B, S_max, H_KV]` | ~34 GB | N/A | DRAM always |
| Hidden state / Q proj | `[B, H]` = `[32, 7168]` | 448 KB | 14.0 KB | L1 |
| Routing scores | `[B, E]` = `[32, 256]` | 16 KB | 0.5 KB | L1 |
| Top-k indices | `[B, k]` = `[32, 8]` | 1 KB | ~0.025 KB | L1 |
| Dispatch/combine buffer | `[C·E_d, H]` = `[32, 7168]` | 448 KB | 14.0 KB | L1 (C=1) |
| FFN intermediate (expert) | `[C·E_d, D_ffn]` (approx) | ~448 KB | ~14.0 KB | L1 if fits |

All sizes in BF16. KV cache estimate assumes $n_{\text{layers}}=64$, $n_{KV}=8$, $H_{KV}=128$, $S_{\text{max}}=4096$.

---

## References

- `wormhole_memory_hierarchy.md` — L1 and DRAM hardware capacities, CB allocation mechanics
- `memory_config_api.md` — `ttnn.MemoryConfig`, `ttnn.to_memory_config`, `ttnn.L1_MEMORY_CONFIG`
- `prefill_memory_strategy.md` — Contrast with prefill phase, where larger tensors require DRAM
- TT-NN source: `ttnn/cpp/ttnn/operations/eltwise/unary/` — example of `memory_config` propagation in unary ops
- Tenstorrent technical note: MoE dispatch buffer sizing (internal)

---

**Next:** [prefill_memory_strategy.md](./prefill_memory_strategy.md)
