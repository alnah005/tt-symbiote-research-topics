# Sparsity Tensor Placement

This file covers where to place the sparsity tensor in device memory, how to handle placement in a multi-device expert-parallel configuration, and how to integrate mask updates into a `ttnn.Trace`-based decode loop.

For the format of the mask itself, see `sparsity_tensor_format.md`. For how to construct the mask from router output, see `constructing_from_router_output.md`.

---

## 1. L1 vs. DRAM: Background

Wormhole B0 has two tiers of on-device memory:

- **L1 (SRAM):** 1.5 MB per Tensix core, 80 cores per chip. Very low latency, high bandwidth. Used for operand staging and small intermediate tensors.
- **DRAM:** ~12 GB per chip (depending on configuration). High capacity, lower bandwidth and higher latency than L1.

During `sparse_matmul` execution, the kernel reads the sparsity tensor once per tile row to decide whether to skip or compute that row. This read occurs in the kernel's inner loop, interleaved with activation and weight tile reads. The memory tier of the mask therefore affects every active tile-row decision.

---

## 2. Mask Size Analysis

The sparsity tensor has shape $[E_d \times M_t, K_t]$ with dtype `uint8` and `TILE_LAYOUT`. One tile = $32 \times 32 \times 1\ \text{byte} = 1024\ \text{bytes}$.

Number of tiles in the mask:

$$\text{tiles} = \left\lceil \frac{E_d \times M_t}{32} \right\rceil \times \left\lceil \frac{K_t}{32} \right\rceil$$

For Qwen3.5-35B ($E_d = 32$, $K_t = 224$):

| Regime | $B$ | $S$ | $C$ | $M_t$ | $E_d \times M_t$ | Tiles in mask | Mask size |
|--------|-----|-----|-----|--------|-------------------|---------------|-----------|
| Decode | 1 | 1 | 1 | 1 | 32 | $\lceil 32/32 \rceil \times \lceil 224/32 \rceil = 1 \times 7 = 7$ | 7168 bytes ≈ 7 KB |
| Decode | 32 | 1 | 1 | 1 | 32 | $1 \times 7 = 7$ | 7168 bytes ≈ 7 KB |
| Small prefill | 4 | 2048 | 256 | 8 | 256 | $\lceil 256/32 \rceil \times 7 = 8 \times 7 = 56$ | 57344 bytes ≈ 56 KB |

Both decode and small-prefill masks fit well within L1 capacity (1.5 MB per core). There is no L1 capacity reason to place the mask in DRAM for these sizes.

---

## 3. L1 Placement (Recommended for Decode)

```python
mask_ttnn = ttnn.from_torch(
    mask_torch,
    dtype=ttnn.uint8,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
```

**Benefits:**
- Mask reads during `sparse_matmul` kernel execution hit L1 SRAM, which has sub-microsecond access latency.
- For decode (7 KB mask), the entire mask is resident in L1 across the full kernel invocation. There is no eviction risk from other tensors given typical L1 utilization patterns.
- No DRAM bandwidth consumed by mask reads, leaving DRAM bandwidth available for weight tile streaming.

> **Tip:** For decode, always prefer `ttnn.L1_MEMORY_CONFIG` for the sparsity tensor. The 7 KB footprint is negligible relative to L1 capacity.

---

## 4. DRAM Placement (Fallback Under L1 Pressure)

```python
mask_ttnn = ttnn.from_torch(
    mask_torch,
    dtype=ttnn.uint8,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

**When to consider DRAM placement:**
- Prefill with large sequence lengths where L1 is heavily occupied by activation tiles and intermediate buffers.
- Profiling shows that L1 capacity pressure from other tensors causes eviction of the mask, defeating L1 placement.

**Cost of DRAM placement:**
- Each mask tile read traverses the NOC and hits DRAM, adding latency on every active tile-row check in the kernel.
- At decode sizes (7 tiles), the absolute bandwidth cost is small (7 × 1024 = 7168 bytes per `sparse_matmul` call).
- At prefill sizes (56 tiles), the cost grows but remains modest relative to weight tile DRAM traffic.

> **Warning:** DRAM mask placement is a performance regression, not a correctness issue. If you observe unexpectedly high `sparse_matmul` latency, check mask memory placement first. See `common_pitfalls.md`, P4.

---

## 5. Sharding for Multi-Device Expert Parallelism

In Qwen3.5-35B's $N=8$ device configuration, each device holds $E_d = 32$ local experts. The sparsity tensor is device-local: device $d$ holds a mask covering its own local experts (global indices $[d \times E_d,\ (d+1) \times E_d)$, local indices $[0, E_d)$).

There is no cross-device mask sharing. Each device independently:
1. Receives the full routing indices (broadcast from host or each device computes its own filter).
2. Filters to its local expert range.
3. Constructs its own $[E_d \times M_t, K_t]$ mask.
4. Calls `sparse_matmul` with its own local activation tensor and mask.

If using `ttnn.MeshDevice` with a mesh mapper, the mask can be created as a sharded tensor where each shard corresponds to one device's $[E_d \times M_t, K_t]$ mask. Alternatively, construct one mask per device independently.

---

## 6. Mask Lifetime: Do Not Cache Across Decode Steps

The routing decisions that produce the sparsity mask change every decode step — each new token is routed to a (potentially) different set of experts. The mask must be recomputed from fresh router output at every step.

**Do not** reuse the mask from the previous decode step. Doing so results in the model computing expert outputs for the wrong experts and skipping experts that should be active. This is a silent correctness bug that does not raise any TTNN error. See `common_pitfalls.md`, P3.

**Exception — static prompts:** If a specific token is decoded with identical routing decisions across multiple steps (which requires that the router's softmax produces the same top-k output for the same input), the mask could theoretically be reused. In practice, this does not occur in autoregressive generation because the KV-cache grows each step and the attention output changes, altering the router input. Treat the mask as always requiring fresh construction.

---

## 7. Integration with TTNN Tracing

`ttnn.Trace` records a fixed execution graph for later replay. The trace captures which operations to execute and which tensor buffers to use, but it does not capture the *values* in those buffers — buffer contents can be updated between trace captures and trace replays.

This property is what allows the sparsity mask to work inside a traced decode loop: the mask values change every step, but the mask tensor's buffer location (and therefore the trace's reference to it) remains constant.

### Correct pattern: pre-allocate mask before tracing

```python
import math
import torch
import ttnn

E_d = 32
K_t = 224
C = 1      # decode capacity
M_t = math.ceil(C / 32)  # = 1

# ── Pre-allocate the mask tensor BEFORE beginning the trace ──────────────────
# Initialize with zeros; actual values will be written each step.
initial_mask = torch.zeros(E_d * M_t, K_t, dtype=torch.uint8)
mask_ttnn = ttnn.from_torch(
    initial_mask,
    dtype=ttnn.uint8,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
# mask_ttnn is now a persistent tensor with a fixed L1 buffer address.

# ── Capture the trace ────────────────────────────────────────────────────────
with ttnn.Trace(mesh_device) as trace:
    # sparse_matmul references mask_ttnn by its buffer address.
    # The trace records this reference, not the values.
    output = ttnn.sparse_matmul(activation_ttnn, weight_ttnn, mask_ttnn)

# ── Decode loop ──────────────────────────────────────────────────────────────
for step in range(max_new_tokens):
    # 1. Run router, get fresh routing indices
    routing_indices = router(current_token_embedding)  # [T, k], int64

    # 2. Build new mask on CPU
    new_mask_torch = build_sparsity_mask_cpu(routing_indices, device_id=0, B=B, S=S)
    # new_mask_torch: [E_d * M_t, K_t], uint8

    # 3. Overwrite the pre-allocated mask buffer in-place
    # This writes new values into the same L1 buffer that the trace references.
    ttnn.copy_(mask_ttnn, ttnn.from_torch(
        new_mask_torch,
        dtype=ttnn.uint8,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    ))

    # 4. Replay the trace with the updated mask values
    ttnn.execute_trace(trace, mesh_device)
    # The replayed kernel reads mask_ttnn, which now contains this step's mask.
```

### What NOT to do inside a trace

Do not construct a new mask tensor inside the trace body. `ttnn.from_torch` involves a host-to-device transfer that creates a new buffer allocation, which is not compatible with trace replay semantics (the trace cannot replay a dynamic allocation with a different value each step).

```python
# WRONG: mask constructed inside trace — creates new buffer each step
with ttnn.Trace(mesh_device) as trace:
    mask_ttnn = ttnn.from_torch(mask_torch, ...)  # This is a new allocation every replay
    output = ttnn.sparse_matmul(activation_ttnn, weight_ttnn, mask_ttnn)
```

> **Warning:** Creating tensors inside a `ttnn.Trace` context that are derived from Python-side values (such as `ttnn.from_torch`) will either fail at trace capture time or produce wrong results at replay time. Always pre-allocate and update in-place.

---

## 8. Concrete Placement Recommendations

| Regime | Mask shape | Size | Recommended placement | Notes |
|--------|-----------|------|-----------------------|-------|
| Decode (any B ≤ 32, S=1) | $[32, 224]$ | 7 KB | `L1_MEMORY_CONFIG` | Always fits; no reason to use DRAM |
| Small prefill (B=4, S=2048) | $[256, 224]$ | 56 KB | `L1_MEMORY_CONFIG` | Fits comfortably in 1.5 MB L1 |
| Large prefill (B=32, S=2048) | $[E_d \times M_t, 224]$ | varies | `L1_MEMORY_CONFIG` if fits, else `DRAM_MEMORY_CONFIG` | Profile L1 utilization before deciding |

> **Tip:** For the decode regime, the allocation cost of calling `ttnn.from_torch` on every step is non-trivial. Pre-allocate the mask tensor once at model initialization and use `ttnn.copy_` to update its values each step. This eliminates repeated L1 allocation/deallocation overhead.

---

## References

- `sparsity_tensor_format.md` — mask shape, dtype, and layout requirements
- `constructing_from_router_output.md` — mask construction from routing indices
- `common_pitfalls.md` — P3 (stale mask), P4 (DRAM placement regression)
- Chapter 4, `sparse_matmul_internals.md` — how the kernel reads the mask during execution
- Chapter 2, TTNN Wormhole primer — L1 capacity (1.5 MB/core), memory config API, `ttnn.Trace` semantics
