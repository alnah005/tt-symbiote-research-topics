# CCL Sensitivity Analysis

## Context

This file addresses:
- **Q1** — What are the actual latency costs of each CCL op, and are the current topology/link/buffer settings optimal for T3K's 1×8 mesh?

This file synthesizes findings from [`all_gather_linear_topology.md`](./all_gather_linear_topology.md) and [`reduce_scatter_ring_topology.md`](./reduce_scatter_ring_topology.md) and connects them to the `TTNNMoE.forward` code structure.

Source range: `moe.py:L1429–L1490`

---

## Which CCL Op Dominates at Batch=1 Decode

The two CCL operations have structurally different characteristics:

| Property | `all_gather_async` | `reduce_scatter_minimal_async` |
|---|---|---|
| Topology | Linear | Ring |
| Direction | gather (no reduction) | scatter + reduce |
| Message size (GLM-4-MoE, batch=1) | 14 KB total | 14 KB in, 1.75 KB out |
| Pipeline depth parameter | — | `chunks_per_sync=10` |
| Worker parallelism | `num_links=1` | `num_workers_per_link=2` |
| Buffering | — | `num_buffers_per_channel=2` |
| Pre-op work | layout cast + typecast (bf16→f32) for gate linear | `ttnn.mul` normalize by `1/n_rs` |

**Why reduce-scatter is likely slower:** The reduce-scatter combines two operations — element-wise accumulation across all 8 devices and scatter of the reduced result. Each Ring step requires: send a chunk, receive a chunk from the adjacent neighbor, accumulate into the local buffer, and forward. The accumulation step is a compute dependency that does not exist in all-gather. Even though Ring topology is pipelined, each step has an accumulate-then-forward dependency chain that Linear all-gather does not.

**First-principles latency estimates (GLM-4-MoE, batch=1):**

All-gather (Linear):
```
T_ag ≈ 7 hops × (3 µs startup + 1.04 µs transfer) ≈ 28 µs lower bound
Typical measured range: 40–80 µs
```

Reduce-scatter (Ring):
```
T_rs ≈ 7 ring steps × (3 µs startup + accumulate_overhead + 0.2 µs transfer)
     ≈ 7 × (3 µs + ~1 µs accumulate) ≈ 28 µs lower bound
Plus: pre-normalize ttnn.mul ≈ 1 µs
Typical measured range: 50–100 µs
```

The reduce-scatter includes an additional compute step (reduction/accumulation) per ring step that the all-gather does not, making it the likely dominant CCL op. However, both are in the same order of magnitude, and the actual dominance ratio is hardware- and firmware-dependent.

**Measurement-based verdict:** Fill in the following table after running the isolation benchmarks described in the preceding files:

| Op | Mean latency (µs) | Fraction of CCL total | Fraction of full MoE forward |
|---|---|---|---|
| `all_gather_async` | ___ | ___ % | ___ % |
| `ttnn.mul` (1/n_rs) | ___ | ___ % | ___ % |
| `reduce_scatter_minimal_async` | ___ | ___ % | ___ % |
| **CCL total** | ___ | 100% | ___ % |
| `TTNNExperts.forward` | ___ | — | ___ % |
| Gate linear + routing | ___ | — | ___ % |
| Shared experts + add | ___ | — | ___ % |
| **MoE forward total** | ___ | — | 100% |

The CCL fraction of total MoE forward time is the primary number to track. If CCL total > 40%, topology and link tuning is the highest-priority optimization.

---

## Are Current Settings Optimal?

### all_gather_async: Linear, num_links=1

**Topology assessment:** Linear is the correct topology for a 1×8 chain without a confirmed wrap-around link. If the wrap-around link is available and Ring topology is tested, the expected improvement at batch=1 is small (both topologies are startup-latency-dominated). Linear is defensively optimal.

**num_links=1 assessment:** At batch=1 with a 14 KB message, the transfer-time fraction of total latency is approximately:

```
T_transfer / T_total ≈ 1.04 µs / 28 µs ≈ 4%
```

Doubling to `num_links=2` reduces this fraction by half, saving approximately 0.52 µs per hop × 7 hops = 3.6 µs — roughly a 5–13% improvement. This is worth measuring but is not a major optimization at batch=1. The dominant cost is per-hop startup latency (`T_start`), which is not reduced by adding links.

**Verdict:** Current all-gather settings are near-optimal for batch=1 decode. Significant improvement requires either reducing per-hop startup latency (firmware/driver work) or overlapping the all-gather with prior work (asynchronous launch opportunities).

### reduce_scatter_minimal_async: Ring, chunks_per_sync=10, num_workers_per_link=2

**chunks_per_sync=10 assessment:** For a 14 KB message with an estimated internal chunk size of ~1 KB, there are approximately 14 chunks in flight. The sync point at chunk 10 occurs once, with 4 remaining chunks completing before the op finishes. Increasing `chunks_per_sync` to 20 or 50 would eliminate this sync point entirely (message completes before sync triggers), but the benefit is at most one sync round-trip saved — approximately 2–4 µs. Worth measuring; not a large win.

**num_workers_per_link=2 assessment:** The estimated DMA worker savings (~2.3 µs for `num_workers_per_link=2` vs 1) are derived in [`reduce_scatter_ring_topology.md`](../ch2_ccl_latency_and_topology/reduce_scatter_ring_topology.md). `num_workers_per_link=4` is worth testing if the API supports it, though diminishing returns are expected.

**Ring topology assessment:** Ring is the standard choice for reduce-scatter and is correct here. Linear topology is suboptimal for reduce-scatter because it does not distribute the accumulation work evenly; the first device would need to wait for the full chain before receiving any accumulation result.

**Verdict:** Current reduce-scatter settings are reasonable. `chunks_per_sync=10` should be swept upward (20, 50) to confirm it is not causing unnecessary sync stalls. `num_workers_per_link=2` is correctly set. The main opportunity is overlap with other work, not parameter tuning.

---

## Feasibility of Overlapping Reduce-Scatter with Shared-Expert Compute

The most impactful CCL optimization available without hardware changes is **op overlap**: launching `reduce_scatter_minimal_async` asynchronously and performing other compute while it runs. The "async" suffix in the op name signals that this design intent exists in the API.

### Current Code Structure (moe.py:L1426–L1494)

```
residual = x                                                    # moe.py:L1426

# Step 4: reduce-scatter
routed_out = ttnn.mul(routed_out, 1.0 / 8.0)          # moe.py:L1477
routed_output = reduce_scatter_minimal_async(routed_out, ...)  # moe.py:L1478–L1490

# Step 5: shared experts (operates on original `residual`, not routed_out)
shared_output = self.shared_experts(residual)           # moe.py:L1493
output = ttnn.add(routed_output, shared_output.to_ttnn)
```

**Data independence:** `shared_experts(residual)` reads `residual` — the original `x` before all-gather, assigned at `moe.py:L1426` before the `all_gather_async` call — not `routed_output`. `routed_output` (the reduce-scatter output) is not read until `ttnn.add`. This dependency is clean, making `shared_experts` a candidate for overlapping with the reduce-scatter.

### What Overlap Would Look Like

In the current sequential execution, the critical path is:

```
reduce_scatter_minimal_async (blocking) → shared_experts → ttnn.add
```

With overlap:

```
reduce_scatter_minimal_async (async launch)
  │  ← shared_experts runs here while scatter is in flight
  ↓
ttnn.add (waits for scatter completion via barrier)
```

The `barrier_semaphore` parameter in the reduce-scatter call is precisely the mechanism for this pattern: the scatter asserts the barrier semaphore on completion, and a subsequent op that depends on its output waits on that semaphore before executing.

### Feasibility Assessment

**Buffer conflict check:** `residual` and `routed_out` (the reduce-scatter input) are separate tensors in DRAM. `shared_output` (produced by `shared_experts`) writes to a new DRAM buffer. There is no aliasing with `routed_out`. Buffer conflict is not a barrier to overlap.

**Code restructuring required:** The current Python code issues `reduce_scatter_minimal_async` and then calls `self.shared_experts(residual)` on the next line. Because the CCL op is async (non-blocking host), `shared_experts` may already begin executing in parallel on the device if the device command queue allows it. However, this depends on the TTNN scheduler and whether the two call sites share a device queue or separate queues.

The explicit restructuring needed:
1. Launch `reduce_scatter_minimal_async` asynchronously (already the case).
2. Immediately enqueue `shared_experts(residual)` work, allowing the device scheduler to run both.
3. Issue `ttnn.add` only after both the scatter and the shared experts have written their outputs.

The barrier semaphore on `reduce_scatter_minimal_async` may already enforce this ordering correctly if `ttnn.add` checks it. Verify by profiling: if Tracy shows `shared_experts` running in the gap while the scatter is in flight, overlap is already occurring. If they are strictly sequential in the timeline, restructuring is needed.

**Potential savings:** If `shared_experts` takes T_se µs and the reduce-scatter takes T_rs µs, and if T_se < T_rs (plausible at batch=1 where shared experts involve small matmuls), overlap eliminates T_se from the critical path entirely. A conservative estimate: shared experts at batch=1 cost 20–50 µs; reduce-scatter costs 50–100 µs. Full overlap saves 20–50 µs per MoE layer.

### Steps to Validate Overlap Feasibility

1. **Profile sequentially:** Using Tracy or TTNN op timers, confirm whether `shared_experts` currently runs before or after the scatter completes. If they already overlap, no code change is needed and the overlap saving is already realized.

2. **Check queue assignment:** Determine whether `reduce_scatter_minimal_async` and the ops inside `shared_experts` are enqueued on the same device command queue. If they share a queue, they are serialized by the queue regardless of Python execution order. Multiple queues are required for true parallelism.

3. **Prototype explicit overlap:** Restructure `moe.py:L1426–L1494` to enqueue the scatter and the shared experts in separate queues, then synchronize before the add. Benchmark the restructured forward pass and compare.

---

## Summary: CCL Optimization Priority Matrix

| Optimization | Expected savings (batch=1) | Implementation effort | Priority |
|---|---|---|---|
| Sweep `chunks_per_sync` upward (20, 50) | 0–4 µs | Trivial (1 param change) | Low |
| Set `num_workers_per_link=1` to confirm benefit of `=2` | Baseline measurement | Trivial | Low (measurement) |
| Increase `num_links` to 2 (all-gather and/or reduce-scatter) | 4–8 µs | Low (1 param change + HW validation) | Medium |
| Overlap reduce-scatter with shared-expert compute | 20–50 µs | Medium (queue management, profiling) | High |
| Firmware-level reduction of per-hop startup latency | 14–35 µs | High (firmware work) | Low (external dependency) |

The highest-leverage action is investigating whether the reduce-scatter and shared-expert compute already overlap and, if not, explicitly enabling that overlap. This does not require topology or link changes and has the largest expected impact at batch=1 decode.

---

**Next:** [Chapter 3 — Expert Dispatch Pipeline Profiling](../ch3_expert_dispatch_pipeline_profiling/index.md)
