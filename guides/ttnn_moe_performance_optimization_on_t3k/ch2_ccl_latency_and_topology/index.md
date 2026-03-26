# Chapter 2: CCL Latency and Topology

## Why CCL Ops Are Primary Suspects at Decode

In the prefill regime, the MoE forward pass is arithmetic-bound: large batches and long sequences keep the expert matmuls saturated, and the collective communication cost is amortized over many tokens. At decode — batch=1, single token — this balance inverts. The expert matmuls process exactly one token's worth of compute (subject to the `SPARSITY_BLOCK_SIZE=32` padding floor), while the two CCL operations — `all_gather_async` and `reduce_scatter_minimal_async` — must still transfer the full hidden-dimension tensor across all 8 devices regardless of token count.

The core problem is message-size independence: CCL latency on a fixed-size message does not shrink as batch size drops to 1. A hidden dimension of 7168 (GLM-4-MoE) or 4096 (Bailing) produces a message that must traverse up to 7 Ethernet hops (Linear topology) or 4 ring steps (Ring topology) with no computational work to hide behind. The matmul compute time, on the other hand, collapses with the token count. At batch=1 decode, it is entirely plausible that the two CCL ops together account for 40–60% of the total MoE forward pass wall-clock time.

This is Q1: **What are the actual latency costs of each CCL op, and are the current topology, link count, and buffer settings optimal for T3K's 1×8 mesh?**

---

## T3K Physical Topology

The T3K is a 1×8 mesh: one row of eight Wormhole chips connected by Ethernet links. When the mesh is described as `device.shape = (1, 8)`, that means:

- `shape[0] = 1` — a single row; the dispatch axis for `all_to_all_dispatch` / `all_to_all_combine`.
- `shape[1] = 8` — eight columns; the communication axis (`cluster_axis=1`) for `reduce_scatter_minimal_async` in `TTNNMoE.forward`. Note that `all_gather_async` does not accept a `cluster_axis` parameter and is not called with one.

The physical links are point-to-point Ethernet connections between adjacent chips. There is no hardware switch; inter-device traffic must be forwarded hop-by-hop. This means:

| Topology | Route for device 0 → device 7 | Hops |
|---|---|---|
| Linear | 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 | 7 |
| Ring | 0 → 1 → 2 → 3 (forward) or 0 → 7 → 6 → 5 → 4 (reverse) | 4 max |

The asymmetry matters: for a pure all-gather (one-to-all broadcast-style), Linear topology requires only that each device forward what it has accumulated so far, so the last device receives all shards after 7 sequential steps. Ring topology can pipeline but adds wrap-around link overhead. The code chooses differently for each op (Linear for all-gather, Ring for reduce-scatter), and Chapter 2 examines whether those choices are optimal.

**Ethernet link bandwidth:** Each point-to-point link on T3K provides approximately 12 GB/s of peak bandwidth in each direction (bidirectional). With `num_links=1`, only one link is exercised per hop. T3K chips expose multiple independent Ethernet ports (typically up to 2 usable between adjacent chip pairs), so there is hardware headroom to increase `num_links` in principle.

---

## CCL Operations in TTNNMoE.forward

Two CCL operations bracket the entire MoE computation:

1. **`all_gather_async`** (`moe.py:L1429–L1436`) — executed before the gate linear and before `TTNNExperts.forward`. Gathers the hidden dimension across all 8 devices so each chip holds the full activation tensor. Uses `Linear` topology, `num_links=1`.

2. **`reduce_scatter_minimal_async`** (`moe.py:L1478–L1490`) — executed after `all_to_all_combine` completes inside `TTNNExperts.forward`. Reduces across the 8-device axis and re-shards the hidden dimension back to tensor-parallel layout. Uses `Ring` topology, `chunks_per_sync=10`, `num_workers_per_link=2`, `num_links=1`.

Only `reduce_scatter_minimal_async` communicates along `cluster_axis=1` (the 8-column axis); `all_gather_async` does not take a `cluster_axis` parameter. Both use a single Ethernet link. The semaphore management is handled by `self.device_state.ccl_manager`, which cycles through pre-allocated semaphore handles to avoid stalls under pipelining.

---

## Files in This Chapter

| File | Contents |
|---|---|
| [`all_gather_linear_topology.md`](./all_gather_linear_topology.md) | Latency model for `all_gather_async` (Linear, `num_links=1`); isolation methodology; whether increasing links or switching to Ring helps at batch=1 decode. |
| [`reduce_scatter_ring_topology.md`](./reduce_scatter_ring_topology.md) | Pipelining mechanics of `reduce_scatter_minimal_async` (Ring, `chunks_per_sync=10`, `num_workers_per_link=2`); parameter sweep methodology for `chunks_per_sync` and `num_workers_per_link`. |
| [`ccl_sensitivity_analysis.md`](./ccl_sensitivity_analysis.md) | Which CCL op dominates; feasibility of overlapping reduce-scatter with shared-expert compute; verdict on current topology/link/buffer settings. |

---

## Research Questions Addressed

This chapter addresses:

- **Q1** — What are the actual latency costs of each CCL op, and are the current topology/link/buffer settings optimal for T3K's 1×8 mesh?

The findings from Chapter 2 feed directly into Chapter 7's optimization priority matrix. If one CCL op dominates, it is the first target for tuning. If neither dominates, the bottleneck lies inside `TTNNExperts.forward` and Chapter 3 becomes the priority.
