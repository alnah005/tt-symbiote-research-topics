# Chapter 1: MoE Forward Pass Anatomy

## Why Read the Code Before Measuring

Profiling a distributed MoE kernel without a precise mental model of its call stack produces numbers that cannot be interpreted. Every latency bucket, every CCL event, every sparsity decision exists because of a specific line of code. Chapter 1 builds that model from the source before any measurement tool is introduced.

The four files in this chapter cover the complete execution path of one MoE forward pass on a T3K mesh — from the initial all-gather that re-assembles tensor-parallel shards through the per-expert sparse matrix multiplications and back out through the reduce-scatter. The CPU fallback path is documented separately so readers can confirm they are measuring the TTNN path and not a silent Python loop.

---

## End-to-End Data Flow

```
Input tensor x  (tensor-parallel sharded across 8 devices)
        │
        ▼  moe.py:L1359–L1366
┌───────────────────────────────────────┐
│  ttnn.experimental.all_gather_async   │  Linear topology, num_links=1
│  → x now full-width on every device  │
└───────────────────────────────────────┘
        │
        ▼  moe.py:L1369–L1393
┌───────────────────────────────────────┐
│  Gate linear  (HiFi4, fp32 acc)       │  x_f32 @ gate_weight → router_logits
│  topk routing → indices + weights     │
└───────────────────────────────────────┘
        │
        ▼  moe.py:L1396–L1401
┌────────────────────────────────────────────────────────────────────┐
│  TTNNExperts.forward                                               │
│                                                                    │
│  pad tokens to SPARSITY_BLOCK_SIZE=32 boundary                     │
│       │                                                            │
│       ▼                                                            │
│  ttnn.all_to_all_dispatch   (cluster_axis=1)                       │
│       │  tokens routed to their assigned expert devices            │
│       ▼                                                            │
│  ttnn.moe_expert_token_remap → sparsity_t                          │
│       │                                                            │
│       ▼                                                            │
│  sparse_matmul(x, w1_proj)  →  w1_out   ┐                         │
│  sparse_matmul(x, w3_proj)  →  w3_out   ├ gate-up projections     │
│       │                                  ┘                         │
│  silu(w1_out) * w3_out  →  intermediate                           │
│       │                                                            │
│  sparse_matmul(intermediate, w2_proj)  →  expert_output           │
│       │  (down projection)                                         │
│       ▼                                                            │
│  ttnn.all_to_all_combine    (cluster_axis=1)                       │
│       │  results gathered back to originating devices              │
│       ▼                                                            │
│  weight application: repeat + permute + mul + sum                  │
│       │                                                            │
│  remove padding                                                    │
└────────────────────────────────────────────────────────────────────┘
        │
        ▼  moe.py:L1404–L1420
┌────────────────────────────────────────┐
│  ttnn.experimental.reduce_scatter_     │  Ring topology
│  minimal_async                         │  chunks_per_sync=10
│  → output sharded across devices again │  num_workers_per_link=2
└────────────────────────────────────────┘
        │
        ▼  moe.py:L1423–L1426
┌──────────────────────────────────┐
│  shared_experts(residual)        │  parallel path on original x
│  ttnn.add(routed_output, shared) │
└──────────────────────────────────┘
        │
        ▼
Output tensor  (tensor-parallel sharded)
```

The two CCL operations (`all_gather_async` and `reduce_scatter_minimal_async`) bracket the entire MoE computation. Everything in between — routing, dispatch, expert compute, combine, weighting — runs between those two synchronization points.

---

## Files in This Chapter

| File | Contents |
|---|---|
| [`ttnn_moe_forward.md`](./ttnn_moe_forward.md) | Annotated walkthrough of `TTNNMoE.forward` (moe.py:L1346–L1496). Covers all-gather, gate linear, reduce-scatter, and the `TTNNBailingMoE` subclass relationship. |
| [`ttnn_experts_forward.md`](./ttnn_experts_forward.md) | Step-by-step walkthrough of `TTNNExperts.forward` (moe.py:L1027–L1343). Covers token padding, all-to-all dispatch, sparse matmul pipeline, all-to-all combine, and weight application. |
| [`cpu_fallback_paths.md`](./cpu_fallback_paths.md) | Documents `Glm4MoeNaiveMoeHybrid` (moe.py:L559–L613), the `ttnn = False` flag, the CPU expert loop, and a checklist for verifying the TTNN path is active. |

---

## Research Questions Addressed in This Chapter

Chapter 1 is prerequisite material for all research questions, but most directly supports:

- **Q1** — What is the per-op breakdown of a single MoE forward pass?
- **Q2** — Which CCL operations dominate end-to-end latency?
- **Q3** — How is sparsity expressed and exploited in the kernel?
- **Q7** — Is the TTNN path actually active in a given deployment?
