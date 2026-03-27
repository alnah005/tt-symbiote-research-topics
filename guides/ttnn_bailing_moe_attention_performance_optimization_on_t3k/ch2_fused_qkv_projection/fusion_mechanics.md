# Fusion Mechanics of `TTNNLinearIColShardedWAllReduced`

## Overview

In a naive implementation of multi-head attention, the Q, K, and V projections are three independent matrix multiplications, each followed by a collective communication operation to combine partial results across the 8 T3K chips. `TTNNLinearIColShardedWAllReduced` eliminates this redundancy by fusing all three weight matrices into one and performing a single all-reduce in place of the three separate collectives. This section describes how that fusion is structured, which CCL primitive is used, what tensor is communicated over Ethernet, and how `num_links` controls the utilization of the available Ethernet bandwidth.

## Weight Fusion: Column-Sharding Across 8 Chips

### Concatenated Weight Matrix

The Q, K, and V projection weights are stored as separate parameter tensors during model loading:

- `W_Q`: shape `(H, Nq·D)` = `(4096, 2048)` — maps the hidden state to 16 query heads
- `W_K`: shape `(H, Nkv·D)` = `(4096, 512)` — maps to 4 key heads
- `W_V`: shape `(H, Nkv·D)` = `(4096, 512)` — maps to 4 value heads

During model preparation, these are concatenated along the output (column) dimension to form the fused weight:

```
W_QKV: shape (H, C) = (4096, 3072)
C = (Nq + 2·Nkv) · D = (16 + 4 + 4) · 128 = 3072
```

This concatenation is a one-time cost at model load time and does not affect decode-step latency.

### Column Sharding

`TTNNLinearIColShardedWAllReduced` distributes `W_QKV` across the 8 chips using a **WIDTH_SHARDED** (column-sharded) layout. Each chip `i` receives a contiguous slice of columns:

```
W_QKV_shard[i]: shape (4096, 384)   # 3072 / 8 columns per chip
```

Because the column dimension (3072) is evenly divisible by 8, each shard is exactly 384 columns wide. The columns are allocated so that each chip holds a coherent set of head projections — specifically, chip `i` holds 2 Q heads, 0.5 K heads, and 0.5 V heads in proportional terms. In practice the allocation is rounded to head boundaries where possible, but for analysis purposes the 384-column-per-chip figure is exact.

The input activation tensor `X: shape (1, H) = (1, 4096)` is broadcast to all 8 chips (replication, not sharding) during the attention input preparation stage. This is a deliberate choice: replicating a small `(1, 4096)` tensor costs far less than sharding it, since the per-chip activation data is only `4096 × 2 bytes = 8 KB` in bfloat16.

### Local Matmul

Each chip independently computes a partial output:

```
Y_partial[i] = X @ W_QKV_shard[i]
Y_partial[i]: shape (1, 384)
```

The 8 partial outputs, when concatenated across chips, reconstitute the full fused projection:

```
Y_full: shape (1, 3072)  # conceptually; not yet assembled
```

At this point the result is **column-sharded**: each chip holds 384 of the 3072 output values. To make the full `(1, 3072)` tensor available on every chip (required for the subsequent attention score computation), an all-reduce is executed.

## The All-Reduce Step

### CCL Primitive Used

The collective used here is an **all-reduce with sum reduction**. This is exposed through TTNN's CCL layer as `ttnn.all_reduce`. Internally on T3K, `ttnn.all_reduce` is typically implemented as a **reduce-scatter followed by all-gather** (also known as the ring all-reduce decomposition), which is bandwidth-optimal for the ring topology formed by the 8-chip Ethernet mesh.

The implementation path in the `tt-metal` repository is:

```
tt_metal/impl/dispatch/kernels/cq_prefetch.cpp   # dispatch layer
ttnn/cpp/ttnn/operations/ccl/all_reduce/         # CCL all-reduce op
ttnn/cpp/ttnn/operations/ccl/reduce_scatter/     # reduce-scatter primitive
ttnn/cpp/ttnn/operations/ccl/all_gather/         # all-gather primitive
```

The Python-facing call in the model code is effectively:

```python
output = ttnn.all_reduce(
    partial_output,          # shape (1, 384) per chip, WIDTH_SHARDED
    math_op=ttnn.ReduceType.Sum,
    num_links=num_links,     # typically 1
    memory_config=output_memory_config,
)
```

After the all-reduce, every chip holds the full `(1, 3072)` output in its L1 or DRAM, depending on `output_memory_config`.

### Tensor Shape Communicated Over Ethernet

During the ring all-reduce, each chip sends and receives data in two sub-phases:

**Phase 1 — Reduce-Scatter:**
Each chip contributes its `(1, 384)` shard. After reduce-scatter, each chip holds a `(1, 48)`-element slice (384 / 8) that is the sum-reduced result for its assigned output columns. The total data moved per chip in this phase is:

```
sent    = (1 × 384) × 2 bytes = 768 bytes   [bfloat16]
received = (1 × 384) × 2 bytes = 768 bytes
```

**Phase 2 — All-Gather:**
Each chip broadcasts its `(1, 48)` reduced slice to the other 7 chips, reassembling the full `(1, 3072)` result everywhere. The total data per chip:

```
sent    = (1 × 48) × 2 bytes × 7 hops ≈ 672 bytes
received = (1 × 48) × 2 bytes × 7 hops ≈ 672 bytes
```

Summing both phases, the total bidirectional Ethernet traffic per chip for this all-reduce is approximately **1.44 KB** [ESTIMATE]. This is extremely small relative to typical CCL payloads; the dominant cost is latency (link setup, kernel dispatch, and synchronization) rather than raw bandwidth.

### How `num_links` Controls Ethernet Utilization

`num_links` controls how many of the 16 available 100 Gb/s Ethernet ports per chip are allocated to the CCL operation; see `num_links_tuning.md` for the quantitative implications.

## Theoretical Bandwidth Model

This section builds a first-principles model comparing the time spent in the fused matmul against the time spent in the all-reduce, as a function of hidden size `H` at fixed `C = 3072` and batch size 1.

### Matmul Compute Time

Each chip performs a `(1, H) × (H, C/P)` matmul, where `P = 8`. For Ling: `(1, 4096) × (4096, 384)`.

Because batch=1 matrix-vector products are memory-bandwidth-bound rather than compute-bound, the relevant bottleneck is DRAM read speed:

```
bytes_loaded = H × (C/P) × 2  [weight matrix, bfloat16]
             = 4096 × 384 × 2 = 3,145,728 bytes ≈ 3.0 MB per chip

bytes_input  = H × 2 = 8,192 bytes per chip  (already in L1)
```

Each chip's on-chip DRAM bandwidth is approximately 288 GB/s [ESTIMATE, Wormhole n300 specification]. At full DRAM bandwidth:

```
t_matmul_bw = 3.0 MB / 288 GB/s ≈ 10.4 µs  [ESTIMATE]
```

This is an optimistic lower bound. Real matmul latency will be in the range **10–40 µs** [ESTIMATE] depending on shard layout and kernel overhead.

### All-Reduce Transfer Time

The all-reduce payload per chip is ~1.44 KB (bidirectional). At `num_links=1` with 100 Gb/s = 12.5 GB/s per direction:

```
t_transfer = 1.44 KB / 12.5 GB/s ≈ 0.00012 µs  [ESTIMATE]
```

Transfer time is negligible. The all-reduce cost is instead dominated by **synchronization and kernel launch latency**, estimated at **3–10 µs** per CCL operation on T3K [ESTIMATE].

### Model Summary

Table: Theoretical time components for the fused QKV projection (Ling hidden_size=4096, batch=1, seq_len=1)

| Component | Dominant cost | Estimated latency |
|---|---|---|
| Fused matmul `(1,4096)×(4096,384)` | DRAM weight load | 10–40 µs [ESTIMATE] |
| All-reduce payload transfer | Link latency, not bandwidth | < 1 µs [ESTIMATE] |
| All-reduce synchronization overhead | Firmware + kernel dispatch | 3–10 µs [ESTIMATE] |
| **Total fused path** | | **13–50 µs** [ESTIMATE] |

**Key insight:** At Ling's hidden size and batch=1, the fused QKV projection is **memory-bandwidth-bound** during the matmul phase and **latency-bound** (not bandwidth-bound) during the all-reduce phase. Increasing `num_links` beyond 1 is unlikely to help unless the all-reduce payload grows substantially (e.g., from increased batch size or hidden size). This motivates the sensitivity analysis in `num_links_tuning.md`.

---

**Next:** [Latency Savings Analysis](./latency_savings_analysis.md)
