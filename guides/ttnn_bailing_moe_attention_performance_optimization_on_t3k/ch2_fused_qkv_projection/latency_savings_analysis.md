# Latency Savings Analysis: Fused vs. Unfused QKV Projection

## Purpose

This file quantifies the latency reduction achieved by `TTNNLinearIColShardedWAllReduced` relative to a naive unfused implementation. It establishes the baseline cost of 3 separate matmuls plus 5 CCL operations, then estimates the fused path cost, and identifies where the remaining bottleneck lies. A brief preview of how to measure the compute/communication split using TTNN op timers is included as a forward reference to Chapter 7.

All latency figures are for the Ling model in decode mode: `hidden_size=4096`, batch=1, `seq_len=1`, on T3K (8-chip, 1×8 mesh).

## Baseline: 3 Separate Matmuls + 5 CCL Operations

### Why Five CCL Operations?

In an unfused implementation the Q, K, and V projections each require their own collective. On a multi-chip tensor parallel setup with column-sharded weights, the typical pattern for each projection is:

1. **All-gather** the input `X` from its sharded form to a replicated form on each chip (if the prior layer left `X` WIDTH_SHARDED rather than replicated).
2. **Local matmul** on the replicated input and the local weight shard.
3. **All-reduce** (or reduce-scatter) to combine partial outputs.

With 3 projections (Q, K, V), the naive pattern produces 6 CCL operations (1 per projection × 2 phases). In practice, TTNN can hoist the input all-gather so it is shared across the three matmuls, reducing the count to **5 CCL operations** (1 shared all-gather for the input + 1 all-reduce each for Q, K, V + sometimes 1 re-scatter to restore the downstream sharding). The exact count depends on the surrounding op graph; 5 is a representative minimum for the unfused case.

### Latency of Each Separate Matmul

Each of the three projections performs a `(1, 4096) × (4096, width)` matmul on each chip:

- `W_Q` shard per chip: `(4096, 256)` — `Nq·D / P = 2048 / 8`
- `W_K` shard per chip: `(4096, 64)` — `Nkv·D / P = 512 / 8`
- `W_V` shard per chip: `(4096, 64)` — same as W_K

Each matmul is independently dispatched to the Tensix array. Key costs per matmul:

Table: Per-matmul weight data loaded from DRAM (bfloat16, per chip)

| Projection | Weight shard shape | DRAM bytes loaded | Estimated matmul latency |
|---|---|---|---|
| Q | `(4096, 256)` | 2.0 MB | 8–28 µs [ESTIMATE] |
| K | `(4096, 64)` | 0.5 MB | 3–10 µs [ESTIMATE] |
| V | `(4096, 64)` | 0.5 MB | 3–10 µs [ESTIMATE] |
| **Total (sequential)** | | **3.0 MB** | **14–48 µs** [ESTIMATE] |

The three matmuls load exactly the same total weight data as the fused matmul (3.0 MB per chip), but they do so in three separate kernel dispatch cycles. Each kernel dispatch incurs a fixed overhead of approximately **2–5 µs** per op [ESTIMATE] for command queue submission, kernel binary DMA to Tensix, and synchronization. Summing dispatch overhead:

```
extra_dispatch_overhead ≈ 2 × 3–5 µs = 6–10 µs  [ESTIMATE]
(2 extra dispatches vs. fused, since fused still has 1 dispatch)
```

### Latency of Each CCL Operation

At `num_links=1`, each individual CCL operation on T3K incurs:

- **Synchronization + kernel launch:** 3–6 µs [ESTIMATE]
- **Data transfer time:** negligible for payloads below ~10 KB (see `fusion_mechanics.md`)

Five CCL operations at 3–6 µs each:

```
t_ccl_total = 5 × (3–6 µs) = 15–30 µs  [ESTIMATE]
```

### Baseline Total

Table: Unfused QKV projection total estimated latency (Ling, decode step, T3K)

| Component | Count | Per-op latency | Subtotal |
|---|---|---|---|
| Matmul (Q) | 1 | 8–28 µs | 8–28 µs |
| Matmul (K) | 1 | 3–10 µs | 3–10 µs |
| Matmul (V) | 1 | 3–10 µs | 3–10 µs |
| CCL operations | 5 | 3–6 µs | 15–30 µs |
| **Unfused total** | | | **29–78 µs** [ESTIMATE] |

The wide range reflects uncertainty in Tensix kernel efficiency for small matmuls and CCL synchronization overhead, both of which require empirical measurement (see "Measuring with TTNN Op Timers" below).

## Fused Path: 1 Matmul + 1 All-Reduce

### Fused Matmul

The fused weight shard per chip is `(4096, 384)`, loading exactly 3.0 MB from DRAM — identical total data as the unfused case, but in a single contiguous kernel dispatch. The Tensix compute array sees a larger, more efficient tile shape:

```
Fused: (1, 4096) × (4096, 384)   — one dispatch, larger tile
Unfused Q: (1, 4096) × (4096, 256)  \
Unfused K: (1, 4096) × (4096, 64)   |  three dispatches, smaller tiles
Unfused V: (1, 4096) × (4096, 64)  /
```

Both require 3.0 MB of DRAM reads, but the fused case allows the TTNN compiler to schedule a single contiguous DRAM read burst, potentially improving DRAM bandwidth utilization and eliminating 2 kernel dispatch overheads.

Estimated fused matmul latency:

```
t_fused_matmul ≈ 10–35 µs  [ESTIMATE]
```

(Slightly better than the unfused sum due to dispatch savings, but DRAM read volume is the same.)

### Single All-Reduce

One `ttnn.all_reduce` on a `(1, 384)` WIDTH_SHARDED tensor at `num_links=1`:

```
t_all_reduce ≈ 3–8 µs  [ESTIMATE]
```

### Fused Path Total

Table: Fused QKV projection total estimated latency (Ling, decode step, T3K)

| Component | Count | Per-op latency | Subtotal |
|---|---|---|---|
| Fused matmul | 1 | 10–35 µs | 10–35 µs |
| All-reduce | 1 | 3–8 µs | 3–8 µs |
| **Fused total** | | | **13–43 µs** [ESTIMATE] |

## Expected Savings

Table: Latency savings from fusion (Ling, decode step, T3K, num_links=1)

| Metric | Unfused | Fused | Savings |
|---|---|---|---|
| Matmul latency | 14–48 µs | 10–35 µs | 4–13 µs |
| CCL latency | 15–30 µs | 3–8 µs | 12–22 µs |
| **Total** | **29–78 µs** | **13–43 µs** | **16–35 µs** [ESTIMATE] |
| **Speedup** | 1× | | **~1.8–2.2×** [ESTIMATE] |

The speedup range is derived directly from the table bounds: best case 29/13 ≈ 2.2×, worst case 78/43 ≈ 1.8×. The majority of the savings (roughly 60–70%) comes from eliminating 4 of the 5 CCL operations. The matmul dispatch savings are secondary. This is a meaningful result: **CCL overhead, not compute, is the dominant cost being eliminated by fusion**.

## Where Does the Remaining Bottleneck Lie?

After fusion, the QKV projection time breaks down approximately as:

```
t_fused_total ≈ t_matmul + t_all_reduce
              ≈ (10–35 µs) + (3–8 µs)
```

The fused matmul accounts for roughly **75–80%** of the remaining time [ESTIMATE]. This matmul is **memory-bandwidth-bound**: the arithmetic intensity of a `(1, 4096) × (4096, 384)` product is extremely low (approximately `1.0 FLOPs/byte` of weight data loaded), far below the Tensix hardware's compute-to-memory ratio. Increasing the number of Tensix cores used does not help; what matters is DRAM read bandwidth.

The all-reduce accounts for approximately **20–25%** of remaining time and is **synchronization-latency-bound**, not bandwidth-bound (the payload is only ~1.44 KB total per chip). Increasing `num_links` beyond 1 offers minimal benefit for this specific operation at batch=1.

To reduce overall QKV projection latency further, the primary lever is **reducing the effective memory bandwidth requirement**, for example by:

- Using weight quantization (int8/fp8) to reduce DRAM read volume by 2×–4×
- Increasing batch size so the matmul becomes more compute-bound (at batch ≥ 32, the matmul shifts toward compute-bound territory [ESTIMATE])
- Exploring alternative weight layouts that improve DRAM prefetch efficiency

## Measuring This Split Using TTNN Op Timers (Preview of Chapter 7)

TTNN provides a profiling interface that can measure per-op device execution time at microsecond resolution. To isolate the matmul and all-reduce contributions:

```python
import ttnn

# Enable op-level profiling
ttnn.enable_program_cache()
ttnn.device.enable_profiling(device)

# Run one decode step
output = model.forward(input_ids)

# Dump op timing report
ttnn.device.dump_device_profiler(device, output_path="profile_ch2.csv")
```

The CSV output contains one row per dispatched op with columns including `op_name`, `device_start_cycle`, `device_end_cycle`, and `duration_us`. Filtering for rows where `op_name` contains `"Matmul"` or `"AllReduce"` isolates the two components analyzed above.

Chapter 7 (`profiling_and_tooling.md`) provides a complete walkthrough of this workflow, including how to interpret the cycle-accurate timestamps and correlate them with the CCL trace on the Ethernet fabric. The key numbers to look for are:

- The `duration_us` for `TTNNLinearIColShardedWAllReduced`'s matmul sub-op
- The `duration_us` for the subsequent `AllReduce` sub-op
- Whether the matmul duration tracks with DRAM bandwidth or compute throughput as batch size is varied

Empirically validating the [ESTIMATE] figures above is the highest-priority measurement task before committing to any optimization changes (see Chapter 7).

---

**Next:** [num_links Tuning](./num_links_tuning.md)
