# `num_links` Tuning for the QKV All-Reduce

## What `num_links=1` Means Physically

Each Wormhole n300 chip on a T3K board is connected to its neighbors via 16 physical Ethernet ports, each running at 100 Gb/s (12.5 GB/s) bidirectional. These ports are exposed to the TTNN CCL runtime as a pool of available links. The parameter `num_links` passed to `ttnn.all_reduce` (and other CCL primitives) selects how many of these physical Ethernet links are allocated to carry the collective's traffic.

With `num_links=1`:

- A **single** 100 Gb/s Ethernet port is used on each chip for both the reduce-scatter and all-gather sub-phases of the all-reduce.
- The remaining 15 ports on each chip are available for other concurrent CCL operations (e.g., from a different layer running in a pipeline), or are simply idle.
- The effective peak bidirectional bandwidth available to this all-reduce is **12.5 GB/s per chip per direction** [ESTIMATE].

With `num_links=2`:

- Two 100 Gb/s ports are bonded, yielding an effective bidirectional bandwidth of **25 GB/s per chip per direction** [ESTIMATE].
- The CCL runtime interleaves data across the two links at the chunk level, so the improvement is approximately linear for large payloads.

With `num_links=4`:

- Four ports are bonded, yielding **50 GB/s per chip per direction** [ESTIMATE].
- At this point the CCL becomes a meaningful consumer of the chip's Ethernet port budget, leaving only 12 ports for concurrent operations.

T3K's total raw Ethernet capacity is ~1,600 GB/s [ESTIMATE] (8 chips × 16 ports × 12.5 GB/s per port; see Chapter 1, `hardware_context.md`). A single CCL ring operation with `num_links=1` uses ~200 GB/s bidirectional [ESTIMATE] (8 chips × 1 link × 12.5 GB/s × 2 directions), which is only 1/8 of the total raw capacity. A single CCL operation thus uses only 1/8 of each chip's available Ethernet port budget.

## Physical Link Topology on T3K

The 1×8 logical mesh maps chips to a ring topology for CCL operations. In the ring all-reduce, data traverses the ring in both clockwise and counterclockwise directions simultaneously (bidirectional ring). Each hop in the ring uses one Ethernet port per direction. With `num_links=1`, this means the ring all-reduce uses exactly **1 Ethernet port per chip in each direction**, for a total of 2 ports per chip per all-reduce.

For the QKV all-reduce specifically, the ring carries the following data per hop:

```
Reduce-scatter phase: 384 / 8 × 2 bytes = 96 bytes per hop
All-gather phase:     48 × 2 bytes × 7 = 672 bytes total received
```

These are extraordinarily small quantities. Even at `num_links=1`, the 100 Gb/s link is capable of transferring 96 bytes in approximately **7.7 nanoseconds** [ESTIMATE], which is far below the synchronization and firmware overhead floor of **~1–3 µs per hop** [ESTIMATE]. The link is idle for essentially the entire duration of the CCL operation.

## Sensitivity Analysis: Latency vs. `num_links`

### At Ling's Hidden Dimension (H=4096, C=3072, batch=1)

The all-reduce payload per chip is ~1.44 KB total. As established in `fusion_mechanics.md`, this is a **latency-dominated** operation: the bottleneck is firmware synchronization (chip-to-chip handshake, kernel dispatch) rather than the time to transmit bits.

The synchronization overhead per CCL operation scales as:

```
t_sync(L) ≈ t_sync_base + t_per_link × L
```

where `t_sync_base` is the fixed cost (estimated **2–4 µs** [ESTIMATE]) and `t_per_link` is the marginal cost of coordinating an additional link (estimated **0.5–1 µs** per additional link [ESTIMATE]).

The transfer time scales as:

```
t_transfer(L) = payload / (L × 12.5 GB/s)
```

For the QKV all-reduce:

```
t_transfer(L=1) = 1.44 KB / 12.5 GB/s ≈ 0.000115 µs
t_transfer(L=2) = 1.44 KB / 25.0 GB/s ≈ 0.000058 µs
t_transfer(L=4) = 1.44 KB / 50.0 GB/s ≈ 0.000029 µs
```

Transfer time is negligible at all link counts. Total estimated all-reduce latency:

Table: Estimated QKV all-reduce latency vs. num_links (Ling, H=4096, C=3072, batch=1)

| `num_links` | Transfer time | Sync overhead | Total all-reduce latency |
|---|---|---|---|
| 1 | ~0.0001 µs | 3–6 µs | **3–6 µs** [ESTIMATE] |
| 2 | ~0.0001 µs | 3.5–7 µs | **3.5–7 µs** [ESTIMATE] |
| 4 | ~0.0001 µs | 4–8 µs | **4–8 µs** [ESTIMATE] |

**Result:** Increasing `num_links` from 1 to 2 or 4 provides **zero improvement** for the QKV all-reduce at batch=1 and hidden_size=4096. In fact, higher `num_links` values may slightly *increase* latency due to the marginal synchronization overhead of coordinating additional links. The payload is too small for the bandwidth gain to offset the coordination cost.

### At Larger Batch Sizes

At batch size `B`, the all-reduce payload scales linearly:

```
payload(B) = B × 384 × 2 bytes  [reduce-scatter input per chip, bfloat16]
```

The crossover point where bandwidth becomes the bottleneck (and increasing `num_links` helps) occurs when:

```
payload / (L × 12.5 GB/s) > t_sync_base
```

For `L=1` and `t_sync_base ≈ 3 µs`:

```
B_crossover = (3 µs × 12.5 GB/s) / (384 × 2 bytes)
            = (3 × 10^-6 s × 12.5 × 10^9 B/s) / 768 B
            = 37,500 / 768
            ≈ 49
```

So at **batch ≥ ~49** [ESTIMATE], the QKV all-reduce becomes bandwidth-bound at `num_links=1`, and increasing `num_links` to 2 would provide a meaningful speedup (approximately 2×). At batch=1 (the Ling decode step), this threshold is not reached.

Table: QKV all-reduce latency vs. batch size and num_links (estimated, H=4096)

| Batch | Payload/chip | `num_links=1` latency | `num_links=2` latency | `num_links=4` latency |
|---|---|---|---|---|
| 1 | 768 B | 3–6 µs | 3.5–7 µs | 4–8 µs |
| 16 | 12 KB | 4–7 µs | 3.5–6 µs | 3.5–6 µs |
| 32 | 24 KB | 5–8 µs | 4–6 µs | 3.5–5.5 µs |
| 64 | 48 KB | 7–11 µs | 5–7 µs | 4–6 µs |
| 128 | 96 KB | 11–18 µs | 7–11 µs | 5–8 µs |

All values [ESTIMATE]. The shift from synchronization-dominated to bandwidth-dominated behavior begins around batch=32–64.

### At Larger Hidden Sizes

For H≥8192, see the revisit-condition table in "Conditions Under Which This Should Be Revisited" below.

## Recommendation

**Use `num_links=1` for the QKV all-reduce in the Ling decode configuration.**

The rationale:

1. The all-reduce payload is ~1.44 KB per chip at batch=1, seq_len=1 — firmly in the synchronization-latency regime.
2. Transfer time is less than 1 ns, while the synchronization floor is ~3–6 µs. No amount of added bandwidth can reduce this floor.
3. Increasing `num_links` to 2 or 4 adds marginal link-coordination overhead without reducing the dominant synchronization cost.
4. Using only 1 link reserves the other 15 Ethernet ports for concurrent CCL operations in other pipeline stages, which may run simultaneously in a multi-chip pipeline.

### Conditions Under Which This Should Be Revisited

The recommendation changes if any of the following conditions arise:

**Batch size increases above ~32.** At batch ≥ 32, the all-reduce payload grows large enough that bandwidth becomes the bottleneck at `num_links=1`. In this regime, `num_links=2` is expected to yield a roughly 1.5–1.9× speedup on the all-reduce [ESTIMATE]. At batch ≥ 64, `num_links=4` may provide additional benefit.

**Hidden size increases substantially.** If `hidden_size` is scaled to 16384 or beyond, revisit the crossover batch size calculation above with the new `C` value.

**A new T3K firmware version changes CCL synchronization overhead.** If Tenstorrent releases a TTNN version that reduces the per-CCL synchronization floor significantly (e.g., to < 1 µs), the crossover batch size drops, and `num_links=1` remains optimal for a wider operating range.

**The QKV projection is pipelined with other CCL operations.** If the model is restructured so that the QKV all-reduce runs concurrently with another CCL op on the same chip (e.g., via multi-stream dispatch), the port sharing budget becomes relevant. In that case, the assignment of links across concurrent CCL ops should be optimized jointly rather than in isolation.

**Empirical measurement disagrees with this analysis.** This entire analysis rests on estimated synchronization overheads and assumed link behaviors. The first action after any deployment should be to profile the all-reduce using TTNN op timers (see `latency_savings_analysis.md` and Chapter 7) and compare measured all-reduce duration across `num_links` values. If the measured data shows a bandwidth-dominated profile at batch=1, the crossover model above should be recalibrated.

## Summary

Table: num_links recommendation summary

| Condition | Recommended `num_links` | Reason |
|---|---|---|
| Ling decode, batch=1, H=4096 | **1** | Synchronization-latency dominated; more links add overhead |
| Ling prefill or batch=32–64 | **2** | Bandwidth starts to matter; 2 links approximately halves transfer time |
| Large batch (≥64) or H≥8192 | **4** | Bandwidth is clearly dominant; 4 links justified |
| Concurrent CCL ops on same chip | **Optimize jointly** | Port budget must be shared across ops |

---

**Next:** [Chapter 3 — Host Round-Trip Replication](../ch3_host_roundtrip_replication/index.md)
