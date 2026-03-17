# Bandwidth Gain Analysis

## Theoretical Peak DRAM Bandwidth on Wormhole B0

Wormhole B0 has 6 DRAM controllers, each driving 2 GDDR6 banks. The aggregate theoretical peak bandwidth across all 6 controllers is approximately **300 GB/s** (indicative; exact figure depends on firmware GDDR6 speed-grade configuration).

This peak is the ceiling against which all measured effective bandwidth is compared. No memory-bound kernel should be designed to assume it will reach this ceiling; practical targets under real workloads are lower due to factors discussed below.

> **Note:** All bandwidth figures in this chapter are indicative estimates based on Wormhole B0 architectural parameters and the Chapter 4 roofline model. Chapter 7, `benchmark_methodology.md`, provides the instrumented measurement harness to produce empirical figures for your specific TTNN version and firmware.

---

## Effective Bandwidth: Interleaved Layout

Under standard `ttnn.DRAM_MEMORY_CONFIG` (interleaved), weight tiles are distributed round-robin across all 12 GDDR6 banks. When 80 Tensix cores issue simultaneous read requests for weight tiles, the NoC links leading to each DRAM controller column become contested: multiple cores compete to read from the same controller at the same time.

See Chapter 4, `bandwidth_estimation.md` for the interleaved and sharded bandwidth tables.

The decode regime is the worse case for interleaved access. With M=1, each Tensix core issues a single tile's worth of reads per matmul call. There are few in-flight requests to hide NoC latency, so each core effectively serializes on DRAM access. The pipeline depth is shallow and NoC stalls are not amortized.

**Why the gap exists:** Chapter 4, `noc_and_dram_access.md` established that interleaved round-robin placement causes two independent sources of inefficiency:

1. **Cross-controller fan-out:** A single core fetching a weight matrix touches all 6 DRAM controller columns in rotation. At M=1, this is unavoidable — there is no locality to exploit.
2. **NoC hotspot formation:** When all 80 cores fetch from interleaved weights simultaneously, the NoC links near popular DRAM columns become saturated, while other links are idle. The effective aggregate bandwidth is limited by the bottleneck link, not by the sum of all links.

---

## Effective Bandwidth: DRAM-Sharded Layout

DRAM-sharded layout eliminates cross-controller fan-out by assigning each contiguous shard of the weight tensor to a specific DRAM bank range. A Tensix core assigned to shard N always reads from the same DRAM bank; the NoC path is deterministic and short.

The target range is approximately 85–95% of the ~300 GB/s theoretical peak. The residual gap arises from bank activation overhead, L1 write latency, and DMA scheduling overhead; see Chapter 4, `bandwidth_estimation.md` for the full explanation.

> **Tip:** To minimize the third overhead, ensure each shard contains at least 4 tile-rows worth of data (128 elements × shard_W). Very small shards — fewer than 2 tile-rows — increase DMA scheduling overhead relative to payload size. See Chapter 5, `shard_shape_alignment_rules.md` Rule 4 for the grid-size constraint that determines shard height.

---

## The Decode Regime: Memory-Bound and High-Impact

Arithmetic intensity = M×N/(M+N) (derived in Chapter 4); at decode, intensity ≈ 1 FLOP/byte; at large prefill, intensity approaches N FLOP/byte.

In single-token decode, the expert FFN computes a matmul of shape `[M, K] × [K, N]` where M=1 (one activation token) and K, N are the weight dimensions. The roofline ridge point on Wormhole B0 is approximately 437 FLOP/byte. At 1 FLOP/byte, the decode expert matmul is **437× below the ridge point** — deeply memory-bound. Every nanosecond spent waiting for DRAM directly translates to latency; compute throughput is irrelevant.

For Mixtral 8x7B (`d_model=4096`, `d_ff=14336`, `num_experts=8`, `top_k=2`), the weight bytes read per decode step for one expert's gate projection:

```
weight_bytes = 4096 × 14336 × 2 bytes (BF16)
             = 117,440,512 bytes
             ≈ 112 MB per expert per projection
```

With `top_k=2`, two experts are active per token. Total weight bytes for gate+up+down projections per decode step (all three projections):

```
total_weight_bytes ≈ 2 experts × 3 projections × 112 MB
                   ≈ 672 MB per token per layer
```

At interleaved effective bandwidth of ~200 GB/s, this takes approximately 3.4 ms per layer. At DRAM-sharded effective bandwidth of ~270 GB/s, the same weight transfer takes approximately 2.5 ms — a ~25% reduction in weight-transfer time, which directly reduces layer latency when the kernel is memory-bound.

> **Note:** These latency figures are derived from the indicative bandwidth estimates above and assume weight transfer dominates kernel time, which is valid at M=1. Activation transfer and compute overlap are not modeled here. See Chapter 7 for a measurement harness that isolates each component.

---

## The Prefill Regime: Diminishing Returns

In prefill, the activation batch has M > 1 (often M=512 to M=8192 for a long-sequence request). At M=512 the kernel is marginally compute-bound (above the ~437 FLOP/byte ridge point; see Chapter 4, `bandwidth_estimation.md`). Two consequences follow:

1. **DRAM bandwidth is not the bottleneck.** The kernel spends most of its time in Tensix compute units, not waiting for DRAM reads. Improving DRAM bandwidth from ~200 GB/s to ~270 GB/s does not reduce total kernel time if the compute phase is longer than the memory phase.
2. **Shard metadata overhead adds cost with no offsetting gain.** The sharded `MemoryConfig` requires extra TTNN bookkeeping during dispatch (shard boundary calculations, shard-to-core mapping resolution). In interleaved mode this overhead is absent. For a compute-bound kernel, this dispatch overhead can increase total op time slightly.

The practical impact is small — typically 0 to +5% slower with sharding in the large-batch prefill case — but it is a reason not to default to DRAM-sharded for all regimes.

---

## Rule of Thumb: The `batch_size × top_k ≤ 16` Threshold

The boundary between the memory-bound and compute-bound regimes occurs near the roofline ridge point. For an expert FFN matmul, the effective M across the active experts is:

```
effective_M = batch_size × top_k
```

because each of the `top_k` selected experts processes all `batch_size` tokens in parallel. The crossover point (where arithmetic intensity equals the ridge point of ~437 FLOP/byte) occurs near:

```
effective_M_crossover ≈ 437 × N / (N − 437)
                      ≈ 437 × 14336 / (14336 − 437)
                      ≈ 451 for Mixtral (N=14336)
```

**DRAM sharding is most valuable when `batch_size × top_k ≤ 16`.** This rule of thumb is conservative: it keeps the workload well within the memory-bound regime where the bandwidth improvement directly reduces latency. The exact crossover is near `effective_M ≈ 451` for Mixtral-scale weights, but the returns diminish well before that point as the kernel transitions from purely memory-bound to mixed.

Model-specific thresholds:

| Model | `d_ff` | `top_k` | Crossover `effective_M` | Max `batch_size` at threshold |
|---|---|---|---|---|
| Mixtral 8x7B | 14336 | 2 | ~451 | ~225 (but use ≤ 8 for high impact) |
| Qwen 235B-A22B | 2048 | 8 | ~556 | ~56 (but use ≤ 2 for high impact) |

The "`≤ 16`" rule targets the high-impact zone, not the boundary: at `effective_M = 16`, DRAM sharding delivers 30–50% bandwidth improvement; at `effective_M = 100`, it may deliver only 5–10%.

---

## Summary: Sharding Impact by Regime

| Regime | `effective_M` | Arithmetic Intensity | Bandwidth-Bound? | Indicative Sharding Gain |
|---|---|---|---|---|
| Decode, single token | 1–2 | ~1–2 FLOP/byte | Yes (strongly) | 30–50% latency reduction |
| Decode, small batch | 4–16 | ~4–15 FLOP/byte | Yes | 15–30% latency reduction |
| Decode, medium batch | 16–64 | 15–57 FLOP/byte | Yes (moderate) | 5–15% latency reduction |
| Prefill, short sequence | 64–256 | ~57–220 FLOP/byte | Transitioning | 0–10% latency reduction |
| Prefill, long sequence | 512+ | >400 FLOP/byte | Compute-bound | Negligible or negative |

---

**Next:** [`shard_setup_overhead.md`](./shard_setup_overhead.md)
