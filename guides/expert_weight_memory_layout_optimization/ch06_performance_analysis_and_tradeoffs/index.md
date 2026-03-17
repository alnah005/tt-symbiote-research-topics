# Chapter 6: Performance Analysis and Trade-offs

## Overview

Chapter 5 established the tile alignment rules that make a shard configuration valid. This chapter asks the next question: once you have a valid DRAM-sharded configuration, how much does it actually help, and under what conditions does the benefit disappear or reverse?

The answer depends on the inference regime. DRAM-sharded expert weights improve latency by eliminating NoC contention and giving each Tensix core a deterministic, short path to its assigned DRAM bank. That benefit is large when the kernel is memory-bound — as it is in decode — and negligible or negative when the kernel is already compute-bound — as it can be in large-batch prefill.

This chapter quantifies the bandwidth improvement, characterizes the one-time cost of resharding weights at load time, and provides a trade-off matrix that maps each combination of batch size and inference phase to a recommended weight layout.

> **Note:** The bandwidth figures in this chapter are indicative estimates derived from Wormhole B0 architectural parameters and the roofline model from Chapter 4. Chapter 7, `benchmark_methodology.md`, provides the measurement harness to validate or refine these estimates on a specific firmware and TTNN version.

---

## Learning Objectives

1. **Estimate the bandwidth efficiency gap** between interleaved and DRAM-sharded layouts and explain its architectural cause.
2. **Apply the decode-regime rule of thumb** (`batch_size × top_k ≤ 16`) to decide whether DRAM sharding is the highest-leverage optimization.
3. **Explain why shard setup overhead is negligible** for deployment when resharding is performed at model load time.
4. **Use the trade-off matrix** to select the correct weight layout for each of the four canonical inference regimes.
5. **Identify the conditions under which sharding adds overhead without benefit**, including large-batch prefill and L1 staging requirements.

---

## Decision Table: DRAM-Sharded vs Interleaved

Use the following table as a first-pass filter before committing to a weight memory layout. The full analysis behind each recommendation is in `bandwidth_gain_analysis.md` and `tradeoff_matrix.md`.

| Regime | `batch_size × top_k` | Primary bottleneck | Recommended layout | Expected latency delta |
|---|---|---|---|---|
| Decode, small batch | ≤ 16 | Memory bandwidth | **DRAM-sharded** | −30 to −50% (improvement) |
| Decode, large batch | 17–64 | Memory bandwidth (moderate) | DRAM-sharded | −10 to −25% |
| Prefill, small batch | 64 < batch_size × top_k ≤ 256 | Mixed | DRAM-sharded (marginal) | −5 to −10% |
| Prefill, large batch | effective_M > 256 | Compute | **Interleaved** | 0 to +5% (no benefit, possible overhead) |

A negative latency delta means the sharded layout is faster. A positive delta means interleaved is preferable.

> **Tip:** For Mixtral 8x7B with `top_k=2`, the boundary is `batch_size ≤ 8`. For Qwen 235B-A22B with `top_k=8`, the boundary is `batch_size ≤ 2`. Decode serving with small batches is almost always in the high-benefit zone.

---

## Prerequisites

| Chapter | Topics Required |
|---|---|
| Chapter 3 | Expert weight shapes; Mixtral and Qwen model parameters |
| Chapter 4 | NoC contention model; roofline analysis; arithmetic intensity for decode vs prefill |
| Chapter 5 | Valid shard configurations; tile alignment rules |

---

## Chapter Structure

| File | Contents |
|---|---|
| [`bandwidth_gain_analysis.md`](./bandwidth_gain_analysis.md) | Theoretical peak vs indicative effective bandwidth; decode vs prefill bottleneck analysis; the `batch_size × top_k ≤ 16` rule |
| [`shard_setup_overhead.md`](./shard_setup_overhead.md) | `ttnn.to_memory_config` cost at load time; recommended load-time resharding pattern; program cache stability |
| [`tradeoff_matrix.md`](./tradeoff_matrix.md) | Four-regime comparison table; when sharding hurts; L1 pressure interaction; T3K multi-chip compounding |

---

## Key Constants (Wormhole B0) — Performance Context

Hardware constants (DRAM bandwidth, ridge point, tile sizes): see Chapter 4, `index.md`.

---

## Next Steps

Read `bandwidth_gain_analysis.md` for the quantitative case behind the decision table above, then `shard_setup_overhead.md` to understand the one-time cost of resharding, and finally `tradeoff_matrix.md` for the complete per-regime guidance including T3K multi-chip interactions.
