# Chapter 7 — Performance Analysis and Bottlenecks

This chapter dissects the measured decode performance of Qwen3.5 on P100A Blackhole —
where the 86 ms/token comes from, why only ~6.8% of theoretical throughput is achieved,
and which bottlenecks block each proposed optimization.

## Reading order

1. [`latency_breakdown.md`](./latency_breakdown.md) — component timing table, theoretical peak, efficiency ratio
2. [`sync_overhead.md`](./sync_overhead.md) — host-device synchronisation cost per operation type
3. [`bottleneck_analysis.md`](./bottleneck_analysis.md) — Python dispatch overhead, device utilisation, optimisation paths

## Prerequisites

- Chapter 2 (GatedDeltaNet host recurrence and fused kernel)
- Chapter 3 (GatedAttention RoPE and sync pattern)
- Chapter 5 (MoE routing host sync)

## Key numbers

| Model | Speed | Latency | Efficiency |
|---|---|---|---|
| Qwen3.5-35B-A3B | 11.7 tok/s | 86 ms/token | ~6.8% of 172 tok/s peak |
| Qwen3.5-27B | 6.28 tok/s | ~159 ms/token | ~3.6% of 172 tok/s peak |

---

**Next:** [`latency_breakdown.md`](./latency_breakdown.md)
