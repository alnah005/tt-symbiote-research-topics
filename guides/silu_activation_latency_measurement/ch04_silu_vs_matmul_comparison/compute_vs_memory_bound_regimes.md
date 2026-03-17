# Compute-Bound vs. Memory-Bound Regimes

This file walks through the transition from memory-bound to compute-bound execution for batched matmul as token count increases, contrasts it with SiLU's permanently memory-bound behavior, and produces a practical decision table for fusion.

---

## 1. Matmul Regime Transition as Batch Size Grows

For a gate_proj matmul of shape `[M, 4096] × [4096, 8192]` on Wormhole B0:

- **M = 1 (decode, single token):** AI ≈ 1 FLOP/byte. The device spends almost all time loading the 4096×8192 weight matrix from DRAM. FPU utilization is negligible. Throughput is ~1/437 of peak FPU = ~0.3 TFLOP/s effective.
- **M = 8:** AI ≈ 8 FLOP/byte. Still deeply memory-bound. 8 tokens reuse each weight element 8 times; effective throughput ~2.4 TFLOP/s.
- **M = 32:** AI ≈ 31 FLOP/byte. Memory-bound, but less severely. Effective throughput ~9.3 TFLOP/s. Runtime scales roughly linearly with M because weights are still the dominant data movement cost.
- **M = 128:** AI ≈ 121 FLOP/byte. FPU starts to constrain. The weight-load cost is amortized over 128 token rows; compute begins to contribute to the bottleneck. Effective throughput ~36 TFLOP/s.
- **M = 512:** AI ≈ 431 FLOP/byte. Approaching the ridge point (437 FLOP/byte). The matmul is near-compute-bound; runtime grows sub-linearly with M.
- **M = 1024:** AI ≈ 745 FLOP/byte. At or above the ridge point. The matmul is compute-bound; further increases in M yield only proportional increases in runtime with no additional DRAM pressure.

The key transition: **matmul crosses into compute-bound territory somewhere between M = 512 and M = 1024 for this weight shape.** The exact threshold depends on weight shape, tile mapping, and whether weights are in DRAM or L1.

---

## 2. SiLU's Permanently Memory-Bound Behavior

SiLU has no weight tensor. Its only data movement is the activation tensor:

- Read input: `num_tokens × ffn_dim × 2` bytes (BF16)
- Write output: `num_tokens × ffn_dim × 2` bytes (BF16)
- Total bytes per call: `4 × num_tokens × ffn_dim`

At `num_tokens=128, ffn_dim=8192`:
- Data movement: `4 × 128 × 8192` = 4,194,304 bytes ≈ 4 MB
- At 300 GB/s DRAM bandwidth: lower bound latency ≈ 4e6 / 300e9 ≈ 13 µs
- Actual measured latency is higher (40–90 µs) due to tile dispatch overhead, grid utilization, and DRAM non-uniformity across banks

SiLU latency scales **linearly** with `num_tokens × ffn_dim` at all practical sizes. There is no compute saturation effect and no weight reuse to amortize.

This stands in direct contrast to matmul: as M increases, matmul amortizes the weight load cost and eventually saturates the FPU. SiLU never amortizes anything — each additional token adds a proportional amount of data movement.

---

## 3. Consequence: Diverging Scaling Curves

As `num_tokens` increases from 1 to 1024, the ratio of SiLU latency to gate_proj matmul latency decreases monotonically:

```
num_tokens    SiLU scaling     Matmul scaling     SiLU / matmul ratio (approx)
----------    ------------     --------------     ----------------------------
1             1×               1×                 ~30%
8             8×               ~7×                ~27%
32            32×              ~24×               ~20%
128           128×             ~55×               ~7%
512           512×             ~140×              ~5%
1024          1024×            ~230×              ~4%
```

Matmul scaling factors above are relative to M=1 and reflect the combined effect of linear growth in FLOPs and sub-linear growth in DRAM access as weights get reused.

---

## 4. Fusion Threshold Analysis

A fused kernel that computes `gate_proj matmul + SiLU (+ optional element-wise multiply)` in a single pass eliminates the separate SiLU (and elem_mul) kernel dispatch and the intermediate DRAM write/read round-trip.

The maximum possible speedup from fusion is bounded by the SiLU + elem_mul fraction of total FFN time:

| num_tokens | SiLU + elem_mul fraction of FFN | Max speedup from fusion |
|---|---|---|
| 1 | ~35–50% | up to 1.7× FFN |
| 8 | ~30–45% | up to 1.8× FFN |
| 16 | ~25–38% | up to 1.6× FFN |
| 32 | ~20–30% | up to 1.4× FFN |
| 64 | ~12–20% | up to 1.25× FFN |
| 128 | ~5–10% | up to 1.11× FFN |
| 256+ | < 5% | < 1.05× FFN |

At `num_tokens >= 64`, fusion eliminates less than 20% of FFN time. The actual speedup will be lower because the fused kernel introduces implementation overhead and may not fully saturate bandwidth on the fused path.

**Practical fusion threshold: num_tokens < 16.** Above this point, the absolute latency reduction from SiLU fusion falls below the typical variance in device kernel scheduling (5–15 µs) and is unlikely to be measurable in end-to-end throughput benchmarks.

---

## 5. SiLU Latency Linear Scaling: Verification

To confirm linear scaling of SiLU latency, plot `DEVICE KERNEL DURATION [ns]` against `num_tokens × ffn_dim`. A memory-bound kernel will produce a straight line. Any deviation indicates:

- Sub-linear: partial L1 caching (tensor fits in L1 at small sizes, switches to DRAM at large sizes)
- Super-linear: DRAM bank conflicts increasing with tensor size, or grid under-utilization at small sizes followed by full-grid utilization at large sizes

If you observe a kink in the scaling curve, identify the `num_tokens × ffn_dim` product at which it occurs and verify the tensor size relative to L1 capacity per core (≈ 1.5 MB per Tensix core on Wormhole).

---

## 6. Summary Decision Table

| num_tokens | SiLU regime | Matmul regime | SiLU / gate_proj ratio | Fusion recommendation |
|---|---|---|---|---|
| 1–8 | Memory-bound | Memory-bound | 20–40% | Fusion likely beneficial; both ops compete for same DRAM bandwidth |
| 9–15 | Memory-bound | Memory-bound | 15–25% | Fusion may be beneficial; measure end-to-end FFN before committing |
| 16–63 | Memory-bound | Transitioning | 8–15% | Fusion marginal; benefit depends on fused kernel quality |
| 64–127 | Memory-bound | Near compute-bound | 5–10% | Fusion unlikely to produce measurable end-to-end speedup |
| 128+ | Memory-bound | Compute-bound | < 5% | SiLU latency negligible; do not fuse; focus on matmul tiling instead |

---

## 7. Implications for MoE Optimization

In a Mixture of Experts forward pass, the dominant use case is decode: one or a few tokens routed to each expert at inference time. This places the workload firmly in the `num_tokens < 16` region where SiLU fusion is most valuable.

Prefill workloads that process long prompts (128–2048 tokens) are compute-bound at the matmul level. SiLU is irrelevant to prefill optimization; effort there should go into matmul tiling, weight prefetching, and expert batching strategies.

---

## Summary

- SiLU latency scales linearly with tensor size at all practical token counts; it is always memory-bound.
- Matmul transitions from memory-bound to compute-bound between M = 512 and M = 1024 for a 4096×8192 weight shape on Wormhole B0.
- The practical fusion threshold is `num_tokens < 16`; above 64 tokens, SiLU latency is negligible relative to total FFN time.
- In MoE decode (the primary workload), SiLU and elem_mul together represent 35–50% of FFN time and are the correct target for kernel fusion.

---

**Next:** [Chapter 5 — Fused Activation Strategies](../ch05_fused_activation_strategies/index.md)
