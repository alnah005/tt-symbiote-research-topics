# When Fusion Helps

This document provides a decision framework for determining whether fusing SiLU into the gate_proj matmul (Pattern A) is worth implementing for a given combination of token count, hidden dimension, and dtype. It covers the decode regime, the prefill regime, T3K multi-chip expert parallelism, and the anti-pattern to avoid.

---

## The Core Question

Fusing SiLU eliminates one kernel dispatch and one L1 read/write round-trip. The value of that savings depends entirely on what fraction of total FFN latency the standalone SiLU kernel occupies. When SiLU is a large fraction, fusion is a first-order optimization. When SiLU is negligible, fusion is still correct but delivers diminishing returns.

The SiLU-to-gate_proj matmul latency ratio is the primary signal. *See Chapter 4, `latency_ratio_by_shape.md` for the full measured table.*

---

## Decision Framework

Use the table below to identify your regime based on `num_tokens`, `hidden_dim` (`d_ff`), and `dtype`. The "Action" column gives the direct recommendation.

| `num_tokens` | `d_ff` | `dtype` | SiLU / gate_proj ratio | Regime | Action |
|---|---|---|---|---|---|
| 1–8 | 2048 | BF16 | 25–40% | Decode, memory-bound | Apply Pattern A; fusion is high value |
| 1–8 | 8192 | BF16 | 15–30% | Decode, memory-bound | Apply Pattern A; fusion is high value |
| 9–16 | 2048–8192 | BF16 | 10–20% | Decode boundary | Apply Pattern A; marginal but worthwhile |
| 17–63 | 2048–8192 | BF16 | 5–10% | Transition zone | Pattern A acceptable; matmul tuning is higher priority |
| 64–128 | 2048–8192 | BF16 | 3–6% | Prefill entry | Fusion optional; focus on matmul program config |
| 128–2048 | 2048–8192 | BF16 | 4–8% at 128 tokens, shrinking to < 5% above ~256 tokens | Prefill, compute-bound | Fusion low priority; standalone `ttnn.silu` acceptable |
| 1–16 | 2048–8192 | BFP8 | 15–35% | Decode, BFP8 | Apply Pattern A with `activation_dtype=ttnn.bfloat8_b`; validate PCC |

The `num_tokens` thresholds are derived from the roofline crossover identified in Chapter 4: the gate_proj matmul transitions from memory-bound to compute-bound near `num_tokens = 32–64` for `d_ff = 2048` on Wormhole B0 (80 Tensix cores, ~300 GB/s DRAM bandwidth, ~131 TFLOP/s BF16 peak). *See Chapter 2, `tensix_compute_engine.md` for hardware constants.*

---

## Decode Phase Analysis (1–16 Tokens)

In the decode phase, the number of tokens processed by each expert FFN is bounded by `batch_size × top_k`. For most MoE inference configurations (batch size 1–4, top-k 2–4), the per-expert token count falls in the range of 1–16 tokens.

**Why both SiLU and gate_proj matmul are memory-bound at decode:**

At `num_tokens = 8`, `d_ff = 2048`, BF16, the gate_proj matmul computes an `[8, 2048] × [2048, 2048]` product. The FLOP count is `2 × 8 × 2048 × 2048 = 67M FLOPs`. The weight tensor size is `2048 × 2048 × 2 = 8 MB`. The arithmetic intensity is approximately `67M / 8M = 8.4 FLOP/byte`, which is well below Wormhole B0's compute-to-memory bandwidth ratio of approximately 437 FLOP/byte (`131 TFLOP/s ÷ 300 GB/s`). The matmul is memory-bound.

SiLU over the same `[8, 2048]` output tensor has arithmetic intensity of approximately 0.5 FLOP/byte — even more memory-bound. Both operations are constrained by the same DRAM bandwidth ceiling.

**The practical consequence:** At decode, the SiLU kernel executes in approximately the same time as a proportionally sized DRAM read/write. This time is 15–40% of the gate_proj matmul time because SiLU reads and writes the same activation tensor that the matmul just produced. Fusion eliminates:

1. One kernel dispatch overhead (estimated 1–3 µs on Wormhole B0 for a SFPU-only kernel).
2. One L1 read/write round-trip: the `[num_tokens, d_ff]` intermediate tensor is never materialized in L1 as a separate allocation. For `num_tokens=8`, `d_ff=2048`, BF16, this is `8 × 2048 × 2 = 32 KB` of avoided L1 traffic.

**Decode regime rule:** Apply Pattern A whenever `batch_size × top_k ≤ 16`. This is the condition under which fusion provides more than 5% end-to-end FFN speedup.

> **Tip:** Even at `num_tokens = 1` (single-token greedy decode), Pattern A is valid and beneficial. The gate_proj matmul is a matrix-vector product; SiLU over a 2048-element vector is a few microseconds of pure DRAM bandwidth consumption. Fusing it saves the full standalone SiLU dispatch cost.

---

## Prefill Phase Analysis (64–2048 Tokens)

In the prefill phase, all tokens in the input prompt are processed simultaneously. The per-expert token count equals the number of prompt tokens routed to that expert, which can range from tens to hundreds depending on router sparsity and sequence length.

**Why fusion benefit shrinks at prefill:**

At `num_tokens = 128`, `d_ff = 2048`, BF16, the gate_proj matmul FLOP count is `2 × 128 × 2048 × 2048 = 1.07G FLOPs`. The arithmetic intensity is approximately `134 FLOP/byte`. This exceeds the memory-bound regime threshold; the matmul is now partially compute-bound, and its latency grows faster than linearly with token count. SiLU latency, however, remains memory-bound and grows linearly with the activation tensor size. The ratio is 4–8% at 128 tokens and shrinks below 5% above approximately 256 tokens.

**Prefill conclusion:** Standalone `ttnn.silu` (Pattern B) is acceptable in the prefill regime. The latency savings from Pattern A are real but small (1–3% of FFN time). Engineering effort is better invested in:

- Matmul program config tuning (block sizes, core grid assignment).
- Sharding strategy for large hidden dimensions.
- Reducing `down_proj` compute overhead.

> **Tip:** In a mixed-batch serving system that handles both decode and prefill requests, implement Pattern A for decode and Pattern B for prefill using conditional logic on `num_tokens`. The overhead of branching is negligible compared to the per-expert FFN latency.

---

## T3K Multi-Chip Context

T3K distributes MoE experts across 8 Wormhole B0 chips connected via Ethernet. Under expert parallelism, each chip hosts a subset of the total experts. When a batch of tokens is routed, each chip receives only the tokens assigned to its local experts.

**Key insight:** Expert parallelism does not increase per-chip token count. If a model has 64 experts and T3K places 8 experts per chip, a token batch of 64 tokens (batch × top_k = 64) distributes as approximately 8 tokens per chip per expert on average. The per-chip token count remains in the decode-phase regime.

**Consequence for fusion:** The decode-phase analysis applies on T3K. Pattern A is the correct choice for expert FFN blocks running under T3K expert parallelism. The `batch_size × top_k ≤ 16` rule applies per chip, not globally.

*See Chapter 4, `compute_vs_memory_bound_regimes.md` for the per-chip roofline derivation.*

> **Tip:** On T3K, verify that the per-chip token count after expert routing matches the expected decode-regime shape before concluding that prefill-phase analysis applies. Router load imbalance can cause individual chips to receive more tokens than the average prediction.

---

## Anti-Pattern: Fusion When the Matmul Output Has Multiple Consumers

`activation="silu"` on `ttnn.matmul` computes SiLU in-place on the matmul output tile buffer. The result is the SiLU-activated tensor; the raw matmul output (before SiLU) is not available as a separate tensor.

**The anti-pattern** occurs when the gate_proj output needs to be consumed by both the SiLU path and another operation downstream. In the standard SwiGLU FFN, this does not arise — the gate_proj output has exactly one consumer (SiLU). However, it can arise in:

- Residual connections that tap the gate_proj intermediate.
- Custom MoE router implementations that reuse expert FFN intermediate activations for auxiliary losses.
- Debugging or profiling code that saves intermediate tensors for inspection.

> **Warning:** Do not apply `activation="silu"` (or `fused_activation` in a program config) when the pre-SiLU gate_proj output is consumed by any op other than the SiLU itself. Doing so will silently produce incorrect results for the second consumer, because the tensor it reads will contain the SiLU-activated values rather than the raw matmul output. There is no TTNN-level error for this misuse.

If the raw matmul output is needed separately, use Pattern B (unfused) and call `ttnn.silu` on the output explicitly after saving or computing with the raw value.

---

## Summary

| Condition | Recommendation |
|---|---|
| `num_tokens ≤ 16`, any `d_ff`, BF16 or BFP8 | Apply Pattern A; fusion is high value |
| `16 < num_tokens < 64`, any `d_ff`, BF16 | Pattern A acceptable; diminishing returns |
| `num_tokens ≥ 64`, BF16, prefill | Pattern B acceptable; focus on matmul tuning |
| T3K expert parallelism, any global batch size | Evaluate per-chip token count; decode analysis likely applies |
| gate_proj output has multiple consumers | Do not fuse; use Pattern B |

---

## Next Steps

Continue to [`configuration_recommendations.md`](configuration_recommendations.md) for the specific TTNN API calls, program config settings, and Tracy profiler verification steps for each of the three production scenarios covered in this chapter.
