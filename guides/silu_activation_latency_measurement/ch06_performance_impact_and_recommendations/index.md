# Chapter 6: Performance Impact and Recommendations

This chapter synthesizes the measurement results, hardware analysis, and fusion strategies from Chapters 3–5 into concrete, actionable configuration recommendations for SiLU and SwiGLU in production MoE inference on Wormhole B0 and T3K. The primary audience is an engineer integrating a SwiGLU-based MoE model (Llama 3, Mixtral, Qwen-MoE, DeepSeek-V3) into a TTNN inference stack who needs to decide whether and how to apply fused activation.

---

## Prerequisites

| Requirement | Details |
|---|---|
| Chapter 3 | Measurement methodology: how SiLU kernel latency is isolated and measured using Tracy profiler CSV; median-over-20-iterations statistical convention |
| Chapter 4 | Roofline analysis and latency ratio results: the `num_tokens < 16` threshold for memory-bound SiLU dominance; the `num_tokens >= 64` prefill crossover |
| Chapter 5 | Fusion mechanisms: Pattern A definition (fused SiLU into gate_proj), Pattern B baseline (4 dispatches), program config `fused_activation` field, `activation_dtype` choices |

---

## Learning Objectives

By the end of this chapter you will be able to:

1. Identify whether your specific `(num_tokens, hidden_dim, dtype)` configuration falls in the decode or prefill performance regime where fusion provides the most benefit.
2. Choose between Pattern A with `activation="silu"` (interleaved/DRAM tensors) and Pattern A with `fused_activation` in a program config (sharded tensors).
3. Configure `activation_dtype` appropriately for BF16 versus BFP8 inference paths.
4. Verify that fusion is active by reading the Tracy profiler CSV and confirming the absence of a standalone `silu` kernel entry.
5. Recognize the anti-pattern where `activation="silu"` is incorrectly applied when the matmul output has multiple consumers.

---

## Key Findings from Chapters 3–5

The following bullets summarize the quantitative and architectural conclusions that underpin the recommendations in this chapter.

**From Chapter 3 (Measurement):**

- SiLU kernel latency is isolated by running `ttnn.silu` on a pre-computed input tensor of the target shape with no surrounding ops, using the `DEVICE KERNEL DURATION [ns]` column from the Tracy profiler CSV, median over 20 warm iterations.
- Absolute SiLU latency on Wormhole B0 ranges from approximately 2 µs (small tensors: `[1, 1, 8, 2048]` BF16) to approximately 20 µs (large tensors: `[1, 1, 2048, 8192]` BF16). *See Chapter 3, `measurement_methodology.md` for the full shape sweep.*
- The standalone measurement is a valid upper bound on the fusion benefit: Pattern A cannot save more than the standalone SiLU latency.

**From Chapter 4 (Roofline and Comparison):**

- SiLU has arithmetic intensity of approximately 0.5 FLOP/byte. It is always memory-bound on Wormhole B0 regardless of tensor size. *See Chapter 4, `roofline_analysis.md` for the roofline plot.*
- At decode token counts (1–8 tokens), SiLU represents 15–40% of the gate_proj matmul latency. Both ops are memory-bound at these small batch sizes, so the ratio remains high.
- At prefill token counts (128+ tokens), the gate_proj matmul becomes compute-bound and its latency grows much faster than SiLU, which remains memory-bound. SiLU falls to 4–8% of gate_proj matmul time at 128 tokens and continues to shrink at larger token counts.
- The `num_tokens = 16` boundary is the practical crossover: above 16 tokens, fusion benefit drops below 5% of FFN latency. Below 16 tokens, fusion is a first-order optimization. *See Chapter 4, `compute_vs_memory_bound_regimes.md` for the threshold derivation.*

**From Chapter 5 (Fusion Strategies):**

- Pattern A (fused) uses 3 kernel dispatches for SwiGLU: fused gate_proj+SiLU, up_proj, mul. *See Chapter 5, `swiglu_fusion_pattern.md` for the code and dispatch count.*
- Pattern B (unfused baseline) uses 4 dispatches: gate_proj, silu, up_proj, mul.
- The `activation` top-level parameter works for interleaved and DRAM-output tensors. The `fused_activation` field inside a program config is required for sharded tensors.
- `activation_dtype=ttnn.bfloat8_b` halves the L1 footprint of the fused SiLU output tensor but requires PCC validation (threshold > 0.999 against a BF16 reference).

---

## Quick-Reference Recommendation Table

| Scenario | `num_tokens` | `d_ff` | `dtype` | Recommendation | Expected SiLU / gate_proj ratio |
|---|---|---|---|---|---|
| Decode, BF16, standard MoE FFN | 1–8 | 2048–8192 | `bfloat16` | Pattern A, `activation="silu"` | 15–40% — fusion critical |
| Decode, BFP8 inference | 1–8 | 2048–8192 | `bfloat8_b` | Pattern A, `fused_activation`, `activation_dtype=ttnn.bfloat8_b`; validate PCC | 15–40% — fusion critical |
| Decode, T3K expert parallelism | 1–16 per chip | 2048 | `bfloat16` | Pattern A; per-chip token count remains low | 15–40% per chip |
| Prefill, 128+ tokens, BF16 | 128–2048 | 2048–8192 | `bfloat16` | Fusion optional; prioritize matmul program config tuning | 4–8% at 128t, < 5% above ~256t — low priority |
| Prefill, sharded large hidden dim | 128–2048 | 8192+ | `bfloat16` | Pattern A via `fused_activation` in program config; L1 savings justify fusion | 4–8% at 128t, < 5% above ~256t; L1 footprint benefit |

---

## Chapter Structure

| File | Contents |
|---|---|
| `index.md` (this file) | Chapter overview, learning objectives, key findings summary, quick-reference table, prerequisites |
| [`when_fusion_helps.md`](./when_fusion_helps.md) | Decision framework mapping `(num_tokens, hidden_dim, dtype)` to latency ratio regime; decode vs. prefill analysis; T3K multi-chip context; anti-pattern warning |
| [`configuration_recommendations.md`](./configuration_recommendations.md) | Recommended config for three production scenarios; program config path for sharded tensors; Tracy profiler verification checklist |
| [`measurement_summary_and_next_steps.md`](./measurement_summary_and_next_steps.md) | Expected latency numbers; interpretation; open questions; pointers to related guides |

---

## Next Steps

Begin with [`when_fusion_helps.md`](when_fusion_helps.md) to understand which combinations of token count, hidden dimension, and dtype place SiLU in the regime where fusion provides a measurable end-to-end speedup. Then proceed to [`configuration_recommendations.md`](configuration_recommendations.md) for the specific API calls and program config settings for each production scenario.
