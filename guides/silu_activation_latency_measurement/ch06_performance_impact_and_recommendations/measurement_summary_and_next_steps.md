# Measurement Summary and Next Steps

This document summarizes the expected absolute latency numbers for SiLU and the surrounding gate_proj matmul on Wormhole B0, provides interpretation guidance for engineers reading their own benchmark results, lists three open research questions, and points to related guides for further optimization work.

---

## Expected Latency Numbers

The table below lists expected SiLU kernel latency (standalone `ttnn.silu`) and gate_proj matmul latency for a representative set of tensor shapes on Wormhole B0. Values are BF16, TILE_LAYOUT, DRAM-input/L1-interleaved-output, median over 20 warm iterations, `DEVICE KERNEL DURATION [ns]` from Tracy CSV divided by 1000. *See Chapter 3, `measurement_methodology.md` for the full measurement procedure.*

All values marked `[MEASURED: ...]` are placeholders to be filled in when the guide is executed against hardware. The accompanying estimates represent the expected range derived from the roofline analysis in Chapter 4 and the hardware constants in Chapter 2.

| `num_tokens` | `d_ff` | `dtype` | Standalone SiLU (µs) | gate_proj matmul (µs) | SiLU / matmul ratio |
|---|---|---|---|---|---|
| 1 | 2048 | BF16 | [MEASURED: ~2 µs] | [MEASURED: ~5–8 µs] | [MEASURED: ~25–40%] |
| 4 | 2048 | BF16 | [MEASURED: ~3 µs] | [MEASURED: ~8–12 µs] | [MEASURED: ~25–35%] |
| 8 | 2048 | BF16 | [MEASURED: ~4–5 µs] | [MEASURED: ~12–18 µs] | [MEASURED: ~22–30%] |
| 8 | 8192 | BF16 | [MEASURED: ~6–8 µs] | [MEASURED: ~25–40 µs] | [MEASURED: ~15–25%] |
| 16 | 2048 | BF16 | [MEASURED: ~5–7 µs] | [MEASURED: ~20–30 µs] | [MEASURED: ~15–25%] |
| 64 | 2048 | BF16 | [MEASURED: ~8–12 µs] | [MEASURED: ~60–100 µs] | [MEASURED: ~8–15%] |
| 128 | 2048 | BF16 | [MEASURED: ~12–16 µs] | [MEASURED: ~180–260 µs] | [MEASURED: ~5–8%] |
| 512 | 2048 | BF16 | [MEASURED: ~15–20 µs] | [MEASURED: ~800–1200 µs] | [MEASURED: ~1–2%] |
| 2048 | 2048 | BF16 | [MEASURED: ~18–22 µs] | [MEASURED: ~3500–5000 µs] | [MEASURED: < 1%] |

**Key observations from the expected ranges:**

1. **SiLU absolute latency is bounded between approximately 2 µs (smallest decode shapes) and 20 µs (large prefill shapes).** The range is narrow because SiLU is always memory-bound; its latency scales linearly with tensor size, and the full tensor size range covered here is 10×.

2. **gate_proj matmul latency spans roughly 5 µs to 5000 µs** — a 1000× range — because it transitions from memory-bound at decode to compute-bound at prefill. The matmul FLOP count scales as `O(num_tokens × d_model × d_ff)` while SiLU scales as `O(num_tokens × d_ff)`.

3. **The crossover is near `num_tokens ≈ 16`** for `d_ff = 2048`, BF16 on Wormhole B0. Below this threshold the gate_proj matmul remains memory-bound and SiLU is 15–40% of its latency; fusion is a first-order optimization. Above `num_tokens ≈ 16` the matmul transitions toward compute-bound and the ratio drops progressively. By `num_tokens ≈ 64` the ratio has fallen below 5% and fusion benefit is negligible. *See Chapter 4, `compute_vs_memory_bound_regimes.md` for the derivation.*

---

## Interpretation Guide

When you run the Chapter 3 measurement harness on your own hardware and fill in the `[MEASURED: ...]` values, use the following interpretation rubric.

**If your measured SiLU / matmul ratio is higher than the estimates above:**

- Verify that the gate_proj matmul is not artificially slow due to a misconfigured program config (e.g., too few cores assigned, suboptimal block sizes). A well-tuned matmul should approach the DRAM bandwidth ceiling at decode token counts.
- Check that the weight tensor (`w1`) is loaded from DRAM and not being re-read from a cold cache on every iteration. Warm iterations should have the weight resident in DRAM (not evicted to host). *See Chapter 3, `isolating_silu_from_matmul.md` for the warmup protocol.*

**If your measured SiLU latency is higher than 20 µs for small tensor shapes:**

- Verify that `TILE_LAYOUT` is set. The SFPU cannot operate on ROW_MAJOR tensors without an implicit layout conversion, which will inflate measured latency. *See Chapter 2, `tensix_compute_engine.md` for the TILE_LAYOUT requirement.*
- Check that the tensor dtype is BF16 or BFP8_B. Mixed-precision tensors may trigger unintended type conversions.

**If Pattern A shows no improvement over Pattern B in your Tracy CSV:**

- Confirm that the standalone SiLU kernel row is absent from the Pattern A trace. If it is still present, the `activation="silu"` parameter is being ignored — a program config without `fused_activation` is likely overriding it. *See [`configuration_recommendations.md`](configuration_recommendations.md), the verification checklist.*
- Confirm that the fused kernel duration is not suspiciously short (matching the standalone matmul only). This can occur if the kernel was compiled without the SFPU pass due to a misconfiguration.

**If BFP8 activation dtype degrades PCC below 0.999:**

- Reduce the scope of BFP8: keep `activation_dtype=ttnn.bfloat16` for SiLU output and apply BFP8 only to weight tensors.
- Investigate whether the accuracy loss concentrates in specific expert slots. If only a small subset of experts degrade, consider per-expert dtype configuration.

> **Tip:** For MoE models with many experts (64+), measure PCC per-expert rather than globally. A global PCC > 0.999 can mask per-expert degradation that affects uncommon but important tokens.

---

## Open Questions

The following three questions are not answered by this guide and represent areas for future investigation.

### 1. Does a Custom Metalium Kernel Fusing Full SwiGLU Provide Additional Benefit?

Pattern A (defined in Chapter 5, `swiglu_fusion_pattern.md`) fuses SiLU into the gate_proj matmul but still requires separate kernel dispatches for up_proj and the element-wise multiply. A fully fused Pattern C kernel — computing gate_proj, SiLU, up_proj, and mul in a single Metalium kernel — would eliminate two more dispatches and two more intermediate tensor allocations.

The question is: at decode token counts (1–16 tokens), what fraction of per-expert FFN latency is consumed by the up_proj matmul dispatch and the `ttnn.mul` dispatch? If these together represent another 10–20% of FFN time, Pattern C could provide a further meaningful speedup.

Investigating this requires authoring a custom Metalium kernel in C++, verifying correctness against the Pattern A baseline, and profiling on representative MoE shapes. This is outside the scope of the current guide. *See the `moe_optimization_techniques_for_ttnn` guide for discussion of custom Metalium kernel authoring.*

### 2. How Does SiLU Latency Scale on T3K with Expert Parallelism Reducing Per-Chip Token Count?

The analysis in `when_fusion_helps.md` predicts that expert parallelism on T3K keeps per-chip token count in the decode regime. This prediction assumes uniform token distribution across experts. In practice, MoE routing is non-uniform: popular experts receive more tokens than rare experts, creating load imbalance.

Under load imbalance, some chips may consistently handle more tokens per expert than the average, pushing their per-chip token count closer to the prefill regime threshold. Measuring the actual per-chip token count distribution on T3K under production routing (with learned router weights on a real dataset) would validate or contradict the decode-regime assumption.

*See the `t3k_mesh_device_optimizations` guide for multi-chip expert dispatch instrumentation.*

### 3. Does BFP8 Activation Dtype Hurt MoE Perplexity at INT8-Quantized Weight Scales?

The PCC threshold of > 0.999 used in this guide is a per-tensor numerical fidelity check, not an end-to-end model quality check. When BFP8 activation dtype is combined with INT8-quantized weight tensors (a common configuration for maximum throughput on Wormhole B0), the compounding quantization errors may degrade model perplexity beyond what per-tensor PCC indicates.

The open question is: at the full model level, does `activation_dtype=ttnn.bfloat8_b` + INT8 weights + BF16 accumulation satisfy the perplexity budget for target benchmarks (WikiText-2 perplexity, MMLU accuracy)?

Answering this requires running end-to-end model evaluation with and without BFP8 activation dtype, with INT8 weights, on a standard benchmark suite. This is an empirical question that depends on the specific model architecture and quantization scheme.

---

## Pointers to Related Guides

| Guide | Relevance |
|---|---|
| `moe_optimization_techniques_for_ttnn` | Batched matmul strategies for MoE expert dispatch; custom Metalium kernel authoring for Pattern C (full SwiGLU fusion); down_proj optimization |
| `t3k_mesh_device_optimizations` | Multi-chip expert dispatch on T3K; per-chip token count measurement; Ethernet communication overhead in expert parallelism |

---

## Guide Summary

This guide has covered the SiLU activation function from first principles through production configuration recommendations for Wormhole B0 and T3K MoE inference.

**What was covered:**

- Chapter 1: SiLU's role in the SwiGLU FFN formula and why it appears at the gate_proj output in production MoE models.
- Chapter 2: How SiLU executes on Wormhole B0 via the SFPU, the L1 round-trip it causes, and the hardware bandwidth ceiling that makes it always memory-bound.
- Chapter 3: How to isolate and measure SiLU kernel latency using Tracy profiler CSV with statistical rigor.
- Chapter 4: The roofline-based comparison of SiLU and gate_proj matmul latency across token counts, and the `num_tokens ≤ 16` decode-regime threshold where SiLU represents 15–40% of matmul time.
- Chapter 5: Pattern A (fused) and Pattern B (unfused baseline) for SwiGLU, the two API paths to enable fusion (top-level `activation=` vs. program config `fused_activation`), and `activation_dtype` for BFP8 precision.
- Chapter 6 (this chapter): Decision framework, production configuration recipes for three scenarios, Tracy profiler verification, and open questions.

**The single most important takeaway:** Apply Pattern A (`activation="silu"` or `fused_activation` in program config) for all SwiGLU FFN blocks in production MoE inference when `batch_size × top_k ≤ 16`. At these token counts, SiLU is 15–40% of gate_proj matmul time and fusion is a first-order optimization requiring only a one-line API change.

---

---

**End of guide.** Return to [Guide Index](../index.md)
