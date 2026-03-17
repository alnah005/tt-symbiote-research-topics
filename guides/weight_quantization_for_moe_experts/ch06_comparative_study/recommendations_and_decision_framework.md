# Recommendations and Decision Framework

## Purpose

This document provides a structured framework for choosing an expert weight quantization
strategy for a bfloat16-trained MoE model running on Tenstorrent Wormhole B0 hardware.
The framework consists of three decision criteria applied in sequence. The output is a
concrete starting configuration — dtype and compute kernel config per projection — together
with explicit fallback conditions.

The framework is validated against two reference points established in this chapter:
DeepSeek-V3 (bfloat4_b gate/up + bfloat8_b down, QAT-trained, PCC ~0.97) and Qwen
235B-A22B (bfloat16 all, no QAT, PCC > 0.999). Both are correct for their respective
contexts. The goal here is to reason about new deployments systematically rather than
copying either design uncritically.

## Decision Criterion 1 — Accuracy Budget

The accuracy budget is the maximum acceptable degradation from the bfloat16 baseline,
expressed as both a per-layer PCC threshold and an end-to-end perplexity delta.

| Budget tier | Per-layer PCC requirement | Perplexity delta | Compatible dtype strategy |
|---|---|---|---|
| Conservative | Full layer PCC ≥ 0.999 | ≤ 0.1 PPL | bfloat16 (no change) |
| Moderate | Full layer PCC ≥ 0.99 | ≤ 0.5 PPL | bfloat8_b all projections + HiFi2 |
| Aggressive | Full layer PCC ≥ 0.97 | ≤ 1.5 PPL | bfloat4_b gate/up + bfloat8_b down |

For context, standard bfloat16 production targets PCC > 0.999. DeepSeek-V3's production
deployment at PCC ~0.97 represents the aggressive tier — achievable only because QAT
shaped the weights to tolerate it.

**Application to Qwen.** For a bfloat16-trained model like Qwen 235B-A22B:

- Start with the **moderate** budget: bfloat8_b all + HiFi2. Expected per-layer PCC ≥
  0.99; perplexity delta ≤ 0.5 PPL from bfloat16 baseline.
- Consider the **aggressive** budget only after the moderate configuration has been
  validated over multiple layers and benchmarks, and only if the additional decode
  throughput improvement justifies the risk.
- Never skip directly from bfloat16 to bfloat4_b gate/up on a model without QAT history.

> **Warning:** The PCC thresholds in the table above are per-layer measurements. Cumulative
> error across N consecutive MoE layers compounds. If any single layer's PCC is below 0.975,
> the multi-layer drift can become significant by layer 20–30. Validate individual layer
> PCC, not just the mean across all layers.

## Decision Criterion 2 — Deployment Regime

The deployment regime determines whether memory bandwidth or compute throughput is the
binding constraint, which in turn determines which projections benefit most from
quantization.

### Decode-Heavy Workloads (batch ≤ 8, interactive inference)

During decode, expert FFN matmuls are memory-bound for all practical batch sizes at all
dtypes (see Chapter 4 for the crossover batch size analysis). In this regime, reducing DRAM
bytes read per forward pass is the dominant performance lever.

**Recommended dtype emphasis:** Maximise bit-width reduction on gate and up projections
(the highest weight-volume pair), because those bytes are read on every decode step for
every active expert. bfloat4_b on gate/up provides 4× DRAM read reduction relative to
bfloat16 for those projections, and 2× relative to bfloat8_b.

**Decode latency summary for gate projection** (`d_model=7168`, `d_ff=2048`, n300 at
576 GB/s peak, ~80% effective = 461 GB/s):

| Dtype | Bytes / projection | Latency estimate |
|---|---|---|
| bfloat16 | 29,360,128 | ~63.7 µs |
| bfloat8_b | 14,680,064 | ~31.8 µs |
| bfloat4_b | 7,340,032 | ~15.9 µs |

For decode-heavy deployments: if the accuracy budget permits the aggressive tier, bfloat4_b
gate/up delivers the largest per-step latency reduction.

### Prefill-Heavy Workloads (batch > 32, throughput-oriented)

During prefill, expert FFN matmuls approach the compute-bound regime as batch size grows.
At batch=32, the arithmetic intensity for bfloat8_b is approximately 64 FLOP/byte —
still below the ridge point, but the gap to compute-bound is narrower. At batch=128 or
above, the matmul becomes compute-bound for bfloat8_b (crossover ≈ batch=219) and
increasingly compute-limited for bfloat16 (crossover ≈ batch=437).

**Recommended dtype emphasis:** For prefill-heavy workloads, bfloat8_b on all projections
provides the best balance — approximately 2× throughput improvement over bfloat16 from
combined DRAM reduction and higher quantized compute throughput, without the accuracy risk
of bfloat4_b on a non-QAT model.

> **Tip:** For mixed workloads (both decode and prefill), choose the quantization tier that
> satisfies the accuracy budget and serves the more latency-sensitive regime. In most
> production deployments, decode latency is the user-visible metric, making the decode-heavy
> analysis the binding constraint.

### T3K Multi-Chip Considerations

On a T3K system (8 Wormhole B0 chips), all-to-all communication for expert routing often
dominates over expert FFN compute for small-batch decode. Quantization's throughput gains
may be partially masked by communication overhead. However, the memory footprint reduction
is always beneficial — it directly increases available DRAM for KV cache regardless of
whether the FFN compute is the bottleneck.

## Decision Criterion 3 — Training History

Training history is the single most important factor distinguishing safe from risky
quantization choices. The key question is: **was this model trained with any form of
quantization simulation?**

| Training history | bfloat8_b all | bfloat4_b gate/up |
|---|---|---|
| FP8 / bfloat4 QAT (e.g., DeepSeek-V3) | Low risk, expect PCC ≥ 0.99 | Low risk, validated PCC ~0.97 |
| bfloat16 standard (e.g., Qwen 235B-A22B) | Low risk, expect PCC ≥ 0.99 | Higher risk, validate per layer before deploying |
| Unknown | Treat as bfloat16 standard | Do not use without empirical validation |

**Why training history matters for bfloat4_b specifically.** The bfloat4_b format provides
only 4 bits per element. The block floating-point encoding uses a shared exponent across a
32×32 tile (1024 elements): one outlier value in the tile forces the shared exponent up,
compressing the representable range for all other elements in the tile and increasing
quantization error across the entire tile. Models trained with QAT have smaller weight
outliers — the optimizer has had thousands of steps to push weights away from values that
cause large tile-level quantization error. bfloat16-trained models have not.

For bfloat8_b, the larger mantissa (8 bits vs 4 bits) provides enough resolution that
bfloat16-trained weight distributions fit within the quantization grid without significant
outlier-driven error. This is why bfloat8_b + HiFi2 is the recommended starting point for
Qwen, while bfloat4_b gate/up is reserved for after empirical validation.

## Recommended Starting Point for Qwen 235B-A22B

Based on the three criteria above applied to Qwen's context (bfloat16-trained, mixed
decode/prefill workload, moderate accuracy budget), the recommended starting configuration
is:

**All projections: `bfloat8_b` + HiFi2**

```python
# HiFi2 config — apply to gate, up, and down projections
WormholeComputeKernelConfig(
    math_fidelity=MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)
```

| Projection | Dtype | Kernel config | Expected PCC |
|---|---|---|---|
| Gate (w1) | `ttnn.bfloat8_b` | HiFi2 | ≥ 0.99 |
| Up (w3) | `ttnn.bfloat8_b` | HiFi2 | ≥ 0.99 |
| Down (w2) | `ttnn.bfloat8_b` | HiFi2 | ≥ 0.99 |

**Expected outcomes:**
- Per-expert memory: `3 × 7168 × 2048 × 1 = 44,040,192 bytes = 42.0 MB` — approximately
  2× reduction from the 84.0 MB bfloat16 baseline.
- DRAM read volume per active expert: 2× reduction, proportionally reducing decode latency.
- System-level (T3K, 128 experts): from ~10.5 GB to ~5.25 GB expert weight footprint.
- Perplexity delta from bfloat16: expected ≤ 0.5 PPL on standard benchmarks.

This configuration is the safest quantization step available for Qwen and provides
meaningful, measurable performance improvement without requiring QAT or per-layer tuning
beyond standard PCC validation.

### Path to the Aggressive Tier

After validating bfloat8_b all + HiFi2:

1. Measure per-layer PCC across all MoE layers in at least two layer depths (early and
   late transformer layers). Confirm all are ≥ 0.99.
2. Run end-to-end perplexity on a calibration set (512-token WikiText-2 or C4 samples).
   Confirm delta ≤ 0.5 PPL from bfloat16.
3. If both pass, trial `bfloat4_b` + LoFi on gate and up projections in a single layer.
   Validate that the full-layer PCC remains ≥ 0.97.
4. If the single-layer trial passes, extend to all layers and repeat perplexity validation.
5. If any layer fails PCC ≥ 0.97, retain bfloat8_b for that projection and consider
   per-layer mixed precision (different dtypes for different layer depths).

## When to Fall Back to bfloat16

Fall back any projection to bfloat16 if:

- The projection output PCC drops below the tier threshold after quantization, where
  thresholds are: gate ≥ 0.96, up ≥ 0.96, down ≥ 0.975, full layer ≥ 0.97.
- End-to-end perplexity delta exceeds the accepted budget for the deployment (≤ 0.5 PPL
  for moderate, ≤ 1.5 PPL for aggressive).
- Cumulative error across consecutive MoE layers causes per-layer PCC drift below 0.975
  at depth N, even if individual shallow layers pass.
- Any non-determinism or layout error causes PCC to be undefined (NaN or negative) — this
  indicates a weight conversion or kernel configuration bug, not a quantization accuracy
  issue, and must be debugged before quantization accuracy can be assessed.

> **Warning:** Falling back to bfloat16 for a single projection does not require reverting
> all projections. Mixed strategies are valid — for example, bfloat8_b gate/up + bfloat16
> down — and may be the correct long-term configuration for layers with high output
> sensitivity. Document the per-layer, per-projection dtype decisions in the model config
> dataclass so that the configuration is reproducible and auditable.

## Summary Decision Tree

```
Is the model QAT-trained (FP8 or bfloat4-aware)?
├── YES → Consider bfloat4_b gate/up + bfloat8_b down (DeepSeek-V3 design)
│         Validate PCC ≥ 0.97 per layer
└── NO (standard bfloat16 training, e.g. Qwen)
    ├── START: bfloat8_b all projections + HiFi2
    │   Validate PCC ≥ 0.99, perplexity delta ≤ 0.5 PPL
    └── AFTER validation passes:
        Consider bfloat4_b gate/up + bfloat8_b down
        Validate PCC ≥ 0.97 per layer, perplexity delta ≤ 1.5 PPL
        └── If any layer fails → retain bfloat8_b for that projection
```

## Next Steps

Chapter 7, `ch07_implementation_workflow/`, provides the end-to-end implementation guide:
weight conversion scripts, per-layer PCC validation harness, throughput profiling
methodology, and iterative tuning decision tree for finalising a quantization configuration
in production.
