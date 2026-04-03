# Decision Matrix

This file synthesizes the CPU baseline profiling and TTNN latency projections into deployment-specific recommendations. Each scenario includes the reasoning, the estimated latency impact, and a clear recommendation.

## Decision Matrix

| # | Scenario | CPU Latency (Mid) | TTNN Latency (Mid) | Recommendation | Rationale |
|---|----------|-------------------|--------------------|--------------------|-----------|
| 1 | Single image, 70-140 tokens | 6-12 ms | 7-9 ms | **CPU acceptable** | TTNN provides marginal or no speedup; porting effort not justified for this case alone |
| 2 | Single image, 280 tokens, offline | ~25 ms | ~12 ms | **CPU acceptable** | 25 ms is tolerable for offline/batch processing; LLM decode dominates total latency |
| 3 | Single image, 280 tokens, latency-sensitive | ~25 ms | ~12 ms | **TTNN preferred** | 2.1x speedup reduces time-to-first-token perceptibly |
| 4 | Single image, 560-1120 tokens | 55-127 ms | 19-34 ms | **TTNN recommended** | 2.8-3.8x speedup; CPU latency becomes user-visible and pipeline-disruptive |
| 5 | Batch >= 4, any token budget | 96-908 ms | 23-158 ms | **TTNN strongly recommended** | 4.1-5.7x speedup; CPU becomes a severe bottleneck |
| 6 | Continuous batching pipeline | Variable | Variable | **TTNN required** | CPU vision encoder stalls the LLM decode pipeline on device |
| 7 | Prefill-dominated (long output) | ~25 ms (once) | ~12 ms (once) | **CPU may be acceptable** | Vision runs once; amortized over hundreds of decode steps |

## Detailed Analysis by Scenario

### Scenario 1: Single-Image, Low Token Budget (70-140 Tokens)

**Use case:** Quick image classification, thumbnail captioning, low-resolution image understanding.

At 70 tokens, the vision encoder processes only ~210 patches. The matmuls are small enough that TTNN's non-matmul overhead (op dispatch, RoPE application, norms) erodes most of the compute advantage. The CPU completes in 6-12 ms, which is:
- Less than a single LLM decode step on Wormhole (~15-25 ms for Gemma 4 31B)
- Invisible to the end user in any interactive scenario

**Recommendation:** Run on CPU. Do not port for this scenario alone. If TTNN is ported for other scenarios, this configuration can use it opportunistically, but it is not the justification.

### Scenario 2: Single-Image, Default Budget (280 Tokens), Offline

**Use case:** Batch image captioning, document understanding pipelines, offline annotation.

CPU latency of ~25 ms is small relative to total inference time. For a 100-token response at ~20 ms per decode step, the LLM takes ~2000 ms. The vision encoder is ~1.2% of the total. Even a 2.1x speedup (saving ~13 ms) is imperceptible.

**Recommendation:** CPU is acceptable. The porting effort is better spent on other optimizations (e.g., LLM decode, speculative decoding).

### Scenario 3: Single-Image, Default Budget (280 Tokens), Latency-Sensitive

**Use case:** Interactive chat with image input, real-time visual Q&A, time-to-first-token SLA.

Here the vision encoder latency is on the critical path to the first token. The 2.1x speedup (25 ms to 12 ms) reduces time-to-first-token by 13 ms. While modest in absolute terms, this matters when:
- The application has a tight P99 latency budget
- Multiple images are processed in a single request (the savings multiply)
- The vision encoder runs synchronously before prefill begins

**Recommendation:** TTNN preferred. The latency reduction is meaningful for user-facing applications.

### Scenario 4: Single-Image, High Token Budget (560-1120 Tokens)

**Use case:** Detailed image analysis, high-resolution document OCR, fine-grained visual reasoning.

At 1120 tokens, CPU latency reaches 127 ms. This is:
- 5-8 decode steps' worth of time on Wormhole
- Perceptible delay before the first token
- A significant fraction of short-response inference time

TTNN reduces this to ~34 ms (3.8x speedup), bringing the vision encoder closer to a single decode step's duration.

> **Risk:** At 1120 tokens, sequence length is ~3360 patches. The attention matmuls are `[3360, 72] x [72, 3360]` per head — large enough to achieve good TTNN utilization. This is the regime where TTNN's advantage is most pronounced. Severity: Low (risk is that estimates are too optimistic). Mitigation: validate with profiling in Phase 3.

**Recommendation:** TTNN recommended. The absolute latency savings (~93 ms) is substantial.

### Scenario 5: Batch Inference (Batch >= 4)

**Use case:** Server-side batch processing, multi-image requests, data pipeline acceleration.

CPU latency scales nearly linearly with batch size, while TTNN scales sub-linearly (better utilization at larger effective batch dimensions). At batch=8, 280 tokens:

- CPU: ~180 ms
- TTNN: ~38 ms
- Speedup: 4.8x

At batch=8, 1120 tokens:

- CPU: ~908 ms
- TTNN: ~158 ms
- Speedup: 5.7x

In batch processing, throughput (images per second) matters more than single-image latency. TTNN processes 8 images in the time CPU processes roughly 1.5-2.

**Recommendation:** TTNN strongly recommended. The throughput advantage is decisive.

### Scenario 6: Continuous Batching Pipeline

**Use case:** Production serving with continuous batching, where the LLM decoder runs on Wormhole and new requests arrive continuously.

In a continuous batching setup, the Wormhole device is continuously executing LLM decode steps. When a new request with an image arrives, the vision encoder must produce soft tokens before the LLM can begin prefill for that request. If the vision encoder runs on CPU:

1. The CPU must complete the vision encoder forward pass before the device can start prefill
2. During this time, the device may have idle cycles (if the vision encoder is on the critical path)
3. Other requests in the batch continue decoding, but the new request is delayed

The fundamental problem is **pipeline coupling**: a CPU-bound stage in a device-bound pipeline creates a stall. Even if the CPU latency is small in absolute terms, the opportunity cost of idle device cycles is high.

With TTNN, the vision encoder runs on a dedicated portion of the device (or on a separate chip in a multi-chip setup), eliminating the CPU-device synchronization bottleneck.

> **Warning:** In a continuous batching system, any CPU-bound operation that gates device work is a throughput limiter. Even 10 ms of CPU vision encoder time per new request can reduce overall system throughput if requests arrive at a rate where the pipeline cannot absorb the delay.

**Recommendation:** TTNN required. CPU execution creates a structural bottleneck in the serving pipeline.

### Scenario 7: Prefill-Dominated Workloads

**Use case:** Long image descriptions, detailed visual analysis responses (500+ output tokens).

When the output is long, the vision encoder runs once but the LLM decodes for many steps:

| Output Length | LLM Decode Time (est.) | Vision Encoder (CPU) | Vision as % of Total |
|--------------|----------------------|---------------------|---------------------|
| 50 tokens | ~1,000 ms | 25 ms | 2.4% |
| 200 tokens | ~4,000 ms | 25 ms | 0.6% |
| 500 tokens | ~10,000 ms | 25 ms | 0.2% |
| 1000 tokens | ~20,000 ms | 25 ms | 0.1% |

At 200+ output tokens, the vision encoder's contribution to total latency is negligible regardless of whether it runs on CPU or TTNN.

**Recommendation:** CPU is acceptable for prefill-dominated workloads at 280-token budget. At higher token budgets (560-1120), re-evaluate using the numbers from Scenario 4.

## How Gemma 3 TTNN Vision Performance Informs Expectations

The Gemma 3 SigLIP encoder shares identical layer dimensions (`hidden_size=1152`, 27 layers, `intermediate_size=4304`) and has been optimized for TTNN. Key lessons from the Gemma 3 experience:

1. **MLP matmuls are well-optimized.** The `1152 x 4304` and `4304 x 1152` matmuls achieve good utilization on Wormhole. These are identical in Gemma 4 and should transfer directly.

2. **Attention at long sequences works well.** The Gemma 3 SigLIP encoder processes 4096 tokens — much longer than Gemma 4's typical 841. If attention worked well at 4096, it will work at 841 (though with lower utilization per head).

3. **Non-matmul overhead is significant.** In the Gemma 3 TTNN encoder, op dispatch and element-wise operations account for a non-trivial fraction of latency. The Gemma 4 encoder has the additional overhead of 2D RoPE application. Tracing is important.

4. **Weight sharding strategy transfers.** The memory layout and sharding decisions for the 27 encoder layers in Gemma 3 can be directly reused for Gemma 4, since the weight shapes are identical.

## Porting Effort vs. Latency Benefit

The decision to port must also account for engineering cost. From [Chapter 6](../ch06_reuse_strategy/index.md) and [Chapter 7](../ch07_implementation_roadmap/index.md):

- **Estimated porting effort:** 5-7 weeks (Phases 1-4)
- **Code reuse from Gemma 3:** ~40-50% direct reuse, ~30% modification

The latency benefit must justify this effort in the context of the target deployment:

| Deployment | Latency Savings per Request | Requests per Day | Daily Time Saved | Justification |
|-----------|---------------------------|-----------------|-----------------|---------------|
| Low-volume offline (100 req/day) | 13 ms | 100 | 1.3 seconds | Not justified |
| Medium-volume serving (10K req/day) | 13 ms | 10,000 | 130 seconds | Marginal |
| High-volume serving (1M req/day) | 13 ms | 1,000,000 | 3.6 hours | Justified |
| High-volume, batch=8, 1120 tokens | 750 ms | 125,000 batches | 26.0 hours | Strongly justified |

> **Tip:** The porting effort produces a reusable codebase that will serve all Gemma 4 deployments on Tenstorrent hardware. The amortized cost across multiple deployment teams and use cases is much lower than the per-team cost suggests.

## Summary of Recommendations

For the typical Tenstorrent deployment scenario — serving Gemma 4 31B on Wormhole hardware with continuous batching and mixed text/image inputs — the recommendation is:

**Port the vision encoder to TTNN.**

The primary justifications are:
1. Continuous batching requires the vision encoder on device to avoid pipeline stalls (Scenario 6)
2. Batch inference provides 4.1-5.7x speedup (Scenario 5)
3. High token budgets provide 2.8-3.8x speedup (Scenario 4)
4. ~40-50% code reuse from Gemma 3 reduces porting effort
5. The porting effort is a one-time cost; the latency benefit accrues on every inference

The only scenario where CPU execution is clearly sufficient is single-image, low token budget (70-140), offline processing. For all other scenarios, TTNN provides a meaningful or essential advantage.

---

**Next:** [Chapter 6 — Reuse Strategy](../ch06_reuse_strategy/index.md) — Concrete file-by-file reuse plan for existing Gemma 3 TTNN modules.
