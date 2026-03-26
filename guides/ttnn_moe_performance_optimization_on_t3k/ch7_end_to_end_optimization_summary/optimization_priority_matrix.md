# Optimization Priority Matrix

## Context

This file addresses all eight research questions (Q1–Q8). It maps each question to the chapter that answers it, the expected latency impact (High/Medium/Low), implementation complexity (Low/Medium/High), and a one-line action.

## Priority Matrix

| # | Research Question (short) | Primary Chapter | Expected Latency Impact | Implementation Complexity | Action |
|---|---|---|---|---|---|
| Q1 | CCL topology/link settings | Ch2 | High (CCL is 55–65% of decode time) | Low (parameter tuning) | Sweep `chunks_per_sync ∈ {1,5,10,20}` and `num_workers_per_link ∈ {1,2}` in `reduce_scatter_minimal_async` |
| Q2 | Expert pipeline bottleneck | Ch3 | High (sparse_matmuls dominate compute) | Medium (requires profiling each stage) | Profile `TTNNExperts.forward` with TTNN op timers to rank stages, then target the slowest |
| Q3 | Program config optimality | Ch4 | Medium (tuning `in0_block_w` and `per_core_M`) | Medium (requires benchmark sweep) | Tune `_make_sparse_matmul_program_config` per model using the sweep harness in Ch4 |
| Q4 | HiFi2 vs LoFi | Ch4 | Medium (~10–20% on expert matmuls) | Low (single constant change) | Run the Ch4 accuracy harness to confirm LoFi is safe, then change `ttnn.MathFidelity.HiFi2` → `ttnn.MathFidelity.LoFi` in expert matmuls |
| Q5 | Weight application overhead | Ch3 | Low–Medium (secondary to sparse_matmuls) | Medium (requires alternative implementation and validation) | Profile the repeat+permute block; if >5% of `TTNNExperts.forward`, implement and benchmark the elementwise-after-reshape alternative |
| Q6 | CPU fallback elimination | Ch6 | High (but only if `Glm4MoeNaiveMoeHybrid` is active) | Low (one-line fix) | Set `ttnn = True` at `moe.py:L569`; run the Ch6 detection harness to confirm no CPU fallback remains |
| Q7 | Router 3-pass centering cost | Ch5 | Low (10–15 µs per pass, conditional) | Medium (requires single-pass alternative and accuracy validation) | Implement single-pass topk baseline from Ch5 and measure agreement rate; consider removing centering only if agreement rate is ≥99.9% |
| Q8 | Profiling toolchain | Ch5 | N/A (enabler, not optimization) | Low | Use TTNN op timers for automated CSV collection; Tracy for interactive timeline exploration |

---

**Previous:** [Chapter 7 Index](index.md) | **Next:** [Recommended Action Plan](recommended_action_plan.md)
