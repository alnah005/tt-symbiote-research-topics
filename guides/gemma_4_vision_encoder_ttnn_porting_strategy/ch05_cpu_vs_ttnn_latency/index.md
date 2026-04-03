# Chapter 5 — CPU vs. TTNN Latency Analysis

## Learning Objectives

After completing this chapter, you will be able to:

- Profile the Gemma 4 vision encoder on CPU and identify which modules dominate execution time
- Estimate TTNN latency from first principles using FLOP counts and Wormhole B0 hardware specs
- Determine the break-even point where TTNN porting becomes worthwhile after accounting for host-to-device transfer overhead
- Apply the decision matrix to choose between CPU and TTNN execution for a given deployment scenario
- Justify the porting effort (or the decision to skip it) with concrete latency numbers

## Prerequisites

- Completion of [Chapter 1 — Gemma 4 Vision Encoder Architecture Overview](../ch01_gemma4_vision_architecture/index.md) (parameter counts, compute profile, token budgets)
- Completion of [Chapter 2 — SigLIP vs. Gemma 4 Comparison](../ch02_siglip_vs_gemma4_comparison/index.md) (reuse potential informs porting effort estimate)
- Familiarity with basic performance analysis: FLOPs, memory bandwidth, roofline model concepts

## Chapter Contents

| File | Topic |
|------|-------|
| [`cpu_baseline_profiling.md`](./cpu_baseline_profiling.md) | CPU profiling methodology, per-module latency breakdown, scaling across token budgets and batch sizes |
| [`ttnn_latency_projection.md`](./ttnn_latency_projection.md) | First-principles TTNN latency estimation, Gemma 3 reference comparison, break-even analysis |
| [`decision_matrix.md`](./decision_matrix.md) | Deployment-scenario recommendations: when CPU is acceptable, when TTNN is required |

## Overview

The Gemma 4 vision encoder has approximately 570M parameters organized into 27 transformer layers with `hidden_size=1152`. Whether this encoder should run on the CPU host or be ported to TTNN on Wormhole hardware depends on the deployment scenario. The purpose of this chapter is to provide the latency data needed to make that decision rigorously.

### The Core Question

Running the vision encoder on CPU is the path of zero porting effort. The question is whether the latency cost is acceptable. There are three scenarios where CPU execution becomes problematic:

1. **Batch inference.** CPU latency scales roughly linearly with batch size, while Wormhole's massively parallel compute grid can absorb batch-level work with sub-linear scaling.

2. **High token budgets.** At 1120 tokens, the encoder processes roughly 4x more patches than at 280 tokens. The quadratic attention cost in the encoder layers makes this regime expensive on CPU.

3. **Continuous batching.** When the vision encoder shares a pipeline with the language model decoder running on Wormhole, a CPU-bound vision encoder stalls the pipeline and wastes device cycles.

This chapter quantifies each scenario with concrete numbers and produces a decision matrix that maps deployment parameters to a CPU-or-TTNN recommendation.

### Decision Framework Summary

The analysis in this chapter leads to the following high-level framework:

| Scenario | Recommendation |
|----------|---------------|
| Single image, 70-140 tokens | CPU likely acceptable |
| Single image, 280 tokens | CPU acceptable for offline; TTNN preferred for latency-sensitive |
| Single image, 560-1120 tokens | TTNN recommended |
| Batch >= 4 at any token budget | TTNN strongly recommended |
| Continuous batching pipeline | TTNN required |
| Prefill-dominated (long output) | CPU may be acceptable (amortized over decode) |

The detailed analysis supporting each cell is in [`decision_matrix.md`](./decision_matrix.md).

### Reading Order

Start with [`cpu_baseline_profiling.md`](./cpu_baseline_profiling.md) to establish the CPU baseline. Then read [`ttnn_latency_projection.md`](./ttnn_latency_projection.md) to understand the projected TTNN performance. Finally, [`decision_matrix.md`](./decision_matrix.md) synthesizes both into actionable recommendations.
