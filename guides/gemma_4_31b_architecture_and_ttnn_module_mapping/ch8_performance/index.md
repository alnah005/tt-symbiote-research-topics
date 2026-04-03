# Chapter 8 --- Performance Analysis and Optimization Roadmap

## Overview

This chapter analyzes the expected performance characteristics of Gemma 4 31B
on the T3K 1x8 Wormhole mesh and identifies the key optimization opportunities
for achieving competitive decode throughput. The analysis draws on the weight
shapes from [Chapter 2](../ch2_projection_shapes/index.md), the sharding
strategy from [Chapter 6](../ch6_tp_sharding/index.md), and the module
structure from [Chapter 7](../ch7_model_assembly/index.md).

After reading this chapter you will know:

- The complete per-device DRAM budget: weight memory, KV cache memory,
  activation memory, and whether the model fits within the 12 GB per-chip limit
  at various sequence lengths and quantization levels.
- The expected per-layer and total decode latency, broken down by component
  (attention projections, RoPE, SDPA, FFN, norms, PLE, CCL).
- Where the primary bottlenecks lie --- memory bandwidth for decode matmuls,
  SDPA latency for global layers at long contexts, and CCL overhead for 120
  all-reduce operations per decode step.
- A prioritized optimization roadmap covering Metal Trace, Multi-CQ, fused
  projections, DRAM-sharded weights, BFP8 KV cache, and other techniques.

## Key Performance Metrics Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Total parameters | ~30.7B | 50 sliding + 10 global layers |
| Weight memory (BFP8, per device) | ~4,143 MB | Fits in 12 GB DRAM |
| Weight memory (BF16, per device) | ~7,513 MB | Tight fit, limited headroom |
| KV cache at S=8,192 (BF16, per device) | 740 MB | 100 MB sliding + 640 MB global |
| KV cache at S=32,768 (BF16, per device) | 2,660 MB | Sliding constant at 100 MB |
| All-reduce ops per decode step | 120 | 2 per layer x 60 layers |
| Max sequence length (BFP8 weights + BF16 KV) | ~65K | ~2.9 GB headroom |
| Max sequence length (BFP8 weights + BFP8 KV) | ~131K | ~2.9 GB headroom |

## Primary Bottlenecks

1. **Memory bandwidth (decode matmuls).** At batch=1 decode, all linear
   projections are memory-bound --- each is a matrix-vector multiply where the
   entire weight matrix must be read from DRAM for a single output token. The
   FFN projections dominate: three matmuls totaling ~86.7 MB of weights per
   device per layer (at BFP8), repeated 60 times.

2. **Global layer SDPA at long contexts.** The 10 global layers attend over
   the full sequence length. At S=32,768, each global SDPA reads a KV cache of
   256 MB per layer per device (BF16). This is 2,560 MB total across 10 global
   layers --- comparable to the entire weight read for 10 layers of FFN.

3. **CCL overhead.** Each decode step requires 120 `ttnn.all_reduce` calls
   (2 per layer x 60 layers). While each payload is small (~10.5 KB at B=1),
   the per-call launch latency accumulates. At an estimated 5--10 us per
   all-reduce, this adds 0.6--1.2 ms of pure CCL overhead per decode step.

4. **Heterogeneous layer structure.** Sliding and global layers have different
   projection shapes, RoPE configurations, and KV cache geometries. This
   complicates Metal Trace capture and program config management, though the
   polymorphic module design from [Chapter 5](../ch5_attention_module_design/index.md)
   ensures a fixed op sequence suitable for tracing.

## Reading Order

1. [`memory_budget.md`](./memory_budget.md) --- Complete DRAM budget analysis
   covering weights, KV cache, activations, and quantization requirements.
2. [`decode_latency_analysis.md`](./decode_latency_analysis.md) --- Per-layer
   and total decode latency breakdown with sliding vs global comparison.
3. [`optimization_roadmap.md`](./optimization_roadmap.md) --- Prioritized
   optimization techniques for achieving competitive decode performance.

## Prerequisites

This chapter builds on:

- [Chapter 2 --- Projection Shapes](../ch2_projection_shapes/index.md): weight
  sizes for all projections.
- [Chapter 6 --- TP Sharding](../ch6_tp_sharding/index.md): per-device weight
  and KV cache sizes after TP=8 sharding.
- [Chapter 7 --- Model Assembly](../ch7_model_assembly/index.md): full model
  structure, decode loop, and KV cache initialization.

---

**Next:** [`memory_budget.md`](./memory_budget.md)
