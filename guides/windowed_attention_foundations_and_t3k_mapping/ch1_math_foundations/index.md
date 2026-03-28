# Chapter 1 — Mathematical Foundations of Windowed Attention

This chapter derives sliding-window attention from first principles, establishes
the notation used throughout the guide, and quantifies exactly how restricting
the query's receptive field translates into compute and memory savings.

## Reading Order

1. [`full_vs_windowed_attention.md`](./full_vs_windowed_attention.md) — formal
   definitions of both attention variants, side-by-side mask diagrams, masked
   softmax formulation, and an O-complexity comparison table.

2. [`window_size_parameter.md`](./window_size_parameter.md) — definition and
   typical values of the window size `w`, its relationship to context length,
   the sink-token extension used by StreamingLLM and Mistral, and how `w` is
   stored in model config and surfaced in tt-transformers.

## Why Window-Bounding the Receptive Field Matters

Bounding each query's receptive field to the most recent `w` tokens converts prefill compute from O(T²) to O(T · w) and caps decode KV-cache bandwidth at O(w) regardless of generation length — both critical on memory-bandwidth-limited hardware. The complexity analysis is in [`full_vs_windowed_attention.md`](./full_vs_windowed_attention.md); the empirical motivation for typical values of `w` is in [`window_size_parameter.md`](./window_size_parameter.md).

The chapters that follow build on this foundation: Chapter 1 establishes the
math, Chapter 2 works out the KV cache lifecycle, Chapter 3 characterises
memory access patterns, Chapters 4 and 5 map the algorithm onto TTNN primitives,
Chapter 6 covers T3K mesh sharding, and Chapter 7 provides a roofline analysis
and kernel survey.
