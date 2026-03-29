# Chapter 2 — ttnn.all_reduce Internal Architecture

This chapter traces the complete call chain from `ttnn.all_reduce` in the Python
`TTNNLinearIColShardedWAllReduced.forward` method down to the specific C++ code paths
selected on a T3K (1×8 logical ring on cluster axis 1). It identifies every
sub-operation that is invoked, characterises the semaphore state those sub-operations
use, and provides a clear verdict on whether that state is compatible with trace
capture and replay.

## Summary answer — the semaphore question

On a T3K 1×8 decode shape with `FABRIC_1D_RING`, `ttnn.all_reduce` takes the
non-composite reduce-scatter + all-gather path, uses no persistent `GlobalSemaphore`
objects, and is **conditionally** trace-compatible. The one precondition is that the
intermediate `scattered_tensor` — a dynamic DRAM allocation created by
`ttnn::reduce_scatter` — must be pre-allocated as a persistent buffer before
`ttnn.begin_trace_capture` is called; without this step the non-composite path carries
the same address-instability risk as the composite path. For the full four-predicate
path-selection logic see [`call_chain.md`](./call_chain.md); for the trace-compatibility
verdict and T3K shape assessment see [Ch3 Q1](../ch3_verdict/q1_trace_compatibility.md).

## Learning objectives

After reading this chapter you will be able to:

1. Follow every function call from `TTNNLinearIColShardedWAllReduced.forward` to the
   hardware-level collective kernel invocation.
2. Identify the four predicates that choose between the composite path and the
   reduce-scatter + all-gather path.
3. Explain why the composite path is incompatible with trace, and why the non-composite
   path is only conditionally compatible (requiring pre-allocation of `scattered_tensor`
   before trace capture).
4. Contrast `ttnn.all_reduce` (no explicit semaphore args) with
   `ttnn.experimental.all_reduce_async` and `ttnn.experimental.reduce_scatter_minimal_async`
   (both require caller-supplied `GlobalSemaphore` handles from `TT_CCL`).

## Reading order

1. [`call_chain.md`](./call_chain.md) — full Python-to-C++ call chain and path-selection
   predicates
2. [`composite_path.md`](./composite_path.md) — composite path (`composite_all_gather` + local reduce); why it
   is incompatible with trace
3. [`reduce_scatter_all_gather_path.md`](./reduce_scatter_all_gather_path.md) —
   synchronous reduce-scatter + all-gather fallback; semaphore and allocation analysis
4. [`contrast_with_async_variants.md`](./contrast_with_async_variants.md) — comparison
   with `ttnn.experimental.all_reduce_async` and `reduce_scatter_minimal_async`; the
   cycling semaphore pattern

## What's next

1. [`call_chain.md`](./call_chain.md)
2. [`composite_path.md`](./composite_path.md)
3. [`reduce_scatter_all_gather_path.md`](./reduce_scatter_all_gather_path.md)
4. [`contrast_with_async_variants.md`](./contrast_with_async_variants.md)
