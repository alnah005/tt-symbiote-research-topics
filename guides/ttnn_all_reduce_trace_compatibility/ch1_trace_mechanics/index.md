# Chapter 1 — Trace Capture Mechanics on MeshDevice

This chapter establishes the mechanical contract that governs `ttnn.begin_trace_capture`, `ttnn.end_trace_capture`, and `ttnn.execute_trace` at the MeshDevice layer. Before a reader can assess whether `ttnn.all_reduce` is safe to use inside a traced region, they need a precise model of what the trace runtime records, what invariants it assumes during replay, and what can break those invariants. This chapter builds that model from first principles using the actual implementation paths in `tt-metal`.

## Context: The `TTNNLinearIColShardedWAllReduced` Question

`TTNNLinearIColShardedWAllReduced` is a symbiote linear module variant that performs a matmul followed by `ttnn.all_reduce` with `topology=Ring` and `cluster_axis=1`. The module inherits `@trace_enabled` from `TTNNLinear`, meaning the symbiote `TracedRun` execution mode will attempt to capture and replay its `forward` method as a device trace.

The central question this guide answers is: **does `ttnn.all_reduce` satisfy the trace replay contract?** That question has two sub-parts, both of which are grounded in the mechanics established here:

1. Do any of the buffers touched by `ttnn.all_reduce` (input, intermediate, output) remain at stable device addresses across all replays?
2. Does `ttnn.all_reduce` use any global semaphore state that persists between executions and could arrive at a wrong value on the second or later replay?

Chapter 1 defines the terms — persistent buffer, global semaphore, local semaphore, replay contract — with enough precision to answer these questions in Chapters 2 and 3.

## Learning Objectives

By the end of this chapter you will be able to:

- Describe what `begin_trace_capture` records and what the `MeshCommandQueue` serializes into a trace buffer.
- Explain why a warm-up (compile) run must precede capture and what the `TracedRun._capture_trace` method does to satisfy this requirement.
- State the buffer-address stability constraint precisely and identify the two buffer categories (persistent input buffers and persistent output buffers) that satisfy it.
- Distinguish between a global semaphore (device L1 location with a persistent value) and a local semaphore (per-program SRAM location re-initialized on every dispatch) and explain why only global semaphores create a re-entry risk in traced executions.

## Reading Order

Read the files in this order:

1. [`what_trace_records.md`](./what_trace_records.md) — What `begin_trace_capture` records, the warm-up run requirement, and the three-phase trace pattern used throughout the symbiote codebase.
2. [`buffer_address_stability.md`](./buffer_address_stability.md) — The buffer-address stability constraint, persistent input and output buffers, and why dynamic tensor allocation inside a trace causes silent corruption.
3. [`semaphore_initialization_and_replay.md`](./semaphore_initialization_and_replay.md) — Global vs local semaphore lifecycle, the re-entry requirement before each replay, and the first observation about `ttnn.all_reduce`'s semaphore argument handling.

## What's next

After completing this chapter, proceed to [Chapter 2 — ttnn.all_reduce Internal Architecture](../ch2_all_reduce_internals/index.md), which traces the full call chain from `ttnn.all_reduce` down to the specific sub-operations invoked on a T3K mesh and identifies what semaphore state, if any, those sub-operations hold between calls.
