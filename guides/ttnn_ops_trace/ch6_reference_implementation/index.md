# Chapter 6 — Putting It All Together

This chapter synthesizes every concept from the preceding five chapters into a single, production-quality reference implementation. Rather than introducing new mechanisms, it shows exactly how the pieces fit: how device initialization from Chapter 1 sets the stage, how async dispatch from Chapter 2 is layered on, how the trace API from Chapter 3 wraps the fixed-shape decode core, how the Chapter 4 decision criteria justify applying trace to this exact workload, and how the Chapter 5 profiling methodology produces the before/after numbers that confirm the optimization is working. The two chapter files are structured so that `traced_decode_loop.md` gives you the annotated implementation and `operational_concerns.md` tells you how to keep it correct over the lifetime of a production deployment.

---

## Learning Objectives

After reading this chapter you will be able to:

- Write a complete traced autoregressive decode loop from scratch, including device initialization, async mode, tensor pre-allocation, capture phase, and replay loop.
- Identify which exact lines in the implementation eliminate host dispatch overhead and which lines retain it.
- Explain why each non-obvious decision (buffer pinning, sync placement, CQ selection, residual pattern) is necessary.
- Interpret before/after profiler output to confirm that the trace is eliminating the overhead it should.
- Enumerate the conditions that invalidate a trace at runtime and describe how to detect them.
- Implement runtime stale-trace detection and structured error recovery.
- Integrate trace capture and validation into a CI test suite.

---

## Map: Prior Chapters to Reference Implementation Sections

Each section of the reference implementation in `traced_decode_loop.md` is directly grounded in a prior chapter. The table below maps each section to its conceptual origin.

| Section in `traced_decode_loop.md` | Grounded in | What the prior chapter established |
|---|---|---|
| Device initialization (`ttnn.open_device`, `num_hw_cqs=1`) | [Ch1 — Command Queues](../ch1_dispatch_fundamentals/command_queues.md) | CQ0 is the primary compute queue; `num_hw_cqs=1` is the standard single-CQ configuration |
| `device.enable_async(True)` | [Ch2 — Async Execution Model](../ch2_async_ops/async_execution_model.md) | Enables the dispatch thread to encode op N+1 while the device executes op N, eliminating some host-wait gaps even without trace |
| Pre-allocated fixed-shape tensors (input, KV-cache, output) | [Ch3 — Trace Internals](../ch3_trace_capture/trace_internals.md) | Buffer aliasing: trace encodes DRAM addresses at capture time; tensors must remain at those addresses for the trace lifetime |
| Warm-up loop before capture | [Ch1 — Host Dispatch Path](../ch1_dispatch_fundamentals/host_dispatch_path.md) | Cold-path kernel selection in phase 2 of dispatch inflates the first few decode steps; warm-up amortizes this before capture |
| `ttnn.synchronize_device` before `begin_trace_capture` | [Ch2 — Pipelining Host and Device](../ch2_async_ops/pipelining_host_and_device.md) | Ensures no in-flight device work overlaps the start of capture; synchronization point semantics are defined in Chapter 2 |
| `ttnn.begin_trace_capture(device, cq_id=0)` | [Ch3 — Trace API](../ch3_trace_capture/trace_api.md) | Opens capture session; returns nothing; all subsequent ops on CQ0 are recorded until `end_trace_capture` |
| `decode_core` (the traced inner function) | [Ch4 — When Not to Trace](../ch4_when_to_use_trace/when_not_to_trace.md) | The traced inner loop / untraced outer wrapper pattern; untraceable ops (sampling, EOS) live outside the boundary |
| Residual connection pattern inside `decode_core` | [Ch3 — Trace Constraints](../ch3_trace_capture/trace_constraints.md) | Residual must reference the per-layer `hidden` state, not the original `input_tensor`, to correctly model the transformer stack |
| `trace_id = ttnn.end_trace_capture(device, cq_id=0)` | [Ch3 — Trace API](../ch3_trace_capture/trace_api.md) | Finalizes the command buffer; returns the `trace_id` handle used by `execute_trace`; capture outputs are valid |
| `ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)` | [Ch3 — Trace API](../ch3_trace_capture/trace_api.md) | Submits replay command; `blocking=False` is consistent with async mode; no re-encoding on replay — this is where dispatch overhead is eliminated |
| Per-step in-place tensor update before `execute_trace` | [Ch3 — Trace Internals](../ch3_trace_capture/trace_internals.md) | Tensors must be updated in-place at the same device address used during capture; allocating a new tensor breaks address fixity |
| End-of-loop `ttnn.synchronize_device` | [Ch2 — Pipelining Host and Device](../ch2_async_ops/pipelining_host_and_device.md) | Mandatory before host-side readback; the one unavoidable synchronization point per step |
| `ttnn.release_trace(device, trace_id)` | [Ch3 — Trace API](../ch3_trace_capture/trace_api.md) | Releases the trace buffer from device DRAM; after this call `execute_trace` with the same `trace_id` is undefined behavior |
| Before/after profiler output | [Ch5 — Profiling Workflow](../ch5_estimating_improvement/profiling_workflow.md) | The two-phase measurement methodology: baseline step latency and dispatch overhead isolation |
| Predicted vs measured speedup | [Ch5 — Estimating Trace Speedup](../ch5_estimating_improvement/estimating_trace_speedup.md) | `speedup = T / (T - D)` applied to the reference model's measured values |

---

## Why the Decode Loop Is the Canonical Case

The decode loop meets all four criteria established in Chapter 4 ([`latency_sensitive_workloads.md`](../ch4_when_to_use_trace/latency_sensitive_workloads.md)): fixed shapes, repeated execution, high op count per step, and dispatch overhead dominating device time. See Chapter 4 for the full analysis and timing breakdown.

---

## Chapter Files

| File | What it covers |
|---|---|
| [`traced_decode_loop.md`](./traced_decode_loop.md) | Fully annotated reference implementation: device init, async mode, tensor pre-allocation, warm-up, capture phase, decode loop with `execute_trace`, before/after profiler output, line-level annotation of where overhead is eliminated vs retained |
| [`operational_concerns.md`](./operational_concerns.md) | Re-capture triggers, runtime stale-trace detection, error handling for trace replay exceptions, CI integration strategies |
