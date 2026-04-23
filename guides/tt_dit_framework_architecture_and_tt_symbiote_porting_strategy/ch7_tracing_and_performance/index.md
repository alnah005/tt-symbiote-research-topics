# Chapter 7: Tracing and Performance

## Prerequisites

- [Chapter 1 -- Architecture Overview](../ch1_architecture_overview/index.md): understanding of TT-DiT's direct-TTNN execution model vs. TT-Symbiote's dispatch interception.
- [Chapter 2 -- Parallelism and CCL](../ch2_parallelism_and_ccl/index.md): awareness of CCL collective operations (all-gather, reduce-scatter) and how they interact with trace capture.
- [Chapter 5 -- Pipelines and Serving](../ch5_pipelines_and_serving/index.md): familiarity with TT-DiT pipeline classes and the denoising loop structure.

## Introduction

Diffusion Transformer inference is dominated by a single, repeated operation: the denoising loop. A typical image generation invocation runs the same transformer forward pass 20--50 times (SD3.5 default: 28 steps, Flux1: 20--50, Motif: 28). A video model like Mochi or Wan may require hundreds of denoising steps per clip. Each forward pass is structurally identical -- the same sequence of matmuls, normalizations, attention computations, and collective operations -- differing only in the input tensors (noisy latents, timestep embedding, and occasionally guidance scalars).

This structural repetition creates a massive opportunity for **trace-based optimization**. Instead of dispatching hundreds of individual TTNN operations to the device on every step, the host can:

1. **Record** the full operation sequence once into a device-side trace buffer.
2. **Update** only the changing input tensors (latents, timestep, sigma difference).
3. **Replay** the entire trace in a single `ttnn.execute_trace` call.

The performance impact is dramatic. Host dispatch overhead -- the time the CPU spends issuing operation commands to the device -- is eliminated for every replayed step. On Tenstorrent hardware, this can improve denoising loop throughput by 10x--50x depending on model size and device topology, as documented in empirical measurements on T3K configurations.

This chapter examines two fundamentally different approaches to tracing:

- **TT-DiT's `Tracer` class and `PipelineTrace` dataclass** -- pipeline-level tracing where the entire denoising step (transformer forward + scheduler arithmetic) is captured as a single trace.
- **TT-Symbiote's `TracedRun` class** -- module-level tracing where individual TTNNModule subclasses are independently traced and cached based on their input signatures.

### Why Two Approaches Exist

The distinction follows directly from the architectural divide described in Chapter 1. TT-DiT controls the full execution graph -- every operation is an explicit TTNN call -- so it can wrap an entire pipeline step in a single trace. TT-Symbiote intercepts PyTorch dispatch at the operation level and manages a heterogeneous collection of modules, some running on device and some falling back to CPU, so it must trace at the module granularity.

## Chapter Contents

| File | Description |
|------|-------------|
| [`tt_dit_tracer.md`](./tt_dit_tracer.md) | TT-DiT's `Tracer` class: function-level trace capture and replay, `_tree_map` traversal, `PipelineTrace` dataclass integration in production pipelines. |
| [`symbiote_traced_run.md`](./symbiote_traced_run.md) | TT-Symbiote's `TracedRun`: module-level tracing with three-phase lifecycle, `@trace_enabled`/`@trace_disabled` decorators, `TTNNLayerStack`, input buffer management, `pre_trace_execute`/`post_trace_execute` hooks. |
| [`integration_strategy.md`](./integration_strategy.md) | Comparison of pipeline-level vs. module-level tracing, recommended approach for porting DiT models to TT-Symbiote, and CCL-aware extensions needed. |

## The TTNN Trace Primitive

Both systems build on the same underlying TTNN trace API. Understanding this primitive is essential before diving into either framework's wrapper.

```
ttnn.begin_trace_capture(device, cq_id=0)  -->  trace_id
  ... execute operations ...
ttnn.end_trace_capture(device, trace_id, cq_id=0)

ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)

ttnn.release_trace(device, trace_id)
```

**Semantics:** Between `begin_trace_capture` and `end_trace_capture`, the device records every operation into a DRAM-resident command buffer instead of executing it. The resulting `trace_id` can then be replayed with `execute_trace`, which re-issues the entire recorded command sequence with near-zero host overhead.

**Constraints:** During trace capture, the device memory allocator is frozen -- no new allocations or deallocations are permitted. All tensor buffers used during capture must be pre-allocated. This means:

- Input tensors must be allocated on-device *before* capture begins.
- Weights must already reside on-device.
- The output tensors produced during capture become the persistent output buffers -- `execute_trace` overwrites them in place on every replay.
- Host-to-device transfers (`ttnn.copy_host_to_device_tensor`) and device-to-device copies (`ttnn.copy`) are used to update input buffers between replays.

**CCL interaction:** Collective communication operations (all-gather, reduce-scatter) issued during trace capture are also recorded. This means the trace replays the full distributed computation graph, including inter-device communication. However, CCL operations must be fully synchronized *before* trace capture begins (see `ttnn.synchronize_device` calls in both TT-DiT and TT-Symbiote implementations).

## Key Takeaways

1. **Denoising loop repetition is the primary performance lever.** Running the same forward pass 20--50+ times with only input tensor changes makes tracing the single most impactful optimization for DiT inference.

2. **The TTNN trace API eliminates host dispatch overhead** by recording the operation sequence once and replaying it from device-side DRAM, reducing per-step host work to input tensor copies.

3. **Memory constraints during trace capture** require all buffers (inputs, weights, outputs) to be pre-allocated before the capture window opens.

4. **TT-DiT and TT-Symbiote wrap the same primitive differently** -- pipeline-level vs. module-level -- reflecting their distinct execution models.

5. **CCL operations are trace-compatible** but require explicit synchronization before capture to avoid in-flight operation conflicts.

---

**Next:** [`tt_dit_tracer.md`](./tt_dit_tracer.md)
