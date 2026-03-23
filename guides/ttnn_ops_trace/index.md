# TTNN Ops Trace — Guide Index

This guide explains how the TTNN runtime dispatches ops to Tenstorrent devices, how async op mode pipelines host and device work, and how trace capture eliminates per-op encoding overhead entirely — giving ML engineers the foundation to decide when trace applies to their workload, measure the expected benefit, and implement a production-quality traced inference loop. The intended reader is an ML engineer who already writes TTNN-based inference models and wants a systematic understanding of the dispatch pipeline and trace mechanism.

---

## How to Use This Guide

Different readers arrive with different goals. Use the table below to find the most direct path to what you need.

| Goal | Recommended path | Direct links |
|---|---|---|
| Understand what "dispatch overhead" means and where it comes from | Read linearly from the start | [Ch 1 — Dispatch Fundamentals](ch1_dispatch_fundamentals/index.md) → [host_dispatch_path.md](ch1_dispatch_fundamentals/host_dispatch_path.md) |
| Understand the trace API and how to use it | Skip to Chapter 3 after reading Chapter 1 | [Ch 1 index](ch1_dispatch_fundamentals/index.md) → [Ch 3 — Trace Capture](ch3_trace_capture/index.md) → [trace_api.md](ch3_trace_capture/trace_api.md) |
| Decide whether trace is worth applying to your model | Read Chapters 4 and 5; skim Chapters 1–3 as needed | [Ch 4 — When to Use Trace](ch4_when_to_use_trace/index.md) → [Ch 5 — Estimating Improvement](ch5_estimating_improvement/index.md) |
| Measure dispatch overhead in your existing model | Go directly to Chapter 5 | [measuring_dispatch_overhead.md](ch5_estimating_improvement/measuring_dispatch_overhead.md) → [profiling_workflow.md](ch5_estimating_improvement/profiling_workflow.md) |
| Implement a traced decode loop from scratch | Read the reference implementation; use prior chapters as reference | [Ch 6 — Reference Implementation](ch6_reference_implementation/index.md) → [traced_decode_loop.md](ch6_reference_implementation/traced_decode_loop.md) |
| Understand async op mode and host-device pipelining | Read Chapter 2 after Chapter 1 | [Ch 2 — Asynchronous Op Execution](ch2_async_ops/index.md) → [pipelining_host_and_device.md](ch2_async_ops/pipelining_host_and_device.md) |
| Understand what cannot be traced and why | Read Chapter 3 trace constraints | [trace_constraints.md](ch3_trace_capture/trace_constraints.md) → [when_not_to_trace.md](ch4_when_to_use_trace/when_not_to_trace.md) |
| Handle trace invalidation and operational concerns in production | Go directly to Chapter 6 operational concerns | [operational_concerns.md](ch6_reference_implementation/operational_concerns.md) |

---

## Chapter Index

| # | Title | Description | Key operations and concepts |
|---|---|---|---|
| 1 | [Ch 1 — Dispatch Fundamentals](ch1_dispatch_fundamentals/index.md) | Establishes the mental model of how TTNN translates a Python op call into device-executable work, covering the four-phase host dispatch path and defining the core vocabulary every subsequent chapter builds on | Host dispatch path, command queues, CQ0/CQ1, dispatch overhead, kernel selection, command encoding |
| 2 | [Ch 2 — Asynchronous Op Execution](ch2_async_ops/index.md) | Explains how TTNN async op mode repositions all four dispatch phases onto a background thread so the Python caller returns immediately, and how this creates host-device pipelining that removes most device idle gaps without any trace | `device.enable_async(True)`, `ttnn.synchronize_device(device)`, synchronization points, dispatch thread, host-device pipelining |
| 3 | [Ch 3 — Trace Capture and Replay](ch3_trace_capture/index.md) | Covers the trace API in full: what the capture phase records (command sequence, kernel arguments, buffer bindings), what it does not record (Python logic, dynamic shapes), how the pre-encoded command buffer is stored on-device, and which operations cannot be traced | `ttnn.begin_trace_capture`, `ttnn.end_trace_capture`, `ttnn.execute_trace`, `ttnn.release_trace`, buffer aliasing, trace constraints, prefill/decode asymmetry |
| 4 | [Ch 4 — When to Use Trace](ch4_when_to_use_trace/index.md) | Synthesizes the prior chapters into a decision framework: conditions under which trace provides a meaningful latency reduction, disqualifying conditions, the decode loop as the canonical trace workload, and the maintenance cost that trace imposes | Decision flowchart, latency vs throughput, decode step, dynamic-shape disqualification, maintenance cost |
| 5 | [Ch 5 — Estimating Improvement](ch5_estimating_improvement/index.md) | Gives a concrete methodology for measuring host dispatch overhead with Tracy or the TTNN profiler, applying the speedup formula, and validating that a captured trace is eliminating the overhead it should | `speedup = T / (T - D)`, Tracy host-device timeline, overhead isolation, profiling workflow, regression testing |
| 6 | [Ch 6 — Reference Implementation](ch6_reference_implementation/index.md) | Synthesizes all prior chapters into a production-quality annotated decode loop, including device initialization, async mode, tensor pre-allocation, capture phase, replay loop, before/after profiler output, and operational concerns for long-running deployments | `ttnn.execute_trace`, `ttnn.copy_`, `ttnn.release_trace`, stale-trace detection, re-capture triggers, CI integration |

---

## Quick Reference

The table below covers the most-used trace and async API calls. For full usage context, follow the links.

| API call | What it does | Where to learn more |
|---|---|---|
| `device.enable_async(True)` | Enables async op mode: op calls return immediately and all four dispatch phases are handled by a background dispatch thread | [async_execution_model.md](ch2_async_ops/async_execution_model.md) |
| `ttnn.synchronize_device(device)` | Blocks the host until all in-flight device work on all command queues is complete; required before host-side readback and before `begin_trace_capture` | [async_execution_model.md](ch2_async_ops/async_execution_model.md), [pipelining_host_and_device.md](ch2_async_ops/pipelining_host_and_device.md) |
| `ttnn.begin_trace_capture(device, cq_id)` | Opens a trace capture session on the specified command queue; all subsequent ops submitted to that CQ are recorded until `end_trace_capture` is called; returns nothing | [trace_api.md](ch3_trace_capture/trace_api.md) |
| `ttnn.end_trace_capture(device, cq_id)` | Closes the capture session and finalizes the pre-encoded command buffer on the device; returns a `trace_id` integer handle used by `execute_trace` and `release_trace` | [trace_api.md](ch3_trace_capture/trace_api.md) |
| `ttnn.execute_trace(device, trace_id, cq_id, blocking)` | Submits a replay command to the device; the device fetches and re-executes the pre-encoded command buffer without the host re-encoding any individual op; pass `blocking=False` when async mode is active | [trace_api.md](ch3_trace_capture/trace_api.md), [traced_decode_loop.md](ch6_reference_implementation/traced_decode_loop.md) |
| `ttnn.release_trace(device, trace_id)` | Releases the device DRAM allocation holding the captured command buffer; after this call, any `execute_trace` with the same `trace_id` is undefined behavior | [trace_api.md](ch3_trace_capture/trace_api.md), [operational_concerns.md](ch6_reference_implementation/operational_concerns.md) |
| `ttnn.copy_(dst, src)` | Performs an in-place device-to-device tensor copy, writing new input values into the buffer at the address recorded during capture; this is the correct pattern for updating per-step inputs before `execute_trace` | [trace_internals.md](ch3_trace_capture/trace_internals.md), [traced_decode_loop.md](ch6_reference_implementation/traced_decode_loop.md) |

---

## Prerequisites

This guide assumes you are already comfortable with the following:

- **TTNN Python API basics** — creating tensors with `ttnn.from_torch`, calling ops such as `ttnn.matmul` and `ttnn.softmax`, moving data between host and device with `ttnn.to_device` and `ttnn.from_device`.
- **TTNN device initialization** — opening and closing a device with `ttnn.open_device` and `ttnn.close_device`, passing device configuration arguments such as `num_hw_cqs`.
- **Autoregressive inference loop structure** — the distinction between a prefill pass (full prompt, variable sequence length, run once) and a decode loop (single token, fixed shapes, run many times).
- **Python async concepts at a surface level** — knowing what it means for a function to return before its underlying work is complete, and that background threads can process a queue of pending tasks. You do not need deep asyncio knowledge; TTNN async mode is controlled by a single flag and is opaque to the caller.
- **Basic familiarity with a profiler** — you have looked at profiler output before and understand the concept of wall-clock time vs CPU time. Chapter 5 teaches TTNN-specific profiling; general profiler fluency is assumed.

No knowledge of TTNN internals, Tenstorrent hardware architecture, or the trace mechanism is assumed. Those are the topics this guide covers.

---

## Source Code Location

This guide covers the TTNN Python API. TTNN is part of the TT-Metalium open-source project. The canonical source repository is:

```
https://github.com/tenstorrent/tt-metal
```

The TTNN Python API layer lives under `ttnn/` in that repository. The dispatch infrastructure, command queue implementation, and trace subsystem are implemented in C++ under `tt_metal/`. When this guide describes internal mechanics — such as how the host encodes a command or how the device fetches a pre-recorded buffer — the descriptions refer to the behavior of that codebase as of the TTNN version documented here. Always cross-reference against the version of `tt-metal` you are running if you observe behavior that differs from what is described.

API reference documentation for TTNN is maintained alongside the source at:

```
https://docs.tenstorrent.com/ttnn/
```
