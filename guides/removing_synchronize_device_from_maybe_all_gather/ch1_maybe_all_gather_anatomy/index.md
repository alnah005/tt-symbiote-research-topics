# Chapter 1 — `_maybe_all_gather`: Role, Call Sites, and the `synchronize_device` Call

Chapter 1 establishes the foundational problem this guide addresses. By the end of this chapter you will know exactly what `_maybe_all_gather` does, where it is invoked in both `TTNNQwen3FullAttention` and `TTNNQwen3LinearAttention`, and why the `ttnn.synchronize_device()` call embedded inside it creates a hard barrier that prevents Metal Trace capture from recording the hybrid attention stack.

---

## Core Problem

`_maybe_all_gather` is a shared helper method present in both `TTNNQwen3FullAttention` and `TTNNQwen3LinearAttention`. Its job is simple: if the model is running across multiple devices in tensor-parallel mode, gather a sharded tensor across the device mesh along the appropriate cluster axis so that the full tensor is available on every device before the next op consumes it. On a single-device deployment it is a no-op; on T3K (1×8 Wormhole mesh) it performs an actual collective communication.

The problem is that the current implementation calls `ttnn.synchronize_device(mesh_device)` inside `_maybe_all_gather` — either immediately before or after the all_gather op. This call drains the device command queue (CQ0) and blocks Python execution until every in-flight kernel on the device has completed. No new TTNN ops can be submitted to the device while the host is waiting.

This host-blocking behavior is fundamentally incompatible with Metal Trace capture. The `ttnn.begin_trace_capture` / `ttnn.end_trace_capture` bracket requires a fully async device command stream — the host may only enqueue device-side commands and must not block waiting for the device inside the bracket. `ttnn.synchronize_device` enqueues a Finish token to CQ0 (which IS recorded and IS replayed as a device fence), but its host-side blocking wait is NOT a device command and is NOT recorded; on replay the host never blocks at that point. If `_maybe_all_gather` is called from within a traced decode region — for example, from inside a `LayerStack` iteration enclosed in a `TracedRun.capture` — the synchronize call executes during capture (blocking the host normally), violating the contract that the capture bracket must be a fully async device command stream. The CQ0 FIFO ordering guarantee already ensures the all_gather output is available to the next enqueued op, making the `synchronize_device` call unnecessary for correctness.

Because every attention layer in the hybrid DeltaNet + full-attention decoder stack calls `_maybe_all_gather`, and because all such layers would be inside the same `LayerStack` trace bracket, the presence of `ttnn.synchronize_device` in `_maybe_all_gather` makes the entire layer stack non-traceable.

---

## Glossary

The following terms are used throughout this chapter and the rest of the guide. A full guide-level glossary is available in `plan.md`.

| Term | Definition |
|---|---|
| `_maybe_all_gather` | A method in `TTNNQwen3FullAttention` and `TTNNQwen3LinearAttention` that conditionally performs a tensor-parallel all_gather when the module is running on multiple devices. On a single device it is a no-op; on T3K it gathers a sharded tensor across the 1×8 mesh before the downstream op reads it. |
| `ttnn.synchronize_device` | A TTNN host API function — called as `ttnn.synchronize_device(mesh_device)` — that enqueues a Finish token to CQ0 (a device-side command) and then blocks Python execution until the device acknowledges it. The Finish token IS recorded by the Metal Trace capture mechanism and IS replayed as a device-level fence. However, the host-side blocking wait is NOT a device command and is NOT recorded; on replay the host never blocks. |
| host-blocking call | Any Python call that halts host execution until a device-side operation or drain completes. Examples include `ttnn.synchronize_device`, `ttnn.to_torch`, and `ttnn.from_device`. While the host is blocked, no new TTNN ops can be submitted to the device. |
| device command queue | The FIFO hardware queue (CQ0 in single-CQ mode) through which the host submits encoded kernel dispatch commands to the device. Ops are executed in the order they are submitted. In single-CQ mode, op N+1 cannot read its inputs until op N has written its outputs — this is the CQ0 ordering guarantee. |
| Metal Trace capture boundary | The region between a `ttnn.begin_trace_capture` call and the matching `ttnn.end_trace_capture` call. Within this region, the host may only enqueue TTNN ops that are device-side recordable; the capture bracket must be a fully async device command stream. Any host-blocking call inside this region (e.g. the host-side wait inside `ttnn.synchronize_device`) executes normally during capture but its blocking behavior is absent from the resulting trace buffer — only the underlying device-side command (e.g. the Finish token) is recorded and replayed. |

---

## Learning Objectives

After reading the three section files in this chapter you will be able to:

1. Describe the call sites of `_maybe_all_gather` in `TTNNQwen3FullAttention.forward` and `TTNNQwen3LinearAttention.forward`, including what tensor is passed in, when it is called relative to QKV projection and SDPA, and what the gating condition is.
2. Explain what `ttnn.synchronize_device(mesh_device)` does at the TTNN/tt-metal level, why it is a host-blocking call, and why the single-CQ FIFO ordering guarantee makes it unnecessary for sequencing purposes.
3. Explain why `ttnn.synchronize_device` inside `_maybe_all_gather` prevents Metal Trace capture from working, and articulate the scope of the problem across the full hybrid decoder layer stack.

---

## What's Next

Read the following files in order:

1. [`call_sites_and_control_flow.md`](./call_sites_and_control_flow.md) — Where `_maybe_all_gather` is called in `TTNNQwen3FullAttention.forward` and `TTNNQwen3LinearAttention.forward`, what tensors are passed, and when the synchronize call is conditionally executed.

2. [`synchronize_device_semantics.md`](./synchronize_device_semantics.md) — What `ttnn.synchronize_device` does at the TTNN/tt-metal level, why it is host-blocking, and the two plausible reasons it could be present in `_maybe_all_gather`.

3. [`why_this_blocks_trace_capture.md`](./why_this_blocks_trace_capture.md) — The Metal Trace capture contract, what happens when `ttnn.synchronize_device` is called inside a trace bracket, and the full scope of the problem across the hybrid decoder layer stack.

---

**Next:** [`call_sites_and_control_flow.md`](./call_sites_and_control_flow.md)
