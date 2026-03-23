# Chapter 3 — Trace Capture and Replay

This chapter introduces TTNN trace as a first-class concept. Chapters 1 and 2 established the four-phase host dispatch path and showed how async op mode pipelines host encoding with device execution — but even a perfectly pipelined async loop still pays per-op encoding costs on every iteration. Trace eliminates those costs entirely by recording a pre-encoded command buffer during a single capture run and replaying that buffer on all subsequent iterations, bypassing phases 1–3 of dispatch on the host.

---

## Learning Objectives

After reading this chapter you will be able to:

- Describe the two-phase trace model: a capture phase and a replay phase, and explain what happens in each.
- Use `ttnn.begin_trace_capture`, `ttnn.end_trace_capture`, and `ttnn.execute_trace` correctly for a fixed-shape decode step.
- Explain what the trace records (command sequence, kernel arguments, buffer bindings) and what it does not record (Python logic, shape changes, dynamic control flow).
- Describe how the captured trace is stored on the device and why replay incurs no host encoding overhead.
- Explain buffer aliasing and state the constraint it imposes on input and output tensor addresses.
- Enumerate the categories of operation that cannot be traced and explain why each category is disqualified.
- Identify the prefill/decode asymmetry and describe how to structure a model to maximize the traceable region.

---

## Conceptual Diagram: Capture Phase vs Replay Phase

The following side-by-side swimlane shows what happens at the host and device levels during the capture phase (left) and a subsequent replay phase (right).

```
CAPTURE PHASE                              REPLAY PHASE
(one-time; slower)                         (every decode step; fast)
─────────────────────────────────          ──────────────────────────────────
Host (Python / TTNN runtime)               Host (Python / TTNN runtime)

  ttnn.begin_trace_capture(device, cq_id)    ttnn.execute_trace(device, trace_id, ...)
         │                                          │
         ▼                                          │  (no phases 1-3)
  [1] Arg validation         ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ (skipped)
  [2] Kernel selection       ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ (skipped)
  [3] Command encoding       ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ (skipped)
         │                                          │
         ▼                                          ▼
  [4] CQ submission +                        [4'] Device fetches pre-encoded
      trace recording                              command buffer from DRAM
         │                                          │
─────────────────────────────────          ──────────────────────────────────
Device (Tenstorrent hardware)              Device (Tenstorrent hardware)

  [5] Kernel execution (live)                [5'] Kernel execution (replay)
      + command buffer                            (same kernels, same
        written to device DRAM                     addresses, no re-encoding)
         │                                          │
         ▼                                          ▼
  ttnn.end_trace_capture(device, cq_id)      (output tensors updated in-place
  → returns trace_id                          at the same device addresses
                                              used during capture)
```

Each replay submits a single execute command — the entire pre-encoded buffer — rather than one command per op.

---

## Relationship to Chapters 1 and 2

**From Chapter 1:** Trace acts on the command encoding and CQ submission phases described in `host_dispatch_path.md`. The capture phase performs these phases exactly once and stores the result. All subsequent replays skip phases 1–3 entirely. The CQ and FIFO ordering guarantees described in `command_queues.md` still apply during replay: the device executes the trace's commands in the order they were recorded.

**From Chapter 2:** Trace and async op mode address different portions of the dispatch cost. Async mode hides encoding latency by overlapping it with device execution; trace eliminates encoding latency altogether. In practice, a traced decode loop uses both: async mode ensures the `ttnn.execute_trace` call is dispatched without blocking the Python thread, and trace ensures that when the device receives the execute command, it can complete the entire step without waiting for the host to encode any individual op.

The synchronization point semantics from `async_execution_model.md` still apply during trace replay. `ttnn.execute_trace` submits a single high-level command to the CQ; the Python thread does not automatically wait for it to finish. You still need `ttnn.synchronize_device()` or an event barrier if you need to observe the replay's output values on the host.

> **Note:** Trace does not replace async op mode. A correctly instrumented production decode loop uses both: `device.enable_async(True)` to pipeline the host dispatch of `ttnn.execute_trace` calls with device execution, and trace to eliminate per-op encoding inside each step.

---

## Chapter Files

| File | What it covers |
|---|---|
| [`trace_api.md`](./trace_api.md) | The three trace API functions; what is recorded and what is not; the minimal code pattern for capturing and replaying a decode step |
| [`trace_internals.md`](./trace_internals.md) | How the captured trace is stored on the device; why replay bypasses host encoding; buffer aliasing and its address-fixity constraint |
| [`trace_constraints.md`](./trace_constraints.md) | Operations that cannot be traced; the prefill/decode asymmetry; how to structure a model to maximize the traceable region |

