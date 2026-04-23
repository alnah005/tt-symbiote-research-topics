# Chapter 1 — Why `ttnn.from_torch` Breaks Metal Trace

This chapter establishes the foundational invariant that governs all Metal Trace usage: every device buffer touched during a trace capture run must exist at the same address on every subsequent replay. Understanding why `ttnn.from_torch` violates this invariant — and why that violation is non-obvious — is prerequisite to understanding the fix described in later chapters.

---

## Learning Objectives

By the end of this chapter you will be able to:

1. Describe the three phases of the Metal Trace lifecycle and explain what is recorded during the capture phase.
2. Explain why `ttnn.from_torch` is a host operation and cannot be safely called inside a traced region.
3. Distinguish between operations that are trace-safe (using pre-allocated device buffers) and operations that are trace-unsafe (allocating new device buffers).
4. Identify the exact call site in `TTNNQwen3FullAttention.forward` where `_ensure_replicated` breaks trace and explain the allocation it triggers.

---

## Trace Lifecycle Diagram

The three-phase lifecycle of a Metal Trace execution is shown below. The annotations mark where `_ensure_replicated` is called relative to the capture bracket.

```
Phase 1 — Compile Run
─────────────────────────────────────────────────────────────────────────
  Python forward() executes normally (no recording).
  Kernels are compiled and cached.
  _ensure_replicated() MAY be called here — safe, no trace active.

Phase 2 — Capture Run
─────────────────────────────────────────────────────────────────────────
  ttnn.begin_trace_capture(mesh_device, cq_id, trace_id)
  │
  │   [ CAPTURE BRACKET OPEN ]
  │
  │   Device commands are recorded to the command buffer:
  │     - kernel dispatches (concrete device addresses baked in)
  │     - DMA transfer descriptors
  │     - semaphore acquire / release operations
  │
  │   _ensure_replicated() called here  <── BUG
  │     └─ detects sharded cos/sin
  │         └─ calls ttnn.from_torch(device=mesh_device)
  │               └─ allocates NEW device buffer  <── TRACE SEES NEW ADDRESS
  │
  │   [ CAPTURE BRACKET CLOSE ]
  │
  ttnn.end_trace_capture(mesh_device, cq_id, trace_id)

Phase 3 — Replay Run(s)
─────────────────────────────────────────────────────────────────────────
  ttnn.execute_trace(mesh_device, cq_id, trace_id, blocking=False)
  │
  │   Recorded commands re-issued verbatim.
  │   No Python re-execution. No host-side allocation.
  │
  │   The buffer allocated by ttnn.from_torch during capture
  │   is NOT re-allocated. Commands reference a stale address.
  │        │
  │        └─► Silent data corruption or device crash.
─────────────────────────────────────────────────────────────────────────
```

> **Warning:** The bug does not surface during Phase 1 (compile run) or Phase 2 (capture run) — it surfaces only during Phase 3 (replay). This delayed failure makes the root cause difficult to diagnose without understanding the trace lifecycle.

---

## Glossary

**Host operation**
Any computation or memory allocation that executes on the host CPU rather than on the Tenstorrent device. Host operations are invisible to the Metal Trace recording mechanism. Examples: `torch.zeros`, `ttnn.from_torch`, Python control flow.

**Device buffer**
A contiguous region of memory allocated in device DRAM and referenced by a concrete physical address. Device buffers are the targets and sources of all kernel dispatches and DMA transfers recorded in a trace.

**Buffer address stability**
The invariant that a device buffer's base address does not change between the trace capture run and any subsequent replay run. Address stability is required for recorded DMA descriptors and kernel arguments to remain valid.

**Command buffer**
The low-level binary artifact produced by `ttnn.end_trace_capture`. It contains the serialized sequence of device commands — including concrete device memory addresses — that will be re-issued verbatim on each `ttnn.execute_trace` call.

**Metal Trace**
The TT-Metal subsystem that records device command sequences during a capture run and replays them at reduced host overhead on subsequent runs. Metal Trace achieves low-latency inference by eliminating Python re-execution and host-side buffer allocation from the critical path.

---

## Files in This Chapter (Reading Order)

1. [`what_trace_records.md`](./what_trace_records.md) — What the Metal Trace command buffer records and what it does not; the buffer address stability invariant stated precisely.
2. [`from_torch_is_a_host_operation.md`](./from_torch_is_a_host_operation.md) — Why `ttnn.from_torch` allocates a new device buffer on every call and why that allocation breaks trace.
3. [`ensure_replicated_call_site.md`](./ensure_replicated_call_site.md) — The exact call site of `_ensure_replicated` in `TTNNQwen3FullAttention.forward`, the original bug it was solving, and the fix required.

---

**Next:** [`what_trace_records.md`](./what_trace_records.md)
