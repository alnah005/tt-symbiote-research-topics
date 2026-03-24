# Chapter 1 — Trace Capture in TTNN: The Core API

## Overview

Trace capture is the mechanism by which TTNN records a sequence of device operations into a
replayable command buffer. Once captured, that buffer can be dispatched to the device repeatedly
with minimal host-side overhead — no Python re-execution, no op re-encoding, no kernel
re-selection. This is the foundation of the decode-loop performance in tt-transformers inference.

This chapter covers the three-phase lifecycle of a trace, the public Python API that drives it,
and the constraints that must be respected to produce a correct and efficient trace.

---

## The Three-Phase Lifecycle

Every trace passes through three distinct phases. Understanding which phase is active at any
moment is essential for debugging and for reasoning about buffer lifetimes.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Python host                                                                          │
│  [1. Compile run]          [2. Capture run]                [3. Replay runs ×N]       │
│  forward_pass()            begin_trace_capture()           execute_trace() ──┐       │
│                            forward_pass()                  execute_trace() ──┤       │
│                            end_trace_capture()             execute_trace() ──┘       │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ TTNN runtime                                                                         │
│  compile kernels           record command buffer           enqueue command buffer    │
│  warm program cache        snapshot buffer addresses       (no re-encoding)          │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ Device                                                                               │
│  upload kernel binaries    execute ops (addresses fixed)   read pre-encoded cmds    │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**Phase 1 — Compile run.** A normal, untraced forward pass. Its purpose is to compile TTNN ops,
warm the program cache, and upload kernel binaries to the device. No command buffer is produced.

**Phase 2 — Capture run.** A second forward pass executed between `ttnn.begin_trace_capture` and
`ttnn.end_trace_capture`. TTNN intercepts every device command and records it, along with the
exact device memory addresses of all buffers in use at that moment, into a command buffer stored
in a reserved DRAM region called the trace region.

**Phase 3 — Replay runs.** Each call to `ttnn.execute_trace` dispatches the pre-encoded command
buffer to the device. The host does not re-encode any commands. Input tensors must remain at the
same device addresses that were recorded during Phase 2.

---

## Glossary

**trace**
A pre-encoded, device-resident command buffer produced by a capture run. A trace encodes the
complete sequence of device commands — kernel launches, NOC transfers, synchronization barriers —
for one full forward pass at fixed buffer addresses.

**command buffer**
The binary representation of the recorded op sequence. Stored in the trace region of DRAM.
On replay, the device firmware reads this buffer directly without further host intervention.

**buffer aliasing**
Reusing the exact DRAM addresses recorded at capture time for all replay inputs; required
because the command buffer hard-codes those addresses. See
[`replay_mechanics.md`](./replay_mechanics.md) for the full constraint and the
`copy_host_to_device` pattern.

**`cq_id`**
Command queue identifier. TTNN dispatches device work through one or more hardware command
queues. `cq_id=0` is the standard single-queue mode used throughout tt-transformers. The trace
is associated with the command queue on which it was captured, and must be replayed on the same
queue.

**`trace_id`**
A `MeshTraceId` value returned by `ttnn.begin_trace_capture`. It uniquely identifies a captured
trace on a given `MeshDevice` and is the handle passed to `ttnn.end_trace_capture` and
`ttnn.execute_trace`.

**trace region**
A dedicated, statically reserved DRAM allocation on the device that holds command buffers. Its
size is configured via `trace_region_size` in `device_params` before the mesh device is opened.
If the trace exceeds the reserved size, an out-of-memory error occurs at capture time.

---

## Learning Objectives

After reading this chapter you should be able to:

1. Describe what each of the three trace phases does and why all three are required.
2. Call `ttnn.begin_trace_capture`, `ttnn.end_trace_capture`, and `ttnn.execute_trace` in the
   correct order and understand what each call returns or seals.
3. Explain why a compile run must precede every capture run and what goes wrong when it is
   skipped.
4. Identify what is recorded in the command buffer and what is explicitly excluded.
5. Explain why replay is faster than live dispatch and what buffer aliasing requires of caller
   code.
6. Choose an appropriate `trace_region_size` value for a given model and device combination,
   using the reference values in `trace_region_config.py`.

---

## Reading Order

Read the files in this chapter in the following order:

1. [`trace_api_overview.md`](./trace_api_overview.md) — the three public API calls and what they
   record
2. [`compile_run_requirement.md`](./compile_run_requirement.md) — why a compile run must precede
   every capture, and how `trace_region_size` is configured
3. [`replay_mechanics.md`](./replay_mechanics.md) — why replay is fast, buffer aliasing, and
   synchronization after `execute_trace`
