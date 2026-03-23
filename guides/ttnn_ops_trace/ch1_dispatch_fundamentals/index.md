# Chapter 1 — TTNN Dispatch Fundamentals

This chapter establishes the mental model you need before tackling async execution or trace. By the end, you will understand how a single Python call like `ttnn.matmul(a, b)` travels from your script through the TTNN runtime and arrives as executable work on the Tenstorrent device, and you will have working definitions for the core vocabulary — dispatch, kernel, and command queue — that every subsequent chapter builds on.

---

## Learning Objectives

After reading this chapter you will be able to:

- Describe each phase of the host dispatch path from Python op call to device kernel execution.
- Explain why host dispatch time is non-trivial and appears as measurable overhead in profiler traces.
- Define a command queue (CQ) and distinguish the roles of CQ0 and CQ1.
- Explain the FIFO ordering guarantee and its implications for correctness.
- Reason about the tradeoffs between single-CQ and dual-CQ configurations.

---

## Conceptual Diagram

The following swimlane shows the three layers of the dispatch path. The host thread (top) does all encoding work; the device (bottom) executes what it receives through the command queue channel.

```
Host (Python / TTNN runtime)
─────────────────────────────────────────────────────────────────────────────
  ttnn.matmul(a, b)
       │
       ▼
  [1] Argument validation
       │
       ▼
  [2] Kernel selection
       │
       ▼
  [3] Command encoding
       │
       ▼
  [4] CQ submission  ──────────────────────────────────────────────────────►
                                                                            │
─────────────────────────────────────────────────────────────────────────────
Device (Tenstorrent hardware)                                               │
─────────────────────────────────────────────────────────────────────────────
                                                        ◄───────────────────┘
                                                    [5] CQ dequeue
                                                         │
                                                         ▼
                                                    [6] Kernel execution
                                                         │
                                                         ▼
                                                    [7] Output written
                                                        to device DRAM
```

Every `ttnn` op call follows this path. The time the host spends in phases 1–4 before returning control to your Python script is the **dispatch overhead** for that op. Chapters 3–5 explain how trace eliminates phases 1–3 on subsequent executions.

---

## Glossary

The terms below are used throughout this guide. All definitions here are scoped to how TTNN uses them; some terms have broader meanings in other contexts.

**dispatch**
The host-side process of validating op arguments, selecting a kernel implementation, encoding a command, and submitting it to the device through a command queue. Dispatch happens on the CPU host thread and is entirely separate from the device executing the resulting kernel.

**kernel**
A compiled program that runs on the Tenstorrent device cores. TTNN selects the appropriate kernel binary for each op based on the tensor data types, memory layout, and tile dimensions involved. The kernel is the actual compute work; dispatch is the bookkeeping that precedes it.

**CQ (command queue)** — The ordered FIFO channel from host to device. See [`command_queues.md`](./command_queues.md) for the full explanation.

**CQ0** — The primary command queue; all compute ops are submitted here by default. See [`command_queues.md`](./command_queues.md) for the full explanation.

**CQ1** — The secondary command queue used for data movement in dual-CQ mode. See [`command_queues.md`](./command_queues.md) for the full explanation.

**dispatch overhead** — The host CPU time spent in phases 1–4 of dispatch before a command reaches the device. See [`host_dispatch_path.md`](./host_dispatch_path.md) for the full explanation.

**command encoding** — The serialization of a kernel invocation into the binary format the device firmware reads from the CQ. See [`host_dispatch_path.md`](./host_dispatch_path.md) for the full explanation.

**synchronization point**
Any operation that causes the host to block until the device has completed a specified amount of in-flight work. Common synchronization points include `ttnn.synchronize_device()`, tensor data readback to host memory, and event barriers. Synchronization points are covered in detail in Chapter 2.

**trace**
A pre-recorded, pre-encoded command buffer stored on the device. When a trace is replayed, the device re-executes the recorded command sequence without the host repeating phases 1–3 of dispatch. Trace is the central subject of Chapter 3.

**decode step**
One autoregressive token generation step. In large language model inference, the decode loop executes the same op sequence repeatedly with a small, fixed-shape tensor representing the current token. Because shapes are fixed and the loop runs thousands of times, the decode step is the primary workload targeted by trace optimization.

---

## What's Next

Read the chapter files in this order:

1. [`host_dispatch_path.md`](./host_dispatch_path.md) — Start here to understand what the host is doing on every op call before you consider how to optimize it.
2. [`command_queues.md`](./command_queues.md) — Understand the channel that carries host work to the device, and the CQ0/CQ1 distinction you will see referenced in later chapters.
