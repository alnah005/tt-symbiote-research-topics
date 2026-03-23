# Chapter 5 — Estimating Improvement from Trace

Chapters 1 through 4 gave you the conceptual foundation: Chapter 1 established that each op dispatch costs 17–63 us across four host-side phases; Chapter 2 showed how async op mode pipelines encoding with device execution; Chapter 3 showed that trace eliminates phases 1–3 on replay by pre-recording a command buffer on the device; and Chapter 4 provided a decision framework for deciding whether trace is worth applying to a given workload. This chapter closes the loop with measurement: before you commit engineering time to implementing trace for a production model, you need concrete profiling data that quantifies how much of your current step latency is host dispatch overhead and how much of that trace can actually eliminate. Two profiling phases are required — a baseline run that establishes ground truth for current step latency, and an overhead-isolation pass that separates host dispatch time from kernel execution time — and together they feed a simple speedup model that predicts the post-trace latency with quantified confidence.

---

## Learning Objectives

After reading this chapter you will be able to:

- Use Tracy or the TTNN built-in profiler to instrument a decode step and capture per-op host dispatch times.
- Isolate host dispatch overhead from kernel execution time by reading the host-device timeline in a Tracy trace.
- Apply the speedup formula `speedup = total_step_time / (total_step_time - dispatch_overhead)` to your measured data.
- Explain what trace eliminates and what it cannot eliminate, and adjust the speedup prediction accordingly.
- Identify the diminishing-returns regime where kernel execution dominates and trace provides marginal benefit.
- Apply the implementation-cost checklist to decide whether the predicted speedup justifies the maintenance burden.
- Execute the end-to-end profiling workflow: instrument, baseline, capture trace, replay, compare, and regression-test.

---

## Conceptual Diagram: Host-Device Timeline and Where Overhead Lives

The following timeline shows a five-op decode step as seen in a Tracy trace. Time advances left to right. The top band is host activity; the bottom band is device activity.

```
Host thread (encoding + submission)
──────────────────────────────────────────────────────────────────────────────
  [enc A]  [enc B]  [enc C]  [enc D]  [enc E]
     │        │        │        │        │
     ▼        ▼        ▼        ▼        ▼
     ─────────────────────────────────────────── CQ submission ────────────►
                                                                            │
──────────────────────────────────────────────────────────────────────────────
Device (kernel execution)                                                   │
──────────────────────────────────────────────────────────────────────────────
         │ idle gap │                                                        │
         ├──────────┤◄── dispatch overhead                                  │
         [kernel A ]  [kernel B ]  [kernel C ]  [kernel D ]  [kernel E ]   │
──────────────────────────────────────────────────────────────────────────────
                                                                            ▼
                                                           step end (last kernel done)
```

The idle gaps on the device — periods where the device is waiting for the host to finish encoding the next command — represent the recoverable dispatch overhead. Tracy labels these as gaps between consecutive kernel execution spans. The sum of those gaps across all ops in a step is `dispatch_overhead`. The total wall-clock time from the first encoding to the last kernel completing is `total_step_time`.

> **Note:** In async op mode (Chapter 2), the host encoding band and the device kernel band overlap: the dispatch thread encodes op N+1 while the device runs op N. The idle gaps that remain visible in a Tracy trace of an async-mode loop are the gaps that trace will eliminate — they represent moments when even the async dispatch thread could not keep the device fed.

---

## Chapter Files

| File | What it covers |
|---|---|
| [`measuring_dispatch_overhead.md`](./measuring_dispatch_overhead.md) | How to use Tracy or the TTNN profiler to isolate host dispatch time from kernel execution time; key metrics to extract; how to read a Tracy trace for dispatch-dominated ops; reference numbers for typical hardware |
| [`estimating_trace_speedup.md`](./estimating_trace_speedup.md) | The speedup model formula; what trace eliminates vs. what it cannot; a worked example with a 32-op decode step; diminishing returns; implementation-cost checklist |
| [`profiling_workflow.md`](./profiling_workflow.md) | End-to-end workflow from instrumentation through regression testing; exact commands and environment variables; numerical validation of trace replay outputs |
