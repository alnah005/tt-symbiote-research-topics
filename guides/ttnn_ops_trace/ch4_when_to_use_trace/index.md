# Chapter 4 — When to Use Trace

Chapters 1 through 3 established the dispatch mechanics and the trace API. Chapter 1 showed that each op call pays 17–63 us in dispatch overhead across four phases; Chapter 2 showed that async op mode pipelines encoding with device execution but does not eliminate encoding costs; Chapter 3 showed that trace eliminates phases 1–3 entirely on replay by pre-recording the command buffer during a single capture run. This chapter synthesizes those foundations into a decision framework: given your workload, should you apply trace, and if so, to which region? The answer depends on two independent questions — whether dispatch overhead is a meaningful fraction of your step latency, and whether your compute sequence satisfies the structural constraints that make tracing safe.

---

## Learning Objectives

After reading this chapter you will be able to:

- Identify the conditions under which trace provides a meaningful latency reduction and distinguish them from the conditions under which trace adds maintenance burden without payoff.
- Explain why the decode loop is the canonical trace workload and why the prefill pass is usually not.
- State the three primary disqualifying conditions (with a closely related fourth pattern covered under Condition 2) for trace and recognize each one in a model's forward pass.
- Describe the maintenance cost that trace imposes and make an informed decision about whether that cost is acceptable for a given deployment scenario.
- Apply the decision flowchart to any candidate loop and produce a trace/no-trace recommendation with a rationale.

---

## The Core Question

Trace is a latency optimization, not a throughput optimization. It reduces the wall-clock time of a single step by removing host dispatch overhead from the critical path. Whether that reduction is worth the added complexity depends on how large a fraction of your step time dispatch overhead actually represents.

The calculation is straightforward: if a decode step takes 5 ms (5,000 us) end-to-end and contains 46 ops, total dispatch overhead is roughly 46 x (17–63 us) = 782 us–2,898 us. Dispatch overhead represents 16–58% of total step latency. Eliminating it via trace produces a proportional speedup. If the same step contained only 5 ops on a device where each kernel runs for 10 ms, dispatch overhead is 5 x 63 us = 315 us against a 50 ms step — less than 1%. Trace would produce a sub-1% improvement while imposing all of its structural constraints and maintenance cost.

The subsequent files in this chapter work through both sides of this decision: workloads where trace is the right tool, and workloads where it is not.

---

## Decision Flowchart: Should I Trace This Loop?

Use the following flowchart to evaluate any candidate compute loop.

```
START: You have a loop or repeated call sequence you want to optimize.
            │
            ▼
┌─────────────────────────────────────────────────────┐
│ Is this loop executed more than once?               │
│ (decode steps, repeated inference requests, etc.)   │
└─────────────────────────────────────────────────────┘
            │ NO                          │ YES
            ▼                             ▼
    ┌───────────────┐       ┌─────────────────────────────────┐
    │ DO NOT TRACE. │       │ Does each iteration execute the │
    │ Trace imposes │       │ same op sequence with fixed     │
    │ a one-time    │       │ tensor shapes throughout?       │
    │ capture cost. │       └─────────────────────────────────┘
    └───────────────┘           │ NO                  │ YES
                                ▼                     ▼
                    ┌─────────────────────┐  ┌─────────────────────────────────┐
                    │ DO NOT TRACE the    │  │ Does the loop body contain any  │
                    │ whole loop. Check   │  │ host readbacks, Python branches  │
                    │ if an inner region  │  │ on device values, or ops that   │
                    │ has fixed shapes.   │  │ self-configure on prior results?│
                    └─────────────────────┘  └─────────────────────────────────┘
                                                 │ YES               │ NO
                                                 ▼                   ▼
                                   ┌─────────────────────┐  ┌───────────────────────┐
                                   │ Refactor: move the  │  │ Is total dispatch     │
                                   │ untraceable parts   │  │ overhead a measurable │
                                   │ outside the capture │  │ fraction of step time?│
                                   │ boundary. Re-check. │  │ (rule of thumb: >5%)  │
                                   └─────────────────────┘  └───────────────────────┘
                                                                 │ NO      │ YES
                                                                 ▼         ▼
                                                        ┌──────────┐  ┌──────────────┐
                                                        │ Consider │  │ TRACE THIS   │
                                                        │ async op │  │ LOOP.        │
                                                        │ mode     │  │ Measure      │
                                                        │ instead. │  │ before and   │
                                                        └──────────┘  │ after.       │
                                                                       └──────────────┘
```

---

## Relationship to Chapters 1, 2, and 3

**From Chapter 1 (`host_dispatch_path.md`):** The 17–63 us per-op overhead across four dispatch phases is the cost that trace eliminates on replay. This chapter uses those numbers to evaluate whether overhead is large enough to justify tracing.

**From Chapter 2 (`pipelining_host_and_device.md`, `async_execution_model.md`):** Async op mode hides encoding latency by overlapping it with device execution. Chapter 4 distinguishes the regime where async mode is sufficient (host is faster than device, device is never starved) from the regime where async mode is not enough (device is faster than the host can encode, causing device stalls that trace eliminates).

**From Chapter 3 (`trace_constraints.md`, `trace_api.md`):** The four constraint categories — dynamic shapes, data-dependent dispatch, mid-loop host readbacks, and self-configuring ops — are the disqualifying checklist applied at the "does the loop contain untraceable ops" node in the flowchart above. This chapter synthesizes those constraints into a decision, not a re-explanation.

---

## Chapter Files

| File | What it covers |
|---|---|
| [`latency_sensitive_workloads.md`](./latency_sensitive_workloads.md) | Why trace is a latency (not throughput) optimization; the decode loop as canonical use case; prefill vs decode asymmetry; batch size effect on host overhead fraction |
| [`when_not_to_trace.md`](./when_not_to_trace.md) | Disqualifying conditions; maintenance cost of captured traces; debugging difficulty; the traced inner loop / untraced outer wrapper pattern |
