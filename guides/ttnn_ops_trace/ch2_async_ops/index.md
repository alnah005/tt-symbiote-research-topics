# Chapter 2 — Asynchronous Op Execution

This chapter explains how TTNN ops can be dispatched asynchronously, allowing the host CPU to continue executing Python code while the device processes previously enqueued work. You will see why synchronous dispatch leaves performance on the table, how the async execution model repositions all four dispatch phases onto a background thread, and where the boundaries of safe asynchronous execution lie.

---

## Learning Objectives

After reading this chapter you will be able to:

- Describe the difference between synchronous and asynchronous dispatch and explain what changes for the calling Python thread in each model.
- Explain how the TTNN runtime tracks in-flight commands and preserves correctness across dependent ops when the host does not wait.
- Identify the three categories of synchronization point and describe what each one costs.
- Explain why the decode loop in autoregressive inference is the primary beneficiary of async execution.
- Predict when host-device pipelining will break down and what to do about it.

---

## Baseline: Synchronous Execution

In synchronous mode, the Python thread is blocked for ~17–63 us per op during the four dispatch phases. If encoding the next op takes longer than executing the current one, the device sits idle waiting for work — this idle gap is the overhead that async mode eliminates.

---

## Async Execution: The Key Difference

In async op mode, `ttnn` op calls return to the Python caller immediately — before any of the four dispatch phases begin. All four phases are handled by a dedicated background dispatch thread that the TTNN runtime maintains per device.

```
Asynchronous execution — host and device from Python's perspective
────────────────────────────────────────────────────────────────────────────────

Python thread (your code)
──────────────────────────────────────────────────────────────────────────────
 ttnn.matmul(a,b)──►return  ttnn.matmul(c,d)──►return  ttnn.softmax(e)──►return
        │                        │                          │
        │ enqueue to             │ enqueue to               │ enqueue to
        │ work queue             │ work queue               │ work queue
        ▼                        ▼                          ▼

Background dispatch thread
──────────────────────────────────────────────────────────────────────────────
               [dispatch A phases 1-4]──►[dispatch B phases 1-4]──►[dispatch C]

Device
──────────────────────────────────────────────────────────────────────────────
                              [kernel A runs]──────►[kernel B runs]──────►...
```

The Python thread queues op requests and returns immediately. The dispatch thread works through those requests sequentially, performing validation, kernel selection, encoding, and CQ submission for each. The device begins executing as soon as the first encoded command reaches the CQ.

This arrangement allows the Python thread to race ahead through the loop — building up a queue of pending op requests — while the dispatch thread and the device work through those requests concurrently. The result is that the device is rarely starved for work: by the time the dispatch thread finishes encoding op N, op N-1 is already executing, and by the time the device finishes op N-1, op N's command is waiting in the CQ.

> **Note:** The background dispatch thread is an internal runtime mechanism. You do not manage it directly. Enabling async mode is done at device open time (covered in [`async_execution_model.md`](./async_execution_model.md)) and the rest of the dispatch machinery is transparent to your Python code.

---

## Timeline Diagram: Host and Device Overlapping

The following timeline shows a sequence of five ops in async mode. Time advances left to right. Each row represents one actor; each labeled block is one unit of work.

```
Async execution timeline — five ops
────────────────────────────────────────────────────────────────────────────────
Time ──────────────────────────────────────────────────────────────────────────►

Python thread
├─[A]┤├─[B]┤├─[C]┤├─[D]┤├─[E]┤  (each bracket = Python enqueues the op)
  (returns ~immediately each time; the whole sequence takes ~5 us total in Python)

Dispatch thread
      ├──dispatch A──┤├──dispatch B──┤├──dispatch C──┤├──dispatch D──┤├──disp E─┤
                                                          (each ~20–40 us)

Device
               ├────── kernel A ──────┤├─── kernel B ───┤├── kernel C ──┤├──...──┤
```

Key observations from this diagram:

1. The Python thread's contribution to total latency is nearly zero — it only takes the time to enqueue five op requests, not to encode or submit them.
2. The dispatch thread and the device overlap: while the dispatch thread encodes B, the device is executing A.
3. The device has work ready in the CQ before it finishes each kernel, so there is no idle gap on the device.
4. The total wall-clock time from the first Python call to the last kernel completing is shorter than in synchronous mode, because the Python thread is not on the critical path.

---

## Chapter Files

| File | What it covers |
|---|---|
| [`async_execution_model.md`](./async_execution_model.md) | The async op model in detail: how the runtime tracks in-flight work, dependency management between ops, synchronization points, and a concrete two-matmul example |
| [`pipelining_host_and_device.md`](./pipelining_host_and_device.md) | Host-device pipelining: when it works, when it breaks down, how the decode loop benefits, and the cost of synchronization barriers |

