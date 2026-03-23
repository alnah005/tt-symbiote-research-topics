# Command Queues

This file defines the command queue (CQ) — the ordered channel through which the host sends work to the Tenstorrent device — and explains how TTNN uses one or two queues to organize compute and data movement work. By the end you will understand how commands are enqueued and consumed, why FIFO ordering matters for correctness, and what it means to choose between single-CQ and dual-CQ operation.

---

## What a Command Queue Is

A command queue is a circular buffer in memory that the host CPU writes to and the device firmware reads from. Every encoded command that results from a TTNN op call is placed into a CQ as the final step of host dispatch. The device firmware polls the CQ continuously; when it finds a new command, it dequeues it and begins setting up the corresponding kernel execution.

The critical property of a CQ is its ordering guarantee: **commands are consumed by the device in exactly the order they were enqueued**. If the host enqueues op A and then op B, the device will always begin executing A before it begins executing B. This guarantee holds even if op A is a heavyweight matmul and op B is a trivial elementwise add — op B cannot skip the queue.

This ordering guarantee is what makes TTNN programs correct without explicit per-op synchronization between ops in the same queue. If op B reads from a buffer that op A writes to, you do not need to insert a barrier between them as long as both are enqueued to the same CQ. The reasoning is a two-step chain: (1) FIFO ordering guarantees that A is *issued* to the device before B is issued, and (2) the device processes commands sequentially within a single CQ — it does not begin executing B until A's execution is complete. Together, these two properties imply that A completes before B starts. This chain is specific to a single CQ; it does not extend across two queues, where neither property holds automatically.

---

## Physical Structure of a CQ

At the hardware level, a CQ is implemented as a region of host-pinned memory (accessible from both the CPU and the device over PCIe or the on-chip fabric, depending on the platform). The host maintains a write pointer; the device firmware maintains a read pointer. The host advances the write pointer after each command is written; the firmware advances the read pointer after each command is dispatched to a device core for execution.

```
Host write pointer
        │
        ▼
┌───────┬───────┬───────┬───────┬───────┬───────┐
│  cmd  │  cmd  │  cmd  │ empty │ empty │ empty │
│   A   │   B   │   C   │       │       │       │
└───────┴───────┴───────┴───────┴───────┴───────┘
                        ▲
                        │
                Device read pointer
```

In this snapshot, the host has enqueued three commands. The device has dispatched A, B, and C to device cores (its read pointer is past all three); C is currently executing on the cores. The host can continue writing new commands to the empty slots without waiting for C to finish.

When the CQ is full — all slots are occupied by commands the device has not yet consumed — the host must stall and wait for the device to free slots before it can enqueue more work. This is the backpressure mechanism that prevents unbounded memory use.

---

## CQ0: The Primary Compute Queue

`CQ0` is the default command queue used by all `ttnn` op calls unless otherwise specified. Every call to `ttnn.matmul`, `ttnn.softmax`, `ttnn.layernorm`, and so on places its encoded command into CQ0.

CQ0 is also used for device-side data movement operations in single-CQ mode: `ttnn.to_device`, `ttnn.from_device`, and similar transfers enqueue DMA commands into CQ0 alongside compute commands.

> **Note:** Because compute and data movement share CQ0 in single-CQ mode, a large tensor transfer that is enqueued before a compute op will block that compute op until the transfer completes. This is correct behavior (the compute op may need the transferred data), but it means the device cannot overlap a tensor load with the preceding op's compute. Dual-CQ mode, described below, addresses this limitation.

You pass a CQ identifier to TTNN APIs that accept one via the `cq_id` parameter. When no `cq_id` is specified, TTNN defaults to `0` (CQ0).

<details>
<summary>Example: explicit CQ selection on a tensor write</summary>

```python
import ttnn

device = ttnn.open_device(device_id=0)

host_tensor = ttnn.from_torch(
    torch.randn(1, 1, 512, 512),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
)

# Explicitly enqueue the device write through CQ0 (the default).
device_tensor = ttnn.to_device(host_tensor, device, cq_id=0)

# Compute op also goes through CQ0.
output = ttnn.softmax(device_tensor, dim=-1)
```

Because both the write and the compute op share CQ0 and FIFO ordering is guaranteed, the softmax will not begin until the tensor write is complete.
</details>

---

## CQ1: The Secondary Queue

`CQ1` is a second, independent command queue that TTNN supports on platforms with dual-CQ capability. CQ1 operates in parallel with CQ0: the device firmware services both queues concurrently, interleaving execution based on availability.

The typical use of CQ1 is to decouple data movement from compute. The conventional split is:

- **CQ0** carries compute ops (the main forward pass).
- **CQ1** carries tensor transfers — moving the next batch of input data from host to device DRAM while the current batch is being processed on CQ0.

This is a recommended pattern, not a hardware restriction: CQ0 is a general-purpose queue and is fully capable of carrying data movement commands even in dual-CQ mode. However, routing transfers through CQ1 by convention allows the device to overlap data movement with compute, hiding transfer latency that would otherwise serialize with computation in a single-CQ setup.

> **Warning:** CQ1 is only useful if you explicitly stage work to it. Simply opening a device with dual-CQ support enabled does not automatically parallelize anything. You must structure your program to enqueue data movement to CQ1 and compute to CQ0, and you must insert explicit synchronization (described below) when work on one queue depends on work in the other.

---

## FIFO Ordering and Cross-Queue Dependencies

Within a single CQ, the FIFO guarantee is total: every command sees the effects of every preceding command on the same queue.

Between two queues (CQ0 and CQ1), there is no automatic ordering. If CQ1 is transferring input data for a kernel that CQ0 will run, you must explicitly synchronize the queues to ensure CQ0's kernel does not start until CQ1's transfer completes.

TTNN provides event-based synchronization for this purpose:

1. The producing queue records an event after its command completes.
2. The consuming queue waits on that event before proceeding.

```
CQ1 (data movement)
─────────────────────────────────────────────────────────────
  [transfer: load next_input to device DRAM]
        │
        ▼
  [record event E1]  ───────────────────────────────────────►
                                                             │
CQ0 (compute)                                               │
─────────────────────────────────────────────────────────────
  [wait on event E1]  ◄─────────────────────────────────────┘
        │
        ▼
  [matmul using next_input]
```

Without the event barrier, CQ0 might attempt to read `next_input` before CQ1 has finished writing it.

> **Note:** Event synchronization introduces a small overhead — typically 2–10 us — at the synchronization point. This cost is worth paying when the parallelism between queues provides more benefit than the sync overhead removes. For short transfers relative to compute time, the overhead may dominate and single-CQ mode may perform better.

---

## Single-CQ vs Dual-CQ: When Each Makes Sense

### Single-CQ mode

All work — compute and data movement — flows through CQ0. This is the simpler configuration and the correct choice in most situations:

- Your model's data is already on the device before the compute loop starts (a common pattern for weights-resident inference).
- The tensor transfers between decode steps are small enough (e.g., a single token embedding) that the serialization cost is negligible.
- You are using trace replay, which pre-encodes commands into a device-side buffer and does not benefit from CQ1 for its internal replay path.
- You want simpler code with no cross-queue synchronization to manage.

Single-CQ mode also has lower synchronization overhead when calling `ttnn.synchronize_device()`, because there is only one queue's worth of in-flight work to drain.

### Dual-CQ mode

CQ1 provides a meaningful benefit when all of the following are true:

- There is a significant volume of host-to-device or device-to-host data transfer that can proceed concurrently with compute.
- The transfer latency on CQ1 is long enough to justify the cross-queue synchronization overhead.
- Your decode or compute loop can be structured to keep CQ0 and CQ1 busy simultaneously (double-buffering the input pipeline).

The canonical example is a streaming inference server loading the next request's input embeddings through CQ1 while processing the current request's decode step through CQ0.

> **Example:** In a decode loop processing one token at a time, the per-step input is a single 1xD tensor (where D is the model's hidden dimension). A typical transfer of 4 KB–32 KB takes 5–20 us. If the decode step itself takes 1 ms (1,000 us), the transfer is fully hidden even in single-CQ mode (the transfer completes long before the step ends). In this case, dual-CQ provides no benefit.
>
> If instead each step loads a new key-value cache shard of several hundred megabytes from host DRAM — a pattern in some very large sparse models — dual-CQ becomes critical for hiding that latency.

---

## How Commands Are Consumed: Device Firmware Perspective

Device firmware polls both CQ0 and CQ1 in an infinite loop and dispatches commands to Tensix cores when new entries are detected; Tensix cores are not aware of the CQ abstraction and simply execute kernels placed in their local instruction memory by the dispatcher. Because polling is not interrupt-driven, there is a 1–5 us latency between when the host advances the write pointer and when the firmware observes the new command — this gap is visible in Tracy and TTNN profiler traces as the dispatch-to-execution interval.

---

**Next:** [Chapter 2 — Asynchronous Op Execution](../ch2_async_ops/index.md)
