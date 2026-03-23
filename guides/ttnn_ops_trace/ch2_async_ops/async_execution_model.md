# Async Execution Model

This file defines the asynchronous op execution model precisely: what it means for a `ttnn` op call to return before device execution completes, how the runtime preserves op-ordering correctness without requiring explicit per-op barriers, what events constitute a synchronization point, and how to reason about two sequential matmuls dispatched in this mode. By the end you will be able to reason about any sequence of async ops and identify exactly when the host and device can diverge and when they must converge.

---

## What "Async" Means in TTNN

In synchronous mode — the default prior to async op mode being enabled — a call like `ttnn.matmul(a, b)` returns to Python only after all four dispatch phases (argument validation, kernel selection, command encoding, CQ submission) have completed and the command is sitting in the CQ. At that point the device may or may not have begun executing the kernel; the host does not know and does not wait. From Chapter 1: the host never waits for kernel execution in either mode. The difference between sync and async is entirely on the host side, before the command reaches the CQ.

In async op mode, `ttnn.matmul(a, b)` returns to Python before any dispatch phase begins. The call places an op request into an internal work queue and returns immediately. A background dispatch thread owned by the TTNN runtime dequeues that request and performs phases 1–4 on a separate CPU thread.

The practical consequence: in async mode, the time between the Python call returning and the command actually reaching the CQ is non-zero and variable. If the dispatch thread is busy encoding a previous op, the new request waits in the work queue. The Python caller cannot assume the command is in the CQ just because the call returned.

---

## Enabling Async Op Mode

Async op mode is enabled per device using `device.enable_async(True)`. The recommended way to enable it for a single device is:

```python
import ttnn

# Open a device.
device = ttnn.open_device(device_id=0)
ttnn.SetDefaultDevice(device)

# Enable async dispatch on the opened device.
device.enable_async(True)
```

After `enable_async(True)`, all subsequent `ttnn` op calls that target this device will dispatch asynchronously. To revert to synchronous dispatch:

```python
device.enable_async(False)
```

> **Note:** `enable_async` is a per-device setting. If you have opened multiple devices, you must call `enable_async` on each one independently. Devices not explicitly set to async mode continue to use synchronous dispatch regardless of what other devices are doing.

---

## The Background Dispatch Thread

When async mode is enabled, the TTNN runtime starts exactly one background dispatch thread per device. This thread runs an infinite loop: dequeue an op request from the work queue, execute phases 1–4 for that request, repeat. The thread is started when async mode is enabled and stopped when async mode is disabled or the device is closed.

The work queue between the Python thread and the dispatch thread is an unbounded FIFO — the Python thread can enqueue op requests faster than the dispatch thread can process them, and the queue simply grows. This means Python-side loop iteration is never blocked by dispatch thread availability. The queue depth is a runtime diagnostic that can indicate whether the dispatch thread is keeping up.

```
Python thread         Work queue (FIFO)         Dispatch thread
──────────────        ─────────────────         ────────────────
  enqueue(A) ──►    │  A  │  B  │  C  │    ──► dequeue → dispatch A phases 1-4
  enqueue(B) ──►    │  B  │  C  │     │                           │
  enqueue(C) ──►    │  C  │     │     │                           ▼
  (Python returns              (grows          [encoded command written to CQ]
   immediately                 if dispatch              │
   after each)                 thread lags)             ▼
                                                  [device reads from CQ]
```

The dispatch thread processes requests in FIFO order, which preserves the op-ordering semantics that the rest of the system depends on. If you enqueue `matmul(a, b)` and then `softmax(c)`, the dispatch thread will always encode and submit the matmul command before the softmax command. Because the device consumes CQ0 in FIFO order (as established in Chapter 1), the device will always execute the matmul before the softmax, even though the Python caller returned from both calls before either command reached the CQ.

> **Note:** The FIFO ordering guarantee established in Chapter 1 for the CQ extends naturally into async mode: the dispatch thread feeds the CQ in the same order that the Python thread enqueued requests, so the device execution order matches the Python call order. Correctness for sequential ops that share data (e.g., the output of op A is the input of op B) is preserved without any extra barriers.

---

## Tracking In-Flight Commands

The runtime tracks the number of in-flight commands — commands that have been submitted to the CQ but whose kernels have not yet signaled completion on the device. The primary bookkeeping mechanism is the CQ's write and read pointer pair described in Chapter 1: the host knows how many commands it has submitted (write pointer), and the device signals completion by advancing its read pointer or writing completion events to a host-mapped register.

From the Python caller's perspective, in-flight tracking is mostly transparent. You do not query a command count or check completion per-op. The runtime uses this accounting internally in two situations:

1. **CQ backpressure** — When the CQ is full (all slots occupied by commands the device has not yet consumed), the dispatch thread must wait before writing the next command. This stall is invisible to the Python thread, which has already returned from the op call. The work queue absorbs the difference: the Python thread continues enqueuing requests; the dispatch thread stalls on CQ full rather than on encoding.

2. **Synchronization points** — When the Python code explicitly requests that the host wait for a specific amount of device work to complete, the runtime uses the in-flight count plus device completion signals to determine when the wait condition is satisfied. Synchronization points are covered in the next section.

---

## Synchronization Points

A synchronization point is any operation that causes the calling Python thread to block until the device has completed a specified set of in-flight work. In async op mode, synchronization points are the only moments when the Python thread re-couples with the device timeline.

There are three categories of synchronization point in TTNN:

### 1. `ttnn.synchronize_device()`

The explicit barrier. Calling `ttnn.synchronize_device(device)` blocks the Python thread until all commands previously submitted to all CQs on that device have completed execution.

```python
import ttnn
import torch

device = ttnn.open_device(device_id=0, num_hw_cqs=1)
device.enable_async(True)

a = ttnn.from_torch(torch.randn(512, 512), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=device)
b = ttnn.from_torch(torch.randn(512, 512), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=device)

# Both calls return immediately. Commands are queued for dispatch.
c = ttnn.matmul(a, b)
d = ttnn.softmax(c, dim=-1)

# The Python thread blocks here until both matmul and softmax have
# finished executing on the device.
ttnn.synchronize_device(device)

# At this point it is safe to inspect device tensors, read profiling
# output, or close the device.
```

`ttnn.synchronize_device()` is a full drain: it waits for the CQ to be empty and for all in-flight kernels to signal completion. After it returns, there is no in-flight work on the device.

The cost of `ttnn.synchronize_device()` is the time the Python thread spends blocked waiting for the device to finish. If called after a long sequence of ops, most of the wait time is real device execution time that you would have paid regardless. If called very frequently — for example, once per op — it destroys the benefit of async dispatch by forcing the host to wait after every single dispatch.

> **Warning:** Calling `ttnn.synchronize_device()` inside a tight decode loop converts async op mode back into something functionally identical to synchronous mode. The correct pattern is to call it once at the end of the loop body, after all ops for the current step have been dispatched, if you need to inspect outputs or control flow depends on the step's outputs. Chapters 3 and 4 show how trace can eliminate even that end-of-step sync for many workloads.

### 2. Tensor Readback to Host Memory

Any call that moves tensor data from device memory to host-accessible memory is an implicit synchronization point. The read cannot return valid data until the device has finished writing that data.

The primary call that triggers this is `ttnn.to_torch()` (or equivalently `ttnn.from_device()` when the destination is host memory):

```python
# This call blocks until the device has finished all ops that
# produced `output`, then copies the result to a torch tensor.
result_host = ttnn.to_torch(output)
```

Under the hood, a readback call inserts a completion barrier for the ops that wrote `output` and waits for that barrier to be signaled before initiating the DMA transfer. The DMA transfer itself is also a blocking operation from the Python caller's perspective.

Readbacks inside a decode loop are a common source of accidental synchronization. A pattern like:

```python
for step in range(num_steps):
    logits = model(token_ids)
    logits_host = ttnn.to_torch(logits)   # implicit sync every step
    next_token = logits_host.argmax(-1)
    token_ids = update_tokens(token_ids, next_token)
```

forces the host to wait for the device to finish the full model forward pass on every iteration, because `ttnn.to_torch(logits)` cannot return until `logits` is ready. The decode loop in this form cannot pipeline at all — the device must finish completely before the host can compute `next_token` and dispatch the next step. `pipelining_host_and_device.md` covers this in detail.

### 3. Event Barriers

TTNN provides a lower-granularity synchronization primitive via device events. An event is a device-side signal that can be recorded after a specific command completes and waited on by either the host or another queue.

```python
# Obtain the command queue object for CQ 0.
cq = device.command_queue(cq_id=0)

# Record an event after the matmul completes on that CQ.
event = ttnn.record_event(cq)

# Wait on that event from the host before reading a result.
ttnn.wait_for_event(cq, event)
```

> **Note:** Verify the exact event API signature against the TTNN documentation for your version.

Event barriers are more precise than `ttnn.synchronize_device()` — they allow you to synchronize on a specific subset of in-flight work rather than draining all queues. The cost is the same class: the host blocks until the device signals the event.

Event barriers are also the mechanism for cross-queue synchronization in dual-CQ mode, as described in Chapter 1's `command_queues.md`. When CQ1 must finish a transfer before CQ0 begins a kernel, an event recorded on CQ1 and waited on by CQ0 provides that ordering guarantee.

The overhead of recording and waiting on a single event is typically 2–10 us, which is low enough that event barriers can be used at fine granularity without significant impact on total step latency.

---

## Concrete Example: Two Sequential Matmuls Dispatched Asynchronously

The following example walks through exactly what happens in time when two matmuls are dispatched in async mode, and identifies the moments when each op is enqueued, dispatched, and executed.

```python
import ttnn
import torch

device = ttnn.open_device(device_id=0, num_hw_cqs=1)
device.enable_async(True)

# Tensors a, b, c are already on the device.
a = ttnn.from_torch(torch.randn(1024, 1024), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=device)
b = ttnn.from_torch(torch.randn(1024, 1024), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=device)
c = ttnn.from_torch(torch.randn(1024, 1024), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=device)

# --- Step 1: Python enqueues matmul_1 ---
# Returns in ~1-2 us. Op request placed in work queue.
# The dispatch thread has not yet started on this request.
result_1 = ttnn.matmul(a, b)          # T=0 us: Python returns

# --- Step 2: Python enqueues matmul_2 ---
# result_1 is a future tensor — its device buffer address is allocated
# but no data exists yet. The runtime records that matmul_2's input
# depends on result_1's output buffer. This is a buffer dependency,
# not an execution barrier: both commands will be in the CQ before
# the device executes either, and the CQ's FIFO order guarantees
# matmul_1 executes first.
result_2 = ttnn.matmul(result_1, c)   # T=~2 us: Python returns

# At T=~2 us, the Python thread has returned from both calls.
# No dispatch has occurred yet; both requests are in the work queue.

# --- Synchronization ---
# The Python thread can continue doing other work from T=~2 us to
# whenever it needs result_2. Here we explicitly wait.
ttnn.synchronize_device(device)       # blocks until T=~1050 us

result_host = ttnn.to_torch(result_2)

ttnn.close_device(device)
```

The timeline above shows that by T=~87 us, both commands are in the CQ — well before the device finishes matmul_1 at ~548 us. The device has continuous work to execute with no idle gap between the two matmuls. The Python thread was free from T=~2 us onward and only rejoins the device timeline at `ttnn.synchronize_device()`.

> **Note:** The buffer addresses for `result_1` and `result_2` are allocated eagerly at the time `ttnn.matmul` is called on the Python thread, even though the tensors contain no valid data yet. This allocation is what allows the dispatch thread to encode matmul_2's command with the correct output buffer addresses for result_1 before matmul_1 has executed. Buffer allocation is synchronous; data production is asynchronous.

---

## What the Runtime Does NOT Do

To avoid common misconceptions, it is worth being explicit about what the async runtime does not do:

- It does not reorder ops. The dispatch thread processes requests in the exact order the Python thread enqueued them.
- It does not analyze data dependencies or insert barriers automatically between ops in different sequences that happen to share buffers. If you write an op sequence with a true data hazard that your code does not protect with an explicit sync, the async runtime will not catch it.
- It does not parallelize dispatch across multiple CPU threads. One dispatch thread processes one work queue sequentially.
- It does not guarantee that two op calls issued from different Python threads on the same device will be serialized in a well-defined order without external synchronization between those threads.

---

**Next:** [`pipelining_host_and_device.md`](./pipelining_host_and_device.md)
