# `synchronize_device` Semantics

This file explains what `ttnn.synchronize_device(mesh_device)` does at the TTNN/tt-metal level, why it constitutes a host-blocking call, and the two plausible reasons it could have been placed inside `_maybe_all_gather`. It then explains why the CQ0 FIFO ordering guarantee makes the call unnecessary as a sequencing mechanism in single-command-queue deployment. The distinction between `ttnn.synchronize_device` (singular) and the less common `ttnn.synchronize_devices` (plural) is also noted.

---

## What `ttnn.synchronize_device` Does at the TTNN/tt-metal Level

When the Python call `ttnn.synchronize_device(mesh_device)` is made, the following sequence occurs inside the TTNN runtime:

1. The host enqueues a **finish command** to CQ0 on the device. This command does not perform any compute; it is a sentinel that the device recognizes as a synchronization point.
2. The host then enters a **polling loop** — or a blocking system call, depending on the driver implementation — waiting for the device to signal that it has processed the finish command. The device processes the command queue in FIFO order, so the finish command is processed only after every command that preceded it in the queue has completed.
3. Once the device signals completion, `ttnn.synchronize_device` returns and Python execution resumes.

The net effect is that after `ttnn.synchronize_device` returns, the host has a guarantee that **every kernel that was submitted to CQ0 before the call has completed execution** and its output has been written to the agreed-upon device memory addresses.

On a mesh device (such as T3K), the call synchronizes all constituent devices — each device's command queue is drained before the call returns.

> **Note:** The singular `ttnn.synchronize_device(mesh_device)` and the less common `ttnn.synchronize_devices(device_list)` (plural form, which accepts an explicit list of devices) are both host-blocking and both drain the device command queue. The singular form with a mesh device handle is the form used in `_maybe_all_gather`. Throughout this guide, "synchronize_device" refers to the singular form unless explicitly stated otherwise.

---

## Why This Is a Host-Blocking Call

The term "host-blocking" means that Python execution on the host CPU stops and cannot proceed until the device-side operation completes. During the time that `ttnn.synchronize_device` is in progress:

- The Python interpreter is suspended inside the TTNN native call. No Python bytecode is executed.
- No new TTNN ops can be submitted to CQ0, because the call is synchronous from the Python perspective: no other code in the calling thread runs.
- Any pending host-side Python work — tensor preprocessing for the next step, sampling logic, cache management — is also blocked because the Python interpreter itself is halted.

This is in contrast to a **device-blocking** call, which would cause the device to stall but would allow the host to continue issuing ops. `ttnn.synchronize_device` is strictly host-blocking.

At decode batch=1 on T3K, the dominant cost of `ttnn.synchronize_device` is the PCIe round-trip time: the host must send the finish command across PCIe to the device, the device must process it (after draining the queue), and the completion signal must travel back to the host across PCIe. On Wormhole hardware this round-trip is approximately 10–30 µs under ideal conditions. If the device queue is not yet empty when the host sends the finish command, the total wait time includes the remaining kernel execution time plus the round-trip.

---

## Two Plausible Reasons `synchronize_device` Is Inside `_maybe_all_gather`

### Reason 1 — Sequencing Concern: Ensuring the Input Tensor Is Ready

The most defensible reason to place `ttnn.synchronize_device` before an all_gather op is to ensure that the preceding async op has finished writing its output before the all_gather reads that tensor as its input. If the preceding op is an async operation (such as a prior `all_gather_async` or a compute kernel that was dispatched non-blocking), the host might reach `_maybe_all_gather` before the async op has completed on the device, and a naive (incorrectly implemented) all_gather might attempt to read a tensor that is still being written.

This would be a legitimate concern in a **multi-CQ** system or with a **poorly designed async op** that does not correctly fence its output. It would not be a concern in a correctly implemented single-CQ system where every op submitted to CQ0 is guaranteed to execute in FIFO order before the next op in the queue starts.

### Reason 2 — Debugging or Stability Artifact

A common pattern during initial development and debugging of tensor-parallel models is to insert `ttnn.synchronize_device` calls at suspected race-condition points to force serial execution and confirm that the model produces correct outputs. Once correctness is confirmed with the synchronize calls in place, the calls are sometimes left in the production code path — either out of caution, because the root cause of the instability was never fully diagnosed, or simply because removing them was deprioritized after correctness was achieved.

In this scenario, `ttnn.synchronize_device` in `_maybe_all_gather` is a debugging artifact that was never removed. It adds host-blocking latency on every call without contributing to the correctness of the op sequence.

### Which Scenario Applies?

Determining which scenario is correct requires **code archaeology**: inspecting the git history for `_maybe_all_gather` to see when the call was added and what commit message or PR description accompanied it. If the call was added alongside a bug fix for a non-deterministic failure, it is likely Reason 1 (even if the correct fix would have been a lighter-weight solution). If it was added in an early commit alongside other diagnostic prints or synchronize calls, it is likely Reason 2.

In the absence of clear git history, the question can also be answered empirically: run `_maybe_all_gather` without `ttnn.synchronize_device` under controlled conditions (see Chapter 7 — Validation) and observe whether correctness is maintained. If correctness is maintained across many runs, the call was unnecessary; if non-deterministic failures appear, there is a sequencing bug that the call was masking.

> **Key finding:** In a single-command-queue (CQ0) TTNN dispatch model, the FIFO ordering guarantee eliminates any sequencing need for `ttnn.synchronize_device` between adjacent ops in the same queue. Chapter 3 investigates which all_gather variant `_maybe_all_gather` uses and delivers the definitive verdict on whether the call is removable.

---

## The CQ0 Ordering Guarantee and Why `synchronize_device` Adds No Sequencing Value

TTNN on T3K uses a single command queue (CQ0) for both compute and CCL (collective communication) operations in trace-compatible deployment. The CQ0 ordering guarantee is:

**Op $N+1$ submitted to CQ0 will not begin executing before op $N$ has completed and its output is available at the agreed-upon device memory address.**

This guarantee holds by the FIFO nature of the hardware command queue: the device processes commands in strict submission order. If op $N$ writes tensor $T$ and op $N+1$ reads tensor $T$, and both are submitted to the same CQ0, then op $N+1$ cannot read $T$ before op $N$ has written it — the queue ordering prevents this.

A concrete example relevant to `_maybe_all_gather`:

```python
# Op N: QKV linear projection — writes xqkv_fused to L1
xqkv_fused = ttnn.linear(x, self.wqkv, ...)

# ttnn.synchronize_device(self.mesh_device) is called here in the current code
# why: this adds no sequencing guarantee — CQ0 already guarantees
#      that the linear projection is complete before the all_gather reads xqkv_fused

# Op N+1: all_gather — reads xqkv_fused written by op N
xqkv_gathered = ttnn.all_gather(xqkv_fused, ...)   # or all_gather_async
```

Because both ops are submitted to the same CQ0, the FIFO guarantee ensures the all_gather cannot read `xqkv_fused` before the linear projection has written it. `ttnn.synchronize_device` adds no ordering guarantee beyond what the queue already provides. Its only effect is to add a host-blocking PCIe round-trip that increases per-step latency without benefiting correctness.

The CQ0 ordering guarantee also extends to async CCL ops (`ttnn.experimental.all_gather_async`, `ttnn.experimental.reduce_scatter_minimal_async`): the device-side semaphore mechanism within the CCL kernel ensures that the output buffer is fully valid before the kernel signals completion. Since the kernel completes before the next CQ0 command is dispatched, any downstream op reading the CCL output will see a valid result.

> **Warning:** The CQ0 ordering guarantee applies only in **single-CQ mode**. If a model uses multi-CQ dispatch (CQ0 for compute, CQ1 for CCL), ops on different queues can interleave arbitrarily, and `ttnn.synchronize_device` or explicit cross-queue barriers may be necessary. Multi-CQ mode is incompatible with Metal Trace in any case. The tt-symbiote attention modules, when run in `TRACED` mode, use single-CQ dispatch exclusively.

---

**Next:** [`why_this_blocks_trace_capture.md`](./why_this_blocks_trace_capture.md)
