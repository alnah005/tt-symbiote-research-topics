# Pipelining Host and Device

This file explains how async op mode enables host-device pipelining — the steady state in which the device is executing op N while the host is simultaneously dispatching op N+1 — and examines the conditions that prevent pipelining from occurring. It then focuses on the decode loop as the primary workload where pipelining matters most, and closes with a realistic accounting of the cost of unavoidable synchronization barriers. By the end you will be able to read a decode loop, identify which constructs enable pipelining and which destroy it, and estimate how much latency synchronization is adding.

---

## What Pipelining Means

Host-device pipelining is the temporal overlap between two activities that would otherwise be sequential:

- The **host activity**: encoding op N+1 (phases 1–4 of dispatch) on the background dispatch thread.
- The **device activity**: executing the kernel for op N.

In synchronous mode, these activities are never truly overlapping from a useful standpoint — the host encodes op N+1 only after it has finished encoding and submitting op N, at which point op N's kernel may already be running. But the overlap is incidental: if kernel N finishes before the host finishes encoding N+1, the device sits idle waiting. There is no mechanism ensuring the host stays ahead.

In async op mode, pipelining is deliberate. Because the Python thread returns immediately from every op call, it races ahead through the loop and fills the work queue with pending requests. The dispatch thread works through those requests at dispatch-thread speed (~20–50 us per op), while the device simultaneously works through the CQ at execution speed (device kernel time per op). As long as the dispatch thread can encode and submit op N+1 before the device finishes executing op N, there is no idle gap on the device.

```
Ideal pipelining — dispatch thread always ahead of device
────────────────────────────────────────────────────────────────────────────────
Time ──────────────────────────────────────────────────────────────────────────►

Dispatch thread
├──── encode A ────┤├──── encode B ────┤├──── encode C ────┤├──── encode D ────┤

Device
                   ├────────── kernel A ──────────┤├──── kernel B ──────┤├──...─┤
```

In this ideal state the dispatch thread is never the bottleneck: it finishes encoding B before the device finishes A. The CQ always has at least one command waiting when the device completes each kernel. Device utilization is 100%.

```
Dispatch thread falling behind — device goes idle
────────────────────────────────────────────────────────────────────────────────
Time ──────────────────────────────────────────────────────────────────────────►

Dispatch thread
├──── encode A ────┤                 ├──── encode B ────┤
                   (encoding delayed)

Device
                   ├── kernel A ──┤  [idle]  ├── kernel B ──┤
                                  ▲
                                  Device finishes A but B is not in CQ yet
```

This idle gap is the dispatch-dominated latency that trace (Chapter 3) eliminates by pre-encoding the entire command sequence at capture time.

---

## When Pipelining Works — and When It Breaks

Pipelining requires two positive conditions:

1. **Async mode is enabled.** Without `device.enable_async(True)`, the Python thread is doing dispatch phases 1–4 itself and returning only after each op's command is in the CQ. The dispatch thread does not exist. The Python thread and the CQ submission are on the same serial path, so the Python thread is always fully occupied during dispatch and cannot be ahead of the device.

2. **The op sequence between synchronization points is long enough to amortize dispatch latency.** If you have only two or three ops between sync points, the pipeline does not have enough depth to provide meaningful overlap. A decode step with 20–100 ops is deep enough to benefit significantly.

The following patterns break pipelining:

### Synchronous Readbacks Inside the Loop

This is the most common pipeline-breaker in production decode loops:

```python
# BAD: readback inside the loop destroys pipelining
for step in range(num_decode_steps):
    logits = model.forward(token_ids)          # dispatched async
    logits_host = ttnn.to_torch(logits)        # SYNC POINT: blocks here
    next_token = logits_host.argmax(-1)        # Python control flow
    token_ids = update_token_ids(token_ids, next_token)
```

At `ttnn.to_torch(logits)`, the Python thread blocks until the device finishes the entire model forward pass for the current step. No dispatch work for the next step has started yet — the Python thread has not even called any ops for step N+1 because it is blocked waiting for step N's outputs. The loop is effectively synchronous: step N must complete entirely before step N+1 can begin dispatching.

### Python Control Flow on Device Outputs

Any Python `if`, `while`, `for`, or similar construct whose condition involves the numeric value of a device tensor is an implicit readback. Even accessing `.shape` on a tensor whose shape depends on a device computation (e.g., after a dynamic op) can force a sync.

Static-shape ops like `ttnn.matmul` with fixed dimensions do not have this problem: the output shape is known statically from the input shapes, and no device read is required to determine it.

### Mixing Profiling Instrumentation That Forces Syncs

Some profiling and debugging patterns insert implicit syncs:

```python
# Calling ttnn.to_torch inside a loop for debugging forces a sync.
for step in range(num_decode_steps):
    output = model.forward(input_ids)
    print(ttnn.to_torch(output)[0, :5])   # sync every step
```

When profiling with Tracy or the TTNN profiler in verbose mode, be aware that some instrumentation options may also insert device-side event flushes that act as partial sync points. Measure overhead-free and overhead-present cases separately to isolate profiling cost.

### CQ Backpressure as a Soft Stall

If the Python thread enqueues op requests faster than the dispatch thread can process them, and the dispatch thread writes to the CQ faster than the device can consume commands, the CQ will eventually fill. When the CQ is full, the dispatch thread stalls waiting for CQ slots to free up. The Python thread is not directly blocked — it can continue enqueuing requests to the work queue — but the effective throughput of the pipeline is limited by the device's consumption rate of the CQ.

This is a soft stall, not a hard synchronization: the device is still executing, just slowly enough that the CQ fills. It is relatively uncommon in practice because device kernel execution time per op tends to be long enough to consume CQ slots quickly. But it can occur when the dispatch thread produces work faster than the device can consume it — i.e., a fast host dispatch thread relative to device execution speed.

---

## The Decode Loop as the Primary Beneficiary

The autoregressive decode loop is the canonical workload that benefits most from async op mode, for three compounding reasons.

**Reason 1: Fixed shapes.** Each decode step operates on tensors whose shapes do not change between steps. The token embedding tensor has shape `[1, 1, 1, D]` regardless of which token is being generated. Because shapes are fixed, there is no dynamic shape computation that would require a device readback for shape inference, and the dispatch thread can encode each step's ops without any host-device synchronization for structural reasons.

**Reason 2: Repeated execution of the same op sequence.** Every decode step executes the same ordered sequence of ops (attention, FFN, norm, etc.) with potentially different tensor values but identical structure. The dispatch thread performs the same four phases in the same order on every iteration. The work queue fills at a predictable rate, and the device and dispatch thread reach a steady pipeline state after the first few iterations.

**Reason 3: High op count per step.** A typical transformer decode step involves 20–100 individual `ttnn` op calls per layer, multiplied by the number of layers. A 32-layer model with 30 ops per layer dispatches 960 ops per step. With per-op dispatch overhead of ~20–50 us, the total dispatch time per step is roughly 19–48 ms (19,000–48,000 us). If the step's actual compute time is 5–10 ms (5,000–10,000 us), at the upper end of these ranges, dispatch overhead can exceed kernel execution time by 4× or more; at the lower end, dispatch overhead is still roughly 2× kernel execution time — but in either case, the device would be significantly starved in synchronous mode. Async mode hides this entirely.

The following code shows a correctly pipelined decode loop:

```python
import ttnn
import torch

device = ttnn.open_device(device_id=0, num_hw_cqs=1)
device.enable_async(True)

# Model weights are already loaded onto the device.
# token_ids is a device tensor that the model reads each step.
token_ids = ttnn.from_torch(
    torch.zeros(1, 1, dtype=torch.int32),
    dtype=ttnn.uint32,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
)

generated_tokens = []

for step in range(max_decode_steps):
    # All ops inside model.forward() are dispatched asynchronously.
    # This call returns in microseconds; no device work has started yet.
    logits = model.forward(token_ids)           # async, returns immediately

    # ttnn.argmax operates on a device tensor and returns a device tensor.
    # No readback; no sync point. Still pipelined.
    next_token_device = ttnn.argmax(logits, dim=-1)

    # Append next_token_device directly to a list — no readback here.
    # The list holds device tensors; their values are resolved only if
    # and when we later call ttnn.to_torch on them.
    generated_tokens.append(next_token_device)

    # Update token_ids on the device using the device-side next token.
    # This is a device-to-device copy; no host readback.
    token_ids = ttnn.reshape(next_token_device, (1, 1))

# After the loop: do a single sync and read all results at once.
ttnn.synchronize_device(device)
results = [ttnn.to_torch(t).item() for t in generated_tokens]

ttnn.close_device(device)
```

The loop body contains no readbacks and no explicit sync calls. Every op is dispatched asynchronously. The single `ttnn.synchronize_device()` after the loop is unavoidable — we need the token values — but it is paid once, not once per step.

> **Note:** In production decode loops, you typically do not collect all tokens and read them at the end; you stream each token to the caller as it is generated. The pattern for streaming without destroying pipelining is to use event barriers: record an event after the `ttnn.argmax` that produces the current step's token, wait on that event in a reader thread or callback, and read the token from a dedicated host-mapped output buffer that the device writes to directly. This avoids a full `synchronize_device()` per step while still providing per-step output. The event overhead (2–10 us per step) is far less than a full sync (kernel execution time per step).

---

## Cost of Synchronization Barriers

Not all synchronization is avoidable. Understanding the cost structure helps you decide when to pay it and when to restructure code to avoid it.

| Synchronization type | Scope | Typical additional overhead | Use when | Notes |
|---|---|---|---|---|
| `ttnn.synchronize_device()` | All in-flight ops on all CQs | 0 (pure wait) | End of loop; model loading; shutdown | Purely a "drain and block" operation; calling inside a loop per-op is equivalent to synchronous mode |
| `ttnn.to_torch()` / readback | All ops that wrote the tensor | 50–500 us DMA (size-dependent) | Token streaming; validation | DMA time depends on tensor size: a 512x512 bfloat16 tensor (512 KB) takes ~50–150 us over PCIe; a scalar result (4 bytes) takes ~5–20 us; for large activations, transfer time can dominate |
| `ttnn.wait_for_event()` | Ops up to the recorded event | 2–10 us | Cross-queue sync; fine-grained output | Does not drain the entire CQ — commands after the recorded event remain in flight on the device |

> **Warning:** The "typical additional overhead" column above excludes the device execution wait time, which dominates in all cases. The numbers in that column represent only the synchronization mechanism's own cost on top of the wait. When evaluating whether a sync is expensive, the primary question is how much in-flight device work you are waiting for, not the overhead of the sync call itself.

---

## Putting It Together: Pipeline Health Checklist

Use the following questions to evaluate whether a decode loop is effectively pipelining:

- Is `device.enable_async(True)` called before the loop?
- Does the loop body contain any `ttnn.to_torch()` or `ttnn.from_device()` call? If yes, is that call strictly necessary, or can it be moved outside the loop?
- Does any Python `if`, `while`, or comparison inside the loop use the numeric value of a device tensor?
- Is `ttnn.synchronize_device()` called inside the loop body?
- Are all output tensors for the step consumed as device tensors by subsequent ops, or are any read back to Python to drive control flow?

A loop that answers "no" to all of the above is correctly structured for pipelining. A loop that answers "yes" to any of them has a synchronization point on the critical path, and async dispatch will not fully hide dispatch overhead for that loop.

---

**Next:** [Chapter 3 — Trace Capture and Replay](../ch3_trace_capture/index.md)
