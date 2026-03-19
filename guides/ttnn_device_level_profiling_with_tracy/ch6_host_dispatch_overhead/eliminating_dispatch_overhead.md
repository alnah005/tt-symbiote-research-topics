# Eliminating Dispatch Overhead with Mesh Trace Capture

Mesh trace eliminates host dispatch overhead by recording device commands once and replaying them with a single MMIO write.

## How Trace Capture Works

Trace capture records the complete device command sequence for a sequence of op dispatches — kernel binaries, runtime arguments, NoC descriptors, core grid configurations, everything — into a pre-built replay buffer on the device. On subsequent executions, the host issues a single "execute trace" MMIO write, and the device firmware executes the entire pre-recorded command sequence directly without any further host involvement.

The three API calls that implement this are:

```python
# Phase 1: Record — run ops normally once while recording commands
ttnn.begin_trace_capture(device, cq_id=0)

# All TTNN op calls inside this block execute normally on the device,
# producing real outputs, while the device command sequence is simultaneously
# recorded into a replay buffer for future use.
output = ttnn.matmul(a, b)
output = ttnn.add(output, bias)
output = ttnn.softmax(output)
# ... additional ops in the captured sequence

trace_id = ttnn.end_trace_capture(device, cq_id=0)

# Phase 2: Replay — execute the full recorded sequence with one host call
ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
```

Between `begin_trace_capture` and `end_trace_capture`, tt-metal captures the complete device command buffer — the fully-serialized sequence of kernel launches, runtime arguments, and device memory writes — without requiring re-dispatch through the Python→C++→MMIO path on each replay.

### What the host does during replay

On a `ttnn.execute_trace` call, the host issues a single MMIO write that tells the device to replay the stored trace. This write is on the order of ~1 µs. The device then executes all captured ops autonomously, reading from and writing to the pre-allocated tensor buffers, without returning to the host between ops. The host can do other work (e.g., preparing the next batch of inputs) while the device is running.

The effective `host_dispatch_time` per op during traced replay is:

```
effective_dispatch_per_op ≈ (single MMIO write time) / (number of ops in trace)
                          ≈ ~1 µs / N_ops
```

For a trace capturing 50 ops, the per-op dispatch contribution from the host drops from ~10 µs to ~0.02 µs.

---

## Expected Speedup

For decode-regime workloads — transformer inference where each decode step repeatedly runs the same sequence of attention, matmul, add, layernorm, and softmax ops on small tensors — trace capture typically delivers:

- **3–10× reduction in total decode step latency** when the workload is dispatch-bound.
- **Near-zero per-op host overhead** for ops inside the trace; the bottleneck shifts to device kernel execution.
- **Linear scaling with decode step repetitions**: the speedup is realized on every replay call, so longer generation sequences benefit proportionally.

The exact speedup depends on the ratio of `host_dispatch_time` to `device_kernel_time` before trace capture. A workload with a 10× dispatch/kernel ratio will see a larger speedup from trace capture than one with a 2× ratio.

> **Note:** Trace capture does not change `DEVICE KERNEL DURATION [ns]` for any individual op. The device executes the same kernel code, on the same cores, with the same tile counts. `DEVICE KERNEL DURATION` values in the CSV during traced replay should be statistically identical to non-traced values (within normal run-to-run variance). If they differ significantly, something in the trace setup is wrong.

---

## Constraints on Trace Capture

Trace capture has important limitations. Violating these constraints results in either silent incorrect output or a runtime assertion.

### Tensor shapes must not change

The trace records the complete command buffer for a fixed set of shapes and data formats. If input tensor shapes change between the `end_trace_capture` call and a `execute_trace` call, the pre-recorded runtime arguments (tile counts, L1 buffer addresses, NoC descriptor addresses) are no longer valid for the new shapes. The device will execute with stale arguments, producing incorrect output without error.

> **Warning:** Shape-dependent branching (e.g., Python `if seq_len > 128: ...` that changes which ops are called) cannot be captured in a single static trace. If your model has shape-conditional paths, only the path active during `begin_trace_capture` will be recorded. The other path will not be executed during replay regardless of the runtime condition.

### Ops requiring host state modification cannot be inside a trace

Ops that modify Python-visible state — reading a tensor back to the host, calling a Python callback, writing to a host-side data structure — cannot run inside a trace capture block. These ops require host involvement by definition; the trace execution model does not allow re-entry to the host mid-replay.

Specifically:

- `ttnn.to_torch()` and `ttnn.from_torch()` inside the trace block will fail or produce wrong results — tensor data cannot be read to or written from the host during replay.
- Any use of Python `print()` or logging that depends on intermediate tensor values is not possible inside a trace.
- Custom Python callbacks or hooks that inspect intermediate tensors cannot be used.

### Output tensors must be pre-allocated on the device

Inside a trace capture block, tensors written by ops must reside in pre-allocated device buffers. Use `ttnn.allocate_tensor_on_device` to create output tensors before the trace begins:

```python
# Allocate output buffers before trace capture
output_buffer = ttnn.allocate_tensor_on_device(
    shape=ttnn.Shape([1, 1, 32, 4096]),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

ttnn.begin_trace_capture(device, cq_id=0)
output_buffer = ttnn.matmul(a, b, output_tensor=output_buffer)
# ... more ops using pre-allocated buffers
trace_id = ttnn.end_trace_capture(device, cq_id=0)
```

If output tensors are allocated implicitly inside the trace block (the default TTNN behavior), the allocations are part of the captured command sequence and will attempt to re-allocate the same memory on every replay — which is incorrect because the memory is already allocated from the first capture pass.

---

## Verifying Trace Execution with the Device Profiler

To confirm that trace capture is working correctly and has not altered kernel behavior, compare `DEVICE KERNEL DURATION [ns]` values between a non-traced run and a traced replay run.

**Procedure:**

1. Run the workload without trace capture with `TT_METAL_DEVICE_PROFILER=1`. Save `ops_perf_results.csv` as `results_baseline.csv`.
2. Run the workload with trace capture (the replay call) with `TT_METAL_DEVICE_PROFILER=1`. Save `ops_perf_results.csv` as `results_traced.csv`.
3. For each op row, compare `DEVICE KERNEL DURATION [ns]` between the two files.

Expected result: `DEVICE KERNEL DURATION [ns]` values are equal (within ±5% for run-to-run thermal and clock variance). If a duration has changed significantly:

- A much shorter duration may indicate the op was not actually inside the trace and was skipped during replay.
- A much longer duration may indicate memory contention from a pre-allocation issue, or that the wrong kernel binary was recorded in the trace.

To verify that Tracy zones collapse during replay (confirming dispatch overhead has been eliminated):

1. Run a Tracy capture during traced replay.
2. Open the `.tracy` file and inspect the Tracy timeline around `ttnn.execute_trace`.
3. You should see a single short `execute_trace` zone (< 5 µs) replacing the sequence of individual op dispatch zones that appeared in the non-traced run. The individual per-op zones (`ttnn::operations::matmul::Matmul`, etc.) should not appear — they are not re-dispatched during replay.

> **Tip:** When profiling a traced workload with both Tracy and the device profiler simultaneously, the Tracy timeline will show a gap between `execute_trace` returning and the device CSV's last recorded `DEVICE KERNEL DURATION` end. This gap is `sync_overhead` — the time the host spends waiting for the device to signal trace completion. Minimizing this gap (by using non-blocking `execute_trace` calls and overlapping host work) is the next optimization step after eliminating dispatch overhead.

---

## When NOT to Use Trace Capture

Trace capture is a production-optimization tool, not a development-time tool. Do not use it when:

**During model development** — while iterating on model architecture, layer configurations, or tensor shapes, trace capture adds complexity (pre-allocated tensors, shape-change constraints) without benefit. The performance cost of re-dispatching is acceptable during development; correctness and iteration speed matter more.

**For ops requiring shape-dependent branching** — if the model uses dynamic control flow based on runtime tensor values (e.g., early exit based on model output), trace capture cannot represent the branching. Use conditional compilation or runtime op selection outside the trace instead.

**When correctness validation requires intermediate tensor reads** — during debugging or accuracy validation, you often need to read intermediate tensors to the host to inspect values. This is impossible inside a trace. Disable trace capture for debugging sessions that require `ttnn.to_torch()` on intermediate tensors.

**For one-shot or low-repetition workloads** — trace capture adds a setup cost: the capture pass runs the op sequence once to record the commands. For a workload that runs only a few times (e.g., prefill of a long prompt), the capture overhead may not be recovered through replay savings. Trace capture pays off when the same op sequence is replayed tens or hundreds of times (e.g., autoregressive token generation).

---

**End of guide.** Return to [Guide Index](../index.md)
