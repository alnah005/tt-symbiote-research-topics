# Trace API

This file introduces the three functions that constitute the TTNN trace API — `ttnn.begin_trace_capture`, `ttnn.end_trace_capture`, and `ttnn.execute_trace` — and gives you a precise understanding of what each one does, what the capture records, what it explicitly does not record, and the minimal code pattern you need to capture and replay a fixed-shape decode step. By the end you will be able to write a correct capture/replay loop and explain why each line is necessary.

---

## The Three API Functions

### `ttnn.begin_trace_capture(device, cq_id)`

Marks the start of a trace capture session on `device` using command queue `cq_id`. After this call returns, all `ttnn` op calls that target `device` on that CQ are intercepted: they execute normally (live dispatch, phases 1–4 as described in `host_dispatch_path.md`) but simultaneously write their encoded commands into a trace recording buffer on the device.

Parameters:

- `device` — the `ttnn.Device` object returned by `ttnn.open_device`.
- `cq_id` — the integer identifier of the command queue to capture on. Pass `0` for CQ0 (the default compute queue). Pass `1` if you are using dual-CQ mode and want to trace on CQ1.

Return value: none. The trace session is associated with `device` and `cq_id` and remains open until `ttnn.end_trace_capture` is called on the same pair.

> **Warning:** Only one capture session may be open at a time per `(device, cq_id)` pair. Calling `ttnn.begin_trace_capture` while a capture is already in progress on the same device and CQ raises a runtime error. Always pair every `begin_trace_capture` with exactly one `end_trace_capture`.

---

### `ttnn.end_trace_capture(device, cq_id)`

Closes the capture session opened by `ttnn.begin_trace_capture` and finalizes the recorded command buffer on the device. After this call, the buffer is locked into device DRAM at a fixed address and assigned a `trace_id`.

Parameters:

- `device` — must match the device passed to `begin_trace_capture`.
- `cq_id` — must match the CQ ID passed to `begin_trace_capture`.

Return value: an integer `trace_id`. Store this value — it is the handle you pass to `ttnn.execute_trace` to replay the recorded sequence.

> **Note:** The ops that ran during capture executed live on the device. Their outputs are available in the output tensors at the moment `end_trace_capture` returns, exactly as they would be without tracing. The capture phase is not a dry run — real device work happened and real outputs were produced. This also means you can validate capture outputs immediately after `end_trace_capture` before committing to the replay loop. The capture run also serves as a warm-up: any kernel selection cold-path cost (described in Chapter 1) is paid during capture, so replay never pays cold-path costs.

---

### `ttnn.execute_trace(device, trace_id, cq_id, blocking)`

Submits a replay command to the device. The device reads the pre-encoded command buffer identified by `trace_id` from its DRAM and re-executes the entire recorded sequence without any host-side re-encoding.

Parameters:

- `device` — the device that owns the trace.
- `trace_id` — the integer returned by `ttnn.end_trace_capture`.
- `cq_id` — the CQ on which to submit the replay command. Must match the CQ used during capture.
- `blocking` — if `True`, the call blocks until the device completes the full replay sequence before returning. If `False`, the call returns immediately after submitting the replay command; you are responsible for synchronizing before consuming replay outputs.

Return value: none. The replay operates on the same device tensors that were bound during capture; outputs appear in-place at the same device addresses used during the capture run.

> **Note:** When using async op mode (`device.enable_async(True)`), pass `blocking=False` to `ttnn.execute_trace`. With blocking=False the execute command is dispatched asynchronously to the CQ — consistent with every other async op. Passing `blocking=True` in async mode forces the calling thread to wait, which defeats the pipelining benefit of async dispatch for the step that follows this one.

---

## What the Trace Records

The trace recording buffer contains a verbatim copy of the binary commands that were encoded and submitted to the CQ during the capture run. Each recorded command includes:

**Command sequence.** The complete ordered list of CQ commands — one per op in the capture sequence — in the same order they were submitted. The order is the FIFO order established by the CQ (described in `command_queues.md` in Chapter 1), so replay preserves the exact execution order of the captured ops.

**Kernel arguments.** For each recorded kernel invocation: the loop bounds, scale factors, and other runtime scalar arguments the kernel reads from L1 at dispatch time. These are the values that were in effect at capture time. If an op's kernel argument changes between steps (for example, a step count or a scaling factor that varies per iteration), the trace will replay the captured values, not the updated values. This is a constraint, not a bug — it is the developer's responsibility to ensure that all kernel arguments that the trace will replay are invariant across iterations.

**Buffer bindings.** The device DRAM and L1 addresses of every input and output tensor buffer that each kernel accesses. These addresses are fixed at capture time. The device uses these same addresses on every replay, which means the physical memory layout of input and output tensors must be identical between capture and replay. This constraint — address fixity — is the central subject of `trace_internals.md`.

---

## What the Trace Does NOT Record

Understanding what is excluded from the recording is as important as understanding what is included.

**Python-side logic.** The Python interpreter, your model's `forward` method, any `if` branches, loop counters, or print statements — none of this is recorded. The trace records only what reaches the CQ as binary commands. A Python function that dispatches 50 ops produces a trace with 50 entries; the Python code that called those ops is not part of the trace and is not re-executed during replay.

**Shape changes.** If any tensor changes shape between capture and replay — including a batch dimension, sequence length, or hidden dimension — the buffer addresses encoded in the trace will no longer match the buffers actually allocated for those tensors. The replay will access wrong addresses. There is no runtime check that detects this silently; the result is incorrect output or a device fault. See `trace_constraints.md` for the full discussion.

**Dynamic control flow.** Any op whose behavior depends on the numeric values of a device tensor at dispatch time — for example, an op that branches on whether a value exceeds a threshold, or a loop that runs for a device-computed number of iterations — cannot be correctly traced. The trace records the commands that were generated during the capture run's specific execution path. If a different execution path would be taken on a subsequent call (because the input values differ), the trace will replay the wrong path.

**Host-side Python computations.** Any work done in Python between op calls — updating a counter, indexing into a Python list, calling a non-TTNN library function — is not part of the trace. This is usually desirable: it means Python-side bookkeeping does not affect the trace at all. But it also means that if your Python code computes a value that it then passes as an op argument (e.g., a scaling factor computed from the current step index), the trace will have recorded the capture-time value of that argument, not the current-step value.

---

## Minimal Code Pattern: Capture and Replay a Decode Step

The following example shows the complete, minimal structure for capturing a fixed-shape decode step and replaying it in a loop. Explanatory comments describe every non-obvious decision.

```python
import ttnn
import torch

# ── Device setup ─────────────────────────────────────────────────────────────

device = ttnn.open_device(device_id=0, num_hw_cqs=1)
ttnn.SetDefaultDevice(device)

# Enable async dispatch so that ttnn.execute_trace calls are non-blocking
# from the Python thread's perspective, allowing host-device pipelining
# across decode steps (as described in Chapter 2).
device.enable_async(True)

# ── Tensor setup ─────────────────────────────────────────────────────────────

# These tensors must be at fixed device DRAM addresses for the duration
# of the trace's lifetime. Allocate them before capture and do not
# deallocate or reallocate them.
input_tensor = ttnn.from_torch(
    torch.zeros(1, 1, 1, 512, dtype=torch.bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
)

# Weight tensors for the model are assumed to be already on the device.
# They are fixed (never written during decode), so their addresses are
# naturally stable across capture and replay.

# ── Capture phase (executed once) ────────────────────────────────────────────

# Synchronize before capture to ensure the device has completed any
# preceding work. This is good practice to ensure a clean capture baseline.
ttnn.synchronize_device(device)

# Open the trace capture session on CQ 0.
ttnn.begin_trace_capture(device, cq_id=0)

# Run the decode step once. This executes live on the device AND records
# all encoded commands into the trace buffer.
#
# model.decode_step is assumed to call a fixed sequence of ttnn ops
# with no dynamic shapes, no host readbacks, and no data-dependent
# branching. All ops must be compatible with tracing (see trace_constraints.md).
output_tensor = model.decode_step(input_tensor)

# Close the capture session. The trace buffer is finalized on the device
# and a trace_id handle is returned. The output_tensor already contains
# valid results from the live capture run.
trace_id = ttnn.end_trace_capture(device, cq_id=0)

# Optional but recommended: validate capture output before entering the
# replay loop. This confirms the model produces correct results with the
# specific input and weight configuration that will be replayed.
ttnn.synchronize_device(device)
capture_output_host = ttnn.to_torch(output_tensor)

# ── Replay loop ───────────────────────────────────────────────────────────────

for step in range(num_decode_steps):
    # Update input_tensor IN-PLACE with the new step's data.
    # This MUST be an in-place device-to-device write to the same
    # buffer that was used during capture — do not allocate a new tensor.
    # A device-side scatter or copy into the existing buffer is the
    # correct pattern.
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(next_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        input_tensor,
    )

    # Replay the captured trace. With blocking=False, this call returns
    # immediately; the device executes the recorded command sequence
    # independently. output_tensor will be updated in-place at the same
    # device address used during capture.
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)

    # The Python thread can do other work here (e.g., prepare the next
    # step's input data) while the device replays the trace.
    next_input = prepare_next_input(step)

# After the loop: synchronize once and read all results.
ttnn.synchronize_device(device)
final_output = ttnn.to_torch(output_tensor)

# ── Cleanup ───────────────────────────────────────────────────────────────────

# Release the trace buffer from device DRAM.
ttnn.release_trace(device, trace_id)

ttnn.close_device(device)
```

<details>
<summary>Context: what model.decode_step must look like for capture to succeed</summary>

For the capture above to produce a valid trace, `model.decode_step` must satisfy the following conditions:

```python
def decode_step(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    # All ops use fixed shapes derived from fixed-size weight tensors.
    # No tensor has a shape that varies with input content.
    q = ttnn.matmul(input_tensor, self.wq)          # fixed shapes
    k = ttnn.matmul(input_tensor, self.wk)          # fixed shapes
    scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1))  # q @ k^T
    attn = ttnn.softmax(scores, dim=-1)             # softmax over q@k^T scores
    x = ttnn.matmul(attn, self.wv)                  # fixed shapes
    x = ttnn.matmul(x, self.w_out)                  # fixed shapes
    x = ttnn.add(x, input_tensor)                   # fixed shapes

    # No host readbacks inside the function.
    # No Python if/while conditions that depend on tensor values.
    # No ops whose behavior changes based on tensor content.

    return x
```

If `model.decode_step` contains any of the disqualifying patterns described in `trace_constraints.md`, the trace will capture an incorrect or incomplete command sequence and replay will produce wrong outputs.
</details>

---

**Next:** [`trace_internals.md`](./trace_internals.md)
