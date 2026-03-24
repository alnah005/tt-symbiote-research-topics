# Replay Mechanics

Once a trace is captured, every subsequent call to `ttnn.execute_trace` dispatches the same
pre-encoded command buffer to the device. This section explains why replay is dramatically faster
than live dispatch, what constraints that imposes on input tensor management, and how to reason
about synchronization when `blocking=False` is used.

---

## Why Replay Is Faster Than Live Dispatch

In a standard (non-traced) forward pass, every TTNN op call on the Python host triggers a chain
of work before any device command is issued:

1. Python calls the TTNN op wrapper.
2. TTNN looks up the op in the program cache and validates the configuration.
3. If the program is found, TTNN encodes the runtime arguments (including current buffer
   addresses) into a host-side command descriptor.
4. The command descriptor is submitted to the host-side command queue.
5. The command queue driver DMA-transfers the descriptor to the device.
6. The device firmware processes the descriptor and launches the kernel.

Steps 1–4 happen on the CPU, in Python and C++, for every op on every decode step. For a
70-layer transformer with attention, feedforward, and normalization ops, this adds up to several
hundred host-side encode operations per decode step. At a batch size of 32 and a decode rate of
hundreds of steps per second, this CPU overhead can consume significant wall time — on the order
of 1–5 ms per step depending on model size and host CPU frequency.

During trace replay, steps 1–4 are eliminated entirely. `ttnn.execute_trace` issues a single
command to the device: "read and execute the command buffer stored at this DRAM address." The
device firmware reads the pre-encoded command sequence directly from the trace region and
dispatches kernels without any host involvement. The host returns from `execute_trace` after
enqueuing that single command, not after encoding hundreds of op descriptors.

> **Key insight:** Trace replay reduces the per-step host overhead from O(num_ops) to O(1).
> The device does not wait for the host to re-encode commands; it reads them from its own DRAM.

The swimlane below contrasts the two dispatch modes:

```
Live dispatch (per decode step):
┌──────────────────────────────────────────────────────────────────┐
│ Python host  │ op1() → op2() → op3() → ... → opN()  [~1–5 ms]  │
├──────────────────────────────────────────────────────────────────┤
│ TTNN runtime │ encode cmd1 → encode cmd2 → ... → encode cmdN    │
├──────────────────────────────────────────────────────────────────┤
│ Device       │          cmd1 ──► cmd2 ──► ... ──► cmdN          │
└──────────────────────────────────────────────────────────────────┘

Trace replay (per decode step):
┌──────────────────────────────────────────────────────────────────┐
│ Python host  │ execute_trace()  [~10–50 us]                      │
├──────────────────────────────────────────────────────────────────┤
│ TTNN runtime │ enqueue "replay trace_id"                         │
├──────────────────────────────────────────────────────────────────┤
│ Device       │ read command buffer from DRAM ──► execute all ops │
└──────────────────────────────────────────────────────────────────┘
```

---

## Buffer Aliasing

The command buffer records device DRAM addresses, not symbolic tensor names. When a kernel is
launched during replay, it uses exactly the same addresses that were present at capture time —
there is no mechanism to redirect a kernel to a different buffer.

This means: **input tensors must always reside at the device addresses that were allocated just
before `ttnn.begin_trace_capture` was called.**

If you allocate a new device tensor between replays (for example, by calling `ttnn.to_device`
without specifying a pre-existing buffer), that tensor lands at a new DRAM address. The trace
kernels will not see it — they will read from the original captured addresses, which still hold
stale data from the previous step.

The correct mechanism for updating trace inputs between replays is `copy_host_to_device` with
the `device_tensors=` parameter, which overwrites the contents of the pre-existing device
buffers in place, without allocating new DRAM.

### How `copy_host_to_device` with `device_tensors=` Works

The function is defined in `models/tt_transformers/tt/common.py`:

```python
def copy_host_to_device(host_tensors, device_tensors=None, mesh_device=None, shard_specs=None):
    if device_tensors is None:
        # Allocates new device buffers and copies into them
        ...
    else:
        # Overwrites existing device buffers; no new allocation
        for i in range(len(host_tensors)):
            ttnn.copy_host_to_device_tensor(host_tensors[i], device_tensors[i])
        return device_tensors
```

When `device_tensors` is supplied, `ttnn.copy_host_to_device_tensor` is called for each tensor.
This writes new data into the buffer at the original captured address. The trace kernels will
read the updated data on the next replay.

### Usage in `_decode_forward_trace_text`

The pattern in `models/tt_transformers/tt/generator.py` (lines 910–920) is:

```python
if reset_inputs:
    copy_host_to_device(
        host_tensors=host_inputs_i,
        device_tensors=self.trace_inputs_decode[sampling_on_device][i],
    )
for i, trace_id in self.trace_ids_decode[sampling_on_device].items():
    ttnn.execute_trace(self.model_args[i].mesh_device, trace_id, cq_id=0, blocking=False)
```

`self.trace_inputs_decode` holds the device tensor list returned by `_capture_decode_trace_text`.
Those tensors are at the captured addresses. `copy_host_to_device` with `device_tensors=` writes
the current step's tokens and positions into those buffers, then `execute_trace` dispatches the
trace. The trace reads from those exact buffers, processes the updated inputs, and writes results
into the captured output buffers.

Similarly, in `_prefill_forward_trace` (lines 265–269):

```python
device_inputs = copy_host_to_device(
    host_inputs, device_tensors=device_inputs, mesh_device=self.model_args[model_id].mesh_device
)
ttnn.execute_trace(self.model_args[model_id].mesh_device, trace_id, cq_id=0, blocking=False)
```

> **Warning:** Never call `ttnn.to_device` or any function that allocates new device buffers
> between `copy_host_to_device(device_tensors=...)` and `ttnn.execute_trace`. A new allocation
> does not update the trace's address table. Only the buffers at the addresses captured during
> Phase 2 will be read by the trace kernels.

---

## `blocking=False` and Host/Device Overlap

All `execute_trace` calls in tt-transformers pass `blocking=False`:

```python
ttnn.execute_trace(self.model_args[i].mesh_device, trace_id, cq_id=0, blocking=False)
```

With `blocking=False`, `execute_trace` enqueues the trace replay command on the hardware command
queue and returns immediately to Python. The device executes the trace asynchronously. The host
is free to proceed: preparing the next batch of inputs, sampling output tokens from a previous
step, or managing scheduling logic.

The output tensor (`tt_out_trace`) returned at capture time remains valid as a handle to the
captured output buffer. Reading its contents before the device finishes writing them would
produce garbage. The caller must synchronize before any CPU-side consumption of output data.

---

## Synchronization After `execute_trace`

Because `blocking=False` leaves the device executing asynchronously, the host must explicitly
synchronize before it reads device outputs back to CPU memory.

Two mechanisms are used in tt-transformers:

### `ttnn.synchronize_device`

A blocking call that waits until all commands enqueued on all command queues for the given device
have completed:

```python
ttnn.synchronize_device(self.model[model_id].mesh_device)
```

This is used in `generator.py` (line 511) after a batch of prefill results has been dispatched
with `blocking=False`. It ensures that `ttnn.to_torch` (or any CPU readback) sees the final
device-side values.

### Implicit synchronization via `.cpu()` or `ttnn.to_torch`

Calling `.cpu(blocking=True)` (the default when no argument is supplied) or `ttnn.to_torch` on
a device tensor blocks the host until all in-flight operations on that tensor's command queue
have completed, then transfers data to host memory. This is the blocking-safe path: the returned
host tensor is immediately valid.

Calling `.cpu(blocking=False)` is different: it returns immediately without waiting. The DMA
transfer is initiated asynchronously, and the returned host tensor is **not valid** until the
transfer completes. Reading from it before completion produces garbage. A subsequent
`ttnn.synchronize_device()` call or an explicit event wait is required before the host tensor
can be safely read.

This is the mechanism used in the decode output path:

```python
outputs = (tt_out[i][0].cpu(blocking=False), tt_out[i][1].cpu(blocking=False))
```

Because `blocking=False` is passed here, the returned host tensors are not immediately safe to
read. `ttnn.record_event` and event-based synchronization are used to track DMA completion in
the multi-model data-parallel decode path (lines 1279–1281 of `generator.py`) before those
tensors are consumed.

### When synchronization is not needed between replays

Synchronization is needed whenever you read device outputs back to host. For output tensors that
are read via `.cpu()` or passed to a callback, call `ttnn.synchronize_device()` before accessing
the result (see the section above for the two mechanisms available).

For the specific case where the only operation between two `execute_trace` calls is
`copy_host_to_device` with `device_tensors=`, and that copy uses the same command queue
(`cq_id=0`) as `execute_trace`, no additional explicit synchronization is needed before writing
to the input buffers. `ttnn.copy_host_to_device_tensor` enqueues the DMA write on `cq_id=0`.
Because the command queue is strictly ordered, the DMA write is guaranteed to complete before
the subsequent trace replay command begins execution on the device.

> **Warning:** This ordering guarantee applies only when inputs are written via
> `copy_host_to_device_tensor` on the same command queue (`cq_id=0`) used by `execute_trace`.
> If a different CQ or a direct host memory write is used, explicit synchronization or event
> insertion is required to avoid a race condition.

---

**Next:** [Chapter 2 — How tt-transformers Uses Trace Capture](../ch2_generator_trace_flows/index.md)
