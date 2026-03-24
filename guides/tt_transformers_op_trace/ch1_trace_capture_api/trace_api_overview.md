# Trace API Overview

TTNN exposes three Python functions that together implement the trace lifecycle. All three are
thin wrappers over a C++ implementation in
`ttnn/cpp/ttnn/operations/trace.cpp` that delegates to `MeshDevice::begin_mesh_trace`,
`MeshDevice::end_mesh_trace`, and `MeshDevice::replay_mesh_trace`.

---

## `ttnn.begin_trace_capture`

```python
trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
```

**What it returns.** A `MeshTraceId` — a thin wrapper around a `uint32_t` — that uniquely
identifies the in-progress trace on this device. Keep this value; it is required for every
subsequent call in the lifecycle.

**What it does internally.**

1. Calls `TracyTTMetalBeginMeshTrace` so that profiling tools can mark the boundary.
2. Asserts that command queue `cq_id` is not already recording a trace (each CQ can record at
   most one trace at a time).
3. Calls `mark_allocations_safe()`, which tells the DRAM allocator that any buffer allocated
   during the capture run will be part of the trace and must not be moved.
4. Creates an empty trace buffer in the active sub-device manager, then calls
   `mesh_command_queues_[cq_id]->record_begin(trace_id, trace_buffer->desc)` to begin
   intercepting all subsequent device commands issued on that CQ.

> **Note:** `ttnn.begin_trace_capture` does not execute any ops itself. It only arms the
> recording mechanism. The ops you call between `begin_trace_capture` and `end_trace_capture`
> are what gets recorded.

---

## `ttnn.end_trace_capture`

```python
ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
```

**What it seals.** Every device command issued on `cq_id` since the matching
`ttnn.begin_trace_capture(mesh_device, cq_id=0)` call is now frozen into the command buffer
associated with `trace_id`. This includes:

- The full sequence of kernel dispatch commands (one per TTNN op).
- The kernel arguments passed to each kernel, including the DRAM buffer addresses of input and
  output tensors as they existed at capture time.
- NOC transfer descriptors, compute program binaries (by reference — not inline), and
  synchronization barrier encodings. Program binaries stored in the program cache are pinned
  for the lifetime of the trace: the trace holds a reference that prevents the cache from
  evicting or relocating the binary, so replay is safe as long as the trace object remains live.

**What it does internally.**

1. Calls `mesh_command_queues_[cq_id]->record_end()` to stop intercepting commands.
2. Calls `MeshTrace::populate_mesh_buffer(...)` to copy the recorded command sequence from the
   staging area into the trace region of DRAM and finalize the `MeshTraceBuffer`.
3. Calls `mark_allocations_unsafe()`, which re-enables normal DRAM allocation checks going
   forward.
4. Calls `TracyTTMetalEndMeshTrace` to close the profiling marker.

> **Warning:** `ttnn.end_trace_capture` must be called with the same `trace_id` that was
> returned by `ttnn.begin_trace_capture`. Passing a mismatched `trace_id` triggers a fatal
> assertion at the C++ layer.

---

## `ttnn.execute_trace`

```python
ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)  # raw API default
```

**What it dispatches.** The pre-encoded command buffer identified by `trace_id`. The device
firmware reads the buffer directly from DRAM and executes each command without any further
encoding by the host CPU.

**The non-blocking dispatch model.** `generator.py` in tt-transformers always passes
`blocking=False` explicitly on every call. See [`replay_mechanics.md`](./replay_mechanics.md)
for a full explanation of the host/device overlap this enables and the synchronization
requirements that follow.

---

## What Is Recorded

During the capture run, TTNN records exactly what the command queues observe:

- **The command sequence.** The ordered list of kernel launch descriptors, each containing the
  program binary reference, the grid of Tensix cores to use, and the runtime argument layout.
- **Kernel arguments.** The actual argument values passed to each kernel at capture time. For
  tensor inputs and outputs, these arguments include the DRAM addresses of the underlying device
  buffers as they exist at the moment the capture run executes.
- **Device buffer addresses.** Because addresses are captured, not recomputed, the same physical
  memory locations must be populated with fresh data before each replay. This is the buffer
  aliasing requirement covered in `replay_mechanics.md`.

---

## What Is NOT Recorded

The following are explicitly excluded from the command buffer:

- **Python control flow.** Any `if`, `for`, or `while` statement in the Python model forward
  pass runs once during the capture run to determine which ops are called, but the branching
  logic itself is not recorded. The trace always replays the same sequence of ops that was
  observed during the capture run.
- **Tensor shape recomputation.** Calls to `.shape`, `.reshape`, or shape-dependent Python
  logic are host-side operations. They execute during the capture run but are not part of the
  device command stream.
- **Host-side sampling logic.** Sampling steps performed on the CPU (argmax, top-k, etc.) are
  not device commands and are never captured.
- **Any op called before `ttnn.begin_trace_capture`.** The compile run forward pass is
  untraced by definition. Ops called in that pass warm the cache but are not part of any trace.

> **Warning:** Ops that change behavior based on runtime Python variables (for example, a
> conditional that depends on a tensor value read back to host) will evaluate that condition only
> once, during the capture run. The captured trace always follows the branch taken at capture
> time, regardless of what the runtime value would be on subsequent replays.

---

## Example

```python
import ttnn

output = model.forward(input_tensor)

trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
output_trace = model.forward(input_tensor)
ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

for step in range(num_decode_steps):
    copy_host_to_device(new_host_inputs, device_tensors=captured_device_inputs)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
```

In `models/tt_transformers/tt/generator.py` this pattern is implemented by
`_capture_decode_trace_text` (Phase 1 + Phase 2) and `_decode_forward_trace_text` (Phase 3).

---

**Next:** [`compile_run_requirement.md`](./compile_run_requirement.md)
