# What Trace Capture Records

This file establishes a precise model of what `ttnn.begin_trace_capture` and `ttnn.end_trace_capture` record and what `ttnn.execute_trace` replays. Understanding this model is a prerequisite for every subsequent analysis in this guide: every claim about whether a given op is trace-compatible is ultimately a claim about what the trace runtime serializes and what it assumes stays constant across replays.

## The Recording Unit: MeshCommandQueue Commands, Not Python Logic

When a Python caller invokes `ttnn.begin_trace_capture(device, cq_id=0)`, the TTNN runtime switches the target `MeshCommandQueue` into recording mode. From that point forward, every TTNN op call that enqueues work on that command queue — kernel dispatches, DMA reads, DMA writes, semaphore wait-and-increment sequences, program argument writes — is serialized into an in-memory trace buffer rather than being dispatched immediately to the device.

The recording captures **device-level commands**, not Python-level logic. Specifically:

- **Kernel dispatch records**: each `EnqueueMeshWorkload` call during capture writes a record that encodes which compiled kernel binary runs on which core grid, with which runtime arguments (crucially, including concrete buffer base addresses embedded in those arguments).
- **Buffer DMA records**: `EnqueueWriteMeshBuffer` and `EnqueueReadMeshBuffer` calls during capture write records of the source/destination addresses and transfer sizes that are current at capture time.
- **Semaphore operation records**: any semaphore write or wait that is expressed as a device-side command (not a host-side spin) is serialized with the semaphore's concrete L1 address on the target core.

Python conditionals, shape inference, tensor reshapes that produce no new device allocation, and any host-CPU computation that happens between op calls are **not recorded**. The trace is a flat sequence of device commands; there is no control flow in the recorded stream.

This distinction has a direct implication: if a Python function called during capture allocates a new device buffer — even if that allocation is invisible at the Python level — the allocation is recorded with its concrete address. That address is baked into the downstream kernel arguments. If the buffer is freed and reallocated between replays, the baked-in address is stale.

## The Warm-Up Run: Why It Must Precede Capture

Tenstorrent devices use an ahead-of-time kernel compilation model. The first time a program is dispatched, the host JIT-compiles the kernel into a binary for the target device architecture, uploads it to device SRAM, and then initiates execution. Subsequent dispatches of the same program reuse the cached binary.

If `begin_trace_capture` is called before at least one compile run, the trace recording phase itself triggers compilation. Compilation involves host-device round trips that can inject commands into the command stream at arbitrary points; those compilation commands are not part of the intended op sequence and produce an unreliable trace buffer.

The symbiote `TracedRun._capture_trace` method enforces the warm-up requirement explicitly:

```python
# From models/experimental/tt_symbiote/core/run_config.py

# Warm-up: runs the module once to trigger kernel compilation and
# register programs in the device's program cache. All kernels used
# by the forward pass are compiled and cached after this call.
trace_output = module.forward(*func_args, **func_kwargs)

# Capture: begin_trace_capture switches cq_id=0 into recording mode.
# Every op call below appends to the trace buffer instead of dispatching.
trace_id = ttnn.begin_trace_capture(device, cq_id=cq_id)
_ = module.forward(*trace_func_args, **func_kwargs)
ttnn.end_trace_capture(device, trace_id, cq_id=cq_id)
ttnn.synchronize_device(device)
```

The warm-up call uses the original (non-persistent) `func_args`. The capture call uses `trace_func_args`, which are the persistent input buffers allocated before the warm-up (see [`buffer_address_stability.md`](./buffer_address_stability.md)). This separation is intentional: the warm-up compiles against the original tensor memory locations, and the capture records against the persistent locations that will also be present at every replay.

The `ttnn.synchronize_device` call after `end_trace_capture` ensures that the trace buffer is fully flushed from the MeshCommandQueue into device memory before the first `execute_trace` call is made.

At the C++ layer, `begin_mesh_trace` calls `this->mark_allocations_safe()` before opening the trace buffer — this permits the allocator to satisfy allocation requests during capture but does not extend the lifetime of the resulting buffers beyond normal Python reference counting; the runtime does not take ownership of buffers allocated during capture. `end_mesh_trace` serializes the recorded command list into device memory and then calls `this->mark_allocations_unsafe()`, disallowing further allocations (any allocation during replay would land at a non-deterministic address not present in the trace). `replay_mesh_trace` calls `enqueue_trace(trace_id, blocking)`, which re-issues the serialized command list to the device without any host-side Python or C++ dispatch logic.

## The Three-Phase Pattern

The codebase uses a consistent three-phase pattern for all traced execution:

```
Phase 1 — Compile run (warm-up):
    Dispatch ops normally so that all kernels are compiled and cached.
    This run produces real outputs but its buffer addresses will not be
    reused in subsequent replays.

Phase 2 — Capture:
    ttnn.begin_trace_capture(device, cq_id=cq_id)
    <op calls that will be recorded into the trace buffer>
    ttnn.end_trace_capture(device, trace_id, cq_id=cq_id)
    ttnn.synchronize_device(device)

Phase 3 — Replay (repeated as many times as needed):
    # Update persistent input buffers with new data (no reallocation).
    ttnn.copy(new_input, trace_input_buffer)
    # Re-execute the recorded command stream.
    ttnn.execute_trace(device, trace_id, cq_id=cq_id, blocking=False)
    # Read the output from the persistent output buffer.
    result = trace_output_buffer
```

This pattern appears in `TracedRun` (symbiote), in `Generator._capture_trace_prefill` / `Generator._easy_trace_prefill` (tt-transformers), and in the distributed trace programming example at `tt_metal/programming_examples/distributed/4_distributed_trace_and_events/`. The invariant across all uses is the same: persistent buffers hold the fixed device addresses that the trace records, and only the contents of those buffers change between replays.

> **Note:** `ttnn.execute_trace` with `blocking=False` enqueues the trace replay command and returns immediately to the host. The device executes asynchronously. The output tensor from trace capture (`trace_output`) is a Python handle pointing to the output device buffer; it is valid to read from this handle only after the device has completed replay, which requires either `blocking=True` or a subsequent synchronization point.

---

**Next:** [`buffer_address_stability.md`](./buffer_address_stability.md)
