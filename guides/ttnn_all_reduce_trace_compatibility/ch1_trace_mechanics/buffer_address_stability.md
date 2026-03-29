# Buffer Address Stability

This file defines the central constraint of trace replay — buffer address stability — and explains the two buffer categories (persistent input buffers and persistent output buffers) that satisfy it. It also describes the failure mode when a device operation allocates a new output tensor inside a captured trace, and explains why weight tensors naturally satisfy the stability constraint without any special handling.

## The Central Constraint

When `ttnn.end_trace_capture` finalizes the trace, every kernel argument that encodes a buffer address is frozen at the address value that was current during capture. When `ttnn.execute_trace` issues a replay, the device firmware re-issues those kernel commands using the frozen addresses verbatim. There is no address fixup step: the device does not consult the current allocator state to resolve where a tensor "should" live now.

The constraint follows directly:

> **Every device buffer that is read or written by any kernel inside a captured trace must reside at the same physical device address during every replay as it did during capture.**

Violating this constraint produces one of two outcomes:
- If the old address is now invalid (the buffer was freed and the allocator reclaimed that memory range), the device reads or writes garbage from an unrelated buffer, producing incorrect outputs with no runtime error.
- If the old address overlaps with a buffer that was subsequently allocated for a different tensor, the kernel silently corrupts that tensor's data.

Neither violation produces a detectable on-device exception in normal fast-dispatch mode. The result is silent numerical corruption that may appear only as a statistical anomaly in model output quality.

## Persistent Input Buffers

A persistent input buffer is a device tensor that is allocated once before `begin_trace_capture`, remains at the same device address for all subsequent replays, and has its contents updated in-place (via `ttnn.copy`) before each replay rather than being reallocated.

`TracedRun._capture_trace` allocates persistent input buffers for every `ttnn.Tensor` argument in the module's forward call:

```python
# From models/experimental/tt_symbiote/core/run_config.py
# _capture_trace (simplified to show the buffer allocation pattern)

mem_config = TracedRun._input_memory_config or ttnn.DRAM_MEMORY_CONFIG

trace_inputs = []
trace_func_args = []

for arg in func_args:
    if isinstance(arg, ttnn.Tensor):
        # Pull to host if the tensor is already on device,
        # then move it back to device at a stable address.
        host_tensor = arg.cpu() if arg.storage_type() != ttnn.StorageType.HOST else arg
        trace_input = ttnn.to_device(host_tensor, device, memory_config=mem_config)
        trace_inputs.append(trace_input)
        trace_func_args.append(trace_input)
    # ... (other argument types elided)
```

`ttnn.to_device` allocates a new device buffer at whatever address the allocator assigns. That address is then captured in the trace when `module.forward(*trace_func_args, ...)` is called inside the `begin_trace_capture` / `end_trace_capture` brackets. The key property is that `trace_input` is kept alive (referenced by `entry.trace_inputs`) for the lifetime of the trace entry. The allocator will not reclaim that address because the buffer object is still live.

At each subsequent replay, `_copy_inputs_to_trace_buffer` writes new data into the persistent buffer without moving it:

```python
# From models/experimental/tt_symbiote/core/run_config.py
# _copy_inputs_to_trace_buffer (simplified)

for arg, trace_input in zip(new_args, entry.trace_inputs):
    if isinstance(arg, ttnn.Tensor):
        # ttnn.copy writes arg's data into trace_input's existing
        # device buffer. The buffer address of trace_input does not change.
        ttnn.copy(arg, trace_input)
```

`ttnn.copy` performs a DMA of the tensor data into the existing device buffer. It does not allocate a new buffer, does not change the buffer's base address, and does not modify the allocator's free list. The device address recorded in the trace remains valid.

> **Warning:** If `ttnn.copy` is replaced by `ttnn.clone` or if the caller inadvertently calls `ttnn.to_device` again (which allocates a new buffer at a potentially different address) and passes the result to `ttnn.execute_trace`, the trace will replay against the original captured address while the new data lives at a different address. The result is that the device reads stale data from the old (now possibly reclaimed) buffer.

## Persistent Output Buffers

The output tensor of the traced forward pass is also a persistent buffer. When `module.forward` is called during capture, the final op in the forward method allocates an output tensor. That output tensor's device buffer address is baked into the kernel arguments of whatever op produced it (as the destination address for the output DMA). The same buffer receives the output on every replay.

`TracedRun._capture_trace` retains the output as `trace_output` in the `TraceEntry`:

```python
entry = TraceEntry(
    trace_id=trace_id,
    trace_inputs=trace_inputs,
    trace_output=trace_output,  # holds the persistent output buffer
    device=device,
)
```

At each replay, `module_run` returns `entry.trace_output` directly without re-running the forward method:

```python
# Cache hit path in TracedRun.module_run
entry = TracedRun._trace_cache[cache_key]
TracedRun._copy_inputs_to_trace_buffer(func_args, entry.trace_inputs)
ttnn.execute_trace(entry.device, entry.trace_id, cq_id=TracedRun._cq_id, blocking=False)
result = entry.trace_output  # same buffer object every time
```

This means the output tensor that the caller receives on every replay is the same Python object wrapping the same device buffer. The device writes new values into that buffer on each replay. The caller must not hold a reference to the buffer and expect it to retain the previous replay's values after the next replay completes.

### Capture-window buffer vs. warm-up buffer

Two terms for persistent output buffers appear throughout later chapters and must be distinguished here:

- **Warm-up buffer** (`TraceEntry.trace_output`): The output tensor produced by the single warm-up call to `module.forward` that `_capture_trace` runs *before* `ttnn.begin_trace_capture`. This buffer is stored in `TraceEntry.trace_output` and is what `TracedRun.module_run` returns to callers on every cache-hit replay. `ttnn.execute_trace` **never** writes into this buffer; it always reflects the warm-up output, not the replay result.

- **Capture-window buffer** (`capture_output`): The output tensor produced by `module.forward` when called *inside* the `begin_trace_capture` / `end_trace_capture` window. `ttnn.execute_trace` writes replay results into this buffer. Callers who need to inspect replay output must retain this buffer (not discard it with `_ = ...`) and read from it after each `ttnn.execute_trace` call.

> **Warning:** Reading `entry.trace_output` (the warm-up buffer) after a replay will always return stale warm-up data. This is a known limitation of the current `TracedRun` implementation. A caller who uses `TracedRun.module_run`'s return value to verify replay correctness will silently read stale data and may incorrectly conclude that the trace is correct. See Chapter 4 (`integration_checklist.md` and `minimal_test_pattern.md`) for the full two-buffer sequence and the correct way to read replay results.

## Dynamic Allocation Inside a Trace: The Failure Mode

If an op allocates an intermediate buffer during capture, its address is baked into the trace; once Python garbage collection reclaims that buffer between replays, the allocator is free to assign the same address to a different tensor. The resulting corruption is probabilistic: it depends on allocator behavior and may not appear until the second or third replay, after enough intervening allocations have recycled the captured address.

> **Key finding:** Any op inside a trace capture that allocates a new device buffer for its output (rather than writing into a pre-allocated buffer) produces a trace that is conditionally correct on the first replay but degrades under sustained load as the allocator recycles the captured addresses. The only safe pattern is to pre-allocate all buffers before the capture *and retain them for the lifetime of the trace* so that the allocator cannot reclaim their addresses between replays.

## Weight Tensors: Natural Address Stability

Weight tensors in symbiote modules satisfy the stability constraint by construction, without any explicit persistent-buffer management:

```python
# From models/experimental/tt_symbiote/modules/linear.py
# TTNNLinear.move_weights_to_device_impl

def move_weights_to_device_impl(self):
    self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
    self.tt_bias = ttnn.to_device(self.tt_bias_host, self.device) if self.tt_bias_host is not None else None
```

`move_weights_to_device_impl` is called by `module_run` before the trace capture check. If the module is entered for the first time (first call to `module_run`), the weights are moved to device. They stay at that address for the lifetime of the module instance because `deallocate_weights_impl` is only called if the module is explicitly destroyed or decorated with `@deallocate_weights_after`. The weight buffer address captured in the trace is the same address that will be present at every subsequent replay.

This is the key reason why tracing a matmul over a fixed weight matrix is straightforward: the weight tensor is a persistent buffer by virtue of being a persistent Python attribute, not because of any explicit persistent-buffer API call. The complexity only arises for activations (input tensors) and intermediate results that are created and discarded on every forward call.

---

**Next:** [`semaphore_initialization_and_replay.md`](./semaphore_initialization_and_replay.md)
