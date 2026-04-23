# `ttnn.from_torch` Is a Host Operation

`ttnn.from_torch` is the primary entry point for staging PyTorch tensors onto Tenstorrent devices. It looks like a TTNN op — it lives in the `ttnn` namespace, accepts TTNN layout and memory configuration arguments, and returns a `ttnn.Tensor`. Despite this surface appearance, every call to `ttnn.from_torch` with `device=mesh_device` allocates a new device buffer. That allocation is a host operation, and it is invisible to Metal Trace.

---

## The Call Chain

When `ttnn.from_torch(torch_tensor, device=mesh_device, ...)` is called, the following sequence of host-side work occurs before any device command is enqueued:

```python
# Simplified call chain — not literal source, illustrative of the phases

# Phase 1: host-side tensor assembly
host_buffer = allocate_host_dram(torch_tensor.nbytes)
# why: data must be in pinned host memory before DMA can transfer it
copy_to_host_pinned(torch_tensor, host_buffer)

# Phase 2: device buffer allocation  <── HOST OPERATION, INVISIBLE TO TRACE
device_buffer = mesh_device.allocate_device_buffer(
    shape, dtype, layout, memory_config
)
# why: a fresh device DRAM region is reserved; its address is new every call

# Phase 3: DMA transfer (this IS a device command, but it references the
#           new device_buffer address, which the trace cannot track)
enqueue_dma(
    src=host_buffer,
    dst=device_buffer.address,   # <── address unknown to prior trace commands
    size=host_buffer.nbytes,
    queue=mesh_command_queue,
)

return ttnn.Tensor(device_buffer)
```

Step 2 is the critical step. `mesh_device.allocate_device_buffer` calls into the TT-Metal memory allocator, which returns a fresh DRAM region at an address that was not in use before the call. This address is not predictable, not stable across calls, and not recorded by the trace.

Step 3 enqueues a DMA command that uses the new address. Even if the DMA command itself were recorded by the trace, the destination address it encodes refers to a buffer that no longer exists on the next replay (because the buffer was allocated inside the capture bracket and will not be re-allocated during replay).

> **Note:** The DMA transfer in Step 3 is a device command and would ordinarily be recordable. The reason it cannot be safely replayed is not that it is unrecordable — it is that the destination address it references is transient. The trace records the address, not the allocation that produced the address.

---

## Why This Is Not Immediately Obvious

Several surface properties of `ttnn.from_torch` suggest it might be trace-safe:

**It has a `ttnn.` prefix.** The `ttnn` namespace is broadly associated with on-device computation. Users familiar with TTNN naturally assume that all `ttnn.*` calls enqueue device work. In reality, `ttnn.from_torch` is a data-staging utility that begins and ends on the host.

**It accepts device arguments.** The `device`, `layout`, and `memory_config` parameters make the call look like a device dispatch. These arguments control *where* the device buffer is allocated, not *whether* a new allocation occurs.

**It produces a `ttnn.Tensor`.** The return value is indistinguishable from a tensor produced by a trace-safe on-device op. There is no flag on the tensor that marks it as having been produced by a host allocation.

**It works correctly in the compile run and capture run.** Because the trace is not yet being replayed, the bug does not appear until `ttnn.execute_trace` is called. The capture run succeeds, giving a false sense of correctness.

> **Warning:** The first failure appears during replay, not during capture. A model that has been successfully traced and executes its first inference correctly can still be subtly broken if `ttnn.from_torch` is called inside the capture bracket, because the same buffer address is replayed against stale data from the capture-time allocation.

---

## The Trace-Safe Alternative: `ttnn.copy` into a Pre-Allocated Buffer

The trace-safe pattern replaces the allocation with a copy into a buffer whose address was established *before* the capture bracket:

```python
# BEFORE the capture bracket — safe, happens once at model initialisation
self.cos_replicated = ttnn.zeros(
    shape=cos_full.shape,
    dtype=cos_full.dtype,
    layout=cos_full.layout,
    device=mesh_device,
    memory_config=replicated_memory_config,
)
# why: the device buffer address is now stable and will not change

# INSIDE the capture bracket — trace-safe
ttnn.copy(src=cos_full, dst=self.cos_replicated)
# why: ttnn.copy enqueues a DMA command that uses self.cos_replicated's
#      pre-existing address; no allocation occurs; the recorded address
#      is valid on every subsequent replay
```

The key distinction is that `ttnn.copy` does not call the device memory allocator. It enqueues a DMA transfer from `src` to a `dst` buffer whose address already exists in the device memory map. The command buffer records the concrete address of `self.cos_replicated`. On every replay, that buffer is still alive at the same address, so the recorded DMA command remains valid.

Stated in terms of the address stability invariant from [`what_trace_records.md`](./what_trace_records.md):

- `ttnn.from_torch` inside the bracket produces an address that is in $B_c$ (recorded at capture) but not guaranteed to be in $B_r^{(i)}$ (present and valid at replay $i$). The invariant $B_c \subseteq B_r^{(i)}$ is violated.
- `ttnn.copy` into a pre-allocated buffer produces a DMA command that references an address in $B_r^{(i)}$ for all $i \geq 1$, because the pre-allocated buffer is never freed. The invariant holds.

> **Key Finding:** The distinction between trace-safe and trace-unsafe operations reduces to a single question: does this operation allocate a new device buffer? If yes, it must not be called inside the capture bracket. `ttnn.from_torch` always allocates. `ttnn.copy` never allocates. This asymmetry is the entire basis for the fix described in [`ensure_replicated_call_site.md`](./ensure_replicated_call_site.md).

---

**Next:** [`ensure_replicated_call_site.md`](./ensure_replicated_call_site.md)
