# TraceNode, the rta_data Snapshot, and Trace Replay

This file documents the mesh trace path used when `ttnn.execute_trace` is invoked on a `MeshDevice` — the target of this guide. The key conclusion is that the semaphore address written by `override_runtime_arguments` during the capture bracket is embedded verbatim into an immutable DRAM command buffer, which the hardware prefetcher replays wholesale on every `execute_trace`. All subsequent chapters build on this fact.

> **Note:** The single-device `HardwareCommandQueue` trace path uses a different mechanism (`rta_updates` entries with `src`/`dst` copy semantics and `update_program_dispatch_commands`). That path is out of scope for this guide, which focuses on `MeshDevice` and `FDMeshCommandQueue`. For single-device trace internals, refer to `tt_metal/impl/program/dispatch.cpp` and `tt_metal/impl/dispatch/hardware_command_queue.cpp`.

---

## The Mesh Trace Path: Capture, Upload, and Replay

### Capture: record_end → assemble_dispatch_commands → ordered_trace_data

During trace capture on a `MeshDevice`, each device's sysmem manager is placed in bypass mode. The dispatch commands that would normally be issued to hardware are instead written into the sysmem bypass buffer. When `end_trace_capture` is called, `FDMeshCommandQueue::record_end` (in `tt_metal/distributed/fd_mesh_command_queue.cpp`) assembles the captured commands:

```cpp
void FDMeshCommandQueue::record_end() {
    trace_ctx_->assemble_dispatch_commands(this->device(), ordered_mesh_trace_md_);
    // ...
}
```

`MeshTraceDescriptor::assemble_dispatch_commands` (in `tt_metal/distributed/mesh_trace.cpp`) reads from the sysmem bypass buffer that was populated during capture and builds `MeshTraceDescriptor::ordered_trace_data` — a vector of `MeshTraceData` entries, each associating a device coordinate range with the raw command words captured for it. The semaphore addresses written by `override_runtime_arguments` are embedded verbatim in those command words.

After `assemble_dispatch_commands` returns, the command data is finalized. Nothing modifies `ordered_trace_data` after this point.

### Upload: populate_mesh_buffer

`MeshTrace::populate_mesh_buffer` (also in `mesh_trace.cpp`) allocates a DRAM trace buffer and uploads each entry in `ordered_trace_data` to that buffer via `enqueue_write_shard_to_sub_grid`. The command stream, with semaphore addresses already embedded, is now resident in DRAM.

### Replay: issue_trace_commands via the hardware prefetcher

When `ttnn.execute_trace` is called on a mesh trace, `FDMeshCommandQueue::enqueue_trace` calls `trace_dispatch::issue_trace_commands`, which issues an `add_prefetch_exec_buf` command to the hardware prefetcher pointing at the pre-built DRAM command buffer:

```cpp
void FDMeshCommandQueue::enqueue_trace(const MeshTraceId& trace_id, bool blocking) {
    // ...
    trace_dispatch::issue_trace_commands(
        mesh_device_, device->sysmem_manager(), dispatch_md, id_, expected_num_workers_completed_, dispatch_core_);
    // ...
}
```

The hardware prefetcher fetches and executes the command buffer from DRAM on each replay. There is no per-replay call to `update_traced_program_dispatch_commands` and no per-replay `std::memcpy` of RTA data. The semaphore addresses are frozen because the DRAM command buffer is immutable once assembled — not because of any per-replay host-side restore.

---

## What Is and Is Not in the Snapshot

| Data | Frozen at capture time? | Notes |
|---|---|---|
| `out_ready_semaphore.address()` per direction | Yes | Embedded in the command buffer assembled by `assemble_dispatch_commands`; the DRAM buffer is immutable after `populate_mesh_buffer` |
| `batch_ready_semaphore.address()` (RS ring only) | Yes | Same path |
| `barrier_semaphore.value().address()` | Yes | Same path |
| Input/output buffer addresses | Yes | Also written by `override_runtime_arguments` and embedded in the command buffer |
| Device L1 semaphore word (the uint32 at `semaphore.address()`) | No | This is in device L1; the trace only captures host-side command stream data |
| Kernel binary | No | The binary is in DRAM/prefetcher cache; the dispatch metadata records whether it is cached or needs to be re-sent |

The absence of the device L1 semaphore word from the snapshot is a separate concern from the RTA address snapshot: the device-side semaphore value must be reset to 0 by `ttnn.reset_global_semaphore_value` before each replay (discussed in Chapter 4), because async CCL kernels leave it at a non-zero value after completion, and the trace does not reset it.

---

## Key Conclusion

The semaphore address recorded at `end_trace_capture` time is the one used on every subsequent `execute_trace`. In the mesh trace path, this address is embedded into the DRAM command buffer by `assemble_dispatch_commands` and replayed wholesale by the hardware prefetcher on each `execute_trace` call. The command buffer is immutable after assembly: no per-replay update can change the embedded addresses, regardless of what `get_and_cycle_*` has returned since capture, and regardless of how the host-side cycling counter has advanced. Every replay is therefore bound to the capture-time semaphore handle's L1 address.

This is the mechanism behind the host-counter / trace-handle mismatch analyzed in Chapter 3.

---

**Next chapter:** [Chapter 3 — The Host-Counter / Trace-Handle Mismatch](../ch3_mismatch_analysis/index.md)
