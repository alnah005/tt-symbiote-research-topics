# The override_runtime_arguments Flow

This file explains how the device operation program cache lifecycle works, when `override_runtime_arguments` is called versus `create_mesh_workload`, and precisely what each of the two CCL operation factories does inside their respective override implementations. The critical point established here is that `override_runtime_arguments` executes on every program cache hit — and a cache hit occurs when the same op is invoked during a `begin_trace_capture` / `end_trace_capture` bracket (because the compile run preceding trace capture has already populated the cache).

---

## Program Cache Lifecycle

The device operation framework maintains a per-mesh program cache keyed by the compile-time properties of each operation. For a given `AllGatherAsync` or `ReduceScatterMinimalAsync` invocation, the cache key is derived from the operation attributes that affect the compiled kernel binary (topology, ring size, number of links, direction counts, data types, etc.) but not from the runtime-variable attributes (buffer addresses, semaphore addresses).

When an operation is dispatched:

1. **Cache miss** (first invocation or changed compile-time properties): `create_mesh_workload` is called. This function compiles the kernel binaries, sets up the `ProgramCommandSequence`, and calls `SetRuntimeArgs` to establish the initial per-core RTA values. The resulting program and its command sequence are stored in the cache under the compile-time key.

2. **Cache hit** (all subsequent invocations with the same compile-time properties): `override_runtime_arguments` is called. This function receives the current operation attributes and output tensor, then overwrites only the per-invocation RTAs (buffer addresses and semaphore addresses) in the already-constructed program command sequence. The compiled binary is not touched.

Both paths end by dispatching the resulting command sequence to the device (or, during trace capture, writing it into the sysmem bypass buffer).

> **Key insight:** `override_runtime_arguments` is not a special trace-only path. It is the standard hot path for every repeated invocation of a cached op. During a `begin_trace_capture` / `end_trace_capture` bracket, the same cache lookup occurs. Because the compile run that immediately precedes trace capture has already populated the cache, the trace capture invocation always follows the cache-hit path, so `override_runtime_arguments` is always the function that writes semaphore addresses into the command sequence that gets frozen into the trace.

---

## AllGatherAsync: DefaultMeshWorkloadFactory::override_runtime_arguments

The entry point is `DefaultMeshWorkloadFactory::override_runtime_arguments` in
`ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.cpp`:

```cpp
void DefaultMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AllGatherAsyncParams& operation_attributes,
    const AllGatherAsyncInputs& tensor_args,
    Tensor& output_tensor) {
    const auto& input = tensor_args.input_tensor;
    const auto& output = output_tensor;

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        auto semaphore = operation_attributes.semaphore;
        auto barrier_semaphore = operation_attributes.barrier_semaphore;

        all_gather_async_minimal_default_helper_override_runtime_arguments(
            program,
            shared_vars.reader_kernel_id,
            shared_vars.writer_kernel_id,
            shared_vars.all_cores,
            operation_attributes.num_links,
            shared_vars.num_directions_per_link,
            shared_vars.num_workers_per_direction,
            shared_vars.num_mux_cores_per_direction_per_link,
            shared_vars.num_cores_per_link,
            barrier_semaphore,
            semaphore,
            input,
            output);
    }
}
```

`operation_attributes.semaphore` is the `std::vector<GlobalSemaphore>` that was passed in at the Python layer — the handle returned by `get_and_cycle_ag_semaphore_handles(cluster_axis)` for the current call. `operation_attributes.barrier_semaphore` is the handle returned by `get_and_cycle_barrier_semaphore_handle(cluster_axis)`.

The helper `all_gather_async_minimal_default_helper_override_runtime_arguments` iterates over all worker cores and writes the L1 addresses directly into the live RTA data via `GetRuntimeArgs`. The helper writes the same slot indices documented in [`rta_vs_compile_time_args.md`](./rta_vs_compile_time_args.md) — reader `[2]`, writer `[3]`, writer `[5]` — via `GetRuntimeArgs` references into the live command stream.

`GetRuntimeArgs` returns a reference into the live `RuntimeArgsData` that is bound directly to the command stream (see the mesh trace capture path described in [`trace_node_rta_snapshot.md`](trace_node_rta_snapshot.md)). Writing into these references is therefore writing into the data that will be read when the command sequence is serialized and dispatched, or when it is assembled into `MeshTraceDescriptor::ordered_trace_data` by `assemble_dispatch_commands`.

---

## ReduceScatterMinimalAsync: Ring and Line Override Implementations

`ReduceScatterMinimalAsync` has two workload factory types. Both follow the same pattern.

### RingReduceScatterMeshWorkloadFactory::override_runtime_arguments

In `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp`:

```cpp
void RingReduceScatterMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const ReduceScatterMinimalAsyncParams& operation_attributes,
    const ReduceScatterMinimalAsyncInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    const auto& input = tensor_args.input_tensor;
    const auto& intermediate = tensor_return_value.at(0);
    const auto& output = tensor_return_value.at(1);

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        ring_reduce_scatter_minimal_async_helper_override_runtime_arguments(
            program,
            shared_vars.reader_kernel_id,
            shared_vars.writer_kernel_id,
            shared_vars.all_cores,
            operation_attributes.num_links,
            shared_vars.num_directions_per_link,
            shared_vars.num_workers_per_direction,
            shared_vars.num_mux_cores_per_direction_per_link,
            shared_vars.num_cores_per_link,
            operation_attributes.barrier_semaphore,
            operation_attributes.semaphore,
            input,
            intermediate,
            output);
    }
}
```

`operation_attributes.semaphore` carries the `std::vector<GlobalSemaphore>` selected by `get_and_cycle_rs_semaphore_handles(cluster_axis)`.

The helper `ring_reduce_scatter_minimal_async_helper_override_runtime_arguments` writes the directional and batch-ready semaphore addresses into the per-core RTA slots documented in [`rta_vs_compile_time_args.md`](./rta_vs_compile_time_args.md).

### LineReduceScatterMeshWorkloadFactory::override_runtime_arguments

```cpp
void LineReduceScatterMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const ReduceScatterMinimalAsyncParams& operation_attributes,
    const ReduceScatterMinimalAsyncInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    // ...
    line_reduce_scatter_minimal_async_helper_override_runtime_arguments(
        program,
        shared_vars.reader_kernel_id,
        shared_vars.writer_kernel_id,
        shared_vars.all_cores,
        operation_attributes.num_links,
        shared_vars.num_directions_per_link,
        shared_vars.num_workers_per_direction,
        shared_vars.num_mux_cores_per_direction_per_link,
        shared_vars.num_cores_per_link,
        operation_attributes.barrier_semaphore,
        operation_attributes.semaphore,
        input,
        intermediate,
        output);
}
```

The helper `line_reduce_scatter_minimal_async_helper_override_runtime_arguments` writes into the per-core RTA slots documented in [`rta_vs_compile_time_args.md`](./rta_vs_compile_time_args.md).

---

## Execution During begin_trace_capture / end_trace_capture

When the Python model calls `ttnn.begin_trace_capture(device)`, the dispatch layer switches the sysmem manager into bypass mode. Subsequent dispatch calls write into a bypass buffer instead of the hardware command queue. From `tt_metal/distributed/fd_mesh_command_queue.cpp`:

```cpp
void FDMeshCommandQueue::record_begin(...) {
    // ...
    for (auto* device : mesh_device_->get_devices()) {
        device->sysmem_manager().set_bypass_mode(/*enable*/ true, /*clear*/ true);
    }
    // ...
}
```

While bypass mode is active, every `enqueue_mesh_workload` call still performs the full program cache lookup and, on a cache hit, calls `override_runtime_arguments`. The resulting `ProgramCommandSequence` — with semaphore addresses freshly written by `override_runtime_arguments` — is then serialized into the bypass buffer via `write_program_command_sequence` inside `capture_program_trace_on_subgrid`.

When `ttnn.end_trace_capture(device, trace_id)` is called, `record_end` calls `trace_ctx_->assemble_dispatch_commands(...)`, which reads from the sysmem bypass buffer that was populated during the capture bracket and builds `MeshTraceDescriptor::ordered_trace_data`. The command stream in `ordered_trace_data` contains the semaphore addresses written by `override_runtime_arguments` verbatim. After `assemble_dispatch_commands` returns, `populate_mesh_buffer` uploads `ordered_trace_data` to a DRAM trace buffer. The hardware prefetcher then replays that immutable DRAM command buffer on each subsequent `execute_trace` call — there is no per-replay update of RTA values.

> **Warning:** The compile run that precedes `begin_trace_capture` is essential for establishing the program cache entry. If the first ever invocation of an async CCL op occurs inside the `begin_trace_capture` / `end_trace_capture` bracket (no compile run beforehand), a cache miss occurs and `create_mesh_workload` runs, not `override_runtime_arguments`. The RTA path is the same in terms of what gets written, but the distinction matters if you are reasoning about which code path determines the slot assignments.

---

## Summary

The writes made by `override_runtime_arguments` during capture are embedded verbatim in `ordered_trace_data`; see [`trace_node_rta_snapshot.md`](./trace_node_rta_snapshot.md) for the assembly and upload mechanics.

---

**Next:** [`trace_node_rta_snapshot.md`](trace_node_rta_snapshot.md)
