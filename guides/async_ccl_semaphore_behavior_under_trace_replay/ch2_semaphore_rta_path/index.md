# Chapter 2 — How Semaphore Addresses Flow into Kernel Runtime Arguments

This chapter traces the complete path from a Python `GlobalSemaphore` handle through the C++ device operation layer, into the per-core runtime argument (RTA) arrays, and finally into the `MeshTraceDescriptor::ordered_trace_data` DRAM command buffer that is assembled at `end_trace_capture` time and replayed immutably on each `execute_trace`. By the end of this chapter you will know exactly where semaphore addresses live in the dispatch data structures, why they are classified as runtime arguments rather than compile-time arguments, how `override_runtime_arguments` writes them on every program cache hit (including the hit that occurs during a trace capture bracket), and how the mesh trace path freezes those addresses into an immutable DRAM buffer via `assemble_dispatch_commands` and `populate_mesh_buffer`.

---

## Prerequisites from Chapter 1

Chapter 1 established the following facts that this chapter builds on directly:

- A `GlobalSemaphore` wraps a sharded L1 buffer. Its `address()` method returns the L1 address at which the semaphore word is stored on each involved core. This address is **stable** for the lifetime of the object — it does not change between calls.
- `TT_CCL` maintains double-buffered semaphore handle arrays (`ag_semaphore_handles`, `rs_semaphore_handles`, `barrier_semaphore_handles`), each indexed by a host-side cycling counter (`ag_semaphores_idx`, `rs_semaphores_idx`, `barrier_semaphore_idx`). The `get_and_cycle_*` methods select the current handle and advance the counter modulo 2.
- `ttnn.reset_global_semaphore_value(semaphore, value)` dispatches a write command that sets the L1 semaphore word to a given value. It operates on the same address returned by `semaphore.address()`.

---

## Chapter Diagram

The data flow from Python handle to frozen DRAM command buffer on a `MeshDevice`:

```
Python GlobalSemaphore handle
        |
        | .address() called in override_runtime_arguments
        v
Per-core RTA slot in bypass-mode command buffer
        |
        | record_end → assemble_dispatch_commands
        v
MeshTraceDescriptor::ordered_trace_data  (immutable DRAM command buffer)
        |
        | execute_trace → issue_trace_commands → add_prefetch_exec_buf
        v
Hardware prefetcher replays immutable DRAM buffer on device
```

The critical observation: `ordered_trace_data` is assembled **once** at `end_trace_capture` time by `assemble_dispatch_commands`, uploaded to DRAM by `populate_mesh_buffer`, and then replayed verbatim on every subsequent `execute_trace`.

---

## Learning Objectives

After reading this chapter you will be able to answer:

1. What is the difference between a compile-time kernel argument and a runtime argument, and why does that distinction determine whether trace replay is affected by a change in semaphore handle?
2. Which specific RTA slot indices carry semaphore addresses in `AllGatherAsync` and `ReduceScatterMinimalAsync`, and in which source files are those assignments made?
3. What does `override_runtime_arguments` do, when is it called, and why does it execute inside a `begin_trace_capture` / `end_trace_capture` bracket?
4. What is `MeshTraceDescriptor::ordered_trace_data`, how is it assembled by `assemble_dispatch_commands`, how is it uploaded to DRAM by `populate_mesh_buffer`, and how does the hardware prefetcher replay it on each `execute_trace`?

---

## What's Next

Read the files in this order:

| File | Topic |
|---|---|
| [`rta_vs_compile_time_args.md`](./rta_vs_compile_time_args.md) | What RTAs are, why semaphore addresses are RTAs, exact slot indices for AG and RS |
| [`override_runtime_arguments_flow.md`](./override_runtime_arguments_flow.md) | Program cache lifecycle, how `override_runtime_arguments` writes semaphore addresses on every cache hit including during trace capture |
| [`trace_node_rta_snapshot.md`](./trace_node_rta_snapshot.md) | Mesh trace capture: `assemble_dispatch_commands` → `ordered_trace_data` → `populate_mesh_buffer` → DRAM → hardware prefetcher replay |
