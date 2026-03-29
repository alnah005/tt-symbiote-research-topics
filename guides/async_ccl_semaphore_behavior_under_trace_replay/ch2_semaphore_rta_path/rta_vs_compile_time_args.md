# Runtime Arguments vs. Compile-Time Kernel Arguments

This file establishes the foundational distinction between compile-time kernel arguments and runtime arguments (RTAs), then shows concretely that global semaphore addresses are passed as RTAs in both `AllGatherAsync` and `ReduceScatterMinimalAsync`, and explains why that classification determines the entire semaphore-in-trace story.

---

## Compile-Time Arguments

Compile-time kernel arguments (also called compile-time defines or `ct_args` in the tt-metal API) are integer constants that are embedded in the kernel binary at the moment the kernel is compiled. They are baked into the RISC-V instruction stream as immediate values. Because they live in the binary itself, changing a compile-time argument requires a full recompile of the kernel — which translates to a program cache miss in the device operation framework. Examples of compile-time arguments in CCL kernels include topology constants, ring size, and direction counts: values that are fixed for a given model configuration and do not change between calls.

The program cache key for a device operation includes all compile-time arguments. Two invocations of the same op with different compile-time arguments produce two distinct cached programs.

## Runtime Arguments

Runtime arguments (RTAs) are per-core scalar values written into a dedicated region of L1 kernel config space immediately before each program dispatch. They are not embedded in the binary; instead, the dispatch pipeline writes them into L1 as part of the `runtime_args_command_sequences` section of the `ProgramCommandSequence`. This means RTAs can change between invocations without invalidating the cached program binary.

From `tt_metal/impl/program/program_command_sequence.hpp`:

```cpp
struct ProgramCommandSequence {
    // ...
    std::vector<HostMemDeviceCommand> runtime_args_command_sequences;
    // ...
    // Note: some RTAs may have their RuntimeArgsData modified so the source-of-truth
    // of their data is the command sequence. They won't be listed in rta_updates.
    std::vector<RtaUpdate> rta_updates;
    // ...
};
```

The `RtaUpdate` struct tracks the source and destination for each RTA block that needs to be propagated:

```cpp
struct RtaUpdate {
    const void* src;
    void* dst;
    uint32_t size;
};
```

Because RTAs are written at dispatch time rather than compile time, they can carry values — such as buffer addresses or semaphore addresses — that change per invocation.

---

## Why Semaphore Addresses Are RTAs

Global semaphore handles passed to async CCL operations are per-invocation values: on each call, `get_and_cycle_*` may return a different handle (from the double-buffer pair) whose L1 address differs from the previous call's handle. If semaphore addresses were compile-time arguments, every handle switch would require a program cache miss and a kernel recompile, making double-buffering prohibitively expensive.

Instead, both `AllGatherAsync` and `ReduceScatterMinimalAsync` treat global semaphore addresses as RTAs. The `override_runtime_arguments` mechanism (described in [`override_runtime_arguments_flow.md`](override_runtime_arguments_flow.md)) updates these slots on every program cache hit, keeping semaphore addresses fresh without touching the compiled binary.

---

## AllGatherAsync: Semaphore RTA Slot Assignments

The assignments are made in `all_gather_async_minimal_default_helper_override_runtime_arguments` in
`ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.cpp`:

```cpp
// For each worker core, per link and per direction:
const auto& out_ready_semaphore = semaphore.at(dir);

// sender reader kernel
auto& worker_reader_sender_runtime_args = reader_runtime_args[core.x][core.y];
worker_reader_sender_runtime_args[0] = input.buffer()->address();
worker_reader_sender_runtime_args[1] = output.buffer()->address();
worker_reader_sender_runtime_args[2] = out_ready_semaphore.address(); // out_ready_semaphore L1 address

// sender writer kernel
auto& worker_writer_sender_runtime_args = writer_runtime_args[core.x][core.y];
worker_writer_sender_runtime_args[0] = output.buffer()->address();
worker_writer_sender_runtime_args[3] = out_ready_semaphore.address(); // out_ready_semaphore L1 address

if (barrier_semaphore.has_value()) {
    worker_writer_sender_runtime_args[5] = barrier_semaphore.value().address(); // barrier_semaphore L1 address
}
```

Summary of semaphore RTA slots for `AllGatherAsync`:

| Kernel | Slot index | Value |
|---|---|---|
| Reader | `[2]` | `out_ready_semaphore.address()` (indexed by direction `dir`) |
| Writer | `[3]` | `out_ready_semaphore.address()` (indexed by direction `dir`) |
| Writer | `[5]` | `barrier_semaphore.value().address()` (when barrier semaphore is present) |

Slots `[0]` and `[1]` on the reader, and `[0]` on the writer, carry buffer addresses — also RTAs, also updated by `override_runtime_arguments` on every cache hit.

> **Note:** `semaphore` in the AllGatherAsync operation attributes is a `std::vector<GlobalSemaphore>` with one entry per direction, shared across all links. `semaphore.at(dir)` selects the semaphore for direction `dir`. `barrier_semaphore` is a `std::optional<GlobalSemaphore>` that is only present when an inter-chip barrier is required.

---

## ReduceScatterMinimalAsync: Semaphore RTA Slot Assignments

`ReduceScatterMinimalAsync` has two topology variants — ring and line — each with its own override helper. Both are in
`ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp`.

### Ring topology

`ring_reduce_scatter_minimal_async_helper_override_runtime_arguments`:

```cpp
// sender reader kernel
auto& worker_reader_sender_runtime_args = reader_runtime_args[core.x][core.y];
worker_reader_sender_runtime_args[0] = input.buffer()->address();
worker_reader_sender_runtime_args[1] = intermed.buffer()->address();
worker_reader_sender_runtime_args[2] = semaphore.at(dir).address(); // out_ready_semaphore L1 address

// sender writer kernel
auto& worker_writer_sender_runtime_args = writer_runtime_args[core.x][core.y];
worker_writer_sender_runtime_args[0] = intermed.buffer()->address();
worker_writer_sender_runtime_args[1] = output.buffer()->address();
worker_writer_sender_runtime_args[4] = semaphore.at(dir).address();                         // out_ready_semaphore L1 address
worker_writer_sender_runtime_args[5] = semaphore.at(num_directions_per_link).address();     // batch_ready_semaphore L1 address

if (barrier_semaphore.has_value()) {
    worker_writer_sender_runtime_args[7] = barrier_semaphore.value().address(); // barrier_semaphore L1 address
}
```

Summary of semaphore RTA slots for ring `ReduceScatterMinimalAsync`:

| Kernel | Slot index | Value |
|---|---|---|
| Reader | `[2]` | `semaphore.at(dir).address()` — out_ready_semaphore |
| Writer | `[4]` | `semaphore.at(dir).address()` — out_ready_semaphore |
| Writer | `[5]` | `semaphore.at(num_directions_per_link).address()` — batch_ready_semaphore |
| Writer | `[7]` | `barrier_semaphore.value().address()` — barrier semaphore (when present) |

### Line topology

`line_reduce_scatter_minimal_async_helper_override_runtime_arguments`:

```cpp
// sender reader kernel
auto& worker_reader_sender_runtime_args = reader_runtime_args[core.x][core.y];
worker_reader_sender_runtime_args[0] = input.buffer()->address();
worker_reader_sender_runtime_args[1] = intermed.buffer()->address();
worker_reader_sender_runtime_args[2] = output.buffer()->address();
worker_reader_sender_runtime_args[3] = semaphore.at(0).address(); // out_ready_semaphore L1 address

// sender writer kernel
auto& worker_writer_sender_runtime_args = writer_runtime_args[core.x][core.y];
worker_writer_sender_runtime_args[0] = intermed.buffer()->address();
worker_writer_sender_runtime_args[1] = output.buffer()->address();
worker_writer_sender_runtime_args[4] = semaphore.at(0).address(); // out_ready_semaphore L1 address

if (barrier_semaphore.has_value()) {
    worker_writer_sender_runtime_args[9] = barrier_semaphore.value().address(); // barrier_semaphore L1 address
}
```

Summary of semaphore RTA slots for line `ReduceScatterMinimalAsync`:

| Kernel | Slot index | Value |
|---|---|---|
| Reader | `[3]` | `semaphore.at(0).address()` — out_ready_semaphore |
| Writer | `[4]` | `semaphore.at(0).address()` — out_ready_semaphore |
| Writer | `[9]` | `barrier_semaphore.value().address()` — barrier semaphore (when present) |

> **Note:** The line topology uses only `semaphore.at(0)` because linear reduce scatter has a single logical direction. The ring topology uses `semaphore.at(dir)` for up to `num_directions_per_link` directional semaphores, plus an additional slot at index `num_directions_per_link` for the batch-ready semaphore.

---

## Implications for Trace Replay

Because semaphore addresses are RTAs, they flow through the `override_runtime_arguments` path described in the next file. This path executes on every program cache hit, including hits that occur inside a `begin_trace_capture` / `end_trace_capture` bracket. The addresses written by `override_runtime_arguments` during the capture bracket are the addresses that appear in the dispatch command stream that gets frozen into the `MeshTraceDescriptor::ordered_trace_data` DRAM command buffer — and therefore the addresses that are replayed verbatim on every subsequent `execute_trace`.

If global semaphore addresses were compile-time arguments, the trace would not record them at all (they would already be baked into the cached binary on first compilation, never changing). The fact that they are RTAs is precisely what makes them susceptible to the capture-time snapshot problem analyzed in Chapter 3.

---

**Next:** [`override_runtime_arguments_flow.md`](override_runtime_arguments_flow.md)
