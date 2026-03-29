# Semaphore Initialization and Replay

This file explains how device semaphores interact with the trace replay contract. It distinguishes global semaphores (device L1 locations with persistent state) from local semaphores (per-program SRAM locations re-initialized on every dispatch), defines the re-entry requirement, and introduces the first observation about how `ttnn.all_reduce` handles its semaphore arguments — an observation that is examined in full in Chapter 2.

## What a Global Semaphore Is

A `GlobalSemaphore` is a sharded L1 buffer, one uint32 per core in the specified `CoreRangeSet`, created by `ttnn.create_global_semaphore`. The C++ constructor allocates the backing buffer immediately and writes the initial value to device:

```cpp
// tt_metal/impl/buffers/global_semaphore.cpp
GlobalSemaphore::GlobalSemaphore(
    IDevice* device, const CoreRangeSet& cores, uint32_t initial_value, BufferType buffer_type)
    : device_(device), cores_(cores) {
    this->setup_buffer(initial_value, buffer_type);
}

void GlobalSemaphore::setup_buffer(
    std::optional<uint32_t> initial_value, BufferType buffer_type, std::optional<uint64_t> address) {
    // Creates a HEIGHT_SHARDED buffer: one uint32 slot per core in the CoreRangeSet.
    // buffer_type must be L1 or L1_SMALL.
    ShardedBufferConfig sem_shard_config = { ... };
    buffer_ = distributed::AnyBuffer::create(sem_shard_config, address);
    if (initial_value.has_value()) {
        this->reset_semaphore_value(initial_value.value());
    }
}
```

The L1 address assigned to the buffer by the allocator is fixed at construction time. Kernels that synchronize using this semaphore receive that address as a compile-time or runtime argument and use it to perform atomic reads and writes to the core's L1 SRAM. The address does not change over the lifetime of the `GlobalSemaphore` object.

The Python API exposes this as:

```python
sem = ttnn.create_global_semaphore(mesh_device, core_range, initial_value=0)
addr = ttnn.get_global_semaphore_address(sem)  # fixed for the object's lifetime
```

## What the Trace Records About Semaphores

When a kernel that interacts with a global semaphore is dispatched during trace capture, the trace records the semaphore's L1 address as part of the kernel's runtime arguments, exactly as it records buffer addresses. Typical operations recorded include:

- **Increment**: a kernel writes `atomic_inc` to the semaphore address on a remote core (signaling that a portion of data is available).
- **Wait**: a kernel spins on the semaphore address until it reaches a threshold value (synchronizing receiver and sender).
- **Reset**: a kernel writes zero (or another baseline value) to the semaphore address before beginning a new phase, so that the next wait loop starts from a known state.

Because the address is frozen in the trace, the replay re-issues these operations against the same L1 address on the same cores. This is correct as long as the semaphore value at the start of each replay is exactly the value that the first replay started with. If the value differs — for example, because a previous replay left a non-zero residual — the wait loop in the replay kernel will either skip its barrier (if the residual value already satisfies the wait condition) or hang indefinitely (if the residual value prevents the condition from being reached on a normal execution path).

## The Re-Entry Requirement

The re-entry requirement states:

> **Before each trace replay, every global semaphore whose initial value is recorded in the trace must be reset to that initial value.**

This is not automatically satisfied by `ttnn.execute_trace`. The trace runtime replays the recorded commands but does not inspect or reset semaphore values before doing so. Semaphore reset is the caller's responsibility.

Two patterns satisfy the re-entry requirement in practice:

**Self-contained lifecycle**: The captured trace includes both the reset-to-zero write at the start and the final reset-to-zero write at the end of each collective operation. If the operation is designed such that it leaves every semaphore at zero when it completes — and the initial value recorded by the first op in the trace is also zero — then the semaphore is already at the correct value when the next replay begins. No external reset is needed.

**Caller-managed initialization**: The semaphore is set to a non-zero value (or to some baseline) before the trace capture begins, and that value is consumed inside the trace. The caller must repeat the initialization — typically by calling `ttnn.reset_global_semaphore_value` — before each replay. Failing to do so causes incorrect behavior on the second and later replays.

A concrete example of the self-contained pattern is a CCL op that cycles through a double-buffered semaphore pair: the first semaphore is incremented and waited on during the first half of the collective, then reset to zero; the second semaphore handles the second half. If the trace captures one complete cycle, both semaphores begin and end at zero, and the lifecycle is self-contained.

A concrete example of the caller-managed pattern is an async collective that uses a barrier semaphore that is pre-initialized to a sentinel value by the host before the collective begins. If the trace starts with a wait on that sentinel, the caller must write the sentinel again before each replay.

> **Warning:** The failure mode for violating the re-entry requirement is the same as the failure mode for buffer address aliasing: it is non-deterministic and depends on execution order. A replay may produce correct output on the first execution if the device happens to complete cleanup before the next replay begins, but fail under load or on the second replay in a rapid loop. This makes semaphore re-entry bugs difficult to reproduce in isolation.

## Resetting Global Semaphore Values

The `ttnn.reset_global_semaphore_value` function issues a **blocking** host-to-device write to reset the semaphore's L1 locations:

```cpp
// tt_metal/impl/buffers/global_semaphore.cpp
void GlobalSemaphore::reset_semaphore_value(uint32_t reset_value) const {
    // Blocking write to ensure the reset value lands on each physical device
    // before the next program runs. This prevents cross-chip writes to the
    // semaphore from being lost due to device skew.
    std::vector<uint32_t> host_buffer(cores_.num_cores(), reset_value);
    auto mesh_buffer = buffer_.get_mesh_buffer();
    distributed::EnqueueWriteMeshBuffer(
        mesh_buffer->device()->mesh_command_queue(), mesh_buffer, host_buffer, true /* blocking */);
}
```

The blocking write ensures that by the time the host calls `ttnn.execute_trace`, the semaphore value is at the correct initial state on all devices in the mesh. For `GlobalSemaphore` objects created with a `MeshDevice*`, this single object is already mesh-aware and handles all devices in the mesh. The `MultiDeviceGlobalSemaphore` type (returned by `ttnn::global_semaphore::create_global_semaphore` when passed a `std::vector<IDevice*>` instead of a `MeshDevice*`) is a distinct public API type that holds one `GlobalSemaphore` per device and resets all per-device semaphores in sequence via the overloaded `reset_global_semaphore_value`. In the common multi-device collective case using a `MeshDevice`, callers receive a `GlobalSemaphore` directly from `ttnn.create_global_semaphore` and do not need `MultiDeviceGlobalSemaphore`.

## Local Semaphores: No Re-Entry Concern

A local semaphore is created as part of a `tt_metal::Program` via `tt_metal::CreateSemaphore`; it is allocated in per-program SRAM and initialized at program dispatch time, not as a persistent L1 allocation managed by the device allocator. When the initialization is expressed as a device-side command and is therefore captured in the trace, every replay re-initializes the local semaphore to its starting value without any external reset. This is why operations that rely only on local semaphores are generally easier to trace correctly than operations that use global semaphores — the trace carries its own initialization. Whether this initialization is always device-side and always captured requires per-implementation verification against `tt_metal::Program::dispatch` and the specific `CreateSemaphore` backend before treating it as a guaranteed invariant.

## First Observation About `ttnn.all_reduce`

`ttnn.all_reduce` passes `std::nullopt` for all three semaphore argument groups
(`barrier_semaphores`, `rs_global_semaphores`, `ag_global_semaphores`) when it calls
`ttnn::experimental::all_reduce_async`. No `GlobalSemaphore` object is created or
consumed at the caller level, so the caller holds nothing to reset between replays.
Whether the downstream path uses internal global semaphores — and whether those are
correctly managed — is determined in [Chapter 2](../ch2_all_reduce_internals/index.md)
and examined in full in [Ch3 Q2](../ch3_verdict/q2_semaphore_state.md).

---

**Next:** [Chapter 2 — ttnn.all_reduce Internal Architecture](../ch2_all_reduce_internals/index.md)
