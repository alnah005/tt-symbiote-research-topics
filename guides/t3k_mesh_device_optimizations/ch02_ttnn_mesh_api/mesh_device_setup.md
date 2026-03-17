# Creating and Configuring a MeshDevice for T3K

> **Quick Reference — TTNN API Symbols Introduced in This File**
>
> | Symbol | Module | Description |
> |---|---|---|
> | `MeshDevice` | `ttnn` | Python class representing a multi-chip device cluster; encapsulates initialization, device ordering, and resource lifetime for all chips in the mesh. |
> | `ttnn.open_mesh_device` | `ttnn` | Factory function that constructs and initializes a `MeshDevice` from a mesh shape and ordered device ID list. |
> | `ttnn.close_mesh_device` | `ttnn` | Teardown function that releases all firmware state, worker threads, and memory contexts associated with a `MeshDevice`. |
> | `MeshShape` | `ttnn` | Named tuple (or integer pair) specifying `(num_rows, num_cols)` for the logical mesh grid. |
> | `DispatchCoreType` | `ttnn` | Enum controlling which on-chip core type is assigned to host-to-device command dispatch. |

The `MeshDevice` is the foundation of all multi-chip TTNN operations on T3K. Every tensor distribution, collective call, and kernel dispatch in this chapter targets a `MeshDevice` instance. This file covers how to construct one correctly, what each constructor parameter controls, how the initialization sequence proceeds internally, and how to tear it down cleanly when done.

---

## The MeshDevice Constructor

`MeshDevice` instances are created via `ttnn.open_mesh_device`. The function signature is:

```python
import ttnn

mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 8),
    device_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    l1_small_size=32768,
    trace_region_size=1048576,
    num_command_queues=1,
    dispatch_core_type=ttnn.DispatchCoreType.WORKER,
)
```

The parameters are discussed individually in the sections below.

### `mesh_shape`

`mesh_shape` is a `ttnn.MeshShape` (or a plain two-element tuple) specifying the logical mesh dimensions as `(num_rows, num_cols)`. For a single T3K board, the correct value is always `ttnn.MeshShape(1, 8)`: one row of eight columns.

This shape defines the coordinate space that subsequent operations use to address devices. A tensor shard placed on logical coordinate `(row=0, col=3)` lands on whichever physical device ID occupies position 3 in the ordered list you provide to `device_ids`. The shape must be consistent with the number of devices in `device_ids`: `num_rows × num_cols` must equal `len(device_ids)`. For a single T3K board, that product must equal 8.

Specifying an inconsistent shape — for example, `ttnn.MeshShape(2, 4)` with eight device IDs — is not necessarily an error at construction time, but it changes the collective routing semantics. A `(2, 4)` shape would create a two-row, four-column mesh, and `cluster_axis=1` for T3K (see `collective_primitives.md` for details) would then address only a four-column axis rather than the expected eight. On a single T3K board this does not correspond to any physical grouping and will produce incorrect collective routing. Always use `ttnn.MeshShape(1, 8)` for single-board T3K deployments.

### `device_ids`

`device_ids` is an ordered list of integer device IDs, one per device in the mesh, assigned to mesh positions in row-major order. For `mesh_shape=(1, 8)`, the first element of the list maps to `(row=0, col=0)`, the second to `(row=0, col=1)`, and so on through `(row=0, col=7)`.

For a standard T3K deployment, the conventional ordering is `[0, 1, 2, 3, 4, 5, 6, 7]`. This assigns logical col=N to the physical device with device ID N, which means logical column distance equals physical Ethernet hop distance (as established in Chapter 1). This alignment simplifies expert placement analysis in Chapter 5.

You may provide device IDs in a different order to reverse or permute the logical-to-physical mapping. For example, `[7, 6, 5, 4, 3, 2, 1, 0]` would assign col=0 to physical device 7 and col=7 to physical device 0. This reverses the linear traversal direction. There is rarely a reason to do this on T3K, and doing so without updating all downstream expert placement assumptions will produce correct but sub-optimal collective routing. The standard ordering `[0, 1, 2, 3, 4, 5, 6, 7]` should be used unless you have an explicit reason to deviate.

The `device_ids` list must contain only valid device IDs enumerated by the driver. Passing an ID not present on the system — for example, `device_id=8` on a single T3K board — raises an error during device enumeration. Passing a duplicate ID (the same device ID twice) is also an error.

### `l1_small_size`

`l1_small_size` specifies the size in bytes of the per-device L1 buffer reserved for small fast allocations (for example, scalar metadata, small routing tables, and operation-level scratchpad structures). This parameter does not control the total L1 available for tensor storage; it carves out a dedicated region for the TTNN runtime's own bookkeeping. The default value used in the example above (32,768 bytes = 32 KiB) is appropriate for most workloads. Increasing it beyond what the runtime needs wastes L1; decreasing it below the runtime's minimum requirement will cause allocation failures during initialization or operation dispatch.

### `trace_region_size`

`trace_region_size` specifies the size in bytes of the per-device DRAM region reserved for the TTNN trace capture mechanism. TTNN's trace mode records a sequence of operations at dispatch time and replays them at reduced host overhead on subsequent invocations — useful for decode loops where the same operation sequence runs thousands of times. The default value (1,048,576 bytes = 1 MiB) is sufficient for most MoE layer sequences. If you are not using trace mode, you may set this to 0 to recover the DRAM space, at the cost of disabling trace-mode replay.

### `num_command_queues`

`num_command_queues` controls how many independent host-to-device command queues are allocated per device. Each queue is an independent dispatch channel: operations submitted to queue 0 and operations submitted to queue 1 can execute concurrently on a device, subject to hardware resource availability. The default value of 1 provides a single serial dispatch channel and is correct for straightforward sequential workloads. Setting `num_command_queues=2` enables two-queue pipelining, which is useful for overlapping compute and communication (for example, submitting the expert matmul to queue 0 while the combine all-to-all runs on queue 1). Two-queue operation is discussed further in `collective_primitives.md` under synchronization semantics.

### `dispatch_core_type`

`dispatch_core_type` selects which type of on-chip core handles host-to-device command dispatch on each Wormhole chip. The default `ttnn.DispatchCoreType.WORKER` assigns dispatch to a regular Tensix worker core. Alternative core types are available for specific firmware configurations. For T3K MoE inference workloads, `ttnn.DispatchCoreType.WORKER` is the appropriate choice and should be used unless a specific firmware requirement dictates otherwise.

---

## Device Ordering Conventions

The ordering of device IDs in the `device_ids` list is the single most consequential parameter for multi-device correctness. Three properties follow from this ordering:

**Logical coordinate assignment.** Position N in `device_ids` maps to logical mesh coordinate `(row=0, col=N)` for a `(1, 8)` mesh. All tensor placement, collective routing, and expert assignment operations address devices by their logical coordinates. If you accidentally swap two device IDs in this list, those two devices will silently exchange logical positions — their local tensor data will be correct but will be routed as if they were the other device, producing incorrect results with no error message.

**Linear traversal direction.** The linear-chain all-to-all algorithm traverses the devices in the order of their logical column indices: col=0, col=1, ..., col=7. With the standard `[0, 1, 2, 3, 4, 5, 6, 7]` ordering, the algorithm traverses physical devices in the same left-to-right order as the PCB Ethernet wiring. This alignment means linear traversal never creates a logical forward step that corresponds to a physical backward step (which would require data to transit through intermediate chips in the "wrong" direction on the board). Maintaining this alignment is the reason the conventional ordering matches the physical device numbering.

**Expert-to-device locality.** Expert placement strategies (Chapter 5) assign experts to devices by logical column. With the standard ordering, `col=0` means physical device 0, so "place expert 0 on col=0" means placing it on the physical chip at the left end of the board. If you use a non-standard device ID ordering, you must translate between logical col indices and physical chip positions whenever discussing expert placement and interconnect locality.

---

## Initialization Sequence

When you call `ttnn.open_mesh_device`, the following sequence occurs internally. You do not need to call these steps manually; they happen automatically. Understanding the sequence helps diagnose initialization failures.

**Step 1: Device enumeration and validation.** TTNN calls into the tt-metal driver to enumerate all devices detected by the host PCIe driver. Each requested device ID is checked against the enumerated list. If any ID is missing — because the device is powered off, the driver has not loaded, or a PCIe link is down — initialization fails with an error identifying the missing ID.

**Step 2: Device opening.** Each device in `device_ids` is opened sequentially. Opening a device allocates the host-side data structures for the device's command queues, memory allocator, and event tracking. The device's firmware is verified to be in a known-good state; if the device was previously used by another process and left in a dirty state (stale kernel programs loaded, buffers allocated), the open sequence attempts to reset it to a clean state.

**Step 3: Mesh firmware programming.** After all devices are open individually, TTNN programs the mesh routing fabric. This involves writing the Ethernet link routing tables into each device's firmware, establishing which logical device-pair connections correspond to which physical Ethernet links, and verifying that all required links are up and passing traffic. This step is where link failures and misconfigured mesh topologies are detected.

**Step 4: Worker thread creation.** TTNN creates one or more host-side worker threads per device to handle command serialization and dispatch. These threads run continuously while the `MeshDevice` is alive and are responsible for translating Python API calls into device command packets and submitting them to the hardware command queues. Increasing `num_command_queues` adds additional threads per device.

**Step 5: Memory context initialization.** Per-device L1 and DRAM allocators are initialized, the `l1_small_size` region is reserved, and the `trace_region_size` DRAM region is reserved. Any L1 allocation of `l1_small_size` or larger that fails here (because the device has insufficient L1 after firmware use) will raise an exception.

The entire sequence for an eight-device T3K mesh typically completes in a few seconds under normal conditions. If any step fails, `ttnn.open_mesh_device` raises an exception with a message identifying the failing step and the specific device involved.

---

## Teardown and Resource Cleanup

Releasing a `MeshDevice` requires explicit cleanup. TTNN does not finalize devices through garbage collection alone; firmware state and device resources must be explicitly freed to leave the devices in a clean state for subsequent processes or re-initialization.

The correct teardown pattern is:

```python
ttnn.close_mesh_device(mesh_device)
```

This single call:
1. Waits for all in-flight operations on all command queues to complete.
2. Deallocates all tensors still resident on any device in the mesh (with a warning if any tensors remain allocated at teardown time, since this may indicate a resource leak).
3. Releases worker threads and command queue resources.
4. Resets device firmware to a clean state.
5. Closes the PCIe driver handles for each device.

You should call `ttnn.close_mesh_device` in a `finally` block or use the `MeshDevice` as a context manager to ensure teardown occurs even if an exception is raised during inference:

```python
import ttnn

mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 8),
    device_ids=[0, 1, 2, 3, 4, 5, 6, 7],
)
try:
    # ... inference operations ...
finally:
    ttnn.close_mesh_device(mesh_device)
```

Alternatively, if your version of TTNN supports the context manager protocol on `MeshDevice`:

```python
with ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 8),
    device_ids=[0, 1, 2, 3, 4, 5, 6, 7],
) as mesh_device:
    # ... inference operations ...
# mesh_device is automatically closed here
```

### Partial Teardown Is Not Supported

You cannot close a single device from a `MeshDevice` while leaving others open. The `MeshDevice` represents the entire mesh as an atomic unit; partial teardown would leave the mesh routing tables in an inconsistent state. If you need to change the set of active devices, close the full `MeshDevice` and open a new one.

### Tensor Lifetime and Teardown Order

All tensors allocated on the mesh must be deleted (or allowed to go out of scope) before `ttnn.close_mesh_device` is called. Calling `close_mesh_device` while tensors still reference device memory produces a warning but proceeds; the tensors are invalidated and accessing them after teardown produces undefined behavior. The safe pattern is to delete or let go out of scope all mesh tensors before the `finally` block executes.

---

## Common Pitfalls

The following errors occur frequently during `MeshDevice` setup. Each is described with its cause and the correct fix.

### Incorrect Mesh Shape

**Symptom:** Operations produce unexpected output shapes, or collective operations fail with a shape mismatch error.

**Cause:** The mesh shape passed to `ttnn.open_mesh_device` does not match the shape assumed by downstream tensor distribution or collective calls. For example, passing `ttnn.MeshShape(2, 4)` when all downstream code assumes `ttnn.MeshShape(1, 8)` will cause any call that uses `cluster_axis=1` for T3K (see `collective_primitives.md` for details) to address a four-column axis rather than the expected eight-column axis, halving the number of participating devices.

**Fix:** Always use `ttnn.MeshShape(1, 8)` for single-board T3K deployments. Add an assertion immediately after `open_mesh_device` to verify the shape:

```python
assert mesh_device.shape == (1, 8), (
    f"Expected mesh shape (1, 8) for T3K; got {mesh_device.shape}"
)
```

### Mismatched Device IDs

**Symptom:** `ttnn.open_mesh_device` raises an error during device enumeration, or devices appear to be assigned to wrong logical positions.

**Cause 1:** A device ID in the list is not present on the system. This can happen if the system has fewer than eight devices (hardware fault, partial board), if the driver has not enumerated all devices, or if a device ID was typed incorrectly.

**Cause 2:** The list has the correct IDs but in the wrong order, silently swapping logical positions for two physical chips.

**Fix for Cause 1:** Before calling `open_mesh_device`, verify that all eight device IDs are enumerated by the driver:

```python
available = ttnn.get_device_ids()
required = [0, 1, 2, 3, 4, 5, 6, 7]
missing = [d for d in required if d not in available]
if missing:
    raise RuntimeError(f"Missing T3K device IDs: {missing}")
```

**Fix for Cause 2:** Use the conventional ordering `[0, 1, 2, 3, 4, 5, 6, 7]` unless you have an explicit requirement for a different ordering, and document any deviation.

### Stale Device State from a Previous Process

**Symptom:** `ttnn.open_mesh_device` succeeds, but the first operation dispatched to the mesh produces garbage output, hangs, or raises an error about unexpected kernel state.

**Cause:** A previous Python process (or a crashed process) left firmware state, allocated buffers, or kernel programs loaded on one or more devices. TTNN's open sequence attempts to reset devices, but in some firmware versions the reset is not fully effective if the previous process was terminated abnormally (for example, with SIGKILL).

**Fix:** Perform a hardware reset of the T3K board through the driver reset interface before re-initializing:

```bash
tt-smi --reset
```

After the driver reset completes, re-run your initialization code. If the problem persists, a full system reboot resolves all residual firmware state.

### Insufficient L1 for `l1_small_size`

**Symptom:** `ttnn.open_mesh_device` raises an L1 allocation failure during Step 5 of the initialization sequence.

**Cause:** The requested `l1_small_size` exceeds the L1 available after firmware use, or the device is in a state where some L1 has been permanently allocated by a previous partially-initialized mesh.

**Fix:** Reduce `l1_small_size` to the minimum needed for your workload (the default of 32 KiB is conservative), or perform a driver reset to clear residual allocations before re-opening.

### Forgetting to Close Before Re-Opening

**Symptom:** The second call to `ttnn.open_mesh_device` in the same process fails or produces stale state.

**Cause:** The first `MeshDevice` was not closed before attempting to open a second one targeting the same device IDs. TTNN tracks open device handles; attempting to open a device that is already open in the same process raises an error or produces undefined state.

**Fix:** Always close the existing `MeshDevice` before opening a new one targeting the same devices:

```python
ttnn.close_mesh_device(old_mesh_device)
new_mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 8),
    device_ids=[0, 1, 2, 3, 4, 5, 6, 7],
)
```
