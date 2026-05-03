# 5.2 CCL Collective Operation Hangs

[Previous: 5.1 Ethernet and Fabric Fundamentals](./01_ethernet_and_fabric_fundamentals.md) | [Next: 5.3 Topology and Mesh Configuration Hangs](./03_topology_and_mesh_configuration_hangs.md)

---

CCL (Collective Communication Library) operations -- `all_gather`, `reduce_scatter`, `all_reduce`, `all_broadcast`, `reduce_to_root`, and `all_to_all_combine` -- are the primary user-facing multi-chip primitives in TTNN. Each operation coordinates computation across every device in a mesh using device-side semaphores, fabric sockets, and the EDM router infrastructure documented in [Section 5.1](./01_ethernet_and_fabric_fundamentals.md). A hang in any CCL operation typically means every device in the participating mesh is stuck, because collective semantics require all participants to complete before any can proceed.

**Prerequisites:** See [Chapter 5 Index prerequisites](./index.md#prerequisites). Also assumes familiarity with the EDM flow control protocol (Section 5.1) and the worker-to-EDM connection protocol (Scenario 5.1.12).

**Reference files:**
- `ttnn/cpp/ttnn/operations/ccl/all_gather/all_gather.cpp` -- `ExecuteAllGather::invoke()`
- `ttnn/cpp/ttnn/operations/ccl/reduce_scatter/reduce_scatter.cpp` -- `ExecuteReduceScatter::invoke()`
- `ttnn/cpp/ttnn/operations/ccl/all_reduce/all_reduce.cpp`, `all_broadcast/all_broadcast.cpp`, `reduce_to_root/reduce_to_root.cpp`, `all_to_all_combine/all_to_all_combine.cpp`
- `ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp` -- Termination master protocol, runtime argument generation
- `ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp` -- Worker synchronization utilities
- `ttnn/core/distributed/fabric_socket.cpp` -- `FabricSocket` send/recv implementation
- `tt_metal/api/tt-metalium/experimental/sockets/mesh_socket.hpp` -- `MeshSocket`, `SocketConfig`, `SocketConnection`
- `tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_deadlock_stability_bh_6U_galaxy.yaml` -- Deadlock stability test configuration

## Part 1: CCL Architecture Background

### Topology Resolution and Multi-Axis Decomposition

CCL operations use either **Ring** or **Linear** topology, determined by the `tt::tt_fabric::Topology` enum. For mesh topologies that are not a simple line (1xN or Nx1), operations are decomposed into sequential 1D operations along each axis. Critically, `all_gather` iterates in reverse order (highest dimension first) while `reduce_scatter` iterates forward:

```cpp
// From all_gather.cpp -- reverse iteration
for (auto it = mesh_view.rbegin(); it != mesh_view.rend(); ++it) {
    tensor = ttnn::all_gather(tensor, dim, axis, ...);
}

// From reduce_scatter.cpp -- forward iteration
for (size_t i = 0; i < mesh_shape.dims(); ++i) {
    tensor = ttnn::reduce_scatter(tensor, dim, i, ...);
}
```

### Termination Master Protocol

CCL operations use a "termination master" pattern for coordinated shutdown. One worker on one device is designated the termination master via runtime arguments from `ccl_common.cpp`:

```cpp
worker_rt_args.push_back(is_termination_master);      // arg offset 1
// ...
worker_rt_args.push_back(termination_master_semaphore_id.value_or(...));
worker_rt_args.push_back(termination_master_virtual_core.x);  // arg offset 15
worker_rt_args.push_back(termination_master_virtual_core.y);  // arg offset 16
```

Non-master workers signal completion by incrementing the master's semaphore. The master waits until the count reaches the expected number, then broadcasts termination to all workers.

---

## Part 2: Hang Scenarios

### 5.2.1 Partial Participation in Collective Operation

**Symptom:** A CCL operation (e.g., `all_gather`) hangs and all devices appear to be waiting at semaphore waits. The watcher shows worker cores stuck at `NSW` (noc_semaphore_wait) on every device in the mesh. One device may show different state -- either the kernel never launched, or the kernel completed abnormally.

**Root Cause:** Every CCL collective requires all devices in the participating group to execute the operation. If one device skips the operation (due to a host-side exception, a conditional branch that bypasses the CCL call on one rank, or a previous operation failure), the remaining devices wait forever for data or semaphore signals from the absent participant.

When the mesh is not a line topology, the code decomposes the operation into per-axis collectives. If a device participates in one axis but not the other, the second axis hangs:

```cpp
if (!mesh_shape.is_line_topology()) {
    for (auto it = mesh_view.rbegin(); it != mesh_view.rend(); ++it) {
        tensor = ttnn::all_gather(tensor, dim, axis, ...);
    }
}
```

**Diagnosis Steps:**
1. Check the watcher on every device in the mesh. Identify any device where the CCL worker kernels are not running or have already completed.
2. Check host-side logs for exceptions or errors on the non-participating device.
3. Verify that the CCL operation was launched on all devices by checking the program launch sequence.
4. Check the `cluster_axis` parameter: if it is `std::nullopt` for a non-line topology, the code produces nested CCL calls.

**Fix:**
```cpp
// BUGGY: CCL call is conditional on one rank, other ranks always execute
if (device_id != SPECIAL_DEVICE) {
    auto output = ttnn::all_gather(input, dim);  // Not called on SPECIAL_DEVICE
}

// CORRECTED: All devices must participate in every collective
auto output = ttnn::all_gather(input, dim);  // Always called on all devices
if (device_id == SPECIAL_DEVICE) {
    // Do special processing AFTER the collective completes
}
```

**Prevention:**
- Treat every CCL call as a barrier: all ranks must execute the same sequence of collectives in the same order.
- Use the MeshDevice abstraction rather than managing individual devices, as it provides coordinated error handling.
- Add assertion checks that verify all devices are in the same program state before launching a collective.

---

### 5.2.2 Tensor Dimension Mismatch Across Ranks

**Symptom:** A CCL operation hangs partway through data transfer. Some devices have received partial data, others are waiting. The hang is deterministic and reproduces with the same input shapes.

**Root Cause:** CCL operations compute the number of packets, chunks, and synchronization points based on the input tensor dimensions. If one device has a tensor with different dimensions (due to asymmetric preprocessing, padding differences, or a sharding bug), the devices disagree on how much data to send. For `reduce_scatter`, the `normalized_dim` determines the scatter dimension:

```cpp
uint32_t normalized_dim = input_tensor.logical_shape().get_normalized_index(dim);
```

If device A has shape `[1, 8, 1024, 1024]` and device B has shape `[1, 8, 1024, 512]`, and the scatter dimension is 3, device A expects to scatter 1024/N elements per rank while device B expects 512/N. The device with fewer elements completes and signals done, while the device with more elements waits for additional data that never arrives.

**Diagnosis Steps:**
1. Before the CCL call, log the input tensor shape on each device.
2. Check `normalized_dim` and verify it resolves to the same dimension index on all devices.
3. Compute the expected per-device contribution: `tensor.shape()[dim] / num_devices`. This must be identical on all ranks.
4. Check the `topology_` parameter: `Linear` vs `Ring` affects the number of expected data transfers.

**Fix:**
```python
# BUGGY: Tensor shapes differ across ranks due to asymmetric padding
tensor = pad(input, target_shape)  # target_shape varies per device!
output = ttnn.all_gather(tensor, dim=3)

# CORRECTED: Ensure uniform shapes before collective
target_shape = get_global_max_shape(input)  # Same on all devices
tensor = pad(input, target_shape)
output = ttnn.all_gather(tensor, dim=3)
```

**Prevention:**
- Add shape assertions before every CCL call: verify that the tensor dimension along `dim` is divisible by the number of participants and is identical on all ranks.
- Use `TT_FATAL` assertions in the CCL operation code to validate shape consistency.

---

### 5.2.3 Topology Mismatch Between Operation and Fabric

**Symptom:** A CCL operation hangs on a multi-chip system where the physical topology does not match the requested logical topology. For example, requesting a `Ring` topology on a physical `Linear` (non-wrapped) connection.

**Root Cause:** A `Ring` topology assumes that the last device connects back to the first, forming a cycle. If the physical Ethernet connections do not form a ring, the CCL operation attempts to send data through a non-existent link, and the EDM sender spins forever. The `get_usable_topology()` function attempts to select an appropriate topology, but if the user explicitly specifies `Topology::Ring` on hardware that only supports `Topology::Linear`, the mismatch is not caught.

**Diagnosis Steps:**
1. Check the `topology` parameter passed to the CCL operation.
2. Verify the physical connectivity: does device N-1 have an Ethernet connection to device 0?
3. Check `SenderReceiverConfig` for edge devices. In a correct Linear topology, `sender_device_id` should be `std::nullopt` for the first device.

**Fix:**
```python
# BUGGY: Requesting Ring on a Linear-only fabric
output = ttnn.all_gather(tensor, dim=0, topology=ttnn.Topology.Ring)

# CORRECTED: Use auto-detect or the topology that matches hardware
output = ttnn.all_gather(tensor, dim=0)  # topology=None -> auto-detect
```

**Prevention:**
- Prefer `topology=None` to let the framework auto-detect the correct topology.
- Use `is_ring_or_torus()` and `is_2D_topology()` utility functions to verify topology compatibility.

---

### 5.2.4 Termination Master Semaphore Coordination Failure

**Symptom:** All CCL worker kernels on all devices have completed their data transfer work, but the kernels do not terminate. The watcher shows workers stuck at a final semaphore wait (`NSMW` waypoint) after all data has been processed.

**Root Cause:** The termination protocol requires each non-master device to increment a specific semaphore on the termination master's core exactly once. Hang causes include:
- **Wrong `termination_master_virtual_core` coordinates**: the signal goes to the wrong core, the master never receives enough signals.
- **Wrong worker count**: the master expects N signals but only N-1 workers signal.
- **Semaphore ID mismatch**: `termination_master_semaphore_id` does not match the actual semaphore allocated on the master core.
- **NOC coordinate system mismatch**: coordinates in NOC 1 when the sender uses NOC 0 (see also Scenario 5.1.12).

**Diagnosis Steps:**
1. Identify the termination master core from the runtime arguments (arg offset 15/16 contain the NOC coordinates).
2. Read the termination semaphore value on that core. Compare it to the expected worker count.
3. If the semaphore is one short, identify which worker did not signal.
4. Verify the semaphore ID matches between the master's allocation and the workers' runtime arguments.

**Fix:**
```cpp
// BUGGY: Termination master coordinates are stale from a previous operation
worker_rt_args.push_back(cached_master_core.x);  // Stale!
worker_rt_args.push_back(cached_master_core.y);  // Stale!

// CORRECTED: Always use freshly computed coordinates
auto master_core = compute_termination_master_core(mesh_device, sub_device_id);
worker_rt_args.push_back(master_core.x);
worker_rt_args.push_back(master_core.y);
```

**Prevention:**
- Use the CCL infrastructure functions (`setup_termination_signal_runtime_args()`) that compute termination master parameters.
- Add a timeout to the termination protocol so a missing signal produces a diagnosable error.

---

### 5.2.5 Fabric Socket Rank Mismatch

**Symptom:** A `FabricSocket::send()` or `FabricSocket::recv()` call hangs. The assertion `check_if_send_socket()` or `check_if_recv_socket()` would fail if assertions were enabled, but in release builds the operation proceeds incorrectly and hangs.

**Root Cause:** The `FabricSocket` validates sender/receiver roles:

```cpp
bool check_if_send_socket(const MeshSocket& mesh_socket) {
    const auto& socket_config = mesh_socket.get_config();
    auto expected_sender_rank = socket_config.distributed_context->rank();
    return (socket_config.sender_rank == expected_sender_rank);
}
```

If `sender_rank` and `receiver_rank` are swapped, data flows in the wrong direction. The sender waits for an incoming tensor that was never sent.

**Diagnosis Steps:**
1. Verify the `sender_rank` and `receiver_rank` in the `SocketConfig` match the actual process ranks.
2. Check `distributed_context->rank()` on each host.
3. Confirm that the process calling `send()` has `rank == sender_rank`.

**Fix:**
```cpp
// BUGGY: Ranks are swapped in socket creation
auto socket = FabricSocket::create(mesh_device,
    /*sender_rank=*/receiver_process_rank,   // Wrong!
    /*receiver_rank=*/sender_process_rank,   // Wrong!
    config);

// CORRECTED: Sender rank is the process that will call send()
auto socket = FabricSocket::create(mesh_device,
    /*sender_rank=*/sender_process_rank,
    /*receiver_rank=*/receiver_process_rank,
    config);
```

**Prevention:**
- Always construct sockets using `MeshSocket::create_socket_pair()` which returns matched sender/receiver pairs.
- Add runtime validation that raises an error (not just an assertion) when a socket operation is called from the wrong rank.

---

### 5.2.6 EriscDataMover Termination Mode Mismatch

**Symptom:** A CCL operation completes on some devices but the ERISC data mover (EDM) on one or more devices does not terminate. The worker kernels have finished and released their EDM connections, but the EDM continues running, holding Ethernet resources and blocking the next operation.

**Root Cause:** The `EriscDataMoverTerminationMode` enum defines how EDM kernels know when to stop:

```cpp
// ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp
enum EriscDataMoverTerminationMode : uint32_t {
    MESSAGE_COUNT_REACHED = 0,  // EDM terminates after processing N messages
    WORKER_INITIATED = 1        // Workers explicitly signal EDM to terminate
};
```

If the CCL operation is configured for `WORKER_INITIATED` but the worker does not send the termination signal (e.g., because it exited early due to zero-size tensor handling), the EDM never receives its termination signal and runs forever. Conversely, if configured for `MESSAGE_COUNT_REACHED` but the expected message count does not match the actual number of packets, the EDM either terminates too early (losing packets) or too late (hanging).

**Diagnosis Steps:**
1. Check the EDM's termination mode via the compile-time arguments.
2. For `WORKER_INITIATED`: verify the worker sends the termination signal before exiting.
3. For `MESSAGE_COUNT_REACHED`: compare `messages_processed` to `expected_message_count`.

**Fix:**
```cpp
// BUGGY: Worker exits without signaling EDM termination
void worker_kernel() {
    edm_connection.open();
    if (tensor_size == 0) {
        return;  // BUG: EDM never gets termination signal
    }
    // ... process data ...
    edm_connection.signal_termination();
    edm_connection.close();
}

// CORRECTED: Always signal termination before exit
void worker_kernel() {
    edm_connection.open();
    if (tensor_size == 0) {
        edm_connection.signal_termination();  // Signal even for zero-size case
        edm_connection.close();
        return;
    }
    // ... process data ...
    edm_connection.signal_termination();
    edm_connection.close();
}
```

**Prevention:**
- Prefer `WORKER_INITIATED` termination mode, as it does not depend on packet count predictions.
- Ensure every code path -- including early exits and error paths -- signals EDM termination.

---

### 5.2.7 Bidirectional Socket Handshake Ordering Violation

**Symptom:** Two processes attempting to create bidirectional fabric sockets deadlock during socket construction. The `MeshSocket` constructor on both sides blocks waiting for the peer's handshake response. The deadlock occurs in socket creation, not during send/recv operations.

**Root Cause:** The `BidirectionalFabricSocket::create()` function creates two underlying `MeshSocket` objects: one for sending and one for receiving. The creation order matters because `MeshSocket` construction involves a handshake with the peer. If both processes create their send sockets first, both block waiting for the peer to create the matching receive socket:

```cpp
// From bidirectional_fabric_socket.cpp:
// Correct ordering: lower-rank process creates send socket first
if (sender_socket_config.sender_rank < recv_socket_config.sender_rank) {
    auto send_socket = MeshSocket(mesh_device, sender_socket_config);
    auto recv_socket = MeshSocket(mesh_device, recv_socket_config);
    return std::make_unique<BidirectionalFabricSocket>(send_socket, recv_socket);
}
// Higher-rank process creates recv socket first
auto recv_socket = MeshSocket(mesh_device, recv_socket_config);
auto send_socket = MeshSocket(mesh_device, sender_socket_config);
```

**Diagnosis Steps:**
1. Check if the hang occurs during socket creation (constructor), not during send/recv.
2. Verify the creation order on each process. The lower-rank process must create its send socket first.
3. Check that `sender_rank != receiver_rank`. The `BidirectionalFabricSocket` throws if ranks are equal.

**Fix:**
```cpp
// BUGGY: Both processes create send socket first -- deadlock
// Process A (rank 0):
auto send = MeshSocket(mesh_device, send_config_A);  // Blocks
auto recv = MeshSocket(mesh_device, recv_config_A);   // Never reached

// CORRECTED: Use BidirectionalFabricSocket::create() which handles ordering
auto bidir_socket = BidirectionalFabricSocket::create(
    mesh_device, peer_rank, socket_config);
```

**Prevention:**
- Always use `BidirectionalFabricSocket::create()` for bidirectional communication.
- For unidirectional sockets, establish a convention where the sender always creates first and the receiver creates second.

---

### 5.2.8 CCL Operation Number-of-Links Mismatch

**Symptom:** A CCL operation hangs on configurations where not all devices are MMIO-capable. The operation works on T3K (all devices MMIO-accessible) but hangs on Galaxy (where only one device per column is MMIO-capable).

**Root Cause:** The number of Ethernet links available for CCL data transfer depends on how many links the dispatch system has already consumed. The dispatch pipeline reserves Ethernet links for the tunnel between MMIO and remote devices. The code comment in `reduce_scatter.cpp` explicitly notes this:

```cpp
// TODO: until #27196 is resolved, the fabric API does not subtract out the
// one link correctly for dispatch used when not all devices are mmio capable.
```

If the CCL operation attempts to use a link reserved for dispatch, the EDM channels collide and the fabric router receives interleaved dispatch and CCL packets, causing corruption and hangs.

**Diagnosis Steps:**
1. Check how many Ethernet links exist between adjacent devices.
2. Check how many are consumed by the dispatch pipeline (inspect `DispatchKernelNode` topology for `FABRIC_MUX` nodes).
3. Verify that `num_links_` does not exceed the remaining available links.

**Fix:**
```python
# BUGGY: Using all available links on Galaxy (some reserved for dispatch)
output = ttnn.all_gather(tensor, dim=0, num_links=4)

# CORRECTED: Let the framework compute the correct value
output = ttnn.all_gather(tensor, dim=0)  # num_links=None
```

**Prevention:**
- Prefer `num_links=None` to let `get_num_links()` compute the correct value.
- Be especially careful on Galaxy configurations where the MMIO topology differs from T3K.

---

### 5.2.9 Composite CCL Operation Deadlock on 2D Mesh

**Symptom:** An `all_gather` or `reduce_scatter` on a 2D mesh (e.g., 2x4) hangs during the second phase. The first dimension's collective completes, but the second dimension's collective never starts or hangs partway.

**Root Cause:** When the mesh is not a line topology, CCL operations decompose into multiple 1D collectives along each mesh dimension. Note the opposite iteration order: `all_gather` uses reverse order, `reduce_scatter` uses forward order. The decomposition assumes each dimension's collective operates on independent subsets of devices. If the intermediate tensor from the first dimension is not properly replicated, or if `use_composite_all_gather()` selects an optimized path with different synchronization requirements, the second phase hangs.

Additionally, for `all_reduce` (implemented as `reduce_scatter` + `all_gather`), if the two phases are not fused and there is no explicit synchronization between them, one device may start `all_gather` before all devices finish `reduce_scatter`.

**Diagnosis Steps:**
1. Check if the operation is using the composite path or the iterative decomposition path.
2. Determine which dimension's collective is hanging (the first or second).
3. Check if all devices have the correct intermediate tensor after the first collective.
4. Verify that `cluster_axis` is incremented correctly across loop iterations.

**Fix:**
```python
# BUGGY: Manual decomposition without synchronization
intermediate = ttnn.reduce_scatter(input, dim=0, cluster_axis=0)
output = ttnn.all_gather(intermediate, dim=0, cluster_axis=0)  # May hang

# CORRECTED: Use built-in all_reduce which handles synchronization
output = ttnn.all_reduce(input, dim=0)
# Or add explicit synchronization between phases
intermediate = ttnn.reduce_scatter(input, dim=0, cluster_axis=0)
mesh_device.synchronize()
output = ttnn.all_gather(intermediate, dim=0, cluster_axis=0)
```

**Prevention:**
- Use the built-in composite operation helpers (`composite_all_gather`, `composite_reduce_scatter`, `composite_all_reduce`).
- Never manually decompose collective operations without explicit mesh-wide synchronization between phases.

---

### 5.2.10 Socket Memory Configuration Mismatch

**Symptom:** A `MeshSocket`-based operation hangs during initial socket setup. The `MeshSocket` constructor does not return, or the first `send()`/`recv()` call blocks indefinitely.

**Root Cause:** The `SocketConfig` specifies memory configuration for data buffers and config buffers. If the `SocketConnection` list does not match the actual MeshDevice layout (e.g., specifying connections to device coordinates that do not exist), the socket allocation fails silently. If a receiver socket is created without sufficient L1 memory for the data buffer, the allocation may return an invalid address, causing secondary hangs (see Chapter 3).

**Diagnosis Steps:**
1. Check the `SocketConfig` for valid `socket_connection_config` entries.
2. Verify that sender/receiver mesh coordinates exist in the MeshDevice.
3. Check L1 allocation status on the receiver devices.
4. Inspect the `MeshBuffer` pointers returned by `get_data_buffer()` and `get_config_buffer()`.

**Fix:**
```cpp
// BUGGY: Socket connection refers to non-existent device coordinate
SocketConnection conn{
    .sender_coord = MeshCoordinate(0, 0),
    .receiver_coord = MeshCoordinate(0, 5)  // Only 4 devices in mesh!
};

// CORRECTED: Validate coordinates against mesh shape
auto mesh_shape = mesh_device->shape();
TT_FATAL(receiver_coord < mesh_shape, "Receiver coordinate out of mesh bounds");
```

**Prevention:**
- Always validate socket connection coordinates against the mesh shape.
- Use `MeshSocket::create_socket_pair()` (see [Scenario 5.2.5](#525-fabric-socket-rank-mismatch)).
- Ensure sufficient L1 memory is available before allocating socket buffers.

---

### 5.2.11 Circular Wait in Ring/Torus CCL Operations

**Symptom:** A CCL operation on a Ring or Torus topology hangs with all devices stuck in a symmetric state. Each device is waiting to receive data from its predecessor while trying to send to its successor. No device can make progress.

**Root Cause:** Ring-based CCL operations pass data around a ring of devices. Each device simultaneously receives from its predecessor and sends to its successor. If the per-device buffer is not large enough to hold incoming data while outgoing data is still in flight, a circular dependency forms. The `chunks_per_sync` parameter controls how many chunks each device processes before synchronizing -- if it is too large, devices try to send too much at once, overwhelming receive buffers.

**Diagnosis Steps:**
1. Check all devices in the ring -- if they all show the same state (trying to send, receiver full), this is a ring deadlock.
2. Examine `chunks_per_sync` and `num_buffers_per_channel` parameters.
3. Check the fabric EDM flow control counters per 5.1.5 to confirm receiver buffers are full.

**Fix:**
```python
# BUGGY: chunks_per_sync is too large, causing ring deadlock
output = ttnn.reduce_scatter(tensor, dim=0, chunks_per_sync=1024)

# CORRECTED: Use framework default or smaller value
output = ttnn.reduce_scatter(tensor, dim=0, chunks_per_sync=4)
```

**Prevention:**
- Use the framework's default `chunks_per_sync` value.
- On Ring/Torus topologies, ensure `enable_deadlock_avoidance` is active at the fabric layer.
- Test with the fabric deadlock stability tests.

---

### 5.2.12 Deadlock Stability Test Failures Indicating Latent Bugs

**Symptom:** The fabric deadlock stability tests hang on specific test patterns, particularly the 2D Torus Y-only configuration. The YAML configuration explicitly documents this:

```yaml
# Hangs: https://github.com/tenstorrent/tt-metal/issues/33456
  # - name: UnicastAlltoAll
  #   fabric_setup:
  #     topology: Torus
  #     torus_config: Y
```

**Root Cause:** The stability tests exercise all fabric routing patterns with various packet types. Known failure modes include:
- **Y-only Torus hangs (issue #33456)**: The routing table for Y-only torus has a specific deadlock scenario where North-South traffic competes with South-North traffic on shared spine links.
- **All-to-all with `unicast_scatter_write`**: Scatter writes generate fine-grained NOC transactions, increasing probability of transaction ID exhaustion on the receiver.
- **Multi-directional multicast (NorthEast, SouthWest, AllDirs)**: These exercise the 2D routing logic in `fabric_edge_node_router.hpp`.

**Diagnosis Steps:**
1. Run the stability tests in isolation to reproduce the hang.
2. Check which specific test pattern hangs by running patterns individually.
3. Cross-reference with the known issues list in the YAML comments.
4. Use fabric telemetry to identify which link/direction is the bottleneck.

**Fix:** These are typically firmware or routing table bugs that require changes to the EDM router or the `ControlPlane` routing table generator.

**Prevention:**
- Run the full deadlock stability test suite before deploying new firmware or routing table changes.
- Do not remove the commented-out test patterns from the YAML -- they serve as documentation of known limitations.

---

## Summary Table

| Scenario | Hang Indicator | Typical Configuration | Fix Category |
|----------|---------------|----------------------|--------------|
| 5.2.1 Partial Participation | All but one device at `NSW` | Any multi-chip | Ensure all ranks execute collective |
| 5.2.2 Tensor Dimension Mismatch | Partial data transfer, deterministic | Any multi-chip | Uniform tensor shapes |
| 5.2.3 Topology Mismatch | EDM sender spins on non-existent link | Ring on Linear HW | Use auto-detect topology |
| 5.2.4 Termination Master | Workers done but not exiting | Any multi-chip CCL | Correct master coordinates |
| 5.2.5 Socket Rank Mismatch | send/recv on wrong rank | Multi-host / distributed | Use `create_socket_pair()` |
| 5.2.6 EDM Termination Mode | EDM runs after workers complete | Any multi-chip CCL | Signal termination on all paths |
| 5.2.7 Bidir Socket Ordering | Hang in MeshSocket constructor | Multi-host | Use `BidirectionalFabricSocket::create()` |
| 5.2.8 Num-Links Mismatch | Link collision with dispatch | Galaxy | Use `num_links=None` |
| 5.2.9 Composite Deadlock | 2nd dimension collective hangs | 2D mesh | Use built-in composite ops |
| 5.2.10 Socket Memory Config | Socket setup or first op hangs | MeshSocket-based ops | Validate coords and L1 |
| 5.2.11 Ring Circular Wait | All devices sending/receiving, stalled | Ring / Torus | Reduce `chunks_per_sync` |
| 5.2.12 Stability Test Failures | Known patterns hang | Galaxy / Torus Y | Track upstream fixes (issue #33456) |

---

[Previous: 5.1 Ethernet and Fabric Fundamentals](./01_ethernet_and_fabric_fundamentals.md) | [Next: 5.3 Topology and Mesh Configuration Hangs](./03_topology_and_mesh_configuration_hangs.md)
