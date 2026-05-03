# 5.3 Topology and Mesh Configuration Hangs

[Previous: 5.2 CCL Collective Operation Hangs](./02_ccl_collective_operation_hangs.md) | [Next: Chapter 6 -- Debugging Tools and Infrastructure](../ch06_debugging_tools/index.md)

---

The physical connectivity of Tenstorrent chips, the logical mesh configuration, and the routing infrastructure that maps between them form the foundation on which the fabric layer (Section 5.1) and CCL operations (Section 5.2) depend. Configuration errors at this level -- wrong chip counts, incorrect routing tables, mismatched mesh descriptors, or improper initialization ordering -- can cause every subsequent fabric and CCL operation to hang. These hangs are often the hardest to diagnose because the symptoms appear in the fabric or CCL layer but the root cause is in the system-level topology configuration.

**Prerequisites:** See [Chapter 5 Index prerequisites](./index.md#prerequisites). Also assumes familiarity with the EDM handshake protocol (Scenario 5.1.4) and the CCL topology resolution (Section 5.2, Part 1).

**Reference files:**
- `tt_metal/api/tt-metalium/mesh_device.hpp` -- `MeshDevice`, `MeshDeviceConfig`, `MeshShape`, `reshape()`
- `tt_metal/api/tt-metalium/experimental/fabric/control_plane.hpp` -- `ControlPlane`, routing tables, `FabricConfig`
- `tt_metal/api/tt-metalium/experimental/fabric/fabric_switch_manager.hpp` -- `FabricSwitchManager` singleton
- `tt_metal/api/tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp` -- `MeshGraphDescriptor` for multi-mesh
- `tt_metal/impl/dispatch/topology.cpp` -- `DispatchKernelNode` definitions, Galaxy 9-chip topology
- `tt_metal/fabric/control_plane.cpp` -- `convert_fabric_routing_table_to_chip_routing_table()`
- `tt_metal/fabric/fabric_context.cpp` -- `FabricContext`, `need_deadlock_avoidance_support()`
- `tt_metal/llrt/rtoptions.hpp` -- `skip_eth_cores_with_retrain`
- `tests/scale_out/4x_bh_quietbox/distributed_reset.sh` -- Distributed reset coordination

## Part 1: Mesh Device and Control Plane Architecture

### MeshDevice Lifecycle

The `MeshDevice` class encapsulates a collection of devices arranged in a 2D grid. It can be reshaped (e.g., 1x8 to 2x4) subject to physical connectivity constraints, and submeshes can be carved from a parent mesh:

```cpp
class MeshDevice : public IDevice, public std::enable_shared_from_this<MeshDevice> {
    static std::shared_ptr<MeshDevice> create(const MeshDeviceConfig& config, ...);
    void reshape(const MeshShape& new_shape);
    std::shared_ptr<MeshDevice> create_submesh(const MeshShape& submesh_shape, ...);
    void quiesce_devices();  // Barrier for overlapping submeshes
};
```

### ControlPlane and Routing

The `ControlPlane` manages the mapping from logical fabric topology to physical chip connectivity:

```cpp
class ControlPlane {
public:
    explicit ControlPlane(
        const ::tt::Cluster& cluster,
        const ::tt::llrt::RunTimeOptions& rtoptions,
        const ::tt::tt_metal::Hal& hal,
        const tt_metal::distributed::multihost::DistributedContext& distributed_context,
        FabricConfig fabric_config = FabricConfig::DISABLED,
        FabricReliabilityMode fabric_reliability_mode =
            FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
        // ...
    );

    void configure_routing_tables_for_fabric_ethernet_channels();
    void write_routing_tables_to_all_chips() const;
};
```

The routing table conversion (`convert_fabric_routing_table_to_chip_routing_table()`) maps source-destination pairs to next-hop Ethernet channels. Errors here produce routing loops or black holes.

### FabricSwitchManager

The `FabricSwitchManager` singleton manages fabric device lifecycle between tests. Failure to call `teardown()` between tests causes handshake deadlocks -- see [Scenario 5.3.3](#533-fabricswitchmanager-teardown-failure-between-tests).

### T3K vs. Galaxy: Architectural Comparison

| Property | T3K (8-chip) | Galaxy (32+ chip) |
|----------|-------------|-------------------|
| Board type | N300 pairs | Galaxy module |
| Ethernet links per chip | 16 | Up to 16 |
| MMIO devices | All 8 accessible from single host | 1 per column (4 per host typically) |
| Fabric topology | Ring or 2D mesh | 2D mesh or torus (4x8, 8x4) |
| Initialization time | Seconds | 30-120 seconds |
| Shutdown order | Any order | Farthest-to-closest from MMIO |
| Link stability | Generally stable | More variable (longer cables, more connectors) |
| Deadlock avoidance | Optional (ring only) | Required for 2D mesh/torus |
| Reset mechanism | `tt-smi -r` per host | `distributed_reset.sh` across hosts |
| Dispatch topology | Direct MMIO or simple tunnel | Multi-hop tunnel through FABRIC_MUX nodes |

---

## Part 2: Hang Scenarios

### 5.3.1 Wrong Chip Connectivity Assumptions in Ring Ordering

**Symptom:** A CCL operation configured for Ring topology hangs immediately on the first data transfer. The first chip sends data, but the intended recipient never receives it.

**Root Cause:** The ring ordering for CCL operations must match the physical Ethernet connectivity. If the ring ordering assumes `0 -> 1 -> 2 -> 3 -> ...` but the physical connectivity follows a different pattern, packets sent from one chip attempt to traverse a non-existent direct link.

The `ControlPlane` constructs routing tables from the physical topology via `convert_fabric_routing_table_to_chip_routing_table()`. If the `MeshGraphDescriptor` incorrectly describes the inter-chip connectivity, the routing table directs packets to incorrect links.

**Diagnosis Steps:**
1. Dump the ControlPlane routing table and verify each entry against the physical Ethernet connections.
2. Use `get_ethernet_sockets(connected_chip_id)` to list physical connections between adjacent chips.
3. Compare against golden routing files (e.g., `ControlPlaneFixture_T3k.yaml`, `ControlPlaneFixture_SingleGalaxy.yaml`).

**Fix:**
```cpp
// BUGGY: Manual ring ordering that does not match physical connectivity
std::vector<int> ring_order = {0, 1, 2, 3, 4, 5, 6, 7};  // Assumes sequential

// CORRECTED: Use the framework's topology solver
auto adjacency = build_adjacency_graph_logical(mesh_graph);
auto routing_table = routing_table_generator->get_intra_mesh_table();
```

**Prevention:**
- Never hardcode chip ordering. Always use the `ControlPlane` and `RoutingTableGenerator`.
- Validate routing tables against golden files in the test suite.

---

### 5.3.2 Routing Table Loop Causing Packet Circulation

**Symptom:** The fabric appears busy (high bandwidth utilization on telemetry) but no CCL operations complete. Similar to Scenario 5.1.8 (packet header corruption), but the root cause is in the ControlPlane routing table rather than data corruption.

**Root Cause:** The `convert_fabric_routing_table_to_chip_routing_table()` function converts the topology solver's routing table into per-chip, per-channel entries. If the solver produces a routing table with a cycle (chip A routes to chip B, chip B routes to chip A for the same destination), packets bounce forever.

This can happen when:
- The `MeshGraphDescriptor` has bidirectional links that the solver treats as two independent unidirectional links.
- The `FabricConfig` enables a routing mode the solver does not properly handle.
- The `FabricReliabilityMode` setting conflicts with the routing algorithm's assumptions.

**Diagnosis Steps:**
1. For each chip pair, trace the routing path from source to destination in the routing table.
2. Check for cycles: if chip A says "to reach chip C, send to chip B" and chip B says "to reach chip C, send to chip A," there is a loop.
3. Compare against expected shortest-path routing.

**Fix:** This requires fixing the `RoutingTableGenerator` or the topology solver. As a workaround, use a known-good `MeshGraphDescriptor`.

**Prevention:**
- Run routing table validation as part of fabric initialization: check for cycles before programming the routes.
- Use golden mapping files in `tests/tt_metal/tt_fabric/golden_mapping_files/` as regression tests.
- Add a TTL (time-to-live) field to packet headers so routing loops are detected at runtime.

---

### 5.3.3 FabricSwitchManager Teardown Failure Between Tests

**Symptom:** The second test in a test suite hangs during fabric initialization. The first test passes, but the second never completes the fabric handshake. `RouterState` on some ERISC cores is `INITIALIZING` and never transitions to `RUNNING`.

**Root Cause:** The `FabricSwitchManager` singleton manages device lifecycle for switch meshes. If `teardown()` is not called between tests, fabric routers from the previous test are still running. The new test's `setup()` attempts to initialize fabric on devices that already have running routers, and the handshake protocol deadlocks because one side expects a fresh start while the other is in `RUNNING` state.

**Diagnosis Steps:**
1. Check if `FabricSwitchManager::teardown()` was called between tests.
2. Check `RouterState` on all ERISC cores: if some are `RUNNING` while others are `INITIALIZING`, this is a teardown failure.
3. Check the `switch_devices_` map in the singleton.

**Fix:**
```cpp
// BUGGY: Test fixture does not call teardown
class MyTestFixture : public ::testing::Test {
    void SetUp() override {
        FabricSwitchManager::instance().setup(FabricConfig::ENABLED);
    }
    // Missing TearDown!
};

// CORRECTED: Always teardown between tests
class MyTestFixture : public ::testing::Test {
    void SetUp() override {
        FabricSwitchManager::instance().setup(FabricConfig::ENABLED);
    }
    void TearDown() override {
        FabricSwitchManager::instance().teardown();
    }
};
```

**Prevention:**
- Use RAII patterns for fabric lifecycle management.
- Add a check in `setup()` that verifies all switch devices are in a clean state.
- Track issue #34040 (keep routers in standby mode instead of fully terminating between workloads).

---

### 5.3.4 MMIO vs Remote Device Tunnel Initialization Hang

**Symptom:** A Galaxy configuration hangs during device initialization. MMIO-mapped devices initialize successfully, but remote devices accessed through Ethernet tunnels never complete initialization.

**Root Cause:** In Galaxy configurations, only a subset of devices are MMIO-mapped (directly accessible from the host PCIe bus). The remaining devices are accessed through Ethernet tunnels. The dispatch topology describes this relationship:

```cpp
// From topology.cpp - Galaxy 9-chip topology:
// { id, device_id, servicing_device_id, cq, fd_kernel, ... }
{0, 0, 1, 0, PREFETCH_H, ...},   // Runs on device 0, services device 1
{1, 0, 1, 0, DISPATCH_H, ...},   // Runs on device 0, services device 1
// ...
{18, 1, x, 0, PREFETCH_D, ...},  // Runs on remote device 1
{19, 1, x, 0, DISPATCH_D, ...},  // Runs on remote device 1
```

The `device_id` indicates where the kernel runs, and `servicing_device_id` indicates which remote device it services. For remote device access, the host communicates through: Host -> PREFETCH_H on MMIO device -> FABRIC_MUX -> Ethernet -> PREFETCH_D on remote device -> DISPATCH_D.

If the fabric between the MMIO device and remote device is not properly initialized before dispatch attempts to use it, commands are lost and the host waits indefinitely. See also Chapter 4 for dispatch pipeline internals.

**Diagnosis Steps:**
1. Check which devices are MMIO-capable: `mesh_device->is_mmio_capable()`.
2. Check if fabric initialization completed on the MMIO device before dispatch was started on remote devices.
3. Look at the FABRIC_MUX and RETURN_FABRIC_MUX nodes in the dispatch topology.
4. Check the watcher on the MMIO device for PREFETCH_H stuck at a barrier.

**Fix:** Ensure fabric initialization (`compile_fabric()`, `configure_fabric()`) completes before any dispatch operations to remote devices. The initialization order must be: open all MMIO devices -> initialize fabric -> complete handshake -> initialize dispatch on MMIO devices -> initialize dispatch tunnels to remote devices.

**Prevention:**
- Use `MeshDevice::create()` which handles initialization ordering correctly.
- Do not manually call deprecated internal functions.
- Test multi-chip initialization with all device counts before deploying.

---

### 5.3.5 Fabric Initialization Timeout on Galaxy

**Symptom:** The first operation on a Galaxy system hangs during fabric initialization. The fabric never reaches `RUNNING` state. The initialization appears to take minutes, eventually timing out.

**Root Cause:** Galaxy systems with 32+ devices require a specific initialization sequence. The `is_handshake_sender` compile-time argument (index `MAIN_CT_ARGS_START_IDX + 4`) determines which side of each link initiates the handshake:

```cpp
constexpr bool is_handshake_sender = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 4) != 0;
constexpr size_t handshake_addr = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 5);
```

If both sides of a link are `is_handshake_sender = true`, neither waits for the other to initiate. If both are `false`, both wait forever.

**Diagnosis Steps:**
1. Check `RouterState` on all Ethernet cores: look for cores stuck in `INITIALIZING`.
2. Identify the specific stuck link by finding a pair of connected Ethernet cores both in `INITIALIZING`.
3. Check the `is_handshake_sender` compile-time argument on both sides -- they should be complementary.

**Fix:**
```cpp
// BUGGY: Both sides are handshake sender
// Side A: is_handshake_sender = true
// Side B: is_handshake_sender = true  // WRONG!

// CORRECTED: One sender, one receiver per link
// Side A: is_handshake_sender = true
// Side B: is_handshake_sender = false
```

**Prevention:**
- The fabric compilation pipeline should enforce complementary handshake roles.
- On Galaxy, expect initialization to take 30-120 seconds and set timeouts accordingly.

---

### 5.3.6 MeshDevice Reshape Violating Physical Connectivity

**Symptom:** A CCL operation hangs after a `MeshDevice::reshape()` call. The operation worked before the reshape, but after reshaping (e.g., from 1x8 to 2x4), operations on the new logical grid hang.

**Root Cause:** `MeshDevice::reshape()` changes the logical mapping. From `mesh_device.hpp`:
- The old_shape volume must equal the new_shape volume
- Line-to-Line Reshaping: Always possible (1xN <-> Nx1)
- Grid-to-Grid Reshaping: Only possible if physical devices can form a connected mesh in the new shape

If a reshape succeeds but the physical connectivity does not support the new grid layout, CCL operations that assume adjacent logical devices are physically connected will try to send over non-existent links:

```
Logical:        Physical connections:
[0, 1, 2, 3]   0-1, 1-2, 2-3 (row 0)
[4, 5, 6, 7]   4-5, 5-6, 6-7 (row 1)
                0-4? (column) -- may not exist!
```

**Diagnosis Steps:**
1. After reshape, verify each pair of logically adjacent devices has a physical Ethernet connection.
2. Check the `MeshDeviceView` for the physical-to-logical mapping.
3. Use `get_ethernet_sockets(connected_chip_id)` on each device to verify physical connectivity.

**Fix:**
```python
# BUGGY: Reshaping to a grid that physical topology does not support
mesh_device.reshape(MeshShape(2, 4))
output = ttnn.all_gather(tensor, dim=0, cluster_axis=0)  # Column-wise: hangs!

# CORRECTED: Verify physical connectivity or use line topology
output = ttnn.all_gather(tensor, dim=0, cluster_axis=None)
```

**Prevention:**
- Before reshape, verify physical connectivity supports the target shape.
- Prefer explicit topology specification over implicit assumptions about the grid layout.

---

### 5.3.7 Multi-Host Synchronization Failure

**Symptom:** A multi-host Galaxy deployment hangs during a collective operation that spans hosts. Operations within a single host's devices work fine, but cross-host operations never complete.

**Root Cause:** Multi-host deployments use `DistributedContext` to coordinate across hosts. Hang causes include:
- **Host rank mismatch**: If host A and host B both think they are rank 0, both try to be the sender.
- **Inter-host Ethernet link failure**: The link between hosts goes down, but intra-host links are fine.
- **Barrier synchronization timeout**: If one host's barrier signal is delayed, others may timeout.

The `MeshGraphDescriptor` maps which meshes are on which hosts:
```cpp
struct LocalMeshBinding {
    std::vector<MeshId> mesh_ids;
    MeshHostRankId host_rank;
};
```

**Diagnosis Steps:**
1. Verify each host's `distributed_context->rank()` returns a unique value.
2. Check inter-host Ethernet link status using watcher and telemetry.
3. Test intra-host operations in isolation to confirm they work.
4. Check the `MeshGraphDescriptor` for correct host-to-mesh mapping.

**Fix:**
```python
# BUGGY: Both hosts create MeshDevice with the same rank
mesh_device = MeshDevice.create(config)  # Both hosts get rank 0

# CORRECTED: Each host must use its proper rank
distributed_context = create_distributed_context(...)
mesh_device = MeshDevice.create(config, distributed_context=distributed_context)
```

**Prevention:**
- Always use the `DistributedContext` to determine host rank rather than hardcoding.
- Test multi-host connectivity independently from multi-host computation.

---

### 5.3.8 Distributed Reset Ordering Violation on Galaxy

**Symptom:** After a hang or error on a Galaxy system, the reset procedure itself hangs. Some devices reset successfully but others remain in a hung state.

**Root Cause:** Galaxy systems require ordered shutdown and reset. The `distributed_reset.sh` script demonstrates the correct procedure:

```bash
parallel-ssh -i -H "host1 host2 host3 host4" \
    "touch $BARRIER_DIR/\$(hostname) && \
     while [ \$(ls $BARRIER_DIR | wc -l) -lt 4 ]; do sleep 0.01; done && \
     tt-smi -r"
```

Key requirements:
1. **All hosts must participate**: If one host skips the barrier, others reset while its devices are still active.
2. **Ordered shutdown**: Devices must be reset from farthest to closest (relative to the MMIO device). If a closer device is reset first, the Ethernet tunnel to farther devices is severed.
3. **Synchronized timing**: The barrier ensures all hosts reach the reset point before any host resets.

**Diagnosis Steps:**
1. Check if all hosts participated in the reset barrier.
2. Verify the reset order: were remote devices reset before the MMIO device?
3. Check if any devices are in a partially reset state via `tt-smi`.

**Fix:**
```bash
# BUGGY: Each host resets independently without synchronization
# Host A: tt-smi -r   (resets immediately)
# Host B: tt-smi -r   (5 seconds later -- too late!)

# CORRECTED: Use the distributed reset script with barrier synchronization
./distributed_reset.sh
```

**Prevention:**
- Always use the distributed reset script for Galaxy configurations.
- On Galaxy, the `MeshDevice::close()` method handles ordered shutdown internally -- do not bypass it.

---

### 5.3.9 DispatchKernelNode Topology Mismatch for Remote Devices

**Symptom:** Operations to specific remote devices in a Galaxy system hang, while operations to other remote devices work fine. The hanging devices are those serviced by a specific FABRIC_MUX node.

**Root Cause:** The dispatch topology in `topology.cpp` defines a `DispatchKernelNode` graph:

```cpp
// Galaxy 9-chip topology:
// FABRIC_MUX on device 0 services remote chips 1-4
{8, 0, x, 0, FABRIC_MUX,
    /*full size*/ {0, 2, 4, 6}, /*header only*/ {1, 3, 5, 7},
    k_fabric_mux_noc, 0},

// FABRIC_MUX on device 0 services remote chips 5-8
{17, 0, x, 0, FABRIC_MUX,
    /*full size*/ {9, 11, 13, 15}, /*header only*/ {10, 12, 14, 16},
    k_fabric_mux_noc, 1},
```

Each `FABRIC_MUX` multiplexes traffic to the fabric. The last parameter (0 or 1) indicates which Ethernet channel it uses. If this channel assignment does not match the actual physical connectivity, the mux sends traffic to the wrong link.

**Diagnosis Steps:**
1. Identify which remote devices are hanging by checking which PREFETCH_D/DISPATCH_D kernels are stuck.
2. Trace back to the FABRIC_MUX node that services those devices.
3. Verify the Ethernet channel assignment matches the physical link to the remote device.
4. Check the `servicing_device_id` in the PREFETCH_H/DISPATCH_H nodes.

**Fix:** The dispatch topology is generated by `DispatchTopology::generate_nodes()`. If the generation produces incorrect channel assignments, the fix is in the topology generation logic.

**Prevention:**
- Do not manually construct `DispatchKernelNode` graphs for production use.
- Validate the generated topology against the physical cluster description.
- Test each remote device individually to identify which mux/channel combination is broken.

---

### 5.3.10 FabricReliabilityMode Strict vs. Relaxed Mismatch

**Symptom:** Fabric initialization fails or hangs on a Galaxy system that has one or two dead Ethernet links. The `ControlPlane` reports validation failures or waits for links that will never come up.

**Root Cause:** `FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE` requires all expected Ethernet links to be live. The `ControlPlane` validates link counts against a golden model derived from the mesh graph descriptor. If any link is missing, the validation fails. In strict mode, this is a fatal condition -- the `ControlPlane` either throws or hangs waiting. In relaxed mode, it would reduce routing planes and route around the dead link.

**Diagnosis Steps:**
1. Check the `ControlPlane` constructor arguments for `fabric_reliability_mode`.
2. Look for validation failure messages in the logs.
3. Use `ControlPlane::print_all_ethernet_connections()` to see which links are detected as active.
4. Compare with the expected link count from the mesh graph descriptor.

**Fix:** Switch to relaxed reliability mode if operating on a partially degraded system. Fix or replace the dead links for production use.

**Prevention:**
- Use a pre-flight check script that verifies all Ethernet links are up before attempting fabric initialization in strict mode.
- Only use strict mode on systems that have passed the pre-flight check.

---

### 5.3.11 Submesh Quiescence Failure Causing Cross-Submesh Hang

**Symptom:** Operations on overlapping submeshes hang. The first submesh's operations complete, but operations on a second submesh that shares physical devices with the first hang.

**Root Cause:** `MeshDevice` supports creating submeshes that share physical devices. The `quiesce_devices()` method exists specifically to handle this:

```cpp
/**
 * @brief Synchronize with all devices derived from this mesh.
 *
 * Blocks until all in-flight work enqueued on every submesh derived
 * from this mesh has completed. Use this to insert a barrier between
 * phases that use overlapping submeshes on the same physical devices.
 * All submeshes must be using the default subdevice manager when this
 * is called.
 */
void quiesce_devices();
```

If `quiesce_devices()` is not called between phases that use overlapping submeshes, the second phase's operations may encounter devices that still have in-flight work from the first phase, corrupting shared state and causing hangs.

**Diagnosis Steps:**
1. Check if the program creates overlapping submeshes.
2. Verify that `quiesce_devices()` is called between phases.
3. Check the sub-device manager state: all submeshes must use the default sub-device manager.

**Fix:**
```python
# BUGGY: Operations on overlapping submeshes without quiescence
submesh_a = mesh_device.create_submesh(MeshShape(1, 4), offset=(0, 0))
submesh_b = mesh_device.create_submesh(MeshShape(1, 4), offset=(0, 2))
# submesh_a and submesh_b share devices at columns 2 and 3

ttnn.all_gather(tensor_a, device=submesh_a)  # Phase 1
ttnn.all_gather(tensor_b, device=submesh_b)  # Phase 2: HANGS!

# CORRECTED: Quiesce between overlapping submesh phases
ttnn.all_gather(tensor_a, device=submesh_a)  # Phase 1
mesh_device.quiesce_devices()                 # Wait for all in-flight work
ttnn.all_gather(tensor_b, device=submesh_b)  # Phase 2: safe
```

**Prevention:**
- Always call `quiesce_devices()` on the parent mesh before switching between overlapping submeshes.
- Prefer non-overlapping submeshes when possible to avoid the need for quiescence.

---

## Summary Table

| Scenario | Hang Indicator | Typical Configuration | Fix Category |
|----------|---------------|----------------------|--------------|
| 5.3.1 Wrong Ring Ordering | First chip sends, recipient silent | T3K / Galaxy CCL | Use ControlPlane routing |
| 5.3.2 Routing Table Loop | High bandwidth, no progress | Any multi-chip fabric | Fix RoutingTableGenerator |
| 5.3.3 SwitchManager Teardown | Second test hangs at handshake | Test suites with fabric | Call `teardown()` |
| 5.3.4 MMIO Tunnel Init | Remote devices never initialize | Galaxy | Correct init ordering |
| 5.3.5 Galaxy Fabric Init | `RouterState::INITIALIZING` stalled | Galaxy 32+ chip | Complementary handshake roles |
| 5.3.6 Reshape Connectivity | CCL hangs after reshape | 2D mesh reshaping | Verify physical connectivity |
| 5.3.7 Multi-Host Sync | Cross-host ops hang, intra-host OK | Multi-host Galaxy | Correct host rank |
| 5.3.8 Distributed Reset | Partial reset state | Galaxy reset | Use `distributed_reset.sh` |
| 5.3.9 DispatchNode Channel | Specific remote devices hang | Galaxy dispatch | Verify FABRIC_MUX channels |
| 5.3.10 Reliability Mode | Init hangs on degraded system | Galaxy with dead links | Use relaxed mode |
| 5.3.11 Submesh Quiescence | Overlapping submesh ops hang | Any multi-chip submesh | Call `quiesce_devices()` |

---

## Cross-Chapter Reference Table

The following table maps Chapter 5 scenarios to related content in other chapters:

| Ch5 Scenario | Related Chapter | Related Scenario / Section |
|-------------|----------------|---------------------------|
| 5.1.1 (Link down) | Ch1, `04_hang_causes_across_architectures.md` | Architecture-specific Ethernet core counts |
| 5.1.5 (Forwarding deadlock) | Ch2, `04_noc_barrier_and_semaphore_hangs.md` | NOC barrier and semaphore hangs (same primitives, different layer) |
| 5.1.7 (Heartbeat stall) | Ch3, `02_dram_and_noc_backpressure.md` | NOC backpressure causing receiver stall |
| 5.2.1 (Device skips collective) | Ch4, `01_dispatch_architecture_and_hang_points.md` | Dispatch waiting for workers (device never starts) |
| 5.2.4 (Termination master) | Ch2, `04_noc_barrier_and_semaphore_hangs.md` | NOC semaphore protocol violations |
| 5.2.10 (Socket memory mismatch) | Ch3, `01_l1_memory_corruption_and_overflow.md` | L1 corruption from buffer overflow |
| 5.3.4 (Tunnel not initialized) | Ch4, `01_dispatch_architecture_and_hang_points.md` | Dispatch pipeline through Ethernet tunnels |
| 5.3.8 (Distributed reset) | Ch1, `01_what_is_a_hang.md` | Board-level reset as last resort recovery |

---

[Previous: 5.2 CCL Collective Operation Hangs](./02_ccl_collective_operation_hangs.md) | [Next: Chapter 6 -- Debugging Tools and Infrastructure](../ch06_debugging_tools/index.md)
