# Chapter 5: Multi-Chip, CCL, and Fabric Hang Causes

Single-chip hangs are disruptive. Multi-chip hangs are catastrophic. When a collective operation stalls across an 8-chip T3K or a 32-chip Galaxy cluster, the blast radius is not one device but every device participating in the operation. The Ethernet fabric that connects these devices introduces an entirely new category of failure modes -- link retraining events, EDM handshake deadlocks, routing table misconfigurations, and cross-chip synchronization failures -- that have no analog in single-chip operation. The debugging surface expands from a single device's five RISC-V cores to hundreds of cores coordinating across physical Ethernet links, each with its own flow control state machine, handshake protocol, and failure modes. A single downed Ethernet link on one chip can cause every device in a 32-chip ring to spin forever, waiting for a packet that will never arrive.

This chapter systematically catalogs every known hang mechanism specific to multi-chip configurations. The scope covers the Ethernet link layer (AERISC/IERISC cores and the watcher link status check), the Fabric Ethernet Data Mover (EDM) router and its flow control protocol, CCL collective operations (`all_gather`, `reduce_scatter`, `all_reduce`, `all_broadcast`, `reduce_to_root`, `all_to_all_combine`), fabric socket programming, mesh device topology configuration, and multi-host synchronization. Every hang cause follows the **Symptom / Root Cause / Diagnosis Steps / Fix / Prevention** format introduced in [Chapter 1](../ch01_anatomy_of_a_hang/01_what_is_a_hang.md).

## Prerequisites

- **Chapter 1, [`01_what_is_a_hang.md`](../ch01_anatomy_of_a_hang/01_what_is_a_hang.md)**: The RISC-V spin-loop hang model, the 5-part diagnostic format, and the `assert_and_hang` mechanism. Multi-chip hangs use the same fundamental spin-loop model but on ERISC cores rather than Tensix RISC-V cores.
- **Chapter 1, [`02_blocking_primitives_taxonomy.md`](../ch01_anatomy_of_a_hang/02_blocking_primitives_taxonomy.md)**: Familiarity with `noc_semaphore_wait`, `noc_async_write_barrier`, and the ethernet-specific blocking primitives (`eth_send_packet`, `eth_txq_is_busy` spin-loops). These primitives are the building blocks for all EDM flow control.
- **Chapter 1, [`04_hang_causes_across_architectures.md`](../ch01_anatomy_of_a_hang/04_hang_causes_across_architectures.md)**: Architectural differences between Wormhole, Blackhole, and Quasar, especially the number of Ethernet cores (16 on WH, up to 16 on BH with dual AERISC/IERISC), link speeds, and the Quasar inter-die fabric.
- **Chapter 2, [`04_noc_barrier_and_semaphore_hangs.md`](../ch02_kernel_and_noc_hangs/04_noc_barrier_and_semaphore_hangs.md)**: NOC semaphore increment/wait mechanics. CCL operations use NOC semaphores for cross-chip synchronization through the fabric, so every semaphore hang mechanism from Chapter 2 can also manifest in multi-chip contexts.
- **Chapter 3, [`02_dram_and_noc_backpressure.md`](../ch03_memory_related_hangs/02_dram_and_noc_backpressure.md)**: NOC backpressure propagation. When a remote device experiences DRAM backpressure, the stall propagates back through Ethernet links to the sending device's EDM, which can appear as a fabric hang.
- **Chapter 4, [`01_dispatch_architecture_and_hang_points.md`](../ch04_dispatch_and_host_device_hangs/01_dispatch_architecture_and_hang_points.md)**: The dispatch topology for multi-chip configurations (PREFETCH_H/DISPATCH_H on the host-side device, PREFETCH_D/DISPATCH_D on remote devices, FABRIC_MUX for multiplexing). Understanding how dispatch kernels are mapped to `device_id` vs `servicing_device_id` is essential for diagnosing remote device dispatch hangs.

## How to Use This Chapter

- **If a watcher reports an Ethernet link went down** (`link_down = 1` in the eth status mailbox): start with [Section 01, Scenario 5.1.1](./01_ethernet_and_fabric_fundamentals.md#511-ethernet-link-down-during-active-operation).
- **If fabric telemetry shows `RouterState::INITIALIZING` that never transitions to `RUNNING`**: start with [Section 01, Scenario 5.1.4](./01_ethernet_and_fabric_fundamentals.md#514-edm-handshake-failure-during-initialization).
- **If the watcher shows ERISC cores stuck at `RW` or the heartbeat counters stop incrementing**: start with [Section 01, Scenario 5.1.7](./01_ethernet_and_fabric_fundamentals.md#517-fabric-telemetry-heartbeat-stall-detection).
- **If a CCL operation (all_gather, reduce_scatter, etc.) hangs across devices**: start with [Section 02](./02_ccl_collective_operation_hangs.md).
- **If the hang occurs only at specific mesh sizes or topologies**: start with [Section 03](./03_topology_and_mesh_configuration_hangs.md).
- **If the hang occurs only in Galaxy (32+ chip) but not T3K (8-chip)**: start with [Section 03, Scenario 5.3.5](./03_topology_and_mesh_configuration_hangs.md#535-fabric-initialization-timeout-on-galaxy).
- **If socket-based communication between meshes hangs**: start with [Section 02, Scenario 5.2.5](./02_ccl_collective_operation_hangs.md#525-fabric-socket-rank-mismatch).
- **If a hang occurs after a link retraining event**: start with [Section 01, Scenario 5.1.3](./01_ethernet_and_fabric_fundamentals.md#513-link-retraining-during-active-data-transfer).

## The Multi-Chip Data Path at a Glance

```
  CHIP 0 (MMIO)                    CHIP 1 (Remote)                   CHIP 2 (Remote)
 +------------------+             +------------------+              +------------------+
 |  Worker Kernels  |             |  Worker Kernels  |              |  Worker Kernels  |
 |   (Tensix cores) |             |   (Tensix cores) |              |   (Tensix cores) |
 +--------+---------+             +--------+---------+              +--------+---------+
          |                                |                                 |
          | NOC write to                   | NOC write to                    | NOC write to
          | EDM sender ch0                 | EDM sender ch0                  | EDM sender ch0
          v                                v                                 v
 +------------------+  Ethernet   +------------------+  Ethernet    +------------------+
 | EDM Router       |  Link       | EDM Router       |  Link        | EDM Router       |
 | (AERISC core)    +------------>| (AERISC core)    +------------->| (AERISC core)    |
 |                  |<------------+                  |<-------------+                  |
 | Sender Ch0 (local worker)      | Sender Ch0 (local worker)       | Sender Ch0 (local)
 | Sender Ch1 (upstream EDM)      | Sender Ch1 (upstream EDM)       | Sender Ch1       |
 | Receiver Ch (from link)        | Receiver Ch (from link)          | Receiver Ch      |
 +------------------+             +------------------+              +------------------+
```

Each arrow in this diagram is a potential hang point:
- **Worker to EDM**: Worker spins on flow control semaphore waiting for EDM to acknowledge space
- **EDM Sender to Receiver**: Sender spins on stream register credits waiting for remote receiver acknowledgement
- **EDM Receiver to local NOC**: Receiver issues NOC writes to local chip; backpressure from NOC stalls the receiver
- **EDM Receiver to downstream EDM**: Multi-hop forwarding; receiver waits for downstream EDM to have buffer space

## Chapter Contents

| # | File | Focus | Key Indicators |
|---|------|-------|----------------|
| 1 | [`01_ethernet_and_fabric_fundamentals.md`](./01_ethernet_and_fabric_fundamentals.md) | ERISC core architecture, Ethernet link status, EDM router, flow control, deadlock avoidance, fabric telemetry (12 scenarios) | `link_down = 1`, `RouterState::INITIALIZING`, stalled heartbeat TX/RX, `eth_txq_is_busy` spin |
| 2 | [`02_ccl_collective_operation_hangs.md`](./02_ccl_collective_operation_hangs.md) | CCL operations, semaphore protocols, fabric sockets, deadlock scenarios, stability tests (12 scenarios) | Missing collective participants, semaphore mismatch, socket rank mismatch, termination master failures |
| 3 | [`03_topology_and_mesh_configuration_hangs.md`](./03_topology_and_mesh_configuration_hangs.md) | MeshDevice setup, ControlPlane/FabricSwitchManager, T3K vs Galaxy, MMIO vs remote, multi-host sync (11 scenarios) | Wrong chip count, routing table loops, `skip_eth_cores_with_retrain`, `distributed_reset.sh`, tunnel init failure |

**Covers research questions:** Q2 (all multi-chip and CCL hang causes), Q9 (multi-chip configuration differences between T3K and Galaxy).

## Multi-Chip Hang Quick Reference

| Category | Typical Symptom | Affected Configurations | Section |
|----------|----------------|------------------------|---------|
| Ethernet link failure | ERISC spins in `while(1)` after `hang_on_down_link()`, watcher shows `link_down = 1` | All multi-chip | 01, Scenarios 5.1.1--5.1.4 |
| Fabric router stall | EDM sender/receiver channels block on flow control, no forward progress | All multi-chip with fabric | 01, Scenarios 5.1.5--5.1.8 |
| Fabric deadlock (circular wait) | All downstream channels full, no bubbles injected | 2D mesh/torus topologies | 01, Scenarios 5.1.5--5.1.6 |
| EDM transmit / credit race | ERISC stuck in `eth_txq_is_busy` spin or credit counters diverge | All multi-chip (BH-specific for credit race) | 01, Scenarios 5.1.9--5.1.10 |
| CCL collective incomplete | One or more ranks never enter collective, others wait forever | All multi-chip CCL | 02, Scenarios 5.2.1--5.2.4 |
| Semaphore protocol violation | Remote semaphore never incremented, local core spins at `noc_semaphore_wait` | All multi-chip CCL | 02, Scenarios 5.2.4--5.2.6 |
| Fabric socket setup failure | Socket handshake incomplete, send/recv mismatch | All multi-chip with sockets | 02, Scenarios 5.2.5--5.2.7 |
| Topology misconfiguration | Wrong device count, incorrect ring ordering, missing mesh connections | T3K, Galaxy | 03, Scenarios 5.3.1--5.3.3 |
| Routing / control plane error | Packets loop or drop, `ControlPlane` routing mismatch | Galaxy with fabric | 03, Scenarios 5.3.2--5.3.4 |
| MMIO/remote device init failure | Remote device operations hang because tunnel not established | Multi-chip (esp. Galaxy) | 03, Scenario 5.3.4 |
| Multi-host sync failure | Cross-host Ethernet link down, barrier timeout | Multi-host Galaxy | 03, Scenarios 5.3.7--5.3.9 |

---

**Next:** [`01_ethernet_and_fabric_fundamentals.md`](./01_ethernet_and_fabric_fundamentals.md)
