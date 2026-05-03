# 5.1 Ethernet and Fabric Hang Fundamentals

[Previous: Chapter 5 Index](./index.md) | [Next: 5.2 CCL Collective Operation Hangs](./02_ccl_collective_operation_hangs.md)

---

Every multi-chip hang on Tenstorrent hardware ultimately traces back to the Ethernet fabric: the physical links that connect chips, the ERISC cores that manage those links, and the Ethernet Data Mover (EDM) router firmware that forwards packets between chips. This section documents the Ethernet core architecture, the EDM router internals, and every known fabric-level hang scenario -- from link failures and handshake deadlocks to flow control stalls and telemetry-detected anomalies.

**Prerequisites:** See [Chapter 5 Index prerequisites](./index.md#prerequisites).

**Reference files:**
- `tt_metal/hw/inc/api/debug/eth_link_status.h` -- `WATCHER_CHECK_ETH_LINK_STATUS()` macro, `hang_on_down_link()` behavior
- `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` -- EDM router main loop
- `tt_metal/fabric/hw/inc/edm_fabric/edm_handshake.hpp` -- EDM handshake protocol (`MAGIC_HANDSHAKE_VALUE`)
- `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp` -- `enable_deadlock_avoidance` compile-time arg
- `tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp` -- deadlock avoidance in packet transmission
- `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp` -- sender channel flow control
- `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp` -- `distance_behind()` function
- `tt_metal/hw/inc/hostdev/fabric_telemetry_msgs.h` -- `FabricTelemetry`, `RouterState`, heartbeat counters
- `tt_metal/impl/debug/watcher_device_reader.cpp` -- `logical_core_to_eth_link_retraining_count` tracking

## Part 1: ERISC Core Architecture

### Active vs. Idle Ethernet Cores

Tenstorrent Ethernet cores come in two categories:

- **AERISC (Active ERISC)**: Ethernet cores connected to active links that have completed link training. These cores run the EDM router firmware and handle actual cross-chip packet traffic. On Blackhole, each active Ethernet port can have two AERISC processors (AERISC 0 and AERISC 1), enabling higher throughput through coordinated context switching.

- **IERISC (Idle ERISC)**: Ethernet cores whose links are not connected or did not complete training. These cores are available for running user kernels but cannot participate in fabric routing.

Only AERISC cores participate in the fabric, and only AERISC cores are monitored by the watcher's Ethernet link status check. If a link that was active at initialization goes down during operation, the AERISC core transitions to a deliberate hang state.

### The Subordinate ERISC Model (Blackhole)

On Blackhole, each Ethernet port has two ERISC processors. The EDM firmware coordinates them via the `CoordinatedEriscContextSwitchState` protocol using stream scratch registers:

```cpp
enum class CoordinatedEriscContextSwitchState : uint32_t {
    NORMAL_EXECUTION = 0,  // Default: both ERISCs executing normally
    RETRAIN_INTENT = 1,    // Master signals intent to begin retrain
    INTENT_ACK = 2,        // Subordinate acknowledges retrain intent
    RETRAIN_COMPLETE = 3,  // Master signals retrain is done
    COMPLETE_ACK = 4,      // Subordinate acknowledges completion
};
```

When the master ERISC (AERISC 0) needs to retrain the link, it must coordinate with the subordinate (AERISC 1) through this state machine. A hang occurs if either side fails to transition states correctly, leaving one ERISC waiting indefinitely for a state that the other never sets.

### The EDM Handshake Protocol

Before any payload data can flow, both sides of an Ethernet link must complete a handshake using `edm_handshake.hpp`:

```cpp
static constexpr uint32_t MAGIC_HANDSHAKE_VALUE = 0xAA;

// Sender side: repeatedly sends magic value until receiver responds
while (handshake_info->local_value != MAGIC_HANDSHAKE_VALUE) {
    if (count == HS_CONTEXT_SWITCH_TIMEOUT) {
        count = 0;
        run_routing();  // Allow context switch
    } else {
        count++;
        internal_::eth_send_packet(0, scratch_addr, local_val_addr, 1);
    }
    invalidate_l1_cache();
}
```

The handshake also exchanges identity information: each side populates `neighbor_mesh_id` and `neighbor_device_id` so the router knows which mesh and device it is connected to. A permanent stall in this handshake produces `RouterState::INITIALIZING` in telemetry.

### Flow Control: The 5-Counter Protocol

EDM-to-EDM flow control uses stream registers to track buffer slot availability across the Ethernet link. Five counters govern the protocol:

| Counter | Maintained By | Purpose |
|---------|---------------|---------|
| `to_receiver_packets_sent` | Sender | Number of buffer slots written to receiver (incremented per packet) |
| `to_sender_0_packets_acked` | Receiver | Packets from sender channel 0 that receiver has seen |
| `to_sender_1_packets_acked` | Receiver | Packets from sender channel 1 that receiver has seen |
| `to_sender_0_packets_completed` | Receiver | Packets from sender channel 0 fully processed |
| `to_sender_1_packets_completed` | Receiver | Packets from sender channel 1 fully processed |

The receiver processes packets through four pointer stages: `ackptr` (receipt) -> `wr_sent_ptr` (NOC write initiated) -> `wr_flush_ptr` (NOC write flushed) -> `completion_ptr` (completion ack sent). A hang occurs whenever any pointer cannot advance because its trailing condition is never satisfied.

---

## Part 2: Hang Scenarios

### 5.1.1 Ethernet Link Down During Active Operation

**Symptom:** All operations on a multi-chip mesh hang simultaneously. The watcher reports `link_down = 1` on one or more Ethernet cores. The affected ERISC core's go message shows `RUN_MSG_DONE`, indicating it exited to base firmware. Other cores across multiple chips are stuck at `noc_semaphore_wait` or EDM flow control spins, waiting for data that will never arrive from the downed link.

**Root Cause:** The Ethernet link between two chips physically goes down (cable issue, transceiver failure, signal integrity degradation). The ERISC core detects this via `is_link_up()` returning false in the `WATCHER_CHECK_ETH_LINK_STATUS()` macro. The core then calls `hang_on_down_link()`:

```c
// From tt_metal/hw/inc/api/debug/eth_link_status.h
void hang_on_down_link() {
    debug_eth_link_t tt_l1_ptr* v = GET_MAILBOX_ADDRESS_DEV(watcher.eth_status);
    v->link_down = 1;

    volatile tt_l1_ptr go_msg_t* go_message_ptr = GET_MAILBOX_ADDRESS_DEV(go_messages[0]);
    go_message_ptr->signal = RUN_MSG_DONE;

    internal_::disable_erisc_app();
#if (defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)) || !defined(ARCH_BLACKHOLE)
    erisc_exit();
#endif
    while (1) { ; }  // Deliberate infinite loop
}
```

**Platform note:** On Blackhole, only AERISC0 calls `erisc_exit()`. AERISC1 enters the `while(1)` loop without the full exit sequence, which may leave it in a partially disabled state (see Scenario 5.1.11).

**Diagnosis Steps:**
1. Check watcher output for `link_down = 1` on any Ethernet core.
2. Check fabric telemetry for `RouterState::RETRAINING` or absent heartbeats on the affected link direction.
3. Use `tt-smi` to verify physical link status and check for cable errors.
4. Cross-reference the downed link with the routing table to identify which chip-to-chip paths are broken.

**Fix:** If the link is intermittently unstable, enable the `skip_eth_cores_with_retrain` runtime option (set env var `TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1`). For permanent link failures, reduce the mesh to exclude the affected chips or replace the cable/transceiver.

**Prevention:**
- Monitor `logical_core_to_eth_link_retraining_count` via watcher for early warning of degrading links.
- Use fabric telemetry heartbeat monitoring (TX/RX heartbeats) to detect link health before operations begin.
- In Galaxy configurations, run pre-flight link health checks before launching large-scale workloads.

---

### 5.1.2 ERISC Context Switch Deadlock (AERISC/IERISC Coordination Failure)

**Symptom:** Both ERISC cores (AERISC0 and AERISC1 on Blackhole) on the same Ethernet port are stuck. One is polling for a state transition that the other never provides. The watcher shows one core at a spin-loop inside the coordinated context switch handshake.

**Root Cause:** The coordinated context switch protocol between AERISC 0 (master) and AERISC 1 (subordinate) has desynchronized. The five-state protocol requires both cores to transition in strict lockstep. If one core misses a state transition (e.g., due to an L1 cache coherence issue on Blackhole, or because one core was buried in a long-running packet processing loop):

```
NORMAL_EXECUTION -> RETRAIN_INTENT -> INTENT_ACK -> RETRAIN_COMPLETE -> COMPLETE_ACK
```

For example, if AERISC0 sets `RETRAIN_INTENT` but AERISC1 never reaches the polling point because it is stuck in a long EDM forwarding operation, the retrain cannot proceed and AERISC0 hangs. The subordinate's L1 cache may contain stale data showing `NORMAL_EXECUTION`, preventing it from ever seeing the intent signal.

**Diagnosis Steps:**
1. Read the `CoordinatedEriscContextSwitchState` from both ERISC cores on the affected port.
2. Compare states: if AERISC0 shows `RETRAIN_INTENT` (1) and AERISC1 shows `NORMAL_EXECUTION` (0), the subordinate missed the intent signal.
3. Check if `invalidate_l1_cache()` was called before reading the state register on the subordinate side.
4. Check AERISC1's waypoint to determine what it is doing (likely stuck in a long EDM operation).

**Fix:** This is a firmware-level issue. Ensure both ERISC cores call `invalidate_l1_cache()` before reading the shared context switch state register, and that the subordinate's main loop checks for `RETRAIN_INTENT` at sufficient frequency (not buried behind a long-running packet processing loop).

**Prevention:**
- Blackhole firmware should use volatile pointers or explicit cache invalidation for all shared state between the two ERISC cores.
- Monitor for `RouterState::RETRAINING` in telemetry to detect when context switches are in progress.
- The context switch timeout provides a fallback, but it is extremely long. Consider reducing it for faster detection.

---

### 5.1.3 Link Retraining During Active Data Transfer

**Symptom:** Intermittent hangs on multi-chip operations. The watcher reports increasing values in `logical_core_to_eth_link_retraining_count` for one or more Ethernet cores. Operations sometimes complete, sometimes hang, with no obvious pattern.

**Root Cause:** Ethernet link retraining is the physical-layer process of re-establishing a link after signal degradation. During retraining, data transfer is suspended. If an EDM sender channel has already written data to the link but the receiver has not acknowledged it, the retraining disrupts the flow control protocol. After retraining completes, the sender's `wrptr` may be ahead of the receiver's `ackptr`, creating a phantom "data in flight" state, and transaction ID tracking (`NUM_TRANSACTION_IDS`, which is 8 when `enable_deadlock_avoidance` is true, 4 otherwise) may become inconsistent.

The watcher tracks retraining events by reading `RETRAIN_COUNT` from each active Ethernet core on device open and device close:
```cpp
// From watcher_device_reader.cpp -- on WatcherDeviceReader destruction
for (const CoreCoord& eth_core : get_active_ethernet_cores(device_id)) {
    read_data = read_core(device_id, virtual_core, RETRAIN_COUNT_ADDR, sizeof(uint32_t));
    uint32_t num_events = read_data[0] - logical_core_to_eth_link_retraining_count[eth_core];
    if (num_events > 0) {
        log_warning("Device {} virtual ethernet core {}: "
                    "Watcher detected {} link retraining events.",
                    device_id, virtual_core, num_events);
    }
}
```

**Diagnosis Steps:**
1. Enable watcher and check for retraining event counts at session end.
2. Check if the hang correlates with specific link directions by inspecting which Ethernet cores show retraining.
3. Use fabric telemetry to correlate `RouterState::RETRAINING` with the time of the hang.

**Fix:** Enable `TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1` to route around unstable links. If the problem persists, the link may need hardware intervention (cable replacement, transceiver check).

**Prevention:**
- Pre-screen links before workloads: run a short fabric stress test and check retraining counts.
- Run the fabric deadlock stability tests (see [Scenario 5.2.12](./02_ccl_collective_operation_hangs.md#5212-deadlock-stability-test-failures-indicating-latent-bugs)) to validate link stability under load.
- In production deployments, monitor retraining counts continuously and proactively remove unstable links.

---

### 5.1.4 EDM Handshake Failure During Initialization

**Symptom:** Fabric initialization hangs. The fabric telemetry shows `RouterState::INITIALIZING` on one or more Ethernet cores that never transitions to `RouterState::RUNNING`. The host may time out waiting for fabric to become ready.

**Root Cause:** The EDM handshake protocol requires both sides of a link to exchange `MAGIC_HANDSHAKE_VALUE` (0xAA). The sender repeatedly sends `eth_send_packet` with the magic value until it receives the magic value back. If the remote side is not running (not yet initialized, crashed, or running different firmware), the handshake spins indefinitely:

```cpp
// The handshake spin-loop from edm_handshake.hpp
while (handshake_info->local_value != MAGIC_HANDSHAKE_VALUE) {
    if (count == HS_CONTEXT_SWITCH_TIMEOUT) {
        count = 0;
        run_routing();  // Context switch to allow other processing
    } else {
        count++;
        eth_send_packet(0, scratch_addr, local_val_addr, 1);
    }
    invalidate_l1_cache();
}
```

This commonly occurs when:
- Devices are initialized in the wrong order (remote device not yet booted when MMIO device starts fabric).
- A previous test left fabric routers in a stale state (see `FabricSwitchManager` teardown requirement in Section 5.3).
- One side of the link had a firmware crash during initialization.

**Diagnosis Steps:**
1. Check `FabricTelemetry::static_info` for the affected core. If `version` is not `FABRIC_TELEMETRY_VERSION`, the telemetry struct was never initialized.
2. Read the `handshake_info_t::local_value` from both sides of the link. If one side shows `0` and the other is attempting sends, the non-zero side is the one waiting.
3. Check initialization order: was the remote device fully booted before fabric initialization began?

**Fix:**
```cpp
// BUGGY: FabricSwitchManager not torn down between tests
// Test 1 completes, leaves routers running
// Test 2 starts, remote routers expect fresh handshake but never get one

// CORRECTED: Always teardown FabricSwitchManager between tests
void run_test() {
    auto& switch_mgr = FabricSwitchManager::instance();
    switch_mgr.setup(fabric_config);
    // ... run test ...
    switch_mgr.teardown();  // CRITICAL: close devices for proper re-handshake
}
```

**Prevention:**
- Always call `FabricSwitchManager::teardown()` between workloads (see [Scenario 5.3.3](./03_topology_and_mesh_configuration_hangs.md#533-fabricswitchmanager-teardown-failure-between-tests) for the full teardown protocol).
- Ensure device initialization follows the correct order: MMIO devices first, then remote devices from closest to farthest.

---

### 5.1.5 EDM Router Receiver Stalled -- Forwarding Deadlock

**Symptom:** The EDM router on one chip stops forwarding packets. Upstream sender channels fill up and block, causing a cascade of stalls across the fabric. The fabric telemetry shows `tx_heartbeat` incrementing (the sender is trying to send) but `rx_heartbeat` has stopped (the receiver is not processing). The `RouterState` remains `RUNNING` but no progress is made.

**Root Cause:** The receiver processes packets through a four-stage pipeline: acknowledge receipt (`ackptr` advances) -> initiate NOC writes (`wr_sent_ptr` advances) -> wait for NOC writes to flush (`wr_flush_ptr` advances) -> send completion notification (`completion_ptr` advances). A forwarding deadlock occurs when the receiver initiates a NOC write to forward a packet to a downstream EDM, but that downstream EDM's receiver buffer is full -- because *it* is also trying to forward to yet another chip. This creates a circular dependency.

**Diagnosis Steps:**
1. Read the five stream-register counters for each EDM channel on the suspected link.
2. Check if `wr_sent_ptr == wr_flush_ptr` (writes initiated but not flushing) or if `ackptr >> completion_ptr` (packets acknowledged but not completed, meaning forwarding is stalled).
3. Trace the downstream path to find the circular dependency.

**Fix:** This is the exact scenario that the `enable_deadlock_avoidance` compile-time argument addresses. When enabled, the EDM uses a "bubble flow control" protocol:

```cpp
// From fabric_edm_packet_transmission.hpp:
FORCE_INLINE void flush_write_to_noc_pipeline(uint8_t rx_channel_id) {
    if constexpr (enable_deadlock_avoidance) {
        auto start_trid = RX_CH_TRID_STARTS[rx_channel_id];
        auto end_trid = start_trid + NUM_TRANSACTION_IDS;  // 8 when DA enabled, 4 otherwise
        for (int i = start_trid; i < end_trid; i++) {
            while (!ncrisc_noc_nonposted_write_with_transaction_id_flushed(..., i));
        }
    }
}
```

**Prevention:**
- Always enable `enable_deadlock_avoidance` for topologies with cycles (Ring, Torus). Linear topologies cannot form circular dependencies and may disable it for performance.
- Note: `super_speedy_mode` is incompatible with deadlock avoidance (the code has a `static_assert` enforcing this). Do not enable both.

---

### 5.1.6 Deadlock Avoidance Bubble Protocol Failure

**Symptom:** In a 2D fabric topology (Blackhole Galaxy), multiple EDM routers become deadlocked despite `enable_deadlock_avoidance` being set. Receiver channels on several chips are full, each waiting for a downstream EDM to free space. Fabric telemetry shows `RouterState::RUNNING` but zero bandwidth.

**Root Cause:** The EDM router implements deadlock avoidance through a "bubble flow control" protocol that reserves at least one buffer slot empty at all times, preventing circular wait conditions. The mechanism is enabled for "turn channels" -- channels that forward traffic between spine (NORTH/SOUTH) and non-spine (EAST/WEST) directions.

The `need_deadlock_avoidance_support()` function in `fabric_context.cpp` determines when deadlock avoidance is needed:

```cpp
// Deadlock avoidance is required for:
// - All directions on Ring topology
// - EAST/WEST directions on Torus-X topologies (the wrapped dimension that forms cycles)
// - NORTH/SOUTH directions on Torus-Y topologies (the wrapped dimension that forms cycles)
// - Both dimensions on full Torus (both X and Y wrapped)
// Not needed for Mesh, Linear, or NeighborExchange topologies
```

The failure occurs when:
- Deadlock avoidance is not enabled for channels on the wrapped dimension of a Torus topology.
- The `need_deadlock_avoidance_support()` function returns `true` but the EDM channel was configured without `enable_deadlock_avoidance`.
- The reserved bubble slot is consumed by a packet before the avoidance protocol can activate.

**Diagnosis Steps:**
1. Check which channels are marked as turn channels via `get_sender_channel_turn_statuses()`.
2. Verify that `enable_deadlock_avoidance` is set to `true` for all turn channels at compile time.
3. Check the receiver buffer occupancy on all involved routers. If all buffers are full in a cycle, the deadlock is confirmed.
4. Check fabric telemetry bandwidth counters: if `num_words_sent` and `num_packets_sent` are stalled, the deadlock is confirmed.

**Fix:** Ensure the `enable_deadlock_avoidance` template parameter is correctly propagated for all router configurations that involve 2D routing with turns. The EDM router's `sender_channels_turn_status` array must accurately reflect which channels handle traffic that changes direction.

**Prevention:**
- Run the fabric deadlock stability tests (`test_fabric_deadlock_stability_bh_6U_galaxy.yaml`) which validate deadlock freedom under adversarial traffic patterns including all cardinal directions.
- Never disable deadlock avoidance on turn channels in production configurations.

---

### 5.1.7 Fabric Telemetry Heartbeat Stall Detection

**Symptom:** The fabric appears to be running (`RouterState::RUNNING`) but no data is being transferred. Host-side monitoring of fabric telemetry shows that TX or RX heartbeat counters have stopped incrementing, indicating that the router is alive but stuck.

**Root Cause:** The fabric telemetry system provides two heartbeat mechanisms per ERISC core:

```cpp
struct EriscDynamicEntry {
    RouterState router_state;
    RiscTimestampV2 tx_heartbeat;  // Incremented when sender queues empty or packet sent
    RiscTimestampV2 rx_heartbeat;  // Incremented when receiver queues empty or packet forwarded
};
```

A stalled TX heartbeat with a running RX heartbeat means the sender side is blocked -- typically waiting for flow control credits from the remote receiver. A stalled RX heartbeat with a running TX heartbeat means the receiver side is blocked -- typically waiting for NOC writes to complete (forwarding stall, see Chapter 3, Section 02 for DRAM backpressure).

The `BandwidthTelemetry` struct provides additional diagnostic data:
```cpp
struct BandwidthTelemetry {
    RiscTimestampV2 elapsed_active_cycles;  // Cycles where work was done
    RiscTimestampV2 elapsed_cycles;         // Total cycles (active + idle)
    uint64_t num_words_sent;
    uint64_t num_packets_sent;
};
```

If `elapsed_cycles` is advancing but `elapsed_active_cycles` is not, the router is spinning idle -- a strong indicator of a flow control deadlock.

**Diagnosis Steps:**
1. Sample `FabricTelemetry.dynamic_info` from the host at two time points.
2. Compare heartbeat deltas: `delta(tx_heartbeat)` and `delta(rx_heartbeat)`.
3. If both are zero but `router_state == RUNNING`, the router is deadlocked.
4. Check `delta(num_words_sent)` and `delta(num_packets_sent)` to confirm no data flow.
5. Check `delta(elapsed_active_cycles) / delta(elapsed_cycles)` for utilization. Near-zero confirms deadlock.

**Fix:** The specific fix depends on the underlying cause (forwarding deadlock per 5.1.5, link issues per 5.1.1, etc.). Telemetry provides the diagnostic signal; the resolution comes from addressing the root cause identified through the counters.

**Prevention:**
- Integrate fabric telemetry sampling into the host-side monitoring loop.
- Set heartbeat stall thresholds: if no heartbeat increment after N seconds, trigger a diagnostic dump.
- Configure `FabricTelemetrySettings` to enable telemetry collection with appropriate overhead.

---

### 5.1.8 Flow Control Semaphore Desynchronization Between Sender and Receiver

**Symptom:** The EDM sender channel stops sending packets even though there is data in its buffer. The sender appears to believe the remote receiver has no space, but the receiver's buffer is actually empty. No progress on any data path that routes through this link.

**Root Cause:** The sender channel tracks remote receiver capacity through stream register counters. If a counter update is lost (due to an Ethernet transient error that does not trigger a full link retraining) or if the counters wrap incorrectly, the sender's view of available space diverges from reality.

The `distance_behind` function in `edm_fabric_flow_control_helpers.hpp` computes available space:

```cpp
FORCE_INLINE uint8_t distance_behind(
    const BufferPtr& trailing_ptr,
    const BufferPtr& leading_ptr,
    uint8_t ptr_wrap_size) {
    bool leading_gte_trailing_ptr = leading_ptr >= trailing_ptr;
    return leading_gte_trailing_ptr
        ? leading_ptr - trailing_ptr
        : ptr_wrap_size - (trailing_ptr - leading_ptr);
}
```

If `ptr_wrap_size` does not match the actual number of buffer slots (due to a configuration mismatch between sender and receiver), the wrapping arithmetic produces incorrect results, causing either buffer overflow (data corruption) or indefinite stall.

**Diagnosis Steps:**
1. Read the five flow control counters on both the sender and receiver side.
2. Verify that `NUM_BUFFERS` is consistent between the sender's `SenderEthChannel<HEADER_TYPE, NUM_BUFFERS>` template parameter and the receiver's buffer allocation.
3. Check the `channel_buffer_size` compile-time argument on both sides.

**Fix:** Ensure compile-time arguments for buffer count and buffer size are identical on both sides of every Ethernet link. The fabric setup code must propagate these consistently.

**Prevention:**
- The `static_assert(sender_channel_free_slots_stream_ids[0] == 22)` etc. assertions in `fabric_erisc_router.cpp` validate stream register assignments at compile time. Ensure these are not bypassed.
- Run the fabric deadlock stability tests (see [Scenario 5.2.12](./02_ccl_collective_operation_hangs.md#5212-deadlock-stability-test-failures-indicating-latent-bugs)).

---

### 5.1.9 eth_txq_is_busy Spin-Loop Deadlock

**Symptom:** An ERISC core is stuck in a tight spin-loop. The watcher waypoints show no progress. Reading the core's program counter reveals it is inside the `send_next_data` function, spinning on `eth_txq_is_busy`.

**Root Cause:** The EDM router's send path has two `eth_txq_is_busy` spin-loops:

```cpp
// From fabric_erisc_router.cpp
template <...>
FORCE_INLINE void send_next_data(...) {
    if constexpr (ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA) {
        while (internal_::eth_txq_is_busy(sender_txq_id)) { };  // SPIN 1
    }
    internal_::eth_send_packet_bytes_unsafe(sender_txq_id, src_addr, dest_addr, payload_size_bytes);

    while (internal_::eth_txq_is_busy(sender_txq_id)) { };  // SPIN 2
    remote_update_ptr_val<to_receiver_pkts_sent_id, sender_txq_id>(1U);
}
```

A permanent stall occurs when:
- The remote receiver is not consuming packets (crashed, reset, or link down but not yet detected).
- The Ethernet PHY's transmit FIFO is full and hardware flow control signals are asserted.
- The `sender_txq_id` is wrong, causing the check to poll the wrong transmit queue.

**Diagnosis Steps:**
1. Read the `eth_txq_is_busy` status register for the specific `sender_txq_id`. If it permanently returns `1`, the hardware transmit queue is stuck.
2. Check the remote side of the link. If the remote ERISC is not running, the transmit queue will remain busy.
3. Check if `is_link_up()` still returns true. The link may be electrically up but the remote firmware is not responding.

**Fix:**
- If the remote ERISC crashed: the link down detection via `WATCHER_CHECK_ETH_LINK_STATUS()` should eventually catch this, but only if the watcher check is called frequently enough.
- If the transmit queue hardware is stuck: device reset is required.

**Prevention:**
- Ensure the EDM main loop calls `WATCHER_CHECK_ETH_LINK_STATUS()` on every iteration, not just when progress is made.
- The `did_something` flag in the router main loop controls whether a context switch is allowed. Verify the threshold is not set too high.

---

### 5.1.10 Multi-TXQ Credit Update Race (Blackhole)

**Symptom:** On Blackhole with `multi_txq_enabled`, packet acknowledgements are lost intermittently. The sender side shows packets sent but the corresponding `ack_counters` or `completion_counters` on the receiver side are behind. The sender eventually stalls because it believes the receiver has no free slots.

**Root Cause:** Blackhole supports multiple transmit queues (`multi_txq_enabled`), which changes the credit update mechanism from stream-register-based to counter-based:

```cpp
struct ReceiverChannelCounterBasedResponseCreditSender {
    FORCE_INLINE void send_completion_credit(uint8_t src_id, uint32_t num_completions) {
        completion_counters[src_id] += num_completions;
        completion_counters_base_ptr[src_id] = completion_counters[src_id];
        update_sender_side_credits();  // eth_send_packet_bytes_unsafe
    }

    FORCE_INLINE void send_ack_credit(uint8_t src_id) {
        ack_counters[src_id]++;
        ack_counters_base_ptr[src_id] = ack_counters[src_id];
        update_sender_side_credits();
    }
};
```

If two credit updates are initiated in rapid succession (e.g., ack followed immediately by completion), and the Ethernet transmit queue serializes them, the second send may overwrite the first before it is transmitted. The code comments warn: "Assumes `!eth_txq_is_busy()` -- PLEASE CHECK BEFORE CALLING."

**Diagnosis Steps:**
1. Read the local `ack_counters` and `completion_counters` arrays on the receiver side.
2. Read the remote counter values on the sender side.
3. If the sender-side values are behind the receiver-side values, a credit update was lost in transit.
4. Check whether `eth_txq_is_busy()` is being checked before each credit send call.

**Fix:** Always check `eth_txq_is_busy()` before calling `send_ack_credit()` or `send_completion_credit()`. The stream-register-based alternative (`ReceiverChannelStreamRegisterFreeSlotsBasedCreditSender`) uses `remote_update_ptr_val` which has its own serialization guarantees.

**Prevention:**
- When using `multi_txq_enabled`, audit all call sites of `send_ack_credit` and `send_completion_credit` for `eth_txq_is_busy()` guards.
- Use the stream-register-based credit sender when possible, as it does not have this race condition.

---

### 5.1.11 Ethernet Send/Receive Asymmetry on Blackhole

**Symptom:** On Blackhole devices, the EDM router works correctly in one direction (e.g., East) but hangs when sending in the opposite direction (West) on the same physical Ethernet link. This manifests as unidirectional fabric connectivity.

**Root Cause:** Blackhole has two AERISC cores per Ethernet port (PHYSICAL_AERISC_ID 0 and 1), unlike Wormhole which has one. The `hang_on_down_link()` function has Blackhole-specific behavior: only AERISC0 calls `erisc_exit()`, governed by the preprocessor guard `#if (defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)) || !defined(ARCH_BLACKHOLE)`. If AERISC1 encounters a link-down condition, it enters the `while(1)` loop without `erisc_exit()`, leaving it in a partially disabled state.

Additionally, if compile-time arguments for the two cores are not consistent, or if the coordinated context switch protocol (5.1.2) fails, one direction works while the other does not.

**Diagnosis Steps:**
1. Check both ERISC cores on the affected Ethernet port.
2. Verify that compile-time arguments (especially `enable_deadlock_avoidance`, `channel_buffer_size`) are consistent between the two cores.
3. Check the `CoordinatedEriscContextSwitchState` for any stuck state transitions.

**Fix:** Ensure both ERISC cores are configured identically for the same Ethernet port. The compile-time argument at `MAIN_CT_ARGS_START_IDX + 2` must match on both cores.

**Prevention:**
- The fabric compilation pipeline should enforce argument consistency between paired ERISC cores.
- Run bidirectional fabric stress tests on all links before production workloads.

---

### 5.1.12 EDM Worker Handshake NOC Mismatch

**Symptom:** A worker kernel connects to an EDM sender channel, but the flow control semaphore increments never arrive at the worker. The worker spins waiting for permission to write its first packet, and the EDM channel shows the connection as established but never receives data.

**Root Cause:** The EDM sender channel sends flow control credits to the worker using a fixed NOC path (`WORKER_HANDSHAKE_NOC`). From `fabric_erisc_datamover_channels.hpp`:

```cpp
noc_semaphore_inc<posted>(worker_semaphore_address, 1, WORKER_HANDSHAKE_NOC);
```

The packet header documentation warns: "ALL PACKETS MUST CONTAIN DESTINATION NOC X/Y AS NOC 0 COORDINATES, REGARDLESS OF THE `noc_index` OF THE SENDER." If a worker provides NOC 1 coordinates when the EDM expects NOC 0, the semaphore increment targets the wrong physical core.

**Diagnosis Steps:**
1. Check the worker's semaphore address and core coordinates stored in the EDM channel's `worker_location_info_ptr`.
2. Verify these are NOC 0 coordinates.
3. Read the semaphore value at the stored address -- if it is 0, the credits are being sent to the wrong location.

**Fix:**
```cpp
// BUGGY: Worker provides NOC 1 coordinates to EDM
auto my_noc1_coords = get_noc_addr(noc_x, noc_y, sem_addr, 1);  // NOC 1
edm_sender.register_worker(my_noc1_coords);

// CORRECTED: Worker must always provide NOC 0 coordinates
auto my_noc0_coords = get_noc_addr(noc_x, noc_y, sem_addr, 0);  // NOC 0
edm_sender.register_worker(my_noc0_coords);
```

**Prevention:**
- Use the `EdmToEdmSender` wrapper from `edm_fabric_worker_adapters.hpp` which handles NOC coordinate translation correctly.
- Add assertions that verify coordinates are in NOC 0 space before writing them to the EDM channel.

---

## Summary Table

| Scenario | Hang Indicator | Primary Component | Affected Archs |
|----------|---------------|-------------------|----------------|
| 5.1.1 Link Down | Watcher `link_down = 1` | Ethernet PHY | WH, BH, Quasar |
| 5.1.2 Context Switch Deadlock | Both ERISCs stuck, handshake state mismatch | AERISC/IERISC coordination | BH only |
| 5.1.3 Link Retraining | Increasing `retraining_count` | Ethernet PHY | All |
| 5.1.4 EDM Handshake Failure | `RouterState::INITIALIZING` permanently | EDM handshake protocol | All |
| 5.1.5 Forwarding Deadlock | `ackptr >> completion_ptr`, no heartbeat | EDM flow control | All |
| 5.1.6 DA Bubble Protocol Failure | All receiver buffers full in cycle | Deadlock avoidance | BH (2D fabric) |
| 5.1.7 Heartbeat Stall | TX/RX heartbeat delta = 0, state = RUNNING | Fabric telemetry | All |
| 5.1.8 FC Desynchronization | Sender sees no space, receiver is empty | Stream register counters | All |
| 5.1.9 eth_txq_is_busy Deadlock | ERISC stuck in tight spin, no waypoints | Ethernet transmit queue | All |
| 5.1.10 Multi-TXQ Credit Race | Sender ack/completion counters behind | Counter-based credit sender | BH only |
| 5.1.11 BH Send/Recv Asymmetry | One direction works, other hangs | Dual AERISC on Blackhole | BH only |
| 5.1.12 NOC Mismatch | Worker waits for FC credit forever | Worker-to-EDM handshake | All |

---

[Previous: Chapter 5 Index](./index.md) | [Next: 5.2 CCL Collective Operation Hangs](./02_ccl_collective_operation_hangs.md)
