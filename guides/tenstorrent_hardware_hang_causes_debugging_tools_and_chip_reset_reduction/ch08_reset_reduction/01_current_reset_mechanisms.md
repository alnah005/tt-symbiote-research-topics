# 01 -- Current Reset Mechanisms: The 5-Level Reset Hierarchy

## Summary

This section documents every reset mechanism available in the Tenstorrent stack, organized from least disruptive (graceful program termination) to most disruptive (full system reboot). For each level, we specify the exact code path, what state is destroyed and preserved, the measured or configured timing, and the conditions under which that level is the minimum necessary intervention. The section covers both the UMD-level API that developers interact with and the kernel driver (tt-kmd) internals that ensure resets are safe in multi-process environments. Understanding this hierarchy is essential for reset reduction: the goal is to always use the *least disruptive* mechanism that restores the system to a known-good state.

## Prerequisites

- Chapter 1 ([`01_what_is_a_hang.md`](../ch01_anatomy_of_a_hang/01_what_is_a_hang.md)): The hang lifecycle and architecture differences.
- Chapter 4 ([`02_host_synchronization_and_timeout_detection.md`](../ch04_dispatch_and_host_device_hangs/02_host_synchronization_and_timeout_detection.md)): Dispatch architecture and `DeviceManager`.
- Chapter 5 ([`01_ethernet_and_fabric_fundamentals.md`](../ch05_multi_chip_and_ccl_hangs/01_ethernet_and_fabric_fundamentals.md)): Multi-chip topology and fabric shutdown.
- Chapter 6 ([`01_watcher_system.md`](../ch06_debugging_tools/01_watcher_system.md)): Watcher and tt-triage as diagnostic tools.

---

## 1. The Reset Hierarchy

The five levels, from least to most destructive:

```
Level 0: Graceful Program Termination     (~ms, no hardware reset)
Level 1: Tensix Per-Core Soft Reset        (~us per core, resets individual RISCs)
Level 2: UMD Warm Reset (ASIC-level)       (~2-20s, resets entire chip)
Level 3: M3/DMC Board-Level Reset          (~20-30s, resets board management controller)
Level 4: Full System Reboot                (~minutes, re-enumerates PCIe)
```

Each level subsumes the previous: a warm reset (Level 2) also resets all individual cores (Level 1), and a full reboot (Level 4) subsumes everything. The key decision at each hang is: *what is the minimum level required?*

### State Destruction by Reset Level

| State Category | Level 0 | Level 1 | Level 2 | Level 3 | Level 4 |
|---|---|---|---|---|---|
| RISC-V register file / PC | Preserved | **Destroyed** | Destroyed | Destroyed | Destroyed |
| L1 SRAM contents | Cleared by close | Preserved | **Destroyed** | Destroyed | Destroyed |
| DRAM contents | Preserved | Preserved | Destroyed (retraining) | **Destroyed** | Destroyed |
| NOC interface state | Preserved | Preserved | **Destroyed** | Destroyed | Destroyed |
| Ethernet links | Preserved | Preserved | **Retrained** | Retrained | Destroyed |
| ARC firmware | Preserved | Preserved | Reloaded | **Destroyed** | Destroyed |
| PCIe link | Preserved | Preserved | Retrained | Preserved | **Destroyed** |
| Host process state | Terminated | Preserved | Preserved | Preserved | **Destroyed** |
| Watcher diagnostic data | Lost at close | Preserved | **Destroyed** | Destroyed | Destroyed |

---

## 2. Level 0: Graceful Program Termination

**What it does:** The host process exits normally or is killed (SIGTERM/SIGKILL), and the runtime's `DeviceManager` destructor runs the ordered shutdown sequence.

**Code path:** `DeviceManager::~DeviceManager()` in `tt_metal/impl/device/device_manager.cpp`:

```cpp
// tt_metal/impl/device/device_manager.cpp
DeviceManager::~DeviceManager() {
    for (const auto& dev : this->devices_) {
        if (dev != nullptr and dev->is_initialized()) {
            dev->close();
        }
    }
    this->devices_.clear();
    init_done_.clear();
    initializers_.clear();
    descriptor_.reset();
}
```

The `close_devices` method handles ordered shutdown for multi-chip configurations. It performs firmware teardown in a specific order:

1. **Dispatch kernel shutdown** (`DispatchKernelInitializer::teardown`)
2. **Fabric firmware shutdown** (`FabricFirmwareInitializer::teardown`)
3. **Profiler shutdown** (`ProfilerInitializer::teardown`)
4. **Command queue shutdown** (`CommandQueueInitializer::teardown`)
5. **Post-teardown cleanup** for each initializer
6. **Individual device close** for each device

**Ordered shutdown for multi-chip:** The `close_devices` method implements a critical ordering requirement: remote devices (accessed via Ethernet tunnels) must be closed *before* the MMIO device that provides their PCIe path. Tunnels are shut down from the farthest device to the closest:

```cpp
// tt_metal/impl/device/device_manager.cpp
// Iterate over all tunnels originating from this MMIO device
for (auto t : tunnels_from_mmio) {
    // Iterate from the farthest tunnel stop back toward MMIO
    for (uint32_t ts = t.size() - 1; ts > 0; ts--) {
        if (this->is_device_active(t[ts])) {
            devices_to_close.push_back(t[ts]);
        }
    }
}
devices_to_close.push_back(mmio_device_id);
```

This ordering is critical for avoiding hangs during shutdown itself. Closing an MMIO device while its remote devices still have active dispatch kernels trying to send messages through the fabric would deadlock the shutdown process.

**Driver-level mechanism (tt-kmd):** On file descriptor close, the kernel driver (`chardev.c`, `tt_cdev_release()`) performs cleanup:
- Executes any registered NOC cleanup action (`noc_cleanup` field in `chardev_private`), writing a single 32-bit value to a specified NOC address -- this can signal device-side firmware that the host has departed
- Releases all held resource locks (ERISC core locks via `resource_lock` bitmap)
- Frees all allocated TLBs
- Decrements the open count; when the last fd closes and `power_policy` is enabled, the driver re-aggregates power state to low-power

The NOC cleanup mechanism is registered via `TENSTORRENT_IOCTL_SET_NOC_CLEANUP`:

```c
struct tenstorrent_set_noc_cleanup {
    __u32 argsz;
    __u32 flags;
    __u8 enabled;
    __u8 x, y, noc;
    __u32 reserved0;
    __u64 addr;
    __u64 data;
};
```

This is a targeted resilience mechanism: the tt-metal runtime registers a cleanup that signals device-side firmware that the host process has departed, allowing the firmware to enter a clean shutdown path rather than hanging indefinitely waiting for host commands that will never arrive.

**What is preserved:** All on-chip state remains accessible -- firmware continues running, no hardware reset occurs. The device is immediately available for the next program.

**What is destroyed:** Program cache, allocator state, command queues are cleared during `Device::close()`.

**Timing:** Milliseconds for single-chip; can extend to seconds for Galaxy (32+ chip) configurations where fabric teardown is involved.

**When sufficient:** The workload has completed or errored, and you want a clean slate for the next run. Also sufficient when a timeout occurred but the chip is still responsive to PCIe reads.

**When insufficient:** The device is truly hung -- `Device::close()` itself hangs because the dispatch or fabric firmware does not respond to shutdown commands.

---

## 3. Level 1: Tensix Per-Core Soft Reset

**What it does:** Resets individual RISC-V processors within a single Tensix core by writing to the `RISCV_DEBUG_REG_SOFT_RESET_0` register. This allows resetting one core without affecting other cores on the same chip.

**Code path:** The UMD-level API is exposed through `Cluster::deassert_risc_reset_at_core()` and `Cluster::assert_risc_reset_at_core()` in `tt_metal/llrt/tt_cluster.cpp`:

```cpp
// tt_metal/llrt/tt_cluster.cpp
void Cluster::deassert_risc_reset_at_core(
    const tt_cxy_pair& core, const tt::umd::RiscType& soft_resets,
    bool staggered_start) const {
    auto core_coord = this->to_umd_coordinate(core);
    this->driver_->deassert_risc_reset(core.chip, core_coord, soft_resets, staggered_start);
}
```

At the device firmware level, the `RISCV_DEBUG_REG_SOFT_RESET_0` register (address `0xFFB121B0` on Blackhole, defined at `RISCV_DEBUG_REGS_START_ADDR | 0x1B0` in `tt_metal/hw/inc/internal/tt-1xx/blackhole/tensix.h`) controls which RISC-V processors within a Tensix core are held in reset.

The `TensixSoftResetOptions` enum (defined in `tt_metal/third_party/umd/device/api/umd/device/types/tensix_soft_reset_options.hpp`) provides fine-grained control:

```cpp
enum class TensixSoftResetOptions : std::uint32_t {
    NONE           = 0,
    BRISC          = ((std::uint32_t)1 << 11),   // bit 11: 0x00800
    TRISC0         = ((std::uint32_t)1 << 12),   // bit 12: 0x01000
    TRISC1         = ((std::uint32_t)1 << 13),   // bit 13: 0x02000
    TRISC2         = ((std::uint32_t)1 << 14),   // bit 14: 0x04000
    NCRISC         = ((std::uint32_t)1 << 18),   // bit 18: 0x40000
    STAGGERED_START = ((std::uint32_t)1 << 31)
};
```

Pre-defined combinations facilitate common operations:

| Constant | Value | Purpose |
|----------|-------|---------|
| `ALL_TENSIX_SOFT_RESET` | BRISC \| NCRISC \| TRISC0-2 \| STAGGERED_START | Reset all cores with staggered startup |
| `TENSIX_ASSERT_SOFT_RESET` | BRISC \| NCRISC \| TRISC0-2 | Assert reset on all cores (hold in reset) |
| `TENSIX_DEASSERT_SOFT_RESET` | NCRISC \| TRISC0-2 \| STAGGERED_START | Release from reset with staggered start |

Firmware helpers in `tt_metal/hw/inc/internal/tt-1xx/risc_common.h` manipulate this register:

```cpp
inline void deassert_all_reset() {
    WRITE_REG(RISCV_DEBUG_REG_SOFT_RESET_0, RISCV_SOFT_RESET_0_NONE);
}
inline void assert_just_ncrisc_reset() {
    WRITE_REG(RISCV_DEBUG_REG_SOFT_RESET_0, RISCV_SOFT_RESET_0_NCRISC);
}
```

The `invert_selected_options` utility computes which cores should remain in reset when releasing specific ones:

```cpp
TensixSoftResetOptions invert_selected_options(TensixSoftResetOptions selected) {
    uint32_t selected_bits = static_cast<uint32_t>(selected);
    uint32_t inverted = (~selected_bits) & static_cast<uint32_t>(ALL_TENSIX_SOFT_RESET);
    return static_cast<TensixSoftResetOptions>(inverted);
}
```

**What is preserved:** All other cores on the chip remain running. DRAM contents, L1 of other cores, NOC routing state, Ethernet links -- all unaffected. L1 of the target core is also NOT cleared by soft reset -- only the RISC-V processor state (registers, PC) is reset.

**What is destroyed:** The target core's RISC-V processor state. The core begins execution from its reset vector.

**Timing:** Microseconds -- this is a single register write via PCIe MMIO.

**When sufficient:** A single core is hung (e.g., stuck in a CB wait or semaphore wait) and no other core depends on it to make forward progress. The watcher has identified the specific hung core, and the hang is not due to a NOC hardware deadlock.

**When insufficient:** The hung core is part of a multi-core dependency chain (common in CB producer/consumer patterns, dispatch kernels, CCL operations). Also insufficient if the NOC itself is deadlocked -- a hardware-level condition that per-core soft reset cannot resolve, since the NOC interface is separate from the RISC-V core and is not affected by soft reset.

**Current limitation:** There is no high-level API for "soft reset this core and restart its firmware." The `assert_risc_reset_at_core` / `deassert_risc_reset_at_core` APIs exist at the cluster level, but using them safely requires understanding the full dependency graph. This gap is addressed by Section 03, Proposal 5 (Partial Device Reset).

---

## 4. Level 2: UMD Warm Reset (ASIC-Level)

This is the workhorse reset mechanism and the one most commonly triggered by `tt-smi -r`. It resets the entire ASIC while keeping the PCIe link and host process alive.

### 4.1 Entry Point and ARM Platform Guard

The warm reset entry point is `WarmReset::warm_reset()` in `tt_metal/third_party/umd/device/warm_reset.cpp`:

```cpp
// tt_metal/third_party/umd/device/api/umd/device/warm_reset.hpp
class WarmReset {
public:
    static void warm_reset(
        std::vector<int> pci_device_ids = {},
        bool reset_m3 = false,
        bool secondary_bus_reset = true);
    static void ubb_warm_reset(
        const std::chrono::milliseconds timeout_ms = timeout::UBB_WARM_RESET_TIMEOUT);
private:
    static constexpr auto POST_RESET_WAIT = std::chrono::milliseconds(2'000);
    ...
};
```

**ARM platform limitation:** Warm reset is unconditionally disabled on ARM-based hosts. The `is_arm_platform()` constexpr check (defined in `tt_metal/third_party/umd/device/common/utils.hpp`) prevents any reset attempt, logging: "Warm reset is disabled on ARM platforms due to instability. Skipping reset." ARM deployments have no recovery path short of a full system reboot (Level 4).

### 4.2 Pre/Post Reset IPC Notification System

Before performing any hardware reset, the system notifies all other processes using the device. The `WarmResetCommunication` IPC mechanism uses Unix Domain Sockets in `/tmp/tt_umd_listeners/`:

1. **Pre-reset notification:** `WarmResetCommunication::Notifier::notify_all_listeners_pre_reset()` sends a `PreReset` message (byte `0x01`) to all connected listeners with a 2-second timeout. This gives other processes time to save state and release device resources.
2. **Reset execution:** The actual hardware reset occurs.
3. **Post-reset notification:** `WarmResetCommunication::Notifier::notify_all_listeners_post_reset()` sends a `PostReset` message (byte `0x02`) to notify processes they can re-acquire devices.

Each listening process registers via `WarmResetCommunication::Monitor::start_monitoring()`, which creates a socket at `/tmp/tt_umd_listeners/client_<PID>.sock` and accepts connections in a detached thread. Two callbacks are registered: `on_cleanup_request` (called on pre-reset) and `post_cleanup_request` (called on post-reset).

**Important:** This IPC mechanism is **system-wide, not per-device.** A warm reset of any device notifies all listeners, regardless of which devices they are using. This is appropriate for the current all-or-nothing reset granularity but would need refinement for per-device reset support.

### 4.3 Architecture-Specific Reset Paths

The warm reset implementation branches based on architecture and driver capability:

```
warm_reset()
  |
  +-- is_arch_agnostic_reset_supported()?
  |     YES --> warm_reset_arch_agnostic()    [newer KMD path]
  |     NO  --> enumerate_devices_info()
  |              |
  |              +-- WORMHOLE_B0 --> warm_reset_wormhole_legacy()
  |              +-- BLACKHOLE   --> warm_reset_blackhole_legacy()
  |              +-- ARM platform --> skip (disabled due to instability)
```

#### 4.3.1 Arch-Agnostic Reset (Modern KMD Path)

`warm_reset_arch_agnostic()` uses kernel driver IOCTLs defined by the `TenstorrentResetDevice` enum in `pci_device.hpp`:

| IOCTL Value | Name | Description |
|-------------|------|-------------|
| 0 | `RESTORE_STATE` | Writes back saved config registers after reset |
| 1 | `RESET_PCIE_LINK` | Full PCIe link retraining (Hot Reset) |
| 2 | `CONFIG_WRITE` | Software-initiated interrupt via config register write |
| 3 | `USER_RESET` | User-triggered device reset |
| 4 | `ASIC_RESET` | Complete ASIC chip reset |
| 5 | `ASIC_DMC_RESET` | Resets the Device Management Controller (M3) |
| 6 | `POST_RESET` | Post-reset initialization procedures |

The sequence for a standard warm reset:
1. If `secondary_bus_reset=true`: `RESET_PCIE_LINK` (Hot Reset)
2. If `reset_m3=true`: `ASIC_DMC_RESET`, else: `ASIC_RESET`
3. Wait for post-reset interval
4. Poll for each device to reappear on PCIe bus
5. `POST_RESET` to finalize

**Post-reset wait timing:** The wait is calculated dynamically:
- With M3 reset: fixed 20 seconds (`WARM_RESET_M3_TIMEOUT`)
- Without M3 reset: `max(2.0, 0.4 * num_devices)` seconds

For a typical single-chip system, this is 2 seconds. For an 8-chip T3K, it is `max(2.0, 0.4 * 8) = 3.2` seconds. For a 32-chip Galaxy, it is `max(2.0, 0.4 * 32) = 12.8` seconds.

**Device reappearance polling:** After reset, each device is polled at 100ms intervals (`WARM_RESET_REAPPEAR_POLL_INTERVAL`) for up to 10 seconds (`WARM_RESET_DEVICES_REAPPEAR_TIMEOUT`) by `wait_for_pci_bdf_to_reappear()`. The function glob-matches `/sys/bus/pci/devices/<BDF>/tenstorrent/tenstorrent!*` and checks that `/dev/tenstorrent/<N>` exists.

#### 4.3.2 Wormhole Legacy Reset

`warm_reset_wormhole_legacy()` communicates directly with the ARC processor via message passing:

1. Issue `RESET_PCIE_LINK` IOCTL to all target devices
2. Wait for ARC to come up (`wait_arc_core_start` with 300-second timeout `ARC_LONG_POST_RESET_TIMEOUT`)
3. Initialize TTDevice instances
4. Record reference clock counter values (`get_refclk_counter()`)
5. Send `MSG_TYPE_ARC_STATE3` (0xA3) to put ARC in reset-ready state
6. Wait 30ms
7. Send `MSG_TYPE_TRIGGER_RESET` (0x56) to trigger the actual reset. With `reset_m3=true`, the argument is `3` (M3 board-level reset); otherwise `0xFFFF` (standard reset)
8. Wait 2 seconds (`POST_RESET_WAIT`)
9. Issue `RESTORE_STATE` IOCTL
10. Verify reset by checking that the reference clock counter has actually reset (current < old means reset occurred)

**Reset verification:** The refclk counter check is a Wormhole-specific validation. If `refclk_values_old[i] < refclk_current[i]` (i.e., the counter did not reset to zero but kept incrementing), the reset did not go through.

#### 4.3.3 Blackhole Legacy Reset

`warm_reset_blackhole_legacy()` uses a config-write based approach:

1. Issue `CONFIG_WRITE` IOCTL
2. Poll the command byte register for each device at 10ms intervals, waiting for bit 1 (the reset bit) to be set
3. Timeout after 2 seconds (`BH_WARM_RESET_TIMEOUT`)
4. Wait 2 seconds (`POST_RESET_WAIT`)
5. Issue `RESTORE_STATE` IOCTL

Note: The `reset_m3` flag has no effect on Blackhole ("Reset M3 flag doesn't influence Blackhole reset.").

### 4.4 Quantitative Timing Summary

| Architecture | Non-M3 Reset Time | M3 Reset Time | Notes |
|---|---|---|---|
| Wormhole (single chip) | ~4s (2s wait + 2s post) | ~22s (20s M3 + 2s post) | Plus ARC startup up to 300s timeout if ARC hung |
| Wormhole (T3K, 8 chips) | ~5.2s (3.2s + 2s post) | ~22s | All devices reset in parallel |
| Blackhole (single chip) | ~4s (2s poll + 2s post) | N/A | |
| Wormhole UBB (6U Galaxy) | ~130s (30s sleep + 100s driver load) | N/A | Uses IPMI, waits for 32 PCIe devices |

### 4.5 Timeout Constants

All warm reset timeouts are defined in `tt_metal/third_party/umd/device/api/umd/device/utils/timeouts.hpp`:

| Constant | Value | Usage |
|----------|-------|-------|
| `WARM_RESET_M3_TIMEOUT` | 20 seconds | Wait time after M3/DMC reset |
| `WARM_RESET_REAPPEAR_POLL_INTERVAL` | 100 ms | BDF polling interval |
| `WARM_RESET_DEVICES_REAPPEAR_TIMEOUT` | 10 seconds | Max wait for device BDF to reappear |
| `UBB_WARM_RESET_TIMEOUT` | 100 seconds | Max wait for all UBB devices |
| `BH_WARM_RESET_TIMEOUT` | 2 seconds | Blackhole legacy reset completion poll |
| `ARC_LONG_POST_RESET_TIMEOUT` | 300 seconds | Wormhole ARC startup after reset |
| `ARC_POST_RESET_TIMEOUT` | 1 second | Standard ARC post-reset wait |

### 4.6 UBB (Universal Backplane Board) Reset for Galaxy

Galaxy 6U systems have a dedicated reset path using IPMI:

```cpp
// tt_metal/third_party/umd/device/warm_reset.cpp
void WarmReset::ubb_warm_reset(const std::chrono::milliseconds timeout_ms) {
    static int constexpr UBB_NUM = 0xF;    // all UBBs
    static int constexpr DEV_NUM = 0xFF;   // all devices
    static int constexpr OP_MODE = 0x0;
    static int constexpr RESET_TIME = 0xF;

    wormhole_ubb_ipmi_reset(UBB_NUM, DEV_NUM, OP_MODE, RESET_TIME);
    sleep(30);  // Fixed 30-second wait after IPMI reset
    ubb_wait_for_driver_load(timeout_ms);
}
```

The `wormhole_ubb_ipmi_reset` function executes: `sudo ipmitool raw 0x30 0x8b <ubb> <dev> <op_mode> <reset_time>`.

After the IPMI reset, it waits for the kernel driver to re-enumerate all 32 PCIe devices (`NUMBER_OF_PCIE_DEVICES = 32`), polling at 1-second intervals for up to 100 seconds.

### 4.7 Post-Reset Topology Discovery

After every warm reset, a topology discovery pass is necessary. The `warm_reset` CLI tool invokes `TopologyDiscovery::discover({})` which scans PCIe BDFs, queries device capabilities, and builds the cluster descriptor. This is essential because device IDs may change across resets (the `/dev/tenstorrent/N` number is re-assigned when the kernel driver re-enumerates).

---

## 5. Kernel Driver Reset Safety Mechanisms (tt-kmd)

The kernel driver provides several mechanisms that ensure resets are safe in multi-process environments. These operate below the UMD layer and are critical for understanding why resets do not corrupt concurrent operations.

### 5.1 The Reset Generation Counter

A critical mechanism is the `reset_gen` counter in `struct tenstorrent_device`. Every time a reset ioctl is issued (any of `USER_RESET`, `ASIC_RESET`, or `ASIC_DMC_RESET`), the driver atomically increments this counter:

```c
atomic_long_inc(&priv->device->reset_gen);
```

When a file descriptor is opened, the current `reset_gen` value is captured in `priv->open_reset_gen`. On every subsequent ioctl call, the driver checks:

```c
if (atomic_long_read(&priv->device->reset_gen) != priv->open_reset_gen) {
    ret = -ENODEV;
    goto out;
}
```

This means that after a reset, all previously-opened file descriptors become permanently invalid with `ENODEV`. The developer cannot accidentally use a stale handle to issue commands to a just-reset chip. Any tt-metal process holding device handles must re-open them after reset.

### 5.2 The Reset Serialization Lock

The driver uses a read-write semaphore (`reset_rwsem` in `struct tenstorrent_device`) to serialize resets against all other operations:

- All non-reset ioctls acquire the lock in shared (read) mode: `down_read(&tt_dev->reset_rwsem)`
- The reset ioctl acquires the lock in exclusive (write) mode: `down_write(&tt_dev->reset_rwsem)`

This ensures that no ioctl is in-flight when a reset begins, and that multiple non-reset operations can proceed concurrently.

### 5.3 The Reset Window

Between the `ASIC_RESET` ioctl and the `POST_RESET` ioctl, the device has `needs_hw_init = true`. During this window, the driver only allows `GET_DEVICE_INFO`, `GET_DRIVER_INFO`, and `RESET_DEVICE` ioctls -- all other operations return `ENODEV`. This prevents any process from issuing NOC transactions or DMA operations to a chip that is in the middle of resetting.

### 5.4 Architecture-Specific Driver Reset Paths

- **Wormhole** (`wormhole_reset()` in `wormhole.c`): The driver first tests if the ARC firmware is responsive by sending a `WH_FW_MSG_NOP` message via the scratch register protocol. If responsive, it sets a reset marker in the PCI command register (parity error response bit) and sends `WH_FW_MSG_TRIGGER_RESET` (0x56). If the firmware is unresponsive, the driver falls back to waiting for the M3 watchdog timer to expire and trigger an automatic reset, polling with `pcie_hot_reset_and_restore_state()` in a loop.

- **Blackhole** (`blackhole_reset()` in `blackhole.c`): For ASIC-only reset, the driver uses `pcie_timer_interrupt()`, which writes to the PCIe interface timer control registers (`INTERFACE_TIMER_CONTROL_OFF` at offset 0x930, `INTERFACE_TIMER_TARGET_OFF` at 0x934) to trigger a hardware timer interrupt that initiates reset. This does not require ARC firmware cooperation.

### 5.5 The PCIe Hot Reset Path

The `pcie_hot_reset_and_restore_state()` function in `pcie.c` performs a PCIe secondary bus reset:

1. Read the bridge control register from the upstream PCIe bridge
2. Assert `PCI_BRIDGE_CTL_BUS_RESET` (write it high)
3. Wait 2ms (the minimum required by the PCIe spec)
4. De-assert `PCI_BRIDGE_CTL_BUS_RESET`
5. Wait 500ms for the link to come back up
6. Poll for the Tenstorrent vendor ID to confirm the device is responsive (up to 10 seconds)
7. Restore PCI configuration state via `safe_pci_restore_state()`

The `safe_pci_restore_state()` function includes a guard: it first reads the vendor ID to confirm the device is responsive before calling `pci_restore_state()`. This prevents a soft lockup that would occur if `pci_restore_state()` tried to scan PCI capabilities on a non-responsive device.

### 5.6 The Reset Marker Mechanism

The driver uses a creative technique to detect when a reset has completed: the PCI command register's parity error response enable bit (`PCI_COMMAND_PARITY`). Before triggering a reset, `set_reset_marker()` sets this bit. After reset, the hardware clears all of PCI configuration space to defaults. The `is_reset_marker_zero()` function checks whether the bit is cleared -- if it is, the reset has completed. This is used by the reset tool's post-reset polling loop and by the `POST_RESET` ioctl handler to verify that the chip has actually been reset before attempting reinitialization.

### 5.7 The Firmware Watchdog System

Both Wormhole and Blackhole have an M3-level watchdog timer that automatically resets the chip if firmware becomes unresponsive. This is configured by the kernel driver during hardware initialization:

- **Wormhole:** `WH_FW_MSG_UPDATE_M3_AUTO_RESET_TIMEOUT` (message 0xBC) sets the timeout. The driver sends `auto_reset_timeout` (default 10 seconds, configurable via the `auto_reset_timeout` module parameter).
- **Blackhole:** `ARC_MSG_TYPE_SET_WDT_TIMEOUT` (message 0xC1) sets the timeout in milliseconds.

If `auto_reset_timeout` is set to 0, the watchdog is disabled and the driver gives up immediately if the firmware is unresponsive ("Watchdog is disabled and device is unresponsive, cannot reset.").

---

## 6. Level 3: M3 / DMC Board-Level Reset

**What it does:** Resets the Device Management Controller (DMC/M3), which is the board management processor responsible for power sequencing, voltage regulation, and ARC processor management. This is a deeper reset than ASIC-level, reaching components that survive a normal warm reset.

**Code path:** Triggered by passing `reset_m3=true` to `WarmReset::warm_reset()`. In the arch-agnostic path, this sends the `ASIC_DMC_RESET` IOCTL (value 5). In the Wormhole legacy path, `MSG_TYPE_TRIGGER_RESET` is sent with argument `3`. In the Blackhole kernel driver path (`blackhole_reset()` in `blackhole.c`), the `ASIC_DMC_RESET` IOCTL triggers a board-level reset through a different mechanism than the UMD legacy path (where `reset_m3` has no effect on Blackhole per Section 4.3.3).

**Timing:** 20 seconds (`WARM_RESET_M3_TIMEOUT`) is the configured wait, compared to 2 seconds for a standard warm reset -- a 10x increase in downtime. DRAM training and Ethernet link training must also be re-performed.

**When required:**
- ARC processor is hung and not responding to standard reset messages
- Board-level power or voltage regulation is in an inconsistent state
- Standard warm reset fails (refclk check fails on Wormhole, device BDF does not reappear)

**Blackhole note:** The codebase explicitly logs: "Reset M3 flag doesn't influence Blackhole reset."

---

## 7. Level 4: Full System Reboot

**What it does:** Complete host system reboot, triggering PCIe device re-enumeration from scratch.

**When required:**
- The PCIe link itself is fully wedged (device reads return `0xFFFFFFFF` consistently)
- The kernel driver has entered an unrecoverable state
- Warm reset times out waiting for devices to reappear (`wait_for_pci_bdf_to_reappear` returns -1)
- Multiple consecutive warm resets have failed
- On ARM platforms, where warm reset is explicitly disabled

**Timing:** Minutes, depending on the system. BIOS POST, PCIe enumeration, kernel driver loading, and topology discovery all must complete.

This is the absolute last resort and represents a total loss of all diagnostic state.

---

## 8. Multi-Host Distributed Reset

For multi-host systems (e.g., 4x Blackhole quietbox), resets must be coordinated across hosts to avoid one host resetting devices while another is mid-operation.

The `distributed_reset.sh` script in `tests/scale_out/4x_bh_quietbox/` implements a barrier-synchronized reset:

```bash
#!/bin/bash
BARRIER_DIR="/nfs/$USER/.barrier"
rm -rf $BARRIER_DIR
mkdir -p $BARRIER_DIR

parallel-ssh -i -H "sjc1-tt-qb-01 sjc1-tt-qb-02 sjc1-tt-qb-03 sjc1-tt-qb-04" \
    "cd /nfs/$USER/tt-smi && source .venv/bin/activate && \
   touch $BARRIER_DIR/\$(hostname) && \
   while [ \$(ls $BARRIER_DIR | wc -l) -lt 4 ]; do sleep 0.01; done && \
   tt-smi -r"

rm -rf $BARRIER_DIR
```

**Mechanism:** Each host creates a marker file on a shared NFS filesystem, then spin-waits until all 4 hosts have created their markers. Once all hosts reach the barrier, they simultaneously execute `tt-smi -r`.

**Limitations:**
- Relies on NFS for coordination, which introduces latency and potential failures
- Uses a fixed host count (4) hard-coded in the poll condition
- No error handling for hosts that fail to reach the barrier (they spin indefinitely)
- No post-reset verification that all devices came back healthy
- The polling interval (10ms) assumes low NFS latency

---

## 9. When Is Each Reset Level Required?

The following decision matrix maps hang conditions to the minimum required reset level:

| Condition | Minimum Reset Level | Rationale |
|---|---|---|
| Host timeout but chip responds to PCIe reads | Level 0 (graceful close) | Kill the process, read diagnostics, restart |
| Single core hung in CB wait, others idle | Level 1 (soft reset) | Reset just that core; but requires manual dep analysis |
| Multiple cores in circular deadlock | Level 2 (warm reset) | No single core can be freed without cascading |
| NOC hardware deadlock (transaction stuck in NOC) | Level 2 (warm reset) | Software cannot resolve NOC hardware state |
| Dispatch firmware unresponsive | Level 2 (warm reset) | Dispatch runs on dedicated cores that cannot be individually recovered |
| Ethernet link permanently down | Level 2 (warm reset) | Link retraining requires reset; firmware exits via `erisc_exit()` |
| ARC processor hung | Level 3 (M3 reset) | ARC controls chip management; standard warm reset requires ARC cooperation |
| PCIe link fully wedged | Level 4 (reboot) | No communication path to device remains |
| ARM platform | Level 4 (reboot) | Warm reset explicitly disabled due to instability |

**Quantitative estimate:** Based on the hang categories from Chapters 2-5:
- ~40% of hangs are kernel-level (CB deadlocks, semaphore misuse) that could theoretically be resolved with Level 1 soft reset if a safe per-core recovery API existed -- but currently require Level 2.
- ~25% are NOC-related (barrier violations, address errors, multicast misconfigurations) that always require Level 2 because the NOC hardware state cannot be repaired from software.
- ~15% are dispatch-related, which require Level 2 because dispatch firmware cannot be restarted independently.
- ~10% are multi-chip (CCL/fabric), which require coordinated Level 2 across all affected devices.
- ~5% are truly unrecoverable without Level 3 or Level 4.
- ~5% are detectable before they become hangs (through static analysis or runtime validation) and could be prevented entirely.

The key insight: nearly half of all resets currently performed are for conditions that, with better tooling, could be resolved at Level 0 or Level 1. This is the primary motivation for the proposals in Section 03.

---

## 10. Reset Granularity: The Current All-or-Nothing Problem

Today, the practical choice is between Level 0 (works only if the chip is responsive) and Level 2 (resets everything). Level 1 (per-core soft reset) exists at the hardware register level but lacks the software infrastructure to use it safely:

1. **No dependency tracking:** There is no runtime system that knows which cores are waiting on which other cores. Resetting a core that another core depends on (via semaphore, CB, or NOC transaction) will cascade into more hangs.

2. **No state restoration:** After soft-resetting a core, there is no mechanism to reload its firmware and resume from a known checkpoint. The core must be fully re-initialized.

3. **No NOC isolation:** The NOC transactions from a hung core may still be in-flight. Resetting the core does not cancel those transactions -- the NOC interface hardware is separate from the RISC-V core.

Bridging this gap -- making Level 1 practically usable -- would eliminate the need for Level 2 resets in a significant fraction of cases. This is the core motivation for Proposal 5 (Partial Device Reset) in Section 03.

---

**Previous:** [`index.md`](./index.md) | **Next:** [`02_reducing_reset_frequency_and_resilience.md`](./02_reducing_reset_frequency_and_resilience.md)
