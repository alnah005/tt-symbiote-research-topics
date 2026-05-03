# Final Plan: Tenstorrent Hardware Hang Causes, Debugging Tools, and Chip Reset Reduction

## Selection Rationale

This final plan is a hybrid constructed from all five candidate plans, guided by the evaluations. The primary structural and technical foundation comes from **Plan V1**, which all evaluators agreed has the deepest code-level specificity and most comprehensive coverage. The following elements are drawn from other plans to address V1's gaps and strengthen the result:

- **From V1 (primary foundation):** The 8-chapter cause-then-tools-then-workflows-then-future structure, the exhaustive source code references (specific files, functions, structs, enum values, line ranges), the dedicated memory chapter, the comprehensive debugging tools catalog (6 files including debug delay/timing perturbation), the detailed watcher system coverage, the thorough future tooling proposals, and the cross-chapter dependency analysis.

- **From V4 (diagnostic format):** The five-part hang scenario presentation format (**Symptom / Root Cause / Diagnosis Steps / Fix / Prevention**) is adopted as the standard for every hang cause description. V4's evaluator and multiple other evaluators identified this as the most practitioner-friendly structure. V4 also contributes its thorough reset hierarchy (five levels from graceful termination to full reboot), the `hang_device` test operation reference, and the NOCDebugState issue type enumeration.

- **From V3 (tt-triage and software stack layer):** V3's extensive tt-triage coverage (14 named scripts with descriptions) is incorporated into Chapter 6. V3's unique "Software Stack and Where Hangs Originate" concept is adapted as a concise orientation file within Chapter 1 rather than a full chapter (to avoid the overlap problem identified by V3's evaluator). V3's allocator/silent OOM coverage and the Symptom/Root Cause/How to Diagnose/How to Fix formatting influence are also incorporated.

- **From V5 (blocking primitives and slow dispatch):** V5's unique "blocking primitives taxonomy" -- a single-file catalog of every device-side API call that can become an infinite wait, mapped to waypoint codes -- is incorporated into Chapter 1. V5's coverage of slow dispatch mode (`TT_METAL_SLOW_DISPATCH_MODE=1`) as a diagnostic isolation technique is added to the workflows chapter. V5's `TT_METAL_DISPATCH_PROGRESS_UPDATE_MS` heartbeat feature and `Device::initialize` minimal mode for attaching to hung chips are incorporated.

- **From V2 (resilience patterns):** V2's defensive programming patterns and resilience strategies (proper CB sizing, NOC transaction ordering, semaphore initialization, timeout wrappers, multi-chip resilience patterns) are incorporated into the reset reduction chapter.

**Key gaps addressed across all plans (flagged by multiple evaluators):**
- tt-triage tool: Missing from V1 and V2; now given dedicated coverage in Chapter 6
- LightMetal replay: Missing from V4; now covered in Chapter 4
- NOC debug dump: Missing from V1; now covered in Chapter 6
- Lightweight kernel asserts: Missing from V1 and V2; now covered in Chapter 6
- Slow dispatch mode: Missing from V1-V4; now covered in Chapter 7
- Debug delay/timing perturbation: Missing from V3 and V5; retained from V1 in Chapter 6
- Grayskull coverage: Thin in V1 and V5; expanded in Chapter 1
- Memory hangs dedicated depth: Weak in V4; retained as dedicated Chapter 3 from V1
- Distinguishing HW vs. SW bugs: Missing from V5; retained from V1 in Chapter 7
- Blocking primitives taxonomy: Unique to V5; incorporated into Chapter 1
- watcher_dump standalone tool: Underserved in V3-V5; retained as dedicated file from V1

---

## Audience

This guide targets **intermediate-to-advanced tt-metal/TTNN developers** who write or debug kernels, ops, and multi-chip workloads on Tenstorrent hardware. Readers are expected to already understand:

- The Tenstorrent Tensix core architecture (BRISC, NCRISC, TRISC processors per core)
- Basic NOC (Network-on-Chip) read/write/multicast semantics
- Circular buffer producer/consumer programming model
- The distinction between host-side (Python/C++) and device-side (RISC-V firmware/kernel) code
- How to build and run programs using tt-metal or TTNN
- Basic familiarity with multi-chip configurations (T3K, Galaxy) is helpful but not required

They do **not** need prior knowledge of watcher internals, dispatch infrastructure details, firmware reset mechanisms, UMD-level driver APIs, or tt-triage/Inspector tooling. All debugging tools and infrastructure are taught from scratch within the guide.

---

## Chapter List

### Chapter 1: Anatomy of a Hang -- Core Concepts and Taxonomy

**Description:** Establishes the foundational vocabulary, execution model, and classification system for all hang types on Tenstorrent hardware, providing the conceptual framework every subsequent chapter builds upon.

**Directory:** `ch01_anatomy_of_a_hang/`

**Files:**

- `01_what_is_a_hang.md`
  - Definition of a "hang" vs. a crash, error, silent corruption, or slow execution
  - Observable symptoms: program stops making progress, host-side timeout (e.g., `Synchronize()` never returns), unresponsive chip, need for `tt-smi` reset or reboot
  - The fundamental model: a hang is always one or more RISC-V processors spinning in a wait loop whose exit condition will never be satisfied
  - The hang lifecycle: trigger condition, silent stall, eventual detection (or lack thereof), diagnostic capture, recovery
  - When a chip reset (`tt-smi` warm reset) is required vs. when the workload can simply be killed
  - **Format:** Each subsequent hang cause in Chapters 2-5 follows the convention: **Symptom** (what the developer observes), **Root Cause** (what is actually happening), **Diagnosis Steps** (which tools to use and what to look for), **Fix** (the code change or configuration needed), **Prevention** (how to avoid the issue in the first place)

- `02_blocking_primitives_taxonomy.md`
  - Catalogs every blocking call in the device-side API that can become an infinite wait (hang point):
    - `cb_reserve_back` -- spins until CB has free space (waypoint: `CRBW`)
    - `cb_wait_front` -- spins until CB has tiles available (waypoint: `CWFW`)
    - `noc_async_read_barrier` -- spins until all outstanding NOC reads complete (waypoint: `NRBW`)
    - `noc_async_write_barrier` -- spins until all outstanding NOC writes are acknowledged (waypoint: `NWBW`)
    - `noc_semaphore_wait` -- spins until semaphore equals target value (waypoint: `NSW`)
    - `noc_semaphore_wait_min` -- spins until semaphore >= target value (waypoint: `NSMW`)
  - For each primitive: the exact spin-loop mechanism, the counter/condition it waits on, the watcher waypoint code that identifies it, and what causes the exit condition to never be met
  - Reference files: `tt_metal/hw/inc/api/dataflow/dataflow_api.h` (lines ~389-1949)
  - This file serves as a single-page lookup table for the most common hang symptoms observed in watcher logs

- `03_hang_taxonomy.md`
  - Complete classification tree of hang root causes organized into six categories:
    1. **Kernel-level deadlocks** -- BRISC/NCRISC/TRISC synchronization failures, circular buffer producer/consumer stalls, subordinate sync protocol violations, inter-RISC semaphore misuse
    2. **NOC transaction failures** -- malformed addresses, alignment violations, linked transaction deadlocks, multicast to invalid targets, backpressure-induced stalls, mcast path reservation hangs
    3. **Memory-related hangs** -- L1 overflow/corruption, DRAM address range violations, circular buffer out-of-bounds access, bank collision stalls, DRAM bandwidth saturation, silent OOM corruption, tile size mismatches
    4. **Dispatch and command queue stalls** -- prefetch/dispatch kernel deadlocks, system memory queue full, worker config buffer exhaustion, trace replay failures
    5. **Multi-chip and CCL hangs** -- Ethernet link failures, cross-chip semaphore protocol violations, all_gather/reduce_scatter deadlocks, topology misconfiguration, fabric router flow control stalls
    6. **Host-device interaction hangs** -- synchronize_device deadlocks, async operation ordering violations, mismatched device/host state after partial errors, LightMetal replay failures
  - For each category: typical symptoms, which RISC processors are involved, and which chapter provides detailed coverage
  - How multiple categories can compound (e.g., a CB overflow corrupts L1, which causes a NOC address violation, which causes a spin-loop hang)
  - Decision tree for initial hang classification based on observable symptoms

- `04_hang_causes_across_architectures.md`
  - Architectural differences relevant to hangs across chip generations:
    - **Grayskull (GS):** Single-chip only, no Ethernet cores, no multi-chip hang scenarios possible. Simpler NOC topology. The baseline architecture for understanding core hang mechanics. Eliminates entire hang categories (Q2, multi-chip CCL)
    - **Wormhole (WH/WH_B0):** NOC coordinate system (NOC0/NOC1 mirroring), virtual vs. physical coordinates (`COORDINATE_VIRTUALIZATION_ENABLED`), harvested row handling. Adds Ethernet cores (ERISC) and multi-chip connectivity. T3K 8-chip and Galaxy 32+ chip configurations. Mcast path reservation hang workaround in dispatch. Active and idle ERISC modes
    - **Blackhole (BH):** Relaxed memory ordering implications, L1 data cache invalidation issues (`enable_hw_cache_invalidation` runtime option). The inline-write-to-L1 hang unique to BH (4-memory-port back-pressure; workaround: write to stream registers via `risc_attribs.h`). Different coordinate virtualization. Subordinate ERISC model
    - **Quasar (tt-2xx):** DM (data mover) core architecture replacing BRISC/NCRISC split, different debug register maps, NEO register set. Implications for synchronization and hang patterns
  - Which hang categories are architecture-specific vs. universal
  - System configuration scale differences: single-chip vs. N300 (2x WH) vs. T3K (8x WH) vs. Galaxy (32+ WH) vs. multi-Blackhole quietbox. How system scale changes the hang surface area

**Covers questions:** 1 (all root cause categories -- overview), 4 (memory causes -- overview), 9 (architecture/configuration differences)

---

### Chapter 2: Kernel-Level and NOC Hang Mechanisms

**Description:** Deep-dive into the on-device hang mechanisms at the firmware and kernel level, covering the exact code paths that lead to spin-loop hangs, with each hang cause presented in the Symptom/Root Cause/Diagnosis Steps/Fix/Prevention format.

**Directory:** `ch02_kernel_and_noc_hangs/`

**Files:**

- `01_risc_synchronization_and_deadlocks.md`
  - BRISC as the main processor, NCRISC and TRISCs as subordinates
  - The `subordinate_sync` mailbox protocol: `RUN_SYNC_MSG_WAITING_FOR_RESET` states
  - The `wait_ncrisc_trisc()` spin in BRISC firmware (`brisc.cc`) and how mismatched kernel configurations cause deadlocks
  - Launch message ring buffer: `launch_msg_rd_ptr`, `go_msg_t` signals (`RUN_MSG_DONE`), and how stale messages cause hangs
  - ERISC context switching and `risc_context_switch()` -- how failing to yield on Ethernet cores causes hangs in fabric routing
  - How the kernel launch / go signal protocol works (`launch_message_ring_buffer_state.cpp`)
  - Deadlocks when one RISC expects a semaphore increment from another RISC that has already exited or is itself blocked
  - Reference files: `tt_metal/hw/firmware/src/tt-1xx/brisc.cc`, `tt_metal/hw/inc/hostdev/dev_msgs.h`

- `02_circular_buffer_deadlocks.md`
  - The `CBInterface` structure and producer/consumer model: `cb_reserve_back`, `cb_push_back`, `cb_wait_front`, `cb_pop_front`
  - How `cb_wait_front` spins forever when the producer never pushes enough tiles
  - How `cb_reserve_back` stalls when the consumer never pops -- the classic circular buffer deadlock
  - Mismatched tile counts, incorrect `num_pages` arguments, and off-by-one errors as common triggers
  - The rule that `ntiles` must evenly divide CB size and must be consistent across calls -- violations cause hangs, not errors
  - The cumulative-total requirement for `cb_wait_front`: calling `cb_wait_front(8)` four times instead of `cb_wait_front(8), cb_wait_front(16), cb_wait_front(24), cb_wait_front(32)` causes incorrect behavior
  - The `RemoteSenderCBInterface` and `RemoteReceiverCBInterface` for cross-core circular buffers and their additional deadlock potential
  - The `cb_addr_shift` mechanism and how it differs between data movers and compute cores
  - WAYPOINT indicators for CB stalls: `CRBW` (CB Reserve Back Wait), `CWFW` (CB Wait Front Wait)
  - Reference files: `tt_metal/hw/inc/api/dataflow/dataflow_api.h` (lines 200-460), `tt_metal/hw/inc/internal/circular_buffer_interface.h`

- `03_noc_address_sanitization_and_violations.md`
  - The complete NOC address validation pipeline in `sanitize.h`:
    - `debug_valid_worker_addr`: L1 base/size bounds checking, read-only mailbox region protection
    - `debug_valid_pcie_addr`: PCIe address range validation using `core_info_msg_t`
    - `debug_valid_dram_addr`: DRAM address range validation
    - `debug_valid_eth_addr`: Ethernet core L1 bounds checking
    - `debug_valid_cb_addr`: Circular buffer out-of-bounds detection (iterates all 32 CBs)
  - All `DebugSanitize*` return codes and what triggers each: `NocAddrUnderflow`, `NocAddrOverflow`, `NocAddrZeroLength`, `NocTargetInvalidXY`, `NocMulticastNonWorker`, `NocMulticastInvalidRange`, `NocAlignment`, `NocMixedVirtualandPhysical`, `InlineWriteDramUnsupported`, `NocAddrMailbox`, `NocLinkedTransactionViolation`, `L1AddrOverflow`, `EthSrcL1AddrOverflow`, `EthDestL1AddrOverflow`, `CBOutOfBounds`
  - The `debug_sanitize_post_addr_and_hang` function: writes violation details to the watcher mailbox, then enters `while(1)` spin-loop (Tensix) or calls `erisc_exit()` (Ethernet cores)
  - Linked transaction validation: how submitting a unicast when a linked multicast is pending causes a deadlock (`DebugSanitizeNocLinkedTransactionViolation`)
  - Reference files: `tt_metal/hw/inc/internal/debug/sanitize.h`

- `04_noc_barrier_and_semaphore_hangs.md`
  - `noc_async_read_barrier` and `noc_async_write_barrier`: spin-waiting on outstanding transaction counters
  - How unbalanced `noc_reads_num_issued` vs. completed reads causes barrier hangs
  - The mcast path reservation hang workaround in `cq_dispatch.cpp`: "Workaround mcast path reservation hangs by always waiting for a write barrier before doing an mcast that isn't linked to a previous mcast" -- a known hardware issue requiring a software workaround
  - `noc_semaphore_wait` and `noc_semaphore_wait_min`: spin-waiting on L1 semaphore values. Subtle difference: `wait` checks `== val`, `wait_min` checks `>= val`
  - How semaphore protocol violations (incrementing wrong semaphore, wrong count, wrong target core, wrong initial value, forgetting to reset between iterations) cause infinite waits
  - Transaction ID-based barriers (`noc_async_read_barrier_with_trid`, `noc_async_write_barrier_with_trid`) and their additional failure modes
  - Reference files: `tt_metal/hw/inc/api/dataflow/dataflow_api.h` (lines 1731-2504)

**Covers questions:** 1 (kernel deadlocks, NOC deadlocks in detail), 4 (L1/DRAM memory hangs, CB overflow -- via NOC sanitization)

---

### Chapter 3: Memory-Related Hang Causes

**Description:** Examines how memory subsystem issues -- L1 corruption, DRAM saturation, alignment violations, allocation failures, and out-of-memory conditions -- lead to hangs that are often harder to diagnose than explicit deadlocks.

**Directory:** `ch03_memory_related_hangs/`

**Files:**

- `01_l1_memory_corruption_and_overflow.md`
  - L1 memory map structure: firmware at address 0 (`fw_launch_addr_value`), mailbox region (read-only protected by watcher), kernel text space, CB space, semaphores, runtime args
  - How L1 overflow corrupts adjacent data structures (e.g., a CB write overflowing into the mailbox region or another CB's space)
  - The watcher's `DumpL1Status()` check for address 0 memory corruption (overwritten firmware launch value)
  - Silent corruption scenarios: when corrupted data does not immediately cause a detectable error but leads to a hang later (wrong NOC address from corrupted runtime args, wrong tile count from corrupted CB metadata)
  - Stack overflow on RISC-V processors: the watcher's `STACK_USAGE` tracking feature and what happens when firmware stack exceeds its allocation
  - The `debug_sanitize_l1_access` check for direct L1 access validation
  - `MEM_L1_BASE`, `MEM_L1_SIZE`, `MEM_ETH_BASE`, `MEM_ETH_SIZE` constants and their role in bounds checking
  - Reference files: `tt_metal/hw/inc/internal/debug/sanitize.h` (lines 535-552), `tt_metal/hw/inc/hostdev/dev_msgs.h`

- `02_dram_and_noc_backpressure.md`
  - DRAM bandwidth saturation: how many concurrent NOC reads/writes to the same DRAM bank can create backpressure
  - Bank collision stalls: when multiple cores target the same L1 or DRAM bank simultaneously
  - How NOC backpressure propagates: a saturated DRAM channel blocks NOC responses, which blocks the issuing core's command buffer, which blocks the kernel spin-loop
  - The relationship between `noc_nonposted_writes_num_issued`, `noc_nonposted_writes_acked`, and write-barrier hangs under backpressure
  - Interleaved DRAM buffer access patterns that create hotspots on specific DRAM channels
  - The DRAM arbiter hang test pattern (`test_kernels/dataflow/dram_arbiter_hang.cpp`)
  - DRAM address validation: `noc_dram_addr_base` and `noc_dram_addr_end` from `core_info_msg_t`
  - Reference files: `tt_metal/hw/inc/internal/debug/sanitize.h` (lines 194-207)

- `03_alignment_and_tile_size_mismatches.md`
  - NOC alignment requirements: `NOC_L1_READ_ALIGNMENT_BYTES`, `NOC_L1_WRITE_ALIGNMENT_BYTES`, `NOC_PCIE_READ_ALIGNMENT_BYTES`, `NOC_DRAM_READ_ALIGNMENT_BYTES`
  - How misaligned DMA transfers can cause silent NOC-level hangs (hardware does not error, just stalls)
  - The alignment cross-check in `debug_sanitize_noc_and_worker_addr`: L1 address alignment must match NOC target alignment
  - Tile size mismatches between reader/writer/compute kernels: how a kernel expecting 32x32 tiles reading from a CB filled with 16x32 tiles can cause address calculation errors leading to NOC violations
  - When the programmed DMA transfer size does not match the actual tile size, the NOC read/write counter never reaches the expected value, causing barriers to hang indefinitely
  - Reference files: `tt_metal/hw/inc/internal/debug/sanitize.h` (lines 468-515)

- `04_allocation_failures_and_silent_oom.md`
  - L1 allocator behavior on out-of-memory: `TT_THROW` vs. silent corruption depending on the code path
  - DRAM allocation failures and their error handling (or lack thereof)
  - The free-list allocator (`free_list_opt.cpp`): fragmentation scenarios that can lead to allocation failure in long-running workloads
  - How allocation failures at the host level can result in kernels receiving garbage buffer addresses, leading to NOC transactions to invalid memory and secondary hangs extremely difficult to trace to their root cause
  - CB overflow: if `cb_push_back` is called with more tiles than the CB can hold, the write pointer wraps and overwrites data the consumer has not yet read, causing data corruption and downstream compute hangs
  - The watcher's CB sanitization feature (`WATCHER_DISABLE_CB_SANITIZE`) and L1 read-only / write-only sanitization (`TT_METAL_WATCHER_DISABLE_SANITIZE_READ_ONLY_L1`, `TT_METAL_WATCHER_DISABLE_SANITIZE_WRITE_ONLY_L1`)
  - Reference files: `tt_metal/hw/inc/internal/debug/sanitize.h`, `tt_metal/impl/allocator/free_list_opt.cpp`

**Covers questions:** 4 (all memory-related hang causes)

---

### Chapter 4: Dispatch, Command Queue, and Host-Device Interaction Hangs

**Description:** Covers hang causes in the host-to-device command dispatch pipeline, including the fast dispatch infrastructure, trace replay, LightMetal deterministic replay, and host-side synchronization mechanisms.

**Directory:** `ch04_dispatch_and_host_device_hangs/`

**Files:**

- `01_dispatch_architecture_and_hang_points.md`
  - Fast dispatch overview: host writes commands to system memory (hugepage), prefetch kernel reads and forwards to dispatch kernel, dispatch kernel writes to worker cores
  - The `SystemMemoryManager` and command queue interface: issue queue and completion queue
  - Command types from `cq_commands.hpp`: `CQ_PREFETCH_CMD_STALL`, `CQ_PREFETCH_CMD_RELAY_*`, `CQ_DISPATCH_CMD_WAIT`, `CQ_DISPATCH_CMD_WRITE_*`, etc.
  - Prefetch kernel (`cq_prefetch.cpp`) hang points: waiting for data from host, waiting for dispatch to consume, `StallState::STALLED` / `StallState::NOT_STALLED` mechanism
  - Dispatch kernel (`cq_dispatch.cpp`) hang points: waiting for prefetch to produce, waiting for workers to become ready
  - Dispatch subordinate (`cq_dispatch_subordinate.cpp`): secondary dispatch path and its synchronization requirements
  - Worker config buffer exhaustion: `WorkerConfigBuffer` space limits and how exceeding them causes dispatch to stall
  - The relay_mux topology for multi-device dispatch and additional synchronization points
  - Reference files: `tt_metal/impl/dispatch/kernels/cq_prefetch.cpp`, `cq_dispatch.cpp`, `cq_dispatch_subordinate.cpp`, `cq_commands.hpp`, `hardware_command_queue.hpp`

- `02_host_synchronization_and_timeout_detection.md`
  - `Synchronize` / `Finish` semantics: how the host waits for device completion
  - The completion queue mechanism: dispatch kernel writes completion signals, host polls for them
  - Timeout scenarios: what happens when the device never writes a completion signal
  - Host-side timeout detection: `llrt.cpp` polls cores with configurable timeout (`TT_METAL_OPERATION_TIMEOUT_SECONDS`). When timeout fires, `MetalContext::on_dispatch_timeout_detected()` serializes Inspector data and optionally executes `TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE` (e.g., `./tools/tt-triage.py`)
  - The `loop_and_wait_with_timeout` mechanism in `system_memory_manager.cpp` that detects when the device fails to consume commands, raising `TIMEOUT: device timeout in fetch queue wait, potential hang detected`
  - `DeviceManager::close_device` and `close_devices` with `skip_synchronize` parameter: when closing a hung device
  - Async operation ordering violations: when operations are enqueued in an order that creates circular dependencies on the device
  - Multi-queue scenarios: when operations on different command queues have implicit dependencies
  - Sub-device managers (`SubDeviceManagerTracker`): how switching sub-device configurations mid-execution can cause state inconsistencies
  - Reference files: `tt_metal/impl/device/device_manager.hpp`, `tt_metal/impl/dispatch/hardware_command_queue.cpp`

- `03_trace_replay_and_lightmetal.md`
  - Trace capture and replay: `TraceBuffer`, how pre-recorded command sequences bypass normal dispatch
  - `RUN_MSG_REPLAY_TRACE` signal in BRISC firmware: resets the launch message read pointer and replays captured command sequences
  - Hang causes specific to trace replay: stale state assumptions, device configuration drift between capture and replay, stale buffer addresses after reallocation
  - Program cache interactions: stale program cache entries pointing to freed L1 memory can cause hangs on re-execution
  - LightMetal capture/replay: `lightmetal_capture.cpp`, `lightmetal_replay_impl.cpp` -- recording and replaying full metal API call sequences
  - How LightMetal replay can be used as a hang reproduction tool: capturing the exact sequence of API calls leading to a hang for deterministic replay on a different device or in simulation
  - Reference files: `tt_metal/impl/trace/trace_buffer.hpp`, `tt_metal/impl/lightmetal/lightmetal_capture.hpp`, `tt_metal/impl/lightmetal/lightmetal_replay_impl.hpp`

**Covers questions:** 3 (all host-device interaction hang causes), 1 (dispatch command queue stalls)

---

### Chapter 5: Multi-Chip, CCL, and Fabric Hang Causes

**Description:** Covers hang causes specific to multi-chip configurations (T3K, Galaxy) including CCL collective operations, Ethernet fabric, and cross-chip synchronization, with each hang scenario in the standard diagnostic format.

**Directory:** `ch05_multi_chip_and_ccl_hangs/`

**Files:**

- `01_ethernet_and_fabric_fundamentals.md`
  - Ethernet core architecture: active erisc (AERISC) vs. idle erisc (IERISC), subordinate erisc cores
  - The Ethernet link status check: `WATCHER_CHECK_ETH_LINK_STATUS()` macro, `is_link_up()` check, `hang_on_down_link()` behavior
  - When a link goes down: the erisc core sets `link_down = 1` in the watcher mailbox, marks itself as `RUN_MSG_DONE`, and exits to base firmware
  - Watcher Ethernet link status tracking: `watcher_device_reader.cpp` tracks `logical_core_to_eth_link_retraining_count` and reports link-down events
  - Fabric EDM (Ethernet Data Mover): `fabric_erisc_router.cpp`, router mux/relay extensions, flow control helpers
  - Fabric deadlock avoidance: the `enable_deadlock_avoidance` template parameter in EDM channels, the `need_deadlock_avoidance_support` mechanism, bubble flow control protocol
  - Fabric telemetry: `FabricTelemetrySettings`, heartbeat TX/RX monitoring, router state, bandwidth tracking via `fabric_telemetry_msgs.h`
  - Reference files: `tt_metal/hw/inc/api/debug/eth_link_status.h`, `tt_metal/fabric/impl/kernels/edm_fabric/`, `tt_metal/hw/inc/hostdev/fabric_telemetry_msgs.h`

- `02_ccl_collective_operation_hangs.md`
  - CCL operation architecture: `all_gather`, `reduce_scatter`, `all_reduce`, `all_broadcast`, `reduce_to_root`, `all_to_all_combine`
  - How CCL operations use device-side semaphores and fabric sockets for cross-chip synchronization
  - Deadlock scenarios: all devices must participate in a collective -- if one device skips or crashes, all others hang waiting. Mismatched tensor dimensions across ranks. Ring/linear topology ordering violations causing circular waits
  - Semaphore protocol violations in CCL: incrementing semaphores out of order, using wrong semaphore IDs, targeting wrong remote cores, termination master semaphore coordination
  - Fabric socket programming: `fabric_socket.cpp`, `bidirectional_fabric_socket.cpp` -- how socket setup errors cause hangs
  - Fabric deadlock stability tests (`test_fabric_deadlock_stability_bh_6U_galaxy.yaml`, `test_fabric_deadlock_stability_6U_galaxy.yaml`) and what they validate
  - Reference files: `ttnn/cpp/ttnn/operations/ccl/`, `ttnn/core/distributed/fabric_socket.cpp`

- `03_topology_and_mesh_configuration_hangs.md`
  - Mesh device setup: `mesh_device.hpp`, mesh graph descriptors, control plane configuration
  - Topology misconfiguration: wrong chip connectivity assumptions, incorrect ring ordering for CCL, wrong number of devices specified
  - The `ControlPlane` and `FabricSwitchManager`: how routing table errors can cause packets to loop or deadlock
  - T3K (8-chip) vs. Galaxy (32+ chip) specific failure modes: link training failures on specific ports, asymmetric bandwidth paths
  - The `skip_eth_cores_with_retrain` runtime option: working around unstable Ethernet links
  - MMIO vs. remote devices: remote devices accessed through Ethernet tunnels from MMIO-mapped devices. If the tunnel fabric is not properly initialized, remote device operations hang
  - `DispatchKernelNode` topology: `device_id` (where the kernel runs) vs. `servicing_device_id` (remote device it services)
  - Multi-host mesh runtime synchronization: `MultiHostMeshRuntime` and cross-host Ethernet link failures
  - Galaxy-specific: fabric initialization can take minutes, ordered shutdown from farthest to closest, distributed reset coordination (`distributed_reset.sh`)
  - Reference files: `tt_metal/api/tt-metalium/experimental/fabric/`, `tt_metal/api/tt-metalium/mesh_device.hpp`

**Covers questions:** 2 (all multi-chip and CCL hang causes), 9 (multi-chip configuration differences)

---

### Chapter 6: Debugging Tools and Infrastructure

**Description:** Comprehensive catalog of all existing Tenstorrent tools and infrastructure for detecting, diagnosing, and recovering from hangs, with practical usage instructions and configuration details.

**Directory:** `ch06_debugging_tools/`

**Files:**

- `01_watcher_system.md`
  - Watcher architecture: `WatcherServer` (host-side polling thread) + device-side mailbox protocol
  - Enabling watcher: `TT_METAL_WATCHER` environment variable, `WatcherSettings` configuration (enabled, dump_all, append, auto_unpause, noinline, interval_ms)
  - What watcher monitors per core (each individually toggleable via `TT_METAL_WATCHER_DISABLE_*` env vars):
    - **Waypoints** (`waypoint.h`): 4-character status codes per RISC processor (I=init, W=wait, R=run, D=done, plus multi-char like NRW=noc read wait)
    - **NOC sanitization** (`sanitize.h`): address validation on every NOC transaction, violation details stored in watcher mailbox
    - **Kernel assertions** (`assert.h`): `ASSERT()` macro that writes line number and assert type to mailbox, then hangs. ERISC variant: calls `erisc_exit()` and sets `RUN_MSG_DONE` instead of infinite loop
    - **Ring buffer** (`ring_buffer.h`): per-core circular debug log (31-element uint32_t buffer) for custom values via `WATCHER_RING_BUFFER_PUSH()`
    - **Pause/resume** (`pause.h`): `PAUSE()` macro for breakpoint-like debugging, host clears flag to resume. `auto_unpause` mode for automated recovery
    - **Stack usage tracking** (`stack_usage.h`): monitors stack watermark per core, detects stack overflow
    - **Ethernet link status** (`eth_link_status.h`): automatic link-down detection on erisc cores, link retraining count tracking
    - **CB sanitization**: detects out-of-bounds circular buffer access, NOC transactions touching active CB regions
    - **Linked transaction validation** (`TT_METAL_WATCHER_ENABLE_NOC_SANITIZE_LINKED_TRANSACTION`): opt-in tracking of posted/non-posted write correlations
  - Full env var reference: `TT_METAL_WATCHER=<interval_ms>`, `TT_METAL_WATCHER_DUMP_ALL`, `TT_METAL_WATCHER_APPEND`, `TT_METAL_WATCHER_NOINLINE`, `TT_METAL_WATCHER_PHYS_COORDS`, `TT_METAL_WATCHER_TEXT_START`, `TT_METAL_WATCHER_SKIP_LOGGING`, `TT_METAL_WATCHER_DISABLE_DISPATCH`, `TT_METAL_WATCHER_DISABLE_ETH`
  - The watcher log file: `generated/watcher/watcher.log` format, legend, kernel ID mapping via `kernel_names.txt`
  - `WatcherDeviceReader`: the host-side component that reads mailboxes from all cores and decodes violations. Sentinel values (`DEBUG_SANITIZE_SENTINEL_OK_64 = 0xbadabadabadabada`)
  - Performance impact of watcher: disables DMA ops, adds sanitization overhead to every NOC transaction, binary size considerations
  - GDB integration: calling `tt::watcher::dump(stderr, true)` from gdb to get device state during a live debug session
  - Reference files: `tt_metal/impl/debug/watcher_server.hpp`, `watcher_server.cpp`, `watcher_device_reader.cpp`, all files under `tt_metal/hw/inc/api/debug/`

- `02_watcher_dump_tool.md`
  - The standalone `watcher_dump` tool: `tt_metal/tools/watcher_dump/watcher_dump.cpp`
  - How it works: creates a minimal device connection (no L1 clear), reads watcher mailboxes, dumps command queue state
  - Usage for post-mortem analysis: running watcher_dump after a hang without rebooting, even when watcher was not enabled during the run
  - `Device::initialize` with `minimal=true`: skips FW/watcher/dprint initialization to allow attaching to a hung chip for diagnostic reads
  - Command queue dump: issue queue and completion queue contents via `dump_cqs()`
  - NOC transfer logging dump: histogram of NOC transfer sizes when `NOC_LOGGING_ENABLED` is set
  - Limitations: device must still be PCIe-accessible (not fully wedged)
  - Reference files: `tt_metal/tools/watcher_dump/watcher_dump.cpp`, `tt_metal/impl/dispatch/debug_tools.hpp`

- `03_dprint_server.md`
  - Device-side print debugging: `DPRINT` macro for printf-like output from kernels
  - Core-specific print macros: `DPRINT_DATA0`, `DPRINT_DATA1`, `DPRINT_MATH`, `DPRINT_PACK`, `DPRINT_UNPACK`
  - `DPrintServer` architecture: device writes to print buffers in L1, host polls and formats output
  - Targeting specific cores/chips/harts: `TT_METAL_DPRINT_CORES`, `TT_METAL_DPRINT_ETH_CORES`, `TT_METAL_DPRINT_CHIPS`, `TT_METAL_DPRINT_RISCVS`, `TT_METAL_DPRINT_FILE`
  - Printing CB data: `print_full_tile`, `print_bf16_pages`, `print_f32_pages` from `dprint_pages.h`
  - Using DPRINT to narrow down hang location: printing progress markers before the hang point
  - DPrint's own hang risk: `server_killed_due_to_hang_` flag triggered when a core appears stalled with outstanding print data. DPrint can itself cause hangs if the print buffer fills up and the kernel blocks waiting for the host to drain it
  - Interaction with watcher: both share L1 buffer space, cannot use NOC logging and DPRINT simultaneously
  - Reference files: `tt_metal/impl/debug/dprint_server.hpp`, `dprint_server.cpp`, `tt_metal/hw/inc/api/debug/dprint.h`

- `04_tt_triage_tool.md`
  - The `tt-triage` system: `tools/tt-triage.py` entry point, script discovery under `tools/triage/` directory
  - Script types: data providers (return diagnostic data) vs. state checkers (log check failures)
  - Key triage scripts and what each diagnoses:
    - `dump_callstacks.py`: Per-core RISC-V program counter and callstack extraction from hung cores
    - `dump_aggregated_callstacks.py`: Grouped view of cores stuck at the same PC
    - `dump_lightweight_asserts.py`: Extracting `ebreak`-triggered assert information
    - `dump_watcher_ringbuffer.py`: Reading ring buffer contents post-hang
    - `dump_fast_dispatch.py`: Dispatcher/prefetcher state analysis
    - `dump_running_operations.py`: Identifying the Metal-level operation that was running when hang occurred
    - `dump_risc_debug_signals.py`: Low-level RISC debug signal state
    - `dump_broken_components.py`: System component failures
    - `check_noc_status.py`: NOC transaction status, stuck transactions
    - `check_noc_locations.py`: NOC address validity
    - `check_eth_status.py`: Ethernet link status
    - `check_arc.py`: ARC processor health
    - `check_cb_inactive.py`: Circular buffer inactivity detection
    - `check_core_magic.py`: Core magic number validation
    - `check_binary_integrity.py`: Binary integrity verification
    - `firmware_versions.py`: Firmware version verification
  - Using ttexalens (the underlying debug framework) for direct register reads
  - tt-triage command-line usage: `--remote-exalens`, `--initialize-with-noc1`, `--dev`, `--run`, `--all-cores`, verbosity levels
  - Auto-trigger on dispatch timeout: `TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE=./tools/tt-triage.py`
  - Inspector integration: always-on Metal host runtime telemetry, RPC server for querying program/workload state, automatic serialization on dispatch timeout via `InspectorSettings`
  - The `hang_device` test operation (`ttnn/cpp/ttnn/operations/experimental/test/hang_device/`) for deliberately inducing hangs to validate tooling
  - Reference files: `tools/tt-triage.py`, `tools/triage/`, `tt_metal/impl/debug/inspector/inspector.cpp`, `data.cpp`, `types.hpp`

- `05_profiler_tracy_and_noc_debug.md`
  - Tracy profiler integration: `tt_metal/tools/profiler/kernel_profiler.hpp`, `tt_metal_tracy.hpp`
  - How profiler data helps debug hangs: identifying the last completed operation before a hang
  - Tracy tools: `process_device_log.py`, `process_ops_logs.py`, `profile_this.py`
  - NOC event profiling: `noc_event_profiler.hpp`, `noc_debugging_profiler.hpp` -- detailed NOC transaction traces with timestamps and addresses for post-mortem deadlock analysis
  - Fabric event profiling: `fabric_event_profiler.hpp` for cross-chip transaction visibility
  - Performance counter modes: `profiler_perf_counter_mode` settings
  - The NOC Debug Dump feature: `TT_METAL_NOC_DEBUG_DUMP=1` environment variable, automatic detection of missing NOC barriers, unflushed async writes
  - The `NOCDebugState` tracking system (`noc_debugging.hpp`): issue types detected -- `WRITE_FLUSH_BARRIER`, `READ_BARRIER`, `UNFLUSHED_WRITE_AT_END`, `WRITE_TO_LOCKED_CORE_LOCAL_MEM`, `WRITE_TO_LOCKED_CB`
  - Host-side NOC logging: `noc_logging.hpp` -- `ClearNocData`, `DumpNocData` for analyzing NOC traffic patterns post-run
  - Lightweight kernel asserts: `TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1`, `ebreak` instruction on failure (assert.h line ~57-60), minimal overhead suitable for production builds. Distinct from watcher `ASSERT` which requires watcher enabled
  - LLK asserts: `static_assert` for compile-time, `LLK_ASSERT` for runtime, TRISC hang on failure
  - Fabric debug tools: `fabric_erisc_dumper.py` (register monitoring, flow control debugging, polling mode), `fabric_binary_analyzer.py`
  - tt-smi: device status monitoring, board-level and chip-level reset capabilities, temperature/power monitoring
  - Reference files: `tt_metal/tools/profiler/`, `tt_metal/impl/debug/noc_debugging.hpp`

- `06_debug_delay_and_timing_perturbation.md`
  - Debug delay feature: `TT_METAL_READ_DEBUG_DELAY_CORES`, `TT_METAL_WRITE_DEBUG_DELAY_CORES`, `TT_METAL_ATOMIC_DEBUG_DELAY_CORES`
  - `WATCHER_DEBUG_DELAY` compile-time constant: number of cycles to delay each transaction
  - How artificial delays help expose race conditions: making timing-dependent hangs reproducible
  - Timing perturbation for compute: `timing_perturbation.h` -- inserting NOPs into unpack/math/pack pipelines
  - The feedback mechanism: `debug_insert_delays_msg_t.feedback` field confirms delays are being applied
  - Dispatch data collection: `TT_METAL_DISPATCH_DATA_COLLECTION` env var
  - Dispatch debug tools: `debug_tools.hpp/cpp` -- `dump_cqs` for dumping host-side CQ state, `wait_for_program_vector_to_arrive_and_compare_to_host_program_vector` for comparing host vs. device program data
  - `TT_METAL_DISPATCH_PROGRESS_UPDATE_MS`: periodic progress heartbeats from dispatch kernels for earlier detection of stalls before full timeout
  - Reference files: `tt_metal/hw/inc/internal/debug/sanitize.h` (lines 685-708), `tt_metal/hw/inc/api/debug/timing_perturbation.h`

**Covers questions:** 5 (all existing tools), 6 (tool usage details for debugging workflows)

---

### Chapter 7: Debugging Workflows and Best Practices

**Description:** Practical, step-by-step workflows for debugging hangs, from initial triage through root cause identification, combining the tools from Chapter 6 into coherent diagnostic procedures.

**Directory:** `ch07_debugging_workflows/`

**Files:**

- `01_initial_triage.md`
  - Step 0: Do not immediately reset the chip -- a hung device preserves diagnostic state
  - Step 1: Recognizing a hang -- timeout messages, unresponsive Python process, `tt-smi` showing unexpected state. Is it a device hang or host hang? Check if the host is blocked in `Synchronize`/`Finish` or in `EnqueueReadMeshBuffer`
  - Step 2: Check watcher log if watcher was enabled -- look for NOC sanitize violations, tripped asserts, stuck waypoints
  - Step 3: Run `tt-triage` for automated system health check: `./tools/tt-triage.py --verbosity=4 --dev=0`
  - Step 4: If watcher was not enabled, use `watcher_dump` tool for post-mortem mailbox inspection
  - Step 5: Check dprint output for last printed message from device
  - Step 6: Check Tracy/profiler data for last completed op. Use Inspector's `dump_running_operations` to see what was executing
  - Decision tree: based on symptoms, route to the appropriate detailed diagnosis procedure in file 02

- `02_diagnosing_by_hang_category.md`
  - **Kernel CB deadlock diagnosis:** Check watcher waypoints for `CRBW`/`CWFW` patterns, verify CB configuration matches between producer and consumer, check for mismatched push/pop counts
  - **NOC hang diagnosis:** Enable `TT_METAL_NOC_DEBUG_DUMP=1`, check for unflushed writes, verify NOC address validity with `check_noc_locations.py`, look for Blackhole-specific inline write issues
  - **Dispatch hang diagnosis:** Run `dump_fast_dispatch`, check CQ fill levels, verify program completion signals, check dispatch core waypoints
  - **Memory corruption diagnosis:** Enable watcher CB sanitization, check for L1 overflow via stack usage tracking, verify buffer allocation boundaries
  - **Multi-chip hang diagnosis:** Check `check_eth_status.py` for link failures, use `fabric_erisc_dumper.py` to monitor flow control, verify CCL participation across all chips
  - **Semaphore deadlock diagnosis:** Look for `NSW` waypoints, verify semaphore addresses are unique per coordination pair, check initial values and reset between iterations
  - Common waypoint patterns indicating specific hang types: `CRBW` on NCRISC + `CWFW` on BRISC = CB deadlock, `NSW` = semaphore hang, `NARW`/`NAWD` = NOC barrier stuck

- `03_narrowing_and_reproducing.md`
  - Binary search with op-level checkpoints: inserting `Synchronize()` calls between ops to find the hanging op
  - Using `null_kernels` mode: replacing all kernel code with no-ops to test dispatch infrastructure in isolation
  - Using `kernels_early_return` mode: kernels remain full-size but skip execution, testing kernel loading without execution
  - Using slow dispatch mode (`TT_METAL_SLOW_DISPATCH_MODE=1`) to isolate whether the hang is in the dispatch system or in kernel execution
  - Single-op isolation: extracting a failing op into a standalone test program
  - Multi-device narrowing: reducing from Galaxy to T3K to single-chip to identify scale-dependent hangs
  - Reducing core grid to isolate the problematic core
  - Strategies for reproducing intermittent hangs: stress testing, timing perturbation (`timing_perturbation.h`), NOC debug delays (`TT_METAL_READ_DEBUG_DELAY_CORES`, `TT_METAL_WRITE_DEBUG_DELAY_CORES`, `TT_METAL_ATOMIC_DEBUG_DELAY_CORES`)
  - Using the `hang_device` test operation to verify that debugging tools correctly detect hangs
  - Using trace capture/replay and LightMetal replay for deterministic reproduction of command streams

- `04_reading_watcher_and_triage_output.md`
  - Decoding waypoint strings: mapping 4-character codes to firmware execution state
  - Interpreting NOC sanitize violations: the `debug_sanitize_addr_msg_t` fields (noc_addr, l1_addr, len, which_risc, is_multicast, is_write, is_target, return_code)
  - Interpreting assert messages: `debug_assert_msg_t` fields (line_num, tripped type, which processor)
  - Reading ring buffer data: understanding application-specific values pushed via `WATCHER_RING_BUFFER_PUSH`
  - Correlating kernel IDs from `kernel_names.txt` with watcher output
  - Using `kernel_elf_paths.txt` to map kernel IDs to ELF binaries for disassembly
  - Reading callstacks from tt-triage: mapping PCs to source locations, understanding firmware vs. kernel code
  - Interpreting sync register inspection (`DumpSyncRegs`) with the caveat that "reading registers while running can cause hangs, only read if requested explicitly"

- `05_distinguishing_hw_vs_sw_bugs.md`
  - Reproducibility as a signal: deterministic hangs strongly suggest software bugs; intermittent hangs may be either
  - Environment variables for hardware fault isolation: `TT_METAL_CLEAR_L1`, `TT_METAL_CLEAR_DRAM` to start from clean state
  - `enable_hw_cache_invalidation` (Blackhole): flushing L1 data cache to expose missing invalidation bugs
  - `disable_relaxed_memory_ordering` (Blackhole): testing whether relaxed ordering causes the hang
  - `validate_kernel_binaries`: verifying kernel binary integrity after loading
  - Checking for Ethernet retrain issues: `skip_eth_cores_with_retrain` runtime option, watcher Ethernet link status feature
  - When to suspect hardware: after exhausting all software debugging, same code works on other chips, ECC errors in `tt-smi`, temperature-related failures, intermittent across different workloads
  - Disabling program cache (`set_program_cache_misses_allowed(true)`) when debugging to eliminate stale-binary issues

**Covers questions:** 6 (all debugging workflow best practices)

---

### Chapter 8: Reset Reduction, Resilience, and Future Improvements

**Description:** Analyzes the current device reset mechanisms, provides strategies for reducing hang frequency and reset needs, documents defensive coding patterns, and proposes future tooling improvements.

**Directory:** `ch08_reset_reduction_and_future/`

**Files:**

- `01_current_reset_mechanisms.md`
  - The reset hierarchy (least to most disruptive):
    1. **Graceful program termination:** killing the host process, allowing the runtime to clean up. `Device::close()` clears program cache, resets allocator, clears command queues
    2. **Tensix soft reset:** per-core reset via `tensix_soft_reset_options.cpp`, resetting individual RISC-V processors without affecting other cores
    3. **UMD warm reset:** `WarmReset::warm_reset()` in `tt_metal/third_party/umd/device/warm_reset.cpp`
       - Architecture-specific paths: `warm_reset_wormhole_legacy`, `warm_reset_blackhole_legacy`, `warm_reset_arch_agnostic`
       - M3 board-level reset via ARC message
       - Secondary bus reset option
       - Pre/post reset notification system: `WarmResetCommunication::Notifier` for coordinating with other processes
    4. **M3 board-level reset:** deeper reset via ARC message
    5. **Full system reboot:** PCIe device re-enumeration when warm reset is insufficient (fully wedged PCIe link)
  - ARM platform limitations: warm reset disabled due to instability
  - Multi-host distributed reset: the `distributed_reset.sh` script for coordinated reset across multiple hosts
  - `DeviceManager::close_devices`: ordered shutdown for multi-chip -- remote devices closed before MMIO devices, tunnels shut down farthest to closest
  - Current reset granularity: all-or-nothing chip reset, no per-core reset capability from the host API
  - When `tt-smi` reset is truly required: firmware corruption, NOC hardware deadlock (no software workaround), Ethernet link permanently down, ARC processor hang

- `02_reducing_reset_frequency_and_resilience.md`
  - **Prevention practices that reduce hang frequency:**
    - Always enable watcher NOC sanitization during development
    - Use lightweight kernel asserts (`TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1`) in production for critical invariants
    - Follow CB API constraints: ntiles must evenly divide CB size, same ntiles across calls, single-thread CB pointer updates
    - Validate NOC addresses before issuing transactions
    - Proper barrier placement: every `noc_async_write` must have a corresponding barrier before dependent reads
    - Semaphore initialization: ensure all semaphores are properly initialized before any core begins execution, unique addresses per coordination pair, reset between iterations
    - NOC transaction ordering: avoid patterns that create circular dependencies between cores
    - Timeout wrappers: wrapping `noc_semaphore_wait` with bounded retry counts and error reporting
  - **Multi-chip resilience patterns:**
    - CCL operation ordering to avoid cross-rank deadlocks
    - Ethernet link health monitoring and workload migration on link degradation
    - Fabric topology validation before launching multi-chip workloads
    - Verifying all devices participate in collectives before launching
  - **Graceful recovery mechanisms:**
    - The erisc special case: erisc cores call `erisc_exit()` on errors instead of hanging, allowing partial recovery
    - Inspector's `serialize_on_dispatch_timeout`: automatically capturing device state before giving up
    - `TT_METAL_DISPATCH_PROGRESS_UPDATE_MS`: periodic dispatch heartbeats for earlier stall detection
    - Firmware watchdog timer registers (`RISCV_DEBUG_REG_WATCHDOG_TIMER`) and their potential for detecting stuck cores
    - Watcher `auto_unpause` mode for automated recovery from paused states
  - **Test infrastructure:**
    - Systematic hang reproduction tests, watcher-based regression tests (test_assert.cpp, test_link_training.cpp, test_stack_usage.cpp, test_pause.cpp, test_noc_sanitize_delays.cpp)
    - Using the `hang_device` test operation for validating tooling

- `03_future_tooling_proposals.md`
  - **Automatic hang detection with root cause classification:** Extending watcher to automatically categorize hang type from waypoint + sanitize state, rather than requiring manual log reading. Combining watcher waypoint staleness + dispatch timeout + NOC sanitize errors into a unified hang classification system
  - **Device-side heartbeat monitoring:** Periodic heartbeat writes from each core that the host monitors -- absence of heartbeat triggers proactive dump before chip becomes unresponsive. Extending fabric telemetry heartbeats (`HEARTBEAT_TX`, `HEARTBEAT_RX`) to all core types, not just ERISC
  - **Automatic state snapshots before reset:** Capturing all L1 contents, NOC state, CB state, semaphore values before warm reset destroys the evidence. Writing diagnostic state to DRAM or host memory so it survives resets
  - **Deterministic replay of command streams:** Extending LightMetal capture to record the exact byte-level command stream written to system memory, enabling exact replay of the workload leading to a hang
  - **Better error propagation:** A structured error channel from firmware to host that propagates error codes, core identity, and context through the completion queue rather than requiring watcher polling. Firmware catches more error conditions and propagates them via mailbox rather than hanging
  - **Partial device reset:** Resetting individual Tensix cores or groups of cores without a full chip reset, using the soft reset register (`RISCV_DEBUG_REG_SOFT_RESET_0`). Host API for per-core soft reset
  - **Firmware watchdog with automatic recovery:** A timer on each core that triggers a controlled exit (like erisc's behavior) if the kernel does not complete within a configurable timeout, rather than requiring host-side detection
  - **Static analysis and pre-flight validation:**
    - Compile-time checking that reader/compute/writer tile counts are consistent
    - CB size validation at program creation time: catching misconfigurations before they reach the device
    - Verifying that every CB producer has a matching consumer, every NOC write has a barrier
  - **Resilient CCL operations:** Timeouts with graceful fallback (retry, reroute around failed links, exclude failed chips from collective). Automatic rerouting on fabric link failure
  - **Enhanced NOC debug infrastructure:** Stabilizing `TT_METAL_NOC_DEBUG_DUMP` as a production-ready missing-barrier detector. NOC transaction replay and analysis for post-mortem deadlock identification. Real-time NOC utilization metrics
  - **Workload resilience techniques:** Checkpoint/restart for long-running workloads, automatic retry of individual ops after soft reset, timeout-based fallback paths in CCL operations

**Covers questions:** 7 (all future tool proposals), 8 (all reset reduction strategies)

---

## Conventions

### Terminology

| Term | Definition |
|------|-----------|
| **Hang** | A state where a program stops making forward progress and the device becomes unresponsive to normal commands, requiring external intervention (reset or reboot) to recover. Always a RISC-V processor spinning in a wait loop whose exit condition cannot be satisfied. |
| **Stall** | A temporary pause in execution that will eventually resolve (e.g., waiting for a NOC transfer to complete). Distinguished from a hang in that stalls are expected behavior. |
| **Deadlock** | A specific type of hang where two or more entities (cores, RISCs, chips) are each waiting for the other, forming a circular dependency. |
| **Spin-loop** | The `while(1){}` infinite loop that firmware enters after detecting an error via watcher sanitization or assertion. This is the immediate mechanism of most soft hangs. |
| **Soft hang** | A firmware-enforced hang (e.g., `while(1){}` after assert failure), distinguishable from a hardware-level lockup. |
| **Watcher** | The combined host-side polling server + device-side mailbox infrastructure for monitoring core health. |
| **Waypoint** | A 4-character status code written by firmware/kernel code to L1 mailbox, identifying where each RISC-V core currently is in execution. |
| **NOC** | Network-on-Chip -- the on-die interconnect for data transfers between cores, DRAM, and PCIe. Two instances (NOC0, NOC1) traverse the chip in opposite directions. |
| **CB** | Circular Buffer -- the producer/consumer FIFO mechanism for passing data between RISC processors within a Tensix core. Implemented in L1 SRAM. |
| **CCL** | Collective Communication Library -- the set of multi-chip collective operations (all_gather, reduce_scatter, etc.). |
| **EDM** | Ethernet Data Mover -- the fabric router firmware running on Ethernet cores. |
| **BRISC/NCRISC/TRISC** | The three types of RISC-V processors within each Tensix core: BRISC (data movement 0), NCRISC (data movement 1), TRISC0/1/2 (compute unpack/math/pack). |
| **ERISC/AERISC** | Ethernet RISC-V cores for inter-chip communication. Variants: active ERISC, idle ERISC, subordinate ERISC. |
| **tt-smi** | Tenstorrent System Management Interface -- the command-line tool for chip status monitoring and reset. |
| **tt-triage** | Post-mortem and live analysis tool that runs a suite of diagnostic scripts on a hung or post-hang system. |
| **Inspector** | Always-on host runtime telemetry system with RPC interface for querying Metal program/workload state. |
| **UMD** | User-Mode Driver -- the low-level driver library for PCIe communication with Tenstorrent chips. |
| **T3K** | A system of 8 Wormhole B0 chips connected via Ethernet. |
| **Galaxy (TG/TGG)** | Large-scale multi-chip systems (32+ chips) with mesh Ethernet topology. |
| **Fast Dispatch** | The default command queuing mode using dedicated RISC-V cores (prefetcher/dispatcher) for asynchronous command processing. |
| **Slow Dispatch** | Alternative mode (`TT_METAL_SLOW_DISPATCH_MODE=1`) that bypasses the command queue for synchronous host-managed execution. Useful for diagnostic isolation. |
| **MMIO Device** | A chip directly accessible by the host via memory-mapped I/O. Remote devices are accessed through Ethernet tunnels from MMIO devices. |
| **Warm Reset** | A chip reset via PCIe secondary bus reset or ARC message that reinitializes the chip without rebooting the host. |

### Notation

- Code references use the format `path/to/file.cpp:LINE` for specific line references, or `path/to/file.cpp` for general file references.
- All paths are relative to the tt-metal repository root.
- Environment variables are written in `MONOSPACE_CAPS` (e.g., `TT_METAL_WATCHER=120`).
- Firmware-side code is marked with `[device]` and host-side code with `[host]` when the distinction matters.
- Register names and hardware constants are written in `ALL_CAPS_WITH_UNDERSCORES`.
- Watcher waypoint codes are written in uppercase monospace (e.g., `CRBW`).
- NOC addresses are shown in hexadecimal.
- Core coordinates are shown as `(x, y)` in logical coordinates unless marked as `[phys]`.
- NOC sanitize return codes reference the `debug_sanitize_noc_return_code_enum` values.

### Formatting Rules

- Each file begins with a one-paragraph summary of its contents and a "Prerequisites" note listing which prior files/chapters should be read first.
- Every hang cause is documented using the five-part format: **(1) Symptom** -- what the developer observes, **(2) Root Cause** -- what is actually happening in hardware/firmware, **(3) Diagnosis Steps** -- which tools to use and what to look for, **(4) Fix** -- the code change or configuration needed, **(5) Prevention** -- how to avoid the issue in future code.
- Code snippets include the source file path as a comment on the first line. Both the buggy pattern (causing the hang) and the corrected pattern are shown where applicable.
- Diagrams use ASCII art for portability.
- "Danger" callouts mark patterns that are known to cause hangs.
- "Tip" callouts mark debugging shortcuts and best practices.
- Tables are used for structured catalogs (env vars, waypoint codes, error codes, triage scripts).
- Cross-references to other chapters/files use the format `(see Chapter N, file_name.md)`.
- Code examples use C++ for device-side code and Python for host-side code.

---

## Cross-Chapter Dependencies

| Chapter | Depends On | Concepts Referenced |
|---------|-----------|-------------------|
| **Ch 2** (Kernel & NOC Hangs) | **Ch 1** | Hang taxonomy categories, blocking primitives taxonomy, architecture differences |
| **Ch 3** (Memory Hangs) | **Ch 1**, **Ch 2** | Hang taxonomy, NOC sanitization return codes, NOC address validation pipeline, CB deadlock patterns (L1 corruption can be caused by CB overflow from Ch2) |
| **Ch 4** (Dispatch & Host-Device) | **Ch 1**, **Ch 2** | Hang taxonomy, kernel synchronization model, NOC semantics |
| **Ch 5** (Multi-Chip & CCL) | **Ch 1**, **Ch 2**, **Ch 4** | Hang taxonomy, semaphore semantics, NOC barriers, dispatch architecture (multi-chip dispatch topology extends single-chip model from Ch4) |
| **Ch 6** (Debugging Tools) | **Ch 1**, **Ch 2** | All hang categories (tools are organized by what they detect), NOC sanitization codes, waypoint meanings. Can be consulted as a reference at any time |
| **Ch 7** (Debugging Workflows) | **Ch 1-6** | All previous chapters -- workflows combine understanding of hang causes with knowledge of tools. The integration chapter |
| **Ch 8** (Reset Reduction & Future) | **Ch 1-7** | All previous chapters -- proposals address gaps identified throughout the guide, resilience patterns build on debugging knowledge |

**Reading order:** Chapters 1-3 form the foundational layer and should be read sequentially. Chapters 4-5 can be read in either order but both require Chapters 1-3. Chapter 6 is a reference chapter that can be consulted at any time but benefits from understanding Chapters 1-5. Chapter 7 synthesizes everything and requires all prior chapters. Chapter 8 is forward-looking and can be read after any level of engagement with the prior material.

**Fast-path for active hangs:** A developer experiencing an active hang may jump directly to Chapter 7 (workflow) with Chapter 1 (taxonomy + blocking primitives) as a prerequisite, then follow references back to the relevant detail chapters (2-5) and tool chapters (6) as needed.

---

## Question Coverage Matrix

| Question | Primary Coverage | Secondary Coverage |
|----------|-----------------|-------------------|
| Q1: All known root cause categories (kernel deadlocks, NOC, L1, dispatch, firmware) | Ch1 (taxonomy + blocking primitives), Ch2 (kernel & NOC detail) | Ch3 (memory), Ch4 (dispatch) |
| Q2: Multi-chip/CCL-specific hang causes | Ch5 (all multi-chip/CCL content) | Ch1 (arch differences for multi-chip configs) |
| Q3: Host-device interaction hang causes | Ch4 (dispatch, sync, trace, LightMetal) | Ch7 (debugging workflows for host-device issues) |
| Q4: Memory-related hang causes | Ch3 (dedicated memory chapter: L1 corruption, DRAM backpressure, alignment, OOM) | Ch2 (CB overflow via NOC sanitization) |
| Q5: Existing debugging tools (tt-smi, watcher, Tracy, dispatch debug, assertions, watchdogs) | Ch6 (complete tool catalog: watcher, watcher_dump, DPrint, tt-triage, profiler/Tracy/NOC debug, debug delay/timing) | -- |
| Q6: Developer debugging workflows and best practices | Ch7 (all debugging workflows: triage, category diagnosis, narrowing, reading output, HW vs SW) | Ch6 (tool usage details) |
| Q7: Future tooling improvements | Ch8 file 03 (all future proposals) | Ch4 file 03 (LightMetal as future replay tool) |
| Q8: Reducing chip reset frequency | Ch8 files 01-02 (reset mechanisms, prevention practices, resilience patterns, graceful recovery) | Ch7 (recovery procedures, avoiding unnecessary resets) |
| Q9: Cross-generation (GS/WH/BH) and cross-config (single/T3K/Galaxy) differences | Ch1 file 04 (architecture + config differences) | Ch5 file 03 (multi-chip topology specifics), Ch8 file 01 (per-arch reset paths) |
