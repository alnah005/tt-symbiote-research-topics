# 7.1 Initial Triage

[Next: Diagnosing by Hang Category](./02_diagnosing_by_hang_category.md)

---

When a hang occurs, the first 60 seconds determine how much diagnostic information survives. A premature chip reset destroys the device-side state -- waypoints, mailbox contents, NOC transaction counters, L1 memory -- that is often the only path to root cause. This section provides a structured first-response workflow: recognize the hang, preserve diagnostic state, run automated triage, and route to the correct category-specific procedure.

**Prerequisites:** [Chapter 1, `01_what_is_a_hang.md`](../ch01_anatomy_of_a_hang/01_what_is_a_hang.md) (hang definition, symptoms), [Chapter 1, `03_hang_taxonomy.md`](../ch01_anatomy_of_a_hang/03_hang_taxonomy.md) (six-category classification), [Chapter 6](../ch06_debugging_tools/) (tool overview -- detailed tool usage is referenced inline).

---

## Step 0: Do NOT Immediately Reset the Chip

> **DANGER:** The single most common mistake during hang debugging is immediately running `tt-smi -r` or rebooting. A hung chip is a *diagnostic goldmine* -- every core's waypoint, every L1 mailbox, every NOC transaction counter, every dispatch queue fill level is frozen in the exact state that caused the hang. Once you reset, all of this is gone.

**Rule:** Never reset until you have either (a) collected all diagnostic data described in Steps 1-6, or (b) confirmed the chip is completely PCIe-inaccessible (in which case there is nothing to collect).

The only exception is when the hang is blocking a shared CI/CD pipeline and other developers need the hardware. Even then, capture what you can in 30 seconds (Steps 1-3) before resetting.

**What to avoid:**
- Do NOT kill the hung process with `kill -9` (this triggers device close, which resets chips)
- Do NOT run `tt-smi -r` (full reset)
- Do NOT start a new program that opens the device (this triggers reset on the previous state)
- Do NOT reboot the machine

**What IS safe:**
- Reading from the device via tt-triage (read-only PCI access)
- Checking host-side log files
- Running `ps`, `top`, `strace -p <pid>` on the host process
- Examining the watcher log file (it is written to disk periodically)
- Examining Inspector log files under `generated/inspector/`

**If the process has already been killed** but the machine has not been rebooted and no other process has opened the devices, the device state is still intact. You can still run tt-triage.

### Multi-Chip Extension: Step 0

On multi-chip systems (T3K, Galaxy), the no-reset rule applies to **ALL** devices in the mesh, not just the one you suspect is hung. A cascading failure typically starts on one device and propagates to others -- resetting any device destroys the evidence chain. Coordinate with other operators or hosts sharing the same mesh before any reset.

---

## Step 1: Recognize the Hang -- Device, Host, or Fabric?

The first diagnostic question is whether the hang is on the device, the host, or involves the multi-chip fabric:

### Symptom: Host Process Appears Frozen

```
Is the host Python/C++ process consuming CPU?
  |
  +-- YES (high CPU, spinning) --> Host is likely blocked in a polling loop
  |   waiting for the device. This is the most common case.
  |   Check: is it in Synchronize/Finish? (host-device sync hang)
  |          is it in EnqueueReadMeshBuffer? (buffer read waiting on device)
  |          is it in SystemMemoryManager fetch queue wait? (dispatch backpressure)
  |
  +-- NO (zero CPU, sleeping) --> Host may be blocked on a mutex, a condition
      variable, or an I/O operation. This suggests a host-side hang, not device.
      Check: is another process holding a device lock?
             is there a host-side deadlock in the async dispatch queue?
```

**Quick test:** If you have a separate terminal, run `tt-smi`. If it can query the device, the chip is not fully wedged and diagnostic reads are possible.

### Symptom: Timeout Error Message

The tt-metal runtime provides two primary timeout messages:

| Message | Source | Meaning |
|---------|--------|---------|
| `TIMEOUT: device timeout, potential hang detected, the device is unrecoverable` | `completion_queue_wait_front` | The host polled the completion queue for longer than `TT_METAL_OPERATION_TIMEOUT_SECONDS` without receiving a completion signal from the device. The device never finished the last dispatched operation. |
| `TIMEOUT: device timeout in fetch queue wait, potential hang detected` | `system_memory_manager.cpp` (`loop_and_wait_with_timeout`) | The host is trying to write new commands to the fetch queue, but the device's prefetch kernel has not consumed previous entries. The prefetch kernel is likely hung or the dispatch pipeline is stalled. |
| `Timed out writing init magic` | `dprint_server.cpp` | The DPrint server could not write initialization magic to a core's DPrint buffer. Usually a symptom of a prior hang, not the root cause. |
| `Timed out waiting on debug print server to read data.` | `dprint_server.cpp` | The DPrint ring buffer is full because the host is not reading fast enough, or the host-side DPrint thread is blocked. |
| `Timed out waiting on watcher server thread to terminate.` | `watcher_server.cpp` | Generally benign -- shutdown ordering issue. |

If a dispatch timeout fires, proceed to Step 2.

### Symptom: tt-smi Shows Unexpected State

Run `tt-smi` from another terminal:

```bash
tt-smi
```

Check for:
- **Device still present on PCIe bus:** If `tt-smi` can query the device, the chip is not fully wedged. Diagnostic reads are possible.
- **Elevated temperature:** If temperature is near or above throttling threshold (typically 85-105C for the ASIC junction), a thermal protection mechanism may have halted the chip.
- **ECC errors reported:** ECC errors suggest hardware-level memory corruption (see [Section 05](./05_distinguishing_hw_vs_sw_bugs.md)).
- **Device not found:** PCIe link is down. The chip is fully wedged. Only a reboot will recover. No diagnostics are possible beyond host-side logs.

### Determining Device Hang vs. Host Hang

```
Can tt-smi still query the device?
  |
  +-- NO --> PCIe link is down. Full reboot required. Skip to "No Diagnostics Possible" below.
  |
  +-- YES
       |
       Does the host process have a timeout error?
         |
         +-- YES --> Device hang. The host dispatched work and the device never completed it.
         |           Proceed to Step 2.
         |
         +-- NO
              |
              Is the host process consuming CPU?
                |
                +-- YES --> Likely device hang. The host is polling for a completion signal.
                |           Proceed to Step 2.
                |
                +-- NO --> Likely host-side issue (lock contention, async queue deadlock,
                           or a bug in host-side code). Attach GDB to the host process:
                           gdb -p <pid>
                           (gdb) thread apply all bt
                           Look for blocked mutexes, condition variable waits, or
                           stuck event loops in the dispatch thread.
```

### Multi-Chip Extension: Multi-Chip Hang Classification

On multi-chip systems, add a third classification: **fabric hang**. A fabric hang is a device-side hang that originates from cross-device communication failure.

```
Multi-chip system? (T3K, Galaxy, N300)
  |
  +-- YES
  |    |
  |    Are some devices visible via tt-smi but others are not?
  |      |
  |      +-- YES --> Mixed failure. The missing devices may have PCIe link issues.
  |      |           Check `dmesg` for PCIe errors on those devices.
  |      |
  |      +-- NO (all visible)
  |           |
  |           Are ERISC cores stuck at fabric wait points on any device?
  |             |
  |             +-- YES --> Fabric hang. Proceed to Step 4 (ERISC State Inspection).
  |             |
  |             +-- NO --> Standard device hang. Continue with Steps 2-3.
  |
  +-- NO --> Single-chip system. Continue with Steps 2-3.
```

---

## Step 2: Check the Watcher Log (If Watcher Was Enabled)

If watcher was enabled during the run (`TT_METAL_WATCHER=<interval_ms>` was set), the most valuable diagnostic data is already on disk.

### Locating the Watcher Log

The watcher log is written to `generated/watcher/watcher.log` relative to `TT_METAL_RUNTIME_ROOT` (or the build output directory). Kernel name mappings are in `generated/watcher/kernel_names.txt`.

```bash
# Check the most recent watcher log entries
tail -100 generated/watcher/watcher.log
```

### What to Look For

1. **NOC sanitize violation:** A line containing `SANITIZE` followed by a return code (e.g., `DebugSanitizeNocAddrOverflow`). This means the watcher detected an illegal NOC transaction and the offending core entered a deliberate `while(1)` hang. The violation details identify the exact root cause. See [Section 04](./04_reading_watcher_and_triage_output.md) for decoding.

2. **Assert tripped:** A line containing `ASSERT` with a line number and file reference. The kernel code hit an `ASSERT()` macro and entered a deliberate hang. The line number and kernel ID tell you exactly where. See [Chapter 6, `01_watcher_system.md`](../ch06_debugging_tools/01_watcher_system.md).

3. **Stuck waypoints:** Multiple consecutive log entries showing the same waypoint code on a core. Common patterns:

   | Stuck Waypoint | Likely Category | Where to Go |
   |---------------|-----------------|-------------|
   | `CRBW` or `CWFW` | Kernel CB deadlock | [Section 02, Kernel CB Deadlock](./02_diagnosing_by_hang_category.md#kernel-cb-deadlock-diagnosis) |
   | `NRBW` or `NWBW` | NOC barrier hang | [Section 02, NOC Hang](./02_diagnosing_by_hang_category.md#noc-hang-diagnosis) |
   | `NSW` or `NSMW` | Semaphore deadlock | [Section 02, Semaphore Deadlock](./02_diagnosing_by_hang_category.md#semaphore-deadlock-diagnosis) |
   | `HQW` | Dispatch prefetch waiting for host | [Section 02, Dispatch Hang](./02_diagnosing_by_hang_category.md#dispatch-hang-diagnosis) |
   | `PWW`/`WCW` | Dispatch waiting for workers | [Section 02, Dispatch Hang](./02_diagnosing_by_hang_category.md#dispatch-hang-diagnosis) |
   | `CBRW` | Dispatch CB page release stall | [Section 02, Dispatch Hang](./02_diagnosing_by_hang_category.md#dispatch-hang-diagnosis) |
   | `RP2W` | NOC command buffer stall | [Section 02, NOC Hang](./02_diagnosing_by_hang_category.md#noc-hang-diagnosis) |

4. **Ethernet link down:** A line reporting a link-down event with a retraining count. This suggests a multi-chip fabric issue. See [Section 02, Multi-Chip Hang](./02_diagnosing_by_hang_category.md#multi-chip-hang-diagnosis).

5. **L1 corruption detected:** A `DumpL1Status` failure indicating the firmware launch address at address 0 was overwritten. This is a memory corruption issue. See [Section 02, Memory Corruption](./02_diagnosing_by_hang_category.md#memory-corruption-diagnosis).

### If the Watcher Log Has Clear Evidence

If the watcher log contains a sanitize violation or a tripped assert, you often have enough information to identify the root cause without further triage. Decode the violation (see [Section 04](./04_reading_watcher_and_triage_output.md)), map the kernel ID to a source file via `kernel_names.txt`, and proceed to the appropriate chapter for the specific scenario:

- Sanitize violations: [Chapter 2, `03_noc_address_sanitization_and_violations.md`](../ch02_kernel_and_noc_hangs/03_noc_address_sanitization_and_violations.md)
- CB-related violations: [Chapter 2, `02_circular_buffer_deadlocks.md`](../ch02_kernel_and_noc_hangs/02_circular_buffer_deadlocks.md)
- Memory-related violations: [Chapter 3](../ch03_memory_related_hangs/)
- Dispatch-related stuck waypoints: [Chapter 4](../ch04_dispatch_and_host_device_hangs/)

### If Watcher Was NOT Enabled

Proceed to Step 3 (tt-triage) and Step 4 (watcher_dump or ERISC state).

> **Tip:** For development and testing, always enable watcher: `export TT_METAL_WATCHER=120`. The 120ms polling interval has minimal performance impact for most workloads and captures critical diagnostic data. Disable only when benchmarking or in production deployments where the binary size overhead matters.

### Multi-Chip Extension: Cross-Host Watcher Logs

On multi-host Galaxy systems, collect watcher logs from **all** hosts. Copy them to a single location and align timestamps across hosts to determine which device exhibited the first anomaly:

```bash
# On each host:
scp generated/watcher/watcher.log central_debug_host:/tmp/watcher_host_N.log
```

---

## Step 3: Run tt-triage for Automated Health Check

`tt-triage` is the most comprehensive single-step diagnostic tool. It runs a suite of scripts that extract callstacks, check NOC status, verify Ethernet links, inspect dispatch state, and more.

### Basic Usage

```bash
# Run all triage scripts on device 0 at maximum verbosity
./tools/tt-triage.py --verbosity=4 --dev=0
```

### Key Options

| Option | Purpose |
|--------|---------|
| `--dev=N` | Target device number (default: 0) |
| `--verbosity=N` | Output detail level (0-4; use 4 for hang diagnosis) |
| `--run=<script>` | Run a specific triage script only |
| `--all-cores` | Inspect all cores, not just those with anomalies |
| `--remote-exalens` | Connect to a remote ttexalens instance |
| `--initialize-with-noc1` | Use NOC1 for register reads (if NOC0 is suspected to be hung) |

### What tt-triage Produces

At verbosity 4, tt-triage runs these key scripts (among others):

| Script | What It Checks | Relevant Hang Categories |
|--------|---------------|--------------------------|
| `dump_callstacks.py` | Per-core RISC-V program counters and callstacks | All -- identifies exactly where each core is stuck |
| `dump_aggregated_callstacks.py` | Groups cores by PC value -- shows common hang points | All -- reveals if many cores are stuck at the same place |
| `dump_fast_dispatch.py` | Prefetch/dispatch kernel state, CQ fill levels | Dispatch hangs ([Ch4](../ch04_dispatch_and_host_device_hangs/)) |
| `check_noc_status.py` | Outstanding NOC transaction counts, stuck transfers | NOC hangs ([Ch2, Section 03-04](../ch02_kernel_and_noc_hangs/)) |
| `check_noc_locations.py` | Validates NOC addresses in pending transactions | NOC hangs, memory hangs |
| `check_eth_status.py` | Ethernet link status, link-down detection | Multi-chip hangs ([Ch5](../ch05_multi_chip_and_ccl_hangs/)) |
| `check_cb_inactive.py` | Detects idle circular buffers (nonzero = NOC likely hung) | Kernel CB deadlocks ([Ch2, Section 02](../ch02_kernel_and_noc_hangs/02_circular_buffer_deadlocks.md)) |
| `check_binary_integrity.py` | Verifies loaded kernel binaries match expected | Memory corruption ([Ch3, Section 01](../ch03_memory_related_hangs/01_l1_memory_corruption_and_overflow.md)) |
| `check_core_magic.py` | Core magic values intact (L1 corruption check) | Memory corruption |
| `dump_running_operations.py` | Identifies which Metal-level operation was executing | All -- tells you what the runtime was doing |
| `dump_watcher_ringbuffer.py` | Ring buffer contents from watcher mailboxes | All -- application-specific debug markers |
| `dump_lightweight_asserts.py` | Extracts `ebreak`-triggered assert information | All -- lightweight asserts ([Ch6, Section 05](../ch06_debugging_tools/05_profiler_tracy_and_noc_debug.md)) |
| `dump_risc_debug_signals.py` | RISC-V debug signal state | All |
| `check_arc.py` | ARC processor health | Hardware issues |

### Auto-Triggering tt-triage on Timeout

To run tt-triage automatically when a dispatch timeout occurs:

```bash
export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="./tools/tt-triage.py --verbosity=4"
```

This configures `MetalContext::on_dispatch_timeout_detected()` to invoke tt-triage before the runtime gives up. The Inspector data (programs, workloads, mesh state) is also serialized automatically on timeout via `InspectorSettings` (see [Chapter 6, `04_tt_triage_tool.md`](../ch06_debugging_tools/04_tt_triage_tool.md)).

---

## Step 4: Use watcher_dump for Post-Mortem Mailbox Inspection / Inspect ERISC State

### watcher_dump

If watcher was NOT enabled during the run, the watcher log will not exist. However, the `watcher_dump` standalone tool can read watcher mailboxes and dispatch queue state from a hung chip *after the fact*.

#### How watcher_dump Works

`watcher_dump` creates a minimal device connection using `Device::initialize` with `minimal=true`, which skips firmware/watcher/DPrint initialization. This allows it to attach to a hung chip without disturbing its state and read:

1. **Watcher mailboxes:** Even without watcher enabled, firmware writes waypoint and status information to fixed L1 addresses. `watcher_dump` reads these and decodes the last known state.
2. **Command queue state:** Issue queue and completion queue fill levels, via `dump_cqs()`.
3. **NOC transfer logging:** If `NOC_LOGGING_ENABLED` was set, histogram data of NOC transfer sizes.

#### Usage

```bash
# From the tt-metal build directory
./build/tools/watcher_dump/watcher_dump
```

#### Limitations

- The device must still be PCIe-accessible. If `tt-smi` cannot see the device, `watcher_dump` will also fail.
- Without watcher enabled during the run, NOC sanitize violations and asserts will not be captured -- `watcher_dump` can only read what firmware wrote to the mailboxes, not what the watcher polling thread would have decoded.
- Waypoint data may be stale if the firmware did not reach a `WAYPOINT()` call before the hang point.

#### When to Prefer watcher_dump Over tt-triage

| Situation | Use |
|-----------|-----|
| First-time hang with no prior setup | tt-triage (more comprehensive) |
| Watcher was disabled, need mailbox data | watcher_dump (reads raw mailboxes) |
| Need dispatch CQ state specifically | watcher_dump with `dump_cqs()` |
| Need callstacks and aggregated analysis | tt-triage (has `dump_callstacks.py`) |
| Remote device without ttexalens | watcher_dump (direct PCIe read) |

Reference: `tt_metal/tools/watcher_dump/watcher_dump.cpp`, `tt_metal/impl/dispatch/debug_tools.hpp` (see [Chapter 6, `02_watcher_dump_tool.md`](../ch06_debugging_tools/02_watcher_dump_tool.md)).

### ERISC State Inspection (Multi-Chip Systems)

On multi-chip systems, the Ethernet RISC (ERISC) processors manage fabric communication. Their state is critical for diagnosing multi-chip hangs.

#### Using fabric_erisc_dumper

```bash
# Monitor fabric EDM state
python3 tools/triage/fabric_erisc_dumper.py --dev=0 --polling
```

The fabric ERISC dumper reads stream register values that control EDM flow control:

| Register Group | Healthy Value | Stuck Indicator |
|---------------|---------------|-----------------|
| BUF_SPACE_AVAILABLE (sender streams) | High (buffer has space) | Zero or very low (buffer full, no consumption) |
| BUF_SPACE_AVAILABLE (ack streams) | Low/Zero (remote consuming immediately) | High (remote side not consuming, credits accumulating) |

If a sender stream's BUF_SPACE_AVAILABLE is zero and the corresponding receiver stream on the remote chip has not consumed data, the fabric is stalled. The root cause is on the receiving side -- the receiver's local worker has not pulled data from the EDM's output buffer.

#### Cross-Device Ethernet Correlation

For each active Ethernet core on a device, there is a corresponding active Ethernet core on the connected remote device. To trace a fabric stall:

1. Identify the stuck ERISC core (from watcher log or fabric_erisc_dumper).
2. Determine which remote chip and ERISC core it connects to (from the mesh topology/control plane).
3. Check the remote ERISC core's state. If both sides are blocked on flow control, follow the chain: the remote ERISC's worker consumer is the next link to check.
4. Continue tracing through the ring/linear topology until you find the device where the original stall occurred.

---

## Step 5: Check DPrint Output

If DPrint was enabled (`TT_METAL_DPRINT_CORES` was set), the device-side print buffer may contain the last message printed before the hang. This is especially valuable for narrowing the hang location within a kernel.

```bash
# Check for DPrint output file
cat dprint_output.txt 2>/dev/null
# Or check the configured DPRINT file
cat $TT_METAL_DPRINT_FILE 2>/dev/null
```

**What to look for:**
- The last printed message indicates the last successful execution point. If your kernel has progress markers (`DPRINT << "reached checkpoint 3\n"`), the absence of "checkpoint 4" tells you the hang is between checkpoints 3 and 4.
- If DPrint shows output from some cores but not others, the silent cores are likely the ones that hung.

**Caveat:** DPrint can itself cause hangs if the print buffer fills up and the kernel blocks waiting for the host to drain it. If you suspect a DPrint-induced hang, disable DPrint and check if the hang persists. The `server_killed_due_to_hang_` flag in `DPrintServer` indicates this condition (see [Chapter 6, `03_dprint_server.md`](../ch06_debugging_tools/03_dprint_server.md)).

**Important:** DPrint and device profiler (Tracy) are mutually exclusive. If Tracy was enabled, DPrint will not be available.

---

## Step 6: Check Tracy/Profiler Data and Inspector State

### Tracy Profiler

If Tracy profiling was enabled, the profiler data shows the last completed operation timeline. This helps answer "which op was running when the hang occurred?"

```bash
# Process device profiler logs
python3 tt_metal/tools/profiler/process_device_log.py
python3 tt_metal/tools/profiler/process_ops_logs.py
```

The last completed zone in the Tracy timeline is the operation that finished immediately before the hanging one. The *next* zone (incomplete) is the one that hung.

### Inspector: dump_running_operations

The Inspector system maintains always-on telemetry about Metal runtime state. Even without Tracy, you can query what was running:

```bash
# If tt-triage is available, this is the targeted approach:
./tools/tt-triage.py --run=dump_running_operations --dev=0
```

This returns the Metal-level operation (e.g., `ttnn.matmul`, `ttnn.all_gather`) that was in progress when the hang occurred, along with its tensor dimensions and configuration. This narrows the search space from "somewhere in the model" to a specific op.

### Inspector Logs (If Inspector Was Enabled)

If `TT_METAL_INSPECTOR=1` was set, Inspector logs are in `generated/inspector/`:

| File | Content |
|------|---------|
| `startup.yaml` | Session start timestamp |
| `kernels.yaml` | Kernel compilation records with watcher IDs and source paths |
| `programs_log.yaml` | Program create/compile/destroy lifecycle |
| `mesh_devices_log.yaml` | Device mesh creation and initialization |
| `mesh_workloads_log.yaml` | Workload execution records |

The `kernels.yaml` file is especially useful for mapping watcher kernel IDs to source code when `kernel_names.txt` is unavailable.

Reference: `tt_metal/impl/debug/inspector/inspector.cpp`, `data.cpp`, `types.hpp` (see [Chapter 6, `04_tt_triage_tool.md`](../ch06_debugging_tools/04_tt_triage_tool.md)).

---

## Decision Tree: Routing to Diagnosis Procedures

After completing Steps 1-6, use this decision tree to route to the appropriate category-specific diagnosis in [Section 02](./02_diagnosing_by_hang_category.md):

```
START: You have a hang. What evidence do you have?
  |
  +-- Watcher log shows SANITIZE violation?
  |   YES --> [Section 02: NOC Hang Diagnosis] or [Ch2, Section 03]
  |           The violation details tell you the root cause directly.
  |
  +-- Watcher log shows ASSERT tripped?
  |   YES --> Decode assert (Section 04). The line number and kernel ID
  |           point directly to the failing code.
  |           If inter-kernel data race assert (codes 4-7): add missing barrier.
  |
  +-- Watcher/triage shows stuck waypoints?
  |   |
  |   +-- CRBW or CWFW on worker cores?
  |   |   --> [Section 02: Kernel CB Deadlock Diagnosis]
  |   |       Cross-ref: [Ch2, 02_circular_buffer_deadlocks.md]
  |   |
  |   +-- NRBW or NWBW on worker cores?
  |   |   --> [Section 02: NOC Hang Diagnosis]
  |   |       Cross-ref: [Ch2, 04_noc_barrier_and_semaphore_hangs.md]
  |   |
  |   +-- NSW or NSMW on worker cores?
  |   |   |
  |   |   Is this a multi-chip system running a CCL op?
  |   |     +-- YES --> [Section 02: Multi-Chip Hang Diagnosis]
  |   |     +-- NO  --> [Section 02: Semaphore Deadlock Diagnosis]
  |   |       Cross-ref: [Ch2, 04_noc_barrier_and_semaphore_hangs.md]
  |   |
  |   +-- HQW / PWW / WCW / CBRW / DCW on dispatch cores?
  |   |   --> [Section 02: Dispatch Hang Diagnosis]
  |   |       Cross-ref: [Ch4, 01_dispatch_architecture_and_hang_points.md]
  |   |
  |   +-- RP2W (NOC command buffer wait)?
  |       --> [Section 02: NOC Hang Diagnosis]
  |           Possible hardware-level NOC stall.
  |           Cross-ref: [Ch2, 04_noc_barrier_and_semaphore_hangs.md]
  |
  +-- ERISC cores stuck at fabric wait points?
  |   YES --> [Section 02: Multi-Chip Hang Diagnosis]
  |           Cross-ref: [Ch5, 01_ethernet_and_fabric_fundamentals.md]
  |
  +-- tt-triage check_eth_status.py reports link down?
  |   YES --> [Section 02: Multi-Chip Hang Diagnosis]
  |           Cross-ref: [Ch5, 01_ethernet_and_fabric_fundamentals.md]
  |
  +-- tt-triage check_cb_inactive shows active CBs?
  |   YES --> NOC is actively hung.
  |           [Section 02: NOC Hang Diagnosis]
  |
  +-- tt-triage check_noc_status shows mismatches?
  |   YES --> NOC transaction incomplete.
  |           [Section 02: NOC Hang Diagnosis]
  |
  +-- tt-triage dump_fast_dispatch shows CQ anomaly?
  |   YES --> [Section 02: Dispatch Hang Diagnosis]
  |           Cross-ref: [Ch4]
  |
  +-- tt-triage check_binary_integrity fails?
  |   YES --> [Section 02: Memory Corruption Diagnosis]
  |           Kernel binary was corrupted after loading.
  |           Cross-ref: [Ch3, 01_l1_memory_corruption_and_overflow.md]
  |
  +-- All cores DONE but host stuck?
  |   YES --> Host-side hang. Attach GDB to the host process.
  |           Not a device hang.
  |
  +-- No clear evidence from any tool?
      --> [Section 03: Narrowing and Reproducing]
          Use binary search, null_kernels, slow dispatch, etc.
          to isolate the problem.
```

### Multi-Chip Hang Fast-Path

If you are debugging on a T3K or Galaxy and the hang is during a CCL operation (the host stack shows `all_gather`, `reduce_scatter`, `all_reduce`, or similar), skip directly to the multi-chip diagnosis in Section 02 regardless of the waypoint pattern. On multi-chip systems, even a `CRBW` (CB wait) on a local Tensix core may be caused by a remote device failing to send data through the fabric, which in turn caused the local CCL worker to stall, which in turn caused the local CB producer to stall.

---

## "No Diagnostics Possible" Path

If `tt-smi` cannot see the device (PCIe link is down):

1. **Capture host-side logs:** Kernel `dmesg` output, syslog, any host-side error messages from the tt-metal runtime.
2. **Check `dmesg` for PCIe errors:** `dmesg | grep -i pcie` -- look for AER (Advanced Error Reporting) messages, link down events.
3. **Record the workload:** Note the exact command, model, and configuration that triggered the hang.
4. **On multi-chip systems, check the other hosts.** Even if your local device is PCIe-dead, the remote devices may still be queryable and their watcher/ERISC state can reveal what the now-dead device was doing when it failed.
5. **Reboot the host.** This is the only recovery path when PCIe is down.
6. **After reboot:** Reproduce with watcher enabled (`TT_METAL_WATCHER=200`) to capture the diagnostic state that was lost.

---

## Quick-Reference: Environment Variables for Triage

These environment variables should be set *before* running the workload that hangs, to maximize diagnostic capture:

| Variable | Value | Effect | When to Use |
|----------|-------|--------|-------------|
| `TT_METAL_WATCHER` | `120` | Enable watcher with 120ms polling | Always during development |
| `TT_METAL_DPRINT_CORES` | `0,0` (or specific range) | Enable DPrint on target cores | When narrowing to specific kernels |
| `TT_METAL_DPRINT_RISCVS` | `0,1,2,3,4` | Which RISC-V harts to print | When narrowing to specific RISCs |
| `TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE` | `./tools/tt-triage.py --verbosity=4` | Auto-run triage on timeout | Always in CI, recommended locally |
| `TT_METAL_OPERATION_TIMEOUT_SECONDS` | `120` (default varies) | Host-side timeout threshold | Lower for faster detection, higher for long ops |
| `TT_METAL_DISPATCH_PROGRESS_UPDATE_MS` | `5000` | Periodic dispatch heartbeats | When debugging dispatch stalls |
| `TT_METAL_NOC_DEBUG_DUMP` | `1` | Track NOC debug state for missing barriers | When suspecting unflushed writes |
| `TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS` | `1` | Enable `ebreak`-based kernel asserts | Low-overhead, suitable for CI |

---

## War Story: The Silent Hang That Was Not a Device Hang

**Symptom:** A model training run hangs after 2 hours. The operator sees 0% device utilization and the process appears stuck.

**Initial Triage:**
- Step 1: `strace -p <pid>` reveals the process is spinning in userspace, consuming 100% CPU on one thread. This is a **host-side hang**, not a device hang.
- Step 2: Watcher log shows all cores at `D` (done).
- Step 3: tt-triage `dump_callstacks --all-cores` confirms all cores have go message `DONE`.
- Resolution: The host-side hang was in the Python data loader, blocked on a deadlocked multiprocessing queue. No chip reset was needed. If the operator had immediately reset the chip, they would have wasted time investigating a non-existent device issue.

**Lesson:** Always classify the hang type before touching the device. Host-side hangs are surprisingly common and do not require any device-side debugging.

---

## War Story: Watcher Catches a NOC Address Overflow in Production

**Symptom:** A batch inference job on T3000 hangs after 1000 iterations. The dispatch timeout fires and the process exits.

**Initial Triage:**
- Step 1: Dispatch timeout message in stderr. `TT_METAL_INSPECTOR_SERIALIZE_ON_DISPATCH_TIMEOUT=1` was set, so Inspector data was saved.
- Step 2: Watcher log (watcher was enabled at 5-second intervals) shows:
  ```
  Watcher detected NOC error and stopped device:
  Device 0 core(x=3,y=4) NCRISC: noc0 unicast write to addr 0x00001200001FFFFF,
  local addr 0x0, len 2048 (NOC target address overflow).
  ```
- Step 3: tt-triage shows core (3,4) stuck with CB0 active, confirming the NOC write was in-flight when sanitize caught it.
- Step 4: The kernel ID in the watcher log maps to a custom data movement kernel. Inspection of the kernel source reveals an off-by-one in the DRAM address calculation that only manifests after 1000 iterations due to a buffer pointer wraparound.

**Lesson:** Watcher NOC sanitize catches the exact transaction, core, and RISC-V that triggered the bad address. Without watcher, this would have been an opaque timeout requiring binary search to narrow down.

---

## Summary: The Initial Triage Checklist

| Step | Action | Time | Multi-Chip Extension |
|------|--------|------|---------------------|
| 0 | **Do NOT reset.** Preserve device state. | 0s | Do not reset ANY chip in the mesh; coordinate across hosts |
| 1 | Determine: device hang, host hang, or fabric hang? Check `tt-smi`. | 10s | Check all hosts; classify as local vs. fabric hang |
| 2 | If watcher enabled: `tail -100 generated/watcher/watcher.log` | 10s | Collect watcher.log from all hosts; align timestamps |
| 3 | Run `./tools/tt-triage.py --verbosity=4 --dev=0` | 30s | Run on all hosts, all devices |
| 4 | If watcher disabled: run `watcher_dump`. On multi-chip: inspect ERISC state. | 15s | Run fabric_erisc_dumper, trace cross-device links |
| 5 | Check DPrint output (if enabled) | 5s | -- |
| 6 | Check Tracy data / `dump_running_operations` / Inspector logs | 10s | Collect from all hosts; note cross-host synchronization state |
| Route | Follow decision tree to [Section 02](./02_diagnosing_by_hang_category.md) | -- | Use multi-chip fast-path for CCL hangs |

### Printable Triage Checklist

```
INITIAL TRIAGE CHECKLIST
========================
[ ] Step 0: Do NOT reset. Preserve device state.
[ ] Step 1: Classify hang type (device / host / fabric)
[ ] Step 1: Run tt-smi to verify PCIe accessibility
[ ] Step 2: Check watcher log (if enabled)
[ ] Step 3: Run tt-triage (./tools/tt-triage.py --verbosity=4 --dev=0)
[ ] Step 4: Run watcher_dump (if watcher was not enabled)
[ ] Step 4: (Multi-chip) Run fabric_erisc_dumper
[ ] Step 5: Check DPrint output (if enabled)
[ ] Step 6: Check Tracy data / Inspector logs
[ ] Route to category-specific diagnosis via decision tree
```

---

**Next:** [02_diagnosing_by_hang_category.md](./02_diagnosing_by_hang_category.md)
