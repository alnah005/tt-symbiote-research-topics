# 6.5 Profiler, Tracy, NOC Debug Dump, and Lightweight Asserts

## Summary

This section covers performance profiling and auxiliary diagnostic tools that complement the core Watcher and DPRINT systems. These tools focus on temporal analysis (when things happen and how long they take), structural validation (detecting missing barriers), system-level monitoring (device health), and lightweight assertions. The tools covered are: Tracy profiler for timeline visualization, NOC debug dump for missing barrier detection, fabric telemetry for multi-chip monitoring, tt-smi for device health checking, lightweight kernel asserts for minimal-overhead error detection, and LLK asserts for compute engine validation.

## Prerequisites

- Tracy profiler client (for visualization)
- Understanding of NOC transaction ordering and barriers (Ch2)
- Multi-chip topology awareness for fabric debug (Ch5)
- Section 6.1 (Watcher, for comparison with lightweight asserts)

## 6.5.1 Tool Selection Decision Tree

```
What kind of information do you need?
|
+-- Performance timeline (when kernels run, how long each takes)
|   --> Tracy Profiler (Section 6.5.2)
|
+-- Missing NOC barrier detection (transactions out of order)
|   --> NOC Debug Dump (Section 6.5.3)
|
+-- Multi-chip fabric health (bandwidth, link status)
|   --> Fabric Telemetry (Section 6.5.4)
|
+-- Quick device health check (power, temp, PCIe status)
|   --> tt-smi (Section 6.5.5)
|
+-- Minimal-overhead assertion in production code
|   --> Lightweight Kernel Asserts (Section 6.5.6)
|
+-- Detailed RISC debug signals and NOC transfer counters
    --> tt-triage scripts (Section 6.4)
```

---

## 6.5.2 Tracy Device Profiler

### Architecture

**Device-side:** The kernel profiler (`tt_metal/tools/profiler/kernel_profiler.hpp`) records timestamped events into per-core profiler buffers, periodically flushed to DRAM.

**Host-side:** The profiler infrastructure reads device buffers, processes them (optionally via C++ post-processing), and pushes events to Tracy for visualization.

### Environment Variable Reference

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `TT_METAL_DEVICE_PROFILER` | flag | disabled | Enable device-level profiling |
| `TT_METAL_DEVICE_PROFILER_DISPATCH` | flag | disabled | Enable profiling on dispatch cores |
| `TT_METAL_PROFILER_SYNC` | flag | disabled | Synchronous profiling (slower, more accurate) |
| `TT_METAL_DEVICE_PROFILER_NOC_EVENTS` | flag | disabled | NOC event profiling |
| `TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH` | path | default | NOC events report path |
| `TT_METAL_PROFILE_PERF_COUNTERS` | flag | disabled | Hardware performance counters |
| `TT_METAL_MEM_PROFILER` | flag | disabled | Memory/buffer profiling |
| `TT_METAL_TRACE_PROFILER` | flag | disabled | Trace-mode profiling |
| `TT_METAL_PROFILER_TRACE_TRACKING` | flag | disabled | Track trace execution |
| `TT_METAL_PROFILER_MID_RUN_DUMP` | flag | disabled | Force profiler buffer dumps during execution |
| `TT_METAL_TRACY_MID_RUN_PUSH` | flag | disabled | Push data to Tracy GUI during execution (real-time) |
| `TT_METAL_PROFILER_CPP_POST_PROCESS` | flag | disabled | Use C++ post-processing (faster than Python) |
| `TT_METAL_PROFILER_SUM` | flag | disabled | Aggregated sum profiling mode |
| `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT` | int | default | Max programs profiler can track |
| `TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES` | flag | `false` | Disable file output |
| `TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY` | flag | `false` | Disable Tracy push |

### Profiler Modes Reference

| Mode | Env Vars Required | Use Case |
|------|-------------------|----------|
| Basic device profiling | `DEVICE_PROFILER=1` | Kernel execution timing |
| Dispatch profiling | `DEVICE_PROFILER=1` + `DEVICE_PROFILER_DISPATCH=1` | Dispatch overhead analysis |
| NOC event profiling | `DEVICE_PROFILER=1` + `DEVICE_PROFILER_NOC_EVENTS=1` | NOC transaction timing |
| Memory profiling | `MEM_PROFILER=1` | L1/DRAM allocation tracking |
| Performance counters | `PROFILE_PERF_COUNTERS=1` | HW counter values (cycles, stalls) |

### Recipe: Basic Tracy Profiling

```bash
# Step 1: Start Tracy capture
tracy-capture -o my_capture.tracy &

# Step 2: Run with profiler enabled
export TT_METAL_DEVICE_PROFILER=1
./my_program

# Step 3: Open capture in Tracy GUI
tracy my_capture.tracy
```

### Recipe: Full Profiling with Dispatch

```bash
export TT_METAL_DEVICE_PROFILER=1
export TT_METAL_DEVICE_PROFILER_DISPATCH=1
export TT_METAL_TRACY_MID_RUN_PUSH=1
./my_program
# Open Tracy client and connect for real-time view
```

### Hang-Relevant Usage

While Tracy is primarily a performance tool, it provides critical hang debugging information:
- **Last kernel to execute** before a hang
- **Kernel duration trends** showing progressive slowdowns leading to hangs
- **Dispatch gaps** indicating dispatch-level stalls
- **Host-device correlation** for timing analysis

---

## 6.5.3 NOC Debug Dump

### Overview

The NOC Debug Dump is an **experimental** feature that uses the profiler infrastructure to record NOC debug packets continuously. Its primary purpose is detecting missing barriers -- situations where a kernel issues NOC transactions that depend on each other without proper ordering guarantees.

**Source:** Enabled via `TT_METAL_NOC_DEBUG_DUMP=1`, implemented through `tt_metal/tools/profiler/noc_debugging_profiler.hpp`.

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `TT_METAL_NOC_DEBUG_DUMP` | disabled | Enable experimental NOC debug dump mode |

### Architecture

When enabled, the NOC debugging profiler records scoped lock events that track address ranges, event types, and timestamps:

```cpp
// From tt_metal/tools/profiler/noc_debugging_profiler.hpp
template <NocDebuggingEventMetadata::NocDebugEventType event_type>
FORCE_INLINE void recordScopedLockEvent(uint32_t locked_address_base, uint32_t num_bytes) {
    NocDebuggingEventMetadata ev_md;
    ev_md.setEventType(event_type);
    ev_md.setLockedRegion(locked_address_base, num_bytes);
    kernel_profiler::timeStampedData<...>(ev_md.asU64());
}
```

### Limitations

- Only works on NCRISC and BRISC (`#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)`)
- Requires profiler infrastructure to be enabled
- Generates large amounts of data; not suitable for long-running workloads
- Experimental: output format and analysis tooling may change
- Adding instrumentation overhead may alter timing of race conditions (observer effect)

### Cross-Reference to Ch2

Missing NOC barriers are a common root cause for intermittent hangs. The NOC debug dump helps identify write-after-write ordering violations, read-before-write-complete races, and transactions that overlap without synchronization.

---

## 6.5.4 Fabric Debug and Telemetry

### Environment Variables

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `TT_METAL_FABRIC_TELEMETRY` | disabled | Enable fabric telemetry. `1` for basic, or detailed spec string |
| `TT_METAL_FABRIC_BW_TELEMETRY` | disabled | Enable fabric bandwidth telemetry |

### Programmatic Configuration

```cpp
struct FabricTelemetrySettings {
    bool enabled = false;
    FabricTelemetrySelection<uint32_t> chips;
    FabricTelemetrySelection<uint32_t> channels;
    FabricTelemetrySelection<uint32_t> eriscs;
    uint8_t stats_mask;  // ROUTER_STATE | BANDWIDTH | HEARTBEAT_TX | HEARTBEAT_RX
};
```

### Fabric Telemetry Features

| Feature | Description | Hang Relevance |
|---------|-------------|----------------|
| Link utilization | Per-link throughput | Low utilization during expected heavy traffic may indicate blocked sender |
| Error counters | CRC errors, retransmissions | Elevated errors indicate physical link issues (Ch5) |
| Congestion metrics | Back-pressure events | Sustained congestion can cause timeouts and hangs |
| Router state | Current routing configuration | Incorrect routing causes packet drops |

### Fabric Debug Scripts

| Tool | Description |
|------|-------------|
| `fabric_erisc_dumper.py` | Dumps ERISC state for fabric debugging |
| `fabric_binary_analyzer.py` | Analyzes fabric binary images |

**Cross-reference:** Chapter 5 covers multi-chip CCL hangs. Fabric telemetry helps identify specific links and routers involved.

---

## 6.5.5 tt-smi (System Management Interface)

`tt-smi` is the system management tool for Tenstorrent devices. It provides hardware-level health information without requiring any tt-metal runtime state. It is the **first tool to reach for** when a device is completely unresponsive.

```
Is the device responsive at all?
|
+-- Unknown --> tt-smi -l  (list devices)
|   |
|   +-- Devices listed --> PCIe-accessible, proceed with higher-level tools
|   +-- No devices / timeout --> Hardware problem: check PCIe, power, cables
|
+-- Device listed but tt-triage fails?
    --> Check temperature (thermal throttling can cause hangs)
    --> Check power (undervoltage can cause corruption)
```

### Key Commands

| Command | Description | Hang Relevance |
|---------|-------------|----------------|
| `tt-smi` / `tt-smi -l` | List devices | Verify devices are visible |
| `tt-smi -a` | Detailed device info | Check for error flags |
| `tt-smi -m` | Monitor continuously | Watch temperature/power during long runs |
| `tt-smi -r <device>` | Reset device | Recovery after a hang |

---

## 6.5.6 Lightweight Kernel Asserts

### Overview

Lightweight asserts provide a minimal-overhead assertion mechanism that works **without the Watcher system**. When `TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1` is set, the `ASSERT()` macro compiles to an `ebreak` instruction.

**Source:** `tt_metal/hw/inc/api/debug/assert.h`

```cpp
#if defined(LIGHTWEIGHT_KERNEL_ASSERTS)

#define ASSERT(condition, ...)      \
    do {                            \
        if (!(condition))           \
            asm volatile("ebreak"); \
    } while (0)

#define ASSERT_ENABLED 1
#define LIGHTWEIGHT_ASSERT_ENABLED 1
#define WATCHER_ASSERT_ENABLED 0

#endif
```

### Three Assert Modes Comparison

| Feature | Watcher Assert | Lightweight Assert | Disabled |
|---------|---------------|-------------------|----------|
| Enable via | `TT_METAL_WATCHER=N` | `TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1` | Default |
| Mechanism | Mailbox write + hang | `ebreak` instruction | Compiled out |
| Overhead | Moderate (mailbox writes) | Minimal (single instruction) | Zero |
| Host notification | Automatic (Watcher poll) | Requires debugger or tt-triage | None |
| Line number info | Yes (in mailbox) | Via debugger PC resolution | N/A |
| Binary size impact | Significant | Minimal | None |
| Production-safe | No (too much overhead) | Yes | -- |

### tt-triage Integration

The `dump_lightweight_asserts.py` triage script detects lightweight assert trips:
1. Scans all cores for RISC-V processors in `ebreak` state
2. Resolves PC to source file and line containing the `ASSERT()`
3. Extracts the assert condition from source code
4. Presents callstack with arguments and local variables

```bash
./tools/tt-triage.py --run=dump_lightweight_asserts
```

---

## 6.5.7 LLK (Low-Level Kernel) Asserts

LLK asserts are embedded in the low-level kernel library (compute engine). They check conditions like valid data formats, tile dimensions, and math unit configuration.

| Condition | Behavior |
|-----------|----------|
| Watcher enabled | LLK assert trip detected by watcher, exception thrown |
| Watcher disabled, lightweight asserts enabled | Trip recorded; detected via tt-triage |
| Both disabled | Assert compiles to no-op; error causes silent corruption or hang |

| Common LLK Assert | Cause | Hang Relationship |
|-------------------|-------|-------------------|
| Invalid data format | Unsupported format to pack/unpack | Compute hang: TRISC stuck |
| Tile dimension mismatch | Wrong tile size | Data corruption, downstream NOC errors |
| Math config invalid | Misconfigured FPU | TRISC infinite loop or garbage output |

---

## 6.5.8 Hang Scenarios

### Scenario 6.5.1: Tracy Profiler Buffer Overflow Masks Hang

**Symptom**: Tracy timeline shows kernel execution up to a point, then no more events. Device is hung but profiler stopped recording.

**Root Cause**: Per-core profiler buffer filled before the hang. Unlike DPRINT, the profiler does not stall the kernel -- it simply stops recording.

**Diagnosis Steps**:
1. Enable `TT_METAL_PROFILER_MID_RUN_DUMP=1` to flush buffers during execution
2. Increase buffer capacity via `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT`
3. Use Tracy alongside Watcher for complementary coverage

**Fix**: Use mid-run dump mode. Combine Tracy (timing) with Watcher (final state) for full picture.

**Prevention**: Always run Watcher alongside Tracy for hang debugging.

### Scenario 6.5.2: NOC Debug Dump Overhead Alters Timing

**Symptom**: Enabling `TT_METAL_NOC_DEBUG_DUMP=1` changes behavior of a timing-sensitive hang (disappears or moves).

**Root Cause**: NOC debug dump adds profiler overhead to every NOC transaction, changing timing relationships (observer effect).

**Diagnosis Steps**:
1. Use in combination with debug delay (Section 6.6) to systematically explore timing variations
2. Use captured data to reason about barrier placement even if the specific hang did not reproduce
3. Consider code review of barrier placement as a complementary approach

**Fix**: This is inherent to any instrumentation. Cross-validate with code analysis.

### Scenario 6.5.3: Lightweight Assert Trip on Production Workload

**Symptom**: Production workload hangs with `TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1`. tt-triage shows a tripped assert.

**Root Cause**: A kernel invariant was violated. The lightweight assert caught it with minimal overhead, but the RISC is now in `ebreak` state.

**Diagnosis Steps**:
1. `./tools/tt-triage.py --run=dump_lightweight_asserts`
2. Identify assert location, source line, and variable values
3. Analyze callstack to understand the call chain leading to failure

**Fix**: Fix the root cause in the kernel code. Consider adding host-side validation.

**Prevention**: Enable watcher during development for real-time detection; use lightweight asserts as a production safety net.

### Scenario 6.5.4: tt-smi Shows Device in Error State

**Symptom**: `tt-smi` shows a device with error flags or abnormal temperature/power readings. Higher-level tools cannot connect.

**Root Cause**: Hardware error state due to thermal throttling, power issues, or ARC firmware crash.

**Diagnosis Steps**:
1. If thermal: check cooling, reduce workload, allow cooling
2. If power: verify power supply meets specifications
3. If ARC crash: chip reset required (`tt-smi -r <device>` or power cycle)
4. After reset, device state is lost -- focus on host-side logs

**Fix**: Address the hardware condition, reset the device.

**Prevention**: Monitor device health via `tt-smi -m` during long runs. Cross-reference with Ch5 for multi-chip thermal scenarios.

---

## 6.5.9 Tool Selection Summary Table

| Symptom | Primary Tool | Secondary Tool | Reference |
|---------|-------------|----------------|-----------|
| Need kernel timing data | Tracy profiler | NOC event profiler | -- |
| Intermittent data corruption | NOC debug dump | Debug delay (Section 6.6) | Ch2 |
| Multi-chip fabric issues | Fabric telemetry | ETH link status (Section 6.1) | Ch5 |
| Device not responding | tt-smi | ARC check (tt-triage) | -- |
| Need production-safe asserts | Lightweight asserts | Watcher full asserts | Section 6.1 |
| Performance degradation pre-hang | Tracy + dispatch profiling | Watcher waypoints | Ch4 |

---

**Cross-references:**
- Watcher assert mechanism: Section 6.1.4
- NOC barrier requirements: Chapter 2, Section 2.4
- Fabric and multi-chip hangs: Chapter 5
- Debug delay for timing exploration: Section 6.6
