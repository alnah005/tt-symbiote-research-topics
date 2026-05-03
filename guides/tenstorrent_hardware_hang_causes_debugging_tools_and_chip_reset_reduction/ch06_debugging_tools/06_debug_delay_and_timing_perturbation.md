# 6.6 Debug Delay, Timing Perturbation, and Dispatch Debug Tools

## Summary

This section covers tools that actively modify device timing to expose or reproduce race conditions and timing-dependent hangs, as well as tools for diagnosing dispatch-level issues through progress monitoring and timeout handling. Debug delay inserts configurable wait cycles before NOC operations (reads, writes, atomics), effectively slowing down specific cores and processors to widen race condition windows. Timing perturbation inserts NOP instructions into compute kernels to alter the relative timing of unpack/math/pack operations. Dispatch debug tools provide visibility into command queue state and progress heartbeat monitoring. Together, these tools transform intermittent "one-in-a-thousand" hangs into reproducible failures.

## Prerequisites

- Watcher must be enabled (`TT_METAL_WATCHER=1`) for debug delay
- NOC sanitization must be enabled (not disabled via `TT_METAL_WATCHER_DISABLE_SANITIZE_NOC`) for debug delay
- Debug builds for timing perturbation (`DEBUG_PRINT_ENABLED` compile flag)
- Chapter 2 (NOC transaction timing, barrier semantics)
- Chapter 4 (dispatch architecture, command queue flow)
- Section 6.1 (Watcher system), Section 6.4 (tt-triage, for timeout integration)

## 6.6.1 Tool Selection for Timing Issues

```
Is the hang intermittent / timing-dependent?
|
+-- YES: Do you know which cores are involved?
|   |
|   +-- YES: Is it a NOC transaction ordering issue?
|   |   |
|   |   +-- YES --> Debug Delay on specific NOC operation type (Section 6.6.2)
|   |   +-- Unsure --> Try Debug Delay on reads, then writes, then atomics
|   |
|   +-- YES: Is it a compute pipeline timing issue (unpack/math/pack)?
|   |   --> Timing Perturbation with compute NOPs (Section 6.6.4)
|   |
|   +-- NO: Hang is in dispatch / command queue?
|       --> Dispatch Progress Heartbeat + Timeout Command (Section 6.6.6)
|
+-- NO: Hang is reproducible but cause is unclear?
    |
    +-- Dispatch-level? --> Dispatch Debug Tools (Section 6.6.5)
    +-- Kernel-level?   --> Debug Delay to narrow timing window
```

---

## 6.6.2 Debug Delay Architecture and Configuration

### How It Works

Debug delay is integrated into the NOC sanitization path. When enabled, the sanitizer inserts a delay loop before allowing the transaction to proceed. This means **Watcher must be enabled** and **NOC sanitization must not be disabled**.

### Validation Assertions (from `rtoptions.cpp`)

```cpp
TT_ASSERT(watcher_settings.enabled,
    "TT_METAL_WATCHER_DEBUG_DELAY requires TT_METAL_WATCHER");
TT_ASSERT(!watcher_disabled_features.contains(watcher_noc_sanitize_str),
    "TT_METAL_WATCHER_DEBUG_DELAY requires TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=0");
```

### Device-Side Implementation

During `WatcherServer::init_device()`, the delay configuration is written to each core's mailbox as a `debug_insert_delays_msg_t` structure:

| Field | Type | Description |
|-------|------|-------------|
| `read_delay_processor_mask` | `uint32_t` | Bitmask of processors to delay on reads |
| `write_delay_processor_mask` | `uint32_t` | Bitmask of processors to delay on writes |
| `atomic_delay_processor_mask` | `uint32_t` | Bitmask of processors to delay on atomics |

The firmware checks these masks before each NOC operation and inserts the configured number of delay cycles if the current processor matches.

### Runtime Features Enum (from `rtoptions.hpp`)

```cpp
enum RunTimeDebugFeatures {
    RunTimeDebugFeatureDprint,
    RunTimeDebugFeatureReadDebugDelay,
    RunTimeDebugFeatureWriteDebugDelay,
    RunTimeDebugFeatureAtomicDebugDelay,
    RunTimeDebugFeatureEnableL1DataCache,
    RunTimeDebugFeatureCount
};
```

---

## 6.6.3 Debug Delay Environment Variable Reference

### Master Configuration

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `TT_METAL_WATCHER_DEBUG_DELAY` | `uint32_t` | `0` (disabled) | Number of delay cycles before each targeted NOC operation. Requires `TT_METAL_WATCHER` + NOC sanitize enabled |

### Per-Operation-Type Targeting

| Operation Type | Core Env Var | RISC Env Var |
|---------------|-------------|-------------|
| NOC Read | `TT_METAL_READ_DEBUG_DELAY_CORES` | `TT_METAL_READ_DEBUG_DELAY_RISCVS` |
| NOC Write | `TT_METAL_WRITE_DEBUG_DELAY_CORES` | `TT_METAL_WRITE_DEBUG_DELAY_RISCVS` |
| NOC Atomic | `TT_METAL_ATOMIC_DEBUG_DELAY_CORES` | `TT_METAL_ATOMIC_DEBUG_DELAY_RISCVS` |

### Core Targeting Syntax

| Format | Example | Description |
|--------|---------|-------------|
| Single core | `0,0` | One core |
| List | `(0,0),(1,1),(2,2)` | Multiple specific cores |
| Range | `(0,0)-(3,3)` | Rectangular region |
| All | `all` | All cores |

### RISC Targeting Syntax

Plus-separated: `BR` (BRISC), `NC` (NCRISC), `TR0` (TRISC0), `TR1` (TRISC1), `TR2` (TRISC2), `ER` (ERISC). If not set, delay applies to all RISCs.

### Additional Targeting (ETH cores, Chips)

| Env Var Suffix | Format | Example |
|---------------|--------|---------|
| `_ETH_CORES` | `all` or `(x,y),...` | `TT_METAL_READ_DEBUG_DELAY_ETH_CORES=(0,0)` |
| `_CHIPS` | comma ints or `all` | `TT_METAL_READ_DEBUG_DELAY_CHIPS=0,1` |

---

## 6.6.4 Timing Perturbation for Compute Kernels

### Source and Compile Gate

**Header:** `tt_metal/hw/inc/api/debug/timing_perturbation.h`

```cpp
#if defined(COMPILE_FOR_TRISC) && defined(DEBUG_PRINT_ENABLED) && !defined(FORCE_DPRINT_OFF)
```

Only available on TRISC processors (compute kernels) in debug builds.

### API Reference

```cpp
namespace tt::compute::common {

// Insert NOPs on each TRISC stage independently
template <
    const int num_unpack_nops,   // NOPs for TRISC0 (unpack)
    const int num_math_nops,     // NOPs for TRISC1 (math)
    const int num_pack_nops,     // NOPs for TRISC2 (pack)
    const int is_riscv_nop,      // 0 = Tensix NOP, 1 = RISC-V NOP
    const int use_loop = 0>      // 0 = inline assembly, 1 = loop-based
inline void add_compute_nops();

// Lower-level: add NOPs on the current TRISC
template <const int num_nops, const int is_riscv_nop, const int use_loop = 0>
inline void add_nops();

}
```

### NOP Type Comparison

| NOP Type | `is_riscv_nop` | Instruction | Use When |
|----------|----------------|-------------|----------|
| Tensix NOP | 0 | `TTI_NOP` | Testing Tensix pipeline timing sensitivity |
| RISC-V NOP | 1 | `nop` (RV32I) | Testing RISC-V control flow timing |

### Implementation Modes

| `use_loop` | Implementation | Trade-off |
|-----------|---------------|-----------|
| 0 (default) | Inline `.rept` assembly | Larger binary, exact cycle count |
| 1 | C++ `for` loop | Smaller binary, loop overhead adds variability |

### Usage Example

```cpp
#include "debug/timing_perturbation.h"

void MAIN {
    for (uint32_t tile = 0; tile < num_tiles; tile++) {
        // 100 Tensix NOPs in unpack, 200 in math, 50 in pack
        tt::compute::common::add_compute_nops<100, 200, 50, 0>();

        acquire_dst(tt::DstMode::Half);
        // ... compute ...
        release_dst(tt::DstMode::Half);
    }
}
```

---

## 6.6.5 Dispatch Debug Tools

### Source Files

| File | Path | Role |
|------|------|------|
| Debug tools | `tt_metal/impl/dispatch/debug_tools.hpp/.cpp` | CQ dump, dispatch state inspection |

### CQ Data Functions

```cpp
namespace internal {
    void wait_for_program_vector_to_arrive_and_compare_to_host_program_vector(
        const char* DISPATCH_MAP_DUMP, IDevice* device);
    void match_device_program_data_with_host_program_data(
        const char* host_file, const char* device_file);
    void dump_cqs(
        std::ofstream& cq_file, std::ofstream& iq_file,
        SystemMemoryManager& sysmem_manager, bool dump_raw_data = false);
}
```

### Dispatch Debug Environment Variables

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `TT_METAL_DISPATCH_DATA_COLLECTION` | flag | disabled | Enable dispatch debug data collection |

---

## 6.6.6 Dispatch Progress Heartbeat and Timeout Pipeline

### Progress Heartbeat

The dispatch kernel periodically writes a progress counter. The host monitors this counter to detect stalled dispatch kernels.

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `TT_METAL_DISPATCH_PROGRESS_UPDATE_MS` | `uint32_t` | `100` | Update period in milliseconds |
| `TT_METAL_OPERATION_TIMEOUT_SECONDS` | float | `0.0` (disabled) | Timeout for device operations |
| `TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE` | string | none | Shell command to run on dispatch timeout |

### Automated Triage Pipeline

This is one of the most powerful automated debugging features. Configuration:

```bash
export TT_METAL_OPERATION_TIMEOUT_SECONDS=30.0
export TT_METAL_DISPATCH_PROGRESS_UPDATE_MS=100
export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE=./tools/tt-triage.py
export TT_METAL_INSPECTOR=1
export TT_METAL_INSPECTOR_SERIALIZE_ON_DISPATCH_TIMEOUT=1
```

**Flow:**
1. Host sends program to device
2. Dispatch kernel processes commands, updating progress counter
3. If dispatch stalls for >30s:
   a. Host detects timeout via progress heartbeat monitoring
   b. Inspector serializes operation/dispatch state to disk
   c. tt-triage runs automatically: `dump_callstacks`, `check_noc_status`, `dump_running_operations`
   d. Diagnostic data saved for post-mortem analysis
4. Program can retry, reset device, or exit with diagnostics

---

## 6.6.7 Debug Delay Usage Recipes

### Recipe: Delay All NOC Reads on All Cores

```bash
export TT_METAL_WATCHER=1
export TT_METAL_WATCHER_DEBUG_DELAY=1000
export TT_METAL_READ_DEBUG_DELAY_CORES=all
./my_program
```

### Recipe: Delay Only BRISC Writes on Specific Cores

```bash
export TT_METAL_WATCHER=1
export TT_METAL_WATCHER_DEBUG_DELAY=500
export TT_METAL_WRITE_DEBUG_DELAY_CORES="(0,0),(1,1),(2,2)"
export TT_METAL_WRITE_DEBUG_DELAY_RISCVS=BR
./my_program
```

### Recipe: Delay Reads and Writes Independently

```bash
export TT_METAL_WATCHER=1
export TT_METAL_WATCHER_DEBUG_DELAY=300
export TT_METAL_READ_DEBUG_DELAY_CORES=all
export TT_METAL_READ_DEBUG_DELAY_RISCVS=BR+NC
export TT_METAL_WRITE_DEBUG_DELAY_CORES="(0,0)-(3,3)"
export TT_METAL_WRITE_DEBUG_DELAY_RISCVS=NC
./my_program
```

### Systematic Race Condition Hunting

```
Step 1: Identify suspect transaction type
  - Semaphore wait hang --> try ATOMIC delay
  - Data arrival hang --> try READ delay
  - Data sending hang --> try WRITE delay

Step 2: Start small, increase
  - TT_METAL_WATCHER_DEBUG_DELAY=1   (minimal perturbation)
  - TT_METAL_WATCHER_DEBUG_DELAY=10  (moderate)
  - TT_METAL_WATCHER_DEBUG_DELAY=100 (aggressive)

Step 3: Narrow the core range
  - Start with all cores
  - Binary search to find which core's delay triggers the hang
  - Then narrow to specific RISCs

Step 4: Combine with Watcher log
  - When hang reproduces, waypoints show exactly where each core stopped
```

---

## 6.6.8 Hang Scenarios

### Scenario 6.6.1: Debug Delay Exposes Race Condition Between BRISC and NCRISC

**Symptom**: Intermittent hang (~1% of runs) where NCRISC's NOC read returns stale data, causing downstream compute to loop forever.

**Root Cause**: BRISC writes data via NOC, NCRISC reads the same location. Missing write barrier means NCRISC sometimes reads before BRISC's write completes.

**Diagnosis Steps**:
1. Add debug delay to BRISC writes:
   ```bash
   export TT_METAL_WATCHER=1
   export TT_METAL_WATCHER_DEBUG_DELAY=5000
   export TT_METAL_WRITE_DEBUG_DELAY_CORES=all
   export TT_METAL_WRITE_DEBUG_DELAY_RISCVS=BR
   ```
2. Run multiple times -- hang should now reproduce reliably (>50% of runs)
3. Use DPRINT on NCRISC to confirm stale data reads
4. Add `noc_async_write_barrier()` after BRISC's write
5. Verify fix by running with delay both enabled and disabled

**Fix**: Insert the missing `noc_async_write_barrier()` (see Ch2 barrier scenarios).

**Prevention**: Use watcher NOC sanitization in CI. Code review for write-then-read patterns across RISCs.

### Scenario 6.6.2: Timing Perturbation Reveals Pack-Before-Math Race

**Symptom**: Adding NOPs to math pipeline causes incorrect results or a hang; original code appears correct.

**Root Cause**: Pack phase reads destination register before math finishes writing. Timing was coincidentally correct without perturbation.

**Diagnosis Steps**:
1. Add pack NOPs: `add_compute_nops<0, 0, 500, 0>()` -- if issue disappears, pack was racing
2. Add math NOPs: `add_compute_nops<0, 500, 0, 0>()` -- if issue now reproduces, math was bottleneck
3. Check for missing `acquire_dst()` / `release_dst()` synchronization

**Fix**: Add proper synchronization between math and pack stages.

**Prevention**: Always use `tile_regs_acquire()` / `tile_regs_release()` between compute stages.

### Scenario 6.6.3: Dispatch Timeout Triggers Automated Triage in CI

**Symptom**: CI test hangs. After 30 seconds, `TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE` fires tt-triage, producing a report showing a specific core stuck in a semaphore wait.

**Root Cause**: A kernel is waiting for a semaphore that will never be incremented because the producer core hit an error and stopped.

**Diagnosis Steps**:
1. Read tt-triage output from CI artifacts
2. `dump_callstacks` shows the waiting core; `check_noc_status` shows the failed producer
3. Investigate why the producer stopped (assert trip, bad address, etc.)

**Fix**: Fix root cause on producer core. Consider adding error propagation so consumer does not wait indefinitely.

**Prevention**: Use the full automated triage pipeline in all CI environments.

### Scenario 6.6.4: Debug Delay Combined with Watcher Provides Full Picture

**Symptom**: Multi-core race condition. Debug delay makes it reproducible; Watcher provides state at hang point.

**Root Cause**: NOC writes from core A and reads from core B with missing barrier on core A.

**Diagnosis Steps**:
1. `TT_METAL_WATCHER=1` provides waypoints and NOC sanitization
2. `TT_METAL_WATCHER_DEBUG_DELAY=20` on core A's writes reproduces the hang
3. Watcher log shows core A at `NWW` and core B at `NRD` with stale data
4. The delay proves core A's write was not completing before core B's read

**Fix**: Add `noc_async_write_barrier()` on core A. Verify: with barrier, delay no longer causes hang.

**Prevention**: Code review for inter-core signaling patterns. Use NOC debug dump (Section 6.5.3) for systematic barrier validation.

### Scenario 6.6.5: Dispatch Progress Heartbeat Detects Stuck Prefetch

**Symptom**: `TT_METAL_DISPATCH_PROGRESS_UPDATE_MS=100` triggers timeout, indicating dispatch prefetch kernel has stalled.

**Root Cause**: Prefetch kernel waiting for PCIe data that the host has not written, due to host-side buffering issue.

**Diagnosis Steps**:
1. Check tt-triage output for prefetch kernel callstack
2. Verify host-side CQ write operations completed
3. Check PCIe link status via tt-smi
4. Cross-reference with Chapter 4, Section 4.1 for prefetch hang scenarios

**Fix**: Fix host-side CQ buffering. Check PCIe link health.

---

## 6.6.9 Combined Tool Usage Matrix

| Investigation Goal | Debug Delay | Timing Perturbation | Dispatch Debug | Watcher | DPRINT | Tracy |
|-------------------|:-----------:|:-------------------:|:--------------:|:-------:|:------:|:-----:|
| Reproduce intermittent NOC race | X | | | X | | |
| Reproduce compute stage race | | X | | X | X | |
| Diagnose dispatch timeout | | | X | X | | |
| Measure dispatch latency | | | X | | | X |
| Detect missing NOC barrier | X | | | X | | |
| Profile kernel hot spots | | | | | | X |
| Validate barrier placement | X | | | X | | |
| Identify pack/unpack bottleneck | | X | | | X | X |
| Auto-triage on hang | | | X | | | |

## 6.6.10 Performance Impact Reference

| Tool | Overhead | Notes |
|------|----------|-------|
| Debug delay | Configurable: `delay_cycles` per NOC op | Can make kernels 10x-100x slower |
| Timing perturbation | Configurable: `num_nops` per iteration | 1000 Tensix NOPs ~= 1us |
| Dispatch progress heartbeat | < 1% | Single counter write per dispatch command |
| Dispatch data collection | 1-5% | Depends on dispatch frequency |
| Auto-trigger command | N/A | Runs only on timeout |

---

**Cross-references:**
- Watcher as prerequisite for debug delay: Section 6.1
- NOC barrier requirements: Chapter 2, Section 2.4
- Compute pipeline synchronization: Chapter 2, Section 2.1
- Dispatch hang scenarios: Chapter 4
- Automated triage integration: Section 6.4
