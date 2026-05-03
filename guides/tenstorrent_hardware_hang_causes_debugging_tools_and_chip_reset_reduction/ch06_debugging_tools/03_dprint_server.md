# 6.3 DPRINT / DEVICE_PRINT Server

## Summary

DPRINT (and the newer DEVICE_PRINT) provides printf-style debugging for device-side kernels and firmware running on Tenstorrent RISC-V processors. On the device side, the `DPRINT` macro writes formatted data into per-RISC L1 buffers. On the host side, the `DPrintServer` spawns a background thread that periodically reads these buffers and outputs the data to files or stdout. The system supports targeting specific cores, RISCs, and devices, and includes specialized features for printing tile data via TileSlice. A critical caveat: if the DPRINT buffer fills up and the host server is not running to drain it, the device kernel will spin-wait forever, causing a hang.

## Prerequisites

- A tt-metal debug build (DPRINT macros compile to no-ops in release builds unless explicitly enabled)
- Understanding of RISC-V processor types per core (BRISC, NCRISC, TRISC0/1/2 for Tensix; ERISC for ethernet)
- Section 6.1 (Watcher system, for comparison)
- Familiarity with circular buffer concepts (for CB data printing features)

## 6.3.1 When to Use DPRINT vs. Alternatives

```
Need to see runtime VALUES from device code?
|
+-- YES: How many cores/RISCs need printing?
|   |
|   +-- Few specific cores --> DPRINT is ideal
|   |   (set TT_METAL_DPRINT_CORES=x,y)
|   |
|   +-- Many/all cores --> Beware buffer saturation
|       |
|       +-- Can you reduce print volume? --> DPRINT with careful filtering
|       +-- No --> Consider Watcher ring buffer (uint32_t values, Section 6.1)
|       +-- Need full timeline? --> Tracy profiler (Section 6.5)
|
+-- NO: Need to see which CODE PATH executes?
    |
    +-- Coarse-grained (which function/block) --> Watcher waypoints (Section 6.1)
    +-- Fine-grained (specific conditions) --> DPRINT with conditional prints
```

---

## 6.3.2 Environment Variable Reference Table

### Core DPRINT Configuration

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `TT_METAL_DPRINT_CORES` | core list | none (disabled) | Worker cores to enable DPRINT on. `all` for all worker cores, or `(x,y),(x,y),...` for specific logical coordinates, or `(x1,y1)-(x2,y2)` for range |
| `TT_METAL_DPRINT_ETH_CORES` | core list | none | Ethernet cores. Same syntax as `DPRINT_CORES` |
| `TT_METAL_DPRINT_DRAM_CORES` | core list | none | DRAM programmable cores (Blackhole DRISC). Same syntax |
| `TT_METAL_DPRINT_CHIPS` | chip list | all | Comma-separated chip IDs: `0,1,2`. Mutually exclusive with `DPRINT_NODES` and `DPRINT_MESH_COORDS` |
| `TT_METAL_DPRINT_NODES` | node list | none | Fabric node IDs: `(M0,D0),(M0,D1)`. Mutually exclusive with `DPRINT_CHIPS` |
| `TT_METAL_DPRINT_MESH_COORDS` | coord list | none | Global system mesh (row,col) coordinates: `(0,0),(1,3)`. Mutually exclusive with `DPRINT_CHIPS` |
| `TT_METAL_DPRINT_RISCVS` | RISC list | all | RISC-V processors. Plus-separated: `BR`, `NC`, `TR0`, `TR1`, `TR2`, `ER`. E.g., `BR+NC+TR0` |
| `TT_METAL_DPRINT_FILE` | path | `generated/dprint/dprint.log` | Output file path. E.g., `/tmp/debug_output.log` |
| `TT_METAL_DPRINT_ONE_FILE_PER_RISC` | flag | `false` | Generate separate output file per RISC-V processor |
| `TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC` | flag | `false` | Prepend device ID, core coordinates, and RISC name to each print line |

---

## 6.3.3 Device-Side API Reference

### Legacy DPRINT Macro

**Header:** `tt_metal/hw/inc/api/debug/dprint.h`

**Compile gate**: `defined(DEBUG_PRINT_ENABLED) && !defined(FORCE_DPRINT_OFF) && !defined(USE_DEVICE_PRINT)`

When the gate is false, `DPRINT` compiles to `if(0) DebugPrinter()` -- completely optimized out by the compiler.

```cpp
// Basic usage (stream-style):
DPRINT << "Hello from core " << (uint32_t)core_id << ENDL();

// Formatted output:
DPRINT << SETW(8) << SETPRECISION(4) << FIXED() << my_float << ENDL();
DPRINT << HEX() << addr << DEC() << ENDL();

// RISC-specific macros:
DPRINT_UNPACK(DPRINT << "Unpack: " << value << ENDL());  // Only on TRISC0
DPRINT_MATH(DPRINT << "Math: " << value << ENDL());      // Only on TRISC1
DPRINT_PACK(DPRINT << "Pack: " << value << ENDL());       // Only on TRISC2
DPRINT_DATA0(DPRINT << "NOC0: " << value << ENDL());      // Only when noc_index == 0
DPRINT_DATA1(DPRINT << "NOC1: " << value << ENDL());      // Only when noc_index == 1
```

### Supported Types

| Type ID | C++ Type | Description |
|---------|----------|-------------|
| `DPrintCSTR` | `const char*` | C string literal |
| `DPrintENDL` | -- | Line terminator (flushes) |
| `DPrintUINT8/16/32/64` | `uint8-64_t` | Unsigned integers |
| `DPrintINT8/16/32/64` | `int8-64_t` | Signed integers |
| `DPrintFLOAT32` | `float` | 32-bit float |
| `DPrintBFLOAT16` | `uint16_t` | BFloat16 value |
| `DPrintCHAR` | `char` | Single character |
| `DPrintTILESLICE` | struct | Print tile data slice |
| `DPrintU32_ARRAY` | `uint32_t*` | Print uint32_t array |
| `SETW`, `SETPRECISION`, `FIXED`, `HEX`, `OCT`, `DEC` | -- | Formatting directives |

### New DEVICE_PRINT Macro

```cpp
// Format-string based (similar to fmt/printf):
DEVICE_PRINT("Value: {}, Count: {}\n", value, count);
```

Uses a single shared per-core buffer and leverages ELF-embedded format strings for more efficient space usage.

---

## 6.3.4 Buffer Layout and Protocol

### Per-RISC Buffer (Legacy DPRINT)

**Defined in:** `tt_metal/hostdevcommon/api/hostdevcommon/dprint_common.h`

```cpp
constexpr static std::uint32_t DPRINT_BUFFER_SIZE = 204;  // per thread

struct DebugPrintMemLayout {
    struct Aux {
        uint32_t wpos;     // Write position (device writes, host reads)
        uint32_t rpos;     // Read position (host writes, device reads)
        uint16_t core_x;
        uint16_t core_y;
    } aux;
    uint8_t data[DPRINT_BUFFER_SIZE - sizeof(Aux)];  // 192 bytes of data
};
```

Each RISC-V processor on each core gets its own 204-byte buffer. With 192 bytes of usable data space, this fills quickly with verbose output.

### Magic Values

| Magic | Value | Meaning |
|-------|-------|---------|
| `DEBUG_PRINT_SERVER_STARTING_MAGIC` | `0x98989898` | Host has initialized buffer, waiting for device to clear |
| `DEBUG_PRINT_SERVER_DISABLED_MAGIC` | `0xf8f8f8f8` | DPRINT disabled for this RISC; device should not use buffer |

### Initialization Protocol

1. Host writes magic value (`0x98989898`) to `wpos` field of each enabled RISC's buffer
2. Host verifies the write landed via MMIO read-back (up to 100K retries)
3. Device firmware checks `wpos` for magic on first `DebugPrinter()` construction
4. If magic found, device resets `wpos` and `rpos` to 0 (one-time init)
5. After init, device writes typed data stream starting at `wpos`, wrapping around buffer
6. Host polls for `wpos != rpos`, reads and parses new data, advances `rpos`

---

## 6.3.5 Host Server Architecture

| Class | Role |
|-------|------|
| `DPrintServer` | Public API wrapper (PIMPL pattern) |
| `DPrintServer::Impl` | Abstract base: owns poll thread, output streams, device-to-core mapping |
| `DPrintImpl` | Legacy DPRINT: per-RISC buffers, uses type ID parsing |
| `DevicePrintImpl` | New DEVICE_PRINT: per-core shared buffer, ELF-based format string resolution |

```cpp
class DPrintServer {
public:
    DPrintServer(llrt::RunTimeOptions& rtoptions);
    void attach_devices();   // Start polling thread, initialize buffers
    void detach_devices();   // Stop polling, flush output
    void set_mute(bool);     // Suppress output without stopping polling
    void await();            // Wait for current print data to be processed
    void clear_log_file();   // Clear log file mid-run
    bool hang_detected();    // Check if WAIT-induced hang was detected
};
```

### Output Paths

| Configuration | Output Destination |
|--------------|-------------------|
| Default (no `DPRINT_FILE`) | `generated/dprint/dprint.log` |
| `TT_METAL_DPRINT_FILE=/path/file` | Specified file path |
| `TT_METAL_DPRINT_ONE_FILE_PER_RISC=1` | `generated/dprint/device_<id>_core_<x>_<y>_risc_<name>.log` |

---

## 6.3.6 TileSlice Debugging

DPRINT supports inspecting tile contents directly on the device:

```cpp
#include "debug/dprint.h"
DPRINT << TileSlice(cb_id, tile_index, SliceRange::hw0_32_16()) << ENDL();
```

### SliceRange Presets

| Preset | Range | Description |
|--------|-------|-------------|
| `hw0_32_16()` | [0:32:16, 0:32:16] | 4 corner values |
| `hw0_32_8()` | [0:32:8, 0:32:8] | 16 evenly spaced elements |
| `hw0_32_4()` | [0:32:4, 0:32:4] | 64 elements |
| `h0_w0_32()` | [0:1:1, 0:32:1] | First row (32 values) |
| `h0_32_w0()` | [0:32:1, 0:1:1] | First column |
| `hw041()` | [0:4:1, 0:4:1] | 4x4 top-left corner |

### TileSlice Return Codes

| Code | Enum | Meaning |
|------|------|---------|
| 2 | `DPrintOK` | Success |
| 3 | `DPrintErrorBadTileIdx` | Invalid tile index for CB |
| 4 | `DPrintErrorBadPointer` | Bad pointer in CB |
| 5 | `DPrintErrorUnsupportedFormat` | Data format not supported |
| 6 | `DPrintErrorMath` | Math TRISC cannot print tiles (no CB access) |
| 7 | `DPrintErrorEthernet` | Ethernet cores cannot print tiles |

---

## 6.3.7 Interaction with Watcher

| Aspect | Interaction |
|--------|-------------|
| **Waypoint codes** | DPRINT sets waypoints `DPW` (DPRINT Wait) and `DPD` (DPRINT Done) visible in watcher logs |
| **ERISC IRAM** | Enabling DPRINT (like Watcher) disables ERISC IRAM mode; may affect ETH kernel performance |
| **Buffer space** | DPRINT buffers occupy L1 space in the debug region; reduces available L1 for kernels |
| **Binary size** | Both are gated by preprocessor flags; a release build excludes both |
| **Dispatch cores** | `DPrintServer::reads_dispatch_cores()` tracks whether DPRINT targets dispatch cores |
| **No mutual exclusion** | Both can run simultaneously, but combined L1 pressure and host read traffic increase |

---

## 6.3.8 Practical Debugging Workflow

### Step 1: Identify the Suspect Core

From Watcher log or tt-triage output, identify which core(s) and RISC(s) are involved in the hang.

### Step 2: Enable Targeted DPRINT

```bash
export TT_METAL_DPRINT_CORES=3,2          # Target specific core
export TT_METAL_DPRINT_RISCVS=BR+NC       # Target BRISC and NCRISC
export TT_METAL_DPRINT_CHIPS=0             # Target device 0
```

### Step 3: Add Strategic Print Statements

```cpp
#include "debug/dprint.h"

void kernel_main() {
    uint32_t arg0 = get_arg_val<uint32_t>(0);
    DPRINT << "arg0=" << arg0 << ENDL();

    DPRINT << "pre-sem-wait addr=" << HEX() << (uint32_t)sem_addr
           << DEC() << " val=" << *sem_addr << ENDL();

    noc_semaphore_wait(sem_addr, expected_val);

    DPRINT << "post-sem-wait" << ENDL();
}
```

### Step 4: Run and Analyze

Output appears in `generated/dprint/`. If the last print is "pre-sem-wait", the kernel is hanging in the semaphore wait.

### Step 5: Clean Up

Remove or guard DPRINT statements before committing.

---

## 6.3.9 Quick-Start Configuration Recipes

### Print from specific cores, BRISC only

```bash
export TT_METAL_DPRINT_CORES="(0,0),(1,1),(2,2)"
export TT_METAL_DPRINT_RISCVS=BR
./my_program
```

### Print from ethernet cores on chip 0

```bash
export TT_METAL_DPRINT_ETH_CORES=all
export TT_METAL_DPRINT_CHIPS=0
./my_program
```

### One file per RISC with device/core prefix

```bash
export TT_METAL_DPRINT_CORES="(0,0)"
export TT_METAL_DPRINT_ONE_FILE_PER_RISC=1
export TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=1
./my_program
# Output: generated/dprint/device_0_core_0_0_risc_brisc.log, etc.
```

---

## 6.3.10 Hang Scenarios

### Scenario 6.3.1: DPRINT Buffer Full Causes Kernel Hang

**Symptom**: Device kernel hangs with waypoint `DPW` (DPRINT Wait). The DPRINT server is either not running or not targeting the core.

**Root Cause**: The per-RISC DPRINT buffer is only 204 bytes. When a kernel writes more data than the buffer can hold, it enters a spin-wait loop waiting for the host to drain the buffer by advancing `rpos`. If the host server is not running (or not targeting this core/RISC), the wait is infinite.

**Diagnosis Steps**:
1. Check watcher log for cores stuck at waypoint `DPW`
2. Verify that `TT_METAL_DPRINT_CORES` includes the hung core
3. Verify that `TT_METAL_DPRINT_RISCVS` includes the hung RISC type
4. Verify that the DPRINT server is running (attached and not detached)

**Fix**:
1. Ensure DPRINT server is always running when DPRINT macros are compiled in
2. Target only the cores/RISCs you need to debug
3. Reduce print volume: print less frequently, use `SETW` to control output size
4. Use `WATCHER_RING_BUFFER_PUSH()` instead for lightweight logging that cannot cause hangs

**Prevention**: Never enable `DPRINT_CORES=all` on a large grid unless print volume is manageable.

### Scenario 6.3.2: DPRINT on Dispatch Core Deadlocks Command Queue

**Symptom**: Enabling DPRINT on dispatch cores causes the entire command queue to freeze.

**Root Cause**: The dispatch kernel fills its DPRINT buffer while processing a CQ command. It stalls waiting for the host print server to drain the buffer, but the print server may be waiting for device data that requires dispatch to continue (circular dependency).

**Diagnosis Steps**:
1. Watcher shows dispatch core waypoints in a DPRINT-related code path
2. The dispatch CQ appears stuck

**Fix**:
1. Avoid DPRINT on dispatch cores unless absolutely necessary
2. If needed, use extremely minimal prints (single uint32_t values)
3. Consider using Watcher waypoints on dispatch cores instead (zero stall risk)

**Prevention**: Use `TT_METAL_WATCHER_DISABLE_DISPATCH=1` if dispatch binary size is also a concern.

### Scenario 6.3.3: DPRINT Targeting Wrong Cores Misses Hang Data

**Symptom**: DPRINT output appears but does not include the core that is hanging.

**Root Cause**: `TT_METAL_DPRINT_CORES` specifies the wrong coordinates, or the hang occurs on a dispatch/ethernet core not covered by worker core targeting.

**Diagnosis Steps**:
1. Cross-reference DPRINT core targeting with Watcher log to verify which cores are actually hanging
2. Update targeting to include the identified core
3. For dispatch cores, use programmatic API. For ETH cores, use `TT_METAL_DPRINT_ETH_CORES`

**Fix**: Always use Watcher first (Section 6.1) to identify the exact hanging core before setting up DPRINT targeting.

### Scenario 6.3.4: Print Server Startup Race Condition

**Symptom**: DPRINT output is garbled or missing for the first kernel invocation, but works correctly for subsequent invocations.

**Root Cause**: The host print server's magic-value handshake with the device has a bootstrapping limitation. If the kernel starts before the magic value is fully written, `wpos`/`rpos` may not be properly initialized.

**Diagnosis Steps**:
1. First kernel's DPRINT output is corrupted or empty; subsequent kernels print correctly

**Fix**: Use `DPrintServer::await()` after the first kernel to ensure synchronization. The `WriteInitMagic()` function includes a 100K-retry wait loop, but in rare cases this may be insufficient.

---

**Cross-references:**
- Watcher waypoints as alternative to DPRINT: Section 6.1
- Ring buffer as lightweight alternative: Section 6.1.4
- Dispatch core hangs from print stalls: Chapter 4
