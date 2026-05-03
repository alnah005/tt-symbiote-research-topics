# 6.1 Watcher System Architecture and Configuration

## Summary

The Watcher is the primary always-on monitoring system for Tenstorrent devices. It consists of a host-side `WatcherServer` that periodically polls device-side mailbox structures (`watcher_msg_t`) from every core on every attached device, checking for errors, recording waypoints, and logging diagnostic data. When Watcher detects an error (NOC address violation, tripped assert, stack overflow, ETH link failure), it immediately throws an exception that halts the program with a detailed diagnostic message. This section provides exhaustive configuration reference tables for every environment variable, every disableable feature, and every log format element, along with decision trees for when to use the Watcher versus alternatives.

## Prerequisites

- Understanding of Tenstorrent core types: Tensix (BRISC/NCRISC/TRISC0-2), Active ETH (ERISC), Idle ETH, DRAM (Blackhole only)
- Familiarity with L1 memory mailbox layout (Ch3)
- Basic understanding of NOC addressing (Ch2)
- Chapter 1 (waypoint conventions, 5-part diagnostic format)

## 6.1.1 When to Use Watcher vs. Alternatives

```
Do you need PROACTIVE error detection during a run?
|
+-- YES: Is binary size a concern (dispatch/eth kernels too large)?
|   |
|   +-- YES --> Enable Watcher with selective feature disabling:
|   |           TT_METAL_WATCHER_NOINLINE=1
|   |           TT_METAL_WATCHER_DISABLE_DISPATCH=1
|   |           TT_METAL_WATCHER_DISABLE_ETH=1
|   |
|   +-- NO  --> Enable full Watcher: TT_METAL_WATCHER=120
|
+-- NO: Do you need POST-MORTEM state from a hung device?
    |
    +-- Process still alive? --> GDB integration (call tt::watcher::dump)
    +-- Process dead, Watcher was on? --> Read generated/watcher/watcher.log
    +-- Process dead, Watcher was off? --> Use watcher_dump tool (Section 6.2)
```

**Watcher is the right choice when:**
- You are developing or testing kernels and want automatic detection of NOC address errors, CB overflows, and assertion failures
- You need continuous state snapshots to diagnose intermittent hangs
- You want waypoint-based visibility into which code path each RISC-V processor last executed

**Watcher is NOT the right choice when:**
- You need printf-style debugging of specific values (use DPRINT, Section 6.3)
- You need detailed performance timelines (use Tracy, Section 6.5)
- You need to diagnose a hang on a device where Watcher was not enabled (use watcher_dump, Section 6.2)
- Watcher overhead is unacceptable for a performance-critical benchmark (consider lightweight asserts, Section 6.5)

---

## 6.1.2 Architecture Overview

### Host-Side Components

| Component | Source File | Role |
|-----------|------------|------|
| `WatcherServer` | `tt_metal/impl/debug/watcher_server.hpp` | Public API: init, attach, detach devices; register kernels; get lock for mailbox writes |
| `WatcherServer::Impl` | `tt_metal/impl/debug/watcher_server.cpp` | Private implementation: owns poll thread, log files, kernel name registry, dump logic |
| `WatcherDeviceReader` | `tt_metal/impl/debug/watcher_device_reader.hpp/.cpp` | Per-device reader: reads mailbox data, decodes all feature-specific fields, formats log output |
| `RunTimeOptions` | `tt_metal/llrt/rtoptions.hpp/.cpp` | Env var parsing and runtime configuration for all watcher settings |

### Device-Side: Mailbox Protocol

Each Tensix/ERISC core has a `watcher_msg_t` structure in its L1 mailbox area (defined in `tt_metal/hw/inc/hostdev/dev_msgs.h`):

```c
struct watcher_msg_t {
    volatile uint32_t enable;
    struct debug_waypoint_msg_t debug_waypoint[MaxProcessorsPerCoreType];
    struct debug_sanitize_addr_msg_t sanitize[MAX_NUM_NOCS_PER_CORE];
    std::atomic<bool> noc_linked_status[MAX_NUM_NOCS_PER_CORE];
    struct debug_eth_link_t eth_status;
    struct debug_assert_msg_t assert_status;
    struct debug_pause_msg_t pause_status;
    struct debug_stack_usage_t stack_usage;
    struct debug_insert_delays_msg_t debug_insert_delays;
    struct debug_ring_buf_msg_t debug_ring_buf;
};
```

### Data Flow

```
Device Cores                    Host
+------------------+            +------------------+
| watcher_msg_t    | <--read--- | WatcherServer    |
|  .enable         |  (MMIO)    |   .poll_thread   |
|  .debug_waypoint |            |   (interval_ms)  |
|  .sanitize[noc]  |            +--------+---------+
|  .assert_status  |                     |
|  .pause_status   |                     v
|  .debug_ring_buf |            +------------------+
|  .stack_usage    |            | WatcherDevice    |
|  .eth_status     |            |   Reader         |
|  .debug_delays   |            |   .Dump()        |
+------------------+            +--------+---------+
                                         |
                                         v
                                  watcher.log
                                  kernel_names.txt
                                  kernel_elf_paths.txt
```

### Lifecycle

1. **`init_devices()`**: Always runs at device init. Writes default `watcher_msg_t` to every core's mailbox -- sets enable flag, initializes waypoints to `'X'`, sets NOC sanitize sentinels to `0xbadabada`, clears assert/ring buffer/pause fields.
2. **`attach_devices()`**: If `TT_METAL_WATCHER` is set, creates log file, instantiates `WatcherDeviceReader` per device, spawns poll thread. Disables DMA ops (`set_disable_dma_ops(true)`) because the UMD DMA library is not thread-safe with concurrent watcher reads.
3. **`poll_watcher_data()`**: Background thread loop. Acquires mutex, calls `Dump()` on each device reader, sleeps for `interval_ms` (checking stop flag every 100ms for responsive shutdown).
4. **`detach_devices()`**: Signals stop, joins poll thread (2s timeout), cleans up. Re-enables DMA ops. Reports ETH link retraining counts.

**Important:** Enabling Watcher or DPRINT automatically disables ERISC IRAM mode (instruction RAM), which may affect performance of ethernet kernels.

---

## 6.1.3 Master Environment Variable Reference Table

### Core Watcher Configuration

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `TT_METAL_WATCHER` | int | disabled | Enables watcher. Bare integer = poll interval in seconds. E.g., `TT_METAL_WATCHER=1` polls every 1 second; `TT_METAL_WATCHER=120` every 120 seconds |
| `TT_METAL_WATCHER_DUMP_ALL` | flag | `false` | Enables dumping potentially unsafe state (sync registers). Can cause hangs if cores are actively running. Use only when cores are known to be idle or stuck |
| `TT_METAL_WATCHER_APPEND` | flag | `false` | Appends to existing watcher.log instead of overwriting |
| `TT_METAL_WATCHER_NOINLINE` | flag | `false` | Disables inlining of watcher device-side functions. Reduces FW binary size at cost of performance; also enables GDB breakpoints on watcher functions |
| `TT_METAL_WATCHER_PHYS_COORDS` | flag | `false` | Adds physical coordinates alongside logical/virtual in log output |
| `TT_METAL_WATCHER_TEXT_START` | flag | `false` | Includes kernel text start addresses in log output (useful for correlating with ELF symbols) |
| `TT_METAL_WATCHER_SKIP_LOGGING` | flag | `false` | Redirects log file to `/dev/null`. Watcher still runs and detects errors, but produces no persistent log. Useful for performance testing with safety net |
| `TT_METAL_WATCHER_TEST_MODE` | flag | `false` | Test-only: catches watcher exceptions instead of crashing, stores error for later assertion in unit tests |

### Feature Disable Flags

Each feature can be individually disabled while keeping the rest of watcher active. All default to **enabled** when watcher is on.

| Environment Variable | Feature Disabled | Internal String Key | Impact When Disabled |
|---------------------|-----------------|---------------------|---------------------|
| `TT_METAL_WATCHER_DISABLE_WAYPOINT` | Waypoint tracking | `WAYPOINT` | No waypoint status in log; `WAYPOINT()` macro compiles to no-op |
| `TT_METAL_WATCHER_DISABLE_SANITIZE_NOC` | NOC address validation | `NOC_SANITIZE` | No runtime NOC transaction checking; removes ~5-10% overhead from NOC operations |
| `TT_METAL_WATCHER_DISABLE_ASSERT` | Device-side asserts | `ASSERT` | `ASSERT()` on device becomes no-op; no assert trip detection |
| `TT_METAL_WATCHER_DISABLE_PAUSE` | Pause/breakpoint support | `PAUSE` | Device-side `PAUSE()` macro ignored; no interactive pause detection |
| `TT_METAL_WATCHER_DISABLE_RING_BUFFER` | Ring buffer logging | `RING_BUFFER` | `WATCHER_RING_BUFFER_PUSH()` becomes no-op; no ring buffer data in log |
| `TT_METAL_WATCHER_DISABLE_STACK_USAGE` | Stack usage tracking | `STACK_USAGE` | No stack usage summary in watcher dumps; no overflow detection |
| `TT_METAL_WATCHER_DISABLE_DISPATCH` | Dispatch state checking | `DISPATCH` | No watcher instrumentation on dispatch kernels (binary size relief) |
| `TT_METAL_WATCHER_DISABLE_ETH` | Ethernet core monitoring | `ETH` | Skips all ethernet cores during dump (both active and idle; binary size relief) |
| `TT_METAL_WATCHER_DISABLE_CB_SANITIZE` | CB bounds checking | `CB_SANITIZE` | No circular buffer overflow/underflow detection in NOC transactions |
| `TT_METAL_WATCHER_DISABLE_SANITIZE_READ_ONLY_L1` | Read-only L1 check | `SANITIZE_READ_ONLY_L1` | Disables checking NOC reads against L1 bounds |
| `TT_METAL_WATCHER_DISABLE_SANITIZE_WRITE_ONLY_L1` | Write-only L1 check | `SANITIZE_WRITE_ONLY_L1` | Disables checking NOC writes against L1 bounds |
| `TT_METAL_WATCHER_DISABLE_ETH_LINK_STATUS` | ETH link down detect | `ETH_LINK_STATUS` | No detection of post-training ethernet link failures |

### Enable Flags (Opt-In)

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `TT_METAL_WATCHER_ENABLE_NOC_SANITIZE_LINKED_TRANSACTION` | `false` | Enables additional check: catches non-multicast transactions submitted while a linked transaction is pending. Requires NOC sanitize to be enabled |

### Debug Delay Configuration (see also Section 6.6)

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `TT_METAL_WATCHER_DEBUG_DELAY` | `uint32_t` | `0` | Number of delay cycles to insert before NOC operations. Requires `TT_METAL_WATCHER` set and NOC sanitize enabled |
| `TT_METAL_READ_DEBUG_DELAY_CORES` | core list | none | Cores to apply read debug delay to. Format: `(x,y),(x,y)` or `(x1,y1)-(x2,y2)` or `all` |
| `TT_METAL_WRITE_DEBUG_DELAY_CORES` | core list | none | Cores to apply write debug delay to |
| `TT_METAL_ATOMIC_DEBUG_DELAY_CORES` | core list | none | Cores to apply atomic debug delay to |
| `TT_METAL_READ_DEBUG_DELAY_RISCVS` | RISC list | all | RISCs to apply read delay to. Format: `BR`, `NC`, `TR0`, `TR1`, `TR2` (plus-separated) |
| `TT_METAL_WRITE_DEBUG_DELAY_RISCVS` | RISC list | all | RISCs to apply write delay to |
| `TT_METAL_ATOMIC_DEBUG_DELAY_RISCVS` | RISC list | all | RISCs to apply atomic delay to |

---

## 6.1.4 Monitoring Features Reference

### Waypoints

**Source:** `tt_metal/hw/inc/api/debug/waypoint.h`

**Compile gate**: `defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_WAYPOINT) && !defined(FORCE_WATCHER_OFF)`

Waypoints are 4-character status markers that each RISC-V processor writes to its mailbox. The `WAYPOINT(x)` macro encodes up to 4 ASCII characters into a `uint32_t`:

```c
template <size_t N, size_t... Is>
constexpr uint32_t fold(const char (&s)[N], std::index_sequence<Is...>) {
    static_assert(sizeof...(Is) <= 4, "Up to 4 characters allowed in WATCHER_WAYPOINT");
    return ((static_cast<uint32_t>(s[Is]) << (8 * Is)) | ...);
}
```

**Standard waypoint codes**:

| Code | Meaning | Context |
|------|---------|---------|
| `X` | Uninitialized | Host-written default before firmware launch |
| `I` | Initializing | Firmware initialization sequence |
| `W` | Waiting | Top of spin loop, waiting for work |
| `R` | Running | Entering kernel execution |
| `D` | Done | Finished spin loop iteration |

**NOC waypoint codes** (set by NOC API):

| Code | Meaning |
|------|---------|
| `NRW` / `NRD` | NOC Read Wait / Done |
| `NWW` / `NWD` | NOC Write Wait / Done |
| `NAW` / `NAD` | NOC Atomic Wait / Done |
| `DPW` / `DPD` | DPRINT Wait / Done |
| `PASW` / `PASD` | Pause Wait / Done |

---

### NOC Sanitization

**Source:** `tt_metal/hw/inc/internal/debug/sanitize.h`

**Compile gate**: `!COMPILE_FOR_TRISC && WATCHER_ENABLED && !WATCHER_DISABLE_NOC_SANITIZE && !FORCE_WATCHER_OFF` (runs on BRISC/NCRISC/ERISC only)

Every NOC transaction is validated before submission, checking target coordinates, L1 address ranges, alignment, zero-length transfers, multicast validity, mailbox writes, CB bounds, and linked transactions.

When a violation is detected, the function `debug_sanitize_post_addr_and_hang()` writes diagnostic data into the `sanitize[noc_id]` mailbox field:

```c
san->noc_addr = noc_addr;       // Full 64-bit NOC address
san->l1_addr = l1_addr;         // Local L1 address
san->len = len;                 // Transfer length
san->which_risc = hw_thread_idx; // Which processor triggered it
san->is_multicast = ...;
san->is_write = ...;
san->is_target = ...;           // Whether the issue is with remote or local address
san->return_code = return_code; // The specific violation type
```

Then the core enters a deliberate hang (`while(1);`), except on ERISC which calls `erisc_exit()`.

**Return codes** (from `debug_sanitize_noc_return_code_enum`):

| Code | Name | Meaning | Cross-Ref |
|------|------|---------|-----------|
| 2 | `DebugSanitizeOK` | No error (sentinel checked for corruption) | -- |
| 3 | `DebugSanitizeNocAddrUnderflow` | Address below valid range | Ch2 |
| 4 | `DebugSanitizeNocAddrOverflow` | Address + length exceeds valid range | Ch2 |
| 5 | `DebugSanitizeNocAddrZeroLength` | Zero-length or overflow in addr+len | Ch2 |
| 6 | `DebugSanitizeNocTargetInvalidXY` | Target NOC coordinate is unknown/invalid | Ch2 |
| 7 | `DebugSanitizeNocMulticastNonWorker` | Multicast to non-Tensix cores | Ch2 |
| 8 | `DebugSanitizeNocMulticastInvalidRange` | Multicast start/end range invalid | Ch2 |
| 9 | `DebugSanitizeNocAlignment` | L1 and NOC address alignment mismatch | Ch2 |
| 10 | `DebugSanitizeNocMixedVirtualandPhysical` | Multicast mixing virtual and physical coords | Ch2 |
| 11 | `DebugSanitizeInlineWriteDramUnsupported` | Inline write targeting DRAM (unsupported) | Ch2 |
| 12 | `DebugSanitizeNocAddrMailbox` | Write to read-only mailbox region | Ch3 |
| 13 | `DebugSanitizeNocLinkedTransactionViolation` | Unicast during pending linked transaction | Ch2 |
| 14 | `DebugSanitizeL1AddrOverflow` | Direct L1 access overflow | Ch3 |
| 15 | `DebugSanitizeEthSrcL1AddrOverflow` | Ethernet source L1 overflow | Ch5 |
| 16 | `DebugSanitizeEthDestL1AddrOverflow` | Ethernet destination L1 overflow | Ch5 |
| 17 | `DebugSanitizeCBOutOfBounds` | NOC transfer exceeds CB allocated region | Ch3 |

**Sentinel values** (detect corrupted sanitization state):

| Width | Name | Hex |
|-------|------|-----|
| 64-bit | `DEBUG_SANITIZE_SENTINEL_OK_64` | `0xbadabadabadabada` |
| 32-bit | `DEBUG_SANITIZE_SENTINEL_OK_32` | `0xbadabada` |
| 16-bit | `DEBUG_SANITIZE_SENTINEL_OK_16` | `0xbada` |
| 8-bit | `DEBUG_SANITIZE_SENTINEL_OK_8` | `0xda` |

**Quasar special handling:** On Quasar, multiple data movers share one NOC, so the sanitization mailbox uses CAS (`__atomic_compare_exchange_n`) to prevent race conditions. The transient value `DebugSanitizeWriteInProgress = 0xDEAD` indicates a DM is mid-write.

---

### Assertions

**Source:** `tt_metal/hw/inc/api/debug/assert.h`

**Compile gate**: `WATCHER_ENABLED && !WATCHER_DISABLE_ASSERT && !FORCE_WATCHER_OFF`

Assert types (from `debug_assert_type_t`):

| Code | Name | Meaning |
|------|------|---------|
| 2 | `DebugAssertOK` | No assertion tripped |
| 3 | `DebugAssertTripped` | Standard `ASSERT()` failure |
| 4-7 | `DebugAssertNCrisc*Tripped` | Specific NCRISC NOC flush assertion failures |
| 8 | `DebugAssertRtaOutOfBounds` | Runtime argument out of bounds |
| 9 | `DebugAssertCrtaOutOfBounds` | Compile-time runtime argument out of bounds |
| 10 | `DebugAssertHwFault` | Hardware fault (Quasar only; includes `mepc`, `mcause`, `mtval` CSR values) |

**Three assert modes** (from `assert.h`):

| Mode | Condition | Behavior |
|------|-----------|----------|
| Watcher assert | `WATCHER_ENABLED` defined | Full mailbox reporting, host notification, line/RISC info |
| Lightweight assert | `LIGHTWEIGHT_KERNEL_ASSERTS` defined | `asm volatile("ebreak")` -- requires debugger/tt-triage to detect |
| Disabled | Neither defined | Assert compiled out entirely |

---

### Ring Buffer

**Source:** `tt_metal/hw/inc/api/debug/ring_buffer.h`

**Compile gate**: `WATCHER_ENABLED && !WATCHER_DISABLE_RING_BUFFER && !FORCE_WATCHER_OFF`

A circular buffer of `uint32_t` values in the watcher mailbox (defined by `DEBUG_RING_BUFFER_ELEMENTS`), accessible via `WATCHER_RING_BUFFER_PUSH(value)`. The starting index is `-1` (`DEBUG_RING_BUFFER_STARTING_INDEX`); first write increments to 0.

```c
struct debug_ring_buf_msg_t {
    int16_t current_ptr;
    uint16_t wrapped;
    uint32_t data[DEBUG_RING_BUFFER_ELEMENTS];
};
```

**Important:** No synchronization between RISCs on the same core. Calling `WATCHER_RING_BUFFER_PUSH()` from different RISCs simultaneously is undefined behavior.

---

### Pause / Stack Usage / ETH Link Status / CB Sanitization

**Pause** (`tt_metal/hw/inc/api/debug/pause.h`): The `PAUSE()` macro halts kernel execution until the host clears the pause flag. ERISC cores context-switch while waiting.

**Stack Usage** (`tt_metal/hw/inc/internal/debug/stack_usage.h`): Stack painting with sentinel `0xBABABABA` before kernel launch; post-kernel scan measures high-water mark. Stored as:

```c
struct debug_stack_usage_per_cpu_t {
    volatile uint16_t min_free;           // Minimum free stack + 1 (0 = unset)
    volatile uint16_t watcher_kernel_id;  // Kernel ID that produced the minimum
};
```

**ETH Link Status** (`tt_metal/hw/inc/api/debug/eth_link_status.h`): On active ERISC cores, checks `is_link_up()` before transactions. A dead link sets `eth_status.link_down = 1` and enters hang. Cross-reference: Ch5 Section 5.1.

**CB Sanitization**: When enabled, NOC transactions are checked against circular buffer bounds. If a transaction would overflow a CB, `DebugSanitizeCBOutOfBounds` is reported. Cross-reference: Ch3 Section 3.4.

**Linked Transaction Sanitization** (`TT_METAL_WATCHER_ENABLE_NOC_SANITIZE_LINKED_TRANSACTION=1`): Catches submitting a non-multicast transaction while a linked transaction is pending. Cross-reference: Ch2.

---

## 6.1.5 Log File Reference

### Output Files

| File | Path | Description |
|------|------|-------------|
| `watcher.log` | `generated/watcher/watcher.log` | Main watcher dump log |
| `kernel_names.txt` | `generated/watcher/kernel_names.txt` | Kernel ID to name mapping |
| `kernel_elf_paths.txt` | `generated/watcher/kernel_elf_paths.txt` | Kernel ID to ELF path mapping |

### Per-Core Log Line Structure

```
Device <id> <type> core(x=<lx>,y=<ly>) virtual(x=<vx>,y=<vy>): <waypoints>  rmsg:<mode><noc_id><state>|<enables> smsg:<sub_states> k_ids:<id0>|<id1>|...
```

| Field | Example | Meaning |
|-------|---------|---------|
| `rmsg` | `D0G\|BRNCTR` | Device dispatch, NOC0, Go state; BRISC+NCRISC+TRISC all enabled |
| `smsg` | `GDDD` | Subordinate: Go, Done, Done, Done |
| `k_ids` | `5\|3\|12\|0\|0` | Kernel IDs per processor (BRISC/NCRISC/TRISC0/1/2) |

### Run State Codes

| Code | Meaning |
|------|---------|
| `I` | Initialized |
| `G` | Go (executing) |
| `D` | Done |
| `R` | Reset read pointer |
| `L` | Loading |
| `W` | Waiting for reset |
| `S` | Initializing sync registers |
| `U` | Unknown/corrupt (triggers TT_THROW) |

---

## 6.1.6 GDB Integration

The `WatcherServer::Impl::dump(FILE* f)` is declared `__attribute__((noinline))` so it can be called from GDB:

```
(gdb) call tt::watcher::dump(stderr, true)
```

This produces a complete watcher dump to the GDB console without restarting or modifying the running program. Use `TT_METAL_WATCHER_NOINLINE=1` to enable GDB breakpoints on watcher functions:

```
(gdb) break debug_sanitize_post_addr_and_hang
```

---

## 6.1.7 Performance Impact Reference

| Feature | Approximate Overhead | Notes |
|---------|---------------------|-------|
| Watcher enabled (all features) | 5-15% | Varies by workload; NOC sanitization is dominant cost |
| NOC sanitization only | 5-10% | Per-transaction address validation |
| Waypoints only | < 1% | Single L1 write per waypoint |
| Ring buffer only | < 1% | Single L1 write per push |
| Stack usage tracking | < 1% | Periodic watermark update |
| `WATCHER_NOINLINE=1` | Reduces FW binary size | May slightly decrease runtime performance due to function call overhead |
| DMA disabled during watcher | N/A | DMA library is not thread-safe; watcher forces MMIO-only reads |

### Recipe: Minimal-Overhead Watcher for CI

```bash
export TT_METAL_WATCHER=120
export TT_METAL_WATCHER_SKIP_LOGGING=1   # No log file, but errors still detected
export TT_METAL_WATCHER_DISABLE_RING_BUFFER=1
export TT_METAL_WATCHER_DISABLE_STACK_USAGE=1
```

### Recipe: Maximum Debugging Information

```bash
export TT_METAL_WATCHER=1
export TT_METAL_WATCHER_NOINLINE=1
export TT_METAL_WATCHER_PHYS_COORDS=1
export TT_METAL_WATCHER_TEXT_START=1
export TT_METAL_WATCHER_ENABLE_NOC_SANITIZE_LINKED_TRANSACTION=1
```

---

## 6.1.8 Hang Scenarios

### Scenario 6.1.1: NOC Address Violation Detected by Watcher

**Symptom**: Watcher error showing `DebugSanitizeNocAddrOverflow` (or other sanitize error) with specific core, RISC, and transaction details.

**Root Cause**: Kernel computed an incorrect NOC address (e.g., buffer base + offset exceeds L1 boundary).

**Diagnosis Steps**:
1. Read the watcher error message: it provides the exact RISC, NOC ID, source L1 address, target core, target address, and transaction length.
2. Check `kernel_names.txt` to map `k_ids` to kernel source files.
3. Cross-reference with the waypoint status to determine where in the kernel the error occurred.

**Fix**: Fix the address computation in the kernel. Use `noc_async_write` with validated bounds.

**Prevention**: Keep `TT_METAL_WATCHER=1` in CI environments with `TT_METAL_WATCHER_SKIP_LOGGING=1` for minimal overhead.

### Scenario 6.1.2: Stack Overflow Detected by Watcher

**Symptom**: Watcher log shows `(OVERFLOW)` in stack usage summary for a specific processor.

**Root Cause**: Kernel or firmware uses more stack space than allocated (deep call chains, large local arrays, or recursive functions).

**Diagnosis Steps**:
1. Identify the processor and kernel from the stack usage summary.
2. Check the kernel for large local variables, deep recursion, or excessive call depth.
3. Use `TT_METAL_WATCHER_NOINLINE=1` to get accurate stack usage (inlining can hide true stack depth).

**Fix**: Reduce stack usage by moving large arrays to L1 buffers, reducing call depth, or using iterative algorithms.

**Prevention**: Monitor stack usage in CI with watcher enabled; alert on values below 64 bytes free.

### Scenario 6.1.3: Watcher Binary Too Large for Dispatch Kernels

**Symptom**: Compilation fails or device errors due to dispatch kernel binary exceeding size limits when Watcher is enabled.

**Root Cause**: Watcher instrumentation adds significant code to dispatch kernels (`cq_prefetch.cpp`, `cq_dispatch.cpp`) which have tight binary size constraints.

**Diagnosis Steps**:
1. Check build output for binary size overflow errors on dispatch or ETH kernels.

**Resolution (progressive)**:
1. `TT_METAL_WATCHER_NOINLINE=1` -- Disable function inlining in watcher code
2. `TT_METAL_WATCHER_DISABLE_DISPATCH=1` -- Compile out debug tools on dispatch kernels entirely
3. `TT_METAL_WATCHER_DISABLE_ETH=1` -- Also remove from ethernet kernels if needed
4. Selectively disable individual features (see Feature Disable Flags table)

**Prevention**: Use selective feature disabling in CI configurations for dispatch-heavy workloads.

### Scenario 6.1.4: ETH Link Down After Training

**Symptom**: Watcher log shows `eth_status.link_down != 0` for an active ethernet core.

**Root Cause**: Physical link failure, cable issue, or signal integrity problem on the ethernet connection between chips.

**Diagnosis Steps**:
1. Check the physical cable connections.
2. Review the retraining event count in the watcher detach log.
3. Cross-reference with Ch5 multi-chip hang scenarios.

**Fix**: Replace cable, check board-level connections, or move to different ethernet ports.

**Prevention**: Monitor retraining events; high counts indicate marginal links.

---

**Cross-references:**
- Waypoint codes for specific blocking primitives: Chapter 1
- NOC sanitization violations: Chapter 2, Sections 2.3-2.4
- CB deadlock scenarios: Chapter 2, Section 2.2; Chapter 3, Section 3.4
- Stack overflow leading to memory corruption: Chapter 3
- Dispatch core watcher monitoring: Chapter 4
- Ethernet link status and multi-chip hangs: Chapter 5
