# 3.1 L1 Memory Corruption and Overflow

L1 memory corruption is the most insidious class of memory-related hangs. Unlike NOC address violations (covered in [Chapter 2, Section 03](../ch02_kernel_and_noc_hangs/03_noc_address_sanitization_and_violations.md)) which are caught by the sanitizer and produce a deliberate hang with a clear error code, L1 corruption often manifests as a **silent, delayed failure**: data is overwritten without immediate detection, and the hang surfaces later when a kernel reads the corrupted data and uses it as a NOC address, CB pointer, or tile count.

Reference files: `tt_metal/hw/inc/internal/debug/sanitize.h` (lines 535-552), `tt_metal/hw/inc/hostdev/dev_msgs.h`, `tt_metal/hw/inc/internal/tt-1xx/wormhole/dev_mem_map.h`, `tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h`, `tt_metal/impl/debug/watcher_device_reader.cpp`

---

## The L1 Memory Map

Each Tensix core has a private L1 SRAM whose layout is defined at compile time via `dev_mem_map.h`. The map is divided into protected system regions and user-accessible regions:

```
Address 0x00000 : MEM_BOOT_CODE_BASE (firmware launch address / fw_launch_addr_value)
Address 0x00004 : MEM_NOC_ATOMIC_RET_VAL_ADDR
Address 0x0000C : MEM_L1_BARRIER
Address 0x00010 : MEM_L1_ARC_FW_SCRATCH (16 bytes)         [BH only]
Address 0x00020 : MEM_L1_INLINE_BASE (inline write staging) [BH only]
Address 0x00060 : MEM_MAILBOX_BASE (96 on BH; 16 on WH)
         ...    : mailboxes_t structure (~12896 bytes)
                   - ncrisc_halt, subordinate_sync
                   - launch_msg_rd_ptr, launch[8], go_messages[9]
                   - watcher_msg_t (waypoints, sanitize, assert, pause, stack_usage, ring_buf)
                   - dprint_buf, core_info
                   - profiler
  MEM_MAILBOX_END:
  MEM_ZEROS_BASE:  (512 bytes of zeros, 32-byte aligned)
  MEM_LLK_DEBUG_BASE:
  MEM_BRISC_FIRMWARE_BASE:
  MEM_NCRISC_FIRMWARE_BASE:
  MEM_TRISC0_FIRMWARE_BASE:
  MEM_TRISC1_FIRMWARE_BASE:
  MEM_TRISC2_FIRMWARE_BASE:
  MEM_NOC_COUNTER_BASE:     (NOC transaction counters)
  MEM_FABRIC_COUNTER_BASE:  (fabric transaction counters)
  MEM_TENSIX_ROUTING_TABLE_BASE:
  MEM_TENSIX_FABRIC_CONNECTIONS_BASE:
  MEM_MAP_READ_ONLY_END:    (writes below this trigger DebugSanitizeNocAddrMailbox)
  MEM_PACKET_HEADER_POOL_BASE:
  MEM_MAP_END:               (everything above is user/kernel space)
     Kernel text, CB configs, semaphores, runtime args, user L1 buffers
  MEM_L1_BASE + MEM_L1_SIZE: End of L1
```

### Architecture-Specific Properties

| Property | Wormhole (WH) | Blackhole (BH) | Quasar |
|----------|---------------|-----------------|--------|
| `MEM_L1_SIZE` | 1464 KB (1,499,136 B) | 1536 KB (1,572,864 B) | 4096 KB (4,194,304 B) |
| `MEM_ETH_SIZE` | ~256 KB - 32 B | 512 KB | 512 KB |
| `MEM_MAILBOX_BASE` | 16 (0x10) | 96 (0x60) | `MEM_MAILBOX_BASE + MEM_L1_UNCACHED_BASE` |
| NCRISC IRAM | Yes (16 KB at 0xFFC00000) | No (all in L1) | Yes |
| Inline write staging | No | Yes (`MEM_L1_INLINE_BASE` at 0x20) | No |
| L1 data cache | No | Yes (`enable_hw_cache_invalidation`) | Yes (uncached region at `MEM_L1_UNCACHED_BASE`) |

### The Read-Only and Read-Write Boundaries

The watcher enforces two L1 protection boundaries via `debug_valid_worker_addr()`:

```c++
// tt_metal/hw/inc/internal/debug/sanitize.h, debug_valid_worker_addr()
inline uint16_t debug_valid_worker_addr(uint64_t addr, uint64_t len, bool write) {
    if (addr + len <= addr) {
        return DebugSanitizeNocAddrZeroLength;
    }
    if (addr < MEM_L1_BASE) {
        return DebugSanitizeNocAddrUnderflow;
    }
    if (addr + len > MEM_L1_BASE + MEM_L1_SIZE) {
        return DebugSanitizeNocAddrOverflow;
    }
#if !defined(DISPATCH_KERNEL) || (DISPATCH_KERNEL == 0)
    if (write && (addr < MEM_MAP_READ_ONLY_END)) {
        return DebugSanitizeNocAddrMailbox;
    }
#endif
    return DebugSanitizeOK;
}
```

- **`MEM_MAP_READ_ONLY_END`**: NOC writes below this address trigger `DebugSanitizeNocAddrMailbox` (return code 12). This protects the mailbox, firmware, routing tables, and fabric connection metadata from accidental overwrites.
- **`MEM_L1_BASE + MEM_L1_SIZE`**: Any access beyond this boundary triggers `DebugSanitizeNocAddrOverflow`.

**Critical limitation:** These checks only apply to **NOC transactions when the watcher is enabled**. Direct RISC-V load/store instructions to L1 are **not checked** by this mechanism. The `debug_sanitize_l1_access` function (Section 3.1.6) provides partial coverage for direct accesses, but only in specific call sites. Dispatch kernels (`DISPATCH_KERNEL == 1`) are exempt from the mailbox write check because they must write to mailbox regions as part of their normal operation.

---

## Hang Cause 3.1.1: Address-0 Firmware Launch Value Corruption

### Symptom

The watcher throws: `"Watcher found corruption at L1[0] on core {x,y}: read {value}"`. The core may also exhibit erratic behavior -- executing garbage instructions, hanging at unexpected waypoints, or failing to respond to go messages. If the watcher is not running, the corruption silently breaks future kernel launches.

### Root Cause

L1 address 0 holds the firmware launch address (`fw_launch_addr_value`). When BRISC firmware boots, it jumps to this address. If any kernel or NOC transaction overwrites address 0, the next kernel launch on that core will jump to a garbage address. The watcher's `DumpL1Status()` function detects this by periodically reading address 0 and comparing it against the expected value:

```cpp
// From watcher_device_reader.cpp, DumpL1Status()
void WatcherDeviceReader::Core::DumpL1Status() const {
    // Read L1 address 0, looking for memory corruption
    std::vector<uint32_t> data;
    data = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        reader_.device_id, virtual_coord_, HAL_MEM_L1_BASE, sizeof(uint32_t));
    TT_ASSERT(programmable_core_type_ == HalProgrammableCoreType::TENSIX);
    uint32_t core_type_idx =
        MetalContext::instance().hal().get_programmable_core_type_index(
            HalProgrammableCoreType::TENSIX);
    auto fw_launch_value =
        MetalContext::instance().hal().get_jit_build_config(
            core_type_idx, 0, 0).fw_launch_addr_value;
    if (data[0] != fw_launch_value) {
        LogRunningKernels();
        TT_THROW("Watcher found corruption at L1[0] on core {}: read {}",
                 virtual_coord_.str(), data[0]);
    }
}
```

Common causes of address-0 corruption:
- A `noc_async_write` with a miscalculated destination address that wraps to 0
- A runtime argument (`get_arg_val<uint32_t>(0)`) used as a destination L1 address but never set (defaults to 0)
- A multicast write with an incorrect address field targeting L1 address 0 on all cores in range

### Diagnosis Steps

1. Enable the watcher (`export TT_METAL_WATCHER=120`) and check `generated/watcher/watcher.log` for the corruption message.
2. Identify which kernels were running on the affected core (the watcher log prints them via `LogRunningKernels()`).
3. Search for any NOC write whose destination address could be 0 or could wrap around to 0.
4. Check for CB configurations where `fifo_limit - fifo_size` could be negative or zero.

### Fix

Correct the address calculation. Ensure the write targets an address within `[MEM_MAP_END, MEM_L1_SIZE)`.

### Prevention

- Enable watcher in CI to catch address-0 corruption early.
- Use `DEBUG_SANITIZE_NOC_WRITE_TRANSACTION` (enabled by default when `WATCHER_ENABLED` is defined) to validate all NOC write destinations before submission.
- Never compute NOC target addresses from unvalidated runtime arguments without bounds checking.

---

## Hang Cause 3.1.2: CB Write Overflowing into Adjacent L1 Regions

### Symptom

A kernel hang on a core where the watcher shows no NOC sanitization violation, but the core is stuck at an unexpected waypoint. Inspecting L1 reveals that data belonging to one CB has overwritten the metadata or data of an adjacent CB, or has overwritten the mailbox region. If CB sanitization is enabled, the watcher may report `DebugSanitizeCBOutOfBounds` (return code 17).

### Root Cause

CB memory regions are contiguous in L1. If a producer writes more data than the CB can hold (exceeding `fifo_limit`), the write pointer advances past the CB boundary and overwrites whatever is at the next higher L1 address. The `cb_push_back` function has a debug assertion but it is compiled out in non-debug builds:

```c++
// tt_metal/hw/inc/api/dataflow/dataflow_api.h, cb_push_back()
void cb_push_back(const int32_t operand, const int32_t num_pages) {
    uint32_t num_words = num_pages * get_local_cb_interface(operand).fifo_page_size;
    pages_received_ptr[0] += num_pages;
    get_local_cb_interface(operand).fifo_wr_ptr += num_words;

    // this will basically reset fifo_wr_ptr to fifo_addr -- no other wrap is legal
    ASSERT(get_local_cb_interface(operand).fifo_wr_ptr <= get_local_cb_interface(operand).fifo_limit);
    if (get_local_cb_interface(operand).fifo_wr_ptr == get_local_cb_interface(operand).fifo_limit) {
        get_local_cb_interface(operand).fifo_wr_ptr -= get_local_cb_interface(operand).fifo_size;
    }
}
```

When `fifo_wr_ptr > fifo_limit`, the ASSERT fires in debug mode, but in release mode the write pointer wraps incorrectly (the subtraction only happens when `== fifo_limit`, not `>`), and subsequent writes go to wrong addresses.

The CB bounds check `debug_valid_cb_addr()` catches NOC transactions that overflow a CB:

```cpp
// From sanitize.h, debug_valid_cb_addr()
inline uint16_t debug_valid_cb_addr(uint32_t l1_addr, uint32_t len) {
    for (uint32_t i = 0; i < NUM_CIRCULAR_BUFFERS; i++) {
        LocalCBInterface& cb = get_local_cb_interface(i);
        if (cb.fifo_size == 0) continue;  // unused CB
        uint32_t cb_start = cb.fifo_limit - cb.fifo_size;
        uint32_t cb_end = cb.fifo_limit;
        if (l1_addr >= cb_start && l1_addr < cb_end) {
            if (static_cast<uint64_t>(l1_addr) + len > cb_end) {
                return DebugSanitizeCBOutOfBounds;
            }
            return DebugSanitizeOK;
        }
    }
    return DebugSanitizeOK;
}
```

**Critical limitation:** This check only validates NOC transactions. Direct L1 memory access from kernel code (e.g., writing through a pointer obtained via `get_write_ptr()`) is *not* validated.

**Buggy code:**
```c++
// Reader kernel -- writes 10 tiles but CB only holds 8
constexpr uint32_t cb_out = tt::CBIndex::c_0;
for (uint32_t i = 0; i < 10; i++) {
    cb_reserve_back(cb_out, 1);
    uint32_t l1_write_addr = get_write_ptr(cb_out);
    noc_async_read(src_noc_addr, l1_write_addr, tile_size);
    noc_async_read_barrier();
    cb_push_back(cb_out, 1);
    // BUG: Consumer only pops 8 tiles, so reserve_back blocks after 8
}
```

**Corrected code:**
```c++
// Ensure the loop count matches what the consumer will pop
constexpr uint32_t num_tiles = 8;  // matches CB capacity and consumer expectation
for (uint32_t i = 0; i < num_tiles; i++) {
    cb_reserve_back(cb_out, 1);
    uint32_t l1_write_addr = get_write_ptr(cb_out);
    noc_async_read(src_noc_addr, l1_write_addr, tile_size);
    noc_async_read_barrier();
    cb_push_back(cb_out, 1);
}
```

### Diagnosis Steps

1. Dump L1 memory for the suspected core and examine CB regions.
2. Compare `fifo_wr_ptr` against `fifo_limit` for each active CB. If `fifo_wr_ptr > fifo_limit`, overflow has occurred.
3. Check whether data at `fifo_limit` through `fifo_limit + overflow_size` belongs to a different CB or system region.
4. Enable watcher CB sanitization (ensure `TT_METAL_WATCHER_DISABLE_CB_SANITIZE` is **not** set).

### Fix

Match the producer tile count to the CB capacity and consumer pop count. See [Chapter 2, Section 02](../ch02_kernel_and_noc_hangs/02_circular_buffer_deadlocks.md) for the full CB protocol requirements.

### Prevention

- Size CBs to accommodate the maximum burst that any single `cb_reserve_back` / `cb_push_back` pair will use.
- Enable `WATCHER_ENABLED` in testing to activate the `ASSERT` in `cb_push_back`.
- Use watcher CB sanitization to catch NOC DMA transfers that exceed CB bounds.

---

## Hang Cause 3.1.3: Silent Corruption of Runtime Arguments

### Symptom

A kernel hangs at `NRBW` (NOC read barrier wait) or `NWBW` (NOC write barrier wait). The watcher (if enabled with NOC sanitization) may report `DebugSanitizeNocTargetInvalidXY` or `DebugSanitizeNocAddrOverflow`. The NOC address logged by the sanitizer appears to be random garbage rather than a plausible core coordinate or L1 offset.

### Root Cause

Runtime arguments (RTAs) are stored in L1 at offsets specified by `kernel_config_msg_t.rta_offset`. If an earlier operation (a CB overflow, an out-of-bounds DMA write, or a stack overflow) corrupts the RTA region, subsequent `get_arg_val<uint32_t>(n)` calls return garbage values. When these values are used to construct NOC addresses:

```c++
// Reader kernel using runtime args for NOC address
uint32_t src_addr = get_arg_val<uint32_t>(0);   // corrupted: should be 0x100000, reads 0xDEADBEEF
uint32_t src_noc_x = get_arg_val<uint32_t>(1);  // corrupted
uint32_t src_noc_y = get_arg_val<uint32_t>(2);  // corrupted
uint64_t src_noc_addr = get_noc_addr(src_noc_x, src_noc_y, src_addr);
noc_async_read(src_noc_addr, dst_l1_addr, tile_size);
noc_async_read_barrier();  // HANGS: the read was issued to an invalid NOC address
```

Without the watcher, the NOC transaction silently fails (no response returns), and the read barrier spins forever.

### Diagnosis Steps

1. If the watcher caught a sanitization violation, examine the logged `noc_addr` -- random-looking values strongly suggest corruption.
2. Read the RTA region from L1 and compare against the host-side values that were programmed.
3. Look for CB overflow or stack overflow on the same core (causes 3.1.2, 3.1.5) as the primary corruption source.
4. Use the watcher ring buffer (`WATCHER_RING_BUFFER`) to trace recent memory accesses.

### Fix

Fix the primary corruption source. The runtime arguments themselves are not the bug -- they are the victim.

### Prevention

- Always pair CB sizing validation with runtime argument layout validation.
- Enable stack usage monitoring (`STACK_USAGE` watcher feature) to detect stack overflow before it corrupts RTAs.
- Enable watcher assertions (`DebugAssertRtaOutOfBounds`, return code 8) to catch runtime argument accesses that exceed the allocated RTA region.

---

## Hang Cause 3.1.4: Corrupted CB Metadata Leading to Infinite Wait

### Symptom

A consumer core hangs at `CWFW` (waiting for tiles) or a producer core hangs at `CRBW` (waiting for free space), but the tile counts are correct on both sides. The issue is that the CB's `fifo_size`, `fifo_limit`, or `fifo_page_size` has been corrupted, causing the free-space or available-tiles calculation to produce incorrect results.

### Root Cause

The `LocalCBInterface` structure is stored in the `cb_interface` array in L1 (or local SRAM for TRISC). If an adjacent memory region overflows into this array, the CB metadata fields are corrupted. The effects depend on which field is corrupted:

- **Corrupted `fifo_num_pages`:** `cb_reserve_back` computes free space as `fifo_num_pages - (pages_received - pages_acked)`. If `fifo_num_pages` is corrupted to a value smaller than `pages_received - pages_acked`, the subtraction underflows (wrapping in unsigned arithmetic), but the cast to `int32_t` produces a negative value, and the loop condition `free_space_pages < num_pages` is always true. The producer spins forever.
- **Corrupted `tiles_received` or `tiles_acked`:** These 16-bit counters drive the producer/consumer synchronization. A corrupted counter can make `cb_wait_front` believe no tiles are available when they actually are, or make `cb_reserve_back` believe the CB is full when it is empty.
- **Corrupted `fifo_page_size`:** `cb_push_back` advances `fifo_wr_ptr` by `num_pages * fifo_page_size`. An incorrect page size causes the write pointer to advance by the wrong amount, leading to a pointer that never reaches `fifo_limit` and never wraps, eventually overflowing L1.

### Diagnosis Steps

1. Use the watcher to identify the hung waypoint (`CWFW` or `CRBW`) and the core.
2. Read the `cb_interface` array from L1 for the affected CB index.
3. Compare the actual values of `fifo_size`, `fifo_limit`, `fifo_page_size`, `fifo_num_pages`, `tiles_acked`, and `tiles_received` against the expected values from the program configuration.
4. If any field does not match, trace the L1 address of that field and identify what could have overwritten it.
5. Dump all CB interfaces, even unused ones (`fifo_size == 0` for unused CBs).

### Fix

Fix the underlying overflow that corrupted CB metadata.

### Prevention

- Enable watcher CB sanitization to detect out-of-bounds writes to CB regions early.
- Verify CB memory layout in host-side tests by reading back CB configurations after programming.

---

## Hang Cause 3.1.5: RISC-V Stack Overflow

### Symptom

The watcher reports `"Watcher detected stack overflow on Device {D} Core {X,Y}: {processor}! Kernel {name} uses (at least) all of the stack."` Or the core hangs in an unpredictable state because the stack overflow corrupted firmware or kernel data.

### Root Cause

Each RISC-V processor on a Tensix core has a small stack allocation in local memory (processor-specific SRAM, not L1). The minimum stack sizes are:

| Processor | Local Memory (WH) | Local Memory (BH) | Min Stack Size |
|-----------|-------------------|-------------------|----------------|
| BRISC | 4 KB | 8 KB | 256 B |
| NCRISC | 4 KB | 8 KB | 256 B |
| TRISC0 | 2 KB | 4 KB | 192 B |
| TRISC1 | 2 KB | 4 KB | 192 B |
| TRISC2 | 2 KB | 4 KB | 256 B |
| IERISC | - | 8 KB | 128 B (WH) / 192 B (BH) |

The watcher tracks stack usage via `debug_stack_usage_t`:

```cpp
// tt_metal/hw/inc/hostdev/dev_msgs.h
struct debug_stack_usage_per_cpu_t {
    volatile uint16_t min_free;           // minimum free stack, offset by +1 (0 == unset)
    volatile uint16_t watcher_kernel_id;  // which kernel had the lowest free stack
};
```

The firmware periodically samples the stack pointer and records the minimum observed free space. A `min_free` value of 1 (representing 0 bytes free after the +1 offset) indicates overflow. The watcher host-side code reports this:

```cpp
// From watcher_device_reader.cpp
if (info.stack_free == 0) {
    fprintf(f, " (OVERFLOW)");
    log_fatal(tt::LogMetal,
        "Watcher detected stack overflow on Device {} Core {}: "
        "{}! Kernel {} uses (at least) all of the stack.",
        device_id, info.virtual_coord.str(), processor_name,
        kernel_names[info.kernel_id].c_str());
} else if (info.stack_free < min_threshold) {
    fprintf(f, " (Close to overflow)");
}
```

**Buggy code:**
```c++
// Kernel with deep recursion that overflows TRISC stack (only 2 KB on WH)
void process_recursive(uint32_t depth, uint32_t* data) {
    uint32_t local_buffer[64];  // 256 bytes on stack per call frame
    if (depth == 0) return;
    process_recursive(depth - 1, local_buffer);
}
```

**Corrected code:**
```c++
// Use iteration and L1 buffers instead of stack-allocated arrays
void kernel_main() {
    uint32_t l1_buffer_addr = get_arg_val<uint32_t>(0);
    volatile tt_l1_ptr uint32_t* work_buf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_buffer_addr);
    for (uint32_t depth = 10; depth > 0; depth--) {
        // ... process data using work_buf in L1 ...
    }
}
```

### Diagnosis Steps

1. Enable watcher stack tracking (on by default unless `TT_METAL_WATCHER_DISABLE_STACK_USAGE=1`).
2. Check `generated/watcher/watcher.log` for `STACK_USAGE` entries. Look for cores where `min_free` is very small (< 32 bytes) or `(OVERFLOW)`.
3. The watcher identifies the specific kernel running when the lowest stack was recorded.

### Fix

Reduce stack usage: (a) replace recursion with iteration, (b) move large local arrays to L1 buffers, (c) reduce call depth by inlining small functions.

### Prevention

- Always enable watcher stack tracking in CI.
- Set a threshold of 64 bytes free as a CI gate.
- Avoid `alloca` and variable-length arrays in device kernels.

---

## Hang Cause 3.1.6: L1 Bounds Overflow via Direct Access

### Symptom

The watcher detects and reports `DebugSanitizeL1AddrOverflow` (return code 14). The core hangs in a deliberate `while(1)` spin loop.

### Root Cause

The `debug_sanitize_l1_access` function validates direct L1 accesses (not NOC transactions) against the L1 size boundary:

```c++
// tt_metal/hw/inc/internal/debug/sanitize.h, lines 535-552
void debug_sanitize_l1_access(uint64_t addr, uint32_t len) {
#if defined(COMPILE_FOR_ERISC)
    constexpr uint64_t l1_overflow_addr = MEM_ETH_SIZE;
#else
    constexpr uint64_t l1_overflow_addr = MEM_L1_SIZE;
#endif
    if (addr + len <= addr || addr + len > l1_overflow_addr) {
        debug_sanitize_post_addr_and_hang(
            0,  // unused (not a noc transaction)
            0,  // unused (not a noc transaction)
            addr, len,
            DEBUG_SANITIZE_NOC_UNICAST,
            DEBUG_SANITIZE_NOC_WRITE,
            DEBUG_SANITIZE_NOC_TARGET,
            DebugSanitizeL1AddrOverflow);
    }
}
```

This check catches two conditions:
1. **Integer overflow**: `addr + len <= addr` (wraparound).
2. **L1 boundary overflow**: `addr + len > MEM_L1_SIZE` (or `MEM_ETH_SIZE` for ethernet cores).

**Important gap:** This only checks the *upper* bound. It does not prevent writes into the system-reserved region (addresses below `MEM_MAP_END`). The mailbox write protection is only enforced for NOC transactions, not direct L1 access.

### Diagnosis Steps

1. The watcher log will contain the violating L1 address and length.
2. Check which RISC-V core triggered the violation (`which_risc` field).
3. Trace back to the kernel code that performed the direct L1 access.

### Fix

Correct the buffer address calculation or reduce the access length to fit within L1 bounds.

### Prevention

- Validate all L1 addresses against architecture-specific `MEM_L1_SIZE` before use.
- Enable watcher to get early detection via `DEBUG_SANITIZE_L1_ADDR`.

---

## Hang Cause 3.1.7: Ethernet L1 Source/Destination Overflow

### Symptom

The watcher detects `DebugSanitizeEthSrcL1AddrOverflow` (return code 15) or `DebugSanitizeEthDestL1AddrOverflow` (return code 16). The ethernet core enters a deliberate hang.

### Root Cause

The `debug_sanitize_eth` function validates both source and destination addresses for ethernet data transfers:

```c++
// tt_metal/hw/inc/internal/debug/sanitize.h
void debug_sanitize_eth(uint32_t src_addr, uint32_t dst_addr, uint32_t len) {
#if defined(COMPILE_FOR_ERISC)
    constexpr uint32_t l1_overflow_addr = MEM_ETH_SIZE;
    if (src_addr + len <= src_addr || src_addr + len > l1_overflow_addr) {
        debug_sanitize_post_addr_and_hang(..., DebugSanitizeEthSrcL1AddrOverflow);
    }
    if (dst_addr + len <= dst_addr || dst_addr + len > l1_overflow_addr) {
        debug_sanitize_post_addr_and_hang(..., DebugSanitizeEthDestL1AddrOverflow);
    }
#endif
}
```

ETH L1 is significantly smaller than Tensix L1 (256 KB on WH, 512 KB on BH), making overflow more likely when kernels designed for Tensix cores are inadvertently deployed to ethernet cores, or when buffer sizes are computed using Tensix L1 constants.

### Diagnosis Steps

1. Check the watcher log for whether the overflow is on the source or destination side.
2. Verify the source and destination buffer addresses against `MEM_ETH_SIZE`.
3. Check whether the kernel was intended for ethernet deployment -- many buffer size calculations use `MEM_L1_SIZE` (Tensix) by default.

### Fix

Use `MEM_ETH_SIZE` rather than `MEM_L1_SIZE` for buffer calculations on ethernet cores.

### Prevention

- Use architecture-aware buffer sizing that checks the core type at compile time.
- Test ethernet kernels with watcher enabled.

---

## Hang Cause 3.1.8: Mailbox Region Write Corruption (Without Watcher)

### Symptom

When the watcher is **enabled**, `DebugSanitizeNocAddrMailbox` (return code 12) is reported and the core enters a deliberate hang. When the watcher is **disabled**, a NOC write to an address below `MEM_MAP_READ_ONLY_END` silently overwrites mailbox data, causing unpredictable symptoms:
- The core never starts (corrupted `go_messages` signal).
- The core launches with wrong kernel configuration (corrupted `launch_msg_t`).
- NOC address validation uses wrong bounds (corrupted `core_info_msg_t.noc_dram_addr_base/end`).

### Root Cause

In production builds without watcher, there is **no hardware memory protection** on the mailbox region. Any NOC write (including multicast writes) that targets an L1 address in the mailbox range will succeed silently. Dispatch kernels are exempt from the check because they legitimately write to the mailbox region.

### Diagnosis Steps

1. If the failure is reproducible, enable the watcher and check for `DebugSanitizeNocAddrMailbox`.
2. Read and compare the mailbox contents against expected values.
3. Look for multicast writes with L1 offsets below `MEM_MAP_READ_ONLY_END`.

### Fix

Correct the NOC write address to target the intended L1 region above `MEM_MAP_END`.

### Prevention

- Always enable watcher during development and in CI.
- Review all multicast write addresses to ensure they target user L1 space.

---

## Hang Cause 3.1.9: Blackhole Inline Write Region Corruption

### Symptom

On Blackhole only: inline writes (`noc_inline_dw_write`) to remote cores carry incorrect data. The remote core reads a garbage value, leading to wrong semaphore increments, corrupted control messages, or secondary NOC transactions to invalid addresses.

### Root Cause

Blackhole emulates inline writes by first writing the value to a dedicated L1 staging region (`MEM_L1_INLINE_BASE`, 16 bytes per NOC) and then issuing a standard NOC async write from that region. If this region is corrupted by an adjacent buffer overflow, the inline write sends garbage data instead of the intended value.

```c++
// blackhole/dev_mem_map.h
#define MEM_L1_INLINE_SIZE_PER_NOC 16
#define MEM_L1_INLINE_BASE 32  // just above ARC FW scratch
```

The inline-write staging area is at addresses 32-95 (for 2 NOCs x 2 processors). This is below `MEM_MAILBOX_BASE` (96 on BH), making it part of the system-critical region. The watcher's mailbox protection should catch writes to this region, but only if NOC sanitization is enabled.

### Diagnosis Steps

1. Confirm the target chip is Blackhole.
2. Read the inline write staging area (L1 addresses 32-95) to see if the values match expected inline write data.
3. Check for adjacent buffer overflows that could write to addresses 32-95.
4. Verify the affected inline write by checking the remote core's received semaphore or control value.

### Fix

Ensure no kernel writes to the L1 region below `MEM_MAILBOX_BASE` on Blackhole.

### Prevention

- On Blackhole, be especially careful with L1 addresses near the base of the address space.
- Enable full NOC sanitization in CI for Blackhole tests.

---

## Hang Cause 3.1.10: BH/Quasar L1 Data Cache Coherence Issues

### Symptom

On BH or Quasar, a kernel reads stale data from L1 despite another core or RISC having written new data. This can cause CB synchronization failures, semaphore misreads, or corrupted buffer contents leading to secondary hangs. The hang is architecture-specific -- the same code works correctly on WH.

### Root Cause

Blackhole and Quasar Tensix cores have an L1 data cache that can serve stale values if not properly invalidated after a remote write modifies L1. Quasar additionally provides an uncached address region:

- **Quasar**: The L1 address space is split into cached (`MEM_L1_BASE` to `MEM_L1_BASE + MEM_L1_SIZE`, 0-4 MB) and uncached (`MEM_L1_UNCACHED_BASE`, 4-8 MB, same physical memory). Mailbox accesses use the uncached base:
  ```c++
  // Quasar mailbox access uses uncached region
  #define GET_MAILBOX_ADDRESS_DEV(x) \
      (&(((mailboxes_t tt_l1_ptr*)(MEM_MAILBOX_BASE + MEM_L1_UNCACHED_BASE))->x))
  ```
- **Blackhole**: Uses `enable_hw_cache_invalidation` runtime option and explicit `invalidate_l1_cache()` calls.

The `invalidate_l1_cache()` call is required in spin-loops that poll L1 values written by remote cores. If this call is missing, the core may:
- Read a stale semaphore value and never see the update, causing a semaphore hang.
- Read a stale CB counter and compute incorrect free space, causing a CB deadlock.
- Read a stale buffer address and issue a NOC transaction to a stale (now-invalid) location.

**Note:** `cb_reserve_back` already includes `invalidate_l1_cache()` in its spin-loop. The risk is in custom polling loops that do not use the standard CB/semaphore API.

### Diagnosis Steps

1. Check if the hang is BH/Quasar-specific. If it works on WH (which has no L1 data cache), suspect a cache coherence issue.
2. Look for custom spin-loops in the kernel that poll L1 values without calling `invalidate_l1_cache()`.
3. Check whether shared data addresses fall in the cached or uncached region (Quasar).
4. On BH, try enabling `enable_hw_cache_invalidation` to see if the hang resolves.

### Fix

Add `invalidate_l1_cache()` at the top of any custom spin-loop that reads a value potentially modified by a remote core. On Quasar, alternatively access shared data through the uncached address region (`MEM_L1_UNCACHED_BASE`).

### Prevention

- On BH and Quasar, always use `invalidate_l1_cache()` in custom polling loops.
- Use the standard blocking primitives (`cb_reserve_back`, `cb_wait_front`, `noc_semaphore_wait`, `noc_semaphore_wait_min`) which already handle cache invalidation correctly.
- Only write custom spin-loops when absolutely necessary.

---

## Hang Cause 3.1.11: Watcher Sanitize State Corruption

### Symptom

The watcher reports `"Watcher unexpected noc debug state on core {X,Y}, reported valid got noc{...} (corrupted noc sanitization state - sanitization memory overwritten)"` or `"corrupted noc sanitization state - unknown failure code"`.

### Root Cause

The watcher's own sanitize state in the mailbox has been overwritten. The `debug_sanitize_addr_msg_t` structure lives inside the `watcher_msg_t` in the mailbox region. If a kernel or NOC transaction corrupts this memory, the watcher reads inconsistent data.

Two specific cases:
1. The `return_code` field reads `DebugSanitizeOK` (sentinel value `DEBUG_SANITIZE_SENTINEL_OK_64 = 0xbadabadabadabada`) but the other fields do not match their sentinel values, indicating partial corruption.
2. The `return_code` field contains an unrecognized value (not in the `debug_sanitize_noc_return_code_enum` range), indicating the field itself was overwritten.

This is a "who watches the watchers" scenario: the watcher cannot identify *what* performed the write because its own diagnostic state is compromised.

### Diagnosis Steps

1. The watcher error message itself is the indicator -- this error means the debugger is compromised.
2. Look for NOC write transactions that could target addresses in the `[MEM_MAILBOX_BASE, MEM_MAILBOX_END)` range.
3. Check for a missing `cb_reserve_back` before a write that could allow a CB to overflow backward into the mailbox region.
4. Check for multicast writes where the L1 offset is in the mailbox range.

### Fix

Identify and fix the corrupting write.

### Prevention

- The mailbox region is protected by the `DebugSanitizeNocAddrMailbox` check for NOC writes, but direct L1 pointer access is not protected. Avoid kernel code that constructs L1 pointers without validation.
- Keep watcher NOC sanitization enabled to catch the writes before they reach the mailbox.

---

## Summary Table

| ID | Hang Cause | Key Indicator | Watcher Return Code | Architecture |
|----|-----------|---------------|--------------------|-|
| 3.1.1 | Address-0 firmware launch corruption | `Watcher found corruption at L1[0]` | N/A (host-side check) | All |
| 3.1.2 | CB write overflow into adjacent regions | `fifo_wr_ptr > fifo_limit` | `DebugSanitizeCBOutOfBounds` (17) | All |
| 3.1.3 | Silent corruption of runtime arguments | Garbage NOC address in sanitize log | `DebugSanitizeNocTargetInvalidXY` (6) | All |
| 3.1.4 | Corrupted CB metadata | Unexpected CB `fifo_size`/`fifo_num_pages` values | None (metadata, not NOC) | All |
| 3.1.5 | RISC-V stack overflow | Watcher `min_free` near 0 | N/A (host-side check) | All (worst on WH TRISC) |
| 3.1.6 | L1 bounds overflow (direct access) | Deliberate hang | `DebugSanitizeL1AddrOverflow` (14) | All |
| 3.1.7 | Ethernet L1 overflow | Deliberate hang | Eth src/dst overflow (15, 16) | All (ETH cores) |
| 3.1.8 | Mailbox region overwrite | `DebugSanitizeNocAddrMailbox` | Return code 12 | All |
| 3.1.9 | BH inline write region corruption | Inline writes carry garbage | Manual inspection | Blackhole only |
| 3.1.10 | BH/Quasar L1 cache coherence | Stale reads in polling loops | None | BH, Quasar |
| 3.1.11 | Watcher sanitize state corruption | Watcher self-check error | Sentinel mismatch | All |

---

**Previous:** [Chapter 3 Index](./index.md)
**Next:** [`02_dram_and_noc_backpressure.md`](./02_dram_and_noc_backpressure.md)
