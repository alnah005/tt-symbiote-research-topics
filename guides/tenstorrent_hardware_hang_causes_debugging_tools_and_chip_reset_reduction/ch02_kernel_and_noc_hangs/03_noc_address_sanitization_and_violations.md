# NOC Address Sanitization and Violations

The NOC (Network-on-Chip) address sanitization layer is the device-side validation system that catches malformed NOC transactions before they cause silent data loss or hardware-level hangs. When enabled (via `WATCHER_ENABLED`), every NOC transaction is validated against a comprehensive set of rules. Violations cause a deliberate hang via `debug_sanitize_post_addr_and_hang`, which records diagnostic data in the sanitize mailbox and then enters a spin loop. When disabled (production builds), the sanitization layer compiles to no-ops, and invalid addresses cause silent failures that eventually manifest as NOC barrier hangs.

This section documents the full validation pipeline, every return code, the deliberate-hang mechanism, the linked transaction validation system, and the debug delay injection mechanism.

**Prerequisites:** [Chapter 1, `02_blocking_primitives_taxonomy.md`](../ch01_anatomy_of_a_hang/02_blocking_primitives_taxonomy.md) (NOC barrier primitives), [Chapter 1, `04_hang_causes_across_architectures.md`](../ch01_anatomy_of_a_hang/04_hang_causes_across_architectures.md) (WH/BH alignment differences). The basic NOC barrier spin-loop code is in Chapter 1; this section focuses on the sanitization checks that precede those barriers.

Reference files: `tt_metal/hw/inc/internal/debug/sanitize.h`

---

## The Validation Pipeline

When a NOC transaction is issued (read, write, atomic, multicast), the sanitization macros call into a validation pipeline that checks the following, in order:

### Step 1: Core Type Resolution

The function `get_core_type(noc_id, x, y, is_virtual_coord)` determines what type of core the NOC address targets. It searches through multiple coordinate spaces:

1. **Physical non-worker cores**: Checks against `core_info_msg_t.non_worker_cores[]` (up to 35 entries covering DRAM, PCIe, and Ethernet cores in physical coordinates).

2. **Virtual non-worker cores** (if `COORDINATE_VIRTUALIZATION_ENABLED`): Checks against `core_info_msg_t.virtual_non_worker_cores[]` (up to 29 entries covering virtualized DRAM, PCIe, and Ethernet cores).

3. **Harvested coordinates** (virtual and physical): Checks against `virtual_harvested_coords[]` and `harvested_coords[]`. If a NOC address targets a harvested row or column, the core type is `AddressableCoreType::HARVESTED`.

4. **Tensix cores** (virtual and physical): Checks if the coordinates fall within the Tensix grid. On Blackhole, Tensix virtual coordinates are not contiguous (requiring full grid bounds), while on Wormhole they use continuous worker grid size.

5. **Unknown**: If no match is found, the core type is `AddressableCoreType::UNKNOWN`, which triggers an `InvalidXY` error.

The possible core types and their validation branches:

| Core Type | Validation Function | Alignment Requirement |
|-----------|-------------------|-----------------------|
| `TENSIX` | `debug_valid_worker_addr` | `NOC_L1_READ/WRITE_ALIGNMENT_BYTES` |
| `DRAM` | `debug_valid_dram_addr` | `NOC_DRAM_READ/WRITE_ALIGNMENT_BYTES` (32B WH, 64B BH/QA) |
| `PCIE` | `debug_valid_pcie_addr` | `NOC_PCIE_READ/WRITE_ALIGNMENT_BYTES` (32B WH, 64B BH/QA) |
| `ETH` | `debug_valid_eth_addr` | `NOC_L1_READ/WRITE_ALIGNMENT_BYTES` |
| `HARVESTED` | Immediate error | N/A |
| `UNKNOWN` | Immediate error | N/A |

> **Danger:** Wormhole uses 32-byte alignment for DRAM and PCIe reads; Blackhole and Quasar use 64-byte alignment. Code that works on WH with 32-byte-aligned addresses will trigger alignment violations on BH if the alignment is not updated.

### Step 2: Address Range Validation

Each core type has its own address range check:

**Worker (Tensix) addresses** -- `debug_valid_worker_addr`:

```c++
// tt_metal/hw/inc/internal/debug/sanitize.h
inline uint16_t debug_valid_worker_addr(uint64_t addr, uint64_t len, bool write) {
    if (addr + len <= addr) return DebugSanitizeNocAddrZeroLength;
    if (addr < MEM_L1_BASE) return DebugSanitizeNocAddrUnderflow;
    if (addr + len > MEM_L1_BASE + MEM_L1_SIZE) return DebugSanitizeNocAddrOverflow;
    if (write && (addr < MEM_MAP_READ_ONLY_END))
        return DebugSanitizeNocAddrMailbox;
    return DebugSanitizeOK;
}
```

**PCIe addresses** -- `debug_valid_pcie_addr`:

```c++
inline uint16_t debug_valid_pcie_addr(uint64_t addr, uint64_t len) {
    if (addr + len <= addr) return DebugSanitizeNocAddrZeroLength;
    if (addr < core_info->noc_pcie_addr_base) return DebugSanitizeNocAddrUnderflow;
    if (addr + len > core_info->noc_pcie_addr_end) return DebugSanitizeNocAddrOverflow;
    return DebugSanitizeOK;
}
```

**DRAM addresses** -- `debug_valid_dram_addr`:

```c++
inline uint16_t debug_valid_dram_addr(uint64_t addr, uint64_t len) {
    if (addr + len <= addr) return DebugSanitizeNocAddrZeroLength;
    if (addr < core_info->noc_dram_addr_base) return DebugSanitizeNocAddrUnderflow;
    if (addr + len > core_info->noc_dram_addr_end) return DebugSanitizeNocAddrOverflow;
    return DebugSanitizeOK;
}
```

**Ethernet L1 addresses** -- `debug_valid_eth_addr`:

```c++
inline uint16_t debug_valid_eth_addr(uint64_t addr, uint64_t len, bool write) {
    if (addr + len <= addr) return DebugSanitizeNocAddrZeroLength;
    if (addr < MEM_ETH_BASE) return DebugSanitizeNocAddrUnderflow;
    if (addr + len > MEM_ETH_BASE + MEM_ETH_SIZE) return DebugSanitizeNocAddrOverflow;
    if (write && (addr < mem_mailbox_end)) return DebugSanitizeNocAddrMailbox;
    return DebugSanitizeOK;
}
```

### Step 3: Multicast Validation

For multicast transactions, additional checks run before the address range validation:

- Start and end coordinates must both be Tensix cores (else `DebugSanitizeNocMulticastNonWorker`)
- Start and end must be in the same coordinate space (else `DebugSanitizeNocMixedVirtualandPhysical`)
- Coordinate ordering depends on NOC ID: NOC0 requires `x_start <= x_end && y_start <= y_end`; NOC1 requires the reverse (else `DebugSanitizeNocMulticastInvalidRange`)

### Step 4: Alignment Validation

After the core type is resolved, the alignment requirement is determined based on core type and direction (read vs. write). The check compares the lower bits of the L1 address and the NOC address:

```c++
if ((worker_addr & alignment_mask) != (noc_addr & alignment_mask))
    -> DebugSanitizeNocAlignment
```

The local L1 address and the remote NOC address must have matching alignment because the NOC DMA engine requires aligned source and destination addresses.

### Step 5: Circular Buffer Bounds Check

If CB sanitization is enabled (`!WATCHER_DISABLE_CB_SANITIZE`), the local L1 address is checked against all configured circular buffers:

```c++
// tt_metal/hw/inc/internal/debug/sanitize.h
inline uint16_t debug_valid_cb_addr(uint32_t l1_addr, uint32_t len) {
    for (uint32_t i = 0; i < NUM_CIRCULAR_BUFFERS; i++) {
        LocalCBInterface& cb = get_local_cb_interface(i);
        if (cb.fifo_size == 0) continue;  // unused CB

        uint32_t cb_start = cb.fifo_limit - cb.fifo_size;
        uint32_t cb_end = cb.fifo_limit;

        if (l1_addr >= cb_start && l1_addr < cb_end) {
            if (static_cast<uint64_t>(l1_addr) + len > cb_end)
                return DebugSanitizeCBOutOfBounds;
            return DebugSanitizeOK;
        }
    }
    return DebugSanitizeOK;  // Not inside any CB
}
```

This iterates all 32 (or 64 on BH) circular buffers. Unused CBs are skipped via `fifo_size == 0`, which is why the BRISC firmware explicitly zeroes all CB interfaces at kernel start when CB sanitization is enabled.

### Step 6: Register Address Exception

Addresses targeting NOC overlay registers or the soft reset register bypass the L1/ETH range checks:

```c++
inline bool debug_valid_reg_addr(uint64_t addr, uint64_t len) {
    return (((addr >= NOC_OVERLAY_START_ADDR) &&
             (addr < NOC_OVERLAY_START_ADDR + NOC_STREAM_REG_SPACE_SIZE * NOC_NUM_STREAMS)) ||
            (addr == RISCV_DEBUG_REG_SOFT_RESET_0)) &&
           (len == 4);
}
```

Register accesses must be exactly 4 bytes and must target valid register ranges.

---

## All `DebugSanitize` Return Codes

```c++
enum debug_sanitize_noc_return_code_enum {
    DebugSanitizeOK                          = 2,  // No error
    DebugSanitizeNocAddrUnderflow            = 3,  // Address below valid range
    DebugSanitizeNocAddrOverflow             = 4,  // Address + length exceeds valid range
    DebugSanitizeNocAddrZeroLength           = 5,  // Zero-length or wraparound-length transfer
    DebugSanitizeNocTargetInvalidXY          = 6,  // NOC coordinates don't map to any core
    DebugSanitizeNocMulticastNonWorker       = 7,  // Multicast target is not a Tensix core
    DebugSanitizeNocMulticastInvalidRange    = 8,  // Multicast start/end ordering violation
    DebugSanitizeNocAlignment                = 9,  // L1 and NOC addr alignment mismatch
    DebugSanitizeNocMixedVirtualandPhysical  = 10, // Multicast start in virtual, end in physical
    DebugSanitizeInlineWriteDramUnsupported  = 11, // Inline write targeting DRAM (unsupported)
    DebugSanitizeNocAddrMailbox              = 12, // Write to read-only mailbox region
    DebugSanitizeNocLinkedTransactionViolation = 13, // Unicast during linked multicast pending
    DebugSanitizeL1AddrOverflow              = 14, // Local L1 access out of bounds
    DebugSanitizeEthSrcL1AddrOverflow        = 15, // Ethernet source L1 address out of bounds
    DebugSanitizeEthDestL1AddrOverflow       = 16, // Ethernet destination L1 address out of bounds
    DebugSanitizeCBOutOfBounds               = 17, // NOC transfer overflows a CB boundary
};
```

Note: values 0 and 1 are deliberately unused because they are common stray-write values. The "OK" value starts at 2 to avoid false negatives from uninitialized memory.

---

## Hang Cause 2.3.1: `debug_sanitize_post_addr_and_hang` Deliberate Hang

### Symptom

A core enters `while(1)` after a NOC sanitization check fails. The watcher can read the sanitize mailbox to determine the exact violation. On Tensix cores, the core spins forever. On ERISC cores, the core exits to base firmware instead.

### Root Cause

When any validation check fails, the violation details are recorded and the core is halted:

```c++
// tt_metal/hw/inc/internal/debug/sanitize.h
void debug_sanitize_post_addr_and_hang(
    uint8_t noc_id, uint64_t noc_addr, uint32_t l1_addr, uint32_t len,
    debug_sanitize_noc_cast_t multicast, debug_sanitize_noc_dir_t dir,
    debug_sanitize_noc_which_core_t which_core, uint16_t return_code) {

    if (return_code == DebugSanitizeOK) return;

    debug_sanitize_addr_msg_t tt_l1_ptr* v = *GET_MAILBOX_ADDRESS_DEV(watcher.sanitize);

    if (v[noc_id].return_code == DebugSanitizeOK) {
        v[noc_id].noc_addr = noc_addr;
        v[noc_id].l1_addr = l1_addr;
        v[noc_id].len = len;
        v[noc_id].which_risc = internal_::get_hw_thread_idx();
        v[noc_id].is_multicast = (multicast == DEBUG_SANITIZE_NOC_MULTICAST);
        v[noc_id].is_write = (dir == DEBUG_SANITIZE_NOC_WRITE);
        v[noc_id].is_target = (which_core == DEBUG_SANITIZE_NOC_TARGET);
        v[noc_id].return_code = return_code;
    }

    // ERISC special handling: exit to base firmware
#if defined(COMPILE_FOR_ERISC)
    go_message_ptr->signal = RUN_MSG_DONE;
    internal_::disable_erisc_app();
    erisc_exit();
#endif

    while (1) { ; }
}
```

Key details:
- The `return_code` is written **last** to the mailbox, preventing a race with the host reader.
- Only one violation per NOC is recorded -- subsequent violations are ignored to preserve the first error.
- ERISC cores mark themselves as done and exit to base firmware, preventing permanent ethernet link loss. The `while(1)` at the end is only reached on Tensix cores.

### Diagnosis Steps

1. Read the sanitize mailbox: `watcher.sanitize[0]` and `watcher.sanitize[1]` (one per NOC).
2. Check `return_code` -- if it is not `DebugSanitizeOK` (2), a violation was recorded.
3. The `which_risc` field identifies the triggering processor.
4. The `noc_addr`, `l1_addr`, `len`, `is_multicast`, `is_write`, and `is_target` fields describe the transaction.
5. Use the `return_code` enum to determine the class of violation.

### Fix

Depends on the specific return code -- see the individual scenarios below.

### Prevention

- Enable watcher during development. The sanitization layer is the primary defense against silent NOC failures.
- When watcher is disabled, all `DEBUG_SANITIZE_*` macros compile to no-ops, and bad addresses cause silent failures.

---

## Hang Cause 2.3.2: Invalid Coordinates (`NocTargetInvalidXY`)

### Symptom

Sanitize mailbox shows `return_code = 6`. Without sanitization, the NOC transaction targets a nonexistent endpoint, the response never arrives, and the core hangs at `NRBW` or `NWBW`.

### Root Cause

The NOC address encodes XY coordinates that do not match any known core. Common causes: using physical coordinates when compiled for virtual (or vice versa), out-of-range computed coordinates, targeting a harvested core.

### Diagnosis Steps

1. Extract X and Y from the NOC address in the sanitize mailbox.
2. Compare against the core coordinate tables in `core_info_msg_t`.

### Fix

Correct the NOC address computation. Use `get_noc_addr()` and similar helper functions rather than manually constructing NOC addresses.

### Prevention

Always use the API-provided address construction functions. Never hardcode NOC coordinates.

---

## Hang Cause 2.3.3: Multicast to Non-Worker Cores (`NocMulticastNonWorker`)

### Symptom

Sanitize mailbox shows `return_code = 7`. A multicast transaction targets a rectangle that includes non-Tensix cores. Without sanitization, the multicast partially delivers to reachable Tensix cores but the `num_dests` count does not match, causing a write barrier hang (`NWBW`).

### Root Cause

NOC multicast is only valid for Tensix worker cores. If either the start or end coordinate of the multicast rectangle resolves to a non-Tensix core type (DRAM, PCIe, Ethernet, Harvested, or Unknown), this error fires. Common causes include using the full chip grid instead of the worker grid, or including DRAM/PCIe rows in the multicast rectangle.

### Diagnosis Steps

1. Read the sanitize mailbox: `return_code = 7` confirms this violation.
2. Extract the start and end coordinates from the NOC address in the mailbox.
3. Cross-reference against the worker grid in `core_info_msg_t` to identify which coordinate resolves to a non-worker core.
4. Check whether the multicast address was constructed manually or via `get_noc_multicast_addr()`.

### Fix

Adjust the multicast rectangle to only include Tensix cores. Use `get_noc_multicast_addr()` with correct grid coordinates.

### Prevention

- Always use `get_noc_multicast_addr()` rather than manually constructing multicast NOC addresses.
- Validate that the start and end coordinates both fall within the Tensix worker grid before issuing multicast transactions.

---

## Hang Cause 2.3.4: Multicast Range Ordering (`NocMulticastInvalidRange`)

### Symptom

Sanitize mailbox shows `return_code = 8`. The multicast start and end coordinates are in the wrong order for the NOC being used. Without sanitization, the hardware interprets the reversed rectangle as undefined behavior, potentially delivering to wrong cores or not delivering at all.

### Root Cause

NOC0 requires `x_start <= x_end && y_start <= y_end`. NOC1 requires the reverse (`x_end <= x_start && y_end <= y_start`). In virtual coordinate space, the NOC0/NOC1 reversal still applies. This typically occurs when manually constructing multicast addresses without accounting for the NOC direction, or when the same multicast address is reused across NOC0 and NOC1.

### Diagnosis Steps

1. Check which NOC was used (from `noc_id` in the sanitize mailbox).
2. Extract start and end coordinates from the NOC address.
3. Verify ordering matches the NOC convention: NOC0 requires ascending, NOC1 requires descending.

### Fix

Swap the start and end coordinates, or switch to the correct NOC. Use the multicast address construction helpers that handle NOC0/NOC1 differences automatically.

### Prevention

- Use `get_noc_multicast_addr()` which automatically handles NOC0/NOC1 coordinate ordering.
- When writing NOC-agnostic code, parameterize the start/end based on the NOC index rather than hardcoding coordinates.

---

## Hang Cause 2.3.5: Mixed Virtual and Physical Coordinates (`NocMixedVirtualandPhysical`)

### Symptom

Sanitize mailbox shows `return_code = 10`. A multicast transaction has start coordinates in one coordinate space and end coordinates in the other. Without sanitization, the multicast targets an unpredictable set of cores.

### Root Cause

The sanitizer resolves both start and end coordinates independently. If one resolves in the virtual lookup and the other in the physical lookup, the `is_virtual_coord` flags differ. This typically happens when code mixes coordinate APIs -- for example, using a virtual helper for the start coordinate and a physical constant for the end.

### Diagnosis Steps

1. Read the sanitize mailbox: `return_code = 10` confirms this violation.
2. Extract start and end coordinates from the NOC address.
3. Determine which coordinate space each resolves in by checking against `core_info_msg_t` virtual vs. physical tables.
4. Trace the source code to find where each coordinate was constructed.

### Fix

Use consistent coordinate APIs. Do not mix `NOC_0_X` (virtual) with `NOC_0_X_PHYS_COORD` (physical). Pick one coordinate space (virtual is preferred for portability) and use it consistently.

### Prevention

- Standardize on virtual coordinates throughout kernel code. Virtual coordinates are portable across harvested configurations.
- When importing coordinates from host-side code, ensure they are converted to the same coordinate space as the device kernel expects.

---

## Hang Cause 2.3.6: Inline Write to DRAM (`InlineWriteDramUnsupported`)

### Symptom

Sanitize mailbox shows `return_code = 11`. An `noc_inline_dw_write` targeted a DRAM address. Without sanitization, the hardware behavior is undefined -- the write may silently fail or cause a NOC stall.

### Root Cause

The `noc_inline_dw_write` API does not support DRAM addresses. The NOC inline write mechanism places data directly in the NOC command packet rather than performing a DMA from L1 to the destination. This mechanism is only supported for L1-to-L1 and L1-to-PCIe transfers. The validation function `debug_throw_on_dram_addr` explicitly checks for this.

### Diagnosis Steps

1. Read the sanitize mailbox: `return_code = 11` confirms this violation.
2. Check the target NOC address -- resolve the coordinates to determine if they map to a DRAM core.
3. Trace the kernel source to find the `noc_inline_dw_write` call and the address computation.

### Fix

Use `noc_async_write` for DRAM targets. `noc_inline_dw_write` only supports Tensix L1 and PCIe targets.

### Prevention

- Use `noc_async_write` as the default for all DRAM transfers.
- Reserve `noc_inline_dw_write` for small L1-to-L1 transfers (e.g., semaphore increments) where the 4-byte inline payload is sufficient.

---

## Hang Cause 2.3.7: Write to Read-Only Mailbox Region (`NocAddrMailbox`)

### Symptom

Sanitize mailbox shows `return_code = 12`. A NOC write targets the firmware mailbox region below `MEM_MAP_READ_ONLY_END`.

### Root Cause

The region below `MEM_MAP_READ_ONLY_END` on Tensix cores contains firmware data structures (launch messages, go signals, watcher state). A stray NOC write to this region would corrupt firmware state. Dispatch kernels are exempted from this check.

### Diagnosis Steps

1. Check the `l1_addr` in the sanitize mailbox.
2. Common cause: using an uninitialized buffer address variable that happens to be 0.

### Fix

Adjust the destination address to target user-accessible L1 memory. Use the `l1_unreserved_start` field from `core_info_msg_t` to determine the safe starting address.

### Prevention

Initialize all buffer address variables before use. Use `get_semaphore_addr()` for semaphore addresses rather than computing them manually.

---

## Hang Cause 2.3.8: Linked Transaction Violation (`NocLinkedTransactionViolation`)

### Symptom

Sanitize mailbox shows `return_code = 13`. The core attempted to issue a unicast NOC transaction while a linked multicast was pending.

### Root Cause

The NOC hardware supports "linked" multicast transactions, where multiple multicasts share a path reservation. Issuing a unicast while a linked multicast is pending corrupts the path reservation state, causing the multicast to deadlock.

```c++
// tt_metal/hw/inc/internal/debug/sanitize.h
inline void debug_sanitize_check_linked_transactions(
    uint8_t noc_id, uint64_t noc_addr, uint32_t l1_addr, uint32_t noc_len,
    debug_sanitize_noc_cast_t multicast, debug_sanitize_noc_dir_t dir) {

    if (multicast == DEBUG_SANITIZE_NOC_UNICAST) {
        auto* watcher_msg = GET_MAILBOX_ADDRESS_DEV(watcher);
        if (watcher_msg->noc_linked_status[noc_id]) {
            debug_sanitize_post_addr_and_hang(...,
                DebugSanitizeNocLinkedTransactionViolation);
        }
    }
}
```

> **Danger:** This check is only active when linked transaction validation is opted in via `TT_METAL_WATCHER_ENABLE_NOC_SANITIZE_LINKED_TRANSACTION`. Without it, a linked transaction violation causes a hardware-level NOC deadlock that requires a chip reset to recover.

### Diagnosis Steps

1. Read the sanitize mailbox: `return_code = 13` confirms this violation.
2. Check the `noc_linked_status` in the watcher mailbox to confirm a linked multicast was in progress.
3. Identify the unicast that triggered the violation from the `noc_addr` in the mailbox.
4. Trace the kernel to determine why a unicast was issued before the linked multicast chain completed.

### Fix

Always complete or unlink a linked multicast chain before issuing unicast transactions on the same NOC. The dispatch kernel (`cq_dispatch.cpp`) contains the primary example of correct linked transaction handling.

### Prevention

- Enable `TT_METAL_WATCHER_ENABLE_NOC_SANITIZE_LINKED_TRANSACTION` during development to catch linked transaction violations before they cause hardware deadlocks.
- Structure dispatch code to complete an entire linked multicast sequence before issuing any unicast operations on the same NOC.
- Without the opt-in check, this violation causes a hardware NOC deadlock that requires a chip reset.

---

## Hang Cause 2.3.9: Local L1 Address Overflow (`L1AddrOverflow`)

### Symptom

Sanitize mailbox shows `return_code = 14`. A local L1 memory access exceeds the valid L1 range.

### Root Cause

```c++
void debug_sanitize_l1_access(uint64_t addr, uint32_t len) {
    constexpr uint64_t l1_overflow_addr = MEM_L1_SIZE;  // or MEM_ETH_SIZE for ERISC
    if (addr + len <= addr || addr + len > l1_overflow_addr) {
        debug_sanitize_post_addr_and_hang(..., DebugSanitizeL1AddrOverflow);
    }
}
```

### Diagnosis Steps

1. Read the sanitize mailbox: `return_code = 14` confirms this violation.
2. Check `l1_addr` and `len` from the mailbox -- the sum exceeds `MEM_L1_SIZE` (or `MEM_ETH_SIZE` for ERISC).
3. For ERISC cores, note that the L1 size is smaller than Tensix L1 -- buffers that fit on Tensix may overflow on ERISC.
4. Trace the buffer allocation to determine if the address or length is incorrect.

### Fix

Reduce the buffer size or fix the address calculation. Validate buffer sizes at allocation time.

### Prevention

- Check buffer allocations against L1 size limits at host-side setup time before launching kernels.
- Use `MEM_L1_SIZE` (Tensix) or `MEM_ETH_SIZE` (ERISC) as upper bounds in buffer allocation logic.
- Account for all L1 consumers (CBs, semaphores, kernel locals, firmware) when computing available L1 space.

---

## Hang Cause 2.3.10: CB Out of Bounds (`CBOutOfBounds`)

### Symptom

Sanitize mailbox shows `return_code = 17`. A NOC transfer's local L1 address falls within a circular buffer's region, but the transfer extends beyond the CB's boundary. Without sanitization, this overwrites adjacent L1 memory, corrupting other CBs or firmware data structures.

### Root Cause

The `debug_valid_cb_addr` function checks every configured CB. If the L1 address is inside a CB but `l1_addr + len > fifo_limit`, the transfer would read or write past the end of the CB. Common causes: tile size mismatch between NOC transfer size and CB page size, incorrect `cb_pages` count leading to undersized CB allocation, or computing addresses beyond the CB's valid region.

### Diagnosis Steps

1. Read the sanitize mailbox: `return_code = 17` confirms this violation.
2. Extract `l1_addr` and `len` from the mailbox.
3. Identify which CB contains `l1_addr` by comparing against `fifo_limit - fifo_size` through `fifo_limit` for each CB.
4. Calculate the overflow: `(l1_addr + len) - fifo_limit` gives the number of bytes overflowing.
5. Compare the NOC transfer size (`len`) against the CB page size to check for tile size mismatches.

### Fix

Reduce the transfer size or increase the CB size. Ensure that NOC read/write sizes match the tile sizes configured for the CB.

### Prevention

- Validate that the CB page size matches the expected tile size at host-side setup time.
- Use `cb_interface.fifo_size` and `cb_interface.fifo_limit` to bounds-check NOC transfers when writing custom data movement kernels.
- See also Hang Cause 2.2.4 (CB Size Divisibility) for related CB sizing constraints.

---

## Hang Cause 2.3.11: Alignment Violation (`NocAlignment`)

### Symptom

Sanitize mailbox shows `return_code = 9`. The local L1 address and the remote NOC address have mismatched alignment. Without sanitization, the NOC transaction silently fails and the subsequent barrier hangs at `NRBW` or `NWBW`.

### Root Cause

The NOC DMA engine requires that source and destination addresses have the same alignment within their respective alignment granularity. The alignment check compares lower bits: `(worker_addr & alignment_mask) != (noc_addr & alignment_mask)`. Wormhole requires 32-byte alignment for DRAM/PCIe reads; Blackhole and Quasar require 64-byte alignment.

> **Danger:** Code that works on Wormhole with 32-byte-aligned DRAM addresses will fail on Blackhole if the addresses are not also 64-byte aligned. This is a common porting issue.

### Diagnosis Steps

1. Read the sanitize mailbox: `return_code = 9` confirms this violation.
2. Extract `l1_addr` and `noc_addr` from the mailbox.
3. Determine the required alignment from the core type (see the alignment table in Step 1 of the Validation Pipeline above).
4. Check the lower bits of both addresses against the alignment mask for the target architecture.

### Fix

Align both source and destination addresses to the required granularity. Use `ALIGN()` macros when allocating L1 buffers and computing DRAM offsets.

### Prevention

- Use architecture-aware alignment constants (`NOC_L1_READ_ALIGNMENT_BYTES`, `NOC_DRAM_READ_ALIGNMENT_BYTES`, etc.) rather than hardcoded alignment values.
- When porting WH kernels to BH, audit all DRAM/PCIe address calculations for 64-byte alignment compliance.
- Allocate L1 buffers with sufficient alignment padding for the target architecture.

---

## When Sanitization Is Disabled

In production builds without `WATCHER_ENABLED`, all `DEBUG_SANITIZE_*` macros compile to no-ops:

```c++
#define DEBUG_SANITIZE_NOC_ADDR(noc_id, a, l) LOG_LEN(l)
#define DEBUG_SANITIZE_NOC_TRANSACTION(noc_id, noc_a, worker_a, l, multicast, dir) LOG_LEN(l)
```

This means:
- Invalid NOC addresses are sent to the hardware
- The hardware may silently drop the transaction, send a response that never arrives, or corrupt memory at the wrong address
- The RISC-V core waiting for the response hangs at the subsequent barrier (`NRBW`, `NWBW`)
- The hang has **no diagnostic data** in the sanitize mailbox

> **Danger:** All sanitization compiles to no-ops without watcher. In production builds, every violation type becomes a silent failure. This is why enabling watcher during development is essential -- it transforms silent hangs into diagnosed failures.

---

## The Debug Delay Mechanism

The sanitization layer includes an optional delay injection mechanism (`WATCHER_DEBUG_DELAY`):

```c++
// tt_metal/hw/inc/internal/debug/sanitize.h
inline void debug_insert_delay(uint8_t transaction_type) {
    debug_insert_delays_msg_t tt_l1_ptr* v =
        GET_MAILBOX_ADDRESS_DEV(watcher.debug_insert_delays);

    bool delay = false;
    switch (transaction_type) {
        case TransactionRead:
            delay = (v[0].read_delay_processor_mask & (1u << get_hw_thread_idx())) != 0;
            break;
        case TransactionWrite:
            delay = (v[0].write_delay_processor_mask & (1u << get_hw_thread_idx())) != 0;
            break;
        case TransactionAtomic:
            delay = (v[0].atomic_delay_processor_mask & (1u << get_hw_thread_idx())) != 0;
            break;
    }
    if (delay) {
        riscv_wait(WATCHER_DEBUG_DELAY);
        v[0].feedback |= (1 << transaction_type);
    }
}
```

This is invoked after every sanitized NOC transaction. The host configures which cores and transaction types receive delays via the `debug_insert_delays_msg_t` mailbox. The `feedback` field confirms that delays are being applied.

**Use case:** Intermittent race-condition hangs can be made reproducible by slowing down one side of a semaphore protocol. Configure delays on specific cores and transaction types via `TT_METAL_READ_DEBUG_DELAY_CORES`, `TT_METAL_WRITE_DEBUG_DELAY_CORES`, and `TT_METAL_ATOMIC_DEBUG_DELAY_CORES` environment variables (see Chapter 6 for full details).

---

## Performance Impact and Configuration

Sanitization adds overhead to every NOC transaction. The following environment variables allow selective control:

| Environment Variable | Effect |
|---------------------|--------|
| `TT_METAL_WATCHER=<ms>` | Enable watcher with polling interval |
| `TT_METAL_WATCHER_DISABLE_NOC_SANITIZE` | Disable NOC address sanitization |
| `TT_METAL_WATCHER_DISABLE_CB_SANITIZE` | Disable CB bounds checking |
| `TT_METAL_WATCHER_DISABLE_SANITIZE_READ_ONLY_L1` | Disable mailbox write protection |
| `TT_METAL_WATCHER_DISABLE_SANITIZE_WRITE_ONLY_L1` | Disable write-only L1 sanitization |
| `TT_METAL_WATCHER_ENABLE_NOC_SANITIZE_LINKED_TRANSACTION` | Enable linked transaction validation (opt-in) |
| `TT_METAL_WATCHER_NOINLINE` | Disable DMA ops in watcher for size savings |

---

## Summary: With vs. Without Watcher

| Violation | With Watcher | Without Watcher |
|-----------|-------------|-----------------|
| Invalid XY coordinates | Sanitize mailbox + deliberate hang | Transaction routed to non-existent endpoint; barrier hangs silently |
| Address overflow/underflow | Sanitize mailbox + deliberate hang | Undefined behavior; may corrupt remote L1, may drop silently |
| Alignment mismatch | Sanitize mailbox + deliberate hang | Transaction silently fails; barrier hangs |
| Multicast to non-worker | Sanitize mailbox + deliberate hang | Partial delivery; `num_dests` mismatch causes barrier hang |
| Multicast range ordering | Sanitize mailbox + deliberate hang | Undefined multicast behavior |
| Mixed virtual/physical | Sanitize mailbox + deliberate hang | Multicast to wrong cores |
| Inline write to DRAM | Sanitize mailbox + deliberate hang | Undefined behavior |
| Linked transaction violation | Sanitize mailbox + deliberate hang | NOC hardware deadlock; requires chip reset |
| Write to mailbox | Sanitize mailbox + deliberate hang | Firmware state corruption; unpredictable behavior |
| L1 address overflow | Sanitize mailbox + deliberate hang | Silent memory corruption |
| CB out-of-bounds | Sanitize mailbox + deliberate hang | Silent memory corruption; secondary hang later |

> **Tip:** The sanitize mailbox records the **first** violation per NOC instance. If multiple violations occur, only the first is visible. In complex operations, the first violation may be a consequence of prior data corruption. When debugging, also check for upstream corruption.

---

**Previous:** [`02_circular_buffer_deadlocks.md`](./02_circular_buffer_deadlocks.md) | **Next:** [`04_noc_barrier_and_semaphore_hangs.md`](./04_noc_barrier_and_semaphore_hangs.md)
