# 3.3 Alignment and Tile Size Mismatches

NOC DMA transfers on Tenstorrent hardware have strict alignment requirements that vary by architecture, memory target type (L1, DRAM, PCIe), and transaction direction (read vs. write). When these requirements are violated, the hardware behavior is architecture-dependent: some violations cause silent stalls (the NOC transaction is accepted but the response never arrives), others cause partial data delivery, and some are caught by the watcher's sanitization layer, which triggers a deliberate hang with `DebugSanitizeNocAlignment` (return code 9). Tile size mismatches between cooperating kernels create a related class of bugs where address calculations drift out of sync, eventually producing either alignment violations or out-of-bounds accesses.

Reference files: `tt_metal/hw/inc/internal/debug/sanitize.h` (lines 468-515), `tt_metal/hw/inc/internal/tt-1xx/wormhole/noc/noc_parameters.h`, `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h`, `tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h`

---

## NOC Alignment Requirements by Architecture

The NOC alignment requirements are defined in architecture-specific `noc_parameters.h` files. The alignment applies to both the local L1 address and the remote NOC address -- both must satisfy the target-specific constraint, and their low bits must match (see the cross-check section below).

### Wormhole (WH) Alignment Constants

```c++
// tt_metal/hw/inc/internal/tt-1xx/wormhole/noc/noc_parameters.h
#define NOC_L1_READ_ALIGNMENT_BYTES    16
#define NOC_L1_WRITE_ALIGNMENT_BYTES   16
#define NOC_PCIE_READ_ALIGNMENT_BYTES  32
#define NOC_PCIE_WRITE_ALIGNMENT_BYTES 16
#define NOC_DRAM_READ_ALIGNMENT_BYTES  32
#define NOC_DRAM_WRITE_ALIGNMENT_BYTES 16
```

### Blackhole (BH) Alignment Constants

```c++
// tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h
#define NOC_L1_READ_ALIGNMENT_BYTES    16
#define NOC_L1_WRITE_ALIGNMENT_BYTES   16
#define NOC_PCIE_READ_ALIGNMENT_BYTES  64   // stricter than WH
#define NOC_PCIE_WRITE_ALIGNMENT_BYTES 16
#define NOC_DRAM_READ_ALIGNMENT_BYTES  64   // stricter than WH
#define NOC_DRAM_WRITE_ALIGNMENT_BYTES 16
```

### Quasar Alignment Constants

```c++
// tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h
#define NOC_L1_READ_ALIGNMENT_BYTES    16
#define NOC_L1_WRITE_ALIGNMENT_BYTES   16
#define NOC_PCIE_READ_ALIGNMENT_BYTES  64
#define NOC_PCIE_WRITE_ALIGNMENT_BYTES 16
#define NOC_DRAM_READ_ALIGNMENT_BYTES  64
#define NOC_DRAM_WRITE_ALIGNMENT_BYTES 16
```

### Derived Alignment Macros

The maximum alignment for each memory type is computed as the max of read and write alignment. These are the macros used for buffer allocation:

```c++
#define L1_ALIGNMENT   (max(NOC_L1_READ_ALIGNMENT_BYTES, NOC_L1_WRITE_ALIGNMENT_BYTES))    // 16 all archs
#define PCIE_ALIGNMENT (max(NOC_PCIE_READ_ALIGNMENT_BYTES, NOC_PCIE_WRITE_ALIGNMENT_BYTES)) // 32 WH, 64 BH/Q
#define DRAM_ALIGNMENT (max(NOC_DRAM_READ_ALIGNMENT_BYTES, NOC_DRAM_WRITE_ALIGNMENT_BYTES)) // 32 WH, 64 BH/Q
```

### Alignment Rules Summary

| Transfer Type | Read Alignment | Write Alignment | Max (Both Dirs) |
|---|---|---|---|
| L1-to-L1 | 16 B (all archs) | 16 B (all archs) | 16 B |
| L1-to-DRAM / DRAM-to-L1 | 32 B (WH) / 64 B (BH, Q) | 16 B (all archs) | 32 B (WH) / 64 B (BH, Q) |
| L1-to-PCIe / PCIe-to-L1 | 32 B (WH) / 64 B (BH, Q) | 16 B (all archs) | 32 B (WH) / 64 B (BH, Q) |

**Key observations:**
1. **Read alignment is always >= write alignment** for DRAM and PCIe. Code that works for writes may fail for reads if the address is only 16-byte aligned but 32-byte (WH) or 64-byte (BH/Quasar) alignment is required.
2. **BH/Quasar require 64-byte DRAM/PCIe read alignment**, which is double the WH requirement. Code that runs correctly on WH may silently hang on BH.
3. **L1 alignment is 16 bytes for both reads and writes** across all architectures -- the most lenient constraint.

---

## The Alignment Cross-Check in `debug_sanitize_noc_and_worker_addr`

The sanitization layer performs a critical cross-check: the local L1 address and the remote NOC address must have **matching low bits** at the alignment granularity of the target memory type. This is more nuanced than simply requiring both addresses to be independently aligned -- the *relative* alignment must match because the NOC DMA engine uses the low bits to position data within the aligned transfer block.

```c++
// tt_metal/hw/inc/internal/debug/sanitize.h, lines 468-515
void debug_sanitize_noc_and_worker_addr(
    uint8_t noc_id, uint64_t noc_addr, uint32_t worker_addr, uint32_t len,
    debug_sanitize_noc_cast_t multicast, debug_sanitize_noc_dir_t dir, bool check_linked) {

    // Step 1: Validate NOC address and get target-specific alignment mask
    uint32_t alignment_mask = debug_sanitize_noc_addr(
        noc_id, noc_addr, worker_addr, len, multicast, dir, check_linked);

    // Step 2: Validate local L1 address bounds
    if (!debug_valid_reg_addr(worker_addr, len)) {
        uint16_t return_code = debug_valid_worker_addr(worker_addr, len, dir == DEBUG_SANITIZE_NOC_READ);
        debug_sanitize_post_addr_and_hang(..., return_code);

        // Step 3: ALIGNMENT CROSS-CHECK
        // L1 address lower bits must match NOC address lower bits within the alignment mask
        if ((worker_addr & alignment_mask) != (noc_addr & alignment_mask)) {
            debug_sanitize_post_addr_and_hang(..., DebugSanitizeNocAlignment);
        }
    }

    // Step 4: CB bounds check (if CB sanitization is enabled)
    debug_sanitize_post_addr_and_hang(..., debug_valid_cb_addr(worker_addr, len));
}
```

The `alignment_mask` is determined by the target core type and transfer direction:

```c++
// From debug_sanitize_noc_addr():
uint32_t alignment_mask =
    (dir == DEBUG_SANITIZE_NOC_READ
        ? NOC_L1_READ_ALIGNMENT_BYTES
        : NOC_L1_WRITE_ALIGNMENT_BYTES) - 1;  // Default: L1

if (core_type == AddressableCoreType::PCIE) {
    alignment_mask = (dir == DEBUG_SANITIZE_NOC_READ
        ? NOC_PCIE_READ_ALIGNMENT_BYTES
        : NOC_PCIE_WRITE_ALIGNMENT_BYTES) - 1;
} else if (core_type == AddressableCoreType::DRAM) {
    alignment_mask = (dir == DEBUG_SANITIZE_NOC_READ
        ? NOC_DRAM_READ_ALIGNMENT_BYTES
        : NOC_DRAM_WRITE_ALIGNMENT_BYTES) - 1;
}
```

Concrete alignment mask values:
- **L1 target**: `0xF` (15) on all architectures
- **DRAM read target**: `0x1F` (WH) or `0x3F` (BH/Q)
- **DRAM write target**: `0xF` (15) on all architectures
- **PCIe read target**: `0x1F` (WH) or `0x3F` (BH/Q)

**Example:** For a DRAM read on BH, the mask is `0x3F`. An L1 address of `0x00020010` has low 6 bits `0x10`, while a DRAM address of `0x00001040` has low 6 bits `0x00`. Since `0x10 != 0x00`, the check fails with `DebugSanitizeNocAlignment`, even though both addresses are individually 16-byte aligned.

---

## Hang Cause 3.3.1: Misaligned DRAM Read (L1 Address Not Aligned)

### Symptom

With watcher: `DebugSanitizeNocAlignment` violation followed by deliberate hang. Without watcher: the core hangs at `NRBW` -- the NOC read transaction is accepted by the NOC command buffer but the response never arrives because the DRAM controller discards or stalls the misaligned request. The hang is reproducible with specific buffer addresses but not others.

### Root Cause

A NOC read from DRAM requires the local L1 address to be aligned to the DRAM read alignment boundary. The most common cause is computing an L1 buffer address by adding a non-aligned offset:

**Buggy code:**
```c++
// L1 buffer at a 16-byte-aligned address, but DRAM read requires 32-byte alignment (WH)
uint32_t base_addr = get_write_ptr(cb_out);  // guaranteed 16-byte aligned by CB
uint32_t offset = header_size;                // e.g., 20 bytes (not aligned)
uint32_t l1_addr = base_addr + offset;       // now misaligned

uint64_t dram_addr = get_noc_addr(dram_x, dram_y, dram_offset);
noc_async_read(dram_addr, l1_addr, tile_size);   // BUG: l1_addr not 32-byte aligned
noc_async_read_barrier();                         // HANGS on WH, HANGS on BH (64-byte)
```

**Corrected code:**
```c++
// Align the DRAM read offset and read aligned, then extract the portion needed
uint32_t aligned_offset = dram_offset & ~(NOC_DRAM_READ_ALIGNMENT_BYTES - 1);  // Round down
uint32_t extra_prefix = dram_offset - aligned_offset;  // Bytes before the data we want
uint32_t aligned_read_size = ((extra_prefix + tile_size + NOC_DRAM_READ_ALIGNMENT_BYTES - 1)
                              & ~(NOC_DRAM_READ_ALIGNMENT_BYTES - 1));
uint32_t l1_addr = (base_addr + DRAM_ALIGNMENT - 1) & ~(DRAM_ALIGNMENT - 1);

uint64_t dram_addr = get_noc_addr(dram_x, dram_y, aligned_offset);
noc_async_read(dram_addr, l1_addr, aligned_read_size);
noc_async_read_barrier();
// Use data starting at l1_addr + extra_prefix
```

### Diagnosis Steps

1. If the watcher is enabled, the `DebugSanitizeNocAlignment` message includes both the L1 and NOC addresses. Extract the low bits of both.
2. Check `target_addr & (NOC_DRAM_READ_ALIGNMENT_BYTES - 1)` -- any non-zero value indicates misalignment.
3. Check the L1 address alignment as well -- the cross-check requires congruent low bits.
4. If watcher is disabled, compare `noc_reads_num_issued` vs. `noc_reads_acked`. A gap indicates a read whose response never arrived.

### Fix

Ensure both addresses satisfy the alignment requirement for the target memory type. Use the architecture-specific `DRAM_ALIGNMENT` or `PCIE_ALIGNMENT` macros instead of hardcoded values.

### Prevention

- Always use aligned allocations for L1 buffers used as DMA source/destination.
- Use `DRAM_ALIGNMENT` when computing buffer addresses for DRAM-related operations.
- Enable watcher NOC sanitization in all test environments.
- Always test on Blackhole when developing code initially written for Wormhole, as the stricter 64-byte DRAM read alignment catches many latent bugs.

---

## Hang Cause 3.3.2: Architecture-Dependent Alignment Failure (WH Code on BH)

### Symptom

Code that runs correctly on WH hangs on BH (or Quasar). With watcher enabled, `DebugSanitizeNocAlignment` is reported. The failing addresses are 32-byte aligned (correct for WH DRAM reads) but not 64-byte aligned (required for BH DRAM reads).

### Root Cause

BH and Quasar require 64-byte DRAM read alignment vs WH's 32-byte requirement. Code that hardcodes `32` as the alignment value will fail on BH:

**Buggy code:**
```c++
// Hardcoded WH alignment -- fails on BH
constexpr uint32_t ALIGN = 32;
uint32_t l1_addr = (raw_addr + ALIGN - 1) & ~(ALIGN - 1);
```

**Corrected code:**
```c++
// Use the architecture-defined alignment constant
uint32_t l1_addr = (raw_addr + DRAM_ALIGNMENT - 1) & ~(DRAM_ALIGNMENT - 1);
```

### Diagnosis Steps

1. Determine the target architecture and look up its alignment requirements.
2. Search the codebase for hardcoded alignment values (32, 0x1F, etc.) that should use the macros.
3. Run the test on WH to confirm it passes there but fails on BH.

### Fix

Replace all hardcoded alignment values with the architecture-defined macros (`DRAM_ALIGNMENT`, `PCIE_ALIGNMENT`, `L1_ALIGNMENT`).

### Prevention

- Forbid hardcoded alignment constants in code reviews.
- Run CI on multiple architectures to catch alignment-dependent failures.
- When adding support for a new architecture, grep for hardcoded alignment values in existing kernels.

---

## Hang Cause 3.3.3: Misaligned PCIe Transfer

### Symptom

A host-to-device data transfer hangs. The kernel reading from PCIe is stuck at `NRBW`. With sanitization enabled, `DebugSanitizeNocAlignment` is reported with a PCIe target. Without watcher, the core hangs at the read or write barrier.

### Root Cause

PCIe read alignment is 32 bytes on WH and 64 bytes on BH/Quasar. PCIe write alignment is 16 bytes on all architectures. Code developed on Wormhole with 32-byte-aligned host buffers fails on Blackhole because the addresses are not 64-byte aligned.

The PCIe address validation also performs bounds checking against per-core limits:

```c++
// sanitize.h, debug_valid_pcie_addr()
inline uint16_t debug_valid_pcie_addr(uint64_t addr, uint64_t len) {
    if (addr + len <= addr) return DebugSanitizeNocAddrZeroLength;
    core_info_msg_t tt_l1_ptr* core_info = GET_MAILBOX_ADDRESS_DEV(core_info);
    if (addr < core_info->noc_pcie_addr_base) return DebugSanitizeNocAddrUnderflow;
    if (addr + len > core_info->noc_pcie_addr_end) return DebugSanitizeNocAddrOverflow;
    return DebugSanitizeOK;
}
```

**Buggy code:**
```c++
// Works on WH (32-byte alignment), FAILS on BH (needs 64-byte alignment)
#define HOST_BUFFER_ALIGNMENT 32
uint32_t host_read_addr = (host_buffer_base + HOST_BUFFER_ALIGNMENT - 1)
                          & ~(HOST_BUFFER_ALIGNMENT - 1);
noc_async_read(get_noc_addr_from_pcie(host_read_addr), l1_dest, size);
```

**Corrected code:**
```c++
// Use PCIE_ALIGNMENT (architecture-aware)
uint32_t host_read_addr = (host_buffer_base + PCIE_ALIGNMENT - 1) & ~(PCIE_ALIGNMENT - 1);
noc_async_read(get_noc_addr_from_pcie(host_read_addr), l1_dest, size);
```

### Diagnosis Steps

1. Verify the target is a PCIe core by checking the NOC address XY against the `core_info_msg_t` non-worker core list.
2. Check `target_addr & (NOC_PCIE_READ_ALIGNMENT_BYTES - 1)` -- must be zero for reads.
3. Check if the hang is architecture-specific (works on WH, fails on BH).
4. Check the host buffer allocation alignment.

### Fix

Align both the PCIe buffer address (on the host) and the L1 address (on the device) to `PCIE_ALIGNMENT` for the target architecture. For portable code, use 64 bytes (the maximum across all architectures).

### Prevention

- Host-side buffer allocations for DMA should use `aligned_alloc` or equivalent with `PCIE_ALIGNMENT`.
- Always use `PCIE_ALIGNMENT` (not a hardcoded value) for host buffer alignment.
- Test PCIe DMA paths on BH where the alignment requirement is strictest.

---

## Hang Cause 3.3.4: L1-to-L1 Alignment Mismatch in Remote Transfers

### Symptom

The watcher reports `DebugSanitizeNocAlignment` (return code 9) for an L1-to-L1 transfer, even though both addresses appear to be 16-byte aligned. Alternatively, without watcher, the transfer delivers garbled data that causes secondary hangs.

### Root Cause

The alignment cross-check requires that `(worker_addr & 0xF) == (noc_addr & 0xF)` for L1 targets. Even if both addresses are independently aligned to 16 bytes, they must be at the *same* offset within a 16-byte boundary. If the local buffer starts at an address like `0x10000` and the remote source starts at `0x10004`, the lower 4 bits differ (`0x0` vs `0x4`), triggering the alignment violation.

This happens when one address comes from the allocator (which respects alignment) and the other is computed via manual pointer arithmetic that adds a non-aligned offset.

**Buggy code:**
```c++
// Source at 16-byte boundary, but destination has 4-byte offset
uint32_t src_addr = remote_base + 4;    // 0x...4 (low bits = 0x4)
uint32_t dst_addr = local_base;         // 0x...0 (low bits = 0x0)
noc_async_read(get_noc_addr(remote_x, remote_y, src_addr), dst_addr, size);
// Alignment check: 0x4 != 0x0 -- FAILS
```

**Corrected code:**
```c++
// Ensure both addresses have matching low-order alignment
uint32_t src_addr = remote_base;        // 0x...0
uint32_t dst_addr = local_base;         // 0x...0
noc_async_read(get_noc_addr(remote_x, remote_y, src_addr), dst_addr, size);
```

### Diagnosis Steps

1. The watcher error includes both the NOC address and the L1 address. Extract the low 4 bits of each.
2. If they differ, the addresses are not congruent modulo 16.
3. Trace both addresses to their allocation points. Check if one was manually calculated and the other came from the allocator.

### Fix

Ensure both addresses have the same low-bit offset. The simplest approach is to align both to the same power-of-two boundary.

### Prevention

- Allocate all NOC transfer buffers with `L1_ALIGNMENT` (16-byte) alignment. The tt-metal allocator does this by default.
- Manual pointer arithmetic that adds non-aligned offsets to allocated addresses breaks this guarantee.

---

## Hang Cause 3.3.5: Tile Size Mismatch Between Reader and Compute Kernels

### Symptom

A compute kernel hangs at `CWFW` or produces incorrect results that lead to a secondary hang. The reader kernel has completed successfully and pushed the expected number of tiles. Inspecting the CB shows that `tiles_received` matches expectations but the data in the CB does not match the format the compute kernel expects.

### Root Cause

The reader kernel writes tiles in one format (e.g., BFP8 with 1088-byte tiles) but the compute kernel expects a different format (e.g., Float16 with 2048-byte tiles). Because the CB's `fifo_page_size` is configured based on one kernel's expectation, the other kernel's tile count calculations are wrong.

When the producer's `fifo_page_size` is 1088 and the consumer's is 2048, each `cb_push_back(cb, 1)` advances the write pointer by 1088 bytes, but the consumer's `cb_pop_front(cb, 1)` advances the read pointer by 2048 bytes. After the first tile:

- Writer's `fifo_wr_ptr` = start + 1088
- Reader's `fifo_rd_ptr` = start + 2048 (still pointing into the middle of the data that was *not* written)

The read pointer never catches up with the write pointer in the expected pattern. Eventually the consumer may attempt to read past the CB's valid data region, processing garbage memory as tile data.

### Diagnosis Steps

1. Compare the tile size used by the reader kernel (the actual `noc_async_read` transfer size) against the CB's `fifo_page_size`.
2. Dump the CB interface for the stuck CB: `fifo_page_size`, `fifo_rd_ptr`, `fifo_wr_ptr`, `tiles_acked`, `tiles_received`.
3. If `tiles_received - tiles_acked` is positive but the consumer is still at `CWFW`, this indicates the tile sizes are mismatched.
4. Check the host-side `CircularBufferConfig` to see what page size was specified.

### Fix

Ensure all kernels sharing a CB agree on the tile format and size. The `CircularBufferConfig` page size must match the actual data format.

**Buggy host-side code:**
```c++
// WRONG: CB configured for Float16 tile size but reader writes BFP8 tiles
CircularBufferConfig cb_config = CircularBufferConfig(
    num_tiles * float16_tile_size,   // total CB size based on Float16
    {{CBIndex::c_0, DataFormat::Float16_b}}  // CB format
);
// But the DRAM tensor is actually in BFP8 format with smaller tiles
```

**Corrected host-side code:**
```c++
// Match the CB data format to the actual tensor format
CircularBufferConfig cb_config = CircularBufferConfig(
    num_tiles * bfp8_tile_size,
    {{CBIndex::c_0, DataFormat::Bfp8_b}}
);
// All three kernels (reader, compute, writer) use the same data format
```

### Prevention

- Always derive CB page size from the actual tensor data format, not a hardcoded value.
- Validate that reader tile size, CB page size, and compute unpack format all agree.
- Use `get_tile_size(cb_id)` in kernel code rather than hardcoded tile sizes.

---

## Hang Cause 3.3.6: DMA Transfer Size Not Matching Tile Size

### Symptom

A core hangs at `NRBW` or `NWBW`. The NOC transaction counter shows `reads_issued != reads_completed` even though the total bytes transferred should have been sufficient. The watcher does not report an alignment violation.

### Root Cause

When a `noc_async_read` is issued with a transfer size that does not match the actual tile size, the NOC transaction counter is incremented once per call, but the data transferred may be more or less than one tile.

There are two sub-cases:

**Case A: Transfer size < actual tile size.** Each read brings fewer bytes than expected. The CB's `fifo_wr_ptr` does not advance enough via `cb_push_back` (which uses `fifo_page_size`, not the actual DMA transfer size), and subsequent reads start at the wrong offset. The compute kernel processes partially-filled tiles containing stale data.

**Case B: Transfer size > actual tile size.** Each read brings more bytes than expected. The reads overflow the CB boundary (if the CB is sized for the smaller tile size), potentially corrupting adjacent L1 regions (see Section 3.1.2).

**Buggy code:**
```c++
// Reader kernel: uses wrong tile size for DMA
constexpr uint32_t tile_size = 2048;  // hardcoded -- should use get_tile_size()
for (uint32_t i = 0; i < num_tiles; i++) {
    cb_reserve_back(cb_out, 1);
    uint32_t l1_write_addr = get_write_ptr(cb_out);
    noc_async_read(src_noc_addr + i * tile_size, l1_write_addr, tile_size);
    noc_async_read_barrier();
    cb_push_back(cb_out, 1);
}
```

**Corrected code:**
```c++
// Use the CB's configured page size for DMA transfers
for (uint32_t i = 0; i < num_tiles; i++) {
    cb_reserve_back(cb_out, 1);
    uint32_t l1_write_addr = get_write_ptr(cb_out);
    uint32_t page_size = get_local_cb_interface(cb_out).fifo_page_size;
    noc_async_read(src_noc_addr + i * page_size, l1_write_addr, page_size);
    noc_async_read_barrier();
    cb_push_back(cb_out, 1);
}
```

### Diagnosis Steps

1. Compare the DMA transfer size in the kernel code against the CB's `fifo_page_size`.
2. Check whether `get_tile_size()` is used consistently (it depends on the data format being defined).
3. Verify that the DRAM tensor's per-tile size matches the kernel's assumption.

### Fix

Use the CB's `fifo_page_size` or `get_tile_size()` to determine the DMA transfer size. Never hardcode tile sizes in kernel code.

### Prevention

- Avoid hardcoded tile sizes in kernel code.
- Use the CB interface to derive transfer sizes.
- Validate tile sizes in host-side setup code.

---

## Hang Cause 3.3.7: Non-Even CB Size / Tile Count Divisibility

### Symptom

A CB deadlock where `cb_reserve_back` or `cb_wait_front` hangs despite apparently correct tile counts. The `fifo_wr_ptr` wraps to an unexpected address or exceeds `fifo_limit`. This is distinct from the CB protocol violations documented in [Chapter 2, Section 02](../ch02_kernel_and_noc_hangs/02_circular_buffer_deadlocks.md) because the tile counts are balanced -- the issue is the CB geometry.

### Root Cause

The CB implementation requires that the total CB size is an **even multiple** of the tile batch size used in `cb_reserve_back`, `cb_push_back`, `cb_wait_front`, and `cb_pop_front`. This is documented in the API:

> Important note: CB total size must be an even multiple of the argument passed to this call.

The pointer wrapping logic in `cb_push_back` assumes the write pointer will exactly hit `fifo_limit`:

```c++
// cb_push_back: wrap logic assumes exact hit at fifo_limit
ASSERT(get_local_cb_interface(operand).fifo_wr_ptr <= get_local_cb_interface(operand).fifo_limit);
if (get_local_cb_interface(operand).fifo_wr_ptr == get_local_cb_interface(operand).fifo_limit) {
    get_local_cb_interface(operand).fifo_wr_ptr -= get_local_cb_interface(operand).fifo_size;
}
```

If the write pointer advances past `fifo_limit` without exactly hitting it (because the batch size does not evenly divide the CB size), the wrap never triggers, and the pointer goes out of bounds. The same applies to `cb_pop_front` with the read pointer.

**Buggy code:**
```c++
// Host side: CB with 10 pages
CircularBufferConfig cb_config(10 * tile_size, {{CBIndex::c_0, format}});
// Kernel side: reserve 3 at a time (10 % 3 != 0)
cb_reserve_back(cb_out, 3);  // After 3 batches (9 pages), pointer near fifo_limit
// 4th batch: pointer skips past fifo_limit, wrap never fires
```

**Corrected code:**
```c++
// Host side: CB with 12 pages (divisible by 3)
CircularBufferConfig cb_config(12 * tile_size, {{CBIndex::c_0, format}});
// Or use batch size 2 or 5 (divisors of 10)
```

### Diagnosis Steps

1. Check the CB's `fifo_num_pages` and the batch sizes used in `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front`.
2. Verify that `fifo_num_pages % batch_size == 0`.
3. Inspect `fifo_wr_ptr` and `fifo_rd_ptr` -- if either exceeds `fifo_limit`, the non-divisibility is the cause.
4. In debug builds, look for the assertion `fifo_wr_ptr <= fifo_limit` firing.

### Fix

Ensure `fifo_num_pages` is evenly divisible by all batch sizes used in the kernel.

### Prevention

- Add compile-time or runtime assertions that `fifo_num_pages % batch_size == 0`.
- Document CB sizing constraints in kernel comments.
- Always create CB sizes as multiples of the processing batch size.

---

## Hang Cause 3.3.8: Transfer Size Not Multiple of NOC Word Size (Stale Bytes)

### Symptom

Data read from DRAM contains unexpected bytes at the end of the transfer. The stale bytes cause incorrect computation, which may lead to a secondary hang (e.g., an invalid address derived from corrupted data). No watcher sanitization error is reported because the addresses and alignment are correct -- only the transfer size is problematic.

### Root Cause

When the NOC transfer size is not a multiple of the NOC word size, the hardware rounds up the actual transfer to the next word boundary. The extra bytes are "stale" -- they contain whatever was previously at the destination address. This is not a hang directly, but it causes incorrect data that can trigger secondary hangs.

The NOC word size varies by architecture:
- **Wormhole:** 256-bit (32-byte) NOC word
- **Blackhole:** 512-bit (64-byte) NOC word
- **Quasar:** 2048-bit (256-byte) NOC word

A transfer of 17 bytes on Wormhole actually transfers 32 bytes, with the last 15 bytes being whatever was in L1 before the transfer. On Quasar, a transfer of 100 bytes actually transfers 256 bytes, leaving 156 stale bytes.

### Diagnosis Steps

1. Check the transfer size against the NOC word size for the target architecture.
2. Dump the destination buffer and look for unexpected data at the end of the transfer.
3. Ensure the kernel does not read past the intended transfer size.

### Fix

Either ensure all transfers are multiples of the NOC word size, or ensure the kernel only reads the intended number of bytes from the destination buffer:

**Buggy code:**
```c++
// Transfer 17 bytes from DRAM -- hardware transfers 32 (WH) or 64 (BH) bytes
noc_async_read(dram_addr, l1_dest, 17);
noc_async_read_barrier();
// Kernel reads 32 bytes from l1_dest -- last 15 bytes are stale!
uint32_t value = *reinterpret_cast<uint32_t*>(l1_dest + 20);  // stale data
```

**Corrected code:**
```c++
// Pad transfer size to NOC word boundary
constexpr uint32_t NOC_WORD_SIZE = NOC_DRAM_READ_ALIGNMENT_BYTES;  // 32 on WH, 64 on BH
uint32_t padded_size = (17 + NOC_WORD_SIZE - 1) & ~(NOC_WORD_SIZE - 1);
noc_async_read(dram_addr, l1_dest, padded_size);
noc_async_read_barrier();
// Only use the first 17 bytes; ignore padding
```

### Prevention

- Pad transfer sizes to the architecture's NOC word size.
- Use tile-based transfers (which are always properly sized) rather than arbitrary byte counts.
- On Quasar, be especially careful: the 256-byte NOC word means even moderately-sized transfers may contain significant padding.

---

## Hang Cause 3.3.9: TRISC `cb_addr_shift` Misinterpretation

### Symptom

A TRISC core computes incorrect addresses for tile data within a CB, leading to reads from wrong L1 locations. The compute produces garbage output that may cause downstream NOC transactions to invalid addresses. On Quasar specifically, the kernel may work on WH/BH but fail on Quasar due to architecture-specific CB interface differences.

### Root Cause

On TRISC cores, CB addresses use a shifted representation: `cb_addr_shift` is `CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT` (value 4), meaning CB addresses are stored as 16-byte word indices rather than byte addresses. On BRISC/NCRISC, `cb_addr_shift` is 0 (byte addresses).

```c++
// tt_metal/hw/inc/internal/circular_buffer_interface.h
#if defined(COMPILE_FOR_TRISC)
constexpr uint32_t cb_addr_shift = CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT;  // 4
#else
constexpr uint32_t cb_addr_shift = 0;
#endif
```

If kernel code designed for a data-movement core (BRISC/NCRISC) is mistakenly compiled for TRISC, all CB pointer arithmetic produces addresses that are 16x too small. A `get_write_ptr(cb_id)` on TRISC returns a word index, not a byte address.

Additionally, on Quasar, the CB interface uses `thread_local` storage:

```c++
#ifdef ARCH_QUASAR
extern thread_local CBInterface cb_interface[NUM_CIRCULAR_BUFFERS];
#else
extern CBInterface cb_interface[NUM_CIRCULAR_BUFFERS];
#endif
```

Quasar also has 4 TRISC cores (vs. 3 on WH/BH) and 8 DM cores. If a kernel is compiled with the wrong `COMPILE_FOR_TRISC` define or the wrong shift value, CB addresses will be miscomputed -- and the Quasar thread-local storage may compound the issue by returning a different CB interface per thread.

### Diagnosis Steps

1. Check which RISC the kernel is compiled for vs. which RISC it was intended for.
2. If a compute kernel is calling NOC API functions (which expect byte addresses) with CB pointers (which are word indices on TRISC), the addresses will be wrong by a factor of 16.
3. Look for `noc_async_read` or `noc_async_write` calls in TRISC kernels -- these should generally not be present.
4. On Quasar, verify the kernel compilation flags for the correct `ARCH_QUASAR` and `COMPILE_FOR_TRISC` defines.

### Fix

Do not mix data-movement API calls with compute-kernel compilation. If a TRISC kernel needs to issue NOC transactions (rare), manually convert addresses: `byte_addr = word_index << cb_addr_shift`.

### Prevention

- Follow the standard tt-metal kernel model: reader (BRISC) handles NOC reads, writer (NCRISC) handles NOC writes, and compute (TRISC) handles only CB-to-register operations. The build system enforces this separation through different compilation flags.
- Use the build system's target specifications rather than manually defining `COMPILE_FOR_TRISC`.
- Test on all target architectures before deployment.

---

## Hang Cause 3.3.10: Transfer Length Zero or Overflow

### Symptom

The watcher reports `DebugSanitizeNocAddrZeroLength` (return code 5). The core enters a deliberate hang.

### Root Cause

The sanitizer checks for zero-length transfers in all address validation functions:

```c++
if (addr + len <= addr) {
    return DebugSanitizeNocAddrZeroLength;
}
```

This condition catches two cases:
1. `len == 0`: a zero-length DMA transfer, which is meaningless and indicates a bug.
2. `addr + len` overflows `uint64_t`: the transfer is so large that the end address wraps around.

The most common cause is a loop that computes `remaining_bytes` as an unsigned integer and subtracts too much:

**Buggy code:**
```c++
uint32_t remaining = total_size - already_read;  // If already_read > total_size, wraps to ~4GB
noc_async_read(src, dst, remaining);             // Zero-length or huge length
```

**Corrected code:**
```c++
if (already_read >= total_size) break;
uint32_t remaining = total_size - already_read;
noc_async_read(src, dst, remaining);
```

Other common causes:
- Computing tile size as `num_tiles * tile_size` where `num_tiles` is 0.
- Using `get_tile_size()` before the data format tables are initialized.
- Integer arithmetic error in page size calculation.

### Diagnosis Steps

1. The watcher log shows the address and length (0 or very large). If close to 2^32, suspect unsigned underflow.
2. Trace back to the kernel code that computed the transfer length.
3. Check runtime arguments that feed into the length computation.
4. Check for conditional logic that might issue a DMA with zero length.

### Fix

Validate that transfer lengths are non-zero before issuing NOC transactions.

### Prevention

- Add explicit checks: `TT_ASSERT(len > 0)` before NOC calls in debug builds.
- Ensure data format initialization happens before any tile size queries.
- Enable NOC sanitization, which catches zero-length and overflow conditions.

---

## Hang Cause 3.3.11: CB Bounds Overflow Detected by Sanitizer

### Symptom

The watcher reports `DebugSanitizeCBOutOfBounds`. The core enters a deliberate hang.

### Root Cause

The `debug_valid_cb_addr` function checks whether a NOC DMA transfer to a local L1 address within a CB stays within that CB's allocated region:

```c++
// tt_metal/hw/inc/internal/debug/sanitize.h
inline uint16_t debug_valid_cb_addr(uint32_t l1_addr, uint32_t len) {
    for (uint32_t i = 0; i < NUM_CIRCULAR_BUFFERS; i++) {
        LocalCBInterface& cb = get_local_cb_interface(i);
        if (cb.fifo_size == 0) continue;  // unused CB

        uint32_t cb_start = cb.fifo_limit - cb.fifo_size;
        uint32_t cb_end = cb.fifo_limit;

        if (l1_addr >= cb_start && l1_addr < cb_end) {
            // Address is in this CB -- verify the full transfer fits
            if (static_cast<uint64_t>(l1_addr) + len > cb_end) {
                return DebugSanitizeCBOutOfBounds;
            }
            return DebugSanitizeOK;
        }
    }
    return DebugSanitizeOK;  // not inside any CB -- other checks validate
}
```

This fires when a NOC read or write lands inside a CB but the transfer extends beyond the CB's `fifo_limit`. Note the `static_cast<uint64_t>` to prevent integer overflow when computing `l1_addr + len`.

**Important:** This check is disabled when `WATCHER_DISABLE_CB_SANITIZE` is defined, and it does not run on ethernet cores (ERISC/IDLE_ERISC). The check iterates over all `NUM_CIRCULAR_BUFFERS` (32 on WH, 64 on BH/Quasar) for every NOC transaction, which adds measurable overhead.

### Diagnosis Steps

1. The watcher log identifies which CB was violated, the L1 address, and the transfer length.
2. Compare `l1_addr + len` against the CB's `fifo_limit`.
3. Determine why the transfer exceeds the CB -- either the tile size is wrong or the write pointer has drifted.

### Fix

Correct the DMA transfer size to match the CB's page size, or increase the CB size to accommodate the transfer.

### Prevention

- Do not disable CB sanitization (`TT_METAL_WATCHER_DISABLE_CB_SANITIZE`) unless absolutely necessary for performance.
- Ensure DMA transfer sizes match CB page sizes.
- After `cb_reserve_back(cb_id, N)` returns, exactly `N * fifo_page_size` bytes of contiguous space are guaranteed starting from `get_write_ptr(cb_id)`.

---

## Hang Cause 3.3.12: Misaligned Multicast Write Causing Partial Delivery

### Symptom

A multicast write appears to succeed (the barrier completes on the sender), but some destination cores received incorrect or partial data. The receiving cores hang when they try to use the corrupted data.

### Root Cause

Multicast writes have the same alignment requirements as unicast writes, but the consequences of misalignment are worse: the data may be partially delivered to some cores and completely missed by others, depending on the NOC routing path and the alignment of the target addresses on each core.

The sanitization layer checks multicast writes for:
1. Both start and end coordinates must be Tensix cores (not DRAM, PCIe, or Ethernet).
2. The address range must be valid on worker cores.
3. Alignment must match across the local and remote addresses.

But if watcher is disabled, a misaligned multicast write simply corrupts data at the destination cores without any indication to the sender.

### Diagnosis Steps

1. Enable watcher and reproduce the hang.
2. Check for `DebugSanitizeNocAlignment` or `DebugSanitizeNocMulticastNonWorker` errors.
3. Verify that the multicast target address is aligned to `NOC_L1_WRITE_ALIGNMENT_BYTES` (16 bytes).

### Fix

Align the multicast target address to `L1_ALIGNMENT`:

**Buggy code:**
```c++
// Multicast destination has misaligned target offset
uint32_t src_addr = aligned_l1_base;    // 0x...0 (aligned)
uint64_t mcast_dst = get_noc_multicast_addr(start_x, start_y, end_x, end_y, target_offset);
// target_offset = 0x104 (not 16-byte aligned)
noc_async_write_multicast(src_addr, mcast_dst, size, num_dests);
```

**Corrected code:**
```c++
uint32_t aligned_target = (target_offset + L1_ALIGNMENT - 1) & ~(L1_ALIGNMENT - 1);
uint64_t mcast_dst = get_noc_multicast_addr(start_x, start_y, end_x, end_y, aligned_target);
uint32_t src_addr = aligned_l1_base;  // Must have same low bits as aligned_target
noc_async_write_multicast(src_addr, mcast_dst, size, num_dests);
```

### Prevention

- Always use aligned addresses for multicast operations.
- Enable watcher during multicast kernel development.
- When using multicast, ensure that the L1 buffer on the sending core is allocated at the same offset (modulo alignment) as the target address on all receiving cores.

---

## Summary Table

| ID | Hang Cause | Key Indicator | Watcher Return Code | Architecture Sensitivity |
|----|-----------|---------------|---------------------|-------------------------|
| 3.3.1 | Misaligned DRAM read | L1/DRAM address not aligned | `DebugSanitizeNocAlignment` | DRAM read: 32 B (WH), 64 B (BH/Q) |
| 3.3.2 | Architecture-dependent alignment failure | Works on WH, hangs on BH | `DebugSanitizeNocAlignment` | BH/Quasar stricter than WH |
| 3.3.3 | Misaligned PCIe transfer | PCIe addr not aligned | `DebugSanitizeNocAlignment` | PCIe read: 32 B (WH), 64 B (BH/Q) |
| 3.3.4 | L1-to-L1 alignment mismatch | `(local & 0xF) != (remote & 0xF)` | `DebugSanitizeNocAlignment` | All (16 B L1 alignment) |
| 3.3.5 | Tile size mismatch (reader vs compute) | CB data format mismatch | None (behavioral: `CWFW`) | All |
| 3.3.6 | DMA transfer size != tile size | Wrong bytes per tile | None (behavioral: `NRBW`) | All |
| 3.3.7 | Non-even CB size / batch divisibility | `fifo_wr_ptr > fifo_limit` | None (debug `ASSERT`) | All |
| 3.3.8 | Transfer size not NOC word multiple | Stale trailing bytes | None (behavioral) | WH: 32 B, BH: 64 B, Q: 256 B |
| 3.3.9 | TRISC `cb_addr_shift` misinterpretation | Address off by 16x | Various (secondary) | All; Quasar thread_local adds risk |
| 3.3.10 | Zero-length or overflow transfer | `len == 0` or overflow | `DebugSanitizeNocAddrZeroLength` | All |
| 3.3.11 | CB bounds overflow (sanitizer) | DMA extends past CB boundary | `DebugSanitizeCBOutOfBounds` | All (not ETH cores) |
| 3.3.12 | Misaligned multicast write | Partial data delivery | `DebugSanitizeNocAlignment` | All |

---

**Previous:** [`02_dram_and_noc_backpressure.md`](./02_dram_and_noc_backpressure.md)
**Next:** [`04_allocation_failures_and_silent_oom.md`](./04_allocation_failures_and_silent_oom.md)
