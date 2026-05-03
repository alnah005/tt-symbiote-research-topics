# Chapter 3: Memory-Related Hang Causes

Memory subsystem issues -- L1 corruption, DRAM saturation, alignment violations, allocation failures, and out-of-memory conditions -- produce hangs that are among the hardest to diagnose on Tenstorrent hardware. Unlike the kernel-level and NOC deadlocks documented in Chapter 2, memory-related hangs frequently manifest as **secondary failures**: the root cause is a corrupted address, an overflowed buffer, or a silent OOM, but the observable symptom is an unrelated NOC transaction stall or barrier hang on a core far from the original fault. A corrupted value may change the exit condition of a spin-loop itself, making the root cause invisible at the point of failure.

This chapter systematically catalogs every known memory-related hang mechanism, organized from the smallest memory scope (L1 corruption on a single core) to the largest (system-wide DRAM bandwidth saturation). Every hang cause uses the **Symptom / Root Cause / Diagnosis Steps / Fix / Prevention** format introduced in [Chapter 1, `01_what_is_a_hang.md`](../ch01_anatomy_of_a_hang/01_what_is_a_hang.md).

## Prerequisites

- **Chapter 1, [`01_what_is_a_hang.md`](../ch01_anatomy_of_a_hang/01_what_is_a_hang.md)**: The RISC-V spin-loop model, the 5-part diagnostic format, and the distinction between deliberate and incidental hangs.
- **Chapter 1, [`02_blocking_primitives_taxonomy.md`](../ch01_anatomy_of_a_hang/02_blocking_primitives_taxonomy.md)**: Familiarity with `noc_async_read_barrier`, `noc_async_write_barrier`, `cb_reserve_back`, and `cb_wait_front` blocking primitives and their waypoint codes (`CRBW`, `CWFW`, `NRBW`, `NWBW`).
- **Chapter 1, [`04_hang_causes_across_architectures.md`](../ch01_anatomy_of_a_hang/04_hang_causes_across_architectures.md)**: Architectural differences between Wormhole, Blackhole, and Quasar, especially L1 sizes, DRAM configurations, and the Blackhole inline-write workaround.
- **Chapter 2, [`02_circular_buffer_deadlocks.md`](../ch02_kernel_and_noc_hangs/02_circular_buffer_deadlocks.md)**: The CB data model (`LocalCBInterface`, `tiles_acked`/`tiles_received`, `fifo_wr_ptr`/`fifo_rd_ptr`) and the producer-consumer protocol. This chapter covers CB *overflow and corruption* rather than CB *deadlocks*.
- **Chapter 2, [`03_noc_address_sanitization_and_violations.md`](../ch02_kernel_and_noc_hangs/03_noc_address_sanitization_and_violations.md)**: The NOC validation pipeline and `DebugSanitize` return codes. This chapter focuses on the *memory-side causes* that produce addresses failing validation, not the validation machinery itself.

## How to Use This Chapter

- **If the watcher reports an L1 or mailbox corruption** (`DumpL1Status`, `DebugSanitizeNocAddrMailbox`, `DebugSanitizeL1AddrOverflow`): start with [Section 01](./01_l1_memory_corruption_and_overflow.md).
- **If multiple cores hang at NOC barriers with no sanitization violation**: start with [Section 02](./02_dram_and_noc_backpressure.md) (DRAM backpressure).
- **If the watcher reports `DebugSanitizeNocAlignment`**: start with [Section 03](./03_alignment_and_tile_size_mismatches.md).
- **If a host-side `Out of Memory` exception precedes the hang**, or if the NOC address looks like garbage: start with [Section 04](./04_allocation_failures_and_silent_oom.md).

## Chapter Contents

| # | File | Focus | Key Return Codes / Waypoints |
|---|------|-------|------|
| 1 | [`01_l1_memory_corruption_and_overflow.md`](./01_l1_memory_corruption_and_overflow.md) | L1 memory map, overflow corrupting adjacent structures, `DumpL1Status` address-0 check, silent corruption scenarios, stack overflow, `debug_sanitize_l1_access`, BH inline write region corruption, BH/Quasar L1 cache coherence, watcher sanitize state corruption (11 scenarios) | `DebugSanitizeL1AddrOverflow`, `DebugSanitizeNocAddrMailbox` |
| 2 | [`02_dram_and_noc_backpressure.md`](./02_dram_and_noc_backpressure.md) | DRAM bandwidth saturation, bank collision stalls, NOC backpressure propagation, command buffer stalls, posted vs non-posted write asymmetry, interleaved DRAM hotspots, BH inline write backpressure, DRAM arbiter hang test, read-write ordering violations (11 scenarios) | `NWBW`, `NRBW` |
| 3 | [`03_alignment_and_tile_size_mismatches.md`](./03_alignment_and_tile_size_mismatches.md) | NOC alignment requirements per architecture, misaligned DMA hangs, alignment cross-check in sanitize, tile size mismatches, DMA size vs tile size mismatch, NOC word size stale bytes, TRISC `cb_addr_shift`, multicast alignment (12 scenarios) | `DebugSanitizeNocAlignment`, `DebugSanitizeNocAddrZeroLength` |
| 4 | [`04_allocation_failures_and_silent_oom.md`](./04_allocation_failures_and_silent_oom.md) | L1/DRAM allocator OOM, `free_list_opt` fragmentation, garbage buffer addresses, CB overflow/overwrite, watcher CB sanitization, RTA out of bounds, `shrink_size`/`reset_size`, size-segregated boundary, silent double-free (12 scenarios) | `DebugSanitizeCBOutOfBounds`, `DebugAssertRtaOutOfBounds` |

**Covers research questions:** Q4 (all memory-related hang causes).

---

## L1 Memory Layout Quick Reference

The following table summarizes the L1 memory map that is central to understanding corruption and overflow scenarios. Detailed per-architecture breakdowns appear in Section 01.

| Region | WH Address Range | BH Address Range | Quasar Address Range |
|--------|-----------------|-----------------|---------------------|
| Boot code / launch addr | `0x0` | `0x0` | `0x0` |
| ARC FW scratch | `0x10` | `0x10` | Varies |
| Inline-write staging | N/A | `0x20` -- `0x5F` | N/A |
| Mailbox (`MEM_MAILBOX_BASE`) | `0x10` -- `MEM_MAILBOX_END` | `0x60` -- `MEM_MAILBOX_END` | Varies (uncached) |
| Firmware (BRISC, NCRISC, TRISC0-2) | After mailbox | After mailbox | After mailbox |
| Read-only boundary (`MEM_MAP_READ_ONLY_END`) | ~29 KB | ~39 KB | Varies |
| System reserved end (`MEM_MAP_END`) | ~32 KB | ~39 KB | Varies |
| Kernel config / CBs / Semaphores / RTAs | After `MEM_MAP_END` | After `MEM_MAP_END` | After `MEM_MAP_END` |
| User L1 buffers | Above kernel config | Above kernel config | Above kernel config |
| Total L1 (`MEM_L1_SIZE`) | 1464 KB (1,499,136 B) | 1536 KB (1,572,864 B) | 4096 KB (4,194,304 B) |
| Total ETH L1 (`MEM_ETH_SIZE`) | ~256 KB - 32 B | 512 KB | 512 KB |

---

**Next:** [`01_l1_memory_corruption_and_overflow.md`](./01_l1_memory_corruption_and_overflow.md)
