# 3.4 Allocation Failures and Silent OOM

Allocation failures are the final link in the memory-related hang chain. When the host-side L1 or DRAM allocator fails to find enough contiguous space, the failure may be handled gracefully (with a `TT_THROW` exception) or may silently produce a garbage address that gets passed to device kernels. These garbage addresses cause secondary NOC violations or out-of-bounds accesses that manifest as hangs far removed from the original allocation failure. Circular buffer overflow -- a device-side "allocation failure" where the producer writes more data than the CB can hold -- creates data corruption that is equally hard to trace. This section examines the allocator implementation, its failure modes, and the downstream hang scenarios that result.

Reference files: `tt_metal/impl/allocator/algorithms/free_list_opt.cpp`, `tt_metal/impl/allocator/algorithms/free_list_opt.hpp`, `tt_metal/hw/inc/internal/debug/sanitize.h` (CB sanitization), `tt_metal/hw/inc/api/dataflow/dataflow_api.h` (CB push/pop), `tt_metal/hw/inc/internal/circular_buffer_interface.h`

---

## The `FreeListOpt` Allocator

Both L1 and DRAM buffer allocations are managed by the `FreeListOpt` allocator, a size-segregated free-list implementation optimized for performance. Understanding its behavior under memory pressure is essential for diagnosing allocation-related hangs.

### Allocator Structure

```c++
// From free_list_opt.hpp
class FreeListOpt : public Algorithm {
    // SoA (Structure of Arrays) for cache efficiency
    std::vector<DeviceAddr> block_address_;       // Start address of each block
    std::vector<DeviceAddr> block_size_;           // Size of each block
    std::vector<ssize_t> block_prev_block_;        // Linked list: previous block index
    std::vector<ssize_t> block_next_block_;        // Linked list: next block index
    std::vector<uint8_t> block_is_allocated_;      // Is this block allocated?
    std::vector<uint8_t> meta_block_is_allocated_; // Is this metadata slot in use?
    std::vector<size_t> free_meta_block_indices_;  // Recycled metadata slot indices

    // Size-segregated free lists (log2-based buckets)
    static constexpr size_t size_segregated_base = 1024;  // bytes
    const size_t size_segregated_count;
    std::vector<std::vector<size_t>> free_blocks_segregated_by_size_;

    // Hash table for O(1) allocated block lookup during deallocation
    static constexpr size_t n_alloc_table_buckets = 512;
    std::vector<std::vector<std::pair<DeviceAddr, size_t>>> allocated_block_table_;

    SearchPolicy policy_;  // BEST fit or FIRST fit
};
```

The two search policies:
- **`SearchPolicy::BEST`**: Finds the smallest block that fits the request (best-fit). Reduces external fragmentation but is slower.
- **`SearchPolicy::FIRST`**: Finds the first block that fits (address-ordered). Faster but can leave fragmentation.

### Allocation Flow

```c++
// From free_list_opt.cpp, allocate()
std::optional<DeviceAddr> FreeListOpt::allocate(
    DeviceAddr size_bytes, bool bottom_up, DeviceAddr address_limit) {
    DeviceAddr alloc_size = align(std::max(size_bytes, min_allocation_size_));

    // Search size-segregated free lists starting from the matching size class
    ssize_t target_block_index = -1;
    size_t size_segregated_index = get_size_segregated_index(alloc_size);
    for (size_t i = size_segregated_index; i < free_blocks_segregated_by_size_.size(); i++) {
        // ... find best/first fit block ...
    }

    if (target_block_index == -1) {
        return std::nullopt;  // OUT OF MEMORY
    }

    // Split block, mark as allocated
    size_t allocated_block_index = allocate_in_block(target_block_index, alloc_size, offset);
    DeviceAddr start_address = block_address_[allocated_block_index];

    if (start_address + offset_bytes_ < address_limit) {
        TT_THROW("Out of Memory: Cannot allocate at an address below {}. Allocation at {}",
                 address_limit, start_address + offset_bytes_);
    }

    return start_address + offset_bytes_;
}
```

**Key observations:**
1. When no block fits, `allocate` returns `std::nullopt`. Whether the caller checks this return value determines whether the failure is handled gracefully or becomes a silent hang.
2. The `address_limit` check throws `TT_THROW` -- this is the "hard" OOM path with a clear error message.
3. The allocator's `init()` creates a single block spanning all memory. On `clear()`, it reinitializes to this state, coalescing all fragments.

### Deallocation and Coalescing

```c++
void FreeListOpt::deallocate(DeviceAddr absolute_address) {
    DeviceAddr addr = absolute_address - offset_bytes_;
    auto block_index_opt = get_and_remove_from_alloc_table(addr);
    if (!block_index_opt.has_value()) {
        return;  // Address not found -- silent no-op
    }
    size_t block_index = block_index_opt.value();
    block_is_allocated_[block_index] = false;

    // Merge with previous block if it's free
    ssize_t prev_block = block_prev_block_[block_index];
    if (prev_block != -1 && !block_is_allocated_[prev_block]) {
        block_size_[prev_block] += block_size_[block_index];
        // ... update links ...
        free_meta_block(block_index);
        block_index = prev_block;
    }

    // Merge with next block if it's free
    ssize_t next_block = block_next_block_[block_index];
    if (next_block != -1 && !block_is_allocated_[next_block]) {
        block_size_[block_index] += block_size_[next_block];
        // ... update links ...
        free_meta_block(next_block);
    }
}
```

**Critical note:** Deallocating an address that was never allocated (or was already freed) is a **silent no-op**. This can hide double-free bugs that eventually lead to corruption of the allocator state.

### Size-Segregated Bucketing

The size classes use logarithmic bucketing:
```
Class 0: [1024, 2048) bytes
Class 1: [2048, 4096) bytes
Class 2: [4096, 8192) bytes
... up to the maximum allocation size
```

A request for 64 KB will search class 6 and above. If all free blocks are in class 0-5 (each smaller than 64 KB), the search fails even though their total may exceed 64 KB.

---

## Hang Cause 3.4.1: L1 Allocator OOM Leading to Garbage Buffer Address

### Symptom

A kernel hangs with a NOC sanitization violation (`DebugSanitizeNocTargetInvalidXY`, `DebugSanitizeNocAddrOverflow`, or similar). The NOC address in the watcher log is clearly invalid -- it may be 0, an address above `MEM_L1_SIZE`, or have X/Y coordinates of 0xFF. Tracing back, the L1 buffer address used by the kernel was never properly allocated.

### Root Cause

When the L1 allocator runs out of space, it returns `std::nullopt`. If the calling code does not check this result and proceeds to compute a NOC address from the (missing) buffer address, the kernel receives a garbage L1 address through its runtime arguments.

The most common paths to L1 OOM:
1. Too many circular buffers configured on one core, consuming the available L1 space.
2. Large kernel text sizes (especially on WH where BRISC/TRISC kernel text competes with CB space in L1).
3. Accumulation of tensor buffers without deallocating previous allocations.

The failure chain:
1. Host-side code calls `allocate()` for an L1 buffer.
2. `allocate()` returns `std::nullopt` due to OOM or fragmentation.
3. The caller does not check, and stores a default or stale address in the kernel's runtime arguments.
4. The kernel reads this address from runtime args and constructs a NOC address.
5. The NOC address points to a non-existent core (if the X/Y bits are garbage) or an out-of-range offset.
6. The NOC transaction either triggers a sanitize violation (deliberate hang) or fails silently (barrier hang).

**Buggy host-side code:**
```c++
// No error check on allocation result
auto result = allocator->allocate(buffer_size);
uint32_t buffer_addr = result.value_or(0);  // BUG: silently uses address 0!
SetRuntimeArgs(program, kernel_id, core, {buffer_addr, ...});
// Address 0 maps to boot code entry point (scenario 3.1.2) -- device will hang
```

**Corrected host-side code:**
```c++
auto result = allocator->allocate(buffer_size);
if (!result.has_value()) {
    auto stats = allocator->get_statistics();
    TT_THROW(
        "Failed to allocate {} bytes on core {}. "
        "Free: {} bytes, Largest block: {} bytes",
        buffer_size, core.str(),
        stats.total_free_bytes, stats.largest_free_block_bytes);
}
uint32_t buffer_addr = result.value();
TT_FATAL(buffer_addr >= MEM_MAP_END, "Buffer at {} is in system-reserved region", buffer_addr);
SetRuntimeArgs(program, kernel_id, core, {buffer_addr, ...});
```

### Diagnosis Steps

1. Decode the garbage NOC address from the watcher's sanitize error. If the X/Y coordinates are 0,0 or 0xFF,0xFF, suspect a default-initialized or uninitialized address.
2. Trace the NOC address back to the kernel's runtime arguments (read from L1).
3. Trace the runtime argument back to the host-side code that set it. Check if it came from an `allocate()` call.
4. Use `allocator.get_statistics()` to check available L1 space:
   ```
   total_allocatable_size_bytes  -- total L1 available
   total_allocated_bytes         -- currently allocated
   total_free_bytes              -- currently free
   largest_free_block_bytes      -- largest contiguous free block
   ```
5. If `largest_free_block_bytes < requested_size` but `total_free_bytes >= requested_size`, the problem is fragmentation (see 3.4.3).

### Fix

Add error checking after every L1 allocation. Use `TT_THROW` or `TT_FATAL` to fail early rather than passing garbage to the device.

### Prevention

- Never use `.value_or(0)` or `.value_or(default)` for device memory allocations. A default address of 0 maps to the boot code entry point.
- Always check `std::optional` return values from allocators.
- Monitor L1 utilization before deploying kernels.
- Use `worker_l1_size` in `MeshDevice::create_unit_meshes` to configure available L1 space.

---

## Hang Cause 3.4.2: DRAM Allocation Failure Passed to Device

### Symptom

Similar to 3.4.1, but the garbage address targets a DRAM core. The watcher may report `DebugSanitizeNocAddrOverflow` on a DRAM address, or the kernel may successfully read from an unallocated DRAM region and get garbage data that causes a secondary hang.

### Root Cause

DRAM allocations can fail due to fragmentation or exhaustion. Unlike L1, where OOM typically causes an immediate error on the host, DRAM OOM may be handled differently depending on the allocation path. The DRAM allocator is also a `FreeListOpt` instance but manages much larger memory (WH: ~12 GB; BH: ~32 GB). DRAM allocation failures are less common than L1 but occur when:
- The model requires more total tensor storage than available DRAM.
- Fragmentation from repeated allocate/deallocate cycles creates a state where no contiguous block is large enough.
- Multiple programs share the same device without proper memory management.

### Diagnosis Steps

1. Check whether the DRAM address in the failing NOC transaction is within the valid range (`noc_dram_addr_base` to `noc_dram_addr_end`).
2. Verify the host-side DRAM allocation succeeded.
3. Check total DRAM utilization via `get_statistics()`.

### Fix

Validate DRAM allocations and fail gracefully if memory is exhausted.

### Prevention

- Track DRAM utilization as part of the workload setup.
- Implement host-side memory budgeting that checks available DRAM before allocating.
- Use `TT_FATAL` on allocation failure.
- Deallocate tensors that are no longer needed promptly.

---

## Hang Cause 3.4.3: Free-List Fragmentation in Long-Running Workloads

### Symptom

A workload that runs successfully for many iterations eventually fails with an allocation error or silent hang. The total free memory is sufficient but no single contiguous block is large enough for the requested allocation. The `get_statistics()` output shows `total_free_bytes >> largest_free_block_bytes`.

### Root Cause

The `FreeListOpt` allocator coalesces adjacent free blocks during deallocation, but coalescing only merges *adjacent* free blocks. If the allocation pattern creates an alternating pattern of allocated and free blocks, no coalescing can occur:

```
Initial state:  [================= 1 MB free =================]

After allocations:
[A 100KB][free 50KB][B 200KB][free 30KB][C 100KB][free 520KB]

After freeing A:
[free 100KB][free 50KB][B 200KB][free 30KB][C 100KB][free 520KB]
                        ^--- B separates the first two free blocks

Cannot coalesce: the first two free blocks are not adjacent.
Requesting 600 KB fails despite 700 KB total free.
```

The size-segregated free lists help find blocks efficiently but do not prevent fragmentation. The `SearchPolicy::BEST` fit policy allocates from the smallest sufficient block, leaving many oddly-sized fragments. `SearchPolicy::FIRST` is faster but may also leave awkward remnants.

### Diagnosis Steps

1. Use `allocator.dump_blocks()` to print the full block table:
   ```
   Block   Address        Size   PrevID   NextID   Allocated
       0         0      1024     none        1   yes
       1      1024      2048        0        2   no
       2      3072      1024        1        3   yes
       ...
   ```
2. Look for many small free blocks separated by allocated blocks.
3. Check `get_statistics().largest_free_block_bytes` -- if this is much smaller than `total_free_bytes`, fragmentation is severe.

### Fix

- Restructure the workload to allocate and deallocate in a stack-like order (LIFO).
- Pre-allocate all long-lived buffers first, then allocate/deallocate short-lived buffers from the remaining space.
- Use `allocator.clear()` between major workload phases to reset the allocator (requires re-allocating all buffers).
- Consider using `SearchPolicy::FIRST` with bottom-up allocation to pack allocations at the bottom and leave a large contiguous block at the top.

### Prevention

- Allocate long-lived buffers first, short-lived buffers last. Deallocate in reverse order.
- Monitor `largest_free_block_bytes` over time to detect creeping fragmentation.
- Use arena-style allocation for per-iteration buffers: allocate a large block once, subdivide it manually within the kernel.
- Consider periodic defragmentation by reallocating all active buffers.

---

## Hang Cause 3.4.4: Double-Free Leading to Allocator State Corruption

### Symptom

Allocation returns an address that overlaps with another allocated buffer. Two different tensors share the same memory, leading to data corruption. The corrupted data causes a secondary hang when used as NOC addresses or control values.

### Root Cause

The `FreeListOpt::deallocate` method silently ignores attempts to deallocate unrecognized addresses:

```c++
void FreeListOpt::deallocate(DeviceAddr absolute_address) {
    auto block_index_opt = get_and_remove_from_alloc_table(addr);
    if (!block_index_opt.has_value()) {
        return;  // Silent no-op for unknown addresses
    }
    // ... proceed with deallocation and coalescing ...
}
```

If the same address is deallocated twice:
1. The first deallocation succeeds and returns the block to the free list.
2. The allocator may re-allocate that block to a new buffer.
3. The second deallocation finds the address in the allocated table (from the new allocation) and frees it again.
4. Now two different callers believe they own the same memory region.

The more dangerous scenario is when the host code frees a buffer and then continues to pass the buffer's address to kernels as a runtime argument. The address may be re-allocated to a different buffer, and now two kernels believe they own the same memory region. This produces data corruption that leads to hangs far removed from the original double-free.

### Diagnosis Steps

1. Add logging around allocate/deallocate calls to trace the lifecycle of each address.
2. Look for two tensors whose address ranges overlap.
3. Check `allocated_addresses()` for duplicates.
4. If two kernels write conflicting data to the same L1 address, suspect a use-after-free.

### Fix

Ensure each allocation is deallocated exactly once. Track buffer ownership carefully.

### Prevention

- Use RAII-style buffer management (buffers are freed when the owning object is destroyed).
- Add debug assertions that check for double-free: maintain a set of freed addresses and assert on re-free.
- Consider adding a `TT_ASSERT` to `deallocate` when the address is not found (rather than silent no-op).
- After deallocating a buffer, set the handle to an invalid value so it cannot be reused.
- Never pass device buffer addresses to kernels after the buffer has been deallocated.

---

## Hang Cause 3.4.5: CB Overflow -- Producer Overwrites Consumer Data

### Symptom

The compute kernel produces incorrect results or hangs at an unexpected location. The CB tile counts (`tiles_received` and `tiles_acked`) appear correct, but the data in the CB is corrupted. The corruption is not detected by the watcher's address sanitization because the addresses are within valid L1 bounds.

### Root Cause

If `cb_push_back` is called with more tiles than the consumer has popped (i.e., the CB is "full" but the producer pushes anyway without a preceding `cb_reserve_back`), the write pointer wraps and overwrites data the consumer has not yet read.

This is distinct from the CB *deadlocks* described in [Chapter 2, Section 02](../ch02_kernel_and_noc_hangs/02_circular_buffer_deadlocks.md). In a deadlock, the producer correctly waits via `cb_reserve_back` and the consumer correctly waits via `cb_wait_front`, but they disagree on tile counts. In an **overflow**, the producer bypasses the waiting mechanism entirely.

The `cb_push_back` function does not check whether the number of tiles pushed exceeds the CB's capacity -- it only asserts that `fifo_wr_ptr <= fifo_limit`:

```c++
// From dataflow_api.h, cb_push_back()
void cb_push_back(const int32_t operand, const int32_t num_pages) {
    uint32_t num_words = num_pages * get_local_cb_interface(operand).fifo_page_size;
    pages_received_ptr[0] += num_pages;
    get_local_cb_interface(operand).fifo_wr_ptr += num_words;

    // Only checks pointer bounds, not tile count vs capacity
    ASSERT(get_local_cb_interface(operand).fifo_wr_ptr <= get_local_cb_interface(operand).fifo_limit);
    if (get_local_cb_interface(operand).fifo_wr_ptr == get_local_cb_interface(operand).fifo_limit) {
        get_local_cb_interface(operand).fifo_wr_ptr -= get_local_cb_interface(operand).fifo_size;
    }
}
```

If the producer pushes tiles faster than the consumer pops them, `tiles_received` eventually exceeds `tiles_acked + fifo_num_pages`, meaning the CB logically contains more tiles than it has physical space for. The write pointer has wrapped and now points to data the consumer has not yet read.

**Buggy code:**
```c++
// Producer bypasses cb_reserve_back -- directly writes and pushes
for (uint32_t i = 0; i < num_tiles; i++) {
    uint32_t l1_addr = get_write_ptr(cb_out);
    noc_async_read(src_addr + i * tile_size, l1_addr, tile_size);
    noc_async_read_barrier();
    cb_push_back(cb_out, 1);  // BUG: no cb_reserve_back! May overflow if consumer is slow
}
```

**Corrected code:**
```c++
for (uint32_t i = 0; i < num_tiles; i++) {
    cb_reserve_back(cb_out, 1);  // Wait for space
    uint32_t l1_addr = get_write_ptr(cb_out);
    noc_async_read(src_addr + i * tile_size, l1_addr, tile_size);
    noc_async_read_barrier();
    cb_push_back(cb_out, 1);
}
```

### Diagnosis Steps

1. Check whether the kernel calls `cb_push_back` without a preceding `cb_reserve_back`.
2. Examine the CB state: if `tiles_received - tiles_acked > fifo_num_pages`, overflow has occurred.
3. Enable the watcher's CB sanitization to catch out-of-bounds NOC writes to CB regions.
4. Dump the CB data region and compare against expected tile contents.

### Fix

Always pair `cb_push_back` with a preceding `cb_reserve_back`.

### Prevention

- Enforce the `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front` protocol in code reviews.
- The `ASSERT` in `cb_push_back` catches some overflow cases in debug builds -- run tests with assertions enabled.
- Add a debug-only counter check that asserts `tiles_received - tiles_acked <= fifo_num_pages`.

---

## Hang Cause 3.4.6: CB Pop Underflow -- Consumer Reads Stale Data

### Symptom

The consumer reads tiles that have already been freed by a previous `cb_pop_front`. The data may have been overwritten by the producer, leading to the consumer processing a mixture of old and new data. This can cause incorrect address calculations and secondary hangs.

### Root Cause

If `cb_pop_front` is called with more tiles than were received, the read pointer advances past the write pointer:

```c++
// cb_pop_front
void cb_pop_front(int32_t operand, int32_t num_pages) {
    pages_acked_ptr[0] += num_pages;
    uint32_t num_words = num_pages * get_local_cb_interface(operand).fifo_page_size;
    get_local_cb_interface(operand).fifo_rd_ptr += num_words;

    ASSERT(get_local_cb_interface(operand).fifo_rd_ptr <= get_local_cb_interface(operand).fifo_limit);
    if (get_local_cb_interface(operand).fifo_rd_ptr == get_local_cb_interface(operand).fifo_limit) {
        get_local_cb_interface(operand).fifo_rd_ptr -= get_local_cb_interface(operand).fifo_size;
    }
}
```

Over-popping inflates `tiles_acked` beyond the actual number of tiles consumed. This has two effects:
1. The producer sees more free space than actually exists via `cb_reserve_back` and may write into the region the consumer is still reading.
2. The consumer's read pointer points to data that has not been written by the producer yet.

Both effects cause data corruption that leads to downstream hangs.

### Diagnosis Steps

1. Check whether `tiles_acked > tiles_received` (accounting for `uint16_t` wraparound).
2. Verify that `cb_pop_front` tile count matches the preceding `cb_wait_front` tile count.
3. Look for mismatched batch sizes between `cb_wait_front(cb, N)` and `cb_pop_front(cb, M)` where `M != N`.

### Fix

Ensure `cb_pop_front` tile count exactly matches the number of tiles actually consumed.

### Prevention

- Use matching constants for `cb_wait_front` and `cb_pop_front` batch sizes.
- Follow the strict producer-consumer protocol: always `cb_wait_front` before accessing tiles, and `cb_pop_front` with the exact number of tiles consumed.
- Add assertions in debug builds to verify `tiles_acked <= tiles_received`.

---

## Hang Cause 3.4.7: Watcher CB Sanitization Disabled -- Silent Out-of-Bounds

### Symptom

A CB out-of-bounds write is not caught by the watcher because CB sanitization has been disabled. The write corrupts adjacent L1 data, leading to a secondary hang that is much harder to diagnose.

### Root Cause

CB sanitization can be disabled via the environment variable `TT_METAL_WATCHER_DISABLE_CB_SANITIZE=1` or the compile-time flag `WATCHER_DISABLE_CB_SANITIZE`. When disabled, the `debug_valid_cb_addr` check is skipped:

```c++
#if !defined(WATCHER_DISABLE_CB_SANITIZE) && !defined(COMPILE_FOR_ERISC) && !defined(COMPILE_FOR_IDLE_ERISC)
    debug_sanitize_post_addr_and_hang(
        noc_id, noc_addr, worker_addr, len, multicast, dir,
        DEBUG_SANITIZE_NOC_LOCAL,
        debug_valid_cb_addr(worker_addr, len));
#endif
```

**Important:** CB sanitization only runs on BRISC/NCRISC (not on ERISC or TRISC). This is because only dataflow kernels (which run on BRISC/NCRISC) issue NOC transactions that interact with CBs. Compute kernels (TRISC) access CBs via the LLK interface, which has its own address validation.

Related watcher disable flags:
- `TT_METAL_WATCHER_DISABLE_SANITIZE_NOC`: Disables all NOC address sanitization.
- `TT_METAL_WATCHER_DISABLE_SANITIZE_READ_ONLY_L1`: Disables the read-only L1 region check (the `DebugSanitizeNocAddrMailbox` check from scenario 3.1.8).
- `TT_METAL_WATCHER_DISABLE_SANITIZE_WRITE_ONLY_L1`: Disables the write-only L1 region check.
- `TT_METAL_WATCHER_DISABLE_STACK_USAGE`: Disables stack usage tracking (scenario 3.1.5).

**Performance impact:** Each `debug_valid_cb_addr()` call iterates over all `NUM_CIRCULAR_BUFFERS` (32 on WH, 64 on BH/Quasar) for every NOC transaction. Combined with the other sanitization checks, the total overhead can be 10-30% of kernel execution time. This is acceptable for debug and CI builds but should be disabled for production performance measurements.

### Diagnosis Steps

1. Check the environment variables for any `WATCHER_DISABLE_*` flags.
2. Re-run the failing test with all sanitization checks enabled.
3. If `DebugSanitizeCBOutOfBounds` is now reported, CB overflow is the root cause.

### Fix

Enable CB sanitization and fix the underlying CB overflow.

### Prevention

- Do not disable CB sanitization in CI.
- Only disable specific sanitization checks when the performance impact is measured and the code paths are well-tested.
- Document why sanitization is disabled whenever `TT_METAL_WATCHER_DISABLE_*` is used.

**Recommended watcher configuration for debugging memory-related hangs:**
```bash
export TT_METAL_WATCHER=60              # Check every 60 seconds
# Keep all sanitization enabled (do not set any DISABLE flags)
# Optionally enable debug delays to expose timing-dependent bugs:
# export WATCHER_DEBUG_DELAY=100
```

---

## Hang Cause 3.4.8: Runtime Argument OOM -- Index Out of Bounds

### Symptom

The watcher reports `DebugAssertRtaOutOfBounds` or `DebugAssertCrtaOutOfBounds`. The core enters a deliberate hang via the assert mechanism.

### Root Cause

Runtime arguments (RTAs) and common runtime arguments (CRTAs) have a finite allocation in L1. The offsets are stored in `kernel_config_msg_t`:

```c++
struct rta_offset_t {
    volatile uint16_t rta_offset;
    volatile uint16_t crta_offset;
};
```

If a kernel calls `get_arg_val<T>(index)` with an index that would access beyond the allocated RTA region, the watcher's RTA bounds check detects the violation:

```c++
// From dev_msgs.h
enum debug_assert_type_t {
    // ...
    DebugAssertRtaOutOfBounds = 8,
    DebugAssertCrtaOutOfBounds = 9
};
```

This happens when:
1. The host-side `SetRuntimeArgs` call provides fewer arguments than the kernel expects.
2. The kernel's argument index is computed dynamically and exceeds the allocated count.
3. The RTA region allocation is too small due to L1 pressure.

### Diagnosis Steps

1. The watcher assert log identifies the core and assert type (`RtaOutOfBounds` or `CrtaOutOfBounds`).
2. Compare the kernel's `get_arg_val` calls against the number of arguments set by `SetRuntimeArgs`.
3. Check `rta_offset` and `crta_offset` in the launch message to determine the allocated region size.

### Fix

Ensure the number of runtime arguments set on the host matches the number accessed by the kernel.

### Prevention

- Use compile-time constants for the number of runtime arguments.
- Add a host-side check that validates argument count before kernel launch.

---

## Hang Cause 3.4.9: Allocator `shrink_size` / `reset_size` Mismatch

### Symptom

An allocation fails unexpectedly after a `shrink_size` call, or a buffer is allocated in a region that was supposed to be shrunk away. The `TT_FATAL` message reads `"Shrink size cuts into allocated block"`.

### Root Cause

The `FreeListOpt` allocator supports dynamic resizing via `shrink_size()` and `reset_size()`. These are used to temporarily reduce the allocatable region (e.g., to reserve space for dispatch kernel buffers) and later restore it:

```c++
// From free_list_opt.cpp
void FreeListOpt::shrink_size(DeviceAddr shrink_size, bool bottom_up) {
    TT_FATAL(bottom_up, "Shrinking from the top is currently not supported");
    // ... finds the block at the shrink boundary ...
    // ... reduces block_size and max_size_bytes ...
    block_size_[block_to_shrink] -= shrink_size;
    max_size_bytes_ -= shrink_size;
    shrink_size_ += shrink_size;
    // ...
}
```

Bugs can occur when:
1. `shrink_size` is called while allocations exist in the shrink region, triggering `TT_FATAL("Shrink size cuts into allocated block")`.
2. `reset_size()` is called but the adjacent block is allocated, so coalescing cannot restore the original contiguous block.
3. Multiple `shrink_size` calls accumulate without corresponding `reset_size` calls, progressively reducing the allocatable region.

### Diagnosis Steps

1. Check the `TT_FATAL` message for `"Shrink size cuts into allocated block"`.
2. Use `dump_blocks()` to visualize the block table before and after shrink operations.
3. Verify that `reset_size()` is called before the next allocation phase.
4. Check whether multiple `shrink_size` calls have accumulated.

### Fix

Ensure that shrink/reset operations are paired and that no allocations exist in the shrink region at the time of shrink.

### Prevention

- Use shrink/reset as a pair around well-defined phases (e.g., dispatch initialization).
- Do not interleave shrink operations with user allocations.
- Track accumulated shrink size and validate it has been fully reset before user allocations.

---

## Hang Cause 3.4.10: Allocator Metadata Exhaustion

### Symptom

The `FreeListOpt` allocator's internal metadata vectors grow unboundedly as allocations fragment the address space. Eventually, the metadata itself consumes significant host memory, and operations on the metadata (scanning for free blocks) become slow enough to cause host-side timeouts that appear as device hangs.

### Root Cause

Each allocation split creates new metadata blocks. The `alloc_meta_block` method grows the metadata vectors:

```c++
size_t FreeListOpt::alloc_meta_block(
    DeviceAddr address, DeviceAddr size, ssize_t prev_block, ssize_t next_block, bool is_allocated) {
    size_t idx;
    if (free_meta_block_indices_.empty()) {
        idx = block_address_.size();
        block_address_.push_back(address);       // grows unboundedly
        block_size_.push_back(size);
        // ... more push_backs ...
    } else {
        idx = free_meta_block_indices_.back();   // reuse freed metadata slot
        // ...
    }
    return idx;
}
```

The `free_meta_block_indices_` provides metadata slot recycling, but if the fragmentation pattern creates many blocks without freeing them, the metadata vectors grow.

Additionally, the `insert_block_to_segregated_list` method becomes expensive:

```c++
void FreeListOpt::insert_block_to_segregated_list(size_t block_index) {
    auto& free_blocks = free_blocks_segregated_by_size_[size_segregated_index];
    if (free_blocks.size() < 30) {
        // Linear scan for small lists
        for (it = free_blocks.begin(); it != free_blocks.end(); it++) { ... }
    } else {
        // Binary search for larger lists
        it = std::lower_bound(free_blocks.begin(), free_blocks.end(), block_index, ...);
    }
    free_blocks.insert(it, block_index);  // O(n) insert into vector
}
```

With thousands of fragments, this becomes O(n) per allocation/deallocation, causing host-side slowdowns.

### Diagnosis Steps

1. Profile host-side allocation time.
2. Call `dump_blocks()` to see the number of active metadata blocks.
3. Check `get_statistics()` for the fragmentation ratio: `total_free_bytes / largest_free_block_bytes`.

### Fix

- Call `allocator.clear()` periodically to reset the allocator when buffers can be re-allocated.
- Use larger minimum allocation sizes to reduce the number of splits.
- Restructure allocation patterns to enable coalescing.

### Prevention

- Monitor the metadata block count over the lifetime of long-running workloads.
- Set an upper bound on the expected number of concurrent allocations.
- Consider periodic defragmentation by reallocating all active buffers.

---

## Hang Cause 3.4.11: Size-Segregated List Boundary Issue

### Symptom

An allocation request that should succeed (based on total free space and largest free block size) unexpectedly returns `std::nullopt`. The allocator statistics show a block large enough exists, but the size-segregated search does not find it.

### Root Cause

The size-segregated lists use a logarithmic bucketing scheme:

```c++
size_t get_size_segregated_index(DeviceAddr size_bytes) const {
    size_t lg = 0;
    size_t n = size_bytes / size_segregated_base;  // base = 1024
    while (n >>= 1) { lg++; }
    return std::min(size_segregated_count - 1, lg);
}
```

A free block of size 2047 bytes maps to `lg = 0` (same bucket as 1024-byte blocks), while a request for 2048 bytes maps to `lg = 1`. The search starts at the request's bucket (`lg = 1`) and searches upward, missing the 2047-byte block in bucket 0. In most cases this is correct (the block genuinely is too small after alignment), but the issue becomes confusing when `BEST` fit policy is used and blocks at the boundary between two size classes appear to be large enough but are not found.

More significantly, when the alignment of the allocation (`align(size_bytes)`) causes the effective size to cross a boundary, the allocator may search a higher size class than needed. If the only suitable block is in the exact matching class but was placed in a lower class due to the unaligned original size, the search misses it.

### Diagnosis Steps

1. Check the size-segregated list contents using `dump_blocks()`.
2. Compute the expected size class for the request and verify which class contains the free blocks.
3. Check if the free block's size class differs from the request's size class by exactly one, indicating a boundary issue.

### Fix

Use allocation sizes that are multiples of `size_segregated_base` (1024 bytes) to avoid boundary confusion. The allocator search iterates from the request's size class upward, so blocks in larger classes will always be found.

### Prevention

- Use allocation sizes that are powers of 2 or multiples of 1024 bytes.
- Monitor the allocator statistics to verify that allocations succeed as expected.
- If unexpected OOM occurs near a class boundary, check the segregated list contents.

---

## Hang Cause 3.4.12: Silent Deallocation of Non-Allocated Address

### Symptom

A buffer is "freed" even though it was never allocated (or was already freed). A subsequent allocation receives the same address, leading to two buffers pointing to the same L1 or DRAM region. Data corruption and hangs follow.

### Root Cause

As shown in Hang Cause 3.4.4, `deallocate()` silently returns when the address is not in the allocation table. This silent no-op means three failure modes go undetected:
- Double-free (second call finds nothing and returns).
- Freeing a never-allocated address.
- Freeing an address that was re-allocated to a new owner -- the stale handle frees the new owner's block, producing overlapping allocations (identical to the 3.4.4 scenario).

Use-after-free is the most dangerous variant: the host passes a freed buffer's address to a kernel as a runtime argument while the allocator has already given that address to a different buffer.

### Diagnosis Steps

1. Add logging around `deallocate()` calls; warn when `get_and_remove_from_alloc_table` returns `std::nullopt`.
2. Track buffer ownership with a host-side debug layer mapping addresses to owning operations.
3. Use `is_address_in_alloc_table` to verify an address is still allocated before deallocating.

### Fix

Add strict ownership tracking. After deallocating a buffer, set the handle to an invalid value.

### Prevention

- Same as Hang Cause 3.4.4: use RAII wrappers, add `TT_ASSERT` on unrecognized deallocation addresses, and never pass device buffer addresses to kernels after deallocation.

---

## Summary Table

| ID | Hang Cause | Key Indicator | Host vs Device | Detection |
|----|-----------|---------------|----------------|-----------|
| 3.4.1 | L1 allocator OOM -> garbage address | NOC violation with random/zero address | Host allocation, device hang | Watcher NOC sanitize |
| 3.4.2 | DRAM allocation failure -> garbage address | NOC DRAM violation | Host allocation, device hang | Watcher NOC sanitize |
| 3.4.3 | Free-list fragmentation | `largest_free_block << total_free` | Host allocator | `get_statistics()` |
| 3.4.4 | Double-free -> overlapping allocations | Two buffers at same address | Host allocator | Address tracking |
| 3.4.5 | CB overflow (producer overwrites consumer) | `tiles_received - tiles_acked > fifo_num_pages` | Device | CB state dump |
| 3.4.6 | CB pop underflow (over-pop) | `tiles_acked > tiles_received` | Device | CB state dump |
| 3.4.7 | CB sanitize disabled -> silent OOB | No watcher detection | Device | Re-enable sanitize |
| 3.4.8 | RTA out of bounds | `DebugAssertRtaOutOfBounds` | Device | Watcher assert |
| 3.4.9 | Shrink/reset mismatch | `TT_FATAL` on shrink | Host allocator | Error message |
| 3.4.10 | Allocator metadata exhaustion | Slow host-side allocation | Host | Profiling, `dump_blocks()` |
| 3.4.11 | Size-segregated list boundary | Unexpected OOM near class boundary | Host | `dump_blocks()`, size class check |
| 3.4.12 | Silent deallocation of non-allocated addr | Two buffers at same address | Host | Address lifecycle tracking |

---

**Previous:** [`03_alignment_and_tile_size_mismatches.md`](./03_alignment_and_tile_size_mismatches.md)
**Next:** [`../ch04_dispatch_and_host_device_hangs/index.md`](../ch04_dispatch_and_host_device_hangs/index.md)
