# Program Cache Issues in Paged SDPA Decode

---

## Issue #21534 — Cache Miss / Wrong Cache Count with BFP8 K/V and BF16 Q

### Symptom

When paged attention is called with:

```
Q dtype:   BF16
K/V dtype: BFP8_B (block floating point 8-bit)
```

tests reported two distinct failures:

1. **Program cache miss on repeated calls**: the compiled kernel was not reused
   on the second call, causing an unexpected recompilation and a test assertion
   on the program cache hit count.

2. **Incorrect cache hit count**: in some configurations the cache count
   incremented by 2 on a single call, suggesting the kernel was being registered
   under two distinct keys.

### Root Cause

The cache key did not correctly encode the mixed-dtype configuration.  BFP8 K/V
with BF16 Q was treated as a distinct case from BF16/BF16 at key-generation time
but the generated kernel was identical in some code paths, causing inconsistent
key collisions.

### Status

Fixed.  The fix landed as part of broader program cache refactoring.  If you see
unexpected cache miss counts with mixed dtype configurations, confirm the TTNN
build includes the #21534 fix.

---

## Page Table Shape Not Included in Cache Key (Pre-#12330)

### Symptom

In early TTNN versions that supported paged attention before the #12330 work,
`page_table_tensor` shape was not included in the program cache key.  If two
consecutive calls used page tables of different shapes (different
`max_num_blocks_per_seq` values, e.g., because one sequence was shorter and used
a smaller pre-allocated table), the second call reused the compiled kernel from
the first call.

The compiled kernel had the wrong loop bounds for the block iteration, causing
it to either read too few blocks (missing valid KV data) or iterate beyond the
allocated table (reading garbage block indices).

### Example

```
Call 1: page_table shape [4 x 32]   → compiled kernel iterates up to 32 blocks
Call 2: page_table shape [4 x 64]   → reuses Call 1 kernel; only reads 32 of 64 blocks
         → tokens in blocks 33–64 are silently ignored
```

### Fix

Page table shape (specifically `max_num_blocks_per_seq`) was added to the
program cache key as part of the #12330 native GQA work.  Calls with different
page table shapes now generate distinct compiled kernels.

---

## Issue #16674 — `paged_update_cache` Hang on Blackhole Hardware

### Symptom

On Blackhole hardware (not Wormhole B0), calls to `paged_update_cache` hang
indefinitely — the device does not complete the operation and no timeout is
triggered in the versions affected.

Note: Wormhole B0 uses 6 DRAM controllers (not 8) and 12 GDDR6 banks.
Blackhole has a different memory subsystem, and the hang is specific to that
architecture.

### Impact on End-to-End Decode

`paged_update_cache` must complete before `sdpa_decode` is called — the write
step fills the KV cache slot that the attention step will read.  A hang in
`paged_update_cache` stalls the entire decode pipeline.  This issue is
unrelated to GQA correctness but is relevant to anyone deploying paged GQA
decode on Blackhole hardware.

### Status

Open as of early 2026.  No upstream fix merged.

**Workaround**: on Blackhole hardware, fall back to a host-side KV cache write
followed by a full device tensor transfer, bypassing `paged_update_cache`.  This
is significantly slower but unblocks functional testing.

---

## Program Cache Diagnostic Snippet

To confirm program cache behavior is correct across calls:

```python
import ttnn

ttnn.enable_program_cache(device)
cache_before = ttnn.num_program_cache_entries(device)

output_1 = sdpa_decode(..., cur_pos=pos_1)
cache_after_first = ttnn.num_program_cache_entries(device)

output_2 = sdpa_decode(..., cur_pos=pos_2)   # same shapes, different cur_pos
cache_after_second = ttnn.num_program_cache_entries(device)

assert cache_after_first - cache_before == 1,      "unexpected compile count on first call"
assert cache_after_second == cache_after_first,    "unexpected recompile on second call"
```

If `cur_pos` is passed as a Python list (not `cur_pos_tensor`), a new compile
is triggered for each unique `cur_pos` tuple.  Use `cur_pos_tensor` (device
tensor) to avoid recompilation across positions.

---

**Next:** [Chapter 6 — Debugging Guide](../ch6_debugging/index.md)
