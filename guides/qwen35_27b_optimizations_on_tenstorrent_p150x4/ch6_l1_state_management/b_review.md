# Chapter 6 Review — Correctness

## Issue 1: Writer kernel compile-time arg index is wrong

**File**: `height_sharded_kernel.md`, line 9 and surrounding code block

The chapter states that `STATE_IS_SHARDED` is at `get_compile_time_arg_val(10)` for both the reader and writer kernels, citing `writer_gdn_fused.cpp`, line 25. In the actual writer source, `STATE_IS_SHARDED` is at compile-time arg index **6**, not 10:

```cpp
// writer_gdn_fused.cpp, line 25
constexpr uint32_t STATE_IS_SHARDED = get_compile_time_arg_val(6);
```

Only the reader uses index 10. The text and code block should distinguish the two kernels.

## Issue 2: Output tensor shape contradicts writer kernel

**File**: `height_sharded_kernel.md`, line 93

The chapter states: "The output tensor `[num_pairs, 1, Dv]` feeds into the subsequent RMS norm and output projection."

The writer kernel header comment (`writer_gdn_fused.cpp`, lines 6-8) explicitly says the opposite:

```
// Writes output tiles to [1, B, value_dim_tp] layout (not [num_pairs, 1, Dv]),
// mapping pair -> (batch_idx, v_head) to place tiles at correct positions.
```

The output layout is `[1, B, value_dim_tp]`, not `[num_pairs, 1, Dv]`.

## Issue 3: Reader kernel sharded-path line range is slightly off

**File**: `height_sharded_kernel.md`, line 29

The chapter cites "lines 268-282" for the HEIGHT_SHARDED path in `reader_gdn_fused.cpp`. The actual `if constexpr (STATE_IS_SHARDED)` block spans lines 268-277. Lines 278-282 are the `else` branch (the non-sharded NOC path). Minor, but the cited range should be 268-277.

---

No other factual errors found. The rolling-window design description, swap mechanism, forward-pass hook logic, memory lifecycle, and SDPA conflict analysis all match the source code accurately.
