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

## Pass 1

**1. Wrong per-layer state size and derived total — `l1_state_design.md`, first paragraph**

The chapter states "approximately 12.6 MB per layer" and a "total state footprint of ~600 MB." The actual size of `rec_states` at `[384, 128, 128]` bfloat16 is:

$$384 \times 128 \times 128 \times 2 = 12{,}582{,}912 \text{ bytes} = 12.0 \text{ MB per layer (exactly)}$$

The correct total for 48 layers is $48 \times 12.0 = 576\ \text{MB}$, not ~600 MB. A reader using the 12.6 MB figure for SRAM/DRAM budget calculations would be off by ~5%. (Criterion a — wrong numerical answer.)

No other factual errors found. The rolling-window design (`enable_l1_state`, `_swap_l1_state`, forward hook, swap timing table), HEIGHT_SHARDED kernel path, shard-config arithmetic, and SDPA conflict analysis all match the source code accurately. All four planned files are present. Navigation footers are present on all content files. All `index.md` file references are clickable links.

## Pass 2

**Pass 1 fix verified.** `l1_state_design.md` first paragraph now reads "12.0 MB per layer" (12,582,912 bytes) and "576 MB" total. Both figures are correct and the fix is confirmed.

**No new issues found.**

All claims cross-checked against source:

- `enable_l1_state()` line range (220-249), `_l1_window = 3` at line 227, `_l1_current_start = 0` at line 248 — all confirmed in `model.py`.
- `_swap_l1_state()` line range (251-278), `output_tensor` usage at line 267, `ttnn.deallocate` at line 268, bounds checks at lines 262-263 and 274 — all confirmed.
- `forward()` guard at line 291, `make_wrapped_forward` factory pattern (lines 321-328), `try/finally` (lines 334-345) — all confirmed.
- `test_e2e_l1_rolling.py` claim "up to 4 layers": test loops `gdn_indices[:4]` (line 79). Consistent.
- `test_e2e_l1_hs.py` `N_L1_LAYERS = 2` (line 42), `assert is_l1` at line 141 — both confirmed.
- HEIGHT_SHARDED shard math: `total_rows = 32 * 12 * 128 = 49152`, `shard_h = 512`, 96-core grid, 128 KB per core, 4 pairs per core — all arithmetic correct and consistent with source lines 86-94.
- `sdpa_l1_conflict.md` `ttnn.deallocate` cite at "line 268 of model.py" — confirmed.
- Navigation footer present on all three content files; all `index.md` links are clickable.

**Chapter 6 approved.**

## Pass 3

**No feedback — chapter approved.**

All claims re-verified against source:

- `l1_state_design.md`: per-layer state size (12.0 MB, 12,582,912 bytes), total 576 MB, NOC transaction counts (384 pairs × 32 transactions = 12,288 per layer, × 48 = 589,824 ≈ 590,000), and DRAM bandwidth (1.2 GB) all check out arithmetically.
- `_l1_current_start` semantics: guide says "0-based index into groups of 3" (i.e., block number). Confirmed: `forward()` stores the block number (0, 1, 2 …) at line 345 and multiplies by W when calling `_swap_l1_state`. Consistent.
- `height_sharded_kernel.md`: HEIGHT_SHARDED shard math (`total_rows = 32 * 12 * 128 = 49152`, `shard_h = 512`, 4 pairs per core at 128 KB each), compile-time arg indices (reader index 10, writer index 6), and output shape `[1, B, value_dim_tp]` all confirmed in source.
- `sdpa_l1_conflict.md`: `ttnn.deallocate` cite at model.py line 268, INTERLEAVED 4-layer / HEIGHT_SHARDED 1-2-layer status table, and SDPA CB watermark description are all consistent with `test_e2e_l1_rolling.py` and `test_e2e_l1_hs.py`.
- Navigation footers present on all three content files; all `index.md` links are clickable; no plain-text display equations found.
