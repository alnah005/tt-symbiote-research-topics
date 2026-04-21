## B Feedback — Pass 1

1. **`prefill_vs_decode_comparison.md`, Image/Video Prefill section** — The formula "Extra random-access data (h + w): 2 × S × rotary_dim/2 × 2 bytes = 2 × 1024 × 32 × 2 bytes = 128 KiB" is numerically incorrect. Using the consistent convention established in `operation_cost_breakdown.md` (each section uses both cos and sin tables: `2*s_i × 2_tables × 2 bytes` per position per section):
   - Height section: `2*s_h × 2 × 2 = 2×11×2×2 = 88 bytes` per position
   - Width section: `2*s_w × 2 × 2 = 2×10×2×2 = 80 bytes` per position
   - h+w total: 168 bytes per position → at S=1024: **~168 KiB**, not 128 KiB.
   The formula incorrectly uses `rotary_dim/2 = 32` as the per-section dimension, ignoring the cos+sin factor that is consistently included elsewhere. The conclusion (< 2 µs at ~150 GB/s) is unaffected (168 KiB / 150 GB/s ≈ 1.1 µs), but the formula should be corrected for consistency with `operation_cost_breakdown.md`. Fix: correct the formula and result to `S × (2*s_h + 2*s_w) × 2 tables × 2 bytes = 1024 × (22 + 20) × 2 × 2 bytes ≈ 168 KiB`.

## B Feedback Application Log — Pass 1

- Fix 1: Corrected the formula in `prefill_vs_decode_comparison.md` Image/Video Prefill section from `2 × S × rotary_dim/2 × 2 bytes = 128 KiB` to `S × (2*s_h + 2*s_w) × 2 tables × 2 bytes = 1024 × 42 × 2 × 2 bytes ≈ 168 KiB`; updated the inline timing from `128 KiB / 150 GB/s ≈ 0.85 µs` to `168 KiB / 150 GB/s ≈ 1.1 µs`.

## B Feedback — Pass 2

No feedback — chapter approved.
