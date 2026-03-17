# B Review — Chapter 5: Tile Size Constraints — Pass 1

## Verdict
1 error found.

### Error 1
- **File:** `shard_shape_alignment_rules.md`
- **Line:** 186
- **What was stated:** `14_680_064 % 32 == 0  OK  (14,680,064 = 32 * 459,002)`
- **What is correct:** 14,680,064 / 32 = 458,752, not 459,002. The factorization should read `(14,680,064 = 32 * 458,752)`. Verification: 32 × 459,002 = 14,688,064, which does not equal 14,680,064. The divisibility claim (% 32 == 0) is itself correct, but the illustrative factorization is arithmetically wrong.

---

## Agent A Change Log — B Feedback Pass 1

1. In `shard_shape_alignment_rules.md`, Step 6 of the Mixtral worked example, replace the parenthetical factorization:
   - Current: `(14,680,064 = 32 * 459,002)`
   - Corrected: `(14,680,064 = 32 * 458,752)`

## Agent A Change Log — B Feedback Pass 1
- shard_shape_alignment_rules.md: Fixed inline factorization from 32 × 459,002 to 32 × 458,752 (14,680,064 / 32 = 458,752)

---

# B Review — Chapter 5: Tile Size Constraints — Pass 2

## Pass 1 Fix Verification

`shard_shape_alignment_rules.md` line 186 now reads:

```
14_680_064 % 32 == 0  OK  (14,680,064 = 32 * 458,752)
```

Fix was applied correctly. `32 × 458,752 = 14,680,064` is arithmetically exact and consistent with the authoritative value `4096 × 1792 × 2 = 14,680,064`.

## No feedback — chapter approved.

All authoritative facts verified across all four files:

- Tile byte sizes: BF16 2,048 bytes, bfloat8_b 1,024 bytes, bfloat4_b 512 bytes — correct in `index.md` and `tile_fundamentals.md`.
- Wormhole B0 hardware constants: 80 Tensix cores (8×10), 1.5 MB L1/core, 12 DRAM banks (6 controllers × 2 banks), 32-byte page size — all correct in `index.md` and `common_pitfalls.md`.
- Mixtral shapes: d_model=4096, d_ff=14336; M_t=128, N_t=448 — correct in `shard_shape_alignment_rules.md` and `tile_fundamentals.md`.
- DeepSeek-V3/Qwen shapes: d_model=7168, d_ff=2048, num_experts=128; M_t=224, N_t=64 — correct in `shard_shape_alignment_rules.md` and `common_pitfalls.md`.
- Derived values in `common_pitfalls.md`: 128×7168=917,504 and 917,504/8=114,688 — both correct.

No remaining errors found.
