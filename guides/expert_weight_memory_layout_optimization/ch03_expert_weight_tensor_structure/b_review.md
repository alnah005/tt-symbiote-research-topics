# B Review — Chapter 3: Expert Weight Tensor Structure — Pass 1

## Verdict

Two confirmed factual errors found; one additional minor arithmetic inconsistency noted.

---

## Error 1 — DeepSeek-MoE-16B per-expert BF16 memory calculation is wrong (SERIOUS)

**File:** `dtype_and_tile_layout.md`, line 195 (footnote) and the summary table at line 192.

**Claimed:** `3 * 2048 * 1408 * 2 = ~12.0 MiB` per expert.

**Actual arithmetic:**
```
3 * 2048 * 1408 * 2 = 17,301,504 bytes
17,301,504 / 1,048,576 = ~16.5 MiB
```

The file states ~12.0 MiB, which is off by ~38%. The value ~12.0 MiB would correspond to d_ff ≈ 1024, not 1408. This is an arithmetic error — the formula is printed correctly but the result is wrong.

---

## Error 2 — DeepSeek-MoE-16B total BF16 budget in summary table is wrong (SERIOUS)

**File:** `dtype_and_tile_layout.md`, line 192.

**Claimed:** Total BF16 for 64 experts = ~0.75 GiB; Total BF8 = ~0.37 GiB.

**Actual arithmetic (using the correct per-expert figure of ~16.5 MiB):**
```
64 * 17,301,504 bytes = 1,107,296,256 bytes
1,107,296,256 / (1024^3) = ~1.03 GiB (BF16)
~0.52 GiB (BF8)
```

The ~0.75 GiB figure flows directly from the wrong per-expert value in Error 1 and is equally incorrect.

---

## Error 3 — "Commonly cited 351.5 MB" figure is wrong (MINOR)

**File:** `dtype_and_tile_layout.md`, line 158.

**Claimed:** "The commonly cited '351.5 MB' figure uses 1 MB = 1,000,000 bytes (SI units)."

**Actual arithmetic:**
```
352,321,536 bytes / 1,000,000 = 352.32 MB  (SI)
```

The SI value is ~352.3 MB, not 351.5 MB. The 351.5 figure does not correspond to any standard unit convention applied to the correct byte count. This is a minor error but should be corrected to avoid confusing readers who try to reproduce the figure.

---

## No Other Errors Found

The following facts were verified and are correct:

- Projection shapes (w1/w3: `[d_model, d_ff]`; w2: `[d_ff, d_model]`) — correct throughout all files.
- Stacked tensor conventions (`[num_experts, d_model, d_ff]` for gate/up; `[num_experts, d_ff, d_model]` for down) — correct.
- Mixtral 8x7B dimensions (d_model=4096, d_ff=14336, num_experts=8, top_k=2) — correct.
- DeepSeek-MoE-16B dimensions (d_model=2048, d_ff=1408) — correctly stated.
- Qwen MoE (235B-A22B) dimensions (d_model=7168, d_ff=2048, num_experts=128, top_k=8) — correct.
- Mixtral per-expert total BF16 bytes: 352,321,536 (~335.9 MiB) — correct.
- Mixtral BF8 total: 176,160,768 bytes (~168.0 MiB) — correct.
- Tile byte sizes (BF16: 2,048 bytes; BF8: 1,024 bytes; BF4: 512 bytes per 32×32 tile) — correct.
- Wormhole B0 hardware: 6 DRAM controllers, 12 GDDR6 banks — correct.
- Worked sharding example: [4096, 14336] WIDTH_SHARDED 8 cores → shard [4096, 1792]; 1792 % 32 = 0 — correct.
- WIDTH_SHARDED constraint formulas — correct.

## Agent A Change Log — B Feedback Pass 1
- dtype_and_tile_layout.md: Fixed DeepSeek-MoE-16B per-expert BF16: 3*2048*1408*2 = 17,301,504 bytes = ~16.5 MiB (was ~12.0 MiB)
- dtype_and_tile_layout.md: Fixed total BF16 for 64 experts: ~1.03 GiB (was ~0.75 GiB); BF8: ~0.52 GiB (was ~0.37 GiB)
- dtype_and_tile_layout.md: Fixed SI byte count: 352.32 MB ≈ 352.3 MB (was 351.5 MB)

---

# B Review — Chapter 3: Expert Weight Tensor Structure — Pass 2

Pass 1 fixes verified. No feedback — chapter approved.

**Pass 1 fix verification:**

- Error 1 fix confirmed: `dtype_and_tile_layout.md` line 195 now reads `3 * 2048 * 1408 * 2 = 17,301,504 bytes = ~16.5 MiB`. Correct.
- Error 2 fix confirmed: summary table line 192 now shows DeepSeek-MoE-16B as `~16.5 MiB*`, `~1.03 GiB`, `~0.52 GiB`. Correct.
- Error 3 fix confirmed: line 158 now reads `"352.3 MB"` and `352,321,536 / 1,000,000 = 352.32 MB ≈ 352.3 MB`. Correct.

**Full scan — no new errors found.** All facts verified against the reference values:

- Model dimensions (d_model, d_ff, num_experts) in `index.md` reference table — correct.
- Mixtral per-expert bytes: `3 * 4096 * 14336 * 2 = 352,321,536` (~335.9 MiB BF16); BF8 ~168.0 MiB; BF4 ~84.0 MiB — correct throughout.
- Mixtral totals (8 experts): `~2.6 GiB` BF16, `~1.3 GiB` BF8 — correctly rounded from 2.625 GiB and 1.312 GiB respectively.
- Qwen MoE per-expert bytes: `3 * 7168 * 2048 * 2 = 88,080,384` (~84.0 MiB BF16) — correct.
- Qwen totals (128 experts): `~10.5 GiB` BF16, `~5.25 GiB` BF8 — correct.
- Tile byte sizes (BF16: 2,048; BF8: 1,024; BF4: 512 per 32×32 tile) — correct.
- Wormhole B0: 6 DRAM controllers, 12 GDDR6 banks — correct (`tensor_to_shard_grid_mapping.md` lines 34–35).
- Projection shapes w1/w3 = `[d_model, d_ff]`; w2 = `[d_ff, d_model]` — correct throughout all files.
- Stacked tensor conventions `[num_experts, d_model, d_ff]` for gate/up; `[num_experts, d_ff, d_model]` for down — correct.
- Worked sharding example: `[4096, 14336]` WIDTH_SHARDED 8 cores → shard `[4096, 1792]`; `1792 % 32 == 0` — correct.

## Agent A Change Log — B Feedback Pass 2

No fixes required — chapter approved.
