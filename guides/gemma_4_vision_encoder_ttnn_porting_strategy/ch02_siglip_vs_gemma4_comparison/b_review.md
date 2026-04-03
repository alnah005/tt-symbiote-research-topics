# Agent B Review: Chapter 2

## Pass 1

### Issue 1 — Incorrect padding amount in tile alignment table (config_diff.md, line 111)

The table states Gemma 3's 588-dimensional patch input needs "4 elements to reach 608" for tile alignment. The correct padding is 608 - 588 = **20 elements**, not 4. Fix the cell to read "20 elements to reach 608".

### Issue 2 — Transposed weight shape for Gemma 4 patch embedding (config_diff.md, line 68)

The table lists the Gemma 4 linear weight tensor shape as `[768, 1152]`. However, `nn.Linear(768, 1152)` stores its weight as `[1152, 768]` (PyTorch convention: `[out_features, in_features]`). The parameter count (884,736) is correct either way, but the shape should read `[1152, 768]` to be factually accurate and consistent with the `nn.Linear(768, 1152)` declaration on the same row.

### Issue 3 — Longest RoPE wavelength arithmetic is off (positional_encoding_shift.md, line 126)

The text claims the longest wavelength (i=17) is approximately 527 positions. Using the formula on line 117, wavelength_17 = 2 * pi * 100^(34/36) = 2 * pi * 10^(1.889) = 2 * pi * 77.4 ≈ **487 positions**. The value 527 appears to be a miscalculation. Correct it to approximately 487.

No other factual issues found. All shared/changed/new parameter classifications, sequence-length calculations, memory estimates, and module-mapping descriptions are consistent with the approved Chapter 1 facts.

## Pass 2

All three Pass 1 issues have been verified as fixed:

1. Tile padding now correctly reads "20 elements to reach 608" (config_diff.md, line 111).
2. Weight shape now correctly reads `[1152, 768]` (config_diff.md, line 68).
3. Longest RoPE wavelength now correctly reads approximately 487 (positional_encoding_shift.md, line 126).

**No feedback — chapter approved.**
