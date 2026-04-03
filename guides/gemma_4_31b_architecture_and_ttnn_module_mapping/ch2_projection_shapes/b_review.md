# Agent B Review: Chapter 2 — Projection Weights and Tensor Shapes

## Pass 1

### Issue 1 — Global layer total parameter count is wrong (index.md, line 109)

The "Per-Layer Parameter Counts — Global Layer" table states the total as **539,067,352**, but the correct sum of the individual rows is **534,009,856**.

Arithmetic:

| Component | Parameters |
|-----------|-----------|
| Q | 88,080,384 |
| K | 11,010,048 |
| V | 0 |
| O | 88,080,384 |
| Gate | 115,605,504 |
| Up | 115,605,504 |
| Down | 115,605,504 |
| Norms | 22,528 |
| **Correct total** | **534,009,856** |

The stated value is 5,057,496 higher than the correct sum. A reader using the per-layer total for memory budgeting would over-estimate global layer weight memory by ~10 MB (BF16) per layer, or ~100 MB across 10 global layers.

### Issue 2 — Full-model formula uses a third, different wrong number (index.md, line 114)

The full-model formula plugs in `539,067,104` for the global layer parameter count:

```
50 × 478,959,104 + 10 × 539,067,104 + 262,144 × 5,376 = 30,747,047,944
```

This value (539,067,104) does not match the table total (539,067,352) and does not match the correct value (534,009,856). There are now three distinct numbers for the same quantity. The correct formula and result should be:

```
50 × 478,959,104 + 10 × 534,009,856 + 262,144 × 5,376 = 30,697,339,904 ≈ 30.7B
```

The rounded approximation (30.7B) happens to survive the error, but the explicit arithmetic is wrong and will mislead anyone who cross-checks it.

## Pass 2

Both Pass 1 issues have been corrected. Verified all remaining arithmetic across all four files:

- Sliding layer total (478,959,104) matches row-by-row sum. Correct.
- Global layer total (534,009,856) matches row-by-row sum. Correct.
- Full-model formula (50 x 478,959,104 + 10 x 534,009,856 + 262,144 x 5,376 = 30,697,339,904) is correct.
- FFN per-layer total (346,816,512) matches row-by-row sum. Correct.
- All individual projection parameter counts (e.g., 5376 x 8192 = 44,040,192) are correct.
- All BF16 byte values are 2x the parameter counts. Correct.
- Norm parameter counts (sliding: 5376 x 4 + 256 x 2 = 22,016; global: 5376 x 4 + 512 x 2 = 22,528) are correct.
- Fused QKV output dims (sliding: 8192+4096+4096 = 16384; global: 16384+2048 = 18432) are correct.
- Fused gate+up shape (43008) and TP=8 per-device shards (2688, 5376) are correct.
- FLOPs claims (gate+up fused: 57.8M; down: 28.9M) are correct.

**No feedback — chapter approved.**
