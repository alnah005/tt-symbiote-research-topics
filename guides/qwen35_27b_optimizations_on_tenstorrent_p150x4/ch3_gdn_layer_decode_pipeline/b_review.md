# Chapter 3 Review -- Correctness

## Issue 1: Recurrence state memory per layer is overstated

**File:** `recurrence_math.md`, "Recurrence State Shape and Memory" section

The chapter states the per-layer recurrence state is "approximately 12.6 MB with any alignment overhead" and extrapolates to "~605 MB" across 48 GDN layers. However, earlier in the same section it correctly computes `384 pairs * 32 KB = 12,288 KB = 12 MB` and notes "no padding overhead since both dimensions are exact multiples of the tile size." If there is no tile padding overhead, the per-layer figure should be 12 MB (not 12.6 MB), and the 48-layer total should be ~576 MB (not ~605 MB). The 12.6 MB figure implies ~5% alignment overhead that contradicts the chapter's own observation about exact tile alignment.

**Fix:** Change "approximately 12.6 MB with any alignment overhead" to "12 MB" and "~605 MB" to "~576 MB", or provide a concrete justification for the extra overhead (e.g., tensor header metadata, DRAM bank alignment).

---

No other factual errors found. The dimension calculations, operation sequences, gate equations, shift register mechanics, and fused/unfused path descriptions all match the source code in `gdn.py` and `model_config.py`.
