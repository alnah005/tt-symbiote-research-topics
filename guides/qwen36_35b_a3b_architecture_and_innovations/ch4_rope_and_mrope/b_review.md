# Critic Review — Chapter 4 (Pass 1)

## Issue 1 — Frequency ordering is inverted throughout partial_rotary_embedding.md (CRITICAL)

**File:** `partial_rotary_embedding.md`, "Frequency Spectrum" section — table column header and surrounding prose.

**Problem:** The table labels pair i=0 as "lowest frequency pair" and pair i=31 as "highest frequency pair". The prose reinforces this: "The slowest pair (i=0) has period 2π positions" and "the fastest pair (i=31) has a period of roughly 40 million positions."

This is backwards. θ_i = rope_theta^(-2i/rotary_dim) is largest at i=0 (θ_0 = 1.0) and smallest at i=31 (θ_31 ≈ 1.58 × 10^-7). Because the rotation angle per position is m·θ_i, larger θ_i means faster oscillation. i=0 is the **highest** frequency pair (period 2π ≈ 6.28 positions — shortest period, fastest change). i=31 is the **lowest** frequency pair (period 2π/(1.58×10^-7) ≈ 40M positions — longest period, slowest change).

A reader implementing a frequency analysis or debugging the spectrum would have the ordering exactly wrong.

**Fix:** Swap the labels: i=0 → "highest frequency pair", i=31 → "lowest frequency pair". In prose, replace "The slowest pair (i=0)" with "The fastest pair (i=0)" and "the fastest pair (i=31)" with "the slowest pair (i=31)". The conceptual point being made (that even the slowest-changing pair barely completes one cycle across the 262K context) is correct — only the labels are inverted.

---

## Issue 2 — Frequency table: i=1 approximate value is wrong (partial_rotary_embedding.md)

**File:** `partial_rotary_embedding.md`, frequency table, row for i=1.

**Problem:** The table states θ_1 ≈ 0.0708. The correct calculation is:

    θ_1 = (10^7)^(-2/64) = 10^(-14/64) = 10^(-0.21875) ≈ 0.604

0.0708 ≈ 10^(-1.15), which does not correspond to any standard evaluation of the formula at i=1. An implementer spot-checking their frequency precomputation against this table would conclude their correct output is wrong.

The values for i=15 (≈ 5.62 × 10^-4) and i=31 (≈ 1.58 × 10^-7) are approximately correct (within rounding); only the i=1 entry is wrong.

**Fix:** Change the i=1 row approximate value from ≈ 0.0708 to ≈ 0.604.

---

## Issue 3 — Navigation "Previous" link in index.md points to wrong chapter (index.md)

**File:** `index.md`, Navigation section.

**Problem:** The footer reads:

    Previous: Chapter 3 — Gated DeltaNet Layers (../ch3_gated_deltanet/index.md)

Per the plan, Chapter 3 is "Qwen3.5 vs Qwen3.6: Exact Differences" located at `ch3_qwen35_vs_qwen36_differences/`. Chapter 2 is the Gated DeltaNet chapter. Both the chapter name and the directory path in the link are wrong.

A reader following the "Previous" link will navigate to a non-existent path and see no content, or (if the directory does not exist) get a 404.

**Fix:** Change the Previous entry to:

    Previous: [Chapter 3 — Qwen3.5 vs Qwen3.6 Differences](../ch3_qwen35_vs_qwen36_differences/index.md)

---

# Critic Review — Chapter 4 (Pass 2)

## Issue 4 — "Why Not Full RoPE" paragraph inverts the index-frequency relationship (CRITICAL)

**File:** `partial_rotary_embedding.md`, "Why Not Full RoPE on 256 Dims?" section (line ~83).

**Problem:** The paragraph states:

> "With full RoPE on d_h = 256, there would be 128 frequency pairs. Pairs at **high indices** would have extremely high base frequencies (short periods), introducing rapid oscillations in the attention logits."

This is factually inverted. The formula θ_i = rope_theta^(-2i/rotary_dim) makes θ_i a decreasing function of i. High-index pairs have the **lowest** θ_i values and therefore the **slowest** oscillation (longest periods). Low-index pairs (i near 0) have the highest θ_i and the fastest oscillation.

After Pass 1 corrected the frequency table to label i=0 as "highest frequency pair" and i=31 as "lowest frequency pair", this paragraph now directly contradicts the table in the same file. A reader who internalized the corrected table will recognize the contradiction; one who read the paragraph first will be re-confused.

The correct argument for why full RoPE on 256 dims is undesirable at long context is: with a small rope_theta (e.g., 10,000), even the fastest pairs (i=0, low index) cycle many times over a 262K-token context, causing the inner product to average out — making distant tokens indistinguishable. The large rope_theta of 10M mitigates this for the 64-dim subspace; expanding back to 256 dims would introduce more slow-frequency pairs (high index), which are not harmful but are unnecessary, while the partial restriction keeps the positional signal concentrated.

**Fix:** Replace "Pairs at high indices would have extremely high base frequencies (short periods)" with "Pairs at low indices have the highest base frequencies (shortest periods)". Revise the surrounding argument to correctly attribute the rapid-oscillation problem to low-index (high-frequency) pairs.

---

## Issue 5 — Frequency table: i=15 approximate value is off by ~7%

**File:** `partial_rotary_embedding.md`, "Frequency Spectrum" table, row for i=15.

**Problem:** The table states θ_15 ≈ 5.62 × 10^-4. The correct calculation is:

    θ_15 = 10^(7 × (−2×15/64)) = 10^(−210/64) = 10^(−3.28125) ≈ 5.24 × 10^-4

5.62 × 10^-4 = 10^-3.2503, which corresponds to an exponent of −3.2503, not −3.28125. The error is approximately 7%. The values for i=0 (1.0) and i=31 (≈ 1.58 × 10^-7) are correct; only the i=15 entry is wrong.

An implementer spot-checking their precomputed frequency array against the table at the midpoint index would find a ~7% discrepancy with a correct implementation and may incorrectly suspect a bug.

**Fix:** Change the i=15 row approximate value from ≈ 5.62 × 10^-4 to ≈ 5.24 × 10^-4.

---

# Critic Review — Chapter 4 (Pass 3)

## Issue 6 — Hidden dimension, query head count, and head dimension are mutually inconsistent (CRITICAL)

**File:** `partial_rotary_embedding.md`, "Dimensions and Parameters" section (lines 12–17).

**Problem:** The document states all three of the following simultaneously:

- Hidden dimension H = 2048
- Query heads n_q = 16
- Head dimension d_h = 256 **(= H / n_q)**

The inline formula annotation `(= H / n_q)` is wrong: 2048 / 16 = 128, not 256. One of the three values must be incorrect. An implementer who trusts all three will compute d_h = 128 (from H and n_q) or set H = 4096 (from n_q and d_h) — either path diverges from what the document intends and will produce a mismatched projection layer.

The downstream consequences are significant: `rotary_dim = d_h × 0.25` depends on d_h directly, so a wrong d_h changes the cos/sin cache shape, the split point between `h_rot` and `h_pass`, and every shape assertion in the implementation.

**Fix:** Audit the actual model config for the correct pair of (H, n_q) that yields d_h = 256. If d_h = 256 is correct, then either H = 4096 (with n_q = 16) or n_q = 8 (with H = 2048). Update whichever value is wrong and remove or correct the formula annotation `(= H / n_q)` to match.

---

## Issue 7 — Frequency table: i=31 approximate value is wrong

**File:** `partial_rotary_embedding.md`, "Frequency Spectrum" table, row for i=31.

**Problem:** The table states θ_31 ≈ 1.58 × 10^-7. The correct calculation is:

    θ_31 = 10^(7 × (−2 × 31 / 64)) = 10^(−6.78125) ≈ 1.655 × 10^-7

1.58 × 10^-7 = 10^-6.8013, which is a ~4.5% underestimate of the correct value. The error is not attributable to rounding (1.655 × 10^-7 rounds to 1.66 × 10^-7 at three significant figures, not 1.58 × 10^-7). This value was present and uncorrected in Passes 1 and 2 (which fixed i=1 and i=15 respectively).

An implementer spot-checking the lowest-frequency pair against the table would find a ~4.5% discrepancy with a correct implementation.

**Fix:** Change the i=31 row approximate value from ≈ 1.58 × 10^-7 to ≈ 1.65 × 10^-7.

---

# Critic Review — Chapter 4 (Pass 4)

No feedback — chapter approved.
