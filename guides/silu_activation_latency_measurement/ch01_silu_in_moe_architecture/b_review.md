# B Review — Chapter 1: SiLU in MoE Architecture — Pass 1

## Verdict

Three correctness issues found. No structural gaps. No missing planned files.

---

## Issues

**1. `compute_role_and_cost_hypothesis.md`, line 15 — Bandwidth formula is missing the write pass**

The text states:

> applying SiLU to a `[1, 14336]` row takes approximately `14336 * 2 bytes / 2e12 ≈ 14 picoseconds`

`14336 * 2 bytes` accounts only for reading the input (bfloat16 = 2 bytes per element). SiLU also writes an output tensor of the same size, so the correct bandwidth-bound estimate is `14336 * 2 * 2 bytes / 2e12 ≈ 28 picoseconds` (or `14336 * 4 bytes / 2e12`). The qualitative conclusion ("negligibly small") is unaffected, but a reader who copies this formula to estimate GPU bandwidth for any activation will get a result that is off by 2×.

Fix: Change `14336 * 2 bytes` to `14336 * 2 * 2 bytes` (or `14336 * 4 bytes`) and update the result to `~28 picoseconds`.

---

**2. `compute_role_and_cost_hypothesis.md`, line 82 — "On-chip DRAM" does not exist on Wormhole**

The text states:

> the L1 working set may exceed per-core SRAM and require on-chip DRAM (L2 or DRAM) access

Wormhole's memory hierarchy is: L1 SRAM (per Tensix core) → DRAM (off-chip HBM/LPDDR). There is no on-chip DRAM and no L2 cache in the conventional sense. A reader building a latency or memory-bandwidth model for Wormhole based on this sentence will incorrectly assume an intermediate cache tier with on-chip latency, leading to wrong performance estimates.

Fix: Replace "on-chip DRAM (L2 or DRAM)" with "off-chip DRAM" (or name the specific DRAM type: LPDDR4 on Wormhole Galaxy boards).

---

**3. `swiglu_variant.md`, line 63 — Dense FFN parameter count uses `[d_model, 4*d_model]` for both W_up and W_down, obscuring the transpose**

The text states:

> Dense FFN: two matrices of `[d_model, 4*d_model]` each

W_up has shape `[d_model, d_ffn]` but W_down has shape `[d_ffn, d_model]` — the transpose. Listing both as `[d_model, 4*d_model]` gives the correct element count (and thus the correct parameter count) only because multiplication is commutative for scalar products, but it misrepresents the actual shapes. A reader who takes these shapes literally and writes code that applies W_down as `[d_model, 4*d_model]` will produce a shape mismatch error (`[T, 4*d_model] @ [d_model, 4*d_model]` is invalid). The SwiGLU side of the same table already uses correct notation (`[d_model, (8/3)*d_model]` for all three projections, implicitly treating W_down's transpose shape the same way), so the inconsistency is not immediately visible to a reader who trusts the example.

Fix: Clarify that W_down is `[d_ffn, d_model]` (or `[4*d_model, d_model]` in the example), or add a parenthetical noting that W_down is the transpose of W_up for the purpose of this parameter-count example.

---

# B Review — Chapter 1: SiLU in MoE Architecture — Pass 2

## Pass 1 issue verification

All three Pass 1 issues are confirmed fixed:
- Issue 1 (`compute_role_and_cost_hypothesis.md` line 15): formula now reads `14336 * 4 bytes / 2e12` — confirmed.
- Issue 2 (`compute_role_and_cost_hypothesis.md` line 82): "off-chip DRAM" — confirmed.
- Issue 3 (`swiglu_variant.md` line 62): W_down now shown as `[d_ffn, d_model]` = `[4*d_model, d_model]` — confirmed.

---

## New Issues

**1. `compute_role_and_cost_hypothesis.md`, line 15 — Result unit is wrong by a factor of 1000**

The text now reads:

> `14336 * 4 bytes / 2e12 ≈ 28 picoseconds`

The arithmetic is: `14336 * 4 = 57344 bytes`; `57344 / 2e12 bytes/s = 2.87e-8 seconds = ~28.7 nanoseconds`. The stated unit "picoseconds" is wrong by a factor of 1000 (1 nanosecond = 1000 picoseconds). A reader who takes this at face value will believe the bandwidth-bound time is 1000× smaller than it actually is and will under-estimate GPU activation latency by three orders of magnitude.

Fix: Change "28 picoseconds" to "~29 nanoseconds" (or "~28.7 ns").

---

**2. `swiglu_variant.md`, line 63 — SwiGLU W_down shape still misrepresented as `[d_model, (8/3)*d_model]`**

The fix from Pass 1 correctly updated the dense FFN row to show W_down as `[4*d_model, d_model]`. However, the SwiGLU row on the immediately following line still reads:

> SwiGLU FFN: three matrices of `[d_model, (8/3)*d_model]`

W_gate and W_up are `[d_model, (8/3)*d_model]`, but W_down must be `[(8/3)*d_model, d_model]` for the matmul `[T, (8/3)*d_model] @ [(8/3)*d_model, d_model]` to be valid. The parameter count in the formula is numerically correct regardless, but stating "three matrices of `[d_model, (8/3)*d_model]`" will lead a reader who implements this to give W_down the wrong shape — the identical shape mismatch bug that motivated the Pass 1 fix on the dense side.

Fix: Change "three matrices of `[d_model, (8/3)*d_model]`" to "W_gate and W_up each `[d_model, (8/3)*d_model]`, W_down `[(8/3)*d_model, d_model]`" — or add a parenthetical: "(W_down is the transpose: `[(8/3)*d_model, d_model]`)".

# B Review — Chapter 1: SiLU in MoE Architecture — Pass 3

## Pass 2 issue verification

Both Pass 2 fixes are confirmed in place:

- Fix 1 (`compute_role_and_cost_hypothesis.md` line 15): Text now reads "~29 nanoseconds (~28.7 ns)" — confirmed. Arithmetic (`14336 * 4 bytes / 2e12 = 28.7e-9 s`) is correct and units are correct.
- Fix 2 (`swiglu_variant.md` line 63): Text now reads "W_gate and W_up each `[d_model, (8/3)*d_model]`, W_down `[(8/3)*d_model, d_model]`" — confirmed. W_gate, W_up, and W_down shapes are now correctly distinguished.

## New Issues

No feedback — chapter approved.

