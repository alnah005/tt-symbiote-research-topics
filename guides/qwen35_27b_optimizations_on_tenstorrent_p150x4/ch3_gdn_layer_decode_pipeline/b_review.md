# Chapter 3 Review -- Correctness

## Issue 1: Recurrence state memory per layer is overstated

**File:** `recurrence_math.md`, "Recurrence State Shape and Memory" section

The chapter states the per-layer recurrence state is "approximately 12.6 MB with any alignment overhead" and extrapolates to "~605 MB" across 48 GDN layers. However, earlier in the same section it correctly computes `384 pairs * 32 KB = 12,288 KB = 12 MB` and notes "no padding overhead since both dimensions are exact multiples of the tile size." If there is no tile padding overhead, the per-layer figure should be 12 MB (not 12.6 MB), and the 48-layer total should be ~576 MB (not ~605 MB). The 12.6 MB figure implies ~5% alignment overhead that contradicts the chapter's own observation about exact tile alignment.

**Fix:** Change "approximately 12.6 MB with any alignment overhead" to "12 MB" and "~605 MB" to "~576 MB", or provide a concrete justification for the extra overhead (e.g., tensor header metadata, DRAM bank alignment).

---

No other factual errors found. The dimension calculations, operation sequences, gate equations, shift register mechanics, and fused/unfused path descriptions all match the source code in `gdn.py` and `model_config.py`.

## Pass 3

**Pass 2 remaining issue — `norm_w`/`rms_scale_tt`/`rms_eps_tt` attribution:** Confirmed fixed. `gdn_decode_flow.md` (lines 85–99) now explicitly states that these arguments are passed into the fused kernel to support the post-kernel RMS norm step, and that they do **not** control the kernel's internal L2 normalization of Q and K — that normalization is weightless. The correction is accurate and clearly written. Resolved.

**New issue — `rms_scale_tt` and `rms_eps_tt` are misdescribed as inputs to the post-kernel `ttnn.rms_norm` call.**

**File:** `gdn_decode_flow.md`, fused-path kernel arguments (lines 85–86)

The guide states: "`rms_scale_tt` and `rms_eps_tt` supply the scale and epsilon for that same call" (i.e., the post-kernel `ttnn.rms_norm`). This is factually wrong. The source code shows:

- `self.rms_scale_tt = _scalar_to_mesh(math.sqrt(self.Dv))` — a scalar tile holding `sqrt(128) ≈ 11.31` (`gdn.py` line 161)
- `self.rms_eps_tt = _scalar_to_mesh(self.Dv * 1e-6)` — a scalar tile holding `128 * 1e-6 = 1.28e-4` (`gdn.py` line 162)

Both are passed as arguments to `gdn_full_fused_inplace` (line 316), i.e., they are consumed inside the fused kernel itself. The actual post-kernel `ttnn.rms_norm` call (line 330) uses a hardcoded `epsilon=1e-6`, not `rms_eps_tt`, and has no separate scale argument — `norm_w` serves as the per-element learned scale. `rms_scale_tt` and `rms_eps_tt` are therefore kernel-internal parameters, not post-kernel `ttnn` call parameters. Describing them as "the scale and epsilon for that same [post-kernel] call" is incorrect and will mislead readers trying to trace the code.

**Fix:** Change the description to state that `rms_scale_tt` (`sqrt(Dv)`) and `rms_eps_tt` (`Dv * 1e-6`) are kernel-internal arguments whose exact use is detailed in Chapter 4, and that the post-kernel `ttnn.rms_norm` uses hardcoded `epsilon=1e-6` with `norm_w` as its learned weight.

## Pass 4

Pass 3 issue confirmed fixed. `gdn_decode_flow.md` (lines 85–86) now explicitly states that `rms_scale_tt` and `rms_eps_tt` are **kernel-internal scalar tiles** passed to `gdn_full_fused_inplace` — `rms_scale_tt = sqrt(Dv)` and `rms_eps_tt = Dv * 1e-6` — and that the post-kernel `ttnn.rms_norm` uses hardcoded `epsilon=1e-6` and receives neither of these tensors. This matches `gdn.py` lines 161–162 and line 330. Resolved.

No new issues found. All formulas, dimension calculations, gate equations, shift register mechanics, fused/unfused path descriptions, navigation footers, and index links are correct. Chapter approved.

**No feedback — chapter approved.**

## Pass 5

Pass 4 approval confirmed. All previously raised issues remain resolved in the current chapter text:

- L2 norm formula (`recurrence_math.md`): `q / sqrt(||q||^2 + ε)` correctly matches `_l2_norm_dev` in `gdn.py` lines 58–67. ✓
- `rms_scale_tt` / `rms_eps_tt` attribution (`gdn_decode_flow.md` lines 85–86): correctly described as kernel-internal scalar tiles (`sqrt(Dv)` and `Dv * 1e-6`), with the post-kernel `ttnn.rms_norm` using hardcoded `epsilon=1e-6`. Matches `gdn.py` lines 161–162 and line 330. ✓
- Memory figures (`recurrence_math.md`): 12 MB per layer, 576 MB across 48 layers — correct and self-consistent. ✓
- All dimension arithmetic verified against `model_config.py` constants (`GDN_QKV_DIM=10240`, `GDN_Z_DIM=6144`, `qkvz_dim_tp=4096`, `qkv_dim_tp=2560`). ✓
- Conv shift register copy order (oldest-first: 0→1→2→3) matches `gdn.py` lines 281–284. ✓
- All navigation footers present; all `index.md` links are clickable relative paths; no plain-text display equations found. ✓

No new issues found.

**No feedback — chapter approved.**

## Pass 1

1. **L2 norm formula is wrong (`recurrence_math.md`, "Preprocessing" section).** The guide writes `q_normed = q / (||q||_2 + ε)`, i.e., epsilon added to the norm after taking the square root. The implementation in `_l2_norm_dev` (`gdn.py` lines 58–67) computes `inv = rsqrt(sum(x*x) + 1e-6)`, i.e., epsilon is added to the *sum of squares* before the square root, giving `q / sqrt(||q||^2 + ε)`. These are numerically different — at small magnitudes the two formulas produce different outputs. A reader implementing the displayed formula would not match the code. The correct display equation is: `q_normed = q / sqrt(||q||^2 + ε)`.

2. **`gdn_decode_flow.md` states `norm_w` is used inside the fused kernel for a "recurrence readout" normalization, yet post-kernel code also calls `ttnn.rms_norm(..., weight=tw["norm_w"])` on the same buffer.** The guide (lines 82–87) lists `tw["norm_w"]` and `self.rms_scale_tt` / `self.rms_eps_tt` as kernel arguments and implies the kernel produces a normalized output. However, source lines 329–330 show a separate `ttnn.rms_norm` call applied *after* the kernel returns. A reader of the fused-path description would reasonably conclude the kernel outputs a fully normalized result and would not expect an additional post-kernel RMS norm call; this is a material misunderstanding of which component performs the normalization. The guide should clarify that the fused kernel outputs raw recurrence readout (one `[1, Dv]` per pair) and the `ttnn.rms_norm` in Stage 4 is responsible for the normalization, with `norm_w`/`rms_scale_tt`/`rms_eps_tt` passed to the kernel serving a different internal purpose.

3. **`gdn_decode_flow.md` gate formula mismatch.** Line 104 in `gdn_decode_flow.md` refers readers to `recurrence_math.md` for the gate formula, and `recurrence_math.md` (line 71) correctly shows `g = neg_exp_A * softplus(a + dt_bias)`. However, `gdn_decode_flow.md` line 86 describes `self.neg_exp_A` as "precomputed `-exp(A_log)`" (correct per `gdn.py` lines 146–148), while the fused-path summary table in `recurrence_math.md` (line 172) labels the fused phase as "`exp`, `log1p`, mul by `neg_exp_A`". The `softplus(x) = log(1 + exp(x))` decomposition means "exp then log1p" is the correct order in the kernel. No cross-document conflict here — this is internally consistent. (No action required; included for completeness of audit trail.)

## Pass 2

Both Pass 1 issues are confirmed fixed.

**Pass 1 Issue 1 — L2 norm formula:** `recurrence_math.md` now displays `q / sqrt(||q||^2 + ε)` (and matching formula for k), which matches `_l2_norm_dev` in `gdn.py` lines 58–67 (`rsqrt(sum(x*x) + 1e-6)`). Resolved.

**Pass 1 Issue 2 — RMS norm ownership:** `gdn_decode_flow.md` now contains an explicit paragraph (after the kernel call block, lines ~85–99) stating that `norm_w`, `rms_scale_tt`, `rms_eps_tt` are passed to the kernel for L2 normalization of Q and K *within* the kernel, and that the post-kernel `ttnn.rms_norm` is a separate `ttnn` operation on the host-visible output. The Stage 4 section (lines ~125–132) reinforces this with the code snippet and plain-English label. Resolved.

**Remaining issue — `norm_w` purpose description is technically imprecise (`gdn_decode_flow.md`, fused-path kernel arguments).**

The guide states that `tw["norm_w"]`, `self.rms_scale_tt`, and `self.rms_eps_tt` "supply the L2 normalization of Q and K within the recurrence itself." `norm_w` is a learned per-element weight loaded for RMS norm (it is used in `ttnn.rms_norm(..., weight=tw["norm_w"])` at lines 330 and 471). An L2 normalization does not use a learned weight — the kernel receives `norm_w` for an internal normalization step, but calling this "L2 normalization of Q and K" conflates two distinct operations. The unfused path L2-normalizes Q and K via `_l2_norm_dev`, which uses no weight parameter at all. Describing `norm_w` as a parameter to the kernel's Q/K L2 norm misleads readers who know that L2 norm is weightless. The correct description is that the kernel receives these arguments for an internal normalization step whose exact type (RMS vs L2) is defined in the Chapter 4 kernel detail, and should not be characterized here as "L2 normalization of Q and K." This is the only unfixed correctness issue. No structural gaps, missing footers, or plain-text display equations were found in any file.
