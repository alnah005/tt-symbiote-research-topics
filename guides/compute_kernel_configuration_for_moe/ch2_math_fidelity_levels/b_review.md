# B Review — Chapter 2: Math Fidelity Levels — Pass 1

1. **`fidelity_and_moe_accuracy.md`, "How K-Loop Depth Amplifies Rounding" section — sqrt(K_t) model contradicts the actual K_t values used to argue down-projection sensitivity.**

   The section invokes "the PCC degradation from LoFi scales roughly with `sqrt(K_t)`" as the mechanism behind down projection sensitivity. It then presents a table where gate/up have K_t=224 and down projection has K_t=64. Under the stated sqrt(K_t) model, gate/up projections accumulate more noise (sqrt(224) ≈ 15) than down projections (sqrt(64) = 8). Yet the text lists "K-loop depth" as factor 1 explaining why down projections are more accuracy-sensitive. This is internally contradictory: if error scales with sqrt(K_t), then the shallower K-loop (down, K_t=64) should be less noisy under that model, not more. The real explanation — stated correctly elsewhere — is the absence of a downstream nonlinearity, not K-loop depth. As written, a reader applying the sqrt(K_t) formula to the table values will reach the opposite conclusion from the one the text asserts. The section should either drop "K-loop depth" as a sensitivity driver for down projections (given the actual K_t values), or explicitly note that K-loop depth is secondary here and the SiLU absorption effect dominates even though K_t is larger for gate/up.

2. **`fidelity_precision_model.md`, HiFi2 throughput claim — relative throughput value is inconsistent with the table.**

   The prose for HiFi2 states "Approximate throughput relative to LoFi: ~0.85× (slightly slower than LoFi, but meaningfully faster than HiFi4)." The table then shows HiFi4 at ~0.5×. A gap from 0.85× (HiFi2) to 0.5× (HiFi4) — a 35-percentage-point drop — is not a minor step. By contrast, HiFi3 is listed at ~0.7×. This makes HiFi2 described as "slightly slower than LoFi" while actually being nearly as fast, and HiFi4 described without comment despite being roughly 40% slower than HiFi2. The characterization "slightly slower than LoFi" for ~0.85× is not a factual error in the number itself, but the prose for HiFi3 (~0.7×) and HiFi4 (~0.5×) has no equivalent hedging phrase, making the relative positions inconsistent in how they are narrated. More concretely: if HiFi2 is ~0.85× and HiFi4 is ~0.5×, then HiFi2 is approximately 1.7× faster than HiFi4 — not a marginal difference. The text should not imply the gap between HiFi2 and HiFi4 is small.

   **Note:** Items 1 and 2 are the only substantive factual issues found. All other verified claims are correct:
   - `torch.corrcoef(torch.stack([a_flat, b_flat]))[0, 1]` — shape [2, N], index [0, 1] (off-diagonal of 2×2 matrix) is correct.
   - `ttnn.WormholeComputeKernelConfig` constructor uses all required keyword arguments correctly in both code examples.
   - `math_fidelity` scope (FPU multiply-accumulate only; does not affect SFPU) is stated correctly.
   - LoFi ~2× over HiFi4 throughput is stated as approximate throughout.
   - PCC range ~0.99–0.999 for LoFi on a 4096×2048 matmul matches the reference table.
   - sqrt(K_t) model is stated as approximate ("roughly", "heuristic" framing via the CLT reference).
   - MoE accuracy reasoning (gate/up tolerate LoFi via SiLU; down needs HiFi2 for residual stream) is correct.

## Agent A Change Log — B Feedback Pass 1
- fidelity_and_moe_accuracy.md: Removed K-loop depth argument from down projection fidelity justification (K_t=64 for down < K_t=224 for gate/up; argument contradicts sqrt(K_t) model); replaced with residual stream sensitivity and lack of nonlinearity absorption as the primary reason
- fidelity_precision_model.md: Clarified HiFi2 throughput from "slightly slower than LoFi" to "approximately 0.85× of LoFi throughput, significantly faster than HiFi4 (~0.5×)"

---

# B Review — Chapter 2: Math Fidelity Levels — Pass 2

## Pass 1 Fix Verification

**Fix 1 — `fidelity_and_moe_accuracy.md` K-loop depth argument:** Confirmed applied. The "Why Down Projections Need HiFi2" section now leads with "Direct residual stream injection" as the primary sensitivity driver, with no mention of K-loop depth as a reason down projection is more sensitive. The separate "K-Loop Depth and Nonlinearity Absorption" section correctly frames sqrt(K_t) as a secondary heuristic, explicitly notes that gate/up have the deeper K-loop (K_t=224 vs K_t=64 for down), and states that SiLU absorption dominates over K-loop depth in determining fidelity sensitivity. The logic is now internally consistent.

**Fix 2 — `fidelity_precision_model.md` HiFi2 throughput description:** Confirmed applied. Line 30 now reads: "Approximate throughput relative to LoFi: ~0.85× — a meaningful reduction from LoFi, but significantly faster than HiFi4 (which runs at ~0.5×, making HiFi2 approximately 1.7× faster than HiFi4)." The "slightly slower than LoFi" language has been removed. The description is now accurate.

## Additional Checks

- `torch.corrcoef(torch.stack([a_flat, b_flat]))[0, 1]` — shape [2, N], index [0, 1] is the off-diagonal of a 2×2 correlation matrix. Correct in both uses (`fidelity_and_moe_accuracy.md` and `fidelity_selection_workflow.md`).
- `ttnn.WormholeComputeKernelConfig` constructor syntax — all four keyword arguments (`math_fidelity`, `math_approx_mode`, `fp32_dest_acc_en`, `packer_l1_acc`) are present and correctly formed in `fidelity_selection_workflow.md`. Correct.
- LoFi ~2× over HiFi4 throughput claim — stated as approximate in both the prose ("approximately 50% of the cycles") and the table (~0.5× for HiFi4). Consistent.

## Minor Observation (Non-Blocking)

`index.md`, line 49 describes `fidelity_and_moe_accuracy.md` as covering "K-loop depth and rounding accumulation" as a key topic alongside "the presence or absence of a downstream nonlinearity." After the Pass 1 fix, K-loop depth is now explicitly a secondary heuristic in that file, not a primary argument. The index description creates a mild expectation mismatch for a reader who has not yet read the file. This is not a factual error in the technical content and does not affect correctness of the chapter — it is a metadata/navigation note only.

## Verdict

All Pass 1 fixes verified. No new factual errors found — chapter approved.
