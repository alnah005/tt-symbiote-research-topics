# Chapter 7 — Agent B (Critic) Review: Factual Correctness

## Verdict

Two factual issues found.

## Issues

### Issue 1: MLP description in phased_plan.md omits the gate projection

**Location:** `phased_plan.md`, line 145 (Optimization Priorities, item 2)

**Text:** "MLP matmuls (up-projection 1152->4304, down-projection 4304->1152)"

**Problem:** The Gemma 4 MLP is a gated MLP with three projections: gate (1152->4304), up (1152->4304), and down (4304->1152). The guide's own Chapter 6 (`direct_reuse_modules.md`) confirms the structure as `down(gelu(gate(x)) * up(x))` with three weight matrices. Describing only two projections understates the MLP compute by roughly one-third and may mislead an engineer sizing sharding strategies or estimating FLOP counts.

**Suggested fix:** Replace with "MLP matmuls (gate-projection 1152->4304, up-projection 1152->4304, down-projection 4304->1152)".

### Issue 2: Vision encoder parameter count inconsistency in risk_register.md

**Location:** `risk_register.md`, line 116

**Text:** "The vision encoder has approximately 550M parameters"

**Problem:** Chapter 1 (`config_parameters.md`) computes the vision encoder total as ~569M parameters (or ~575M including the multimodal projection). Stating "approximately 550M" is a ~19M / ~3.3% undercount. While the surrounding claim ("roughly 2% of the 31B total") still holds either way, the parameter count should be consistent with the figure established in Chapter 1.

**Suggested fix:** Replace "approximately 550M" with "approximately 570M".

## Pass 2

Both Pass 1 issues have been fixed:

1. **MLP gate projection (phased_plan.md, line 145):** Now correctly lists all three projections: "gate-projection 1152->4304, up-projection 1152->4304, down-projection 4304->1152". Verified.
2. **Parameter count (risk_register.md, line 116):** Now reads "approximately 570M parameters", consistent with the Chapter 1 derivation. Verified.

No new factual issues found across index.md, phased_plan.md, or risk_register.md. Checked: phase counts, risk counts, timeline arithmetic, PCC thresholds, tile-alignment calculations, parameter-count ratios, data-transfer size estimates, MLP dimensions, head-dim padding overhead percentages, and RoPE identity-padding values. All are internally consistent and consistent with prior chapters.

**No feedback — chapter approved.**
