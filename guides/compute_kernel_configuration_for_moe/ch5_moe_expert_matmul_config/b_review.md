# B Review — Chapter 5: MoE Expert Matmul Configuration — [PENDING]

# B Review — Chapter 5: MoE Expert Matmul Config — Pass 1

## Verdict

5 errors found — all in `qwen_moe_current_state.md`. The file uses Qwen2-57B-A14B dimensions throughout, but the authoritative reference model for this chapter is Qwen 235B-A22B. This produces a cascade of wrong values for `d_model`, `d_ff`, `num_experts`, and derived quantities (`K_t` for down projection, `K_t` for gate/up projections).

**Error 1**
- File: `qwen_moe_current_state.md`, line 29
- Stated: Reference model is `Qwen2-57B-A14B`
- Correct: The authoritative model is Qwen 235B-A22B (Qwen2-235B-A22B)

**Error 2**
- File: `qwen_moe_current_state.md`, line 30
- Stated: `d_model = 2048`
- Correct: `d_model = 7168` (for Qwen 235B-A22B)

**Error 3**
- File: `qwen_moe_current_state.md`, line 31
- Stated: `d_ff per expert = 768`
- Correct: `d_ff per expert = 2048` (for Qwen 235B-A22B)

**Error 4**
- File: `qwen_moe_current_state.md`, line 32
- Stated: `num_experts = 64`
- Correct: `num_experts = 128` (for Qwen 235B-A22B)

**Error 5**
- File: `qwen_moe_current_state.md`, line 34 and line 51
- Stated: `K_t = d_ff / 32 = 24` for down projection; `K_t = d_model / 32 = 2048 / 32 = 64` for gate/up projections
- Correct: With d_ff=2048, K_t for down projection = 2048/32 = **64** (matching DeepSeek-V3, not 24). With d_model=7168, K_t for gate/up projections = 7168/32 = **224** (not 64). The 98.4% savings figure cited for gate/up is also wrong for the correct model; it only holds incidentally for the wrong (57B-A14B) model. The down projection savings cited as 95.8% (23/24) must be recomputed from the correct K_t=64, yielding 63/64 = 98.4%.

---

## Agent A Change Log — B Feedback Pass 1

1. **Replace the reference model**: Change `Qwen2-57B-A14B` to `Qwen2-235B-A22B` (Qwen 235B-A22B) throughout `qwen_moe_current_state.md`.
2. **Correct `d_model`**: Change `2048` to `7168`.
3. **Correct `d_ff per expert`**: Change `768` to `2048`.
4. **Correct `num_experts`**: Change `64` to `128`.
5. **Recompute all K_t values and bandwidth savings**:
   - Down projection: K_t = 2048/32 = 64; savings = 63/64 = 98.4% (update the 95.8% figure and remove the "K_t=24" / "23 unnecessary DRAM round-trips" language).
   - Gate/up projections: K_t = d_model/32 = 7168/32 = 224; savings = 223/224 ≈ 99.6% (replace the incorrect 98.4% figure and the erroneous `d_model=2048` substitution on line 51).
   - Update the Summary table (line 67-69) to reflect corrected savings percentages.

---

## Agent A Change Log — B Feedback Pass 1
- qwen_moe_current_state.md: Fixed model reference from Qwen2-57B-A14B to Qwen2-235B-A22B
- qwen_moe_current_state.md: Fixed d_model from 2048 to 7168
- qwen_moe_current_state.md: Fixed d_ff per expert from 768 to 2048
- qwen_moe_current_state.md: Fixed num_experts from 64 to 128
- qwen_moe_current_state.md: Fixed K_t(down)=64/98.4%; K_t(gate/up)=224/99.6%

---

# B Review — Chapter 5: MoE Expert Matmul Config — Pass 2

## Pass 1 Fix Verification

**Fix 1 — Model name (Qwen2-57B-A14B → Qwen2-235B-A22B):** APPLIED. `qwen_moe_current_state.md` line 26 reads "Qwen2-235B-A22B (Qwen MoE 235B-A22B)". Correct.

**Fix 2 — d_model (2048 → 7168):** APPLIED. Dimensions table shows `d_model = 7168`. Correct.

**Fix 3 — d_ff per expert (768 → 2048):** APPLIED. Dimensions table shows `d_ff per expert = 2048`. Correct.

**Fix 4 — num_experts (64 → 128):** APPLIED. Dimensions table shows `Number of experts = 128`. Correct.

**Fix 5 — K_t(down)=64/98.4% and K_t(gate/up)=224/99.6%:** APPLIED. Down projection: K_t=64, savings=63/64=98.4% (line 34, 45). Gate/up projections: K_t=7168/32=224, savings=223/224≈99.6% (line 49). Summary table (lines 65–67) reflects both corrected figures. Correct.

All 5 Pass 1 fixes confirmed applied correctly.

## No feedback — chapter approved.
