# B Review — Chapter 7 — Pass 1

## Item 1 (REJECTED — not an error) — `rotary_dim = 64` claim

**File:** `existing_ttnn_primitives_survey.md`, Section 7.

**B's claim:** `rotary_dim = 64` is wrong for Gated Attention with `d_h = 256`; should be 256.

**Rejection reason:** This is intentional partial RoPE. Chapter 3 (`gated_attention_formulation.md`, Section 5) establishes that Gated Attention applies RoPE to only 64 of the 256 head dimensions — a design choice in Qwen3.5 to reduce the fraction of head capacity consumed by positional encoding. `rotary_dim = 64 = d_h / 4` is correct.

**No changes required.**

---

No other issues found by Pass 1 review.

**No feedback — chapter approved.**
