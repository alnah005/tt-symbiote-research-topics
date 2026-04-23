# Chapter 6 — Recommendations and Implementation Guide

This chapter consolidates all findings from the preceding five chapters into a concrete fix recommendation, a precondition policy, and a verification checklist. Every research question from the original topic is answered in the summary table below before the derivations are presented in the chapter files.

---

> **Key Finding:** The correct fix is Strategy C — precompute a cos/sin table of shape `[max_seq_len, head_dim]` with identity values (`cos=1.0, sin=0.0`) at passthrough positions and real rotation values at the `rotary_dim/2` rotation positions. This construction is correct for any `rotary_dim <= head_dim` with `rotary_dim % 2 == 0` (including non-tile-aligned values), adds no runtime overhead, and is fully trace-compatible. For currently supported Qwen3-family models the bug is latent dead code, so the fix is not urgent — but it should be in place before any new model with non-tile-aligned `rotary_dim` is brought up.

---

## Research Question Summary Table

| Research question | Concise answer | Reference |
|---|---|---|
| What exact shape does `ttnn.experimental.rotary_embedding` require for cos/sin? | `cos.shape[-1] == head_dim`, enforced by `TT_FATAL` in `RotaryEmbeddingOperation::invoke` and `RotaryEmbedding::validate`. | Ch2: [`../ch2_op_shape_contract/shape_validation_in_invoke.md`](../ch2_op_shape_contract/shape_validation_in_invoke.md) |
| Does zero-padding cos/sin from `rotary_dim` to `nearest_32(rotary_dim)` produce correct partial RoPE? | No. The padding target is wrong (`nearest_32(rotary_dim)` instead of `head_dim`), and even if cos/sin were padded all the way to `head_dim` with zeros, the kernel's fixed `head_dim/2`-split pairing still produces wrong output at all 128 positions. | Ch3: [`../ch3_bug_root_cause/step_by_step_failure_trace.md`](../ch3_bug_root_cause/step_by_step_failure_trace.md), [`../ch3_bug_root_cause/correct_partial_rope_reference.md`](../ch3_bug_root_cause/correct_partial_rope_reference.md) |
| Which implementation strategy should be used? | Strategy C: identity-filled precomputed cos/sin table of shape `[max_seq_len, head_dim]`. Correct for all `rotary_dim <= head_dim` configurations, trace-compatible, zero runtime overhead. | Ch4: [`../ch4_implementation_strategies/strategy_c_precomputed_full_head_cos_sin.md`](../ch4_implementation_strategies/strategy_c_precomputed_full_head_cos_sin.md); Ch6: [`recommended_fix.md`](./recommended_fix.md) |
| Should `TTNNRotaryPositionEmbedding` enforce `rotary_dim % 32 == 0`? | No. Strategy C handles non-tile-aligned `rotary_dim` correctly. Enforce `head_dim % 64 == 0` (the op's two-tile constraint) instead. A warning (not an error) when `rotary_dim % 32 != 0` is appropriate to surface unexpected configurations. | Ch4: [`../ch4_implementation_strategies/strategy_b_enforce_tile_alignment.md`](../ch4_implementation_strategies/strategy_b_enforce_tile_alignment.md); Ch6: [`precondition_policy.md`](./precondition_policy.md) |
| Is the bug currently affecting any production-supported model? | No. All Qwen3-family models in tt-symbiote use `partial_rotary_factor=0.5, head_dim=128 → rotary_dim=64`, which is tile-aligned. The bug is latent dead code. | Ch5: [`../ch5_model_config_audit/is_this_dead_code.md`](../ch5_model_config_audit/is_this_dead_code.md) |

---

## Recap of All Prior Chapter Prerequisites

- **Ch1** established the partial RoPE math (rotate-half within `rotary_dim`, passthrough for `[rotary_dim, head_dim)`), the tile-alignment requirement (`TILE_WIDTH=32`), and the `TTNNRotaryPositionEmbedding` zero-padding behavior. ([`../ch1_rope_fundamentals/index.md`](../ch1_rope_fundamentals/index.md))
- **Ch2** traced `ttnn.experimental.rotary_embedding` through C++ validation to the compute kernel, establishing: `cos.shape[-1] == head_dim` is enforced by `TT_FATAL`; `head_dim % 64 == 0` is required; the kernel's rotate-half split is always at `head_dim/2`, not `rotary_dim/2`. ([`../ch2_op_shape_contract/index.md`](../ch2_op_shape_contract/index.md))
- **Ch3** derived the two failure paths for `rotary_dim=48, head_dim=128` (TT_FATAL in Path A; PCC ~0.71 in Path B), showed element-level corruption for Path B, and proved no zero-padding scheme produces correct partial RoPE from this op. ([`../ch3_bug_root_cause/index.md`](../ch3_bug_root_cause/index.md))
- **Ch4** presented three strategies (Slice-Apply-Concat, Enforce Precondition, Precomputed Full-Head cos/sin), derived the identity-filled cos/sin construction for Strategy C, and proved trace compatibility. ([`../ch4_implementation_strategies/index.md`](../ch4_implementation_strategies/index.md))
- **Ch5** audited all currently supported Qwen3-family models, confirmed all have `rotary_dim=64` (tile-aligned), and concluded the bug is latent dead code. ([`../ch5_model_config_audit/index.md`](../ch5_model_config_audit/index.md))

---

## Files in Reading Order

1. [**`recommended_fix.md`**](./recommended_fix.md) — Step-by-step implementation of Strategy C in `TTNNRotaryPositionEmbedding.__init__` and `forward`; the precondition to add; the `ttnn.pad` call to remove.
2. [**`precondition_policy.md`**](./precondition_policy.md) — Which constraints must be enforced as hard errors, which should warn, and whether Strategy B remains appropriate as a short-term option.
3. [**`verification_checklist.md`**](./verification_checklist.md) — Five concrete test cases covering tile-aligned partial RoPE, non-tile-aligned partial RoPE, full-head RoPE, trace compatibility, and an edge case; the PyTorch reference formula for each.
