# Chapter 5 — Model Configuration Audit: Which Models Exercise the Non-Tile-Aligned rotary_dim Path?

This chapter audits all currently supported Qwen3-family models in tt-symbiote to determine whether any production configuration exercises the non-tile-aligned `rotary_dim` code path in `TTNNRotaryPositionEmbedding`. The audit answer is stated first; the supporting files derive it.

---

> **Key Finding:** All currently supported Qwen3-family models in tt-symbiote have tile-aligned `rotary_dim`. The non-tile-aligned path — exemplified by `rotary_dim=48` — is not exercised by any production-supported configuration. The bug documented in Chapters 1–3 is latent: present in code, reachable in principle, but not triggered by any model that ships today.

---

## Audit Conclusion

The table below summarizes every Qwen3-family configuration that passes through `TTNNRotaryPositionEmbedding`. All have `partial_rotary_factor=0.5` and `head_dim=128`, yielding `rotary_dim=64`. Because `64 % 64 == 0`, the two-tile constraint is satisfied and the zero-padding branch in `TTNNRotaryPositionEmbedding` is never reached.

| Model | partial\_rotary\_factor | head\_dim | rotary\_dim (derived) | rotary\_dim tile-aligned? | Bug path reached? |
|---|---|---|---|---|---|
| Qwen3.5-35B-A3B (attention layers) | 0.5 | 128 | 64 | Yes (`64 % 64 == 0`) | No |
| Qwen3.6-35B-A3B (attention layers) | 0.5 | 128 | 64 | Yes (`64 % 64 == 0`) | No |
| Qwen3.6-35B-A3B (linear attention / DeltaNet layers) | 0.5 | 128 | 64 | Yes (`64 % 64 == 0`) | No |
| Hypothetical: any model with partial\_rotary\_factor=0.375, head\_dim=128 | 0.375 | 128 | 48 | No (`48 % 32 == 16 != 0`) | Yes — bug triggered |

The `rotary_dim=48` scenario (`partial_rotary_factor=0.375`) is a synthetic configuration that is not associated with any currently supported production model. It was constructed to expose the bug.

---

## Learning Objectives

By the end of this chapter you will be able to:

1. Identify which of the two RoPE classes in tt-symbiote is responsible for non-tile-aligned `rotary_dim` handling, and explain under what conditions each class is used.
2. Derive `rotary_dim` from `partial_rotary_factor` and `head_dim` for any Qwen3-family model and determine whether the derived `rotary_dim` is tile-aligned.
3. Explain what "latent dead code" means in the context of this bug and why fixing a latent bug is still warranted.
4. Describe the two failure paths (Path A: `TT_FATAL`; Path B: silent PCC ~0.71 corruption) that would occur if a non-tile-aligned model were brought up today.

---

## Recap of Chapter 1 Prerequisite

- `TTNNRotaryPositionEmbedding` is used when `partial_rotary_factor < 1.0` (partial RoPE path).
- `TTNNDistributedRotaryPositionEmbedding` is used when `partial_rotary_factor == 1.0` or when the distributed (tensor-parallel) path is active.
- When `rotary_dim % 32 != 0`, `TTNNRotaryPositionEmbedding` calls `ttnn.pad` to extend cos/sin from `rotary_dim` to `nearest_32(rotary_dim)` — which is not the correct padding target (`head_dim` is required). (Ch1: [`../ch1_rope_fundamentals/tile_alignment_in_ttnn.md`](../ch1_rope_fundamentals/tile_alignment_in_ttnn.md))

---

## Files in Reading Order

1. [**`which_models_use_ttnn_rope.md`**](./which_models_use_ttnn_rope.md) — Distinguishes the two RoPE classes, enumerates all Qwen3-family models, derives `rotary_dim` for each, and checks tile alignment. Explains the investigation method (grep `partial_rotary_factor`).
2. [**`is_this_dead_code.md`**](./is_this_dead_code.md) — Concludes that the non-tile-aligned path is latent dead code for all currently supported models; explains the implications for fix urgency and strategy selection.
3. [**`the_rotary_dim_48_test_case.md`**](./the_rotary_dim_48_test_case.md) — Reconstructs the synthetic `rotary_dim=48, head_dim=128` test case that exposed the bug; traces both failure paths through `TTNNRotaryPositionEmbedding.forward`.

---

## What's Next

Chapter 6 consolidates all findings into a concrete fix recommendation, a precondition policy, and a verification checklist.

**Next:** [Chapter 6 — Recommendations and Implementation Guide](../ch6_recommendations/index.md)
