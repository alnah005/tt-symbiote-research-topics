# Chapter 1 — Model Architecture Overview

This chapter establishes the two Qwen3.5 model variants implemented in this codebase
(27B dense and 35B-A3B MoE), their layer composition, hyperparameter tables, and
the hybrid DeltaNet + full-attention design. No implementation details are introduced
here — those are covered in Chapters 2–5.

---

## Files in This Chapter

| File | Description |
|------|-------------|
| [`model_variants.md`](./model_variants.md) | Side-by-side comparison of all four Qwen3.5 variants — architecture, layer counts, hardware targets, and performance |
| [`layer_types_and_hyperparams.md`](./layer_types_and_hyperparams.md) | Per-layer hyperparameters for DeltaNet layers, full-attention layers, MoE MLPs, and the vocabulary / embedding setup |

---

## Reading Order

1. [`model_variants.md`](./model_variants.md) — start here for the big picture: which models
   exist, what hardware they target, and why 35B-A3B is the recommended entry point.
2. [`layer_types_and_hyperparams.md`](./layer_types_and_hyperparams.md) — drill into the exact
   hyperparameter values for each layer type. The values introduced here (especially `head_dim`,
   `hidden_size`, `layer_types`, `linear_num_key_heads`, `linear_num_value_heads`) are referenced
   throughout later chapters.

---

## Cross-References

- **Chapter 2 — GatedDeltaNet** uses `linear_num_key_heads`, `linear_num_value_heads`,
  `linear_key_head_dim`, `linear_value_head_dim`, and `linear_conv_kernel_dim` introduced in
  [`layer_types_and_hyperparams.md`](./layer_types_and_hyperparams.md).
- **Chapter 3 — GatedAttention** uses `n_heads`, `n_kv_heads`, `head_dim`,
  `partial_rotary_factor`, and `rope_theta` introduced in
  [`layer_types_and_hyperparams.md`](./layer_types_and_hyperparams.md).
- **Chapter 4 — Decoder Block** explains how the `layer_types` list drives per-layer dispatch,
  first described in [`model_variants.md`](./model_variants.md).
- **Chapter 5 — Mixture of Experts** expands on the MoE hyperparameters (num_experts,
  num_experts_per_tok, moe_intermediate_size, shared_expert_intermediate_size) from
  [`layer_types_and_hyperparams.md`](./layer_types_and_hyperparams.md).
