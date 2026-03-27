# Agent B Review — Chapter 1 — Pass 1

1. [`ling_model_overview.md`, ~line 44] Issue: The `head_dim` table cell contains the parenthetical `hidden_size / num_heads = 4096 / 32`, but `num_heads` is defined as 16 (not 32). Following the formula as written, `4096 / 16 = 256`, which does not equal 128. The expression `4096 / 32 = 128` accidentally produces the correct numeric answer through an incorrect operand, but a reader who tries to verify or generalize the derivation will get the wrong result. Fix: Change `4096 / 32` to the correct formula. Since `head_dim` is an independent parameter (as the note below the table explains), the cleanest fix is to remove the parenthetical derivation entirely from the table cell, or replace it with the correct statement that `head_dim` is not derived from `hidden_size / num_heads` here.

# Agent B Review — Chapter 1 — Pass 2

**No feedback — chapter approved.**

# Agent B Review — Chapter 1 — Pass 3

1. [`index.md`, Research Questions Map table, lines 29–36] Critical structural gap: all eight research-question rows link to chapter directories that do not exist on disk (`ch2_fused_qkv_projection` through `ch7_profiling_and_bottleneck_identification`). Only `ch1_model_and_hardware_context` is present. Every link in the table is currently broken. This is a navigation failure for any reader who clicks from the index into a downstream chapter. The links themselves are correctly formed relative paths and require no syntax fix — the target directories simply need to be created before the index is published or shared.
