# Compatibility Verdict

This document provides per-question compatibility analysis for the Qwen3.6-35B-A3B checkpoint against the existing Qwen3.5-35B-A3B TTNN module suite in TT-Symbiote.

---

## 1. Weight Tensor Shapes

**Finding:** All backbone weight tensor shapes are numerically identical between the two checkpoints. Every hyperparameter that governs tensor shapes (`hidden_size`, `intermediate_size`, `moe_intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `vocab_size`) is unchanged between Qwen3.5-35B-A3B and Qwen3.6-35B-A3B (see `../ch2_weight_shapes/`).

**Evidence:** The governing hyperparameters in `config.json` are identical across both checkpoints. Because all matmul shapes, shard configurations, and TTNN program configs are derived from these hyperparameters, no changes to any TTNN matmul program config are needed. The MTP head in Qwen3.6 adds new weight keys (`model.future_prediction[0].*`) but does not rename or alter any existing backbone keys.

**Risk level:** **none**

**Required action:** None.

---

## 2. `partial_rotary_factor` Promotion

**Finding:** `partial_rotary_factor: 0.25` is promoted to the top level of `config.json` in Qwen3.6. For Qwen3.6, `config.partial_rotary_factor = 0.25` is accessible as a top-level attribute via `AutoConfig`. For Qwen3.5, this attribute does not exist at the top level — it is nested inside `config.rope_parameters`. The rotary dimension is identical for both checkpoints: `rotary_dim = int(128 * 0.25) = 32`. `TTNNRotaryPositionEmbedding` cos/sin table shapes are unchanged (see `../ch3_partial_rotary_factor/`).

**Evidence:** Ch3 traced the HuggingFace config resolution path end-to-end. `AutoConfig.from_pretrained` produces `config.partial_rotary_factor = 0.25` for both checkpoints when loaded via `AutoConfig`. The value is identical, so no numerical difference exists in RoPE computation.

**Risk level:** **low** — code that reads `config.partial_rotary_factor` directly as a top-level attribute now works with Qwen3.6 but would raise `AttributeError` on Qwen3.5. This is a backward compatibility issue for Qwen3.5, not a Qwen3.6 blocker.

**Required action:** Add defensive fallback (see `migration_steps.md` Step 2).

---

## 3. `bos_token_id: 248044`

**Finding:** This field is new in Qwen3.6 and absent from Qwen3.5. The value 248,044 exceeds `vocab_size = 151,936` — it is out of range for the embedding table by more than 96,000 indices. If any code path uses this ID for an embedding lookup, TTNN device behavior is undefined (silent garbage values or out-of-bounds access with no exception raised). The field has no effect when pre-formed `input_ids` are used from the tokenizer, which is standard practice (see `../ch4_bos_token_id/`).

**Evidence:** Ch4 traced `GenerationMixin.generate()` and TT-Symbiote's custom generation loop. The failure path is triggered only when `input_ids is None` at `generate()` entry (causing HuggingFace to construct a batch from `bos_token_id`) or when the TT-Symbiote generation loop reads `config.bos_token_id` directly to initialize the token sequence.

**Risk level:** **medium** — silent failure on the TTNN device with no exception raised. The failure is not caught at the Python layer.

**Required action:** Suppress auto-prepend; audit generation loop; add CI bounds check (see `migration_steps.md` Steps 4 and 5).

---

## 4. MTP Head Weight Keys

**Finding:** Qwen3.6 adds `model.future_prediction[0].*` keys to the checkpoint (nine weight tensors, approximately 304.6 MiB). These keys are not consumed by any existing TTNN module. The MTP head is training-only; `model.generate()` never invokes it due to the `labels is not None AND self.training is True` gate in the forward pass. No backbone weight keys are renamed or otherwise changed (see `../ch5_mtp_head/loading_recipe.md`).

**Evidence:** Ch5 traced the HuggingFace module structure and confirmed the `labels is not None AND self.training is True` gate. Ch2 confirmed that no existing TTNN module key patterns match the `model.future_prediction` prefix. The extra keys must be filtered before TT-Symbiote's weight preprocessing to prevent unrecognized-key warnings or errors during loading.

**Risk level:** **low** — extra keys that must be filtered, not weights that conflict with or replace existing backbone modules.

**Required action:** Add key filter in weight preprocessing pipeline (see `migration_steps.md` Step 3).

---

## 5. Overall Verdict

> **The existing Qwen3.5-35B-A3B TTNN module suite (`TTNNQwen3FullAttention`, `TTNNQwen3LinearAttention`, `TTNNQwen3MoE`, and related helpers) can run Qwen3.6-35B-A3B weights without modification to any backbone TTNN module. Four targeted changes to the loading and generation pipeline are required, all low-to-medium effort. Expected PCC against reference: identical to Qwen3.5 baselines (architecture is unchanged; only weight values differ).**
