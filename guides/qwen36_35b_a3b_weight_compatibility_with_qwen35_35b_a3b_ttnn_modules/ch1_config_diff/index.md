# Chapter 1 --- Config Diff: Qwen3.6 vs Qwen3.5

## Overview

This chapter performs a field-by-field comparison of the `config.json` files
for `Qwen/Qwen3.5-35B-A3B` and `Qwen/Qwen3.6-35B-A3B`. It identifies every
changed, added, and removed field, explains what each change means at the
Python and architecture level, and establishes which chapters carry the
downstream TTNN impact analysis.

The central finding of this chapter is that the two configs share every
structural hyperparameter that governs weight tensor shapes. The only material
differences are three additions to the Qwen3.6 config: a top-level promotion of
`partial_rotary_factor`, a new `bos_token_id`, and a new `mtp_num_hidden_layers`
field. None of these additions alter the dimensions of any weight matrix in the
shared backbone.

After reading this chapter you will know:

- Which config fields are identical between the two versions and why that
  guarantees shape compatibility for all existing TTNN weight tensors.
- What `partial_rotary_factor: 0.25` being promoted to the top level means for
  HuggingFace config resolution and for `TTNNRotaryPositionEmbedding`.
- What `bos_token_id: 248044` is, how the generation stack uses it, and whether
  it touches any TTNN tensor.
- What `mtp_num_hidden_layers: 1` declares, whether the MTP head is
  inference-active, and whether its weight keys can interfere with the existing
  TTNN loading path.

## Prerequisites

This chapter assumes the following background.

| Prerequisite | What is assumed | Where to review |
|---|---|---|
| HuggingFace `config.json` format | Familiarity with how model architecture fields (e.g., `hidden_size`, `num_attention_heads`) are stored in a JSON config and consumed by `AutoConfig` / `AutoModelForCausalLM` | HuggingFace Transformers docs, `AutoConfig` reference |
| Qwen3.5-35B-A3B architecture | Know that this model is a hybrid MoE transformer with GQA, `hidden_size=7168`, 94 decoder layers, and YaRN-extended context via `rope_scaling` | Qwen3 technical report; tt-symbiote model card |
| TTNN weight loading | Awareness that TT-Symbiote maps HuggingFace weight key names to on-device TTNN tensors, and that changes to key names or tensor shapes require explicit loader updates | tt-symbiote TTNN onboarding docs |

## Reading Order

1. [`structural_fields.md`](./structural_fields.md) --- All fields that are
   identical between the two configs; why unchanged structural hyperparameters
   guarantee that all weight tensor shapes are preserved; fields that changed
   numerically but have no tensor-shape impact.
2. [`new_and_modified_fields.md`](./new_and_modified_fields.md) --- The three
   fields added or promoted in Qwen3.6 (`partial_rotary_factor`,
   `bos_token_id`, `mtp_num_hidden_layers`) and every other field that
   changed; the meaning of each change and which chapter carries the TTNN
   impact analysis.

## Quick-Reference Diff Table

The table below lists every field that is **different** between the two
`config.json` files. Fields that are present and identical in both versions are
not listed here; they are documented in
[`structural_fields.md`](./structural_fields.md).

| Field | Qwen3.5-35B-A3B value | Qwen3.6-35B-A3B value | Change type | TTNN impact analysed in |
|---|---|---|---|---|
| `partial_rotary_factor` | _(absent at top level; present only inside `rope_parameters`)_ | `0.25` | Promoted to top level | [Chapter 3](../ch3_partial_rotary_factor/index.md) |
| `bos_token_id` | _(absent)_ | `248044` | Added | [Chapter 4](../ch4_bos_token_id/index.md) |
| `mtp_num_hidden_layers` | _(absent)_ | `1` | Added | [Chapter 5](../ch5_mtp_head/index.md) |
