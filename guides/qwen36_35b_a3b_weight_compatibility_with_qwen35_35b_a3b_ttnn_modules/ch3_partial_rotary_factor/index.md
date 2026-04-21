# Chapter 3: `partial_rotary_factor` Promotion and RoPE Resolution

## Finding Summary

The promotion of `partial_rotary_factor` to the top-level config in **Qwen3.6** is a defensive no-op. The numeric value does not change — it is `0.25` in both the nested `rope_scaling` dict (present in both checkpoints) and the new top-level key (present only in **Qwen3.6**). The computed rotary dimension is:

```
rotary_dim = int(head_dim × partial_rotary_factor) = int(128 × 0.25) = 32
```

This value is identical for **Qwen3.5** and **Qwen3.6**. No TTNN cos/sin table shapes change. `TTNNRotaryPositionEmbedding` produces an identical embedding module for both checkpoints.

## The One Actionable Risk

Any TT-Symbiote code that bypasses `Qwen3_5MoeConfig.__init__` and reads `config.partial_rotary_factor` as a bare attribute on a raw config object will raise `AttributeError` for **Qwen3.5** config JSON, where `partial_rotary_factor` is not a top-level key. **Qwen3.6** adds the top-level key as a defensive fix for this code path. Code working with raw config objects (bypassing `__init__`) and handling both checkpoints requires the guard pattern (code using properly loaded `AutoConfig` objects does not strictly need it) — see [`ttnn_rope_impact.md` Section 1](./ttnn_rope_impact.md) for the guard expression.

## Prerequisites

- Chapter 1 (`new_and_modified_fields.md`) — establishes the config diff and the dual-location placement of `partial_rotary_factor`
- Familiarity with `TTNNRotaryPositionEmbedding` and how it consumes model config to build cos/sin tables

## Contents

- [`hf_config_resolution.md`](./hf_config_resolution.md) — traces how HuggingFace `AutoConfig` resolves `partial_rotary_factor` through every layer of the config stack for both checkpoints
- [`ttnn_rope_impact.md`](./ttnn_rope_impact.md) — maps the resolved value to `TTNNRotaryPositionEmbedding`, confirms identical `rotary_dim`, and documents the `AttributeError` guard pattern
