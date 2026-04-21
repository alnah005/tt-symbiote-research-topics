# Chapter 2 — M-RoPE in Qwen3.6-35B-A3B: Configuration and Reference Implementation

## Prerequisites

- Chapter 1: [RoPE Foundations](../ch1_rope_foundations/index.md)

---

## Contents

| File | Description |
|---|---|
| [`qwen36_rope_config.md`](./qwen36_rope_config.md) | Config fields, `mrope_section` breakdown, `partial_rotary_factor` derivation |
| [`hf_reference_implementation.md`](./hf_reference_implementation.md) | HuggingFace `apply_multimodal_rotary_pos_emb` walkthrough |
| [`position_id_construction.md`](./position_id_construction.md) | How position IDs are built for text-only, text+image, and text+video inputs |
