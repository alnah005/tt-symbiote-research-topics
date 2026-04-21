# Chapter 3: Text-Only Behavior — Does M-RoPE Reduce to Standard RoPE?

## Answer-First Summary

M-RoPE reduces to standard 1D RoPE for text-only inputs. The condition is that all three rows of the `[3, batch, seq_len]` position ID tensor must be identical — which is exactly how HuggingFace constructs them for text-only sequences. As a consequence, the existing `TTNNRotaryPositionEmbedding` text-only path is numerically correct for Qwen3.6-35B-A3B inference and requires no modification.

## Prerequisites

- **Ch1 `section_dimension_assignment.md`**: section boundary derivation from `mrope_section = [11, 11, 10]`, the three-gather + duplication assembly procedure, and the full dimension map for columns 0–63.
- **Ch2 `position_id_construction.md`**: how HuggingFace constructs the `[3, batch, seq_len]` position ID tensor for text-only sequences (Section 2 of that file covers the degenerate case directly).

## Contents

| File | Description |
|---|---|
| [`mathematical_equivalence_proof.md`](./mathematical_equivalence_proof.md) | Formal proof that equal position IDs across all three axes produce identical output to standard 1D partial RoPE; coverage argument; silent-failure caveat |
| [`practical_implications_for_text_inference.md`](./practical_implications_for_text_inference.md) | What the equivalence means for the HuggingFace text-only path and the existing TTNN implementation; scoping M-RoPE support to vision inputs only |
| [`mrope_section_always_active.md`](./mrope_section_always_active.md) | Why the section split is always structurally present in the forward pass but its effect on output values depends entirely on position ID content |
