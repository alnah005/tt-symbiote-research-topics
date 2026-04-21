# Chapter 4: M-RoPE TTNN Implementation Strategy

> **Key Finding:** The cos/sin frequency table does not change for M-RoPE. M-RoPE reuses the same `[max_seq_len, rotary_dim/2]` table as standard partial RoPE — the only difference is that three independent row gathers (one per position axis) replace the single contiguous slice. The frequency values and table shape are identical.

## Decision Framework Summary

Two implementation options are available:

**Option A — Extend `TTNNRotaryPositionEmbedding`:** Add a `use_mrope: bool` flag and `mrope_section: list[int]` constructor parameter to the existing class. When `use_mrope=False` (default), the class is bit-for-bit identical to the current implementation. When `use_mrope=True`, the forward accepts a `[3, batch, seq_len]` position ID tensor and performs the three-gather construction. This is the lower-risk path for initial bring-up.

**Option B — New `TTNNMRoPERotaryPositionEmbedding` class:** A standalone class that handles only M-RoPE, with no backward-compatibility branching. The forward interface always accepts `[3, batch, seq_len]` position IDs. This eliminates the conditional branch overhead in the decode hot path and provides clean test isolation. Preferable for production.

For TT-Symbiote initial M-RoPE bring-up, Option A is recommended. When M-RoPE is validated and becomes a first-class inference mode, refactor to Option B.

## Prerequisites

- Chapter 1: RoPE foundations and section dimension assignment (`mrope_section = [11, 11, 10]`)
- Chapter 2: Qwen3.6 M-RoPE configuration (`rotary_dim = 64`, `rope_theta = 1000000.0`, `head_dim = 128`)
- Chapter 3: Text-only equivalence proof — sequential position IDs cause the three-gather path to produce output numerically identical to standard 1D RoPE

## Contents

- [`existing_ttnn_rope_gap_analysis.md`](./existing_ttnn_rope_gap_analysis.md) — What `TTNNRotaryPositionEmbedding` currently does and the two gaps that block M-RoPE
- [`extension_approach.md`](./extension_approach.md) — Option A: adding `use_mrope` flag to the existing class
- [`new_class_approach.md`](./new_class_approach.md) — Option B: standalone `TTNNMRoPERotaryPositionEmbedding` class
- [`pre_computed_cos_sin_strategy.md`](./pre_computed_cos_sin_strategy.md) — Why a single cos/sin table suffices for M-RoPE (no per-section tables needed)
- [`gather_operation_on_ttnn.md`](./gather_operation_on_ttnn.md) — How `ttnn.embedding` implements the per-axis row gather and decode/prefill considerations
