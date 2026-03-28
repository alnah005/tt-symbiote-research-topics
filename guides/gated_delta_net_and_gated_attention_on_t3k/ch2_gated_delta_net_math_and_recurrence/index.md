# Chapter 2 — Gated Delta Net: Mathematical Formulation and Recurrence Structure

## Overview

This chapter derives the Gated Delta Net state-update rule from first principles, defines every tensor and its shape for the Qwen3.5-35B-A3B configuration, and analyzes whether and how the recurrence can be parallelized across the sequence dimension.

**This chapter covers Gated Delta Net only.** Gated Attention — the softmax-based co-located attention layers — is treated in Chapter 3.

## Learning Objectives

After reading this chapter you will be able to:

- Write down the Gated Delta Net recurrence from memory and explain the role of each term (decay gate, update rate, delta correction, outer-product write).
- Trace every projection in one Gated Delta Net layer of Qwen3.5-35B-A3B, with full input/output shapes.
- Explain why the raw recurrence is not directly associative and what the chunkwise WY-decomposition enables.
- Quantify the per-layer recurrent state memory and compare it to the KV cache of a co-located Gated Attention layer at T = 256K.

## Questions Answered

This chapter answers **Q1** (what is the precise mathematical formulation of the Gated Delta Net recurrence?) and **Q3** (what is the parallelism structure, and when is the recurrence sequential vs. parallelizable?) from the guide specification.

## Sections

| File | Contents |
|------|----------|
| [`gated_delta_rule_formulation.md`](./gated_delta_rule_formulation.md) | Core recurrence derivation, per-term interpretation, decay gate derivation, full projection inventory, state matrix size |
| [`parallelism_and_scan.md`](./parallelism_and_scan.md) | Sequential dependency analysis, chunkwise WY-decomposition, associative scan feasibility, decode vs. prefill complexity |
| [`state_vs_kv_cache_memory.md`](./state_vs_kv_cache_memory.md) | Step-by-step memory calculations for recurrent state vs. KV cache, crossover analysis, T3K DRAM and L1 constraints |
