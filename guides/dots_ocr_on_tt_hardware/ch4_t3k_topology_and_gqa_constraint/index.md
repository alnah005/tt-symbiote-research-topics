# Chapter 4: T3K Topology and GQA Constraint

## Overview

This chapter explains why dots.ocr is restricted to tensor parallelism degree TP≤2 on T3K hardware, derives that limit mathematically from the model's GQA head configuration, and describes the submesh approach that satisfies it. It also covers the environment variables and chunked prefill mechanism that make long-context OCR workloads practical within these constraints.

## Reading Order

| File | Description |
|------|-------------|
| [gqa_tp_constraint.md](gqa_tp_constraint.md) | Mathematical derivation of the TP≤2 limit from `num_key_value_heads=2` and what failure looks like at TP>2 |
| [t3k_submesh_and_env_vars.md](t3k_submesh_and_env_vars.md) | How `open_dots_mesh_device()` carves a 1×2 logical submesh from the full 1×8 T3K parent mesh, plus all relevant env vars |
| [chunked_prefill.md](chunked_prefill.md) | Why OCR sequences require chunked prefill on Wormhole devices and how chunk size is controlled |

## Quick Reference

| Config | Value | Implication |
|--------|-------|-------------|
| `num_key_value_heads` | 2 | TP ∈ {1, 2} only |
| `num_attention_heads` | 12 | gcd(12, 2) = 2 → TP ≤ 2 |
| T3K devices | 8 | 6 devices are idle at TP=2 |
| Submesh shape | 1×2 | logical TP=2 group carved from 1×8 parent |
| `DOTS_T3K_TP` | 1 or 2 | selects submesh width |
| `DOTS_LM_HEAD_MAX_COLUMNS_PER_DEVICE` | 2048 | prevents L1 overflow at `vocab_size=151936` |

**Next:** [GQA TP Constraint](gqa_tp_constraint.md)
