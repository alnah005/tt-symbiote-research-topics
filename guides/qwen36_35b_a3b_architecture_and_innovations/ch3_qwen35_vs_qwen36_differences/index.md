# Chapter 3 — Qwen3.5 vs Qwen3.6: Exact Differences

## Overview

This chapter provides a precise, exhaustive comparison of Qwen3.6-35B-A3B against Qwen3.5-35B-A3B at the config, weight, and behavior levels. The goal is to establish definitively what changed between the two model versions and what remained identical — particularly from the perspective of a hardware-level TTNN implementation.

**The central finding of this chapter:** the neural architecture is completely identical between Qwen3.5 and Qwen3.6. Every architectural hyperparameter, every operator type, every weight tensor shape, and every data type is unchanged. The differences are entirely in post-training (alignment, RLHF, RL on agentic tasks) and a small set of config metadata fields that carry no architectural significance.

This means any TTNN implementation that correctly runs Qwen3.5-35B-A3B will also correctly run Qwen3.6-35B-A3B without any model code changes — only the checkpoint weights need to be swapped.

## Learning Objectives

After completing this chapter, you will be able to:

1. Enumerate every field that differs between the Qwen3.5 and Qwen3.6 `config.json` files and explain why none of them represent architectural changes.
2. Explain what post-training means in this context and identify the specific agentic capabilities that were improved.
3. Interpret benchmark tables showing Qwen3.5 vs Qwen3.6 performance and identify which task categories saw the largest gains.
4. State with confidence why no changes to TTNN op graphs, kernel dispatch, or weight loading are required to support Qwen3.6.

## Key Finding Preview

| Dimension | Qwen3.5-35B-A3B | Qwen3.6-35B-A3B | Changed? |
|-----------|-----------------|-----------------|----------|
| HuggingFace model class | `Qwen3_5MoeForConditionalGeneration` | `Qwen3_5MoeForConditionalGeneration` | No |
| `model_type` | `qwen3_5_moe` | `qwen3_5_moe` | No |
| Number of layers | 40 | 40 | No |
| `hidden_size` | 2048 | 2048 | No |
| MoE configuration | identical | identical | No |
| Attention configuration | identical | identical | No |
| Vision encoder | identical | identical | No |
| Weight tensor shapes | all identical | all identical | No |
| Weight tensor values | (pre-trained base) | (post-trained alignment) | **Yes** |
| Config metadata strings | Qwen3.5 | Qwen3.6 | Yes (metadata only) |
| `transformers_version` | `4.57.0.dev0` | `4.57.1` | Yes (metadata only) |

The table above captures everything that differs. There are no architectural changes whatsoever.

## Chapter Contents

- [`config_diff.md`](./config_diff.md) — Field-by-field side-by-side comparison of `config.json` for Qwen3.5 and Qwen3.6, with a definitive conclusion on architectural equivalence and TTNN implications.

- [`post_training_differences.md`](./post_training_differences.md) — What post-training means, what changed in alignment (agentic RL, Thinking Preservation), and why weight values differ while shapes and dtypes are identical.

- [`benchmark_comparison.md`](./benchmark_comparison.md) — Benchmark tables for agentic coding, general reasoning, and vision tasks, with deltas and analysis of where gains originate.

## Navigation

| | |
|---|---|
| Previous chapter | [Chapter 2 — Gated DeltaNet Deep Dive](../ch2_gated_deltanet/index.md) |
| Next chapter | [Chapter 4 — Partial Rotary Embedding and M-RoPE](../ch4_rope_and_mrope/index.md) |
| Guide root | [Qwen3.6-35B-A3B Architecture and Innovations](../index.md) |
