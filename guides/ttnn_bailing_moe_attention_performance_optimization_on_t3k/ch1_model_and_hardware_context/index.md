# Chapter 1 — Model and Hardware Context

This chapter establishes the two fixed reference points that every subsequent optimization decision in this guide depends on: the Ling (BailingMoeV2) model's attention configuration, and the T3K hardware topology on which it runs. Readers who already have deep familiarity with one of these subjects can read selectively, but the conventions and notation introduced here are used without re-definition throughout Chapters 2–7.

## Scope

Chapter 1 does not propose any optimization. It exists to give precise, shared definitions for:

- The attention hyperparameters (`num_heads`, `num_kv_heads`, `head_dim`, `partial_rotary_factor`, `use_qk_norm`) that appear in every performance equation later in the guide.
- The distinction between prefill and decode execution, which determines which performance profile matters for a given use case.
- The T3K chip count, mesh topology, Ethernet link count, and per-link bandwidth that set the ceiling on CCL performance.
- The TTNN sharding primitives (`TensorMemoryLayout`, `ShardSpec`, `MemoryConfig`) that are the vocabulary of every tensor layout transition discussed in Chapters 3 and 4.

## Reading Order

Read the two content files in order. The model overview comes first because the hardware topology discussion references GQA head counts and tensor shapes that are defined there.

1. [`ling_model_overview.md`](./ling_model_overview.md) — Ling architecture, GQA configuration, prefill vs. decode paths.
2. [`t3k_topology_primer.md`](./t3k_topology_primer.md) — T3K physical layout, Ethernet CCL characteristics, TTNN sharding primitives.

**Start reading:** [Ling Model Overview](ling_model_overview.md)
