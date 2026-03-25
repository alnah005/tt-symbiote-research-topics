# Chapter 7 — Llama 3 End-to-End Walkthrough

## Overview

The previous six chapters built every optimization primitive in isolation: hardware tile execution and data formats (Ch1), attention kernels and paged KV cache (Ch2), matmul program configs and weight quantization (Ch3), memory hierarchy and sharding (Ch4), tensor parallelism and collective communication (Ch5), and the LLM inference pipeline with KV cache capacity planning (Ch6). This chapter assembles all of those primitives into a concrete end-to-end walkthrough of a single decode step on real production configurations.

The walkthrough proceeds in two stages:

1. **Llama 3.1 8B on a single N150** (`single_device_decode.md`) — A Wormhole chip running one model. The walkthrough traces a complete decode step through all 32 decoder layers, naming at each operation which optimization from which chapter applies, and showing the two precision configurations (accuracy mode vs performance mode) and their throughput implications.

2. **Llama 3.1 70B on a T3K** (`multi_device_decode.md`) — An 8-chip Wormhole Ethernet ring running tensor parallelism at TP=8. The walkthrough shows how the single-device picture is extended: which weights are column-parallel, which are row-parallel, where collective operations are inserted, and how the GQA head distribution works out exactly at TP=8.

The goal is not to re-teach any individual optimization. It is to show how the optimizations compose — which choices are independent, which are coupled, and what the overall execution sequence looks like when everything runs together.

---

## Prerequisites

Before reading this chapter you should have covered all prior chapters, or at minimum be comfortable with:

- Tensix tile execution, BFP formats, and math fidelity pairings — Ch1 (`math_fidelity_and_data_formats.md`)
- FlashAttention-2 decode (paged SDPA), GQA, and paged KV cache shape conventions — Ch2 (`flash_decode.md`, `paged_attention_kv_cache.md`)
- Weight layout (pre-transposed `[K, N]`), BFP8/BFP4 quantization, and DRAM-sharded matmul — Ch3 (`weight_layout_and_quantization.md`, `matmul_program_configs.md`)
- DRAM vs L1 memory hierarchy and why decode is bandwidth-bound — Ch4 (`memory_hierarchy.md`)
- Tensor parallelism, column-parallel / row-parallel weight splitting, and CCL operations (`ttnn.all_gather`, `ttnn.reduce_scatter`, `ttnn.all_reduce`) — Ch5 (`tensor_parallelism.md`, `ccl_and_ethernet.md`)
- Decode loop structure, `current_pos` scalar semantics, warm-up vs trace capture, and KV cache capacity — Ch6 (`prefill_decode_pipeline.md`, `kv_cache_capacity_planning.md`)

---

## What You Will Be Able to Do After This Chapter

- Identify which TTNN optimization primitive applies at each operation in a Llama 3.1 transformer decode step
- Explain the concrete effect of switching between BFP4 (performance mode, ~28 t/s/u) and BFP8 (accuracy mode, ~23 t/s/u) on the N150
- State which math fidelity level is paired with each data format (BFP4 + LoFi, BFP8 + HiFi2, BF16 + HiFi4) and why
- Describe how 64 Q heads and 8 KV heads are distributed across 8 devices in a T3K TP=8 configuration without KV head replication
- Identify where `ttnn.all_gather`, `ttnn.reduce_scatter`, and `ttnn.all_reduce` appear in the TP=8 decode step and explain what each collects or reduces
- Read a T3K decode step description and predict which tensors each device holds before and after each collective

---

## Files in This Chapter

| File | Description |
|---|---|
| [single_device_decode.md](./single_device_decode.md) | Llama 3.1 8B on N150: weight loading, 13-step decode sequence traced through one decoder layer, optimization cross-references, BFP4 vs BFP8 precision configurations |
| [multi_device_decode.md](./multi_device_decode.md) | Llama 3.1 70B on T3K: TP=8 distribution, per-device head counts, collective insertion points, full decode step with 8-chip scaling |

---

## Reading Order

Read the files in the order listed below:

1. [single_device_decode.md](./single_device_decode.md) — Read first. The single-device walkthrough establishes the baseline decode step sequence and names every optimization in context. The multi-device file extends this baseline rather than re-explaining the primitives.
2. [multi_device_decode.md](./multi_device_decode.md) — Read second. The multi-device walkthrough assumes familiarity with the single-device sequence and focuses on how each step changes under TP=8 and what collective operations are inserted.

---

## Key Concepts

| Concept | Short Definition | Where Covered |
|---|---|---|
| Accuracy mode | BFP8 MLP + BFP8 attn (HiFi2 for both); ~23 t/s/u on N150 | `single_device_decode.md` |
| Performance mode | BFP4 MLP + BFP8 attn (LoFi MLP, HiFi2 attn); ~28 t/s/u on N150 | `single_device_decode.md` |
| BFP4 bandwidth multiplier | 3.56x vs BF16; approximately 22% higher throughput than BFP8 mode | `single_device_decode.md` |
| Pre-transposed weight layout | Weights stored as `[K, N]` = `[in_features, out_features]` at load time; no runtime transpose | `single_device_decode.md` |
| DRAM-sharded matmul | All decode linear layers use DRAM sharding; weights stay in DRAM, not loaded into L1 before matmul | `single_device_decode.md` |
| Paged SDPA decode | Flash decode kernel operating on paged KV cache; GQA-aware | `single_device_decode.md` |
| TP=8 column-parallel | QKV and FF1/FF3 projections split along the output dimension across 8 devices | `multi_device_decode.md` |
| TP=8 row-parallel | Output projection and FF2 split along the input dimension; post-op `ttnn.reduce_scatter` | `multi_device_decode.md` |
| GQA at TP=8 | 8 KV heads / 8 devices = 1 KV head per device; no KV head replication required | `multi_device_decode.md` |
| Vocab-parallel LM head | LM head weight split along vocab dimension; post-op `ttnn.all_reduce` | `multi_device_decode.md` |

---

## Relationship to Prior Chapters

- **Ch1** introduced BFP formats and math fidelity. This chapter applies those as concrete configuration choices at every linear layer.
- **Ch2** introduced paged SDPA decode and GQA. This chapter shows exactly where and how `paged_scaled_dot_product_attention_decode` appears in the decode sequence.
- **Ch3** introduced DRAM-sharded matmul and pre-transposed weight layout. This chapter applies those to every linear layer in the decode step and shows the format choice per layer.
- **Ch4** explained why decode is memory-bandwidth-bound. This chapter quantifies the consequence: BFP4 weights (3.56x bandwidth vs BF16) yield ~22% higher throughput than BFP8 weights (1.88x bandwidth vs BF16) because the MLP is the dominant bandwidth consumer.
- **Ch5** introduced tensor parallelism and CCL operations. This chapter places each collective in its precise position in the TP=8 decode step.
- **Ch6** established the decode loop, `current_pos` scalar semantics, and warm-up vs trace capture. This chapter assumes those conventions are in effect during the walkthrough.

---

**Start:** [`single_device_decode.md`](./single_device_decode.md)
