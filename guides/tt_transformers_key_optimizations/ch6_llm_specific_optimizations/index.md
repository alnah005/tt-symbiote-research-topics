# Chapter 6 — LLM-Specific Optimizations: Prefill/Decode Pipeline and KV Cache Capacity

## Overview

The previous five chapters established the hardware foundations and individual optimization primitives: FlashAttention-2 and paged KV cache (Ch2), matmul program configs and weight quantization (Ch3), memory hierarchy and sharding patterns (Ch4), and tensor parallelism with collective communication (Ch5). Chapter 6 assembles those primitives into a complete LLM inference pipeline and addresses two practical engineering concerns that span every layer of the stack:

1. **Prefill/decode pipeline mechanics** — how chunked prefill works, how the decode loop is structured, how batching affects memory-bandwidth utilization, and what initialization sequence (JIT warm-up, trace capture) is required before production serving can begin.

2. **KV cache capacity planning** — how to calculate the DRAM budget for a given model, context length, and batch configuration; how to size the paged block pool; and the tradeoffs between KV cache capacity and decode throughput.

Both topics require understanding the exact API signatures of the forward functions, the shape conventions for KV cache and page table tensors, and the behavioral differences between compile-time warm-up and trace capture. These details are pinned to confirmed facts throughout.

---

## Prerequisites

Before reading this chapter you should be comfortable with:

- Paged KV cache addressing and `paged_update_cache` — covered in Ch2 (`paged_attention_kv_cache.md`)
- Weight layout, BFP8/BFP4 formats, and math fidelity pairings — covered in Ch3 (`weight_layout_and_quantization.md` and `matmul_program_configs.md`)
- DRAM vs L1 memory hierarchy and the decode memory-bandwidth bottleneck — covered in Ch4 (`memory_hierarchy.md`)
- Tensor parallelism and CCL operations — covered in Ch5 (`tensor_parallelism.md` and `ccl_and_ethernet.md`)

---

## What You Will Be Able to Do After This Chapter

- Write a correct chunked prefill loop that bounds iteration by `prompt_lens.max().item()`, not by a tensor dimension
- Explain why `current_pos` in `decode_forward` is a scalar (not per-sequence), and what that implies for batch homogeneity requirements
- Distinguish warm-up (JIT compilation only) from trace capture, and state the required call order
- Calculate the per-token, per-layer, and total KV cache DRAM footprint for a given model configuration
- Size a paged block pool for a target batch size, context length, and block size
- Articulate the capacity vs throughput tradeoff when increasing batch size under a fixed KV cache DRAM budget

---

## Files in This Chapter

| File | Description |
|---|---|
| [prefill_decode_pipeline.md](./prefill_decode_pipeline.md) | Chunked prefill loop mechanics, decode loop structure, batching strategies, warm-up and trace capture initialization sequence |
| [kv_cache_capacity_planning.md](./kv_cache_capacity_planning.md) | KV cache memory budget calculation, block pool sizing, page table constraints, capacity vs throughput tradeoffs |

---

## Key Concepts

| Concept | Short Definition | Where Covered |
|---|---|---|
| Chunked prefill | Splitting a long prompt into fixed-size chunks processed sequentially, each updating the KV cache incrementally | `prefill_decode_pipeline.md` |
| `MAX_PREFILL_CHUNK_SIZE` | Hardware/model-specific ceiling on chunk length; constrained by the attention score matrix fitting in L1 | `prefill_decode_pipeline.md` |
| `prefill_forward_text` | Forward function for prefill; accepts `tokens [batch, seq_len]`, `page_table`, `kv_cache_len`, and `prompt_lens` | `prefill_decode_pipeline.md` |
| `decode_forward` | Forward function for single-token generation; `current_pos` is a single scalar int, not per-sequence | `prefill_decode_pipeline.md` |
| Warm-up (JIT compilation) | `warmup_model_decode()` and `warmup_model_prefill()` trigger JIT kernel compilation; they do NOT capture traces | `prefill_decode_pipeline.md` |
| Trace capture | Separate step using `ttnn.begin_trace_capture` / `ttnn.end_trace_capture`; required for production decode performance | `prefill_decode_pipeline.md` |
| Static batching | All sequences complete together; simpler scheduling, no mid-batch insertions | `prefill_decode_pipeline.md` |
| Continuous batching | New requests inserted as sequences complete; reduces head-of-line blocking | `prefill_decode_pipeline.md` |
| KV cache shape | `[1, n_kv_heads, n_blocks * block_size, head_dim]` — first dim is 1 (not batch); pool is shared across all sequences | `kv_cache_capacity_planning.md` |
| Per-token KV footprint | For Llama 3.1 8B in BF16: 128 KB per token across all 32 layers | `kv_cache_capacity_planning.md` |
| Block pool sizing | `n_blocks = ceil(batch * max_seq_len / block_size)` as a starting formula; refined by capacity vs throughput tradeoff | `kv_cache_capacity_planning.md` |
| Page table constraints | `[batch, max_pages]` int32; shape must remain constant across decode steps to avoid TTNN recompilation | `kv_cache_capacity_planning.md` |

---

## Reading Order

Read the files in the following order:

1. [prefill_decode_pipeline.md](./prefill_decode_pipeline.md) — Read first. The pipeline mechanics establish the runtime structure (warm-up, trace capture, chunked prefill loop, decode loop) that motivates the KV cache sizing decisions in the next file.
2. [kv_cache_capacity_planning.md](./kv_cache_capacity_planning.md) — Read second. Capacity planning assumes familiarity with the paged block pool structure and the batch size / `current_pos` semantics established in the pipeline file.

---

## Relationship to Prior Chapters

- **Ch2** introduced the paged KV cache structure and `paged_update_cache`. This chapter does not re-derive those; it uses them as given and focuses on sizing and operational sequencing.
- **Ch3** established weight quantization (BFP8/BFP4) and math fidelity pairings. The capacity calculations in this chapter extend those bandwidth principles to the KV cache itself.
- **Ch4** established that KV cache is stored in DRAM (not L1) and that decode is memory-bandwidth-bound. The capacity tradeoffs in this chapter flow directly from that constraint.
- **Ch5** established tensor parallelism and per-device KV head distribution. KV cache capacity calculations in this chapter cover the single-device case; multi-device sharding of the KV cache follows the head-sharding rules from Ch5.

---

**Next:** [`prefill_decode_pipeline.md`](./prefill_decode_pipeline.md)
