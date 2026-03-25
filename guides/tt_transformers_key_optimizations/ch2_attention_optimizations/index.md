# Chapter 2 — Attention Optimizations: FlashAttention, GQA, and Paged Decode

## Overview

Attention is the most memory-bandwidth-intensive operation in transformer inference. For a sequence of length S and head dimension D, naive attention materializes an S×S score matrix to DRAM at O(S²) cost — a bottleneck that compounds with batch size and number of heads. This chapter covers how tt-transformers eliminates that bottleneck through two distinct code paths that share a common TTNN API surface but use fundamentally different parallelization strategies:

**Prefill path:** The model processes a prompt of S tokens in a single forward pass. The key operation is `ttnn.transformer.scaled_dot_product_attention`, which implements FlashAttention-2 tiling — parallelizing Q chunks across Tensix cores, computing online softmax incrementally over KV blocks, and keeping all intermediates in L1 (120 KB per core) without a single DRAM write for the score matrix.

**Decode path:** The model generates one new token per step, so Q is a single-token vector attending to a KV cache that grows with each step. The key operation is `ttnn.transformer.scaled_dot_product_attention_decode`, which implements Flash-Decode — parallelizing over the KV sequence dimension rather than the Q dimension, with optional paged KV cache for memory-efficient batch serving.

Understanding which path is active at any point in inference is essential for reading tt-transformers source code. Prefill and decode use different program configs, different parallelization axes, and different performance bottlenecks.

---

## Prerequisites

Before reading this chapter, you should have completed Chapter 1 or be comfortable with:

- Tensix core architecture: L1 SRAM (120 KB), DRAM, RISC-V kernel structure (BRISC, NCRISC, TRISC0/1/2)
- TTNN tensor layout: `TilizedLayout`, `MemoryConfig`, `TensorMemoryLayout`, sharding strategies
- Math fidelity levels and `WormholeComputeKernelConfig`
- The high-level distinction between prefill (processing a full prompt) and decode (generating one token at a time)

---

## What You Will Be Able to Do After This Chapter

- Explain why FlashAttention-2 is compute-bound rather than memory-bandwidth-bound, and trace that claim to specific Tensix hardware properties
- Describe causality-aware load balancing and explain why pairing Q_low and Q_high chunks per core achieves ~1.6× speedup
- Construct a `SDPAProgramConfig` for prefill, choosing `q_chunk_size` and `kv_chunk_size` to fit within 120 KB L1
- Explain Flash-Decode's KV-parallel strategy and contrast it with prefill's Q-parallel strategy
- Configure GQA head grouping in `SDPAMultiCoreProgramConfig` and understand how multiple Q heads per KV group map to core parallelism
- Explain paged KV cache addressing and describe how `paged_update_cache` fuses K and V writes for parallel DRAM throughput
- Identify when to use `paged_scaled_dot_product_attention_decode` vs the standard decode path

---

## TTNN API Entry Points

The following TTNN operations are the primary entry points for attention in tt-transformers. All are in the `ttnn.transformer` namespace:

| Operation | Use Case | Parallelizes Over |
|---|---|---|
| `ttnn.transformer.scaled_dot_product_attention` | Prefill: multi-token Q, full KV | Q sequence chunks across cores |
| `ttnn.transformer.scaled_dot_product_attention_decode` | Decode: single-token Q, KV cache | Batch and KV sequence across cores |
| `ttnn.transformer.paged_scaled_dot_product_attention_decode` | Decode with paged KV cache | Batch and KV sequence; K/V fetched via page table |
| `ttnn.transformer.ring_distributed_scaled_dot_product_attention` | Multi-device ring topology | KV sequence split across devices; partial outputs reduced |

Each operation accepts a program config object that controls core grid size, chunk dimensions, and parallelization strategy. The config type differs between prefill and decode: prefill uses `SDPAProgramConfig`; decode uses `SDPAMultiCoreProgramConfig`.

---

## Files in This Chapter

| File | Description |
|---|---|
| [flash_attention_prefill.md](./flash_attention_prefill.md) | FlashAttention-2 tiling on Tensix, causality-aware load balancing, double-buffering, sliding window attention, `SDPAProgramConfig` |
| [flash_decode_and_gqa.md](./flash_decode_and_gqa.md) | Flash-Decode algorithm, GQA/MQA head grouping, `SDPAMultiCoreProgramConfig`, `cur_pos_tensor`, ring-distributed SDPA |
| [paged_attention_kv_cache.md](./paged_attention_kv_cache.md) | Paged KV cache addressing, page table tensor, `paged_update_cache`, `paged_update_cache` core sharding, Multi-Latent Attention (MLA) |

---

## Reading Order

Read the files in the order below. Each file builds directly on the one before:

1. [flash_attention_prefill.md](./flash_attention_prefill.md) — Start here. The FlashAttention-2 tiling strategy is the conceptual foundation for understanding why Flash-Decode takes a different approach.
2. [flash_decode_and_gqa.md](./flash_decode_and_gqa.md) — Read second. Flash-Decode's KV-parallel strategy contrasts explicitly with the Q-parallel approach from file 1; GQA/MQA is covered here.
3. [paged_attention_kv_cache.md](./paged_attention_kv_cache.md) — Read last. Paged KV is an extension of the decode path; understanding the standard decode kernel first makes the page table indirection easier to follow.

---

**Next:** [`flash_attention_prefill.md`](./flash_attention_prefill.md)
