# Chapter 5 — Paged KV Cache Interaction

Chapters 1–4 built the complete picture of windowed attention in the non-paged
setting: the mathematics of the window constraint, the circular-buffer eviction
policy, the memory access patterns, and the concrete TTNN primitives and tensor
shapes. Chapter 5 steps sideways from the circular-buffer model and asks what
happens when the underlying KV storage layer is not a single fixed-shape tensor
but a paged memory system managed by a virtual-to-physical page table.

This question is practical because tt-transformers ships a production-quality
paged KV cache whose decode primitive — `paged_sdpa_decode` — is the recommended
entry point for serving workloads with variable-length or dynamically-batched
sequences. Any deployment of windowed attention in that serving stack must either
integrate cleanly with the existing paging infrastructure or deliberately bypass
it. This chapter provides the analysis needed to make that choice.

## Prerequisites

This chapter requires two pieces of prior content:

- The circular-buffer layout and `pos_offset` scalar introduced in
  [`../ch2_kv_cache_management/circular_buffer_layout.md`](../ch2_kv_cache_management/circular_buffer_layout.md).
  Page-aware windowing is best understood as mapping the circular-buffer
  abstraction onto a paged physical memory backend.

- The `paged_sdpa_decode` interface, the `ttnn.update_cache` write primitive,
  and the GQA tensor shapes `[B, H_q, 1, d]` / `[B, H_kv, w, d]` from
  [`../ch4_ttnn_primitives/decode_primitives.md`](../ch4_ttnn_primitives/decode_primitives.md).
  The interface analysis in this chapter refers to specific arguments and
  program config fields established there.

## Reading Order

1. [`paged_sdpa_and_windowing.md`](./paged_sdpa_and_windowing.md)
2. [`eviction_and_page_reuse.md`](./eviction_and_page_reuse.md)

## Chapter Scope

[`paged_sdpa_and_windowing.md`](./paged_sdpa_and_windowing.md) covers the
mechanics of paged KV caches — page tables, block size, virtual-to-physical
mapping — and then analyses two concrete strategies for combining paging with
windowed attention:

- **Page-aware windowing**, in which only the pages containing tokens inside
  the current window are loaded, with the page table acting as the recency
  index.
- **Circular-buffer-as-pages**, in which exactly `ceil(w / block_size)` pages
  are allocated per sequence slot and overwritten in round-robin order,
  mirroring the non-paged circular buffer directly in the paging layer.

The file determines which strategy is more compatible with the existing
`paged_sdpa_decode` interface and identifies the minimal interface changes
required for each.

[`eviction_and_page_reuse.md`](./eviction_and_page_reuse.md) drills into the
correctness and memory-efficiency concerns that arise once a windowed sequence
grows beyond `w` tokens:

- How the window eviction policy translates to page eviction events.
- The risk of stale page table entries and the correctness invariants that
  prevent silent data corruption.
- Memory fragmentation behaviour when sequences share a fixed-size windowed
  page pool.

## What Comes Next

Chapter 6 — T3K Mesh Sharding and CCL Implications — uses the tensor shapes
established in Chapter 4 to compute per-device shard sizes and collective
communication volumes across the 8-device T3K mesh. The paging layer discussed
here is a single-device concern; Chapter 6 assumes paging has been resolved and
focuses on the multi-device distribution of the resulting KV tensors.
