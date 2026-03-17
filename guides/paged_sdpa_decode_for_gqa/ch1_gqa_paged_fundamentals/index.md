# Chapter 1: GQA and Paged Attention Fundamentals

This chapter establishes the conceptual vocabulary required for the rest of this guide. It covers two independent ideas — Grouped Query Attention (GQA) and paged KV caches — and then explains how they interact in autoregressive decode. Before reading Chapter 2 onward, you must be comfortable with all terms and invariants defined here.

---

## Learning Objectives

By the end of this chapter you will be able to:

1. State the head-count relationship `nh = nkv * group_size` and explain why it must hold exactly.
2. Identify the KV memory reduction factor from GQA for a given `nh` / `nkv` configuration.
3. Describe the paged KV cache block model and explain the role of `page_table_tensor`.
4. Recite the tensor shape change when moving from contiguous to paged K/V storage.
5. Explain why GQA expansion must never be applied before writing K/V into a paged cache.

---

## Prerequisites

Before reading this chapter, confirm you can check each item:

- [ ] You understand standard scaled dot-product attention: `softmax(QK^T / sqrt(dh)) * V`.
- [ ] You know what a KV cache is and why it exists in autoregressive decoding.
- [ ] You can distinguish prefill (processing all prompt tokens) from decode (generating one token at a time).
- [ ] You are familiar with basic TTNN tensor operations: creating tensors, reading `.shape`, applying memory configs.
- [ ] You know what Multi-Head Attention (MHA) is: each of the `nh` query heads has a dedicated K and V head.

You do **not** need to know TTNN Flash-Decode internals; those are introduced in Chapter 2.

---

## Chapter Map

| File | Contents |
|------|----------|
| `gqa_concept.md` | MHA/MQA/GQA definitions; the `nh = nkv * group_size` invariant; memory reduction; broadcast vs. native implementations; historical `repeat_interleave` workaround |
| `paged_kv_cache_concept.md` | Paged block model; supported block sizes; `page_table_tensor` shape and semantics; contiguous vs. paged K/V tensor shapes; motivation for long-context decode |
| `gqa_plus_paging_interaction.md` | How paged storage and GQA interact; the `max_num_blocks` invariant; common mistake: expanding KV heads before writing to cache |

---

## Key Terms (Quick Reference)

| Term | Definition |
|------|-----------|
| `nh` | Number of query heads (e.g., 16 in Ling) |
| `nkv` | Number of KV heads (e.g., 4 in Ling) |
| `group_size` | `nh / nkv`; query heads per KV head |
| `dh` | Head dimension |
| `b` | Batch size |
| `s` | Full KV cache sequence capacity (max storable tokens) |
| `block_size` | Tokens per paged KV block; one of 32, 64, 128 |
| `max_num_blocks_per_seq` | Maximum blocks per sequence: `ceil(s / block_size)` |
| `max_num_blocks` | Total physical blocks allocated: `b * max_num_blocks_per_seq` |
| `cur_pos[i]` | Count of valid KV tokens for batch element `i` after the current write |
| `page_table_tensor` | Shape `[b x max_num_blocks_per_seq]`; maps logical block index to physical block index |

Full terminology table is in the [plan](../plan.md#3-conventions).

---

## Forward References to Later Chapters

The concepts here are intentionally kept at the model level. The following topics are deferred:

- **Chapter 2** covers the complete `ttnn.transformer.scaled_dot_product_attention_decode` API signature, including every parameter shape and the difference between paged and non-paged call modes.
- **Chapter 3** covers exactly how TTNN expects `nkv` and `nh` to appear in tensor memory after padding to tile-alignment multiples of 32, and the silent correctness bug that arises when padding changes the effective `group_size`.
- **Chapter 4** covers the precise semantics of `cur_pos` and `cur_pos_tensor`, including the off-by-one interpretation and the `-1` skip behavior.
- **Chapter 5** catalogs known open bugs, including the sporadic PCC failures at block boundaries (issue `tenstorrent/tt-metal#30362`).

---

## Next Steps

Read the section files in order:

1. `gqa_concept.md` — understand GQA before paging
2. `paged_kv_cache_concept.md` — understand paging before their interaction
3. `gqa_plus_paging_interaction.md` — combine both concepts

Then proceed to **Chapter 2: TTNN paged_sdpa_decode API** (`../ch2_ttnn_api/index.md`).
