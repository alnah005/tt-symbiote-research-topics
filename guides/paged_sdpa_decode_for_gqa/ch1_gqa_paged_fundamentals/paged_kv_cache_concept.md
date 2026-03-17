# Paged KV Cache Concept

This file explains what a paged KV cache is, how TTNN implements the paged block model, and how paged storage changes the tensor shapes for K and V. It also explains why paged attention is necessary for long-context decode and dynamic batching.

---

## 1. The Problem with Contiguous KV Caches

In a contiguous KV cache, every batch element pre-allocates a contiguous tensor slice of length `s` (the maximum sequence length). This has two costs:

1. **Static allocation waste:** If most sequences are short, most of the allocated memory is empty but still reserved.
2. **Fragmentation under dynamic batching:** When sequences complete and new ones start, their length histories differ. Fitting new sequences into a contiguous allocator causes fragmentation similar to heap fragmentation in C.

For long-context models (128K token context windows), even a single sequence occupies gigabytes per layer in a full contiguous allocation. Maintaining a pool of such allocations for a dynamic batch is impractical.

Paged KV caches solve both problems by borrowing the virtual memory paging concept: allocate storage in fixed-size blocks and map logical positions to physical blocks on demand.

---

## 2. The Paged Block Model

A paged KV cache divides the token dimension into **logical blocks** of a fixed `block_size`. Each logical block holds `block_size` consecutive token positions. Physical storage is a pool of fixed-size blocks that can be assigned to any sequence, in any order.

The mapping from a sequence's logical block index to a physical block index is stored in a **page table**.

```
Sequence (logical view):
  token 0  ... token 31    →  logical block 0
  token 32 ... token 63    →  logical block 1
  token 64 ... token 95    →  logical block 2
  ...

Page table (per sequence):
  logical block 0  →  physical block 7
  logical block 1  →  physical block 2
  logical block 2  →  physical block 15
  ...

Physical KV storage:
  block 0:  [32 token slots x nkv x dh]
  block 1:  [32 token slots x nkv x dh]
  ...
  block 7:  ← holds tokens 0–31 of this sequence
  ...
```

Physical blocks are shared across all sequences in the batch. A block not yet assigned to any sequence is "free" and can be claimed at runtime as sequences grow longer.

---

## 3. Supported Block Sizes in TTNN

TTNN's Flash-Decode kernel supports three block sizes:

| `block_size` | Token slots per physical block |
|--------------|-------------------------------|
| 32 | 32 |
| 64 | 64 |
| 128 | 128 |

The `block_size` must be chosen at model initialization time and kept consistent across all calls to `paged_update_cache` and `scaled_dot_product_attention_decode` for a given KV cache tensor.

> **Tip:** Larger block sizes reduce page table overhead (fewer entries, fewer block pointer lookups) but increase the granularity of wasted space at sequence ends (a sequence that is 1 token into a new block still occupies a full block). For 128K-token sequences, `block_size=128` is typical.

---

## 4. The Page Table Tensor

The page table is represented in TTNN as `page_table_tensor`, a device-resident integer tensor with shape:

```
page_table_tensor: [b x max_num_blocks_per_seq]
```

Where:
- `b` is the batch size
- `max_num_blocks_per_seq = ceil(s / block_size)` is the maximum number of logical blocks any single sequence can use

`page_table_tensor[i, j]` is the **physical block index** for logical block `j` of batch element `i`. The kernel uses this to compute the starting address of the K/V data for any token position.

For a token at position `pos` in batch element `i`, see `gqa_plus_paging_interaction.md` for a worked example of physical block address computation including GQA head selection.

> **Warning:** `page_table_tensor` must be row-major int32 on device. Using the wrong dtype (e.g., int64 or float32) or the wrong layout (e.g., tile layout) causes the kernel to read incorrect physical block indices. This is a silent failure: no error is raised and the output is numerically wrong.

---

## 5. Tensor Shape: Contiguous vs. Paged

### Contiguous (Non-Paged) K/V Shape

In the non-paged mode, the KV cache is a single contiguous tensor per K and V:

```
K: [b x nkv x s x dh]
V: [b x nkv x s x dh]
```

Dimension breakdown:
- `b`: batch axis (one slice per sequence)
- `nkv`: KV head axis
- `s`: token/sequence axis (full capacity, `s` positions pre-allocated)
- `dh`: head dimension axis

---

### Paged K/V Shape

In paged mode, the K and V tensors are reshaped to a flat pool of physical blocks:

```
K: [max_num_blocks x nkv x block_size x dh]
V: [max_num_blocks x nkv x block_size x dh]
```

Dimension breakdown:
- `max_num_blocks = b * max_num_blocks_per_seq`: total physical blocks in the pool
- `nkv`: KV head axis (same as contiguous)
- `block_size`: token slots within a single block
- `dh`: head dimension axis

The batch dimension `b` is **gone** from the K/V tensor shape. Batch identity is entirely encoded in the `page_table_tensor`. Two sequences can share physical blocks only if the page table routes them to different block indices; in practice, blocks are not shared between sequences (copy-on-write sharing is not a TTNN feature).

---

### Shape Comparison

| Mode | K/V shape | Batch encoded by |
|------|-----------|-----------------|
| Contiguous | `[b x nkv x s x dh]` | Axis 0 (direct slice) |
| Paged | `[max_num_blocks x nkv x block_size x dh]` | `page_table_tensor` indirection |

---

## 6. Writing to the Paged Cache: `paged_update_cache`

During each decode step, one new K/V token is computed for each batch element. TTNN provides the `paged_update_cache` op to write these tokens into the correct physical block positions.

The input to `paged_update_cache` for each decode step is:

```
new_K: [b x nkv x 1 x dh]   (one token per batch element)
new_V: [b x nkv x 1 x dh]
```

> **Note:** The `1` in the third dimension is the single-token decode dimension — one new token is produced per batch element per decode step. In prefill mode, `paged_update_cache` takes `s_chunk` tokens instead of `1`, where `s_chunk` is the number of tokens in the current prefill chunk.

The op consults the `page_table_tensor` and `cur_pos` (or `cur_pos_tensor`) to determine the physical block and offset where each token should be written. After the write, the paged K/V cache tensors reflect the updated content.

> **Warning:** The new K/V input to `paged_update_cache` must have `nkv` heads, not `nh` heads. Passing a tensor with `nh` heads (i.e., the expanded GQA layout) writes `nh` KV slices per token instead of `nkv`. This corrupts the cache by overwriting the wrong block offsets with Q-head-structured data. See `gqa_plus_paging_interaction.md` for a detailed explanation.

---

## 7. Why Paged Attention Enables Long-Context Decode

### Long-Context Scenario

A 128K-token context window with `nkv=4`, `dh=128`, BF16, and 32 layers:

| Cache type | Memory per layer | Total (32 layers) |
|------------|-----------------|-------------------|
| Contiguous, b=1 | `4 * 128 * 131072 * 2` = 128 MB | 4 GB |
| Contiguous, b=8 | 1 GB per layer | 32 GB |

Pre-allocating 32 GB of contiguous KV cache for 8 simultaneous sequences at 128K tokens each is feasible but inflexible. With paged storage, blocks are allocated only as sequences grow. An 8-sequence batch where sequences average 10K tokens uses `~10%` of the peak allocation.

### Dynamic Batching Scenario

In continuous batching (serving multiple users with independent sequence lengths):
- Sequence A completes at 3000 tokens. Its physical blocks are freed.
- Sequence B starts and is assigned those same physical blocks.
- No memory copy or defragmentation is required.

The contiguous model cannot do this: once sequence A's 3000-token slot is freed in a contiguous allocator, the space can only be reused by another sequence that fits within that exact allocation shape.

> **Tip:** Even if your current use case is batch=1 with a known sequence length, using paged attention makes your code compatible with production serving infrastructure that uses dynamic batching. Starting with a paged implementation avoids a costly migration later.

---

## Summary

| Concept | Detail |
|---------|--------|
| Physical block | Fixed-size token slot pool, shared across batch |
| Supported block sizes | 32, 64, 128 tokens |
| `page_table_tensor` shape | `[b x max_num_blocks_per_seq]`, int32, row-major |
| Paged K/V shape | `[max_num_blocks x nkv x block_size x dh]` |
| Contiguous K/V shape | `[b x nkv x s x dh]` |
| `paged_update_cache` input | `[b x nkv x 1 x dh]` per decode step |

---

## Next Steps

Continue to `gqa_plus_paging_interaction.md` to see how paged storage and GQA head counts interact, and where the most common conceptual mistake occurs.
