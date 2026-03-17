# GQA and Paged KV Cache Interaction

This file explains how Grouped Query Attention (GQA) and paged KV caches interact during autoregressive decode. It establishes the key invariants, illustrates the physical block addressing model, and identifies the most common conceptual mistake that causes correctness failures.

---

## 1. The Two Dimensions Are Independent (At First)

It helps to understand that paged block addressing and GQA head grouping operate on different axes of the K/V tensor:

- **Paged block addressing** concerns the `block_size` dimension (token positions within a block) and `max_num_blocks` (which physical block to access). It uses `page_table_tensor` as the indirection layer.
- **GQA head grouping** concerns the `nkv` dimension (which K/V head to load for a given Q head). It uses the formula `kv_head_idx = q_head_idx // group_size`.

These two operations act on different axes of the paged K/V tensor `[max_num_blocks x nkv x block_size x dh]`:
- The block dimension (`max_num_blocks`) is resolved by the page table.
- The head dimension (`nkv`) is resolved by the GQA group mapping.
- The within-block token offset (`block_size`) is resolved from `cur_pos[i] % block_size`.

Because they are orthogonal, each can be reasoned about independently — as long as the K/V tensor carries `nkv` heads (not `nh` heads). If `nh` heads are stored in the cache, the GQA grouping and the block addressing both break.

---

## 2. The `max_num_blocks` Invariant

For a batch of `b` sequences, each capable of storing up to `max_num_blocks_per_seq` logical blocks, the total physical block pool must satisfy:

```
max_num_blocks = b * max_num_blocks_per_seq
```

This invariant ensures that the physical K/V tensor has enough slots to assign `max_num_blocks_per_seq` blocks to every batch element without any block being shared. In practice, at initialization time you allocate:

```python
max_num_blocks_per_seq = math.ceil(max_seq_len / block_size)
max_num_blocks = batch * max_num_blocks_per_seq

kv_cache_shape = [max_num_blocks, nkv, block_size, dh]
page_table = torch.arange(max_num_blocks).reshape(batch, max_num_blocks_per_seq)
```

The default page table is simply a sequential assignment: batch element `i` owns physical blocks `[i * max_num_blocks_per_seq, ..., (i+1) * max_num_blocks_per_seq - 1]`. In a production serving system, a block allocator manages these assignments dynamically. For testing, the sequential default is sufficient.

> **Tip:** When debugging, print both `page_table_tensor.shape` and `kv_cache.shape[0]`. The first dimension of the KV cache must equal the total number of entries in the page table: `batch * max_num_blocks_per_seq`.

---

## 3. Physical Block Addressing: Step-by-Step

Given a query at decode step where batch element `i` has `cur_pos[i]` valid KV tokens, the kernel retrieves the K/V data for token at position `pos` (where `0 <= pos < cur_pos[i]`) as follows:

```
Step 1: Compute logical block and offset
        logical_block   = pos // block_size
        offset_in_block = pos % block_size

Step 2: Look up physical block from page table
        physical_block  = page_table_tensor[i, logical_block]

Step 3: Index into physical K/V storage
        K_token = K_cache[physical_block, :, offset_in_block, :]
        V_token = V_cache[physical_block, :, offset_in_block, :]
```

At step 3, `K_token` has shape `[nkv x dh]` — it contains all `nkv` KV heads for that one token position. The GQA mapping then selects the head slice for query head `q`:

```
        kv_head_idx = q // group_size
        K_for_q     = K_token[kv_head_idx, :]   # shape: [dh]
```

The block dimension and the head dimension are resolved in sequence, but they are independent lookups.

---

## 4. Illustration: 2-Sequence Batch with 4 KV Heads

Consider: `b=2`, `nkv=4`, `nh=16`, `dh=64`, `block_size=32`, `max_num_blocks_per_seq=4`.

```
max_num_blocks = 2 * 4 = 8

Physical KV cache shape: [8 x 4 x 32 x 64]
                          ^   ^   ^    ^
                          |   |   |    head dim
                          |   |   tokens per block
                          |   KV heads
                          physical blocks (0..7)

Page table: [2 x 4]
  seq 0 →  [0, 1, 2, 3]   (physical blocks 0–3)
  seq 1 →  [4, 5, 6, 7]   (physical blocks 4–7)

To access token 50 of sequence 1 with query head 9:
  logical_block    = 50 // 32 = 1
  offset_in_block  = 50 % 32  = 18
  physical_block   = page_table[1, 1] = 5
  K slice          = K_cache[5, :, 18, :]   # shape [4 x 64], all 4 KV heads
  kv_head_idx      = 9 // 4   = 2
  K for Q head 9   = K_cache[5, 2, 18, :]   # shape [64]
```

This is the kernel's internal logic. The caller only provides the `page_table_tensor`, `cur_pos`, and the paged K/V tensor; the kernel performs these index computations on device.

---

## 5. The Common Conceptual Mistake: Expanding KV Before the Cache Write

The most frequent correctness error when porting a GQA model to paged TTNN attention is applying the GQA expansion (from `nkv` heads to `nh` heads) **before** writing K/V into the paged cache. This happens when a developer:

1. Correctly understands that the attention computation needs all `nh` KV slices (one per Q head), and
2. Incorrectly applies `repeat_interleave` at the projection output stage before `paged_update_cache`.

### What Goes Wrong

If K/V tensors with `nh=16` heads are written into a cache initialized for `nkv=4` heads, the shapes are inconsistent in one of two ways:

**Scenario A: Cache was initialized with `nkv=4`.**
The `paged_update_cache` op receives a `[b x 16 x 1 x dh]` input when the cache tensor has shape `[max_num_blocks x 4 x block_size x dh]`. This is a shape mismatch that may or may not raise an error depending on TTNN version; if it silently proceeds, it writes 16 KV slices into a 4-slot buffer, overflowing into adjacent head or block storage.

**Scenario B: Cache was (wrongly) initialized with `nkv=16`.**
The developer expanded the cache to match the expanded K/V. Now the cache holds `nh=16` KV heads per token. When the Flash-Decode kernel is called with `nkv=16` and `nh=16`, it computes `group_size = 1`. Q head `i` attends to expanded KV head `i`, which is a copy of original KV head `i // group_size` — the correct GQA mapping. The attention output is numerically **correct**. However, this is a **performance and memory waste, not a correctness failure**: the cache stores `nh` KV heads instead of `nkv`, discarding GQA's memory reduction benefit entirely and consuming `group_size×` more memory and bandwidth than necessary.

### The Correct Approach

```
                         ┌──────────────────────────────────────────┐
                         │  CORRECT GQA + PAGED DECODE PIPELINE     │
                         └──────────────────────────────────────────┘

  Linear projections:
    Q:  [b x nh  x 1 x dh]   (16 heads)
    K:  [b x nkv x 1 x dh]   (4 heads — do NOT expand)
    V:  [b x nkv x 1 x dh]   (4 heads — do NOT expand)

  Write to paged cache:
    paged_update_cache(K, V, cache, page_table, cur_pos)
    → cache shape remains [max_num_blocks x nkv x block_size x dh]

  Attention call:
    output = scaled_dot_product_attention_decode(
        Q,              # [1 x b x nh  x dh]  (16 heads)
        K_cache,        # [max_num_blocks x nkv x block_size x dh]  (4 heads)
        V_cache,        # same
        page_table_tensor=page_table,
        cur_pos=cur_pos,
    )
    # Kernel internally maps: kv_head_idx = q_head_idx // group_size
```

The key rule: **K and V travel through the system as `nkv`-head tensors at all times.** The GQA broadcast happens inside the kernel, not in Python-level model code.

---

## 6. Head Count Invariants Summary

| Invariant | Expression | If violated |
|-----------|-----------|-------------|
| GQA head divisibility | `nh % nkv == 0` | Wrong K/V head selected for some Q heads; silent wrong output |
| Cache head count matches `nkv` | `kv_cache.shape[1] == nkv` | Block addressing corrupted or wrong group_size |
| Physical block pool size | `max_num_blocks == b * max_num_blocks_per_seq` | Page table addresses physical blocks that don't exist |
| Page table shape | `page_table.shape == [b, max_num_blocks_per_seq]` | Kernel reads wrong physical block for some batch elements |
| `cur_pos[i]` count semantics | `cur_pos[i]` = tokens written, not index being written | Attention mask off-by-one; current token attends to wrong positions |

---

## 7. Forward Reference: Padding Effects on `nkv`

TTNN's tile layout requires tensor dimensions to be multiples of 32. For a model with `nkv=4` and `nh=16`, padding may change the effective head counts seen by the kernel:

- `nkv=4` padded to `nkv_padded=32`
- `nh=16` padded to `nh_padded=32`
- Effective `group_size = 32 / 32 = 1` — kernel sees MHA behavior (group_size=1: each Q head gets its own KV head), not GQA grouping

This is a silent correctness failure. The full treatment of padding and its implications for GQA group size is in **Chapter 3: GQA Tensor Layout Requirements** (`../ch3_gqa_tensor_layout/gqa_grouping_in_kernel.md`).

---

## Next Steps

You have now covered all of Chapter 1. Proceed to **Chapter 2: TTNN paged_sdpa_decode API** (`../ch2_ttnn_api/index.md`) to see how these conceptual shapes map to the exact parameter types and constraints of the `ttnn.transformer.scaled_dot_product_attention_decode` function call.
