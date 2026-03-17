# Paged KV Cache Layout for GQA

## Overview

This file describes the shape of the paged KV (key/value) cache, what each dimension represents, and the silent correctness mistake that occurs when expanded Q-head-count data is written into the cache instead of KV-head-count data.

---

## Paged KV Cache Shape

**Shape: `[max_num_blocks x nkv x block_size x dh]`**

| Axis | Symbol | Meaning |
|------|--------|---------|
| 0 | `max_num_blocks` | Total number of blocks available across all batch elements |
| 1 | `nkv` | KV head count — the **actual model KV head count**, not expanded to nh |
| 2 | `block_size` | Number of consecutive token positions stored per block |
| 3 | `dh` | Head dimension |

The `max_num_blocks` axis is the paged dimension. Each batch element is assigned a subset of blocks via a block table (a mapping from logical sequence positions to physical block indices). This is what makes the layout "paged": different batch elements can hold different numbers of tokens, and blocks need not be contiguous in memory.

Each block stores `block_size` consecutive token positions for **all `nkv` KV heads simultaneously**. When the kernel reads a block for attention, it retrieves all heads for those token positions in one operation.

---

## `paged_update_cache` Input Shape

When writing a new decode token into the paged cache, `paged_update_cache` expects the input tensor in this shape:

**Shape per decode step: `[b x nkv x 1 x dh]`**

| Axis | Symbol | Meaning |
|------|--------|---------|
| 0 | `b` | Batch size |
| 1 | `nkv` | KV head count — **actual model nkv, not nh** |
| 2 | `1` | One token position (the current decode step) |
| 3 | `dh` | Head dimension |

The `1` in axis 2 mirrors the leading `1` in the Q tensor shape: one token is being appended per decode step per batch element.

---

## The Expanded-Head-Count Write Mistake

A common mistake is passing a tensor of shape `[b x nh x 1 x dh]` — using the **Q head count** instead of the KV head count — to `paged_update_cache`.

This mistake arises because the Q tensor after `nlp_create_qkv_heads_decode` has `nh` heads, and it can be tempting to write the corresponding new KV values using the same head axis size. However, the KV cache stores **one copy of each KV head**, not one copy per Q head.

### Why This Is Wrong

GQA's memory reduction comes from storing `nkv` KV heads, not `nh`. Multiple Q heads share a single KV head during attention. If you write `nh` copies of the KV data:

1. The cache grows by a factor of `group_size` unnecessarily, consuming `group_size x` more memory.
2. The paged block table maps logical positions to blocks sized for `nkv` heads. Reading back `nh`-sized writes returns misaligned data — the kernel reads the wrong tokens for the wrong heads.
3. No error is raised. The attention scores are computed over incorrect KV data.

> **Warning:** Do not pass `[b x nh x 1 x dh]` to `paged_update_cache`. The paged KV cache is sized and indexed for `nkv` heads. Writing `nh` heads bloats the cache by `group_size x` and causes the kernel to read back wrong KV data during attention — silently, with no runtime error.

### Correct vs. Incorrect Write

```python
# Correct: write nkv heads (actual KV head count)
new_kv_for_cache  # shape [b x nkv x 1 x dh]
paged_update_cache(kv_cache, new_kv_for_cache, update_idxs, ...)

# Incorrect: write nh heads (expanded Q head count)
new_kv_for_cache  # shape [b x nh x 1 x dh]  ← wrong
# This will silently write group_size x more data per step,
# misaligning the block table and producing wrong attention scores.
```

---

## Why the Cache Uses nkv, Not nh

The entire purpose of GQA is that multiple Q heads share one KV head. The KV cache exists to store the keys and values that the Q heads attend over. Since each KV head is shared by `group_size` Q heads, there is only one unique KV value per KV head — storing it once is correct. Storing `nh` copies would re-introduce the memory cost that GQA was designed to eliminate, while also corrupting the block layout the kernel depends on.

```
GQA memory reduction (correct):
  KV cache entries per token = nkv x dh x 2   (K and V)
  e.g., nkv=4, dh=128: 1024 values per token

Expanded-head-count write (incorrect):
  KV cache entries per token = nh x dh x 2
  e.g., nh=16, dh=128: 4096 values per token  ← 4x bloat, wrong indexing
```

---

## Summary

| Property | Correct Value | Common Mistake |
|----------|--------------|----------------|
| Paged KV cache shape | `[max_num_blocks x nkv x block_size x dh]` | Using `nh` for axis 1 |
| `paged_update_cache` input shape | `[b x nkv x 1 x dh]` | `[b x nh x 1 x dh]` |
| Head count stored per token | `nkv` (actual model KV heads) | `nh` (expanded Q heads) |

---

## Next Steps

This completes Chapter 3. Proceed to **Chapter 4** for the end-to-end decode loop: combining the tensor shapes from this chapter with `paged_sdpa_decode`, `paged_update_cache`, and the block table to run a complete autoregressive decode step.
