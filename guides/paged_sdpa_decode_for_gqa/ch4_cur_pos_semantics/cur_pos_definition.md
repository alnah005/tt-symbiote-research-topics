# `cur_pos` Definition

## Precise Semantics

`cur_pos[i]` is the **length of the valid prefix** in the KV cache for batch
element `i`.  It means: tokens at physical positions `[0, cur_pos[i])` are
valid; everything at index `cur_pos[i]` and beyond is garbage or padding.

The kernel enforces this as a causal mask: for query `q` at step `t`, key
position `k` is included in the softmax iff `k < cur_pos[i]`.

**Critical distinction — 0-indexed length, not the slot being written:**

```
Step 0 (prefill done, 0 decode tokens written): cur_pos[i] = 0
Step 1 (first decode token written):            cur_pos[i] = 1
Step 2 (second decode token written):           cur_pos[i] = 2
...
Step t:                                          cur_pos[i] = t
```

`cur_pos[i]` is incremented **after** `paged_update_cache` writes the new
token and **before** (or simultaneously with) the SDPA call that reads it.
The new token must be visible to the attention kernel, so `cur_pos[i]` must
already reflect the write.

## Two Passing Modes

### Mode 1 — Python list (CPU tensor, triggers recompilation)

```python
# cur_pos is a plain Python list of length b
cur_pos = [64, 128, 32, 96]   # one entry per batch element

output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
    query,         # [1, b, nh, dh]
    key_cache,     # [max_num_blocks, nkv, block_size, dh]
    value_cache,
    cur_pos=cur_pos,          # Python list — compiled per unique tuple
    page_table=page_table,
    ...
)
```

Each unique `cur_pos` tuple triggers a new program compilation.  Acceptable
for small batch sizes or static-length workloads; becomes a bottleneck when
positions change every step with a large batch.

### Mode 2 — Device tensor (avoids recompilation)

```python
# cur_pos_tensor lives on device; shape [b] or [1, b], dtype uint32
cur_pos_tensor = ttnn.from_torch(
    torch.tensor([64, 128, 32, 96], dtype=torch.int32),
    device=device,
    layout=ttnn.ROW_MAJOR_LAYOUT,
)

output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
    query,
    key_cache,
    value_cache,
    cur_pos_tensor=cur_pos_tensor,   # device tensor — no recompile
    page_table=page_table,
    ...
)
```

The kernel reads positions from device memory at runtime.  Program is
compiled once and reused regardless of position values.  Preferred for
production inference loops.

**Do not pass both `cur_pos` and `cur_pos_tensor`** — the kernel accepts one
or the other, not both.

## The `-1` Sentinel

Setting `cur_pos[i] = -1` tells the kernel to skip all computation for batch
slot `i`.  The output tensor at index `i` is left undefined (may contain
zeros, garbage, or stale data — do not read it).

Intended use: **variable-length batches** where some slots are inactive.

```python
# Batch of 4; slots 1 and 3 are inactive padding slots
cur_pos = [64, -1, 32, -1]

output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
    query, key_cache, value_cache,
    cur_pos=cur_pos,
    page_table=page_table,
    ...
)
# output[:, 0, :, :] and output[:, 2, :, :] are valid
# output[:, 1, :, :] and output[:, 3, :, :] are UNDEFINED — do not use
```

Padding slots must also have a valid (but irrelevant) page table row; the
kernel may or may not dereference it depending on implementation version.
Safe practice: set the page table row for `-1` slots to all-zeros.
