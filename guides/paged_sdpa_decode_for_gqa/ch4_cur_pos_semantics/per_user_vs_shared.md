# Per-User vs. Shared `cur_pos`

## Per-User Semantics

Each batch element has a fully independent `cur_pos[i]`.  This is what makes
continuous batching possible: requests join and leave the batch at different
times, so their KV caches grow at different rates.

```
Batch element 0: started 200 steps ago  → cur_pos[0] = 200
Batch element 1: started   5 steps ago  → cur_pos[1] =   5
Batch element 2: inactive (padding)     → cur_pos[2] =  -1
Batch element 3: started  47 steps ago  → cur_pos[3] =  47
```

The kernel applies a separate causal mask per element — element 0 attends to
200 KV positions, element 1 attends to 5, element 2 is skipped entirely.

---

## Common Mistake 1 — Scalar Instead of a Length-`b` List

```python
b = 4
seq_len = 512

# WRONG — scalar, not a list
cur_pos = seq_len          # Python int, not list

# CORRECT
cur_pos = [seq_len] * b    # [512, 512, 512, 512]
```

Passing a scalar is either rejected by the API or silently applied as a
shared value.  Even if it happens to work today, it bypasses per-element
masking and will produce incorrect output when batch elements diverge.

---

## Common Mistake 2 — Off-by-One: Token Index vs. Cache Length

This is the most dangerous mistake because it silently corrupts output with
no error or assert.

```
After writing token t=0 (the first decode token):
  Cache contains 1 valid token at index 0.
  Correct:   cur_pos[i] = 1   (length of valid prefix)
  WRONG:     cur_pos[i] = 0   (index of the slot just written)
```

Effect of the off-by-one:

```
Correct  (cur_pos=1): causal mask allows keys at positions {0}
Wrong    (cur_pos=0): causal mask allows keys at positions {}   ← empty!
                      the current query attends to nothing
```

At later steps the mask shifts by one, causing each query to miss the most
recently written token and potentially attend past the valid window.

```python
# WRONG — using the write index
for step in range(max_steps):
    paged_update_cache(new_kv, cache, page_table, cur_pos_write=step)
    cur_pos = [step] * b          # off-by-one: this is the index, not length

    output = sdpa_decode(..., cur_pos=cur_pos)

# CORRECT — length = index + 1
for step in range(max_steps):
    paged_update_cache(new_kv, cache, page_table, cur_pos_write=step)
    cur_pos = [step + 1] * b      # length of valid prefix after write

    output = sdpa_decode(..., cur_pos=cur_pos)
```

---

## Common Mistake 3 — Sharing One `cur_pos` When Sequences Have Diverged

```python
# Scenario: two sequences with different lengths
lengths = [200, 5]

# WRONG — using the max or a single value for all elements
shared_pos = max(lengths)         # 200
cur_pos = [shared_pos] * b        # [200, 200]
# Element 1 now attends to 195 positions of garbage beyond its valid prefix.

# CORRECT — use per-element lengths
cur_pos = lengths                 # [200, 5]
```

Over-masking (value too low): the query cannot attend to recently written
tokens — generation becomes incoherent.

Under-masking (value too high): the query attends to uninitialized or
stale KV slots from a previous request — this is a data leak and a
correctness failure.

---

## Side-by-Side Correct vs. Incorrect Usage

```python
import ttnn, torch

b, nh, nkv, dh = 4, 32, 8, 128
block_size, max_num_blocks = 32, 64

# --- INCORRECT ---
cur_pos_wrong = 47                          # scalar — mistake 1
cur_pos_wrong = [46] * b                    # off-by-one — mistake 2 (step=46, length=47)
cur_pos_wrong = [max(seq_lengths)] * b      # shared max — mistake 3

# --- CORRECT ---
seq_lengths = [47, 12, 200, 3]             # actual lengths after this step's write
cur_pos_correct = seq_lengths              # list of length b, each = valid prefix length

output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
    query,          # [1, b, nh, dh]
    key_cache,      # [max_num_blocks, nkv, block_size, dh]
    value_cache,
    cur_pos=cur_pos_correct,
    page_table=page_table,
    scale=dh**-0.5,
)
```
