# Chapter 4: `cur_pos` Semantics

## Quick Reference

`cur_pos[i]` = number of valid tokens currently in the KV cache for batch element `i`.

- It is the length of the valid prefix, not an index into it.
- After the first decode step, `cur_pos[i] = 1` (one token has been written).
- Special value `-1`: skip all computation for batch index `i`; output undefined.

## Learning Objectives

By the end of this chapter you should be able to:

1. State the exact definition of `cur_pos[i]` and distinguish it from the token index being written.
2. Choose between the Python-list form (`cur_pos`) and the device-tensor form (`cur_pos_tensor`) and explain the compilation trade-off.
3. Identify the three most common off-by-one and shape mistakes and know how to avoid them.
4. Explain how `cur_pos[i]` drives both causal masking and physical block selection in paged KV mode.
5. Describe the block-boundary PCC failure mode tracked in GitHub Issue #30362.

## Prerequisites

- **Chapter 1** — SDPA decode overview: Q/K/V tensor shapes, GQA head mapping.
- **Chapter 2** — Paged KV layout: block table structure, `max_num_blocks`, `block_size`.
- **Chapter 3** — `paged_update_cache`: how tokens are written into physical blocks before `cur_pos` is incremented.

## File Map

| File | Content |
|---|---|
| [`cur_pos_definition.md`](./cur_pos_definition.md) | Precise semantics, passing modes, and the `-1` sentinel |
| [`per_user_vs_shared.md`](./per_user_vs_shared.md) | Per-user independence, common mistakes with code examples |
| [`cur_pos_in_paged_mode.md`](./cur_pos_in_paged_mode.md) | Block selection, partial-block masking, Issue #30362 |

## Tensor Shape Reminder

```
Q:             [1  x b  x nh  x dh]
K / V (dense): [b  x nkv x s   x dh]
K / V (paged): [max_num_blocks x nkv x block_size x dh]
paged_update_cache input: [b x nkv x 1 x dh]
```

GQA mapping: `kv_head_idx = q_head_idx // group_size`

## Why `cur_pos` Matters

Every decode call needs to know two things per batch element:

1. **Where to read**: which KV positions contain valid data.
2. **Where to write**: which cache slot will receive the new token (handled by `paged_update_cache` before this call, so by the time SDPA runs the write has already happened).

`cur_pos[i]` answers both questions.  Getting it wrong by even one position
silently corrupts attention output — the causal mask shifts, letting the
current query attend to a position it should not see, or masking out a
position it should see.
