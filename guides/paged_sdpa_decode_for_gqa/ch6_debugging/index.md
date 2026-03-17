# Chapter 6: Debugging Incorrect Decode Output

## Overview

This chapter provides a structured debugging workflow for incorrect output from
`ttnn.transformer.paged_scaled_dot_product_attention_decode` with GQA. Each section
targets a distinct failure class. Work through the ladder in order — earlier checks
are cheap and eliminate a large fraction of bugs before reaching the expensive ones.

## Prerequisites

Before starting, you need a **reference PyTorch implementation** of the same attention
operation using the identical Q/K/V tensors (including padding zeros), the same scale
factor, and the same `is_causal` setting. Without a reference you cannot measure PCC
or confidently attribute a mismatch to a specific cause.

Minimum reference:

```python
import torch
import torch.nn.functional as F

def ref_sdpa(Q, K, V, scale, is_causal=False):
    # Q: [1, b, nh, dh] -> [b, nh, 1, dh]
    q = Q.squeeze(0).permute(0, 2, 1, 3)          # [b, nh, 1, dh]
    # K/V: [b, nkv, s, dh] -> expand to [b, nh, s, dh] for GQA
    group_size = Q.shape[2] // K.shape[1]
    k = K.repeat_interleave(group_size, dim=1)     # [b, nh, s, dh]
    v = V.repeat_interleave(group_size, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=is_causal)
    return out.permute(0, 2, 1, 3).unsqueeze(0)    # [1, b, nh, dh]
```

## Debug Ladder (cheapest → most expensive)

| Step | File | Cost | Catches |
|------|------|------|---------|
| 1 | `shape_validation_checklist.md` | seconds | wrong rank, bad GQA ratio, wrong axis ordering |
| 2 | `cur_pos_validation.md` | minutes | off-by-one, batch scalar vs list, wrong block count |
| 3 | `pcc_comparison_workflow.md` | minutes–hours | numerical divergence, layout bugs, boundary bugs |
| 4 | `root_cause_isolation.md` | hours | paging logic vs attention logic vs kernel |

## File Map

```
ch6_debugging/
├── index.md                      ← this file
├── shape_validation_checklist.md ← Step 1: tensor shape audit
├── cur_pos_validation.md         ← Step 2: cur_pos correctness
├── pcc_comparison_workflow.md    ← Step 3: numerical comparison
└── root_cause_isolation.md       ← Step 4: root cause flowchart
```

## Tensor Shape Quick Reference

| Tensor | Expected shape |
|--------|----------------|
| Q | `[1, b, nh, dh]` |
| K / V (contiguous) | `[b, nkv, s, dh]` |
| K / V (paged) | `[max_num_blocks, nkv, block_size, dh]` |
| page table | `[b, max_num_blocks_per_seq]`, int32 |
| `paged_update_cache` input | `[b, nkv, 1, dh]` |

## Key Identities

- GQA group size: `group_size = nh / nkv`
- Correct padding: `nkv_padded = nh_padded / original_group_size`
- `cur_pos[i]` = post-write KV length (number of tokens already in cache)
- Issue #30362 is open as of early 2026; see `ch5_known_issues/` for details
