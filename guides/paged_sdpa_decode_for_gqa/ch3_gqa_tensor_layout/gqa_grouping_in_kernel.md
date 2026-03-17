# GQA Grouping in the Flash-Decode Kernel

## Overview

This file explains the single formula by which the Flash-Decode kernel implements Grouped-Query Attention (GQA), the hard correctness requirement on head counts, and the silent padding bug that can collapse GQA into Multi-Head Attention (MHA) behavior.

---

## The Kernel's GQA Formula

The Flash-Decode kernel maps each Q head to a KV head using integer division:

```python
kv_head_idx = q_head_idx // group_size
```

This one line is the **sole mechanism** by which multiple Q heads share a single KV head. There is no other GQA-specific logic in the kernel. The formula works correctly as long as `group_size = nh / nkv` is an integer and the padded head counts preserve this ratio.

### Example

With `nh = 16`, `nkv = 4`, `group_size = 4`:

| q_head_idx | kv_head_idx = q_head_idx // 4 |
|-----------|-------------------------------|
| 0, 1, 2, 3 | 0 |
| 4, 5, 6, 7 | 1 |
| 8, 9, 10, 11 | 2 |
| 12, 13, 14, 15 | 3 |

Each KV head is used by exactly 4 Q heads. The GQA memory reduction is achieved because the KV cache stores only 4 heads rather than 16.

---

## Correctness Requirement: `nh % nkv == 0`

The kernel requires that `nh` is evenly divisible by `nkv`. This is the mathematical precondition for `group_size` to be a well-defined integer.

**The kernel does NOT check this condition.** If `nh % nkv != 0`, the integer division produces a non-uniform mapping — some KV heads receive more Q heads than others — and the attention output is incorrect. No error or warning is raised.

```python
# Correct: nh is evenly divisible by nkv
nh, nkv = 16, 4     # group_size = 4, nh % nkv == 0  ✓

# Silent failure: uneven mapping
nh, nkv = 17, 4     # group_size = 4 (floor), but Q head 16 maps to KV head 4
                     # KV head 4 does not exist for nkv=4 — undefined behavior, no error
```

---

## The TTNN 32-Padding Silent Bug

`nlp_create_qkv_heads_decode` pads `nh` and `nkv` to multiples of 32 to satisfy hardware tile requirements. If this padding is applied independently to both axes, the `group_size` ratio can change silently.

### Failure Scenario

Starting configuration: `nh = 16`, `nkv = 4`, `group_size = nh / nkv = 4`.

If both are padded independently to the nearest multiple of 32:

```
pnh        = 32   (16 padded to next multiple of 32)
nkv_padded = 32   (4 padded to next multiple of 32)

effective_group_size = pnh / nkv_padded = 32 / 32 = 1
```

An effective `group_size` of 1 means every Q head maps to a unique KV head — this is **MHA behavior**, not GQA. All 16 padded Q heads now attend to the same single-KV-head block (since nkv_padded = 32 but the model only has 4 real KV heads, padded entries are zeros or garbage). The model computes attention with wrong KV data. No error is raised.

> **Warning:** If `nkv` is padded independently to a multiple of 32, and `nkv_padded != pnh / original_group_size`, the effective group_size seen by the kernel changes silently. The model does not raise an error but produces incorrect attention outputs — GQA collapses to MHA or worse. Always derive `nkv_padded` from `pnh` and `original_group_size`, not by independently rounding `nkv`.

### Correct Padding Strategy

Pad `nkv` in a way that keeps `pnh / nkv_padded == original_group_size`:

```python
original_group_size = nh // nkv          # e.g., 16 // 4 = 4

pnh = ((nh + 31) // 32) * 32             # pad nh to nearest multiple of 32
                                          # e.g., pnh = 32

nkv_padded = pnh // original_group_size  # derive nkv_padded from pnh
                                          # e.g., 32 // 4 = 8

# Verify:
effective_group_size = pnh // nkv_padded  # 32 // 8 = 4  ✓ matches original
```

### Comparison Table

| Scenario | pnh | nkv_padded | effective_group_size | Correct? |
|----------|-----|------------|---------------------|----------|
| nh=16, nkv=4, padded correctly | 32 | 8 | 4 | Yes |
| nh=16, nkv=4, nkv padded independently | 32 | 32 | 1 | No — silent MHA collapse |
| nh=32, nkv=4, padded correctly | 32 | 4 | 8 | Yes |

---

## Summary

- The kernel computes `kv_head_idx = q_head_idx // group_size`. This is the entirety of GQA in the kernel.
- `nh % nkv == 0` is required but not checked. Violation produces wrong output, not an error.
- Padding `nkv` independently can silently change `effective_group_size` to 1, collapsing GQA to MHA with no error raised.
- Always compute `nkv_padded = pnh // original_group_size` to preserve the head ratio across padding.

---

## Next Steps

Continue to [`paged_layout_for_gqa.md`](paged_layout_for_gqa.md) to see how the paged KV cache stores GQA tensors, and why writing expanded Q-head-count data into the cache is a common and silent correctness mistake.
