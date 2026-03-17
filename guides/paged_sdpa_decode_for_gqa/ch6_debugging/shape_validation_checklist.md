# Shape Validation Checklist

Run this checklist before any numerical comparison. Shape bugs produce misleading PCC
values and are almost always cheaper to fix than tracing a kernel.

## 1. Extracting Shapes

For host tensors:

```python
print(tensor.shape)          # torch.Size([...])
```

For device tensors (TT):

```python
print(tensor.shape)                  # preferred — returns Shape object
print(tensor.get_legacy_shape())     # fallback for older API versions
```

Always print both the logical shape and the memory layout/shard spec when in doubt.

## 2. Q Tensor

Expected: `[1, b, nh, dh]`

- Axis 0 must be **1** (single decode step, not a sequence).
- Axis 1 is the batch dimension `b`.
- Axis 2 is `nh` (query heads).
- Axis 3 is `dh` (head dimension).

Common mistake: transposing `b` and `nh`, or passing `[b, 1, nh, dh]` instead of
`[1, b, nh, dh]`.

## 3. K / V Tensors (Contiguous Path)

Expected: `[b, nkv, s, dh]`

- Axis 0 is **batch** (`b`). This is a known historical layout bug — older code placed
  `nkv` on axis 0 and `b` on axis 1. Confirm axis 0 == `b` before proceeding.
- Axis 1 is `nkv` (KV heads, fewer than query heads in GQA).
- Axis 2 is `s` (current sequence length / cache capacity).
- Axis 3 is `dh`.

## 4. K / V Tensors (Paged Path)

Expected: `[max_num_blocks, nkv, block_size, dh]`

- `max_num_blocks` = total block pool size (across all batch elements).
- `block_size` is typically 32 or 64; must match the value used in page table arithmetic.
- The paged tensor does **not** have a `b` axis — batch is addressed indirectly via the
  page table.

## 5. GQA Ratio Preservation After Padding

Two conditions must both hold: `nh % nkv == 0` (heads divisible before padding), and `nh_padded // nkv_padded == nh // nkv` (group size preserved after padding). Do **not** pad `nkv` independently of `nh`; this changes the group size and produces silently wrong attention weights. Verify both assertions hold; for the full derivation and code, see Chapter 3, `gqa_grouping_in_kernel.md`.

## 6. Page Table

Expected: `[b, max_num_blocks_per_seq]`, dtype `int32`, row-major, on device.

Checks:

```python
assert page_table.shape == (b, max_num_blocks_per_seq)
assert page_table.dtype == torch.int32          # host side before transfer
assert max_num_blocks_per_seq * block_size >= max_seq_len, \
    "page table cannot address the full sequence"
```

Each entry is a block index into the paged K/V tensor. Unwritten entries should be
initialized to a sentinel (e.g., 0 or a designated invalid index) and never read by
the attention op during a valid decode step.

## 7. `paged_update_cache` Input Shape

Expected: `[b, nkv, 1, dh]`

This is a common source of bugs. The write op takes **KV heads**, not query heads.
Passing `[b, nh, 1, dh]` will silently write the wrong number of head slices.

```python
assert kv_write.shape == (b, nkv, 1, dh), \
    f"expected [{b}, {nkv}, 1, {dh}], got {kv_write.shape}"
```

## 8. Checklist Summary

| # | Check | Pass condition |
|---|-------|---------------|
| 1 | Q rank and axis ordering | `Q.shape == (1, b, nh, dh)` |
| 2 | K/V axis 0 is batch | `K.shape[0] == b` |
| 3 | K/V contiguous shape | `K.shape == (b, nkv, s, dh)` |
| 4 | K/V paged shape | `K.shape == (max_num_blocks, nkv, block_size, dh)` |
| 5 | `nh % nkv == 0` | integer assertion passes |
| 6 | GQA ratio preserved after padding | `nh_padded // nkv_padded == nh // nkv` |
| 7 | Page table shape | `(b, max_num_blocks_per_seq)`, int32 |
| 8 | Page table capacity | `max_num_blocks_per_seq * block_size >= max_seq_len` |
| 9 | Cache write input | `(b, nkv, 1, dh)` — not `(b, nh, 1, dh)` |

---

**Next:** [`cur_pos_validation.md`](./cur_pos_validation.md)
