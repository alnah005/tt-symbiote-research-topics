# Tensor Shape Reference

This file is the authoritative shape table for all tensors consumed and
produced by `ttnn.transformer.scaled_dot_product_attention_decode`. Use it as
a checklist when debugging a correctness failure: verify each tensor against
the table before investigating the kernel itself.

---

## Master Shape Table

| Tensor | Non-paged shape | Paged shape | Layout | Dtype | Notes |
|--------|-----------------|-------------|--------|-------|-------|
| Q input | `[1 x b x nh x dh]` | same | Tile | BF16 | `nh` padded to `pnh=ceil(nh/32)*32` by upstream op |
| K input | `[b x nkv x s x dh]` | `[max_num_blocks x nkv x block_size x dh]` | Tile | BF16 or BFP8 | `nkv` is model nkv, not query-head count |
| V input | same as K | same as K | Tile | BF16 or BFP8 | must match K dtype exactly |
| Output | `[1 x b x pnh x dh]` | same | Tile | BF16 | `pnh=ceil(nh/32)*32`; slice to `nh` if `pnh != nh` |
| `page_table_tensor` | N/A | `[b x max_num_blocks_per_seq]` | Row-major | int32 | must be on device |
| `cur_pos_tensor` | `[b]` | `[b]` | Row-major | int32 | on device; alternative to `cur_pos` Python list |

### Paged-mode dimension definitions

| Symbol | Meaning |
|--------|---------|
| `max_num_blocks` | Total physical KV blocks in the pool = `b x max_num_blocks_per_seq` |
| `max_num_blocks_per_seq` | Maximum blocks any single sequence can occupy |
| `block_size` | Tokens per physical block; must be 32, 64, or 128 |
| `nkv` | Number of KV heads (not expanded to match `nh`) |

---

## Layout Requirements

TTNN tile layout packs data into 32×32 element tiles. All compute tensors (Q,
K, V, output) must be in tile layout before they are passed to the kernel.
Tensors that contain integer indices (page table, cur_pos) cannot be expressed
in tile layout because TTNN does not support int32 in tile layout — they must
be row-major.

| Tensor category | Required layout | Reason |
|-----------------|-----------------|--------|
| Q, K, V, output | Tile (32x32) | Hardware matrix engine requires tiled input |
| `page_table_tensor` | Row-major | int32 dtype; no tile-layout support for int32 |
| `cur_pos_tensor` | Row-major | int32 dtype; no tile-layout support for int32 |

Passing a tile-layout tensor where row-major is required (or vice versa) raises
a layout mismatch error at dispatch time. This error is one of the few
layout-related failures that is not silent.

---

## Dtype Constraints

> **[SILENT FAILURE]** `page_table_tensor` must be int32. If it is int64 or
> float32, the kernel reads the raw byte buffer as if it were packed int32
> values. A float32 value of `2.0` has bit pattern `0x40000000 = 1073741824`,
> so "block index 2" becomes "block index 1073741824" — far outside the
> allocated pool. No out-of-bounds error is raised; the kernel reads from an
> arbitrary memory location and produces silently wrong KV values.

> **[SILENT FAILURE]** K and V must have the same dtype. If K is BFP8 and V is
> BF16 (or any mismatched pair), the kernel does not validate this. The
> mismatched V bytes are interpreted as BFP8, producing scaled-down attention
> values that appear numerically plausible but are wrong.

Supported dtype combinations:

| K/V dtype | Notes |
|-----------|-------|
| BFloat16 | Standard; higher precision |
| BFP8_b | Block floating point 8-bit; saves memory; small accuracy trade-off |

Q and output are always BFloat16; no other dtype is accepted for those tensors.

---

## The `num_cores >= b * nkv` Constraint

The kernel parallelizes the attention computation by distributing work across
the compute grid. Each core is assigned exactly one `(batch_element, kv_head)`
pair. With `b=8` and `nkv=4`, there are `8 * 4 = 32` pairs, so at least 32
cores are required.

**When the constraint is violated:** If the grid has fewer than `b * nkv`
cores, some `(batch_element, kv_head)` pairs are never processed. The output
tensor positions corresponding to those pairs are left at their initialized
values (typically zeros or garbage from a previous operation in the same L1
buffer).

**Symptom:** Attention output for some batch elements or some KV heads is
wrong or zero, while others are correct. This looks like a pattern: the first
`num_cores` pairs are correct and the rest are wrong — but because the mapping
between cores and pairs is not documented in the public API, diagnosing this
requires counting cores and comparing to `b * nkv`.

**Fix:** Set `SDPAProgramConfig.compute_with_storage_grid_size` so that
`cols * rows >= b * nkv`. For the example above: `(8, 4)` gives 32 cores,
exactly matching the requirement.

---

## Cache Layout History

`[v0.55+]`

> **[SILENT FAILURE]** The non-paged KV cache layout changed between versions.
> The old layout was `[nkv x b x s x dh]` (KV heads in axis 0, batch in axis
> 1). The current layout is `[b x nkv x s x dh]` (batch in axis 0, KV heads
> in axis 1). If you pass a tensor created with the old layout to a kernel
> expecting the new layout, axis 0 is interpreted as the batch dimension.
> With a typical model where `nkv=4` and `b=8`, the kernel sees "8 KV heads
> and 4 batch elements" — no shape error is raised because the product of the
> first two axes is the same in both layouts. The resulting head indexing is
> entirely wrong and the attention output is silently incorrect.

To verify which layout your KV cache uses, inspect axis 0: if it equals `b`
you have the new layout; if it equals `nkv` you have the old layout.

---

## GQA Padding Gotcha

When `nkv` and `nh` are not multiples of 32, they must be padded to the
nearest 32 for tile alignment. The padding must preserve the GQA ratio.

> **[SILENT FAILURE]** Suppose `nkv=4` and `nh=16` (group_size = 4). If both
> are naively padded to 32 independently — `nkv_padded=32` and `pnh=32` — the
> kernel computes an effective group_size of `32 / 32 = 1`. This is MHA
> behavior, not GQA. Each query head attends to its own KV head instead of
> sharing KV heads across a group of 4. The kernel produces wrong attention
> weights with no error.

Correct approach: pad both `nh` and `nkv` so that the group_size ratio is
preserved. The procedure is:

1. Compute `pnh = ceil(nh / 32) * 32` — pad `nh` to the nearest tile multiple.
2. Compute `nkv_padded = pnh / group_size` — pad `nkv` to preserve the ratio.
3. Verify: `pnh / nkv_padded == group_size` ✓

For `nkv=4, nh=16, group_size=4`:
- `pnh = ceil(16 / 32) * 32 = 32`
- `nkv_padded = 32 / 4 = 8`
- Effective group_size = `32 / 8 = 4` ✓

The paged K/V tensors must be allocated with `nkv_padded` KV heads (8 in this
example), not the original `nkv=4`. Allocating K/V with `nkv=4` while using
`pnh=32` in Q leaves the group_size broken.
