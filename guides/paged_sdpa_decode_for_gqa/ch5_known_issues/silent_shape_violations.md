# Silent Failure Modes — Shape and Layout Violations

All four failure modes described here produce wrong numerical output with no
exception, no assertion, and no TTNN error message.  They are detectable only
by comparing output against a reference.

---

## 1. Padding-Induced GQA Collapse

### Mechanism

TTNN requires tensor dimensions to be multiples of 32 (tile alignment).  When
`nh` or `nkv` is not already a multiple of 32, padding is applied.  The
invariant that must be preserved is:

```
nkv_padded / nh_padded == nkv / nh    (original group_size must be unchanged)
```

Correct padding formula: `nkv_padded = nh_padded / original_group_size`

If padding is applied independently to `nh` and `nkv` — for example by rounding
each to the next multiple of 32 separately — the ratio changes.  The kernel
reads `group_size` from the padded counts:

```
effective_group_size = nh_padded / nkv_padded
```

If this differs from `nh / nkv`, every `kv_head_idx = q_head_idx // group_size`
lookup returns the wrong KV head.  No error is raised.

### Example

```
nh = 20, nkv = 5   → group_size = 4

Incorrect padding:
  nh_padded  = 32   (next multiple of 32)
  nkv_padded = 32   (next multiple of 32)
  effective_group_size = 32 / 32 = 1  → kernel behaves as MQA

Correct padding:
  nh_padded  = 32
  nkv_padded = 32 / 4 = 8
  effective_group_size = 32 / 8 = 4  → correct GQA
```

---

## 2. Cache Layout Mismatch (Old vs. Current Dense K/V Layout)

### Mechanism

This issue applies to the **dense** (non-paged) K/V cache path.  Prior to the
native GQA work tracked in #12330, the dense KV cache convention placed the
KV-head dimension first:

```
[nkv x b x S x dh]   ← old dense layout (wrong after #12330)
```

After #12330, the current expected layout for the dense KV cache is:

```
[b x nkv x S x dh]   ← current dense layout (correct)
```

If a dense KV tensor built with the old layout is passed to the current kernel,
the kernel's indexing arithmetic reads head `h` from the wrong physical offset.
The symptom is that all query heads appear to attend to the same K/V content —
specifically, the data that lives at offset 0 in what the kernel treats as the
KV-head dimension.

This mismatch can arise when KV cache buffers are allocated and populated in
one subsystem (e.g., a model loader that predates the layout change) and
consumed in another (e.g., a current TTNN inference loop).

**Paged KV cache note**: The paged KV cache uses a distinct shape that has no
old/new distinction in this context.  The paged layout is always:

```
[max_num_blocks x nkv x block_size x dh]   ← paged cache shape
```

The dense-layout change described above does not affect paged tensors.

### Detection

Compare per-head output against a reference computed with the correct layout.
All heads beyond head 0 will differ from reference if the layout is swapped.

---

## 3. `paged_update_cache` Head Count Mismatch

### Mechanism

`paged_update_cache` writes one new token into the paged KV cache.  Its input
tensor has shape `[b x nkv x 1 x dh]`.  A common mistake in GQA models is to
pass a tensor of shape `[b x nh x 1 x dh]` — the full query head count — either
because the variable name is wrong or because the tensor was taken from a Q
computation path instead of a K/V computation path.

When `nh > nkv`, writing `nh` heads into a cache dimensioned for `nkv` heads
corrupts the block contents for all heads beyond index `nkv - 1`.  On
subsequent decode steps, those corrupted slots produce wrong attention values.
No error is raised at the write site.

### Invariant to check

```python
assert kv_for_cache.shape == (b, nkv, 1, dh), (
    f"Expected nkv={nkv} heads, got {kv_for_cache.shape[1]}"
)
```

---

## 4. `page_table_tensor` Dtype and Layout Errors

### Required specification

```
dtype:   ttnn.int32
layout:  ttnn.ROW_MAJOR_LAYOUT
device:  on-device tensor (not host)
shape:   [b x max_num_blocks_per_seq]
```

### Failure modes

| Error | Symptom |
|---|---|
| Wrong dtype (e.g., `uint16`, `int64`) | Block index read is bit-truncated or zero-extended; kernel reads from wrong physical block; output is numerically wrong |
| Tiled layout instead of row-major | Index values are read from transposed positions within the tile; wrong blocks fetched |
| Host tensor (not on device) | Undefined behavior or silent zero-block reads depending on TTNN version |

None of these conditions raise an exception.  The kernel proceeds with whatever
bits it reads from the page table memory, treating them as valid block indices.
If the resulting indices are in-bounds for the allocated block buffer, no memory
fault occurs; the wrong KV data is silently used.

### Defensive construction

```python
page_table = ttnn.from_torch(
    page_table_host,                        # torch.int32
    dtype=ttnn.int32,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
)
assert page_table.dtype == ttnn.int32
assert page_table.layout == ttnn.ROW_MAJOR_LAYOUT
```

---

**Next:** [`gqa_workaround_history.md`](./gqa_workaround_history.md)
