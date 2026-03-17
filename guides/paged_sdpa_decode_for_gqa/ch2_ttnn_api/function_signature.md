# Function Signature: scaled_dot_product_attention_decode

`[v0.55+]`

---

## Complete Python Signature

```python
ttnn.transformer.scaled_dot_product_attention_decode(
    input_tensor_q,        # [1 x b x nh x dh], BFloat16, tile layout
    input_tensor_k,        # non-paged: [b x nkv x s x dh]
                           # paged:     [max_num_blocks x nkv x block_size x dh]
    input_tensor_v,        # same shape as K
    *,
    is_causal=True,
    attn_mask=None,        # [b x 1 x s x s], optional
    cur_pos=None,          # List[int] of length b
    cur_pos_tensor=None,   # ttnn.Tensor shape [b], row-major int32 on device
    scale=None,            # float, defaults to 1/sqrt(dh)
    program_config=None,   # SDPAProgramConfig
    compute_kernel_config=None,
    sliding_window_size=None,
    page_table_tensor=None,  # [b x max_num_blocks_per_seq], row-major int32 on device
    memory_config=None,
) -> ttnn.Tensor  # [1 x b x pnh x dh]
```

All parameters after `input_tensor_v` are keyword-only (enforced by the bare
`*`).

---

## Parameter Reference

### `input_tensor_q`

**Type:** `ttnn.Tensor` — BFloat16, tile layout, shape `[1 x b x nh x dh]`.
Required.

The leading dimension of 1 encodes the single decode step: at decode time only
one new token is attended from per batch element. `b` is batch size, `nh` is
the number of query heads, and `dh` is the per-head dimension. In practice `nh`
is almost always padded to `pnh = ceil(nh / 32) * 32` by the upstream call to
`nlp_create_qkv_heads_decode` — the kernel receives and operates on `pnh`
heads, not `nh`. The caller is responsible for tracking the real `nh` and
slicing the output accordingly.

---

### `input_tensor_k` and `input_tensor_v`

**Type:** `ttnn.Tensor` — BFloat16 or BFP8, tile layout. Required. Both must
have the same shape and dtype.

Two modes are supported and selected implicitly by whether `page_table_tensor`
is provided:

- **Non-paged:** shape `[b x nkv x s x dh]`. The full KV cache for each batch
  element sits in a single contiguous allocation. `s` is the maximum sequence
  length.
- **Paged:** shape `[max_num_blocks x nkv x block_size x dh]`. The KV cache is
  split into equal-sized physical blocks. `max_num_blocks = b x
  max_num_blocks_per_seq` is the total block pool size; `nkv` is the number of
  KV heads; `block_size` is the number of tokens per block (32, 64, or 128).

You cannot mix paged K with non-paged V or vice versa — both tensors must be
in the same mode. Mixing them raises no error but produces incorrect attention
values.

> **[SILENT FAILURE]** The non-paged KV cache layout changed at v0.55. See `tensor_shape_reference.md` for the full axis-order constraint and concrete diagnostic example.

---

### `is_causal`

**Type:** `bool`. Default: `True`.

When `True` the kernel applies a causal mask so each query position can only
attend to KV positions up to and including its own sequence position. For
decode (single-step inference) this is almost always `True`. Setting it to
`False` without supplying an explicit `attn_mask` is rarely correct.

---

### `attn_mask`

**Type:** `ttnn.Tensor` or `None`. Default: `None`. Shape when provided:
`[b x 1 x s x s]`, BFloat16.

An explicit additive attention mask added to the pre-softmax logits. When
`is_causal=True` and `attn_mask` is also provided, both are applied (the
causal mask and the additive mask are summed). Most decode paths leave this
`None` and rely solely on `is_causal` plus `cur_pos`.

---

### `cur_pos`

**Type:** `List[int]` of length `b` or `None`. Default: `None`.

A Python list where `cur_pos[i]` is the number of valid KV tokens currently
stored for batch element `i`. This is a count (length), not a write index. For
example, if batch element 0 has filled 64 tokens, pass `cur_pos[0] = 64`. The
kernel uses this to mask out unwritten cache positions beyond the valid range.

> **[SILENT FAILURE]** Passing a plain Python `int` instead of a length-`b`
> list is accepted without error. The scalar is interpreted as the position for
> only the first batch element; remaining elements get undefined masking
> behavior, leading to incorrect attention scores for all but the first batch
> element.

When `cur_pos` is used, the kernel is retraced and recompiled for each unique
combination of values. In production with variable sequence lengths this
accumulates many cached programs. Prefer `cur_pos_tensor` for production
serving.

---

### `cur_pos_tensor`

**Type:** `ttnn.Tensor` or `None`. Default: `None`. Shape: `[b]`, row-major,
int32, on device.

A device tensor carrying the same per-element valid-length information as
`cur_pos`, but read at runtime rather than at compile time. Because the values
are not embedded in the program, the kernel does not recompile when sequence
lengths change. Use this for production inference. Exactly one of `cur_pos` or
`cur_pos_tensor` must be provided; providing both or neither raises an error.

---

### `scale`

**Type:** `float` or `None`. Default: `None`.

The scalar multiplier applied to Q before the dot product with K. When `None`,
the kernel computes `1 / sqrt(dh)` internally. Pass an explicit value only if
you have already pre-scaled Q (e.g., for quantization reasons) and do not want
double-scaling. Providing the wrong scale shifts the softmax temperature and
degrades output quality silently.

---

### `program_config`

**Type:** `SDPAProgramConfig` or `None`. Default: `None`.

Controls the compute grid size and chunking strategy. When `None` the kernel
selects a default grid, which is often suboptimal for small-batch GQA
configurations (e.g., b=1, nkv=4). See `sdpa_program_config.md` for field
details and a worked example.

---

### `compute_kernel_config`

**Type:** TTNN compute kernel config object or `None`. Default: `None`.

Controls low-level math settings (e.g., fp32 accumulation, math fidelity).
Passing `None` uses the device default. For most correctness debugging work
this can be left as `None`; it is relevant when investigating precision
regressions between math-fidelity modes.

---

### `sliding_window_size`

**Type:** `int` or `None`. Default: `None`.

When set, restricts each query to attend only to the most recent
`sliding_window_size` KV positions. Used for long-context models that limit
attention span. Leave `None` for standard full-context decode.

---

### `page_table_tensor`

**Type:** `ttnn.Tensor` or `None`. Default: `None`. Shape:
`[b x max_num_blocks_per_seq]`, row-major, int32, on device.

Activates paged mode when provided. Each row `i` contains the physical block
indices assigned to batch element `i` in sequence order. The kernel uses this
table to translate a logical token position into a physical block address
within the K/V cache pool.

> **[SILENT FAILURE]** See `tensor_shape_reference.md` for the full page_table_tensor dtype/layout constraint and the silent-failure consequences.

---

### `memory_config`

**Type:** TTNN memory config or `None`. Default: `None`.

Specifies where the output tensor is allocated (e.g., L1 or DRAM, interleaved
or sharded). When `None` the output follows the default memory placement
policy. For performance-sensitive paths you may want to pin the output to L1
interleaved to avoid DRAM round-trips before the next operation.

---

## Output Tensor

**Shape:** `[1 x b x pnh x dh]`, BFloat16, tile layout.

`pnh = ceil(nh / 32) * 32`. When `nh` is already a multiple of 32 (e.g., 32
or 64), `pnh == nh` and no slicing is needed. When `nh` is not a multiple of
32 (e.g., 16), `pnh > nh` and the extra head positions contain undefined
values. The caller must slice the output before downstream use:

```python
# nh=16, pnh=32
output = ttnn.transformer.scaled_dot_product_attention_decode(q, k, v, ...)
# output shape: [1, b, 32, dh]
output = output[:, :, :nh, :]  # drop padding heads -> [1, b, 16, dh]
```

Forgetting this slice passes padding-head garbage into the subsequent linear
projection, which silently corrupts all subsequent logits.
