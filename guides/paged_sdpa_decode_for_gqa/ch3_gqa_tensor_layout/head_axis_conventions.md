# Head Axis Conventions for Decode

## Overview

TTNN's `paged_sdpa_decode` uses a specific axis ordering for Q, K, and V tensors that differs from the training/prefill convention. Passing tensors in the wrong layout produces incorrect attention scores without raising an error.

---

## Tensor Shapes at Decode Time

### Q (Query) Tensor

**Shape: `[1 x b x nh x dh]`**

| Axis | Symbol | Meaning |
|------|--------|---------|
| 0 | `1` | Single decode step (fixed to 1, not a sequence length) |
| 1 | `b` | Batch size |
| 2 | `nh` | Query (Q) head count |
| 3 | `dh` | Head dimension |

### K and V (Key/Value) Tensors

**Shape: `[b x nkv x s x dh]`**

| Axis | Symbol | Meaning |
|------|--------|---------|
| 0 | `b` | Batch size |
| 1 | `nkv` | KV (key/value) head count |
| 2 | `s` | Max sequence length (contiguous cache) |
| 3 | `dh` | Head dimension |

---

## Why Q Has a Leading `1`

The leading `1` in the Q shape is not a placeholder — it encodes the decode contract.

Flash-Decode (the algorithm underlying `paged_sdpa_decode`) processes **one decode token per batch element per call**. During autoregressive generation, each forward pass produces exactly one new token. The query tensor therefore always has a sequence length of 1. Rather than omitting the sequence axis entirely, TTNN keeps it explicit as `1` so that the kernel can share index arithmetic with prefill code paths, where the sequence axis is greater than 1.

Concretely: the tensor shape `[1 x b x nh x dh]` means "for each of `b` batch elements, query `nh` heads each with dimension `dh` over a single decode step." It is not a GQA-specific convention — it applies equally to Multi-Head Attention (MHA) and GQA decode.

---

## `nlp_create_qkv_heads_decode` and 32-Padding

When TTNN pads nh and nkv independently, the effective group_size can silently change; see `gqa_grouping_in_kernel.md` for the full analysis.

---

## KV Cache Layout Change (Issue #12330)

The layout of the contiguous (non-paged) KV cache changed in tt-metal:

| Version | Shape |
|---------|-------|
| Old layout | `[nkv x b x S x dh]` |
| Current layout | `[b x nkv x S x dh]` |

The batch axis and the KV-head axis were **swapped**. This change was tracked in tt-metal issue #12330.

### Why This Matters

The shapes `[nkv x b x S x dh]` and `[b x nkv x S x dh]` can have identical total element counts whenever `nkv == b`. In that case, passing an old-layout KV cache to the current `paged_sdpa_decode` produces **wrong attention scores with no error raised**. The tensor passes shape validation because the product of dimensions is unchanged, but the kernel reads batch elements as KV heads and vice versa.

> **Warning:** If your KV cache was constructed or loaded using code written before the layout change in issue #12330, verify the axis order before passing it to `paged_sdpa_decode`. The shape `[nkv x b x S x dh]` and `[b x nkv x S x dh]` are numerically indistinguishable when `nkv == b`. There is no runtime error — the model will silently compute attention over the wrong tokens.

---

---

**Next:** [`gqa_grouping_in_kernel.md`](./gqa_grouping_in_kernel.md)
