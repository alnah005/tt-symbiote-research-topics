# GQA Concept: Multi-Head, Multi-Query, and Grouped Query Attention

This file defines MHA, MQA, and GQA precisely, derives the head-count relationship that the TTNN Flash-Decode kernel enforces, and explains the memory reduction that motivates GQA. It also traces the historical path from the `repeat_interleave` workaround to native GQA support in TTNN.

---

## 1. Multi-Head Attention (MHA)

In standard MHA, every query head has a dedicated, independent K and V head. For a model with `nh` heads:

- Q tensor: `nh` heads, each of dimension `dh`
- K tensor: `nh` heads, each of dimension `dh`
- V tensor: `nh` heads, each of dimension `dh`

During attention for head `i`, the computation is:

```
output[i] = softmax(Q[i] @ K[i].T / sqrt(dh)) @ V[i]
```

The KV cache stores `nh` K heads and `nh` V heads for every cached token. For a model with `nh=16` and `dh=128`, a single token occupies `2 * 16 * 128 * sizeof(dtype)` bytes in the KV cache — one factor of 2 for K and V separately.

MHA is the baseline: `nkv = nh`, so `nh / nkv = 1` (reduction factor = 1×, no reduction).

---

## 2. Multi-Query Attention (MQA)

MQA is the extreme opposite: all `nh` query heads share a **single** K head and a **single** V head.

- Q tensor: `nh` heads
- K tensor: 1 head
- V tensor: 1 head

The KV cache shrinks by a factor of `nh` compared to MHA. For `nh=16`, this is a 16x reduction. MQA was introduced to reduce the memory bandwidth cost of loading K/V tensors at each decode step, since bandwidth — not arithmetic — is the bottleneck for autoregressive generation.

In TTNN terminology, MQA is GQA with `nkv=1`.

---

## 3. Grouped Query Attention (GQA)

GQA sits between MHA and MQA. The `nh` query heads are divided into `nkv` groups of equal size. Every query head in a group shares the same K head and V head.

### Head-Count Invariant

```
nh = nkv * group_size
```

This must hold exactly — `nh` must be divisible by `nkv`. The Flash-Decode kernel computes the KV head index for query head `i` as:

```python
kv_head_idx = i // group_size   # where group_size = nh // nkv
```

If `nh % nkv != 0`, this integer division produces an incorrect mapping for some query heads, causing wrong attention scores with no error raised.

### Concrete Example: Ling (16 Q heads, 4 KV heads)

| Parameter | Value |
|-----------|-------|
| `nh` | 16 |
| `nkv` | 4 |
| `group_size` | 4 |

The 16 query heads are organized as 4 groups of 4:

```
Group 0:  Q heads  0,  1,  2,  3  →  KV head 0
Group 1:  Q heads  4,  5,  6,  7  →  KV head 1
Group 2:  Q heads  8,  9, 10, 11  →  KV head 2
Group 3:  Q heads 12, 13, 14, 15  →  KV head 3
```

The K and V tensors only need 4 heads, not 16. KV cache memory is reduced by `group_size = 4x` relative to MHA.

---

## 4. KV Cache Memory Reduction

For a sequence of length `s`, the KV cache footprint per layer is:

| Attention type | Cache size (bytes per layer per token) |
|----------------|----------------------------------------|
| MHA (`nkv = nh = 16`) | `2 * 16 * dh * sizeof(dtype)` |
| GQA (`nkv = 4, nh = 16`) | `2 * 4 * dh * sizeof(dtype)` |
| MQA (`nkv = 1, nh = 16`) | `2 * 1 * dh * sizeof(dtype)` |

With `dh=128` and BF16 (2 bytes), the GQA cache costs `2 * 4 * 128 * 2 = 2048` bytes per token per layer, compared to `8192` bytes for MHA. Across 32 layers and a 128K-token context, this difference is gigabytes.

> **Tip:** The memory savings from GQA are realized only if the KV cache is stored with `nkv` heads — not `nh` heads. Any code path that expands K/V from `nkv` to `nh` heads before writing to the cache defeats the reduction.

---

## 5. Broadcast vs. Native Group Implementations

There are two ways to implement GQA at the compute level:

### Broadcast (Expand-then-Attend)

The caller expands the K and V tensors from `nkv` heads to `nh` heads by repeating each KV head `group_size` times, then calls a standard MHA kernel:

```python
# Broadcast implementation (pre-native GQA)
k_expanded = k.repeat_interleave(group_size, dim=1)   # [b x nh x s x dh]
v_expanded = v.repeat_interleave(group_size, dim=1)   # [b x nh x s x dh]
output = mha_sdpa(q, k_expanded, v_expanded)
```

This works correctly but has two costs:
1. Memory: the expanded tensors are `group_size` times larger.
2. Bandwidth: reading `nh` KV heads instead of `nkv` heads at every decode step.

### Native Group (Kernel-Aware)

The kernel accepts K and V with `nkv` heads and internally computes `kv_head_idx = q_head_idx // group_size` to select the correct KV head per query. No expansion is performed; the kernel reads each KV head multiple times, once per query head in the group.

TTNN's Flash-Decode kernel (`ttnn.transformer.scaled_dot_product_attention_decode`) supports native GQA. The caller passes K/V with `nkv` heads and Q with `nh` heads.

> **Note:** Do not mix these two approaches carelessly. See Scenario B in `gqa_plus_paging_interaction.md` for the full analysis of why this wastes memory without affecting correctness.

---

## 6. Historical Context: `repeat_interleave` Workaround and Issue #12330

Before native GQA support was added to the Flash-Decode kernel, TTNN model code (including early ports of GQA models) used the broadcast approach:

```python
# Historical workaround — no longer needed for Flash-Decode
key_layer = ttnn.repeat_interleave(key_layer, group_size, dim=1)
value_layer = ttnn.repeat_interleave(value_layer, group_size, dim=1)
```

This workaround was correct but wasteful. It also required the paged KV cache to store `nh` heads rather than `nkv` heads, inflating the cache size by `group_size`.

`tenstorrent/tt-metal#12330` (Round 3 of GQA support work) added native GQA to the Flash-Decode kernel. After that change, the caller can pass K/V tensors with `nkv` heads directly, and the kernel handles the grouping internally.

> **Warning:** Some codebases written before issue #12330 still contain `repeat_interleave` on K/V. If such code is run against a newer TTNN version with native GQA, the behavior depends on whether the call is in paged or non-paged mode and how the tensor shapes are interpreted. Auditing for `repeat_interleave` or `expand` on K/V before the decode call is a required step in any correctness investigation.

The paged KV cache shape implications of this history are covered in `gqa_plus_paging_interaction.md`.

---

## Summary

| Concept | MHA | MQA | GQA |
|---------|-----|-----|-----|
| `nkv` | `= nh` | `= 1` | `1 < nkv < nh` |
| `group_size` | 1 | `nh` | `nh / nkv` |
| KV cache reduction vs. MHA | 1x | `nh`x | `group_size`x |
| TTNN Flash-Decode support | Native | Native (MQA = GQA with nkv=1) | Native (post #12330) |

---

**Next:** [`paged_kv_cache_concept.md`](./paged_kv_cache_concept.md)
