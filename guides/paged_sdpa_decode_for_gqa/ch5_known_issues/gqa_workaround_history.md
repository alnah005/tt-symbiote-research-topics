# GQA Workaround History — `repeat_interleave` and Native GQA Support

---

## Pre-Issue-#12330 Era: The `repeat_interleave` Workaround

Before native GQA support was added to the Flash-Decode kernel, the SDPA decode
operation required `nh` K heads and `nh` V heads — it had no mechanism to share
a single KV head across a group of query heads internally.

The standard workaround was to expand K and V to full query head count before
the SDPA call:

```python
# Pre-#12330 workaround
# k_cache: [b x nkv x s x dh]
# v_cache: [b x nkv x s x dh]
group_size = nh // nkv
k_expanded = k_cache.repeat_interleave(group_size, dim=1)  # [b x nh x s x dh]
v_expanded = v_cache.repeat_interleave(group_size, dim=1)  # [b x nh x s x dh]
output = sdpa_decode(q, k_expanded, v_expanded, ...)
```

This workaround was functionally correct.  Its cost was proportional to
`group_size`: for a model with `nh = 32` and `nkv = 8` (`group_size = 4`), KV
memory usage per decode step was 4x the compressed size.  For paged KV caches,
the expanded tensor had to be materialized in full before each call, eliminating
the memory benefit of paging.

---

## Issue #12330 Round 3: Native GQA Added to Flash-Decode Kernel

Issue #12330 (tracked across multiple rounds) added native GQA support to the
TTNN Flash-Decode kernel.  After this change:

- The kernel accepts `k_cache: [b x nkv x s x dh]` and `v_cache: [b x nkv x s x dh]` directly.
- Internally it computes `kv_head_idx = q_head_idx // group_size` and reads the
  correct KV head without any expansion in device memory.
- `group_size` is inferred from `nh` and `nkv` at kernel launch time.

The `repeat_interleave` workaround became unnecessary after this change landed.

---

## Risk: Mixing Old Workaround with Native GQA Kernel

If code written before #12330 is used with a post-#12330 TTNN build without
removing the `repeat_interleave` call, the following happens:

1. `repeat_interleave` expands K/V from `nkv` heads to `nh` heads.
2. The kernel receives `nh` K heads and `nh` V heads.
3. The kernel infers `effective_nkv = nh` and `effective_group_size = nh / nh = 1`.
4. Every query head gets its own dedicated KV head — MQA behavior, not GQA.

This is a correctness bug.  The model silently runs as MQA instead of GQA.
Because attention is still computed (just with the wrong sharing structure), the
output is numerically plausible but wrong.  PCC against a reference will be
below 1.0 but may remain above a loose threshold if the reference was also
computed with the expanded layout.

No error is raised.  The bug is undetectable without a reference computed from
the unexpanded `nkv`-head K/V.

---

## Audit Checklist

Search all code paths that lead to `sdpa_decode` for any of the following
patterns on the K or V tensor:

```python
# Patterns to eliminate after #12330
k.repeat_interleave(...)
v.repeat_interleave(...)
k.expand(...)            # if used to broadcast nkv → nh
v.expand(...)
torch.repeat_interleave(k, ...)
```

For each occurrence, confirm whether it predates #12330 and whether the current
TTNN build includes native GQA support.  If both are true, remove the expansion
and pass the `[b x nkv x s x dh]` tensor directly.

---

## Summary Table

| Era | K/V shape passed to SDPA | Kernel behavior | Correct? |
|---|---|---|---|
| Pre-#12330, with workaround | `[b x nh x s x dh]` | MHA path; no GQA sharing | Yes (functionally correct, expensive) |
| Post-#12330, no workaround | `[b x nkv x s x dh]` | Native GQA; `group_size` applied | Yes |
| Post-#12330, workaround left in | `[b x nh x s x dh]` | Native GQA with `group_size = 1` | No — silently MQA |
