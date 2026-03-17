# PCC Comparison Workflow

Pearson Correlation Coefficient (PCC) measures numerical alignment between the TT
output and a reference. It is more informative than absolute error for diagnosing
layout and scaling bugs.

## 1. Setting Up the Reference

The reference must match the TT op exactly:

- Same Q/K/V **values**, including any padding zeros added for alignment.
- Same **scale** (`1 / sqrt(dh)` unless overridden).
- Same **`is_causal`** flag.
- GQA expansion via `repeat_interleave` (not `expand`) to avoid aliased memory.

```python
import torch
import torch.nn.functional as F

def ref_sdpa_decode(Q_tt, K_tt, V_tt, cur_pos, scale, group_size):
    """
    Q_tt : [1, b, nh, dh]  — host float32 copy of TT Q tensor
    K_tt : [b, nkv, s, dh] — full cache (contiguous path)
    """
    b, nh, dh = Q_tt.shape[1], Q_tt.shape[2], Q_tt.shape[3]
    nkv = K_tt.shape[1]

    q = Q_tt.squeeze(0).permute(0, 2, 1, 3)           # [b, nh, 1, dh]
    k = K_tt.repeat_interleave(group_size, dim=1)      # [b, nh, s, dh]
    v = V_tt.repeat_interleave(group_size, dim=1)

    # mask future tokens using cur_pos
    outputs = []
    for i in range(b):
        qi = q[i:i+1]                                  # [1, nh, 1, dh]
        ki = k[i:i+1, :, :cur_pos[i], :]              # [1, nh, cur_pos[i], dh]
        vi = v[i:i+1, :, :cur_pos[i], :]
        oi = F.scaled_dot_product_attention(qi, ki, vi, scale=scale)
        outputs.append(oi)

    out = torch.cat(outputs, dim=0)                    # [b, nh, 1, dh]
    return out.permute(0, 2, 1, 3).unsqueeze(0)        # [1, b, nh, dh]
```

## 2. PCC Formula

```python
def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    cc = torch.corrcoef(torch.stack([a_flat, b_flat]))
    return cc[0, 1].item()   # off-diagonal of the 2×2 matrix
```

`torch.corrcoef` returns a 2×2 matrix; the cross-correlation is at `[0, 1]` (== `[1, 0]`).
Do not use `[0, 0]` or `[1, 1]` — those are autocorrelations (always 1.0).

## 3. PCC Thresholds

| PCC value | Interpretation |
|-----------|----------------|
| > 0.9999 | Floating-point rounding only; expected for bfloat16 vs float32 |
| > 0.99 | Good agreement; minor quantization effects |
| 0.98 – 0.99 | Borderline; investigate scale or masking differences |
| < 0.98 | Systematic error; likely wrong mask, wrong `cur_pos`, or wrong scale |
| < 0.9 | Severe mismatch; almost always a layout or shape bug |

When PCC < 0.98, do not tune thresholds — find the cause.

## 4. Narrowing with Binary Search Over cur_pos

Issue #30362 and similar boundary bugs manifest only at specific `cur_pos` values
(e.g., when a sequence crosses a page block boundary). Binary search over `cur_pos`
to find the first failing position:

```python
lo, hi = 1, max_seq_len
while lo < hi:
    mid = (lo + hi) // 2
    cur_pos_test = [mid] * b
    tt_out = run_decode(q, k, v, cur_pos=cur_pos_test)
    ref_out = ref_sdpa_decode(Q, K, V, cur_pos_test, scale, group_size)
    if pcc(tt_out, ref_out) < 0.99:
        hi = mid
    else:
        lo = mid + 1
print(f"First failing cur_pos: {lo}")
# Examine: lo % block_size == 0 ? → page boundary bug
```

If the first failure is exactly at a multiple of `block_size`, this is a strong signal
of a paged block boundary bug (consistent with #30362).

## 5. Isolating the KV Write Path

Before blaming the attention kernel, verify that `paged_update_cache` writes correct
values:

```python
# After paged_update_cache, read back the block containing cur_pos[0]
block_idx = page_table[0, (cur_pos[0] - 1) // block_size]
slot_in_block = (cur_pos[0] - 1) % block_size

written_k = paged_k_cache[block_idx, :, slot_in_block, :]   # [nkv, dh]
expected_k = K_new[0, :, 0, :]                               # [nkv, dh]

cache_pcc = pcc(written_k, expected_k)
print(f"KV write PCC: {cache_pcc:.6f}")
assert cache_pcc > 0.999, "Cache write is incorrect — bug is in paged_update_cache"
```

If the write PCC is high but attention output PCC is low, the bug is in the attention
read path or kernel, not the write path.

## 6. Workflow Summary

1. Build reference using the exact same tensors and parameters.
2. Compute PCC on the full output tensor.
3. If PCC < 0.99: binary-search `cur_pos` to find the first failing position.
4. Check if failure aligns with `block_size` multiples.
5. Isolate KV write path before blaming the attention kernel.
