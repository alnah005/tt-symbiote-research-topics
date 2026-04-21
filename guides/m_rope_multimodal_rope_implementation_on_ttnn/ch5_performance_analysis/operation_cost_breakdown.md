# Operation Cost Breakdown: Standard RoPE vs. M-RoPE

## Standard Partial RoPE on TTNN (Baseline)

At decode time with seq_len=1, batch=B, num_heads=H, rotary_dim=D (D=64 for Qwen3.6-35B-A3B):

**Op 1 — Contiguous table slice:**

```text
cos_sin[pos]  →  shape [1, D]
Data read: D * 2 * 2 bytes  (cos row + sin row, each BF16)
         = 64 * 4 = 256 bytes  (at D=64)
Access pattern: single contiguous row read
```

**Op 2 — Elementwise rotate-half multiply (applied separately for Q and K):**

```text
[B, H, 1, D] × [1, D]  →  [B, H, 1, D]
Operations: B * H * D  BF16 multiplies
Data moved: B * H * D * 2 bytes  read  +  B * H * D * 2 bytes  write
```

**Total kernel dispatches (standard RoPE): ~2** (slice + multiply, or fused into 1 if the implementation combines them).

---

## M-RoPE on TTNN at Decode Time (seq_len=1)

M-RoPE requires independent position lookups for each of the three mrope sections: temporal (s_t=11), height (s_h=11), width (s_w=10), where `s_t + s_h + s_w = 32 = rotary_dim/2`. Each section index covers `2 * s_i` real dimensions (cos + sin interleaved or stored as two halves).

**Ops 1–3 — Three `ttnn.embedding` lookups:**

```text
Temporal:  position_ids[0]  →  shape [B, 1] indices
           gather rows from cos/sin table  →  [B, 1, 2*s_t]
           data read: B * 2*s_t * 2 bytes = B * 44 * 2 bytes

Height:    position_ids[1]  →  shape [B, 1] indices
           gather rows from cos/sin table  →  [B, 1, 2*s_h]
           data read: B * 2*s_h * 2 bytes = B * 44 * 2 bytes

Width:     position_ids[2]  →  shape [B, 1] indices
           gather rows from cos/sin table  →  [B, 1, 2*s_w]
           data read: B * 2*s_w * 2 bytes = B * 40 * 2 bytes

Total data read across all 3 lookups:
  B * (2*s_t + 2*s_h + 2*s_w) * 2 bytes
  = B * 2 * rotary_dim * 2 bytes
  = B * 256 bytes  (same total as standard RoPE, split across 3 dispatches)
```

**Ops 4–5 — Two `ttnn.concat` operations:**

```text
concat([B, 1, 2*s_t], [B, 1, 2*s_h])  →  [B, 1, 2*s_t + 2*s_h]
concat([B, 1, 2*s_t + 2*s_h], [B, 1, 2*s_w])  →  [B, 1, rotary_dim]

Data written: rotary_dim * 2 bytes = 128 bytes  (intermediate + final)
```

**Op 6 — Elementwise rotate-half multiply (same as standard RoPE):**

```text
[B, H, 1, rotary_dim] × [B, 1, rotary_dim]  →  [B, H, 1, rotary_dim]
Applied separately for Q and K  (×2 dispatches)
Identical cost to standard RoPE multiply
```

**Total kernel dispatches (M-RoPE): ~7** (3 embedding + 2 concat + 2 multiply) versus ~2 for standard RoPE — **5 additional dispatches**.

---

## Summary Comparison Table

| Phase | Operation | Standard RoPE | M-RoPE |
|---|---|---|---|
| Decode | Table lookup | 1 contiguous slice | 3 `ttnn.embedding` lookups |
| Decode | Assembly | None | 2 `ttnn.concat` operations |
| Decode | Rotate-half (Q) | 1 elementwise multiply | 1 elementwise multiply (same) |
| Decode | Rotate-half (K) | 1 elementwise multiply | 1 elementwise multiply (same) |
| Decode | **Total dispatches** | **~2** | **~7** |
| Decode | Data volume (lookup) | 256 bytes | 256 bytes total (same) |
| Prefill (text) | Table lookup | 1 contiguous slice | 3 sequential embedding lookups |
| Prefill (text) | Access pattern | Sequential rows `[0, S)` | Sequential rows `[0, S)` × 3 axes |
| Prefill (image) | Table lookup | N/A | 3 random-access embedding lookups |
| Prefill (image) | Access pattern | N/A | Non-sequential DRAM row reads |

---

## Arithmetic Intensity Note

At decode batch=1, the rotate-half multiply operates on a tensor of `num_heads * rotary_dim` elements:

```math
arithmetic_intensity = MACs / bytes_moved
                     = (num_heads * rotary_dim) / (num_heads * rotary_dim * 2 bytes)
                     = 0.5 MACs/byte
```

This is well below the ridge point of Wormhole (approximately 300 GFLOPS / 288 GB/s ≈ 1 MAC/byte), meaning the rotate-half step is **bandwidth-bound** even at batch=1. M-RoPE does not change this arithmetic intensity — the total data volume accessed across all three embedding lookups equals the single contiguous read in standard RoPE. The 5 additional dispatches are the only non-trivial M-RoPE overhead, not extra compute or extra bytes transferred.

At larger batch sizes (B > 1), the multiply tensor scales as `B * H * D` while the lookup table read remains fixed at `rotary_dim * 2 bytes`. Dispatch overhead is amortized over the larger multiply cost, making M-RoPE's relative overhead smaller as batch increases.

## References

- [Chapter 4: TTNN Implementation Strategy](../ch4_ttnn_implementation/index.md) — establishes `ttnn.embedding` as the recommended gather op and the 3-lookup + 2-concat forward path
- [Chapter 2: Qwen3.6 M-RoPE Configuration](../ch2_qwen36_mrope_config/index.md) — defines `mrope_section = [11, 11, 10]`, `rotary_dim = 64`
