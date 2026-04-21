# Memory Access Analysis

## Cos/Sin Table Characteristics

The cos/sin table is a single precomputed tensor stored in DRAM with shape `[max_seq_len, rotary_dim]`. For Qwen3.6-35B-A3B with `max_seq_len=32768` and `rotary_dim=64`:

```text
cos table:  32768 rows × 64 columns × 2 bytes (BF16) = 4 MiB
sin table:  32768 rows × 64 columns × 2 bytes (BF16) = 4 MiB
Total:      8 MiB
```

Key structural properties:

- Each row corresponds to a single sequence position and contains the rotation angles for all `rotary_dim` dimensions at that position.
- Standard RoPE and M-RoPE share the same table layout — M-RoPE performs three independent row-index lookups instead of one contiguous range read (see [Chapter 4](../ch4_ttnn_implementation/index.md)).
- The hardware DRAM prefetcher on Wormhole is effective for sequential row access patterns; random row access incurs page-miss overhead.
- At 8 MiB total size, the table may partially fit within L2/LLC on Wormhole, depending on chip-level cache configuration. If the table is resident in cache, all access patterns become equivalent and the M-RoPE vs. standard RoPE bandwidth difference disappears.

---

## Standard RoPE Access at Decode (seq_len=1)

At each decode step, standard partial RoPE reads a single row from the cos/sin table at the current position index `pos`:

```text
Access: cos_sin[pos]  →  contiguous read of rotary_dim * 2 bytes = 128 bytes
DRAM rows touched: 1 (cos row at pos) + 1 (sin row at pos) = 2 rows
Cache lines touched: 128 bytes / 64 bytes per cache line = 2 cache lines
Access pattern: maximally cache-friendly; single sequential stride
```

The DRAM prefetcher requires no warmup for a single-row access — this is the smallest possible read granularity and is bounded only by DRAM latency, not bandwidth.

---

## M-RoPE Access at Decode (seq_len=1)

M-RoPE performs three independent `ttnn.embedding` lookups, one per mrope section axis, using position indices `(t, h, w)` from the three rows of the `[3, batch, seq_len]` position_ids tensor.

### Text tokens at decode

For text tokens, all three axes carry identical sequential position values (established in [Chapter 3](../ch3_text_only_reduction/index.md)):

```text
t == h == w == pos
```

All three embedding lookups read the **same row** of the cos/sin table:

```text
Lookup 1 (temporal):  cos_sin[pos]  →  128 bytes (hits cache)
Lookup 2 (height):    cos_sin[pos]  →  128 bytes (cache hit, already loaded)
Lookup 3 (width):     cos_sin[pos]  →  128 bytes (cache hit, already loaded)
Total DRAM traffic: 128 bytes (single load; subsequent lookups served from cache)
```

The hardware either detects the repeated access and serves from cache, or the data is already in L1 from the first lookup. The bandwidth cost is identical to standard RoPE; the only overhead is 2 additional dispatch invocations.

### Vision tokens at decode

For vision tokens (image/video), the three axes carry distinct position values from the spatial grid:

```text
t = frame index (e.g., 0 for a single image)
h = grid row index for this patch
w = grid col index for this patch
```

The three lookups read **different rows**:

```text
Lookup 1 (temporal):  cos_sin[t]  →  128 bytes from row t
Lookup 2 (height):    cos_sin[h]  →  128 bytes from row h
Lookup 3 (width):     cos_sin[w]  →  128 bytes from row w

Total DRAM traffic: 3 × 128 bytes = 384 bytes
Standard RoPE baseline: 128 bytes
Absolute difference: 256 bytes
Time difference at 288 GB/s peak: 256 / (288 × 10^9) ≈ 0.9 ns
```

The absolute bandwidth difference is sub-nanosecond — negligible at any batch size.

---

## M-RoPE Access at Prefill (seq_len=S)

Prefill processes S tokens in a single forward pass. The access pattern depends on the token type.

### Text-only prefill

Position IDs for all S text tokens are identical across all three axes: `position_ids[0,b,i] == position_ids[1,b,i] == position_ids[2,b,i] == i` for each batch element `b` and position `i`.

```text
Standard RoPE: rows [0, S) read once, sequentially → S * 128 bytes
M-RoPE (text): rows [0, S) read three times, sequentially
               → same sequential access pattern × 3
               → hardware prefetcher handles each scan independently
               → 3 × S * 128 bytes transferred
               → extra bandwidth vs. standard RoPE: 2 × S * 128 bytes
```

At S=1024 on P150 (288 GB/s):

```text
Extra data: 2 × 1024 × 128 bytes = 256 KiB
Extra time at peak bandwidth: 256 KiB / 288 GB/s ≈ 0.9 µs
```

The hardware prefetcher is equally effective for all three sequential scans. The only material difference versus standard RoPE prefill is the 5 additional kernel dispatches.

### Image/video prefill

For S image patches (e.g., S=1024 from a 32×32 grid), the position IDs follow the grid rasterization layout (row-major patch ordering):

```text
Temporal axis:  all S patches from the same frame → same integer t repeated S times
                → S reads of the same row → best case: 1 DRAM read + (S-1) cache hits

Height axis:    patch row index, repeated num_patches_w times per row
                Pattern: [0,0,...,0, 1,1,...,1, ..., num_patches_h-1,...,num_patches_h-1]
                → num_patches_h distinct rows, each read num_patches_w times
                → stride access; hardware prefetcher partially effective
                → DRAM page access pattern: sequential across rows but not within stride

Width axis:     patch col index, cycles [0, 1, ..., num_patches_w-1] for each row
                → num_patches_w distinct rows, read num_patches_h times each
                → tight inner loop; prefetcher effective within each row stripe
```

Worst-case analysis (all S position IDs distinct, no spatial locality):

```text
Per-section distinct rows: up to S distinct row reads
3 sections × S random row reads vs. S sequential row reads for standard RoPE
Bytes transferred: same total (S × 128 bytes per section × 3 sections)
Cache miss rate: higher for random access → effective bandwidth reduced

Estimated effective bandwidth for random access: 50–70% of peak
= 144–200 GB/s (vs. 288 GB/s peak for sequential)

Extra latency from reduced bandwidth (height + width sections):
  Data: 2 × 1024 × 128 bytes = 256 KiB
  At 150 GB/s effective: 256 KiB / 150 GB/s ≈ 1.7 µs
```

### Cache residency effect

At `max_seq_len=32768` and `rotary_dim=64`, the 8 MiB cos/sin table is large enough that it may not fit entirely in L1 or L2 on a single Wormhole core. However:

- If a prefill pass repeatedly accesses the same table rows (e.g., all patches from a single image frame share `t=0`), those rows warm up in cache across sections.
- The 8 MiB total size is within range of LLC on Wormhole-class hardware. If the table is resident in LLC, all access patterns — sequential and random alike — incur similar latency.
- Random-access overhead is most significant when the table does **not** fit in any level of cache and the access pattern has poor spatial locality. For the image prefill case, the height section accesses rows strided by `rotary_dim = 64 columns`, which corresponds to 128 bytes per row — a single DRAM cache line per row access. The page-miss overhead dominates over the transfer time in this regime.

> **Key Finding:** The M-RoPE cos/sin table access overhead is sub-microsecond at decode time in all cases, and at most 1–5 µs at prefill time even for worst-case random-access image token patterns. The 8 MiB table size means cache residency effects are hardware-configuration-dependent; measure on actual TT hardware before optimizing.

## References

- [Chapter 4: TTNN Implementation Strategy](../ch4_ttnn_implementation/index.md) — establishes the single shared cos/sin table and `ttnn.embedding` as the gather mechanism
- [Chapter 3: Text-Only Reduction](../ch3_text_only_reduction/index.md) — establishes that text tokens have identical position IDs across all three axes
- [Chapter 2: Qwen3.6 M-RoPE Configuration](../ch2_qwen36_mrope_config/index.md) — defines `rotary_dim=64`, `mrope_section=[11,11,10]`
