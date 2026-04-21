# Kernel Launch Overhead

## Baseline Dispatch Count: Standard Partial RoPE

Standard partial RoPE on TTNN at decode time (seq_len=1) requires the following dispatch sequence:

| Dispatch | Operation | Description |
|---|---|---|
| 1 | Table slice | `cos_sin[pos]` — contiguous row read at current position |
| 2 | Elementwise multiply | Rotate-half applied to Q and K jointly, or two separate dispatches |

**Total dispatches: ~2** (or 1 if the slice and multiply are fused into a single op).

This is the minimum achievable dispatch count for any RoPE implementation that precomputes the cos/sin table and applies it at runtime.

---

## M-RoPE Dispatch Count

M-RoPE replaces the single contiguous slice with three independent embedding lookups, followed by two concatenations to reassemble the full cos/sin tensor before the rotate-half multiply:

| Dispatch | Operation | Shape out | Notes |
|---|---|---|---|
| 1 | `ttnn.embedding` (temporal) | `[B, 1, 2*s_t]` | `s_t=11`; reads 22 columns from cos/sin table |
| 2 | `ttnn.embedding` (height) | `[B, 1, 2*s_h]` | `s_h=11`; reads 22 columns |
| 3 | `ttnn.embedding` (width) | `[B, 1, 2*s_w]` | `s_w=10`; reads 20 columns |
| 4 | `ttnn.concat` (first) | `[B, 1, 2*s_t + 2*s_h]` | Combines temporal + height sections |
| 5 | `ttnn.concat` (second) | `[B, 1, rotary_dim]` | Adds width section; final cos/sin tensor |
| 6 | Elementwise multiply (Q) | `[B, H, 1, rotary_dim]` | Rotate-half for query |
| 7 | Elementwise multiply (K) | `[B, H, 1, rotary_dim]` | Rotate-half for key |

**Total dispatches: ~7** versus ~2 for standard RoPE.

**Additional dispatches introduced by M-RoPE: 5** (3 embedding lookups + 2 concatenations).

---

## Dispatch Latency Estimate

TTNN host-side op dispatch includes Python overhead, argument validation, tensor metadata operations, and command queue submission. The estimated range per dispatch is 5–10 µs on current TT hardware.

```text
Additional dispatches:        5
Estimated dispatch cost:      5–10 µs per dispatch
Total additional dispatch:    25–50 µs per attention layer per decode step

Decode step latency (P150, Qwen3.6-35B-A3B):  ~250 ms (estimated; bandwidth-bound)
Overhead fraction:            25–50 µs / 250,000 µs = 0.01–0.02%
```

Status: `[placeholder — dispatch latency should be measured on actual TT hardware]`

The 250 ms decode step estimate is based on DRAM bandwidth requirements for loading MoE expert weights and attention projection weights during a single autoregressive step on a single P150 chip. RoPE dispatch overhead is four to five orders of magnitude smaller than this step latency.

---

## Comparison to Total RoPE Cost

RoPE has two cost components at runtime: the table lookup and the rotate-half multiply. Their relative weight shifts with batch size.

### At batch=1 (single-stream decode)

```text
Lookup data volume:  128 bytes (standard) or 384 bytes (M-RoPE, vision token)
Time at 288 GB/s:    0.4 ns – 1.3 ns
Multiply data:       num_heads * rotary_dim * 2 bytes = 128 * 64 * 2 = 16 KiB
Time at 288 GB/s:    ~56 ns

Dispatch overhead (M-RoPE, 7 dispatches):  35–70 µs
```

At batch=1, **dispatch overhead dominates both the lookup and multiply costs by three orders of magnitude**. The actual compute and DRAM transfer time for RoPE at decode is negligible; the only non-trivial RoPE cost is host-side dispatch latency.

### At batch=32

```text
Multiply data scales:  32 × 16 KiB = 512 KiB per Q/K → ~1.8 µs at 288 GB/s
Dispatch overhead:     same 35–70 µs (does not scale with batch)
Overhead fraction of RoPE time:  35 µs / (35 µs + 3.6 µs) ≈ 91% still dispatch
```

Even at larger batch sizes, dispatch overhead remains the dominant RoPE cost at decode time. M-RoPE's 5 additional dispatches are proportionally the same overhead regardless of batch.

### At prefill (seq_len=S=1024)

```text
Multiply data:         B * H * S * rotary_dim * 2 bytes = 1 * 128 * 1024 * 64 * 2 = 16 MiB
Time at 288 GB/s:      ~57 µs
Dispatch overhead:     35–70 µs (fixed; does not scale with S)
```

At S=1024, the multiply cost and dispatch overhead become comparable. The 5 additional dispatches add 25–50 µs to a 57 µs multiply step — approximately 44–88% relative overhead on the RoPE portion. However, RoPE is a small fraction of total prefill cost (attention is O(S²), FFN dominates token throughput).

---

## Fusion Opportunity

### Theoretical case for fusion

The 3 gather + 2 concat operations could in principle be fused into a single custom TTNN kernel that reads the three relevant cos/sin row segments and writes the assembled output tensor in one pass, eliminating 4 of the 5 additional dispatches.

### Obstacle: mrope_section sizes vs. TTNN tile size

TTNN's compute engine processes tensors in tiles of 32×32 elements. Efficient custom kernels require tensor dimensions to align to tile boundaries (multiples of 32 in the relevant dimension).

For Qwen3.6-35B-A3B, the section sizes in real (non-complex) dimensions are:

```text
s_t (temporal):  11 pairs → 2 * 11 = 22 real dims  (not a multiple of 32)
s_h (height):    11 pairs → 2 * 11 = 22 real dims  (not a multiple of 32)
s_w (width):     10 pairs → 2 * 10 = 20 real dims  (not a multiple of 32)
rotary_dim:      64 real dims                       (multiple of 32 — the full tensor is tile-aligned)
```

The individual sections are not tile-aligned. A fused gather-concat kernel operating on these sections would either pad each section to 32 (wasting 30–40% of tile bandwidth) or use a non-tiled scatter-gather path with lower efficiency.

### Recommendation

Implement the naive 3-lookup + 2-concat approach first. Profile with Tracy on actual TT hardware to determine whether the 25–50 µs dispatch overhead is measurable in production decode latency. Only invest in a fused kernel if profiling demonstrates RoPE is in the critical path — current analysis strongly suggests it will not be.

> **Document as:** Fusion opportunity exists but is not immediately tractable given mrope_section sizes `[11, 11, 10]` (none are multiples of 32 required for efficient TTNN tile alignment). Implement naive approach first; revisit after profiling.

Status of fusion benchmark: `[placeholder — to be filled during research]`

---

## T3K (Multi-Chip) Consideration

On T3K (4 Wormhole chips with tensor parallelism), num_heads per device is reduced by the tensor-parallel degree. At TP=4 with 128 heads:

```text
Heads per chip: 128 / 4 = 32
Multiply data per chip: 32 * 64 * 2 bytes = 4 KiB per Q/K
```

The multiply cost is 4× smaller per chip, but dispatch overhead is constant per chip. The M-RoPE dispatch overhead fraction is slightly higher on T3K than on a single P150 because the per-chip multiply work is smaller. This remains negligible in absolute terms (25–50 µs against a per-chip decode step latency that is still dominated by MoE expert routing and weight loading).

## References

- [Chapter 4: TTNN Implementation Strategy](../ch4_ttnn_implementation/index.md) — establishes the 3-lookup + 2-concat M-RoPE forward path and `ttnn.embedding` as the recommended gather op
- [Chapter 2: Qwen3.6 M-RoPE Configuration](../ch2_qwen36_mrope_config/index.md) — defines `mrope_section=[11,11,10]`, `rotary_dim=64`, `num_heads=128`
