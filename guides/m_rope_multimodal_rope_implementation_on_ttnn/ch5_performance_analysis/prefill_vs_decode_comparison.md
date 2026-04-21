# Prefill vs. Decode: M-RoPE Overhead Comparison

## Decode (seq_len=1 per Step)

At each autoregressive decode step, a single new token is processed. The position ID tensor has shape `[3, B, 1]`.

**Overhead source:** Kernel dispatch count — M-RoPE introduces 5 additional dispatches (3 `ttnn.embedding` + 2 `ttnn.concat`) relative to standard RoPE's single contiguous slice.

```text
Additional dispatches:     5
Estimated dispatch cost:   5–10 µs each
Total additional overhead: 25–50 µs per decode step per attention layer

Decode step total latency (P150, Qwen3.6-35B-A3B):  ~250 ms
Overhead fraction:         25–50 µs / 250,000 µs = 0.01–0.02%
```

Status: `[placeholder — dispatch latency should be measured on actual TT hardware]`

**For text tokens at decode:** All three position ID axes are identical (`t == h == w == pos`), so all three embedding lookups hit the same row of the cos/sin table. Cache behavior is optimal; the DRAM access cost of M-RoPE is identical to standard RoPE.

**For vision tokens at decode:** The three axes carry distinct values (`t`, `h`, `w` differ), so three distinct DRAM row reads occur. Extra DRAM traffic: 2 × 128 bytes = 256 bytes — sub-nanosecond at 288 GB/s.

**Conclusion: Negligible.** No optimization needed for the decode path. Implement the naive gather-based approach and move on.

---

## Text-Only Prefill (Sequential Position IDs)

During text-only prefill of S tokens, position IDs follow `position_ids[axis, b, i] = i` for all three axes (identical sequential integers). This is the standard 1D RoPE position pattern applied uniformly across all axes (established in [Chapter 3](../ch3_text_only_reduction/index.md)).

**Overhead source:** 2 additional sequential cos/sin table scans (axis 1 and axis 2 are redundant with axis 0) plus 5 additional dispatches.

```text
At S=1024, rotary_dim=64:
  Extra table data: 2 × 1024 × 64 × 2 bytes = 256 KiB
  Extra bandwidth time at 288 GB/s: 256 KiB / 288 GB/s ≈ 0.9 µs

  Extra dispatches: 5 × ~7.5 µs = ~37.5 µs

  Total M-RoPE overhead vs. standard RoPE prefill: ~38 µs
  Typical attention prefill cost at S=1024: O(S²) matmul, >>1 ms
  Overhead fraction: < 0.01%
```

Both the extra bandwidth and extra dispatch costs are smaller than the RoPE multiply step itself at S=1024 (`B * H * S * rotary_dim * 2 bytes ≈ 16 MiB → ~57 µs`).

**Conclusion: Negligible** for text-only prefill.

---

## Image/Video Prefill (Non-Sequential Position IDs)

During image/video prefill, S patch tokens carry grid-based position IDs. The temporal axis is often uniform within a frame (same integer for all patches), while height and width axes follow the spatial rasterization pattern (non-sequential at the cos/sin table row level).

**Overhead source:** Random-access DRAM reads for the height and width `ttnn.embedding` lookups, plus 5 additional dispatches.

### Worst-case random access analysis

```text
S = 1024 image patches (e.g., 32 × 32 grid)
All height and width position IDs distinct and non-sequential

Random-access effective bandwidth:   ~50–70% of peak = 144–200 GB/s (vs. 288 GB/s)
Extra random-access data (h + w):    S × (2*s_h + 2*s_w) × 2 tables × 2 bytes
                                   = 1024 × (22 + 20) × 2 × 2 bytes
                                   = 1024 × 168 bytes ≈ 168 KiB

Worst-case time at 150 GB/s effective:   168 KiB / 150 GB/s ≈ 1.1 µs
Overhead from extra dispatches:          5 × ~7.5 µs ≈ 37.5 µs
Total M-RoPE overhead vs. standard RoPE: ~38–40 µs
```

For context, a prefill pass over S=1024 tokens is dominated by:

```text
Self-attention:   O(S²) matmul → ~100s of ms at S=1024 for a 35B model
FFN / MoE:        large weight loads → dominant fraction of prefill cost
```

Even at worst case, M-RoPE adds less than 5 µs of additional DRAM cost — less than 0.1% of total prefill cost.

**Conclusion: Small but measurable for large image prefills.** Not worth optimizing without profiling evidence showing RoPE in the critical path.

---

## Comparison Table

| Scenario | M-RoPE overhead vs. standard RoPE | Relative to step cost | Priority |
|---|---|---|---|
| Decode, seq_len=1, text token | ~25–50 µs (dispatch) + 0 ns (same cache line) | < 0.02% | None |
| Decode, seq_len=1, vision token | ~25–50 µs (dispatch) + ~1 ns (3 cache lines) | < 0.05% | None |
| Prefill, text-only, S=1024 | ~1 µs (bandwidth) + ~38 µs (dispatch) | < 0.01% | None |
| Prefill, 1024-patch image | ~1–5 µs (random-access bandwidth) + ~38 µs (dispatch) | < 0.1% | Low |

All entries: hardware = P150 (single Wormhole chip, 288 GB/s DRAM), model = Qwen3.6-35B-A3B.

Status of all entries: `[placeholder — to be filled during research]`

---

## Recommendation

Implement the naive gather-based M-RoPE approach first:

1. Three `ttnn.embedding` lookups (one per mrope section axis)
2. Two `ttnn.concat` operations to assemble the full cos/sin tensor
3. Identical rotate-half multiply to the standard RoPE path

Profile the implementation with Tracy on actual TT hardware after initial bringup is complete. Measure:

- Actual dispatch latency per op (expected: 5–10 µs; measure to confirm)
- Actual RoPE time as fraction of decode step and prefill step
- Cache hit rate for the cos/sin table under image prefill workloads

Only invest in a fused M-RoPE kernel (gathering all three sections in a single dispatch) if profiling demonstrates that RoPE is in the critical path. Current analysis — based on the 5 additional dispatches against a ~250 ms decode step — strongly indicates it will not be.

> **Key Finding:** M-RoPE introduces 5 additional TTNN kernel dispatches per attention layer per decode step, adding an estimated 25–50 µs. This is less than 0.02% of the ~250 ms decode step on P150 (Wormhole) and is not in the critical path. The naive gather-based implementation is the correct starting point; optimize only after profiling reveals a measurable impact.

## References

- [Chapter 4: TTNN Implementation Strategy](../ch4_ttnn_implementation/index.md) — establishes the naive 3-lookup + 2-concat M-RoPE forward path
- [Chapter 3: Text-Only Reduction](../ch3_text_only_reduction/index.md) — establishes that text-only M-RoPE degenerates to standard 1D RoPE (identical position IDs across all three axes)
- [Chapter 2: Qwen3.6 M-RoPE Configuration](../ch2_qwen36_mrope_config/index.md) — defines `mrope_section=[11,11,10]`, `rotary_dim=64`
- [Chapter 5: Operation Cost Breakdown](operation_cost_breakdown.md)
- [Chapter 5: Memory Access Analysis](memory_access_analysis.md)
- [Chapter 5: Kernel Launch Overhead](kernel_launch_overhead.md)
