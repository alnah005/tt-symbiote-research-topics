# Chapter 5: Performance Cost Analysis — M-RoPE vs. Standard RoPE on TTNN

## Prerequisites

- [Chapter 1: RoPE Foundations](../ch1_rope_foundations/index.md)
- [Chapter 2: Qwen3.6 M-RoPE Configuration](../ch2_qwen36_mrope_config/index.md)
- [Chapter 3: Text-Only Reduction](../ch3_text_only_reduction/index.md)
- [Chapter 4: TTNN Implementation Strategy](../ch4_ttnn_implementation/index.md)

## Overview

This chapter quantifies the overhead of M-RoPE's per-section gather approach versus standard RoPE's contiguous slice on TTNN. The analysis covers four cost dimensions:

1. **Operation count** — how many kernel dispatches M-RoPE adds relative to standard RoPE
2. **Memory access patterns** — contiguous vs. random-access reads into the cos/sin table
3. **Kernel launch overhead** — host-side dispatch latency at estimated 5–10 µs per op
4. **Prefill vs. decode** — how overhead scales with sequence length and position ID structure

> **Key Finding:** The dominant M-RoPE overhead is 5 additional TTNN kernel dispatches per decode step (3 embedding lookups + 2 concatenations, versus 1 contiguous slice for standard RoPE). At decode time (seq_len=1), this adds an estimated 25–50 µs of host-dispatch overhead — less than 0.02% of the ~250 ms decode step on P150. The M-RoPE overhead is negligible at decode time and small at prefill time for text-only inputs (sequential access pattern). Prefill with image/video tokens introduces non-sequential DRAM access that may reduce effective bandwidth by up to 2–3× for the gather step, but RoPE remains a small fraction of total prefill cost regardless.

All performance numbers in this chapter are **estimates** based on known hardware specifications (P150 Wormhole, 288 GB/s DRAM bandwidth) and TTNN dispatch characteristics. Entries marked `[placeholder]` require measurement on actual TT hardware.

## Files in This Chapter

| File | Content |
|---|---|
| `operation_cost_breakdown.md` | Op-by-op comparison of standard RoPE vs. M-RoPE; dispatch counts; arithmetic intensity |
| `memory_access_analysis.md` | Cos/sin table access patterns; sequential vs. random reads; cache behavior |
| `kernel_launch_overhead.md` | Host dispatch latency; fusion opportunities; batch scaling |
| `prefill_vs_decode_comparison.md` | Per-scenario overhead estimates; recommendation to implement naive approach first |

## References

- [Chapter 1: RoPE Foundations](../ch1_rope_foundations/index.md)
- [Chapter 2: Qwen3.6 M-RoPE Configuration](../ch2_qwen36_mrope_config/index.md)
- [Chapter 3: Text-Only Reduction](../ch3_text_only_reduction/index.md)
- [Chapter 4: TTNN Implementation Strategy](../ch4_ttnn_implementation/index.md)
