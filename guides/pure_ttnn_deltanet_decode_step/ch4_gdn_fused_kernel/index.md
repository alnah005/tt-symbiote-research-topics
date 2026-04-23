# Chapter 4 — The `gdn_full_fused_inplace` Kernel: Reuse vs. Adapt

The `gdn_full_fused_inplace` kernel targets Blackhole and likely requires tuning before running correctly on Wormhole T3K. The key concern is CB (circular buffer) size constants that were calibrated for Blackhole's 2 MB L1 — they must be rechecked against Wormhole's 1.5 MB. The algorithmic structure (6 fused ops, state in L1, DMA streams) is portable and the kernel is classified as `[REUSABLE — port and tune]`. This chapter documents what to check and what to change.

> **Key Finding:** The composed TTNN form from Chapter 2 is the immediate, correct fix for Metal Trace compatibility — no new kernel is required to unblock tracing. The `gdn_full_fused_inplace` kernel is a latency optimization over the composed form (one dispatch per layer instead of 12), not a prerequisite for trace. Chapters 4 and 5 in the implementation roadmap (Chapter 7) sequence work accordingly: wire the composed form first, then port the fused kernel.

---

## Chapter Goal

Chapter 2 derived a 12-operation TTNN composition that implements the DeltaNet decode step entirely on-device. That composed form is trace-compatible and correct. This chapter answers a separate question: does a better starting point already exist in the form of the `gdn_full_fused_inplace` kernel from the Qwen3.5-27B Blackhole implementation, and if so, how much work is required to run it on Wormhole T3K?

The answer drives Task 6 in the Chapter 7 implementation roadmap. If the kernel is `REUSABLE_WITH_TUNING`, porting it is lower-effort than writing a new kernel from scratch. If it is `REQUIRES_REWRITE`, the composed TTNN form (Chapter 2) provides the algorithmic baseline and the Mamba SSM kernel patterns (Chapter 5) provide useful implementation idioms for a new kernel.

---

## Prerequisites

- Chapter 2 (`ch2_ttnn_decomposition/`): establishes the 6-operation structure that the fused kernel must implement; the L1 feasibility result from `state_tensor_memory_config.md` (per-head state = 32 KB, well within 1.5 MB Wormhole L1) is the key input to the CB layout analysis in this chapter.
- Chapter 1 (`ch1_trace_breakage_audit/`): documents that the existing `recurrent_gated_delta_rule` host fallback is the primary decode latency bottleneck; the fused kernel is the path to eliminating that latency.

---

## Reuse Classification

| Classification | Meaning | Outcome |
|---|---|---|
| `REUSABLE_AS_IS` | Compiles and passes correctness on Wormhole T3K without source changes | Best case; only requires verification testing |
| `REUSABLE_WITH_TUNING` | Architecturally compatible; CB size constants, core grid, or data format flags need Wormhole-specific values | A handful of `constexpr` changes; kernel logic unchanged |
| `REQUIRES_REWRITE` | Relies on a Blackhole-specific hardware feature unavailable on Wormhole | New TT-Metalium kernel using the Chapter 2 algorithm and Mamba-derived idioms |

The analysis in `gdn_full_fused_inplace_analysis.md` establishes that the expected classification is **`REUSABLE_WITH_TUNING`**: the 6-operation algorithmic structure and DMA streaming pattern are hardware-agnostic, but CB size constants calibrated for Blackhole's 2 MB L1 must be rechecked against Wormhole's 1.5 MB, and any FP32_DEST_ACC accumulation paths must be reviewed.

---

## Files in Reading Order

1. [`gdn_full_fused_inplace_analysis.md`](./gdn_full_fused_inplace_analysis.md) — What the kernel computes, where to find its source, key parameters to document, and architecture-specific concerns for the Blackhole-to-Wormhole port. Establishes the `REUSABLE_WITH_TUNING` classification.
2. [`wormhole_t3k_adaptation.md`](./wormhole_t3k_adaptation.md) — Required constant changes for Wormhole T3K, CB layout, core grid assignment for 4 heads per device, multi-device sharding note, and verification test specification.

---

## What's Next

After this chapter, the kernel reuse decision is made: port and tune `gdn_full_fused_inplace` for Wormhole T3K (Task 6 in Chapter 7). The exact list of constant changes is in `wormhole_t3k_adaptation.md`.

Chapter 5 (`ch5_scan_primitives_survey/`) surveys the broader landscape of scan and recurrence kernels in tt-metal to confirm whether any other existing primitive is a closer match for the DeltaNet decode step. The conclusion there reinforces the Chapter 4 strategy: no scan primitive applies; the composed TTNN form (Chapter 2) or the ported fused kernel (this chapter) are the two correct paths.
