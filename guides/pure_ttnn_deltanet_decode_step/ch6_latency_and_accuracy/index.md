# Chapter 6 — Latency Impact and Numerical Accuracy

This chapter measures the latency cost of the current PyTorch fallback path for `recurrent_gated_delta_rule` at decode batch size B=1, estimates the expected latency for a pure on-device implementation using both the composed TTNN form and the fused kernel form, and establishes PCC accuracy thresholds that the on-device implementation must meet for the DeltaNet state update. The goal is to give the implementing engineer a clear before/after picture: how slow is the current host-CPU round-trip, how fast can the on-device forms go, and what numerical accuracy is required throughout.

> **Key Finding:** The host-CPU round-trip for the current `recurrent_gated_delta_rule` fallback costs approximately 300–700 µs per DeltaNet layer. At 30 DeltaNet layers, this contributes 9–21 ms per decode step — the dominant latency cost for Qwen3.6-35B-A3B linear attention inference on T3K. The composed on-device TTNN form (12 ops, no new kernel) eliminates the PCIe transfer entirely; its bottleneck is dispatch overhead, with a total estimated cost of approximately 1 ms for all 30 layers. The fused kernel form (`gdn_full_fused_inplace`, `[REUSABLE_WITH_TUNING]`) reduces this to approximately 177 µs total by collapsing 12 dispatches to 1 per layer. The PCC threshold for correctness is 0.999 per decode step, measured for both `S_new` and `o_t` against a PyTorch reference. State errors do not accumulate exponentially because the DeltaNet decay gate `g_t < 1` contracts errors from prior steps by `g_t^{T-t}`.

## Files in This Chapter

Read in this order:

1. `host_roundtrip_latency.md` — Breaks down the per-layer PCIe transfer and kernel execution costs that make up the 300–700 µs fallback cost; explains why the 128 KB `S_prev` transfer dominates.
2. `on_device_latency_estimate.md` — Derives analytic estimates for the composed TTNN form (~1 ms) and fused kernel form (~177 µs); explains why dispatch overhead, not DRAM bandwidth, is the bottleneck for the composed form.
3. `pcc_accuracy_thresholds.md` — Establishes the 0.999 PCC threshold, explains the error-decay argument from the DeltaNet decay gate, and defines the measurement methodology for validating correctness across 200 decode steps.

## What's Next

Chapter 7 synthesizes all findings from Chapters 1–6 into a concrete, prioritized implementation roadmap. It provides a numbered task list with priorities and complexity ratings, a step-by-step Metal Trace integration checklist, and a verification and test matrix. After completing Chapter 7, the implementing engineer has a complete specification for landing a trace-compatible, fully on-device DeltaNet decode step in `TTNNQwen3LinearAttention`.
