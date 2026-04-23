# Chapter 5 — Scan and Recurrence Primitives Survey

No existing scan or recurrence kernel in tt-metal can be adapted for DeltaNet decode. The Mamba SSM kernel provides useful implementation patterns (DMA streaming, L1 state management, outer product idioms) but its inner loop is structurally different from DeltaNet's non-separable retrieval-then-write update. Parallel prefix scan is irrelevant at decode time (T=1). The correct starting point is the composed TTNN form from Chapter 2.

> **Key Finding:** The DeltaNet decode step is structurally distinguished from all surveyed scan primitives by a single property: the outer product write `k̃_t ⊗ error` depends on `retrieval = S_{t-1}^T @ k̃_t`, which is itself a read from the current state. This data dependency makes the DeltaNet update non-separable — the write cannot be computed without first reading the state. No existing tt-metal scan or recurrence kernel encodes this read-before-write pattern. All candidates are classified `PARTIAL_REUSE` (for borrowed idioms) or not applicable.

---

## Chapter Goal

Chapter 2 established that the composed TTNN form is the correct immediate implementation path — all 12 operations are available today. This chapter asks a complementary question: is there an existing scan or recurrence primitive in tt-metal that is a closer starting point than composing from scratch? If yes, it would accelerate the fused kernel development in Chapter 4. If no, the Chapter 4 strategy (port `gdn_full_fused_inplace` from Blackhole, or use the Mamba SSM idioms as reference) stands.

The answer is: none of the surveyed primitives can be directly adapted. The Mamba SSM kernel is the most structurally similar candidate, and its DMA and outer product patterns are borrowable for Chapter 4 kernel development — but its inner loop logic is wrong for DeltaNet. Parallel prefix scan does not apply at T=1. No GLA or RetNet kernel exists in tt-metal.

---

## Prerequisites

- Chapter 2 (`ch2_ttnn_decomposition/`, particularly `recurrence_math_and_tensor_ops.md`): the six DeltaNet decode operations and the non-separability argument are derived there; this chapter references that derivation when explaining why each candidate fails.
- Chapter 4 (`ch4_gdn_fused_kernel/`): the Mamba SSM kernel patterns identified in this chapter are inputs to the Chapter 4 kernel development strategy; read Chapter 4 alongside or after this chapter.

---

## Files in Reading Order

1. [`mamba_ssm_kernel_review.md`](./mamba_ssm_kernel_review.md) — Review of the Mamba SSM selective scan kernel; structural comparison with DeltaNet; reuse classification `[PARTIAL_REUSE]`; list of borrowable idioms for Chapter 4.
2. [`parallel_prefix_scan_review.md`](./parallel_prefix_scan_review.md) — Why parallel prefix scan is irrelevant for DeltaNet decode (T=1 means one recurrence step; nothing to parallelize); why it also does not apply to prefill (intra-chunk WY step is not associative).
3. [`gla_and_related_kernel_survey.md`](./gla_and_related_kernel_survey.md) — Survey of GLA, RetNet, and related linear attention kernels in tt-metal; expected finding: none exist; summary table; conclusion.

---

## Survey Summary

| Candidate | What it computes | Key structural difference from DeltaNet | Reuse classification |
|---|---|---|---|
| Mamba SSM selective scan | `h_t = A * h_{t-1} + B_t x_t`; `y_t = C_t h_t` | Write `B_t ⊗ x_t` is independent of current state — no retrieval step | `[PARTIAL_REUSE]` — outer product and DMA idioms borrowable |
| Parallel prefix scan | Associative scan over a sequence; `⊕` operator must be associative | At T=1 (decode), there is exactly one recurrence step — nothing to scan over | Not applicable for decode |
| GLA / RetNet kernels | Gated linear attention state update with scalar per-channel decay | No tt-metal implementation found | `[GAP — requires new kernel]` if needed |
| Composed TTNN (Chapter 2) | All 6 DeltaNet ops via TTNN primitives | This IS the correct implementation; all ops available | `[AVAILABLE — needs wiring]` |

---

## What's Next

After this chapter, the scan primitive survey is complete. The conclusion reinforces the implementation strategy from Chapters 2 and 4:

- **Immediate path (trace compatibility):** Wire the composed TTNN form from Chapter 2. All ops are available; no kernel development required.
- **Latency path:** Port or tune `gdn_full_fused_inplace` from Chapter 4, borrowing the Mamba SSM DMA and outer product idioms identified in `mamba_ssm_kernel_review.md`.

Chapter 6 (`ch6_latency_and_accuracy/`) quantifies the latency gap between the host fallback, the composed TTNN form, and the fused kernel, and establishes the PCC accuracy thresholds for the TTNN implementation.
