# Chapter 6 — Sequence Length Scaling Analysis

## Overview

This chapter explains how to determine whether and how the 16ms gap identified in Chapter 5
scales with sequence length. The scaling behavior of a latency gap is one of the strongest
signals available for identifying its root cause: a gap that grows linearly with `seq_len`
almost certainly originates in compute or communication, while a gap that stays constant
regardless of `seq_len` points to a synchronization barrier or a fixed-cost host operation.

The method here is experimental: you run the MoE forward pass at a controlled sweep of
sequence lengths, measure the gap at each point using the same Tracy CSV and device profiler
CSV techniques from Chapter 5, and fit a scaling model to the resulting data. The chapter
closes with an interpretation guide that maps scaling exponents to root causes and
recommended follow-up actions.

---

## Prerequisites

This chapter requires:

- **Chapter 4** — The canonical MoE op sequence and per-op latency budgets for each phase.
  `scaling_theory.md` in this chapter references those per-phase latency estimates directly
  and extends them into a scaling model.
- **Chapter 5** — The gap measurement methodology (Tracy CSV export, device profiler CSV
  correlation, and the repeatability criterion). This chapter inherits that methodology and
  applies it across multiple `seq_len` values rather than at a single operating point.

No additional build flags or environment variables beyond those introduced in Chapters 2–3
are required to run the scaling sweep.

---

## Learning Objectives

After completing this chapter you will be able to:

1. **Explain why sequence length is the correct independent variable** for a MoE scaling
   experiment and why batch size and num_experts are held fixed.
2. **Predict the expected scaling exponent** for each op class in the MoE forward pass:
   matrix multiplications, softmax/topk, CCL collectives, host-side Python operations, and
   synchronization barriers.
3. **Design a controlled scaling experiment** using the standard seven-point `seq_len` sweep
   `{64, 128, 256, 512, 1024, 2048, 4096}`, including warm-up protocol, sample size, and
   confounder controls.
4. **Automate the sweep** using a parameterized pytest fixture or a standalone Python script
   that writes results to a structured CSV.
5. **Plot and interpret scaling results** using log-log axes, read the scaling exponent from
   the slope of the best-fit line, and apply the decision table to select the most likely
   root cause.
6. **Decompose a mixed gap** into its constant and linear components using linear regression,
   and attribute each component to a different root cause.

---

## The Four Expected Scaling Behaviors

Every gap observed in the MoE forward pass will exhibit one of four scaling behaviors as
`seq_len` increases. Understanding these behaviors before running the experiment lets you
form a falsifiable prediction for each gap pattern from Chapter 5.

### Behavior 1: Constant (O(1))

The gap duration does not change as `seq_len` increases. In log-log space, the best-fit
slope is approximately 0.

**What it means:** The gap originates in an operation whose cost is independent of the number
of tokens. Device synchronization barriers (`ttnn.synchronize_device`,
`ttnn.wait_for_event`) and program cache misses (Pattern D from Chapter 5) are the primary
sources of constant gaps. The barrier blocks the host for a fixed duration determined by
device queue drain time, which is set by the last enqueued kernel's latency — not by the
current token count.

**Investigation path:** Apply Method 2 from Chapter 5 (`gap_attribution.md`) to identify the
synchronization call. If the gap is present only on the first call, Pattern D (program cache
miss) is confirmed.

### Behavior 2: Linear (O(seq_len))

The gap duration doubles when `seq_len` doubles. In log-log space, the best-fit slope is
approximately 1.0.

**What it means:** The gap originates in an operation whose cost scales with the number of
tokens. Candidates include:
- CCL all-to-all latency, which scales with total message size
  (`num_active_tokens × d_model × 2 bytes`).
- Host-side Python operations that iterate over token assignments in a loop (Pattern A).
- Memory-bound matrix multiplications in the regime where `seq_len` is small enough that
  the operation is not compute-bound.

**Investigation path:** Compare the observed gap duration against the theoretical CCL latency
formula (see `scaling_theory.md`). If the gap magnitude matches, CCL is the primary
contributor (Pattern C). If the gap is larger than the CCL estimate, host-side Python
overhead is present in addition to or instead of CCL.

### Behavior 3: Sublinear (0 < slope < 1)

The gap grows with `seq_len` but slower than linearly. In log-log space, the slope is
between 0 and 1.

**What it means:** This behavior occurs when a matrix multiplication transitions from
memory-bound to compute-bound as `seq_len` increases, or when the gap is a superposition
of a constant term (synchronization) and a linear term (CCL or host Python). The slope
reflects the mix of the two components rather than a single clean scaling law.

**Investigation path:** Fit a linear regression model `gap = a + b × seq_len` to the data
in linear (not log-log) space. The constant term `a` corresponds to a synchronization
barrier; the slope `b` corresponds to a linear component. See `interpreting_scaling_results.md`
for the full procedure.

### Behavior 4: Non-monotonic

The gap is generally increasing with `seq_len` but has one or more discontinuous jumps or
flat regions that break the trend.

**What it means:** The MoE layer is hitting tile-count boundaries. Tenstorrent Wormhole B0
processes data in 32×32 tiles; when `seq_len` crosses a multiple of 32 that changes the
tile grid shape, the compiled kernel changes, which may trigger a program cache miss at that
specific `seq_len` value. Non-monotonic behavior always warrants investigation of
tile-alignment boundaries.

**Investigation path:** Check whether the discontinuities occur at `seq_len` values that are
multiples of 32 or multiples of 64. Run an additional fine-grained sweep around the
discontinuity (e.g., `{480, 512, 544}`) to confirm the boundary. See
`interpreting_scaling_results.md` for the full procedure.

---

## Pattern Predictions from Chapter 5

The following table maps each Chapter 5 gap pattern to its predicted scaling behavior
before the experiment is run. This table should be consulted before designing the sweep
to ensure the sweep points cover the relevant range for discriminating between patterns.

| Chapter 5 Pattern | Description | Predicted Scaling | Expected Log-Log Slope |
|---|---|---|---|
| **Pattern A** | Gap after `ttnn.topk` — unannotated host-side index construction | Linear (O(seq_len)) if using Python loops; O(1) if tensor-ized | ~1.0 (Python loop) or ~0 (tensor-ized) |
| **Pattern B** | Gap between last expert matmul and first combine op | Constant (O(1)) — sync barrier whose duration is independent of token count | Constant (slope ≈ 0) |
| **Pattern C** | Gap that scales with `num_active_tokens` — CCL all-to-all latency | Linear (O(seq_len)) because `num_active_tokens = seq_len × top_k` at fixed `top_k` | ~1.0 |
| **Pattern D** | Gap at the start of the MoE layer — program cache miss | Constant (O(1)) — compilation cost is independent of token count | ~0 |

---

## Chapter Structure

| File | Contents |
|---|---|
| `scaling_theory.md` | Why seq_len is the primary variable; expected scaling by op class; CCL message size formula |
| `experiment_design.md` | Sweep point selection; confounder controls; warm-up protocol; sweep automation script |
| `interpreting_scaling_results.md` | Log-log plotting; reading the scaling exponent; decision table; mixed-gap decomposition; non-monotonic results |

---

## Key Model Configuration (Used Throughout This Chapter)

All scaling formulas and example calculations in this chapter use the configuration from Chapter 4.
Model configuration constants (d_model, d_ff, top_k, num_experts for Qwen/DeepSeek): see Chapter 4, `index.md`.

The following T3K-specific value is used in CCL latency calculations throughout this chapter:

| Parameter | Value |
|---|---|
| Model | DeepSeek-V3 / Qwen 235B-A22B (same MoE dimensions) |
| Ethernet bandwidth | ~7 GB/s effective per inter-chip link |

---

## Next Steps

Proceed to [`scaling_theory.md`](./scaling_theory.md) to understand the theoretical basis
for each scaling behavior before designing the experiment.
