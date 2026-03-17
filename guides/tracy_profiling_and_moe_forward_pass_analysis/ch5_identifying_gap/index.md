# Chapter 5 — Identifying the 16ms Gap

## Overview

This chapter teaches how to read a Tracy trace to locate latency gaps between MoE forward pass
ops, correlate unaccounted time with specific synchronization points or communication ops, and
determine whether the observed 16ms gap originates in host-side Python dispatch, a device
synchronization barrier, or CCL collective communication.

The investigation method here is empirical: you start from a captured `.tracy` file and a
`ops_perf_results_<timestamp>.csv` from the device profiler, measure gap sizes programmatically
using `tracy-csvexport`, and then apply a three-method attribution process to assign each gap
to a root cause. The chapter closes with a catalog of four common gap patterns drawn from
real MoE workloads on Wormhole B0 and T3K hardware.

---

## Prerequisites

This chapter requires:

- **Chapter 2** — Tracy trace files, `tracy-csvexport`, and the `ops_perf_results_<timestamp>.csv`
  device profiler output format. The CSV column names used in this chapter (`DEVICE KERNEL
  DURATION [ns]`, `OP TO OP LATENCY [ns]`) are defined there.
- **Chapter 3** — The Tracy zone naming convention (`MoE/forward`, `MoE/dispatch`,
  `MoE/expert_matmul`, `MoE/combine`) established in `annotating_your_code.md`. Gap analysis
  in this chapter uses these zone names as anchors.
- **Chapter 4** — The canonical MoE op sequence and expected per-op latency budgets. This
  chapter treats Chapter 4's op table as the ground truth against which Tracy observations
  are compared.

---

## The Three Gap Hypotheses

Every latency gap in the MoE forward pass that is not accounted for by a named Tracy zone or
device profiler kernel entry falls under one of three hypotheses:

**Hypothesis 1 — Host-side Python overhead.**
The host thread is executing Python between TTNN op dispatches: index tensor construction,
Python loops over expert assignments, shape manipulation, or logging. This time does not appear
as a device profiler kernel entry and may not have a Tracy zone if the code was not annotated.
Characteristic Tracy evidence: the CPU thread timeline shows idle time on the main thread that
is not inside any custom zone.

**Hypothesis 2 — Device synchronization barrier.**
An explicit `ttnn.synchronize_device` or `ttnn.wait_for_event` call blocks the host thread
until the device queue drains. The host cannot dispatch further ops until the device finishes
all previously enqueued work. This gap is host-wall-clock time, not device idle time, but it
causes a visible stall before the next op is dispatched. Characteristic Tracy evidence: a
`ttnn.synchronize_device` zone appears on the CPU thread timeline; its duration matches the gap.

**Hypothesis 3 — CCL collective latency.**
An all-to-all or reduce-scatter collective communication op on the T3K mesh has non-trivial
latency that is either not annotated as a Tracy zone or is reported as a device-side gap
because the host is waiting for the CCL op to complete before enqueuing the next op. Latency
scales with message size (num_active_tokens × d_model bytes). Characteristic Tracy evidence: the
gap scales with seq_len and is larger than expected for a synchronization barrier.

---

## Decision Tree: Attributing a Latency Gap

Use the following decision tree when you find a gap larger than 1ms between consecutive Tracy
zones in the MoE forward pass:

```
Is there a ttnn.synchronize_device or ttnn.wait_for_event zone
spanning the gap duration?
      │
      ├── YES → Hypothesis 2: device synchronization barrier.
      │         Cross-check: does a corresponding event appear in the
      │         device profiler CSV as a gap between consecutive kernel
      │         start times? If yes, barrier is confirmed.
      │
      └── NO  →
            Does the gap appear between ttnn.gather (end of dispatch phase)
            and ttnn.matmul (start of expert compute)?
                  │
                  ├── YES →
                  │         Does the gap duration scale with seq_len
                  │         across multiple runs?
                  │               │
                  │               ├── YES → Hypothesis 3: CCL all-to-all latency.
                  │               │         Verify: gap_ms ≈ (seq_len × top_k × d_model
                  │               │                          × 2 bytes) / link_bandwidth
                  │               │
                  │               └── NO (constant duration) →
                  │                         Is the gap present only on the first call?
                  │                               ├── YES → Pattern D: program cache miss.
                  │                               └── NO  → Hypothesis 2 (unlabeled barrier).
                  │
                  └── NO  →
                        Does the gap appear immediately after ttnn.topk?
                              │
                              ├── YES → Pattern A: unannotated index construction.
                              │         Add a Tracy zone around the index build loop.
                              │
                              └── NO  → Add Tracy zones to narrow the gap location,
                                        then re-run and re-apply this tree.
```

---

## Learning Objectives

After completing this chapter you will be able to:

1. **Navigate the Tracy GUI** to locate the MoE forward pass in a timeline, identify zone
   nesting, and measure gap durations using the built-in ruler tool.
2. **Export Tracy zone data** to CSV and write a Python script that identifies all gaps
   larger than a configurable threshold.
3. **Apply the three-method attribution process** — host dispatch comparison, synchronization
   zone search, CCL alignment check — to assign each gap to a specific root cause.
4. **Recognize the four common gap patterns** (A–D) by their zone-level signatures and
   propose the correct diagnostic follow-up for each.
5. **Distinguish measurement noise from real gaps** by computing mean and standard deviation
   over multiple iterations and applying a repeatability criterion.

---

## Chapter Structure

| File | Contents |
|---|---|
| [`reading_tracy_traces.md`](./reading_tracy_traces.md) | Tracy GUI orientation; zooming into the MoE forward pass; CSV gap measurement script |
| [`gap_attribution.md`](./gap_attribution.md) | Three attribution methods; 16ms hypothesis table; ruling out noise |
| [`common_gap_patterns.md`](./common_gap_patterns.md) | Patterns A–D: signatures, root causes, diagnostic steps |

---

## Key Model Configuration (Used Throughout This Chapter)

Latency estimates and gap sizes in this chapter use DeepSeek-V3 MoE dimensions unless
otherwise noted, as that is the configuration in which the 16ms gap was originally observed.
All other dimensions (`d_model`, `d_ff`, `num_experts`, `top_k`, dtype) are defined in
[Chapter 4's configuration table](../ch4_moe_op_breakdown/index.md#key-model-configuration).

The ch5-specific values needed for gap analysis are:

| Parameter | Value |
|---|---|
| `expert_capacity` | 64 (at seq_len=1024) |
| Hardware | Wormhole B0 on T3K (8-chip mesh) |

Qwen 235B MoE uses the same `d_model`, `d_ff`, and `num_experts` values; latency estimates
transfer directly.

---

## Next Steps

Proceed to [`reading_tracy_traces.md`](./reading_tracy_traces.md) to learn how to navigate
the Tracy GUI and measure gap durations from both the visual timeline and a CSV export.
