# Chapter 5 — Low FPU Utilization: Causes and Remediation

## Overview

Chapter 4 established how to classify a kernel as compute-bound, bandwidth-bound, or overhead-bound using `FPU UTIL` and `NOC BW UTIL`. This chapter addresses the case that surprises practitioners most: the kernel is a large matmul, arithmetic intensity is well above the ridge point (8.0 FLOPs/byte), yet `FPU UTIL` is low — far below 0.7.

This situation means the FPU is not being fed work at its theoretical rate. Something upstream of the math engine, or structural about the problem shape, is leaving cycles on the table. There are seven distinct causes, each with a characteristic CSV signature and a targeted fix.

---

## Learning Objectives

By the end of this chapter you will be able to:

1. Name all seven root causes of low `FPU UTIL` on Wormhole B0.
2. Identify each cause from a specific pattern in the Tracy CSV columns.
3. Apply the correct remediation lever for each cause without disturbing unrelated parameters.
4. Follow the diagnostic checklist to efficiently rule out causes in order of ease.
5. Determine when to escalate to a kernel-level investigation (Cause 7) versus when a config change suffices.

---

## Prerequisites

- **Chapter 3** — CSV Column Reference: `FPU UTIL`, `NOC BW UTIL`, `TRISC0 KERNEL DURATION [ns]`, `TRISC1 KERNEL DURATION [ns]`, `TRISC2 KERNEL DURATION [ns]`, `DEVICE KERNEL DURATION [ns]`, `PM IDEAL`, `CORE COUNT`, `DATA FORMAT`, `MATH FIDELITY`.
- **Chapter 4** — Compute vs. Bandwidth Classification: roofline model, arithmetic intensity, the ridge point (AI_ridge = 8.0 FLOPs/byte for Wormhole B0), and the FPU UTIL classification thresholds (>0.7 compute-bound, <0.3 bandwidth-bound).

---

## Summary Table

The table below is a quick-reference digest. Full cause descriptions, rationale, and examples are in [`causes_of_low_fpu_util.md`](./causes_of_low_fpu_util.md). CSV signatures with exact column patterns are in [`csv_signatures.md`](./csv_signatures.md). Remediation parameters and guidelines are in [`remediation_levers.md`](./remediation_levers.md).

| # | Cause | Key CSV Signature | Fix |
|---|---|---|---|
| 1 | Insufficient tile count (small M, K, or N) | `FPU UTIL < 0.2`, `CORE COUNT` exceeds effective tile parallelism | Reduce `compute_with_storage_grid_size` |
| 2 | Sub-optimal data format (FP32 where BF16 suffices) | `DATA FORMAT == "FLOAT32"`, `FPU UTIL ≈ 0.5 × expected_BF16_util` | Convert operands to `ttnn.bfloat16` |
| 3 | Math fidelity mismatch (`HiFi4` where `LoFi` tolerable) | `MATH FIDELITY == "HiFi4"`, `TRISC1 KERNEL DURATION [ns] ≈ 4 × LoFi baseline` | Set `math_fidelity=MathFidelity.LoFi` |
| 4 | TRISC0/TRISC2 pipeline stalls (unpacker latency) | `TRISC0 KERNEL DURATION > 1.2 × TRISC1 KERNEL DURATION` | Increase L1 double-buffering; use sharded memory |
| 5 | NoC contention (too many active cores) | `NOC BW UTIL > 0.8`, `FPU UTIL < 0.3` | Reduce core count; interleave read patterns |
| 6 | Program cache miss (recompilation on first call) | First-call `DEVICE KERNEL DURATION >> 10 ×` steady-state duration | `ttnn.enable_program_cache()` |
| 7 | Incorrect loop count in kernel (padded shapes) | `FPU UTIL` stable but consistently below 0.4, no other cause present | Kernel-level investigation |

> **Note:** The diagnostic checklist in [`csv_signatures.md`](./csv_signatures.md) specifies the recommended order for ruling out causes — start with Cause 6 (easiest to dismiss) and proceed toward Cause 7 (requires kernel inspection).

---

## Chapter Contents

| File | Description |
|---|---|
| [`causes_of_low_fpu_util.md`](./causes_of_low_fpu_util.md) | Detailed explanation of all seven causes: mechanism, observable effect, and targeted fix |
| [`csv_signatures.md`](./csv_signatures.md) | Exact CSV column patterns for each cause; diagnostic checklist with recommended ruling-out order |
| [`remediation_levers.md`](./remediation_levers.md) | API parameters and configuration options for each fix, with usage guidelines |

---

## Navigation

- **Previous:** [Chapter 4 — Compute-Bound vs. Bandwidth-Bound Classification](../ch4_compute_vs_bandwidth/index.md)
- **Next:** [Chapter 6 — Host Dispatch Overhead vs. Device Kernel Time](../ch6_host_dispatch_overhead/index.md)
