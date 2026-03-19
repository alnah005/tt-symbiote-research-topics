# Chapter 4 — Compute-Bound vs. Bandwidth-Bound Classification

## Overview

Every kernel running on a Wormhole B0 Tensix core is ultimately constrained by one of two physical limits: the floating-point unit (FPU) throughput, or the rate at which data can be moved across the NoC or from DRAM. Understanding which limit applies to a given op is the prerequisite for any meaningful optimization effort — an op that is compute-bound will not benefit from reducing memory traffic, and an op that is bandwidth-bound will not benefit from algorithmic tricks that reduce FLOP count.

This chapter answers the central question:

> **Is this kernel limited by FPU throughput, or by NoC/DRAM memory bandwidth?**

The answer comes directly from the CSV columns produced by the device profiler and discussed in Chapter 3. No additional instrumentation is required.

---

## Learning Objectives

By the end of this chapter you will be able to:

1. Apply the roofline model to a Wormhole B0 core given hardware ceiling values.
2. Compute a kernel's arithmetic intensity from its input/output shapes and data format.
3. Use `FPU UTIL` and `NOC BW UTIL` to classify any kernel as compute-bound, bandwidth-bound, or overhead-bound.
4. Interpret the ratio `DEVICE KERNEL DURATION / PM IDEAL` to detect unexplained overhead beyond the modeled bound.
5. Corroborate your classification using the `TRISC0 DURATION`, `TRISC1 DURATION`, and `TRISC2 DURATION` breakdown.

---

## Prerequisites

- **Chapter 3** — CSV Column Reference: `PM IDEAL`, `FPU UTIL`, `TRISC0 DURATION`, `TRISC1 DURATION`, `TRISC2 DURATION`, `DEVICE KERNEL DURATION`, and `NOC BW UTIL` must be familiar before proceeding.

---

## Quick-Reference Decision Table

Use this table for a fast first-pass classification. Full derivation and edge cases are in [`classification_method.md`](./classification_method.md).

| `FPU UTIL` | `NOC BW UTIL` | Classification | Primary Bottleneck |
|---|---|---|---|
| High (> 0.7) | Low (< 0.4) | Compute-bound | FPU throughput |
| Low (< 0.3) | High (> 0.7) | Bandwidth-bound (NoC) | NoC data movement |
| Low (< 0.3) | Low (< 0.3) | Overhead-bound | Dispatch, stalls, or pipeline inefficiency |
| Medium | Medium | Balanced | Both limits simultaneously active |

> **Note:** These thresholds are starting points. Always cross-check against `DEVICE KERNEL DURATION / PM IDEAL` and the TRISC duration breakdown before drawing a final conclusion.

---

## Chapter Contents

| File | Description |
|---|---|
| [`roofline_model_primer.md`](./roofline_model_primer.md) | Roofline model theory, Wormhole B0 hardware ceilings, arithmetic intensity, ridge point |
| [`classification_method.md`](./classification_method.md) | Step-by-step classification procedure using CSV columns; decision flowchart; full threshold table |
| [`worked_examples.md`](./worked_examples.md) | Three end-to-end worked examples: large matmul, small matmul, elementwise op |

---

## Navigation

- **Previous:** [Chapter 3 — CSV Column Reference](../ch3_csv_reference/index.md)
- **Next:** [Chapter 5 — Low FPU Utilization: Causes and Remediation](../ch5_low_fpu_util/index.md)
