# Chapter 3 — CSV Reference: `ops_perf_results.csv`

## Overview

Chapter 2 showed how to set `TT_METAL_DEVICE_PROFILER=1`, run a TTNN program, and invoke `process_ops_logs.py` to produce `ops_perf_results.csv`. Chapter 3 explains what is inside that CSV — every column, what units it uses, and how the numbers relate to each other.

By the end of this chapter you will be able to:

- Identify any column in `ops_perf_results.csv` and state what it measures.
- Convert raw cycle counts to nanoseconds for Wormhole B0.
- Explain why `DEVICE KERNEL DURATION` differs from individual RISC durations.
- Interpret `PM IDEAL` and `FPU UTIL` and know when an op is compute-bound or memory-bound.

## Prerequisites

- **Chapter 2** — specifically:
  - How `TT_METAL_DEVICE_PROFILER=1` enables per-core cycle-counter instrumentation.
  - How `process_ops_logs.py` aggregates the raw per-core logs from `tt_metal/tools/profiler/logs/` into a single `ops_perf_results.csv`.
  - The names and locations of the output artifacts produced by the post-processing step.

## Files in This Chapter

| File | What it covers |
|---|---|
| [`csv_column_definitions.md`](./csv_column_definitions.md) | How the CSV is produced, complete column reference, cycle-to-ns conversion |
| [`pm_ideal_and_fpu_util.md`](./pm_ideal_and_fpu_util.md) | PM IDEAL roofline model, FPU UTIL derivation, interpreting efficiency ratios |

## Quick-Reference Column Cheat Sheet

For the complete column reference, see [`csv_column_definitions.md`](./csv_column_definitions.md). The four most-used non-obvious columns at a glance:

| Column | One-liner |
|---|---|
| `DEVICE KERNEL DURATION [ns]` | Wall-clock span from first core start to last core end, in nanoseconds — the true elapsed time for the op |
| `FPU UTIL` | `PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles` — fraction of peak FPU throughput achieved (0.0–1.0) |
| `PM IDEAL [ns]` | `max(compute_cycles, memory_cycles)` converted to ns — the roofline lower bound on kernel duration |
| `NOC BW UTIL` | Fraction of peak NoC bandwidth actually used (0.0–1.0) — the memory-bound equivalent of FPU UTIL |

---

**Next:** [`csv_column_definitions.md`](./csv_column_definitions.md)
