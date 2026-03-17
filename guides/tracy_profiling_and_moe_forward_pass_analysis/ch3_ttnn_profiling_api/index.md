# Chapter 3: TTNN Op-Level Profiling API

## Overview

This chapter covers the TTNN (TT Neural Network) op-level profiling API — the tools and techniques for measuring how long each individual TTNN operation spends executing on the Tenstorrent device. Where Chapter 1 introduced the Tracy two-process model and Chapter 2 covered building and capturing a trace, this chapter focuses on understanding what happens *inside* those traces at the per-operation granularity.

By the end of this chapter you will be able to profile a MoE (Mixture of Experts) forward pass at per-op resolution, annotate your own code sections with custom timing zones, and reconstruct the complete time budget of a forward pass by summing hardware kernel durations and accounting for the gaps.

---

## Learning Objectives

After completing this chapter you will be able to:

1. **Wrap a TTNN code block with the device profiler** to capture device-side timing for every TTNN operation that executes inside the block.
2. **Identify TTNN op names** as they appear in the profiler CSV output — understanding the naming convention so you can match CSV rows back to source code calls.
3. **Run `process_ops_logs.py`** and read the resulting timeline report — converting raw cycle counts to nanoseconds and understanding the output columns.
4. **Add custom Tracy zone markers** around user-defined code sections — using the Tracy Python bindings to create named zones visible in the Tracy GUI.
5. **Sum op durations to find gaps** between TTNN-reported device kernel time and wallclock measurement — the first step in diagnosing overhead sources.

---

## Prerequisites

This chapter builds directly on:

- **Chapter 1** — Tracy two-process model: the distinction between the Tracy *server* (GUI/capture process) and the *instrumented process*, and what CPU-side zones represent.
- **Chapter 2** — Build setup and capture workflow: how to build tt-metal with `ENABLE_TRACY=ON`, how to launch the capture server, and how to replay a saved trace file.

If you have not completed Chapters 1 and 2, review them before continuing. Code examples in this chapter assume you have a working Tracy capture environment and a tt-metal build with Tracy support enabled.

---

## Three Approaches to Per-Op Timing

TTNN exposes three complementary mechanisms for measuring operation latency. Understanding when to use each one is important before diving into the API details.

| Approach | What it measures | Where it appears | When to use it |
|---|---|---|---|
| **Tracy zones (CPU-side)** | Time between zone entry and exit on the host CPU — includes host dispatch overhead | Tracy GUI timeline, saved `.tracy` file | Measuring Python-level and dispatch latency; identifying CPU-side stalls |
| **Device profiler CSV** (`TT_METAL_DEVICE_PROFILER=1`) | On-device Tensix kernel execution time measured by hardware cycle counters | `ops_perf_results_<timestamp>.csv`, processed HTML/ODS report | Measuring true hardware execution cost of each op, independent of host overhead |
| **`ttnn.tracer`** | Python-level graph capture — records the sequence and shapes of TTNN ops as a graph IR | Python object, exportable to JSON | Understanding op graph structure; debugging unexpected op dispatch order |

These approaches are **not mutually exclusive**. The recommended workflow for MoE forward pass analysis is to use all three together: Tracy zones to find host-side gaps, the device profiler CSV to quantify hardware execution costs per op, and `ttnn.tracer` to verify the op dispatch sequence matches expectations.

---

## Chapter Structure

This chapter contains three sections:

1. [`device_profiler_api.md`](./device_profiler_api.md) — How to enable and use the device profiler, what it records, and how to post-process the CSV output.
2. [`annotating_your_code.md`](./annotating_your_code.md) — How to add custom Tracy zone markers in Python for fine-grained timing of your own code sections.
3. [`reading_op_timing_output.md`](./reading_op_timing_output.md) — How to read the profiler output, interpret CSV columns, reconstruct total MoE time, and diagnose gaps.

---

## Quick Reference

### Enabling the device profiler

```bash
# Preferred: set in the shell before running your script
TT_METAL_DEVICE_PROFILER=1 python run_moe_forward.py
```

Full details and caveats: [Enabling the Device Profiler](./device_profiler_api.md#enabling-the-device-profiler).

### Custom Tracy zone (Python)

```python
import tracy

with tracy.zone("MoE/dispatch"):
    # code inside this block appears as a named zone in the Tracy GUI
    expert_indices = compute_routing(hidden_states)
```

### Processing the CSV

```bash
python tt_metal/tools/profiler/process_ops_logs.py --csv ops_perf_results_<timestamp>.csv
```

---

## Next Steps

Continue to [`device_profiler_api.md`](./device_profiler_api.md) to learn how to enable device-side cycle counter collection and interpret what the profiler records for each TTNN op.
