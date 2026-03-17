# Chapter 1 — Tracy Profiler Overview

This chapter introduces the Tracy profiler from first principles and places it in context alongside the TTNN device profiler. After reading this chapter you will understand what each tool measures, how they are architecturally different, and which questions each one is designed to answer. All subsequent chapters — from build setup through gap attribution — assume the vocabulary and mental model established here.

No prior profiling experience with Tracy or the TTNN device profiler is required. The prerequisite background is described in the section below.

---

## Prerequisites

Readers are expected to arrive with the following background. If any item is unfamiliar, resolve it before proceeding, as later chapters build directly on these concepts.

- **TTNN basics**: you can write a forward pass that uses `ttnn.matmul`, `ttnn.to_device`, and `ttnn.MemoryConfig`; you understand that ops are dispatched from a Python host to a Tenstorrent device.
- **MoE architecture**: you know what a gating network, top-K routing, and the dispatch/combine pattern are at a conceptual level.
- **Python development**: you are comfortable running Python scripts on a machine with tt-metal installed, reading stack traces, and setting environment variables.
- **General profiling intuition**: you understand that profiling means measuring where time is spent; you have used `time.perf_counter` or similar tools in Python.

---

## What This Chapter Covers

| File | Topic |
|---|---|
| `what_is_tracy.md` | Tracy's origins, data model, two-process architecture, and how tt-metal instruments itself with Tracy macros |
| `tracy_vs_device_profiler.md` | The distinction between Tracy (CPU-side host timing) and the TTNN device profiler (on-device kernel timing), and how to combine them |

---

## Learning Objectives

After completing this chapter, you will be able to:

1. **Explain what Tracy records** and how it differs from a sampling profiler: Tracy is an instrumentation-based profiler that records named CPU zones with nanosecond-precision timestamps; it does not sample call stacks.

2. **Describe the Tracy two-process model**: identify the distinct roles of the profiled application process and the capture server process.

3. **Identify the compilation flag** that enables Tracy in tt-metal (`TRACY_ENABLE`).

4. **Name the categories of events** that tt-metal annotates by default with Tracy zones: op dispatch, program enqueue, and mesh trace lifecycle events.

5. **Distinguish Tracy from the TTNN device profiler** on the axes of measurement location, output format, and blind spots.

6. **State the combined profiling workflow**: run Tracy first to identify which host-side phase has unexpected latency, then run the device profiler to determine whether the time within that phase is dominated by kernel execution or by host-device synchronization overhead.

---

## Quick Reference: Tracy vs. TTNN Device Profiler

The table below summarizes the two tools at a glance. For the full comparison, see `tracy_vs_device_profiler.md`.

| Dimension | Tracy | TTNN Device Profiler |
|---|---|---|
| **What it times** | Host CPU op dispatch and Python overhead | Tensix kernel execution (hardware cycle counts) |
| **Activation** | `TRACY_ENABLE` compile flag + `tracy-capture` server | `TT_METAL_DEVICE_PROFILER=1` env var at runtime |
| **Output format** | `.tracy` binary (Tracy GUI or `tracy-csvexport`) | `profile_log_device.csv` (post-processed via `process_ops_logs.py`) |
| **Key question** | When did the host enqueue this op? How long did dispatch take? | How long did the kernel actually run on Tensix? |
| **Blind spot** | Cannot see on-device kernel execution time | Cannot see host dispatch overhead or Python-level gaps |

---

## How to Read This Chapter

Read the two files in the following order:

1. **`what_is_tracy.md`** — Establishes what Tracy is, how its data model works, and how tt-metal has integrated it. This file is the prerequisite for every other chapter in the guide.

2. **`tracy_vs_device_profiler.md`** — Builds on the Tracy foundation to introduce the TTNN device profiler and explain how the two tools complement each other. The combined workflow described in this file is the methodology used in Chapters 5 and 6 for gap attribution and scaling analysis.

---

## Relationship to Later Chapters

- **Chapter 2** (`ch2_tracy_setup/`) requires the Tracy data model and two-process architecture introduced in `what_is_tracy.md`. Readers who try to follow the build-flag instructions in Chapter 2 without first reading `what_is_tracy.md` will encounter undefined references to "zones", "the capture server", and `TRACY_ENABLE`.

- **Chapter 3** (`ch3_ttnn_profiling_api/`) relies on the Tracy-vs.-device-profiler distinction from `tracy_vs_device_profiler.md` to explain when to use the Python `ttnn.device_profiler_state` context manager versus adding custom Tracy zone markers.

- **Chapter 5** (`ch5_identifying_gap/`) applies the methodology outlined in `tracy_vs_device_profiler.md` — specifically, comparing Tracy CPU zone end timestamps against device profiler kernel start times — to attribute the 16 ms MoE forward-pass gap to a specific root cause. The terminology defined in this chapter (Tracy zone, device profiler, `.tracy` file) is used throughout Chapter 5 without redefinition.

---

## Next Steps

Proceed to [`what_is_tracy.md`](what_is_tracy.md) to understand the Tracy profiler's architecture, data model, and integration with tt-metal before reading anything else in this guide.
