# TT Transformers Op Trace: From Capture to Profiling

## Introduction

This guide is written for ML engineers and systems engineers working with the tt-transformers model stack on Tenstorrent hardware. It covers the full lifecycle of TTNN trace capture — from the core three-function API, through the concrete decode and prefill trace flows in the `Generator` class, model warm-up, Tracy profiling, and a worked end-to-end example that ties all the pieces together. Readers are expected to be comfortable with Python and basic C++, and to have some familiarity with ML inference workflows, including the distinction between prefill and decode phases and the concept of on-device tensors.

---

## Guide Overview

### [Chapter 1: Trace Capture in TTNN: The Core API](ch1_trace_capture_api/index.md)

Introduces the three-phase trace lifecycle — compile run, capture run, and replay — and the three public API calls (`ttnn.begin_trace_capture`, `ttnn.end_trace_capture`, `ttnn.execute_trace`) that drive it. Establishes the foundational concepts, constraints, and terminology that all subsequent chapters build on.

### [Chapter 2: How tt-transformers Uses Trace Capture](ch2_generator_trace_flows/index.md)

Walks through the concrete trace capture and replay code paths in the `Generator` class, covering the decode and prefill trace flows, the keying strategy that maps sequence length and model ID to stored trace handles, and the configuration tables in `model_config.py` and `trace_region_config.py` that govern when tracing is allowed.

### [Chapter 3: Model Warm-Up and Its Relationship to Trace Capture](ch3_warmup/index.md)

Explains the warm-up phase that runs before any user request is served, showing how `WarmupForwardMixin` orchestrates compile runs and trace captures across all supported sequence lengths and sampling configurations, and how to distinguish warm-up calls from production calls at both the Python level and in profiling output.

### [Chapter 4: Tracy Profiling with Trace Capture](ch4_tracy_profiling/index.md)

Covers how to run tt-transformers under Tracy profiling, which environment variables and CLI flags activate trace-aware profiling, what C++ markers are emitted at trace boundaries, and how to read the resulting `ops_perf_results_<date>.csv` to separate warm-up, trace-capture, and production-replay rows.

### [Chapter 5: Putting It Together: A Worked Example](ch5_worked_example/index.md)

Ties all prior chapters into a single, concrete end-to-end scenario: launching a Llama model with trace and Tracy profiling enabled, observing the warm-up and capture phases in stdout, and interpreting the ops report to isolate each runtime phase and compute per-decode-step device time.

---

## How to Use This Guide

The chapters are designed to be read sequentially: each one builds on concepts introduced in the chapters before it, and reading them out of order will leave gaps. Readers who are new to TTNN trace capture should start at Chapter 1 and work forward. Chapter 5 is where all concepts converge — it introduces no new material but provides an integrated, annotated reference that is most useful after the earlier chapters have been read. Readers who already understand the trace API and generator flows may use individual chapters as standalone references for the specific topic they need.
