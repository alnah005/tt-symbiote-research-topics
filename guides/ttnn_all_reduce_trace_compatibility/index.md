# ttnn.all_reduce Trace Compatibility

This guide examines whether `ttnn.all_reduce` can be captured inside a `ttnn` device trace on a T3K MeshDevice and defines the exact conditions required to do so safely. It is written for ML engineers integrating collective operations into traced inference pipelines.

---

## How to Use This Guide

| Goal | Recommended path | Direct links |
|------|-----------------|--------------|
| Understand how trace capture works on MeshDevice | Read Ch 1 first | [Trace Capture Mechanics](ch1_trace_mechanics/index.md) |
| Understand what all_reduce does internally | Read Ch 2 | [all_reduce Internal Architecture](ch2_all_reduce_internals/index.md) |
| Get the go/no-go verdict and requirements | Jump to Ch 3 | [Compatibility Verdict](ch3_verdict/index.md) |
| Integrate all_reduce into a traced model | Read Ch 3, then Ch 4 | [Verdict](ch3_verdict/index.md), [Integration Checklist](ch4_integration/index.md) |
| Run tests to validate trace compatibility | Go directly to Ch 4 | [Integration Checklist and Test Strategy](ch4_integration/index.md) |

---

## Chapter Index

| # | Title | Description | Key concepts |
|---|-------|-------------|--------------|
| 1 | [Ch 1 — Trace Capture Mechanics on MeshDevice](ch1_trace_mechanics/index.md) | How `ttnn.begin_trace_capture` / `ttnn.execute_trace` record and replay programs on multi-chip devices | Trace buffer, program cache, persistent buffers, `MeshDevice` |
| 2 | [Ch 2 — ttnn.all_reduce Internal Architecture](ch2_all_reduce_internals/index.md) | Code paths inside `ttnn.all_reduce`: composite vs. non-composite, semaphore handling, intermediate allocation | `reduce_scatter`, `all_gather`, `all_broadcast`, `scattered_tensor`, semaphore `std::nullopt` |
| 3 | [Ch 3 — Trace Compatibility Verdict and Requirements](ch3_verdict/index.md) | Definitive verdict on which paths are trace-safe and the precise buffer pre-allocation requirements | Non-composite path, composite path incompatibility, `scattered_tensor` pre-allocation, bias buffer |
| 4 | [Ch 4 — Integration Checklist and Test Strategy](ch4_integration/index.md) | Step-by-step checklist for wrapping a traced model that calls `all_reduce`, plus a test matrix | `TTNNLinearIColShardedWAllReduced`, `@trace_enabled`, persistent buffer lifecycle, test cases |

---

## Quick Reference

| API / concept | What it does | Where to learn more |
|---------------|-------------|---------------------|
| `ttnn.begin_trace_capture(device, trace_buffer_size)` | Starts recording Metal programs into a replayable trace | [Ch 1](ch1_trace_mechanics/index.md) |
| `ttnn.execute_trace(device, trace_id)` | Replays the captured program sequence without re-dispatching host-side ops | [Ch 1](ch1_trace_mechanics/index.md) |
| `ttnn.all_reduce` — non-composite path | Synchronous reduce-scatter + all-gather using only per-program local semaphores; trace-safe on T3K decode shapes | [Ch 2](ch2_all_reduce_internals/index.md), [Ch 3](ch3_verdict/index.md) |
| `ttnn.all_reduce` — composite path | Uses `all_broadcast` + `concat` + `local_sum`; allocates dynamic intermediates; trace-incompatible | [Ch 2](ch2_all_reduce_internals/index.md), [Ch 3](ch3_verdict/index.md) |
| `scattered_tensor` | Intermediate output of `ttnn.reduce_scatter` inside `all_reduce`; must be pre-allocated as a persistent buffer before trace capture | [Ch 3](ch3_verdict/index.md), [Ch 4](ch4_integration/index.md) |
| Semaphore argument (`std::nullopt`) | `ttnn.all_reduce` passes `std::nullopt` for all semaphore args; no persistent global semaphore state crosses trace boundaries | [Ch 2](ch2_all_reduce_internals/index.md) |
| `TTNNLinearIColShardedWAllReduced` | Linear layer subclass that calls `all_reduce`; inherits `@trace_enabled` from `TTNNLinear` | [Ch 4](ch4_integration/index.md) |
| `self.tt_bias` | Bias buffer that must be pre-loaded at a fixed device address before trace capture | [Ch 3](ch3_verdict/index.md), [Ch 4](ch4_integration/index.md) |

---

## Prerequisites

Readers should be familiar with the following before using this guide:

- Basic `ttnn` tensor and device API (`ttnn.from_torch`, `ttnn.to_device`, `ttnn.Tensor`)
- `MeshDevice` concepts: device grid, sharding, multi-chip communication
- What collective operations are: reduce-scatter, all-gather, all-reduce semantics
- General understanding of how Metal programs and command queues operate on Tenstorrent hardware

---

## Source Code Location

All paths are relative to the root of the `tt-metal` repository.

| Component | Path |
|-----------|------|
| `ttnn.all_reduce` entry point | `ttnn/cpp/ttnn/operations/ccl/all_reduce/all_reduce.cpp` |
| Reduce-scatter implementation | `ttnn/cpp/ttnn/operations/ccl/reduce_scatter/` |
| All-gather implementation | `ttnn/cpp/ttnn/operations/ccl/all_gather/` |
| Trace capture / execute API | `ttnn/cpp/ttnn/operations/trace.cpp` |
| `TTNNLinear` and `TTNNLinearIColShardedWAllReduced` | `models/experimental/llama/tt/ttnn_linear.py` |
| MeshDevice trace buffer management | `tt_metal/impl/device/mesh_device.cpp` |
