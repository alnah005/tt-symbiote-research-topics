# Chapter 2: Invoking the Profiler — Build, Environment, and Capture

## Overview

Chapter 1 established the conceptual model: Tracy is the C++ CPU-side profiler that records named zones in a binary `.tracy` database via the two-process capture model; the device profiler is the on-device cycle-counter profiler controlled by `TT_METAL_DEVICE_PROFILER=1` that writes `ops_perf_results.csv`. This chapter converts that model into a working capture session.

The chapter proceeds in three stages: (1) confirming the build is correctly configured to produce profiler instrumentation, (2) setting the environment variables that gate both profilers at runtime, and (3) executing the full capture workflow from a clean shell to two output artifacts on disk.

## Learning Objectives

By the end of this chapter you will be able to:

- Identify whether a tt-metal build was compiled with `ENABLE_PROFILER=ON` by inspecting the binary's symbol table.
- Recite the five environment variables that control profiler behavior and state what each one does.
- Start `tracy-capture` and a pytest session in the correct order and verify that both profilers produced output.
- Diagnose the two most common capture failures: Tracy client/server version mismatch and the case where `TT_METAL_DEVICE_PROFILER=1` is set but `ENABLE_PROFILER` was not active at build time.
- Choose the correct environment variable set for latency measurement versus throughput measurement.

## Prerequisites

Before working through this chapter, confirm the following from Chapter 1:

- [ ] **Tracy zone model** — You understand that Tracy records named CPU zones using lock-free ring buffers and that zones are zero-cost when `TRACY_ENABLE` is not defined at compile time.
- [ ] **Two-process architecture** — You can describe why Tracy uses a separate `tracy-capture` server process to drain events from the profiled process over a local socket, and why both processes must be built from the same Tracy tag.
- [ ] **`TRACY_ENABLE` define** — You know that `TRACY_ENABLE` is a preprocessor symbol that activates all Tracy instrumentation macros; without it, every `ZoneScoped` and `FrameMark` expands to nothing.
- [ ] **Device profiler vs. Tracy distinction** — You can state which tool answers "how long did this kernel run on Tensix cores" (device profiler / `ops_perf_results.csv`) versus "how long did the host spend dispatching this op" (Tracy / `profile.tracy`), and why both are needed to evaluate total op latency. See Ch6 for the full latency decomposition.

## Five-Step Setup Checklist

The following checklist takes you from a clean source checkout to two output artifacts: a `.tracy` binary capture file and an `ops_perf_results.csv` device profiler log. See [`capture_workflow.md`](./capture_workflow.md) for full step details.

- [ ] **Step 1 — Build** with profiler instrumentation enabled. See [`build_requirements.md`](./build_requirements.md).
- [ ] **Step 2 — Configure** required environment variables for device profiling and Tracy capture. See [`env_vars_and_flags.md`](./env_vars_and_flags.md).
- [ ] **Step 3 — Start `tracy-capture`** in a separate terminal before launching the profiled process.
- [ ] **Step 4 — Run your pytest** with profiling env vars active.
- [ ] **Step 5 — Verify** that `profile.tracy` is non-zero and `ops_perf_results.csv` exists under `generated_profile_log_{test_name}/`.

## Files in This Chapter

| File | Description |
|---|---|
| [`build_requirements.md`](./build_requirements.md) | The `ENABLE_PROFILER` CMake flag, the `TRACY_ENABLE` preprocessor define, how to verify the build, and the runtime overhead warning for production benchmarks. |
| [`env_vars_and_flags.md`](./env_vars_and_flags.md) | Complete reference table of all runtime environment variables, interaction rules, and pytest-specific configuration patterns. |
| [`capture_workflow.md`](./capture_workflow.md) | Step-by-step capture procedure, output artifact locations, common failure modes, and a minimal working pytest example. |

---

**Start here:** [`build_requirements.md`](./build_requirements.md)

---

**Previous:** [Chapter 1 — Tracy Profiler Fundamentals](../ch1_tracy_fundamentals/index.md)
