# Chapter 1: Tracy Profiler Fundamentals for TTNN

## Overview

This chapter establishes the conceptual foundation for all profiling work in this guide. Before capturing your first trace or reading a CSV column, you need to know what each tool measures and where its boundaries lie.

Tracy records named CPU zones on the host with nanosecond resolution; the device profiler records kernel execution time directly on Tensix hardware using on-device cycle counters and emits `ops_perf_results.csv`. Neither tool alone gives the full picture — the child files in this chapter explain each tool's scope, blind spots, and when to use each.

## Learning Objectives

By the end of this chapter you will be able to:

- Explain what Tracy records, how it records it, and why its instrumentation is zero-cost when disabled.
- Describe the two-process model (profiled app + `tracy-capture` server) and name the output artifact it produces.
- Identify which Tracy zones tt-metal emits by default for TTNN ops.
- State what question Tracy answers and what question the device profiler answers, and recognize which tool to reach for first given a performance symptom.
- Recite the latency decomposition equation and assign the correct tool to each term.

## Prerequisites

Before working through this chapter, confirm the following:

- [ ] You have SSH or console access to a machine with a Wormhole B0 card installed and recognized by `tt-smi`.
- [ ] You have a local clone of `tt-metal` at a commit that includes the profiler infrastructure (`tt_metal/tools/profiler/` exists).
- [ ] You can build `tt-metal` from source — specifically, you have CMake 3.17+, a C++17 toolchain, and the dependencies listed in `tt-metal/INSTALLING.md`.
- [ ] You understand that profiler support requires a build with `ENABLE_PROFILER=ON` (covered in Chapter 2); for this chapter, no profiler build is required — we are reading concepts only.

## Files in This Chapter

| File | Description |
|---|---|
| [`what_is_tracy.md`](./what_is_tracy.md) | Tracy's architecture, data model, two-process capture model, and how it integrates with tt-metal instrumentation macros. |
| [`two_profilers_compared.md`](./two_profilers_compared.md) | Side-by-side comparison of Tracy and the device CSV profiler: what each answers, blind spots of each, and when to use one, the other, or both. |
