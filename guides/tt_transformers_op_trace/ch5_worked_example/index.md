# Chapter 5 — Putting It Together: A Worked Example

This chapter ties together every concept from Chapters 1–4 in a single, concrete end-to-end scenario: launching a Llama model with trace and Tracy profiling enabled, observing the warm-up and capture phases in stdout, and reading the resulting ops report to isolate warm-up, trace-capture, and production-replay rows. No new concepts are introduced; the goal is to show how the pieces fit together in practice.

## Learning Objectives

After reading this chapter you should be able to:

1. Set the three required environment variables and assemble a `run_device_profiler` call that produces a complete `ops_perf_results_<date>.csv` for a Llama decode run with trace enabled.
2. Identify the four runtime phases in stdout — compile run, warm-up, trace-capture pass, and production replay — and know which Tracy markers and CSV column values mark each boundary.
3. Use `find_repeated_runs` and `split_compile_and_trace` to programmatically slice the CSV into compile-phase and trace-replay DataFrames.
4. Read `METAL TRACE ID` and `METAL TRACE REPLAY SESSION ID` from a row and immediately classify it as a warm-up row, a capture row, or a replay row.
5. Sum `DEVICE KERNEL DURATION [ns]` over replay rows to compute per-decode-step device time.
6. Recognise the common pitfalls — silent `num_runs=1` misuse, `.loc` end-inclusivity, and unchecked `left == -1` — before they produce misleading results.

## What's Next

Read the files in this chapter in the following order:

1. [`running_the_example.md`](./running_the_example.md) — prerequisites, the full launch command, expected stdout, and output file locations.
2. [`annotated_ops_report.md`](./annotated_ops_report.md) — walking through the CSV phase by phase, with a representative annotated table and step-by-step Python snippets.

---

**Previous:** [Chapter 4 — Tracy Profiling and the Op Trace](../ch4_tracy_profiling/index.md)
