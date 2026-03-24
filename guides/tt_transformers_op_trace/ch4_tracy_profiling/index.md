# Chapter 4 — Tracy Profiling with Trace Capture

This chapter explains how the Tracy profiler integrates with TT-Metal's trace capture machinery to produce the per-op performance CSV that downstream analysis tools consume. Chapters 1–3 established what traces are and how they are recorded in device firmware; this chapter shows how `python3 -m tracy -r` orchestrates the capture pipeline, how C++ macros stamp trace boundaries into the Tracy event stream, and how the post-processing scripts turn that stream into structured profiling data.

## Recap of earlier chapters

- **Chapter 1** covered the TT-Metal trace capture API: `BeginTrace`, `EndTrace`, `ReplayTrace`, and how the firmware captures and replays command sequences.
- **Chapter 2** described the trace-enabled generator flows in `tt_transformers`, including how `decode_forward` selects the trace path at runtime.
- **Chapter 3** explained the warm-up sequence that compiles ops and populates caches before a trace is captured.

## The `python3 -m tracy -r` subprocess flow

When `-r` (report mode) is active, `python3 -m tracy -r` runs three processes in sequence:

```
┌────────────────────────────────────────────────────────────────────────┐
│  python3 -m tracy -r -- pytest <test_file> ...                        │
│                                                                        │
│  1. Tracy capture process   ──────────────────────────────────────┐   │
│     (tracy-capture -o tracy_profile_log_host.tracy -f -p <port>)  │   │
│                                                                    │   │
│  2. Test process            ──────────────────────────────────────┤   │
│     python3 -m tracy [options] -- pytest <test_file> ...          │   │
│     (env: TTNN_OP_PROFILER=1, TT_METAL_DEVICE_PROFILER=1,        │   │
│           TT_METAL_PROFILER_TRACE_TRACKING=1)                     │   │
│                                                                    │   │
│  3. Post-processing (after test exits)  ──────────────────────────┘   │
│     a. csvexport -u -p TT_  →  tracy_ops_times.csv                    │
│     b. csvexport -m -s ";"  →  tracy_ops_data.csv                     │
│     c. process_ops(...)     →  ops_perf_results_<date>.csv            │
└────────────────────────────────────────────────────────────────────────┘
```

The capture process listens on a TCP port; the test process connects to it as a Tracy client and streams events. Once the test exits, `generate_report` in `tools/tracy/__init__.py` runs `csvexport` twice to produce `tracy_ops_times.csv` and `tracy_ops_data.csv`, then calls `process_ops` which in turn calls `process_ops_logs.py` to produce the final `ops_perf_results_<date>.csv`.

## Learning objectives

After reading this chapter you will be able to:

1. Invoke `python3 -m tracy -r` with the correct flags for a trace-capture profiling run.
2. Explain what each environment variable set by `-r` mode does and which C++ code path each one unlocks.
3. Identify the Tracy C++ macros that stamp trace begin/end/replay boundaries into the event stream.
4. Parse `ops_perf_results_<date>.csv` to separate compile-run rows, trace-capture rows, and production-replay rows.
5. Use `METAL TRACE ID` and `METAL TRACE REPLAY SESSION ID` columns to isolate per-iteration replay performance.
6. Distinguish trace-replayed ops from normally dispatched ops in the profiling output.

## What's next

- [Running tests with Tracy (`running_with_tracy.md`)](./running_with_tracy.md)
- [Tracy markers for trace boundaries (`tracy_markers_for_trace.md`)](./tracy_markers_for_trace.md)
- [Reading profiling output (`reading_profiling_output.md`)](./reading_profiling_output.md)
- [Differentiating trace ops from normal ops (`differentiating_trace_ops_from_normal_ops.md`)](./differentiating_trace_ops_from_normal_ops.md)
