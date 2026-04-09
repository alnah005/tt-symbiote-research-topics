# Chapter 4 — Performance Analysis Tools

TT-Lang ships three complementary profiling modes, each activated by an environment variable. All three read the same underlying data source — `profile_log_device.csv` produced by tt-metal's device profiler — but present different views of that data. A fourth env var launches a Perfetto trace server for interactive visualization.

## Prerequisites

Every profiling mode requires:

1. **`TT_METAL_DEVICE_PROFILER=1`** — tells tt-metal to instrument kernels with hardware cycle counters and write `profile_log_device.csv` into `$TT_METAL_HOME/generated/profiler/.logs/`.
2. **`TT_METAL_HOME`** — points to the tt-metal installation root. The profiler CSV and NOC trace JSONs are located at `$TT_METAL_HOME/generated/profiler/.logs/`.

Some modes have additional prerequisites noted in their respective sections.

## Profiling Modes at a Glance

| Env Var | Mode | What It Shows | Key Module |
|---|---|---|---|
| `TTLANG_AUTO_PROFILE=1` | [Auto-profile](./auto_profile.md) | Per-source-line cycle counts, CB flow attribution, roofline analysis | `ttl._src.auto_profile` |
| `TTLANG_SIGNPOST_PROFILE=1` | [Signpost profile](./signpost_profile.md) | User-defined profiling zones with aggregated cycle counts | `ttl._src.signpost_profile` |
| `TTLANG_PERF_DUMP=1` | [Perf dump](./perf_dump_and_perfetto.md) | NOC profiler summary, CB flow graph, pipe graph | `ttl._src.perf_summary` |
| `TTLANG_PERF_SERV=1` | [Perfetto server](./perf_dump_and_perfetto.md) | Interactive Perfetto trace in browser | `ttl._src.perf_trace_server` |

## Execution Order in the Runtime

All profiling hooks run **after** kernel execution inside `pykernel_gen`'s wrapper (see [Chapter 2 — Compilation Pipeline](../ch2_compilation_pipeline/index.md) for how `pykernel_gen` works). The order is fixed:

```
1. compiled_kernel(*args)            # Execute on device
2. _run_profiling_pipeline(...)      # TTLANG_AUTO_PROFILE
3. _run_perf_dump(...)               # TTLANG_PERF_DUMP
4. _run_signpost_profile(...)        # TTLANG_SIGNPOST_PROFILE
5. serve_trace(...)                  # TTLANG_PERF_SERV (blocks until Enter)
```

Each hook calls `ttnn.ReadDeviceProfiler(device)` to flush profiler data from the device before parsing the CSV. The `TTLANG_PERF_SERV` hook runs last because it blocks on user input (it starts an HTTP server and waits for Enter).

## Data Flow

```
TT-Lang kernel
    |
    v
[Device execution with signpost instrumentation]
    |
    v
profile_log_device.csv   noc_trace_*.json   cb_flow_graph.json
    |                         |                    |
    +-- auto_profile.py       |                    |
    +-- signpost_profile.py   |                    |
    +-- perf_trace_server.py  |                    |
    |                         |                    |
    +-------------------------+--------------------+
                              |
                         perf_summary.py
```

## Compilation Pipeline Integration

Two profiling modes inject MLIR passes during compilation (defined in `ttl_api.py`):

- **Auto-profile** adds `ttl-dump-cb-flow-graph` to produce `cb_flow_graph.json`, which maps CB wait/reserve ops to their DMA sources for attribution in the report.
- **Perf dump** also adds `ttl-dump-cb-flow-graph` (writing to `/tmp/ttlang_cb_flow_graph.json`).
- Both modes rely on `ttl-lower-signpost-to-emitc`, which converts `ttl.signpost` ops into `DeviceZoneScopedN` C++ calls that the device profiler captures as `ZONE_START`/`ZONE_END` rows in the CSV.

## Chapter Contents

- [`auto_profile.md`](./auto_profile.md) — Automatic per-line profiling with CB flow attribution and roofline analysis
- [`signpost_profile.md`](./signpost_profile.md) — User-defined profiling zones via `ttl.signpost`
- [`perf_dump_and_perfetto.md`](./perf_dump_and_perfetto.md) — NOC profiler summary, CB/pipe graph dump, and Perfetto trace server

## Key Takeaways

1. **All profiling reads the same CSV.** The device profiler CSV (`profile_log_device.csv`) is the single source of truth. Different tools just slice it differently.
2. **Signposts are the instrumentation primitive.** Whether inserted automatically (auto-profile) or manually (`ttl.signpost`), profiling zones become `ZONE_START`/`ZONE_END` pairs in the CSV keyed by signpost name and RISC thread.
3. **Profiling is post-execution.** Data is flushed from the device via `ttnn.ReadDeviceProfiler(device)` after the kernel completes — there is no live streaming.
4. **Compilation and runtime cooperate.** The MLIR pass `ttl-dump-cb-flow-graph` runs at compile time to produce structural metadata; the Python profiling modules consume it at runtime alongside the device CSV.
5. **`TTLANG_PERF_SERV` is additive.** It can be combined with any other profiling flag to get a Perfetto visualization of the same run.
