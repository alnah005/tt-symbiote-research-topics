# What Is Tracy?

## Origins

Tracy was created by Bartosz Taudul as a real-time, low-overhead C++ profiler originally designed for game engine development, where microsecond-accurate CPU and GPU timing is essential for maintaining smooth frame rates. Game engines demand profiling that is always on in development builds — instrumentation that can be left in hot paths without meaningfully changing the behavior being observed. These same requirements make Tracy well-suited for systems-level performance tooling in hardware-adjacent software like tt-metal: the instrumentation overhead is low enough to leave enabled during normal development without distorting the measurements you care about.

Tracy has since been adopted beyond game engines into graphics drivers, language runtimes, and ML framework kernels, wherever developers need to understand how CPU work maps to time at sub-millisecond granularity.

## What Tracy Records

Tracy captures several categories of events:

- **CPU zones** — the primary unit of Tracy data. A zone is a named, time-bounded span associated with a specific thread. You mark the start and end of a zone with macros; Tracy records the start timestamp (ns), end timestamp (ns), thread ID, source file, line number, and an optional color for the GUI. Zones can be nested, giving you a call-stack-like view of time.
- **GPU zones** — time ranges submitted to a GPU command queue, correlated with CPU-side submission time. (Used in graphics workloads; not the primary mechanism for Tensix device timing in tt-metal.)
- **Memory allocations** — heap allocation and deallocation events with call-site information, enabling memory usage timelines.
- **Frame markers** — logical "frame" boundaries used in game engines to align time series data; available as a lightweight annotation in non-game workloads too.
- **Free-form text messages** — arbitrary string payloads attached to a point in time on a thread, useful for logging op names, tensor shapes, or status strings alongside the timing data.

> **Note:** For TTNN profiling, CPU zones and free-form messages are the events you will encounter most. GPU zones and frame markers are not central to the Tensix workflow.

## The Tracy Data Model

Every event Tracy records shares a common structure. A CPU zone entry contains:

| Field | Type | Description |
|---|---|---|
| Zone name | string | Human-readable label, e.g., `"ttnn::matmul"` |
| Start timestamp | uint64 (ns) | Nanoseconds since a monotonic epoch anchored at process start |
| End timestamp | uint64 (ns) | Nanoseconds at zone exit |
| Thread ID | uint32 | OS thread ID of the recording thread |
| Source location | string | File and line number of the instrumentation macro (compile-time constant) |
| Payload | optional string | Free-form annotation attached at zone start or end |

Zones are hierarchical: a zone that begins while another is active on the same thread becomes a child. Tracy reconstructs the full nesting tree and displays it as a flame graph in the GUI.

> **Note:** Timestamps are in nanoseconds relative to a per-process epoch. They are not wall-clock UTC times. When correlating Tracy data with device profiler data across separate runs, you must use a common reference event (such as a dispatch sync point) rather than absolute timestamps.

## The Two-Process Model

Tracy uses a client/server architecture to minimize the overhead on the profiled application:

1. **The profiled process (client)** — your application, compiled with Tracy's client library linked in. At runtime, Tracy opens a local TCP socket and streams zone events to the capture server as a compact binary protocol. The client maintains a lock-free ring buffer per thread; background threads drain the buffer and send events without blocking your application's hot path.

2. **The capture server (`tracy-capture`)** — a separate process that receives the event stream, compresses the data, and writes a binary `.tracy` file to disk. Events arrive pre-timestamped: the client timestamps each event before placing it in the ring buffer, so `tracy-capture` only compresses and writes the already-timestamped data. `tracy-capture` can run on the same machine as the profiled process (connected via loopback) or on a remote machine.

The output artifact is a binary `.tracy` file. This file is loaded by the Tracy GUI (`tracy`) for interactive inspection, or processed by Tracy's analysis tools for scripted queries. The binary format is versioned; the client and server must be built from the same Tracy release to avoid a version mismatch (a common failure mode covered in Chapter 2).

```bash
# Start the capture server before launching your workload
tracy-capture -o profile.tracy -f

# In a separate terminal, run your profiled workload (Tracy is activated at build time via TRACY_ENABLE; no runtime env var is needed)
pytest tests/ttnn/my_test.py -s
```

> **Warning:** If you start the profiled process before `tracy-capture` is listening, Tracy will attempt to connect, fail, and drop all events silently. Always start `tracy-capture` first.

## How Tracy Integrates with tt-metal

tt-metal's integration with Tracy is controlled by a single compile-time define: `TRACY_ENABLE`.

When `TRACY_ENABLE` is defined (which happens automatically when you build with `-DENABLE_PROFILER=ON`), the macros in `tt_metal/tools/profiler/tt_metal_tracy.hpp` expand to real Tracy instrumentation calls that record zone entry and exit events. When `TRACY_ENABLE` is absent — the default for production builds — every macro in that header expands to nothing. There is no branch, no function call, no memory touch. The instrumentation is truly zero-cost when disabled.

The key macros you will encounter in tt-metal source:

| Macro | Expands to (when enabled) | Purpose |
|---|---|---|
| `ZoneScoped` | Tracy zone covering the current C++ scope | Records the duration of the enclosing function or block |
| `ZoneScopedN("name")` | Named Tracy zone | Same, but with an explicit string name instead of the function name |
| `TracyMessage(str, len)` | Free-form text message | Attaches a string payload to the current point in time |
| `FrameMark` | Frame boundary marker | Not used for TTNN op timing; appears in some loop-boundary instrumentation |

> **Tip:** You do not need to modify tt-metal source to get useful Tracy data. The default instrumentation already covers the most important host-side spans for op profiling.

## What tt-metal Annotates by Default

With `ENABLE_PROFILER=ON`, tt-metal automatically instruments the following host-side events:

- **Op dispatch calls** — the entry point of each TTNN op's C++ dispatch function is wrapped in a named zone, giving you the total host-side time from Python call to enqueue completion for every op invocation.
- **Program enqueue events** — the point at which a compiled program (kernel binary + runtime arguments) is written into the device command queue. This zone separates program-creation time from the MMIO write latency.
- **Trace lifecycle events** — `begin_trace_capture`, `end_trace_capture`, and `execute_trace` are each wrapped in zones, allowing you to see how long trace capture and replay take at the host level.

These default annotations are sufficient to answer the primary Tracy question: "When did the host dispatch this op, and how long did dispatch take?" You do not need to add your own instrumentation to start profiling TTNN workloads.

> **Note:** If you want finer-grained host-side zones inside a specific op's dispatch path — for example, to isolate program creation time from argument validation time — you can add `ZoneScopedN` macros to the relevant C++ functions. Those changes require a rebuild with `ENABLE_PROFILER=ON`.

---

**Next:** [`two_profilers_compared.md`](./two_profilers_compared.md)
