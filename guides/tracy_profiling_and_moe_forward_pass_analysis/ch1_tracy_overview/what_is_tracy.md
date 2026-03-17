# What Is Tracy?

Tracy is a real-time, low-overhead C++ profiler designed for systems-level performance tooling. It was originally created by Bartosz Taudul for game engine development — a domain where single-millisecond latency budgets are the norm and profiler overhead must be negligible — but its design makes it equally well-suited for any latency-sensitive C++ application. tt-metal uses Tracy to instrument the host-side op dispatch path, giving engineers nanosecond-precision visibility into exactly when each TTNN operation was handed to the device runtime.

Understanding Tracy's architecture before touching build flags or capture commands prevents a class of common mistakes: trying to read a `.tracy` file in a text editor, launching the profiled application without the capture server, or wondering why all Tracy macros appear to be no-ops after a standard tt-metal build.

---

## Origins and Design Philosophy

Tracy emerged from the game engine community's need for a profiler that could:

- Record tens of thousands of events per frame without perceptibly slowing the program being profiled.
- Present results in real time so that a developer could observe a performance regression as it occurred, not after the fact.
- Produce a self-contained binary file that could be shared with a colleague and replayed without the original binary.

These requirements led to three design decisions that distinguish Tracy from sampling profilers (e.g., `perf`, `gprof`, Instruments) and from logging-based tools (e.g., Python's `cProfile`):

1. **Instrumentation, not sampling.** Every event Tracy records must be explicitly marked in source code with a Tracy macro. There is no call-stack unwinding or periodic interruption; only what is explicitly annotated appears in the trace. This makes Tracy accurate for latency-sensitive measurements but requires that the code under analysis be instrumented in advance.

2. **Lock-free ring buffer per thread.** Each thread writes its events to a per-thread ring buffer using lock-free atomic operations. A background Tracy thread drains the buffer and serializes events over a socket. This design keeps the overhead of emitting an event in the sub-100 ns range on modern hardware.

3. **Two-process separation.** The application being profiled and the Tracy server that records and displays the trace are separate processes. The profiled application is a normal production binary (with Tracy instrumentation compiled in); the Tracy server is a separate GUI or command-line binary. This separation means that the cost of storing and analyzing the trace does not affect the process being measured.

---

## What Tracy Records

Tracy records four categories of events, each corresponding to a distinct concept in the Tracy data model:

**CPU Zones** are the primary unit of Tracy instrumentation. A zone is a named, time-bounded scope on a CPU thread. Each zone has:
- A name (string literal, resolved at compile time for zero runtime string allocation cost).
- A start timestamp and an end timestamp, measured in nanoseconds using the platform high-resolution clock.
- The thread ID of the thread that emitted the zone.
- An optional color and an optional 64-bit integer value payload that can carry arbitrary user data (e.g., a tensor shape dimension).

Zones can nest: if zone B starts after zone A starts but before zone A ends, B is a child of A. The Tracy GUI displays nested zones as an indented timeline — the "flame graph" view.

**GPU Zones** (not used directly by tt-metal's Tracy integration) are similar to CPU zones but associated with a GPU command queue timestamp rather than a CPU clock. Tenstorrent device timing is handled separately via the on-device cycle counter profiler rather than through Tracy GPU zones.

**Memory Allocations** are recorded by patching `malloc`/`free` or by explicit annotation. Tracy tracks allocation site, size, and lifetime. tt-metal does not currently surface Tensix DRAM allocation events as Tracy memory events, though host-side C++ allocations can be captured.

**Frame Markers** delimit discrete units of work (frames, in game engine parlance; inference iterations in an ML context). Frame markers are optional but useful for computing per-iteration statistics in the Tracy GUI.

---

## The Tracy Data Model

Every event in a Tracy trace is associated with a zone, and every zone is identified by the following fields:

| Field | Type | Notes |
|---|---|---|
| Zone name | Compile-time string | Interned once; zero runtime allocation after the first use |
| Source location | File, function, line number | Captured by C++ `__FILE__` / `__LINE__` macros at compile time |
| Thread ID | OS thread ID | Identifies which thread recorded the zone |
| Start timestamp | `int64` nanoseconds | Monotonic clock; platform-specific (TSC on x86, `clock_gettime` on Linux) |
| End timestamp | `int64` nanoseconds | Same clock source as start |
| Color | `uint32` RGBA | Optional; used for visual grouping in the Tracy GUI |
| Value | `int64` | Optional user payload; can carry e.g. a tensor batch dimension |

The `.tracy` file is a binary serialization of all accumulated zone records plus the metadata needed to reconstruct source locations, thread names, and frame markers. It is not human-readable and is not a text or CSV file; it must be opened in the Tracy GUI (`tracy-profiler`) or exported to CSV via `tracy-csvexport`.

> **Tip:** Zone names that follow a hierarchical slash convention (e.g., `MoE/dispatch/topk`) will display as a collapsible tree in the Tracy GUI's statistics panel, making it easy to aggregate timing by functional subsystem. This guide adopts that convention for all custom zone annotations added in Chapter 3.

---

## The Two-Process Model

Deploying Tracy requires two processes to be running simultaneously:

**Process 1: the profiled application.** This is a tt-metal or TTNN workload compiled with Tracy instrumentation enabled. It contains the Tracy client library, which manages per-thread ring buffers and a background network thread. When the application starts, the Tracy client attempts to connect to a Tracy server on `localhost:8086` (the default port). **By default, if no server is listening, the Tracy client blocks at program startup and waits until a server connects** — the application will not proceed past initialization until a Tracy server is available. Silent event discarding (non-blocking behavior where the application runs without a server and events are simply dropped) requires building Tracy with the `TRACY_ON_DEMAND` compile-time option defined. When using tt-metal's standard build, be aware of this blocking behavior: always start the Tracy server (`tracy-capture` or `tracy-profiler`) before launching the profiled application, or build with `TRACY_ON_DEMAND` if you need the application to run independently of a server.

**Process 2: the Tracy server.** This is either the Tracy GUI binary (`tracy-profiler`) or the headless capture binary (`tracy-capture`). It listens on `localhost:8086`, accepts the connection from the profiled application, and receives the event stream. The server writes events to a `.tracy` file on disk as they arrive. The capture binary (`tracy-capture`) is the recommended approach for automated profiling pipelines because it produces a `.tracy` file without requiring a display.

```
┌─────────────────────────────────┐        ┌──────────────────────────────────┐
│  Profiled Application           │        │  Tracy Server                    │
│                                 │        │  (tracy-capture or tracy-profiler)│
│  ┌──────────────────────────┐   │  TCP   │                                  │
│  │  Tracy client library    │───┼───────▶│  Receives event stream           │
│  │  (per-thread ring buffer)│   │ :8086  │  Writes to output.tracy          │
│  └──────────────────────────┘   │        │                                  │
│                                 │        │                                  │
│  tt-metal op dispatch           │        │                                  │
│  TTNN Python frontend           │        │                                  │
└─────────────────────────────────┘        └──────────────────────────────────┘
```

> **Warning:** The Tracy client library version compiled into the profiled application must exactly match the Tracy server version. A version mismatch causes the server to reject the connection and log a protocol error. Always build `tracy-capture` and `tracy-profiler` from the same Tracy commit that is used as the tt-metal submodule. Chapter 2 covers how to verify this.

The two-process model has one important consequence: if the profiled application exits before all events have been flushed from the ring buffer to the server, the tail of the trace will be truncated. The environment variable `TRACY_NO_EXIT=1` instructs the Tracy client to hold the process alive after `main()` returns until all buffered events have been sent. Chapter 2 covers this in detail.

---

## Tracy Integration in tt-metal

tt-metal's Tracy integration is controlled by two build-time switches:

- **`TRACY_ENABLE`**: the C preprocessor define that activates all Tracy macros in tt-metal's codebase. When this define is absent, every Tracy macro expands to nothing — zero code is generated, and there is no runtime overhead of any kind.
- **`ENABLE_PROFILER`**: the CMake flag that controls `TRACY_ENABLE` and also enables the on-device cycle-counter profiler. In a standard tt-metal build without `ENABLE_PROFILER=ON`, Tracy instrumentation is fully compiled out. **Important: in the tt-metal build system, `ENABLE_PROFILER=ON` unconditionally activates both Tracy (host-side) and the on-device cycle-counter profiler together.** `TRACY_ENABLE` is not independently toggleable via the standard CMake build interface — it is set as a consequence of `ENABLE_PROFILER=ON`. This means there is no supported build configuration that enables Tracy-only (without the device profiler overhead). If minimizing overhead is a requirement, the only supported option is to disable `ENABLE_PROFILER` entirely, which compiles out both profilers.

The Tracy macros used in tt-metal are wrapped in a tt-metal-specific header:

```cpp
// tt_metal/tools/profiler/tt_metal_tracy.hpp
// (simplified illustration — actual file may differ)
#ifdef TRACY_ENABLE
  #include <tracy/Tracy.hpp>
  #define TracyTTMetalBeginMeshTrace(name)   ZoneScopedN("MeshTrace/begin/" name)
  #define TracyTTMetalEndMeshTrace(name)     ZoneScopedN("MeshTrace/end/" name)
  #define TracyTTMetalReplayMeshTrace(name)  ZoneScopedN("MeshTrace/replay/" name)
  #define TracyTTMetalReleaseMeshTrace(name) ZoneScopedN("MeshTrace/release/" name)
  #define TracyTTMetalEnqueueMeshWorkloadTrace(name) ZoneScopedN("MeshTrace/enqueue/" name)
#else
  #define TracyTTMetalBeginMeshTrace(name)
  #define TracyTTMetalEndMeshTrace(name)
  #define TracyTTMetalReplayMeshTrace(name)
  #define TracyTTMetalReleaseMeshTrace(name)
  #define TracyTTMetalEnqueueMeshWorkloadTrace(name)
#endif
```

> **Warning:** The macro expansions shown above are a simplified illustration of the pattern. The actual `tt_metal_tracy.hpp` may use different macro signatures or underlying Tracy zone types. Always consult the source file in the tt-metal repository you are building from. Never assume that a Tracy zone created via `ZoneScopedN` vs. `ZoneNamedN` vs. `FrameMarkNamed` has the same behavior — they differ in whether the zone scope is tied to the C++ block scope or must be manually opened and closed.

---

## What tt-metal Annotates by Default

When `TRACY_ENABLE` is active, tt-metal instruments the following categories of host-side events with Tracy zones. These zones appear in every Tracy trace produced from a tt-metal workload without requiring any user-side annotation:

**Op dispatch calls.** The entry into the TTNN op dispatch path — the function that validates arguments, selects a kernel program, and enqueues it to the device command queue — is wrapped in a Tracy zone. The zone name encodes the op class (e.g., `ttnn::matmul`, `ttnn::softmax`). This means that a Tracy trace of a TTNN forward pass will contain one zone per dispatched op, with start and end timestamps bracketing the host-side dispatch work for that op.

**Program enqueue events.** The call to `EnqueueProgram` (the tt-metal API that writes a compiled kernel program into the device command queue) is instrumented as a Tracy zone. This is distinct from the TTNN-level op dispatch zone: the TTNN zone includes argument validation and kernel selection overhead; the `EnqueueProgram` zone covers only the command-queue write itself.

**Mesh trace lifecycle events.** When TTNN trace capture is used (via `ttnn.begin_trace_capture` / `ttnn.end_trace_capture` / `ttnn.execute_trace`), the tt-metal layer annotates each stage of the trace lifecycle with a dedicated Tracy macro:

| Macro | When it fires | What it marks |
|---|---|---|
| `TracyTTMetalBeginMeshTrace` | Entry into trace capture mode | Start of the op sequence being recorded |
| `TracyTTMetalEndMeshTrace` | Exit from trace capture mode | End of capture; trace is compiled |
| `TracyTTMetalReplayMeshTrace` | Each call to `ttnn.execute_trace` | Start of one trace replay |
| `TracyTTMetalReleaseMeshTrace` | Trace handle destruction | Trace memory is freed |
| `TracyTTMetalEnqueueMeshWorkloadTrace` | Enqueue of a compiled trace workload to a mesh device | The low-level dispatch of the compiled trace object |

These lifecycle zones are important for MoE profiling because the decode phase of MoE inference commonly uses trace capture to eliminate per-op host dispatch overhead. Understanding where `TracyTTMetalReplayMeshTrace` appears in the Tracy timeline relative to the op-level zones is key to interpreting the 16 ms gap analyzed in Chapter 5.

> **Tip:** In a trace-replay workload, individual op dispatch zones (e.g., `ttnn::matmul`) will not appear during replays — they only appear during the initial capture. The only zone visible during replay is `TracyTTMetalReplayMeshTrace` (plus any user-added zones that were annotated at the replay call site rather than inside the captured region). This is expected behavior, not a profiling failure.

---

## Summary

Tracy is an instrumentation-based, low-overhead C++ profiler that records named CPU zones with nanosecond timestamps. Its two-process architecture separates the profiled application from the recording server; the `.tracy` binary output requires the Tracy GUI or `tracy-csvexport` for analysis. In tt-metal, Tracy is activated by the `TRACY_ENABLE` compile flag and instruments op dispatch, program enqueue, and mesh trace lifecycle events by default. When the flag is absent, all Tracy macros are zero-cost no-ops with no effect on runtime behavior.

---

---

**Next:** [`tracy_vs_device_profiler.md`](./tracy_vs_device_profiler.md)
