# Build Flags for Tracy and the On-Device Profiler

To capture any Tracy trace from a tt-metal workload, the binary must be compiled with profiling support enabled. This is not enabled by default. This file explains the relevant CMake flags, how they interact, how to verify the resulting build, and how to control blocking behavior at runtime.

---

## The `ENABLE_TRACY` Flag

The primary CMake flag is `ENABLE_TRACY`. Setting it to `ON` activates two separate subsystems simultaneously:

- **Tracy** (host-side): the `TRACY_ENABLE` preprocessor define is set, which causes all Tracy zone macros in tt-metal source to expand into real instrumentation calls instead of no-ops.
- **Tensix on-device cycle-counter profiler** (device-side): `kernel_profiler.hpp` is activated, injecting cycle-counter reads into device kernels so that per-core execution times can be recorded.

These two subsystems are coupled by the CMake build system. You cannot enable one without the other through the standard CMake interface. If you need host-side Tracy tracing, you will always also incur the device-side instrumentation overhead, and vice versa.

> **Warning:** `ENABLE_TRACY=ON` adds measurable runtime overhead from the cycle-counter instrumentation injected into device kernels. Do not use a profiler-enabled binary for SLA-critical benchmarks or production latency measurements. Keep a separate non-instrumented build for performance baselines.

---

## The `TRACY_ENABLE` Preprocessor Define

`TRACY_ENABLE` is a preprocessor symbol consumed by the Tracy client library itself. When it is defined, Tracy's C++ macros (`ZoneScoped`, `ZoneScopedN`, and the tt-metal wrapper macros in `tt_metal_tracy.hpp`) expand to real code that allocates zone structs, records timestamps, and enqueues events into Tracy's internal ring buffer for transmission to the capture server.

When `TRACY_ENABLE` is not defined — which is the default in a release or debug build without `ENABLE_TRACY=ON` — every Tracy macro in the codebase expands to nothing. There is literally zero runtime cost: no function calls, no memory accesses, no branches. The preprocessor removes all instrumentation before compilation.

CMake sets `TRACY_ENABLE` automatically when you pass `-DENABLE_TRACY=ON`. You should not need to set it independently in most workflows.

---

## CMake Invocation

```bash
cmake -DENABLE_TRACY=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -B build -S .
cmake --build build --target tt_metal -j$(nproc)
```

`-DCMAKE_BUILD_TYPE=Release` is recommended for profiling. Debug builds introduce artificial overhead from extra bounds checks and disabled inlining that obscures the timing you are trying to measure.

---

## The Tracy Submodule and Version Pinning

tt-metal includes Tracy as a git submodule at `tt_metal/third_party/tracy`. The Tracy client library compiled into `libtt_metal.so` must be the exact same version as the `tracy-capture` and `tracy-profiler` binaries you use to record and view traces. Tracy enforces this at the protocol level: a version mismatch causes the capture server to reject or corrupt the connection immediately.

Check the pinned submodule commit before building your capture tools:

```bash
git submodule status tt_metal/third_party/tracy
```

Build `tracy-capture` from that same submodule source, not from a system package or a separately cloned Tracy repository:

```bash
cd tt_metal/third_party/tracy/capture/build/unix
make
```

The resulting binary is at `tt_metal/third_party/tracy/capture/build/unix/tracy-capture`. Use this binary — not any other installation — for all captures against this tt-metal build.

---

## Verifying the Build

After building, confirm that Tracy symbols are present in the shared library:

```bash
nm build/lib/libtt_metal.so | grep Tracy
```

A profiler-enabled build will show multiple Tracy symbols (e.g., `Tracy::Profiler`, `TracyAlloc`, zone constructors). A non-profiler-enabled build will show no Tracy symbols because the macros compiled away to nothing.

If `nm` returns no output from the `grep`, the build did not pick up `TRACY_ENABLE`. Re-run CMake, confirm the flag is in `build/CMakeCache.txt`, and rebuild.

---

## `TRACY_ON_DEMAND`: Non-Blocking Startup

By default, when a Tracy-instrumented process starts, the Tracy client blocks in its startup routine until a capture server (`tracy-capture`) accepts a TCP connection on port 8086. This ensures no events are lost, but it means the process will hang indefinitely if no server is running.

`TRACY_ON_DEMAND` is a compile-time preprocessor define — it must be set when building, not at runtime. Add `-DTRACY_ON_DEMAND=1` to your CMake flags alongside `ENABLE_TRACY=ON` to compile non-blocking Tracy behavior into the binary. When compiled with `TRACY_ON_DEMAND`, the client does not block at startup. If no server is connected, events are silently discarded. When a server does connect later, recording begins from that point forward.

```bash
cmake -DENABLE_TRACY=ON -DTRACY_ON_DEMAND=1 \
      -DCMAKE_BUILD_TYPE=Release \
      -B build -S .
```

> **Tip:** Use a binary compiled with `TRACY_ON_DEMAND=1` in automated test pipelines or CI environments where a live Tracy server is not guaranteed. For deliberate profiling sessions where you want complete data from process start, omit this define and ensure `tracy-capture` is running before you launch the workload.

---

---

**Next:** [`capture_workflow.md`](./capture_workflow.md)
