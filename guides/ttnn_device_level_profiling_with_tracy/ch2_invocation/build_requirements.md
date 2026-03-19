# Build Requirements

## The `ENABLE_PROFILER` CMake Flag

`ENABLE_PROFILER` is the single CMake option that arms both instrumentation layers simultaneously:

1. **Tracy CPU-side instrumentation** — Causes CMake to pass `-DTRACY_ENABLE` to the compiler for all tt-metal translation units that include Tracy headers. Every `ZoneScoped`, `ZoneScopedN`, `FrameMark`, and related macro in the tt-metal source tree activates. Without this flag those macros expand to nothing and the profiled process never opens a Tracy connection.

2. **On-device cycle-counter profiler** (`kernel_profiler.hpp`) — Enables the device-side profiling infrastructure that reads hardware cycle counters on each Tensix core and prepares per-core timing data for upload to the host. This is the data source that ultimately produces `ops_perf_results.csv`. The relevant header lives at `tt_metal/include/tt_metal/tools/profiler/kernel_profiler.hpp` and is compiled into kernel binaries only when `ENABLE_PROFILER=ON`.

Both layers are gated by the same flag because a capture session almost always requires both: Tracy records the host-side dispatch zones while the device profiler records the on-die execution time. Enabling only one half produces an incomplete picture of total op latency. See Ch6 for the full latency decomposition.

## The `TRACY_ENABLE` Preprocessor Define

`TRACY_ENABLE` is a C/C++ preprocessor symbol defined by Tracy's own CMake machinery. You do not set it directly; setting `ENABLE_PROFILER=ON` causes tt-metal's CMake infrastructure to set it for you.

The define controls every Tracy macro in the codebase. From Tracy's headers:

```cpp
#ifndef TRACY_ENABLE
#  define ZoneScoped
#  define ZoneScopedN(x)
#  define FrameMark
   // ... all other macros become empty
#endif
```

When `TRACY_ENABLE` is absent, all Tracy macros expand to empty at the source level — no ring buffer is allocated, no background thread is started, and no port is opened. This is the zero-overhead guarantee Chapter 1 described. Whether the Tracy client library is actually linked into the final binary depends on the specific CMake configuration: some configurations skip the Tracy link entirely when `ENABLE_PROFILER=OFF`, so do not rely on the library being present in that case.

### Verifying `TRACY_ENABLE` Is Active in the Build

**Method 1 — Symbol table check.**

After building, inspect a tt-metal binary that exercises the profiler path:

```bash
nm -C build/tt_metal/libtt_metal.so | grep -i tracy | head -20
```

If `TRACY_ENABLE` was active you will see symbols such as:

```
0000000000000000 T tracy::Profiler::SendCallstack(...)
0000000000000000 T tracy::Zone::Zone(...)
0000000000000000 T tracy::InitRPMalloc()
```

If the output is empty, `TRACY_ENABLE` was not defined and the build does not include Tracy instrumentation.

**Method 2 — `ldd` / shared library check.**

Tracy can be built as a shared library depending on CMake configuration. Run:

```bash
ldd build/tt_metal/libtt_metal.so | grep tracy
```

A match confirms the shared Tracy client is linked. Absence is not conclusive (static linking is common) — fall back to the symbol table method.

**Method 3 — `compile_commands.json` inspection.**

`TRACY_ENABLE` is a compiler preprocessor define, not a CMake BOOL cache variable, so it does not appear in `CMakeCache.txt`. Instead, check the compilation database (requires `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` at configure time):

```bash
grep -m1 "DTRACY_ENABLE" build/compile_commands.json
```

A match confirms that `-DTRACY_ENABLE` was passed to the compiler for at least one translation unit. If `compile_commands.json` is not present, fall back to the symbol table method (Method 1), which is authoritative.

## Example CMake Invocation

A minimal invocation that enables the profiler:

```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DENABLE_PROFILER=ON \
  -DTT_METAL_BUILD_TESTS=ON \
  ..

cmake --build build --target tt_metal ttnn -j$(nproc)
```

`TRACY_ENABLE` is implied by `ENABLE_PROFILER=ON` — you do not pass it explicitly on the cmake command line. The `RelWithDebInfo` build type is strongly preferred for profiling: it preserves debug symbols (needed for Tracy's call stack unwinding) while keeping compiler optimizations enabled (so measured timings reflect production performance).

> **Note:** If you have an existing build directory that was previously configured without `ENABLE_PROFILER=ON`, you must either delete `build/CMakeCache.txt` and reconfigure, or pass `-DENABLE_PROFILER=ON` to a fresh `cmake -B build` invocation. An incremental rebuild that only adds the flag may not re-compile all affected translation units if CMake does not detect the define change as a dependency trigger on your build system version.

## The Tracy Submodule and Version Matching

The Tracy client embedded in tt-metal lives at:

```
tt_metal/third_party/tracy/
```

This is a Git submodule pinned to a specific Tracy release tag. The Tracy capture server (`tracy-capture`) and the Tracy GUI (`tracy`) **must be built from the same submodule commit** as the client that is compiled into tt-metal. Tracy's wire protocol is not stable across versions — a version mismatch causes `tracy-capture` to print a version mismatch error and refuse the connection. The profiled process continues running but receives no connection from the capture server, so no events are recorded and the output `.tracy` file will be empty or absent.

To build the matching capture server:

```bash
cd tt_metal/third_party/tracy/capture/build/unix
make -j$(nproc)
# Produces: tracy-capture binary in the same directory
```

To build the matching GUI (optional, for interactive inspection):

```bash
cd tt_metal/third_party/tracy/profiler/build/unix
make -j$(nproc)
# Produces: tracy binary (the GUI)
```

> **Tip:** Add the directory containing the freshly built `tracy-capture` to your `PATH` so that shell scripts and CI steps always invoke the version-matched binary, not any system-installed Tracy binary.

## Build-Time Warning: Runtime Overhead

> **Warning:** `ENABLE_PROFILER=ON` adds nonzero runtime overhead from two sources: (1) Tracy's lock-free zone recording allocates and writes into a ring buffer on every `ZoneScoped` entry and exit, and (2) the on-device cycle-counter profiler reads hardware performance counters on each Tensix core after every kernel dispatch and DMAs that data back to the host. Both costs are small per operation but accumulate across a long workload. Do **not** use a profiler build for SLA-critical production benchmarks or throughput measurements that will be compared against non-profiler numbers. Always note `ENABLE_PROFILER=ON` in any performance report generated from a profiler build.

## Verifying the Build Artifact Before Attempting a Capture

Run the following checklist before starting any capture session:

1. **Confirm Tracy symbols are present** — use the `nm` command from Method 1 above. If the output is empty, the build does not have `ENABLE_PROFILER=ON`.

2. **Confirm the version-matched `tracy-capture` binary exists:**

```bash
ls -lh tt_metal/third_party/tracy/capture/build/unix/tracy-capture
```

If any of these checks fail, revisit the CMake configuration step before proceeding to environment variable setup and capture.

---

**Next:** [`env_vars_and_flags.md`](./env_vars_and_flags.md)
