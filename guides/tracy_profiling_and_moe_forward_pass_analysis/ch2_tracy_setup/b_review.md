# B Review — Chapter 2: Setting Up Tracy Profiling — Pass 1

1. **Wrong CMake flag name throughout (`build_flags.md`, `index.md`, `capture_workflow.md`).** The guide consistently instructs readers to build with `-DENABLE_PROFILER=ON`. This flag does not exist in tt-metal. The actual CMake option is `ENABLE_TRACY` (defined in `cmake/project_options.cmake` as `option(ENABLE_TRACY "Enable Tracy Profiling" OFF)`). Every cmake invocation in the guide that uses `ENABLE_PROFILER=ON` will silently have no effect, and the resulting binary will contain no Tracy instrumentation. The redundant `-DTRACY_ENABLE=ON` flag passed alongside it is similarly not a CMake option — `TRACY_ENABLE` is a preprocessor define set automatically by `cmake/tracy.cmake` when `ENABLE_TRACY` is ON.

2. **`TRACY_ON_DEMAND` is a compile-time preprocessor define, not a runtime environment variable (`build_flags.md`, `capture_workflow.md`).** The guide instructs users to set `TRACY_ON_DEMAND=1` as a shell environment variable at process launch time. Tracy does not read this at runtime: there is no `GetEnvVar("TRACY_ON_DEMAND")` call anywhere in the Tracy client source. `TRACY_ON_DEMAND` is a compile-time `#define` that must be set when building the Tracy client library. Setting it as an env var has no effect. By contrast, `TRACY_NO_EXIT=1` is correctly described as a runtime environment variable (Tracy does call `GetEnvVar("TRACY_NO_EXIT")` in `TracyProfiler.cpp`).

3. **Wrong Tracy submodule path (`build_flags.md`).** The guide states the submodule is at `third_party/tracy` (e.g., `git submodule status third_party/tracy`, `cd third_party/tracy`). The actual path, as recorded in `.gitmodules`, is `tt_metal/third_party/tracy`. Commands using the wrong path will fail with "no such path in the working tree."

4. **Wrong build method for `tracy-capture` (`build_flags.md`).** The guide shows building `tracy-capture` via `cmake -B build -S . && cmake --build build --target tracy-capture`. The Tracy submodule used by tt-metal does not expose a `tracy-capture` CMake target — its `CMakeLists.txt` only builds the client library. The tt-metal build system builds the capture tool using its Unix Makefile (`tt_metal/third_party/tracy/capture/build/unix/Makefile`), as shown in `cmake/tracy.cmake`. The correct manual build is `cd tt_metal/third_party/tracy/capture/build/unix && make`, not the cmake invocation shown.

5. **Wrong CSV column names for `tracy-csvexport --unwrap` output (`output_format.md`).** The guide documents the per-zone (unwrap) columns as `ns_start`, `ns_end`, `ns_duration`, and `thread_id`. The actual columns produced by the csvexport source are `ns_since_start`, `exec_time_ns`, and `thread` — there is no `ns_end`, no `ns_duration`, and no `thread_id`. Additionally, `ns_since_start` is time elapsed since the trace start (not an absolute wall-clock timestamp), so the claim that it is anchored to `CLOCK_MONOTONIC` process start is incorrect for this field. The pandas gap-analysis snippet in the same file uses `df["ns_start"]` and `df["ns_end"]` column names that do not exist, so it will raise a `KeyError` at runtime.

## Agent A Change Log — B Feedback Pass 1
- build_flags.md: Replaced ENABLE_PROFILER=ON with ENABLE_TRACY=ON (correct CMake option name)
- build_flags.md: Changed TRACY_ON_DEMAND from "runtime env var" to "compile-time define (-DTRACY_ON_DEMAND=1)"
- build_flags.md: Fixed Tracy submodule path from third_party/tracy to tt_metal/third_party/tracy
- build_flags.md/capture_workflow.md: Fixed tracy-capture build command to use Makefile in tt_metal/third_party/tracy/capture/build/unix/
- output_format.md: Fixed CSV column names to match actual tracy-csvexport output (ns_since_start, exec_time_ns, thread); fixed pandas gap-computation snippet

---

# B Review — Chapter 2: Setting Up Tracy Profiling — Pass 2

Pass 1 fixes verified for `build_flags.md`, `capture_workflow.md`, and `output_format.md`. One residual error remains in `index.md`, which was not updated during Pass 1.

1. **`index.md`, lines 13 and 26 — wrong CMake flag `ENABLE_PROFILER=ON` still present.**
   - Line 13 (Learning Objectives, item 1): `"State the CMake flags needed to enable Tracy and the on-device cycle-counter profiler in a tt-metal build (ENABLE_PROFILER=ON, TRACY_ENABLE=ON)."` — both flag names are wrong. `ENABLE_PROFILER` does not exist; `TRACY_ENABLE` is a preprocessor define, not a CMake option. The correct CMake flag is `ENABLE_TRACY=ON`.
   - Line 26 (Setup Checklist, step 1): `"Build tt-metal with ENABLE_PROFILER=ON and TRACY_ENABLE=ON"` — same error. The correct invocation is `-DENABLE_TRACY=ON` (which sets `TRACY_ENABLE` automatically; the user does not pass it separately).
   - Wrong claim: `ENABLE_PROFILER=ON` and `TRACY_ENABLE=ON` as independent CMake flags.
   - Correct value: `-DENABLE_TRACY=ON` as the single CMake option; `TRACY_ENABLE` is a preprocessor define set automatically by CMake, not a flag the user passes.

## Agent A Change Log — B Feedback Pass 2
- index.md: Fixed ENABLE_PROFILER=ON → ENABLE_TRACY=ON; removed spurious -DTRACY_ENABLE=ON from cmake invocation

---

# B Review — Chapter 2: Setting Up Tracy Profiling — Pass 3

Pass 2 fix verified. `index.md` lines 13 and 26 are correct: `ENABLE_TRACY=ON` is the sole CMake option stated, `ENABLE_PROFILER` is absent, and `TRACY_ENABLE` is not presented as a user-facing CMake flag at either location.

One residual error found in `build_flags.md`.

1. **`build_flags.md`, line 24 — `ENABLE_PROFILER=ON` still present in explanatory prose.** The sentence reads: *"which is the default in a release or debug build without `ENABLE_PROFILER=ON`"*. `ENABLE_PROFILER` does not exist as a CMake option. The correct flag name is `ENABLE_TRACY=ON`. This is the same class of error fixed in `index.md` during Pass 2; it was missed in this file because it appears in a prose sentence rather than a code block or checklist item.

## Agent A Change Log — B Feedback Pass 3
- build_flags.md: Fixed residual ENABLE_PROFILER=ON in prose sentence at ~line 24 → ENABLE_TRACY=ON

---

# B Review — Chapter 2: Setting Up Tracy Profiling — Pass 4

Pass 3 fix verified. `build_flags.md` line 24 now reads "without `ENABLE_TRACY=ON`" — `ENABLE_PROFILER` is absent. Full scan of all four content files (`index.md`, `build_flags.md`, `capture_workflow.md`, `output_format.md`) finds zero remaining occurrences of `ENABLE_PROFILER`.

Pass 3 fix verified. No feedback — chapter approved.
