## Pass 7

**No feedback — chapter approved.**

All issues raised in Passes 1–6 have been resolved in the current file content. Every prior correctness flag was verified against the live text:

- Tracy GUI "relay mode" invention (Pass 5 item 1): removed; the mutual-exclusion description is now accurate.
- conftest.py warning now explicitly names both `TT_METAL_CLEAR_L1` and `TT_METAL_DEVICE_PROFILER` as initialization-time variables (Pass 4 item 2): fixed.
- `sync_overhead` mis-characterization (Pass 4 item 3): sentence removed from the current content.
- CMake cache grep (Pass 1 item 1): replaced with `compile_commands.json` grep for `-DTRACY_ENABLE`, which is the correct artifact.
- Python path-discovery command (Pass 1 item 2): removed; only the `${TT_METAL_HOME:-.}` pattern is used.
- Interaction table omission of tracy-capture pre-start requirement (Pass 1 item 3): note added.
- "library always linked" overclaim (Pass 1 item 4): now correctly qualified with "some configurations skip the Tracy link entirely".
- CSV path contradiction across files (Pass 2 item 1): unified to `tt_metal/tools/profiler/logs/ops_perf_results.csv`.
- `tracy-capture --version` invalid flag (Pass 2 item 2): replaced with correct instruction to read the startup banner.
- index.md missing navigation footer (Pass 2 item 3 / Pass 3 item 2): `**Previous:**` link now present.
- Contradictory version-mismatch failure visibility (Pass 4 item 1): both files now describe the failure as a visible "Protocol mismatch" message.

No new correctness issues found in the current content of any of the four files.

---

## Pass 6

**No feedback — chapter approved.**

All issues raised in Passes 1–5 have been resolved in the current file content. One candidate new issue was examined and assessed below for completeness.

**Candidate: `build_requirements.md` lines 37 vs. 130 — inconsistent `libtt_metal.so` path between Method 1 and the pre-capture checklist.**
Method 1 (line 37) uses `nm -C build/tt_metal/libtt_metal.so`; the pre-capture verification checklist (line 130) uses `nm build/lib/libtt_metal.so`. One of these paths will fail on any given build. However, both commands include `2>/dev/null` or equivalent context, and both already note "the exact library name may vary by build configuration; adjust the path if your build places it elsewhere." Because the file explicitly warns the reader to adjust the path, a reader who encounters an empty result is directed to re-examine the path — they are not materially misled into a wrong conclusion. The conceptual instruction (use `nm` and grep for `tracy::` symbols) is correct in both places. This does not rise to criterion (b) or (c) as written. No flag raised.

---

## Pass 5

1. **capture_workflow.md line 11 — Factually wrong claim about Tracy GUI "relay mode" through `tracy-capture`.**
   Line 11 states the Tracy GUI "can connect to the same `tracy-capture` server in relay mode, giving you a live view of incoming zones while the test runs." Tracy does not have a relay mode where the GUI connects through `tracy-capture`. The profiled process accepts exactly one incoming TCP connection; either `tracy-capture` connects (for batch capture) or the GUI connects directly (for interactive viewing), but not both simultaneously — which is what the immediately following note (line 13) also says. The "relay mode" description invents a workflow that does not exist in Tracy's architecture. A reader who follows this instruction will start both `tracy-capture` and the GUI, find that only the first one to connect succeeds, and be confused when the other is silently refused. Fix: replace line 11 with the correct description — the GUI connects directly to the profiled process as an alternative to `tracy-capture`, not through it.

2. **capture_workflow.md line 267 — Broken "Next" footer link to non-existent `ch3_csv_reference/index.md`.**
   The footer `**Next:** [Chapter 3 — Reading the ops_perf_results CSV](../ch3_csv_reference/index.md)` links to a directory that does not exist. This was flagged in Pass 3 and remains unresolved. A reader who clicks it gets a 404. Fix: remove the hyperlink and replace with a plain-text forward reference, e.g. `**Next:** Chapter 3 — Reading the ops_perf_results CSV (coming soon)`, until the chapter is written.

3. **build_requirements.md line 127 — Hardcoded `blank_kernel.o` path does not correspond to actual tt-metal build output locations.**
   The verification command `nm build/tt_metal/kernels/compute/blank_kernel.o 2>/dev/null | grep profiler` references a path that is not where tt-metal's CMake build system places compiled kernel objects. Kernel `.o` files are placed in CMake-internal directories under `build/` (e.g., `build/tt_metal/CMakeFiles/...`) — not at `build/tt_metal/kernels/compute/`. On a correctly configured profiler build this command will always fail with "no such file or directory" (silently, because of `2>/dev/null`), producing empty output. A reader interpreting empty output as "profiler not compiled in" will incorrectly conclude their build is broken and attempt an unnecessary reconfigure. Fix: remove this command from the checklist (it is unreliable regardless of build state), or replace it with a check against an artifact that is reliably present — for example, checking whether the Tracy symbol count from the Method 1 `nm` command on `libtt_metal.so` is non-zero, which is already listed as the authoritative check.

## Pass 4

1. **build_requirements.md line 96 vs. capture_workflow.md line 108 — contradictory descriptions of version-mismatch failure visibility.**
   `build_requirements.md` line 96 states that a Tracy client/server version mismatch causes the server to "silently accept the TCP connection and then immediately close it." `capture_workflow.md` line 108 (Failure Mode 1) describes the same scenario with an explicit visible symptom: `tracy-capture` prints "Connection dropped" or "Protocol mismatch". A reader cannot know whether to watch for a printed error or a silent failure. If the failure is truly silent (as `build_requirements.md` claims), a reader following the `capture_workflow.md` diagnostic loop — waiting for an error message that never comes — will waste significant time. Fix: reconcile to a single description. Tracy's wire protocol handshake does emit a "Protocol mismatch" or equivalent message on the server side in most versions; if that is the actual behavior, remove the word "silently" from `build_requirements.md` line 96 and align both files on the visible-error description.

2. **env_vars_and_flags.md lines 139–142 — conftest.py warning names only `TT_METAL_CLEAR_L1` but the same initialization-time read issue applies to `TT_METAL_DEVICE_PROFILER`.**
   The warning states: "For environment variables that are read once at device initialization time (such as `TT_METAL_CLEAR_L1`), the fixture approach works correctly only if the device is initialized after the fixture runs." The parenthetical example `TT_METAL_CLEAR_L1` is illustrative but incomplete. `TT_METAL_DEVICE_PROFILER=1` is equally — arguably more critically — read at device initialization: if the device is already open when the fixture sets it, the CSV profiler is never activated and the test produces no CSV output with no error message. A reader who sees only `TT_METAL_CLEAR_L1` called out will assume `TT_METAL_DEVICE_PROFILER` is safe to set after device open, leading to a silently empty CSV. Fix: change the parenthetical to list both variables, e.g. "(such as `TT_METAL_CLEAR_L1` and `TT_METAL_DEVICE_PROFILER`)".

3. **env_vars_and_flags.md line 85 — states `sync_overhead` "becomes non-zero only when `TT_METAL_PROFILER_SYNC=1` is set", which is factually incorrect.**
   The sentence reads: "The `sync_overhead` term in `total_op_latency = host_dispatch_time + device_kernel_time + sync_overhead` becomes non-zero only when `TT_METAL_PROFILER_SYNC=1` is set." Sync overhead is a real hardware latency that exists in every dispatched op regardless of the flag. The flag does not cause sync overhead to appear; it serializes the dispatch pipeline so that the overhead becomes attributable to individual ops rather than being buried in pipeline overlap. Telling a reader that `sync_overhead` is zero when the flag is unset is factually wrong and could lead them to conclude their unsynchronized per-op numbers are complete (criterion a). The qualifying sentence that follows ("but `host_dispatch_time` and `device_kernel_time` measured individually may not add up to a meaningful per-op total because of pipeline overlap") partially clarifies, but the initial claim remains a wrong numerical statement. Fix: change "becomes non-zero only when `TT_METAL_PROFILER_SYNC=1` is set" to something like "is only explicitly measurable when `TT_METAL_PROFILER_SYNC=1` is set; without the flag, sync overhead is real but absorbed into unmeasured pipeline overlap".

## Pass 1

1. **build_requirements.md, line 66 — CMake cache grep will always return empty, misleading the reader into thinking the build is broken.**
   The command `grep -i tracy build/CMakeCache.txt | grep -i enable` is presented as a way to confirm that `TRACY_ENABLE` is active. However, `TRACY_ENABLE` is a C preprocessor define passed via `-DTRACY_ENABLE` compiler flags; it is not a CMake `BOOL` cache variable and does not appear as `TRACY_ENABLE:BOOL=ON` in `CMakeCache.txt`. On a correctly configured profiler build this grep returns nothing, which the text implies means the build is wrong. Fix: replace with a grep against `CMAKE_CXX_FLAGS` or `COMPILE_DEFINITIONS` where the define actually appears, e.g. `grep "TRACY_ENABLE" build/CMakeCache.txt` targeting `CMAKE_CXX_FLAGS_RELWITHDEBINFO` or similar, or remove Method 3 and direct the reader to rely on the symbol-table check (Method 1) as the authoritative verification.

2. **capture_workflow.md, lines 86–88 — CSV path discovery command produces a silently wrong path.**
   The snippet `python3 -c "import tt_metal; print(tt_metal.__file__)"` returns the filesystem path of the `tt_metal` Python package file (e.g. `/path/to/site-packages/tt_metal/__init__.py`), not the repository root. Appending `/tools/profiler/logs/ops_perf_results.csv` to that string creates an invalid path (it includes `__init__.py` as a directory component). A reader who follows the "if the path above doesn't resolve, use the default relative path" fallback is safe, but the primary command as written will silently produce a wrong path and `ls` will fail with a confusing error. Fix: remove the python3 path-discovery attempt entirely and use only the direct relative path `tt_metal/tools/profiler/logs/ops_perf_results.csv`, or replace it with a repo-root detection approach (e.g. `git rev-parse --show-toplevel`).

3. **env_vars_and_flags.md, lines 57–63 — Interaction table omits a required condition for Tracy-only capture.**
   The table row "Tracy capture only" lists `TRACY_NO_EXIT=1` as the only required variable and notes the build requirement parenthetically. This is incomplete in a way that causes implementation errors: Tracy is activated at runtime by the profiled process attempting to open a TCP connection to the capture server; if `tracy-capture` is not already running, the Tracy client silently skips connecting and no `.tracy` file is produced. The table does not mention that `tracy-capture` must be started before the profiled process, which is the most common failure mode for Tracy-only capture. A reader consulting only this table will set `TRACY_NO_EXIT=1`, run pytest, and get no Tracy output with no error message. Fix: add a note to the Tracy-only row stating that `tracy-capture` must be running before the profiled process starts, or add a footnote referencing capture_workflow.md step 3.

4. **build_requirements.md, line 28 — Incorrect zero-overhead claim for the unbuilt case.**
   The text states that when `TRACY_ENABLE` is absent, "the Tracy client library is still linked (the submodule is always compiled) but it is fully inert — no ring buffer is allocated, no background thread is started, and no port is opened." This contradicts the earlier accurate statement in line 7 that "Without this flag those macros expand to nothing and the profiled process never opens a Tracy connection." The claim that the library is "still linked" in all configurations is not universally true — some CMake configurations of tt-metal skip linking Tracy entirely when `ENABLE_PROFILER=OFF`. A reader could incorrectly conclude that even a non-profiler build carries the Tracy shared library as a dependency, which affects their reasoning about production binary sizes and attack surface. Fix: qualify the statement to "depending on CMake configuration, the Tracy client may or may not be linked" and direct the reader to use `ldd` (Method 2) to determine the actual link state for their build.

## Pass 3

1. **capture_workflow.md, line 267 — "Next" footer links to a chapter that does not exist.**
   The footer `**Next:** [Chapter 3 — Reading the ops_perf_results CSV](../ch3_csv_reference/index.md)` references `../ch3_csv_reference/index.md`. No `ch3_csv_reference/` directory exists anywhere under `tt-symbiote-research-topics/guides/ttnn_device_level_profiling_with_tracy/`. A reader who clicks this link gets a 404. Fix: either remove the "Next" footer until Chapter 3 is written, or replace it with a placeholder that names the target file without a hyperlink.

2. **index.md — Navigation footer from Pass 2 remains unresolved.**
   Pass 2 flagged that `index.md` has no navigation footer. The file still ends with the inline `**Start here:** [\`build_requirements.md\`](./build_requirements.md)` line and carries no `**Previous:**` link back to the Chapter 1 index. Every content file in the chapter has a `**Next:**` footer; `index.md` is the only entry point that leaves the reader with no backward navigation. Fix: add a `---` footer section with `**Previous:** [Chapter 1](../ch1_tracy_fundamentals/index.md)` (or equivalent) before or after the "Start here" line.

## Pass 2

1. **capture_workflow.md, lines 88 vs. 101 — CSV path is contradictory within the same file.**
   The shell snippet on line 88 sets `CSV_PATH` to `${TT_METAL_HOME:-.}/generated/profiler/.logs/ops_perf_results.csv` (note: `generated/profiler/.logs/`), but the Output Artifacts table on line 101 lists the path as `tt_metal/tools/profiler/logs/ops_perf_results.csv`. The minimal working example in the same file (line 178) also uses `tt_metal/tools/profiler/logs/`. These three references cannot all be correct simultaneously. A reader who follows the shell snippet will look in a different directory than the one the artifacts table describes, causing silent verification failure. Fix: resolve to a single authoritative path and use it consistently across all three references in the file.

2. **capture_workflow.md, line 125 — `tracy-capture --version` is not a valid flag in Tracy's Unix build.**
   The "confirm versions match" step instructs the reader to run `./tracy-capture --version` and expect it to print a version tag. Tracy's `capture` binary (built from the Unix Makefile in `capture/build/unix`) does not accept a `--version` argument — it will either exit with an error or print a usage/help message, not a version string. The actual version is visible in the startup banner when `tracy-capture` is launched normally (`Tracy Profiler capture - 0.10.0`). This makes the version-confirmation step unreliable and potentially confusing. Fix: replace `./tracy-capture --version` with the instruction to start `./tracy-capture` without arguments and read the version from the first line of its startup output, or cross-reference the submodule tag directly via `git -C tt_metal/third_party/tracy describe --tags`.

3. **index.md — Missing navigation footer.**
   Every content file in this chapter (`build_requirements.md`, `env_vars_and_flags.md`, `capture_workflow.md`) ends with a `**Next:**` footer linking to the following file. `index.md` has no footer at all. A reader who lands on the index after finishing Chapter 1 has no navigational link to proceed. Fix: add a footer to `index.md` linking forward to `build_requirements.md` (Step 1 of the checklist) and/or back to the Chapter 1 index.
