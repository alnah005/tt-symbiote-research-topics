# B Review — Chapter 1: Tracy Profiler Overview — Pass 1

1. **`what_is_tracy.md`, line 71** — Default Tracy port stated as `localhost:8086`. The Tracy default listen port is **8086** for the GUI (`tracy-profiler`) but the headless capture binary (`tracy-capture`) also uses 8086 by default; however, the Tracy project's actual default port is **8086 only for the GUI** — the capture binary defaults to the same port. This number is correct in current Tracy releases (v0.9+). No factual error here on the port number itself, BUT the file states the client connects to `localhost:8086` and if no server is listening, "the client silently discards all events." In reality, Tracy's default behavior when no server is present is that the client **blocks at startup** waiting for a server connection (not silently discard). Silent discarding requires compiling with `TRACY_ON_DEMAND` defined. A reader who follows the instruction as written may be surprised when their profiled application hangs waiting for a connection rather than running normally. **Fix:** Note that silent discarding only applies when Tracy is compiled with the `TRACY_ON_DEMAND` option; the default behavior is to block at startup until a server connects.

2. **`what_is_tracy.md`, line 100 / `index.md`, line 55** — The file states that Tracy is activated by the `TRACY_ENABLE` compile flag, and separately says `ENABLE_PROFILER` is the CMake flag that controls `TRACY_ENABLE`. This is presented as a clean two-level mapping. In the actual tt-metal build system, `ENABLE_PROFILER=ON` enables **both** Tracy and the on-device cycle-counter profiler together. A reader who wants only Tracy (host-side) without the device profiler overhead cannot achieve that by reading this chapter — they would need to know whether `TRACY_ENABLE` can be set independently of `ENABLE_PROFILER`. The chapter does not clarify this, which could cause a reader to incorrectly assume the two are always coupled and that enabling Tracy always incurs device-profiler overhead. **Fix:** Explicitly state whether `TRACY_ENABLE` can be passed independently to CMake/the build system to enable only Tracy, or whether `ENABLE_PROFILER=ON` is the only supported path and always enables both simultaneously.

3. **`tracy_vs_device_profiler.md`, line 57 / `index.md`, line 57** — The granularity row states device profiler resolution is "~1 cycle (~1 ns at 1 GHz)." Tenstorrent Tensix cores do not run at 1 GHz; the Wormhole B0 AICLK is nominally **~1 GHz** for the AI clock but this is approximate and the actual frequency used for cycle-to-time conversion matters for correctness. More critically, the parenthetical `~1 ns at 1 GHz` is arithmetically correct but the implied precision is misleading — the actual conversion depends on the runtime device clock frequency reported by the driver, not a fixed 1 GHz. If a reader uses `1 ns/cycle` as a hard conversion constant rather than querying the actual device frequency, their time calculations will be wrong. **Fix:** State that the 1 GHz figure is approximate and that accurate cycle-to-time conversion must use the device's reported AICLK frequency (available from `tt-smi` or from the device profiler post-processing script), not a hardcoded constant.

No further correctness issues found beyond the three above.

# B Review — Chapter 1: Tracy Profiler Overview — Pass 2

All three Pass 1 issues have been resolved:

1. **Pass 1 Issue 1 (blocking behavior) — VERIFIED FIXED.** `what_is_tracy.md` line 71 now correctly states that the Tracy client blocks at startup when no server is present, and that `TRACY_ON_DEMAND` is required for non-blocking silent-discard behavior.

2. **Pass 1 Issue 2 (ENABLE_PROFILER couples both profilers) — VERIFIED FIXED.** `what_is_tracy.md` line 100 and `index.md` line 55 both now explicitly state that `ENABLE_PROFILER=ON` unconditionally activates Tracy and the on-device cycle-counter profiler together, with no supported build path to enable only one of them.

3. **Pass 1 Issue 3 (1 GHz AICLK illustrative only) — VERIFIED FIXED.** `tracy_vs_device_profiler.md` line 21 and `index.md` line 57 both now carry explicit notes that the 1 GHz figure is illustrative, and that accurate cycle-to-nanosecond conversion requires the device's actual AICLK frequency from `tt-smi` or `process_ops_logs.py`.

No new correctness issues were found in Pass 2.

**No feedback — chapter approved.**
