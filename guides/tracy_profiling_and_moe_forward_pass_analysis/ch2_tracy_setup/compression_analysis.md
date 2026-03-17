# Compression Analysis: ch2_tracy_setup

## Summary

- **Files analyzed:** 4 (`index.md`, `build_flags.md`, `capture_workflow.md`, `output_format.md`)
- **Estimated current line count:** 390 lines (index: 53, build_flags: 97, capture_workflow: 116, output_format: 124)
- **Estimated post-compression line count:** ~345 lines
- **Estimated reduction:** ~11–12%

The chapter is well-structured overall. The largest duplication is a verbatim code block repeated across two files. A secondary duplication is a "Output Artifacts" summary section in `capture_workflow.md` that pre-empts content covered fully in `output_format.md`. Minor redundancies are scattered across "Next Steps" footers and one repeated compile-time vs. runtime clarification.

---

## CRUCIAL Suggestions

### 1. Verbatim build command for `tracy-capture` duplicated in `build_flags.md` and `capture_workflow.md`

**Files:** `build_flags.md` lines 55–60, `capture_workflow.md` lines 95–100

Both files contain the identical shell block:

```bash
cd tt_metal/third_party/tracy/capture/build/unix
make
```

In `build_flags.md` (lines 55–60) this appears under "The Tracy Submodule and Version Pinning" as the primary instruction for building `tracy-capture`. In `capture_workflow.md` (lines 95–100) it reappears verbatim under "Common Failure Mode: Client/Server Version Mismatch" as the fix.

**What should keep it:** `build_flags.md` is the authoritative location for build commands. `capture_workflow.md` should keep only a prose fix instruction and a cross-reference: e.g., "Rebuild `tracy-capture` from the pinned submodule commit (see `build_flags.md` — Tracy Submodule and Version Pinning)." This removes approximately 7 lines from `capture_workflow.md`.

---

### 2. "Output Artifacts" section in `capture_workflow.md` duplicates the opening of `output_format.md`

**Files:** `capture_workflow.md` lines 104–110, `output_format.md` lines 1–9 and lines 68–71

`capture_workflow.md` contains an "Output Artifacts" section (lines 104–110) that describes both `moe_trace.tracy` and `profile_log_device.csv`, including what each file is, its readability, and that it needs post-processing. `output_format.md` opens with the same characterization of these two files and then expands on each in full.

The `capture_workflow.md` section adds no information beyond what `output_format.md` provides; it is a forward-summary that the "Next Steps" link already handles. The section also partially duplicates the cross-reference role of `index.md`'s Chapter Contents table.

**What should keep it:** `output_format.md` is the authoritative location. The "Output Artifacts" section in `capture_workflow.md` (lines 104–110, approximately 7 lines) should be removed. The existing "Next Steps" line at the bottom of `capture_workflow.md` already directs readers to `output_format.md` and is sufficient.

---

## MINOR Suggestions

### 1. "Next Steps" footers in three sub-files duplicate `index.md` navigation

`build_flags.md` (lines 94–96), `capture_workflow.md` (lines 113–115), and `output_format.md` (lines 121–123) each end with a "Next Steps" section pointing to the next file in sequence. `index.md` already defines the reading order in both its "Setup Checklist" (lines 24–31) and its "Reading Order" section (lines 44–47). The footer sections are minor navigational aids that add 3 lines each but are redundant with the chapter index. They are low-risk to remove if the guide is consumed via `index.md` as the entry point; they are harmless to keep if readers may open sub-files directly.

### 2. `TRACY_ON_DEMAND` compile-time clarification repeated in two files

`build_flags.md` (lines 78–90) provides the full explanation that `TRACY_ON_DEMAND` is a compile-time preprocessor define and must not be set as a shell variable at launch. `capture_workflow.md` (lines 50–52, inside the environment variables table) repeats this same clarification inline: "This is a compile-time define, not a runtime env var... Setting it as a shell variable at launch has no effect. See `build_flags.md`."

The table entry in `capture_workflow.md` is technically appropriate as a quick reminder — it prevents a common mistake at the point of use — but the full sentence is longer than necessary. It could be trimmed to: "Compile-time define only; see `build_flags.md`. Setting as a shell variable has no effect." This is a minor tightening rather than a removal.

### 3. Opening prerequisite statement in `capture_workflow.md` restates `index.md` checklist steps

`capture_workflow.md` lines 4–5 ("This file assumes you have already built tt-metal with `ENABLE_TRACY=ON` and built `tracy-capture` from the `tt_metal/third_party/tracy` submodule. If you have not done both, return to `build_flags.md`.") restates what `index.md` already enforces through the Setup Checklist (steps 1–2) and Reading Order. This is a minor 2-line observation and the statement is a useful guard; it could be condensed to a one-line note rather than removed entirely.

---

## Load-Bearing Evidence

The following facts are uniquely present and must not be removed during compression:

1. **`build_flags.md` lines 13–14** — The CMake build system couples Tracy and the Tensix on-device cycle-counter profiler; you cannot enable one without the other through the standard CMake interface. This coupling constraint is non-obvious and not stated elsewhere.

2. **`build_flags.md` lines 22–24** — When `TRACY_ENABLE` is not defined, every Tracy macro compiles to a literal no-op: no function calls, no memory accesses, no branches. This explains why a release build without the flag carries zero profiling overhead and justifies maintaining two separate builds.

3. **`capture_workflow.md` lines 79–81** — `TRACY_NO_EXIT=1` works by inserting a blocking wait into the Tracy client's shutdown path; the process does not exit until the ring buffer is empty and the capture server has acknowledged the final batch. This is the only file that explains the mechanism (not just the symptom).

4. **`output_format.md` lines 34** — There is no `ns_end` column in the `tracy-csvexport` CSV. End time must be computed as `ns_since_start + exec_time_ns`. This is a non-obvious schema fact required to write correct analysis queries.

5. **`output_format.md` lines 95–97** — The `DURATION[ns]` column in `profile_log_device.csv` uses the actual AICLK read from the device at runtime, not a fixed 1 GHz assumption. The accompanying warning ("Do not assume AICLK is 1 GHz") is critical for correctness of any downstream timing arithmetic.

6. **`output_format.md` lines 113–117** — Tracy host timestamps use `CLOCK_MONOTONIC` anchored to process start; device profiler timestamps use hardware cycle counts converted via AICLK. The two clocks are not automatically synchronized; `TT_METAL_DEVICE_PROFILER_DISPATCH_CORES` controls the anchor markers used for cross-tool correlation. This is the only place this synchronization limitation is explained.

7. **`build_flags.md` lines 44–45** — The Tracy client library compiled into `libtt_metal.so` must be the exact same version as `tracy-capture` and `tracy-profiler` at the protocol level; a version mismatch causes the capture server to reject or corrupt the connection immediately. The causal mechanism (protocol-level enforcement) is stated only here.

---

## VERDICT

Crucial updates: yes

## Agent A Change Log — C Feedback Pass 1
- capture_workflow.md: Replaced duplicate tracy-capture build command block with cross-reference to build_flags.md
- capture_workflow.md: Removed "Output Artifacts" section; replaced with one-sentence pointer to output_format.md

---

# Compression Analysis: Chapter 2 — Setting Up Tracy Profiling — Pass 2

## Summary

- **Files analyzed:** 4 (`index.md`, `build_flags.md`, `capture_workflow.md`, `output_format.md`)
- **Actual line counts after Pass 1:** index.md: 53, build_flags.md: 97, capture_workflow.md: 108, output_format.md: 124 — **total: 382 lines**
- **Pass 1 reduction achieved:** 8 lines removed from `capture_workflow.md` (390 → 382; ~2% reduction)
- **Note on estimated vs. actual:** Pass 1 estimated ~45 lines of reduction; actual reduction was 8 lines. The "Output Artifacts" section and build command block together comprised fewer lines than estimated, which was stated as approximately 14 lines combined in the Pass 1 analysis.

### Pass 1 Fix Verification

**Fix 1 — Duplicate `tracy-capture` build command (CONFIRMED APPLIED):** The verbatim shell block (`cd tt_metal/third_party/tracy/capture/build/unix` / `make`) that previously appeared in `capture_workflow.md` under "Common Failure Mode: Client/Server Version Mismatch" has been replaced with a prose cross-reference: "Build `tracy-capture` first as described in `build_flags.md`; the binary will be at `tt_metal/third_party/tracy/capture/build/unix/capture`." The authoritative shell block remains only in `build_flags.md` lines 55–58. No duplication exists.

**Fix 2 — "Output Artifacts" section removed (CONFIRMED APPLIED):** The "Output Artifacts" section that previously occupied `capture_workflow.md` lines 104–110 is gone. In its place, line 101 reads: "For a full description of the `.tracy` binary and `tracy-csvexport` output format, see `output_format.md`." This is the one-sentence pointer called for in the Pass 1 recommendation. The existing "Next Steps" footer at lines 105–107 also continues to direct readers to `output_format.md`.

---

## CRUCIAL Suggestions

None.

Both Pass 1 crucial duplications have been resolved. A survey of the post-fix file set finds no remaining verbatim block duplications, no section-level content pre-emptions, and no cases where one file renders another section redundant in a way that obscures authoritative information. The remaining minor issues identified in Pass 1 are carried forward below and remain the only candidates for further reduction.

---

## MINOR Suggestions

### 1. (Carried from Pass 1) "Next Steps" footers in three sub-files duplicate `index.md` navigation

`build_flags.md` lines 94–96, `capture_workflow.md` lines 105–107, and `output_format.md` lines 121–123 each end with a "Next Steps" section pointing to the next file in reading order. `index.md` already defines this order explicitly in the Setup Checklist (lines 24–30) and the Reading Order section (lines 44–46). The footer sections add 3 lines each (9 lines total across the three files) and are redundant when readers follow the chapter via `index.md`. They are harmless for readers who open sub-files directly, so removal is judgment-dependent. No new information is at risk; this is a pure navigational duplication.

### 2. (Carried from Pass 1) `TRACY_ON_DEMAND` compile-time clarification repeated in two files

`build_flags.md` lines 78–90 provides the full explanation that `TRACY_ON_DEMAND` is a compile-time preprocessor define and must not be set as a shell variable at launch. `capture_workflow.md` line 50 repeats this in the environment variables table: "This is a compile-time define, not a runtime env var. Non-blocking behavior must be compiled in with `-DTRACY_ON_DEMAND=1` at build time. Setting it as a shell variable at launch has no effect. See `build_flags.md`." The table row is technically appropriate as a point-of-use warning preventing a common mistake; the full sentence could be trimmed to: "Compile-time define only; see `build_flags.md`. Setting as a shell variable has no effect." This is a minor tightening, not a removal.

### 3. (Carried from Pass 1) Opening prerequisite statement in `capture_workflow.md` restates `index.md` checklist

`capture_workflow.md` lines 4–5 state: "This file assumes you have already built tt-metal with `ENABLE_TRACY=ON` and built `tracy-capture` from the `tt_metal/third_party/tracy` submodule. If you have not done both, return to `build_flags.md`." This restates what `index.md` already enforces through the Setup Checklist steps 1–2 and the Reading Order. The statement is a useful guard for readers who open the file directly; it could be condensed to a one-line note rather than removed.

---

## Load-Bearing Evidence

The following facts are confirmed present and must not be removed during any further compression pass:

1. **`build_flags.md` lines 13–14** — The CMake build system couples Tracy and the Tensix on-device cycle-counter profiler; enabling one through the standard CMake interface always enables the other. This coupling constraint is non-obvious and stated only here.

2. **`build_flags.md` lines 22–24** — When `TRACY_ENABLE` is not defined, every Tracy macro compiles to a literal no-op with zero runtime cost: no function calls, no memory accesses, no branches. This justifies maintaining two separate builds and is stated only here.

3. **`capture_workflow.md` lines 80–81** — `TRACY_NO_EXIT=1` inserts a blocking wait into the Tracy client's shutdown path; the process does not exit until the ring buffer is empty and the capture server has acknowledged the final batch. This is the only file that explains the mechanism rather than just the symptom.

4. **`output_format.md` line 34** — There is no `ns_end` column in the `tracy-csvexport` CSV; end time must be computed as `ns_since_start + exec_time_ns`. This non-obvious schema fact is required for correct analysis queries and stated only here.

5. **`output_format.md` lines 95–97** — The `DURATION[ns]` column in `profile_log_device.csv` uses the actual AICLK read from the device at runtime, not a fixed 1 GHz assumption. The warning against assuming 1 GHz is critical for downstream timing correctness and stated only here.

6. **`output_format.md` lines 113–117** — Tracy host timestamps use `CLOCK_MONOTONIC` anchored to process start; device profiler timestamps use hardware cycle counts converted via AICLK. The two clocks are not automatically synchronized; `TT_METAL_DEVICE_PROFILER_DISPATCH_CORES` controls anchor markers for cross-tool correlation. This synchronization limitation is explained only here.

7. **`build_flags.md` lines 44–45** — The Tracy client library compiled into `libtt_metal.so` must be the exact same version as `tracy-capture` and `tracy-profiler` at the protocol level; a mismatch causes the capture server to reject or corrupt the connection immediately. The causal mechanism (protocol-level enforcement) is stated only here.

8. **`capture_workflow.md` lines 93–97** — After Pass 1, the fix for the version mismatch failure mode is now expressed as a prose cross-reference to `build_flags.md` rather than a repeated shell block. The cross-reference is load-bearing: it is the only in-context pointer from the failure mode description back to the authoritative build instruction. It must not be further abbreviated or removed.

---

## VERDICT

Crucial updates: no
