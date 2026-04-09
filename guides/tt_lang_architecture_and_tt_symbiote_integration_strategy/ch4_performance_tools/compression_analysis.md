# Chapter 4 Change Log

## 2026-04-09 — auto_profile.md: Fix signpost naming convention documentation

**Problem:** The guide presented `generate_signpost_name()` (`auto_profile.py:116`) as "the canonical form" for signpost naming. This function is dead code — it is defined but never called anywhere in the codebase. The actual signpost names are constructed directly in `ttl_ast.py` by `_emit_line_signpost_if_needed()` (line 232) and `_emit_op_signposts()` (line 264), and they never include the `_C{col}` column suffix that `generate_signpost_name()` would produce.

**Fix:** Replaced the dead-code function listing with documentation of the actual naming convention as implemented in `ttl_ast.py`:
- Line-level: `f"{self.name}_L{file_lineno}"` (e.g., `compute_L52`)
- Op-level: `f"{self.name}_L{file_lineno}_{prefix}{op_name}"` (e.g., `dm_read_L52_cb_wait`)

Added a callout noting that `generate_signpost_name()` exists but is dead code.

**Source files verified:**
- `/localdev/salnahari/testing_dir/tt-lang/python/ttl/_src/auto_profile.py` (line 116 — dead code definition)
- `/localdev/salnahari/testing_dir/tt-lang/python/ttl/_src/ttl_ast.py` (lines 232, 264 — actual construction)
- `grep` confirmed zero callers of `generate_signpost_name` outside its own definition.

---

## 2026-04-09 — Compression Analysis, Pass 1

### Summary

Chapter 4 (index + 3 sub-pages, ~470 lines total) is well-structured but contains repeated explanations of the same mechanisms across files: the CSV flush call `ttnn.ReadDeviceProfiler(device)`, the logs path `$TT_METAL_HOME/generated/profiler/.logs/`, the `ZONE_START`/`ZONE_END` pairing mechanic, and the signpost lowering pipeline are each explained multiple times. There is also a comparison table in `signpost_profile.md` that largely restates information already present in `index.md`'s "Profiling Modes at a Glance" table. Several prose passages use hedging or verbose phrasing that could be tightened.

### CRUCIAL Suggestions

None.

### MINOR Suggestions

1. **Repeated `ttnn.ReadDeviceProfiler(device)` flush explanation (3 files).** `index.md` line 35 explains "Each hook calls `ttnn.ReadDeviceProfiler(device)` to flush profiler data from the device before parsing the CSV." `auto_profile.md` line 52 repeats "Calls `ttnn.ReadDeviceProfiler(device)` to flush profiler data." `signpost_profile.md` line 64 repeats "Calls `ttnn.ReadDeviceProfiler(device)` to flush data." Since the index already establishes this as a universal step, the sub-pages could simply reference the index or omit the detail.

2. **Repeated logs path explanation (3 files).** The path `$TT_METAL_HOME/generated/profiler/.logs/` appears as prose in `index.md` lines 10-11, `auto_profile.md` lines 46 and 53, `signpost_profile.md` line 65, and `perf_dump_and_perfetto.md` line 26. The index prerequisite section already establishes this; sub-pages could use a short reference like "the standard profiler logs directory (see Prerequisites)."

3. **Repeated ZONE_START/ZONE_END pairing explanation (3 files).** The mechanic of matching `ZONE_START`/`ZONE_END` pairs is explained in `index.md` line 74, `auto_profile.md` lines 44 and 91, `signpost_profile.md` lines 55 and 80, and `perf_dump_and_perfetto.md` line 134. The first mention in the index is sufficient; sub-pages could abbreviate to "pairs zones as described in the chapter introduction."

4. **Repeated signpost lowering pipeline (2 files).** `auto_profile.md` lines 42-44 describes the `ttl-lower-signpost-to-emitc` pass converting to `DeviceZoneScopedN` macros. `signpost_profile.md` lines 50-55 repeats the same three-step lowering path. Since the index (lines 59-63) already covers this, the sub-pages could cross-reference rather than re-explain.

5. **Redundant comparison table in signpost_profile.md.** The "Relationship to Auto-Profile" table (lines 128-136) restates what is already conveyed by the "Profiling Modes at a Glance" table in `index.md` (lines 16-21) plus the narrative text. The table adds CB attribution and roofline rows, but these could be folded into a single sentence rather than a full duplicate table.

6. **Verbose prose in perf_dump_and_perfetto.md.** Line 13: "Where auto-profile and signpost profiling focus on cycle counts per code region, perf dump answers questions about memory bandwidth, transfer patterns, and data movement topology." The contrast clause ("Where auto-profile and signpost profiling focus on...") is unnecessary since the reader has already read those sections. Tighten to: "Perf dump provides a hardware-level summary of NOC traffic, memory bandwidth, and data movement topology."

7. **Hedging language in auto_profile.md.** Line 46: "The output path is either `$TT_METAL_HOME/generated/profiler/.logs/cb_flow_graph.json` or the directory of `$TTLANG_PROFILE_CSV`." The "either...or" could be simplified to a primary path with a parenthetical override note.

8. **Over-long code comment in signpost_profile.md.** The `signpost()` docstring (lines 17-27) repeats usage that is immediately demonstrated in the following code block (lines 33-45). The docstring's example (`with ttl.signpost("my_region"): ...`) is redundant with the full kernel example 5 lines later.

### Load-Bearing Evidence

- **index.md** line 35: `"Each hook calls ttnn.ReadDeviceProfiler(device) to flush profiler data from the device before parsing the CSV."`
- **auto_profile.md** line 52: `"Calls ttnn.ReadDeviceProfiler(device) to flush profiler data."`
- **signpost_profile.md** lines 50-55: `"1. AST to MLIR... 2. ttl-lower-signpost-to-emitc... 3. Device execution: The macro calls produce ZONE_START/ZONE_END entries in the device profiler CSV."` (restates index lines 59-63 and auto_profile lines 42-44)
- **perf_dump_and_perfetto.md** line 13: `"Where auto-profile and signpost profiling focus on cycle counts per code region, perf dump answers questions about..."` (verbose contrast clause)

### VERDICT

Crucial updates: no. Eight minor compression opportunities identified, primarily repeated cross-file explanations of shared infrastructure (CSV flush, logs path, zone pairing, signpost lowering). Estimated recoverable lines: ~30-40 if duplicates are replaced with cross-references.
