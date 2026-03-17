# B Review — Chapter 3: TTNN Op-Level Profiling API — Pass 1

## Summary

Four files reviewed: `index.md`, `device_profiler_api.md`, `annotating_your_code.md`, `reading_op_timing_output.md`.

---

## Factual Errors

### Error 1 — Wrong output CSV filename (device_profiler_api.md, reading_op_timing_output.md)

**What the files state:** The device profiler writes to `profile_log_device.csv`.

**What is correct:** The canonical output filename is `ops_perf_results_<timestamp>.csv`. The name `profile_log_device.csv` does not match the specified authoritative fact. This incorrect filename appears consistently throughout `device_profiler_api.md` (lines 29, 134, 145, 148) and `reading_op_timing_output.md` (multiple steps). It also appears in the quick-reference section of `index.md`. Any reader following these instructions will look for the wrong file.

---

### Error 2 — Wrong description of `OP TO OP LATENCY [ns]` (device_profiler_api.md line 48, reading_op_timing_output.md line 72)

**What the files state:** `OP TO OP LATENCY [ns]` is described as "Time from the end of the previous op to the start of this op — includes host dispatch overhead."

**What is correct:** `OP TO OP LATENCY [ns]` is the device-side time between op boundaries; it includes dispatch overhead but is measured on the device side, not as a generic host-to-host interval. Characterizing it purely as a host-measured inter-op gap misrepresents what the column captures. The authoritative fact specifies it as "device-side time between op boundaries, includes dispatch overhead." The files' framing conflates device-side boundary time with host-side Python latency in a way that will lead readers to misinterpret large `OP TO OP LATENCY` values.

---

### Error 3 — `ttnn.ReadDeviceProfiler` auto-flush behavior not mentioned; ring-buffer behavior implicitly excluded but never stated (device_profiler_api.md)

**What the files state:** The section on the device profiler never mentions `ttnn.ReadDeviceProfiler(device)`, its auto-flush threshold, or the fact that it reads and resets profiler state. The file describes the profiler as simply appending rows to a CSV as ops run, with no discussion of the 1000-op auto-flush boundary.

**What is correct:** The authoritative facts specify that `ttnn.ReadDeviceProfiler(device)` auto-flushes at 1000 ops (it is not a ring buffer / silent overwrite) and that calling it reads and resets state. Omitting this is a material factual gap: a reader profiling a long forward pass with many ops could silently lose data if they do not understand the 1000-op flush boundary and the need to call `ttnn.ReadDeviceProfiler` explicitly.

---

### Error 4 — Capture/replay Tracy zone warning is internally contradictory and contradicts authoritative fact (annotating_your_code.md lines 146, 172)

**What the files state:** The section states two contradictory things:
1. (line 146) "Tracy zones inside the capture block will appear once (during capture) but will not appear on subsequent replays" — this is the correct behavior and matches the authoritative fact.
2. (line 172, the Warning box) "Zones inside `ttnn.begin_trace_capture` / `ttnn.end_trace_capture` blocks execute on every replay if the captured Python code is re-run."

**What is correct:** The authoritative fact is clear: annotate `ttnn.execute_cached_trace()`, not the capture site, because zones during capture repeat on every replay. The Warning box on line 172 contradicts the body text immediately above it and also contradicts the authoritative fact. The Warning implies zones inside capture blocks could execute on every replay "if the captured Python code is re-run," which confuses the trace-capture path with a plain Python loop. The correct and consistent statement is: zones inside the capture block appear only once (during capture) and do not repeat on replay — annotate the `execute_cached_trace()` call site instead.

---

### Error 5 — `scatter` is absent from MoE op filter lists; `all_to_all` is added without authoritative basis (reading_op_timing_output.md lines 90, 107)

**What the files state:** The MoE op filter table (lines 87–95) lists `all_to_all` as a filter term alongside `all_gather`. The Python filter list (line 107) does not include `scatter`. The authoritative MoE op name list is: `matmul`, `topk`, `softmax`, `silu`, `all_gather`, `gather`, `scatter`.

**What is correct:** `scatter` is a named MoE op that must appear in the filter list; it is absent from both the table and the Python code. `all_to_all` is not in the authoritative op name list and should not be presented as a known MoE CSV op name. A reader filtering the CSV using the code as written will miss all `scatter` rows and may add a spurious `all_to_all` filter that matches nothing.

---

## Agent A Change Log — B Feedback Pass 1
- device_profiler_api.md + reading_op_timing_output.md: Fixed CSV filename from profile_log_device.csv to ops_perf_results_<timestamp>.csv
- device_profiler_api.md + reading_op_timing_output.md: Corrected OP TO OP LATENCY description to device-side between-op boundary time
- device_profiler_api.md: Added ReadDeviceProfiler auto-flush at 1000 ops note
- annotating_your_code.md: Removed contradictory Warning box; kept correct body text about capture/replay zone behavior
- reading_op_timing_output.md: Added scatter to MoE op filter list; added version caveat for all_to_all/CCL op names

---

# B Review — Chapter 3: TTNN Op-Level Profiling API — Pass 2

## Pass 1 Fix Verification

All five Pass 1 fixes were applied correctly in `device_profiler_api.md`, `annotating_your_code.md`, and `reading_op_timing_output.md`.

- Fix 1 (CSV filename): Confirmed corrected to `ops_perf_results_<timestamp>.csv` throughout `device_profiler_api.md` and `reading_op_timing_output.md`.
- Fix 2 (OP TO OP LATENCY): Confirmed corrected to device-side between-op boundary description in both files.
- Fix 3 (auto-flush): Confirmed added to `device_profiler_api.md` with correct 1000-op threshold and read-and-reset behavior.
- Fix 4 (capture/replay contradiction): Confirmed contradictory Warning box removed; correct body text retained in `annotating_your_code.md`.
- Fix 5 (scatter/all_to_all): Confirmed `scatter` added to both the table and the Python filter list in `reading_op_timing_output.md`; `all_to_all` addressed with a caveat.

## Remaining Error Found

### Error 6 — `index.md` was not updated for the CSV filename fix (index.md lines 41, 87)

**What the file states:**

- Line 41 (Three Approaches table, "Device profiler CSV" row): `profile_log_device.csv, processed HTML/ODS report`
- Line 87 (Quick Reference bash command): `python tt_metal/tools/profiler/process_ops_logs.py --csv profile_log_device.csv`

**What is correct:** The canonical output filename is `ops_perf_results_<timestamp>.csv`. Pass 1 identified `index.md` as one of the files containing this error (the Pass 1 Error 1 write-up explicitly names "the quick-reference section of `index.md`"), but the fix was applied only to `device_profiler_api.md` and `reading_op_timing_output.md`. Both surviving occurrences in `index.md` still show the wrong filename. A reader who only consults the quick-reference in `index.md` will look for the wrong file.

## Agent A Change Log — B Feedback Pass 2
- index.md line 41: Fix CSV filename in Three Approaches table from `profile_log_device.csv` to `ops_perf_results_<timestamp>.csv`
- index.md line 87: Fix CSV filename in Quick Reference bash command from `--csv profile_log_device.csv` to `--csv ops_perf_results_<timestamp>.csv`

## Agent A Change Log — B Feedback Pass 2
- index.md: Fixed 2 remaining occurrences of profile_log_device.csv → ops_perf_results_<timestamp>.csv (Three Approaches table ~line 41, Quick Reference command ~line 87)

---

# B Review — Chapter 3: TTNN Op-Level Profiling API — Pass 3

Pass 2 fix verified. Both occurrences of `profile_log_device.csv` in `index.md` (line 41 Three Approaches table, line 87 Quick Reference bash command) are replaced with `ops_perf_results_<timestamp>.csv`. Zero occurrences of `profile_log_device.csv` remain across all four files.

All other key facts confirmed correct across all four files:
- `DEVICE KERNEL DURATION [ns]`: correctly described as on-device hardware execution time from hardware cycle counters.
- `OP TO OP LATENCY [ns]`: correctly described as device-side time between consecutive op boundaries, not host CPU wall-clock, in both `device_profiler_api.md` and `reading_op_timing_output.md`.
- `ttnn.ReadDeviceProfiler(device)` auto-flush at 1000 ops with read-and-reset behavior: correctly documented in `device_profiler_api.md`.
- Tracy capture/replay zone behavior: contradictory Warning box removed; body text correctly states zones inside the capture block appear once (during capture) and do not repeat on replay; `execute_cached_trace()` is correctly identified as the annotation site.
- MoE op filter: all required terms (`matmul`, `topk`, `softmax`, `silu`, `all_gather`, `gather`, `scatter`) present in both the filter table and the Python filter list in `reading_op_timing_output.md`; `all_to_all` addressed with an appropriate caveat.

No feedback — chapter approved.
