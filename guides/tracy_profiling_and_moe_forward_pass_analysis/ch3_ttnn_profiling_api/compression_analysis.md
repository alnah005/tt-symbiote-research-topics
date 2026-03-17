# Compression Analysis: Chapter 3 TTNN Op-Level Profiling API

## Summary
- Files analyzed: `index.md`, `device_profiler_api.md`, `annotating_your_code.md`, `reading_op_timing_output.md`
- Estimated current line count: 714
- Estimated post-compression line count: 570
- Estimated reduction: ~20%

---

## CRUCIAL Suggestions

### C-1: CSV Column Table Duplicated Verbatim Across Two Files

**Files:** `device_profiler_api.md` lines 40–51 and `reading_op_timing_output.md` lines 65–77

Both files contain an 8-column table describing the fields in `ops_perf_results_<timestamp>.csv`. The columns listed — `OP TYPE`, `PROGRAM ID`, `DEVICE ID`, `CORE GRID`, `DEVICE KERNEL START CYCLE`, `DEVICE KERNEL END CYCLE`, `DEVICE KERNEL DURATION [ns]`, `OP TO OP LATENCY [ns]` — are identical in both tables. The descriptions are near-verbatim; the only difference is that `reading_op_timing_output.md` adds a `When to use it` third column. The core content (column names and definitions) is duplicated.

**Canonical home:** `device_profiler_api.md` — it is the dedicated API reference file. The table there should be the single source of truth and expanded to include the `When to use it` column from `reading_op_timing_output.md`.

**Action for `reading_op_timing_output.md`:** Replace the full table (~13 lines) with a cross-reference sentence: *"For a description of each CSV column, see the [CSV column reference in device_profiler_api.md](./device_profiler_api.md#what-the-profiler-records-per-op)."* Then retain only the `DEVICE KERNEL DURATION [ns]` and `OP TO OP LATENCY [ns]` tip callout (line 77), which adds interpretive guidance specific to the analysis context.

**Estimated line savings:** ~12 lines.

---

### C-2: `process_ops_logs.py` Workflow Steps Duplicated Across Two Files

**Files:** `device_profiler_api.md` lines 130–153 and `reading_op_timing_output.md` lines 9–58

`device_profiler_api.md` contains a "Typical workflow" section with four numbered bash steps (enable profiler → run model → post-process → open HTML). `reading_op_timing_output.md` opens with an identical five-step workflow section covering the same four steps, extended by a step to locate the CSV file. Both sections share:

- The same `TT_METAL_DEVICE_PROFILER=1 python run_moe_forward.py` command.
- The same `python tt_metal/tools/profiler/process_ops_logs.py --csv ops_perf_results_<timestamp>.csv` command.
- The same `xdg-open` / `open` HTML report commands (`device_profiler_api.md` lines 150–151, `reading_op_timing_output.md` lines 53–55).
- The same ODS sort tip (`device_profiler_api.md` line 154, `reading_op_timing_output.md` line 58).

**Canonical home:** `reading_op_timing_output.md` — it is the dedicated file for "how to read the output." The full end-to-end workflow, including the CSV-locating step, belongs there.

**Action for `device_profiler_api.md`:** The "Typical workflow" subsection (lines 137–154) should be replaced with a single-line cross-reference: *"For the complete end-to-end workflow from run to report, see [reading_op_timing_output.md](./reading_op_timing_output.md)."* Retain only the "Basic usage" command (line 131–132) and the surrounding paragraph (lines 126–135) as the minimal API documentation for `process_ops_logs.py`.

**Estimated line savings:** ~20 lines from `device_profiler_api.md`.

---

### C-3: Device Profiler Enable Block Duplicated Across Three Files

**Files:** `index.md` lines 62–72, `device_profiler_api.md` lines 14–27, `reading_op_timing_output.md` lines 13–19

All three files show code to set `TT_METAL_DEVICE_PROFILER=1`. In `device_profiler_api.md` this is the full canonical explanation (bash preferred method + Python `os.environ` alternative with the important caveat that it must precede tt-metal import). In `index.md` the same two code blocks appear verbatim in the "Quick Reference" section. In `reading_op_timing_output.md` only the `export` form appears as Step 1 of the workflow.

**Canonical home:** `device_profiler_api.md` — already has the correct caveat and both forms.

**Action for `index.md`:** The Quick Reference "Enabling the device profiler" subsection (lines 60–72) should be replaced with a single cross-reference line. The one-liner bash form may be kept as a true quick-reference snippet (1–2 lines), but the Python `os.environ` block and its comment should be removed to avoid duplicating the canonical form with its warning.

**Action for `reading_op_timing_output.md`:** Step 1 (lines 13–19) can be reduced to a single sentence with a link to `device_profiler_api.md`, since the reader has already passed through that file.

**Estimated line savings:** ~15 lines combined.

---

### C-4: MoE Dispatch Order / Op Names Presented Twice in Different Formats

**Files:** `device_profiler_api.md` lines 67–89 and `reading_op_timing_output.md` lines 81–99

`device_profiler_api.md` contains a Python-call-to-CSV-op-name mapping table (6 rows) followed by a MoE dispatch order sequence (`topk → softmax → all_gather → matmul → silu → matmul → matmul → reduce_scatter/all_reduce`). `reading_op_timing_output.md` contains a filter-term table (9 rows) that lists the same op types (matmul, topk, softmax, silu, all_gather, reduce_scatter, scatter, gather, moe) with short descriptions of what each matches. The op set is the same; only the presentation format differs.

These two tables serve related but distinct purposes: the first maps Python calls to CSV names (lookup); the second lists filter terms for post-processing (action). They are not purely duplicative — both are load-bearing — but the overlap in op names is high enough to justify a consolidation note.

**Recommended action:** Add a cross-reference from the filter-term table in `reading_op_timing_output.md` to the naming-convention table in `device_profiler_api.md`: *"For the mapping from Python call to CSV op name, see [TTNN Op Names in the CSV](./device_profiler_api.md#ttnn-op-names-in-the-csv)."* No content needs to be deleted; only a link added.

**Estimated line savings:** 2–3 lines (the added link replaces no content but prevents readers from needing to look up the connection manually).

---

## MINOR Suggestions

### M-1: `index.md` Quick Reference Previews Content Already Covered in Sub-files

`index.md` lines 59–90 contain a "Quick Reference" section with three code blocks: enabling the profiler, a custom Tracy zone, and the `process_ops_logs.py` command. These are useful as at-a-glance reminders but directly copy syntax that is fully explained in the sub-files. This is acceptable for an index file, but the Python `os.environ` block (lines 63–66) is a partial copy that omits the important warning from `device_profiler_api.md` (the caveat about not hardcoding AICLK, and the no-runtime-toggle rule). A reader who only reads the index may miss the caveats.

**Recommendation:** Keep the bash one-liner and the Tracy zone snippet as true quick-reference items. Remove the Python `os.environ` block from the index and note that the full form with caveats is in `device_profiler_api.md`.

**Estimated line savings:** ~5 lines.

---

### M-2: "Next Steps" Sections Are Navigation Boilerplate With No Technical Content

Each of the four files ends with a "Next Steps" section (index.md line 92–94, device_profiler_api.md lines 176–178, annotating_your_code.md lines 187–189, reading_op_timing_output.md lines 244–253). The first three are single-sentence navigation links. `reading_op_timing_output.md` Next Steps (lines 246–253) is a bulleted summary of what the chapter covered followed by a forward reference to Chapter 4. The bulleted recap partially restates the Learning Objectives from `index.md` lines 13–19.

**Recommendation:** These are acceptable navigational aids and should be kept short. The bulleted summary in `reading_op_timing_output.md` (lines 246–252) could be cut to a single closing sentence plus the Chapter 4 forward reference, saving ~5 lines without losing any technical content.

---

### M-3: Overview Paragraphs in Sub-files Restate Index Content

`device_profiler_api.md` lines 4–7 and `reading_op_timing_output.md` lines 3–6 each open with an Overview paragraph that summarizes what the file covers. These accurately preview the content and are fine as orientation paragraphs. However, both descriptions partially restate the chapter overview in `index.md` lines 4–7 and the Chapter Structure list at lines 50–54. This is low-severity; no action required unless line count is a hard constraint.

---

## Load-Bearing Evidence

The following facts and code fragments must NOT be removed regardless of consolidation actions:

1. **`TT_METAL_DEVICE_PROFILER=1` cannot be toggled at runtime** — `device_profiler_api.md` lines 12–13. This is a correctness constraint that prevents silent data loss. Must stay in the canonical location with the warning.

2. **1000-op auto-flush and `ttnn.ReadDeviceProfiler(device)` call** — `device_profiler_api.md` lines 31–32. This is the only mention of the buffer size limit across all files. Removing it would cause users to silently lose profiling data on long workloads.

3. **AICLK warning: do not hardcode 1 GHz; 5–15% error range** — `device_profiler_api.md` lines 162. Specific, quantitative, and not restated elsewhere. Must be preserved.

4. **`tt-smi | grep AICLK` command and example output (`AICLK: 1202 MHz`)** — `device_profiler_api.md` lines 167–169. The only place a reader learns how to manually check AICLK. Must be preserved.

5. **Program cache warm-up: "at least 2 warm-up iterations"; first-call inflation caveat** — `device_profiler_api.md` lines 100–120. Removing this would cause users to collect invalid profiler data during compilation. The specific advice ("at least 2 warm-up iterations" and the note that MoE models may need more) is load-bearing.

6. **Tracy stub / no-op warning** — `annotating_your_code.md` lines 27–28. The warning that a no-op stub "silently does nothing" and that users must verify capture is receiving data is a correctness check that appears only here.

7. **Tracy zones inside `ttnn.begin_trace_capture` appear only once (during capture) and do NOT appear on replay** — `annotating_your_code.md` lines 146–147. This is a behavioral fact about trace/replay interaction that is not restated anywhere else. Must be preserved.

8. **`all_to_all` is not a confirmed canonical TTNN/CCL op name** — `reading_op_timing_output.md` lines 97 and 113. This is a correctness note preventing users from using a filter term that matches nothing. Must be preserved with the recommendation to use `all_gather` and `reduce_scatter` instead.

9. **Cross-device CSV alignment by `PROGRAM ID`** — `reading_op_timing_output.md` lines 196–197. The only mention of how to align multi-device CSV files. Must be preserved.

10. **Gap interpretation table with specific thresholds (> 5 ms, 1–5 ms, < 1 ms)** — `reading_op_timing_output.md` lines 162–167. Quantitative thresholds for diagnosing overhead sources. Must be preserved.

11. **MoE dispatch order sequence** — `device_profiler_api.md` lines 85–89 (`topk → softmax → all_gather → matmul (gate) → silu → matmul (up) → matmul (down) → reduce_scatter/all_reduce`). This is the only place the expected op dispatch order is listed as a canonical reference. Must be preserved.

12. **`DEVICE KERNEL DURATION [ns]` vs. `OP TO OP LATENCY [ns]` semantic distinction** — `device_profiler_api.md` line 50 and `reading_op_timing_output.md` line 72 both define `OP TO OP LATENCY` as "device-side time between consecutive op boundaries on the device timeline — includes dispatch overhead measured at the device, not a host CPU wall-clock measurement." This precise phrasing must be preserved in whichever file holds the canonical table.

---

## VERDICT: Crucial updates: yes

---

## Agent A Change Log — C Feedback Pass 1

### Fix C-1: Merge CSV column tables; `device_profiler_api.md` is canonical

**File to edit:** `device_profiler_api.md`
- In the table at lines 40–51, add a third column `When to use it` with the values from `reading_op_timing_output.md` lines 65–77.

**File to edit:** `reading_op_timing_output.md`
- Replace the full "Key CSV Columns" table (lines 64–77, ~14 lines) with a single cross-reference sentence plus the `DEVICE KERNEL DURATION` / `OP TO OP LATENCY` tip callout. Saves ~12 lines.

---

### Fix C-2: Move end-to-end workflow to `reading_op_timing_output.md`; trim `device_profiler_api.md`

**File to edit:** `device_profiler_api.md`
- Delete the "Typical workflow" subsection (lines 137–154, ~18 lines).
- Replace with one sentence: *"For the complete end-to-end workflow from run to report, see [Reading and Interpreting Profiler Output](./reading_op_timing_output.md)."*
- Retain the "Basic usage" paragraph (lines 129–135) and the `xdg-open` note already in `reading_op_timing_output.md` is the single copy.
- Saves ~16 lines.

---

### Fix C-3: Remove duplicated enable-profiler blocks from `index.md` and shorten Step 1 in `reading_op_timing_output.md`

**File to edit:** `index.md`
- In the "Quick Reference" section (lines 60–72), remove the Python `os.environ` block (lines 63–67). Keep only the bash one-liner (lines 69–72) as a true quick-reference.
- Add a note: *"Full details and caveats: [Enabling the Device Profiler](./device_profiler_api.md#enabling-the-device-profiler)."*
- Saves ~5 lines.

**File to edit:** `reading_op_timing_output.md`
- Replace Step 1 (lines 13–19) with: *"Enable `TT_METAL_DEVICE_PROFILER=1` in your shell before launching the process (see [device_profiler_api.md](./device_profiler_api.md#enabling-the-device-profiler) for the full options and the no-runtime-toggle caveat)."*
- Saves ~7 lines.

---

### Fix C-4: Add cross-reference from filter-term table to op-naming table

**File to edit:** `reading_op_timing_output.md`
- After the filter-term table (after line 99), add one sentence: *"For the mapping from Python call to CSV op name (e.g., that `ttnn.linear` lowers to `tt::operations::primary::matmul`), see [TTNN Op Names in the CSV](./device_profiler_api.md#ttnn-op-names-in-the-csv)."*
- No lines deleted; ~2 lines added for clarity.

## Agent A Change Log — C Feedback Pass 1
- reading_op_timing_output.md: Replaced duplicate CSV field table (~lines 65-77) with cross-reference to device_profiler_api.md
- device_profiler_api.md: Trimmed duplicate process_ops_logs.py workflow to cross-reference to reading_op_timing_output.md
- index.md + reading_op_timing_output.md: Replaced TT_METAL_DEVICE_PROFILER=1 enable blocks with cross-references to device_profiler_api.md
- reading_op_timing_output.md + device_profiler_api.md: Added cross-reference links between MoE op name sections

---

# Compression Analysis: Chapter 3 TTNN Op-Level Profiling API — Pass 2

## Summary
- Pass 1 fixes: all 4 verified correctly applied
- Current line count after Pass 1: 683 (index.md: 90, device_profiler_api.md: 162, annotating_your_code.md: 190, reading_op_timing_output.md: 241)
- New crucial duplications: none

## CRUCIAL Suggestions

None.

All four Pass 1 fixes were applied correctly:
- C-1: `reading_op_timing_output.md` line 60 has the cross-reference sentence; the full table is gone. `device_profiler_api.md` lines 41–51 remain as canonical.
- C-2: `device_profiler_api.md` lines 128–137 retains only the Basic usage paragraph and command, with a single cross-reference line at 137. The full 5-step workflow lives exclusively in `reading_op_timing_output.md`.
- C-3: `index.md` lines 62–67 retains only the bash one-liner and a cross-reference; the Python `os.environ` block is gone. `reading_op_timing_output.md` Step 1 (lines 13–15) is a single sentence pointing to `device_profiler_api.md`.
- C-4: `reading_op_timing_output.md` line 84 adds the cross-reference from the filter-term table to the op-naming table in `device_profiler_api.md`.

## MINOR Suggestions

### M-carry-1: Intra-file self-duplication in `reading_op_timing_output.md` (carry-forward of M-2 spirit)

The "Complete Worked Example" bash script (lines 192–227) re-runs `TT_METAL_DEVICE_PROFILER=1 python run_moe_forward.py` and `process_ops_logs.py --csv ...`, which also appear in the Step-by-Step Workflow section (lines 19–40 of the same file). This is within one file and serves as a consolidated copy-paste script, so it is acceptable. No action required unless line count is a hard constraint; the worked example's value as a self-contained runnable script justifies the overlap.

### M-carry-2: `moe_filter_terms` appears three times in `reading_op_timing_output.md`

The filter term set appears as (1) a conceptual table (lines 71–81), (2) a Python list in the `is_moe_op` function (lines 97–98), and (3) a second Python list in the worked example script (lines 206–207). All three are in the same file. The two Python lists could be extracted to a shared snippet with a comment, but given they serve different contexts (standalone function vs. inline script), this is low priority.

## Load-Bearing Evidence

All load-bearing items confirmed still present:

1. **1000-op buffer auto-flush limit** — `device_profiler_api.md` line 31: "The device profiler buffer auto-flushes after 1000 ops." PRESENT.
2. **AICLK 5–15% error range** — `device_profiler_api.md` line 145: "Hardcoding 1 GHz can introduce errors of 5–15% in your nanosecond estimates." PRESENT.
3. **2-iteration warm-up minimum** — `device_profiler_api.md` line 120: "Always run at least 2 warm-up iterations before collecting profiler data." PRESENT.
4. **Tracy no-op stub silent-failure warning** — `annotating_your_code.md` lines 27–28: "Zones created against a no-op stub produce no data and no error — they silently do nothing." PRESENT.
5. **Trace-capture/replay zone behavior** — `annotating_your_code.md` lines 146–147: "Tracy zones inside the capture block will appear once (during capture) but will not appear on subsequent replays." PRESENT.
6. **`all_to_all` non-canonical name warning** — `reading_op_timing_output.md` lines 82 and 99–101: "`all_to_all` is not a confirmed canonical TTNN/CCL op name and may match nothing in the CSV." PRESENT in both the filter table note and the code comment.
7. **Gap interpretation thresholds (>5 ms / 1–5 ms / <1 ms)** — `reading_op_timing_output.md` lines 151–154: table with all three threshold bands. PRESENT.

## VERDICT: Crucial updates: no
