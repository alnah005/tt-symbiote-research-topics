## Agent A Change Log — B Review Pass 1
- pm_ideal_and_fpu_util.md: Fixed ambiguous "M × K × K × N" matmul notation to unambiguous "(M×K) × (K×N)" form.
- pm_ideal_and_fpu_util.md: Removed self-contradictory "impossible" assertion about FPU UTIL > 1.0; now accurately describes valid range.
- pm_ideal_and_fpu_util.md: Qualified duration decomposition equation to apply only for compute-bound ops (TRISC1-dominant); added memory-bound case.
- csv_column_definitions.md: Replaced sfpu_clock_rate() API call with safe clock-frequency guidance via device_params.json or arch-based lookup.

## Agent A Change Log — B Review Pass 2
- pm_ideal_and_fpu_util.md: Fixed inverted memory-bound causality — corrected to: NCRISC is too slow to keep TRISC1 fed; TRISC1 stalls waiting for tile data.
- csv_column_definitions.md: Replaced "sum of RISC durations" with "maximum RISC duration" — RISCs run concurrently, not sequentially.

## Agent A Change Log — B Review Pass 3
- pm_ideal_and_fpu_util.md: Fixed tile-unit FLOPs multiplier from 1024 to 32768 (32^3 MACs per tile-triplet = 32768 MACs = 65536 FLOPs).
- pm_ideal_and_fpu_util.md: Fixed stall_cycles formula to TRISC1_cycles × (1 − FPU_UTIL); was incorrectly equating cycle delta to dimensionless fraction.

---

# Compression Analysis: Chapter 3 — Reading the ops_perf_results CSV — Pass 1

## Summary
- Total files analyzed: 3
- Estimated current line count: ~352 lines
- Estimated post-compression line count: ~295 lines
- Estimated reduction: ~16%

## CRUCIAL Suggestions

### index.md ~lines 28–53
**Issue:** The "Quick-Reference Column Cheat Sheet" table (lines 28–53) duplicates information already present in the complete column reference table in `csv_column_definitions.md`. Every column name and its one-liner description is restated there verbatim or near-verbatim (e.g., `FPU UTIL` → "Fraction of peak FPU throughput actually achieved (0.0–1.0)" appears identically in both files). The cheat-sheet provides no additional information not already accessible one file-link away.
**Suggestion:** Replace the full cheat-sheet table with a two-sentence pointer: "For a quick overview of all columns, see the Complete Column Reference in `csv_column_definitions.md`. The table there covers every column with units and a description." This removes ~20 lines of pure duplication. Alternatively, keep only the 4–5 least-obvious columns (e.g., `PM IDEAL`, `FPU UTIL`, `NOC BW UTIL`) in a shortened "non-obvious columns" table.

### csv_column_definitions.md ~lines 95–117
**Issue:** The cycle-to-nanosecond conversion section (lines 95–117) restates the 1 GHz = 1 cycle/ns equivalence three times: once in the formula block (line 101), once in the simplified form (lines 103–105), and once as a prose summary (lines 107). The two-option clock-lookup note (lines 109–115) then repeats "1 GHz" and "1 cycle = 1 ns" a fourth time. The Warning callout (lines 117) is the only non-redundant sentence in the section.
**Suggestion:** Collapse the three restatements of the 1 GHz equivalence into one sentence. Keep the formula, strike the "That is, on Wormhole B0…" prose restatement (lines 107), and fold the two clock-lookup options into a single note. Estimated savings: ~8 lines.

### pm_ideal_and_fpu_util.md ~lines 105–148
**Issue:** The "Relationship Between PM IDEAL and Actual Duration" section (lines 105–148) restates the compute-bound vs. memory-bound distinction already established in lines 13–14 of the same file and in `csv_column_definitions.md` lines 137. The final Note callout (lines 148) fully restates the memory-bound interpretation already covered in the FPU UTIL table (lines 80–87) and the preceding tip callout (line 87).
**Suggestion:** Cut the final Note callout (lines 148) entirely — it is a word-for-word restatement of the memory-bound interpretation already given in the FPU UTIL table and tip. Trim the compute-bound/memory-bound decomposition prose (lines 126–144) to a single annotated code block showing both cases, removing the repeated sentence "Applying the TRISC1 formula to a memory-bound op produces a meaningless or negative result" since the code block makes this self-evident. Estimated savings: ~12 lines.

## MINOR Suggestions

### csv_column_definitions.md ~lines 29–33
**Issue:** The Warning and Tip callouts after the script invocation block (lines 31–33) both explain what happens if you skip post-processing, but this is already implied by the section heading "Running the Post-Processing Script." The Tip callout ("Re-running is idempotent") is a trivial operational note that does not aid understanding of the CSV columns.
**Suggestion:** Drop the Tip callout (line 33). The Warning is worth keeping because it prevents a concrete mistake; the Tip is low-value reassurance.

### pm_ideal_and_fpu_util.md ~lines 91–101
**Issue:** "Why FPU UTIL Can Exceed 1.0" (lines 91–101) ends with a Warning callout (lines 101) that identifies values above 1.05 as a bug signal. The preceding prose already established 1.0–1.05 as the valid range and named the two causes. The Warning repeats the 1.05 threshold and lists the same two root causes (model parameters, log aggregation) already named in the prose.
**Suggestion:** Fold the Warning into the closing sentence of the prose: "Values substantially above 1.05 indicate a model or tooling bug — check FPU peak parameters and `process_ops_logs.py` aggregation logic." Remove the separate Warning block. Saves ~3 lines and removes the repetition.

### index.md ~lines 14–20
**Issue:** The Prerequisites section (lines 14–20) contains a three-bullet list under "Chapter 2" that itemizes exactly what Chapter 2 covered. This is backstory, not a prerequisite the reader must act on before proceeding. Two of the three bullets (artifact names, script name) are restated in the very next section's table and in `csv_column_definitions.md` line 13.
**Suggestion:** Collapse to a single line: "**Chapter 2** — familiarity with `TT_METAL_DEVICE_PROFILER=1`, `process_ops_logs.py`, and the output artifacts in `tt_metal/tools/profiler/logs/`." Saves 3 lines and removes the redundant artifact enumeration.

## Load-Bearing Evidence
- `csv_column_definitions.md` line ~66: `"FPU UTIL | fraction (0.0–1.0) | Ratio of actual FPU throughput to peak FPU throughput: PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles."` — load-bearing because this is the only place in the chapter where the exact arithmetic definition of FPU UTIL is given as a column-table entry; the derivation section in `pm_ideal_and_fpu_util.md` expands on it but the table entry is the canonical one-line reference.
- `csv_column_definitions.md` line ~129: `"DEVICE KERNEL DURATION = max(all core end timestamps) − min(all core start timestamps)"` — load-bearing because this is the only place the aggregation formula is stated explicitly; it cannot be inferred from the prose description alone.
- `pm_ideal_and_fpu_util.md` line ~10: `"PM_IDEAL_cycles = max(compute_cycles, memory_cycles)"` — load-bearing because this roofline reduction formula is the definitional equation for PM IDEAL; removing or paraphrasing it would strip the quantitative grounding from all downstream interpretation.
- `pm_ideal_and_fpu_util.md` line ~40: the FPU peak ops/cycle table (`BFLOAT16 / HiFi4 → 256`, etc.) — load-bearing because these hardware constants are required to manually verify PM IDEAL values from the CSV; they appear nowhere else in the chapter.

## VERDICT
- Crucial updates: yes

## Agent A Change Log — C Compression Pass 1
- index.md: Replaced full 19-column cheat sheet with 4-column quick-reference + pointer to csv_column_definitions.md.
- csv_column_definitions.md: Removed three redundant restatements of 1 GHz = 1 ns/cycle; kept one clear formula.
- pm_ideal_and_fpu_util.md: Removed final Note callout (word-for-word restatement of FPU UTIL table); trimmed decomposition prose to new content only.

---

# Compression Analysis: Chapter 3 — Reading the ops_perf_results CSV — Pass 2

## Summary
- Total files analyzed: 3
- Estimated current line count: ~322 lines (index.md ~42, csv_column_definitions.md ~134, pm_ideal_and_fpu_util.md ~146)
- Estimated post-compression line count: ~313 lines
- Estimated reduction: ~3%

## CRUCIAL Suggestions
none — all Pass 1 items resolved

Pass 1 items confirmed fixed:
1. index.md cheat sheet — now a 4-row quick-reference table plus a pointer to `csv_column_definitions.md`. No duplication of the complete column reference remains.
2. csv_column_definitions.md — "1 GHz = 1 ns/cycle" now stated exactly once (line 101, inline with the formula). Not repeated elsewhere in that file.
3. pm_ideal_and_fpu_util.md — the trailing Note callout that word-for-word restated the FPU UTIL table has been removed. File ends cleanly at the "Next:" link.

## MINOR Suggestions

### pm_ideal_and_fpu_util.md ~lines 122–138
**Issue:** The formula `stall_cycles = TRISC1_KERNEL_DURATION_cycles × (1 − FPU_UTIL)` appears twice in the same section. It is introduced at line 123 as part of the two-part gap breakdown, and then printed again verbatim at line 138 after the compute-bound code block, introduced with "The stall cycles attributable to FPU inefficiency are:". The second occurrence adds no new context — the variable names and semantics are identical.
**Suggestion:** Delete the second appearance of the formula (lines 137–139: the sentence introducing it plus the code block). The first occurrence at line 123 is already inside the enumerated list where it is first defined; readers do not need it restated four lines later.

### pm_ideal_and_fpu_util.md ~line 16
**Issue:** "On Wormhole B0 at 1 GHz, the two columns are numerically identical (one cycle = one nanosecond)." repeats the same 1 GHz = 1 ns/cycle fact that `csv_column_definitions.md` line 101 already states, and which was the subject of a Pass 1 fix in that file. The cross-file restatement is minor but adds nothing for a reader who reads the files in order.
**Suggestion:** Replace with a forward pointer: "See `csv_column_definitions.md` for the cycle-to-nanosecond conversion." (~1 line saved, removes the only remaining cross-file 1 GHz restatement).

### csv_column_definitions.md ~lines 125–127
**Issue:** The third bullet in the "Key implications" list ("The five RISC processors on a single core run concurrently. Their durations do not add; instead, the longest one among them determines that core's contribution to the device kernel duration.") restates the same point made in the second bullet ("A large gap between DEVICE KERNEL DURATION and the maximum RISC duration on one core indicates…"). Both bullets are expressing that RISC processors overlap and the longest one governs. The third bullet is the stronger, clearer statement; the second bullet's phrasing is weaker and partially redundant.
**Suggestion:** Merge the two bullets. Keep the third bullet's direct statement about concurrency; fold the load-balancing observation (currently in bullet 2) into it as a subordinate clause. Net saving: ~2 lines.

### csv_column_definitions.md ~line 129 vs. pm_ideal_and_fpu_util.md ~line 87
**Issue:** The Tip callout in `csv_column_definitions.md` (line 129: "compare TRISC1 KERNEL DURATION vs. NCRISC KERNEL DURATION to determine compute-bound vs. memory-bound") and the Tip callout in `pm_ideal_and_fpu_util.md` (line 87: "When FPU UTIL is low but NCRISC KERNEL DURATION is high, the op is memory-bound") convey the same diagnostic heuristic. A reader working through both files will see the same decision rule twice.
**Suggestion:** Remove the Tip from `csv_column_definitions.md` line 129 (the shallower statement of the two) and leave the richer version in `pm_ideal_and_fpu_util.md` where it is contextually motivated by the FPU UTIL interpretation section. Saves ~2 lines.

## Load-Bearing Evidence
- `csv_column_definitions.md` line ~101: "For **Wormhole B0** at the nominal **1 GHz** clock, this simplifies to `duration_ns = cycles` (one cycle = one nanosecond)" — load-bearing because this is now the single authoritative statement of the Wormhole B0 clock simplification across all three files; removing it would leave the conversion formula with no worked example.
- `pm_ideal_and_fpu_util.md` line ~10: `"PM_IDEAL_cycles = max(compute_cycles, memory_cycles)"` — load-bearing because this is the definitional equation for the entire PM IDEAL concept; every downstream formula in the file derives from it.
- `pm_ideal_and_fpu_util.md` lines ~40–46: the FPU peak ops/cycle table (BFLOAT16/HiFi4 → 256, etc.) — load-bearing because these hardware constants are required to manually verify any PM IDEAL value; they appear nowhere else in the chapter.
- `csv_column_definitions.md` line ~119: `"DEVICE KERNEL DURATION = max(all core end timestamps) − min(all core start timestamps)"` — load-bearing because this is the only explicit aggregation formula in the chapter; all load-balancing reasoning in the surrounding prose depends on it.

## VERDICT
- Crucial updates: no
