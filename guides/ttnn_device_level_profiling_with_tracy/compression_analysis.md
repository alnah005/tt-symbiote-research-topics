# Compression Analysis: Cross-Chapter Final Pass — Pass 2

## Summary
- Total files analyzed: 23
- Estimated current line count: ~2890 lines
- Estimated post-compression line count: ~2810 lines
- Estimated reduction: ~3%

## CRUCIAL Suggestions
None — all CRUCIAL items from Pass 1 have been resolved.

## MINOR Suggestions

### [ch2/capture_workflow.md] ~lines 3–15 (Three Components / mutually-exclusive note)
**Issue:** Lines 3–9 of `capture_workflow.md` begin with "A complete profiling session involves two processes" and then enumerate three items (profiled process, `tracy-capture`, Tracy GUI). The "OR" separator between items 2 and 3 is rendered in bold and as an indented line, which is visually inconsistent. More importantly, the note on lines 14–15 ("tracy-capture and the Tracy GUI are mutually exclusive") restates a constraint already implied by "accepts exactly one receiver" in line 7. The mutually-exclusive sentence is therefore partially duplicated within the same section, across only 8 lines.
**Suggestion:** Remove the standalone mutually-exclusive note block (lines 14–15) and fold it into the existing "accepts exactly one receiver" sentence in item 1 (line 7): "The Tracy client opens a single outbound TCP connection and accepts exactly one receiver — either `tracy-capture` (batch) or the Tracy GUI (live), but not both." This eliminates the orphaned note without losing the warning. Saves ~3 lines.

### [ch4/roofline_model_primer.md] ~lines 91–96 (FPU throughput paragraph doubled)
**Issue:** Lines 93–94 state the FPU ceiling as "128 BF16 FMA operations per cycle = 256 FLOPs/cycle" and then lines 96–97 re-derive the identical figure via the tile-triplet calculation (32×32×32 MACs → 65,536 FLOPs → 256 FLOPs/cycle). The paragraph beginning "For matrix operations" on line 96 is a second derivation of the same 256 FLOPs/cycle number that was already stated as a flat assertion on line 93. The derivation adds the intermediate steps (tile-triplet MAC count, cycles per tile-triplet), but the final figure is not new.
**Suggestion:** Merge lines 93–96 into a single paragraph: assert the 128 FMA/cycle figure, then provide the tile-triplet derivation once as the justification. Remove the lead sentence of line 96 ("For matrix operations the hardware operates on 32×32 element tiles") since that context is already established by the paragraph heading. Saves ~3 lines.

### [ch5/causes_of_low_fpu_util.md] ~lines 69–77 (Cause 3 mechanism — hedged FPU UTIL drop language)
**Issue:** The Cause 3 mechanism explanation (lines 75–77) contains a nested hedge: "A drop in FPU UTIL attributable to fidelity only occurs when the extra overhead from higher fidelity ... exceeds the proportional increase that PM IDEAL already models — for example, if HiFi4 introduces micro-architectural stalls on top of the 4× iteration count that PM IDEAL does not capture." This edge-case qualification occupies three lines and describes a condition the guide elsewhere never measures or diagnoses, making it dead weight. The adjacent observable-effect bullet already captures the actionable version ("a significant FPU UTIL drop relative to a LoFi baseline indicates overhead beyond what PM IDEAL models").
**Suggestion:** Remove the nested "A drop in FPU UTIL attributable to fidelity only occurs when..." sentence from the mechanism paragraph. The observable-effect section already states the same condition more precisely. Saves ~3 lines.

### [ch6/what_is_dispatch_overhead.md] ~lines 88–108 (small-op dispatch example)
**Issue:** The worked numeric example in lines 92–108 for `ttnn.add` on a [32, 4096] tensor gives stage-by-stage estimates (~0.3 µs kernel, ~6 µs dispatch, ~0.5 µs sync), then writes the sum equation, then states "More than 88% of the op's observed latency is `host_dispatch_time`." The 88% figure is just `6/6.8` rounded; any reader can verify it from the numbers already shown. The closing sentence adds no diagnostic information beyond what the equation already provides.
**Suggestion:** Remove the closing "More than 88% of the op's observed latency is `host_dispatch_time`." sentence. The equation itself communicates the dominant term; the percentage restatement is redundant. Saves ~1 line.

### [ch3/csv_column_definitions.md] ~lines 113–130 (DEVICE KERNEL DURATION section — BRISC paragraph)
**Issue:** Lines 120–121 state that "each core runs five independent RISC processors (BRISC, NCRISC, TRISC0, TRISC1, TRISC2) in parallel" in the context of the `DEVICE KERNEL DURATION` vs. individual RISC durations section. The BRISC enumeration here is accurate, but the following row entry for `BRISC KERNEL DURATION [ns]` in the duration column table (line 54) already explains BRISC is not present in the CSV. The text at line 120 therefore contradicts the reader's natural understanding (if five RISCs run, why are only four in the CSV?) without immediately resolving the tension. The table row for `BRISC KERNEL DURATION [ns]` resolves it, but a reader who reaches line 120 via the section heading will not have read that table entry.
**Suggestion:** In line 120–121, replace "five independent RISC processors (BRISC, NCRISC, TRISC0, TRISC1, TRISC2)" with "four profiled RISC processors (NCRISC, TRISC0, TRISC1, TRISC2) — BRISC is excluded from per-cycle profiling and does not appear in the CSV." This avoids the apparent contradiction without adding a new explanation block. Saves ~1 line (the inline clarification replaces the mismatch that otherwise forces a parenthetical or footnote). Net cost: 0 lines added, confusion removed.

### [guide-level index.md] ~lines 35–44 (Quick Reference table — one-liner mismatch with Ch3 cheat sheet)
**Issue:** Pass 1 flagged this table and suggested condensing it to env-var and tool entries not covered in Ch3's cheat sheet. The Ch3 index cheat sheet (ch3/index.md lines 33–37) carries four column definitions. The guide-level index.md Quick Reference table (lines 35–44) also carries those same four definitions in slightly different prose. Pass 1's CRUCIAL item on this was classified as MINOR, but the fix was not applied — the table still has all four column one-liners. The overlap is still present.
**Suggestion:** Replace the four column rows (`DEVICE KERNEL DURATION`, `FPU UTIL`, `PM IDEAL`, `NOC BW UTIL`) in the guide-level Quick Reference with a single row: "CSV column definitions — see [Ch 3 Quick-Reference Cheat Sheet](ch3_csv_reference/index.md)". Keep the four non-column rows (`TT_METAL_DEVICE_PROFILER`, `TT_METAL_PROFILER_SYNC`, `tracy-capture`, `ttnn.execute_trace`), which do not appear in the Ch3 cheat sheet. Saves ~4 lines.

## Load-Bearing Evidence

- `ch1/what_is_tracy.md` line ~42: "The Tracy client opens a local TCP socket and streams zone events to the capture server as a compact binary protocol. The client maintains a lock-free ring buffer per thread; background threads drain the buffer and send events without blocking your application's hot path." — Load-bearing because this is the only place in the guide that explains the client-side ring-buffer architecture, which is the reason `TRACY_NO_EXIT=1` is required; removing or shortening this explanation would leave the env-var requirement without a causal justification.
- `ch2/build_requirements.md` line ~96: "The Tracy client embedded in tt-metal lives at: `tt_metal/third_party/tracy/` ... must be built from the same submodule commit" — Load-bearing because the version-mismatch failure mode is the most common capture blocker; the submodule path and the must-match requirement are actionable facts not stated elsewhere in the guide.
- `ch3/csv_column_definitions.md` lines ~113–130 (DEVICE KERNEL DURATION vs. individual RISC durations section) — Load-bearing because the three bullet implications (≥ any RISC duration; load-imbalance gap; RISCs run concurrently, durations do not add) are the prerequisite for interpreting every multi-RISC comparison in Ch4 and Ch5; collapsing this section would leave those chapters without their foundation.
- `ch4/classification_method.md` lines ~130–169 (Decision Flowchart ASCII diagram) — Load-bearing because the flowchart integrates all five classification steps into a single navigable path and is not representable as prose without destroying its utility as a reference; it is also the only place that explicitly handles the "medium FPU UTIL + medium NOC BW UTIL → Balanced" branch.
- `ch5/csv_signatures.md` lines ~9–21 (Diagnostic Checklist ordering) — Load-bearing because the recommended rule-out order (Cause 6 first, Cause 7 last) encodes the key diagnostic efficiency insight — the ordering is not derivable from the individual cause descriptions alone, and reversing it would cause practitioners to spend time on kernel-level investigation before ruling out the one-line `DATA FORMAT` check.
- `ch6/eliminating_dispatch_overhead.md` lines ~62–101 (Constraints section: fixed shapes, no host state modification, pre-allocated output tensors) — Load-bearing because each of the three constraints is accompanied by a specific correctness failure mode (stale runtime arguments → silent wrong output; `ttnn.to_torch()` inside trace → wrong results; implicit allocation inside trace → re-allocation on replay) that is not mentioned elsewhere; removing any constraint block would leave a silent correctness hazard uncovered.
- `ch6/measuring_dispatch_vs_kernel.md` lines ~96–117 (Worked Data Table and dispatch/kernel crossover analysis) — Load-bearing because the four-row measurement table is the only place in the guide that provides concrete observed numbers (dispatch: 6–12 µs; kernel: 0.3–200 µs) anchoring the ~100 µs crossover rule-of-thumb; the crossover figure is referenced in the Ch6 index overview and is meaningless without the empirical grounding in this table.

## VERDICT
- Crucial updates: no

---

# Compression Analysis: Cross-Chapter Final Pass — Pass 1

## Summary
- Total files analyzed: 23
- Estimated current line count: ~2952 lines
- Estimated post-compression line count: ~2500 lines
- Estimated reduction: ~15%

## CRUCIAL Suggestions

### [ch3/csv_column_definitions.md + ch6/measuring_dispatch_vs_kernel.md] ~lines 113–131 and ~lines 28–48
**Issue:** `DEVICE KERNEL DURATION` is defined with its formula (`max(all core end timestamps) − min(all core start timestamps)`) in three separate files: `csv_column_definitions.md` lines 116–128, `measuring_dispatch_vs_kernel.md` lines 28–34, and `pm_ideal_and_fpu_util.md` lines 108–117. The full definitional paragraph is restated nearly verbatim in each location, including the same bullet list of implications (always ≥ RISC durations, load-imbalance gap, parallel RISCs do not add).
**Suggestion:** In `measuring_dispatch_vs_kernel.md` and `pm_ideal_and_fpu_util.md`, replace the full restated definition with a single sentence and a cross-reference pointer: "See [Ch 3 — csv_column_definitions.md] for the full definition." Retain only the formula line that is immediately used in the local context. Saves ~25 lines across two files.

### [ch4/roofline_model_primer.md + ch3/pm_ideal_and_fpu_util.md] ~lines 87–126 and ~lines 24–66
**Issue:** The Wormhole B0 hardware ceilings (FPU: 128 FMA ops/cycle = 256 FLOPs/cycle; NoC: 32 bytes/cycle; ridge point: 8.0 FLOPs/byte) and the FPU peak throughput derivation are spelled out in full in both files. `roofline_model_primer.md` lines 91–125 and `pm_ideal_and_fpu_util.md` lines 38–47 both enumerate the hardware FPU rate, cite the 256 FLOPs/cycle figure, and explain what lower math-fidelity modes do to that figure.
**Suggestion:** In `pm_ideal_and_fpu_util.md`, replace the repeated hardware-ceiling prose and the FMA-rate explanation with a single reference sentence ("Hardware ceilings and the ridge-point derivation are in [roofline_model_primer.md].") and keep only the fidelity lookup table, which is unique to that file. Saves ~15 lines.

### [ch2/index.md + ch2/capture_workflow.md] Five-step checklist duplicated
**Issue:** `ch2/index.md` lines 30–45 contains a five-step setup checklist (Build, env vars, start tracy-capture, run pytest, verify artifacts) that restates exactly the narrative procedure spelled out step-by-step in `capture_workflow.md` lines 17–96 using the same steps, same subprocess, same `-s` flag rationale, and same verification commands. The index checklist is not a genuine summary — it is a shorter duplicate.
**Suggestion:** Condense the five-step checklist in `ch2/index.md` to three bullets (configure, capture, verify) with cross-reference links to the detailed sections, removing the re-explanation of `-o`, `-f`, and `-s` flags that are already covered in full within `capture_workflow.md`. Saves ~10 lines.

### [ch5/causes_of_low_fpu_util.md + ch5/csv_signatures.md] Cause descriptions partially restated in signatures file
**Issue:** For each of the seven causes, `csv_signatures.md` re-describes the mechanism alongside the CSV pattern. For Causes 1, 4, 5, and 7, the mechanism paragraph in `csv_signatures.md` is between 3–6 lines and repeats information that is covered at depth in `causes_of_low_fpu_util.md`. Examples: Cause 1 "distinguishing from other causes" block (lines 59–64 of csv_signatures.md) restates the same TRISC1/NOC observations already enumerated in causes_of_low_fpu_util.md lines 20–23; Cause 5 "verification step" (lines 143–145 of csv_signatures.md) restates the AI > 8.0 ridge-point check from causes_of_low_fpu_util.md lines 143–148.
**Suggestion:** In `csv_signatures.md`, reduce each "distinguishing from other causes" block to the single diagnostic condition that is unique to each cause (i.e., the CSV predicate only), and cut the mechanism re-explanation. Add a one-line pointer to the corresponding section in `causes_of_low_fpu_util.md` for context. Saves ~30 lines.

### [ch6/what_is_dispatch_overhead.md + ch1/two_profilers_compared.md + ch6/index.md] Latency decomposition equation restated four times
**Issue:** The equation `total_op_latency = host_dispatch_time + device_kernel_time + sync_overhead` appears at:
- `two_profilers_compared.md` line 26 (with bullet definitions of all three terms, ~7 lines)
- `what_is_dispatch_overhead.md` line 103 (with inline numeric example, ~3 lines)
- `ch6/index.md` line 14 (as the chapter's central recap, ~2 lines)
- `ch2/index.md` line 26 (as a prerequisite item, ~1 line)
Each appearance re-defines the same three terms. The definitional block in `two_profilers_compared.md` is the canonical home.
**Suggestion:** In `what_is_dispatch_overhead.md`, `ch6/index.md`, and `ch2/index.md`, replace the full three-term re-definition with the equation alone (no bullet re-definitions) and a cross-reference to `two_profilers_compared.md`. The numeric example in `what_is_dispatch_overhead.md` line 103–108 is unique and should be kept. Saves ~12 lines.

## MINOR Suggestions

### [ch1/what_is_tracy.md] ~lines 1–6
**Issue:** The opening "Origins" section (lines 1–6) provides background on Tracy's game-engine history. It is informative but does not affect any diagnostic procedure in the guide. It is the only section across all files that is purely historical context with no actionable consequence.
**Suggestion:** Condense the Origins section from 6 prose lines to 2: "Tracy is a low-overhead C++ profiler by Bartosz Taudul, originally designed for game engines. Its always-on, zero-cost-when-disabled instrumentation model makes it suitable for ML framework profiling." Remove the elaboration on graphics drivers, language runtimes, and ML framework kernels. Saves ~4 lines.

### [ch2/env_vars_and_flags.md] ~lines 86–143 (pytest-specific section)
**Issue:** The `conftest.py` fixture code block (lines 101–138) includes 30 lines of Python to demonstrate restoring environment variables after a test. The multi-line env-var save/restore loop is a general Python pattern; only the fact that `TT_METAL_DEVICE_PROFILER` must precede device init is guide-specific. The warning note at line 139 is load-bearing; the fixture body is not.
**Suggestion:** Replace the full fixture body with a 5-line abbreviated version showing just the `os.environ` assignments, keeping the warning note. The env-var restoration idiom is standard Python and does not need to be shown in full. Saves ~15 lines.

### [ch4/worked_examples.md] ~lines 7–62 (Example 1 verbosity)
**Issue:** Example 1 (Large Matmul) has a Step 3 paragraph (lines 38–41) that explains "The NoC is reading 50 MB of operand data, but the FPU is consuming it at 256 FLOPs/cycle, meaning the read bandwidth requirement per FLOP is very low." This quantitative reasoning duplicates the arithmetic-intensity analysis in Step 1 of the same example and adds no new information.
**Suggestion:** In Step 3, replace the explanatory paragraph with a single sentence: "AI ≈ 683 FLOPs/byte is far above the ridge point, confirming the NoC is lightly loaded." Saves ~3 lines.

### [ch3/csv_column_definitions.md] ~lines 93–110 (cycle-to-ns conversion section)
**Issue:** The "Cycle-to-Nanosecond Conversion" section (lines 93–110) contains two nested approaches for finding clock frequency, plus a warning about non-WH-B0 devices. The first sub-approach ("Preferred — read device_params.json") restates `device_params.json` as the source of clock frequency, which is already stated in `pm_ideal_and_fpu_util.md` line 16. The warning at line 109 is a third place where non-WH-B0 users are warned.
**Suggestion:** Merge the two approaches into one: remove the numbered approach sub-structure and keep only: "Check `device_params.json` alongside the CSV for the `aiclk` field; do not call `device.sfpu_clock_rate()` as it may not exist in all tt-metal versions." The non-WH-B0 warning can be reduced to one sentence. Saves ~8 lines.

### [ch6/eliminating_dispatch_overhead.md] ~lines 130–143 ("When NOT to Use" section)
**Issue:** The "When NOT to Use Trace Capture" section (lines 130–143) contains four separately headed scenarios, each restating a constraint already mentioned in the constraints section (lines 62–101) just above. "During model development" is a restatement of the shape-change constraint; "For ops requiring shape-dependent branching" is identical to the warning on line 67–68; "one-shot or low-repetition workloads" is a corollary of the capture overhead described at lines 32–34.
**Suggestion:** Replace the four-scenario breakdown with a compact 3-item bullet list in a single paragraph. Keep "correctness validation requires intermediate tensor reads" as a distinct item since it is not covered elsewhere in the constraints section. Saves ~8 lines.

### [ch5/remediation_levers.md] ~lines 46–66 (math_fidelity table)
**Issue:** The math fidelity relative-throughput table in `remediation_levers.md` lines 63–65 lists the same FMA-iteration-per-tile values (1×, 2×, 4×) that are also presented in `pm_ideal_and_fpu_util.md` (the effective FLOPs/cycle table, lines 43–47) and referenced in `causes_of_low_fpu_util.md` line 73. Three files carry overlapping fidelity-to-throughput mappings.
**Suggestion:** In `pm_ideal_and_fpu_util.md` and `causes_of_low_fpu_util.md`, replace the inline fidelity-iteration restatements with a pointer to the authoritative table in `remediation_levers.md`. Saves ~4 lines scattered across files.

### [guide-level index.md] ~lines 35–44 (Quick Reference table)
**Issue:** The Quick Reference table in `index.md` lines 35–44 repeats one-liner definitions for `DEVICE KERNEL DURATION`, `FPU UTIL`, `PM IDEAL`, and `NOC BW UTIL`. These are identical to the one-liners in the Ch 3 index quick-reference cheat sheet (`ch3_csv_reference/index.md` lines 33–38). Two tables in two index files carry the same four column summaries.
**Suggestion:** In the guide-level `index.md`, condense the Quick Reference table to the two env-var entries (`TT_METAL_DEVICE_PROFILER`, `TT_METAL_PROFILER_SYNC`) and the two tool entries (`tracy-capture`, `ttnn.execute_trace`) that do not appear in the Ch3 cheat sheet, and add a pointer to Ch3's cheat sheet for the column definitions. Saves ~4 lines.

## Load-Bearing Evidence

- `ch1/two_profilers_compared.md` line ~26: `"total_op_latency = host_dispatch_time + device_kernel_time + sync_overhead"` — load-bearing because this is the canonical home of the decomposition equation; all later references in Ch2, Ch6 index, and Ch6 what_is_dispatch_overhead.md should defer to this definition.
- `ch2/build_requirements.md` line ~96: `"The Tracy client embedded in tt-metal lives at: tt_metal/third_party/tracy/ ... must be built from the same submodule commit"` — load-bearing because version-mismatch is the most common capture failure; removing the submodule path or the must-match warning would leave users without actionable recovery steps.
- `ch3/pm_ideal_and_fpu_util.md` line ~40: `"| BFLOAT16 | HiFi4 | 256 |"` (the fidelity-to-effective-FLOPs/cycle table) — load-bearing because this is the authoritative lookup table for PM IDEAL computation; all ch4 and ch5 references to fidelity-adjusted throughput trace back here.
- `ch4/classification_method.md` lines ~131–169 (decision flowchart) — load-bearing because the ASCII flowchart is the only place in the guide that integrates all five classification steps into a single navigable decision procedure; collapsing it into prose would destroy its utility as a diagnostic reference.
- `ch4/worked_examples.md` line ~213: the three-row Summary Comparison table (`Large matmul / Small matmul / silu`) — load-bearing because it is the only place in the guide that juxtaposes the three canonical op archetypes in a scannable side-by-side format; removing or merging it would require readers to compare across three separate example sections.
- `ch5/csv_signatures.md` lines ~10–21 (Diagnostic Checklist ordering) — load-bearing because the recommended rule-out sequence (Cause 6 first, Cause 7 last) is the operationally critical piece; the ordering is not derivable from the individual cause descriptions alone.
- `ch6/eliminating_dispatch_overhead.md` lines ~62–101 (Constraints section) — load-bearing because the three correctness constraints (fixed shapes, no host state modification, pre-allocated output tensors) are each accompanied by a `Warning` or specific API example that is not duplicated elsewhere; cutting any of the three constraint blocks would leave users without guidance on a silent correctness failure mode.

## VERDICT
- Crucial updates: yes

---

## Agent A Change Log — Final Pass B Review Pass 1
- ch3/pm_ideal_and_fpu_util.md: Qualified clock frequency claim — Wormhole B0 TRISC nominally ~1 GHz but varies; readers should verify from device_params.json; use [ns] columns directly to avoid clock dependency.

## Agent A Change Log — Final Pass B Review Pass 2
- ch1/two_profilers_compared.md: Corrected "run twice" instruction to "run simultaneously" — both Tracy and device profiler active in same run with TT_METAL_PROFILER_SYNC=1; cross-references Ch6.

## Agent A Change Log — Final Pass B Review Pass 3
- ch2/env_vars_and_flags.md: Fixed interaction table — replaced ENABLE_PROFILER=ON with TRACY_ENABLE for Tracy capture build requirement.

## Agent A Change Log — Final Pass B Review Pass 4
- ch4/classification_method.md: Corrected TRISC0 role from "data reader" to "math unpacker (L1 → FPU registers)"; NCRISC is the data reader.
- ch4/classification_method.md: Corrected "Tracy profiler" attribution to "device profiler / ops_perf_results.csv".

## Agent A Change Log — Final Pass B Review Pass 5
- ch4/classification_method.md: Fixed interpretation table — changed TRISC0 DURATION >> TRISC1 to NCRISC DURATION >> TRISC1 for read-bandwidth-bound pattern.

## Agent A Change Log — Final Pass B Review Pass 6
- ch5/csv_signatures.md: Replaced non-existent INPUT 0 SHAPE / INPUT 1 SHAPE with actual Ch3 column names INPUT_0_Y, INPUT_0_X, INPUT_1_Y, INPUT_1_X.

## Agent A Change Log — Final Pass B Review Pass 7
- ch4/worked_examples.md: Corrected tiles-per-core threshold from 8-16 to ≥4 to match Ch5's authoritative M_t×N_t/core_count ≥ 4 guideline.

## Agent A — Final Pass 9 Fixes
- ch3/pm_ideal_and_fpu_util.md (Fix 1): Replaced both occurrences of `stall_cycles = TRISC1_KERNEL_DURATION_cycles × (1 − FPU_UTIL)` with the physically meaningful direct subtraction form `stall_cycles = TRISC1_KERNEL_DURATION_cycles − PM_IDEAL_cycles`. The two forms are algebraically equivalent; the subtraction form references only directly observable CSV columns without requiring knowledge of FPU_UTIL's derivation.
- ch4/index.md (Fix 2): Replaced "CSV columns produced by the Tracy profiler" with "CSV columns produced by the device profiler". Those columns (FPU UTIL, NOC BW UTIL, PM IDEAL) come from `process_ops_logs.py` processing on-device cycle-counter logs, not from Tracy.
- ch2/capture_workflow.md (Fix 3): Replaced three occurrences of the hardcoded path `tt_metal/tools/profiler/logs/ops_perf_results.csv` with the correct pattern `generated_profile_log_{test_name}/ops_perf_results_{timestamp}.csv`. Affected locations: the shell verification snippet (line ~90), the Output Artifacts table (line ~103), and the end-of-section prose (line ~265).
- ch2/index.md (Fix 3): Updated Step 5 checklist item — replaced `tt_metal/tools/profiler/logs/` prefix with `generated_profile_log_{test_name}/` to match the corrected output path pattern.

## Agent A — Final Pass 11 Fixes
- ch2/capture_workflow.md (Fix 1): Corrected the bash verification block comment — replaced "The CSV is written to tt_metal/tools/profiler/logs/ relative to the repo root." with "The CSV is written to generated_profile_log_{test_name}/ops_perf_results_{timestamp}.csv". The comment now matches the CSV_PATH variable immediately below it.
- ch3/csv_column_definitions.md (Fix 2): Replaced `tt_metal/tools/profiler/logs/` with `generated_profile_log_{test_name}/` in the device_params.json note (clock frequency section, Preferred approach). The file lands in the same output directory as the CSV, not in the old logs path.
- ch5/causes_of_low_fpu_util.md (Fix 3): Removed the parenthetical "(or DRAM via the NoC)" from the TRISC0 bullet in the Cause 4 mechanism paragraph. TRISC0 is the math unpacker and reads exclusively from L1; DRAM-to-L1 DMA is NCRISC's exclusive role. The erroneous parenthetical conflated TRISC0's job with NCRISC's.

## Agent A — Final Pass 12 Fixes
- ch4/roofline_model_primer.md (Fix 1): Removed the erroneous "32 × 2 = 64 FLOPs/cycle for simple element-wise paths" statement. Replaced with the correct figure: peak FPU throughput is 128 FMA ops/cycle = 256 FLOPs/cycle for ALL op paths (matmul and element-wise). Added a clarifying note that element-wise ops are in practice memory-bound and rarely approach FPU saturation.
- ch5/causes_of_low_fpu_util.md (Fix 2): Corrected TRISC0/NCRISC attribution throughout Cause 4 and Cause 5. Specifically: (a) Cause 4 mechanism — replaced "tiles not yet in L1 and must be fetched from DRAM [by TRISC0]" with "NCRISC has not yet delivered tiles to L1 for TRISC0 to unpack (TRISC0 reads only from L1; DRAM-to-L1 DMA is NCRISC's role)"; (b) Cause 4 observable effect — changed "intermittent DRAM reads that TRISC0 is waiting on" to "DRAM-to-L1 DMA traffic that NCRISC is issuing, with TRISC0 stalled waiting for tiles to arrive in L1"; (c) Cause 4 fix — changed "TRISC0 reads from local L1 rather than DRAM" to tiles being pre-resident in L1, eliminating the need for NCRISC to DMA them at compute time; (d) Cause 5 mechanism — changed "each core issuing DRAM read requests" to "NCRISC on each core issuing DRAM-to-L1 DMA read requests", and attributed tile delivery delay to NCRISC; (e) Cause 5 fix and tip — updated DMA scheduling language and the coexistence tip to consistently attribute DRAM reads to NCRISC.

## Agent A — Final Pass 13 Fixes
- ch5/causes_of_low_fpu_util.md (Fix 1): Corrected the Cause 4 mechanism statement — replaced "Each Tensix core runs three independent RISC-V processors: TRISC0, TRISC1, TRISC2" with the accurate "Each Tensix core runs **five** independent RISC-V processors: BRISC, NCRISC, TRISC0, TRISC1, TRISC2." Added bullet descriptions for BRISC (host-facing dispatcher) and NCRISC (NoC/DMA manager) to match the existing TRISC0/TRISC1/TRISC2 bullets.

## Agent A — Final Pass 10 Fixes
- ch4/roofline_model_primer.md (Fix 1): Corrected tile FLOPs calculation. Replaced wrong values (32×32 = 1024 MACs, 2048 FLOPs per tile, 8 cycles per tile) with correct values for a full BF16 matmul tile-triplet: 32×32×32 = 32,768 MACs, 65,536 FLOPs per tile-triplet, 256 cycles per tile-triplet (at 256 FLOPs/cycle). The derived ceiling of 256 FLOPs/cycle and the ridge point of 8.0 FLOPs/byte are unchanged.
- ch2/capture_workflow.md (Fix 2): Replaced hardcoded static CSV path (`PROFILER_LOG_DIR / "ops_perf_results.csv"` pointing to `tt_metal/tools/profiler/logs/`) in the Minimal Working Example Python block with a dynamic glob-based lookup (`pathlib.Path(".").glob("generated_profile_log_*")` → `ops_perf_results_*.csv`) that correctly discovers the timestamped output under `generated_profile_log_{test_name}/ops_perf_results_{timestamp}.csv`. Added `import glob` and an explanatory comment documenting the path pattern.

## Agent A — Final Pass 14 Fixes
- ch3/pm_ideal_and_fpu_util.md (Fix 1): Renamed misleading table column header "FPU peak ops/cycle (per core)" to "Effective FLOPs/cycle for PM IDEAL". The original values of 512 (BF16/HiFi2), 512 (BFLOAT8_B/HiFi2), and 1024 (BFLOAT8_B/LoFi) exceed the actual hardware FPU ceiling of 256 FLOPs/cycle and would be interpreted by readers as impossible hardware specs. Replaced the existing Note with a corrected one clarifying: values above 256 reflect reduced loop iterations in lower-fidelity modes (fewer accumulation passes per tile), not a higher hardware FPU rate; hardware FPU peak is always 256 FLOPs/cycle (128 FMA ops/cycle).

## Agent A — Final Pass 15 Fixes
- ch4/classification_method.md (Fix 1): Corrected the Step 5 preamble from "The **three** TRISC processors" to "The four per-core RISC processors profiled in the CSV". NCRISC is not a TRISC processor — it is a separate NoC RISC; the CSV profiles four processors (NCRISC, TRISC0, TRISC1, TRISC2) with BRISC excluded because it lacks per-cycle counters in ops_perf output. Also updated the two "All three durations" rows in the interpretation table to "All four durations" for consistency.
- ch5/causes_of_low_fpu_util.md (Fix 2): Corrected Cause 3 (Math Fidelity Mismatch) to be consistent with Ch3's fidelity-aware PM IDEAL. Removed the erroneous claim that "FPU UTIL will be ~0.25× at HiFi4 relative to LoFi" and removed the "FPU UTIL ≈ 0.25" observable effect bullet. PM IDEAL scales proportionally with fidelity (256/512/1024 effective FLOPs/cycle for HiFi4/HiFi2/LoFi respectively), so switching fidelity increases both TRISC1_KERNEL_DURATION and PM_IDEAL_cycles by the same factor — FPU UTIL stays roughly constant for a well-tuned op. The correct statement is that HiFi4 causes ~4× slower absolute kernel time, and FPU UTIL only drops if HiFi4 introduces overhead (pipeline stalls, register pressure) beyond what PM IDEAL already models.

## Agent A — Final Pass 17 Fixes
- ch1/two_profilers_compared.md (Fix 1): Corrected the RISC processor list from "(BRISC, NCRISC, TRISC0, TRISC1, TRISC2)" to "(NCRISC, TRISC0, TRISC1, TRISC2)". Added an explicit note that BRISC is excluded from per-cycle profiling because it has no per-cycle counters and does not appear in `ops_perf_results.csv`.
- ch3/csv_column_definitions.md (Fix 2): Added a clear "Not present in `ops_perf_results.csv`" note to the `BRISC KERNEL DURATION [ns]` table entry, explaining that BRISC lacks per-cycle counters and this column does not appear in the CSV.
- ch4/roofline_model_primer.md (Fix 3): Removed the misleading "SIMD width: 512 bits / 16 bits per element = 32 elements per cycle" derivation. Replaced with the correct direct statement: the Wormhole B0 Tensix FPU is a matrix engine that executes 128 BF16 FMA operations per cycle = 256 FLOPs/cycle, without a false SIMD-width derivation.

## Agent A — Final Pass 18 Fixes
- ch1/two_profilers_compared.md (Fix 1): Corrected the "When to Use" table row for "Analyzing per-RISC duration breakdown" — replaced `(BRISC, NCRISC, TRISC0–2)` with `(NCRISC, TRISC0, TRISC1, TRISC2)`. BRISC has no per-cycle counters and does not appear in `ops_perf_results.csv`.

## Agent A — Final Pass 19 Fixes
- ch2/capture_workflow.md (Fix 1): Corrected the `duration_col` assignment in the Minimal Working Example Python block — replaced `"DEVICE KERNEL DURATION"` with `"DEVICE KERNEL DURATION [ns]"` to match the actual CSV column name including its unit suffix.
- ch2/capture_workflow.md (Fix 2): Corrected the print statement label in the same block — changed `"cycles"` to `"ns"` since the column contains nanoseconds, not raw clock cycles.

## Agent A — Final Pass 16 Fixes
- ch5/csv_signatures.md (Fix 1): Corrected the Cause 3 CSV pattern. Removed the erroneous claims that `FPU UTIL ≈ 0.25 × LoFi_util` for HiFi4 and `≈ 0.5 × LoFi_util` for HiFi2. Because PM IDEAL accounts for fidelity proportionally (256 effective FLOPs/cycle for HiFi4, 512 for HiFi2, 1024 for LoFi), FPU UTIL stays roughly constant across fidelity modes for a well-tuned op. Replaced these claims with the correct diagnostic signal: the absolute `TRISC1 KERNEL DURATION` is ~4× longer at HiFi4 versus a LoFi baseline (due to ~4× more compute iterations per tile), while FPU UTIL itself is NOT depressed. Updated the CSV pattern block, the explanatory paragraph, and the "How to check" guidance to direct readers to compare `TRISC1 KERNEL DURATION` across fidelity settings rather than looking for low `FPU UTIL`.

## Agent A — Final Pass 20 Fixes
- ch5/low_fpu_util/index.md (Fix 1): Corrected the Cause 3 row in the summary table — replaced `FPU UTIL ≈ 0.25 × LoFi_util` with `TRISC1 KERNEL DURATION [ns] ≈ 4 × LoFi baseline` in the "Key CSV Signature" column. FPU UTIL is not the correct diagnostic for math fidelity mismatch; the correct signal is a ~4× longer absolute TRISC1 kernel duration at HiFi4 versus a LoFi baseline. This aligns the summary table with the corrections already made to ch5/csv_signatures.md (Final Pass 16) and ch5/causes_of_low_fpu_util.md (Final Pass 15).

## Agent A — Final Pass 21 Fixes
- ch3/pm_ideal_and_fpu_util.md (Fix 1): Removed self-contradictory expression `K_t × 32 = 32` in the compute-cycles numerator breakdown (line 36). The original text implied K_t = 1 while K_t is actually the total number of K-dimension tiles. Replaced "accumulates K_t × 32 = 32 partial products" with "accumulates 32 partial products from the 32 K-dimension elements of the paired input tile". The per-tile-triplet accumulation count is simply 32 (the K-dimension size of one tile); K_t tiles are iterated over in the outer loop and are already captured by the `M_t × K_t × N_t` factor in the numerator.

## Agent A — Final Pass 22 Fixes
- ch4/classification_method.md (Fix 1): Corrected TRISC2 role in Step 5 role table — replaced "Data writer: moves output tiles from L1 to DRAM/NoC" with "Math packer: packs computed results from FPU output registers back into L1". TRISC2 is the math packer stage, not a data writer; NCRISC handles DMA/NoC data movement.
- ch4/classification_method.md (Fix 2): Corrected the interpretation table entry for high TRISC2 duration — replaced "Write-bandwidth-bound: data writer is stalling the pipeline" with "Output-buffer-stall: TRISC2 (math packer) is stalling because its L1 output buffer is full, typically because NCRISC or a downstream stage is not consuming tiles fast enough". This accurately describes the stall mechanism: TRISC2 stalls on L1 buffer backpressure, not on DRAM/NoC write bandwidth.

## Agent A — Final Pass 23 Fixes
- ch5/causes_of_low_fpu_util.md (Fix 1): Removed "(or DRAM)" from the TRISC2 bullet in the five-processor list. TRISC2 (math packer) writes exclusively to L1; it has no direct DRAM write path. The erroneous parenthetical would mislead readers into attributing high TRISC2 duration to DRAM write bandwidth rather than L1 output buffer backpressure.
- ch4/worked_examples.md (Fix 2): Corrected the TRISC2 description in the Example 3 (SiLU / bandwidth-bound) Step 5 breakdown — replaced "the writer is continuously pushing output tiles" (implying NoC traffic) with "TRISC2 packs math results into L1 output buffers; it is NCRISC that subsequently pushes those tiles over the NoC." TRISC2 operates solely within L1; NoC output traffic is driven by NCRISC.

## Agent A — Final Pass 24 Fixes
- ch4/classification_method.md (Fix 1): Corrected NCRISC role in the Step 5 role table — replaced "Data reader: moves input tiles from DRAM/NoC into L1" with "NoC DMA reader/writer: moves input tiles from DRAM/NoC into L1 and writes output tiles from L1 to DRAM/NoC." NCRISC handles all DRAM/NoC traffic in both directions (reads and writes), not just reads.

## Agent A — Cross-Chapter Compression Pass 1 Changes

**Item 1 — DEVICE KERNEL DURATION re-definitions:**
- `ch6/measuring_dispatch_vs_kernel.md`: Replaced the full DEVICE KERNEL DURATION definition paragraph (formula + wall-clock explanation) with a one-line cross-reference: "See Ch3 for the `DEVICE KERNEL DURATION` definition." The authoritative definition remains in `ch3/csv_column_definitions.md`.
- `ch3/pm_ideal_and_fpu_util.md`: No standalone re-definition block was present in the current file (prior passes had already removed the verbatim formula); no change required beyond Item 2 below.

**Item 2 — Wormhole B0 hardware ceilings re-derivation:**
- `ch3/pm_ideal_and_fpu_util.md`: Replaced the Note below the fidelity table that re-stated the 256 FLOPs/cycle and 128 FMA ops/cycle hardware ceilings with a one-line cross-reference: "See Ch4 for the Wormhole B0 hardware ceilings (256 FLOPs/cycle, 8.0 FLOPs/byte ridge point)." The fidelity lookup table itself (unique to Ch3) was preserved. The full derivation remains in `ch4/roofline_model_primer.md`.

**Item 3 — Ch2 setup checklist duplication:**
- `ch2/index.md`: Condensed the five-step checklist from verbose entries with full flag names and step descriptions down to five brief one-line bullets (step name + one-sentence description), with a top-level "See `capture_workflow.md` for full step details" pointer. Removed the duplicated `-o`, `-f`, `-s` flag explanations and the full narrative that was already in `capture_workflow.md`.

**Item 4 — csv_signatures.md mechanism re-explanations:**
- `ch5/csv_signatures.md`, Cause 1: Removed the three-bullet "Distinguishing from other causes" block; replaced with a single sentence noting the key distinguishing signals (`TRISC1 KERNEL DURATION` is short, `NOC BW UTIL` is low) and a pointer to `causes_of_low_fpu_util.md`.
- `ch5/csv_signatures.md`, Cause 2: Removed the three-bullet "Distinguishing from Cause 3" block; replaced with a one-sentence note that both causes can coexist, plus a pointer to `causes_of_low_fpu_util.md`.
- `ch5/csv_signatures.md`, Cause 3: Trimmed the mechanism explanation from three paragraphs to two sentences covering only the key diagnostic implication (do not expect low FPU UTIL; check TRISC1 KERNEL DURATION instead), with pointer to `causes_of_low_fpu_util.md`.
- `ch5/csv_signatures.md`, Cause 4: Replaced the four-component mechanism + distinguishing-from-Cause-5 block with a two-sentence summary of stall signal and how Cause 4 differs from Cause 5 (`NOC BW UTIL < 0.8`), plus pointer to `causes_of_low_fpu_util.md`.
- `ch5/csv_signatures.md`, Cause 5: Replaced the three-paragraph mechanism + verification + warning block with a two-sentence summary (both conditions required, AI > 8.0 context), plus pointer to `causes_of_low_fpu_util.md`.
- `ch5/csv_signatures.md`, Cause 7: Replaced the multi-sentence "Quantitative signal" paragraph with a single sentence describing the padding loop-count mechanism, with pointer to `causes_of_low_fpu_util.md` for investigation steps.

**Item 5 — total_op_latency equation re-definitions:**
- `ch1/two_profilers_compared.md`: Replaced the full three-term block (equation + three bullet definitions) in the "Why Both Are Needed" section with a single sentence containing the inline equation and a "See Ch6 for the full latency decomposition" pointer. The subsequent non-definition prose was retained.
- `ch2/index.md` (prerequisites): Replaced the inline equation in the Device profiler vs. Tracy prerequisite bullet with "total op latency" prose and a "See Ch6 for the full latency decomposition" pointer.
- `ch2/build_requirements.md`: Replaced the inline `total_op_latency = ...` equation in the ENABLE_PROFILER description with prose referencing host-side and on-die terms, plus a "See Ch6 for the full latency decomposition" pointer.
- `ch6/index.md`: Replaced the fenced code block containing the equation with an inline mention and a pointer to `what_is_dispatch_overhead.md` for full term definitions and the numeric example. Also trimmed the prerequisites bullet that re-stated the full equation.
- `ch6/what_is_dispatch_overhead.md`: Full equation and numeric example (lines 100–108) retained unchanged as the authoritative definition home.
