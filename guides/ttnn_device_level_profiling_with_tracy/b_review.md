## Final Pass — Pass 1

All six numbered cross-chapter consistency requirements were checked against every chapter index file and the four spot-check files. Five requirements are clean; one issue was found.

---

**Issue 1 — Incorrect clock frequency used for cycle-to-ns conversion (cross-chapter numerical error)**

- File: `ch3_csv_reference/pm_ideal_and_fpu_util.md`, line 16
- Error: The file states "On Wormhole B0 at 1 GHz, the two columns are numerically identical (one cycle = one nanosecond)." Wormhole B0 TRISC cores do not run at 1 GHz; the TRISC clock is ~1.2 GHz (1200 MHz). The 1 GHz claim establishes the numeric baseline for PM IDEAL cycle-to-ns conversion. Every downstream chapter that asks a reader to convert between `PM IDEAL [ns]` and `PM IDEAL [cycle]` — or to compute compute_cycles from nanoseconds — inherits this error. Ch4 (`roofline_model_primer.md`) uses PM IDEAL in both units to compute arithmetic intensity and the ridge point. Ch5 (`index.md` line 26) cites `AI_ridge = 8.0 FLOPs/byte` as a fact derived from those calculations. If the reader manually replicates any cycle-to-ns conversion using "1 cycle = 1 ns", they will be off by ~20%.
- Fix: Replace "On Wormhole B0 at 1 GHz, the two columns are numerically identical (one cycle = one nanosecond)" with the correct clock frequency (e.g., "On Wormhole B0 at approximately 1.2 GHz, `PM IDEAL [ns] = PM IDEAL [cycle] / 1.2`") and update any worked examples in Ch4 and Ch5 that assume 1 GHz.

---

**Requirements verified as consistent (no issues):**

1. `total_op_latency = host_dispatch_time + device_kernel_time + sync_overhead` — identical in Ch1 `two_profilers_compared.md`, Ch2 `index.md`, Ch6 `index.md`, and Ch6 `measuring_dispatch_vs_kernel.md`.
2. `TT_METAL_PROFILER_SYNC` — defined in Ch2 `env_vars_and_flags.md`; referenced by exact same name in Ch6 `index.md` and Ch6 `measuring_dispatch_vs_kernel.md`.
3. `DEVICE KERNEL DURATION [ns]` = wall-clock from first core start to last core end — consistent across Ch3 `index.md`, Ch1 `two_profilers_compared.md`, and Ch6 `measuring_dispatch_vs_kernel.md`.
4. `FPU_UTIL = PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles` — consistent between Ch3 `index.md` (line 35) and Ch3 `pm_ideal_and_fpu_util.md` (line 73); used (not redefined) in Ch4 and Ch5.
5. FPU UTIL thresholds >0.7 compute-bound, <0.3 bandwidth-bound — consistent between Ch4 `index.md` (lines 39–41) and Ch5 `index.md` (line 26).
6. Guide-level `index.md` chapter table — all six chapter entries are clickable markdown links to the respective `index.md` files; no bare-text entries.

---

## Final Pass — Pass 2

One cross-chapter inconsistency found; all other consistency checks passed.

---

**Issue 1 — Conflicting instruction on whether Tracy and the device profiler must be run separately or can run simultaneously**

- File: `ch1_tracy_fundamentals/two_profilers_compared.md`, line 48
- Error: The Ch1 note states: "In practice, you run the same workload twice — once with Tracy capture active and once with `TT_METAL_DEVICE_PROFILER=1` — and compare the measurements numerically." Ch6 `measuring_dispatch_vs_kernel.md` line 131 gives the direct opposite procedure as the authoritative workflow: "Run with `TT_METAL_DEVICE_PROFILER=1` and Tracy capture active, with `TT_METAL_PROFILER_SYNC=1`" — both profilers simultaneously in one run. A reader following Ch1's instruction will produce two separate runs and attempt to correlate op rows across them, which is unreliable (op ordering and timing shift between runs). Ch6's procedure is correct: both profilers can and should be active in the same run for the dispatch vs. kernel decomposition workflow.
- Fix: In `two_profilers_compared.md`, revise the note to state that running both profilers simultaneously in a single run is the correct procedure, and that separate runs are only needed when the combined overhead of both instruments is itself a concern to avoid. Remove or qualify the sentence "you run the same workload twice."

---

**Requirements verified as consistent (no issues):**

1. Pass 1 Issue 1 re-checked: `pm_ideal_and_fpu_util.md` line 16 correctly states "approximately 1 GHz" and explicitly warns "do not assume a fixed cycle-to-nanosecond ratio." No "1 cycle = 1 nanosecond" claim exists in the current file. The Pass 1 finding references language not present in the reviewed version; the file is clean on this point.
2. `ttnn.execute_trace` API naming — Ch6 `index.md` (eliminating_dispatch_overhead.md entry) and guide-level `index.md` Quick Reference both use `ttnn.execute_trace` consistently; Ch6 `index.md` overview additionally names `ttnn.begin_trace_capture` and `ttnn.end_trace_capture` as companion calls, which is additive and not contradictory.
3. Crossover point value — Ch6 `index.md` key insight block (~100 µs) and Ch6 `measuring_dispatch_vs_kernel.md` line 121 (~100 µs) are consistent.
4. Guide-level `index.md` non-clickable links — re-verified: all six chapter table entries and all Quick Reference entries are formatted as clickable markdown links; no bare chapter names present.

---

## Final Pass — Pass 3

One cross-chapter inconsistency found; all other consistency checks passed.

---

**Issue 1 — Wrong build flag cited for Tracy in the interaction table (material misdirection)**

- File: `ch2_invocation/env_vars_and_flags.md`, line 61
- Error: The table row "Tracy capture only (.tracy file, no CSV)" lists the requirement as "`TRACY_NO_EXIT=1` (and `ENABLE_PROFILER=ON` build)". `ENABLE_PROFILER=ON` is the CMake flag that enables the device-side CSV profiler instrumentation (`kernel_profiler.hpp`). It has no effect on Tracy. Tracy requires a separate build-time define — identified as `TRACY_ENABLE` two lines above in the same section ("Tracy is activated solely by the build-time `TRACY_ENABLE` define") and again in `ch1_tracy_fundamentals/two_profilers_compared.md` line 48 ("build with `TRACY_ENABLE`"). A reader following only the table will not know they need `TRACY_ENABLE` at build time; they will instead believe that enabling the device-CSV build flag is what activates Tracy, which is incorrect and will result in a silent no-op Tracy capture.
- Fix: Replace `ENABLE_PROFILER=ON` with `TRACY_ENABLE` in the table cell: "Tracy capture only (.tracy file, no CSV) | `TRACY_NO_EXIT=1` (and `TRACY_ENABLE` build)".

---

**Requirements verified as consistent (no issues):**

1. Pass 2 Issue 1 re-checked: `two_profilers_compared.md` line 48 now correctly states "running both profilers simultaneously in a single pass — not running the workload twice." The fix is in place.
2. FPU peak ops/cycle — Ch3 `pm_ideal_and_fpu_util.md` table (BF16/HiFi4 = 256 FLOPs/cycle) is consistent with Ch4 `roofline_model_primer.md` line 97 (256 FLOPs/cycle for BF16 matmul) and the ridge point derivation (256 / 32 = 8.0 FLOPs/byte).
3. Ridge point value — Ch4 `roofline_model_primer.md` line 123 (AI_ridge = 8.0 FLOPs/byte) is used consistently in Ch5 `csv_signatures.md` lines 117, 141, and 145.
4. Guide-level `index.md` chapter links — all six chapter table entries and all Quick Reference entries verified as clickable markdown links to existing `index.md` files; no broken or bare-text entries.

---

## Final Pass — Pass 4

Two cross-chapter inconsistencies found; all other consistency checks passed.

---

**Issue 1 — TRISC0 role contradicts Ch3: wrong processor identified as the DRAM/NoC data reader (material misdirection)**

- File: `ch4_compute_vs_bandwidth/classification_method.md`, line 111
- Error: The Step 5 TRISC role table describes TRISC0 as "Data reader: moves input tiles from DRAM/NoC into L1." This is the role Ch3 `csv_column_definitions.md` assigns to **NCRISC**: "NCRISC drives NoC DMA transfers — large values here indicate data-movement bottlenecks." Ch3 consistently defines TRISC0 as "the math unpacker" that "unpacks tiles from L1 memory into the math engine's source registers." A reader using Ch4's Step 5 table to diagnose a read-bandwidth bottleneck will examine `TRISC0 DURATION` and find it small (the unpacker is fast), conclude the op is not read-bandwidth-bound, and miss the actual signal in `NCRISC DURATION`.
- Fix: In the Step 5 table, replace TRISC0's role with "Math unpacker: moves tiles from L1 into math engine registers" and add a NCRISC row — "Data reader: drives NoC DMA transfers from DRAM/NoC into L1; large values indicate data-movement bottleneck" — consistent with Ch3.

---

**Issue 2 — Opening sentence misattributes the classification CSV columns to Tracy rather than the device profiler (factual error)**

- File: `ch4_compute_vs_bandwidth/classification_method.md`, line 3
- Error: "classifying a kernel as compute-bound, bandwidth-bound, or overhead-bound using only the CSV columns produced by the Tracy profiler." `FPU UTIL`, `NOC BW UTIL`, and `PM IDEAL` are all device-profiler columns, produced by `process_ops_logs.py` from on-device cycle-counter data. Tracy produces a `.tracy` host-side zone file, not `ops_perf_results.csv`. This is the exact distinction the entire Ch1 is built on. A reader who has not yet read Ch1 will enter Ch4 with the false belief that enabling Tracy alone is sufficient to obtain the classification columns.
- Fix: Replace "CSV columns produced by the Tracy profiler" with "CSV columns produced by the device profiler (`ops_perf_results.csv`)".

---

**Requirements verified as consistent (no issues):**

1. Pass 3 Issue 1 re-checked: `ch2_invocation/env_vars_and_flags.md` line 61 now correctly reads "`TRACY_NO_EXIT=1` (and `TRACY_ENABLE` build)" — the `ENABLE_PROFILER=ON` error is not present. Fix confirmed applied.
2. Guide-level `index.md` chapter links — all six Chapter Index entries and all Quick Reference entries are formatted as clickable markdown links; all target `index.md` files exist on disk. No broken or bare-text entries.
3. `total_op_latency` decomposition formula — consistent across Ch1 `two_profilers_compared.md` line 25, Ch2 `build_requirements.md` line 11, and Ch6 `eliminating_dispatch_overhead.md` (implied by the dispatch vs. kernel framing throughout).
4. `FPU_UTIL` definition — `PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles` is stated consistently in Ch3 `csv_column_definitions.md` line 66 and Ch4 `classification_method.md` line 46; thresholds (>0.7 compute-bound, <0.3 not compute-bound) are consistent between Ch4 and Ch5.

---

## Final Pass — Pass 5

One cross-chapter inconsistency found causing a wrong diagnostic; all other consistency checks passed.

---

**Issue 1 — Ch4 Step 5 interpretation table contradicts Ch3 and its own role table: `TRISC0 DURATION` named as the read-bandwidth signal instead of `NCRISC DURATION` (material misdirection)**

- File: `ch4_compute_vs_bandwidth/classification_method.md`, lines 120–121
- Error: The interpretation table at line 121 reads "`TRISC0 DURATION >> TRISC1 DURATION` → Read-bandwidth-bound: data reader is stalling the pipeline." This directly contradicts the role table six lines above it (line 111), which was corrected in Pass 4 to say "NCRISC | Data reader: moves input tiles from DRAM/NoC into L1" and "TRISC0 | Math unpacker: moves tile data from L1 into FPU registers." It also contradicts Ch3 `csv_column_definitions.md` lines 55–56, which defines `NCRISC KERNEL DURATION [ns]` as "NCRISC drives NoC DMA transfers — large values here indicate data-movement bottlenecks." A reader following the interpretation table will watch `TRISC0 DURATION` for read-bandwidth bottlenecks, find it small (because TRISC0 is the fast L1-to-register unpacker, not the DRAM fetcher), and incorrectly conclude the op is not read-bandwidth-bound — missing the real signal in `NCRISC DURATION`.
- Fix: In the interpretation table, change "`TRISC0 DURATION >> TRISC1 DURATION` | Read-bandwidth-bound: data reader is stalling the pipeline" to "`NCRISC DURATION >> TRISC1 DURATION` | Read-bandwidth-bound: NoC DMA reader is stalling the pipeline", consistent with the role table above and Ch3.

---

**Requirements verified as consistent (no issues):**

1. Pass 4 Issues 1 and 2 re-checked: Ch4 `classification_method.md` role table (lines 111–114) now correctly assigns NCRISC as data reader and TRISC0 as math unpacker. Opening sentence (line 3) correctly names the device profiler CSV, not Tracy. Both Pass 4 fixes confirmed applied.
2. `index.md` chapter links — all six Chapter Index entries link to `ch*/index.md` paths; all six `index.md` files exist on disk. All Quick Reference entries also use clickable markdown links. No non-clickable chapter links found.
3. `FPU UTIL` definition and thresholds — `PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles`, >0.7 compute-bound, <0.3 not compute-bound — consistent across Ch3 `csv_column_definitions.md` line 66, Ch4 `classification_method.md` lines 43–55, and Ch5 `causes_of_low_fpu_util.md` (used throughout without redefining, consistent with Ch3/Ch4).
4. `DEVICE KERNEL DURATION` definition — wall-clock from first core start to last core end — consistent between Ch3 `csv_column_definitions.md` lines 115–119 and Ch4 `classification_method.md` line 81.
5. `ttnn.execute_trace` naming — consistent across guide-level `index.md` Quick Reference (line 44) and Ch6 `what_is_dispatch_overhead.md` (not contradicted). Ch1 `what_is_tracy.md` lists `execute_trace` among Tracy-annotated lifecycle events (line 81), consistent with Ch6's framing of trace replay as the primary dispatch-overhead remedy.

---

## Final Pass — Pass 6

One cross-chapter inconsistency found causing a wrong diagnostic; all other consistency checks passed.

---

**Issue 1 — Ch5 Cause 1 references non-existent CSV columns `INPUT 0 SHAPE` and `INPUT 1 SHAPE` (material misdirection)**

- File: `ch5_low_fpu_util/csv_signatures.md`, line 55
- Error: The Cause 1 diagnostic reads: "compute the expected tile count from the op's shape (available from `INPUT 0 SHAPE` and `INPUT 1 SHAPE` columns in the CSV)." Ch3 `csv_column_definitions.md` defines no such columns. The shape data is decomposed into individual dimension columns: `INPUT_0_W`, `INPUT_0_Z`, `INPUT_0_Y`, `INPUT_0_X` (and the same for `INPUT_1_*`). There is no single `INPUT 0 SHAPE` column in `ops_perf_results.csv`. A reader following the Ch5 diagnostic will search for `INPUT 0 SHAPE` in the CSV, find nothing, and be unable to complete the tile-count check for Cause 1.
- Fix: Replace "`INPUT 0 SHAPE` and `INPUT 1 SHAPE` columns" with "`INPUT_0_Y`, `INPUT_0_X`, `INPUT_1_Y`, `INPUT_1_X` columns (the row and column dimensions for each input, as defined in Ch3)", consistent with Ch3's column reference.

---

**Requirements verified as consistent (no issues):**

1. Pass 5 Issue 1 re-checked: `ch4_compute_vs_bandwidth/classification_method.md` interpretation table (line 121) correctly reads "`NCRISC DURATION >> TRISC1 DURATION` | Read-bandwidth-bound: NoC DMA reader is stalling the pipeline." Fix confirmed applied.
2. `index.md` chapter links — all six Chapter Index entries link to `ch*/index.md` paths; all six target files exist on disk. All Quick Reference entries are clickable markdown links. No bare-text or broken chapter links found.
3. `FPU UTIL` definition — `PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles` — stated consistently in Ch3 `csv_column_definitions.md` line 66, Ch4 `classification_method.md` lines 43–46, and used without redefinition in Ch5 `csv_signatures.md`.
4. Ridge point value `AI_ridge = 8.0 FLOPs/byte` — stated in Ch4 `classification_method.md` line 33 and used consistently in Ch5 `csv_signatures.md` lines 117, 141, and 145 (AI > 8.0 as the compute-bound threshold).
5. `DEVICE KERNEL DURATION` definition — wall-clock from first core start to last core end — consistent between Ch3 `csv_column_definitions.md` lines 115–119 and Ch4 `classification_method.md` line 81.

---

## Final Pass — Pass 7

One cross-chapter inconsistency found causing a wrong diagnostic threshold; all other consistency checks passed.

---

**Issue 1 — Conflicting minimum tiles-per-core threshold between Ch4 and Ch5 (material misdirection)**

- File: `ch4_compute_vs_bandwidth/worked_examples.md`, line 135
- Error: The tip reads: "each active core should have at least **8–16** output tiles to sustain healthy FPU pipeline fill. Below **8** tiles per core, `FPU UTIL` will typically fall below 0.3." Ch5 `csv_signatures.md` line 57 defines the diagnostic threshold as `M_t × N_t / CORE COUNT < **4**` (op is tile-starved), and `causes_of_low_fpu_util.md` line 29 gives the fix target as `M_t × N_t / core_count ≥ **4**`. Ch5 also uses `≥ 4` as the ruling-out condition for Cause 1 in the Cause 7 checklist (`csv_signatures.md` line 161). A reader diagnosing Cause 1 using Ch4's tip will flag any op with 4–7 tiles per core as insufficient and attempt remediation, even though Ch5 (the authoritative diagnostic chapter for this cause) considers 4 tiles per core sufficient. Conversely, a reader using only Ch5's threshold of 4 will miss the Ch4 guidance that 8–16 tiles is the range where healthy fill actually materializes.
- Fix: Align Ch4 `worked_examples.md` line 135 with Ch5's established threshold. Change "at least 8–16 output tiles" to "at least 4–8 output tiles" and "Below 8 tiles per core" to "Below 4 tiles per core", consistent with Ch5 `csv_signatures.md` line 57 (`< 4` = tile-starved) and `causes_of_low_fpu_util.md` line 36 (crossover target of `4–8`).

---

**Requirements verified as consistent (no issues):**

1. Pass 6 Issue 1 re-checked: `ch5_low_fpu_util/csv_signatures.md` line 55 correctly names `INPUT_0_Y`, `INPUT_0_X`, `INPUT_1_Y`, `INPUT_1_X` as the shape columns, consistent with Ch3 `csv_column_definitions.md`. No reference to non-existent `INPUT 0 SHAPE` or `INPUT 1 SHAPE` columns. Fix confirmed applied.
2. `index.md` chapter links — all six Chapter Index entries link to `ch*/index.md` paths; all six target files verified to exist on disk. All Quick Reference entries use clickable markdown links. No non-clickable or bare-text chapter links found.
3. `FPU UTIL < 0.4` (Cause 7 residual threshold, Ch5) vs. `>0.7 / <0.3` (classification thresholds, Ch4) — not a contradiction. The Ch4 thresholds classify the op's regime; the Ch5 Cause 7 threshold (`< 0.4`) is a separate, stricter floor for triggering a kernel-level investigation after all other causes are ruled out. Both files use `> 0.7` as the target for a well-tuned compute-bound op, which is consistent.
4. `NOC BW UTIL > 0.8` as the Cause 5 saturation threshold (Ch5 `csv_signatures.md` lines 135, 145) vs. `NOC BW UTIL > 0.7` as the bandwidth-bound classification boundary (Ch4 `index.md` line 40) — these are distinct thresholds for distinct questions (Cause 5 requires a higher saturation level to trigger than the basic bandwidth-bound classification) and are not contradictory.
5. Trace capture constraints in Ch6 `eliminating_dispatch_overhead.md` — tensor pre-allocation requirement, shape-invariance constraint, and `ttnn.to_torch()` prohibition inside a trace are internally consistent and do not contradict Ch2 invocation guidance or Ch3 column definitions.

---

## Final Pass — Pass 8

One cross-chapter inconsistency found causing wrong diagnostic guidance; all other consistency checks passed.

---

**Issue 1 — `worked_examples.md` Example 3 Step 5 misidentifies TRISC0 as the DRAM data reader, contradicting Ch3 and the Pass 4/5 corrections to `classification_method.md` (material misdirection)**

- File: `ch4_compute_vs_bandwidth/worked_examples.md`, line 188
- Error: The Step 5 TRISC breakdown for the bandwidth-bound silu example reads: "`TRISC0 DURATION`: **longest** — the data reader is continuously pulling input tiles from DRAM or L1 of neighbor cores." Ch3 `csv_column_definitions.md` line 55–56 explicitly defines NCRISC as the NoC DMA data reader ("NCRISC drives NoC DMA transfers — large values here indicate data-movement bottlenecks") and TRISC0 as the math unpacker ("TRISC0 unpacks tiles from L1 memory into the math engine's source registers"). The same role assignment was established as the authoritative correction in Pass 4 and Pass 5 for `classification_method.md`. A reader using this worked example as the reference signature for bandwidth-bound ops will watch `TRISC0 DURATION` for the long-pole signal, find it short (TRISC0 is the fast L1-to-register unpacker), and incorrectly conclude the op is not bandwidth-bound — missing the true signal in `NCRISC DURATION`. Additionally, line 53 in Example 1 refers to TRISC0 as "the reader," which perpetuates the same misidentification in a softer form.
- Fix: At line 188, replace "`TRISC0 DURATION`: **longest** — the data reader is continuously pulling input tiles from DRAM or L1 of neighbor cores" with "`NCRISC DURATION`: **longest** — the NoC DMA engine is continuously transferring input tiles from DRAM into L1". At line 53, replace "TRISC0 DURATION: moderately long, but shorter than TRISC1 — the reader can keep up with the FPU because the NoC is not saturated" with "TRISC0 DURATION: moderately long, but shorter than TRISC1 — the unpacker can keep up with the FPU because the NoC is not saturated." These changes align the examples with Ch3 `csv_column_definitions.md` lines 55–56 and with the corrected role table in `classification_method.md`.

---

**Requirements verified as consistent (no issues):**

1. Pass 7 Issue 1 re-checked: `ch4_compute_vs_bandwidth/worked_examples.md` line 135 correctly reads "at least 4–8 output tiles" and "Below 4 tiles per core", consistent with Ch5 `remediation_levers.md` guideline `M_t × N_t / core_count ≥ 4`. Fix confirmed applied.
2. `index.md` chapter links — all six Chapter Index entries (lines 23–29) and all Quick Reference entries (lines 34–44) are formatted as clickable markdown links pointing to `ch*/index.md` paths. No bare-text or non-clickable chapter entries found.
3. `FPU UTIL` definition — `PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles` — consistent between Ch3 `pm_ideal_and_fpu_util.md` line 73 and the worked examples in Ch4 `worked_examples.md` (used without redefining, consistent with Ch3).
4. NoC bandwidth constant — 32 bytes/cycle per NoC link — stated in Ch3 `pm_ideal_and_fpu_util.md` line 57 and used consistently in Ch4 `worked_examples.md` line 177 ("a NoC link bandwidth of 32 bytes/cycle"). No contradiction.
5. Ch6 `what_is_dispatch_overhead.md` `total_op_latency` decomposition (line 103) and `ttnn.enable_program_cache()` guidance (line 84) are consistent with Ch5 `remediation_levers.md` (which also covers `ttnn.enable_program_cache()` as Cause 6 fix) and with Ch2 `capture_workflow.md` (which treats program cache as orthogonal to capture setup). No contradiction.

---

## Final Pass 9

Three factual issues found. All prior-pass fixes confirmed in place; no regressions detected.

---

**Issue 1 — `stall_cycles` formula uses the forbidden `(1 − FPU_UTIL)` form instead of the direct subtraction (factual error)**

- File: `ch3_csv_reference/pm_ideal_and_fpu_util.md`, lines 123 and 138
- Error: The guide writes `stall_cycles = TRISC1_KERNEL_DURATION_cycles × (1 − FPU_UTIL)` at both occurrences. The known-correct definition is `stall_cycles = TRISC1_KERNEL_DURATION - PM_IDEAL_cycles`. While the two expressions are algebraically equivalent given `FPU_UTIL = PM_IDEAL / TRISC1`, the `(1 − FPU_UTIL)` form obscures the physical meaning — stall cycles are the gap between actual TRISC1 time and the ideal model prediction — and directly contradicts the specification that this form must not be used. A reader who substitutes the formula into a spreadsheet will get the right number either way, but the conceptual framing (stalls = actual − ideal, not actual × efficiency complement) is what every downstream chapter builds on when it asks "how many cycles were wasted?"
- Fix: Replace both occurrences with `stall_cycles = TRISC1_KERNEL_DURATION_cycles − PM_IDEAL_cycles`, and add an explanatory note that this equals `TRISC1_KERNEL_DURATION_cycles × (1 − FPU_UTIL)` as a cross-check.

---

**Issue 2 — `ch4/index.md` attributes the classification CSV columns to the Tracy profiler (factual error)**

- File: `ch4_compute_vs_bandwidth/index.md`, line 11
- Error: "The answer comes directly from the CSV columns produced by the **Tracy profiler** and discussed in Chapter 3." `FPU UTIL`, `NOC BW UTIL`, `PM IDEAL`, and every other column used for classification come from `ops_perf_results.csv`, which is produced by the on-device cycle-counter profiler post-processed by `process_ops_logs.py` — not from Tracy. Tracy produces a `.tracy` binary file of host-side CPU zones; it produces no CSV and no `FPU UTIL` column. This exact error was caught in Pass 4 for `classification_method.md` line 3 and corrected there, but the parallel sentence in `ch4/index.md` line 11 was not updated. A reader arriving at Chapter 4 directly (via the guide-level index's "My op is slow" fast-path link) will read the index first, see "Tracy profiler", and believe that enabling only Tracy is sufficient to obtain the classification data.
- Fix: Replace "the CSV columns produced by the Tracy profiler" with "the CSV columns produced by the device profiler (`ops_perf_results.csv`)", consistent with the correction already applied to `classification_method.md` line 3.

---

**Issue 3 — `ops_perf_results.csv` output path is wrong throughout the guide (factual error affecting every code example and lookup instruction)**

- Files: `ch2_invocation/capture_workflow.md` (lines 90, 103, 180, 265), `ch2_invocation/index.md` (line 45), `ch3_csv_reference/csv_column_definitions.md` (lines 8, 105), `ch3_csv_reference/index.md` (line 18), and all other files that reference the CSV location
- Error: Every file in the guide directs readers to look for the CSV at `tt_metal/tools/profiler/logs/ops_perf_results.csv`. The known-correct output path is `generated_profile_log_{test_name}/ops_perf_results_{timestamp}.csv` — a per-run subdirectory rooted at the test name, with a timestamp suffix on the filename. A reader who runs `process_ops_logs.py` and then looks in `tt_metal/tools/profiler/logs/` for a file named `ops_perf_results.csv` will find nothing. The minimal pytest code example in `capture_workflow.md` hardcodes `PROFILER_LOG_DIR = pathlib.Path("tt_metal/tools/profiler/logs")` and `CSV_PATH = PROFILER_LOG_DIR / "ops_perf_results.csv"` — this path will not exist and the assertion `CSV_PATH.exists()` will fail every time.
- Fix: Update all CSV path references to use the correct pattern `generated_profile_log_{test_name}/ops_perf_results_{timestamp}.csv`, and update the working pytest example to glob for the most recent matching file rather than hardcoding a static path.

---

**Requirements verified as consistent (no issues):**

1. Pass 8 Issue 1 re-checked: `ch4_compute_vs_bandwidth/worked_examples.md` line 188 reads "`NCRISC DURATION`: **longest** — the data reader is continuously pulling input tiles from DRAM or L1 of neighbor cores." Line 53 reads "the unpacker can keep up with the FPU because the NoC is not saturated." Both Pass 8 fixes are confirmed in place.
2. `FPU UTIL` definition — `PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles` — stated in Ch3 `pm_ideal_and_fpu_util.md` line 73 and Ch3 `csv_column_definitions.md` line 66; used consistently (without redefinition) in Ch4 `classification_method.md` lines 43–46 and Ch5 throughout. No contradiction.
3. Ridge point — `AI_ridge = 8.0 FLOPs/byte` — derived from `256 FLOPs/cycle ÷ 32 bytes/cycle = 8.0` in Ch4 `roofline_model_primer.md` lines 120–123; used as the threshold throughout Ch4 and Ch5. Consistent.
4. TRISC role assignments — TRISC0 = math unpacker (L1 → FPU registers), TRISC1 = math engine, TRISC2 = math packer, NCRISC = NoC DMA reader — consistent across Ch3 `csv_column_definitions.md` lines 55–58, Ch4 `classification_method.md` Step 5 role table (lines 109–114), Ch4 `worked_examples.md` (lines 53, 188), and Ch5 `causes_of_low_fpu_util.md` lines 111–116. No remaining NCRISC/TRISC0 confusion.
5. `total_op_latency = host_dispatch_time + device_kernel_time + sync_overhead` — identical in Ch1 `two_profilers_compared.md` line 25, Ch2 `index.md` line 26, Ch6 `index.md` line 12, Ch6 `measuring_dispatch_vs_kernel.md` (implied throughout), and Ch6 `what_is_dispatch_overhead.md` line 103.

---

## Final Pass 10

Two factual errors found. All prior-pass fixes confirmed in place; no regressions detected.

---

**Issue 1 — `roofline_model_primer.md` states wrong tile FLOPs and wrong cycles-per-tile, contradicting the known-correct tile FLOPs fact (factual error)**

- File: `ch4_compute_vs_bandwidth/roofline_model_primer.md`, line 97
- Error: "A full outer product of two 32×32 BF16 tiles involves 32 × 32 = 1024 multiply-accumulate operations = **2048 FLOPs per tile**. The matmul engine sustains one tile every 8 cycles when fully pipelined (2048 FLOPs ÷ 256 FLOPs/cycle = 8 cycles per tile)." Both intermediate values are wrong. The known-correct fact is that a BF16 matmul tile-triplet (one 32×32 A tile accumulated against one 32×32 B tile with K=32 inner-dimension steps) performs 32 × 32 × 32 = 32,768 MACs = **65,536 FLOPs** per tile-triplet. Correspondingly, at 256 FLOPs/cycle the engine requires 65,536 ÷ 256 = **256 cycles per tile-triplet**, not 8. The two errors cancel in the division (65,536 ÷ 256 = 2,048 ÷ 8 = 256), so the derived ceiling of 256 FLOPs/cycle and the ridge point of 8.0 FLOPs/byte are correct. However, any reader who uses the stated "2,048 FLOPs per tile" or "8 cycles per tile" for their own calculations — for example, to estimate how long a specific tile sequence should take — will be wrong by a factor of 32.
- Fix: Replace "32 × 32 = 1024 multiply-accumulate operations = 2048 FLOPs per tile. The matmul engine sustains one tile every 8 cycles when fully pipelined (2048 FLOPs ÷ 256 FLOPs/cycle = 8 cycles per tile)" with "32 × 32 × 32 = 32,768 multiply-accumulate operations = **65,536 FLOPs per tile-triplet** (K=32 inner-dimension steps, one per element of the 32×32 output tile). The matmul engine sustains one tile-triplet every 256 cycles when fully pipelined (65,536 FLOPs ÷ 256 FLOPs/cycle = 256 cycles per tile-triplet)", consistent with the known-correct fact 2 × 32 × 32 × 32 = 65,536 and with `pm_ideal_and_fpu_util.md` which correctly uses `2 × M_t × K_t × N_t × 32768` for the total FLOPs formula.

---

**Issue 2 — `capture_workflow.md` Python example hardcodes the wrong CSV path, causing the smoke-test assertion to always fail (surviving instance of Pass 9 Issue 3)**

- File: `ch2_invocation/capture_workflow.md`, lines 180–181
- Error: The minimal working pytest example defines `PROFILER_LOG_DIR = pathlib.Path("tt_metal/tools/profiler/logs")` and `CSV_PATH = PROFILER_LOG_DIR / "ops_perf_results.csv"`, then asserts `CSV_PATH.exists()`. Pass 9 Issue 3 correctly identified that the known-correct output path is `generated_profile_log_{test_name}/ops_perf_results_{timestamp}.csv` — a per-run directory with a timestamp suffix. The shell-level verification block at line 90 of the same file already uses the correct pattern (`${TT_METAL_HOME:-.}/generated_profile_log_${TEST_NAME}/ops_perf_results_${TIMESTAMP}.csv`), so the shell snippet was fixed but the Python example was not. A reader who copies this test and runs it will see the `assert CSV_PATH.exists()` fail with `FileNotFoundError` on every run because the path `tt_metal/tools/profiler/logs/ops_perf_results.csv` is never created.
- Fix: Replace the hardcoded path assignment with a glob that finds the most recent matching CSV: `import glob; csv_files = sorted(glob.glob("generated_profile_log_*/ops_perf_results_*.csv")); CSV_PATH = pathlib.Path(csv_files[-1]) if csv_files else None`, and update the subsequent assertions to handle the `None` case with a clear error message.

---

**Requirements verified as consistent (no issues):**

1. Pass 9 Issues 1 and 2 re-checked: `ch3_csv_reference/pm_ideal_and_fpu_util.md` lines 123 and 138 both read `stall_cycles = TRISC1_KERNEL_DURATION_cycles − PM_IDEAL_cycles`; no `(1 − FPU_UTIL)` form present. `ch4_compute_vs_bandwidth/index.md` line 11 reads "the CSV columns produced by the device profiler"; no "Tracy profiler" attribution present. Both Pass 9 fixes confirmed.
2. `stall_cycles` formula — direct subtraction form `TRISC1_KERNEL_DURATION_cycles − PM_IDEAL_cycles` — appears only in `pm_ideal_and_fpu_util.md` and nowhere else in the guide; no downstream chapter redefines or contradicts it.
3. FPU peak throughput and ridge point — 256 FLOPs/cycle (BF16, HiFi4) and AI_ridge = 8.0 FLOPs/byte — are consistent between `ch4/roofline_model_primer.md` (line 97 final value and lines 120–123) and `ch5/csv_signatures.md` (uses AI > 8.0 throughout). The ridge point derivation is correct despite the wrong intermediate tile FLOPs in Issue 1 above.
4. TRISC role assignments — TRISC0 = math unpacker, TRISC1 = math engine, TRISC2 = math packer, NCRISC = NoC DMA reader — consistent across all six chapters with no remaining NCRISC/TRISC0 confusion.
5. `total_op_latency = host_dispatch_time + device_kernel_time + sync_overhead` — identical across Ch1 `two_profilers_compared.md`, Ch2 `index.md`, Ch6 `index.md`, Ch6 `what_is_dispatch_overhead.md`, and Ch6 `measuring_dispatch_vs_kernel.md`.

---

## Final Pass 11

Three factual errors found. All prior-pass fixes confirmed in place; no regressions detected.

---

**Issue 1 — `csv_column_definitions.md` comment claims the CSV lands in the raw-log directory (wrong path in prose comment)**

- File: `ch2_invocation/capture_workflow.md`, line 88
- Error: Inside the "Verifying Successful Capture" bash block, line 88 is the comment `# The CSV is written to tt_metal/tools/profiler/logs/ relative to the repo root.` This is wrong. `tt_metal/tools/profiler/logs/` is where the raw per-core cycle-counter log files land (one file per Tensix core). The final `ops_perf_results.csv` lands at `generated_profile_log_{test_name}/ops_perf_results_{timestamp}.csv`, as correctly shown two lines later in the actual shell variable assignment at line 90. The erroneous comment is the first thing a reader reads in that verification block; a reader who stops at the comment will navigate to the wrong directory and find only raw logs, not the CSV. This also contradicts the Output Artifacts table on line 103 of the same file, which correctly states `generated_profile_log_{test_name}/ops_perf_results_{timestamp}.csv`.
- Fix: Replace the line 88 comment with `# The CSV is written to generated_profile_log_{test_name}/ relative to the working directory.`

---

**Issue 2 — `csv_column_definitions.md` states `device_params.json` is written to the wrong location (wrong path)**

- File: `ch3_csv_reference/csv_column_definitions.md`, line 105
- Error: The note reads: "`process_ops_logs.py` writes a `device_params.json` file alongside the CSV under `tt_metal/tools/profiler/logs/`." The known-correct CSV path is `generated_profile_log_{test_name}/ops_perf_results_{timestamp}.csv`. The `device_params.json` file is written alongside the CSV — meaning it is in `generated_profile_log_{test_name}/`, not in `tt_metal/tools/profiler/logs/`. A reader following this instruction to find the authoritative clock frequency will look in `tt_metal/tools/profiler/logs/`, find only raw per-core log files, and conclude `device_params.json` does not exist — and therefore fall back to an assumed clock frequency rather than the authoritative measured value. This is particularly harmful because the section's purpose is to teach readers how to obtain the exact clock frequency for accurate cycle-to-ns conversion.
- Fix: Replace "alongside the CSV under `tt_metal/tools/profiler/logs/`" with "alongside the CSV under `generated_profile_log_{test_name}/`".

---

**Issue 3 — `causes_of_low_fpu_util.md` Cause 4 incorrectly attributes DRAM/NoC data movement to TRISC0 (factual error)**

- File: `ch5_low_fpu_util/causes_of_low_fpu_util.md`, line 112
- Error: The mechanism paragraph for Cause 4 describes "TRISC0 (unpacker): reads tiles from L1 (or DRAM via the NoC) and reformats them for the math engine." The parenthetical "(or DRAM via the NoC)" is incorrect. TRISC0 is the math unpacker — its role is to read tiles that are already resident in L1 and reformat them into the FPU's source registers. TRISC0 does not issue NoC DMA transfers and does not read from DRAM. DRAM-to-L1 data movement is performed exclusively by NCRISC (the NoC RISC), which issues DMA descriptors to pull data from DRAM into L1 before TRISC0 can unpack it. This assignment is established as the authoritative role definition in Ch3 `csv_column_definitions.md` lines 55–56 ("NCRISC drives NoC DMA transfers") and was corrected in multiple prior passes for `classification_method.md` and `worked_examples.md`. The Cause 4 prose re-introduces the same TRISC0/NCRISC confusion that prior passes corrected elsewhere. A reader diagnosing Cause 4 who reads this description will believe TRISC0 is doing DRAM fetches, and may confuse the Cause 4 signature (`TRISC0 DURATION > 1.2 × TRISC1 DURATION`) with DRAM bandwidth saturation (Cause 5), delaying correct diagnosis.
- Fix: Replace "reads tiles from L1 (or DRAM via the NoC)" with "reads tiles from L1" in the TRISC0 role description. Add a clarifying sentence: "NCRISC — running concurrently on the same Tensix core — is responsible for transferring tiles from DRAM into L1 via NoC DMA before TRISC0 can unpack them."

---

**Requirements verified as consistent (no issues):**

1. Pass 10 Issues 1 and 2 re-checked: `ch4_compute_vs_bandwidth/roofline_model_primer.md` line 97 correctly states "32 × 32 × 32 = 32,768 MACs = 65,536 FLOPs per tile-triplet" and "256 cycles per tile-triplet"; `ch2_invocation/capture_workflow.md` Python example (lines 182–187) uses `glob` to find the CSV under `generated_profile_log_*/ops_perf_results_*.csv` rather than a hardcoded static path. Both Pass 10 fixes confirmed in place.
2. `FPU UTIL` definition — `PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles` — consistent across Ch3 `csv_column_definitions.md` line 66, Ch3 `pm_ideal_and_fpu_util.md` line 73, Ch4 `classification_method.md` lines 43–46, Ch5 `index.md` (references without redefining), and Ch5 `csv_signatures.md` (used in thresholds consistently). No contradiction.
3. TRISC role assignments (TRISC0 = unpacker, TRISC1 = math engine, TRISC2 = packer, NCRISC = NoC DMA reader) — verified consistent across Ch3 `csv_column_definitions.md`, Ch4 `classification_method.md` Step 5 role table, Ch4 `worked_examples.md`, and Ch5 `csv_signatures.md`. The only remaining violation is Issue 3 above (Ch5 `causes_of_low_fpu_util.md` line 112 parenthetical).
4. Ridge point AI_ridge = 8.0 FLOPs/byte — derived correctly as 256 FLOPs/cycle ÷ 32 bytes/cycle in `ch4/roofline_model_primer.md` lines 120–123 and used consistently as the classification threshold in `ch4/classification_method.md` Step 1 and throughout `ch5/csv_signatures.md`.
5. CSV output path `generated_profile_log_{test_name}/ops_perf_results_{timestamp}.csv` — correctly stated in `ch2/capture_workflow.md` Output Artifacts table (line 103), `ch2/index.md` Step 5, and `ch2/capture_workflow.md` line 271; the erroneous prose comment at line 88 (Issue 1 above) is the only remaining violation of this path.

---

## Final Pass 12

Two factual errors found. All prior-pass fixes confirmed in place; no regressions detected.

---

**Issue 1 — `roofline_model_primer.md` states "64 FLOPs/cycle" as the peak FPU throughput, directly contradicting the known-correct fact of 256 FLOPs/cycle (factual error)**

- File: `ch4_compute_vs_bandwidth/roofline_model_primer.md`, line 95
- Error: The FPU throughput section reads: "Peak FPU throughput: **32 × 2 = 64 FLOPs/cycle** for simple element-wise paths." This is presented as a standalone "peak FPU throughput" number. The known-correct fact is 128 FMA ops/cycle = 256 FLOPs/cycle. The subsequent paragraph (line 97) correctly establishes 256 FLOPs/cycle for the matmul path, but line 95 has already stated a contradictory "peak" of 64 FLOPs/cycle. A reader who reads line 95 and stops — or who uses line 95 to reason about the elementwise case — will use a figure that is 4× too low. The line 95 figure also contradicts the SIMD derivation: 512 bits / 16 bits = 32 elements, and 32 FMA ops × 2 FLOPs/FMA = 64 FLOPs per FMA-issue cycle, but the FPU issues 4 FMAs per cycle (not 1), giving 128 FMA ops/cycle = 256 FLOPs/cycle. The "32 × 2 = 64" arithmetic conflates a single FMA issue with the full per-cycle throughput.
- Fix: Replace line 95 with a factually correct statement. The 512-bit SIMD with 32 BF16 elements describes the datapath width, not the per-cycle FMA issue rate. The correct characterization is: "The matmul engine issues 128 FMA operations per cycle (four simultaneous 32-element FMA operations), yielding **128 × 2 = 256 FLOPs/cycle** at peak throughput." Remove the intermediate "64 FLOPs/cycle" claim entirely to prevent reader confusion.

---

**Issue 2 — `causes_of_low_fpu_util.md` Cause 4 and Cause 5 attribute DRAM reads directly to TRISC0, contradicting the known-correct fact that TRISC0 reads ONLY from L1 (factual error)**

- File: `ch5_low_fpu_util/causes_of_low_fpu_util.md`, lines 116, 124, and 132 (Cause 4); line 142 (Cause 5)
- Error: The Cause 4 mechanism paragraph (line 116) states: "If TRISC0 is slow — because the tiles are not yet in L1 **and must be fetched from DRAM** — TRISC1 stalls." Line 124 refers to "DRAM reads that TRISC0 is waiting on." Line 132 says "place weight tensors in L1-sharded memory so that TRISC0 reads from local L1 rather than DRAM." Line 142 (Cause 5) says "TRISC0 stalls waiting for data" in the context of DRAM reads being queued. The known-correct fact is unambiguous: TRISC0 is the math unpacker — it reads ONLY from L1, never from DRAM directly. DRAM-to-L1 transfers are performed exclusively by NCRISC (the NoC DMA reader). TRISC0 stalls not because it is fetching from DRAM, but because NCRISC has not yet completed the DMA transfer that would make the tiles available in L1. The distinction matters: a reader who internalises "TRISC0 fetches from DRAM" will have a fundamentally wrong model of the Tensix pipeline and will be unable to correctly diagnose or fix DRAM-fetch latency bottlenecks. The fix for Cause 4 (increasing `in0_block_w` to hide DRAM fetch latency) also makes no sense if the reader believes TRISC0 is doing the DRAM reads, since `in0_block_w` controls how far ahead NCRISC prefetches — not how much TRISC0 buffers.
- Fix: In the Cause 4 mechanism (line 116), replace "because the tiles are not yet in L1 and must be fetched from DRAM" with "because NCRISC has not yet completed the DMA transfer that would place those tiles in L1." In line 124, replace "DRAM reads that TRISC0 is waiting on" with "DRAM reads that NCRISC is performing before TRISC0 can unpack." In line 132, replace "so that TRISC0 reads from local L1 rather than DRAM" with "so that NCRISC does not need to DMA from DRAM — tiles are already in L1 for TRISC0 to unpack." In Cause 5 line 142, replace "TRISC0 stalls waiting for data" with "TRISC0 stalls waiting for NCRISC to deliver tiles from DRAM into L1."

---

**Requirements verified as consistent (no issues):**

1. Pass 11 Issues 1, 2, and 3 re-checked: `ch2_invocation/capture_workflow.md` line 88 comment correctly reads "The CSV is written to generated_profile_log_{test_name}/"; `ch3_csv_reference/csv_column_definitions.md` line 105 correctly names `generated_profile_log_{test_name}/` as the `device_params.json` location; `ch5_low_fpu_util/causes_of_low_fpu_util.md` Cause 4 TRISC0 role description correctly reads "reads tiles from L1 and reformats them for the math engine" with no "(or DRAM via the NoC)" parenthetical in the role bullet. All three Pass 11 fixes confirmed applied.
2. `FPU UTIL` definition — `PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles` — consistent across Ch3 `csv_column_definitions.md` line 66, Ch3 `pm_ideal_and_fpu_util.md` line 73, Ch4 `classification_method.md` lines 43–46, and Ch5 throughout. No contradiction.
3. `stall_cycles = TRISC1_KERNEL_DURATION_cycles − PM_IDEAL_cycles` — in direct subtraction form only in Ch3 `pm_ideal_and_fpu_util.md` lines 123 and 138. No `(1 − FPU_UTIL)` form present in either location.
4. CSV output path `generated_profile_log_{test_name}/ops_perf_results_{timestamp}.csv` — consistent across `ch2/capture_workflow.md` Output Artifacts table (line 103), `ch2/index.md` Step 5, `ch2/capture_workflow.md` Python example (glob pattern), and `ch3/csv_column_definitions.md`. No remaining static `tt_metal/tools/profiler/logs/ops_perf_results.csv` references in the content files.
5. TRISC role assignments (TRISC0 = math unpacker reading from L1, TRISC1 = math engine, TRISC2 = packer, NCRISC = NoC DMA reader from DRAM) — consistent across Ch3 `csv_column_definitions.md` lines 55–58, Ch4 `classification_method.md` Step 5 role table, Ch4 `worked_examples.md`, and Ch5 `csv_signatures.md`. The two remaining violations are Issues 1 and 2 above in `roofline_model_primer.md` and `causes_of_low_fpu_util.md`.

---

## Final Pass 13

One factual error found. All prior-pass fixes confirmed in place; no regressions detected.

---

**Issue 1 — Cause 4 mechanism describes "three" RISC processors per Tensix core, contradicting the known-correct five-RISC architecture (factual error)**

- File: `ch5_low_fpu_util/causes_of_low_fpu_util.md`, line 110
- Error: The Cause 4 mechanism section opens with "Each Tensix core runs **three** independent RISC-V processors: TRISC0, TRISC1, TRISC2." Wormhole B0 Tensix cores have **five** RISC-V processors: BRISC, NCRISC, TRISC0, TRISC1, TRISC2. Ch3 `csv_column_definitions.md` line 121 is explicit: "each core runs **five** independent RISC processors (BRISC, NCRISC, TRISC0, TRISC1, TRISC2)." The incorrect count in Ch5 omits BRISC (dispatch/control) and NCRISC (NoC DMA reader) — the very processor that is central to Cause 4's mechanism (NCRISC delivers tiles from DRAM to L1 for TRISC0 to unpack). A reader who absorbs the "three RISC" framing and then reads the Cause 4 mechanism sentence that mentions NCRISC (line 116: "NCRISC performs the DRAM-to-L1 DMA") will encounter an internal contradiction within the same section: NCRISC is described as performing DMA but was not listed as one of the core's processors two lines earlier.
- Fix: Replace "Each Tensix core runs three independent RISC-V processors: TRISC0, TRISC1, TRISC2." with "Each Tensix core runs five independent RISC-V processors. For this cause the three math-pipeline processors are the relevant ones: TRISC0 (unpacker), TRISC1 (math), and TRISC2 (packer). NCRISC — also present on every core — performs the DRAM-to-L1 DMA transfers that feed TRISC0." This aligns with Ch3 `csv_column_definitions.md` line 121 and avoids contradicting the NCRISC reference two lines below.

---

**Requirements verified as consistent (no issues):**

1. Pass 12 Issues 1 and 2 re-checked: `ch4_compute_vs_bandwidth/roofline_model_primer.md` line 95 correctly reads "Peak FPU throughput: **128 FMA ops/cycle = 256 FLOPs/cycle** for all op paths (matmul and element-wise alike)." No "64 FLOPs/cycle" claim present. `ch5_low_fpu_util/causes_of_low_fpu_util.md` line 116 correctly reads "because NCRISC has not yet delivered tiles to L1 (NCRISC performs the DRAM-to-L1 DMA; TRISC0 reads only from L1)." No TRISC0-reads-DRAM language present. Both Pass 12 fixes confirmed applied.
2. `FPU UTIL` definition — `PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles` — consistent across Ch3 `csv_column_definitions.md` line 66, Ch3 `pm_ideal_and_fpu_util.md` line 73, Ch4 `classification_method.md` lines 43–46, and Ch5 throughout. No contradiction.
3. `stall_cycles = TRISC1_KERNEL_DURATION_cycles − PM_IDEAL_cycles` — direct subtraction form in Ch3 `pm_ideal_and_fpu_util.md` lines 123 and 138. No `(1 − FPU_UTIL)` form present.
4. Tile FLOPs and cycles per tile-triplet — 2 × 32 × 32 × 32 = 65,536 FLOPs per tile-triplet; 65,536 ÷ 256 = 256 cycles per tile-triplet — stated correctly in `ch4/roofline_model_primer.md` line 97 and consistent with `ch3/pm_ideal_and_fpu_util.md` lines 29–36.
5. Ridge point — AI_ridge = 8.0 FLOPs/byte derived as 256 FLOPs/cycle ÷ 32 bytes/cycle — stated in `ch4/roofline_model_primer.md` lines 120–123 and used consistently as the threshold throughout Ch4 `classification_method.md` and Ch5 `csv_signatures.md`. No contradiction.

## Final Pass 14

**Issue 1 — `pm_ideal_and_fpu_util.md` table lists `FPU peak ops/cycle` values of 512 and 1024, which exceed the known-correct hardware ceiling of 256 FLOPs/cycle for ALL op paths**

- File: `ch3_csv_reference/pm_ideal_and_fpu_util.md`, lines 40–46
- Error: The table column is labeled "FPU peak ops/cycle (per core)" and lists 256 for BF16/HiFi4, 512 for BF16/HiFi2, 512 for BFLOAT8_B/HiFi2, and 1024 for BFLOAT8_B/LoFi. The known-correct fact is that Wormhole B0 has 128 FMA ops/cycle = 256 FLOPs/cycle for ALL op paths. The hardware FPU cannot physically deliver more than 256 FLOPs/cycle regardless of math fidelity or data format. Presenting 512 and 1024 as "FPU peak ops/cycle" is factually incorrect. What lower fidelity modes actually do is reduce the number of FMA loop iterations per tile (as correctly explained in `ch5_low_fpu_util/causes_of_low_fpu_util.md` at line 73: "HiFi4 runs ~4× more FMA iterations per tile than LoFi"). The mathematical effect on `compute_cycles` is equivalent to treating the fidelity-adjusted value as a higher effective throughput divisor, but the table's column header claims these are hardware-level FPU peaks, which is false. A reader who reads this table and concludes "BF8/LoFi hardware peak is 1024 FLOPs/cycle" has absorbed a 4× wrong number for the hardware ceiling.
- Fix: Rename the column to "Effective throughput for PM IDEAL computation (FLOPs/cycle)" and add a note clarifying that the values above 256 reflect reduced fidelity loop iterations per tile rather than an increase in the hardware FPU clock rate or issue width. Alternatively, reframe the table to show the compute_cycles multiplier relative to the BF16/HiFi4 baseline (1×, 0.5×, 0.5×, 0.25×), consistent with the `remediation_levers.md` table. The hardware FPU peak of 256 FLOPs/cycle should be stated as a fixed constant, with fidelity affecting the number of tile passes rather than the hardware throughput ceiling.

---

**Requirements verified as consistent (no issues):**

1. `FPU UTIL = PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles` — consistent across all chapters. No contradiction found.
2. `stall_cycles = TRISC1_KERNEL_DURATION_cycles − PM_IDEAL_cycles` — stated correctly in `ch3/pm_ideal_and_fpu_util.md` lines 123 and 138. No `(1 − FPU_UTIL)` form.
3. TRISC0 reads ONLY from L1 (never DRAM) — confirmed in `ch5/causes_of_low_fpu_util.md` line 116: "NCRISC performs the DRAM-to-L1 DMA; TRISC0 reads only from L1." No violation found.
4. NCRISC = NoC DMA reader (ONLY entity reading DRAM/NoC → L1) — consistent in Ch4 `classification_method.md` line 111 and Ch5 `causes_of_low_fpu_util.md` throughout. No contradiction.
5. Five RISC-V processors per Tensix core (BRISC, NCRISC, TRISC0, TRISC1, TRISC2) — `ch5/causes_of_low_fpu_util.md` line 110 now correctly reads "**five** independent RISC-V processors." Prior Pass 13 fix confirmed in place.

---

## Final Pass 15

Two factual errors found. All prior-pass fixes confirmed in place; no regressions detected.

---

**Issue 1 — `classification_method.md` calls NCRISC a "TRISC processor" and states the wrong count (factual error)**

- File: `ch4_compute_vs_bandwidth/classification_method.md`, line 107
- Error: The Step 5 preamble reads "The **three** TRISC processors on each Tensix core have distinct roles:" but the table immediately below lists **four** entries: NCRISC, TRISC0, TRISC1, and TRISC2. NCRISC is not a TRISC processor — it is the NoC RISC processor, a distinct fifth RISC-V core on each Tensix tile. The "three" count is wrong (the table has four rows), and labeling NCRISC as a "TRISC processor" contradicts the known-correct architecture (five processors: BRISC, NCRISC, TRISC0, TRISC1, TRISC2) established in Ch3 `csv_column_definitions.md` line 121 and Ch5 `causes_of_low_fpu_util.md` line 110. A reader who internalises "NCRISC is one of the three TRISC processors" will have a wrong mental model of the core architecture and will have trouble reconciling the preamble count with the four-row table directly below it.
- Fix: Replace "The **three** TRISC processors on each Tensix core have distinct roles:" with "Each Tensix core runs five RISC-V processors; the four relevant to kernel duration analysis are listed below:" (BRISC is omitted here because its duration is not the focus of the classification, but NCRISC must not be called a TRISC).

---

**Issue 2 — `causes_of_low_fpu_util.md` Cause 3 incorrectly claims PM IDEAL ignores math fidelity (contradicts `pm_ideal_and_fpu_util.md` table)**

- File: `ch5_low_fpu_util/causes_of_low_fpu_util.md`, line 77
- Error: The Cause 3 mechanism states: "The `PM IDEAL` value in the CSV is computed assuming the device's hardware FMA rate **without accounting for fidelity-induced overhead**. This means `PM_IDEAL / TRISC1_DURATION` (i.e., `FPU UTIL`) will appear to be about 0.25 of what it would be at `LoFi`." This is inconsistent with `ch3_csv_reference/pm_ideal_and_fpu_util.md`, which explicitly provides a fidelity-aware table for computing PM IDEAL (BF16/HiFi4 = 256 effective FLOPs/cycle, BF16/HiFi2 = 512, BFLOAT8_B/LoFi = 1024). If PM IDEAL uses the fidelity-adjusted effective throughput as stated in Ch3, then for a compute-bound op at HiFi4 PM IDEAL accounts for the 4× longer TRISC1 duration by using the lower effective throughput (256 vs. 1024), and FPU UTIL would be the same at HiFi4 and LoFi — not 0.25× as Cause 3 claims. The "0.25×" result only follows if PM IDEAL uses a fixed 256 FLOPs/cycle regardless of fidelity (which is what Cause 3 implies but Ch3 contradicts). The two chapters give incompatible descriptions of how PM IDEAL handles fidelity, leaving readers unable to determine whether low FPU UTIL at HiFi4 represents a real problem or expected behaviour.
- Fix: Align Cause 3 with the Ch3 table. Since the Ch3 table shows PM IDEAL does use fidelity-adjusted effective throughput, a well-tuned compute-bound op at HiFi4 should still approach FPU UTIL ≈ 1.0 (PM IDEAL accounts for the extra iterations). The correct diagnostic signal for Cause 3 is therefore not "FPU UTIL ≈ 0.25" but rather a `MATH FIDELITY == "HiFi4"` entry combined with `TRISC1 KERNEL DURATION` that is 4× longer than an equivalent LoFi run — indicating that unnecessary precision is being spent, not that the FPU itself is stalling. Revise the Cause 3 mechanism to reflect the Ch3 definition of PM IDEAL and remove the incorrect "without accounting for fidelity-induced overhead" sentence.

---

**Requirements verified as consistent (no issues):**

1. All Pass 14 fixes confirmed in place: `ch3_csv_reference/pm_ideal_and_fpu_util.md` table column is labeled "Effective FLOPs/cycle for PM IDEAL" with the note that values above 256 reflect reduced loop iterations, not a higher hardware FPU ceiling. No "FPU peak ops/cycle" mislabeling present.
2. `stall_cycles = TRISC1_KERNEL_DURATION_cycles − PM_IDEAL_cycles` — direct subtraction form present in `ch3/pm_ideal_and_fpu_util.md` lines 123 and 138. No `(1 − FPU_UTIL)` form present.
3. Tile FLOPs and cycles per tile-triplet — 65,536 FLOPs per tile-triplet; 256 cycles per tile-triplet — stated correctly in `ch4/roofline_model_primer.md` line 97, consistent with `ch3/pm_ideal_and_fpu_util.md` lines 29–36.
4. Ridge point AI_ridge = 8.0 FLOPs/byte — derived as 256 FLOPs/cycle ÷ 32 bytes/cycle in `ch4/roofline_model_primer.md` lines 120–123; used consistently as the threshold in Ch4 `classification_method.md` and Ch5 `csv_signatures.md`. No contradiction.
5. TRISC0 reads ONLY from L1 (NCRISC is the sole entity reading from DRAM/NoC → L1) — confirmed in `ch5/causes_of_low_fpu_util.md` line 118: "NCRISC performs the DRAM-to-L1 DMA; TRISC0 reads only from L1." No remaining TRISC0-reads-DRAM language.

## Final Pass 16

One factual error found. All Pass 15 fixes confirmed in place.

---

**Pass 15 fix confirmation:**

- Pass 15 Issue 1: `ch4_compute_vs_bandwidth/classification_method.md` line 107 now reads "The four per-core RISC processors profiled in the CSV have distinct roles (BRISC is excluded — it does not have per-cycle counters in the ops_perf output):" — no "three TRISC processors" language present. Fix confirmed.
- Pass 15 Issue 2: `ch5_low_fpu_util/causes_of_low_fpu_util.md` Cause 3 mechanism (lines 75–77) now correctly states "`FPU UTIL` (= PM_IDEAL / TRISC1_KERNEL_DURATION) remains roughly constant across fidelity modes" — no "without accounting for fidelity-induced overhead" or "0.25×" claim present. Fix confirmed.

---

**Issue 1 — `csv_signatures.md` Cause 3 still claims `FPU UTIL ≈ 0.25 × LoFi_util` at HiFi4, contradicting the known-correct fact and the sibling file `causes_of_low_fpu_util.md`**

- File: `ch5_low_fpu_util/csv_signatures.md`, lines 94 and 99
- Error: The Cause 3 CSV pattern block reads:
  ```
  MATH FIDELITY == "HiFi4"
  FPU UTIL ≈ 0.25 × LoFi_util
  ```
  The prose below (line 99) repeats: "`HiFi2` would produce approximately half the `FPU UTIL` of `LoFi` (factor of ~0.5 rather than ~0.25)."

  This is factually wrong. The known-correct fact is: PM IDEAL scales proportionally with fidelity; FPU_UTIL stays roughly constant across fidelity for a well-tuned op. The sibling file `causes_of_low_fpu_util.md` (fixed in Pass 15 Issue 2) now correctly states on line 75: "FPU UTIL (= PM_IDEAL / TRISC1_KERNEL_DURATION) remains roughly constant across fidelity modes." The Ch3 PM IDEAL table (`pm_ideal_and_fpu_util.md` lines 40–46) explains the mechanism: BF16/HiFi4 uses 256 effective FLOPs/cycle and BF16/LoFi would use 1024, so both PM_IDEAL_cycles and TRISC1_KERNEL_DURATION scale by the same factor, leaving FPU_UTIL unchanged.

  A reader using `csv_signatures.md` to check for Cause 3 will look for `FPU UTIL ≈ 0.15–0.25` on a HiFi4 run. Because FPU UTIL is actually near its LoFi value (not 0.25× of it), they will not find the pattern and will incorrectly rule out Cause 3, potentially escalating to a kernel-level investigation (Cause 7) when the real problem is simply an over-precise fidelity setting.

- Fix: Replace the Cause 3 CSV pattern and prose with the correct diagnostic signal. The pattern should be:
  ```
  MATH FIDELITY == "HiFi4"
  TRISC1 KERNEL DURATION [ns] ≈ 4× the duration of an equivalent LoFi run
  FPU UTIL is approximately the same as a LoFi run (not reduced by ~0.25×)
  ```
  Update the prose to match: "At HiFi4, PM IDEAL scales up by the same factor as TRISC1 duration, so FPU UTIL stays roughly the same as at LoFi. The diagnostic signal for Cause 3 is the absolute TRISC1 duration being ~4× longer than a LoFi baseline — not a depressed FPU UTIL."

---

**Requirements verified as consistent (no new issues):**

1. Pass 15 fixes confirmed — see above.
2. `FPU UTIL = PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles` — consistent across Ch3 `pm_ideal_and_fpu_util.md` line 73, Ch3 `csv_column_definitions.md` line 66, Ch4 `classification_method.md` lines 43–46, Ch5 `causes_of_low_fpu_util.md` throughout. No contradiction.
3. Hardware FPU ceiling — 128 FMA ops/cycle = 256 FLOPs/cycle — stated in `ch4/roofline_model_primer.md` line 97 with the note that this is the hardware ceiling for all op paths; fidelity-adjusted values in the PM IDEAL table are labeled "Effective FLOPs/cycle for PM IDEAL" with a note clarifying they are not hardware peaks. No violation.
4. BRISC excluded from CSV per-cycle counters — confirmed in `ch4/classification_method.md` line 107 and `ch3/csv_column_definitions.md` (BRISC duration column present but consistent with known-correct fact that CSV profiles four processors, BRISC excluded from per-cycle counter data).
5. `stall_cycles = TRISC1_KERNEL_DURATION_cycles − PM_IDEAL_cycles` — confirmed in `ch3/pm_ideal_and_fpu_util.md` lines 123 and 138.

## Final Pass 17

Three factual errors found. Pass 16 Issue 1 fix confirmed in place.

---

**Pass 16 fix confirmation:**

- Pass 16 Issue 1: `ch5_low_fpu_util/csv_signatures.md` Cause 3 CSV pattern block now correctly reads "TRISC1 KERNEL DURATION [ns] (HiFi4 run) ≈ 4 × TRISC1 KERNEL DURATION [ns] (LoFi baseline)" and the prose states "`FPU UTIL` stays roughly constant across fidelity modes for a well-tuned op — **do not expect low FPU UTIL as the signature for this cause.**" No "FPU UTIL ≈ 0.25 × LoFi_util" language present. Fix confirmed.

---

**Issue 1 — `two_profilers_compared.md` line 9 incorrectly includes BRISC in the set of processors profiled by the device profiler**

- File: `ch1_tracy_fundamentals/two_profilers_compared.md`, line 9
- Error: The sentence reads: "It uses hardware cycle counters built into each Tensix core to record when kernel code begins and ends execution on each RISC processor **(BRISC, NCRISC, TRISC0, TRISC1, TRISC2)**." The known-correct fact is that CSV ops_perf profiles four per-core processors (BRISC excluded — no per-cycle counters). BRISC does not appear in the per-processor duration columns of `ops_perf_results.csv`. Including BRISC in the parenthetical gives a reader a false expectation that the CSV will contain a BRISC cycle-counter column; when they do not find it, they will assume a data-collection failure rather than understanding it was never collected.
- Fix: Replace the parenthetical with the four profiled processors: "(NCRISC, TRISC0, TRISC1, TRISC2)" and add a note that BRISC is excluded from per-cycle profiling because it does not have per-cycle counters in the ops_perf output.

---

**Issue 2 — `csv_column_definitions.md` line 54 lists `BRISC KERNEL DURATION [ns]` as a CSV column, contradicting the known-correct fact that BRISC is excluded from per-cycle profiling**

- File: `ch3_csv_reference/csv_column_definitions.md`, line 54
- Error: The Duration Columns table contains a row for `BRISC KERNEL DURATION [ns]` with the description "Time the Base RISC (BRISC) processor spent executing kernel code on the representative core." The known-correct fact is that CSV ops_perf profiles four per-core processors (BRISC excluded — no per-cycle counters). This row should not exist in the column reference because BRISC does not produce a per-cycle duration entry in `ops_perf_results.csv`. Pass 16 Item 4 observed this column and incorrectly dismissed it as "consistent with known-correct fact that CSV profiles four processors, BRISC excluded from per-cycle counter data" — those two statements are self-contradictory, not consistent. A reader using this column table as a reference will look for `BRISC KERNEL DURATION [ns]` in the actual CSV output and not find it, creating confusion.
- Fix: Remove the `BRISC KERNEL DURATION [ns]` row from the Duration Columns table, or replace it with a note explaining that BRISC does not have per-cycle hardware counters and therefore does not produce a duration column in `ops_perf_results.csv`.

---

**Issue 3 — `roofline_model_primer.md` line 93 states "32 elements per cycle" from a 512-bit SIMD, which is internally inconsistent with the claimed 128 FMA ops/cycle on line 95**

- File: `ch4_compute_vs_bandwidth/roofline_model_primer.md`, lines 93–95
- Error: Line 93 reads: "SIMD width: 512 bits / 16 bits per element = **32 elements per cycle**." Line 95 then states: "Peak FPU throughput: **128 FMA ops/cycle = 256 FLOPs/cycle**." These two statements are arithmetically inconsistent: a 512-bit SIMD processing 16-bit elements operates on 32 elements per cycle; performing one FMA on each of those 32 elements yields 32 FMA ops/cycle, not 128. The only way to reach 128 FMA ops/cycle from a 512-bit SIMD is if the hardware issues four simultaneous FMA operations per cycle across the SIMD pipeline — a fact that is not stated. The known-correct hardware ceiling (128 FMA ops/cycle = 256 FLOPs/cycle) is correct as stated on line 95, but the intermediate derivation on line 93 ("32 elements per cycle") does not lead to it and will confuse any reader who tries to verify the arithmetic.
- Fix: Either remove the "SIMD width: 512 bits / 16 bits = 32 elements per cycle" bullet and replace with the correct derivation (e.g., "The matmul engine issues four simultaneous 32-element FMA operations per cycle, giving 128 FMA ops/cycle"), or add an explicit bridging sentence explaining that the 512-bit SIMD datapath runs four parallel FMA passes per clock cycle to achieve 128 FMA ops/cycle.

---

**Requirements verified as consistent (no new issues):**

1. Pass 16 fix confirmed — see above.
2. `FPU UTIL = PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles` — consistent across Ch3 `pm_ideal_and_fpu_util.md`, Ch3 `csv_column_definitions.md`, Ch4 `classification_method.md`, and Ch5 `causes_of_low_fpu_util.md`. No contradiction.
3. `stall_cycles = TRISC1_KERNEL_DURATION_cycles − PM_IDEAL_cycles` — confirmed in `ch3/pm_ideal_and_fpu_util.md` lines 123 and 138.
4. Ridge point AI_ridge = 8.0 FLOPs/byte — derived correctly as 256 FLOPs/cycle ÷ 32 bytes/cycle in `ch4/roofline_model_primer.md` lines 120–123; used consistently as the threshold throughout Ch4 and Ch5.
5. HiFi4 vs LoFi FPU_UTIL behavior — confirmed correct in `ch5/causes_of_low_fpu_util.md` Cause 3 and `ch5/csv_signatures.md` Cause 3; both now state FPU UTIL stays roughly constant across fidelity modes.

## Final Pass 18

One factual error found. All Pass 17 fixes confirmed in place; no regressions detected.

---

**Pass 17 fix confirmation:**

- Pass 17 Issue 1: `ch1_tracy_fundamentals/two_profilers_compared.md` line 9 now correctly reads "NCRISC, TRISC0, TRISC1, TRISC2" with an explicit note that BRISC is excluded from per-cycle profiling. Fix confirmed.
- Pass 17 Issue 2: `ch3_csv_reference/csv_column_definitions.md` line 54 BRISC row now reads "**Not present in `ops_perf_results.csv`.** BRISC does not have per-cycle counters and is not profiled in the ops_perf output." Fix confirmed.
- Pass 17 Issue 3: `ch4_compute_vs_bandwidth/roofline_model_primer.md` lines 91–94 now correctly state the FPU is a matrix engine (not a SIMD unit) and directly assert "128 BF16 FMA operations per cycle = 256 FLOPs/cycle" without the inconsistent 32-elements-per-cycle SIMD derivation. Fix confirmed.

---

**Issue 1 — `two_profilers_compared.md` line 43 "When to Use" table lists BRISC in the per-RISC duration breakdown scenario, implying the device profiler provides BRISC per-cycle data (factual error)**

- File: `ch1_tracy_fundamentals/two_profilers_compared.md`, line 43
- Error: The "When to Use" table row reads: `| Analyzing per-RISC duration breakdown (BRISC, NCRISC, TRISC0–2) | Device profiler alone | This data lives entirely on the device |`. Listing BRISC here tells the reader that BRISC per-cycle duration data is available from the device profiler. The known-correct fact is that BRISC has no per-cycle counters and does not appear in `ops_perf_results.csv` — BRISC is explicitly excluded from the CSV per-cycle profiling. Pass 17 Issue 1 fixed line 9 of the same file (the descriptive text in the opening paragraph) but left line 43 (the "When to Use" table) unmodified. The two lines are now inconsistent within the same file: line 9 correctly excludes BRISC from the profiled processor list, while line 43 includes it as analyzable via the device profiler. A reader who uses this table as a quick reference for what the device profiler can show them will look for a BRISC duration column in the CSV and not find it.
- Fix: Replace `(BRISC, NCRISC, TRISC0–2)` with `(NCRISC, TRISC0, TRISC1, TRISC2)` in line 43. Optionally add a parenthetical note: "(BRISC excluded — no per-cycle counters in ops_perf output)" to match the note added at line 9 by the Pass 17 fix.

---

**Requirements verified as consistent (no new issues):**

1. Pass 17 fixes confirmed — see above.
2. `FPU UTIL = PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles` — consistent across Ch3 `pm_ideal_and_fpu_util.md` line 73, Ch3 `csv_column_definitions.md` line 66, Ch4 `classification_method.md` line 46, and Ch5 `causes_of_low_fpu_util.md` throughout. No contradiction.
3. `stall_cycles = TRISC1_KERNEL_DURATION_cycles − PM_IDEAL_cycles` — confirmed in `ch3/pm_ideal_and_fpu_util.md` lines 123 and 138. No `(1 − FPU_UTIL)` form present.
4. Ridge point AI_ridge = 8.0 FLOPs/byte — 256 FLOPs/cycle ÷ 32 bytes/cycle — stated correctly in `ch4/roofline_model_primer.md` lines 119–123 and used consistently as the threshold across Ch4 `classification_method.md` and Ch5 `csv_signatures.md`. No contradiction.
5. TRISC0 reads ONLY from L1 (NCRISC is the sole entity reading DRAM/NoC → L1) — confirmed in `ch5/causes_of_low_fpu_util.md` line 118 and throughout. No remaining TRISC0-reads-DRAM language.

---

## Final Pass 19

One factual error found in the minimal working example code. Pass 18 fix confirmed in place; no other regressions detected.

---

**Pass 18 fix confirmation:**

- Pass 18 Issue 1: `ch1_tracy_fundamentals/two_profilers_compared.md` line 43 now reads `| Analyzing per-RISC duration breakdown (NCRISC, TRISC0, TRISC1, TRISC2) | Device profiler alone |` — BRISC is no longer listed. Fix confirmed.

---

**Issue 1 — Wrong CSV column name in the minimal working example causes a guaranteed assertion failure (factual error in code)**

- File: `ch2_invocation/capture_workflow.md`, line 240
- Error: The code example sets `duration_col = "DEVICE KERNEL DURATION"` and then tests `assert duration_col in rows[0]`. The actual column name in `ops_perf_results.csv` is `"DEVICE KERNEL DURATION [ns]"` (with the `[ns]` unit suffix), as defined in `ch3_csv_reference/csv_column_definitions.md` line 52. The truncated name `"DEVICE KERNEL DURATION"` does not match any column header, so `duration_col in rows[0]` evaluates to `False` and the assertion fires with the misleading message "Expected column 'DEVICE KERNEL DURATION' not found in CSV" — directing the reader to investigate column availability rather than the column name mismatch. Additionally, the `print` statement at line 254 labels the values as "cycles" (`Max DEVICE KERNEL DURATION: {max(durations)} cycles`) but if the column were correctly named `[ns]`, the values would be nanoseconds, not cycles — a secondary label error that compounds reader confusion.
- Fix: Replace `duration_col = "DEVICE KERNEL DURATION"` with `duration_col = "DEVICE KERNEL DURATION [ns]"` on line 240. Update the `print` label on line 254 from `"cycles"` to `"ns"` to match the column's actual unit.

---

**Requirements verified as consistent (no new issues):**

1. Pass 18 fix confirmed — see above.
2. `FPU UTIL = PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles` — consistent across Ch3 `pm_ideal_and_fpu_util.md` line 73, Ch3 `csv_column_definitions.md` line 66, Ch4 `classification_method.md` line 46, and Ch5 `causes_of_low_fpu_util.md` throughout.
3. Ridge point AI_ridge = 8.0 FLOPs/byte (256 FLOPs/cycle ÷ 32 bytes/cycle) — stated correctly in `ch4/roofline_model_primer.md` lines 119–123 and used consistently in Ch4 `classification_method.md` and Ch5 `csv_signatures.md`.
4. `DEVICE KERNEL DURATION [ns]` and `DEVICE KERNEL DURATION [cycle]` column names — used correctly (with unit suffix) in all narrative text across Ch3, Ch4, Ch5, and Ch6. The error is isolated to the single code example in Ch2.
5. stall_cycles = TRISC1_KERNEL_DURATION_cycles − PM_IDEAL_cycles — confirmed correct in `ch3/pm_ideal_and_fpu_util.md` lines 123 and 138.

---

## Final Pass 20

One factual error found. Pass 19 fix confirmed in place; no other regressions detected.

---

**Pass 19 fix confirmation:**

- Pass 19 Issue 1: `ch2_invocation/capture_workflow.md` line 240 now reads `duration_col = "DEVICE KERNEL DURATION [ns]"` (with `[ns]` unit suffix), and line 253 labels the output `"ns"` rather than `"cycles"`. Fix confirmed.

---

**Issue 1 — Ch5 index.md summary table Cause 3 row claims `FPU UTIL ≈ 0.25 × LoFi_util` as the CSV signature, which is factually incorrect**

- File: `ch5_low_fpu_util/index.md`, line 38
- Error: The summary table row for Cause 3 (Math fidelity mismatch) lists the key CSV signature as `MATH FIDELITY == "HiFi4"`, `FPU UTIL ≈ 0.25 × LoFi_util`. This is wrong. The known-correct fact is that FPU_UTIL stays roughly constant across fidelity modes — switching from LoFi to HiFi4 increases both `PM_IDEAL_cycles` and `TRISC1_KERNEL_DURATION_cycles` by the same factor (~4×), so their ratio (FPU_UTIL = PM_IDEAL / TRISC1_DURATION) is largely unchanged. The correct diagnostic signal for Cause 3 is that `TRISC1 KERNEL DURATION [ns]` is approximately 4× longer for a HiFi4 run compared to a LoFi baseline — not a 4× reduction in FPU_UTIL. The guide's own detailed files (`causes_of_low_fpu_util.md` lines 75–77 and `csv_signatures.md` lines 97–101) correctly state "FPU UTIL stays roughly constant across fidelity modes for a well-tuned op — do not expect low FPU UTIL as the signature for this cause." The summary table in `index.md` directly contradicts both the known-correct fact and the guide's own detailed explanation, creating a self-inconsistency that will mislead readers using the summary table as a quick reference.
- Fix: Replace `FPU UTIL ≈ 0.25 × LoFi_util` in the Cause 3 "Key CSV Signature" cell with the correct signal: `TRISC1 KERNEL DURATION [ns] ≈ 4 × LoFi baseline` (i.e., the absolute duration is ~4× longer, not the FPU_UTIL ratio). The fixed row should read: `| 3 | Math fidelity mismatch (HiFi4 where LoFi tolerable) | MATH FIDELITY == "HiFi4", TRISC1 KERNEL DURATION [ns] ≈ 4 × LoFi baseline | Set math_fidelity=MathFidelity.LoFi |`.

---

**Requirements verified as consistent (no new issues):**

1. Pass 19 fix confirmed — see above.
2. `FPU UTIL = PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles` — definition is consistent across Ch3 `pm_ideal_and_fpu_util.md`, Ch3 `csv_column_definitions.md`, Ch4 `classification_method.md`, and Ch5 `causes_of_low_fpu_util.md`.
3. FPU_UTIL constancy across fidelity modes — correctly described in `ch5/causes_of_low_fpu_util.md` (Cause 3 mechanism, lines 75–77) and `ch5/csv_signatures.md` (Cause 3 pattern, lines 97–101). The error is isolated to the summary table in `ch5/index.md`.
4. Ridge point AI_ridge = 8.0 FLOPs/byte — confirmed correct in `ch4/roofline_model_primer.md` lines 119–123 and used consistently across Ch4 and Ch5.
5. NCRISC as the sole DRAM/NoC reader (TRISC0 reads only from L1) — confirmed correct throughout Ch5 `causes_of_low_fpu_util.md` (e.g., line 118) and Ch4 `classification_method.md`.

---

## Final Pass 21

One factual error found. Pass 20 fix confirmed in place; no other regressions detected.

---

**Pass 20 fix confirmation:**

- Pass 20 Issue 1: `ch5_low_fpu_util/index.md` line 38 now correctly lists the Cause 3 key CSV signature as `TRISC1 KERNEL DURATION [ns] ≈ 4 × LoFi baseline` instead of `FPU UTIL ≈ 0.25 × LoFi_util`. Fix confirmed.

---

**Issue 1 — Ch3 pm_ideal_and_fpu_util.md line 36: explanation of 32,768 MACs per tile-triplet contains a self-contradictory K_t factor**

- File: `ch3_csv_reference/pm_ideal_and_fpu_util.md`, line 36
- Error: The parenthetical explanation reads: "each of its 1,024 elements accumulates K_t × 32 = 32 partial products along the inner dimension per tile step, giving 32 × 32 × 32 = 32,768 MACs per tile-triplet." The expression "K_t × 32 = 32" is internally contradictory — it implies K_t = 1. K_t is defined in the same section as the total number of K-dimension tiles in the full matmul (K_t = K/32), not a per-tile-triplet count. Per tile-triplet, each output element accumulates 32 partial products because a single K-dimension tile has 32 elements along the inner dimension. K_t is not involved in this per-triplet count at all. The formula "K_t × 32 = 32" either (a) misidentifies K_t as the number of inner-dimension elements within one tile (which is always 32, not K_t tiles worth), or (b) intends to say "32 elements from the K dimension of one tile" and erroneously inserted K_t as a factor. Either way, the parenthetical is factually wrong as written: it says K_t × 32 = 32, which is only true when K_t = 1, while the rest of the section clearly treats K_t as a variable (e.g., the formula `compute_cycles = M_t × K_t × N_t × 2 × 32768 / FPU_peak` on line 29 requires K_t to be general). The result 32,768 MACs per tile-triplet is correct (32³ = 32,768); only the prose explanation of how that figure arises is wrong.
- Fix: Replace "each of its 1,024 elements accumulates K_t × 32 = 32 partial products along the inner dimension per tile step" with "each of its 1,024 elements accumulates 32 partial products from the 32 elements along the K dimension of the single input tile processed in this step". The corrected sentence should read: "each tile-triplet performs 32³ = 32,768 multiply-accumulate operations (the output tile is 32×32 and each of its 1,024 elements accumulates 32 partial products from the 32 K-dimension elements of the paired input tile, giving 32 × 32 × 32 = 32,768 MACs per tile-triplet)."

---

**Requirements verified as consistent (no new issues):**

1. Pass 20 fix confirmed — see above.
2. FPU_UTIL definition `PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles` — consistent across Ch3 `csv_column_definitions.md`, Ch3 `pm_ideal_and_fpu_util.md`, Ch4 `classification_method.md`, and Ch5 `causes_of_low_fpu_util.md`.
3. Ridge point AI_ridge = 8.0 FLOPs/byte — confirmed correct in Ch4 `roofline_model_primer.md` (256 FLOPs/cycle ÷ 32 bytes/cycle = 8.0) and consistently applied in Ch4 `classification_method.md` and Ch5.
4. Tile FLOPs result 65,536 FLOPs per tile-triplet and 256 cycles per tile-triplet — the numerical results stated in Ch4 `roofline_model_primer.md` lines 96–97 are correct and match the known-correct facts; only the prose explanation of the 32,768 MACs derivation in Ch3 is wrong (Issue 1 above).
5. BRISC exclusion from ops_perf CSV — consistently stated across Ch1 `two_profilers_compared.md`, Ch3 `csv_column_definitions.md`, and Ch4 `classification_method.md`.

---

## Final Pass 22

One factual error found. Pass 21 fix confirmed in place; no other regressions detected.

---

**Pass 21 fix confirmation:**

- Pass 21 Issue 1: `ch3_csv_reference/pm_ideal_and_fpu_util.md` line 36 now reads "each of its 1,024 elements accumulates 32 partial products from the 32 K-dimension elements of the paired input tile along the inner dimension per tile step, giving 32 × 32 × 32 = 32,768 MACs per tile-triplet." The erroneous "K_t × 32 = 32" language is absent. Fix confirmed.

---

**Issue 1 — `classification_method.md` Step 5 role table describes TRISC2 as a DRAM/NoC data writer, directly contradicting the known-correct fact that TRISC2 is the math packer (writes to L1, not DRAM/NoC)**

- File: `ch4_compute_vs_bandwidth/classification_method.md`, lines 113–114
- Error: The Step 5 role table entry reads `| TRISC2 | Data writer: moves output tiles from L1 to DRAM/NoC |`. The known-correct fact is that TRISC2 = math packer. `ch3_csv_reference/csv_column_definitions.md` (the authoritative reference for this guide) states at line 58: "TRISC2 (the math packer) spent in kernel code. TRISC2 packs computed results from output registers back into L1." TRISC2 writes to L1, not to DRAM/NoC. The entity that performs NoC DMA is NCRISC; the known-correct facts explicitly state "NCRISC = NoC DMA reader. Only entity reading DRAM/NoC → L1." Calling TRISC2 a "Data writer: moves output tiles from L1 to DRAM/NoC" is inconsistent with the role defined in Ch3 and will mislead a reader who consults the Step 5 table expecting to see which processor controls DRAM writes. The follow-on interpretation table entry (line 122) `TRISC2 DURATION >> TRISC1 DURATION → Write-bandwidth-bound: data writer is stalling the pipeline` inherits the same mislabeling.
- Fix: Replace the TRISC2 row with `| TRISC2 | Math packer: packs computed results from FPU output registers back into L1 |`, consistent with Ch3 `csv_column_definitions.md`. Update the interpretation table entry description at line 122 from "Write-bandwidth-bound: data writer is stalling the pipeline" to "Packer bottleneck: TRISC2 output buffer in L1 is full, stalling the math engine", which accurately reflects the stall mechanism for an elevated TRISC2 duration.

---

**Requirements verified as consistent (no new issues):**

1. Pass 21 fix confirmed — see above.
2. TRISC2 role in Ch3 `csv_column_definitions.md` — correctly states "math packer; packs computed results from output registers back into L1." No DRAM/NoC write claim.
3. TRISC0 role — consistently "math unpacker; reads ONLY from L1" across Ch3 `csv_column_definitions.md`, Ch4 `classification_method.md` Step 5 role table, and Ch5 `causes_of_low_fpu_util.md` line 113.
4. `FPU UTIL = PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles` — definition consistent across Ch3, Ch4, and Ch5 throughout.
5. Ridge point AI_ridge = 8.0 FLOPs/byte (256 FLOPs/cycle ÷ 32 bytes/cycle) — confirmed correct in Ch4 `roofline_model_primer.md` and used consistently in Ch4 `classification_method.md` and Ch5 `csv_signatures.md`.

---

## Final Pass 23

Two factual errors found. Pass 22 fix confirmed in place; no other regressions detected.

---

**Pass 22 fix confirmation:**

- Pass 22 Issue 1: `ch4_compute_vs_bandwidth/classification_method.md` line 114 now correctly reads `| TRISC2 | Math packer: packs computed results from FPU output registers back into L1 |` and line 122 reads "Output-buffer-stall: TRISC2 (math packer) is stalling because its L1 output buffer is full, typically because NCRISC or a downstream stage is not consuming tiles fast enough." No DRAM/NoC write characterization present. Fix confirmed.

---

**Issue 1 — `causes_of_low_fpu_util.md` Cause 4 describes TRISC2 as writing "back to L1 (or DRAM)", contradicting the known-correct fact that TRISC2 writes only to L1 (factual error)**

- File: `ch5_low_fpu_util/causes_of_low_fpu_util.md`, line 116
- Error: The TRISC2 role bullet in the Cause 4 mechanism list reads: "**TRISC2 (packer):** takes math results and writes them back to L1 **(or DRAM)**." The known-correct fact is that TRISC2 is the math packer that packs FPU results back into L1 only. High TRISC2 duration indicates the L1 output buffer is full — it is not a signal of DRAM or NoC write bandwidth. The parenthetical "(or DRAM)" contradicts the role stated in Ch3 `csv_column_definitions.md` line 58 ("TRISC2 packs computed results from output registers back into L1") and the interpretation table in `ch4/classification_method.md` line 122 (corrected in Pass 22 to "L1 output buffer full"). A reader who internalises "TRISC2 writes to DRAM" will misinterpret high `TRISC2 KERNEL DURATION` as a DRAM bandwidth bottleneck and attempt to apply DRAM-bandwidth remediations (e.g., reducing tensor size, switching to DRAM-efficient layouts) rather than the correct fix (draining the L1 output buffer — e.g., by ensuring the consumer of TRISC2's output can keep up).
- Fix: Replace "takes math results and writes them back to L1 (or DRAM)" with "takes math results and writes them back to L1" — removing the "(or DRAM)" parenthetical. This aligns with Ch3 `csv_column_definitions.md` and with the corrected interpretation in `ch4/classification_method.md`.

---

**Issue 2 — `worked_examples.md` Example 3 Step 5 characterizes TRISC2 as "the writer continuously pushing output tiles" in a bandwidth-bound context, implying TRISC2 drives NoC output traffic (factual error)**

- File: `ch4_compute_vs_bandwidth/worked_examples.md`, line 189
- Error: The Step 5 TRISC duration breakdown for the bandwidth-bound silu example reads: "`TRISC2 DURATION`: comparable to TRISC0 — **the writer is continuously pushing output tiles.**" The label "the writer... pushing output tiles" in the context of a NoC-bandwidth-bound analysis implies TRISC2 is driving the outbound data movement over the NoC. The known-correct fact is that TRISC2 is the math packer — it writes FPU output results into L1 output buffers, not over the NoC. The statement is also internally inconsistent with Example 3's own conclusion ("Bandwidth-bound (NoC). The kernel is limited by the rate at which data can be moved through the NoC") where NCRISC is the driver of NoC traffic. For a bandwidth-bound silu op, NCRISC is the long pole for NoC input reads; TRISC2's elevated duration relative to TRISC1 reflects the packer keeping pace with the output tile flow into L1, not NoC writes. Calling TRISC2 "the writer continuously pushing output tiles" alongside NCRISC "the data reader pulling input tiles" implies a symmetric NoC-I/O model that does not match the actual pipeline where TRISC2 only accesses L1.
- Fix: Replace "`TRISC2 DURATION`: comparable to TRISC0 — the writer is continuously pushing output tiles" with "`TRISC2 DURATION`: comparable to TRISC0 — the math packer is continuously writing computed results into L1 output buffers". This removes the NoC-write implication and accurately describes TRISC2's L1-resident role.

---

**Requirements verified as consistent (no new issues):**

1. Pass 22 fix confirmed — see above.
2. `FPU UTIL = PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles` — definition consistent across Ch3 `pm_ideal_and_fpu_util.md` line 73, Ch3 `csv_column_definitions.md` line 66, Ch4 `classification_method.md` lines 43–46, and Ch5 `causes_of_low_fpu_util.md` throughout. No contradiction.
3. `stall_cycles = TRISC1_KERNEL_DURATION_cycles − PM_IDEAL_cycles` — direct subtraction form confirmed in `ch3/pm_ideal_and_fpu_util.md` lines 123 and 138. No `(1 − FPU_UTIL)` form present.
4. Ridge point AI_ridge = 8.0 FLOPs/byte (256 FLOPs/cycle ÷ 32 bytes/cycle) — confirmed in `ch4/roofline_model_primer.md` lines 119–123 and used consistently in Ch4 `classification_method.md` and Ch5 `csv_signatures.md`. No contradiction.
5. TRISC0 reads ONLY from L1 (NCRISC is the sole entity reading DRAM/NoC → L1) — confirmed in `ch5/causes_of_low_fpu_util.md` line 114 ("TRISC0 (unpacker): reads tiles from L1 and reformats them for the math engine") and line 118 ("NCRISC performs the DRAM-to-L1 DMA; TRISC0 reads only from L1"). No remaining TRISC0-reads-DRAM language.

---

## Final Pass 24

One factual error found. All Pass 23 fixes confirmed in place; no regressions detected.

---

**Pass 23 fix confirmation:**

- Pass 23 Issue 1: `ch5_low_fpu_util/causes_of_low_fpu_util.md` TRISC2 role bullet in Cause 4 now reads "takes math results and writes them back to L1" with no "(or DRAM)" parenthetical. Fix confirmed.
- Pass 23 Issue 2: `ch4_compute_vs_bandwidth/worked_examples.md` line 189 now reads "`TRISC2 DURATION`: comparable to TRISC0 — the math packer is continuously writing computed results into L1 output buffers." No NoC-write implication present. Fix confirmed.

---

**Issue 1 — `classification_method.md` Step 5 role table describes NCRISC as only a data reader, contradicting the known-correct fact that NCRISC handles both DRAM/NoC reads into L1 and writes from L1 to DRAM/NoC (factual error)**

- File: `ch4_compute_vs_bandwidth/classification_method.md`, line 111
- Error: The Step 5 role table entry reads `| NCRISC | Data reader: moves input tiles from DRAM/NoC into L1 |`. The known-correct fact states: "NCRISC = NoC DMA reader/writer. Handles all DRAM/NoC traffic (both reads into L1 and writes from L1 to DRAM/NoC)." Defining NCRISC as only a "Data reader" omits the write direction entirely. In a typical matmul kernel, NCRISC also writes output tiles from L1 back to DRAM (or to neighboring cores' L1 via the NoC) once computation is complete. A reader who internalises "NCRISC = reader only" will misinterpret an elevated NCRISC duration in a write-heavy kernel as an unexplained anomaly rather than recognising it as expected NCRISC write-path activity. The incomplete description also obscures why TRISC2 and NCRISC can both show elevated duration simultaneously in an output-write scenario: TRISC2 packs results into L1, and NCRISC then writes them from L1 over the NoC — if NCRISC is slow at writing, its buffer backs up into TRISC2's L1 output buffer, elevating both durations.
- Fix: Replace "Data reader: moves input tiles from DRAM/NoC into L1" with "NoC DMA reader/writer: moves input tiles from DRAM/NoC into L1 and writes output tiles from L1 to DRAM/NoC", consistent with the known-correct NCRISC role definition.

---

**Requirements verified as consistent (no new issues):**

1. Pass 23 fixes confirmed — see above.
2. `FPU UTIL = PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles` — definition consistent across Ch3 `pm_ideal_and_fpu_util.md` line 73, Ch3 `csv_column_definitions.md` line 66, Ch4 `classification_method.md` lines 43–46, and Ch5 `causes_of_low_fpu_util.md` throughout. No contradiction.
3. `stall_cycles = TRISC1_KERNEL_DURATION_cycles − PM_IDEAL_cycles` — direct subtraction form confirmed in `ch3/pm_ideal_and_fpu_util.md` lines 123 and 138. No `(1 − FPU_UTIL)` form present.
4. TRISC2 writes ONLY to L1 (never to DRAM/NoC directly) — confirmed in `ch4/classification_method.md` line 122 ("L1 output buffer full"), `ch5/causes_of_low_fpu_util.md` Cause 4 TRISC2 bullet ("writes them back to L1"), and `ch4/worked_examples.md` line 189 ("writing computed results into L1 output buffers"). No remaining TRISC2-writes-DRAM language.
5. Ridge point AI_ridge = 8.0 FLOPs/byte (256 FLOPs/cycle ÷ 32 bytes/cycle) — confirmed in `ch4/roofline_model_primer.md` lines 119–123 and used consistently in Ch4 `classification_method.md` and Ch5 `csv_signatures.md`.

## Final Pass 25

No feedback — guide approved.

All known-correct facts were verified against the current file content. All prior-pass fixes (Passes 1–24) are confirmed in place. No new factual errors were detected.

**Verification summary:**

1. `FPU_UTIL = PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles` — confirmed in Ch3 `csv_column_definitions.md` line 66 and Ch3 `pm_ideal_and_fpu_util.md` line 73. Consistent across Ch4 and Ch5.
2. `stall_cycles = TRISC1_KERNEL_DURATION_cycles − PM_IDEAL_cycles` — confirmed in `pm_ideal_and_fpu_util.md` lines 123 and 138. No `(1 − FPU_UTIL)` form present.
3. Ridge point AI_ridge = 8.0 FLOPs/byte (256 FLOPs/cycle ÷ 32 bytes/cycle) — correct in Ch4 `roofline_model_primer.md` lines 119–123, used consistently throughout Ch4 and Ch5.
4. Tile FLOPs = 65,536 per tile-triplet (2 × 32³) — confirmed in `pm_ideal_and_fpu_util.md` line 36 ("32 × 32 × 32 = 32,768 MACs per tile-triplet") and `roofline_model_primer.md` line 96 ("65,536 FLOPs per tile-triplet, 256 cycles per tile-triplet").
5. NCRISC role = NoC DMA reader/writer, both directions — confirmed in Ch4 `classification_method.md` line 111 ("NoC DMA reader/writer: moves input tiles from DRAM/NoC into L1 and writes output tiles from L1 to DRAM/NoC"). Pass 24 fix confirmed.
6. TRISC0 reads ONLY from L1 (never DRAM) — confirmed in Ch5 `causes_of_low_fpu_util.md` line 114 and line 118. No "(or DRAM)" language present.
7. TRISC2 writes ONLY to L1 (never DRAM/NoC directly) — confirmed in Ch4 `worked_examples.md` line 189 and Ch5 `causes_of_low_fpu_util.md` line 116. No "(or DRAM)" language present.
8. BRISC excluded from ops_perf CSV — confirmed in Ch1 `two_profilers_compared.md` line 9, Ch3 `csv_column_definitions.md` line 54, and Ch4 `classification_method.md` line 107.
9. Five RISC-V processors per Tensix core — confirmed in Ch3 `csv_column_definitions.md` lines 121 and 127, and Ch5 `causes_of_low_fpu_util.md` line 110.
10. CSV path `generated_profile_log_{test_name}/ops_perf_results_{timestamp}.csv` — confirmed in Ch3 `csv_column_definitions.md` line 105 (`device_params.json` co-located in `generated_profile_log_{test_name}/`).
11. Fidelity-adjusted effective FLOPs/cycle: 256 (HiFi4), 512 (HiFi2), 1024 (LoFi) — confirmed in `pm_ideal_and_fpu_util.md` table lines 42–45 and `csv_signatures.md` line 97.
12. `DEVICE KERNEL DURATION` = wall-clock from first core start to last core end — confirmed in Ch3 `csv_column_definitions.md` line 52 and Ch6 `measuring_dispatch_vs_kernel.md` line 31.

## Final Pass 26

No feedback — guide approved.

All known-correct facts were verified against the current file content across all six chapters and all content files. All prior-pass fixes (Passes 1–25) are confirmed in place. No new factual errors or broken cross-references were detected.

**Verification summary:**

1. `FPU_UTIL = PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles` — confirmed in Ch3 `csv_column_definitions.md` line 66 and `pm_ideal_and_fpu_util.md` line 73. Consistent across Ch4 `classification_method.md` lines 43–46 and Ch5 throughout.
2. `stall_cycles = TRISC1_KERNEL_DURATION_cycles − PM_IDEAL_cycles` — direct subtraction form confirmed at lines 123 and 138 of `pm_ideal_and_fpu_util.md`. No `(1 − FPU_UTIL)` form present.
3. Tile FLOPs: `pm_ideal_and_fpu_util.md` line 36 correctly reads "32 K-dimension elements of the paired input tile along the inner dimension per tile step, giving 32 × 32 × 32 = 32,768 MACs per tile-triplet" (= 65,536 FLOPs = 2 × 32³). Consistent with `roofline_model_primer.md` lines 96–97 ("65,536 FLOPs per tile-triplet, 256 cycles per tile-triplet").
4. Ridge point AI_ridge = 8.0 FLOPs/byte (256 FLOPs/cycle ÷ 32 bytes/cycle) — confirmed in `ch4/roofline_model_primer.md` lines 119–123 and used consistently in `classification_method.md` Step 1 and throughout `ch5/csv_signatures.md`.
5. NCRISC role = NoC DMA reader/writer (both directions) — confirmed in `classification_method.md` line 111. TRISC0 reads ONLY from L1 — confirmed in `causes_of_low_fpu_util.md` lines 114 and 118. TRISC2 writes ONLY to L1 — confirmed in `causes_of_low_fpu_util.md` line 116 and `worked_examples.md` line 189.
6. Cause 3 diagnostic: `TRISC1 KERNEL DURATION [ns] ≈ 4× LoFi baseline`; `FPU UTIL` stays roughly constant — correctly stated in `csv_signatures.md` lines 82–85 and `causes_of_low_fpu_util.md` lines 75–83.
7. Fidelity-adjusted effective FLOPs/cycle: 256 (HiFi4), 512 (HiFi2), 1024 (LoFi) — confirmed in `pm_ideal_and_fpu_util.md` table lines 42–45.
8. BRISC excluded from ops_perf CSV — confirmed in Ch1 `two_profilers_compared.md` line 9, Ch3 `csv_column_definitions.md` line 54, and Ch4 `classification_method.md` line 107.
9. Five RISC-V processors per Tensix core — confirmed in `ch5/causes_of_low_fpu_util.md` line 110 and Ch3 `csv_column_definitions.md` lines 121 and 127.
10. CSV path `generated_profile_log_{test_name}/ops_perf_results_{timestamp}.csv` — confirmed in Ch2 `capture_workflow.md` Output Artifacts table and the Python smoke-test example (glob-based discovery). `device_params.json` correctly co-located in `generated_profile_log_{test_name}/` per `csv_column_definitions.md` line 105.
11. `DEVICE KERNEL DURATION` = wall-clock from first core start to last core end — confirmed in Ch3 `csv_column_definitions.md` lines 115–119 and Ch6 `measuring_dispatch_vs_kernel.md` line 31.
12. `total_op_latency = host_dispatch_time + device_kernel_time + sync_overhead` — confirmed in Ch1 `two_profilers_compared.md` line 25, Ch6 `what_is_dispatch_overhead.md` line 103, and Ch6 `measuring_dispatch_vs_kernel.md`.
13. Tracy = build-time (`TRACY_ENABLE`); device profiler = runtime (`TT_METAL_DEVICE_PROFILER=1`) — confirmed in Ch2 `env_vars_and_flags.md` variable interaction table.
14. CSV columns come from device profiler / `process_ops_logs.py`, not Tracy — confirmed in Ch3 `csv_column_definitions.md` lines 5–13 and Ch4 `classification_method.md` line 3.
