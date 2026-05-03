# Agent B Review -- Chapter 6

## Pass 1

**Issues found:** 5

---

### Issue 1: NOC barrier cross-references point to wrong Ch2 sections

- **File:** `06_debug_delay_and_timing_perturbation.md`, `05_profiler_tracy_and_noc_debug.md`, `01_watcher_system.md`
- **Location:** `06_debug_delay_and_timing_perturbation.md` line 426; `05_profiler_tracy_and_noc_debug.md` line 369; `01_watcher_system.md` line 485
- **Category:** Cross-chapter inconsistency
- **Problem:** Multiple files state "NOC barrier requirements: Chapter 2, Sections 2.1-2.3" but NOC barriers are covered in Chapter 2, Section 2.4 (`04_noc_barrier_and_semaphore_hangs.md`). Sections 2.1-2.3 cover RISC synchronization, circular buffer deadlocks, and NOC address sanitization respectively -- none of which are primarily about barrier requirements. The correct reference should include Section 2.4 (or at minimum say "Sections 2.3-2.4" to include both the sanitization that precedes barriers and the barriers themselves).
- **Evidence:** Ch2 index lists file 04 (`04_noc_barrier_and_semaphore_hangs.md`) as Section 2.4 with focus on "Read/write barriers, mcast path reservation workaround, semaphore protocols, TRID barriers" and key waypoints `NRBW`, `NWBW`, `NSW`, `NSMW`. The barrier-specific content (noc_async_read_barrier, noc_async_write_barrier, noc_semaphore_wait) is all in Section 2.4.
- **Suggested fix:** Change all occurrences of "Chapter 2, Sections 2.1-2.3" that reference NOC barriers to "Chapter 2, Section 2.4" (or "Sections 2.3-2.4" where the intent is to cover both sanitization and barriers).

---

### Issue 2: "Compute pipeline synchronization" cross-reference points to wrong Ch2 section

- **File:** `06_debug_delay_and_timing_perturbation.md`
- **Location:** Line 427
- **Category:** Cross-chapter inconsistency
- **Problem:** The cross-references section states "Compute pipeline synchronization: Chapter 2, Section 2.3" but Section 2.3 is `03_noc_address_sanitization_and_violations.md`, which covers the NOC address validation pipeline, not compute pipeline synchronization. Compute pipeline synchronization (TRISC unpack/math/pack coordination, acquire_dst/release_dst) is covered in Chapter 2, Section 2.1 (`01_risc_synchronization_and_deadlocks.md`) which documents BRISC/NCRISC/TRISC synchronization protocols.
- **Evidence:** The Ch2 index describes Section 2.1 as covering "BRISC/NCRISC/TRISC synchronization protocols, subordinate mailbox, ERISC context switching" and Section 2.3 as covering "NOC validation pipeline, all DebugSanitize return codes, deliberate hang mechanism."
- **Suggested fix:** Change "Compute pipeline synchronization: Chapter 2, Section 2.3" to "Compute pipeline synchronization: Chapter 2, Section 2.1".

---

### Issue 3: DPRINT default output destination is contradictory within the same file

- **File:** `03_dprint_server.md`
- **Location:** Line 51 (env var table) vs. line 175 (output paths table)
- **Category:** Factual error (internal inconsistency)
- **Problem:** The environment variable reference table on line 51 states the default for `TT_METAL_DPRINT_FILE` is `stdout`. However, the Output Paths table on line 175 states that the default (when no `DPRINT_FILE` is set) is `generated/dprint/dprint.log`. These two statements directly contradict each other. A reader following the env var table would expect output on stdout, while a reader following the output paths table would look for a log file.
- **Evidence:** Line 51: `| TT_METAL_DPRINT_FILE | path | stdout | Output file path. |` vs. Line 175: `| Default (no DPRINT_FILE) | generated/dprint/dprint.log |`
- **Suggested fix:** Determine the actual default behavior from the source code and make both tables consistent. Based on the code pattern (similar to watcher writing to `generated/watcher/watcher.log`), the default is likely `generated/dprint/dprint.log`, and the env var table's "Default" column should say `generated/dprint/dprint.log` rather than `stdout`.

---

### Issue 4: CB sanitization cross-reference points to Ch3 Section 3.2 instead of Section 3.4

- **File:** `01_watcher_system.md`
- **Location:** Line 325 and line 486
- **Category:** Cross-chapter inconsistency
- **Problem:** Line 325 states "Cross-reference: Ch3 Section 3.2" in the context of CB (circular buffer) sanitization and `DebugSanitizeCBOutOfBounds`. Line 486 states "CB deadlock scenarios: Chapter 2, Section 2.2; Chapter 3, Section 3.2". However, Ch3 Section 3.2 is `02_dram_and_noc_backpressure.md`, which covers DRAM bandwidth saturation and NOC backpressure -- not CB bounds checking. The CB overflow/overwrite content and watcher CB sanitization (`DebugSanitizeCBOutOfBounds`) are documented in Ch3 Section 3.4 (`04_allocation_failures_and_silent_oom.md`), as confirmed by the Ch3 index which lists `DebugSanitizeCBOutOfBounds` as a key return code for Section 3.4.
- **Evidence:** Ch3 index: Section 3.4 covers "CB overflow/overwrite, watcher CB sanitization" with key code `DebugSanitizeCBOutOfBounds`. Section 3.2 covers "DRAM bandwidth saturation, bank collision stalls, NOC backpressure propagation" with key waypoints `NWBW`, `NRBW`.
- **Suggested fix:** Change "Ch3 Section 3.2" to "Ch3 Section 3.4" on line 325. Change the Ch3 part of line 486 from "Chapter 3, Section 3.2" to "Chapter 3, Section 3.4".

---

### Issue 5: Scenarios use "Resolution" instead of the guide-standard "Fix" label

- **File:** All scenario files in Ch6 (`01_watcher_system.md`, `02_watcher_dump_tool.md`, `03_dprint_server.md`, `04_tt_triage_tool.md`, `05_profiler_tracy_and_noc_debug.md`, `06_debug_delay_and_timing_perturbation.md`)
- **Location:** Every scenario section across all six content files
- **Category:** Format violation
- **Problem:** The guide-wide 5-part diagnostic format established in Chapter 1 and used consistently throughout Chapters 2-5 is: **Symptom / Root Cause / Diagnosis Steps / Fix / Prevention**. Chapter 6 scenarios use **Symptom / Root Cause / Diagnostic Steps / Resolution / Prevention** -- deviating in two labels: "Diagnostic Steps" instead of "Diagnosis Steps", and "Resolution" instead of "Fix". While the semantic meaning is preserved, this creates an inconsistency with all other chapters in the guide and makes keyword-based searching across chapters unreliable.
- **Evidence:** Ch2 `02_circular_buffer_deadlocks.md` uses `### Diagnosis Steps` and `### Fix`. All Ch6 scenarios use `**Diagnostic Steps**:` and `**Resolution**:`. The same pattern appears in all 22 scenarios across Ch6.
- **Suggested fix:** Rename "Diagnostic Steps" to "Diagnosis Steps" and "Resolution" to "Fix" in all 22 scenarios to match the established guide convention. The heading style difference (bold inline vs. `###` heading) is a less critical stylistic choice and could optionally be harmonized as well.

---

## Format Compliance

Scenario numbering follows the `6.X.Y` format correctly with no gaps or duplicates (6.1.1-6.1.4, 6.2.1-6.2.3, 6.3.1-6.3.4, 6.4.1-6.4.4, 6.5.1-6.5.4, 6.6.1-6.6.5). All 22 scenarios contain the five conceptual parts (symptom, root cause, diagnosis, resolution, prevention), but the labels deviate from the guide standard as noted in Issue 5. The overall structure is consistent and well-organized within the chapter.

## What the Chapter Gets Right

- The progressive diagnosis pipeline (Stage 0 through Stage 5) in the index is an excellent contribution that provides a clear methodology for approaching any hang investigation.
- Environment variable reference tables are exhaustive and well-structured, covering every configuration option with types, defaults, and descriptions.
- Decision trees at the start of each section help readers quickly navigate to the right tool.
- The watcher `debug_sanitize_noc_return_code_enum` table (Section 6.1.4) is fully consistent with the authoritative listing in Ch2 Section 2.3.
- Internal cross-references between Ch6 sections (e.g., Section 6.1 <-> 6.2, 6.4 <-> 6.6) are correct and well-placed.
- Practical recipes (bash code blocks with environment variable configurations) are actionable and immediately useful.
- The combined tool usage matrix in Section 6.6.9 is a valuable quick-reference for selecting tool combinations.

---

**Verdict:** NEEDS REVISION
