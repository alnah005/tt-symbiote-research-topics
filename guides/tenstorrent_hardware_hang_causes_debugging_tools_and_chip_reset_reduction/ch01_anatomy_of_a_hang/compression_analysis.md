# Compression Analysis -- Change Log

## 2026-05-02: Agent B Feedback Applied (2 issues)

### Issue 1: Fixed chapter cross-references in `03_hang_taxonomy.md`

Four of six category cross-references pointed to wrong chapters. Corrected:

| Location | Was | Now | Reason |
|---|---|---|---|
| Category 2 (NOC) -- "NOC debugging tools" | Chapter 3 | Chapter 6 | Debugging tools are in Chapter 6 |
| Category 3 (Memory) -- "Address sanitization" | Chapter 4 | Chapter 6 | Debugging tools are in Chapter 6 |
| Category 4 (Dispatch) -- "Dispatch hang debugging" | Chapter 5 | Chapter 4 | Dispatch hangs are covered in Chapter 4 |
| Category 6 (Host-Device) -- "Host-device debugging" | Chapter 6 | Chapter 4 | Host-device hangs are covered in Chapter 4 |

Two references were already correct and left unchanged:
- Category 1 (Kernel-Level) -> Chapter 2
- Category 5 (Multi-Chip) -> Chapter 5

### Issue 2: Fixed 5-part diagnostic format in `01_what_is_a_hang.md`

Replaced non-conforming format labels with the plan-specified labels:

| Position | Was | Now |
|---|---|---|
| 1 | Mechanism | Symptom |
| 2 | Symptom | Root Cause |
| 3 | Root Cause | Diagnosis Steps |
| 4 | Diagnosis | Fix |
| 5 | Mitigation | Prevention |

Also updated the descriptions for each label and the closing summary sentence to reflect the new format structure.

---

# Compression Analysis: Chapter 1 — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~1906 lines
- Estimated post-compression line count: ~1680 lines
- Estimated reduction: ~12%

## CRUCIAL Suggestions
### [01_what_is_a_hang.md] ~lines 67-86
**Issue:** The `cb_reserve_back` code snippet and accompanying explanation are fully duplicated. The identical code block appears again in `02_blocking_primitives_taxonomy.md` lines 38-57, with an even more detailed walkthrough. Section 01 already explains the concept of wait-loops with a simpler pseudocode pattern at lines 42-49; including the full `cb_reserve_back` implementation here is redundant with the dedicated taxonomy file.
**Suggestion:** Replace the full `cb_reserve_back` code listing and explanation in `01_what_is_a_hang.md` (lines 67-86) with a brief 2-3 line forward reference: state that `cb_reserve_back` is a representative example of this spin-wait pattern, and point the reader to Section 02 for the full code and failure mode analysis. This removes ~20 lines of duplicated code and prose.

### [02_blocking_primitives_taxonomy.md] ~lines 554-561
**Issue:** The `assert_and_hang` entry in Section 02 explicitly acknowledges it is "documented in detail in Section 1 (01_what_is_a_hang.md)" and then proceeds to re-summarize it anyway. The three bullet points (assert mailbox contents, watcher can read it, ERISC exits to base firmware) repeat information already covered in `01_what_is_a_hang.md` lines 266-306. The only net-new sentence is the triage advice ("Always check the assert mailbox early in diagnosis"), which could be a one-liner cross-reference.
**Suggestion:** Collapse the `assert_and_hang` subsection in `02_blocking_primitives_taxonomy.md` to a single short paragraph: state that `assert_and_hang` is technically a blocking primitive (`while(1)` loop), note it is distinguishable via the assert mailbox, and point the reader to `01_what_is_a_hang.md` for full details. Retain only the triage-relevant sentence about checking the assert mailbox early. This saves ~8 lines.

### [01_what_is_a_hang.md] ~lines 5-17 + [03_hang_taxonomy.md] ~lines 24-28
**Issue:** The failure-mode comparison table in `01_what_is_a_hang.md` (lines 9-15) describes hang symptoms (no forward progress, no error signal, preserved data, requires kill or chip reset). Then `03_hang_taxonomy.md` Category 1 "Symptoms" section (lines 24-28) restates the same observable characteristics (frozen waypoints, inside user kernel phase, NOC registers typically clean, cascade pattern). While the framing differs slightly (general failure-mode taxonomy vs. category-specific symptoms), the description of what a kernel-level hang "looks like" is already well-established by the end of file 01. The symptoms subsection in Category 1 of file 03 adds little that the reader has not already internalized.
**Suggestion:** Trim the Category 1 "Symptoms" subsection in `03_hang_taxonomy.md` to a 2-line summary that references the general hang definition in file 01 and states only the distinguishing characteristic (NOC registers clean, hang is in CB/semaphore logic). The full five-bullet symptom list can be cut. Saves ~6 lines while keeping the essential diagnostic distinction.

### [04_hang_causes_across_architectures.md] ~lines 29-53
**Issue:** The "Universal Hang Causes" section restates concepts already thoroughly covered in files 01, 02, and 03. Specifically: "CB Protocol Violations" (lines 33-37) re-describes `cb_reserve_back`/`cb_wait_front` and the `pages_acked`/`pages_received` protocol, which is the primary subject of `02_blocking_primitives_taxonomy.md`. "NOC Barrier Deadlocks" (lines 39-41) restates that NRBW/NWBW function identically, already covered in file 02. "Semaphore Overshoot" (lines 43-45) restates the NSW exact-match problem, already covered in file 02 lines 330-331. "Go-Signal Failure" (lines 47-49) restates the GW loop, covered in both file 01 and file 02. Each subsection adds only one architecture-differential sentence (e.g., "BH supports 64 circular buffers").
**Suggestion:** Replace the five "Universal Hang Causes" subsections with a single compact paragraph or table stating: "The following hang patterns are universal across all architectures (see Sections 01 and 02 for full details): CB deadlocks (CRBW/CWFW), NOC barrier stalls (NRBW/NWBW), semaphore overshoot (NSW), go-signal failure (GW), and NOC address errors." Then keep ONLY the architecture-differential notes (BH 64 CBs, alignment differences) as a short bulleted addendum. This saves ~20 lines.

## MINOR Suggestions
### [01_what_is_a_hang.md] ~lines 152-173
**Issue:** The two signal tables (Go-Signal Protocol and Subordinate Synchronization) list raw hex values for every signal constant. These constants are already defined in source code, and the hex values are not needed for conceptual understanding of the hang lifecycle. The tables occupy ~22 lines of vertical space.
**Suggestion:** Merge the two tables into a single compact table with columns: Signal Name, Direction (host->device or inter-core), and Purpose. Drop the hex value column. Readers needing exact values can consult the source. Saves ~8 lines.

### [02_blocking_primitives_taxonomy.md] ~lines 7-11
**Issue:** The "Understanding Waypoint Codes" preamble re-explains the waypoint mechanism (4-character ASCII codes, packed into 32-bit word, WAYPOINT macro). This is already explained in `01_what_is_a_hang.md` lines 51-65 with the actual implementation code. The repetition in file 02 is a lighter re-statement without code, but it covers the same ground.
**Suggestion:** Replace with a one-line forward reference: "For how waypoint codes work, see Section 01. The convention: suffix `W` = waiting, suffix `D` = done." Saves ~4 lines.

### [03_hang_taxonomy.md] ~lines 249-263
**Issue:** The "Symptoms Cross-Reference Matrix" is a useful lookup table, but three of its rows are low-signal: "Heartbeat stopped" is yes for nearly every category, "Non-deterministic" is "Rare" for nearly every category, and "No waypoint (core never entered kernel)" only distinguishes two categories. These three rows add visual bulk without strong diagnostic utility.
**Suggestion:** Remove the "Heartbeat stopped," "Non-deterministic," and "No waypoint" rows from the matrix, or collapse them into a footnote. Saves ~3 lines and sharpens the table's signal-to-noise ratio.

### [04_hang_causes_across_architectures.md] ~lines 435-464
**Issue:** The "Architecture-Specific vs. Universal Hang Categories" section contains two tables that largely re-list information already presented earlier in the same file. The "Universal" table (lines 437-443) restates the five universal causes from lines 29-53. The "Architecture-Specific" table (lines 447-464) is a summary of the GS/WH/BH/QA subsections that precede it.
**Suggestion:** Keep only the "Architecture-Specific" table (it serves as a useful index). Remove the "Universal" table entirely since those causes are covered in the paragraph suggested above and exhaustively documented in files 01-02. Saves ~8 lines.

### [04_hang_causes_across_architectures.md] ~lines 392-429
**Issue:** The "Scale-Dependent Hang Patterns" section contains six subsections (Single Chip, N300, T3K, Galaxy, Multi-BH, Multi-QA). The Single Chip subsection (lines 396-399) states the obvious: Categories 1-4 and 6 apply, Category 5 does not. The Multi-QA subsection (lines 425-429) contains only speculative statements ("expected to differ," "TBD"). Both subsections are low-information-density.
**Suggestion:** Remove the Single Chip subsection (its content is self-evident from the category definitions). Trim Multi-QA to a single sentence noting the architecture is emerging. Saves ~10 lines.

## Load-Bearing Evidence
(Not required -- verdict is "Crucial updates: yes")

## VERDICT
- Crucial updates: yes

---

## 2026-05-02: CRUCIAL Compression Suggestions Applied (4 edits)

### Edit 1: `01_what_is_a_hang.md` -- Removed duplicated `cb_reserve_back` code listing
Replaced the full `cb_reserve_back` code block (~lines 67-86) with a 3-line forward reference to `02_blocking_primitives_taxonomy.md`. The code was duplicated verbatim in file 02. Saved ~18 lines.

### Edit 2: `02_blocking_primitives_taxonomy.md` -- Collapsed `assert_and_hang` subsection
Replaced the 8-line subsection (which explicitly acknowledged duplicating file 01 content) with a single-paragraph cross-reference to `01_what_is_a_hang.md`. Retained the triage-relevant advice about checking the assert mailbox early. Saved ~5 lines.

### Edit 3: `03_hang_taxonomy.md` -- Trimmed Category 1 "Symptoms" subsection
Replaced the 5-bullet symptom list with a 2-line summary referencing `01_what_is_a_hang.md` and retaining only the category-specific distinguishing feature (NOC registers clean, hang is in CB/semaphore logic). Saved ~4 lines.

### Edit 4: `04_hang_causes_across_architectures.md` -- Condensed "Universal Hang Causes" section
Replaced five subsections (~25 lines of restated content from files 01/02) with a single paragraph referencing `02_blocking_primitives_taxonomy.md` followed by a short bulleted list of only the architecture-differential notes (BH 64 CBs, NOC barrier implementation differences, alignment requirements). Saved ~15 lines.

---

## 2026-05-02: Agent B Feedback Applied (1 issue, round 2)

### Issue: Fixed fabricated details in `assert_and_hang` cross-reference in `02_blocking_primitives_taxonomy.md`

The compressed `assert_and_hang` subsection (line 556) introduced two fabricated details:
1. A waypoint code `AH` -- `assert_and_hang` does not call WAYPOINT(); it writes to the assert mailbox, not the waypoint mailbox
2. A function name `erisc_exit()` -- no such function exists in the chapter; `01_what_is_a_hang.md` says ERISC cores "exit back to base firmware"

Fixed by replacing the inaccurate sentence with accurate language matching `01_what_is_a_hang.md`: describes the assert mailbox fields (line number, RISC-V processor, assert type), states ERISC cores exit back to base firmware, and explicitly notes that `assert_and_hang` does not use the WAYPOINT mechanism.

---

# Compression Analysis: Chapter 1 — Pass 2

## Summary
- Total files analyzed: 4
- Estimated current line count: ~1860 lines
- Estimated post-compression line count: ~1820 lines
- Estimated reduction: ~2%

## CRUCIAL Suggestions
None

## MINOR Suggestions
### [02_blocking_primitives_taxonomy.md] ~lines 17-27 vs. ~lines 602-622
**Issue:** The "Overview of Primary Blocking Primitives" table (6 rows, lines 19-26) overlaps with the "Summary Table: Blocking Primitives Quick Reference" (17 rows, lines 604-622). Both share columns for Waypoint and Function, and the 6 primary primitives appear in both. However, each table has unique columns -- the overview has "Core(s)" and "Done Waypoint"; the summary has "Most Common Hang Cause" -- so neither is a strict subset. The overlap is modest (6 shared rows with different column sets), but a reader scanning the file encounters the same primitive-to-waypoint mapping twice.
**Suggestion:** Consider merging the overview table's unique "Core(s)" column into the summary table (adding a 5th column) and removing the overview table, replacing it with a sentence pointing to the summary table at the end. This would save ~10 lines while consolidating the lookup surface. The tradeoff is that a larger table at the end is slightly less convenient for quick orientation at the start of the file.

### [04_hang_causes_across_architectures.md] ~lines 417-425
**Issue:** The "Universal (All Architectures)" table (5 rows) restates the universal causes already described in the condensed paragraph at lines 29-31 of the same file ("CB deadlocks (CRBW/CWFW), NOC barrier stalls (NRBW/NWBW), semaphore overshoot (NSW), go-signal failure (GW), and NOC address errors"). This was flagged as MINOR in Pass 1; it remains a minor intra-file echo. The table adds the "Root Cause" and "Affected Architectures" columns, but the latter is "GS, WH, BH, Quasar" for every row (which is what "universal" means), and the root causes are one-phrase summaries of file 02 content.
**Suggestion:** Remove the "Universal (All Architectures)" table and add a one-line note above the "Architecture-Specific" table: "For universal hang causes (CB deadlocks, NOC barriers, semaphore errors, go-signal failures, command buffer stalls), see [02_blocking_primitives_taxonomy.md](./02_blocking_primitives_taxonomy.md)." Saves ~8 lines.

### [01_what_is_a_hang.md] ~lines 115-126 vs. [04_hang_causes_across_architectures.md] ~lines 336-345
**Issue:** The Quasar `wait_subordinates()` code block (the `while` loop checking `subordinate_sync->allDMs`, `allNeo0-3`) appears in both files. In file 01, it illustrates Phase 5 of the kernel lifecycle for the Quasar variant. In file 04, it appears under QA-2 with a function wrapper and additional analysis of the 16-TRISC hang surface. The code overlap is 6 lines of `while` loop body. Each instance serves a distinct analytical purpose (lifecycle context vs. architecture-specific hang surface), so this is a minor duplication rather than bloat.
**Suggestion:** In `01_what_is_a_hang.md`, replace the Quasar code block (lines 117-126) with a brief note: "On Quasar (tt-2xx), the subordinate wait extends to four Neo engines and multiple DM cores -- see [04_hang_causes_across_architectures.md](./04_hang_causes_across_architectures.md#qa-2-four-trisc-cores-per-neo-engine) for the full code and hang surface analysis." This removes 10 lines of duplicated code while preserving the lifecycle narrative. The authoritative code listing remains in file 04 where it is analyzed in depth.

## Load-Bearing Evidence
- `01_what_is_a_hang.md` line ~67: "A representative example is `cb_reserve_back`, which spins waiting for a circular buffer consumer to free space. It sets waypoint `CRBW` before the loop and `CRBD` after." -- load-bearing because this forward reference replaced the previously duplicated code block; removing or altering it would break the conceptual bridge between the generic wait-loop model (lines 40-49) and the detailed taxonomy in file 02.
- `02_blocking_primitives_taxonomy.md` line ~556: "The `assert_and_hang` mechanism is documented in [01_what_is_a_hang.md](./01_what_is_a_hang.md#the-assert_and_hang-pattern-a-hang-by-design)." -- load-bearing because this cross-reference replaced the previously duplicated `assert_and_hang` description; it is the sole pointer from the blocking-primitives catalog to the authoritative `assert_and_hang` documentation.
- `04_hang_causes_across_architectures.md` line ~31: "The blocking primitives documented in [02_blocking_primitives_taxonomy.md](./02_blocking_primitives_taxonomy.md) are universal across all architectures: CB deadlocks (CRBW/CWFW), NOC barrier stalls (NRBW/NWBW), semaphore overshoot (NSW), go-signal failure (GW), and NOC address errors" -- load-bearing because this condensed paragraph replaced five previously expanded subsections and is the only place in file 04 that establishes which hang patterns are universal vs. architecture-specific.
- `03_hang_taxonomy.md` line ~24: "Kernel-level hangs exhibit the general hang characteristics described in [01_what_is_a_hang.md](./01_what_is_a_hang.md#observable-symptoms)." -- load-bearing because this cross-reference replaced a previously redundant symptom list and is the sole link from the taxonomy's Category 1 back to the foundational hang definition.
- `02_blocking_primitives_taxonomy.md` line ~286: "Blackhole inline-write back-pressure: On Blackhole, inline writes to L1 use all four memory ports and can hang when there is back-pressure." -- load-bearing because this is the only mention of BH inline-write back-pressure as a failure mode of the NWBW primitive; removing it would leave the NWBW failure mode table incomplete for BH-specific issues.

## VERDICT
- Crucial updates: no
