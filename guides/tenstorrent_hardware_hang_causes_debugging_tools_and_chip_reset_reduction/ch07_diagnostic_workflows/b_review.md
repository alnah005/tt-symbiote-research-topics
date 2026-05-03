# Agent B Review -- Chapter 7

## Pass 1

**Issues found:** 5

---

### Issue 1: Index title is Chapter 8's title, not Chapter 7's

- **File:** `index.md`
- **Location:** Line 1
- **Category:** Factual error
- **Problem:** The title reads `# Chapter 7: Chip Reset Reduction, Graceful Recovery, and Resilience` but this is the topic description of Chapter 8. Chapter 7 covers "Diagnostic Workflows and End-to-End Triage." The body text of the same index.md (line 3 onward) correctly describes diagnostic workflows, creating a contradiction between the title and the content.
- **Evidence:** Compare with `../ch08_reset_reduction/index.md` line 1 which reads `# Chapter 8: Reset Reduction, Resilience, and Future Improvements`. Additionally, `index.md` line 5 says "This chapter addresses research questions Q7 (chip reset reduction strategies), Q8 (future tooling and recovery)" -- these are Chapter 8's research questions, not Chapter 7's. The rest of the index body, the file list, and all five content files are clearly about diagnostic workflows.
- **Suggested fix:** Change line 1 to `# Chapter 7: Diagnostic Workflows and End-to-End Triage`. Update line 5 to reference the correct research questions for Chapter 7 (diagnostic workflows, tool integration, triage methodology).

---

### Issue 2: Scenario cross-reference 2.2.3 should be 2.2.4

- **File:** `02_diagnosing_by_hang_category.md`
- **Location:** Line 123, in the "Step 2: Map the RISC waypoint pattern to a specific scenario" table
- **Category:** Factual error (incorrect cross-chapter scenario number)
- **Problem:** The table row reads: `| CRBW | CWFW | R | Both data-mover RISCs stuck | CB size indivisible by tile count | Ch 2, Scenario 2.2.3 |`. However, Hang Cause 2.2.3 in Chapter 2 (`02_circular_buffer_deadlocks.md`) is "The Cumulative-Total Requirement Violation," not "CB size indivisible by tile count." The correct scenario is Hang Cause 2.2.4: "CB Size Not Evenly Divisible by Tile Count."
- **Evidence:** In `../ch02_kernel_and_noc_hangs/02_circular_buffer_deadlocks.md`: line 161 shows `## Hang Cause 2.2.3: The Cumulative-Total Requirement Violation` and line 224 shows `## Hang Cause 2.2.4: CB Size Not Evenly Divisible by Tile Count`.
- **Suggested fix:** Change `Ch 2, Scenario 2.2.3` to `Ch 2, Scenario 2.2.4` in that table row.

---

### Issue 3: Scenario 2.2.7 cross-reference is wrong

- **File:** `02_diagnosing_by_hang_category.md`
- **Location:** Line 152, in the "Step 5: Verify producer-consumer consistency" table
- **Category:** Factual error (incorrect cross-chapter scenario number)
- **Problem:** The table row reads: `| cb_addr_shift | Is the kernel compiled for the correct core type (data mover vs. compute)? | Scenario 2.2.7: Wrong core type compilation |`. However, Hang Cause 2.2.7 in Chapter 2 is "`cb_push_back` Overflow (Silent Corruption Leading to Hang)," not "Wrong core type compilation." The `cb_addr_shift` / wrong core type issue is discussed in the preamble of `02_circular_buffer_deadlocks.md` (lines 44-56) but does not have its own numbered scenario.
- **Evidence:** In `../ch02_kernel_and_noc_hangs/02_circular_buffer_deadlocks.md`: line 365 shows `## Hang Cause 2.2.7: cb_push_back Overflow (Silent Corruption Leading to Hang)` and line 480 in the summary table confirms `2.2.7 Push overflow`.
- **Suggested fix:** Remove the reference to "Scenario 2.2.7" and replace with a direct reference to the `cb_addr_shift` discussion in Ch2 Section 02, e.g., `Ch 2, 02_circular_buffer_deadlocks.md (cb_addr_shift section)`.

---

### Issue 4: Inconsistent heading format for scenarios 7.5.12 and 7.5.13

- **File:** `05_distinguishing_hw_vs_sw_bugs.md`
- **Location:** Lines 624 and 644
- **Category:** Format violation
- **Problem:** Sections 7.5.12 and 7.5.13 use the heading prefix `## Scenario 7.5.12:` and `## Scenario 7.5.13:` while all other sections in the same file (7.5.1 through 7.5.11 and 7.5.14) use the format `## 7.5.X` without the "Scenario" prefix. These two are the only sections in the entire chapter that use the 5-part diagnostic format (Symptom / Root Cause / Diagnosis Steps / Fix / Prevention), which is correct for scenarios. However, the inconsistent heading style breaks the otherwise uniform numbering pattern of the file.
- **Evidence:** Line 15: `## 7.5.1 Reproducibility as the Primary Signal`; Line 581: `## 7.5.10 War Story: ...`; Line 624: `## Scenario 7.5.12: Clean State Reveals Stale Semaphore Initialization`. The "Scenario" prefix appears only on 7.5.12 and 7.5.13.
- **Suggested fix:** For consistency with the rest of the file, change to `## 7.5.12 Clean State Reveals Stale Semaphore Initialization` and `## 7.5.13 Blackhole Relaxed Memory Ordering Exposes Missing Barrier` (dropping the "Scenario" prefix). Alternatively, if the intent is to distinguish scenario-format sections from non-scenario sections, apply the same convention throughout the guide.

---

### Issue 5: Index.md research question attribution is incorrect for Chapter 7

- **File:** `index.md`
- **Location:** Line 5 and line 40
- **Category:** Factual error
- **Problem:** The index states "This chapter addresses research questions Q7 (chip reset reduction strategies), Q8 (future tooling and recovery), and partially Q5 (automated debugging)." and line 40 repeats "Covers research questions: Q7 (chip reset reduction strategies), Q8 (future tooling and recovery procedures), and partially Q5 (automated debugging)." These research questions (Q7 on chip reset reduction, Q8 on future tooling/recovery) describe Chapter 8's scope, not Chapter 7's. Chapter 7's content is about diagnostic workflows and end-to-end triage methodology -- it integrates tools from Chapter 6 into practical procedures. The research questions for diagnostic workflows would more naturally be about systematic debugging methodology and tool integration.
- **Evidence:** Chapter 8's index.md explicitly covers reset reduction and future improvements. Chapter 7's actual content files (01-05) contain no reset reduction strategies or future tooling proposals -- they are entirely about diagnostic workflows, triage procedures, and interpreting tool output.
- **Suggested fix:** Update the research question attribution to accurately reflect Chapter 7's content. If the guide's research question mapping explicitly assigns Q7/Q8 to Chapter 7, then the content files are misaligned with the index -- but the more likely explanation is that the index metadata was copied from Chapter 8's template.

---

## Format Compliance

The two scenarios present in Chapter 7 (7.5.12 and 7.5.13 in `05_distinguishing_hw_vs_sw_bugs.md`) correctly use the 5-part format: **(1) Symptom**, **(2) Root Cause**, **(3) Diagnosis Steps**, **(4) Fix**, **(5) Prevention**. The numbering uses the `7.X.Y` format as required (7.5.12, 7.5.13) with no gaps or duplicates. However, they use a numbered-prefix style `(1)` / `(2)` / etc. rather than bold labels, which is a minor style difference from other chapters but acceptable.

The remaining content in Chapter 7 consists of workflow procedures, mapping tables, decision trees, and checklists rather than traditional scenarios -- this is appropriate given that Chapter 7 is an integration/workflow chapter, not a hang-catalog chapter. No format violations exist in the procedural content.

## What the Chapter Gets Right

- **Cross-chapter links are almost entirely correct.** All directory names (`ch01_anatomy_of_a_hang`, `ch02_kernel_and_noc_hangs`, etc.) and file names match the actual filesystem. No broken directory references.
- **The waypoint code tables are comprehensive and accurate.** The complete waypoint reference in `04_reading_watcher_and_triage_output.md` matches the blocking primitives described in Chapter 1 and the dispatch waypoints in Chapter 4.
- **The NOC error message mapping table is thorough.** All sanitize return codes (3-17) are documented with correct enum names and actionable diagnosis steps.
- **The decision trees and flowcharts are well-structured.** They provide clear routing from symptoms to category-specific procedures with appropriate cross-references.
- **The war stories are instructive and grounded.** They illustrate realistic debugging scenarios and reinforce the workflows described in the procedural sections.
- **Tool integration is excellent.** The chapter successfully weaves together watcher, tt-triage, watcher_dump, DPrint, Inspector, Tracy, and debug delays into coherent workflows.

---

**Verdict:** NEEDS REVISION

The title/metadata mismatch in `index.md` (Issues 1 and 5) is the most impactful problem since it misidentifies the entire chapter's purpose and research question scope. The two incorrect scenario cross-references (Issues 2 and 3) could misdirect developers to the wrong hang cause in Chapter 2. All five issues are straightforward to fix.
