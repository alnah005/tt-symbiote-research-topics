# Agent B Review: Chapter 1 -- Pass 1

## Issue 1: Incorrect chapter cross-references in hang taxonomy will send readers to wrong chapters
**File:** 03_hang_taxonomy.md
**Lines:** ~98-99, ~131-132, ~170, ~243
**Problem:** Four of the six category "Chapter References" footers point to the wrong chapters. According to the plan: Chapter 2 = Kernel-Level and NOC Hang Mechanisms, Chapter 3 = Memory-Related Hang Causes, Chapter 4 = Dispatch and Host-Device Interaction Hangs, Chapter 5 = Multi-Chip and CCL Hangs, Chapter 6 = Debugging Tools. The current text has:
- Category 2 (NOC): "NOC debugging tools: Chapter 3" -- should be Chapter 6 (Debugging Tools)
- Category 3 (Memory): "Address sanitization: Chapter 4" -- should be Chapter 2, file 03 (NOC address sanitization)
- Category 4 (Dispatch): "Dispatch hang debugging: Chapter 5" -- should be Chapter 4
- Category 6 (Host-Device): "Host-device debugging: Chapter 6" -- should be Chapter 4
A reader following these references will land in the wrong chapter and miss the relevant content entirely.
**Suggested fix:** Correct the references to:
- Category 2: "NOC hang mechanisms: Chapter 2. NOC debugging tools: Chapter 6."
- Category 3: "Memory-related hang causes: Chapter 3. NOC address sanitization: Chapter 2, `03_noc_address_sanitization_and_violations.md`."
- Category 4: "Dispatch hang causes and debugging: Chapter 4."
- Category 6: "Host-device interaction hangs: Chapter 4."

## Issue 2: 5-part diagnostic format in ch01 contradicts the plan and will produce inconsistent content downstream
**File:** 01_what_is_a_hang.md
**Lines:** ~348-356
**Problem:** The plan specifies the canonical 5-part format as: **(1) Symptom, (2) Root Cause, (3) Diagnosis Steps, (4) Fix, (5) Prevention**. The chapter instead defines: **(1) Mechanism, (2) Symptom, (3) Root Cause, (4) Diagnosis, (5) Mitigation**. These are materially different. "Fix" (a specific code change) vs. "Mitigation" (a risk reduction strategy) have different semantics. "Diagnosis Steps" (actionable procedure) vs. "Diagnosis" (analytical observation) set different expectations. Authors of Chapters 2-5 who follow the ch01 format will produce content structurally inconsistent with the plan's intended practitioner-friendly structure. Readers expecting a concrete "Fix" will instead get a vague "Mitigation".
**Suggested fix:** Align the 5-part format with the plan: **(1) Symptom** -- what the developer observes, **(2) Root Cause** -- what is actually happening, **(3) Diagnosis Steps** -- which tools to use and what to look for, **(4) Fix** -- the code change or configuration needed, **(5) Prevention** -- how to avoid the issue in the first place. If the "Mechanism" field is considered valuable, it can be added as supplementary context within each entry rather than replacing one of the canonical five fields.

---

# Agent B Review: Chapter 1 -- Pass 2

**No feedback -- chapter approved.**

Both issues from Pass 1 have been resolved:

- The 5-part diagnostic format in `01_what_is_a_hang.md` (lines 350-355) now reads Symptom / Root Cause / Diagnosis Steps / Fix / Prevention, matching the plan.
- The chapter cross-references in `03_hang_taxonomy.md` have been corrected. All six categories now point to appropriate chapters consistent with the guide structure.

No new factual errors, critical coherence problems, or critical structural gaps were identified in this pass.

---

# Agent B Review: Chapter 1 -- Pass 3

## Issue 1: Fabricated waypoint code `AH` in compressed `assert_and_hang` cross-reference
**File:** `02_blocking_primitives_taxonomy.md`
**Line:** 556
**Problem:** The compressed `assert_and_hang` subsection states: "It uses waypoint `AH` and enters `while(1){}` on Tensix or calls `erisc_exit()` on Ethernet cores." Two factual errors were introduced during compression:

1. **Waypoint `AH` does not exist.** The actual `assert_and_hang` code (shown in `01_what_is_a_hang.md` lines 252-262) contains no `WAYPOINT()` call. The function writes diagnostic data to the assert mailbox (`v->line_num`, `v->tripped`, `v->which`) and then enters `while(1) { ; }`. A developer building diagnostic tooling that scans for waypoint `AH` to detect assertion failures would be looking for a signal that is never emitted. The correct diagnostic indicator is the assert mailbox contents, not a waypoint.

2. **`erisc_exit()` is not mentioned in the source material.** The detailed description in `01_what_is_a_hang.md` (line 273) says ERISC cores "exit back to base firmware" -- it does not name a specific function `erisc_exit()`. Introducing this unverified function name could cause a developer to search for a symbol that may not exist under that exact name.

**Suggested fix:** Replace the sentence with: "It enters `while(1){}` on Tensix cores (with no waypoint written) or exits back to base firmware on Ethernet cores." This matches the source material in `01_what_is_a_hang.md` without introducing fabricated details.

---

# Agent B Review: Chapter 1 -- Pass 4

**No feedback -- chapter approved.**

The Pass 3 issue (fabricated `AH` waypoint and `erisc_exit()` function name in the `assert_and_hang` cross-reference in `02_blocking_primitives_taxonomy.md`) has been properly fixed. The corrected text at line 556 now accurately states that `assert_and_hang` writes diagnostic data to the assert mailbox, enters `while(1){}` on Tensix cores, exits back to base firmware on ERISC cores, and does not use the WAYPOINT mechanism. This matches the source material in `01_what_is_a_hang.md`.

No new factual errors, misleading claims, or structural issues were identified across all four content files and the index.
