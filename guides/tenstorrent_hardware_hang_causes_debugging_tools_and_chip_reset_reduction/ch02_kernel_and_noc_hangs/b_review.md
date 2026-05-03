# Agent B Review — Chapter 2

## Pass 1

**Issues found:** 2

### Issue 1: NKFW assert types incorrectly attributed to the BRISC firmware check
- **File:** `01_risc_synchronization_and_deadlocks.md`
- **Location:** Hang Cause 2.1.12, lines 637-643 (the "assert types provide specific information" paragraph)
- **Category:** Factual error
- **Problem:** The chapter states that the NKFW post-kernel check in `brisc.cc` produces four specific assert types: `DebugAssertNCriscNOCReadsFlushedTripped` (4), `DebugAssertNCriscNOCNonpostedWritesSentTripped` (5), `DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped` (6), and `DebugAssertNCriscNOCPostedWritesSentTripped` (7). This is incorrect. The actual NKFW code in `brisc.cc` (lines 545-549) calls the generic `ASSERT(condition)` macro without passing a specific assert type, so all five checks produce `DebugAssertTripped = 3` when they fail. The specialized assert types (values 4-7) ARE used, but in the kernel wrappers (`brisck.cc`, `ncrisck.cc`, `active_erisck.cc`, `idle_erisck.cc`) — not in the BRISC firmware NKFW check. A developer reading the watcher assert mailbox after a NKFW failure and seeing `tripped = 3` would be confused by the chapter's claim that they should see values 4-7.
- **Evidence:** `brisc.cc` lines 545-549 use `ASSERT(ncrisc_dynamic_noc_reads_flushed(noc))` (no second argument), which defaults to `DebugAssertTripped = 3` per `assert.h` line 43-47. The specific types are used in `brisck.cc` lines 87-90: `ASSERT(ncrisc_noc_reads_flushed(NOC_INDEX), DebugAssertNCriscNOCReadsFlushedTripped)`.
- **Suggested fix:** Revise the paragraph to state that the NKFW check in `brisc.cc` uses the generic `DebugAssertTripped` (value 3) for all five assertions, and note that the specialized assert types (values 4-7) are used in the post-kernel checks in the kernel wrapper files (`brisck.cc`, `ncrisck.cc`, etc.), not in the NKFW firmware check itself. Alternatively, clarify that both the firmware and kernel wrapper perform similar checks, but with different assert type granularity.

### Issue 2: Alignment Violation section lacks hang cause numbering in the 2.X.Y scheme
- **File:** `03_noc_address_sanitization_and_violations.md`
- **Location:** Line 525, section heading "## Alignment Violation (`NocAlignment`)"
- **Category:** 5-part format violation (structural)
- **Problem:** Every other hang cause in this file uses the "## Hang Cause 2.3.X: Title" numbering format (2.3.1 through 2.3.10). The Alignment Violation section follows 2.3.10 but is titled simply "## Alignment Violation (`NocAlignment`)" without a number. The chapter index (`index.md`) states that "Each scenario is numbered with a 2.X.Y scheme for precise cross-referencing." A reader attempting to cross-reference this section by number cannot do so. The section does contain all five parts (Symptom/Root Cause/Diagnosis Steps/Fix/Prevention), so the content structure is correct — only the numbering is missing.
- **Evidence:** All ten preceding sections in the same file use the numbered format: "Hang Cause 2.3.1" through "Hang Cause 2.3.10". The Summary table at the end of the file references scenarios 2.3.1-2.3.10 but does not include the alignment violation, further confirming the omission.
- **Suggested fix:** Rename the heading to "## Hang Cause 2.3.11: Alignment Violation (`NocAlignment`)" and add a corresponding row to the Summary table.

**Verdict:** NEEDS REVISION
