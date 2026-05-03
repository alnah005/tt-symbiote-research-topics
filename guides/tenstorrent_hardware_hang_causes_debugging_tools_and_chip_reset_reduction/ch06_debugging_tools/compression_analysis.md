# Chapter 6 Compression Analysis

**Total lines across 7 content files:** ~2,347 (excluding b_review.md)
**Estimated compressible lines:** ~95-115

---

## 1. Duplicate "When to Use X vs. Alternatives" Decision Trees

### Redundancy
The `index.md` contains a master "Tool Selection Decision Tree" (lines 19-53) and a "Tool Selection Quick Reference" table (lines 89-102) that substantially overlap with the per-file "When to Use" decision trees in every content file:
- `01_watcher_system.md` Section 6.1.1 (lines 14-44)
- `02_watcher_dump_tool.md` Section 6.2.1 (lines 14-48)
- `03_dprint_server.md` Section 6.3.1 (lines 14-34)
- `04_tt_triage_tool.md` Section 6.4.1 (lines 18-36)
- `05_profiler_tracy_and_noc_debug.md` Section 6.5.1 (lines 14-36)
- `06_debug_delay_and_timing_perturbation.md` Section 6.6.1 (lines 18-38)

The per-file decision trees repeat the same routing logic from the index -- e.g., "process dead, Watcher was on? --> read watcher.log" appears in index.md, 01_watcher_system.md, and 02_watcher_dump_tool.md. Similarly, "need printf-style debugging --> use DPRINT" appears in both 01_watcher_system.md and index.md.

### Suggested Fix
Keep the master decision tree and quick-reference table in `index.md` only. In each content file, replace the full decision tree with a 2-3 line summary: "Use this tool when [primary use case]. See index.md Tool Selection Decision Tree for alternatives." Keep only the "right choice when / NOT right choice when" bullet lists which contain tool-specific nuance.

### Estimated Savings
~60 lines total (roughly 10 lines per file x 6 files)

---

## 2. Duplicate Debug Delay Environment Variable Tables

### Redundancy
The debug delay environment variables are documented in full in two places:
- `01_watcher_system.md` Section 6.1.3 "Debug Delay Configuration" (lines 152-162) -- 11 lines
- `06_debug_delay_and_timing_perturbation.md` Section 6.6.3 (lines 84-118) -- 35 lines with expanded detail

Both list `TT_METAL_WATCHER_DEBUG_DELAY`, `TT_METAL_READ_DEBUG_DELAY_CORES`, `TT_METAL_WRITE_DEBUG_DELAY_CORES`, `TT_METAL_ATOMIC_DEBUG_DELAY_CORES`, and the RISC targeting variables.

### Suggested Fix
In `01_watcher_system.md`, replace the Debug Delay Configuration subsection with a single-row summary entry and a forward reference: "Debug delay configuration (7 env vars): See Section 6.6.3 for complete reference." Keep only `TT_METAL_WATCHER_DEBUG_DELAY` in the 6.1 table since it is the master enable flag and is relevant context for the watcher env var reference.

### Estimated Savings
~8 lines

---

## 3. Repeated "watcher_dump then tt-triage" Workflow

### Redundancy
The recommended post-mortem workflow "1. watcher_dump first, 2. tt-triage second, 3. cross-reference" is stated nearly identically in three places:
- `02_watcher_dump_tool.md` Section 6.2.7 (lines 161-169) -- "This order matters because tt-triage's initialization may modify some device state..."
- `04_tt_triage_tool.md` Section 6.4.6 Workflow 3 (lines 213-215) -- identical 3-step sequence
- `index.md` Progressive Diagnosis Pipeline, Stage 1 (lines 64-67) -- same ordering

### Suggested Fix
Keep the detailed explanation in `02_watcher_dump_tool.md` (Section 6.2.7) as the canonical location since it explains the rationale. In `04_tt_triage_tool.md`, replace Workflow 3 with a single line: "See Section 6.2.7 for the recommended watcher_dump + tt-triage combined workflow and ordering rationale."

### Estimated Savings
~6 lines

---

## 4. Repeated ERISC IRAM Disabling Note

### Redundancy
The fact that enabling Watcher or DPRINT disables ERISC IRAM mode is stated in:
- `01_watcher_system.md` line 108: "Enabling Watcher or DPRINT automatically disables ERISC IRAM mode..."
- `03_dprint_server.md` line 219 (Interaction with Watcher table): "Enabling DPRINT (like Watcher) disables ERISC IRAM mode; may affect ETH kernel performance"

### Suggested Fix
Keep both mentions. The first is in watcher context, the second in DPRINT context, and both are brief. Readers of either file independently need this information. **No action needed** -- this is acceptable cross-referencing, not bloat.

### Estimated Savings
0 lines (not a genuine redundancy)

---

## 5. Repeated Three Assert Modes Comparison

### Redundancy
The three assert modes (Watcher assert, Lightweight assert, Disabled) are documented:
- `01_watcher_system.md` Section 6.1.4 "Three assert modes" (lines 280-287) -- 8 lines
- `05_profiler_tracy_and_noc_debug.md` Section 6.5.6 "Three Assert Modes Comparison" (lines 249-258) -- 10-line table

The 6.5 version is more detailed (includes binary size, production-safe columns). The 6.1 version is briefer.

### Suggested Fix
In `01_watcher_system.md`, replace the three-row "Three assert modes" table with a forward reference: "For a full comparison of all three assert modes (Watcher, Lightweight, Disabled), see Section 6.5.6." Keep only the Watcher assert mode description inline since it is the focus of Section 6.1.

### Estimated Savings
~5 lines

---

## 6. Duplicate Tool Selection Summary Tables

### Redundancy
`index.md` has a "Tool Selection Quick Reference" (lines 89-102, 10 rows) and `05_profiler_tracy_and_noc_debug.md` has a "Tool Selection Summary Table" at Section 6.5.9 (lines 354-365, 6 rows). Four of the six rows in 6.5.9 repeat content from the index table:
- "Need kernel timing data --> Tracy" (index row 8)
- "Device not responding --> tt-smi" (index row 7)
- "Need production-safe asserts --> Lightweight asserts" (index row 9)
- "Intermittent data corruption --> NOC debug dump" (similar to index row 6)

### Suggested Fix
Remove the 6.5.9 summary table entirely. Its content is covered by the index table. The two unique rows ("Multi-chip fabric issues" and "Performance degradation pre-hang") can be added to the index table if not already present.

### Estimated Savings
~12 lines

---

## 7. Repeated "Device Hard-Hung, Reset Required" Pattern

### Redundancy
The sequence "device is hard-hung --> tt-smi -r to reset --> L1 state is lost --> focus on host-side logs" appears in:
- `02_watcher_dump_tool.md` Scenario 6.2.3 (lines 209-221)
- `04_tt_triage_tool.md` Scenario 6.4.4 (lines 319-331)
- `05_profiler_tracy_and_noc_debug.md` Scenario 6.5.4 (lines 339-351)

Each scenario adds tool-specific context (watcher_dump hangs during attach; tt-triage cannot connect; tt-smi shows error state), so they are not pure duplicates. However, the diagnosis steps and fix sections are nearly identical.

### Suggested Fix
Keep all three scenarios (they each describe a different tool's failure mode). Compress the shared "Fix" and "Prevention" blocks by referencing a canonical description. Add a brief note to `index.md` prerequisites or a new subsection: "When the device is completely unresponsive, a chip reset via `tt-smi -r <device>` is required. All L1 state is lost; focus on host-side logs (watcher.log, Inspector serialized data)." Then in each scenario, the Fix/Prevention can be shortened to "See index.md for device reset procedure."

### Estimated Savings
~8 lines (3 scenarios x ~3 lines each, minus the canonical note)

---

## 8. Verbose Cross-Reference Blocks at File Endings

### Redundancy
Every content file ends with a "Cross-references" section listing chapter and section references. Some of these repeat information already in the index.md cross-reference table (lines 115-123). For example:
- `01_watcher_system.md` lines 483-489 list 6 cross-references, 5 of which are in the index table.
- `05_profiler_tracy_and_noc_debug.md` lines 367-372 list 4 cross-references, all in the index table.

### Suggested Fix
Keep per-file cross-references as they serve readers who land directly on a specific file. However, remove any cross-references that simply restate what is obvious from the section title (e.g., "Watcher as prerequisite for debug delay: Section 6.1" in 06 is already stated in the prerequisites). Trim to only non-obvious or actionable cross-references.

### Estimated Savings
~6 lines total (1-2 lines per file from the most redundant entries)

---

## 9. index.md "Progressive Diagnosis Pipeline" Partially Duplicates Per-File Content

### Redundancy
The 6-stage pipeline in `index.md` (lines 57-87) restates tool usage that each section covers in depth. This is an intentional quick-reference and is valuable as a navigation aid.

### Suggested Fix
**No action needed.** The pipeline serves as an orientation map. It is concise (30 lines) and does not contain explanatory prose that duplicates per-file content.

### Estimated Savings
0 lines

---

## 10. Combined Tool Usage Matrix in 06 Overlaps with Index Quick Reference

### Redundancy
`06_debug_delay_and_timing_perturbation.md` Section 6.6.9 "Combined Tool Usage Matrix" (lines 399-411, 13 lines) overlaps with the `index.md` "Tool Selection Quick Reference" table. Both map symptoms/goals to tools.

### Suggested Fix
The 6.6.9 matrix is specifically scoped to timing-perturbation investigations and includes tools from other sections (Watcher, DPRINT, Tracy). This is a useful localized reference for the timing-perturbation workflow. **No action needed** -- its scope is sufficiently different from the index table.

### Estimated Savings
0 lines

---

## Summary

| # | Redundancy | Files Affected | Est. Lines Saved |
|---|-----------|----------------|-----------------|
| 1 | Duplicate per-file decision trees vs. index master tree | All 6 content files + index.md | ~60 |
| 2 | Debug delay env vars duplicated in 6.1 and 6.6 | 01, 06 | ~8 |
| 3 | watcher_dump + tt-triage workflow stated 3 times | 02, 04, index | ~6 |
| 4 | ERISC IRAM note (acceptable duplication) | 01, 03 | 0 |
| 5 | Three assert modes table in both 6.1 and 6.5 | 01, 05 | ~5 |
| 6 | Tool selection summary table duplicates index | 05, index | ~12 |
| 7 | "Device hard-hung, reset required" pattern x3 | 02, 04, 05 | ~8 |
| 8 | Verbose ending cross-reference blocks | All 6 content files | ~6 |
| **Total** | | | **~105** |

**Compression ratio:** ~105 / 2,347 = ~4.5%

The chapter is well-structured with relatively low redundancy. The largest compression opportunity (item 1) comes from the per-file decision trees that restate the index-level routing logic. The remaining items are modest but cumulatively meaningful.
