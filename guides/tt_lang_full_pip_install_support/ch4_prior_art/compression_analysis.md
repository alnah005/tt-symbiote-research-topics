# Compression Analysis: Prior Art — Pass 1

## Summary
- Total files analyzed: 3
- Estimated current line count: ~440 lines
- Estimated post-compression line count: ~330 lines
- Estimated reduction: ~25%

## CRUCIAL Suggestions

### C1. Duplicate summary table in index.md vs. case_studies.md

`index.md` lines 7-12 contain a full comparison table (Build Backend, LLVM Strategy, Wheel Size, Platform Support) for all four projects. Every cell in this table is restated — with more detail — in `case_studies.md`. The index table should be reduced to a one-line-per-project bullet list pointing readers to the case studies, or removed entirely since the chapter is only two clicks deep.

**Estimated savings:** ~10 lines from index.md

### C2. "Common Patterns" section in index.md duplicates lessons_learned.md

`index.md` lines 14-38 describe three patterns: pre-built LLVM in CI, bundled shared libraries, and separate toolchain/runtime wheels. These are restated with more depth in `lessons_learned.md` sections 2 (toolchain wheel pattern, lines 45-72), 3 (auditwheel/RPATH, lines 75-107), and 1 (build backend, lines 7-42). The index.md "Common Patterns" section is essentially a preview of lessons_learned.md content. It should be collapsed to a brief enumeration (3 bullets, ~5 lines) rather than full subsections with examples.

**Estimated savings:** ~20 lines from index.md

### C3. Toolchain wheel pattern table restated three times

The "toolchain wheel" concept appears in three locations:
1. `index.md` lines 30-38 (pattern 3: "Separate Toolchain / Compiler / Runtime Wheels")
2. `case_studies.md` in each "Key Takeaway for TT-Lang" section (lines 68-70, 125-127, 180-182, 219-221) — each reiterates how the respective project's approach maps to TT-Lang
3. `lessons_learned.md` lines 49-56 (a dedicated table mapping each project's slow/fast separation)

The table at `lessons_learned.md:49-56` and the prose at `index.md:30-38` convey the same information. One should be the single source of truth. The four per-case-study "Key Takeaway" paragraphs in `case_studies.md` each partly restate it again. These takeaways could be consolidated into a single "Implications for TT-Lang" paragraph at the end of `case_studies.md`.

**Estimated savings:** ~25 lines across all three files

### C4. auditwheel/delocate/delvewheel explanation duplicated

Platform-specific wheel repair tools are explained twice:
1. `index.md` lines 24-28 (3 bullets under "Bundled Shared Libraries")
2. `case_studies.md` lines 60-64 (torch-mlir's platform support section)
3. `lessons_learned.md` lines 79-83 (what auditwheel does) and 103-106 (macOS/Windows equivalents)

The index.md bullets and lessons_learned.md sections overlap almost entirely. The case_studies.md mention is project-specific and appropriate, but `index.md` should not also explain the mechanism.

**Estimated savings:** ~6 lines from index.md

## MINOR Suggestions

### M1. Hedging in lessons_learned.md section 1
Lines 9-10: "None of them have migrated to `scikit-build-core`, despite it being the recommended PEP 517 build backend for CMake-based projects. The reasons vary, but the trade-offs are clear:" — the phrase "The reasons vary, but the trade-offs are clear" is filler. Cut it.

### M2. Verbose advantage/disadvantage lists in lessons_learned.md
Lines 11-34: The `setuptools` vs. `scikit-build-core` comparison uses full prose paragraphs for each bullet. These could be tightened to single-line bullets (e.g., "Maximum control over CMake invocation" instead of "Maximum control over the CMake invocation (flags, targets, install components).").

### M3. Redundant "Key Takeaway for TT-Lang" preamble in each case study
Each of the four case studies ends with a "Key Takeaway for TT-Lang" subsection. The preamble phrasing is formulaic ("X demonstrates... TT-Lang could adopt..."). These could be merged into a single concluding section in `case_studies.md`, or shortened to one sentence each.

### M4. Wheel size tables appear in both index.md and case_studies.md
`index.md` line 9 shows "~222 MB" for torch-mlir; `case_studies.md` lines 52-58 provide a more detailed per-platform breakdown. The index table's wheel-size column is redundant given the detail in case_studies.md.

### M5. sdist section (lessons_learned.md lines 110-151) is thorough but over-explained
The section walks through the sdist problem, how each project handles it (all the same way: they don't), and then provides a sample `MANIFEST.in` before concluding that sdist is not viable. The conclusion at line 151 makes the preceding 40 lines somewhat unnecessary. Could be cut to ~15 lines: state the problem, note all projects skip sdist, give the recommendation.

### M6. Triton build caching section (case_studies.md lines 101-105) is tangential
Kernel-level compilation caching is a runtime concern, not a packaging/pip-install concern. These 5 lines could be removed or reduced to a single sentence noting its existence.

## Load-Bearing Evidence

- **index.md**: The summary comparison table (lines 7-12) and the three "Common Patterns" subsections (lines 14-38) are the primary sources of redundancy — every fact in them is covered more thoroughly in the other two files. However, the chapter navigation links (lines 40-45) are load-bearing and must remain.
- **case_studies.md**: The per-project "Key Takeaway for TT-Lang" subsections (lines 68-70, 125-127, 180-182, 219-221) collectively restate the toolchain-wheel insight four times with minor variation. The detailed build-backend, LLVM-strategy, and wheel-structure sections are load-bearing and should not be compressed.
- **lessons_learned.md**: The summary table at lines 157-163 is a clean, non-redundant distillation. The body sections (1-4) contain the substantive analysis. The toolchain-wheel table at lines 51-56 duplicates index.md's pattern description but is the more complete version and should be kept as the single source of truth.

## VERDICT
- Crucial updates: yes

---

## Change Log

### 2026-04-09 -- CRUCIAL suggestions applied (Agent A)

All four CRUCIAL items (C1-C4) applied:

- **C1 applied**: Replaced the 6-row summary comparison table in `index.md` with a 4-bullet "Projects Surveyed" list that points readers to `case_studies.md` for details.
- **C2 applied**: Collapsed the three "Common Patterns" subsections in `index.md` (Pre-built LLVM, Bundled Shared Libraries, Separate Toolchain Wheels) from full subsections with examples into a 3-bullet enumeration with cross-references to `lessons_learned.md`.
- **C3 applied**: Removed the four per-case-study "Key Takeaway for TT-Lang" subsections from `case_studies.md` and consolidated them into a single "Implications for TT-Lang" section at the end of the file. The toolchain-wheel table in `lessons_learned.md` remains the single source of truth.
- **C4 applied**: The `auditwheel`/`delocate`/`delvewheel` mechanism explanation was removed from `index.md` as part of the C2 collapse; `index.md` now links to `lessons_learned.md` section 3 instead. The project-specific mentions in `case_studies.md` were preserved as appropriate.

Navigation footers preserved in all files. MINOR suggestions (M1-M6) not applied in this pass.

---

# Compression Analysis: Prior Art — Pass 2

## Re-check of CRUCIAL Items

### C1. Summary table in index.md duplicated by case_studies.md — RESOLVED

`index.md` lines 5-11 now contain a 4-bullet "Projects Surveyed" list with one-line descriptions per project. Each bullet includes a brief stat (e.g., "~222 MB wheels") that also appears in `case_studies.md`, but these serve as navigational hints to help the reader decide whether to read the full case study. This is not redundant duplication — it is appropriate index-level summarization. No further action needed.

### C2. "Common Patterns" in index.md previews lessons_learned.md — RESOLVED

`index.md` lines 14-21 now contain exactly 3 numbered bullets, each one sentence plus a cross-reference link to the relevant `lessons_learned.md` section. No mechanism explanations, no examples, no subsections. This is the correct level of detail for a chapter index page. No further action needed.

### C3. Toolchain wheel pattern in 3 locations — RESOLVED

The pattern now lives in two locations:
1. `lessons_learned.md` lines 45-71 — the single source of truth with a detailed comparison table and TT-Lang-specific application.
2. `case_studies.md` lines 209-216 — a consolidated "Implications for TT-Lang" section that references the pattern alongside other cross-cutting observations.

The four per-case-study "Key Takeaway" subsections have been removed. The remaining mention in `case_studies.md`'s concluding section is a natural summary for that file, not a redundant restatement. `index.md` line 20 contains only a one-line pointer. No further action needed.

### C4. auditwheel explanation in index.md and lessons_learned.md — RESOLVED

`index.md` line 19 now names the tools parenthetically and links to `lessons_learned.md` section 3 — no mechanism explanation remains. `lessons_learned.md` lines 75-107 is the sole detailed treatment. `case_studies.md` mentions `auditwheel repair` in project-specific contexts (e.g., torch-mlir line 62, Triton line 119), which is appropriate since those are documenting what each project actually does, not explaining the tool itself. No further action needed.

## Load-Bearing Evidence

- **index.md**: The "Projects Surveyed" bullets (lines 7-10) and "Common Patterns" enumeration (lines 16-20) now serve purely as navigation aids with cross-reference links. The chapter-contents list and navigation footer (lines 22-27) are structural and must remain.
- **case_studies.md**: The four detailed case studies (torch-mlir, Triton, IREE, CIRCT) are load-bearing primary research. The "Implications for TT-Lang" section (lines 209-216) is the sole location for cross-project synthesis within this file.
- **lessons_learned.md**: All four lesson sections (1-4) contain substantive, non-duplicated analysis. The summary table (lines 157-163) is a clean distillation that does not repeat prose from elsewhere.

## MINOR Suggestions

### M1. Hedging filler in lessons_learned.md line 9
"The reasons vary, but the trade-offs are clear:" adds no information. Replace with a colon or cut entirely. (~8 words saved)

### M2. Verbose advantage/disadvantage lists in lessons_learned.md lines 11-34
The `setuptools` vs. `scikit-build-core` comparison uses parenthetical expansions in every bullet (e.g., "Maximum control over the CMake invocation (flags, targets, install components)."). The parentheticals could be cut since the reader understands "maximum control" in context. (~30 words saved)

### M3. sdist section (lessons_learned.md lines 110-151) conclusion undercuts its own length
Lines 110-151 walk through the sdist problem in detail, then conclude "sdist-only installs are not viable" (line 151). The sample `MANIFEST.in` (lines 142-149) is for a path the text explicitly recommends against. Consider cutting the `MANIFEST.in` example and reducing this section to ~15 lines: state the problem, note that all four projects skip sdist, give the recommendation. (~25 lines saved)

### M4. Triton build caching (case_studies.md lines 97-101)
Kernel-level compilation caching (`~/.triton/cache/`) is a runtime optimization concern, not a packaging concern. These 5 lines could be reduced to one sentence noting the feature exists, or removed entirely from a packaging-focused chapter. (~4 lines saved)

## VERDICT
- Crucial updates: no

---

## Change Log

### 2026-04-09 -- Pass 2 feedback applied (Agent A)

Three issues from Agent B's pass 2 review applied:

- **Issue 1 fixed (index.md line 18):** Cross-reference anchor for "Pre-built LLVM in CI" corrected from `#1-scikit-build-core-vs-setuptools--custom-cmakebuild` to `#2-the-toolchain-wheel-pattern`, which is the section that actually discusses pre-building LLVM.
- **Issue 2 fixed (case_studies.md line 211):** Changed "three patterns emerge" to "four patterns emerge" to match the four bullet points that follow.
- **Issue 3 fixed (index.md line 19):** Changed "all four projects" to "the three PyPI-publishing projects" since CIRCT does not use wheel repair tools (`auditwheel`, `delocate`, `delvewheel`).
