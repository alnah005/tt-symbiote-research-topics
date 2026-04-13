# Compression Analysis: Two-Phase Wheel Architecture — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~796 lines
- Estimated post-compression line count: ~620 lines
- Estimated reduction: ~22%

## CRUCIAL Suggestions

### C1: Duplicate CMake build invocations across three files

The exact same `cmake -G Ninja ... -DTTLANG_BUILD_TOOLCHAIN=ON` invocation appears in:
- `toolchain_wheel_design.md` lines 258-263 (build script)
- `build_pipeline.md` lines 38-43 (toolchain pipeline steps)
- `build_pipeline.md` lines 197-200 (developer "Building Your Own Toolchain" section)

The strip command (`find ... -name '*.so' -exec strip --strip-unneeded {} +`) also appears in both `toolchain_wheel_design.md` line 266 and `build_pipeline.md` lines 46-47.

**Recommendation:** The build script in `toolchain_wheel_design.md` already shows the canonical build flow. `build_pipeline.md` should reference it rather than repeating it. The toolchain pipeline steps (lines 38-47) can say "Run `scripts/build-toolchain-wheel.sh` (see toolchain_wheel_design.md)" instead of re-listing the cmake and strip commands. The developer section (Option A) similarly duplicates the cmake invocation.

**Estimated savings:** ~25 lines

### C2: Toolchain wheel contents described three times

The contents of each wheel are enumerated in:
1. `index.md` lines 20-31 — the two-package ASCII block listing contents
2. `toolchain_wheel_design.md` lines 9-53 — detailed tables of all artifacts
3. `toolchain_wheel_design.md` lines 59-103 — the package layout tree

Items (2) and (3) convey nearly identical information in two different formats (tabular vs. directory tree). Every artifact appears in both the tables AND the tree. Merging these into a single annotated directory tree would be more scannable and eliminate the duplication.

**Recommendation:** Replace the four artifact tables (sections 1-4, lines 9-53) and the directory tree (lines 59-103) with a single annotated directory tree that includes purpose notes as inline comments. Keep the tables only for artifacts whose purpose is non-obvious.

**Estimated savings:** ~30 lines

### C3: `ttl-toolchain` install command repeated verbatim across files

The command `pip install ttl-toolchain==0.1.250413 --index-url https://internal.example.com/simple/` appears at:
- `build_pipeline.md` line 78 (extension pipeline)
- `build_pipeline.md` line 141 (cibuildwheel before-build)
- `build_pipeline.md` line 171 (developer first-time setup)

Additionally, `pip install ttl-toolchain==0.1.250413` (without index URL) appears in `main_wheel_design.md` line 197 (editable install workflow).

**Recommendation:** Show the full command once (in the developer workflow or CI overview), then reference it elsewhere. The cibuildwheel config already shows it in TOML; the narrative step doesn't need to repeat it.

**Estimated savings:** ~8 lines

### C4: Editable install workflow duplicated across two files

`main_wheel_design.md` lines 191-207 and `build_pipeline.md` lines 161-188 both describe the editable install workflow with nearly identical commands and explanations. Both show `pip install ttl-toolchain`, then `pip install -e python/`, then explain that C++ changes require recompilation.

**Recommendation:** Keep the developer workflow in `build_pipeline.md` (its natural home) and remove the "Editable Install Workflow" section from `main_wheel_design.md`, replacing it with a cross-reference.

**Estimated savings:** ~18 lines

## MINOR Suggestions

### M1: "Why Not a Single Large Wheel?" in index.md (lines 53-57)
The four reasons listed are solid but points 1-2 are already covered by the "Build Time Asymmetry" table and the preceding paragraph (lines 14). Consider condensing to 2-3 bullet points.

### M2: Approach A in main_wheel_design.md (lines 122-134)
Approach A (RPATH patching) is introduced as "Recommended" in the heading but immediately explained as impractical. The heading should say "RPATH Patching (Impractical)" or the recommendation label should be removed. Beyond the labeling issue, the explanation of why RPATH fails is verbose — the key point (cross-package RPATH is unpredictable) can be stated in 2 sentences instead of 5.

### M3: Approach C in main_wheel_design.md (lines 165-167)
Three lines stating `LD_LIBRARY_PATH` is not recommended. This could be a single sentence appended to the Approach B discussion rather than a separate subsection.

### M4: Version "0.1.250413" hardcoded throughout
The literal version string `0.1.250413` appears ~15 times across all four files. While this is intentional for concreteness, a brief note in `index.md` that this is an example version would reduce the impression that the docs are tightly coupled to a single release.

### M5: Wheel metadata block in main_wheel_design.md (lines 177-187)
The TOML metadata block duplicates the dependency list already shown in the `pyproject.toml` install requirements block (lines 100-108), just in a slightly different format. One of these could be removed with a note that setuptools generates the metadata from pyproject.toml.

### M6: "Caching" note in build_pipeline.md (line 66)
The ccache explanation is useful but the parenthetical "(already configured in `BuildLLVM.cmake` and `BuildTTMetal.cmake`)" could be trimmed — readers of this chapter likely already know the cmake modules.

### M7: auditwheel explained twice in build_pipeline.md
The `auditwheel` skip rationale appears in both the toolchain cibuildwheel section (line 122) and the main wheel auditwheel section (lines 150-159). The explanation of why auditwheel repair is skipped (it would duplicate LLVM libs) is the same concept applied to both wheels. A single "auditwheel strategy" subsection could cover both.

## Load-Bearing Evidence

All CRUCIAL items involve content that exists in multiple locations with no cross-referencing. Here are the key duplicated fragments:

1. **cmake invocation** (`cmake -G Ninja -S . -B build -DTTLANG_BUILD_TOOLCHAIN=ON ...`): `toolchain_wheel_design.md:258`, `build_pipeline.md:38`, `build_pipeline.md:197`
2. **strip command** (`find ... strip --strip-unneeded`): `toolchain_wheel_design.md:266`, `build_pipeline.md:47`
3. **Artifact tables vs directory tree**: `toolchain_wheel_design.md:9-53` vs `toolchain_wheel_design.md:59-103`
4. **Editable install flow**: `main_wheel_design.md:191-207` vs `build_pipeline.md:161-188`
5. **pip install ttl-toolchain command**: `build_pipeline.md:78,141,171`, `main_wheel_design.md:197`

## VERDICT
- Crucial updates: yes

---

## Change Log (Applied by Agent A)

### Agent B Fixes
1. **B1: Approach A mislabel.** Renamed heading from "Approach A: RPATH Patching (Recommended)" to "Approach A: RPATH Patching (Impractical)" in `main_wheel_design.md`. Condensed the verbose RPATH explanation to two sentences (also addresses M2).
2. **B2: Broken Chapter 7 link.** Changed `build_pipeline.md` navigation footer from a broken link to plain text: "Chapter 7 -- Wheel Packaging and Platform Compliance (forthcoming)".

### Crucial Compressions
3. **C1: Duplicate CMake invocations.** Replaced the inline cmake + strip commands in `build_pipeline.md` toolchain pipeline steps 2-3 with a reference to `scripts/build-toolchain-wheel.sh` (canonical source in `toolchain_wheel_design.md`). Replaced the Option A cmake invocation in "Building Your Own Toolchain" with a `bash scripts/build-toolchain-wheel.sh` call.
4. **C2: Artifact tables + directory tree merged.** Replaced the four artifact tables and the separate directory tree in `toolchain_wheel_design.md` with a single annotated directory tree. Each artifact has an inline comment noting its purpose and whether it is build-time only.
5. **C3: `pip install ttl-toolchain` repetition.** Removed the instance from `main_wheel_design.md` (subsumed by C4). Remaining occurrences in `build_pipeline.md` serve distinct contexts (CI step, TOML config literal, developer instructions) and are kept.
6. **C4: Editable install workflow deduplication.** Removed the full editable-install section from `main_wheel_design.md` and replaced it with a cross-reference to `build_pipeline.md#developer-workflow`.

### Minor Items Also Applied
7. **M2: Approach A verbosity.** Condensed the RPATH explanation from a code block + 5 sentences to 2 sentences (done as part of B1).
8. **M3: Approach C folded.** Removed the separate "Approach C: LD_LIBRARY_PATH" subsection and appended its key point as a sentence in the "Recommended Approach" paragraph.
9. **M4: Example version note.** Added a note in `index.md` clarifying that `0.1.250413` is a concrete example version, not a hardcoded requirement.

---

# Compression Analysis: Two-Phase Wheel Architecture — Pass 2

## Re-Check of CRUCIAL Items (C1--C4)

### C1: Duplicate CMake build invocations — RESOLVED
The inline cmake and strip commands in `build_pipeline.md` have been replaced with references to `scripts/build-toolchain-wheel.sh`. The canonical invocation lives solely in `toolchain_wheel_design.md` lines 212-216. The "Building Your Own Toolchain" section (line 184) now calls `bash scripts/build-toolchain-wheel.sh` instead of repeating the cmake invocation. No residual duplication found.

### C2: Artifact tables + directory tree — RESOLVED
`toolchain_wheel_design.md` now has a single annotated directory tree (lines 13-57) with inline comments indicating purpose and build-time-only markers. The separate four-table + tree representation has been eliminated. The introductory prose (line 7) still references four categories by number, and the tree uses matching `(1)`, `(2)`, `(3)`, `(4)` annotations, so these remain consistent.

### C3: `pip install ttl-toolchain` repetition — RESOLVED
Three instances remain in `build_pipeline.md` (extension pipeline step at line 66, cibuildwheel TOML at line 129, developer setup at line 159). Each serves a distinct context (CI narrative, config literal, developer instructions). The fourth instance in `main_wheel_design.md` was removed as part of C4. No further compression warranted.

### C4: Editable install workflow deduplication — RESOLVED
`main_wheel_design.md` lines 175-177 now contain only a cross-reference: "see the [Developer Workflow](./build_pipeline.md#developer-workflow) section in `build_pipeline.md`." The full workflow lives exclusively in `build_pipeline.md` lines 149-197. No residual duplication.

## VERDICT
- Crucial updates: no

## Load-Bearing Evidence

- **`index.md`**: Line 33 version note ("The version string `0.1.250413` is used throughout this chapter as a concrete example") confirms M4 was applied. No duplicate content with other files.
- **`toolchain_wheel_design.md`**: Lines 13-57 contain the sole artifact layout (single annotated tree). Lines 212-216 contain the sole cmake invocation (canonical build script). No content is duplicated elsewhere.
- **`main_wheel_design.md`**: Lines 175-177 contain a cross-reference to `build_pipeline.md#developer-workflow` instead of a duplicated workflow. The editable install content is gone.
- **`build_pipeline.md`**: Line 36 references `scripts/build-toolchain-wheel.sh` instead of inlining cmake commands. Line 184 likewise references the script. The three `pip install ttl-toolchain` occurrences serve non-overlapping contexts (CI step, TOML config, developer setup).

## MINOR Suggestions

### M5 (carried from pass 1): Wheel metadata block still overlaps with install requirements
`main_wheel_design.md` lines 161-171 (wheel metadata TOML) list the same five dependencies already shown in the `[project] dependencies` block at lines 100-108. Since setuptools generates the metadata from `pyproject.toml`, one of these blocks could be replaced with a note that the metadata is auto-generated. Estimated savings: ~10 lines.

### M8: Step numbering gap in `build_pipeline.md`
The toolchain pipeline steps jump from step 3 (line 38) to step 5 (line 47). Step 4 is missing. This appears to be a remnant of the C1 compression removing a step without renumbering. Should be renumbered to 1-2-3-4-5.
