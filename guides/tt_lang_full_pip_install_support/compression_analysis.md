# Cross-Chapter Compression Analysis — Pass 1

## Summary
- Total chapters analyzed: 8
- Cross-chapter redundancies found: 12
- Estimated line savings: ~120 lines

## CRUCIAL Suggestions

None. The redundancies identified are all minor — repetitions serve as contextual reminders appropriate for a multi-chapter guide where readers may enter at different chapters. No verbatim tables, code blocks, or explanations are duplicated at a level that would confuse readers or create maintenance hazards.

## MINOR Suggestions

### M1. Extension module table repeated across Ch3 index, Ch6 main_wheel_design, and Ch7 so_bundling_and_rpath

The `_ttlang` / `_ttmlir` extension module descriptions (CMake targets, source files, EMBED_CAPI_LINK_LIBS) appear in near-identical form in:
- Ch3 `index.md` lines 16-19 (authoritative definition)
- Ch6 `main_wheel_design.md` lines 12-17 (repeated with slightly different column headers)
- Ch7 `so_bundling_and_rpath.md` lines 11-18 (repeated again with CAPI link details)

**Suggestion:** Ch6 and Ch7 should reference Ch3's table via a cross-link ("see Ch3 for the full extension module catalogue") and include only the columns specific to their chapter's concern (wheel contents for Ch6, RPATH for Ch7). Savings: ~20 lines.

### M2. `MLIR_PYTHON_PACKAGE_PREFIX=ttl.` explained three times

The compile definition and its purpose are explained in:
- Ch3 `index.md` lines 38-48 (authoritative)
- Ch7 `mlir_dialect_bindings.md` lines 36-61 (full re-explanation with comparison table)
- Ch3 `mlir_dependency_chain.md` line 190 (brief back-reference)

**Suggestion:** Ch7's `mlir_dialect_bindings.md` could shorten the prefix explanation to a single sentence with a cross-link to Ch3, keeping only the comparison table (which adds new value). Savings: ~15 lines.

### M3. `TTLangPythonCAPI` library described in Ch3 index, Ch6 main_wheel_design, and Ch7 so_bundling_and_rpath

The CAPI library's role and construction (`add_mlir_python_common_capi_library`) is described in:
- Ch3 `index.md` lines 51-86 (authoritative, with full dependency tree)
- Ch6 `main_wheel_design.md` lines 17-19 (brief but restates creation mechanism)
- Ch7 `so_bundling_and_rpath.md` lines 22-28 (restates creation and versioned soname detail)

**Suggestion:** Ch6 and Ch7 should use a one-line summary with a cross-link to Ch3's dependency tree rather than re-explaining the `add_mlir_python_common_capi_library` mechanism. Savings: ~10 lines.

### M4. `auditwheel repair` explained in Ch4 lessons_learned and Ch7 so_bundling_and_rpath

Both files explain what `auditwheel repair` does (vendoring, RPATH rewriting, tag rewriting):
- Ch4 `lessons_learned.md` lines 79-101 (general lessons from prior art)
- Ch7 `so_bundling_and_rpath.md` lines 74-126 (TT-Lang-specific application)

The Ch7 version is the actionable one with TT-Lang-specific `--exclude` flags and concrete commands. The Ch4 version is more general.

**Suggestion:** Acceptable overlap — Ch4 teaches the concept from prior art, Ch7 applies it. No change needed, but Ch7 could add a "For background on auditwheel across MLIR projects, see Ch4 lessons_learned" cross-link. Savings: ~5 lines if Ch4's explanation is trimmed.

### M5. `scikit-build-core` vs setuptools trade-off discussed in Ch4 lessons_learned and Ch5 pyproject_toml_changes

Both files evaluate the setuptools vs scikit-build-core choice:
- Ch4 `lessons_learned.md` lines 9-42 (general comparison with pros/cons)
- Ch5 `pyproject_toml_changes.md` lines 86-112 (TT-Lang-specific evaluation)

Both reach the same conclusion (start with setuptools, migrate to scikit-build-core later).

**Suggestion:** Ch5 should reference Ch4's general comparison and focus only on the TT-Lang-specific decision. The duplicated pros/cons list could be removed from Ch5. Savings: ~15 lines.

### M6. Five build phases table appears in Ch1 index and Ch1 cmake_architecture

The five-phase summary table in Ch1 `index.md` lines 9-16 and the CMake architecture file's include-order diagram (lines 33-49) convey overlapping information. This is intra-chapter overlap and acceptable for an index-then-detail structure.

**Suggestion:** No change needed.

### M7. Toolchain wheel pattern described in Ch4 lessons_learned and Ch6 index

Ch4 `lessons_learned.md` lines 45-72 discusses the "toolchain wheel pattern" abstractly with a comparison table. Ch6 `index.md` lines 8-14 presents the same concept as the TT-Lang-specific "build time asymmetry" table.

**Suggestion:** Ch6 could reference Ch4's comparison table instead of restating the general pattern. The TT-Lang-specific duration/trigger table is unique to Ch6 and should stay. Savings: ~10 lines.

### M8. `MANIFEST.in` discussed in Ch4 lessons_learned and Ch5 setup_py_fixes

Both files discuss sdist correctness and `MANIFEST.in`:
- Ch4 `lessons_learned.md` lines 110-151 (general lesson with minimal example)
- Ch5 `setup_py_fixes.md` lines 215-251 (complete TT-Lang-specific MANIFEST.in)

**Suggestion:** Acceptable overlap. Ch4 teaches the lesson, Ch5 provides the implementation. A cross-link from Ch5 to Ch4's lesson would help readers understand the rationale. Savings: ~5 lines if Ch4's minimal example is removed (it adds little over Ch5's complete version).

### M9. `cibuildwheel` configuration appears in Ch7 index and Ch6 build_pipeline

- Ch7 `index.md` lines 39-54 quotes existing `[tool.cibuildwheel]` config
- Ch6 `build_pipeline.md` lines 86-136 provides full cibuildwheel config for both wheels

**Suggestion:** Ch7 should reference Ch6's pipeline config rather than re-quoting the same `pyproject.toml` snippet. Savings: ~10 lines.

### M10. Version generation discussed in Ch2 cmake_build_class and Ch5 setup_py_fixes

Both discuss the date-based version in `setup.py` and the git-tag version in CMake:
- Ch2 `cmake_build_class.md` lines 209-254 (documents the problem)
- Ch5 `setup_py_fixes.md` lines 170-213 (proposes the fix)

**Suggestion:** Acceptable overlap — Ch2 documents the as-is state, Ch5 proposes the fix. The problem statement in Ch5 could be shortened to a cross-link to Ch2. Savings: ~10 lines.

### M11. `setup.py` `cwd` bug described in Ch2 why_pip_install_fails and Ch5 setup_py_fixes

- Ch2 `why_pip_install_fails.md` lines 42-60 (documents the bug with scenario table)
- Ch5 `setup_py_fixes.md` lines 8-40 (restates problem and provides fix)

**Suggestion:** Ch5 could reference Ch2's scenario table instead of re-explaining the working-directory mismatch. Savings: ~10 lines.

### M12. Sim-only requirements listed in Ch8 index and Ch8 design_options

The runtime dependency list for the simulator appears in:
- Ch8 `index.md` lines 64-71
- Ch8 `design_options.md` lines 31-33 (referenced but not fully listed)

**Suggestion:** Minor intra-chapter overlap; acceptable for readability.

## Load-Bearing Evidence

1. **Extension module descriptions (M1):** Ch3 `index.md` lines 16-19 defines `_ttlang` and `_ttmlir` with CMake targets, source files, and purpose. Ch6 `main_wheel_design.md` lines 12-17 repeats the same information with the column "What It Contains" that closely mirrors Ch3's "Purpose" column. Ch7 `so_bundling_and_rpath.md` lines 11-18 repeats again with EMBED_CAPI_LINK_LIBS details already present in Ch3 `index.md` lines 121-124.

2. **MLIR_PYTHON_PACKAGE_PREFIX (M2):** Ch3 `index.md` lines 38-48 provides the cmake snippet and explains the `ttl.` prefix convention with references to IREE and torch-mlir. Ch7 `mlir_dialect_bindings.md` lines 36-61 re-explains the same concept with an identical cmake snippet (`add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=ttl.")`) and a similar comparison table (IREE `iree.compiler.`, torch-mlir `torch_mlir.`, TT-Lang `ttl.`).

3. **scikit-build-core evaluation (M5):** Ch4 `lessons_learned.md` lines 13-35 lists setuptools advantages (maximum control, env-var overrides, battle-tested) and scikit-build-core advantages (PEP 517 compliance, editable installs, declarative config). Ch5 `pyproject_toml_changes.md` lines 108-112 restates the same pros/cons and reaches the identical conclusion: "Start with Option A (keep setuptools), migrate later."

4. **auditwheel --exclude (M4):** Ch4 `lessons_learned.md` lines 87-97 shows an `auditwheel repair --exclude` example with torch libraries. Ch7 `so_bundling_and_rpath.md` lines 94-107 shows the TT-Lang-specific `--exclude` with LLVM/tt-metal libraries. The commands differ in their excluded libraries, so this is legitimate parallel structure rather than pure duplication.

5. **cibuildwheel config (M9):** Ch7 `index.md` lines 42-44 quotes `build = "cp311-manylinux_x86_64*"`, `skip = "*-musllinux_*"`, `build-verbosity = 2`. Ch6 `build_pipeline.md` lines 119-123 quotes the identical three lines. Both then extend the config with different additions (Ch6 adds `before-build` and `environment`, Ch7 discusses IN_CIBW_ENV).

## VERDICT
- Crucial updates: no
