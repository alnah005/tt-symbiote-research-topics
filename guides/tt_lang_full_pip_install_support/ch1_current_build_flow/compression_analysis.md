# Compression Analysis: Current Build and Installation Flow -- Pass 1

## Summary
- Total files analyzed: 3
- Estimated current line count: ~520 lines
- Estimated post-compression line count: ~455 lines
- Estimated reduction: ~12%

## CRUCIAL Suggestions

### [cmake_architecture.md] ~lines 103-113 + ~lines 289-299
**Issue:** ccache detection is shown twice with near-identical code blocks: once in the `BuildLLVM.cmake` section ("ccache forwarding" subsection, showing `find_program(CCACHE_PROGRAM ccache)` + conditional) and again in the `TTLangCompilerSetup.cmake` section (showing the same `find_program` pattern with launcher variables). The LLVM subsection exists solely to show code that duplicates the compiler-setup section.
**Suggestion:** Remove the "ccache forwarding" subsection (lines 103-113) from the `BuildLLVM.cmake` section entirely. Replace with a single sentence: "ccache detection from `TTLangCompilerSetup.cmake` is forwarded to the LLVM build via `LLVM_CCACHE_BUILD`." This eliminates ~10 lines of duplicated code and prose.

### [index.md] ~lines 12, 17 + [cmake_architecture.md] ~lines 72, 186
**Issue:** The concept that LLVM and tt-metal are "configure-time builds" (run inside `cmake -B`, not `cmake --build`) is stated four times: (1) index.md phase table row 2: "This happens *inside* CMake configure, not during `cmake --build`", (2) index.md line 17: "The critical architectural detail is that phases 2 and 3 are **configure-time builds**", (3) cmake_architecture.md line 72: "builds LLVM/MLIR from ... at **CMake configure time** using `execute_process`", and (4) cmake_architecture.md line 186: "builds tt-metal from ... at **configure time**, similar to how `BuildLLVM.cmake` builds LLVM."
**Suggestion:** Keep the authoritative statement in index.md line 17 and the first mention in `BuildLLVM.cmake` (line 72). Remove the italicized clause from the phase table row 2 (line 12) since line 17 makes the same point emphatically. In the `BuildTTMetal.cmake` opening (line 186), replace "at **configure time**, similar to how `BuildLLVM.cmake` builds LLVM" with "(also a configure-time build)."

### [index.md] ~lines 56-68 + [cmake_architecture.md] ~lines 92-101
**Issue:** The LLVM rebuild-skip mechanism (checking for `MLIRConfig.cmake`) is explained in two places with code blocks: (1) index.md shows the shell-script detection (`do_configure()` checking for `MLIRConfig.cmake` and setting `TTLANG_USE_TOOLCHAIN=ON`), and (2) cmake_architecture.md shows the CMake-side check under "Rebuild skip logic." Both describe the same net effect: "if LLVM is already built, skip the rebuild." The shell-level code in index.md is an implementation detail of the same skip logic.
**Suggestion:** In index.md, reduce the `do_configure()` section (lines 56-68) to a 2-line summary: "The `do_configure()` function auto-detects a pre-built toolchain by checking for `MLIRConfig.cmake` in `TTLANG_TOOLCHAIN_DIR` (see cmake_architecture.md for the CMake-level rebuild skip logic)." Remove the code block. Saves ~8 lines.

## MINOR Suggestions

### [index.md] ~line 3
**Issue:** Verbose introductory sentence with hedging: "Understanding this pipeline is essential context for designing a proper `pip install` experience, because every phase described here currently happens either at CMake configure time or as a post-configure shell step -- none of it is integrated with Python packaging."
**Suggestion:** Shorten to: "None of this pipeline is integrated with Python packaging today, which is the core obstacle to `pip install` support."

### [environment_assumptions.md] ~lines 1-3
**Issue:** The opening has a second sentence that restates the chapter-level framing: "These implicit assumptions are a key reason `pip install` does not work today -- the build expects a specific host environment that Python packaging tools do not provide."
**Suggestion:** Cut the second sentence. This framing is already established in index.md. Keep only: "This file documents the environment variables, tool requirements, and activation script that TT-Lang's build system depends on."

### [cmake_architecture.md] ~lines 52-55
**Issue:** The "Simulator-only early exit" subsection restates what is already shown in two other places: the include-order diagram (line 39: `[TTLANG_SIM_ONLY early return]`) and the option declaration (line 23: "Set up Python environment for simulator only (skip compiler build)").
**Suggestion:** Cut this 4-line subsection. The information is fully conveyed by the option description and the diagram annotation.

### [cmake_architecture.md] ~line 258
**Issue:** In the `TTLangPython.cmake` venv search case 4, the parenthetical "(actual creation is deferred to `BuildLLVM.cmake`)" restates the responsibility split already explained in the opening paragraph of the section (line 249).
**Suggestion:** Remove the parenthetical from case 4 to avoid restating the same point within the same section.

### [environment_assumptions.md] ~line 18
**Issue:** The `Python3_ROOT_DIR` table entry explains the GitHub Actions workaround, duplicating the explanation in cmake_architecture.md lines 280-282.
**Suggestion:** Shorten to: "Unset during venv activation to prevent system Python override (see `TTLangPython.cmake`)."

### [environment_assumptions.md] ~lines 107-109
**Issue:** Points 1 and 2 under "Why `source env/activate` Is Currently Required" re-explain that `PYTHONPATH` and `LD_LIBRARY_PATH` are needed, which the `env/activate.in` template section immediately above (lines 48-79) already demonstrates with the actual script content.
**Suggestion:** Merge points 1 and 2 into a single sentence: "Built packages are not installed into `site-packages` and shared objects lack `RPATH`, requiring manual `PYTHONPATH` and `LD_LIBRARY_PATH` settings (as shown in the template above)."

## Load-Bearing Evidence
- `index.md` lines ~9-15: The five-phase table is the authoritative pipeline reference that all other sections depend on -- cannot be cut.
- `cmake_architecture.md` lines ~33-49: The include-order diagram is the single canonical source for CMake module dependency ordering -- not duplicated elsewhere.
- `environment_assumptions.md` lines ~48-79: The `env/activate.in` template listing is the only place showing the actual generated activation script content -- load-bearing for understanding runtime environment requirements.

## VERDICT
- Crucial updates: yes

---

## Change Log

### 2026-04-09 -- Applied Agent B review feedback (b_review.md), all 4 items

1. **cmake_architecture.md line 3**: Changed "six cmake modules" to "seven cmake modules" to include `TTLangUtils.cmake`. The diagram order (TTLangCompilerSetup before TTLangPython) was already correct and required no change.
2. **cmake_architecture.md line 168**: Changed LLK header count from "13" to "14" and expanded the prose to mention fabric 1D/2D routing, fabric API, and register API headers, matching the 14 entries in `BuildTTMLIRMinimal.cmake` lines 108-123.
3. **cmake_architecture.md lines 247-258 (TTLangPython.cmake section)**: Clarified that `TTLangPython.cmake` performs discovery and path resolution only. Venv creation is handled by `BuildLLVM.cmake` (lines 139-167) and the `TTLANG_SIM_ONLY` block in `CMakeLists.txt`.
4. **cmake_architecture.md line 175**: Changed `MLIRTTMetalDialect -- TTMetal IR` to `MLIRTTMetalDialect -- TTMetal IR and transforms`, reflecting that `BuildTTMLIRMinimal.cmake` processes both `TTMetal/IR` (line 75) and `TTMetal/Transforms` (line 79).

### 2026-04-09 -- Applied compression pass 1 CRUCIAL suggestions (3 items)

1. **[cmake_architecture.md] ccache duplication (CRUCIAL 1):** Removed the "ccache forwarding" subsection (~10 lines including code block) from the `BuildLLVM.cmake` section. Replaced with single sentence: "ccache detection from `TTLangCompilerSetup.cmake` is forwarded to the LLVM build via `LLVM_CCACHE_BUILD`." The authoritative ccache code block remains in the `TTLangCompilerSetup.cmake` section.
2. **[index.md + cmake_architecture.md] "configure-time build" repetition (CRUCIAL 2):** Removed italicized clause ("This happens *inside* CMake configure, not during `cmake --build`.") from the phase table row 2 in index.md. Replaced `BuildTTMetal.cmake` opening sentence in cmake_architecture.md with "(also a configure-time build)." Authoritative statement retained at index.md line 17 and first mention in `BuildLLVM.cmake` section.
3. **[index.md] LLVM rebuild-skip duplication (CRUCIAL 3):** Reduced `do_configure()` section in index.md from ~13 lines (including code block) to a 2-line summary referencing cmake_architecture.md for the CMake-level rebuild skip logic. Code block removed.

# Compression Analysis: Current Build and Installation Flow -- Pass 2

## Scope

Re-check of the three CRUCIAL items from pass 1 to verify they were applied correctly and no residual duplication remains.

## Re-check Results

### CRUCIAL 1: ccache detection duplication in cmake_architecture.md

**Status: RESOLVED**

The `BuildLLVM.cmake` section (line 103) now contains only: "ccache detection from `TTLangCompilerSetup.cmake` is forwarded to the LLVM build via `LLVM_CCACHE_BUILD`." The duplicated `find_program(CCACHE_PROGRAM ccache)` code block has been removed. The sole authoritative code block remains in the `TTLangCompilerSetup.cmake` section (lines 284-289).

### CRUCIAL 2: "Configure-time build" repetition across index.md and cmake_architecture.md

**Status: RESOLVED**

Four occurrences reduced to three non-redundant mentions:
1. `index.md` line 11 -- phase table mentions "configure-time builds" as part of the phase description (necessary context within the table row).
2. `index.md` line 17 -- authoritative bolded statement retained.
3. `cmake_architecture.md` line 72 -- first technical mention in `BuildLLVM.cmake` section retained.
4. `cmake_architecture.md` line 176 -- shortened to parenthetical "(also a configure-time build)" rather than a full re-explanation.

The removed italicized clause from the phase table ("This happens *inside* CMake configure, not during `cmake --build`.") was the right cut -- line 17 makes this point more emphatically one paragraph later.

### CRUCIAL 3: LLVM rebuild-skip explained twice with code blocks

**Status: RESOLVED**

`index.md` lines 56-58 now contain a 2-line summary: "The `do_configure()` function auto-detects a pre-built toolchain by checking for `MLIRConfig.cmake` in `TTLANG_TOOLCHAIN_DIR` (see `cmake_architecture.md` for the CMake-level rebuild skip logic)." The code block has been removed. The single authoritative code block and explanation remains at `cmake_architecture.md` lines 94-101.

## Load-Bearing Evidence

- **cmake_architecture.md line 103** -- The replacement sentence "ccache detection from `TTLangCompilerSetup.cmake` is forwarded to the LLVM build via `LLVM_CCACHE_BUILD`" cannot be cut because it documents the cross-module forwarding relationship; without it, readers would not know how the LLVM sub-build gets ccache support.
- **cmake_architecture.md line 176** -- The parenthetical "(also a configure-time build)" cannot be cut because it is the only indication in the `BuildTTMetal.cmake` section that tt-metal shares the same configure-time execution model as LLVM; removing it would require readers to infer this from code blocks alone.
- **index.md lines 56-58** -- The compressed `do_configure()` summary with its cross-reference to `cmake_architecture.md` cannot be cut because it is the only place documenting the shell-script-level toolchain auto-detection behavior, which is distinct from the CMake-level skip logic.

## MINOR Suggestions

### [cmake_architecture.md] ~lines 94-101: Rebuild skip code block trailing sentence
The sentence after the code block (line 103) -- "ccache detection from `TTLangCompilerSetup.cmake` is forwarded to the LLVM build via `LLVM_CCACHE_BUILD`" -- is placed under the "Rebuild skip logic" subsection heading but is not about rebuild skipping. It should be moved up to the "Key LLVM build flags" bullet list (after line 86) as a fifth bullet: `- \`LLVM_CCACHE_BUILD\` -- Forwarded from parent ccache detection in \`TTLangCompilerSetup.cmake\``. This would also allow cutting the standalone sentence, saving 2 lines and improving section coherence.

### [environment_assumptions.md] ~lines 90-91: TT_METAL_RUNTIME_ROOT row
The table row for `TT_METAL_RUNTIME_ROOT` says "Alias used by tt-metal's runtime layer" and the template itself (line 74) shows it is always set to `${TT_METAL_HOME}`. This row adds no new information beyond "it's an alias." Consider collapsing into the `TT_METAL_HOME` row with a note: "(also exported as `TT_METAL_RUNTIME_ROOT`)".

### [index.md] ~lines 42-54: Multi-stage Docker example
The 4-stage bash example and the preceding sentence ("The script documents a multi-stage workflow for Docker image construction:") could be tightened. The comments inside the code block duplicate the flag descriptions from the table immediately above. Removing the inline comments from the code block would save 4 lines while the table remains the authoritative reference.

## VERDICT
- Crucial updates: no
