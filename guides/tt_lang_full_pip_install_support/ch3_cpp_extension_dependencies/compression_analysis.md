# Compression Analysis: C++ Extension Build Dependencies -- Pass 1

## Summary
- Total files analyzed: 3
- Estimated current line count: ~641 lines
- Estimated post-compression line count: ~530 lines
- Estimated reduction: ~17%

## CRUCIAL Suggestions

### C1: Duplicate "Compile Definitions and Nanobind Domain" section

**Files:** `index.md` lines 24-47 and `mlir_dependency_chain.md` lines 199-218

The nanobind domain (`MLIR_BINDINGS_PYTHON_NB_DOMAIN=ttl`) and the package prefix (`MLIR_PYTHON_PACKAGE_PREFIX=ttl.`) are explained in full in both files. `index.md` covers them in "The Shared Nanobind Domain" and "The Package Prefix" subsections; `mlir_dependency_chain.md` repeats the same information in its "Compile Definitions and Nanobind Domain" section, including restating the same CMake variable names, line references, and behavioral consequences.

**Recommendation:** Remove the "Compile Definitions and Nanobind Domain" section (lines 199-218) from `mlir_dependency_chain.md` entirely and add a one-line cross-reference to `index.md`. This eliminates ~20 lines of pure duplication.

### C2: Duplicate CAPI library descriptions across files

**Files:** `index.md` lines 51-110 and `mlir_dependency_chain.md` lines 103-136

`index.md` documents `TTLangPythonCAPI`, `TTMLIRMinimalCAPI`, and `TTLangCAPI` with a dependency tree diagram. `mlir_dependency_chain.md` re-describes both `TTMLIRMinimalCAPI` and `TTLangCAPI` under "tt-mlir CAPI Libraries" with overlapping detail (source files, link dependencies, function names). The link dependency information is largely the same data presented in two formats (prose + ASCII tree in index.md, prose + inline lists in mlir_dependency_chain.md).

**Recommendation:** Consolidate CAPI library details in one location. Keep the detailed source-file-level breakdown in `mlir_dependency_chain.md` (since that file is the "full breakdown") and simplify `index.md` to a high-level summary with a cross-reference. Alternatively, keep `index.md` as the authoritative CAPI section and have `mlir_dependency_chain.md` only add the source-file details not already covered. Estimated savings: ~15 lines.

### C3: Duplicate Python ODS bindings table

**Files:** `mlir_dependency_chain.md` lines 30-34 and lines 184-195

The three-dialect table under `declare_mlir_dialect_python_bindings` (lines 30-34) lists the same TD-file-to-output mapping that is restated in prose form in the "Python ODS Bindings TableGen" section (lines 184-195). The second occurrence adds only a sentence of framing ("trigger a separate TableGen invocation") before repeating the same three dialect/file pairs.

**Recommendation:** Merge the "Python ODS Bindings TableGen" subsection into the earlier table by adding a "TableGen stage" column or a note, and remove lines 184-195. Estimated savings: ~12 lines.

## MINOR Suggestions

### M1: Verbose CMake helper descriptions in mlir_dependency_chain.md

Lines 14-55 of `mlir_dependency_chain.md` explain five CMake helper functions (`declare_mlir_python_sources`, `declare_mlir_dialect_python_bindings`, `declare_mlir_python_extension`, `add_mlir_python_common_capi_library`, `add_mlir_python_modules`). The last two (lines 45-55) are single-paragraph descriptions that mostly restate what `index.md` already shows via concrete CMake snippets. These could be reduced to one-line definitions.

### M2: Hedging in discovery_mechanisms.md "Why this matters for pip-install"

Lines 114-122 of `discovery_mechanisms.md` present a two-option analysis of pip-install implications for tt-mlir source consumption. While relevant, it is somewhat speculative and could be a single sentence: "Because tt-mlir is consumed as source, an sdist must bundle the submodule or a separate pre-built package must be provided."

### M3: Verbose post-discovery setup explanation

Lines 59-76 of `discovery_mechanisms.md` list four `include()` calls with a paragraph of explanation. The `list(APPEND CMAKE_MODULE_PATH ...)` lines and the four includes could be presented as a compact code block with a one-line description instead of the current block + paragraph format.

### M4: ASCII art dependency graph in mlir_dependency_chain.md

Lines 224-245 restate the build flow that is already implicitly documented by the preceding sections. The ASCII diagram adds visual clarity but duplicates the narrative. Consider whether readers need both.

## Load-Bearing Evidence

- **index.md** line 27: `set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "ttl")` -- duplicated at mlir_dependency_chain.md line 212
- **mlir_dependency_chain.md** line 206: `MLIR_PYTHON_PACKAGE_PREFIX=ttl.` -- duplicated from index.md line 42
- **discovery_mechanisms.md** line 117: "Bundle the tt-mlir submodule in the source distribution, or" -- low-value speculative content that could be one sentence

## VERDICT
- Crucial updates: yes

---

# Compression Analysis -- Change Log

## 2026-04-09 -- Applied Agent B Pass 1 Feedback

### Change 1: Fixed simulator-only mode variable name and attribution

**File:** `discovery_mechanisms.md`, "Simulator-only mode" subsection

- Renamed `TTLANG_SIMULATOR_ONLY` to `TTLANG_SIM_ONLY` (the actual variable
  declared at line 22 of the top-level `CMakeLists.txt`).
- Corrected the description: `TTLANG_SIM_ONLY` triggers an early-return block
  at line 65 of the top-level `CMakeLists.txt`, *before* `BuildTTMetal.cmake`
  is included. The previous text incorrectly implied the skip happened inside
  `BuildTTMetal.cmake`.

**Verified against:** `tt-lang/CMakeLists.txt` lines 22 and 65.

### Change 2: Added `MLIRCAPITransforms` to TTMLIRMinimalCAPI dependency diagram

**File:** `index.md`, CAPI Library Dependency Summary diagram

- Added `MLIRCAPITransforms` to the `TTMLIRMinimalCAPI` sub-tree alongside
  `MLIRIR` and `MLIRSupport`. This library is publicly linked at line 26 of
  `lib/ttmlir-minimal/CAPI/CMakeLists.txt` and provides pass-manager CAPI
  symbols (`mlirPassManagerCreate`, `mlirPassManagerRunOnOp`, etc.).

**Verified against:** `tt-lang/lib/ttmlir-minimal/CAPI/CMakeLists.txt` line 26.

## 2026-04-09 -- Applied CRUCIAL Compression Suggestions (C1, C2, C3)

### C1: Removed duplicate "Compile Definitions and Nanobind Domain" section

**File:** `mlir_dependency_chain.md`

- Replaced the full "Compile Definitions and Nanobind Domain" section (former
  lines 199-218) with a one-line cross-reference to `index.md#the-shared-nanobind-domain`.
- Eliminated ~17 lines of pure duplication. `index.md` remains the authoritative
  source for `MLIR_BINDINGS_PYTHON_NB_DOMAIN` and `MLIR_PYTHON_PACKAGE_PREFIX`.

### C2: Deduplicated CAPI library descriptions

**File:** `mlir_dependency_chain.md`

- Added a cross-reference to `index.md#capi-library-dependency-summary` at the
  top of the "tt-mlir CAPI Libraries" section.
- Removed the "Links publicly against..." prose from both `TTMLIRMinimalCAPI`
  and `TTLangCAPI` subsections, since that link-dependency information is already
  fully captured in `index.md`'s dependency tree diagram.
- Retained the source-file-level details (`.cpp` files, function names) which
  are unique to `mlir_dependency_chain.md`.

### C3: Merged duplicate Python ODS bindings table

**File:** `mlir_dependency_chain.md`

- Added a "TableGen stage" column to the `declare_mlir_dialect_python_bindings`
  table (lines 30-34) and a follow-up sentence noting the enum generation.
- Removed the separate "Python ODS Bindings TableGen" subsection (former lines
  184-195) which restated the same three dialect/file pairs in prose form.
- Net savings: ~10 lines.

# Compression Analysis: C++ Extension Build Dependencies -- Pass 2

## Re-check of Previous CRUCIAL Items

### C1: Nanobind domain + package prefix duplication — RESOLVED

`mlir_dependency_chain.md` line 190 now contains a single cross-reference sentence pointing to `index.md#the-shared-nanobind-domain`. The full explanations of `MLIR_BINDINGS_PYTHON_NB_DOMAIN` and `MLIR_PYTHON_PACKAGE_PREFIX` exist only in `index.md` (lines 24-47). No residual duplication.

### C2: CAPI library descriptions overlap — RESOLVED

`mlir_dependency_chain.md` lines 109-110 cross-reference `index.md#capi-library-dependency-summary` for the dependency tree. The remaining content in `mlir_dependency_chain.md` (lines 113-137) covers only source-file-level detail (`.cpp` filenames, CAPI function names) that does not appear in `index.md`. The two files are now complementary, not overlapping.

### C3: Python ODS bindings table duplication — RESOLVED

The `declare_mlir_dialect_python_bindings` table at `mlir_dependency_chain.md` lines 30-34 now includes a "TableGen stage" column. The former second occurrence (prose restatement of the same three dialect/file pairs) has been removed. No duplicate remains.

## VERDICT

- Crucial updates: no

## Load-Bearing Evidence

- **index.md** lines 24-47: sole location of nanobind domain and package prefix explanations; `mlir_dependency_chain.md` line 190 correctly defers here with a cross-reference.
- **mlir_dependency_chain.md** lines 113-137: CAPI source-file details (`Dialects.cpp`, `TTLAttrs.cpp`, function names) are unique to this file; link-dependency data lives only in `index.md` lines 89-110.
- **discovery_mechanisms.md** lines 211-238: discovery-order summary is self-contained with no content duplicated from the other two chapter files.

## MINOR Suggestions

### M1: Cross-reference anchor text could be more specific

`mlir_dependency_chain.md` line 190 links to `index.md#the-shared-nanobind-domain` but the anchor text reads "The Shared Nanobind Domain and The Package Prefix." The `index.md` heading is actually "The Shared Nanobind Domain" (line 24) and "The Package Prefix" (line 37) — two separate subsections. The link target resolves to only the first heading. Consider linking to the parent section or adding both anchors for precision.

### M2: Summary Dependency Graph in mlir_dependency_chain.md could note it is a build-order view

The ASCII diagram at `mlir_dependency_chain.md` lines 196-217 shows the CMake declaration flow but does not label itself as such. A one-line caption ("Build-time declaration order for the Python package") would distinguish it from the link-time dependency tree in `index.md`.
