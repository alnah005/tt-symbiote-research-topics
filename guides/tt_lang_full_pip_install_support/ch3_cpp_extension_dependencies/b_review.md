## Pass 1

### Issue 1 — Wrong CMake variable name for simulator-only mode

**File:** `discovery_mechanisms.md`, "tt-metal Discovery" section, "Simulator-only mode" subsection

**Claim:** "When `TTLANG_SIMULATOR_ONLY=ON`, tt-metal is skipped entirely..."

**Actual:** The variable is `TTLANG_SIM_ONLY`, not `TTLANG_SIMULATOR_ONLY` (declared at line 22 of the top-level `CMakeLists.txt`). A reader who sets `TTLANG_SIMULATOR_ONLY=ON` would get no effect — the tt-metal build would proceed normally.

Additionally, `TTLANG_SIM_ONLY` is handled by an early-return block in the top-level `CMakeLists.txt` (line 65), *before* `BuildTTMetal.cmake` is ever included. It is not a discovery mode inside `BuildTTMetal.cmake` as the section structure implies. The three `BuildTTMetal.cmake` discovery modes are: (1) pre-built toolchain, (2) submodule build on Linux, and (3) macOS skip (the `APPLE` guard at line 22 of `BuildTTMetal.cmake`).

**Severity:** Implementation error — wrong variable name and wrong attribution of the mechanism.

---

### Issue 2 — `MLIRCAPITransforms` missing from TTMLIRMinimalCAPI dependency diagram

**File:** `index.md`, lines 95-103 (CAPI Library Dependency Summary diagram)

**Claim:** The `TTMLIRMinimalCAPI` sub-tree lists seven tt-mlir libraries plus `MLIRIR` and `MLIRSupport`.

**Actual:** `lib/ttmlir-minimal/CAPI/CMakeLists.txt` also links `MLIRCAPITransforms` publicly (line 26). This library provides the pass-manager CAPI symbols (`mlirPassManagerCreate`, `mlirPassManagerRunOnOp`, etc.) that `TTMLIRMinimalCAPI` re-exports. A reader using the diagram to reconstruct the link set would get undefined-symbol errors at link time.

**Severity:** Material omission — would cause a link failure if someone built from the diagram.

---

No other issues found. Navigation footers are present on all three files. The two "Further Reading" links in `index.md` and the inter-file `**Next:**` links use correct relative paths.

## Pass 2

Both Pass 1 issues have been resolved:

- **Issue 1 (variable name):** `discovery_mechanisms.md` now correctly uses `TTLANG_SIM_ONLY` and accurately describes the early-return mechanism in the top-level `CMakeLists.txt` (line 65), distinct from the `BuildTTMetal.cmake` discovery modes.
- **Issue 2 (missing dependency):** The CAPI Library Dependency Summary diagram in `index.md` now includes `MLIRCAPITransforms` under `TTMLIRMinimalCAPI`.

Verified against source:

- Line numbers cited in `discovery_mechanisms.md` (BuildLLVM.cmake lines 126, 174; top-level CMakeLists.txt lines 22, 65, 140-141) are accurate.
- Line numbers cited in `index.md` (python/CMakeLists.txt lines 9, 13, 236, 261, 276) are accurate.
- `EMBED_CAPI_LINK_LIBS` tables match the actual `declare_mlir_python_extension()` calls in `python/CMakeLists.txt`.
- `TTMLIRMinimalCAPI` link libraries match `lib/ttmlir-minimal/CAPI/CMakeLists.txt` exactly.
- `TTLangCAPI` link libraries and `DEPENDS` match `lib/CAPI/CMakeLists.txt` exactly.
- Navigation footers present on all three files. Internal links (`./mlir_dependency_chain.md`, `./discovery_mechanisms.md`) resolve correctly. Forward link to `../ch4_prior_art/index.md` targets a not-yet-written chapter, which is expected per the plan.

**No feedback — chapter approved.**

## Pass 3

Re-verified all key claims against source:

- `python/CMakeLists.txt` line numbers confirmed: `MLIR_PYTHON_PACKAGE_PREFIX` (line 9), `MLIR_BINDINGS_PYTHON_NB_DOMAIN` (line 13), `add_mlir_python_common_capi_library` (line 236), `add_mlir_python_modules` (line 261), `COMMON_CAPI_LINK_LIBS` (line 276), `add_dependencies(TTLangPythonModules TTLangGeneratedElementwise)` (line 281).
- CAPI library dependency diagram in `index.md` includes `MLIRCAPITransforms` under `TTMLIRMinimalCAPI` (fix from Pass 1 still present).
- `discovery_mechanisms.md` correctly uses `TTLANG_SIM_ONLY` and describes the early-return mechanism (fix from Pass 1 still present).
- Navigation footers present on all three files. Internal links (`./mlir_dependency_chain.md`, `./discovery_mechanisms.md`) resolve correctly. Anchor links (`#capi-library-dependency-summary`, `#the-shared-nanobind-domain`) match their target headings. Forward link to `../ch4_prior_art/index.md` targets a not-yet-written chapter, expected per the plan.

**No feedback — chapter approved.**
