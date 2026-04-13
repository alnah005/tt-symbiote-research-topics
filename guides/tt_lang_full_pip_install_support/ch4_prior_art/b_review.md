## Pass 1

**Verdict: No blocking issues found. Chapter approved with one minor factual note below.**

I verified the key technical claims against the actual project repositories and PyPI:

- **IREE build backend:** Chapter correctly states `setuptools` + CMake. Confirmed by inspecting `iree-org/iree/compiler/setup.py` (imports `setuptools`, defines custom `CMakeBuildPy` and `CMakeExtension` classes). Note: the plan (`plan.md`) incorrectly says IREE uses `scikit-build-core` -- the chapter is right, the plan is wrong.
- **CIRCT build backend:** Chapter correctly states `setuptools` + CMake via `setup.py`. Confirmed by inspecting `llvm/circt/lib/Bindings/Python/setup.py`. The plan incorrectly says `scikit-build-core` with `flit_core` -- again, the chapter is right.
- **Triton LLVM strategy:** Chapter correctly states auto-download of pre-built LLVM from Azure Blob Storage (`oaitriton.blob.core.windows.net`). The plan incorrectly says Triton "builds LLVM from source during `pip install`" -- the chapter is right.
- **torch-mlir manylinux_2_28:** Confirmed via the current `build_linux_packages.sh` which defaults to `quay.io/pypa/manylinux_2_28`.
- **torch-mlir dual package (`torch-mlir` vs `torch-mlir-ext`):** Confirmed in `llvm/torch-mlir/setup.py` -- the `TORCH_MLIR_ENABLE_ONLY_MLIR_PYTHON_BINDINGS` flag switches the package name.
- **IREE `manylinux_2_27`:** Confirmed on PyPI -- wheels carry dual tags `manylinux_2_27.manylinux_2_28`.
- **IREE wheel sizes (~83 MB compiler, ~8 MB runtime):** Confirmed on PyPI for recent versions.

**Factual note (non-blocking):** The torch-mlir wheel sizes in `case_studies.md` cite version `20221213.686` (December 2022). Current torch-mlir wheels are significantly smaller (e.g., ~70 MB for aarch64). The numbers are correct for the stated version, but a downstream reader may assume these reflect current sizes. Consider updating to a recent release, or adding a date qualifier.

**Navigation footers:** `index.md` has a "Next" link to `case_studies.md`. `case_studies.md` has a "Next" link to `lessons_learned.md`. `lessons_learned.md` has a "Next" link to Chapter 5. All navigation links are present and use correct relative paths.

**Clickable links in `index.md`:** Both content links (`case_studies.md`, `lessons_learned.md`) and the cross-chapter link to Chapter 5 use correct relative markdown paths.

No feedback items to return. Chapter approved.

## Pass 2

**3 issues found.**

### Issue 1 — Incorrect cross-reference anchor in `index.md` (line 18)

Pattern 1 ("Pre-built LLVM in CI") links to `lessons_learned.md#1-scikit-build-core-vs-setuptools--custom-cmakebuild`, which is the section about build backend choice (setuptools vs. scikit-build-core). That section does not discuss pre-building LLVM. The link should point to section 2 (`#2-the-toolchain-wheel-pattern`), which is the section that actually covers how each project separates slow LLVM builds from fast project builds.

**File:** `index.md`, line 18
**Fix:** Change the link target from `./lessons_learned.md#1-scikit-build-core-vs-setuptools--custom-cmakebuild` to `./lessons_learned.md#2-the-toolchain-wheel-pattern`.

### Issue 2 — "Three patterns" but four bullets in `case_studies.md` (lines 211-216)

The "Implications for TT-Lang" section opens with "three patterns emerge" but then lists four bullet points. Either change the count to "four" or fold the CIRCT bullet (which describes current state rather than a forward pattern) into a separate sentence outside the bulleted list.

**File:** `case_studies.md`, line 211
**Fix:** Change "three patterns emerge" to "four patterns emerge", or restructure to match the stated count.

### Issue 3 — Factual contradiction: "all four projects" use wheel repair tools (`index.md` line 19)

Pattern 2 in `index.md` states "all four projects use platform-specific wheel repair tools (`auditwheel`, `delocate`, `delvewheel`)." However, the CIRCT case study in `case_studies.md` (line 203) explicitly says the opposite: "there is no need for `auditwheel`, `cibuildwheel`, or manylinux compliance." CIRCT does not publish wheels to PyPI and does not run any wheel repair tooling. The claim should be scoped to the three projects that publish to PyPI (torch-mlir, Triton, IREE).

**File:** `index.md`, line 19
**Fix:** Change "all four projects" to "all three PyPI-publishing projects" or similar qualifier.

## Pass 3

All three Pass 2 issues have been resolved:

1. **Issue 1 (incorrect anchor):** `index.md` line 18 now links to `#2-the-toolchain-wheel-pattern`. Confirmed correct.
2. **Issue 2 (count mismatch):** `case_studies.md` line 211 now says "four patterns emerge" matching the four bullets. Confirmed correct.
3. **Issue 3 (CIRCT overclaim):** `index.md` line 19 now says "the three PyPI-publishing projects". Confirmed correct.

Navigation footers verified: `index.md` links to `case_studies.md`, `case_studies.md` links to `lessons_learned.md`, `lessons_learned.md` links to Chapter 5. The Chapter 5 target does not exist yet (only chapters 1-4 are present), which is expected for an in-progress guide. All internal cross-references (`index.md` anchors into `lessons_learned.md`, back-references to chapters 1 and 2) resolve to correct headings.

No feedback — chapter approved.
