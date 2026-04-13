## Pass 1

### Issue 1 — `sim` package is NOT routed through CMake (factual error)

**File:** `cmake_build_class.md`, paragraph after the package table (around "Only `ttl` depends on compiled C++ extensions...")

**Claim:** "The remaining five packages are pure Python, though they are still routed through the CMake build system via `declare_mlir_python_sources` so they end up in the `python_packages/` output directory alongside the compiled artifacts."

**Fact:** `sim` does not appear anywhere in `python/CMakeLists.txt`. It is not declared via `declare_mlir_python_sources` and is not part of any CMake target. It exists only in the `setup.py` `packages` list and is handled solely by setuptools' `build_py` phase. The statement is true for `ttl._src`, `utils`, and `pykernel`/`pykernel._src`, but false for `sim`.

**Impact:** A reader implementing the pip install path would incorrectly assume `sim` will be produced by the CMake build and look for it in `python_packages/`, when in fact it must come from the source tree via setuptools.

**Fix:** Change the sentence to clarify that `sim` is handled exclusively by setuptools `build_py`, while the others are routed through CMake.

---

### Issue 2 — `pykernel` is a separate CMake target, not part of `TTLangPythonModules` (materially misleading)

**File:** `cmake_build_class.md`, Step 3 / install section

**Claim:** "This copies the entire `python_packages/` tree (containing `ttl/` and `pykernel/` as top-level directories) into `CMAKE_INSTALL_PREFIX`"

**Fact:** `PykernelPythonModules` is a separate `add_mlir_python_modules()` target in `python/CMakeLists.txt` (lines 252-257). It is NOT a dependency of `TTLangPythonModules`. The build step (`cmake --build ... -- TTLangPythonModules`) does not build `PykernelPythonModules`. Therefore, after Step 2, `python_packages/pykernel/` may not exist, and the Step 3 install would not contain it.

**Impact:** A reader would believe the three-step CMake process produces a complete `pykernel/` package under `python_packages/`, when in fact only `ttl/` is guaranteed to be there. This matters for anyone debugging why pykernel files are missing from wheels or implementing an alternative build flow.

**Fix:** Either note that `PykernelPythonModules` is a separate target that must also be built, or clarify that the `pykernel` pure-Python files in the final wheel come from setuptools `build_py` (using `package_dir` from `setup.py`), not from the CMake install.

---

### Issue 3 — Navigation footer missing "Previous" link on `index.md`

**File:** `index.md`

The `index.md` file has a "Next" footer pointing to `cmake_build_class.md`, but no "Previous" link back to Chapter 1. Similarly, `cmake_build_class.md` has "Next" but no "Previous" back to `index.md`. `why_pip_install_fails.md` has "Next" to Chapter 3 but no "Previous" to `cmake_build_class.md`.

This is a structural gap only if the plan or other chapters establish a convention of bidirectional navigation footers. If the convention is "Next only," this is fine. Flagging for awareness.

---

No other issues found. The `pyproject.toml` snippet, `setup.py` code, `CMakeBuild` class logic, `GetVersionFromGit.cmake` representation, CI-mode behavior, path-assumption analysis, and the five failure modes in `why_pip_install_fails.md` all cross-check correctly against the actual source files. Links in `index.md` are correct relative paths.

## Pass 2

No feedback — chapter approved.

All three pass 1 issues have been addressed:

- **Issue 1 (sim not routed through CMake):** Fixed. The package table and surrounding text in `cmake_build_class.md` now correctly state that `sim` is handled exclusively by setuptools' `build_py` phase and is not declared in `python/CMakeLists.txt`.
- **Issue 2 (pykernel separate CMake target):** Fixed. The Step 3 / install section now notes that `PykernelPythonModules` is a separate target not built by `TTLangPythonModules`, and that `pykernel` files in the final wheel come from setuptools `build_py` via `package_dir`.
- **Issue 3 (navigation):** Unchanged (still "Next" only), which is acceptable as a consistent convention across the chapter.

Verified against source: `pyproject.toml`, `python/setup.py`, `python/CMakeLists.txt` (install component at line 295-298, `PykernelPythonModules` at line 252, `TTLangPythonCommon.Utils` at line 210), `cmake/modules/GetVersionFromGit.cmake`, and absence of `MANIFEST.in`. All code snippets, path claims, CMake target relationships, and failure-mode analyses are factually correct.

## Pass 3

No feedback — chapter approved.

Verified all code snippets and factual claims against source files: `pyproject.toml` (repo root), `python/setup.py`, `python/CMakeLists.txt` (TTLangPythonWheel at line 297, PykernelPythonModules at line 252, TTLangPythonCommon.Utils at line 210, TTLangPythonModules declared sources at lines 261-278), and `cmake/modules/GetVersionFromGit.cmake`. All pass 1 fixes remain correct. Navigation footers are consistent ("Next" only). Internal cross-links (`cmake_build_class.md#the-package-list`, `why_pip_install_fails.md#3-path-assumptions-cwdparent--build`) target valid sections. Forward link to Chapter 3 (`../ch3_cpp_extension_dependencies/index.md`) is structurally correct pending that chapter's creation.
