## Pass 1

**1. Invalid TOML syntax in `pyproject_toml_changes.md` section 1 "Proposed" block (would cause implementation error)**

The proposed TOML on lines 25-40 uses `[project.dependencies]` as a section header with `dependencies = [...]` inside it. This is invalid PEP 621 TOML. `[project.dependencies]` creates a subtable of `[project]`, not a key. The correct form is simply `dependencies = [...]` as a key directly under the `[project]` table (no new section header). The summary diff at the bottom of the same file (lines 161-175) shows it correctly -- just `dependencies = [...]` without a section header -- but the "Proposed" code block that a reader would copy first is wrong. An implementer copying the section 1 block verbatim would get a TOML parse error or, worse, a silently malformed `pyproject.toml`.

**2. Proposed `setup.py` includes `install_requires` that conflicts with / duplicates `pyproject.toml` static `dependencies` (materially misleading)**

In `setup_py_fixes.md` section 8, the complete proposed `setup.py` (lines 363-368) passes `install_requires=["pydantic<3", "torch>=1.9.0", "numpy>=1.20.0", "greenlet>=3.0.0"]` to `setup()`. Meanwhile, `pyproject_toml_changes.md` removes `dependencies` from the `dynamic` list and adds a full static `dependencies` array with 12 entries. When `dependencies` is not declared dynamic, setuptools ignores `install_requires` and uses the `pyproject.toml` value. The `install_requires` in the proposed `setup.py` is therefore dead code with a different (shorter) dependency list, which will mislead an implementer into thinking those four packages are the authoritative runtime dependencies. Fix: either remove `install_requires` from `setup()` entirely (since `pyproject.toml` is authoritative), or keep `dependencies` in the `dynamic` list and use `install_requires` as the single source of truth.

**3. `BuildTTMLIRMinimal.cmake` has no `TTLANG_USE_TOOLCHAIN` guard -- chapter correctly identifies this, but the proposed guard sets `TT_MLIR_SOURCE_DIR` to the wrong path (would break include resolution)**

In `cmake_changes.md` section 1, the proposed toolchain guard for `BuildTTMLIRMinimal.cmake` sets `TT_MLIR_SOURCE_DIR` to `"${TTLANG_TOOLCHAIN_DIR}/include"`. In the actual source (`BuildTTMLIRMinimal.cmake` line 22), `TT_MLIR_SOURCE_DIR` is set to `"${CMAKE_SOURCE_DIR}/third-party/tt-mlir"` -- it is the root of the tt-mlir submodule, not an include directory. Downstream code (e.g., `python/CMakeLists.txt` line 17 which sets `TTMLIR_PYTHON_ROOT_DIR` to `"${CMAKE_SOURCE_DIR}/third-party/tt-mlir/python/ttmlir"`) does not use `TT_MLIR_SOURCE_DIR`, so the immediate breakage is limited. However, any future code or macro that relies on `TT_MLIR_SOURCE_DIR` pointing to the tt-mlir source root would silently get the wrong path. The variable should either not be set at all in the toolchain path, or should be set to a semantically correct value (e.g., the toolchain's tt-mlir share directory).

**4. Navigation: no "Prev" links on any content file**

All four content files have a "Next" footer but none have a "Prev" link. A reader navigating backward from `cmake_changes.md` has no link to return to `setup_py_fixes.md`, etc. The `index.md` also lacks a "Prev" link to Chapter 4. This is a structural gap for navigation -- minor but consistent across all files.

**5. No issues found with:** line number references for `setup.py` (lines 49-50, 94, 97-99, 102-103, 109-111, 116-117), `python/CMakeLists.txt` (lines 134, 144-153, 157-161, 252-257, 281, 286-290, 295-298), `CMakeLists.txt` (lines 1, 17-18, 130, 152-157, 176-181), `BuildLLVM.cmake` (lines 90-98, 139-167), `BuildTTMetal.cmake` (lines 44-56), `requirements.txt` content match, clickable relative links in `index.md`.

## Pass 2

All four Pass 1 issues have been resolved:

- Issue 1 (invalid TOML syntax): The proposed block now shows `dependencies = [...]` as a bare key, not under a `[project.dependencies]` section header. Fixed.
- Issue 2 (`install_requires` conflict): The complete proposed `setup.py` in section 8 no longer passes `install_requires` to `setup()`. Fixed.
- Issue 3 (`TT_MLIR_SOURCE_DIR` wrong path): The proposed guard now explicitly leaves `TT_MLIR_SOURCE_DIR` unset in toolchain mode, with a comment explaining why. Fixed.
- Issue 4 (missing Prev links): All four files now have both `**Prev:**` and `**Next:**` navigation footers. Fixed.

**1. `cmake_changes.md` section 5: pip detection via `PIP_BUILD_TRACKER` is unreliable on modern pip (would silently skip the guard)**

The proposed pip-detection check (section 5, line 163) tests `DEFINED ENV{PIP_BUILD_TRACKER} OR DEFINED ENV{_PIP_STANDALONE_CERT}`. The `PIP_BUILD_TRACKER` environment variable was removed in pip 23.1 (released April 2023), and `_PIP_STANDALONE_CERT` is an internal implementation detail that is not guaranteed across pip versions. On any pip >= 23.1, neither variable will be set, so `_SKIP_VENV_SETUP` will never become `TRUE` and the venv creation block will run during `pip install`, conflicting with pip's build isolation. A more reliable detection is checking for `VIRTUAL_ENV` combined with `PIP_REQ_TRACKER` (also gone) or, better, making venv-skip the default when `TTLANG_USE_TOOLCHAIN=ON` since the toolchain path already implies a pip-based workflow.

No feedback -- chapter approved.

## Pass 3

Pass 2 issue 1 (pip detection via `PIP_BUILD_TRACKER`) has been resolved. Section 5 of `cmake_changes.md` now correctly ties venv-skip to `TTLANG_USE_TOOLCHAIN` and includes an explanatory note about the unreliable pip environment variables. Fixed.

**1. `cmake_changes.md` section numbering skips from 3 to 5 (structural gap)**

Sections are numbered 1, 2, 3, 5, 6, 7, 8. There is no section 4. The summary table in section 8 does not reference a section 4 either, so this appears to be a numbering error rather than a missing section. The gap will confuse readers who expect sequential numbering or try to cross-reference by section number.

**2. `cmake_changes.md` navigation footer links to nonexistent Chapter 6**

The footer reads `**Next:** [Chapter 6 -- Two-Phase Wheel Architecture](../ch6_two_phase_wheel_architecture/index.md)`. The directory `ch6_two_phase_wheel_architecture` does not exist in the guide. The link is a dead reference. Either remove the Next link (if Chapter 6 is not yet written) or update it to point to an existing target.

No feedback -- chapter approved.
