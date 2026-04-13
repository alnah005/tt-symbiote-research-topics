# `pyproject.toml` Changes

The current `pyproject.toml` (at repo root) has three problems that block a working `pip install`:

1. **`dependencies` is declared dynamic** (line 35) but `setup.py` only supplies `pydantic<3`, missing most runtime dependencies.
2. **No mechanism to pass CMake variables** (`TTLANG_USE_TOOLCHAIN`, `TTLANG_TOOLCHAIN_DIR`) through the build backend.
3. **Build-system requires are incomplete** for the toolchain-based pip build (missing `nanobind` version pin, missing `scikit-build-core` if we switch backends).

## 1. Static `[project.dependencies]`

Replace the dynamic dependencies declaration with a static list derived from `requirements.txt`. This makes the package installable without `setup.py` having to re-parse requirements files.

### Current (`pyproject.toml`, lines 34-35)

```toml
# We will dynamically provide `version` and `dependencies` in setup.py
dynamic = ["version", "dependencies", "readme"]
```

### Proposed

Remove `dependencies` from the `dynamic` list and declare the full dependency array as a static key under `[project]`. The complete list (12 entries, derived from `requirements.txt`) is shown in the [Summary Diff](#summary-of-pyprojecttoml-diff) at the bottom of this file.

The `version` field remains dynamic because it is derived from git tags at CMake configure time (via `GetVersionFromGit.cmake`, line 17-18 of `CMakeLists.txt`). The `readme` field remains dynamic because `setup.py` reads `README.md` from the project root (line 109-111 of `python/setup.py`).

> **Note:** `pybind11` and `pytest` from `requirements.txt` are excluded -- `pybind11` is a build dependency (already in `[build-system].requires`), and `pytest` belongs in `[project.optional-dependencies].test`.

## 2. Optional Dependency Extras

Add extras for simulator-only usage, development, and testing:

```toml
[project.optional-dependencies]
sim = [
    "torch>=1.9.0",
    "numpy>=1.20.0",
    "greenlet>=3.0.0",
    "pydantic<3",
]
dev = [
    "black",
    "pre-commit",
    "pyright",
    "lit",
]
test = [
    "pytest>=7.0",
    "pytest-order>=1.0.0",
    "pytest-xdist>=3.0",
]
```

This aligns with the existing `dev-requirements.txt` (which includes `-r requirements.txt`, `black`, `lit`, `pre-commit`, `pyright`, `pytest-order`, `pytest-xdist`) and allows users to do `pip install ttl[dev,test]`.

## 3. Build Backend Options

### Option A: Stay with `setuptools.build_meta` (recommended for now)

The current `[build-system]` section is:

```toml
[build-system]
requires = ["setuptools>=61.0", "cmake", "nanobind", "wheel", "pip", "ninja"]
build-backend = "setuptools.build_meta"
```

This works. The `CMakeBuild` class in `setup.py` already invokes CMake manually. The changes needed are in `setup.py` itself (see [`setup_py_fixes.md`](./setup_py_fixes.md)), not in the build backend.

**Additions to `[build-system].requires`:**

```toml
requires = [
    "setuptools>=61.0",
    "cmake>=3.28",
    "nanobind>=2.9,<3.0",
    "wheel",
    "ninja",
]
```

Changes:
- Pin `cmake>=3.28` to match `cmake_minimum_required(VERSION 3.28.0)` in `CMakeLists.txt` line 1.
- Pin `nanobind>=2.9,<3.0` to match `requirements.txt` line 9.
- Remove `pip` -- it is not a valid build requirement (pip is the installer, not a build dependency).

### Option B: Switch to `scikit-build-core`

`scikit-build-core` provides native CMake integration with `pyproject.toml` and eliminates the need for a custom `CMakeBuild` class in `setup.py`:

```toml
[build-system]
requires = [
    "scikit-build-core>=0.10",
    "nanobind>=2.9,<3.0",
]
build-backend = "scikit_build_core.build"

[tool.scikit-build]
cmake.build-type = "Release"
cmake.targets = ["TTLangPythonModules"]
install.components = ["TTLangPythonWheel"]
cmake.define = {TTLANG_USE_TOOLCHAIN = "ON"}

[tool.scikit-build.cmake.define]
TTLANG_TOOLCHAIN_DIR = {env = "TTLANG_TOOLCHAIN_DIR", default = "/opt/ttlang-toolchain"}
```

**Pros:** Eliminates all of `setup.py`'s `CMakeBuild` class. CMake variable passthrough is declarative. Editable installs and wheel builds work out of the box.

**Cons:** Adds a new build dependency. Requires that `python/CMakeLists.txt` be invocable as a top-level project (or that we point scikit-build at the repo root and rely on `TTLANG_USE_TOOLCHAIN` to skip heavy builds). The MLIR Python binding macros (`AddMLIRPython`, `declare_mlir_python_extension`) expect the full MLIR CMake infrastructure to be available, which requires `find_package(MLIR)` to succeed -- this is the same requirement either way.

**Recommendation:** Start with Option A (keep `setuptools.build_meta`, fix `setup.py`). The custom `CMakeBuild` class is only ~70 lines and already handles the CI/local distinction. Once the toolchain path works end-to-end, switching to `scikit-build-core` is a follow-up that deletes code rather than adding it.

## 4. CMake Variable Passthrough

Regardless of backend choice, the user must be able to set `TTLANG_TOOLCHAIN_DIR` via environment variable. With the setuptools backend, this is handled in `setup.py` (see [`setup_py_fixes.md`](./setup_py_fixes.md)). The `pyproject.toml` needs no additional configuration for this -- CMake variables are passed programmatically.

For users who prefer `CMAKE_ARGS` (a convention from scikit-build), we can also support:

```bash
CMAKE_ARGS="-DTTLANG_TOOLCHAIN_DIR=/opt/ttlang-toolchain" pip install .
```

This is implemented in the `setup.py` changes by parsing `os.environ.get("CMAKE_ARGS", "")`.

## Summary of `pyproject.toml` Diff

```diff
 [build-system]
-requires = ["setuptools>=61.0", "cmake", "nanobind", "wheel", "pip", "ninja"]
+requires = ["setuptools>=61.0", "cmake>=3.28", "nanobind>=2.9,<3.0", "wheel", "ninja"]
 build-backend = "setuptools.build_meta"

 [project]
 name = "ttl"
 ...

-# We will dynamically provide `version` and `dependencies` in setup.py
-dynamic = ["version", "dependencies", "readme"]
+dynamic = ["version", "readme"]
+
+dependencies = [
+    "pydantic<3",
+    "torch>=1.9.0",
+    "numpy>=1.20.0",
+    "greenlet>=3.0.0",
+    "pandas<3",
+    "nanobind>=2.9,<3.0",
+    "PyYAML>=5.4.0,<=6.0.1",
+    "typing_extensions>=4.12.2",
+    "ml_dtypes>=0.1.0,<=0.6.0; python_version<'3.13'",
+    "ml_dtypes>=0.5.0,<=0.6.0; python_version>='3.13'",
+    "loguru>=0.6.0",
+    "graphviz",
+    "seaborn>=0.13.2",
+]
+
+[project.optional-dependencies]
+sim = ["torch>=1.9.0", "numpy>=1.20.0", "greenlet>=3.0.0", "pydantic<3"]
+dev = ["black", "pre-commit", "pyright", "lit"]
+test = ["pytest>=7.0", "pytest-order>=1.0.0", "pytest-xdist>=3.0"]
```

**Prev:** [`index.md`](./index.md) | **Next:** [`setup_py_fixes.md`](./setup_py_fixes.md)
