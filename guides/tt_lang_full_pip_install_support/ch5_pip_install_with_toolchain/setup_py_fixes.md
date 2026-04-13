# `setup.py` Fixes

The current `python/setup.py` contains a `CMakeBuild` class that orchestrates the CMake build during `pip install`. It has several bugs and missing features that prevent it from working with a pre-built toolchain. This file details the required fixes.

## 1. Fix `cwd` / Repo Root Resolution

### Problem

`python/setup.py`, line 49:

```python
cwd = pathlib.Path().absolute()
```

When pip runs `setup.py`, the working directory is pip's temporary build directory (e.g., `/tmp/pip-req-build-XXXXXXXX/`), not the repository root. The code then computes `build_dir = cwd.parent / "build"` (line 50), which resolves to a path like `/tmp/pip-req-build-XXXXXXXX/../build` -- completely wrong.

### Fix

Resolve the repo root relative to `setup.py`'s own location, which is stable regardless of pip's working directory:

```python
def build_(self, ext):
    build_lib = self.build_lib
    if not os.path.exists(build_lib):
        return

    extension_path = pathlib.Path(self.get_ext_fullpath(ext.name))
    print(f"Running cmake to install ttlang at {extension_path}")

    # setup.py lives at <repo>/python/setup.py -- repo root is one level up
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    build_dir = pathlib.Path(self.build_temp).resolve() / "cmake-build"

    install_dir = pathlib.Path(self.build_lib).resolve()
```

Key changes:
- `repo_root` is derived from `__file__`, not `os.getcwd()`.
- `build_dir` uses `self.build_temp` (setuptools' designated temp directory), not a hardcoded sibling of `cwd`.
- `install_dir` is resolved to an absolute path.

## 2. Pass Toolchain CMake Variables

### Problem

`setup.py` lines 59-68 construct `cmake_args` but never pass `TTLANG_USE_TOOLCHAIN` or `TTLANG_TOOLCHAIN_DIR`. The full project build is always triggered.

### Fix

Read `TTLANG_TOOLCHAIN_DIR` from the environment and pass both variables to CMake:

```python
# Toolchain configuration
toolchain_dir = os.environ.get("TTLANG_TOOLCHAIN_DIR", "")
use_toolchain = "ON" if toolchain_dir else "OFF"

# Support CMAKE_ARGS passthrough (scikit-build convention)
extra_cmake_args = os.environ.get("CMAKE_ARGS", "").split()

cmake_args = [
    "-G", "Ninja",
    "-S", str(repo_root),
    "-B", str(build_dir),
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_INSTALL_PREFIX=" + str(install_dir),
    "-DCMAKE_C_COMPILER=clang",
    "-DCMAKE_CXX_COMPILER=clang++",
    "-DTTLANG_USE_TOOLCHAIN=" + use_toolchain,
]

if toolchain_dir:
    cmake_args.append("-DTTLANG_TOOLCHAIN_DIR=" + toolchain_dir)

cmake_args.extend(extra_cmake_args)
```

When `TTLANG_USE_TOOLCHAIN=ON`, the existing guards in `BuildLLVM.cmake` (line 90-98), `BuildTTMetal.cmake` (line 44-56), and `BuildTTMLIRMinimal.cmake` will skip the heavy submodule builds. See [`cmake_changes.md`](./cmake_changes.md) for the additional guards needed in `BuildTTMLIRMinimal.cmake`.

## 3. Target Only `TTLangPythonModules`

### Current (correct)

`setup.py` line 94 already does this correctly:

```python
self.spawn(["cmake", "--build", str(build_dir), "--", "TTLangPythonModules"])
```

This targets only the Python modules, not the full project. No change needed here, but it is important to verify that `TTLangPythonModules` and all its transitive dependencies can be satisfied when `TTLANG_USE_TOOLCHAIN=ON`. See [`cmake_changes.md`](./cmake_changes.md) for the CMake-side changes.

## 4. Fix the `TTLangPythonWheel` Install Component

### Current

`setup.py` lines 97-99:

```python
self.spawn(
    ["cmake", "--install", str(build_dir), "--component", "TTLangPythonWheel"]
)
```

This installs into `CMAKE_INSTALL_PREFIX`, which is set to `self.build_lib`. The `TTLangPythonWheel` component (defined at `python/CMakeLists.txt` line 295-298) installs the contents of `${CMAKE_BINARY_DIR}/python_packages/` into the install prefix root:

```cmake
install(DIRECTORY ${CMAKE_BINARY_DIR}/python_packages/
  DESTINATION .
  COMPONENT TTLangPythonWheel
  EXCLUDE_FROM_ALL)
```

This produces a layout like:

```
build_lib/
  ttl/
    __init__.py
    _mlir_libs/
      _ttlang.cpython-311-x86_64-linux-gnu.so
      _ttmlir.cpython-311-x86_64-linux-gnu.so
      libTTLangPythonCAPI.so
      ...
    dialects/
    ...
  pykernel/
    __init__.py
    ...
```

This is correct for wheel packaging -- setuptools expects top-level packages directly under `build_lib`. The install component logic is sound; the only issue is that `install_dir` was wrong due to the `cwd` bug (fixed in section 1).

### Additional Fix: Ensure `pykernel` is included

The `PykernelPythonModules` target (line 252-257 of `python/CMakeLists.txt`) is built separately from `TTLangPythonModules`. The `--build` step in `setup.py` must also build `PykernelPythonModules`:

```python
self.spawn([
    "cmake", "--build", str(build_dir), "--",
    "TTLangPythonModules", "PykernelPythonModules",
])
```

## 5. Remove CI-Specific Path Hacks

### Problem

`setup.py` lines 56-58 and 70-92 contain CI-specific logic that sources `env/activate` and adjusts `install_dir` when `IN_CIBW_ENV=ON`. This is fragile and unnecessary when the toolchain is pre-built.

### Fix

Remove the `in_ci()` method and the associated branching. The toolchain-based build does not need a pre-existing `env/activate` because all dependencies come from the toolchain directory and pip's own build isolation. The CI pipeline should set `TTLANG_TOOLCHAIN_DIR` instead of `IN_CIBW_ENV`.

```python
# Remove these:
# def in_ci(self) -> bool:
#     return os.environ.get("IN_CIBW_ENV") == "ON"
#
# if self.in_ci():
#     install_dir = cwd / "build" / install_dir.name
#
# if not self.in_ci():
#     cmake_args.extend(["-S", str(cwd.parent)])
#
# if self.in_ci():
#     subprocess.run(..., shell=True, ...)
```

The `-S` flag (source directory) should always be passed, pointing to `repo_root`.

## 6. Fix Version Handling

### Problem

`setup.py` lines 102-103:

```python
date = datetime.now().strftime("%y.%m.%d")
version = "0.1." + date + ".dev0"
```

This generates a date-based version (`0.1.26.04.09.dev0`) that is completely disconnected from the git-tag-based version used by CMake (`GetVersionFromGit.cmake`). The `version` field is declared `dynamic` in `pyproject.toml`, so `setup.py` must provide it.

### Fix

Read the version from the CMake-generated `config.py` if it exists, or fall back to a PEP 440-compliant dev version:

```python
def _get_version():
    """Read version from CMake-generated config, or compute a fallback."""
    # After CMake configure, config.py exists in the build tree
    config_py = pathlib.Path(__file__).resolve().parent / "ttl" / "config.py"
    if config_py.exists():
        ns = {}
        exec(config_py.read_text(), ns)
        return ns.get("VERSION", "0.0.0.dev0")

    # Fallback: try git describe
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--dirty"],
            capture_output=True, text=True, check=True,
            cwd=pathlib.Path(__file__).resolve().parent.parent,
        )
        tag = result.stdout.strip()
        # Convert "v0.2.0-5-gabcdef" to "0.2.0.dev5"
        tag = tag.lstrip("v")
        parts = tag.split("-")
        if len(parts) >= 3:
            return f"{parts[0]}.dev{parts[1]}"
        return parts[0]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "0.0.0.dev0"
```

## 7. `MANIFEST.in`

No `MANIFEST.in` currently exists in the repository. When building an sdist (`python -m build --sdist`), setuptools includes only files tracked by version control plus those matching `MANIFEST.in` patterns. The CMake build during `pip install` needs C++ sources, `.td` files, CMake modules, and third-party Python sources.

Create `MANIFEST.in` at the repo root:

```
# CMake build system
include CMakeLists.txt
recursive-include cmake *.cmake
include python/CMakeLists.txt

# C++ sources for nanobind extensions
recursive-include python/ttlang *.cpp *.h
recursive-include python/ttmlir *.cpp *.h
recursive-include lib/ttmlir-minimal *.cpp *.h

# MLIR dialect definitions (TableGen)
recursive-include include *.td *.h *.h.in *.def

# Third-party tt-mlir Python sources needed during build
recursive-include third-party/tt-mlir/python *.py *.td *.cpp

# Python sources
recursive-include python/ttl *.py *.py.in
recursive-include python/pykernel *.py
recursive-include python/sim *.py
recursive-include python/utils *.py
include python/gen_elementwise.py

# Package metadata
include README.md
include LICENSE*
include requirements.txt
```

> **Note:** The sdist will be large (~tens of MB) because it includes C++ sources and third-party submodule files. This is expected -- the sdist is for building from source, not for end-user distribution. Pre-built wheels are the intended distribution mechanism.

## 8. Reconstructing the Full `setup.py`

The complete proposed `setup.py` is assembled by applying fixes 1-7 in order to the existing `python/setup.py`. Each section above contains the exact code for its fix. The key structural changes are:

- **Imports:** `os`, `pathlib`, `subprocess`, `setuptools.Extension`, `setuptools.setup`, `setuptools.command.build_ext.build_ext`.
- **`_get_version()`:** New function (section 6) replaces the date-based version.
- **`CMakeBuild.build_()`:** Uses `__file__`-relative paths (section 1), passes toolchain variables (section 2), builds both `TTLangPythonModules` and `PykernelPythonModules` (sections 3-4), removes CI hacks (section 5).
- **`setup()` call:** No `install_requires` (dependencies are static in `pyproject.toml`). Declares `packages`, `package_dir`, `ext_modules`, `cmdclass`, and `long_description`.

**Prev:** [`pyproject_toml_changes.md`](./pyproject_toml_changes.md) | **Next:** [`cmake_changes.md`](./cmake_changes.md)
