# tt-lang Full Install & Hardware Test — From Scratch

## Prerequisites

- Clang 17+ (`/usr/bin/clang-17`, `/usr/bin/clang++-17`)
- CMake >= 3.28, Ninja
- Python 3.11+
- Tenstorrent device (Wormhole B0)

---

## Step 1: Clone and init submodules

```bash
git clone <tt-lang-repo-url>
cd tt-lang
git submodule update --init --recursive --depth 1
```

---

## Step 2: Patch files (8 files)

The upstream repo has bugs that prevent toolchain-based builds and pip install from working. Apply these patches before building.

### 2a. `python/setup.py` — Replace entire file

```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
# tt-lang Python package setup

import os
import pathlib
import subprocess

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

readme = None


class TTLangExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            if "ttl" in ext.name:
                self.build_(ext)
            else:
                raise Exception("Unknown extension")

    def build_(self, ext):
        build_lib = self.build_lib
        if not os.path.exists(build_lib):
            return

        extension_path = pathlib.Path(self.get_ext_fullpath(ext.name))
        print(f"Running cmake to install ttlang at {extension_path}")

        repo_root = pathlib.Path(__file__).resolve().parent.parent
        build_dir = pathlib.Path(self.build_temp).resolve() / "cmake-build"
        install_dir = pathlib.Path(self.build_lib).resolve()

        toolchain_dir = os.environ.get("TTLANG_TOOLCHAIN_DIR", "")
        use_toolchain = "ON" if toolchain_dir else "OFF"
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

        self.spawn(["cmake", *cmake_args])

        self.spawn([
            "cmake", "--build", str(build_dir), "--",
            "TTLangPythonModules", "PykernelPythonModules",
        ])

        self.spawn(
            ["cmake", "--install", str(build_dir), "--component", "TTLangPythonWheel"]
        )


def _get_version():
    config_py = pathlib.Path(__file__).resolve().parent / "ttl" / "config.py"
    if config_py.exists():
        ns = {}
        exec(config_py.read_text(), ns)
        return ns.get("VERSION", "0.0.0.dev0")
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--dirty"],
            capture_output=True, text=True, check=True,
            cwd=pathlib.Path(__file__).resolve().parent.parent,
        )
        tag = result.stdout.strip().lstrip("v")
        parts = tag.split("-")
        if len(parts) >= 3:
            return f"{parts[0]}.dev{parts[1]}"
        return parts[0]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "0.0.0.dev0"


version = _get_version()

ttlang_c = TTLangExtension("ttl")

readme_path = pathlib.Path(__file__).absolute().parent.parent / "README.md"
with open(str(readme_path), "r", encoding="utf-8") as f:
    readme = f.read()

setup(
    name="ttl",
    version=version,
    packages=["ttl", "ttl._src", "pykernel", "pykernel._src", "sim", "utils"],
    package_dir={
        "ttl": "ttl",
        "ttl._src": "ttl/_src",
        "pykernel": "pykernel",
        "pykernel._src": "pykernel/_src",
        "sim": "sim",
        "utils": "utils",
    },
    ext_modules=[ttlang_c],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    long_description=readme,
    long_description_content_type="text/markdown",
)
```

**What changed and why:**
- `pathlib.Path().absolute()` → `pathlib.Path(__file__).resolve().parent.parent` — fixes cwd resolving to pip's temp dir
- Added `TTLANG_TOOLCHAIN_DIR` / `TTLANG_USE_TOOLCHAIN` cmake passthrough — enables pre-built toolchain mode
- Always passes `-S repo_root` to cmake — was conditional on CI mode
- Builds both `TTLangPythonModules` and `PykernelPythonModules` — was missing pykernel
- Removed `in_ci()`, `rmdir()`, `import shutil`, `from datetime import datetime` — CI-specific hacks
- Replaced date-based version with `_get_version()` using git describe — proper PEP 440 versioning
- Removed `install_requires` — dependencies are now static in pyproject.toml

### 2b. `pyproject.toml` — Three edits

**Edit 1** — Line 2, fix build requires:

```toml
requires = ["setuptools>=61.0", "cmake>=3.28", "nanobind>=2.9,<3.0", "wheel", "ninja"]
```

(Pin cmake>=3.28, pin nanobind>=2.9,<3.0, remove "pip")

**Edit 2** — Lines 33-35, replace dynamic declaration:

Change:
```toml
dynamic = ["version", "dependencies", "readme"]
```

To:
```toml
readme = "README.md"
dynamic = ["version"]

dependencies = [
    "pydantic<3",
    "torch>=1.9.0",
    "numpy>=1.20.0",
    "greenlet>=3.0.0",
    "pandas<3",
    "nanobind>=2.9,<3.0",
    "PyYAML>=5.4.0,<=6.0.1",
    "typing_extensions>=4.12.2",
    "ml_dtypes>=0.1.0,<=0.6.0; python_version<'3.13'",
    "ml_dtypes>=0.5.0,<=0.6.0; python_version>='3.13'",
    "loguru>=0.6.0",
    "graphviz",
    "seaborn>=0.13.2",
]

[project.optional-dependencies]
sim = ["torch>=1.9.0", "numpy>=1.20.0", "greenlet>=3.0.0", "pydantic<3"]
dev = ["black", "pre-commit", "pyright", "lit"]
test = ["pytest>=7.0", "pytest-order>=1.0.0", "pytest-xdist>=3.0"]
```

**Edit 3** — Remove the `License :: OSI Approved` classifier from the `classifiers` list (conflicts with PEP 639 `license = "Apache-2.0"` field):

Delete this line from the classifiers array:
```
  "License :: OSI Approved :: Apache Software License",
```

**What changed and why:**
- Dependencies were declared dynamic but setup.py only listed `pydantic<3` — now static with all 13 entries
- `readme` moved from dynamic to static field
- Added optional-dependencies for sim/dev/test extras
- License classifier conflicts with PEP 639 SPDX license expression — newer setuptools rejects the combination

### 2c. `CMakeLists.txt` (root) — Two edits around line 176

**Edit 1** — Wrap subdirectories (replace the three `add_subdirectory` lines):

```cmake
if(NOT TTLANG_USE_TOOLCHAIN)
  add_subdirectory(include)
  add_subdirectory(lib)
  add_subdirectory(tools)
endif()
```

**Edit 2** — Add this block BEFORE `add_subdirectory(python)`:

```cmake
if(TTLANG_USE_TOOLCHAIN AND NOT TARGET TTLangCAPI)
  # Import ttlang dialect/transform libraries from the toolchain.
  foreach(_lib MLIRTTLDialect TTLangTTLTransforms TTLangTTLPipelines TTLangTTKernelTransforms)
    if(NOT TARGET ${_lib})
      add_library(${_lib} STATIC IMPORTED)
      set_target_properties(${_lib} PROPERTIES
        IMPORTED_LOCATION "${TTLANG_TOOLCHAIN_DIR}/lib/lib${_lib}.a"
      )
    endif()
  endforeach()

  # Create obj.TTLangCAPI OBJECT IMPORTED for MLIR aggregation support.
  if(NOT TARGET obj.TTLangCAPI)
    add_library(obj.TTLangCAPI OBJECT IMPORTED)
    file(GLOB_RECURSE _ttlang_capi_objects
      "${TTLANG_TOOLCHAIN_DIR}/lib/objects-Release/obj.TTLangCAPI/*.o"
    )
    set_target_properties(obj.TTLangCAPI PROPERTIES
      IMPORTED_OBJECTS "${_ttlang_capi_objects}"
    )
  endif()

  add_library(TTLangCAPI STATIC IMPORTED)
  set_target_properties(TTLangCAPI PROPERTIES
    IMPORTED_LOCATION "${TTLANG_TOOLCHAIN_DIR}/lib/libTTLangCAPI.a"
    MLIR_AGGREGATE_OBJECT_LIB_IMPORTED "obj.TTLangCAPI"
    MLIR_AGGREGATE_DEP_LIBS_IMPORTED "MLIRCAPIIR;MLIRIR;MLIRSupport;MLIRFuncDialect;TTLangTTLTransforms;TTLangTTLPipelines;TTLangTTKernelTransforms;MLIRTTLDialect"
  )
endif()
```

**What changed and why:**
- Skips building C++ compiler/tools during pip install (only Python bindings needed)
- Creates imported TTLangCAPI target from pre-built toolchain artifacts
- Uses `GLOB_RECURSE` (not `GLOB`) to find object files in nested directories — critical for MLIR aggregation to include all CAPI symbols

### 2d. `cmake/modules/BuildTTMLIRMinimal.cmake` — Two edits

**Edit 1** — Insert this block after the header comments (line ~21), BEFORE the existing `set(TT_MLIR_SOURCE_DIR ...)` line:

```cmake
# ---------------------------------------------------------------------------
# When consuming a pre-built toolchain, C++ dialect libraries and CAPI are
# pre-built.  Import them as IMPORTED targets.  But we still need the
# submodule source for TableGen (.td files) and C++ headers used by the
# Python bindings.
# ---------------------------------------------------------------------------
if(TTLANG_USE_TOOLCHAIN)
  include_directories(SYSTEM "${TTLANG_TOOLCHAIN_DIR}/include")

  if(NOT TARGET obj.TTMLIRMinimalCAPI)
    add_library(obj.TTMLIRMinimalCAPI OBJECT IMPORTED)
    file(GLOB_RECURSE _ttmlir_capi_objects
      "${TTLANG_TOOLCHAIN_DIR}/lib/objects-Release/obj.TTMLIRMinimalCAPI/*.o"
    )
    set_target_properties(obj.TTMLIRMinimalCAPI PROPERTIES
      IMPORTED_OBJECTS "${_ttmlir_capi_objects}"
    )
  endif()

  foreach(_lib
      MLIRTTCoreDialect MLIRTTTransforms MLIRTTKernelDialect
      MLIRTTKernelTransforms MLIRTTMetalDialect TTMLIRTTKernelToEmitC
      TTKernelTargetCpp)
    if(NOT TARGET ${_lib})
      add_library(${_lib} STATIC IMPORTED)
      set_target_properties(${_lib} PROPERTIES
        IMPORTED_LOCATION "${TTLANG_TOOLCHAIN_DIR}/lib/lib${_lib}.a"
      )
    endif()
  endforeach()

  if(NOT TARGET TTMLIRMinimalCAPI)
    add_library(TTMLIRMinimalCAPI STATIC IMPORTED)
    set_target_properties(TTMLIRMinimalCAPI PROPERTIES
      IMPORTED_LOCATION "${TTLANG_TOOLCHAIN_DIR}/lib/libTTMLIRMinimalCAPI.a"
      MLIR_AGGREGATE_OBJECT_LIB_IMPORTED "obj.TTMLIRMinimalCAPI"
      MLIR_AGGREGATE_DEP_LIBS_IMPORTED "MLIRTTCoreDialect;MLIRTTTransforms;MLIRTTKernelDialect;MLIRTTKernelTransforms;MLIRTTMetalDialect;TTMLIRTTKernelToEmitC;TTKernelTargetCpp;MLIRIR;MLIRCAPITransforms;MLIRSupport"
    )
  endif()
endif()
```

**IMPORTANT: Do NOT use `return()` here.** The rest of the file (TableGen steps, include paths, `.td` file processing) must still run — the Python bindings need the generated `.h.inc` files.

**Edit 2** — Find the section that builds C++ libraries (LLK headers, `add_subdirectory(lib/ttmlir-minimal)`, warning suppression) and wrap it:

```cmake
if(NOT TTLANG_USE_TOOLCHAIN)
  # ... LLK generated headers ...
  # ... add_subdirectory("${CMAKE_SOURCE_DIR}/lib/ttmlir-minimal" ...) ...
  # ... warning suppression ...
endif()
```

This skips only the C++ compilation while keeping TableGen and include path setup.

**What changed and why:**
- Creates imported TTMLIRMinimalCAPI target with proper MLIR aggregation properties
- Uses `GLOB_RECURSE` to find all object files including those in nested `__/__/__/third-party/` paths
- TableGen must still run to generate `.h.inc` files needed by Python binding compilation
- Only the C++ library build (`lib/ttmlir-minimal`) is skipped in toolchain mode

### 2e. `cmake/modules/BuildLLVM.cmake` — Add around line 139

Add before the existing venv creation block:

```cmake
if(TTLANG_USE_TOOLCHAIN)
  if(NOT DEFINED Python3_EXECUTABLE)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  endif()
  set(_SKIP_VENV_SETUP TRUE)
endif()
```

Then wrap the existing venv creation block with:

```cmake
if(NOT _SKIP_VENV_SETUP)
  # ... existing venv creation and pip install code ...
endif()
```

**What changed and why:**
- During pip install, pip already provides an isolated Python environment
- Creating a venv inside pip's build conflicts with pip's isolation
- The guard skips venv creation when using a pre-built toolchain

### 2f. `cmake/modules/TTLangCompilerSetup.cmake` — Line 17

Change:
```cmake
add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter)
```

To:
```cmake
add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter -Wno-unknown-warning-option)
```

**What changed and why:**
- The toolchain may be built with clang-17 but some flags (like `-Wno-deprecated-literal-operator`) are only valid in clang-18+
- `-Wno-unknown-warning-option` prevents `-Werror` from failing on unrecognized warning flags

### 2g. `test/CMakeLists.txt` — Around line 25

Change:
```cmake
set(TTLANG_TEST_DEPENDS)
list(APPEND TTLANG_TEST_DEPENDS ttlang-opt ttlang-translate)
```

To:
```cmake
set(TTLANG_TEST_DEPENDS)
if(NOT TTLANG_USE_TOOLCHAIN)
  list(APPEND TTLANG_TEST_DEPENDS ttlang-opt ttlang-translate)
endif()
```

**What changed and why:**
- In toolchain mode, `ttlang-opt` and `ttlang-translate` are pre-built binaries in the toolchain, not CMake targets
- Without the guard, CMake fails trying to resolve non-existent targets

### 2h. `setup.py` (NEW file at repo root)

Create this file at the repository root (next to `pyproject.toml`):

```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
# Root-level setup.py that delegates to the build logic in python/setup.py
# This file exists so that `pip install .` works from the repo root,
# where pyproject.toml lives.

import os
import pathlib
import subprocess

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class TTLangExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            if "ttl" in ext.name:
                self.build_(ext)
            else:
                raise Exception("Unknown extension")

    def build_(self, ext):
        build_lib = self.build_lib
        if not os.path.exists(build_lib):
            return
        extension_path = pathlib.Path(self.get_ext_fullpath(ext.name))
        print(f"Running cmake to install ttlang at {extension_path}")

        repo_root = pathlib.Path(__file__).resolve().parent
        build_dir = pathlib.Path(self.build_temp).resolve() / "cmake-build"
        install_dir = pathlib.Path(self.build_lib).resolve()

        toolchain_dir = os.environ.get("TTLANG_TOOLCHAIN_DIR", "")
        use_toolchain = "ON" if toolchain_dir else "OFF"
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

        self.spawn(["cmake", *cmake_args])
        self.spawn(["cmake", "--build", str(build_dir), "--",
                     "TTLangPythonModules", "PykernelPythonModules"])
        self.spawn(["cmake", "--install", str(build_dir), "--component", "TTLangPythonWheel"])


def _get_version():
    config_py = pathlib.Path(__file__).resolve().parent / "python" / "ttl" / "config.py"
    if config_py.exists():
        ns = {}
        exec(config_py.read_text(), ns)
        return ns.get("VERSION", "0.0.0.dev0")
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--dirty"],
            capture_output=True, text=True, check=True,
            cwd=pathlib.Path(__file__).resolve().parent,
        )
        tag = result.stdout.strip().lstrip("v")
        parts = tag.split("-")
        if len(parts) >= 3:
            return f"{parts[0]}.dev{parts[1]}"
        return parts[0]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "0.0.0.dev0"


version = _get_version()
ttlang_c = TTLangExtension("ttl")

setup(
    name="ttl",
    version=version,
    packages=["ttl", "ttl._src", "pykernel", "pykernel._src", "sim", "utils"],
    package_dir={
        "ttl": "python/ttl",
        "ttl._src": "python/ttl/_src",
        "pykernel": "python/pykernel",
        "pykernel._src": "python/pykernel/_src",
        "sim": "python/sim",
        "utils": "python/utils",
    },
    ext_modules=[ttlang_c],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
```

**What changed and why:**
- `pyproject.toml` lives at the repo root but `setup.py` only existed in `python/`
- pip expects `setup.py` next to `pyproject.toml`
- This root-level version uses `package_dir` pointing to `python/` subdirectories
- Uses `repo_root = pathlib.Path(__file__).resolve().parent` (not `.parent.parent` like the one in `python/`)

---

## Step 3: Build the toolchain (~1-3 hours, one-time)

```bash
export CC=/usr/bin/clang-17
export CXX=/usr/bin/clang++-17
export TTLANG_TOOLCHAIN_DIR=$HOME/ttlang-toolchain
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)

cmake -G Ninja -B build-toolchain \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX \
    -DTTLANG_TOOLCHAIN_DIR=$TTLANG_TOOLCHAIN_DIR \
    -DTTLANG_BUILD_TOOLCHAIN=ON

cmake --build build-toolchain
cmake --install build-toolchain --prefix $TTLANG_TOOLCHAIN_DIR
```

This builds LLVM/MLIR (~5400 targets), tt-metal (~1100 targets), and tt-lang (~86 targets), then installs everything into the toolchain directory (~6.2 GB).

**Note:** Clang 14 will fail with `error: no member named 'source_location' in namespace 'std'`. You need clang 17+.

---

## Step 4: Run a hardware test

```bash
source build-toolchain/env/activate
unset TT_VISIBLE_DEVICES
pytest test/me2e/test_compute_ops.py -v -k "add"
```

---

## Step 5 (optional): pip install into an external venv

```bash
source /path/to/your/venv/bin/activate
pip install setuptools cmake nanobind wheel ninja
export TTLANG_TOOLCHAIN_DIR=$HOME/ttlang-toolchain
export CC=/usr/bin/clang-17
export CXX=/usr/bin/clang++-17
pip install . --no-build-isolation
```

`--no-build-isolation` is required because `setup.py` resolves the repo root via `__file__`, which breaks under pip's isolated temp directory.

---

## Step 6 (optional): Run all tests

```bash
source build-toolchain/env/activate
cmake --build build-toolchain --target check-ttlang-all
```

### Expected results

| Suite | Expected |
|-------|----------|
| check-ttlang-mlir | 107/107 pass |
| check-ttlang-python-bindings | 3/3 pass |
| check-ttlang-me2e | ~528 pass, ~23 xfailed |
| check-ttlang-pytest | ~1115 pass, 1 skipped |
| check-ttlang-python-lit | 6/55 pass (49 fail — fabric control plane issue, not a build problem) |

---

## Troubleshooting

### `std::source_location` errors during toolchain build
Use clang 17+, not clang 14.

### `Physical chip id 0 not found in control plane chip mapping`
Use `source build-toolchain/env/activate` (not your own venv's ttnn). The toolchain's tt-metal is version-matched; external tt-metal builds may have fabric API mismatches.

### `undefined symbol: ttmlirTTKernelL1AddrTypeGet`
The `GLOB_RECURSE` fix in patches 2c and 2d was not applied. `file(GLOB ...)` misses object files in nested directories; `file(GLOB_RECURSE ...)` is required.

### pip install fails with "No configuration found for dynamic 'readme'"
The `pyproject.toml` patch (2b) must change `dynamic = ["version", "dependencies", "readme"]` to `dynamic = ["version"]` and add `readme = "README.md"` as a static field.

### `License classifiers have been superseded by license expressions`
Remove the `"License :: OSI Approved :: Apache Software License"` classifier from `pyproject.toml` — it conflicts with the PEP 639 `license = "Apache-2.0"` field.

### `No space left on device` during build
This often means `/tmp` is full, not `/localdev`. Set `TMPDIR` to a location with space:
```bash
export TMPDIR=/localdev/salnahari/tmp
mkdir -p $TMPDIR
```
Also the toolchain install defaults to `$HOME/ttlang-toolchain` — if `/home` is small, use `/localdev`:
```bash
export TTLANG_TOOLCHAIN_DIR=/localdev/salnahari/ttlang-toolchain
```

### Missing `.h.inc` files after toolchain build
The `cmake --install` for the toolchain may not copy TableGen-generated `.h.inc` files. Copy them manually:
```bash
cp -r build-toolchain/include/ttmlir $TTLANG_TOOLCHAIN_DIR/include/
cp -r build-toolchain/include/ttlang $TTLANG_TOOLCHAIN_DIR/include/
```

### `ModuleNotFoundError: No module named 'ttl.config'`
`config.py` is generated during CMake configure. Make sure you ran `cmake --build build` (or `build-toolchain`) and that `source build-toolchain/env/activate` is active (sets PYTHONPATH).
