# Discovery Mechanisms

This file documents how the TT-Lang build system locates its four major
external dependencies at configure time: LLVM/MLIR, tt-mlir, tt-metal, and
Python development packages.

---

## LLVM/MLIR Discovery

Handled entirely by `cmake/modules/BuildLLVM.cmake`, which supports two modes.

### Mode A: Pre-built LLVM (recommended for pip-install)

The user supplies either `MLIR_PREFIX` or `MLIR_DIR`:

```cmake
# BuildLLVM.cmake, line 126
if(DEFINED MLIR_PREFIX)
  set(MLIR_DIR "${MLIR_PREFIX}/lib/cmake/mlir" CACHE PATH "MLIR CMake dir" FORCE)
endif()
```

Then:

```cmake
# BuildLLVM.cmake, line 174
find_package(MLIR REQUIRED CONFIG)
```

`find_package(MLIR REQUIRED CONFIG)` searches for `MLIRConfig.cmake` at the
path `${MLIR_DIR}`.  On success it transitively provides `LLVM_DIR`,
`LLVM_INCLUDE_DIRS`, `MLIR_INCLUDE_DIRS`, `MLIR_CMAKE_DIR`, `LLVM_CMAKE_DIR`,
and all MLIR CMake functions (`AddMLIR`, `AddMLIRPython`, `TableGen`, etc.).

The toolchain variant (`TTLANG_USE_TOOLCHAIN=ON`) sets `MLIR_PREFIX` to
`${TTLANG_TOOLCHAIN_DIR}` automatically, then uses the same `find_package`
path.

A SHA verification step compares the installed LLVM commit against the SHA
recorded in TT-Lang's `third-party/llvm-project` submodule to catch version
mismatches:

```cmake
ttlang_verify_llvm_sha("${LLVM_INSTALL_DIR}" "${_TTLANG_EXPECTED_LLVM_SHA}")
```

### Mode B: Build from submodule

When neither `MLIR_PREFIX` nor `MLIR_DIR` is provided, `BuildLLVM.cmake`
builds LLVM/MLIR from the `third-party/llvm-project` git submodule at
**configure time** using `execute_process()`.  The build is configured with
Ninja, targets host-only, enables MLIR and Python bindings, and installs to
`${CMAKE_BINARY_DIR}/llvm-install` (or `${TTLANG_TOOLCHAIN_DIR}` if set).

After the build completes, `find_package(MLIR REQUIRED CONFIG)` runs against
the freshly installed tree.

### Post-discovery setup

Regardless of mode, `BuildLLVM.cmake` appends the MLIR and LLVM CMake module
paths and includes four standard LLVM/MLIR modules:

```cmake
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
```

These provide `mlir_tablegen()`, `add_mlir_dialect_library()`,
`add_mlir_python_common_capi_library()`, and the rest of the MLIR build
vocabulary used by all downstream `CMakeLists.txt` files.

---

## tt-mlir Source Discovery

Handled by `cmake/modules/BuildTTMLIRMinimal.cmake`.

tt-mlir is consumed as a **source submodule** at a hardcoded path:

```cmake
# BuildTTMLIRMinimal.cmake, line 22
set(TT_MLIR_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third-party/tt-mlir")
```

The submodule is ensured to be checked out:

```cmake
ttlang_ensure_submodules(third-party/tt-mlir)
```

Patches from `third-party/patches/ttmlir-*.patch` are applied automatically
for LLVM API compatibility fixes.

No `find_package()` or config-mode discovery is used for tt-mlir.  Instead,
TT-Lang compiles a curated subset of tt-mlir sources directly using
`add_mlir_dialect_library()` and `add_mlir_conversion_library()` in
`lib/ttmlir-minimal/CMakeLists.txt`.  TableGen is run on tt-mlir's `.td` files
by processing its `include/` directories via `add_subdirectory()`.

### Key paths derived from tt-mlir

| Variable | Value | Used for |
|----------|-------|----------|
| `TT_MLIR_SOURCE_DIR` | `${CMAKE_SOURCE_DIR}/third-party/tt-mlir` | Root of tt-mlir source tree |
| `TT_MLIR_INCLUDE_DIR` | `${TT_MLIR_SOURCE_DIR}/include` | TableGen `.td` resolution and C++ includes |
| `TTMLIR_PYTHON_ROOT_DIR` | `${CMAKE_SOURCE_DIR}/third-party/tt-mlir/python/ttmlir` | Python dialect binding sources |

### Why this matters for pip-install

Because tt-mlir is consumed as source (not as a pre-built library), any
pip-install solution must either:

1. Bundle the tt-mlir submodule in the source distribution, or
2. Pre-build the tt-mlir dialect libraries and provide them as a separate
   package that TT-Lang links against.

---

## tt-metal Discovery

Handled by `cmake/modules/BuildTTMetal.cmake`.

Like tt-mlir, tt-metal is consumed from a submodule:

```cmake
# BuildTTMetal.cmake, line 15
set(TT_METAL_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third-party/tt-metal")
```

The build produces three variables that downstream code and the runtime
activation script consume:

| Variable | Meaning |
|----------|---------|
| `TT_METAL_HOME` | Root of tt-metal source/build tree; used at runtime for JIT header lookup |
| `TT_METAL_PYTHON_PATH` | Paths added to `PYTHONPATH` for `ttnn` and tools packages |
| `TT_METAL_LIB_PATH` | Path added to `LD_LIBRARY_PATH` for tt-metal shared libraries |

Three discovery modes exist:

### Pre-built toolchain (`TTLANG_USE_TOOLCHAIN=ON`)

tt-metal artifacts are expected at `${TTLANG_TOOLCHAIN_DIR}/tt-metal/`.  No
build is performed:

```cmake
set(TT_METAL_HOME "${TTMETAL_BUILD_DIR}")
set(TT_METAL_PYTHON_PATH "${TTMETAL_BUILD_DIR}/python_packages/ttnn:...")
set(TT_METAL_LIB_PATH "${TTMETAL_BUILD_DIR}/lib")
```

### Submodule build (Linux)

tt-metal is built from `third-party/tt-metal` at configure time, including its
nested submodules (tracy, tt_llk, umd).

### Simulator-only mode

When `TTLANG_SIM_ONLY=ON` (declared at line 22 of the top-level
`CMakeLists.txt`), the entire compiler build is short-circuited by an
early-return block at line 65 of the top-level `CMakeLists.txt` -- *before*
`BuildTTMetal.cmake` is ever included.  The early-return creates a Python venv,
installs runtime requirements, generates an activate script, and then stops.
No LLVM, tt-mlir, or tt-metal build is performed, so the three tt-metal
variables are never set.  This mode is the most relevant for a lightweight
pip-install scenario that only needs the simulator.

---

## Python Dev Package Discovery

Handled by the top-level `CMakeLists.txt` (lines 140-141):

```cmake
include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()
```

`MLIRDetectPythonEnv` is a CMake module provided by the MLIR installation.  It:

1. Locates the Python 3 interpreter, development headers, and libraries using
   `find_package(Python3 COMPONENTS Interpreter Development.Module)`.
2. Locates **nanobind** (the binding library used for `_ttmlir` and `_ttlang`).
3. Sets `MLIR_PYTHON_NANOBIND_DIR` and other variables consumed by
   `declare_mlir_python_extension()`.

This must run **after** `BuildLLVM.cmake` (which may set `Python3_EXECUTABLE`
to a venv interpreter) and **before** `add_subdirectory(python)`.

### Python venv management

`BuildLLVM.cmake` manages a Python virtual environment at
`${TTLANG_PYTHON_VENV}` (defaulting to `${CMAKE_BINARY_DIR}/venv`).  If the
venv does not exist, it is created from the system Python and runtime
requirements are installed from `requirements.txt`.  The venv's interpreter
is then set as `Python3_EXECUTABLE` so all downstream `find_package(Python3)`
calls and `mlir_configure_python_dev_packages()` use it.

When building LLVM from submodule, MLIR's own Python requirements
(`mlir/python/requirements.txt`, which includes nanobind and PyYAML) are also
installed into this venv.

---

## Summary: Discovery Order

The following sequence in the top-level `CMakeLists.txt` establishes all
dependencies:

```
1. include(BuildLLVM)
   - Creates/activates Python venv
   - find_package(MLIR REQUIRED CONFIG)  [or builds from submodule first]
   - Provides: MLIR_DIR, LLVM_DIR, all MLIR CMake functions

2. include(BuildTTMLIRMinimal)
   - Locates third-party/tt-mlir submodule
   - Runs TableGen on tt-mlir .td files
   - Builds tt-mlir dialect C++ libraries

3. include(BuildTTMetal)
   - Locates third-party/tt-metal submodule
   - Builds tt-metal (or skips in simulator-only mode)
   - Sets TT_METAL_HOME, TT_METAL_PYTHON_PATH, TT_METAL_LIB_PATH

4. include(MLIRDetectPythonEnv) + mlir_configure_python_dev_packages()
   - Detects Python3, nanobind
   - Must run after BuildLLVM (Python3_EXECUTABLE may have changed)

5. add_subdirectory(python)
   - Uses all of the above to build extensions and stage the package
```

**Next:** [Chapter 4 -- Prior Art](../ch4_prior_art/index.md)
