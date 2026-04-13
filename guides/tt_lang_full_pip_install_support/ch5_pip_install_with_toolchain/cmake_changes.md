# CMake Changes

The CMake build system must work in two modes: the full from-source build (existing behavior) and a lightweight pip-install build that consumes a pre-built toolchain. This file details the changes needed so that `python/CMakeLists.txt` can produce the `TTLangPythonModules` target when invoked with `TTLANG_USE_TOOLCHAIN=ON`.

## 1. Skip Heavy Builds When `TTLANG_USE_TOOLCHAIN=ON`

### `BuildLLVM.cmake` -- Already handled

`BuildLLVM.cmake` (lines 90-98) already handles the toolchain case: when `TTLANG_USE_TOOLCHAIN=ON`, it sets `MLIR_PREFIX` from `TTLANG_TOOLCHAIN_DIR` and calls `find_package(MLIR REQUIRED CONFIG)` against the pre-built install. No LLVM submodule build is triggered.

### `BuildTTMetal.cmake` -- Already handled

`BuildTTMetal.cmake` (lines 44-56) returns early when `TTLANG_USE_TOOLCHAIN=ON`, setting `TT_METAL_HOME`, `TT_METAL_PYTHON_PATH`, and `TT_METAL_LIB_PATH` from the toolchain directory.

### `BuildTTMLIRMinimal.cmake` -- Needs a toolchain guard

`BuildTTMLIRMinimal.cmake` is always included (line 130 of the root `CMakeLists.txt`) and currently has no `TTLANG_USE_TOOLCHAIN` guard. When using a toolchain, the tt-mlir dialects (C++ libraries, TableGen outputs) are already pre-built and installed alongside MLIR. The entire file should be skipped.

**Proposed change** -- add an early return at the top of `BuildTTMLIRMinimal.cmake`:

```cmake
# BuildTTMLIRMinimal.cmake, after the file header comment

# ---------------------------------------------------------------------------
# When consuming a pre-built toolchain, all tt-mlir artifacts (C++ dialect
# libraries, TableGen-generated .inc files, CAPI) are pre-installed.  Skip
# the submodule build entirely.
# ---------------------------------------------------------------------------
if(TTLANG_USE_TOOLCHAIN)
  # The toolchain install includes tt-mlir headers and libraries.
  # In a source build TT_MLIR_SOURCE_DIR points to the tt-mlir submodule root
  # (${CMAKE_SOURCE_DIR}/third-party/tt-mlir).  In toolchain mode we do not
  # have the submodule, so leave TT_MLIR_SOURCE_DIR unset -- no downstream
  # code in the pip-install path references it.
  include_directories(SYSTEM "${TTLANG_TOOLCHAIN_DIR}/include")

  # TTMLIRMinimalCAPI is pre-built in the toolchain; create an imported
  # target so that python/CMakeLists.txt's EMBED_CAPI_LINK_LIBS can find it.
  if(NOT TARGET TTMLIRMinimalCAPI)
    add_library(TTMLIRMinimalCAPI STATIC IMPORTED)
    set_target_properties(TTMLIRMinimalCAPI PROPERTIES
      IMPORTED_LOCATION "${TTLANG_TOOLCHAIN_DIR}/lib/libTTMLIRMinimalCAPI.a"
    )
  endif()

  return()
endif()
```

This ensures that when pip invokes the full `CMakeLists.txt` with `TTLANG_USE_TOOLCHAIN=ON`, the tt-mlir submodule build (TableGen, C++ dialect libraries, LLK headers) is entirely skipped.

### Root `CMakeLists.txt` -- Guard non-Python targets

Lines 176-179 of the root `CMakeLists.txt` unconditionally add `include`, `lib`, `tools`, and `test` subdirectories:

```cmake
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
```

These contain the TTL dialect C++ implementation, the `ttl-opt` tool, and the compiler library. During a pip install, we only need the Python bindings. Wrap them in a guard:

```cmake
if(NOT TTLANG_USE_TOOLCHAIN)
  add_subdirectory(include)
  add_subdirectory(lib)
  add_subdirectory(tools)
endif()
```

The `TTLangCAPI` target (referenced by `python/CMakeLists.txt` line 134 via `EMBED_CAPI_LINK_LIBS TTLangCAPI`) is defined in `lib/`. When using the toolchain, this must be an imported target, similar to `TTMLIRMinimalCAPI` above:

```cmake
if(TTLANG_USE_TOOLCHAIN AND NOT TARGET TTLangCAPI)
  add_library(TTLangCAPI STATIC IMPORTED)
  set_target_properties(TTLangCAPI PROPERTIES
    IMPORTED_LOCATION "${TTLANG_TOOLCHAIN_DIR}/lib/libTTLangCAPI.a"
  )
endif()
```

This block should go in the root `CMakeLists.txt` just before `add_subdirectory(python)` (line 181).

## 2. Ensure Toolchain Installs CAPI Libraries

For the imported targets above to work, the toolchain build must install `libTTMLIRMinimalCAPI.a` and `libTTLangCAPI.a`. These are currently not installed by any `install()` command.

Add install rules to `lib/ttmlir-minimal/CAPI/CMakeLists.txt` and `lib/CAPI/CMakeLists.txt` (or wherever the CAPI targets are defined):

```cmake
# In the CMakeLists.txt that defines TTMLIRMinimalCAPI:
install(TARGETS TTMLIRMinimalCAPI
  ARCHIVE DESTINATION lib
  COMPONENT TTLangToolchain
  EXCLUDE_FROM_ALL)

# In the CMakeLists.txt that defines TTLangCAPI:
install(TARGETS TTLangCAPI
  ARCHIVE DESTINATION lib
  COMPONENT TTLangToolchain
  EXCLUDE_FROM_ALL)
```

The `TTLangToolchain` component should be installed by the toolchain build script (see [Chapter 4](../ch4_prior_art/index.md) for the toolchain-build pattern).

## 3. `config.py.in` and `_generated_elementwise.py` -- No Changes Needed

Both generated files work as-is during a toolchain pip install:

- **`config.py`** (`python/CMakeLists.txt` lines 157-161): Generated via `configure_file()` from variables set by the root `CMakeLists.txt` (`TTLANG_VERSION` at line 18, `TTLANG_HAS_DEVICE_INT` at line 152). Both variables are set regardless of `TTLANG_USE_TOOLCHAIN`, and `ttlang_check_device_available()` has no build dependencies (it checks device nodes only).
- **`_generated_elementwise.py`** (`python/CMakeLists.txt` lines 144-153): A pure-Python `gen_elementwise.py` script run against a `.def` file in the source tree. No C++ compilation or MLIR tools involved.

## 4. Fix `Python3_EXECUTABLE` for pip Builds

When pip invokes `setup.py`, it runs inside a build-isolation venv. The `TTLangPython.cmake` module (line 80-132) searches for a venv at `TTLANG_PYTHON_VENV`, then falls back to creating one at `${CMAKE_BINARY_DIR}/venv`. During pip install, this venv creation is unnecessary and harmful -- pip already provides an isolated environment.

The most reliable detection is to tie venv-skip to `TTLANG_USE_TOOLCHAIN` itself, since the toolchain path already implies a pip-based workflow. Environment variables like `PIP_BUILD_TRACKER` (removed in pip 23.1) and `_PIP_STANDALONE_CERT` (internal, not guaranteed) are unreliable across pip versions.

Add the following to the root `CMakeLists.txt` (or `BuildLLVM.cmake`), before the venv creation block:

```cmake
# When using a pre-built toolchain (the pip-install path), skip venv creation.
# pip's build isolation already provides the correct Python environment.
if(TTLANG_USE_TOOLCHAIN)
  if(NOT DEFINED Python3_EXECUTABLE)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  endif()
  set(_SKIP_VENV_SETUP TRUE)
endif()
```

Then, in `BuildLLVM.cmake`, wrap the venv creation block (lines 139-167) with:

```cmake
if(NOT _SKIP_VENV_SETUP)
  # ... existing venv creation and requirements installation ...
endif()
```

## 5. `CMAKE_INSTALL_PREFIX` -- No Change Needed

The `setup.py` fix (see [`setup_py_fixes.md`](./setup_py_fixes.md) section 1) sets `CMAKE_INSTALL_PREFIX` to setuptools' `build_lib` directory. The `TTLangPythonWheel` install component uses `DESTINATION .`, so files land directly into `build_lib` as setuptools expects.

## 6. `python/CMakeLists.txt` Include Path for `_ttmlir` Extension

Lines 286-290 add include directories for the `_ttmlir` extension:

```cmake
target_include_directories(TTLangPythonModules.extension._ttmlir.dso PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/ttmlir"
  "${CMAKE_SOURCE_DIR}/third-party/tt-mlir/python"
  "${CMAKE_SOURCE_DIR}/lib/ttmlir-minimal/CAPI"
)
```

When `TTLANG_USE_TOOLCHAIN=ON`, `third-party/tt-mlir/python` may not exist (if git submodules are not initialized). The `.cpp` files in `python/ttmlir/` include headers from the tt-mlir Python sources (e.g., `TTModule.cpp`, `TTKernelModule.cpp`).

Two options:

**Option A (recommended):** Require the tt-mlir submodule to be checked out for source builds, even with toolchain. The submodule only needs to be initialized (no recursive build). This is already the case in practice -- `python/CMakeLists.txt` line 17 references `third-party/tt-mlir/python/ttmlir/` for Python source declarations.

**Option B:** Copy the required headers into the toolchain install and adjust the include path:

```cmake
if(TTLANG_USE_TOOLCHAIN)
  target_include_directories(TTLangPythonModules.extension._ttmlir.dso PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/ttmlir"
    "${TTLANG_TOOLCHAIN_DIR}/share/tt-mlir/python"
    "${TTLANG_TOOLCHAIN_DIR}/include/ttmlir-minimal/CAPI"
  )
else()
  target_include_directories(TTLangPythonModules.extension._ttmlir.dso PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/ttmlir"
    "${CMAKE_SOURCE_DIR}/third-party/tt-mlir/python"
    "${CMAKE_SOURCE_DIR}/lib/ttmlir-minimal/CAPI"
  )
endif()
```

Option A is simpler and avoids duplicating sources into the toolchain.

## 7. Summary of CMake Changes

| File | Change | Why |
|------|--------|-----|
| `cmake/modules/BuildTTMLIRMinimal.cmake` | Add `if(TTLANG_USE_TOOLCHAIN) return()` guard with imported `TTMLIRMinimalCAPI` target | Skip submodule build; provide CAPI library for linking |
| `CMakeLists.txt` (root, ~line 176) | Wrap `add_subdirectory(include/lib/tools)` in `if(NOT TTLANG_USE_TOOLCHAIN)` | Skip compiler C++ build during pip install |
| `CMakeLists.txt` (root, before line 181) | Add imported `TTLangCAPI` target when using toolchain | Provide CAPI library for nanobind extension linking |
| `cmake/modules/BuildLLVM.cmake` (~line 139) | Skip venv creation when `TTLANG_USE_TOOLCHAIN=ON` | Avoid conflicting with pip's build isolation |
| `lib/ttmlir-minimal/CAPI/CMakeLists.txt` | Add `install(TARGETS TTMLIRMinimalCAPI ...)` | Make CAPI available in toolchain installs |
| `lib/CAPI/CMakeLists.txt` | Add `install(TARGETS TTLangCAPI ...)` | Make CAPI available in toolchain installs |
| `python/CMakeLists.txt` (~line 286) | Conditional include paths for toolchain vs. submodule | Find headers regardless of build mode |
| `python/CMakeLists.txt` (config.py, elementwise) | No change needed | Both generators work as-is (see section 3) |
| Root `CMakeLists.txt` install prefix | No change needed | `setup.py` sets `CMAKE_INSTALL_PREFIX` correctly (see section 6) |

> **Note:** For the end-to-end flow of how these changes interact during `pip install`, see [index.md -- How It Fits Together](./index.md#how-it-fits-together).

**Prev:** [`setup_py_fixes.md`](./setup_py_fixes.md) | **Next:** [Chapter 6 -- Two-Phase Wheel Architecture](../ch6_two_phase_wheel_architecture/index.md)
