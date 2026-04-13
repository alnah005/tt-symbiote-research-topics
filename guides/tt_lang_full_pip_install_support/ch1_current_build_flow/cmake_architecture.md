# CMake Architecture

This file documents the CMake build system that drives TT-Lang. The root `CMakeLists.txt` acts as the top-level orchestrator, and seven cmake modules under `cmake/modules/` implement the heavy lifting.

## Root `CMakeLists.txt` Structure

The root `CMakeLists.txt` follows a carefully ordered sequence. The ordering matters because several modules have data dependencies on earlier ones (e.g., `BuildLLVM.cmake` needs the Python venv from `TTLangPython.cmake`).

### Option declarations

Three key options control the build mode:

```cmake
# CMakeLists.txt
option(TTLANG_USE_TOOLCHAIN "Use pre-built LLVM from ttlang toolchain" OFF)
option(TTLANG_BUILD_TOOLCHAIN
  "Build LLVM and tt-metal into a reusable toolchain directory" OFF)
option(TTLANG_SIM_ONLY "Set up Python environment for simulator only (skip compiler build)" OFF)
```

- `TTLANG_USE_TOOLCHAIN` -- Consume a pre-built toolchain at `TTLANG_TOOLCHAIN_DIR` (skips LLVM and tt-metal builds entirely)
- `TTLANG_BUILD_TOOLCHAIN` -- Build a fresh toolchain from submodules into `TTLANG_TOOLCHAIN_DIR` (mutually exclusive with `TTLANG_USE_TOOLCHAIN`)
- `TTLANG_SIM_ONLY` -- Create a minimal Python venv for the simulator, then `return()` early -- no compiler, LLVM, or tt-metal build

The `TTLANG_TOOLCHAIN_DIR` variable is resolved from three sources in priority order:

1. Explicit `-DTTLANG_TOOLCHAIN_DIR=<path>` on the cmake command line
2. The `TTLANG_TOOLCHAIN_DIR` environment variable
3. Default `/opt/ttlang-toolchain` when `TTLANG_USE_TOOLCHAIN=ON`

### Include order and `add_subdirectory` calls

```
CMakeLists.txt
  |-- include(TTLangUtils)              # Utility functions (version, submodule init, pip helpers)
  |-- include(GetVersionFromGit)        # Version from git tags (e.g., v0.2.0 -> 0.2.0.dev5)
  |-- include(TTLangCompilerSetup)      # clang/lld detection, ccache, compiler flags
  |-- include(TTLangPython)             # Python venv creation/discovery
  |-- [TTLANG_SIM_ONLY early return]    # If sim-only, configure activate script and stop
  |-- include(BuildLLVM)                # LLVM/MLIR dependency (pre-built or submodule build)
  |-- include(BuildTTMLIRMinimal)       # tt-mlir dialect TableGen + C++ libraries
  |-- include(BuildTTMetal)             # tt-metal configure-time build
  |-- include(MLIRDetectPythonEnv)      # MLIR's Python dev package detection
  |-- add_subdirectory(include)         # tt-lang dialect .td files and headers
  |-- add_subdirectory(lib)             # tt-lang C++ libraries
  |-- add_subdirectory(tools)           # Compiler driver (ttlang-opt, etc.)
  |-- add_subdirectory(python)          # Python bindings (pybind11/nanobind)
  |-- add_subdirectory(test)            # lit tests
```

The ordering constraint is explicit in the source comments: `TTLangPython` must run before `BuildLLVM` so that `find_package(Python3)` resolves against the venv, not a system Python. Similarly, `BuildLLVM` must complete before `BuildTTMLIRMinimal` because TableGen requires LLVM/MLIR include directories.

### Simulator-only early exit

When `TTLANG_SIM_ONLY=ON`, the root CMakeLists.txt creates a minimal venv, installs `requirements.txt`, generates `env/activate`, and calls `return()`. No LLVM, tt-mlir, or tt-metal build occurs. This mode exists for users who only need the Python simulator (`ttlang-sim`).

## `BuildLLVM.cmake`

This module implements dual-mode LLVM dependency management.

### Mode A: Pre-built LLVM

When the user provides `MLIR_PREFIX`, `MLIR_DIR`, or uses `TTLANG_USE_TOOLCHAIN=ON`, the module runs `find_package(MLIR REQUIRED CONFIG)` against the existing installation. It then derives `LLVM_INSTALL_DIR` by stripping `lib/cmake/mlir` from the MLIR config path:

```cmake
# cmake/modules/BuildLLVM.cmake
get_filename_component(LLVM_INSTALL_DIR "${MLIR_DIR}/../../.." ABSOLUTE)
```

### Mode B: Build from submodule

When no pre-built MLIR is available, the module builds LLVM/MLIR from `third-party/llvm-project` at **CMake configure time** using `execute_process`. This is the expensive path -- three sequential `execute_process` calls for configure, build, and install:

```cmake
# cmake/modules/BuildLLVM.cmake
execute_process(COMMAND ${CMAKE_COMMAND} ${_LLVM_CMAKE_ARGS} ...)   # Configure
execute_process(COMMAND ${CMAKE_COMMAND} --build "${LLVM_BUILD_DIR}" ...)  # Build
execute_process(COMMAND ${CMAKE_COMMAND} --install "${LLVM_BUILD_DIR}" ...) # Install
```

Key LLVM build flags:

- `LLVM_ENABLE_PROJECTS=mlir` -- Only MLIR, no clang/libc++/etc.
- `LLVM_TARGETS_TO_BUILD=host` -- Host architecture only
- `MLIR_ENABLE_BINDINGS_PYTHON=ON` -- Python bindings for MLIR
- `LLVM_CCACHE_BUILD` -- Forwarded from parent ccache detection

### SHA verification

The module reads the expected LLVM commit SHA from the submodule's git HEAD and verifies it against the installed LLVM's `VCSRevision.h`. This uses the `ttlang_verify_llvm_sha()` utility from `TTLangUtils.cmake`, which parses the `LLVM_REVISION` macro from the header and compares SHAs via `scripts/verify-sha.sh`. On mismatch, it emits `FATAL_ERROR` unless `TTLANG_ACCEPT_LLVM_MISMATCH=ON`.

### Rebuild skip logic

If `${LLVM_INSTALL_DIR}/lib/cmake/mlir/MLIRConfig.cmake` already exists (and `TTLANG_BUILD_TOOLCHAIN` is not set), the entire LLVM build is skipped:

```cmake
# cmake/modules/BuildLLVM.cmake
if(EXISTS "${LLVM_INSTALL_DIR}/lib/cmake/mlir/MLIRConfig.cmake"
   AND NOT TTLANG_BUILD_TOOLCHAIN)
  message(STATUS "LLVM/MLIR already built at ${LLVM_INSTALL_DIR}, skipping rebuild")
```

ccache detection from `TTLangCompilerSetup.cmake` is forwarded to the LLVM build via `LLVM_CCACHE_BUILD`.

### Common setup

After either mode completes, the module appends MLIR and LLVM cmake directories to `CMAKE_MODULE_PATH` and includes standard LLVM/MLIR cmake helpers:

```cmake
# cmake/modules/BuildLLVM.cmake
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
```

## `BuildTTMLIRMinimal.cmake`

This module builds only the tt-mlir MLIR dialects needed by TT-Lang, directly from the `third-party/tt-mlir` submodule. It does not build the full tt-mlir project.

### TableGen processing

The module uses `add_subdirectory()` on tt-mlir's include directories to drive MLIR TableGen. Each `add_subdirectory` call sets `BINARY_DIR` to mirror the source layout under `${CMAKE_BINARY_DIR}/include/`, so generated `.inc` files land at canonical include paths:

```cmake
# cmake/modules/BuildTTMLIRMinimal.cmake
add_subdirectory(
  "${TT_MLIR_INCLUDE_DIR}/ttmlir/Dialect/TTCore/IR"
  "${CMAKE_BINARY_DIR}/include/ttmlir/Dialect/TTCore/IR")
```

The dialects processed are:

| Dialect | Subdirectories processed |
|---------|------------------------|
| **TTCore** | `IR`, `Transforms` |
| **TTMetal** | `IR`, `Transforms` |
| **TTKernel** | `IR`, `Transforms` |
| **Conversion** | Conversion passes (stablehlo branch skipped via unset `TTMLIR_ENABLE_STABLEHLO`) |

### LLK header generation

For the `TTKernelToCpp` target translation, the module generates C++ headers from LLK (Low Level Kernel) `.h` files. Each source header is converted to a raw string literal header using `GenerateRawStringHeader.cmake`:

```cmake
# cmake/modules/BuildTTMLIRMinimal.cmake
add_custom_command(
  OUTPUT ${output_file}
  COMMAND ${CMAKE_COMMAND}
    -DINPUT_FILE=${llk_header}
    -DOUTPUT_FILE=${output_file}
    -DVARIABLE_NAME=${header_name}_generated
    -P "${TT_MLIR_SOURCE_DIR}/cmake/modules/GenerateRawStringHeader.cmake"
  DEPENDS ${llk_header}
)
```

There are 14 LLK headers covering tilize/untilize operations, SFPI invocation, dataflow API, matmul, padding, coordinate translation, fabric topology, fabric 1D/2D routing, fabric API, register API, and semaphores.

### C++ dialect libraries

The actual C++ libraries are built via `add_subdirectory(lib/ttmlir-minimal)`, which uses MLIR's `add_mlir_dialect_library()`. The components built are:

- `MLIRTTCoreDialect` -- TTCore IR and transforms
- `MLIRTTMetalDialect` -- TTMetal IR and transforms
- `MLIRTTKernelDialect` and `MLIRTTKernelTransforms` -- TTKernel IR and transforms
- `TTMLIRTTKernelToEmitC` -- TTKernel to EmitC conversion
- `TTKernelTargetCpp` -- TTKernel to C++ target translation

### Patches

Before building, the module applies any patches matching `third-party/patches/ttmlir-*.patch` to the tt-mlir source tree. The `ttlang_apply_patches()` function skips patches that are already applied (checked via `git apply --reverse --check`).

## `BuildTTMetal.cmake`

This module builds tt-metal from `third-party/tt-metal` (also a configure-time build).

### Nested submodule initialization

tt-metal has its own nested submodules (tracy, tt_llk, umd) required for building. The module checks for sentinel files in each nested submodule and runs recursive submodule init if any are missing:

```cmake
# cmake/modules/BuildTTMetal.cmake
foreach(_sub tt_metal/third_party/tracy/CMakeLists.txt
  tt_metal/third_party/tt_llk/README.md
  tt_metal/third_party/umd/CMakeLists.txt)
  if(NOT EXISTS "${TT_METAL_SOURCE_DIR}/${_sub}")
    set(_nested_missing TRUE)
    break()
  endif()
endforeach()
```

### Sentinel-based rebuild skip

The module uses `_ttnn.so` as a sentinel file to determine whether tt-metal is already built:

```cmake
# cmake/modules/BuildTTMetal.cmake
set(_TTNN_SO "${TTMETAL_BUILD_DIR}/ttnn/_ttnn.so")
if(EXISTS "${_TTNN_SO}")
  message(STATUS "tt-metal already built at ${TTMETAL_BUILD_DIR}, skipping rebuild")
```

If the sentinel does not exist, any stale build directory is removed entirely before reconfiguring, avoiding `CMakeCache.txt` conflicts.

### Build configuration

The tt-metal build is configured with these notable flags:

- `TT_UNITY_BUILDS=ON` -- Unity builds for faster compilation
- `WITH_PYTHON_BINDINGS=ON` -- Build the `_ttnn.so` and `_ttnncpp.so` Python extensions
- `ENABLE_TRACY=${TTLANG_ENABLE_PERF_TRACE}` -- Tracy profiler integration
- `BUILD_SHARED_LIBS=ON` -- Shared libraries
- `CPM_SOURCE_CACHE` -- Forwarded from the environment or defaulting to `${TT_METAL_SOURCE_DIR}/.cpmcache`

### Artifact installation into toolchain

After building, several artifact-copying operations occur:

1. `_ttnn.so` and `_ttnncpp.so` are copied into the tt-metal source tree (`ttnn/ttnn/`) so `import ttnn` works from the source layout
2. Runtime artifacts (linker scripts, LLK headers, SoC descriptors) are saved via `scripts/copy-ttmetal-runtime-artifacts.sh`
3. If `TTLANG_TOOLCHAIN_DIR` is defined, all artifacts are installed into `${TTLANG_TOOLCHAIN_DIR}/tt-metal/` using `scripts/install-ttmetal.sh`

### SHA verification

The module verifies the tt-metal submodule SHA against the version pinned in `third-party/tt-mlir/third_party/CMakeLists.txt` (the `TT_METAL_VERSION` variable). On mismatch, it raises `FATAL_ERROR` unless `TTLANG_ACCEPT_TTMETAL_MISMATCH=ON`.

### Variables exported

After the module completes, three variables are set for use by `env/activate.in`:

- `TT_METAL_HOME` -- Root of tt-metal source (or toolchain install)
- `TT_METAL_PYTHON_PATH` -- Colon-separated paths for `PYTHONPATH` (ttnn and tools directories)
- `TT_METAL_LIB_PATH` -- Colon-separated paths for `LD_LIBRARY_PATH` (lib, tt_metal, ttnn, tt_stl, fmt, umd directories)

## `TTLangPython.cmake`

This module performs Python virtual environment **discovery and path resolution** for the build. It does **not** create the venv or install packages -- that responsibility belongs to `BuildLLVM.cmake` (for the main build path) and the `TTLANG_SIM_ONLY` block in the root `CMakeLists.txt` (for the simulator-only path).

### Venv search order

The search proceeds through four cases:

1. **User override** -- If `-DTTLANG_PYTHON_VENV=<path>` is set and the directory exists with a working interpreter, use it directly
2. **Toolchain venv** -- If `TTLANG_TOOLCHAIN_DIR` is defined and `${TTLANG_TOOLCHAIN_DIR}/venv` exists, use it (fatal error if interpreter is broken)
3. **Local project venv** -- If `${CMAKE_BINARY_DIR}/venv` exists and has a working interpreter, use it
4. **No venv yet** -- Set `TTLANG_PYTHON_VENV` to either `${TTLANG_TOOLCHAIN_DIR}/venv` or `${CMAKE_BINARY_DIR}/venv` for creation later (actual creation is deferred to `BuildLLVM.cmake`)

### Interpreter discovery

The `_ttlang_find_venv_python()` function searches for interpreters in this order:

1. `${venv_dir}/bin/python3`
2. `${venv_dir}/bin/python`
3. Versioned names matching `${venv_dir}/bin/python3.*`

Each candidate is verified by running `--version` to catch dangling symlinks.

### `VIRTUAL_ENV` activation

The `_ttlang_activate_venv()` macro sets the environment so that downstream `find_package(Python3)` uses the venv:

```cmake
# cmake/modules/TTLangPython.cmake
set(ENV{VIRTUAL_ENV} "${venv_dir}")
set(Python3_FIND_VIRTUALENV ONLY)
set(Python_FIND_VIRTUALENV ONLY)
unset(ENV{Python3_ROOT_DIR})
```

The `Python3_ROOT_DIR` unset is specifically to work around GitHub Actions' `setup-python`, which sets this variable to the runner's system Python and would otherwise override the venv.

## `TTLangCompilerSetup.cmake`

This module handles C/C++ compiler and linker configuration.

### ccache detection

ccache is auto-detected and enabled if found:

```cmake
# cmake/modules/TTLangCompilerSetup.cmake
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()
```

### Compiler flags

The module sets strict warning flags: `-Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter`.

### LLD linker detection

When using Clang, the module searches for a version-matched `ld.lld` (e.g., `ld.lld-18` for Clang 18). It extracts the Clang major version, searches for the versioned LLD binary, and verifies the major versions match before setting `CMAKE_LINKER_TYPE=LLD`:

```cmake
# cmake/modules/TTLangCompilerSetup.cmake
set(LD_LLD_EXECUTABLE_VERSIONED "ld.lld-${CLANG_VERSION_MAJOR}")
find_program(LLD NAMES ${LD_LLD_EXECUTABLE_VERSIONED} ld.lld)
```

If versions do not match, the default system linker is used instead.

---

**Next:** [`environment_assumptions.md`](./environment_assumptions.md)
