# Environment Assumptions

This file documents the environment variables, tool requirements, and activation script that TT-Lang's build system depends on. These implicit assumptions are a key reason `pip install` does not work today -- the build expects a specific host environment that Python packaging tools do not provide.

## Environment Variables Consumed

The build system reads the following environment variables at various points during configure and build:

| Variable | Where consumed | Purpose |
|----------|---------------|---------|
| `TTLANG_TOOLCHAIN_DIR` | `CMakeLists.txt`, `BuildLLVM.cmake`, `build-and-install.sh` | Root directory for the reusable toolchain (LLVM, tt-metal, venv). Defaults to `/opt/ttlang-toolchain` in the shell script. |
| `CMAKE_BINARY_DIR` | `build-and-install.sh` | Overrides the build directory. Defaults to `build` (or `build-toolchain` for `--toolchain-only` mode). |
| `IN_CIBW_ENV` | Not currently referenced in cmake modules but reserved for `cibuildwheel` integration. Indicates the build is running inside a `cibuildwheel` container. |
| `TT_METAL_SIMULATOR` | `TTLangUtils.cmake` (`ttlang_check_device_available`) | When set, the build assumes a Tenstorrent device is available via the simulator, bypassing `/dev/tenstorrent*` detection. |
| `CPM_SOURCE_CACHE` | `BuildTTMetal.cmake` | Directory for CPM's download cache (tt-metal uses CPM for its C++ dependencies). Defaults to `${TT_METAL_SOURCE_DIR}/.cpmcache`. |
| `TTLANG_CMAKE_DEBUG` | `TTLangUtils.cmake` (`ttlang_debug_message`) | When defined, enables verbose debug output during CMake configuration. |
| `VIRTUAL_ENV` | `TTLangPython.cmake` | Set by `_ttlang_activate_venv()` to direct `find_package(Python3)` to the venv interpreter. |
| `Python3_ROOT_DIR` | `TTLangPython.cmake` | Explicitly unset during venv activation to prevent GitHub Actions' `setup-python` from overriding venv discovery. |

## Pre-installed Tool Requirements

The build system assumes the following tools are available on the host system before CMake is invoked:

### Compilers and linkers

- **clang / clang++** -- The default C/C++ compilers (set in `CMakeLists.txt` via `CMAKE_C_COMPILER` and `CMAKE_CXX_COMPILER` if not already defined). The build does not support GCC.
- **lld** -- The LLVM linker, specifically a version-matched `ld.lld-<N>` where `<N>` matches the Clang major version. Detected by `TTLangCompilerSetup.cmake`. Falls back to the default linker if not found.

### Build tools

- **CMake >= 3.28** -- Enforced by `cmake_minimum_required(VERSION 3.28.0)` in the root `CMakeLists.txt`.
- **Ninja** -- The build system generator. The `build-and-install.sh` script hardcodes `-G Ninja`.
- **ccache** *(optional)* -- Automatically detected and enabled for both the tt-lang build and forwarded to LLVM and tt-metal builds.

### Runtime tools

- **Python >= 3.11** -- Required for MLIR Python bindings, tt-metal's ttnn bindings, and tt-lang's own Python packages. The exact minimum is implicitly enforced by dependency versions in `requirements.txt`.
- **git** -- Required for submodule initialization, SHA verification, version detection from tags, and patch application. The build gracefully degrades when `.git` is absent (e.g., Docker contexts) but many features are lost.

## The `env/activate.in` Template

The file `env/activate.in` is a CMake `configure_file` template that generates the shell activation script at `${CMAKE_BINARY_DIR}/env/activate`. CMake substitutes `@VARIABLE@` placeholders with their values at configure time.

### Variables set by the activation script

The generated script sets the following environment:

```bash
# env/activate.in (after CMake substitution)

# Python venv activation
source "@TTLANG_PYTHON_VENV@/bin/activate"

# Core paths
export TT_LANG_HOME="@TT_LANG_HOME@"
export TTLANG_HAS_DEVICE="@TTLANG_HAS_DEVICE_INT@"
export LLVM_INSTALL_DIR="@LLVM_INSTALL_DIR@"

# PATH: build tools, LLVM tools, tt-lang scripts, venv
export PATH="@CMAKE_BINARY_DIR@/bin:${TT_LANG_HOME}/bin:${LLVM_INSTALL_DIR}/bin:@TTLANG_PYTHON_VENV@/bin:$PATH"

# PYTHONPATH: built MLIR + tt-mlir + ttlang bindings, source trees
export PYTHONPATH="\
@CMAKE_BINARY_DIR@/python_packages:\
${TT_LANG_HOME}/python:\
${TT_LANG_HOME}/test:\
${PYTHONPATH:-}"

# LD_LIBRARY_PATH: built shared libraries + LLVM libs
export LD_LIBRARY_PATH="@CMAKE_BINARY_DIR@/lib:${LLVM_INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"

# tt-metal runtime
export TT_METAL_HOME="@TT_METAL_HOME@"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_HOME}"
export PYTHONPATH="@TT_METAL_PYTHON_PATH@:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="@TT_METAL_LIB_PATH@:${LD_LIBRARY_PATH:-}"

export TTLANG_ENV_ACTIVATED=1
```

### What each path does

| Variable | Content | Why needed |
|----------|---------|------------|
| `TT_LANG_HOME` | TT-Lang repository root | Used by Python code and scripts to locate source files, examples, and test data |
| `LLVM_INSTALL_DIR` | LLVM/MLIR install prefix | Tools like `llvm-lit` and MLIR Python bindings need to locate LLVM libraries |
| `PATH` additions | Build bin, repo bin, LLVM bin, venv bin | Makes `ttlang-opt`, `ttlang-sim`, `llvm-lit`, `FileCheck`, and the venv Python available |
| `PYTHONPATH` additions | Build python_packages, `python/`, `test/`, ttnn, tools | Makes `import ttlang`, `import ttnn`, `import ttmlir`, and test utilities importable |
| `LD_LIBRARY_PATH` additions | Build lib, LLVM lib, tt-metal lib dirs (6 paths) | Shared libraries for MLIR, tt-metal runtime, ttnn, fmt, umd |
| `TT_METAL_HOME` | tt-metal source or toolchain root | JIT compilation at runtime resolves firmware sources and headers via this path |
| `TT_METAL_RUNTIME_ROOT` | Same as `TT_METAL_HOME` | Alias used by tt-metal's runtime layer |

### Guard against re-activation

The script checks `TTLANG_ENV_ACTIVATED` and calls `deactivate` (from the Python venv) before re-activating, preventing `PATH` and `PYTHONPATH` from accumulating duplicate entries across repeated `source env/activate` calls.

## Why `source env/activate` Is Currently Required

After every build (or rebuild), the user must run:

```bash
source build/env/activate
```

This is required because:

1. **`PYTHONPATH` is not baked into the wheel.** The built Python packages (`ttlang`, `ttmlir` bindings, `ttnn`) are not installed into the venv's `site-packages` via pip. Instead, they sit in `${CMAKE_BINARY_DIR}/python_packages/` and the source tree's `python/` directory, which are only reachable via `PYTHONPATH`.

2. **`LD_LIBRARY_PATH` is not encoded in the shared objects.** The built `.so` files (MLIR libraries, tt-metal runtime, ttnn) do not have `RPATH` entries pointing to their dependencies. The activation script manually adds six tt-metal library directories and two LLVM/tt-lang library directories to `LD_LIBRARY_PATH`.

3. **`TT_METAL_HOME` must point to the source tree.** The tt-metal JIT build system resolves firmware source files, linker scripts, and LLK headers relative to `TT_METAL_HOME` at device runtime. Without this variable, any operation that touches hardware will fail.

4. **Build-directory-relative paths.** The generated `env/activate` hardcodes `@CMAKE_BINARY_DIR@` at configure time. Moving the build directory or running from a different shell session requires re-sourcing the script.

This manual activation step is the single most important reason TT-Lang cannot be distributed as a standard Python package today. A proper `pip install` must eliminate the need for `source env/activate` by:

- Installing Python packages into `site-packages` (eliminating `PYTHONPATH`)
- Setting `RPATH` in shared objects or bundling libraries (eliminating `LD_LIBRARY_PATH`)
- Embedding or discovering `TT_METAL_HOME` equivalent paths relative to the installed package
- Removing hard-coded build directory paths

---

**Next:** [Chapter 2 -- Python Packaging As-Is](../ch2_python_packaging_as_is/index.md)
