# Main Wheel Design (`ttl`)

This file specifies the contents of the `ttl` wheel, its build-time and runtime dependencies on `ttl-toolchain`, and the mechanisms by which the compiled extensions find their CAPI and MLIR shared library dependencies at runtime.

## What Goes Into the Wheel

The `ttl` wheel contains everything produced by the `TTLangPythonModules` CMake target (see `python/CMakeLists.txt`) plus the additional pure-Python packages declared in `setup.py`. It does **not** contain any LLVM, MLIR, or tt-metal libraries -- those come from the `ttl-toolchain` dependency.

### Compiled Extensions

Three shared objects are compiled from TT-Lang sources against the pre-built toolchain:

| Artifact | CMake Target | What It Contains |
|----------|-------------|-----------------|
| `_ttlang.so` | `TTLangPythonExtensions.Main` | Nanobind bindings for the TTL dialect: ops, attributes, types, passes. Source: `python/ttlang/TTLangModule.cpp`, `python/ttlang/TTLModule.cpp` |
| `_ttmlir.so` | `TTMLIRMinPythonExtensions.Main` | Nanobind bindings for TTCore, TTKernel, TTMetal dialects and pass registration. Source: `python/ttmlir/TT_MLIRMinimal*.cpp` (4 files) |
| `TTLangPythonCAPI.so` | `TTLangPythonCAPI` | Shared CAPI library that aggregates upstream MLIR CAPI, `TTMLIRMinimalCAPI`, and `TTLangCAPI`. Built by `add_mlir_python_common_capi_library()` in `python/CMakeLists.txt` |

As documented in [Chapter 3](../ch3_cpp_extension_dependencies/index.md), `_ttlang.so` and `_ttmlir.so` both use the nanobind domain `"ttl"` (set by `MLIR_BINDINGS_PYTHON_NB_DOMAIN`) so that MLIR C types are interoperable across the two modules. Both extensions `EMBED_CAPI_LINK_LIBS` into `TTLangPythonCAPI.so`, meaning they dlopen that shared library at import time rather than statically linking the CAPI.

### Pure Python Packages

| Package | Install Location | Source |
|---------|-----------------|--------|
| `ttl` | `site-packages/ttl/` | `python/ttl/` -- compiler API, layouts, operators, diagnostics |
| `ttl._src` | `site-packages/ttl/_src/` | AST, profiling, tensor registry internals |
| `ttl._mlir_libs` | `site-packages/ttl/_mlir_libs/` | Extension loader, site initialization (`_site_initialize_0.py`, `_site_initialize_1.py`) |
| `ttl.dialects` | `site-packages/ttl/dialects/` | ODS-generated dialect bindings (TTL, TTCore, TTKernel) |
| `pykernel` | `site-packages/pykernel/` | Kernel authoring DSL |
| `sim` | `site-packages/sim/` | Device simulator |
| `utils` | `site-packages/utils/` | Block allocation, correctness utilities |

### Generated Files

Two files are generated during the CMake build and must be included in the wheel:

| File | Generator | Notes |
|------|-----------|-------|
| `ttl/config.py` | `configure_file()` from `python/ttl/config.py.in` | Contains `TTLANG_HAS_DEVICE` flag and path constants |
| `ttl/_generated_elementwise.py` | `python/gen_elementwise.py` from `include/ttlang/Dialect/TTL/TTLElementwiseOps.def` | Elementwise op bindings generated from the `.def` file |

Both are produced by the `CMakeBuild.build_()` step in `setup.py` before the wheel is assembled.

## Build-Time Dependency on `ttl-toolchain`

The `ttl` wheel requires `ttl-toolchain` to be installed **before** the build starts, because the CMake configure step needs to `find_package(MLIR)` against the toolchain's CMake configs.

### `pyproject.toml` Build Requirements

```toml
[build-system]
requires = [
    "setuptools>=61.0",
    "cmake>=3.28",
    "nanobind",
    "wheel",
    "ninja",
    "ttl-toolchain==0.1.250413",   # <-- pre-built LLVM/MLIR + tt-metal
]
build-backend = "setuptools.build_meta"
```

When pip processes `pip install ttl` (or `pip install .`), it reads `[build-system].requires`, installs `ttl-toolchain` into the build isolation environment, then invokes the build backend. The `CMakeBuild` class in `setup.py` locates the toolchain via the Python API:

```python
# setup.py (modified CMakeBuild.build_)
import ttl_toolchain

cmake_args = [
    "-G", "Ninja",
    "-B", str(build_dir),
    "-S", str(source_dir),
    "-DCMAKE_BUILD_TYPE=Release",
    f"-DMLIR_DIR={ttl_toolchain.get_mlir_cmake_dir()}",
    f"-DTTLANG_TOOLCHAIN_DIR={ttl_toolchain.get_toolchain_dir()}",
    "-DTTLANG_USE_TOOLCHAIN=ON",
]
```

This replaces the current approach (Chapter 5) of requiring `TTLANG_TOOLCHAIN_DIR` as an environment variable. The environment variable still works as an override for developers who maintain their own toolchain build, but the default path uses the pip-installed package.

### What CMake Needs from the Toolchain

The CMake configure step uses the toolchain for:

1. **`find_package(MLIR REQUIRED CONFIG)`** -- resolved via `MLIR_DIR` pointing into `ttl_toolchain/mlir/lib/cmake/mlir/`. This provides all MLIR CMake macros (`add_mlir_dialect`, `declare_mlir_python_extension`, `mlir_tablegen`, etc.).

2. **`find_package(LLVM)`** -- transitively pulled in by MLIRConfig.cmake. Provides `LLVM_INCLUDE_DIRS`, `LLVM_DEFINITIONS`, and the `AddLLVM` module.

3. **TableGen code generation** -- `mlir-tblgen` from `ttl_toolchain/mlir/bin/` is invoked to generate `.h.inc` files for the TTL, TTCore, and TTKernel dialects from their `.td` definitions.

4. **Header inclusion** -- nanobind extension source files include MLIR and LLVM headers (`mlir/IR/Operation.h`, `mlir-c/IR.h`, etc.) from the toolchain's `include/` directories.

5. **CAPI library linking** -- `_ttlang.so` and `_ttmlir.so` link against `TTLangPythonCAPI.so`, which in turn links the static CAPI libraries (`libTTMLIRMinimalCAPI.a`, `libTTLangCAPI.a`, `libMLIRCAPIIR.a`, etc.) from the toolchain.

## Runtime Dependency on `ttl-toolchain`

### `pyproject.toml` Install Requirements

```toml
[project]
dependencies = [
    "ttl-toolchain==0.1.250413",
    "pydantic<3",
    "torch>=1.9.0",
    "numpy>=1.20.0",
    "greenlet>=3.0",
]
```

At runtime, the extensions need to load shared libraries from the toolchain:

- `TTLangPythonCAPI.so` links against `libMLIRIR.so`, `libMLIRPass.so`, and other MLIR shared libraries.
- `_ttnn.so` (from tt-metal, re-exported via `ttl_toolchain`) links against `libtt_metal.so` and `libdevice.so`.

### How Extensions Find Shared Libraries at Runtime

The key challenge is ensuring that when Python loads `_ttlang.so`, the dynamic linker can find the MLIR shared libraries that `TTLangPythonCAPI.so` depends on. There are three viable approaches:

#### Approach A: RPATH Patching (Impractical)

Cross-package RPATH is impractical because `site-packages/ttl/_mlir_libs/TTLangPythonCAPI.so` cannot predict the relative path to `site-packages/ttl_toolchain/mlir/lib/` -- pip may install packages in different `site-packages` directories (e.g., user vs. system).

#### Approach B: `ctypes` Pre-loading (Recommended)

The `ttl/_mlir_libs/__init__.py` module (the MLIR extension loader) pre-loads the toolchain's shared libraries using `ctypes.CDLL` with `RTLD_GLOBAL` before any extension is imported:

```python
# ttl/_mlir_libs/__init__.py
import ctypes
import sys
import ttl_toolchain

def _preload_toolchain_libs():
    """Pre-load LLVM/MLIR shared libraries so extensions can resolve symbols."""
    tc = ttl_toolchain.get_toolchain_dir()
    mlir_lib = tc / "mlir" / "lib"
    ttmetal_lib = tc / "ttmetal" / "lib"

    # Load order matters: LLVM first, then MLIR, then tt-metal
    for lib_dir in [mlir_lib, ttmetal_lib]:
        for so_file in sorted(lib_dir.glob("*.so")):
            try:
                ctypes.CDLL(str(so_file), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass  # Some .so files are plugins, not loadable directly

_preload_toolchain_libs()
```

This is the pattern used by torch-mlir and IREE's Python bindings. It works regardless of install layout because it resolves the toolchain path through the Python import system at runtime, not through filesystem-relative RPATHs. The `RTLD_GLOBAL` flag makes the loaded symbols available to subsequently loaded extensions.

### Recommended Approach

**Use Approach B (`ctypes` pre-loading) as the primary mechanism.** It is the most robust for pip-installed packages and follows established patterns in the MLIR ecosystem. The `_mlir_libs/__init__.py` already exists in the current codebase (it hosts the site initialization chain); extending it with pre-loading logic is a natural fit. The current `LD_LIBRARY_PATH`-based approach (via `env/activate`) is unsuitable for pip-installed packages because it requires manual environment setup and conflicts with other packages.

For editable installs (`pip install -e .`), the same mechanism works because `ttl_toolchain` is still a regular pip-installed package with a fixed `site-packages` location.

## Wheel Metadata

```toml
# Metadata in the built wheel (not a source file -- generated by setuptools)
Metadata-Version: 2.4
Name: ttl
Version: 0.1.250413
Requires-Dist: ttl-toolchain ==0.1.250413
Requires-Dist: pydantic <3
Requires-Dist: torch >=1.9.0
Requires-Dist: numpy >=1.20.0
Requires-Dist: greenlet >=3.0
```

The `ttl-toolchain` pin uses `==` (exact match on the base version) to ensure ABI compatibility. The `ttl` wheel and `ttl-toolchain` wheel share the same base version string; they are always released in lockstep.

## Editable Install Workflow

For the editable-install workflow (including first-time setup and day-to-day iteration), see the [Developer Workflow](./build_pipeline.md#developer-workflow) section in `build_pipeline.md`.

**Prev:** [`toolchain_wheel_design.md`](./toolchain_wheel_design.md) | **Next:** [`build_pipeline.md`](./build_pipeline.md)
