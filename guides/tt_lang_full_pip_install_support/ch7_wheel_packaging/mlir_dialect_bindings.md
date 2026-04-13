# MLIR Dialect Bindings in the Wheel

This section covers how MLIR's Python binding infrastructure generates and copies dialect files into the wheel, how the `ttl.` package prefix convention works, and which generated files must be present for the package to function.

## Background: MLIR Python Binding Infrastructure

MLIR provides a CMake-based system for declaring Python bindings for dialects. Three key macros drive this (all from `AddMLIRPython.cmake`, included at line 1 of `python/CMakeLists.txt`):

| Macro | Purpose |
|-------|---------|
| `declare_mlir_python_sources` | Declares a group of `.py` source files to include in the package |
| `declare_mlir_dialect_python_bindings` | Declares a dialect's ODS-generated Python bindings (ops + enums) |
| `add_mlir_python_modules` | Copies all declared sources into the output directory, creating the final package tree |

The final aggregation step happens at line 261:

```cmake
add_mlir_python_modules(TTLangPythonModules
  ROOT_PREFIX "${TTLANG_PYTHON_PACKAGES_DIR}/ttl"
  INSTALL_PREFIX "python_packages/ttl"
  DECLARED_SOURCES
    MLIRPythonSources                        # upstream MLIR
    MLIRPythonExtension.RegisterEverything   # upstream registration
    TTMLIRMinPythonSources                   # tt-mlir dialects
    TTMLIRMinPythonExtensions                # _ttmlir nanobind module
    TTLangPythonSources                      # TTL dialect bindings
    TTLangPythonExtensions                   # _ttlang nanobind module
    TTLangPythonCommon                       # runtime Python code
  COMMON_CAPI_LINK_LIBS
    TTLangPythonCAPI
)
```

`ROOT_PREFIX` determines the filesystem destination. Because `TTLANG_PYTHON_PACKAGES_DIR` is `${CMAKE_CURRENT_BINARY_DIR}/python_packages` (set at line 180 of the top-level `CMakeLists.txt`), everything lands under `build/python_packages/ttl/`.

## The `MLIR_PYTHON_PACKAGE_PREFIX` Convention

Line 9 of `python/CMakeLists.txt` sets:

```cmake
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=ttl.")
```

This compile definition tells the MLIR Python binding infrastructure to prefix all internal import paths with `ttl.`. Without it, MLIR's generated code would try to import from a bare `_mlir_libs` or `dialects` package at the Python root, which conflicts with other MLIR-based projects installed in the same environment.

The convention is borrowed from other MLIR-based projects:

| Project | Prefix | Top-level import |
|---------|--------|-----------------|
| IREE | `iree.compiler.` | `from iree.compiler._mlir_libs import ...` |
| torch-mlir | `torch_mlir.` | `from torch_mlir._mlir_libs import ...` |
| **TT-Lang** | `ttl.` | `from ttl._mlir_libs import ...` |

With this prefix, the ODS-generated dialect files import their C++ backends as:

```python
# Inside ttl/dialects/_ttl_ops_gen.py (auto-generated)
from ttl._mlir_libs._ttlang import ...
```

Rather than the bare `from _mlir_libs._ttlang import ...` that would be generated without the prefix.

## Dialect Source File Layout

### TTL Dialect (TT-Lang Native)

Declared at line 102 of `python/CMakeLists.txt`:

```cmake
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TTLangPythonSources.Dialects
  ROOT_DIR "${TTLANG_PYTHON_ROOT_DIR}"
  TD_FILE dialects/TTLBinding.td
  GEN_ENUM_BINDINGS ON
  GEN_ENUM_BINDINGS_TD_FILE dialects/TTLEnumBinding.td
  SOURCES dialects/ttl.py
  DIALECT_NAME ttl
)
```

This declaration does the following at build time:

1. Runs `mlir-tblgen -gen-python-op-bindings` on `dialects/TTLBinding.td` (which includes `ttlang/Dialect/TTL/IR/TTLOps.td`) to produce `dialects/_ttl_ops_gen.py`
2. Runs `mlir-tblgen -gen-python-enum-bindings` on `dialects/TTLEnumBinding.td` (which includes `ttlang/Dialect/TTL/IR/TTLOpsEnums.td`) to produce `dialects/_ttl_enum_gen.py`
3. Copies the hand-written `dialects/ttl.py` alongside the generated files

The hand-written `python/ttl/dialects/ttl.py` serves as the public API for the TTL dialect. It re-exports the generated ops and enums and provides the `ensure_dialects_registered()` helper:

```python
from ttl._mlir_libs import get_dialect_registry
from .._mlir_libs import _ttlang
from .._mlir_libs._ttlang import ttl_ir as ir
from ._ttl_enum_gen import *
from ._ttl_ops_gen import *
```

### TTCore Dialect (from tt-mlir)

Declared at line 30:

```cmake
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TTMLIRMinPythonSources.Dialects
  ROOT_DIR "${TTMLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/TTCoreBinding.td
  GEN_ENUM_BINDINGS ON
  GEN_ENUM_BINDINGS_TD_FILE dialects/TTCoreEnumBinding.td
  SOURCES dialects/ttcore.py
  DIALECT_NAME ttcore
)
```

The source `.td` and `.py` files come from `third-party/tt-mlir/python/ttmlir/dialects/`. The `ROOT_DIR` is set to `TTMLIR_PYTHON_ROOT_DIR` (line 17), which points there. After `add_mlir_python_modules` copies everything under `python_packages/ttl/`, the ttcore files end up at `ttl/dialects/ttcore.py`, `ttl/dialects/_ttcore_ops_gen.py`, and `ttl/dialects/_ttcore_enum_gen.py`.

### TTKernel Dialect (from tt-mlir)

Declared at line 40, following the same pattern as TTCore:

```cmake
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TTMLIRMinPythonSources.Dialects
  ROOT_DIR "${TTMLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/TTKernelBinding.td
  GEN_ENUM_BINDINGS ON
  GEN_ENUM_BINDINGS_TD_FILE dialects/TTKernelEnumBinding.td
  SOURCES dialects/ttkernel.py
  DIALECT_NAME ttkernel
)
```

Produces `ttl/dialects/ttkernel.py`, `_ttkernel_ops_gen.py`, and `_ttkernel_enum_gen.py`.

### Upstream MLIR Scaffolding

The `MLIRPythonSources` group (from upstream LLVM/MLIR) provides foundational files that the dialect bindings depend on:

- `ttl/dialects/__init__.py` -- the dialects package initializer
- `ttl/dialects/_ods_common.py` -- shared ODS utility functions used by all `_*_ops_gen.py` files
- `ttl/_mlir_libs/__init__.py` -- the MLIR Python libs package initializer (provides `get_dialect_registry()`)
- `ttl/ir.py`, `ttl/passmanager.py`, etc. -- upstream MLIR Python API, now available under the `ttl.` prefix

## Site Initialization Chain

MLIR's Python infrastructure uses a site-initialization mechanism to register dialects when the package is first imported. Files named `_site_initialize_*.py` in `_mlir_libs/` are discovered and executed in lexicographic order.

TT-Lang uses two site initializers:

**`_mlir_libs/_site_initialize_0.py`** (loaded first):
```python
from . import _ttmlir

def register_dialects(registry):
    _ttmlir.register_dialects(registry)
```

Registers TTCore and TTKernel dialects from the `_ttmlir` nanobind extension.

**`_mlir_libs/_site_initialize_1.py`** (loaded second):
```python
from .._mlir_libs import _ttlang

def register_dialects(registry):
    _ttlang.register_dialects(registry)
```

Registers the TTL dialect from the `_ttlang` nanobind extension.

> **Import path asymmetry:** `_site_initialize_0.py` uses `from . import _ttmlir` (a same-package import) while `_site_initialize_1.py` uses `from .._mlir_libs import _ttlang` (traverses up to `ttl/` and back down). Both forms are functionally correct -- `_ttlang` and `_ttmlir` both live in `_mlir_libs/`. The longer form in `_site_initialize_1.py` mirrors the import style used by the rest of the TT-Lang Python code and is intentional, not an error.

The numeric suffixes (`_0`, `_1`) enforce ordering: tt-mlir dialects must be registered before TTL because TTL ops may reference TTCore types.

## Generated Files

Two files are generated at build time and must be included in the wheel. They are declared in `TTLangPythonCommon.Generated` (line 189) with `ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/ttl"`, meaning they come from the build directory, not the source tree.

### `_generated_elementwise.py`

Generated by `python/gen_elementwise.py` from `include/ttlang/Dialect/TTL/TTLElementwiseOps.def`:

```cmake
set(TTL_ELEMENTWISE_DEF "${CMAKE_SOURCE_DIR}/include/ttlang/Dialect/TTL/TTLElementwiseOps.def")
set(GEN_ELEMENTWISE_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/gen_elementwise.py")
set(GENERATED_ELEMENTWISE_PY "${CMAKE_CURRENT_BINARY_DIR}/ttl/_generated_elementwise.py")

add_custom_command(
  OUTPUT ${GENERATED_ELEMENTWISE_PY}
  COMMAND ${Python3_EXECUTABLE} ${GEN_ELEMENTWISE_SCRIPT}
          ${TTL_ELEMENTWISE_DEF} -o ${GENERATED_ELEMENTWISE_PY}
  DEPENDS ${TTL_ELEMENTWISE_DEF} ${GEN_ELEMENTWISE_SCRIPT}
  ...
)
```

This file defines Python wrappers for all elementwise operations (add, mul, exp, etc.) and is imported by `ttl/__init__.py`, `ttl/operators.py`, and `ttl/ttl_math.py`. A build dependency (`add_dependencies(TTLangPythonModules TTLangGeneratedElementwise)` at line 281) ensures it is created before the Python modules target copies files into the package tree.

### `config.py`

Generated by CMake's `configure_file()` from `python/ttl/config.py.in`:

```cmake
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/ttl/config.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/ttl/config.py
  @ONLY
)
```

The template substitutes CMake variables:

```python
# Auto-generated by CMake - do not edit manually
HAS_TT_DEVICE = @TTLANG_HAS_DEVICE_INT@ == 1
VERSION = "@TTLANG_VERSION@"
```

This is how the Python package knows at runtime whether device support was compiled in and what version string to report.

## Packaging Implications

### Verifying the Wheel Contents

For a functional `ttl` wheel, the following generated/copied files are mandatory. Missing any of them causes import errors at runtime. The `cmake --install` step with `--component TTLangPythonWheel` (as invoked by `setup.py` at line 98) copies the entire `python_packages/` tree, which includes all of the below.

| File | Source | Generator |
|------|--------|-----------|
| `ttl/dialects/_ttl_ops_gen.py` | `TTLBinding.td` | `mlir-tblgen` |
| `ttl/dialects/_ttl_enum_gen.py` | `TTLEnumBinding.td` | `mlir-tblgen` |
| `ttl/dialects/_ttcore_ops_gen.py` | `TTCoreBinding.td` | `mlir-tblgen` |
| `ttl/dialects/_ttcore_enum_gen.py` | `TTCoreEnumBinding.td` | `mlir-tblgen` |
| `ttl/dialects/_ttkernel_ops_gen.py` | `TTKernelBinding.td` | `mlir-tblgen` |
| `ttl/dialects/_ttkernel_enum_gen.py` | `TTKernelEnumBinding.td` | `mlir-tblgen` |
| `ttl/_generated_elementwise.py` | `TTLElementwiseOps.def` | `gen_elementwise.py` |
| `ttl/config.py` | `config.py.in` | `cmake configure_file()` |
| `ttl/dialects/__init__.py` | upstream MLIR | `add_mlir_python_modules` copy |
| `ttl/dialects/_ods_common.py` | upstream MLIR | `add_mlir_python_modules` copy |
| `ttl/_mlir_libs/__init__.py` | upstream MLIR | `add_mlir_python_modules` copy |

After building, verify all mandatory files are present with:

```bash
$ unzip -l dist/ttl-0.1.250413-cp311-cp311-manylinux_2_28_x86_64.whl | grep -E "dialects/|_generated|config\.py|_mlir_libs/"
```

The output should include every file from the table above, plus the hand-written dialect modules (`ttl.py`, `ttcore.py`, `ttkernel.py`), the site initializers, the nanobind extensions, and `libTTLangPythonCAPI.so.20`.

If any dialect files are missing, check that the corresponding `declare_mlir_dialect_python_bindings` declaration is listed under `DECLARED_SOURCES` in the `add_mlir_python_modules(TTLangPythonModules ...)` call, either directly or through a parent source group.

**Next:** [Chapter 8 -- Sim-Only Installation Mode](../ch8_sim_only_mode/index.md)
