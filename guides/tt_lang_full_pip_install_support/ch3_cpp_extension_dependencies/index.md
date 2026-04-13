# Chapter 3 -- C++ Extension Build Dependencies

This chapter catalogues the two nanobind C++ extension modules that TT-Lang
builds, the shared CAPI library that backs them, and the full dependency graph
that must be satisfied before any of these extensions can compile.

---

## Extension Modules at a Glance

TT-Lang produces two native extension shared objects that are loaded at runtime
by the `ttl` Python package.  Both use the **nanobind** binding library (not
pybind11) and share a single nanobind domain so that C types are compatible
across modules.

| Extension | CMake target | Source files | Purpose |
|-----------|-------------|--------------|---------|
| `_ttlang` | `TTLangPythonExtensions.Main` | `python/ttlang/TTLangModule.cpp`, `python/ttlang/TTLModule.cpp` | TTL dialect Python bindings -- ops, attributes, passes |
| `_ttmlir` | `TTMLIRMinPythonExtensions.Main` | `python/ttmlir/TT_MLIRMinimalExtension.cpp`, `python/ttmlir/TT_MLIRMinimalPasses.cpp`, `python/ttmlir/TT_MLIRMinimalTTModule.cpp`, `python/ttmlir/TT_MLIRMinimalTTKernelModule.cpp` | TTCore / TTKernel / TTMetal dialect bindings and pass registration |

Both extensions are declared with `declare_mlir_python_extension()` in
`python/CMakeLists.txt` and specify `PYTHON_BINDINGS_LIBRARY nanobind`.

### The Shared Nanobind Domain

Line 13 of `python/CMakeLists.txt` sets:

```cmake
set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "ttl")
```

This ensures that nanobind type objects registered by `_ttmlir` (e.g.,
`MlirModule`, `MlirContext`) are visible to `_ttlang` and vice-versa.  Without
a shared domain, passing an `MlirModule` obtained in one extension to a
function in the other would fail with a type-mismatch error at runtime.

### The Package Prefix

Line 9 of `python/CMakeLists.txt` adds a compile definition:

```cmake
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=ttl.")
```

This tells upstream MLIR's Python binding scaffolding to look for modules under
`ttl.` rather than at the top-level `mlir.` namespace.  The same pattern is
used by IREE (`iree.compiler.`) and Torch-MLIR (`torch_mlir.`).

---

## The `TTLangPythonCAPI` Shared Library

Rather than having each extension link its own copy of MLIR and dialect
libraries, TT-Lang builds a single **common CAPI shared library** called
`TTLangPythonCAPI`.  This is created by `add_mlir_python_common_capi_library()`
at line 236 of `python/CMakeLists.txt`:

```cmake
add_mlir_python_common_capi_library(TTLangPythonCAPI
  INSTALL_DESTINATION python_packages/ttl/_mlir_libs
  OUTPUT_DIRECTORY "${TTLANG_PYTHON_PACKAGES_DIR}/ttl/_mlir_libs"
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.RegisterEverything
    TTMLIRMinPythonSources
    TTMLIRMinPythonExtensions
    TTLangPythonSources
    TTLangPythonExtensions
)
```

`TTLangPythonCAPI` aggregates three layers of CAPI libraries:

1. **Upstream MLIR CAPI** -- `MLIRCAPIIR`, `MLIRCAPITransforms`, and every
   dialect registration pulled in by `MLIRPythonExtension.RegisterEverything`.
2. **`TTMLIRMinimalCAPI`** (defined in `lib/ttmlir-minimal/CAPI/CMakeLists.txt`)
   -- wraps TTCore, TTKernel, and TTMetal dialect registration, attribute
   accessors, type accessors, pass registration, and the TTKernelToCpp
   translation entry point.
3. **`TTLangCAPI`** (defined in `lib/CAPI/CMakeLists.txt`) -- wraps the TTL
   dialect registration, TTL attribute accessors, and TTL pass registration.

At link time both `_ttlang` and `_ttmlir` resolve their CAPI symbols against
`TTLangPythonCAPI` via the `COMMON_CAPI_LINK_LIBS` parameter of
`add_mlir_python_modules()` (line 276).

### CAPI Library Dependency Summary

```
TTLangPythonCAPI
  |
  +-- MLIRCAPIIR, MLIRCAPITransforms          (upstream MLIR)
  +-- MLIRPythonExtension.RegisterEverything   (upstream MLIR)
  |
  +-- TTMLIRMinimalCAPI                        (lib/ttmlir-minimal/CAPI/)
  |     +-- MLIRTTCoreDialect
  |     +-- MLIRTTTransforms
  |     +-- MLIRTTKernelDialect
  |     +-- MLIRTTKernelTransforms
  |     +-- MLIRTTMetalDialect
  |     +-- TTMLIRTTKernelToEmitC
  |     +-- TTKernelTargetCpp
  |     +-- MLIRIR, MLIRCAPITransforms, MLIRSupport
  |
  +-- TTLangCAPI                               (lib/CAPI/)
        +-- MLIRTTLDialect
        +-- TTLangTTLTransforms
        +-- MLIRCAPIIR, MLIRIR, MLIRSupport
        +-- MLIRFuncDialect
```

---

## `EMBED_CAPI_LINK_LIBS` per Extension

Each extension also declares which CAPI libraries its symbols depend on via
`EMBED_CAPI_LINK_LIBS`.  This does not cause separate linking; instead it tells
`AddMLIRPython` which symbols the extension expects the common CAPI library to
provide.

| Extension | `EMBED_CAPI_LINK_LIBS` |
|-----------|----------------------|
| `_ttmlir` | `MLIRCAPIIR`, `MLIRCAPITransforms`, `TTMLIRMinimalCAPI` |
| `_ttlang` | `TTLangCAPI` |

---

## Module Aggregation

`add_mlir_python_modules(TTLangPythonModules ...)` at line 261 is the final
aggregation step.  It takes all declared Python sources and extensions, links
them against `TTLangPythonCAPI`, and stages the complete `ttl` package tree
under `${TTLANG_PYTHON_PACKAGES_DIR}/ttl/`.

An additional dependency is wired manually:

```cmake
add_dependencies(TTLangPythonModules TTLangGeneratedElementwise)
```

This ensures the code-generated `_generated_elementwise.py` file (produced from
`include/ttlang/Dialect/TTL/TTLElementwiseOps.def`) exists before the Python
package is assembled.

---

## Further Reading

- [MLIR Dependency Chain](./mlir_dependency_chain.md) -- full breakdown of LLVM/MLIR
  libraries linked, TableGen artifacts required, and the tt-mlir CAPI layers.
- [Discovery Mechanisms](./discovery_mechanisms.md) -- how LLVM, tt-mlir, tt-metal,
  and Python are located at configure time.

**Next:** [`mlir_dependency_chain.md`](./mlir_dependency_chain.md)
