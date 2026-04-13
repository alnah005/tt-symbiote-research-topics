# MLIR Dependency Chain

This file traces the full chain of MLIR CMake infrastructure, LLVM/MLIR
libraries, tt-mlir CAPI layers, and TableGen-generated artifacts that must be
available before the `_ttlang` and `_ttmlir` nanobind extensions can compile.

---

## MLIR CMake Infrastructure Used

TT-Lang relies heavily on MLIR's Python-binding CMake helpers, all provided by
the `AddMLIRPython` module (included at line 1 of `python/CMakeLists.txt`).

### `declare_mlir_python_sources`

Registers a named group of `.py` files that will be copied into the final
package tree.  Used for upstream MLIR re-exports, dialect bindings, site-init
scripts, and the TTL runtime library:

- `TTMLIRMinPythonSources` -- tt-mlir dialect `.py` files
- `TTLangPythonSources` -- TTL dialect `.py` files
- `TTLangPythonCommon` -- runtime library (`ttl_api.py`, `layouts.py`, etc.)

### `declare_mlir_dialect_python_bindings`

A specialization that runs `mlir-tblgen -gen-python-op-bindings` on a `.td`
file and declares the resulting `.py` as a source.  TT-Lang uses it three
times:

| Dialect | TD file | Enum TD file | Output | TableGen stage |
|---------|---------|-------------|--------|----------------|
| `ttcore` | `dialects/TTCoreBinding.td` | `dialects/TTCoreEnumBinding.td` | `dialects/ttcore.py` | `mlir-tblgen -gen-python-op-bindings` |
| `ttkernel` | `dialects/TTKernelBinding.td` | `dialects/TTKernelEnumBinding.td` | `dialects/ttkernel.py` | `mlir-tblgen -gen-python-op-bindings` |
| `ttl` | `dialects/TTLBinding.td` | `dialects/TTLEnumBinding.td` | `dialects/ttl.py` | `mlir-tblgen -gen-python-op-bindings` |

Each row triggers a separate TableGen invocation that produces both the op
bindings `.py` and the corresponding enum bindings from the `*EnumBinding.td`
file.

### `declare_mlir_python_extension`

Declares a nanobind C++ extension module.  Each call specifies:

- `MODULE_NAME` -- the importable name (`_ttmlir` or `_ttlang`)
- `PYTHON_BINDINGS_LIBRARY nanobind` -- selects nanobind over pybind11
- `SOURCES` -- the `.cpp` files implementing the module
- `EMBED_CAPI_LINK_LIBS` -- which CAPI libraries provide the needed symbols

### `add_mlir_python_common_capi_library`

Builds the single shared CAPI library (`TTLangPythonCAPI`) that all extensions
link against.  This avoids duplicate MLIR symbols and ensures a single
`MLIRContext` instance at runtime.

### `add_mlir_python_modules`

The final aggregation command.  Combines all declared sources and extensions
into a staged Python package under `ROOT_PREFIX`, linking extensions against the
common CAPI library via `COMMON_CAPI_LINK_LIBS`.

---

## LLVM/MLIR Libraries Linked

### Upstream MLIR (via `MLIRPythonExtension.RegisterEverything`)

The `RegisterEverything` source group pulls in registration for all upstream
MLIR dialects (func, arith, scf, memref, tensor, bufferization, emitc, quant,
etc.) and their transforms.  Key libraries transitively linked into
`TTLangPythonCAPI`:

| Library | Role |
|---------|------|
| `MLIRCAPIIR` | Core IR CAPI (context, module, operation, type, attribute) |
| `MLIRCAPITransforms` | Pass-manager CAPI (run passes, register passes) |
| `MLIRIR` | C++ core IR library |
| `MLIRSupport` | String refs, diagnostics, threading |
| `MLIRPass` | Pass infrastructure |
| `MLIRFuncDialect` | `func.func` / `func.return` |
| `MLIRSCFDialect` | `scf.for` / `scf.if` / `scf.while` |
| `MLIRArithToEmitC` | Arith-to-EmitC lowering (used by TTKernelToEmitC) |
| `MLIRMemRefToEmitC` | MemRef-to-EmitC lowering |
| `MLIRSCFToEmitC` | SCF-to-EmitC lowering |
| `MLIREmitCDialect` | EmitC ops (target of kernel lowering) |
| `MLIRTargetCpp` | C++ code emission from EmitC |
| `MLIRQuantDialect` | Quantization types (TTCore dependency) |
| `MLIRTensorDialect` | Tensor ops (TTMetal dependency) |
| `MLIRBufferizationTransforms` | One-shot bufferize (TTCore transforms) |
| `MLIRTransformUtils` | Transform pass utilities |

### tt-mlir Dialect Libraries (built by `lib/ttmlir-minimal/CMakeLists.txt`)

These are compiled directly from `third-party/tt-mlir` sources:

| Library | Source dialect | Links to |
|---------|--------------|----------|
| `MLIRTTCoreDialect` | TTCore IR | `MLIRQuantDialect` |
| `MLIRTTTransforms` | TTCore Transforms | `MLIRBufferizationTransforms`, `MLIRTTCoreDialect` |
| `MLIRTTMetalDialect` | TTMetal IR | `MLIRTTCoreDialect`, `MLIRSCFDialect`, `MLIRTensorDialect` |
| `MLIRTTKernelDialect` | TTKernel IR | `MLIRTTMetalDialect`, `MLIRTTCoreDialect`, `MLIRSCFDialect` |
| `MLIRTTKernelTransforms` | TTKernel Transforms | `MLIRTTKernelDialect` |
| `TTMLIRTTKernelToEmitC` | TTKernelToEmitC conversion | `MLIRArithToEmitC`, `MLIRMemRefToEmitC`, `MLIRSCFToEmitC`, `MLIREmitCDialect`, `MLIRTargetCpp`, `MLIRTTKernelDialect` |
| `TTKernelTargetCpp` | TTKernel-to-C++ translation | `MLIRTTKernelDialect`, `MLIRTTMetalDialect`, `MLIRTTCoreDialect`, `MLIRTargetCpp` |

---

## tt-mlir CAPI Libraries

For the high-level dependency tree and link-time resolution of these libraries,
see [index.md -- CAPI Library Dependency Summary](./index.md#capi-library-dependency-summary).
This section documents only the source-file-level details.

### `TTMLIRMinimalCAPI`

Defined in `lib/ttmlir-minimal/CAPI/CMakeLists.txt` using
`add_mlir_public_c_api_library()`.  Sources:

- `lib/ttmlir-minimal/CAPI/Dialects.cpp` -- dialect registration
  (`MLIR_DEFINE_CAPI_DIALECT_REGISTRATION` for TT, TTKernel, TTMetal),
  `ttmlirMinimalRegisterAllDialects()`, `ttmlirMinimalRegisterPasses()`,
  `ttmlirMinimalRunTTKernelToEmitC()`, `ttmlirMinimalTranslateKernelToCpp()`
- `third-party/tt-mlir/lib/CAPI/TTCoreAttrs.cpp` -- TTCore attribute accessors
- `third-party/tt-mlir/lib/CAPI/TTKernelTypes.cpp` -- TTKernel type accessors

### `TTLangCAPI`

Defined in `lib/CAPI/CMakeLists.txt` using `add_mlir_library()` with
`ENABLE_AGGREGATION`.  Sources:

- `lib/CAPI/Dialects.cpp` -- TTL dialect registration
  (`MLIR_DEFINE_CAPI_DIALECT_REGISTRATION` for TTL),
  `ttlangRegisterAllDialects()`, `ttlangRegisterTTLDialect()`,
  `ttlangRegisterPasses()`
- `lib/CAPI/TTLAttrs.cpp` -- TTL attribute accessors

Depends on `MLIRTTLOpsAttributesIncGen` (a TableGen-generated artifact from
TT-Lang's own TTL dialect).

---

## TableGen-Generated Artifacts Required Before Compilation

TableGen `.inc` files must be generated before any C++ source that includes the
corresponding dialect headers can compile.  The dependency chain involves two
separate sets of TableGen runs.

### tt-mlir TableGen (via `BuildTTMLIRMinimal.cmake`)

`BuildTTMLIRMinimal.cmake` processes tt-mlir's `include/` directories using
`add_subdirectory()`, which invokes `mlir_tablegen()` in each dialect's
`CMakeLists.txt`.  Generated `.inc` files land under
`${CMAKE_BINARY_DIR}/include/ttmlir/`:

| CMake target | Generated artifacts |
|-------------|-------------------|
| `MLIRTTCoreOpsIncGen` | `TTCoreOps.h.inc`, `TTCoreOps.cpp.inc`, `TTCoreDialect.h.inc`, `TTCoreDialect.cpp.inc` |
| `MLIRTTCoreOpsEnumsIncGen` | `TTCoreOpsEnums.h.inc`, `TTCoreOpsEnums.cpp.inc` |
| `MLIRTTCoreOpsAttributesIncGen` | `TTCoreOpsAttributes.h.inc`, `TTCoreOpsAttributes.cpp.inc` |
| `TTCoreAttrInterfacesIncGen` | TTCore attribute interface `.inc` files |
| `MLIRTTCorePassesIncGen` | `TTCorePasses.h.inc`, `TTCorePasses.cpp.inc` |
| `MLIRTTMetalOpsIncGen` | TTMetal ops `.inc` files |
| `MLIRTTMetalOpsEnumsIncGen` | TTMetal enums `.inc` files |
| `MLIRTTMetalOpsAttributesIncGen` | TTMetal attributes `.inc` files |
| `TTMetalAttrInterfacesIncGen` | TTMetal attribute interface `.inc` files |
| `MLIRTTKernelOpsIncGen` | TTKernel ops `.inc` files |
| `MLIRTTKernelOpsEnumsIncGen` | TTKernel enums `.inc` files |
| `MLIRTTKernelOpsAttributesIncGen` | TTKernel attributes `.inc` files |
| `MLIRTTKernelPassesIncGen` | TTKernel passes `.inc` files |
| `TTMLIRConversionPassIncGen` | Conversion passes `.inc` files |

Additionally, `TTKernelGeneratedLLKHeaders` generates `*_generated.h` files
from LLK header sources using `GenerateRawStringHeader.cmake`.  These embed
LLK C++ headers as raw string literals for the TTKernelToCpp translation.

### TT-Lang TableGen (TTL dialect)

TT-Lang's own TTL dialect has its own set of TableGen-generated artifacts
(processed from `include/ttlang/Dialect/TTL/`):

| CMake target | Purpose |
|-------------|---------|
| `MLIRTTLOpsIncGen` | TTL ops `.inc` files |
| `MLIRTTLOpsAttributesIncGen` | TTL attributes `.inc` files (required by `TTLangCAPI`) |
| `MLIRTTLOpsEnumsIncGen` | TTL enums `.inc` files |

---

## Compile Definitions and Nanobind Domain

See [index.md -- The Shared Nanobind Domain and The Package Prefix](./index.md#the-shared-nanobind-domain) for full details on `MLIR_BINDINGS_PYTHON_NB_DOMAIN` and `MLIR_PYTHON_PACKAGE_PREFIX`.

---

## Summary Dependency Graph

```
                    AddMLIRPython (CMake module)
                           |
            +--------------+--------------+
            |              |              |
  declare_mlir_       declare_mlir_    add_mlir_python_
  python_sources      python_extension common_capi_library
            |              |              |
            |         +---------+    TTLangPythonCAPI
            |         |         |         |
            |      _ttmlir   _ttlang      |
            |         |         |         |
            |         +----+----+         |
            |              |              |
            +---------+    |    +---------+
                      |    |    |
               add_mlir_python_modules
                      |
               TTLangPythonModules
                      |
          staged ttl/ package tree
```

**Next:** [`discovery_mechanisms.md`](./discovery_mechanisms.md)
