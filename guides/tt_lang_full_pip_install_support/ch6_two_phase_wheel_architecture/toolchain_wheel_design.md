# Toolchain Wheel Design (`ttl-toolchain`)

This file specifies the contents, package layout, versioning scheme, and size management strategy for the `ttl-toolchain` wheel -- the platform-specific binary wheel that ships pre-built LLVM/MLIR and tt-metal artifacts.

## What Goes Into the Wheel

The toolchain wheel bundles everything that the existing `TTLANG_TOOLCHAIN_DIR` directory contains (as described in [Chapter 5](../ch5_pip_install_with_toolchain/index.md)), repackaged into a pip-installable layout. The contents fall into four categories: (1) LLVM/MLIR shared libraries and bindings (from `BuildLLVM.cmake`), (2) tt-metal shared libraries and Python packages (from `BuildTTMetal.cmake`), (3) tt-mlir CAPI libraries and headers (from `lib/ttmlir-minimal/` and `lib/CAPI/`), and (4) build support files.

## Package Layout

The wheel installs into `site-packages/` with the following annotated structure:

```
site-packages/
  ttl_toolchain/
    __init__.py                    # Package entry point: get_toolchain_dir(), __version__
    mlir/                          # (1) LLVM/MLIR — outputs of BuildLLVM.cmake
      lib/
        libMLIRIR.so               # Core MLIR infrastructure (IR, passes, transforms)
        libMLIRPass.so             #   ~120 shared libraries total
        libMLIRTransforms.so
        libLLVMSupport.so          # LLVM support libraries (ADT, Support, TableGen runtime)
        libLLVMDemangle.so
        ...
      lib/cmake/
        mlir/MLIRConfig.cmake      # find_package(MLIR) entry point [build-time only]
        llvm/LLVMConfig.cmake      # Transitively required by MLIRConfig [build-time only]
      include/                     # [build-time only] Headers for nanobind extension compilation
        mlir/**/*.h
        mlir-c/**/*.h
        llvm/**/*.h
        llvm-c/**/*.h
      bin/                         # [build-time only] Dialect code generation
        mlir-tblgen
        llvm-tblgen
      python/
        mlir/                      # Upstream MLIR Python bindings (mlir.ir, mlir.dialects, etc.)
    ttmetal/                       # (2) tt-metal — outputs of BuildTTMetal.cmake
      lib/
        libtt_metal.so             # Core tt-metal runtime
        libdevice.so               # Device management library
        _ttnn.so                   # TTNN Python C extension
        _ttnncpp.so
      python_packages/
        ttnn/                      # `import ttnn`
        tools/                     # Runtime utilities
      runtime/                     # JIT kernel compilation artifacts
    ttmlir/                        # (3) tt-mlir CAPI — for linking _ttmlir and _ttlang extensions
      lib/
        libTTMLIRMinimalCAPI.a     # Static CAPI for tt-mlir dialects [build-time only]
        libTTLangCAPI.a            # Static CAPI for TTL dialect [build-time only]
      include/                     # [build-time only]
        ttmlir/**/*.h
        ttmlir/**/*.h.inc          # Pre-generated TableGen outputs (TTCore, TTKernel, TTMetal)
    cmake/
      TTLToolchainConfig.cmake     # Wrapper CMake config for find_package(TTLToolchain)
```

### The `__init__.py` Entry Point

```python
"""ttl-toolchain: pre-built LLVM/MLIR and tt-metal for TT-Lang."""

import pathlib

__version__ = "0.1.250413"

def get_toolchain_dir() -> pathlib.Path:
    """Return the root of the installed toolchain tree."""
    return pathlib.Path(__file__).parent

def get_mlir_cmake_dir() -> str:
    """Return the path to MLIRConfig.cmake for find_package(MLIR)."""
    return str(get_toolchain_dir() / "mlir" / "lib" / "cmake" / "mlir")

def get_llvm_cmake_dir() -> str:
    """Return the path to LLVMConfig.cmake for find_package(LLVM)."""
    return str(get_toolchain_dir() / "mlir" / "lib" / "cmake" / "llvm")

def get_ttmetal_dir() -> str:
    """Return the path to tt-metal artifacts."""
    return str(get_toolchain_dir() / "ttmetal")
```

This module allows the `ttl` build to locate the toolchain without environment variables:

```python
# In ttl's setup.py or CMake integration:
import ttl_toolchain
cmake_args.append(f"-DMLIR_DIR={ttl_toolchain.get_mlir_cmake_dir()}")
```

### The CMake Config Wrapper

`TTLToolchainConfig.cmake` is a thin wrapper that sets the variables the TT-Lang CMake build expects:

```cmake
# TTLToolchainConfig.cmake -- installed by the ttl-toolchain wheel.
# Usage: find_package(TTLToolchain REQUIRED)

get_filename_component(_TTL_TC_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

set(TTLANG_TOOLCHAIN_DIR "${_TTL_TC_ROOT}" CACHE PATH
  "ttl-toolchain install root (from pip)" FORCE)
set(TTLANG_USE_TOOLCHAIN ON CACHE BOOL
  "Use pre-built LLVM from ttl-toolchain" FORCE)
set(MLIR_DIR "${_TTL_TC_ROOT}/mlir/lib/cmake/mlir" CACHE PATH
  "MLIR CMake dir" FORCE)

# tt-metal paths
set(TT_METAL_HOME "${_TTL_TC_ROOT}/ttmetal" CACHE PATH
  "tt-metal root" FORCE)
set(TT_METAL_LIB_PATH "${_TTL_TC_ROOT}/ttmetal/lib" CACHE PATH
  "tt-metal library path" FORCE)

# tt-mlir CAPI paths (for linking _ttmlir and _ttlang extensions)
set(TTMLIR_CAPI_LIB_DIR "${_TTL_TC_ROOT}/ttmlir/lib" CACHE PATH
  "tt-mlir CAPI library dir" FORCE)
set(TTMLIR_INCLUDE_DIR "${_TTL_TC_ROOT}/ttmlir/include" CACHE PATH
  "tt-mlir headers" FORCE)

unset(_TTL_TC_ROOT)
```

## Version Pinning

The toolchain version must uniquely identify the combination of LLVM and tt-metal sources it was built from. The scheme uses the TT-Lang release version plus the short SHAs of both submodules:

```
ttl-toolchain==0.1.250413+llvm.abc1234.ttm.def5678
```

Where:
- `0.1.250413` -- TT-Lang base version (matches the `ttl` wheel version epoch)
- `llvm.abc1234` -- first 7 characters of the LLVM submodule SHA (read by `cmake/modules/BuildLLVM.cmake` line 23 via `ttlang_get_submodule_sha`)
- `ttm.def5678` -- first 7 characters of the tt-metal submodule SHA (read by `cmake/modules/BuildTTMetal.cmake` line 115)

The `+` local version separator is used because these SHAs are build metadata, not release identifiers. For PyPI uploads, the local segment is dropped and the version is pinned by the base version alone. For internal indexes (e.g., a Tenstorrent-hosted simple repository), the full local version provides traceability.

### Version Verification at Import Time

The `ttl` package verifies at import time that the installed `ttl-toolchain` version is compatible:

```python
# ttl/__init__.py (excerpt)
import importlib.metadata

_tc_version = importlib.metadata.version("ttl-toolchain")
if not _tc_version.startswith("0.1.250413"):
    import warnings
    warnings.warn(
        f"ttl-toolchain {_tc_version} may be incompatible with ttl {__version__}",
        stacklevel=2,
    )
```

## Size Considerations

The uncompressed toolchain contents are large:

| Component | Unstripped | Stripped | Notes |
|-----------|-----------|---------|-------|
| LLVM/MLIR shared libs | ~2.4 GB | ~800 MB | `strip --strip-unneeded` on all `.so` files |
| LLVM/MLIR headers + CMake | ~180 MB | ~180 MB | Cannot strip; needed at `ttl` build time |
| MLIR Python bindings | ~15 MB | ~15 MB | Pure Python + small `.so` stubs |
| tt-metal shared libs | ~1.2 GB | ~500 MB | Stripping removes debug info |
| tt-metal Python packages | ~30 MB | ~30 MB | Pure Python |
| tt-mlir CAPI + headers | ~50 MB | ~20 MB | Static libs strip well |
| **Total** | **~3.9 GB** | **~1.55 GB** | |

### Reduction Strategies

To bring the wheel to a manageable size:

1. **Strip all shared libraries.** Apply `strip --strip-unneeded` to every `.so` before packaging. This is the single biggest win, cutting LLVM libraries from 2.4 GB to ~800 MB.

2. **Selective LLVM inclusion.** The full LLVM install includes libraries for targets and features TT-Lang does not use. The `BuildLLVM.cmake` configuration already limits the build to `-DLLVM_TARGETS_TO_BUILD=host` and `-DLLVM_ENABLE_PROJECTS=mlir`, but the install still includes ~120 shared libraries. A post-build pruning step can remove libraries that are not in the transitive dependency closure of `libMLIRIR.so`, `libMLIRPass.so`, and the dialects used by tt-mlir. Estimated savings: 100--200 MB.

3. **Exclude build-only artifacts from the runtime wheel.** Headers, CMake configs, TableGen binaries, and static CAPI libraries are only needed when building `ttl` from source. They are not needed when a pre-built `ttl` wheel is installed. Two approaches:
   - **Single wheel, accept the size.** Include everything; the ~180 MB of headers is a small fraction of the total.
   - **Three wheels** (`ttl-toolchain-runtime`, `ttl-toolchain-dev`, `ttl`). This adds complexity for modest savings and is not recommended for the initial implementation.

4. **Wheel compression.** Zip compression within the `.whl` format provides ~50-60% compression on shared libraries. The 1.55 GB stripped total compresses to approximately **600--700 MB** as a wheel file.

5. **Split LLVM into a shared `libLLVM.so`.** Building LLVM with `-DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON` produces a single `libLLVM-19.so` (~150 MB stripped) instead of ~60 individual LLVM component libraries. MLIR libraries then link against this single shared library. This can reduce the LLVM portion by 200--300 MB due to eliminated cross-library symbol duplication. However, it changes the ABI surface and requires testing with the MLIR Python bindings.

### Size Budget

| Scenario | Estimated Wheel Size |
|----------|---------------------|
| Stripped, all libraries included | ~600-700 MB |
| Stripped + selective LLVM pruning | ~500-600 MB |
| Stripped + `libLLVM.so` dylib mode | ~400-500 MB |

For comparison, the PyTorch nightly wheel (`torch`) is ~800 MB for CUDA builds. A 400--600 MB toolchain wheel is large but within the range that internal package indexes and `pip install --index-url` can handle. PyPI's 100 MB limit means the toolchain wheel would be hosted on a Tenstorrent-managed index (e.g., AWS CodeArtifact or a simple S3-backed repository), not on pypi.org directly.

## Build Script

The toolchain wheel is built by a dedicated script that wraps the existing CMake toolchain build:

```bash
#!/usr/bin/env bash
# scripts/build-toolchain-wheel.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build-toolchain"
INSTALL_DIR="${BUILD_DIR}/install"
WHEEL_DIR="${BUILD_DIR}/wheelhouse"

# Phase 1: Build LLVM + tt-metal into a toolchain directory
cmake -G Ninja -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
  -DTTLANG_BUILD_TOOLCHAIN=ON \
  -DTTLANG_TOOLCHAIN_DIR="${INSTALL_DIR}" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "${BUILD_DIR}"

# Phase 2: Strip shared libraries
find "${INSTALL_DIR}" -name '*.so' -exec strip --strip-unneeded {} +

# Phase 3: Package as wheel
python -m ttl_toolchain_builder \
  --toolchain-dir "${INSTALL_DIR}" \
  --output-dir "${WHEEL_DIR}" \
  --version "$(cat "${REPO_ROOT}/VERSION")"
```

The `ttl_toolchain_builder` module (a small build helper, not shipped in the wheel) takes the install directory, copies artifacts into the `ttl_toolchain/` package layout, generates `__init__.py` with the correct version, and invokes `wheel pack` to produce the `.whl` file.

**Prev:** [`index.md`](./index.md) | **Next:** [`main_wheel_design.md`](./main_wheel_design.md)
