# Case Studies

This file examines how four MLIR-based projects package their Python bindings into distributable wheels. Each case study covers the build backend, LLVM integration strategy, wheel structure, and platform support.

---

## torch-mlir

**Repository:** [llvm/torch-mlir](https://github.com/llvm/torch-mlir)

### Build Backend

torch-mlir uses `setuptools` with a custom `CMakeBuild` command class in `setup.py`. The build system defines several custom setuptools command overrides:

- **`CMakeBuild`** -- invokes CMake to configure and compile the project, then copies artifacts into the wheel staging area.
- **`CustomBuild`** -- redirects the build directory to `setup_build/` to avoid collision with CMake's standard `build/` directory.
- **`CMakeExtension`** -- a placeholder extension that triggers the CMake build without running the default `build_ext` logic.

Key environment variables control the build:

```bash
# Point to a pre-built LLVM installation for out-of-tree builds
export LLVM_INSTALL_DIR=/path/to/llvm-install

# Skip CMake compilation when artifacts already exist
export TORCH_MLIR_CMAKE_ALREADY_BUILT=1
export TORCH_MLIR_CMAKE_BUILD_DIR=build/

# Build the wheel
python setup.py bdist_wheel
```

### LLVM Strategy

torch-mlir supports three build modes:

| Mode | LLVM source | Typical use |
|------|------------|-------------|
| In-tree | Bundled via `externals/llvm-project` submodule | CI wheel builds |
| Out-of-tree | Pre-built LLVM pointed to by `-DLLVM_DIR` / `-DMLIR_DIR` | Developer iteration |
| Wheel | Either of the above, orchestrated by `setup.py` | PyPI publishing |

For CI wheel builds, LLVM is compiled from the in-tree submodule as part of the same CMake invocation. The `CMakeLists.txt` detects an in-tree build when `CMAKE_SOURCE_DIR` matches `CMAKE_CURRENT_SOURCE_DIR` and registers torch-mlir as an `LLVM_EXTERNAL_PROJECTS` target.

### Wheel Structure and Size

torch-mlir ships two distinct wheel variants:

- **`torch-mlir`** -- pure MLIR Python bindings (built when `TORCH_MLIR_ENABLE_ONLY_MLIR_PYTHON_BINDINGS=1`).
- **`torch-mlir-ext`** -- full package including PyTorch native extensions, requiring a matching PyTorch version.

Wheel sizes on PyPI (version 20221213.686):

| Platform | Size |
|----------|------|
| Linux manylinux x86-64 | ~222 MB |
| macOS universal2 | ~181 MB |
| Windows x86-64 | ~23 MB |

### Platform Support and Wheel Repair

Linux wheels are built inside `manylinux_2_28` Docker containers. After compilation, `auditwheel repair` bundles dynamic libraries (excluding PyTorch's own `.so` files) to produce self-contained manylinux-compatible wheels. The build script is `build_tools/python_deploy/build_linux_packages.sh`.

macOS wheels use `delocate-wheel` for the equivalent library bundling. Windows uses `delvewheel`, explicitly excluding PyTorch DLLs (`c10.dll`, `torch_python.dll`, `torch_cpu.dll`).

Python versions: CPython 3.10, 3.11, 3.12 (configurable via `TM_PYTHON_VERSIONS`).

---

## Triton (OpenAI / triton-lang)

**Repository:** [triton-lang/triton](https://github.com/triton-lang/triton)

### Build Backend

Triton uses `setuptools` with a custom CMake extension in `python/setup.py`. When a user runs `pip install`, setuptools invokes CMake with `-DTRITON_BUILD_PYTHON_MODULE=ON`, which compiles C++ extensions as shared libraries placed in `python/triton/_C/`.

The C++ to Python interface is implemented via **pybind11**, listed as a build-time dependency in `python/requirements.txt`. The resulting `triton._C.libtriton` module exposes IR construction and pass management to Python.

### LLVM Strategy

Triton pins a specific LLVM commit in `cmake/llvm-hash.txt`. The LLVM dependency is resolved through one of two paths:

1. **Auto-download (default):** `setup.py` automatically downloads pre-built LLVM static libraries from Azure Blob Storage. These binaries are built by the `llvm-build.yml` GitHub Actions workflow for multiple platforms. No LLVM compilation is needed on the user's machine.

2. **Custom LLVM:** Developers can point to a self-built LLVM via environment variables:

```bash
export LLVM_BUILD_DIR=/path/to/llvm/build
export LLVM_INCLUDE_DIRS=/path/to/llvm/include
export LLVM_LIBRARY_DIR=/path/to/llvm/lib
pip install -e python/
```

This is the closest analogue to TT-Lang's `TTLANG_USE_TOOLCHAIN` mode -- the project pre-builds LLVM once and reuses it across many incremental `pip install` cycles.

### Build Caching

Triton implements kernel-level compilation caching at `~/.triton/cache/` (configurable via `TRITON_HOME`). Cache keys include source code hash, argument specialization, and compiler options. Cached artifacts span the full compilation pipeline: TTIR, TTGIR, LLIR, PTX/AMDGCN, and final binaries (cubin/hsaco).

For build-time caching, the `--no-build-isolation` pip flag is important: it reuses CMake symlinks across rebuilds, preventing Ninja from re-linking LLVM static libraries unnecessarily.

### Wheel Structure and Size

The built wheel contains:

- `triton/_C/libtriton.so` -- the compiled pybind11 extension
- `triton/language/`, `triton/compiler/`, `triton/runtime/` -- pure Python modules
- `triton/backends/{nvidia,amd}/` -- hardware backend modules copied from `third_party/` at build time
- LLVM's `FileCheck` utility for running lit tests

Wheel sizes (version 3.6.0):

| Platform | Size |
|----------|------|
| Linux manylinux x86-64 | ~188 MB |
| Linux manylinux aarch64 | ~176 MB |

CI uses **cibuildwheel** to build for multiple Python versions (3.10 through 3.14, including free-threaded 3.13t and 3.14t), then applies `auditwheel repair` for manylinux compliance.

---

## IREE

**Repository:** [iree-org/iree](https://github.com/iree-org/iree)

### Build Backend

IREE uses `setuptools` with CMake, but unlike the other projects it maintains **separate `setup.py` files** for the compiler and runtime packages:

- `compiler/setup.py` -- builds and packages `iree-base-compiler`
- `runtime/setup.py` -- builds and packages `iree-base-runtime`

Both scripts use Ninja as the CMake generator and support editable installs via `CMAKE_INSTALL_MODE=ABS_SYMLINK`. The runtime build is relatively fast; the compiler build is heavier due to LLVM/MLIR inclusion.

Key environment variables:

```bash
# Custom build directory
export IREE_RUNTIME_API_CMAKE_BUILD_DIR=build/runtime

# Build type
export IREE_CMAKE_BUILD_TYPE=Release
```

### LLVM Strategy

LLVM is pre-built in CI as part of the compiler wheel build. The CMake configuration disables everything except what the compiler needs (`-DIREE_BUILD_COMPILER=ON -DIREE_BUILD_SAMPLES=OFF -DIREE_BUILD_TESTS=OFF`). Link-time optimization (LTO) is enabled by default for release builds via `IREE_RUNTIME_OPTIMIZATION_PROFILE`.

### Wheel Structure and Size

The split into two packages is IREE's most distinctive packaging decision:

| Package | Contents | Linux x86-64 size |
|---------|----------|--------------------|
| `iree-base-compiler` | MLIR/LLVM compiler stack, dialect Python bindings under `iree.compiler._mlir_libs`, command-line tools | ~83 MB |
| `iree-base-runtime` | HAL runtime, device drivers, `iree-run-module` and `iree-benchmark-executable` utilities | ~8 MB |

The compiler wheel bundles Python bindings for MLIR dialects in `iree.compiler._mlir_libs`, following the upstream MLIR Python bindings layout. The runtime wheel is lightweight and only pulls in what is needed to execute compiled models.

IREE's release wheels use the **Python Stable ABI** (`abi3`): `cp312-abi3` wheels are compatible with any standard CPython 3.12+ interpreter. Separate `cp310` and `cp311` wheels are provided for older versions, and a free-threaded `cp313t` wheel is available.

### Platform Support

| Platform | Compiler | Runtime |
|----------|----------|---------|
| Linux x86-64 (manylinux_2_27) | Yes | Yes |
| Linux aarch64 (manylinux_2_27) | Yes | Yes |
| macOS universal2 | Yes | Yes |
| Windows x86-64 | Yes | Yes |

---

## CIRCT

**Repository:** [llvm/circt](https://github.com/llvm/circt)

### Build Backend

CIRCT uses a `setup.py` script located in `lib/Bindings/Python` that orchestrates a unified CMake build of LLVM/MLIR and CIRCT together. The script accepts environment variables for customization:

```bash
# Use an existing CMake build directory (skip rebuild)
export CIRCT_CMAKE_BUILD_DIR=/path/to/build

# Point to an alternate LLVM directory
export CIRCT_LLVM_DIR=/path/to/llvm

# Build and install
pip install lib/Bindings/Python
```

The resulting wheel is named `circt_core-<version>-<python_version>-<platform>.whl`.

### LLVM Strategy

By default, the `setup.py` performs a unified CMake build that compiles LLVM/MLIR and CIRCT in a single invocation with `-DMLIR_ENABLE_BINDINGS_PYTHON=ON`. This means a from-source `pip install` triggers a full LLVM build, which is time-consuming but requires no manual setup.

Users can skip the LLVM build by setting `CIRCT_LLVM_DIR` to a pre-built LLVM installation or `CIRCT_CMAKE_BUILD_DIR` to an existing build tree.

### Distribution Model

CIRCT does **not** publish wheels to PyPI. Wheels are generated locally via `pip wheel lib/Bindings/Python` or installed directly during development. This limits CIRCT's reach compared to the other projects but simplifies its packaging story -- there is no need for `auditwheel`, `cibuildwheel`, or manylinux compliance.

Pre-built wheels are available only through project-specific CI artifacts, not through a public package index.

---

## Implications for TT-Lang

Across all four projects, four patterns emerge that map directly to TT-Lang's packaging strategy:

- **Custom `CMakeBuild` in `setup.py`** is the universal approach (torch-mlir, Triton, IREE, CIRCT). TT-Lang already uses this pattern (see [Chapter 2](../ch2_python_packaging_as_is/index.md)).
- **Pre-built LLVM with transparent download** (Triton's auto-download from Azure, torch-mlir's `LLVM_INSTALL_DIR`, IREE's CI pre-build) eliminates the largest build bottleneck. TT-Lang could host pre-built LLVM+tt-metal toolchain tarballs and download them during `pip install` when `TTLANG_USE_TOOLCHAIN` is not set.
- **Compiler/runtime wheel split** (IREE's `iree-base-compiler` + `iree-base-runtime`, torch-mlir's `torch-mlir` + `torch-mlir-ext`) maps onto a `tt-lang-compiler` wheel (LLVM + tt-mlir dialects + Python bindings) and a `tt-lang-runtime` wheel (ttnn runtime + pykernel). IREE's use of the Python Stable ABI (`abi3`) is also worth considering to reduce per-Python-version wheel builds.
- **CIRCT represents TT-Lang's current state**: a `setup.py` wrapping CMake that works locally but is not designed for public distribution. The gap between CIRCT and the PyPI-publishing projects defines the work ahead.

---

**Next:** [`lessons_learned.md`](./lessons_learned.md)
