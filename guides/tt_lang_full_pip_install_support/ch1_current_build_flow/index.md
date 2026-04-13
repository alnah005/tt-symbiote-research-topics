# Chapter 1 -- Current Build and Installation Flow

This chapter documents the end-to-end build pipeline for TT-Lang as it exists today. Understanding this pipeline is essential context for designing a proper `pip install` experience, because every phase described here currently happens either at CMake configure time or as a post-configure shell step -- none of it is integrated with Python packaging.

## The Five Major Phases

A full TT-Lang build proceeds through five sequential phases:

| Phase | What happens | Orchestrated by |
|-------|-------------|-----------------|
| **1. Configure** | CMake configures the project, creates a Python venv, installs pip requirements, and triggers configure-time builds of LLVM and tt-metal via `execute_process` | `cmake -G Ninja -B build ...` |
| **2. LLVM Build** | LLVM/MLIR is either located from a pre-built toolchain (`TTLANG_USE_TOOLCHAIN=ON`) or built from the `third-party/llvm-project` submodule. | `BuildLLVM.cmake` |
| **3. tt-metal Build** | tt-metal is built from `third-party/tt-metal` at configure time. Nested submodules (tracy, tt_llk, umd) are initialized automatically. The sentinel `_ttnn.so` is used to skip rebuilds. | `BuildTTMetal.cmake` |
| **4. tt-lang Build** | The actual Ninja build: MLIR TableGen processing of tt-mlir dialect `.td` files, C++ dialect libraries, Python bindings, and compiler tools | `cmake --build build` |
| **5. Install/Finalize** | Artifacts are installed into `TTLANG_TOOLCHAIN_DIR`, the toolchain is normalized and cleaned up, and optionally the build directory is removed | `cmake --install` + `do_finalize()` |

The critical architectural detail is that phases 2 and 3 are **configure-time builds** -- they run inside `cmake -B build`, not during `cmake --build build`. This means the CMake configure step itself can take 30-60 minutes on a cold build.

## The `scripts/build-and-install.sh` Driver

The shell script `scripts/build-and-install.sh` orchestrates these phases. It accepts mutually exclusive mode flags that control which subset of the pipeline runs:

| Flag | Phases executed | Use case |
|------|----------------|----------|
| *(default, no flag)* | All five: configure + install-ttmetal + build + install + finalize | Full from-scratch build |
| `--toolchain-only` | Configure + install-ttmetal + finalize (no tt-lang build) | Building a reusable toolchain for CI |
| `--configure-only` | Configure only (build dirs preserved) | First stage of multi-stage Docker build |
| `--install-ttmetal` | Install tt-metal artifacts into toolchain | Second stage of multi-stage Docker build |
| `--build-and-install` | Build tt-lang + install (assumes configure already ran) | Third stage after configure |
| `--finalize` | Normalize toolchain + cleanup | Final stage; optionally removes build dir |
| `--test-toolchain` | Build in a fresh directory using installed toolchain, run tests | Validation of a toolchain install |

Additional option flags:

- `--force-rebuild` -- Forces LLVM and tt-metal rebuild even if cached artifacts exist
- `--remove-build-dir` -- Removes `CMAKE_BINARY_DIR` after finalize (for Docker builds where disk space is limited)

### Typical multi-stage usage

The script documents a multi-stage workflow for Docker image construction:

```bash
# Stage 1: Build LLVM + tt-metal (expensive, cached)
build-and-install.sh --configure-only

# Stage 2: Install tt-metal into toolchain prefix
build-and-install.sh --install-ttmetal

# Stage 3: Build tt-lang compiler and Python bindings
build-and-install.sh --build-and-install

# Stage 4: Clean up build artifacts, normalize paths
build-and-install.sh --finalize --remove-build-dir
```

### Configure function internals

The `do_configure()` function auto-detects a pre-built toolchain by checking for `MLIRConfig.cmake` in `TTLANG_TOOLCHAIN_DIR` (see [`cmake_architecture.md`](./cmake_architecture.md) for the CMake-level rebuild skip logic). After CMake configure completes, the function sources the generated `env/activate` script and installs Python runtime dependencies into the toolchain venv.

## Chapter Contents

- [`cmake_architecture.md`](./cmake_architecture.md) -- Root `CMakeLists.txt` structure, and detailed walkthrough of each CMake module (`BuildLLVM.cmake`, `BuildTTMLIRMinimal.cmake`, `BuildTTMetal.cmake`, `TTLangPython.cmake`, `TTLangCompilerSetup.cmake`)
- [`environment_assumptions.md`](./environment_assumptions.md) -- Environment variables, pre-installed tool requirements, the `env/activate.in` template, and why `source env/activate` is required after every build

---

**Next:** [`cmake_architecture.md`](./cmake_architecture.md)
