# TT-Lang Full pip install Support

This guide is for TT-Lang developers and build engineers who want to make the TT-Lang compiler and DSL runtime installable via standard `pip install .` workflows. It covers the current build system, Python packaging gaps, C++ extension dependencies, prior art from other MLIR projects, and concrete designs for toolchain-based pip install, two-phase wheel splitting, platform-compliant wheel packaging, and a sim-only installation mode.

---

## How to Use This Guide

| Goal | Recommended Path | Direct Links |
|------|-----------------|--------------|
| Understand the current build pipeline | Ch 1 | [Current Build Flow](ch1_current_build_flow/index.md) |
| Learn why `pip install .` fails today | Ch 1 then Ch 2 then Ch 3 | [Current Build Flow](ch1_current_build_flow/index.md), [Python Packaging As-Is](ch2_python_packaging_as_is/index.md), [C++ Extension Dependencies](ch3_cpp_extension_dependencies/index.md) |
| Evaluate packaging approaches from other MLIR projects | Ch 4 | [Prior Art](ch4_prior_art/index.md) |
| Implement pip install with a pre-built toolchain | Ch 2 then Ch 3 then Ch 5 | [Python Packaging As-Is](ch2_python_packaging_as_is/index.md), [C++ Extension Dependencies](ch3_cpp_extension_dependencies/index.md), [pip install with Toolchain](ch5_pip_install_with_toolchain/index.md) |
| Design the toolchain and main wheel split | Ch 3 then Ch 5 then Ch 6 | [C++ Extension Dependencies](ch3_cpp_extension_dependencies/index.md), [pip install with Toolchain](ch5_pip_install_with_toolchain/index.md), [Two-Phase Wheel Architecture](ch6_two_phase_wheel_architecture/index.md) |
| Ship manylinux-compliant wheels | Ch 6 then Ch 7 | [Two-Phase Wheel Architecture](ch6_two_phase_wheel_architecture/index.md), [Wheel Packaging](ch7_wheel_packaging/index.md) |
| Set up a lightweight sim-only install | Ch 8 | [Sim-Only Mode](ch8_sim_only_mode/index.md) |
| End-to-end: full pip install support from scratch | Ch 1 through Ch 8 in order | All chapters below |

---

## Chapter Index

| # | Chapter | Description | Key Concepts |
|---|---------|-------------|--------------|
| 1 | [Ch 1 --- Current Build and Installation Flow](ch1_current_build_flow/index.md) | Detailed walkthrough of the existing CMake-driven build pipeline from configure through install. | `build-and-install.sh`, `BuildLLVM.cmake`, `BuildTTMetal.cmake`, `TTLangPython.cmake`, five build phases |
| 2 | [Ch 2 --- Python Packaging As-Is](ch2_python_packaging_as_is/index.md) | How `pyproject.toml` and `python/setup.py` work together and why `pip install .` fails today. | `CMakeBuild` class, `TTLangExtension`, dynamic version, missing `MANIFEST.in`, path assumptions |
| 3 | [Ch 3 --- C++ Extension Build Dependencies](ch3_cpp_extension_dependencies/index.md) | LLVM/MLIR libraries, tt-mlir artifacts, and tt-metal components required before nanobind modules compile. | `_ttlang`, `_ttmlir`, `TTLangPythonCAPI`, `AddMLIRPython`, TableGen artifacts, `MLIR_PYTHON_PACKAGE_PREFIX` |
| 4 | [Ch 4 --- Prior Art](ch4_prior_art/index.md) | How torch-mlir, Triton, IREE, and CIRCT handle `pip install` with heavy C++ toolchain dependencies. | scikit-build-core, toolchain wheel pattern, `auditwheel`, LLVM caching strategies |
| 5 | [Ch 5 --- pip install with Pre-Built Toolchain](ch5_pip_install_with_toolchain/index.md) | Concrete changes to support `TTLANG_TOOLCHAIN_DIR=/path pip install .` using a pre-built toolchain. | `TTLANG_USE_TOOLCHAIN`, `pyproject.toml` changes, `setup.py` fixes, CMake variable passthrough |
| 6 | [Ch 6 --- Two-Phase Wheel Architecture](ch6_two_phase_wheel_architecture/index.md) | Splitting distribution into a `ttl-toolchain` wheel (LLVM/tt-metal) and a `ttl` main wheel. | `ttl-toolchain`, `ttl`, version pinning, CI pipeline, editable install workflow |
| 7 | [Ch 7 --- Wheel Packaging and Platform Compliance](ch7_wheel_packaging/index.md) | Bundling nanobind extensions in wheels, RPATH handling, `auditwheel`, and MLIR dialect bindings. | `$ORIGIN` RPATH, `auditwheel repair`, `_mlir_libs/` layout, manylinux compliance, ODS-generated bindings |
| 8 | [Ch 8 --- Sim-Only Installation Mode](ch8_sim_only_mode/index.md) | Lightweight simulator-only installation that skips the compiler and tt-metal dependencies. | `TTLANG_SIM_ONLY`, `ttl-sim` package, `ttl[sim]` extras group, build-time flag approach |

---

## Quick Reference

| Concept / Tool | What It Does | Where to Learn More |
|----------------|-------------|---------------------|
| `scripts/build-and-install.sh` | Orchestrates the five-phase CMake build (configure, LLVM, tt-metal, tt-lang, finalize). | [Ch 1](ch1_current_build_flow/index.md) |
| `TTLANG_TOOLCHAIN_DIR` | Points to pre-built LLVM/MLIR + tt-metal + tt-mlir artifacts for toolchain-based builds. | [Ch 1](ch1_current_build_flow/index.md), [Ch 5](ch5_pip_install_with_toolchain/index.md) |
| `TTLANG_USE_TOOLCHAIN=ON` | CMake flag that skips LLVM/tt-metal/tt-mlir submodule builds and consumes pre-built artifacts. | [Ch 5](ch5_pip_install_with_toolchain/index.md) |
| `CMakeBuild` class in `python/setup.py` | Custom setuptools build command that invokes CMake to build nanobind extensions. | [Ch 2](ch2_python_packaging_as_is/index.md) |
| `TTLangPythonCAPI` | Shared CAPI library aggregating MLIR, tt-mlir, and TT-Lang Python bindings. | [Ch 3](ch3_cpp_extension_dependencies/index.md) |
| `_ttlang` / `_ttmlir` | The two nanobind extension modules providing Python access to the compiler and MLIR dialects. | [Ch 3](ch3_cpp_extension_dependencies/index.md) |
| `MLIR_PYTHON_PACKAGE_PREFIX=ttl.` | Compile definition routing upstream MLIR Python modules under `ttl.ir`, `ttl.dialects`, etc. | [Ch 3](ch3_cpp_extension_dependencies/index.md), [Ch 7](ch7_wheel_packaging/index.md) |
| `auditwheel repair` | Vendors transitive `.so` dependencies into a wheel for manylinux compliance. | [Ch 4](ch4_prior_art/index.md), [Ch 7](ch7_wheel_packaging/index.md) |
| `ttl-toolchain` wheel | Pre-built wheel containing LLVM/MLIR and tt-metal shared libraries and CMake config. | [Ch 6](ch6_two_phase_wheel_architecture/index.md) |
| `TTLANG_SIM_ONLY` | CMake option for a pure-Python simulator-only installation with no compiler build. | [Ch 8](ch8_sim_only_mode/index.md) |

---

## Prerequisites

- **Python packaging fundamentals**: Familiarity with wheels, sdists, `pyproject.toml`, and setuptools.
- **CMake**: Intermediate knowledge of CMake build systems (`find_package`, targets, install components).
- **MLIR concepts**: General awareness of MLIR-based compiler projects (dialects, TableGen, CAPI). No deep MLIR expertise is required; relevant details are introduced in [Ch 3](ch3_cpp_extension_dependencies/index.md).
- **TT-Lang stack**: Understanding of the roles of tt-metal, tt-mlir, and LLVM/MLIR in the TT-Lang compilation pipeline. [Ch 1](ch1_current_build_flow/index.md) provides a full walkthrough for those unfamiliar.
- **Tools**: `clang`/`clang++`, `lld`, Ninja, CMake >= 3.28, Python >= 3.11, and `git` installed on the build host.

---

## Source Code Locations

| Component | Repository Path |
|-----------|----------------|
| TT-Lang repository root | `/localdev/salnahari/testing_dir/tt-lang/` |
| Root `pyproject.toml` | `/localdev/salnahari/testing_dir/tt-lang/pyproject.toml` |
| Python packaging (`setup.py`) | `/localdev/salnahari/testing_dir/tt-lang/python/setup.py` |
| CMake modules | `/localdev/salnahari/testing_dir/tt-lang/cmake/modules/` |
| Build orchestration script | `/localdev/salnahari/testing_dir/tt-lang/scripts/build-and-install.sh` |
| Nanobind extensions (C++) | `/localdev/salnahari/testing_dir/tt-lang/python/TTLangModule.cpp`, `TT_MLIRMinimalExtension.cpp` |
| MLIR dialect definitions | `/localdev/salnahari/testing_dir/tt-lang/lib/Dialect/` |
| tt-mlir submodule | `/localdev/salnahari/testing_dir/tt-lang/third-party/tt-mlir/` |
| Simulator package | `/localdev/salnahari/testing_dir/tt-lang/sim/` |
