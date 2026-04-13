# TT-Lang Full pip install Support — Research Guide Plan

## Audience

This guide is for **TT-Lang developers and build engineers** who want to make the TT-Lang compiler and DSL runtime installable via standard `pip install .` workflows. Readers are expected to be familiar with:

- Python packaging concepts (wheels, sdists, `pyproject.toml`, setuptools)
- CMake build systems at an intermediate level
- The general architecture of MLIR-based compiler projects (dialects, tablegen, CAPI)
- The role of tt-metal, tt-mlir, and LLVM/MLIR in the TT-Lang stack

Readers do **not** need prior experience with `scikit-build-core`, `auditwheel`, `cibuildwheel`, or wheel repair tooling — those topics are covered in the guide.

---

## Chapter List

### Chapter 1: Current Build and Installation Flow
**Description:** A detailed walkthrough of the existing CMake-driven build pipeline, from `scripts/build-and-install.sh` through LLVM/tt-metal compilation to Python package installation.

**Directory:** `ch1_current_build_flow`

**Files:**

- `index.md`
  - Overview of the end-to-end build pipeline and its five major phases (configure, LLVM build, tt-metal build, tt-lang build, install/finalize)
  - The role of `scripts/build-and-install.sh` and its mode flags (`--toolchain-only`, `--configure-only`, `--build-and-install`, `--finalize`, `--test-toolchain`)

- `cmake_architecture.md`
  - Root `CMakeLists.txt` structure: option declarations (`TTLANG_USE_TOOLCHAIN`, `TTLANG_SIM_ONLY`, `TTLANG_BUILD_TOOLCHAIN`), the include order of cmake modules, and the `add_subdirectory` calls
  - `BuildLLVM.cmake`: dual-mode LLVM dependency (pre-built via `MLIR_PREFIX`/`MLIR_DIR` vs. submodule build with `execute_process`), SHA verification, ccache forwarding
  - `BuildTTMLIRMinimal.cmake`: TableGen processing of tt-mlir dialect `.td` files, LLK header generation, the C++ dialect libraries built (TTCore, TTMetal, TTKernel)
  - `BuildTTMetal.cmake`: configure-time tt-metal build from submodule, nested submodule initialization, sentinel-based rebuild skip (`_ttnn.so`), artifact installation into toolchain
  - `TTLangPython.cmake`: venv creation and discovery logic (toolchain venv, local project venv, user override), `VIRTUAL_ENV` activation for `find_package(Python3)`
  - `TTLangCompilerSetup.cmake`: clang/lld detection, ccache, compiler flags

- `environment_assumptions.md`
  - Environment variables consumed: `TTLANG_TOOLCHAIN_DIR`, `CMAKE_BINARY_DIR`, `IN_CIBW_ENV`, `TT_METAL_SIMULATOR`, `CPM_SOURCE_CACHE`, `TTLANG_CMAKE_DEBUG`
  - Pre-installed tool requirements: clang/clang++, lld, Ninja, cmake >= 3.28, Python >= 3.11, git
  - The `env/activate.in` template and what it sets: `PYTHONPATH`, `LD_LIBRARY_PATH`, `TT_LANG_HOME`, `TT_METAL_HOME`, `LLVM_INSTALL_DIR`
  - Why `source env/activate` is currently required after every build

### Chapter 2: Python Packaging As-Is
**Description:** How the existing `pyproject.toml` and `python/setup.py` work together, what the custom `CMakeBuild` class does, and why `pip install .` fails today.

**Directory:** `ch2_python_packaging_as_is`

**Files:**

- `index.md`
  - The relationship between root `pyproject.toml` (build-system requires, project metadata, dynamic fields) and `python/setup.py` (the actual build logic)
  - Why `pyproject.toml` lives at the repo root but `setup.py` lives in `python/` — and how setuptools resolves this

- `cmake_build_class.md`
  - The `TTLangExtension` / `CMakeBuild` class in `python/setup.py`: how it invokes `cmake -G Ninja -B build`, builds the `TTLangPythonModules` target, and installs the `TTLangPythonWheel` component
  - The CI mode (`IN_CIBW_ENV=ON`): `env/activate` sourcing, adjusted install directory
  - The package list: `ttl`, `ttl._src`, `pykernel`, `pykernel._src`, `sim`, `utils` — which are pure Python and which depend on compiled extensions
  - Dynamic version generation from date stamps vs. git tags (`GetVersionFromGit.cmake`)

- `why_pip_install_fails.md`
  - Missing pre-built LLVM/MLIR: `CMakeBuild.build_()` assumes `cmake -B build` will find an already-configured build directory or will configure from scratch, but LLVM submodule build takes hours
  - Missing tt-metal: not built by `setup.py`'s cmake invocation
  - Path assumptions: `cwd.parent / "build"` assumes `python/` is the working directory; `pip install .` from repo root breaks this
  - No `MANIFEST.in` or `setuptools` source inclusion for C++ sources, `.td` files, cmake modules
  - The `python_packages/` output directory structure (flat `ttl/`, `pykernel/`) vs. what setuptools expects

### Chapter 3: C++ Extension Build Dependencies
**Description:** The specific LLVM/MLIR libraries, tt-mlir artifacts, and tt-metal components that must be available before the nanobind modules can compile.

**Directory:** `ch3_cpp_extension_dependencies`

**Files:**

- `index.md`
  - Overview of the two nanobind extension modules: `_ttlang` (TTLangModule.cpp, TTLModule.cpp) and `_ttmlir` (TT_MLIRMinimalExtension.cpp, TT_MLIRMinimalPasses.cpp, TT_MLIRMinimalTTModule.cpp, TT_MLIRMinimalTTKernelModule.cpp)
  - The shared CAPI library `TTLangPythonCAPI`: what it aggregates (MLIRPythonSources, MLIRPythonExtension.RegisterEverything, TTMLIRMinPythonSources/Extensions, TTLangPythonSources/Extensions)

- `mlir_dependency_chain.md`
  - MLIR CMake infrastructure used: `AddMLIRPython`, `declare_mlir_python_sources`, `declare_mlir_python_extension`, `add_mlir_python_common_capi_library`, `add_mlir_python_modules`
  - LLVM/MLIR libraries linked: `MLIRCAPIIR`, `MLIRCAPITransforms`, and everything transitively pulled by `MLIRPythonSources`
  - tt-mlir CAPI: `TTMLIRMinimalCAPI` (from `lib/ttmlir-minimal/CAPI/`), `TTLangCAPI`
  - TableGen-generated artifacts required before compilation: ODS-generated `.py` and `.h.inc` files for TTCore, TTKernel, TTL dialects
  - The `MLIR_PYTHON_PACKAGE_PREFIX=ttl.` compile definition and `MLIR_BINDINGS_PYTHON_NB_DOMAIN=ttl` nanobind domain

- `discovery_mechanisms.md`
  - How LLVM/MLIR is currently discovered: `find_package(MLIR REQUIRED CONFIG)` using `MLIR_DIR` set from `LLVM_INSTALL_DIR/lib/cmake/mlir`
  - How tt-mlir sources are found: hardcoded `third-party/tt-mlir` submodule path
  - How tt-metal is found: `TT_METAL_HOME`, `TT_METAL_PYTHON_PATH`, `TT_METAL_LIB_PATH` — all set by `BuildTTMetal.cmake`
  - Python dev packages: `MLIRDetectPythonEnv` + `mlir_configure_python_dev_packages()` — must run after BuildLLVM sets `Python3_EXECUTABLE`

### Chapter 4: Prior Art — MLIR-Based pip install Approaches
**Description:** How other MLIR-based Python projects (torch-mlir, Triton, IREE, circt) handle `pip install` with heavy C++ toolchain dependencies.

**Directory:** `ch4_prior_art`

**Files:**

- `index.md`
  - Summary comparison table: project, build backend, LLVM strategy, wheel size, platform support
  - Common patterns: pre-built LLVM in CI, bundled shared libraries, separate toolchain wheels

- `case_studies.md`
  - **torch-mlir**: uses `setup.py` with CMake, pre-builds LLVM in CI, bundles all `.so` files in wheel, uses `auditwheel repair` for manylinux compliance, ships ~200MB wheels
  - **Triton (OpenAI)**: uses `setup.py` with CMake, builds LLVM from source during `pip install` (with caching via `TRITON_CACHE_DIR`), uses `pybind11`, ships as a single `triton` package
  - **IREE**: splits into `iree-compiler` and `iree-runtime` wheels, uses `scikit-build-core`, pre-builds LLVM in CI, bundles dialect Python bindings under `iree.compiler._mlir_libs`
  - **CIRCT**: uses `scikit-build-core` with `flit_core`, distributes pre-built wheels only, no from-source pip install path

- `lessons_learned.md`
  - scikit-build-core vs. setuptools+CMakeBuild: trade-offs (PEP 517 compliance, editable installs, configuration passthrough)
  - The "toolchain wheel" pattern: separating multi-hour LLVM builds from fast extension builds
  - `auditwheel` and RPATH considerations for bundled `.so` files
  - The importance of `MANIFEST.in` / `pyproject.toml` `[tool.setuptools.packages]` for sdist correctness

### Chapter 5: pip install with Pre-Built Toolchain
**Description:** Concrete changes to `pyproject.toml`, `setup.py`, and `CMakeLists.txt` to support `pip install .` when LLVM/tt-mlir/tt-metal are pre-built at `TTLANG_TOOLCHAIN_DIR`.

**Directory:** `ch5_pip_install_with_toolchain`

**Files:**

- `index.md`
  - Design goals: `TTLANG_TOOLCHAIN_DIR=/opt/ttlang-toolchain pip install .` should produce a working `ttl` package
  - Scope: only the nanobind extensions + pure Python packages are compiled; LLVM/tt-metal are consumed from the toolchain

- `pyproject_toml_changes.md`
  - Moving from `setuptools.build_meta` to either `scikit-build-core` or a fixed `CMakeBuild` in setuptools
  - Declaring `cmake` and `ninja` as build-system requires (already present)
  - Adding `[tool.scikit-build]` or `[tool.setuptools.cmake]` configuration for CMake variable passthrough (`TTLANG_TOOLCHAIN_DIR`, `TTLANG_USE_TOOLCHAIN=ON`)
  - Proper `[project.dependencies]` instead of dynamic: `pydantic<3`, `torch>=1.9.0`, `numpy>=1.20.0`, `greenlet>=3.0.0`
  - `[project.optional-dependencies]` for `sim`, `dev`, `test` extras

- `setup_py_fixes.md`
  - Fixing the `CMakeBuild.build_()` path logic: `cwd` should resolve to the repo root regardless of pip's working directory
  - Passing `TTLANG_USE_TOOLCHAIN=ON` and `TTLANG_TOOLCHAIN_DIR` as CMake defines
  - Ensuring `cmake --build` targets only `TTLangPythonModules` (not the full project)
  - Handling the `TTLangPythonWheel` install component to place files where setuptools expects them
  - Adding `MANIFEST.in` entries for C++ sources, `.td` files, cmake modules, and `third-party/tt-mlir/python/` sources needed during the build

- `cmake_changes.md`
  - Making `python/CMakeLists.txt` work standalone when invoked from pip: the `TTLangPythonModules` target must find pre-installed MLIR CMake config, pre-built CAPI libraries, and pre-generated TableGen outputs
  - Ensuring `TTLANG_USE_TOOLCHAIN=ON` skips `BuildLLVM`, `BuildTTMetal`, and `BuildTTMLIRMinimal` submodule builds (already partially supported)
  - Setting `CMAKE_INSTALL_PREFIX` to the setuptools build directory
  - Handling `config.py.in` and `_generated_elementwise.py` generation during pip build

### Chapter 6: Two-Phase Wheel Architecture
**Description:** Splitting the distribution into a toolchain wheel (`ttl-toolchain`) containing pre-built LLVM/MLIR/tt-metal shared libraries and a main wheel (`ttl`) that depends on it.

**Directory:** `ch6_two_phase_wheel_architecture`

**Files:**

- `index.md`
  - Rationale: separating the multi-hour toolchain build (LLVM ~30min, tt-metal ~20min) from the fast ttl extension build (~2min)
  - Package dependency graph: `ttl` depends-on `ttl-toolchain` (runtime), `ttl-toolchain` is platform-specific binary wheel

- `toolchain_wheel_design.md`
  - Contents of `ttl-toolchain` wheel: LLVM/MLIR shared libraries (`libMLIR*.so`, `libLLVM*.so`), MLIR Python bindings base (`_mlir_libs/__init__.py`), tt-metal shared libraries (`_ttnn.so`, `_ttnncpp.so`, `libtt_metal.so`, etc.), tt-metal Python packages (`ttnn/`), MLIR CMake config files (for downstream builds)
  - Package layout: `ttl_toolchain/mlir/`, `ttl_toolchain/ttmetal/`, `ttl_toolchain/cmake/`
  - Version pinning: toolchain version encodes LLVM SHA + tt-metal SHA for reproducibility
  - Size considerations: stripped LLVM install ~800MB, tt-metal libs ~500MB — wheel compression and selective inclusion strategies

- `main_wheel_design.md`
  - Contents of `ttl` wheel: `_ttlang.so`, `_ttmlir.so`, `TTLangPythonCAPI.so`, all pure Python packages (`ttl/`, `pykernel/`, `sim/`, `utils/`), MLIR dialect Python bindings (ODS-generated `.py` files)
  - Build-time dependency: `ttl-toolchain` must be installed for CMake to find MLIR config and link against CAPI libraries
  - Runtime dependency: `ttl-toolchain` provides shared libraries loaded at import time
  - How `_ttlang.so` and `_ttmlir.so` find `TTLangPythonCAPI.so` and its transitive MLIR dependencies at runtime (RPATH or `__init__.py` `ctypes.CDLL` preloading)

- `build_pipeline.md`
  - CI workflow: build `ttl-toolchain` wheel once per LLVM/tt-metal version bump, cache/publish to internal PyPI; build `ttl` wheel on every PR
  - Developer workflow: `pip install ttl-toolchain` from internal index, then `pip install -e .` for editable development
  - `cibuildwheel` integration: using `CIBW_BEFORE_BUILD` to install `ttl-toolchain`, passing `TTLANG_TOOLCHAIN_DIR` via environment

### Chapter 7: Wheel Packaging and Platform Compliance
**Description:** How to correctly bundle compiled nanobind extensions in wheels, handle RPATHs, run `auditwheel`, and include MLIR dialect bindings.

**Directory:** `ch7_wheel_packaging`

**Files:**

- `index.md`
  - Overview of manylinux compliance requirements for PyPI distribution
  - The challenge: TT-Lang wheels link against LLVM/MLIR `.so` files that are not system libraries

- `so_bundling_and_rpath.md`
  - How `.so` files should be laid out in the wheel: `ttl/_mlir_libs/_ttlang.cpython-311-x86_64-linux-gnu.so`, `ttl/_mlir_libs/_ttmlir.cpython-311-x86_64-linux-gnu.so`, `ttl/_mlir_libs/libTTLangPythonCAPI.so`
  - RPATH strategy: set `$ORIGIN` RPATH at build time so extensions find `libTTLangPythonCAPI.so` in the same `_mlir_libs/` directory
  - `auditwheel repair` usage: vendoring transitive `.so` dependencies into the wheel, the `--exclude` flag for libraries provided by `ttl-toolchain`
  - `auditwheel show` output analysis and troubleshooting

- `mlir_dialect_bindings.md`
  - The MLIR Python binding file layout: `ttl/dialects/ttl.py`, `ttl/dialects/ttcore.py`, `ttl/dialects/ttkernel.py`, `ttl/_mlir_libs/_site_initialize_0.py`, `ttl/_mlir_libs/_site_initialize_1.py`
  - How `add_mlir_python_modules` copies these into `python_packages/ttl/` and how the wheel install must preserve this structure
  - The `ttl.` package prefix convention and how `MLIR_PYTHON_PACKAGE_PREFIX` ensures upstream MLIR modules land under `ttl.ir`, `ttl.dialects`, etc.
  - Generated files that must be included: `_generated_elementwise.py`, `config.py`

### Chapter 8: Sim-Only Installation Mode
**Description:** How to expose a lightweight simulator-only installation that skips the compiler and tt-metal dependencies entirely.

**Directory:** `ch8_sim_only_mode`

**Files:**

- `index.md`
  - Current sim-only support: `TTLANG_SIM_ONLY` CMake option creates a venv with only runtime requirements, no compiler build
  - Three possible approaches: separate package (`ttl-sim`), extras group (`pip install ttl[sim]`), or conditional dependencies with a build flag

- `design_options.md`
  - **Option A — Separate package (`ttl-sim`)**: a pure-Python wheel containing only `sim/`, `pykernel/`, `utils/` packages with minimal dependencies (`torch`, `greenlet`, `pydantic`). Pros: simple, no C++ build. Cons: code duplication, two packages to maintain
  - **Option B — Extras group (`pip install ttl[full]`)**: the base `ttl` package includes only pure Python (sim + pykernel + utils); `ttl[full]` or `ttl[compiler]` adds the compiled extensions and MLIR dependencies. Pros: single package. Cons: the base install cannot include compiled extensions at all, which may confuse users expecting `import ttl` to work with the compiler
  - **Option C — Build-time flag**: `TTLANG_SIM_ONLY=ON` environment variable during `pip install` skips extension compilation. Pros: single package and single `pip install` command. Cons: non-standard, build flags are not discoverable
  - Recommended approach and rationale

---

## Conventions

1. **File paths** are always given relative to the TT-Lang repo root (`/localdev/salnahari/testing_dir/tt-lang`), e.g., `python/setup.py`, `cmake/modules/BuildLLVM.cmake`.

2. **CMake variables** are written in `UPPER_SNAKE_CASE` and formatted as inline code, e.g., `TTLANG_TOOLCHAIN_DIR`.

3. **Environment variables** are prefixed with `$` when showing shell usage, e.g., `$TTLANG_TOOLCHAIN_DIR`, but written without `$` when discussing the variable name itself.

4. **Python package names** use the canonical PyPI name (lowercase with hyphens), e.g., `ttl`, `ttl-toolchain`. Python import names use underscores, e.g., `import ttl`, `import ttl_toolchain`.

5. **Shell commands** are shown in fenced code blocks with `bash` syntax highlighting. CMake code uses `cmake` highlighting.

6. **Terminology:**
   - "Toolchain" refers to the pre-built LLVM/MLIR + tt-metal + tt-mlir artifact set installed at `TTLANG_TOOLCHAIN_DIR`.
   - "Extension" or "nanobind extension" refers to the compiled `.so` modules (`_ttlang`, `_ttmlir`).
   - "CAPI library" refers to `TTLangPythonCAPI`, the shared library aggregating all C API symbols.
   - "Dialect bindings" refers to the ODS-generated Python files in `ttl/dialects/`.
   - "Wheel" refers to a Python `.whl` distribution file.

7. **Code snippets** from the TT-Lang codebase include the source file path as a comment on the first line.

8. **Cross-references** between chapters use relative markdown links, e.g., `[see Chapter 3](../ch3_cpp_extension_dependencies/index.md)`.

---

## Cross-Chapter Dependencies

| Chapter | Depends On | Reason |
|---------|-----------|--------|
| Ch2 (Python Packaging As-Is) | Ch1 (Current Build Flow) | Ch2 references the CMake targets and build phases explained in Ch1 |
| Ch3 (C++ Extension Dependencies) | Ch1 (Current Build Flow) | Ch3 discusses LLVM/tt-mlir/tt-metal artifacts whose build process is covered in Ch1 |
| Ch4 (Prior Art) | Ch2 (Python Packaging As-Is), Ch3 (C++ Extension Dependencies) | Ch4 compares external approaches against TT-Lang's current packaging and dependency structure |
| Ch5 (pip install with Toolchain) | Ch1 (Current Build Flow), Ch2 (Python Packaging As-Is), Ch3 (C++ Extension Dependencies) | Ch5 proposes concrete changes to files analyzed in Ch1-Ch3 |
| Ch6 (Two-Phase Wheel Architecture) | Ch3 (C++ Extension Dependencies), Ch4 (Prior Art), Ch5 (pip install with Toolchain) | Ch6 extends the single-wheel approach from Ch5 into a split architecture, informed by prior art from Ch4 |
| Ch7 (Wheel Packaging) | Ch3 (C++ Extension Dependencies), Ch5 (pip install with Toolchain), Ch6 (Two-Phase Wheel Architecture) | Ch7 addresses the `.so` bundling and RPATH details for the wheel layouts designed in Ch5-Ch6 |
| Ch8 (Sim-Only Mode) | Ch1 (Current Build Flow), Ch2 (Python Packaging As-Is), Ch6 (Two-Phase Wheel Architecture) | Ch8 references the existing `TTLANG_SIM_ONLY` CMake option from Ch1 and proposes packaging alternatives informed by Ch2 and Ch6 |
