# Chapter 6 -- Two-Phase Wheel Architecture

Chapter 5 showed how `pip install .` can work against a pre-built toolchain directory on disk. That design has a critical limitation: the user must already possess a correctly built `TTLANG_TOOLCHAIN_DIR`, which today means either running the full CMake build or obtaining a Docker image that contains one. This chapter proposes splitting the deliverable into **two pip-installable wheels** so that the toolchain itself becomes a pip dependency -- no Docker image, no manual `cmake` invocation, no environment variables.

## The Problem: Build Time Asymmetry

The TT-Lang build has two phases with radically different rebuild frequencies and durations:

| Phase | Components | Typical Duration | Rebuild Trigger |
|-------|-----------|-----------------|-----------------|
| **Toolchain** | LLVM/MLIR (from `third-party/llvm-project`), tt-metal (from `third-party/tt-metal`), tt-mlir CAPI libraries | 30--50 minutes | LLVM submodule bump, tt-metal submodule bump |
| **Extension** | `_ttlang.so`, `_ttmlir.so`, `TTLangPythonCAPI.so`, pure Python packages, generated dialect bindings | 1--2 minutes | Any change to `python/`, `lib/`, `include/` |

The toolchain changes roughly once per release cycle (when submodule SHAs are updated). The extension changes on every PR. Building both from source on every `pip install` is wasteful; bundling both into a single wheel produces an artifact that is too large for PyPI (the LLVM install alone is ~800 MB unstripped) and takes too long to rebuild in CI.

## The Two-Wheel Split

The solution is two separate Python packages:

```
ttl-toolchain        (platform wheel, ~400-600 MB compressed)
  Contains: LLVM/MLIR shared libs, MLIR Python bindings base,
            MLIR CMake configs, tt-metal shared libs + Python packages,
            tt-mlir CAPI static libs + headers

ttl                  (platform wheel, ~15-25 MB compressed)
  Contains: _ttlang.so, _ttmlir.so, TTLangPythonCAPI.so,
            all pure Python (ttl, pykernel, sim, utils),
            MLIR dialect bindings (ODS-generated)
  Depends:  ttl-toolchain (exact version pin)
```

*Note:* The version string `0.1.250413` is used throughout this chapter as a concrete example. In practice, the version is determined by the release tag at build time (see the [Version Pinning](./toolchain_wheel_design.md#version-pinning) section).

### Package Dependency Graph

```
pip install ttl
    |
    +---> ttl-toolchain==0.1.250413   (auto-installed as dependency)
    |       Provides at runtime:
    |         - libMLIR*.so, libLLVM*.so
    |         - MLIRConfig.cmake (used only at ttl build time)
    |         - tt-metal shared libs (libtt_metal.so, _ttnn.so, etc.)
    |         - tt-metal Python packages (ttnn, tools)
    |
    +---> pydantic<3                   (pure Python dependency)
    +---> torch>=1.9.0                  (runtime dependency)
    +---> numpy>=1.20.0                (runtime dependency)
```

At **build time**, `ttl` declares `ttl-toolchain` as a build dependency in `pyproject.toml`'s `[build-system].requires` so that CMake can `find_package(MLIR)` against the installed toolchain. At **runtime**, `ttl` declares `ttl-toolchain` as an install dependency so that shared libraries are available for `import ttl`.

### Why Not a Single Large Wheel?

1. **PyPI size limit.** PyPI enforces a 100 MB limit per file (with exceptions granted to ~200 MB). A combined wheel with LLVM libraries would exceed this even with aggressive stripping and compression.
2. **CI rebuild cost.** Rebuilding LLVM on every PR is a 30-minute cost that delays iteration. A cached toolchain wheel eliminates this entirely.
3. **Version decoupling.** The toolchain wheel can be versioned and released independently. Multiple `ttl` versions can share the same `ttl-toolchain` as long as the ABI is compatible.
4. **Developer ergonomics.** `pip install ttl-toolchain && pip install -e .` gives developers a fast editable-install workflow without touching CMake directly.

## Chapter Contents

| File | Description |
|------|-------------|
| [`toolchain_wheel_design.md`](./toolchain_wheel_design.md) | Contents, layout, versioning, and size considerations for the `ttl-toolchain` wheel |
| [`main_wheel_design.md`](./main_wheel_design.md) | Contents of the `ttl` wheel, build-time and runtime dependencies on `ttl-toolchain`, RPATH and library resolution |
| [`build_pipeline.md`](./build_pipeline.md) | CI workflow (build `ttl-toolchain` once, `ttl` on every PR), developer workflow, `cibuildwheel` integration |

## Relationship to Earlier Chapters

- [Chapter 3](../ch3_cpp_extension_dependencies/index.md) catalogues the extension modules (`_ttlang`, `_ttmlir`) and their CAPI dependency graph. This chapter uses that catalogue to draw the boundary between the two wheels.
- [Chapter 4](../ch4_prior_art/index.md) documents the toolchain-wheel pattern used by IREE (separate compiler and runtime wheels) and Triton (pre-built LLVM download). This chapter applies those lessons to TT-Lang's specific dependency structure.
- [Chapter 5](../ch5_pip_install_with_toolchain/index.md) established the `TTLANG_USE_TOOLCHAIN` CMake mode and the `setup.py` changes needed for `pip install` against a local toolchain directory. The two-phase architecture builds on that foundation by packaging the toolchain directory itself as a wheel.

**Next:** [`toolchain_wheel_design.md`](./toolchain_wheel_design.md)
