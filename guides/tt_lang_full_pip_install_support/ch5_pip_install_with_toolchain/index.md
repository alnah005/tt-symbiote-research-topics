# Chapter 5 -- `pip install` with Pre-Built Toolchain

This chapter presents the concrete changes needed so that

```bash
TTLANG_TOOLCHAIN_DIR=/opt/ttlang-toolchain pip install .
```

produces a working `ttl` package -- complete with the nanobind `_ttlang` and `_ttmlir` C extensions, the upstream MLIR Python bindings, and every pure-Python module (`ttl`, `pykernel`, `sim`, `utils`).

## Design Goals

1. **LLVM, tt-metal, and tt-mlir are never compiled during `pip install`.** The pre-built toolchain at `TTLANG_TOOLCHAIN_DIR` supplies all MLIR CMake configs, pre-built CAPI libraries (`.so` / `.a`), pre-generated TableGen `.inc` files, and tt-metal runtime artifacts. Only the nanobind extensions and pure Python code are compiled.

2. **A single environment variable is the contract.** Setting `TTLANG_TOOLCHAIN_DIR` (or passing it via `CMAKE_ARGS`) is the only requirement beyond a working C++ toolchain (clang, cmake, ninja). The CMake option `TTLANG_USE_TOOLCHAIN=ON` is implied automatically when invoked from pip.

3. **Standard pip workflows work.** `pip install .`, `pip install -e .`, `pip wheel .`, and `python -m build` must all succeed. The package metadata in `pyproject.toml` is self-contained -- no dynamic version hacks, no `requirements.txt` re-parsing at build time.

4. **The build is fast.** By targeting only `TTLangPythonModules` (not the full project), pip builds compile roughly a dozen C++ source files against pre-built libraries. A clean build should complete in under two minutes on a modern workstation.

## Scope of Changes

The proposal is split across three files, each covering a layer of the packaging stack:

| File | Layer | What Changes |
|------|-------|-------------|
| [`pyproject_toml_changes.md`](./pyproject_toml_changes.md) | Package metadata | Static dependencies, build-system requirements, optional extras, scikit-build-core evaluation |
| [`setup_py_fixes.md`](./setup_py_fixes.md) | Build orchestration | `CMakeBuild.build_()` path resolution, toolchain passthrough, MANIFEST.in, install component handling |
| [`cmake_changes.md`](./cmake_changes.md) | CMake build system | `python/CMakeLists.txt` standalone operation, toolchain skip guards, generated file handling |

## How It Fits Together

The existing build (described in [Chapter 2](../ch2_python_packaging_as_is/index.md)) already has the pieces -- `TTLANG_USE_TOOLCHAIN`, `TTLangPythonWheel` install component, `CMakeBuild` in `setup.py` -- but they do not work end-to-end because:

- `setup.py` (at `python/setup.py`, line 49) computes `cwd` as `pathlib.Path().absolute()`, which resolves to pip's temporary build directory, not the repo root.
- `pyproject.toml` declares `dependencies` as `dynamic` (line 35) but `setup.py` only lists `pydantic<3` (line 117), omitting `torch`, `numpy`, `greenlet`, and every MLIR runtime dependency from `requirements.txt`.
- The CMake configure invoked by `setup.py` runs the *full* project (`CMakeLists.txt` at repo root), which triggers `BuildLLVM`, `BuildTTMLIRMinimal`, and `BuildTTMetal` -- a multi-hour build that defeats the purpose of a pre-built toolchain.
- `config.py.in` and `_generated_elementwise.py` are generated during the CMake build, but their generation depends on targets that may not exist when only `TTLangPythonModules` is requested.

The three change files below address each of these issues with minimal, targeted modifications to the existing codebase.

**Prev:** [Chapter 4 -- Prior Art](../ch4_prior_art/index.md) | **Next:** [`pyproject_toml_changes.md`](./pyproject_toml_changes.md)
