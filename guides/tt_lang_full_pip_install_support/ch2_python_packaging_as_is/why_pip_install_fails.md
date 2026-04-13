# Why `pip install .` Fails Today

Running `pip install .` from the TT-Lang repo root does not produce a working
wheel. This file catalogs every failure mode, from missing native dependencies
to structural assumptions baked into the current `setup.py`.

## 1. Missing pre-built LLVM/MLIR

The `CMakeBuild.build_()` method runs the CMake configure step
(see [`cmake_build_class.md`, Step 1](./cmake_build_class.md#step-1-configure)
for the full invocation). This configures the **entire** TT-Lang CMake project from scratch, which
includes LLVM and MLIR as submodule dependencies. Building LLVM from source
takes several hours on typical hardware and requires substantial disk space
(tens of GB).

There is no mechanism in `setup.py` to:

- Download a pre-built LLVM/MLIR toolchain
- Skip the LLVM build if it already exists
- Use the `TTLANG_USE_TOOLCHAIN` / `TTLANG_TOOLCHAIN_DIR` CMake options that
  the normal developer workflow supports

In the CI path (`IN_CIBW_ENV=ON`), this is sidestepped by assuming the build
directory is already configured with a pre-built LLVM. But a bare
`pip install .` has no such pre-existing state.

## 2. Missing tt-metal

The `setup.py` CMake invocation only builds the `TTLangPythonModules` target.
However, the full CMake configure step requires the `tt-metal` submodule to be
present and its dependencies to be available. The `TTLangPythonModules` target
depends (transitively) on:

- `TTLangCAPI` — which links against dialect libraries that depend on tt-mlir
- `TTMLIRMinimalCAPI` — the minimal CAPI for tt-mlir dialects
- `MLIRCAPIIR` and `MLIRCAPITransforms` — from the LLVM/MLIR build

None of these are built or provided by `setup.py` in isolation. The build simply
fails at the CMake configure or build step with missing targets.

## 3. Path assumptions: `cwd.parent / "build"`

The `build_()` method computes the build directory as:

```python
# python/setup.py
cwd = pathlib.Path().absolute()
build_dir = cwd.parent / "build"
```

This assumes `cwd` is `python/`, making `build_dir` resolve to
`<repo_root>/build`. But when pip runs `setup.py`, the working directory depends
on the pip version and build isolation settings:

| Scenario | Working directory | `cwd.parent / "build"` resolves to |
|---|---|---|
| `cd python && python setup.py bdist_wheel` | `python/` | `<repo_root>/build` (correct) |
| `pip install .` from repo root | repo root (or a temp copy) | `<repo_root_parent>/build` (wrong) |
| `pip install .` with build isolation | isolated temp dir | arbitrary path (wrong) |

Similarly, the README is located via:

```python
# python/setup.py
readme_path = pathlib.Path(__file__).absolute().parent.parent / "README.md"
```

This works when `setup.py` is at `python/setup.py` (so `.parent.parent` is the
repo root), but breaks if pip copies the source tree to a temporary location
with a different directory structure.

## 4. No `MANIFEST.in` or source inclusion for C++ files

The repository has **no `MANIFEST.in` file**. When pip builds a wheel with build
isolation enabled (the default), it first creates a source distribution (sdist)
of the project. Without a `MANIFEST.in`, setuptools includes only:

- Files listed in `packages` / `package_dir` (the `.py` files)
- `pyproject.toml`, `setup.py`, `setup.cfg`

This means the sdist will be **missing**:

- All C++ source files (`lib/`, `include/`)
- MLIR TableGen definitions (`.td` files)
- CMake modules (`cmake/modules/`)
- The `python/CMakeLists.txt` that defines `TTLangPythonModules`
- The root `CMakeLists.txt`
- The `env/activate` script
- Submodule contents (`third-party/llvm-project/`, `third-party/tt-mlir/`)

Even if all other issues were resolved, the CMake configure step would fail
because none of the source files it needs would be present in the isolated build
directory.

## 5. Output directory structure mismatch

The CMake build produces a flat directory structure under
`<build_dir>/python_packages/`:

```
python_packages/
  ttl/
    __init__.py
    _mlir_libs/
      _ttlang.cpython-311-x86_64-linux-gnu.so
      _ttmlir.cpython-311-x86_64-linux-gnu.so
      libTTLangPythonCAPI.so
      ...
    dialects/
      ttl.py
      ttcore.py
      ttkernel.py
      ...
    ttl_api.py
    layouts.py
    ...
  pykernel/
    __init__.py
    _src/
      ...
```

The `TTLangPythonWheel` install component copies this entire tree into
`CMAKE_INSTALL_PREFIX` (i.e., `self.build_lib`).

However, setuptools also expects to find the packages declared in `package_dir`
**in the source tree** during the `build_py` phase (which runs *before*
`build_ext`). Setuptools' `build_py` command copies `.py` files from
`python/ttl/`, `python/pykernel/`, etc. into `self.build_lib`. Then `build_ext`
(the CMake build) overwrites some of those files with the CMake-generated
versions. This creates a race condition:

- `build_py` copies `python/ttl/__init__.py` to `build/lib/ttl/__init__.py`
- `build_ext` (CMake install) overwrites `build/lib/ttl/__init__.py` with the
  version from `python_packages/ttl/__init__.py`

For generated files (like `_generated_elementwise.py` or `config.py`) that only
exist in the CMake output, `build_py` will fail to find them (or copy stale
versions) and `build_ext` must correctly overwrite them.

## Summary of blockers

| # | Blocker | Severity |
|---|---|---|
| 1 | LLVM/MLIR not available as pre-built artifact | Critical — hours-long build |
| 2 | tt-metal not built by `setup.py` | Critical — missing link targets |
| 3 | `cwd.parent / "build"` path assumption | High — wrong build directory |
| 4 | No `MANIFEST.in` for C++/cmake/td files | High — empty sdist |
| 5 | `python_packages/` vs. setuptools `build_py` | Medium — file overwrite race |

These blockers are addressed in subsequent chapters, starting with
[Chapter 3 — C++ Extension Build Dependencies](../ch3_cpp_extension_dependencies/index.md).

---

**Next:** [Chapter 3 — C++ Extension Build Dependencies](../ch3_cpp_extension_dependencies/index.md)
