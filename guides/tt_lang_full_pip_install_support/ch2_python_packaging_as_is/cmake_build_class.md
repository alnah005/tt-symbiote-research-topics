# The CMakeBuild Class and Package Structure

This file documents the `TTLangExtension` / `CMakeBuild` class in
`python/setup.py` — the setuptools machinery that bridges Python packaging with
the CMake build system — as well as the package list, CI mode, and version
generation.

## `TTLangExtension` — a stub extension

```python
# python/setup.py
class TTLangExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])
```

`TTLangExtension` declares a setuptools `Extension` with **no source files**.
Its sole purpose is to trigger the `build_ext` command so that `CMakeBuild.run()`
executes. Without at least one `ext_modules` entry, setuptools would skip the
`build_ext` phase entirely.

A single instance is created:

```python
# python/setup.py
ttlang_c = TTLangExtension("ttl")
```

## `CMakeBuild` — the build_ext override

`CMakeBuild` subclasses `setuptools.command.build_ext.build_ext` and overrides
`run()` to dispatch to `build_()`:

```python
# python/setup.py
class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            if "ttl" in ext.name:
                self.build_(ext)
            else:
                raise Exception("Unknown extension")
```

### `build_()` — the CMake invocation

The `build_()` method performs three sequential steps:

1. **Configure** — invoke `cmake -G Ninja -B <build_dir>` with Release mode
2. **Build** — invoke `cmake --build <build_dir> -- TTLangPythonModules`
3. **Install** — invoke `cmake --install <build_dir> --component TTLangPythonWheel`

#### Step 1: Configure

```python
# python/setup.py  (inside build_)
cwd = pathlib.Path().absolute()
build_dir = cwd.parent / "build"

install_dir = pathlib.Path(self.build_lib)

cmake_args = [
    "-G", "Ninja",
    "-B", str(build_dir),
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_INSTALL_PREFIX=" + str(install_dir),
    "-DCMAKE_C_COMPILER=clang",
    "-DCMAKE_CXX_COMPILER=clang++",
]

if not self.in_ci():
    cmake_args.extend(["-S", str(cwd.parent)])
```

Key assumptions:

- **`cwd` is `python/`**, so `cwd.parent` is the repo root and `build_dir`
  resolves to `<repo_root>/build`. See
  [`why_pip_install_fails.md`, section 3](./why_pip_install_fails.md#3-path-assumptions-cwdparent--build)
  for failure scenarios when this assumption breaks.
- `CMAKE_INSTALL_PREFIX` is set to setuptools' `self.build_lib`, where
  setuptools expects to find installed Python files before bundling into a wheel.
- In non-CI mode, `-S <repo_root>` is passed so CMake knows where the top-level
  `CMakeLists.txt` lives.

#### Step 2: Build

```python
# python/setup.py
self.spawn(["cmake", "--build", str(build_dir), "--", "TTLangPythonModules"])
```

This builds only the `TTLangPythonModules` CMake target, which includes:

- Upstream `MLIRPythonSources` and `MLIRPythonExtension.RegisterEverything`
- `TTMLIRMinPythonSources` and `TTMLIRMinPythonExtensions` (the `_ttmlir`
  nanobind module)
- `TTLangPythonSources` and `TTLangPythonExtensions` (the `_ttlang` nanobind
  module)
- `TTLangPythonCommon` (runtime Python files like `ttl_api.py`, `layouts.py`)
- Generated files: `_generated_elementwise.py` (from `TTLElementwiseOps.def`)
  and `config.py` (from `config.py.in`)

The output lands in `TTLANG_PYTHON_PACKAGES_DIR`, which is typically
`<build_dir>/python_packages/`.

#### Step 3: Install

```python
# python/setup.py
self.spawn(
    ["cmake", "--install", str(build_dir), "--component", "TTLangPythonWheel"]
)
```

The `TTLangPythonWheel` component is defined in `python/CMakeLists.txt`:

```cmake
# python/CMakeLists.txt
install(DIRECTORY ${CMAKE_BINARY_DIR}/python_packages/
  DESTINATION .
  COMPONENT TTLangPythonWheel
  EXCLUDE_FROM_ALL)
```

This copies the entire `python_packages/` tree into `CMAKE_INSTALL_PREFIX`,
which `build_()` set to `self.build_lib`. However, because `build_()` only
builds the `TTLangPythonModules` target in Step 2, only `ttl/` (and `utils/`)
are guaranteed to exist under `python_packages/` at this point.

**Note on `pykernel`:** `PykernelPythonModules` is a **separate**
`add_mlir_python_modules()` target in `python/CMakeLists.txt` (lines 252-257).
It is **not** a dependency of `TTLangPythonModules`, so `cmake --build ...
-- TTLangPythonModules` does not build it. The `pykernel/` directory will only
appear under `python_packages/` if `PykernelPythonModules` is built separately
(or if a higher-level target like `all` is invoked). In the current wheel-build
flow, the `pykernel` files that end up in the final wheel come from setuptools'
`build_py` phase via the `package_dir` mapping in `setup.py`, not from the
CMake install step.

## CI mode (`IN_CIBW_ENV=ON`)

When the `IN_CIBW_ENV` environment variable is set to `ON`, two things change:

### 1. `env/activate` is sourced before configure

```python
# python/setup.py
if self.in_ci():
    subprocess.run(
        " ".join([
            "cd", str(cwd.parent), "&&",
            ".", "env/activate", "&&",
            "cmake", *cmake_args,
        ]),
        shell=True,
        check=True,
    )
```

This sources the `env/activate` script (which sets `PATH`, `LD_LIBRARY_PATH`,
and other environment variables for the pre-built LLVM/MLIR toolchain) before
running CMake. In CI, the build directory is assumed to already be configured.

### 2. Install directory is adjusted

```python
# python/setup.py
if self.in_ci():
    install_dir = cwd / "build" / install_dir.name
```

Instead of installing to `self.build_lib` (which in cibuildwheel may be an
absolute path outside the repo), CI mode installs to
`python/build/<build_lib_name>`. This is because cibuildwheel runs the build
inside a container with different path conventions.

Also note that in CI mode, the `-S <repo_root>` flag is **not** passed to
CMake — the assumption is that the build directory was already configured by
a prior step.

## The package list

The `setup()` call declares six top-level packages:

```python
# python/setup.py
packages=["ttl", "ttl._src", "pykernel", "pykernel._src", "sim", "utils"],
```

| Package | Type | Description |
|---|---|---|
| `ttl` | Compiled extensions + pure Python | Main package. Contains MLIR Python bindings, nanobind extensions (`_ttmlir.so`, `_ttlang.so`), ODS-generated dialect wrappers, and runtime Python code (`ttl_api.py`, `layouts.py`, etc.) |
| `ttl._src` | Pure Python | Internal implementation: AST helpers, profiling, tensor registry |
| `pykernel` | Pure Python | Kernel DSL AST and type definitions, declared as a separate `add_mlir_python_modules` target in CMake |
| `pykernel._src` | Pure Python | Kernel AST internals: `base_ast.py`, `kernel_ast.py`, `kernel_types.py` |
| `sim` | Pure Python | Simulator: block state, DFB, greenlet scheduler, torch utilities |
| `utils` | Pure Python | Utility functions: block allocation, correctness checking |

Only `ttl` depends on compiled C++ extensions. Of the remaining five pure-Python
packages, `ttl._src` and `utils` are declared via `declare_mlir_python_sources`
and built as part of the `TTLangPythonModules` CMake target, so they end up in
the `python_packages/` output directory alongside the compiled artifacts.
`pykernel` and `pykernel._src` are under a separate CMake target
(`PykernelPythonModules` — see Step 3 note above). `sim` is **not** declared in
`python/CMakeLists.txt` at all; it is handled exclusively by setuptools'
`build_py` phase using the `package_dir` mapping in `setup.py`.

## Dynamic version generation

TT-Lang uses two independent version strategies depending on context:

### In `python/setup.py` — date-stamp version

```python
# python/setup.py
date = datetime.now().strftime("%y.%m.%d")
version = "0.1." + date + ".dev0"
```

This produces versions like `0.1.26.04.09.dev0`. This is the version that ends
up in the wheel metadata when building via `setup.py`.

### In CMake — git-tag version (`GetVersionFromGit.cmake`)

```cmake
# cmake/modules/GetVersionFromGit.cmake
execute_process(
  COMMAND ${GIT_EXECUTABLE} describe --tags --match "v[0-9]*" --abbrev=0
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_TAG
  ...
)

if(GIT_TAG)
  string(REGEX REPLACE "^v" "" TTLANG_VERSION "${GIT_TAG}")
  # ... parse major.minor.patch ...
  # If commits exist since tag, append .dev<count>
  if(COMMITS_SINCE_TAG AND NOT COMMITS_SINCE_TAG EQUAL "0")
    set(TTLANG_VERSION "${TTLANG_VERSION}.dev${COMMITS_SINCE_TAG}")
  endif()
else()
  set(TTLANG_VERSION "0.2.0.dev0")
endif()
```

The CMake version uses `git describe --tags` to find the most recent `v*` tag,
then appends `.dev<N>` where `N` is the number of commits since that tag. If no
tags exist, it falls back to `0.2.0.dev0`.

These two versioning schemes are **not synchronized**. The `setup.py` date-stamp
version is what pip sees in the wheel filename and metadata, while the CMake
version is used internally (e.g., in `config.py` generated from `config.py.in`).
Unifying these is one of the tasks for full pip install support.

---

**Next:** [`why_pip_install_fails.md`](./why_pip_install_fails.md)
