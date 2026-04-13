# Build Pipeline

This file describes the CI workflow for building both wheels, the developer workflow for local iteration, and the `cibuildwheel` integration for producing platform-compatible artifacts.

## CI Workflow Overview

The two-phase architecture maps directly to two CI pipelines with different trigger conditions:

```
Toolchain Pipeline                    Extension Pipeline
(runs on LLVM/tt-metal bump)          (runs on every PR)

  checkout tt-lang                      checkout tt-lang
       |                                     |
  build LLVM (~30 min)                  pip install ttl-toolchain
       |                                (from internal index, ~2 min)
  build tt-metal (~20 min)                   |
       |                                cmake configure (~10 sec)
  strip + package                       cmake build TTLangPythonModules (~90 sec)
       |                                     |
  upload ttl-toolchain wheel            build ttl wheel
  to internal index                     run tests
       |                                     |
  trigger extension pipeline            upload ttl wheel
  (to verify compatibility)             to internal index
```

### Toolchain Pipeline

**Trigger:** Changes to `third-party/llvm-project` submodule SHA, `third-party/tt-metal` submodule SHA, or `cmake/modules/BuildLLVM.cmake` / `cmake/modules/BuildTTMetal.cmake`.

**Steps:**

1. **Checkout** the TT-Lang repository with submodules (`git submodule update --init --recursive`).

2. **Build the toolchain and strip shared libraries** by running `scripts/build-toolchain-wheel.sh` (see the [Build Script](./toolchain_wheel_design.md#build-script) section in `toolchain_wheel_design.md` for the canonical cmake and strip invocations).

3. **Package the toolchain wheel** by running the toolchain wheel builder script:
   ```bash
   python scripts/build-toolchain-wheel.py \
     --toolchain-dir build/toolchain-install \
     --output-dir dist/ \
     --llvm-sha "$(git -C third-party/llvm-project rev-parse --short=7 HEAD)" \
     --ttmetal-sha "$(git -C third-party/tt-metal rev-parse --short=7 HEAD)"
   ```

4. **Upload** the wheel to the internal Python package index:
   ```bash
   twine upload --repository internal dist/ttl_toolchain-*.whl
   ```

5. **Trigger the extension pipeline** as a downstream job to verify that the new toolchain wheel builds and passes tests with the current `main` branch.

**Caching:** The LLVM and tt-metal builds use `ccache` (already configured in `BuildLLVM.cmake` and `BuildTTMetal.cmake`). CI runners maintain a persistent ccache directory across runs. A full cold build takes ~50 minutes; a warm rebuild (after minor LLVM patches) takes ~10-15 minutes.

### Extension Pipeline

**Trigger:** Every push to a PR branch, every merge to `main`.

**Steps:**

1. **Checkout** the TT-Lang repository (submodules not required -- the toolchain wheel supplies everything).

2. **Install the toolchain:**
   ```bash
   pip install ttl-toolchain==0.1.250413 --index-url https://internal.example.com/simple/
   ```

3. **Build the `ttl` wheel** using `pip wheel`:
   ```bash
   pip wheel python/ --no-deps --wheel-dir dist/
   ```
   This invokes `setup.py`'s `CMakeBuild` class, which calls CMake with `TTLANG_USE_TOOLCHAIN=ON` and targets only `TTLangPythonModules`. Total build time: ~90 seconds.

4. **Install and test:**
   ```bash
   pip install dist/ttl-*.whl
   pytest test/ -x --timeout=300
   ```

5. **Upload** the wheel to the internal index (on `main` branch merges only):
   ```bash
   twine upload --repository internal dist/ttl-*.whl
   ```

## `cibuildwheel` Integration

Both wheels are platform-specific (they contain `.so` files compiled for a specific OS, architecture, and Python version). `cibuildwheel` handles the matrix of Python versions and platform tags.

### Toolchain Wheel `cibuildwheel` Config

The toolchain wheel does not depend on a specific Python version (the shared libraries are Python-version-agnostic), but it must be tagged with the correct platform. The `cibuildwheel` config builds a single wheel per platform:

```toml
# pyproject.toml for ttl-toolchain
[tool.cibuildwheel]
build = "cp311-manylinux_x86_64"
skip = "*-musllinux_*"

# The toolchain build happens outside cibuildwheel;
# cibuildwheel only packages and repairs the pre-built artifacts.
[tool.cibuildwheel.linux]
before-all = "scripts/build-toolchain-wheel.sh"
repair-wheel-command = ""  # Skip auditwheel; libraries are self-contained

[tool.cibuildwheel.environment]
TTLANG_BUILD_TOOLCHAIN = "ON"
```

Because the toolchain's shared libraries form a self-contained set (LLVM libraries link against each other and system libraries only), `auditwheel repair` is skipped. The wheel is tagged `manylinux_2_28_x86_64` (or the appropriate manylinux tag for the build environment's glibc version).

**Python version independence:** The toolchain wheel is built for a single Python version (e.g., `cp311`) but the shared libraries inside work with any Python 3.x. To support multiple Python versions, the wheel tag can use `py3-none-manylinux_2_28_x86_64` if no Python-version-specific code is included. However, since the MLIR Python bindings base (installed under `ttl_toolchain/mlir/python/`) is compiled for a specific Python version, one toolchain wheel per Python minor version is required.

### Main Wheel `cibuildwheel` Config

The existing `pyproject.toml` already has a `[tool.cibuildwheel]` section:

```toml
[tool.cibuildwheel]
build = "cp311-manylinux_x86_64*"
skip = "*-musllinux_*"
build-verbosity = 2
```

This needs two additions:

```toml
[tool.cibuildwheel.linux]
before-build = "pip install ttl-toolchain==0.1.250413 --index-url https://internal.example.com/simple/"
repair-wheel-command = ""  # TTLangPythonCAPI.so links dynamically; repaired via ctypes preloading

[tool.cibuildwheel.environment]
IN_CIBW_ENV = "ON"
```

The `before-build` hook installs the toolchain into the cibuildwheel build environment before the wheel build starts. The `IN_CIBW_ENV` variable is already checked by `setup.py` (line 38: `return os.environ.get("IN_CIBW_ENV") == "ON"`) to adjust path resolution for the cibuildwheel sandbox.

### `auditwheel` Considerations

The `ttl` wheel's extensions (`_ttlang.so`, `_ttmlir.so`, `TTLangPythonCAPI.so`) link against shared libraries from `ttl-toolchain`. Running `auditwheel repair` on the `ttl` wheel would attempt to bundle those libraries into the wheel, duplicating the entire LLVM/MLIR stack. This is undesirable because:

1. It would make the `ttl` wheel ~600 MB instead of ~15 MB.
2. The bundled libraries would conflict with the identical libraries in `ttl-toolchain`.

Instead, `auditwheel repair` is skipped for the `ttl` wheel. The `ctypes` pre-loading mechanism described in [main_wheel_design.md](./main_wheel_design.md) ensures that the libraries from `ttl-toolchain` are loaded before the extensions, satisfying all dynamic linker requirements.

The wheel is tagged with `linux_x86_64` (not `manylinux`) since it depends on the `ttl-toolchain` wheel for its platform-specific shared libraries rather than bundling them. If PyPI compliance is needed in the future, `auditwheel addtag` can be used to apply the correct `manylinux` tag after verifying glibc compatibility.

## Developer Workflow

### First-Time Setup

```bash
# Clone the repo
git clone https://github.com/tenstorrent/tt-lang.git
cd tt-lang

# Install the toolchain (one-time, ~2 min download)
pip install ttl-toolchain==0.1.250413 --index-url https://internal.example.com/simple/

# Editable install (compiles extensions, ~90 sec)
pip install -e python/
```

### Day-to-Day Iteration

```bash
# After modifying Python files: no rebuild needed (editable install)
pytest test/unit/ -x

# After modifying C++ extension sources:
pip install -e python/   # Recompiles only changed extensions

# After modifying .td dialect definitions:
pip install -e python/   # Reruns tablegen + recompiles
```

### Building Your Own Toolchain

Developers who need to modify LLVM or tt-metal (e.g., for debugging or testing patches) can still build a local toolchain:

```bash
# Option A: CMake toolchain build (existing workflow)
# See scripts/build-toolchain-wheel.sh for the full cmake invocation.
git submodule update --init --recursive
bash scripts/build-toolchain-wheel.sh

# Use the local toolchain instead of the pip-installed one
TTLANG_TOOLCHAIN_DIR=build-toolchain/install pip install -e python/

# Option B: Build and install as a local wheel
python scripts/build-toolchain-wheel.py \
  --toolchain-dir build/toolchain-install \
  --output-dir dist/
pip install dist/ttl_toolchain-*.whl --force-reinstall
pip install -e python/
```

The `TTLANG_TOOLCHAIN_DIR` environment variable takes precedence over the pip-installed `ttl-toolchain` package in `setup.py`'s `CMakeBuild`, so existing developer workflows continue to work unchanged.

## Release Process

A release involves building and publishing both wheels in sequence:

1. **Tag the release** on the `main` branch (e.g., `v0.1.250413`).
2. **Toolchain pipeline** runs, producing `ttl_toolchain-0.1.250413-cp311-cp311-manylinux_2_28_x86_64.whl`.
3. **Extension pipeline** runs against the new toolchain, producing `ttl-0.1.250413-cp311-cp311-manylinux_2_28_x86_64.whl`.
4. Both wheels are uploaded to the internal index.
5. Users install with: `pip install ttl==0.1.250413 --index-url https://internal.example.com/simple/`

The exact version pin (`ttl-toolchain==0.1.250413`) in `ttl`'s dependencies ensures that pip automatically pulls the correct toolchain wheel.

## Summary

| Aspect | Toolchain Pipeline | Extension Pipeline |
|--------|-------------------|-------------------|
| **Trigger** | LLVM/tt-metal submodule bump | Every PR, every merge |
| **Duration** | 30--50 min (cold), 10--15 min (warm) | ~2 min |
| **Output** | `ttl-toolchain` wheel (~500 MB) | `ttl` wheel (~15 MB) |
| **Frequency** | ~1--2x per release cycle | 10--50x per day |
| **Caching** | ccache for LLVM/tt-metal | Toolchain wheel from index |

**Prev:** [`main_wheel_design.md`](./main_wheel_design.md) | **Next:** [Chapter 7 — Wheel Packaging and Platform Compliance](../ch7_wheel_packaging/index.md)
