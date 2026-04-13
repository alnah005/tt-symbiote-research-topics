# Lessons Learned

This file distills cross-cutting lessons from the [case studies](./case_studies.md) that are directly applicable to TT-Lang's `pip install` strategy.

---

## 1. `scikit-build-core` vs. `setuptools` + Custom `CMakeBuild`

All four surveyed projects use `setuptools` with a custom CMake build command class. None of them have migrated to `scikit-build-core`, despite it being the recommended PEP 517 build backend for CMake-based projects. The reasons vary, but the trade-offs are clear:

### `setuptools` + Custom `CMakeBuild` (status quo)

**Advantages:**
- Maximum control over the CMake invocation (flags, targets, install components).
- Easy to add environment-variable-driven overrides (e.g., `TORCH_MLIR_CMAKE_ALREADY_BUILT`, `CIRCT_CMAKE_BUILD_DIR`).
- All four projects already have battle-tested implementations.

**Disadvantages:**
- Not PEP 517 compliant by default -- the custom `build_ext` command works with `python setup.py bdist_wheel` but may not work with `pip install .` in isolated build mode.
- Editable installs (`pip install -e .`) require manual `sys.path` manipulation.
- No standard way to pass CMake configuration options through `pip install` (each project invents its own environment variables).

### `scikit-build-core`

**Advantages:**
- Full PEP 517 compliance: works with `pip install .`, `python -m build`, and build isolation out of the box.
- Editable installs are supported natively (using symlinks or redirect imports).
- CMake configuration can be passed through `pyproject.toml` `[tool.scikit-build.cmake]` or via `pip install . --config-settings cmake.args=-DFOO=ON`.
- Automatic wheel tagging and platform detection.

**Disadvantages:**
- Less control over the build lifecycle -- the CMake invocation is managed by `scikit-build-core`, so projects with unusual build steps (configure-time sub-builds, artifact copying, venv management) may need workarounds.
- Newer ecosystem with fewer examples of MLIR-scale projects using it in production.
- Migration cost for projects that already have a working `setup.py`.

### Recommendation for TT-Lang

TT-Lang should evaluate `scikit-build-core` for its PEP 517 compliance and editable-install support. However, given the complexity of TT-Lang's configure-time builds (LLVM, tt-metal) described in [Chapter 1](../ch1_current_build_flow/index.md), a phased approach is safer:

1. **Phase 1:** Fix the existing `setuptools` + `CMakeBuild` approach to work with `pip install .` (not just `python setup.py bdist_wheel`).
2. **Phase 2:** Migrate to `scikit-build-core` once the toolchain-wheel split is in place and the CMake build is simplified to a single `cmake --build` step.

---

## 2. The "Toolchain Wheel" Pattern

The most impactful pattern across these projects is the separation of slow, infrequently-changing dependencies (LLVM, runtime frameworks) from the fast, frequently-changing project code.

### How Each Project Implements It

| Project | Slow dependency | Fast project code | Separation mechanism |
|---------|----------------|-------------------|---------------------|
| **Triton** | LLVM static libraries | Triton compiler + pybind11 extension | Pre-built LLVM downloaded from Azure Blob Storage at build time |
| **torch-mlir** | LLVM/MLIR libraries | torch-mlir dialects + PyTorch integration | `LLVM_INSTALL_DIR` env var points to pre-built LLVM; CI builds LLVM in a cached Docker layer |
| **IREE** | LLVM/MLIR compiler stack | Runtime HAL drivers | Two separate wheels (`iree-base-compiler`, `iree-base-runtime`) |
| **CIRCT** | LLVM/MLIR libraries | CIRCT dialects | `CIRCT_LLVM_DIR` env var; no wheel-level split |

### Applying This to TT-Lang

TT-Lang's build has three slow components (see [Chapter 1](../ch1_current_build_flow/index.md)):

1. **LLVM/MLIR** (~30-60 minutes from source)
2. **tt-metal** (~20-40 minutes from source)
3. **tt-lang itself** (~5-10 minutes)

The toolchain wheel pattern would separate these into:

- A **`tt-lang-toolchain`** wheel (or downloadable tarball) containing pre-built LLVM and tt-metal artifacts, published infrequently (weekly or per-release).
- A **`tt-lang`** wheel that depends on the toolchain and contains only the tt-mlir dialects, Python bindings, and compiler tools, rebuilt on every commit.

Triton's auto-download approach is particularly attractive: `setup.py` checks for a local toolchain, and if absent, downloads the matching pre-built archive transparently. This eliminates the need for users to manually set `TTLANG_USE_TOOLCHAIN=ON` and manage toolchain paths.

---

## 3. `auditwheel` and RPATH Considerations

Every project that ships Linux wheels uses `auditwheel repair` to produce manylinux-compatible wheels. The key considerations:

### What `auditwheel repair` Does

1. Scans the wheel's `.so` files for dynamic library dependencies.
2. Copies dependent libraries that are not part of the manylinux allowlist into the wheel (under a `.libs/` or similar directory).
3. Rewrites RPATHs in the `.so` files to point to the bundled copies via `$ORIGIN`.

### Common Pitfalls

**Over-bundling framework libraries.** torch-mlir explicitly excludes PyTorch's `.so` files from `auditwheel repair` because users already have PyTorch installed. Similarly, CIRCT excludes LLVM libraries when the user provides their own LLVM. TT-Lang will need to exclude `_ttnn.so` and related tt-metal libraries if those are expected to come from a separate package.

The exclusion is done via `auditwheel repair --exclude`:

```bash
auditwheel repair dist/*.whl \
    --exclude libtorch.so \
    --exclude libtorch_cpu.so \
    --exclude libc10.so \
    -w wheelhouse/
```

**RPATH conflicts with editable installs.** When using `pip install -e .`, the `.so` files live in the source tree rather than in `site-packages`. If `auditwheel` has rewritten RPATHs to be relative to the installed location, editable installs may fail to resolve libraries. Triton works around this by using `--no-build-isolation` during development, which skips the wheel repair step entirely.

**manylinux version selection.** All surveyed projects target `manylinux_2_27` or `manylinux_2_28` (glibc 2.27 or 2.28). The `_2_28` variant covers all Linux distributions from 2018 onward (Ubuntu 20.04+, RHEL 8+, Debian 10+). TT-Lang should target `manylinux_2_28` unless there is a specific need for older distribution support.

### macOS and Windows Equivalents

- **macOS:** `delocate-wheel` performs library bundling and sets `@loader_path`-relative install names. torch-mlir defaults to `CMAKE_OSX_DEPLOYMENT_TARGET=11.1` with universal2 (arm64 + x86-64) architectures.
- **Windows:** `delvewheel` bundles DLLs. torch-mlir uses it to bundle `TorchMLIRAggregateCAPI.dll` while excluding PyTorch DLLs.

---

## 4. sdist Correctness: `MANIFEST.in` and `pyproject.toml` Package Discovery

A working `pip install .` from a git checkout is necessary but not sufficient. For the package to be installable from a source distribution (`sdist`), all required files must be included in the tarball. This is a common source of subtle bugs.

### The Problem

When `pip install` runs in an isolated build environment (the default since PEP 517), it first builds an sdist, unpacks it into a temporary directory, and then builds a wheel from there. If the sdist is missing files -- CMakeLists.txt, `.td` files, C++ sources, submodule contents -- the build fails.

### How Projects Handle This

**Triton** avoids the problem by not supporting sdist installation at all -- the PyPI releases are wheels only. Source installation requires a git clone.

**torch-mlir** also distributes wheels only on PyPI. From-source builds require a git clone with initialized submodules (`externals/llvm-project`).

**IREE** distributes wheels only. The `compiler/setup.py` and `runtime/setup.py` are designed for use within the git repository, not from an sdist.

### What This Means for TT-Lang

If TT-Lang only publishes pre-built wheels to PyPI (the recommended approach), sdist correctness is less critical. However, for developers who want to `pip install .` from a git checkout, the following must be correct:

1. **`pyproject.toml` `[tool.setuptools.packages]`** must list all Python packages, including generated ones (e.g., the MLIR Python bindings output directory).

2. **`MANIFEST.in`** (or `pyproject.toml` `[tool.setuptools.package-data]`) must include:
   - All `CMakeLists.txt` files in subdirectories
   - TableGen `.td` files
   - C++ source and header files needed by CMake
   - The `cmake/` directory with module files

3. **Submodule content** cannot be included in an sdist. This is why every surveyed project requires a git clone for from-source builds -- git submodules are fundamentally incompatible with Python sdists.

A minimal `MANIFEST.in` for a project like TT-Lang would look like:

```
recursive-include cmake *.cmake
recursive-include lib *.cpp *.h *.td
recursive-include include *.h *.td
recursive-include python *.py
include CMakeLists.txt
include pyproject.toml
```

But because TT-Lang depends on submodules (llvm-project, tt-metal), a true sdist-only install path is not feasible. The practical approach is: **publish wheels to PyPI, support `pip install .` from git clones, and do not attempt to support `pip install` from sdist tarballs.**

---

## Summary

| Lesson | Implication for TT-Lang |
|--------|------------------------|
| All successful projects pre-build LLVM in CI | TT-Lang must not require end-users to build LLVM |
| `setuptools` + `CMakeBuild` is the proven pattern | Start with fixing the existing `setup.py`; migrate to `scikit-build-core` later |
| The toolchain wheel pattern separates slow from fast builds | Split into `tt-lang-toolchain` (LLVM + tt-metal) and `tt-lang` (compiler + bindings) |
| `auditwheel repair` requires explicit exclusions | Exclude tt-metal and PyTorch `.so` files from repair |
| sdist-only installs are not viable for submodule-heavy projects | Publish wheels; support `pip install .` from git clones only |

---

**Next:** [Chapter 5 -- pip install with Pre-Built Toolchain](../ch5_pip_install_with_toolchain/index.md)
