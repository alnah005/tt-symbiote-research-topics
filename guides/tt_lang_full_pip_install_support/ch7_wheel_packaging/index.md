# Chapter 7 -- Wheel Packaging and Platform Compliance

The previous chapters designed a [two-phase wheel architecture](../ch6_two_phase_wheel_architecture/index.md) that splits TT-Lang into a heavyweight `ttl-toolchain` wheel and a lightweight `ttl` extension wheel. This chapter addresses the next problem: making those wheels actually installable via `pip install` on any compatible Linux machine, without requiring the user to have LLVM, MLIR, or tt-metal already present on their system.

That requirement boils down to **manylinux compliance** -- the set of rules PyPI enforces so that a pre-built binary wheel works across Linux distributions.

## The manylinux Problem

PyPI does not accept wheels tagged `linux_x86_64`; it requires a [manylinux tag](https://peps.python.org/pep-0600/) such as `manylinux_2_28_x86_64`. A manylinux-tagged wheel promises that every `.so` file inside it either:

1. Links only against a small allow-list of system libraries (`libc.so.6`, `libm.so.6`, `libpthread.so.0`, `libdl.so.2`, `librt.so.1`, `libstdc++.so.6`, etc.), or
2. Vendors (bundles) any non-system `.so` dependency inside the wheel itself.

TT-Lang wheels violate this promise out of the box. The native extensions `_ttlang.cpython-311-x86_64-linux-gnu.so` and `_ttmlir.cpython-311-x86_64-linux-gnu.so` link against:

- `libTTLangPythonCAPI.so` -- the unified CAPI library built by `add_mlir_python_common_capi_library()` in `python/CMakeLists.txt`
- Transitive LLVM/MLIR `.so` files (`libMLIR.so`, `libLLVM.so`, etc.)
- tt-metal runtime libraries (when device support is enabled)

None of these appear on the manylinux allow-list.

## The Two Sub-Problems

Making TT-Lang wheels manylinux-compliant requires solving two interrelated problems, each covered in its own section:

| Section | Problem | Solution |
|---------|---------|----------|
| [`.so` Bundling and RPATH](./so_bundling_and_rpath.md) | Extension `.so` files cannot find `libTTLangPythonCAPI.so` at runtime | Set `$ORIGIN` RPATH, use `auditwheel repair` to vendor transitive deps |
| [MLIR Dialect Bindings](./mlir_dialect_bindings.md) | ODS-generated Python files and generated code must land at correct paths | Leverage `add_mlir_python_modules` with `MLIR_PYTHON_PACKAGE_PREFIX=ttl.` |

## Wheel File Layout (Target State)

After `auditwheel repair`, the `ttl` wheel should be self-contained: pure Python runtime files, ODS-generated dialect bindings under `ttl/dialects/`, nanobind extensions and the unified CAPI library under `ttl/_mlir_libs/`, and build-generated files (`config.py`, `_generated_elementwise.py`) at the package root. For the full file listing and a verification command, see [Verifying the Wheel Contents](./mlir_dialect_bindings.md#verifying-the-wheel-contents).

The `ttl-toolchain` wheel contains the heavy LLVM/MLIR shared libraries and tt-metal runtime. Whether those `.so` files live in the toolchain wheel (vendored) or are `--exclude`-d from `auditwheel` and expected to come from the toolchain depends on the [two-wheel boundary](../ch6_two_phase_wheel_architecture/main_wheel_design.md) chosen at build time.

## cibuildwheel Configuration

The existing `pyproject.toml` already declares the target platform:

```toml
[tool.cibuildwheel]
build = "cp311-manylinux_x86_64*"
skip = "*-musllinux_*"
build-verbosity = 2
```

This tells `cibuildwheel` to:

1. Build only for CPython 3.11 on `manylinux_x86_64`
2. Skip musl-based distributions (Alpine)
3. Use verbose output during the build

The `setup.py` `CMakeBuild` class (see [Chapter 2](../ch2_python_packaging_as_is/cmake_build_class.md)) detects the `IN_CIBW_ENV` environment variable to adjust the install directory when running inside a cibuildwheel container.

## What Comes Next

The following two sections walk through the implementation details:

- [`.so` Bundling and RPATH](./so_bundling_and_rpath.md) -- how shared libraries are laid out, how RPATH makes them discoverable, and how `auditwheel` vendors transitive dependencies
- [MLIR Dialect Bindings](./mlir_dialect_bindings.md) -- how MLIR's `add_mlir_python_modules` infrastructure copies dialect sources into the wheel, the role of `MLIR_PYTHON_PACKAGE_PREFIX`, and the generated files that must be included

**Next:** [`so_bundling_and_rpath.md`](./so_bundling_and_rpath.md)
