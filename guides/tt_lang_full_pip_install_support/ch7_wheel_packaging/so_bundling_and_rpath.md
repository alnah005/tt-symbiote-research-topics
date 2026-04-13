# `.so` Bundling and RPATH Strategy

This section covers the native `.so` layout inside the wheel and the `auditwheel repair` workflow.

## Shared Library Layout Inside the Wheel

After `cmake --install` runs the `TTLangPythonWheel` component (see [Chapter 2 -- `CMakeBuild`](../ch2_python_packaging_as_is/cmake_build_class.md)), the build tree under `python_packages/ttl/_mlir_libs/` contains three categories of `.so` files:

### 1. Nanobind Extension Modules

These are the Python-importable C++ extension modules built by `declare_mlir_python_extension()`:

```
ttl/_mlir_libs/_ttlang.cpython-311-x86_64-linux-gnu.so
ttl/_mlir_libs/_ttmlir.cpython-311-x86_64-linux-gnu.so
```

`_ttlang` is declared in `python/CMakeLists.txt` (line 126) with `EMBED_CAPI_LINK_LIBS TTLangCAPI` and `_ttmlir` (line 73) with `EMBED_CAPI_LINK_LIBS MLIRCAPIIR MLIRCAPITransforms TTMLIRMinimalCAPI`. Both use nanobind as the binding library (`PYTHON_BINDINGS_LIBRARY nanobind`) and share a common nanobind domain `"ttl"` so that C types are compatible across modules.

### 2. The Unified CAPI Library

```
ttl/_mlir_libs/libTTLangPythonCAPI.so.20
```

This is produced by `add_mlir_python_common_capi_library(TTLangPythonCAPI ...)` at line 236 of `python/CMakeLists.txt`. It statically links all declared CAPI sources (upstream MLIR, tt-mlir minimal, and ttlang) into a single shared library. The extension modules (`_ttlang.so`, `_ttmlir.so`) dynamically link against it.

The `CMAKE_PLATFORM_NO_VERSIONED_SONAME ON` setting (line 4) prevents CMake from creating versioned symlink chains (`libFoo.so -> libFoo.so.20 -> libFoo.so.20.1`), which simplifies wheel packaging.

### 3. Vendored Transitive Dependencies

After `auditwheel repair`, additional `.so` files from LLVM/MLIR and system toolchains may be copied into `_mlir_libs/`:

```
ttl/_mlir_libs/libLLVM-20.so           # if not excluded
ttl/_mlir_libs/libMLIR.so.20           # if not excluded
ttl/_mlir_libs/libz.so.1               # zlib, pulled in by LLVM
```

Which of these get vendored vs. excluded depends on the two-wheel boundary, discussed below.

## RPATH Strategy

When Python does `from ttl._mlir_libs import _ttlang`, the dynamic linker must resolve `_ttlang.so`'s dependency on `libTTLangPythonCAPI.so`. On a developer machine this works because `LD_LIBRARY_PATH` points at the build directory. Inside a wheel, there is no `LD_LIBRARY_PATH` -- the `.so` must know where to look.

### Setting `$ORIGIN` RPATH

The solution is to set the RPATH of each extension module to `$ORIGIN`, which tells the dynamic linker to search the directory containing the `.so` itself. Since both `_ttlang.so` and `libTTLangPythonCAPI.so` live in `ttl/_mlir_libs/`, `$ORIGIN` resolves correctly.

This should be set in CMake:

```cmake
# In python/CMakeLists.txt or a shared helper
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
```

Alternatively, `auditwheel repair` will rewrite RPATHs automatically when it vendors libraries, but setting the RPATH at build time is more predictable and enables the wheel to work even before `auditwheel` processing (useful for local development).

You can verify RPATH settings with `patchelf`:

```bash
$ patchelf --print-rpath ttl/_mlir_libs/_ttlang.cpython-311-x86_64-linux-gnu.so
$ORIGIN
```

Or with `readelf`:

```bash
$ readelf -d ttl/_mlir_libs/_ttlang.cpython-311-x86_64-linux-gnu.so | grep RPATH
 0x000000000000000f (RPATH)  Library rpath: [$ORIGIN]
```

## `auditwheel repair`

[auditwheel](https://github.com/pypa/auditwheel) is the standard tool for making Linux wheels manylinux-compliant. It performs two functions:

1. **Vendoring** -- copies non-system `.so` dependencies into the wheel and rewrites RPATHs to point at them.
2. **Tag rewriting** -- changes the wheel filename from `linux_x86_64` to `manylinux_2_28_x86_64` (or whichever tag matches).

### Basic Usage

```bash
# Build the wheel first (produces a linux_x86_64 wheel)
cd python && pip wheel . --no-deps -w dist/

# Repair it
auditwheel repair dist/ttl-0.1.250413-cp311-cp311-linux_x86_64.whl \
    --plat manylinux_2_28_x86_64 \
    -w dist/repaired/
```

### The `--exclude` Flag

In the two-wheel architecture ([Chapter 6](../ch6_two_phase_wheel_architecture/index.md)), LLVM/MLIR `.so` files live in the `ttl-toolchain` wheel, not the `ttl` wheel. If `auditwheel` vendors them into the `ttl` wheel, the combined size becomes unmanageable (~800 MB+). The `--exclude` flag prevents vendoring specific libraries:

```bash
auditwheel repair dist/ttl-*.whl \
    --plat manylinux_2_28_x86_64 \
    --exclude libLLVM-20.so \
    --exclude libMLIR.so.20 \
    --exclude libMLIRPythonCAPI.so \
    --exclude libtt_metal.so \
    -w dist/repaired/
```

Each `--exclude` tells auditwheel: "this library will be provided at runtime by another package (the toolchain wheel); do not vendor it." The `ttl` wheel then declares a runtime dependency on `ttl-toolchain` (with an exact version pin), ensuring the excluded `.so` files are present when the wheel is installed.

### How Vendoring Works Internally

When `auditwheel repair` vendors a library, it:

1. Copies the `.so` into a `.libs/` subdirectory (or alongside the extension, depending on the `--strip` behavior)
2. Renames it with a hash suffix to avoid collisions: `libz-a1b2c3d4.so.1`
3. Patches the extension's `DT_NEEDED` entry to reference the renamed file
4. Sets the RPATH of the extension to `$ORIGIN/.libs` (or the directory containing the vendored file)

After repair, the wheel structure might look like:

```
ttl/_mlir_libs/_ttlang.cpython-311-x86_64-linux-gnu.so     # RPATH=$ORIGIN
ttl/_mlir_libs/libTTLangPythonCAPI.so.20                    # vendored in-place
ttl.libs/libz-a1b2c3d4.so.1                                # vendored + renamed
ttl.libs/libzstd-e5f6a7b8.so.1                             # vendored + renamed
```

## `auditwheel show` -- Diagnosis and Troubleshooting

Before running `repair`, `auditwheel show` reports the current compliance status:

```bash
$ auditwheel show dist/ttl-0.1.250413-cp311-cp311-linux_x86_64.whl

ttl-0.1.250413-cp311-cp311-linux_x86_64.whl is consistent with
the following platform tag: "linux_x86_64".

The wheel references external versioned symbols in these
system-provided shared libraries: libc.so.6 with GLIBC_2.28,
libstdc++.so.6 with GLIBCXX_3.4.29, libm.so.6 with GLIBC_2.28

The following external shared libraries are not in the manylinux_2_28
policy allow-list and would need to be bundled:
    libTTLangPythonCAPI.so.20
    libMLIR.so.20
    libLLVM-20.so
    libz.so.1
```

This output tells you exactly which libraries block manylinux compliance. The remediation strategy for each:

| Library | Action |
|---------|--------|
| `libTTLangPythonCAPI.so.20` | Already in the wheel; `auditwheel` just needs to confirm RPATH |
| `libMLIR.so.20`, `libLLVM-20.so` | `--exclude` (provided by `ttl-toolchain`) |
| `libz.so.1` | Let `auditwheel` vendor it (small, ~100 KB) |
| `libstdc++.so.6`, `libc.so.6`, `libm.so.6` | Already on the allow-list; no action needed |

### Common Failure Modes

**"Cannot repair wheel -- could not find library"**

This means a `.so` referenced by `DT_NEEDED` is not on `LD_LIBRARY_PATH` or in the wheel. It typically happens when building outside the cibuildwheel container and the LLVM install prefix is not in the library search path. Fix:

```bash
export LD_LIBRARY_PATH=/path/to/toolchain/lib:$LD_LIBRARY_PATH
auditwheel repair ...
```

**"Wheel is not consistent with any manylinux policy"**

This means the wheel uses glibc symbols newer than what any manylinux tag permits. Check which glibc version you built against:

```bash
$ auditwheel show dist/ttl-*.whl 2>&1 | grep GLIBC
```

If it shows `GLIBC_2.35`, you need to build in a `manylinux_2_28` container (which provides glibc 2.28) to produce a compatible wheel.

**Extension loads but crashes with "symbol not found"**

This happens when `--exclude` was used for a library that is not actually provided at runtime. Verify that `ttl-toolchain` exposes the excluded library and that Python can find it:

```python
import ttl_toolchain
print(ttl_toolchain.get_lib_dir())  # Should print path containing libMLIR.so.20
```

## Integration With the Build Pipeline

In CI, the full sequence is:

```bash
# 1. Build inside manylinux container (via cibuildwheel or manually)
pip wheel python/ --no-deps -w dist/

# 2. Repair with exclusions for toolchain-provided libs
auditwheel repair dist/ttl-*.whl \
    --plat manylinux_2_28_x86_64 \
    --exclude libLLVM-20.so \
    --exclude libMLIR.so.20 \
    --exclude libtt_metal.so \
    -w dist/repaired/

# 3. Verify the repaired wheel
auditwheel show dist/repaired/ttl-*.whl
# Expected: "is consistent with manylinux_2_28_x86_64"

# 4. Quick smoke test
pip install dist/repaired/ttl-*.whl
python -c "import ttl; print(ttl.__version__)"
```

For more detail on how this fits into the overall CI/CD pipeline, see the [build pipeline design](../ch6_two_phase_wheel_architecture/build_pipeline.md).

**Next:** [`mlir_dialect_bindings.md`](./mlir_dialect_bindings.md)
