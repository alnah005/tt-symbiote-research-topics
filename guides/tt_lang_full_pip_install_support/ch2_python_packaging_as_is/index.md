# Chapter 2 — Python Packaging As-Is

This chapter examines how TT-Lang currently exposes itself as a Python package:
the split between `pyproject.toml` (declarative metadata) and `python/setup.py`
(imperative build logic), and how setuptools resolves the two.

## Files covered

- [`cmake_build_class.md`](./cmake_build_class.md) — The `TTLangExtension` / `CMakeBuild` class, CI mode, package list, and version generation
- [`why_pip_install_fails.md`](./why_pip_install_fails.md) — Why a naive `pip install .` from the repo root cannot succeed today

## The split: `pyproject.toml` vs. `python/setup.py`

TT-Lang declares its Python package metadata in two places:

| Concern | File | Location |
|---|---|---|
| PEP 517 build-system declaration | `pyproject.toml` | repo root |
| Actual build logic and package list | `python/setup.py` | `python/` subdirectory |

### `pyproject.toml` — declarative metadata at repo root

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=61.0", "cmake", "nanobind", "wheel", "pip", "ninja"]
build-backend = "setuptools.build_meta"

[project]
name = "ttl"
description = "Python Bindings and Package for TT-Lang Compiler Project"
requires-python = ">=3.11"
dynamic = ["version", "dependencies", "readme"]
```

Key points:

1. **`build-system.requires`** lists the host dependencies pip must install into
   the isolated build environment before invoking the backend. This is where
   `cmake`, `ninja`, and `nanobind` enter the picture — pip will `pip install`
   them into a temporary venv.

2. **`dynamic = ["version", "dependencies", "readme"]`** tells setuptools that
   these three fields are *not* declared statically in `pyproject.toml`. Instead,
   they will be supplied at build time by `setup()` in `python/setup.py`.

3. **No `[tool.setuptools]` section** defines `package-dir`, `packages`, or
   `py-modules`. All of that is delegated to the imperative `setup()` call.

### `python/setup.py` — the imperative build logic

```python
# python/setup.py
setup(
    name="ttl",
    version=version,
    ext_modules=[ttlang_c],       # triggers build_ext -> CMakeBuild
    cmdclass={"build_ext": CMakeBuild},
    packages=[...],               # 6 packages — see cmake_build_class.md
    package_dir={...},            # relative paths from python/
    ...
)
```

The `package_dir` mapping uses *relative paths from `python/`*, because
`setup.py` lives there and setuptools resolves these paths relative to the
directory containing `setup.py`. For the full package list, `package_dir`
mapping, and per-package descriptions, see
[`cmake_build_class.md` — The package list](./cmake_build_class.md#the-package-list).

### How setuptools resolves the two files

When you run `pip install .` from the repo root, the PEP 517 build frontend:

1. Reads `pyproject.toml` at the repo root.
2. Installs the build requirements (`setuptools`, `cmake`, `nanobind`, etc.)
   into an isolated build environment.
3. Calls `setuptools.build_meta` as the build backend.
4. The build backend looks for `setup.py` **in the current working directory** —
   which is the repo root, not `python/`.

This is the first critical mismatch: `setup.py` lives in `python/`, not at the
repo root. Setuptools will either fail to find `setup.py` or, if it does find
it, the `package_dir` relative paths (`"ttl": "ttl"`) resolve against the repo
root instead of `python/`, so the source packages cannot be located.

The existing workflow sidesteps this by never running `pip install .` from the
repo root. Instead, the CMake build system handles Python packaging through the
`TTLangPythonModules` target and the `TTLangPythonWheel` install component (see
[`cmake_build_class.md`](./cmake_build_class.md) for details).

### Why the split exists

The split is a pragmatic consequence of TT-Lang's CMake-first architecture:

- The **repo root** is the CMake source directory (`CMAKE_SOURCE_DIR`), hosting
  `CMakeLists.txt`, C++ sources, MLIR TableGen files, and the LLVM/tt-mlir
  submodules.
- The **`python/` subdirectory** is the Python-specific subtree, containing
  `setup.py`, the `ttl`, `pykernel`, `sim`, and `utils` packages, and the
  `CMakeLists.txt` that declares MLIR Python module targets.
- `pyproject.toml` must live at the repo root because PEP 517 frontends
  (`pip`, `build`) always look for it there.
- `setup.py` lives in `python/` because its `package_dir` paths, its README
  resolution (`pathlib.Path(__file__).absolute().parent.parent / "README.md"`),
  and its CMake invocation (`cwd.parent / "build"`) all assume `python/` as the
  working directory.

This split works when the CMake build drives everything, but it is a fundamental
obstacle to a standard `pip install .` flow. [Chapter 3](../ch3_cpp_extension_dependencies/index.md)
discusses the C++ dependencies that compound this problem.

---

**Next:** [`cmake_build_class.md`](./cmake_build_class.md)
