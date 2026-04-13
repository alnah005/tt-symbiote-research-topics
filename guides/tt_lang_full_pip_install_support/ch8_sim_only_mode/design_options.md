# Sim-Only Installation -- Design Options

This section evaluates three approaches for exposing the TT-Lang simulator as a pip-installable package that requires no C++ compilation, no LLVM toolchain, and no tt-metal.

## Option A -- Separate Package (`ttl-sim`)

Create a standalone pure-Python package called `ttl-sim` that ships only the simulator, pykernel, and utils modules.

### Package structure

```
ttl-sim/
  pyproject.toml
  src/
    sim/           # copied or symlinked from python/sim/
    pykernel/      # copied or symlinked from python/pykernel/
    utils/         # copied or symlinked from python/utils/
```

### `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "ttl-sim"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    # sim-only deps — see index.md "What the Simulator Actually Needs"
]
```

Because there are no C extensions, this produces a `py3-none-any` universal wheel. The build takes seconds and requires only `setuptools`.

### Pros

- **Simplest build.** No CMake, no compiled extensions, no platform-specific wheels.
- **Independent release cadence.** Sim-only fixes ship without waiting for a compiler release.
- **Clear separation of concerns.** Users who only need the simulator never encounter compiler-related errors or dependencies.
- **Small wheel size.** Roughly 100KB of Python source plus metadata, versus the multi-hundred-MB full `ttl` wheel.

### Cons

- **Code duplication.** The `sim/`, `pykernel/`, and `utils/` source files exist in two packages. Without careful automation (CI sync, monorepo tooling), they will drift.
- **Two packages to maintain.** Separate release workflows, separate version numbers, separate PyPI entries.
- **Import conflict risk.** If a user installs both `ttl` and `ttl-sim`, both provide the `sim` package at the top level. Python's import system will pick one unpredictably. This requires either namespacing (e.g., `ttl_sim.sim`) or an explicit `Conflicts-With` mechanism (which pip does not enforce).

### Mitigation for code duplication

The `ttl-sim` wheel can be built from the same monorepo by pointing setuptools at the existing `python/sim/`, `python/pykernel/`, and `python/utils/` directories using `package_dir`:

```python
# python/setup_sim.py
setup(
    name="ttl-sim",
    packages=["sim", "pykernel", "pykernel._src", "utils"],
    package_dir={
        "sim": "sim",
        "pykernel": "pykernel",
        "pykernel._src": "pykernel/_src",
        "utils": "utils",
    },
)
```

This eliminates file duplication at the source level, though it still produces a separate distributable artifact.

---

## Option B -- Extras Group (`pip install ttl[sim]` / `pip install ttl[full]`)

Use a single `ttl` package with optional dependency groups. The base install includes only pure-Python modules; users opt into the compiler via an extras specifier.

### Dependency layout

```toml
[project]
name = "ttl"
dependencies = [
    # sim-only deps — see index.md "What the Simulator Actually Needs"
]

[project.optional-dependencies]
compiler = [
    "ttl-toolchain",       # pre-built LLVM/tt-metal (from Ch6)
    "ml_dtypes>=0.1.0",
    "loguru>=0.6.0",
]
full = [
    "ttl[compiler]",       # alias for convenience
]
dev = [
    "ttl[full]",
    "pytest>=7.0",
    "black",
    "pyright",
]
```

### How the wheel is built

The key challenge is that `pip install ttl` (without `[full]`) should not trigger C++ compilation. This requires conditional extension building in `setup.py` or `pyproject.toml`:

```python
# python/setup.py
import os

ext_modules = []
cmdclass = {}

if os.environ.get("TTLANG_SIM_ONLY") != "ON":
    # Only attempt C++ build when not in sim-only mode
    # AND when the toolchain is available
    toolchain_dir = os.environ.get("TTLANG_TOOLCHAIN_DIR")
    if toolchain_dir and os.path.isdir(toolchain_dir):
        ext_modules = [TTLangExtension("ttl")]
        cmdclass = {"build_ext": CMakeBuild}

setup(
    name="ttl",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    # ...
)
```

However, this creates an awkward situation: the *same* package name `ttl` produces different wheel contents depending on build-time environment variables. A `ttl` wheel built with sim-only will lack `_ttlang.so`, while one built with the toolchain will include it. Both claim to be `ttl` version X.Y.Z.

### Pros

- **Single package identity.** One name on PyPI, one `pip install` command, one `import ttl`.
- **Natural upgrade path.** A user who starts with sim-only can later install the compiler with `pip install ttl[compiler]` (assuming the toolchain wheel is available).
- **Extras are a well-understood pattern.** Projects like `pandas[sql]`, `httpx[http2]`, and `celery[redis]` use this widely.

### Cons

- **Compiled extensions in a "base" install are ambiguous.** If the base `ttl` wheel on PyPI is built without extensions, users who `pip install ttl` and then try `from ttl import _ttlang` get an `ImportError`. The error message must be clear and actionable.
- **Two distinct wheel flavors under one name.** PyPI does not support uploading two wheels with the same name-version-platform but different contents. The project must decide: does PyPI host the sim-only wheel or the full wheel?
- **Extras cannot add compiled extensions.** `[project.optional-dependencies]` adds Python package dependencies, not compiled extensions. The compiler extras (`ttl[compiler]`) can pull in `ttl-toolchain`, but the nanobind extensions (`_ttlang.so`, `_ttmlir.so`) must either be in the base wheel (making it non-pure) or in a separate extension package. Refining this to split extensions into a separate `ttl-compiler` package effectively collapses back into Option A.

---

## Option C -- Build-Time Flag

Keep a single `ttl` package and use the environment variable `TTLANG_SIM_ONLY=ON` during `pip install` to skip extension compilation.

### Usage

```bash
# Sim-only (no compiler, no toolchain needed)
TTLANG_SIM_ONLY=ON pip install .

# Full (requires toolchain)
TTLANG_TOOLCHAIN_DIR=/opt/ttlang-toolchain pip install .
```

### Implementation

The `setup.py` `CMakeBuild` class checks the environment variable:

```python
class CMakeBuild(build_ext):
    def run(self):
        if os.environ.get("TTLANG_SIM_ONLY") == "ON":
            # Skip all extension compilation
            return
        for ext in self.extensions:
            self.build_(ext)
```

The resulting wheel includes the pure-Python packages (`sim/`, `pykernel/`, `utils/`, `ttl/`) but no `.so` files. Dependencies could also be conditioned, though this is harder -- `pyproject.toml`'s `[project.dependencies]` is static.

### Pros

- **Single package, single command.** No new package names, no extras syntax to learn.
- **Matches existing CMake pattern.** Developers already familiar with `TTLANG_SIM_ONLY` in CMake will recognize the convention.
- **Simple implementation.** A few lines of conditional logic in `setup.py`.

### Cons

- **Non-standard and undiscoverable.** Environment variables during `pip install` are not part of Python packaging conventions. Users will not know `TTLANG_SIM_ONLY` exists unless they read documentation.
- **Broken metadata.** The wheel's `METADATA` file will list `build-system.requires = ["setuptools", "cmake", "nanobind", "ninja"]` regardless. A sim-only user still needs `cmake` and `ninja` installed to run `pip install`, even though they are not used, because `pyproject.toml`'s `[build-system].requires` is unconditional.
- **Wheel reuse problems.** `pip` caches wheels by name and version. A sim-only wheel cached as `ttl-0.1.0-cp311-cp311-linux_x86_64.whl` will be reused for a later full install, silently missing the extensions.
- **No `pip install ttl` from PyPI.** A pre-built wheel on PyPI is either sim-only or full. There is no mechanism for pip to rebuild with a flag at install time from a binary wheel.

---

## Comparison Summary

| Criterion | A: Separate `ttl-sim` | B: Extras group | C: Build flag |
|-----------|----------------------|----------------|---------------|
| pip installable (no CMake) | Yes | Partially (base only) | No (`cmake` in build-requires) |
| Single package name | No | Yes | Yes |
| PyPI publishable | Yes (pure wheel) | Yes (but wheel content ambiguity) | No (flag not portable) |
| No code duplication | With `package_dir` trick | Yes | Yes |
| Clear user experience | `pip install ttl-sim` | `pip install ttl` vs `ttl[full]` | `TTLANG_SIM_ONLY=ON pip install .` |
| Import conflict risk | Yes (`sim` namespace) | No | No |
| Extras can add extensions | N/A | No (needs sub-package) | N/A |

---

## Recommended Approach

**Option A (separate `ttl-sim` package) with monorepo source sharing**, refined with a namespace to prevent import conflicts.

### Rationale

1. **Pure-Python wheels are the right artifact for a pure-Python package.** The simulator has zero compiled code. Packaging it as a pure wheel (`py3-none-any`) means it installs instantly on any platform, requires no build tools, and can be hosted on PyPI without platform-specific builds.

2. **Options B and C cannot cleanly separate compiled and pure content.** Option B's extras mechanism can add *dependencies* but not compiled extensions. Option C requires build tools even for sim-only installs and produces ambiguous cached wheels. Both options conflate two fundamentally different distribution types (pure Python vs. platform-specific binary) under one package name.

3. **Code duplication is solvable.** By building `ttl-sim` from the same monorepo source tree using `package_dir` (as shown in Option A's mitigation section), there is no file duplication. A single CI job produces the `ttl-sim` wheel from `python/sim/`, `python/pykernel/`, and `python/utils/` directly.

4. **The import conflict is solvable with namespacing.** Instead of top-level `sim` and `pykernel` packages, both `ttl` and `ttl-sim` should use a `ttl.sim`, `ttl.pykernel` namespace. The recommended approach is to make the full `ttl` package depend on `ttl-sim` (step 5 below), so `ttl-sim` owns the `ttl.sim` and `ttl.pykernel` subpackages while `ttl` owns the compiler-related subpackages and `ttl/__init__.py`. This avoids the complications of implicit namespace packages (which require omitting `__init__.py`, breaking regular package semantics) by using a straightforward dependency relationship instead.

### Concrete next steps

1. **Restructure `python/sim/` to `python/ttl/sim/`** (and similarly for `pykernel/` and `utils/`). Update all internal imports from `from sim.foo import bar` to `from ttl.sim.foo import bar`. This is a prerequisite that benefits the full package as well -- [Chapter 2](../ch2_python_packaging_as_is/index.md) already identified the flat top-level namespace as a packaging issue.

2. **Create `python/setup_sim.py`** (or a separate `pyproject-sim.toml`) that builds the `ttl-sim` wheel from the shared source. Use `package_dir` to point at the existing files.

3. **Add a CI job** that builds and publishes `ttl-sim` on every release tag, using the same version number as the full `ttl` package.

4. **Deprecate the CMake `TTLANG_SIM_ONLY` option** in favor of `pip install ttl-sim`. The CMake path can remain for backward compatibility but should print a deprecation notice pointing users to the pip package.

5. **Make the full `ttl` package depend on `ttl-sim`** (or at minimum declare it as a known alternative) so that `pip install ttl` automatically includes the simulator, while `pip install ttl-sim` gives the lightweight option.

### Relationship to the two-phase architecture

The [two-phase wheel architecture](../ch6_two_phase_wheel_architecture/index.md) from Chapter 6 already splits the build into `ttl-toolchain` (LLVM + tt-metal) and `ttl` (extensions + Python). Adding `ttl-sim` creates a three-tier hierarchy:

```
ttl-sim          Pure Python: sim, pykernel, utils
  ^
  |  (depends on)
ttl              Extensions: _ttlang.so, _ttmlir.so, dialect bindings + ttl-sim
  ^
  |  (depends on)
ttl-toolchain    Pre-built: LLVM, MLIR, tt-metal shared libraries
```

A user who needs only the simulator installs one package. A user who needs the compiler installs `ttl`, which transitively pulls in both `ttl-sim` and `ttl-toolchain`. This is clean, standard, and discoverable.

---

**End of guide.** Return to [Guide Index](../index.md)
