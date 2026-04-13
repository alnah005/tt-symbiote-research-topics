# Chapter 8 -- Sim-Only Installation Mode

The previous chapters designed a full `pip install` path for TT-Lang, culminating in [two-phase wheel architecture](../ch6_two_phase_wheel_architecture/index.md) and [manylinux-compliant packaging](../ch7_wheel_packaging/index.md). That path assumes the user wants the compiler -- nanobind extensions, MLIR dialect bindings, and a pre-built toolchain containing LLVM and tt-metal.

Many users do not need the compiler. The TT-Lang simulator (`python/sim/`) and the pykernel authoring library (`python/pykernel/`) are pure Python. Researchers prototyping kernel algorithms, CI jobs running functional correctness checks, and developers on machines without Tenstorrent hardware all need only the simulator. Forcing them through a toolchain-dependent build is unnecessary friction.

This chapter examines what TT-Lang already provides for this use case and proposes packaging designs that expose a lightweight sim-only installation via standard `pip install` workflows.

## Current Sim-Only Support

TT-Lang already has a CMake option for simulator-only mode. In the root `CMakeLists.txt`:

```cmake
# CMakeLists.txt
option(TTLANG_SIM_ONLY "Set up Python environment for simulator only (skip compiler build)" OFF)
```

When `TTLANG_SIM_ONLY=ON`, the build creates a venv, installs runtime requirements, stubs out compiler variables, and returns before the LLVM/tt-metal build:

```cmake
if(TTLANG_SIM_ONLY)
  # Create venv if it doesn't exist
  _ttlang_find_venv_python("${TTLANG_PYTHON_VENV}" _VENV_PYTHON)
  if(NOT _VENV_PYTHON)
    message(STATUS "Creating Python venv at ${TTLANG_PYTHON_VENV}...")
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -m venv --prompt ttlang "${TTLANG_PYTHON_VENV}"
      RESULT_VARIABLE _VENV_RESULT
    )
    # ... venv creation, pip upgrade ...
  endif()

  # Install runtime requirements (torch, greenlet, pydantic, etc.)
  ttlang_pip_install_requirements("${_VENV_PYTHON}"
    "${CMAKE_SOURCE_DIR}/requirements.txt" FATAL)

  # Set variables needed by env/activate.in to harmless defaults
  set(TT_LANG_HOME "${CMAKE_CURRENT_LIST_DIR}")
  set(TTLANG_HAS_DEVICE_INT 0)
  set(LLVM_INSTALL_DIR "")
  set(TT_METAL_HOME "")
  # ...

  configure_file("${PROJECT_SOURCE_DIR}/env/activate.in"
    "${PROJECT_BINARY_DIR}/env/activate" @ONLY)

  return()   # <-- Skip compiler build entirely
endif()
```

## What the Simulator Actually Needs

The pure-Python packages that constitute the sim-only surface are:

| Package | Source path | Purpose |
|---------|------------|---------|
| `sim` | `python/sim/` | Core simulator: scheduling, dataflow, math ops, torch-based execution |
| `pykernel` | `python/pykernel/` | Kernel AST definition and type system for authoring kernels in Python |
| `utils` | `python/utils/` | Shared utilities: block allocation, correctness checking |

These packages have 24 Python source files in `python/sim/`, 6 in `python/pykernel/`, and 3 in `python/utils/`. None contain C extensions or import from the compiled `_ttlang` or `_ttmlir` nanobind modules.

Their runtime dependencies (drawn from `requirements.txt`) are:

- `torch>=1.9.0` -- the simulator executes operations using PyTorch tensors
- `greenlet>=3.0.0` -- cooperative scheduling for the simulator's coroutine-based execution model
- `pydantic<3` -- data validation for kernel parameters and configuration
- `numpy>=1.20.0` -- array utilities used by sim and utils
- `PyYAML>=5.4.0,<=6.0.1` -- configuration file parsing

Notably absent from this list are the build-only or compiler-only dependencies: `nanobind`, `cmake`, `ninja`, `ml_dtypes`, and the entire LLVM/MLIR/tt-metal stack.

## Limitations of the Current Approach

The existing `TTLANG_SIM_ONLY` mechanism works but has significant limitations in a pip-installable world:

1. **Requires CMake.** Users must run `cmake -B build -DTTLANG_SIM_ONLY=ON` and `source build/env/activate`. This is not a standard Python workflow.
2. **No package metadata.** The resulting venv has the simulator code on `PYTHONPATH` via the activate script, but there is no installed package. `pip list` does not show `ttl` or `sim`.
3. **Full `requirements.txt` install.** The sim-only mode installs *all* runtime requirements, including packages like `ml_dtypes` and `loguru` that are only needed by tt-metal or the compiler's Python bindings.
4. **No versioning or distribution.** There is no wheel, no sdist, and no way to declare a dependency on the sim-only package from another project's `pyproject.toml`.

## Design Goals

A proper sim-only installation mode should:

- Install via `pip install` with no CMake, clang, or system-level toolchain required
- Produce a proper Python package visible in `pip list` and `pip show`
- Pull in only the dependencies the simulator actually needs
- Be publishable to PyPI or an internal package index
- Coexist cleanly with the full compiler package (no import conflicts)

The next section evaluates three design options for achieving these goals.

---

**Next:** [`design_options.md`](./design_options.md)
