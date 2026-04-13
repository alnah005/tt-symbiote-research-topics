# Chapter 4 -- Prior Art: MLIR-Based `pip install` Approaches

Before designing a `pip install` strategy for TT-Lang, it is worth studying how other MLIR-based projects have solved the same fundamental problem: shipping a compiled MLIR/LLVM stack inside a Python wheel. This chapter surveys four prominent projects -- torch-mlir, Triton, IREE, and CIRCT -- and distills the recurring patterns they share.

## Projects Surveyed

- **torch-mlir** -- custom `CMakeBuild` in `setup.py`; in-tree LLVM submodule built in CI; ~222 MB wheels
- **Triton** -- auto-downloads pre-built LLVM from Azure Blob Storage; ~188 MB wheels
- **IREE** -- separate compiler (~83 MB) and runtime (~8 MB) wheels; pre-built LLVM in CI
- **CIRCT** -- unified CMake build of LLVM + CIRCT; local wheels only (no PyPI release)

See [`case_studies.md`](./case_studies.md) for full details on each project.

## Common Patterns

Three architectural patterns recur across these projects:

1. **Pre-built LLVM in CI** -- no successful MLIR wheel project builds LLVM from source on the end-user's machine. See [`lessons_learned.md` section 2](./lessons_learned.md#2-the-toolchain-wheel-pattern).
2. **Bundled shared libraries** -- the three PyPI-publishing projects use platform-specific wheel repair tools (`auditwheel`, `delocate`, `delvewheel`) to bundle `.so`/`.dylib`/`.dll` files. See [`lessons_learned.md` section 3](./lessons_learned.md#3-auditwheel-and-rpath-considerations).
3. **Separate toolchain / compiler / runtime wheels** -- the largest projects split deliverables to manage size and rebuild frequency. See [`lessons_learned.md` section 2](./lessons_learned.md#2-the-toolchain-wheel-pattern).

## Chapter Contents

- [`case_studies.md`](./case_studies.md) -- Detailed case studies of torch-mlir, Triton, IREE, and CIRCT packaging
- [`lessons_learned.md`](./lessons_learned.md) -- Cross-cutting lessons on build backends, `auditwheel`, sdist correctness, and the toolchain wheel pattern

**Next:** [`case_studies.md`](./case_studies.md)
