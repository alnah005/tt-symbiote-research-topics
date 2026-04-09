# Chapter 2 — TT-Lang Compilation Pipeline

This chapter traces the full lowering path that transforms a Python DSL kernel into C++ source code ready for JIT execution on Tenstorrent hardware. The pipeline spans four distinct stages:

1. **Python to MLIR** — A Python AST visitor (`TTLGenericCompiler`) translates decorated kernel functions into the TTL MLIR dialect.
2. **MLIR Pass Pipeline** — A 17-pass sequence lowers TTL MLIR through the Compute and TTKernel dialects, performing DST allocation, subblock tiling, loop generation, hardware init insertion, and EmitC conversion.
3. **C++ Codegen** — EmitC output is serialized to C++ source files, written to disk, and wrapped in kernel descriptors.
4. **JIT Execution** — `CompiledTTNNKernel` builds `KernelSpec` objects and dispatches to the device via `ttnn.generic_op`.

## Entry Point: `pykernel_gen`

The user-facing decorator `@ttl.pykernel_gen` (aliased as `@ttl.operation`) is defined in `python/ttl/ttl_api.py`. It accepts grid dimensions, indexing maps, iterator types, memory space, and compiler options, then returns a wrapper that:

1. Resolves the grid (static tuple, callable, or `"auto"` device query).
2. Merges `CompilerOptions` from three priority tiers: `sys.argv` > `TTLANG_COMPILER_OPTIONS` env var > decorator `options=` string.
3. Checks a per-kernel compilation cache keyed on tensor shapes, dtypes, memory spaces, layouts, mesh shape, and compiler options.
4. On cache miss, calls `_compile_kernel()` to run the full pipeline.
5. Executes the resulting `CompiledTTNNKernel` (unless `TTLANG_COMPILE_ONLY=1`).

```
@ttl.pykernel_gen(grid=(4, 4), num_outs=1, memory_space="L1")
def my_kernel(inp, out):
    @ttl.compute()
    def compute_thread(inp, out):
        ...
    @ttl.datamovement()
    def dm_reader(inp, out):
        ...
    @ttl.datamovement()
    def dm_writer(inp, out):
        ...
```

## Pipeline Architecture

```
  Python source
       │
       ▼
  ast.parse() ─────────────────────────► Python AST
       │
       ▼
  TTLGenericCompiler.visit()  ──────────► TTL MLIR Module
       │                                  (one func.func per thread)
       ▼
  PassManager.parse(pipeline_str) ──────► 17 MLIR passes
       │
       ├─ TTL Dialect passes (compute config, DST, subblocks, loops, CB annotations)
       ├─ TTKernel Dialect passes (inits, L1 accumulation, pack combining)
       ├─ Standard passes (canonicalize, CSE, lower-affine)
       └─ EmitC conversion (convert-ttkernel-to-emitc, symbol-dce)
       │
       ▼
  ttkernel_to_cpp_by_name() ────────────► C++ source strings
       │
       ▼
  _write_kernel_to_tmp() ──────────────► /tmp/{user}/ttlang_kernel_{name}_{hash}.cpp
       │
       ▼
  CompiledTTNNKernel ──────────────────► KernelSpec + ttnn.generic_op dispatch
```

## Chapter Files

| File | Content |
|------|---------|
| [`python_to_mlir.md`](./python_to_mlir.md) | `TTLGenericCompiler`: Python AST visitor, capture collection, tensor type construction, source location tracking |
| [`mlir_passes.md`](./mlir_passes.md) | The full 17-pass pipeline from `_compile_kernel()`, each pass with its C++ source and purpose. `CompilerOptions` dataclass and priority merge. |
| [`codegen_and_execution.md`](./codegen_and_execution.md) | C++ source generation, kernel file writing, `CompiledTTNNKernel`, `KernelSpec`, `run_kernel_on_device()`, compilation caching |

## Key Takeaways

- The compilation pipeline is a single-function entry point (`pykernel_gen`) that orchestrates AST parsing, MLIR lowering, pass execution, codegen, and device dispatch.
- Thread functions decorated with `@ttl.compute()` and `@ttl.datamovement()` are compiled independently by `TTLGenericCompiler`, then merged into a single MLIR module before the pass pipeline runs.
- The 17-pass MLIR pipeline is ordered carefully: TTL-level transformations (DST allocation, subblock tiling, loop lowering) precede TTKernel-level transformations (hardware init insertion, pack combining) which precede EmitC conversion.
- `CompilerOptions` is a frozen, hashable dataclass with a three-tier priority merge (`sys.argv` > env var > decorator), making it safe to use as a cache key component.
- Compilation results are cached per-kernel by tensor metadata and compiler options, so repeated calls with the same shapes and dtypes skip recompilation entirely.
- The final execution path goes through `run_kernel_on_device()`, which builds `KernelDescriptor`, `CBDescriptor`, and `ProgramDescriptor` objects for `ttnn.generic_op`.
