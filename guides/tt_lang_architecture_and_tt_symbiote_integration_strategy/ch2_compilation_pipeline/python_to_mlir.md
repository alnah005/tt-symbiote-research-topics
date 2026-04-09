# Python to MLIR: TTLGenericCompiler

**Source:** `python/ttl/_src/ttl_ast.py`

This stage transforms a Python function body into TTL MLIR operations. The compiler is an AST visitor that walks the parsed Python source and emits MLIR ops into an in-memory module.

## Compilation Trigger

When a thread-decorated function (e.g., `@ttl.compute()`, `@ttl.datamovement()`) is called, the `_compile()` decorator in `ttl_api.py` performs these steps:

```python
# In _compile() -> _decorator -> _wrapper:
source_code = _cleanup_source_code(f)        # Strip decorators, normalize indentation
m = ast.parse(source_code)                    # Python AST

b = TTLGenericCompiler(
    f.__name__,
    kernel_type,                              # "compute" or "datamovement"
    _collect_captures(f),                     # Closure variables
    *args,
    _globals=f.__globals__,
    _source_file=source_file,
    _source_lines=source_lines,
    _line_offset=_get_source_line_offset(f),
    debug_locations=True,
    **kwargs,                                 # grid, memory_space, tiled
)

b.visit(m)                                    # Walk AST, emit MLIR
b.module.operation.verify()                   # Validate MLIR
```

## TTLGenericCompiler Class

`TTLGenericCompiler` extends `TTCompilerBase` (from `pykernel._src.kernel_ast`) and maintains:

| Field | Purpose |
|-------|---------|
| `self.context` | `CompilerContext(grid, memory_space, tiled)` — immutable compilation context |
| `self.captures` | Dict of captured closure variables (ints, floats, `CircularBuffer`, ttnn tensors) |
| `self.fn_globals` | Function's `__globals__` dict for resolving module-level constants |
| `self.streams` | Set of stream names encountered during compilation |
| `self._cb_info` | List of CB metadata dicts: `{name, shape, element_type, cb_index}` |
| `self._fn_map` | Registry of syntax handlers (populated from `TTLGenericCompiler._syntax` class variable) |
| `self.source_file` | Path to the original Python source (for error reporting) |
| `self.line_offset` | Offset to convert parsed AST line numbers to actual file line numbers |

## Capture Collection

Before AST compilation, `_collect_captures(f)` inspects the function's `__closure__` and converts each cell:

```python
def _collect_captures(f):
    if f.__closure__ is None:
        return {}
    return {
        n: convert(n, c.cell_contents)
        for n, c in zip(f.__code__.co_freevars, f.__closure__)
    }
```

Supported capture types:
- `int`, `float` — passed as MLIR constants
- `CircularBuffer` — mapped to MLIR CB references (with `_cb_index` attribute)
- ttnn tensors — passed through for shape/dtype extraction

Unsupported types raise `TypeError`.

## Tensor Type Construction

The function `_build_tensor_type()` constructs MLIR `RankedTensorType` with Tenstorrent-specific layout encoding:

```python
def _build_tensor_type(ctx, tensor, grid, tiled, memory_space):
    # 1. Validate: must be tiled, L1 or DRAM, 2D grid, >= 2D shape
    # 2. Detect memory layout (interleaved, sharded) from ttnn tensor
    # 3. Build LayoutConfig with logical shape, grid, dtype, memory layout
    # 4. Create TileType element (32x32 tiles by default)
    # 5. Compute device shape: batch dims + ceil(rows/32) x ceil(cols/32)
    return RankedTensorType.get(device_shape, element_type, layout)
```

For a tensor with shape `[B, M, N]` and 32x32 tiles:

$$\text{device\_shape} = \left[B, \left\lceil \frac{M}{32} \right\rceil, \left\lceil \frac{N}{32} \right\rceil\right]$$

The `TTLLayoutAttr` encoding carries grid dimensions, memory layout (interleaved vs. sharded), and data type through all subsequent MLIR passes.

## Source Location Tracking

Every MLIR operation receives an `mlir.Location` for error diagnostics. The compiler provides two location modes:

1. **File locations** (always enabled for error messages): `Location.file(source_file, lineno + offset, col + 1, ctx)`
2. **Name locations** (fallback): `Location.name(function_name)`

The helper `_make_file_loc()` converts Python AST node positions to MLIR file locations:

```python
def _make_file_loc(ctx, source_file, node, line_offset=0):
    return Location.file(
        source_file, node.lineno + line_offset, node.col_offset + 1, ctx
    )
```

When a pass fails, the error handler in `_compile_kernel()` calls `format_mlir_error()` which maps MLIR locations back to Python source lines, producing user-friendly error messages with source context.

## AST Visitor Methods

Key visitor overrides in `TTLGenericCompiler`:

| Method | Purpose |
|--------|---------|
| `visit_Assign` | Handles tuple unpacking (e.g., `cx, cy = core(dims=2)`) |
| `visit_Call` | Dispatches function calls via `_fn_map`, injects auto-profiling signposts |
| `visit_BinOp` | Maps Python `+`, `-`, `*` to TTL dialect ops |
| `visit_Print` | Translates `print()` with keyword args to `ttl.dprint` ops |

The `_fn_map` registry maps Python function names to MLIR emission handlers. For example, calling `cb.wait()` in Python dispatches to a handler that emits `ttl.cb_wait` in MLIR.

## Auto-Profiling Signpost Injection

When `TTLANG_AUTO_PROFILE=1` is set, the compiler emits signpost operations at Python source line boundaries:

```python
def _emit_line_signpost_if_needed(self, node):
    file_lineno = node.lineno + self.line_offset
    if self._current_signpost_line == file_lineno:
        return
    # Close previous signpost
    self._emit_signpost(f"{self.name}_L{self._current_signpost_line}", is_end=True)
    # Open new signpost
    self._emit_signpost(f"{self.name}_L{file_lineno}")
```

These signposts survive through the pass pipeline (lowered to EmitC verbatim calls by `ttl-lower-signpost-to-emitc`) and appear in hardware profiler traces, enabling line-level performance attribution.

## Output: Per-Thread MLIR Modules

Each thread decorator call produces a `TTLGenericCompiler` instance with a `func.func` operation containing the thread's MLIR. The parent `_compile_kernel()` function then:

1. Collects all compiled threads from the thread registry
2. Sets `ttl.base_cta_index` and `ttl.crta_indices` attributes on each function (for tensor accessor lowering)
3. Merges all `func.func` operations into a single `builtin.module`
4. Passes the module to the MLIR pass pipeline

The merged module typically contains exactly 3 functions: one compute thread and two data movement threads (reader + writer), matching the Tensix core's TRISC + NCRISC + BRISC thread model described in [Chapter 1](../ch1_programming_model/index.md).

---

**Next:** [`mlir_passes.md`](./mlir_passes.md)
