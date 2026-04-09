# Decorators and Threads

TT-Lang uses three decorator tiers to structure every kernel: `@ttl.operation` defines the kernel entry point, while `@ttl.compute` and `@ttl.datamovement` mark the inner thread functions that run on each Tensix core's processors.

## The Three Decorator Tiers

### `@ttl.operation(grid=...)` — Kernel Entry Point

The outermost decorator wraps a function that accepts `ttnn.Tensor` arguments and defines the kernel's dataflow buffers and threads. Its `grid` parameter specifies the core grid topology.

```python
@ttl.operation(grid="auto")
def eltwise_add(a_in: ttnn.Tensor, b_in: ttnn.Tensor, out: ttnn.Tensor) -> None:
    # 1. Compute tiling parameters
    # 2. Create dataflow buffers
    # 3. Define @ttl.compute and @ttl.datamovement closures
    ...
```

**Source:** `python/sim/operation.py`

Under the hood, `@ttl.operation` performs the following steps:

1. **Grid resolution.** If `grid="auto"`, the actual grid shape is read from `get_context().config.default_auto_grid`. Otherwise, the user-supplied tuple (e.g., `(8, 4)`) is used directly.
2. **Function rewriting.** A new `FunctionType` is created with `grid` injected into its globals, so inner code can reference it as a bare name.
3. **Thread collection.** When the decorated function body executes, each `@ttl.compute` and `@ttl.datamovement` closure self-registers into the thread registry.
4. **Validation.** Exactly 3 threads must be registered: 1 compute + 2 data-movement. This mirrors the Tensix core's physical RISC-V processor layout.
5. **Program construction.** A `Program` object is created with the ordered thread templates and the resolved grid, then invoked with the tensor arguments.

```python
# From sim/operation.py — thread count enforcement
if len(compute_threads) != 1:
    raise ValueError(
        f"Kernel must define exactly 1 compute thread, got {len(compute_threads)}"
    )
if len(dm_threads) != 2:
    raise ValueError(
        f"Kernel must define exactly 2 datamovement threads, got {len(dm_threads)}"
    )
```

### `@ttl.compute()` — Compute Thread

Marks a closure as the compute thread. This thread performs arithmetic on blocks acquired from dataflow buffers.

```python
@ttl.compute()
def compute():
    with (
        a_dfb.wait() as a_blk,
        b_dfb.wait() as b_blk,
        out_dfb.reserve() as out_blk,
    ):
        out_blk.store(a_blk + b_blk)
```

**Source:** `python/sim/decorators.py`

The decorator creates a `ComputeTemplate` class whose `thread_type` attribute is `ThreadType.COMPUTE`. The template registers itself in the thread registry immediately upon decoration.

```python
# From sim/decorators.py
def compute() -> Callable[[FunctionType], BindableTemplate]:
    def decorator(func: FunctionType) -> BindableTemplate:
        class ComputeTemplate:
            __name__ = func.__name__
            __wrapped__ = func
            thread_type = ThreadType.COMPUTE

            def bind(self, ctx: Dict[str, Any]) -> Callable[[], Any]:
                bound_func = rebind_func_with_ctx(func, ctx)
                return bound_func

        template = ComputeTemplate()
        _register_thread(template)
        return template
    return decorator
```

### `@ttl.datamovement()` — Data Movement Thread

Marks a closure as a data-movement thread. Each kernel must define exactly two of these — conventionally named `read` (DM0) and `write` (DM1).

```python
@ttl.datamovement()
def read():
    with a_dfb.reserve() as a_blk:
        tx = ttl.copy(a_in[r0:r1, col:col+1], a_blk)
        tx.wait()

@ttl.datamovement()
def write():
    with out_dfb.wait() as out_blk:
        tx = ttl.copy(out_blk, out[r0:r1, col:col+1])
        tx.wait()
```

The `DMTemplate` is structurally identical to `ComputeTemplate` but carries `ThreadType.DM`. Both DM threads share the same template class; they are distinguished only by their position in the registry (DM0 vs DM1).

## Thread Registration Mechanism

Registration uses a context-local list stored in the simulation's global context object:

```python
# From sim/decorators.py
def _register_thread(thread_template: BindableTemplate) -> None:
    """Register a thread template during decoration."""
    get_context().thread_registry.append(thread_template)

def get_registered_threads() -> List[BindableTemplate]:
    """Get all registered threads and clear the registry."""
    registry = get_context().thread_registry
    threads = list(registry)
    registry.clear()
    return threads
```

The lifecycle is:

1. `@ttl.operation` calls `clear_thread_registry()` before executing the kernel body.
2. Each `@ttl.compute()` / `@ttl.datamovement()` decoration appends a template.
3. After the kernel body completes, `get_registered_threads()` harvests all three and clears the registry.

## The `BindableTemplate` Protocol

Every thread template must satisfy the `BindableTemplate` protocol (defined in `sim/typedefs.py`). The key method is `bind(ctx)`, which takes a per-core context dictionary and returns a zero-argument callable:

```
bind(ctx: Dict[str, Any]) -> Callable[[], Any]
```

The `ctx` dictionary contains:
- `_core` — linear core index (0 to $N-1$ where $N$ is total cores)
- `grid` — the grid tuple (e.g., `(8, 4)`)
- Per-core copies of all `DataflowBuffer` objects
- Shared references to input/output `Tensor` objects
- A custom `print` function for debug output

### `rebind_func_with_ctx` — Per-Core Closure Binding

The critical mechanism that makes DFB-per-core isolation work is `rebind_func_with_ctx`. When `Program` binds a template for core $i$, it must replace the closure cells that the inner function captured (e.g., `a_dfb`, `out_dfb`) with fresh, core-local copies.

```python
# From sim/decorators.py
def rebind_func_with_ctx(func: FunctionType, ctx: Dict[str, Any]) -> FunctionType:
    """
    Create a new function from `func` but with:
      - globals = func.__globals__ + ctx
      - closure cells rebuilt from ctx when possible
    """
    freevars = func.__code__.co_freevars
    orig_closure = func.__closure__ or ()
    new_cells = []
    for name in freevars:
        if name in ctx:
            new_cells.append(_make_cell(ctx[name]))
        else:
            new_cells.append(orig_cell_map[name])

    new_globals = dict(func.__globals__)
    new_globals.update(ctx)

    return types.FunctionType(
        func.__code__, new_globals, func.__name__,
        func.__defaults__, tuple(new_cells)
    )
```

This ensures that when core 0's `read()` thread references `a_dfb`, it gets core 0's private `DataflowBuffer` instance — not a shared object.

## Simulator vs Compiler Equivalents

For the dual-backend architecture overview, see [`index.md`](./index.md#position-in-the-software-stack). The decorator-specific differences are:

| Aspect | Simulator (`sim/`) | Compiler (`ttl/ttl_api.py`) |
|--------|--------------------|-----------------------------|
| `@ttl.compute()` | Creates `ComputeTemplate` with `ThreadType.COMPUTE` | Collects decorated function into `_thread_registry` for AST compilation |
| `@ttl.datamovement()` | Creates `DMTemplate` with `ThreadType.DM` | Same collection mechanism, tagged as DM |
| `@ttl.operation(grid=...)` | Resolves grid, executes body, builds `Program`, runs `GreenletScheduler` | Resolves grid, compiles threads to MLIR, generates C++ kernel sources |
| Thread execution | Greenlet-based cooperative scheduling across all cores | Hardware: each core's 3 RISC-V processors run in parallel |

## Thread Execution Model

Once `Program` has bound all templates, it schedules execution using the `GreenletScheduler`:

```python
# From sim/program.py (simplified)
for core in range(total_cores):
    core_context = self._build_core_context(core)
    for name, tmpl in [("compute", compute_tmpl), ("dm0", dm0_tmpl), ("dm1", dm1_tmpl)]:
        bound_func = tmpl.bind(core_context)
        scheduler.add_thread(f"core{core}-{name}", bound_func, thread_type)
scheduler.run()
```

For a grid of $R \times C$ cores, this creates $3 \times R \times C$ greenlet threads that cooperatively yield at blocking points (`dfb.wait()`, `dfb.reserve()`, `tx.wait()`). This faithfully simulates the concurrent execution model of the real hardware.

---

**Next:** [`dataflow_buffers.md`](./dataflow_buffers.md)
