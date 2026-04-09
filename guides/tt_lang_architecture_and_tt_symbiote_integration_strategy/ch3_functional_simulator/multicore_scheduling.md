# Multi-Core Scheduling

The simulator must emulate dozens of concurrent threads -- three per core (compute, dm0, dm1) across an entire grid -- without real OS threads. TT-Lang achieves this with **cooperative scheduling** built on Python greenlets.

All source references point into `python/sim/greenlet_scheduler.py`, `python/sim/program.py`, and `python/sim/decorators.py`.

## GreenletScheduler Overview

`GreenletScheduler` (in `greenlet_scheduler.py`) is the heart of the simulator's concurrency model. Each thread function runs inside its own `greenlet` -- a user-space coroutine that can be suspended and resumed without OS involvement.

```python
class GreenletScheduler:
    def __init__(self):
        self._active: Dict[str, Tuple[greenlet, Any, str, ThreadType, str, Optional[Tuple[str, int]]]]
        self._completed: List[str]
        self._main_greenlet: Optional[greenlet]
        self._current_name: Optional[str]
        self._last_run: Dict[str, int]      # thread_name -> timestamp (for fair scheduling)
        self._timestamp: int                  # global monotonic counter
        self._has_made_progress: Dict[str, bool]
```

Key concepts:

- **`_active`** maps thread names (e.g., `"core3-compute"`) to a tuple of the greenlet, its current blocking object, the blocked operation name, the thread type, and source location info.
- **`_main_greenlet`** is the scheduler's own greenlet. When a thread blocks, it switches back here.
- **`_timestamp`** is a monotonically increasing counter used by the fair scheduling algorithm.

### Adding Threads

```python
scheduler.add_thread("core0-compute", bound_func, ThreadType.COMPUTE)
```

Each call wraps `bound_func` in a greenlet. The wrapped function calls `self._mark_completed(name)` on successful return, which removes the thread from `_active` and appends it to `_completed`.

### Blocking: `block_current_thread()`

When a thread calls `dfb.wait()` or `dfb.reserve()` and the DFB cannot immediately satisfy the request, it calls:

```python
scheduler.block_current_thread(blocking_obj=dfb, operation="wait")
```

This records the blocking object and operation in the `_active` entry, captures the user-code source location for diagnostics, and **switches to `_main_greenlet`** -- returning control to the scheduler's `run()` loop.

### The `block_if_needed()` Free Function

`block_if_needed(obj, operation)` is the primary synchronization primitive called by DFB operations. Its behavior varies by scheduling algorithm:

**Greedy mode:**
1. Calls `obj.can_{operation}()`. If `True`, marks progress and returns immediately.
2. If `False`, calls `scheduler.block_current_thread()` to yield.

**Fair mode:**
1. Marks progress, then **always yields** via `block_current_thread()` -- even if the operation could proceed -- to give other threads a chance to run.
2. On resume, re-checks `can_{operation}()`. If still blocked, yields again.

This makes fair mode strictly more conservative about scheduling order, which is useful for catching concurrency bugs that greedy mode might mask.

## Scheduling Algorithms

The algorithm is configured globally via `set_scheduler_algorithm()` and defaults to `"fair"`:

```python
set_scheduler_algorithm("greedy")  # or "fair" (default)
```

### Greedy Algorithm

Threads are tried in dictionary insertion order. A thread runs until it blocks, and the scheduler immediately tries the next active thread. Simple and fast, but the execution order is sensitive to thread registration order.

### Fair Algorithm

The fair algorithm has two phases:

**Phase 1 -- Initialization (`_initialization_phase()`):**
Every thread is run sequentially until it first blocks. This ensures all threads have their `blocking_obj` set so that `can_{operation}()` checks work correctly from the start. Threads that make progress (pass at least one `block_if_needed` check) receive a timestamp; threads that block immediately keep timestamp 0, giving them priority.

**Phase 2 -- Main loop:**
Threads are sorted by `_last_run` timestamp in ascending order (least recently run first). Ties are broken alphabetically by name for determinism. For each candidate:

1. If the thread is blocked, call `blocking_obj.can_{blocked_op}()`. Skip if still blocked.
2. If unblocked, switch into its greenlet.
3. After the greenlet yields or completes, increment `_timestamp` and record it against the thread.

The fair algorithm approximates round-robin scheduling across all cores and threads, preventing any single core from monopolizing execution.

## Deadlock Detection

If a full pass over all active threads makes **zero progress** (no thread was unblocked or advanced), the scheduler declares a deadlock:

```python
if not any_progress and self._active:
    # Group blocked threads by (operation, object, location)
    # Print diagnostic with core ranges
    raise RuntimeError("Deadlock detected: all generators blocked")
```

The diagnostic groups threads that are blocked on the same operation at the same source location and formats core ranges compactly (e.g., `cores: 0-15`). Each group gets a pretty-printed error with the source file and line number via `print_diagnostic_error()`.

## Program: Binding Threads to Cores

`Program` (in `program.py`) bridges the gap between the kernel's thread templates and the scheduler. It is constructed by `operation()` with the three ordered thread templates and the grid shape.

### Per-Core Context Building

For each core in `range(total_cores)` (where `total_cores = prod(grid)`), `Program._build_core_context()` creates an isolated context dictionary:

- **`Tensor` arguments** are shared across cores (same input data).
- **`DataflowBuffer` instances** are **freshly constructed** per core -- each core gets its own independent ring buffer with the same configuration as the original.
- **All other values** are `copy.deepcopy()`-ed to prevent cross-core interference.
- **Module objects** are shared by reference (no need to copy).
- The per-core context also injects `_core` (the core index), `grid`, and a custom `print` function for debug output.

### Thread Registration via Decorators

The `@compute()` and `@datamovement()` decorators (in `decorators.py`) each create a template class (`ComputeTemplate` / `DMTemplate`) that:

1. Stores the original function as `__wrapped__` and sets a `thread_type` class attribute (`ThreadType.COMPUTE` or `ThreadType.DM`).
2. Provides a `bind(ctx)` method that calls `rebind_func_with_ctx(func, ctx)` -- this creates a new function object with globals and closure cells replaced by the per-core context, so that names like `out_dfb` resolve to the core-local DFB.
3. Registers itself in the global thread registry via `_register_thread()`.

### Execution Flow

```
operation() wrapper
  |-- executes kernel body (registers 3 thread templates)
  |-- validates: exactly 1 compute + 2 DM threads
  |-- Program(compute_tmpl, dm0_tmpl, dm1_tmpl, grid=(R, C))
       |-- _run_cooperative(total_cores=R*C, ...)
            |-- checks DFB count and L1 limits (warnings)
            |-- for each core:
            |     build per-core context
            |     bind each template to context
            |     scheduler.add_thread("core{N}-compute", ...)
            |     scheduler.add_thread("core{N}-dm0", ...)
            |     scheduler.add_thread("core{N}-dm1", ...)
            |-- scheduler.run()
            |-- validate_dataflow_buffers()  # no pending blocks
```

### Post-Execution Validation

After `scheduler.run()` completes, `Program._validate_dataflow_buffers()` iterates over every `DataflowBuffer` in every core context and calls `validate_no_pending_blocks()`. This catches a common bug where a block's data was used as an arithmetic operand (`assign_src`) but the result was never stored -- the block's `_store_confirmation_pending` flag would still be set.

## Error Reporting

When a thread raises an exception, `_format_and_raise_thread_error()` walks the traceback to find the first non-simulator frame (user code) and calls `print_diagnostic_error()` with the file, line, and column. The re-raised `RuntimeError` includes the thread name (e.g., `core5-dm0: RuntimeError: ...`) so the developer knows exactly which core and thread type failed.

---

**Next:** [`resource_limits.md`](./resource_limits.md)
