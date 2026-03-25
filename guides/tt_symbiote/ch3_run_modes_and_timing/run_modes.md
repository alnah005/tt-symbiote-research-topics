# Run Modes — the eight execution strategies that control how every ATen operation and every module forward pass is dispatched.

TT Symbiote intercepts PyTorch's dispatch mechanism through a `TorchTTNNTensor` wrapper subclass. The selected run mode determines what happens inside `__torch_dispatch__` for each intercepted ATen operation and inside the module-level `module_run` wrapper. Eight modes are registered in `_RUN_MODE_REGISTRY` (defined at line 1167 of `core/run_config.py`).

## Selecting a run mode

### Via environment variable (takes precedence)

```python
export TT_SYMBIOTE_RUN_MODE=NORMAL
```

### Via `set_run_mode()` (programmatic, one-shot)

```python
from models.experimental.tt_symbiote.core.run_config import set_run_mode
set_run_mode("DPL")
```

`set_run_mode` asserts that the mode has not already been set to a different value; it cannot be changed mid-execution. Note: the warning about env var override is printed by `get_tensor_run_implementation()`, not by `set_run_mode` itself — `set_run_mode` simply sets `_current_run_mode` and does not inspect the environment variable.

`get_tensor_run_implementation()` is the internal function that resolves the final choice each time a `TorchTTNNTensor` is constructed. It reads `TT_SYMBIOTE_RUN_MODE` first, then `_current_run_mode`, and defaults to `"NORMAL"` if neither is set.

### Custom modes

```python
from models.experimental.tt_symbiote.core.run_config import add_run_mode
add_run_mode("MY_MODE", MyRunClass)
```

`add_run_mode` raises `ValueError` if the key already exists.

---

## The eight run modes

### Mode registry

```python
_RUN_MODE_REGISTRY = {
    "LIGHTWEIGHT":          LightweightRun,
    "NORMAL":               NormalRun,
    "NORMAL_WITH_FALLBACK": NormalRunWithFallback,
    "SEL":                  SELRun,
    "DPL":                  DPLRun,
    "DPL_NO_ERROR_PROP":    DPLRunNoErrorProp,
    "CPU":                  CPU,
    "TRACED":               TracedRun,
}
```

> **Note:** The plan spec described six modes. The source defines eight. All eight are documented below.

---

### `NORMAL` — `NormalRun`

Full TTNN execution. This is the default mode.

**ATen-level behavior (`torch_dispatch`):**

1. Calls `can_dispatch_to_ttnn(func.name(), args, kwargs)` and records its elapsed time to `DispatchManager`.
2. If dispatchable: calls `DispatchManager.dispatch_to_ttnn_wrapper`, which invokes `dispatch_to_ttnn` and records elapsed time under the `"TTNN"` backend.
3. If not dispatchable: calls `DispatchManager.dispatch_to_torch_wrapper`, which unwraps tensors to plain PyTorch, executes the native ATen op (or a custom torch dispatcher if one exists), re-wraps results as `TorchTTNNTensor`, and records elapsed time under the `"Torch"` backend.

**Module-level behavior (`module_run`):**

1. Applies the transform chain `wrap_to_torch_ttnn_tensor -> to_ttnn_wrap -> set_device_wrap(device)` to all input tensors to move them onto the TTNN device. **Exception:** keyword arguments whose key contains the substring `"past_key_value"` are excluded from transformation and passed through to `self.forward()` unchanged. They are not wrapped, converted to TTNN, or moved to device by this step.
2. Calls `preprocess_weights()` and `move_weights_to_device()`, recording both via `DispatchManager`.
3. Optionally emits a Tracy signpost if `TT_SYMBIOTE_SIGNPOST_MODE` is set.
4. Calls `self.forward(...)` and records the forward duration.
5. Records the total module wall-clock time under the `"TorchModules"` backend.

`NormalRun` cannot be instantiated directly (`__new__` raises `TypeError`); its static methods are mixed into `TorchTTNNTensor`.

---

### `LIGHTWEIGHT` — `LightweightRun`

CPU-only dispatch at the ATen level, without TTNN involvement.

**Inherits from:** `NormalRun`

**ATen-level behavior:**

Overrides `torch_dispatch` to call only `DispatchManager.dispatch_to_torch_wrapper(func, args, kwargs, wrap=False)`. The `wrap=False` argument means results are **not** re-wrapped as `TorchTTNNTensor` after execution, making this the lowest-overhead path.

> **Warning:** `LIGHTWEIGHT` and `TRACED` require the CPU dispatcher to be active. `get_tensor_run_implementation()` asserts `get_active_dispatcher() == cpu_dispatcher`; set `TT_SYMBIOTE_DISPATCHER=CPU` before using either mode.

---

### `NORMAL_WITH_FALLBACK` — `NormalRunWithFallback`

TTNN execution with automatic per-operation PyTorch fallback on error.

**Inherits from:** `NormalRun`

**ATen-level behavior:**

Wraps the dispatch logic in a `try/except`. If `dispatch_to_ttnn_wrapper` (or the TTNN dispatch check) raises any exception, it catches it, prints a message, and falls back to `dispatch_to_torch_wrapper`.

**Module-level behavior:**

Applies the transform chain `wrap_to_torch_ttnn_tensor -> to_ttnn_wrap -> set_device_wrap(device)` to **all** keyword arguments including any whose key contains `"past_key_value"`. Unlike `NormalRun.module_run` and `TracedRun.module_run`, `NormalRunWithFallback.module_run` does **not** exclude `past_key_value` kwargs from transformation: all kwargs are passed through `tree_map(transform, kwds)` without filtering. A transformer model that passes KV-cache tensors as `past_key_value` kwargs under `NORMAL_WITH_FALLBACK` will have those tensors wrapped and moved to device — different behavior from `NORMAL` mode, which passes them through unchanged.

Adds a `try/except` around `self.forward(...)`. On failure it falls back to `self.torch_layer(*args, **kwds)` (the reference PyTorch layer). If `device` is `None` it also falls back immediately. The fallback requires `self.torch_layer` to be set; an assertion error is raised otherwise.

---

### `SEL` (Segment Each Layer) — `SELRun`

PyTorch and TTNN both execute; outputs are compared with PCC, and the result tensor carries the TTNN backend data.

**Inherits from:** `NormalRun`

**ATen-level behavior:**

1. Copies all input tensors to pure PyTorch via `copy_to_torch(func)` (deep clone, `ttnn_tensor = None`).
2. Runs the copied inputs through `dispatch_to_torch_wrapper` to get a PyTorch reference output.
3. If the op is TTNN-dispatchable, also runs the original (TTNN-backed) inputs through `dispatch_to_ttnn_wrapper`.
4. Calls `compare_fn_outputs` on inputs, then on outputs.
5. Calls `create_new_ttnn_tensors_using_torch_output(result, ttnn_output)` — assigns the TTNN tensor pointer into the PyTorch-shaped result wrapper (with `assign_ttnn_to_torch=False`, so `elem` is cleared).

**Module-level behavior:**

Runs `self.torch_layer` first (using cloned inputs), then if a device is set also runs `self.forward` with TTNN tensors, compares, and merges via `create_new_ttnn_tensors_using_torch_output`.

> **Note:** SEL is named "Segment Each Layer" to reflect that it validates every layer independently. It is useful for pinpointing the first layer where TTNN diverges from PyTorch.

---

### `DPL` (Debug Per Layer) — `DPLRun`

TTNN and PyTorch run in parallel; outputs are compared; the result carries **both** the PyTorch `elem` and the TTNN tensor.

**Inherits from:** `NormalRun`

**ATen-level behavior:**

Identical structure to `SELRun` except `create_new_ttnn_tensors_using_torch_output` is called with `assign_ttnn_to_torch=True`, meaning `elem` is **kept** alongside `ttnn_tensor`. This lets TTNN errors propagate forward through the graph: subsequent ops receive tensors whose `.elem` is still the live PyTorch value, so PyTorch divergence can accumulate.

**Module-level behavior:**

Asserts `self.torch_layer is not None`. Runs both paths and compares. Uses `assign_ttnn_to_torch=True` at the module level as well.

---

### `DPL_NO_ERROR_PROP` — `DPLRunNoErrorProp`

DPL variant where TTNN receives freshly-copied TTNN tensors derived from the latest PyTorch values, preventing error accumulation.

**Inherits from:** `NormalRun`

**ATen-level behavior:**

1. Copies inputs to torch via `copy_to_torch(func)` and runs PyTorch.
2. Separately copies inputs to fresh TTNN tensors via `copy_to_ttnn(func)` — this re-creates the `ttnn_tensor` from the current `elem`, preserving layout and device.
3. Runs TTNN with these fresh tensors.
4. Compares inputs and outputs; merges with `assign_ttnn_to_torch=True`.
5. Prints `"DPLNoErrorPropRun: Done Executing {func.name()}"` after each dispatchable op.

**Module-level behavior:**

Uses `copy_to_ttnn` on the incoming args before running the TTNN forward path, so every module starts from a clean PyTorch-derived state.

---

### `CPU` — `CPU`

CPU-only execution using the PyTorch reference layer; no TTNN involvement.

**Inherits from:** `NormalRun`

**ATen-level behavior:**

Overrides `torch_dispatch` to always call `DispatchManager.dispatch_to_torch_wrapper`, printing the function name first.

**Module-level behavior:**

Wraps all tensors with `wrap_to_torch_ttnn_tensor`, calls `self.torch_layer`, and wraps outputs. Device state is ignored.

> **Note:** `CPU` mode is appropriate for validating model correctness on a machine without Tenstorrent hardware, or as a reference baseline before enabling TTNN.

---

### `TRACED` — `TracedRun`

TTNN trace capture and replay for maximum throughput on repeated inference.

**Inherits from:** `LightweightRun`

**Class-level configuration:**

```python
TracedRun.configure(
    device=mesh_device,
    cq_id=0,
    input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

`configure()` calls `release_all()` as its first action, which iterates over every cached `TraceEntry` and calls `ttnn.release_trace(entry.device, entry.trace_id)` to free the GPU-side trace allocation, then clears the Python-side `_trace_cache` dict. Any previously captured traces are fully released — both device memory and the cache — before the new configuration is stored. A call to `configure()` on a model with populated trace cache will force a full re-capture on the next forward pass.

**Module-level behavior:**

1. Applies the same input transform chain as `NormalRun`.
2. Calls `preprocess_weights()` and `move_weights_to_device()`.
3. Checks `is_trace_enabled(self)` — trace-enablement is inherited via `isinstance`, so a subclass decorated with `@trace_disabled` can opt out of a parent's `@trace_enabled`. See the [decorator documentation below](#trace_enabled-and-trace_disabled-decorators) for the full `isinstance`-based check.
4. Before the `is_trace_enabled` check, `TracedRun.module_run` acquires `_TRACE_RUNNING_LOCK` and sets `_TRACE_RUNNING = True` if it was `False`. Only after this does it evaluate `if not is_trace_enabled(self) or already_running`. On the early-return (non-trace-enabled or nested-guard) path it calls `self.forward(...)` with `_TRACE_RUNNING = True`, then resets the flag to `False` on exit (only if this call was the one that set it). **Important:** this means that if a non-trace-enabled module wraps trace-enabled submodules, those submodules will also see `_TRACE_RUNNING = True` during the outer `forward()` call and will take their own early-return paths — their traces will never be captured.
5. Builds a cache key via `_make_cache_key(module_name, func_args, func_kwargs)`, which hashes module name plus the shape/dtype/layout signature of each positional input tensor **and** each keyword-argument tensor. Both positional and keyword tensor arguments are included in the key — a model that passes tensors as keyword arguments (e.g., `attention_mask`) will produce a different cache key if those kwarg tensor shapes differ.
6. **Cache miss (first run):** Sets `_TRACE_RUNNING = True`, calls `_capture_trace`, which does one warm-up forward, then captures with `ttnn.begin_trace_capture` / `ttnn.end_trace_capture` / `ttnn.synchronize_device`. Stores a `TraceEntry(trace_id, trace_inputs, trace_output, device)`.
7. **Cache hit (subsequent runs):** Calls `_copy_inputs_to_trace_buffer` to copy new input data into the pre-allocated trace buffers via `ttnn.copy`, then replays with `ttnn.execute_trace(..., blocking=TracedRun._blocking)`. The `blocking` flag defaults to `True` and is controlled by the `blocking` parameter passed to `TracedRun.configure()`. Do not assume non-blocking execution; the default replay is blocking.

**`@trace_enabled` and `@trace_disabled` decorators:**

```python
from models.experimental.tt_symbiote.core.run_config import trace_enabled, trace_disabled

@trace_enabled
class MyLinear(TTNNLinear):
    ...

@trace_disabled
class MyLinearDebug(MyLinear):
    ...
```

`trace_enabled(cls)` adds `cls` to `_TRACE_ENABLED_CLASSES`. `trace_disabled(cls)` adds `cls` to `_TRACE_DISABLED_CLASSES`. `is_trace_enabled(module)` checks both sets. A subclass can opt out even if its parent opted in.

**`disable_trace` function decorator:**

```python
from models.experimental.tt_symbiote.core.run_config import disable_trace

@disable_trace
def my_function_that_must_not_be_traced(...):
    ...
```

`disable_trace` wraps a function so that `_TRACE_RUNNING` is set to `True` for its duration, preventing any nested trace capture.

**Releasing traces:**

```python
TracedRun.release_all()           # release all cached traces
TracedRun.release("model.layer0") # release traces for a specific module
```

Both call `ttnn.release_trace` on the stored `trace_id`.

---

## The `no_dispatch()` context manager

```python
@contextlib.contextmanager
def no_dispatch() -> Iterator[None]:
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard
```

`no_dispatch()` wraps `torch._C._DisableTorchDispatch()`. It is used inside `DispatchManager.dispatch_to_torch_wrapper` to prevent infinite recursion: when falling back to a native ATen op from within `__torch_dispatch__`, the call would otherwise re-enter the same dispatch hook. Disabling the dispatch mechanism for the duration of the fallback avoids this loop.

---

## Helper transform functions

These pure functions are used to prepare tensors before passing them to a backend. They are applied via `tree_map` from `core/utils.py`, which recursively walks nested lists, tuples, and dicts.

| Function | Signature | What it does |
|----------|-----------|-------------|
| `unwrap_to_torch(func)` | `(func) -> (e -> tensor)` | Returns a function that extracts the plain `torch.Tensor` from a `TorchTTNNTensor` (via `.to_torch`), passes through existing `torch.Tensor`s, and converts bare `ttnn.Tensor`s by wrapping then converting. |
| `wrap_from_torch(e)` | `(e) -> TorchTTNNTensor or e` | Wraps a plain `torch.Tensor` as a `TorchTTNNTensor`; passes through anything that is already a `TorchTTNNTensor` or non-tensor. |
| `copy_to_torch(func)` | `(func) -> (e -> TorchTTNNTensor)` | Returns a function that deep-clones the PyTorch side of a `TorchTTNNTensor` and sets `ttnn_tensor = None`, producing a CPU-only clone. Used in `SELRun` and `DPLRun` to isolate the reference path. |
| `copy_to_ttnn(func)` | `(func) -> (e -> TorchTTNNTensor)` | Returns a function that creates a fresh `ttnn.Tensor` from `elem.clone()`, preserving layout and moving to the original device. **Only operates when both `e.elem is not None` and `e.ttnn_tensor is not None`**; if either field is absent, the original tensor is returned unchanged and no fresh copy is made. Used in `DPLRunNoErrorProp` to prevent TTNN error propagation — but only effective for tensors that carry both a PyTorch `elem` and an active `ttnn_tensor`. |
| `set_device_wrap(device)` | `(device) -> (e -> e)` | Returns a function that calls `ttnn.to_device` to move the tensor onto `device` if it is not already there. Handles both raw `ttnn.Tensor` objects (moves the tensor directly) and `TorchTTNNTensor` objects with a non-`None` `ttnn_tensor` field (moves the `ttnn_tensor` field in-place). Non-tensor inputs and `TorchTTNNTensor` objects with `ttnn_tensor = None` are returned unchanged. |
| `create_new_ttnn_tensors_using_torch_output` | `(torch_output, ttnn_output, assign_ttnn_to_torch=False)` | Merges a TTNN result back into a PyTorch-shaped wrapper. If `assign_ttnn_to_torch=False` (SEL): clears `elem`. If `assign_ttnn_to_torch=True` (DPL): keeps `elem`. Asserts shape agreement. Handles both single-tensor and list/tuple outputs. |

### `unwrap_to_torch` vs. `copy_to_torch`

`unwrap_to_torch` simply returns a reference to the existing PyTorch tensor — it does not copy. `copy_to_torch` clones the data and severs the TTNN link, ensuring the PyTorch reference path is fully independent.

### `set_device_wrap` vs. `to_ttnn_wrap`

`to_ttnn_wrap` (defined just above `set_device_wrap` in the source) triggers lazy TTNN conversion of a `TorchTTNNTensor` and returns the resulting raw `ttnn.Tensor`; it does not move the result to any specific device. `set_device_wrap(device)` handles both raw `ttnn.Tensor` objects and `TorchTTNNTensor` objects whose `ttnn_tensor` field is set; it is safe to call on either type. In `NormalRun.module_run` they are composed in order: `wrap_to_torch_ttnn_tensor -> to_ttnn_wrap -> set_device_wrap(device)`. Because `to_ttnn_wrap` returns a raw `ttnn.Tensor`, `set_device_wrap` receives a `ttnn.Tensor` in this chain — but when used in isolation or with `TorchTTNNTensor` inputs, it also handles those correctly.

---

Next: [dispatch_manager.md](dispatch_manager.md)
