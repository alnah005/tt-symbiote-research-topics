# The Dispatcher System — routing ATen operations to TTNN or PyTorch

The dispatcher is the decision engine that sits between `TorchTTNNTensor.__torch_dispatch__` and the actual execution of every ATen operation. It answers two questions: can this operation run on TTNN right now, and if so, how?

## Architecture

The system is split across two layers:

```
core/
├── dispatcher.py                  # Public API — thin wrappers over the registry
└── dispatchers/
    ├── __init__.py                # Re-exports registry functions
    ├── dispatcher_config.py       # Registry, selection logic, auto-registration
    ├── default_dispatcher.py      # Standard TTNN implementation
    ├── debug_dispatcher.py        # Verbose logging variant
    ├── cpu_dispatcher.py          # Forces PyTorch CPU execution
    └── tensor_operations_dispatcher.py  # Tensor-op focused variant
```

`dispatcher.py` is the file user code imports. It defines the two canonical public execution functions `can_dispatch_to_ttnn(func_name, args, kwargs)` and `dispatch_to_ttnn(func_name, args, kwargs)` with real bodies that call `get_active_dispatcher()` and delegate to it. These functions are the authoritative public interface — they are not pass-throughs. The registry management functions (`register_dispatcher`, `set_dispatcher`, `list_available_dispatchers`) are re-exported from `dispatchers/dispatcher_config.py` without modification; only that group contains no logic of its own in `dispatcher.py`. This separation means adding a new built-in dispatcher requires no changes to the public API.

## Registry functions

The five stable public functions are declared in `dispatcher.py`'s `__all__` and are safe to import from `core/dispatcher.py`:

| Function | Signature | Purpose |
|---|---|---|
| `can_dispatch_to_ttnn` | `(func_name: str, args=None, kwargs=None) -> bool` | Asks the active dispatcher whether the named ATen operation can be handled |
| `dispatch_to_ttnn` | `(func_name: str, args, kwargs) -> Any` | Executes the operation via the active dispatcher; returns a `TorchTTNNTensor` in almost all cases |
| `register_dispatcher` | `(name: str, dispatcher_module: Any) -> None` | Adds a dispatcher to the registry; raises `ValueError` if the module lacks `can_dispatch_to_ttnn` or `dispatch_to_ttnn` |
| `set_dispatcher` | `(name: str) -> None` | Sets the programmatic active dispatcher; raises `ValueError` if the name is not registered |
| `list_available_dispatchers` | `() -> list[str]` | Returns all registered names as a list |

> **Note on `get_active_dispatcher`:** This function is accessible as a module attribute of `dispatcher.py` (it is re-imported from `dispatchers/dispatcher_config.py`) but it is **not** in `dispatcher.py`'s `__all__`. It should not be treated as a stable public import. The five functions in the table above are the declared stable interface. `can_dispatch_to_ttnn` and `dispatch_to_ttnn` call `get_active_dispatcher()` internally, so there is no need for user code to call it directly.

## Active dispatcher selection order

`get_active_dispatcher` applies rules in this order:

1. If `TT_SYMBIOTE_DISPATCHER` is set in the environment **and** the value matches a registered name, that dispatcher is used. The environment variable always takes precedence over the programmatic setting.

> **Warning:** The behavior when `TT_SYMBIOTE_DISPATCHER` is set to an unrecognized name depends on whether `set_dispatcher()` has been called:
> - **No prior `set_dispatcher()` call (fresh process):** A `RuntimeError` is raised immediately — the CPU default is unreachable when the env var is present but unrecognized.
> - **After a prior `set_dispatcher()` call:** The unrecognized env var is ignored and the programmatically selected dispatcher is used.
>
> In either case, a typo in the env var name will not silently use the CPU fallback. Always verify the intended dispatcher name with `list_available_dispatchers()` before setting the env var.

2. If `TT_SYMBIOTE_DISPATCHER` is not set **and** no dispatcher has been selected programmatically (`_current_dispatcher is None`), the **CPU** dispatcher is returned as the default.
3. Otherwise the programmatically selected dispatcher (`_current_dispatcher`) is used.

> **Note:** The default when neither the environment variable nor a programmatic call has been made is **CPU**, not DEFAULT. This is intentional: the CPU dispatcher is a safe, universally available fallback. To enable full TTNN dispatch you must either set `TT_SYMBIOTE_DISPATCHER=DEFAULT` or call `set_dispatcher("DEFAULT")` explicitly.

## Environment variable reference

| Variable | Values | Effect |
|---|---|---|
| `TT_SYMBIOTE_DISPATCHER` | `DEFAULT`, `DEBUG`, `CPU`, `TENSOR_OPS`, or any registered name | Overrides the programmatic dispatcher selection for the duration of the process |

Example:

```bash
export TT_SYMBIOTE_DISPATCHER=DEFAULT
python run_model.py

# Or inline for a single command:
TT_SYMBIOTE_DISPATCHER=DEBUG pytest tests/test_vit.py
```

## Auto-registered dispatchers

`dispatcher_config.py` calls `_auto_register_dispatchers()` at module import time. The function attempts to import each built-in dispatcher and registers it under a fixed name. Import failures are silently swallowed (`except ImportError: pass`), so optional dependencies do not prevent the others from loading.

| Registered name | Module | Description |
|---|---|---|
| `DEFAULT` | `dispatchers/default_dispatcher.py` | Full TTNN implementation: binary ops, matmul, activations, tensor manipulation, comparisons |
| `DEBUG` | `dispatchers/debug_dispatcher.py` | Same operations as DEFAULT with verbose logging for tracing dispatch decisions |
| `CPU` | `dispatchers/cpu_dispatcher.py` | Falls through to PyTorch CPU for every operation; used as the safe default |
| `TENSOR_OPS` | `dispatchers/tensor_operations_dispatcher.py` | Focused on tensor manipulation operations |

## Public interface

Only two functions are needed during normal forward-pass execution. Both are imported from `models.experimental.tt_symbiote.core.dispatcher`:

```python
from models.experimental.tt_symbiote.core.dispatcher import can_dispatch_to_ttnn, dispatch_to_ttnn

if can_dispatch_to_ttnn("aten::add.Tensor", args, kwargs):
    result = dispatch_to_ttnn("aten::add.Tensor", args, kwargs)
```

`func_name` is the full ATen name string as passed by `__torch_dispatch__` (e.g., `"aten::add.Tensor"`, `"aten::mm"`). Always call `can_dispatch_to_ttnn` before `dispatch_to_ttnn`; behaviour when dispatching an unsupported operation is dispatcher-defined. See the registry functions table above for full signatures and return types.

## Registering a custom dispatcher at runtime

Any module that exports `can_dispatch_to_ttnn` and `dispatch_to_ttnn` can be registered at runtime:

```python
import my_project.custom_dispatcher
from models.experimental.tt_symbiote.core.dispatcher import register_dispatcher, set_dispatcher

register_dispatcher("custom", my_project.custom_dispatcher)
set_dispatcher("custom")
```

After this, all subsequent ATen operations intercepted by `TorchTTNNTensor.__torch_dispatch__` will be routed through `my_project.custom_dispatcher`.

To make a dispatcher auto-register with the framework (useful for built-in additions), add it to `_auto_register_dispatchers()` in `models/experimental/tt_symbiote/core/dispatchers/dispatcher_config.py` following the existing pattern.

## Dispatcher contract

Every dispatcher module — built-in or custom — must expose exactly two callables:

```python
def can_dispatch_to_ttnn(func_name: str, args=None, kwargs=None) -> bool:
    """Return True if this dispatcher can handle the given ATen operation."""
    ...

def dispatch_to_ttnn(func_name: str, args, kwargs):
    """Execute the operation and return the result (typically a TorchTTNNTensor)."""
    ...
```

`register_dispatcher` validates the presence of both at registration time and raises `ValueError` if either is missing.

---

**Next:** [Chapter 3 — Run Modes and the Dispatch Manager](../ch3_run_modes_and_timing/index.md)
