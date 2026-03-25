# Chapter 6 — Fallback and Debugging

This file covers the tooling available to debug a custom `TTNNModule`, explains
when the fallback PyTorch layer is used, and lists the most common authoring mistakes.

---

## The `_fallback_torch_layer` attribute

Every `TTNNModule` has a `_fallback_torch_layer` attribute (exposed as the `.torch_layer`
property). When `from_torch` is implemented correctly, this holds the original
`nn.Module` that was replaced. Several run modes depend on it.

```python
@property
def torch_layer(self):
    """Get fallback PyTorch layer."""
    return self._fallback_torch_layer
```

Source: `module.py`, `TTNNModule.torch_layer` property.

---

## Run modes and how they use the fallback layer

All run modes are selected via the `TT_SYMBIOTE_RUN_MODE` environment variable (or
programmatically via `set_run_mode`). The full registry (from `run_config.py`) is:

| `TT_SYMBIOTE_RUN_MODE` value | Class | Fallback behaviour |
|---|---|---|
| `NORMAL` (default) | `NormalRun` | No fallback; errors propagate |
| `NORMAL_WITH_FALLBACK` | `NormalRunWithFallback` | Catches exceptions in `forward` and falls back to `torch_layer` |
| `SEL` | `SELRun` | Runs both `torch_layer` and `forward`, compares outputs with PCC |
| `DPL` | `DPLRun` | Runs both `torch_layer` and `forward`, compares outputs with PCC; propagates TTNN tensors downstream (torch wrapper retains both elem and ttnn_tensor, unlike SEL which clears elem) |
| `DPL_NO_ERROR_PROP` | `DPLRunNoErrorProp` | Feeds fresh (non-accumulated-error) tensors to each op; compares PCC |
| `CPU` | `CPU` | Always runs `torch_layer`; never calls `forward` |
| `LIGHTWEIGHT` | `LightweightRun` | Runs CPU dispatch; no TTNN |
| `TRACED` | `TracedRun` | Captures and replays `ttnn` traces |

### `NORMAL_WITH_FALLBACK`

When the device is set and `forward` raises an exception, the run mode catches it and
calls `self.torch_layer(*args, **kwargs)` instead:

```python
# NormalRunWithFallback.module_run (run_config.py)
try:
    result = post_process_ttnn_module_output(self, self.forward(*func_args, **func_kwargs))
except Exception as e:
    print(f"Error {e} in {self.__class__.__name__} forward, falling back to torch")
    assert self.torch_layer is not None, ...
    result = self.torch_layer(*args, **kwds)
```

If the device is not set at all, it falls back unconditionally:

```python
else:
    print("Device not set, falling back to torch")
    assert self.torch_layer is not None, ...
    result = self.torch_layer(*args, **kwds)
```

This mode is useful for incremental bring-up: you can run the full model with some
modules not yet working, provided every module has `_fallback_torch_layer` set.

### `DPL` (Dual Path Logging)

`DPLRun.module_run` runs the `torch_layer` path first (to get a reference output), then
runs the TTNN `forward` path. It calls `compare_fn_outputs` on both inputs and outputs to
log PCC discrepancies. The TTNN outputs are then propagated downstream (they replace the
torch outputs for subsequent modules).

```python
# DPLRun.module_run (run_config.py)
torch_output = tree_map(wrap_to_torch_ttnn_tensor, self.torch_layer(*func_args, **func_kwargs))
...
ttnn_output = post_process_ttnn_module_output(self, self.forward(*func_args, **func_kwargs))
compare_fn_outputs(torch_output, ttnn_output, self.__class__.__name__)
result = create_new_ttnn_tensors_using_torch_output(torch_output, ttnn_output, assign_ttnn_to_torch=True)
```

`torch_layer` must not be `None` for `DPLRun`; an assertion fires if it is.

---

## Isolating a single module's numerical error with `DPL_NO_ERROR_PROP`

The `DPL_NO_ERROR_PROP` mode is specifically designed to prevent accumulated numerical
error from propagating through the network, making it easier to identify which module
introduces a PCC regression.

In `DPLRunNoErrorProp.module_run` (source: `run_config.py`), each module receives a
**fresh copy** of the torch tensors converted to TTNN (via `copy_to_ttnn`), rather than
the TTNN tensors produced by earlier modules:

```python
ttnn_no_error_prop_args = tree_map(copy_to_ttnn(self.__class__.__name__), args)
ttnn_no_error_prop_kwargs = tree_map(copy_to_ttnn(self.__class__.__name__), kwds)
transform = compose_transforms(wrap_to_torch_ttnn_tensor, to_ttnn_wrap, set_device_wrap(self.device))
func_args = tree_map(transform, ttnn_no_error_prop_args)
...
ttnn_output = post_process_ttnn_module_output(self, self.forward(*func_args, **func_kwargs))
compare_fn_outputs(torch_output, ttnn_output, self.__class__.__name__)
```

**Usage:**

```bash
TT_SYMBIOTE_RUN_MODE=DPL_NO_ERROR_PROP python my_model_script.py
```

Any module whose PCC drops below threshold in this mode is the direct source of the
error, not a downstream consumer of someone else's accumulated error.

---

## Using `TT_SYMBIOTE_DISPATCHER=DEBUG` for op-level logging

The `DEBUG` dispatcher (registered in `core/dispatchers/dispatcher_config.py` as `"DEBUG"`) wraps the
`DEFAULT` dispatcher and logs every dispatched ATen operation via Python's `logging`
module at `DEBUG` level:

```python
# debug_dispatcher.py
def can_dispatch_to_ttnn(func_name, args=None, kwargs=None):
    result = default_can_dispatch(func_name, args, kwargs)
    if not result:
        logger.debug(f"  Cannot dispatch {func_name} to TTNN")
    return result

def dispatch_to_ttnn(func_name, args, kwargs):
    try:
        result = default_dispatch(func_name, args, kwargs)
        logger.debug(f"  Successfully dispatched {func_name}")
        return result
    except Exception as e:
        logger.error(f"  Failed to dispatch {func_name}: {e}")
        raise
```

**Usage:**

```bash
TT_SYMBIOTE_DISPATCHER=DEBUG python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
# ... run your model
"
```

This tells you, for every ATen op, whether the dispatcher routed it to TTNN or fell back
to Torch. Ops that `can_dispatch_to_ttnn` returns `False` for will appear in the debug
log as `"Cannot dispatch <op> to TTNN"`.

The `DEBUG` dispatcher uses Python's standard `logging` module, not `print`, so you must
configure a handler at `DEBUG` level to see its output.

---

## Using `TT_SYMBIOTE_RUN_MODE=SEL` to compare outputs with PCC

`SELRun` (Selective Execution Logging) runs both the torch and TTNN paths at the module
boundary and compares them:

```python
# SELRun.module_run (run_config.py)
torch_output = tree_map(wrap_to_torch_ttnn_tensor, self.torch_layer(*func_args, **func_kwargs))
...
ttnn_output = post_process_ttnn_module_output(self, self.forward(*func_args, **func_kwargs))
compare_fn_outputs(copied_torch_tensors_args, func_args, self.__class__.__name__)   # inputs
compare_fn_outputs(torch_output, ttnn_output, self.__class__.__name__)              # outputs
result = create_new_ttnn_tensors_using_torch_output(torch_output, ttnn_output)
```

Unlike `DPL`, `SELRun` propagates the torch-backed tensors downstream (with the TTNN
tensor attached), so TTNN errors do not accumulate. At the op level (`SELRun.torch_dispatch`),
the same dual-run and comparison happens for every individual ATen operation.

**Usage:**

```bash
TT_SYMBIOTE_RUN_MODE=SEL python my_model_script.py
```

`compare_fn_outputs` is from `models/experimental/tt_symbiote/core/utils.py` and prints
PCC results to stdout.

---

## Reading the `DispatchManager` timing CSV

`DispatchManager` (in `run_config.py`) records a timing entry for every TTNN op dispatch,
every torch fallback, every `preprocess_weights`, every `move_weights_to_device`, and
every `forward` call. Entries are keyed by `backend`, `module_name`, and `func_name`.

### Saving timing data

```python
from models.experimental.tt_symbiote.core.run_config import DispatchManager

# ... run your model ...

DispatchManager.save_stats_to_file("timings.csv")
```

This writes two files:

- `timings.csv` — one row per recorded event.
- `timings_pivot.csv` — pivot table aggregated by `(func_name, module_name)` with
  per-backend totals, min, max, and row count columns.

### Confirming TTNN ops are being dispatched

Open `timings_pivot.csv` and look for rows where the `TTNN` column has a non-zero
duration and the `Torch` column is zero (or absent). If a module you expect to run on
TTNN shows only `Torch` entries, it means either:

- The dispatcher is returning `False` from `can_dispatch_to_ttnn` for those ops (use
  `TT_SYMBIOTE_DISPATCHER=DEBUG` to confirm).
- The module's `forward` raised an exception and the mode fell back to torch
  (relevant only in `NORMAL_WITH_FALLBACK`).

### Backend labels used in `timings.csv`

| Backend label | Meaning |
|---|---|
| `TTNN` | Dispatched to TTNN hardware |
| `Torch` | Fell back to CPU torch |
| `TorchModules` | Full module invocation timing (wraps everything in `module_run`) |

---

## Common mistakes

### 1. Forgetting to call `super().deallocate_weights_impl()`

If your `deallocate_weights_impl` frees its own tensors but does not call
`super().deallocate_weights_impl()`, any child `TTNNModule` instances stored in
`self.__dict__` will not have their device tensors freed. The base class implementation
recurses over all `TTNNModule`-typed dict values.

**Wrong:**
```python
def deallocate_weights_impl(self):
    ttnn.deallocate(self.tt_weight)
    # super() not called — child modules leak device memory
```

**Correct:** See the canonical pattern in [implementation_guide.md — Step 5](implementation_guide.md#step-5--deallocate_weights_impl).

### 2. Returning the wrong tensor type from `forward`

`forward` must return a `ttnn.Tensor` (or a structure containing `ttnn.Tensor`s). The
run-mode `post_process_ttnn_module_output` wraps the result in `TorchTTNNTensor` for the
downstream torch-dispatch machinery, but if your module returns a raw `torch.Tensor`
instead of a `ttnn.Tensor`, the wrapping will treat it as a plain torch tensor with no
TTNN backend, defeating the purpose of the module.

If you need to access the underlying `ttnn.Tensor` from a `TorchTTNNTensor` that arrives
in your `forward`, use `.to_ttnn`:

```python
def forward(self, x):
    tt = x.to_ttnn            # unwrap TorchTTNNTensor -> ttnn.Tensor
    result = ttnn.relu(tt)    # returns ttnn.Tensor
    return result             # correct: return ttnn.Tensor
```

### 3. Not calling `set_device` (i.e., `to_device`) before `move_weights_to_device`

`TTNNModule.move_weights_to_device` asserts `self.device is not None`. If you call
`module.move_weights_to_device()` directly (outside the standard `module_run` path)
without first calling `module.to_device(device)`, you will get:

```
AssertionError: Device must be set for <name> before moving weights to device.
```

When using the standard setup sequence via `register_module_replacement_dict`, call
`to_device` on every module in the returned dict before the first forward pass. When
calling lifecycle methods manually (e.g., in tests), always call `to_device` first.

### 4. Setting `_fallback_torch_layer` to `None` and then using a fallback run mode

`DPLRun.module_run` and `DPLRunNoErrorProp.module_run` both contain:

```python
assert self.torch_layer is not None, \
    f"torch_layer must be set for DPLRun, {self} does not have torch_layer set."
```

`NormalRunWithFallback.module_run` also asserts `torch_layer is not None` at the point
of fallback. If you override `from_torch` and forget to set `_fallback_torch_layer`, any
run mode that needs it will raise `AssertionError` at runtime.

### 5. Calling `preprocess_weights()` inside `from_torch`

`from_torch` is a factory, not a setup routine. The `preprocess_weights()` method is
guarded by `_preprocessed_weight` and is called lazily by `module_run`. If you call it
inside `from_torch`, the weight tensors (e.g., `self.weight`, `self.bias`) must already
be set as attributes — which is fine — but then the `_preprocessed_weight` flag is
already `True`, so a second call from `module_run` will be a no-op. This is not wrong
per se, but it means weight preprocessing happens at model-load time rather than at first
inference, which may not be what you want and is inconsistent with the framework
convention. `TTNNLinear.from_parameters` uses this pattern intentionally (it calls
`preprocess_weights()` during factory construction and then deletes the raw attributes),
but `TTNNLinear.from_torch` does not.

### 6. Using `@run_on_devices` without the `MESH_DEVICE` environment variable set

The `run_on_devices` decorator reads `os.environ.get("MESH_DEVICE")` at call time. If
`MESH_DEVICE` is not set, it raises:

```
RuntimeError: <ClassName>: Unable to determine device architecture from MESH_DEVICE
environment variable.
```

Set `MESH_DEVICE` to one of the string values in `MeshShapeToDeviceArch` (e.g., `N150`,
`T3K`, `P150`) before running a model that uses `@run_on_devices`.

---

**Next:** [Chapter 7 — End-to-End Use Cases and Performance Benchmarking](../ch7_use_cases_and_benchmarking/index.md)
