# TT Symbiote Internals

Source file: `models/experimental/tt_symbiote/core/module.py`

---

## `TTNNModule` base class

`TTNNModule` is the root of every accelerated module in the TT Symbiote framework. It deliberately does **not** subclass `torch.nn.Module`: attribute access and method dispatch on `nn.Module` carry measurable host overhead, and Symbiote avoids it by maintaining its own child-traversal logic.

```python
from models.experimental.tt_symbiote.core.module import TTNNModule
```

### Construction and identity

The `__init__` method initialises seven instance attributes that the framework relies on:

```python
def __init__(self):
    self._device = None
    self._preprocessed_weight = False
    self._weights_on_device = False
    self._fallback_torch_layer = None
    self._unique_name = None
    self._device_state: Optional[DistributedConfig] = None
    self._model_config = {}
```

`_unique_name` is assigned either by `register_module_replacement_dict` at replacement time or lazily as `f"{self.__class__.__name__}_{id(self)}"` when `module_name` is first accessed. The name propagates to children via `override_children_module_names()`, which calls `set_module_name_recursively()` to walk `__dict__`, lists, tuples, and dicts.

### `from_torch()`

The canonical factory method for creating a `TTNNModule` from a PyTorch layer:

```python
@classmethod
def from_torch(cls, torch_layer, *args, **kwargs):
    new_layer = cls(*args, **kwargs)
    new_layer._fallback_torch_layer = torch_layer
    return new_layer
```

The original `torch_layer` is retained as `_fallback_torch_layer` and exposed via the `torch_layer` property. This is used by DPL and fallback run modes to execute the reference implementation alongside or instead of the TTNN path.

### Weight lifecycle

Symbiote enforces a strict three-phase lifecycle for weights:

**Phase 1 — preprocess on host**

```python
def preprocess_weights(self):
    if not self._preprocessed_weight:
        self._preprocessed_weight = True
    else:
        return
    self.preprocess_weights_impl()
```

`preprocess_weights_impl()` is the override point. The default implementation recurses into every `TTNNModule` child found in `__dict__`. A subclass overrides it to perform dtype conversion, quantisation, or layout transformation on host tensors.

**Phase 2 — move to device**

```python
def move_weights_to_device(self):
    assert self._preprocessed_weight, ...
    assert self.device is not None, ...
    if not self._weights_on_device:
        self._weights_on_device = True
    else:
        return
    self.move_weights_to_device_impl()
```

The guard assertions ensure that `preprocess_weights()` has been called first and that a device has been assigned via `to_device(device)`. `move_weights_to_device_impl()` is the override point; the default recurses into children.

**Phase 3 — optional deallocation**

```python
def deallocate_weights(self):
    self.deallocate_weights_impl()
    self._weights_on_device = False
```

The `@deallocate_weights_after` decorator from `models/experimental/tt_symbiote/core/module.py` wraps a `forward` implementation to call `self.deallocate_weights()` immediately after the forward pass returns. This is useful when a module's weights are large and should not occupy device memory between invocations.

```python
from models.experimental.tt_symbiote.core.module import deallocate_weights_after

class MyModule(TTNNModule):
    @deallocate_weights_after
    def forward(self, x):
        ...
```

### `forward()`

`forward()` raises `NotImplementedError` in the base class. Every concrete `TTNNModule` subclass must override it. The method is **not** called directly by user code; it is invoked by the run-mode implementation inside `__call__`.

---

## `__call__` and run-mode dispatch

```python
def __call__(self, *args, **kwds):
    return TENSOR_RUN_IMPLEMENTATION.module_run(self, *args, **kwds)
```

`TENSOR_RUN_IMPLEMENTATION` is a class object selected once at module import time by `get_tensor_run_implementation()` from `models/experimental/tt_symbiote/core/run_config.py`. It reads the `TT_SYMBIOTE_RUN_MODE` environment variable and returns the corresponding class from `_RUN_MODE_REGISTRY`:

```python
_RUN_MODE_REGISTRY = {
    "LIGHTWEIGHT": LightweightRun,
    "NORMAL": NormalRun,
    "NORMAL_WITH_FALLBACK": NormalRunWithFallback,
    "SEL": SELRun,
    "DPL": DPLRun,
    "DPL_NO_ERROR_PROP": DPLRunNoErrorProp,
    "CPU": CPU,
    "TRACED": TracedRun,
}
```

The default mode when no environment variable is set is `NORMAL`. Each mode's `module_run` static method defines what happens when a `TTNNModule` is called:

> **Note:** The TTNN path in any run mode only executes when `self.device is not None`. Always call `set_device(model, ttnn_device)` before using any debug mode for TTNN validation.

**Class hierarchy quick reference**

| Run mode | Superclass | CPU dispatcher required? |
|---|---|---|
| `LIGHTWEIGHT` | `NormalRun` | Yes (`TT_SYMBIOTE_DISPATCHER=CPU`) |
| `NORMAL` | — | No |
| `NORMAL_WITH_FALLBACK` | `NormalRun` | No |
| `SEL` | `NormalRun` | No |
| `DPL` | `NormalRun` | No |
| `DPL_NO_ERROR_PROP` | `NormalRun` | No |
| `CPU` | `NormalRun` | No |
| `TRACED` | `LightweightRun` | Yes (`TT_SYMBIOTE_DISPATCHER=CPU`) |

*`issubclass` is reflexive, so the `issubclass(result, LightweightRun)` guard in `get_tensor_run_implementation` fires for both `LIGHTWEIGHT` and `TRACED`.*

- **LIGHTWEIGHT** (`LightweightRun`): Overrides `torch_dispatch` to skip TTNN routing — all tensor operations go directly to the torch path. `module_run` is fully inherited from `NormalRun`, so weight preprocessing, device movement, tensor wrapping, and timing instrumentation still run; only per-op dispatch is bypassed.
- **NORMAL**: Wraps inputs as `TorchTTNNTensor`, converts to TTNN, moves to device, calls `preprocess_weights()` and `move_weights_to_device()`, then calls `self.forward()`. Records per-operator timing via `DispatchManager`.
- **NORMAL_WITH_FALLBACK**: Same as `NORMAL`, but applies fallback at two levels. At the **per-op level**: individual TTNN ops that throw are silently retried on the PyTorch path. At the **module level**: if the entire `forward()` throws, the whole module re-runs via `_fallback_torch_layer`. Prefer `DPL` when diagnosing partial dispatch failures, as silent per-op retries can obscure which operations are not dispatching correctly.
- **SEL**: A dual-path validation mode. `SELRun.module_run` runs both `_fallback_torch_layer` and the TTNN path for every layer, then compares outputs using PCC via `compare_fn_outputs`. Requires `_fallback_torch_layer` to be set.
- **DPL** (Dual-Path Logging): Runs both the TTNN path and `_fallback_torch_layer`, then calls `compare_fn_outputs` to compare outputs using PCC. A warning is printed when PCC < 0.999, but no exception is raised — check console output actively. Requires `_fallback_torch_layer` to be set.
- **DPL_NO_ERROR_PROP**: Structurally similar to `DPL` at the module level (runs both PyTorch and TTNN paths, compares with PCC, does not raise on mismatch). The distinction is in `torch_dispatch`: rather than re-using the same tensor objects for TTNN dispatch, `DPL_NO_ERROR_PROP` creates separate TTNN tensor copies via `copy_to_ttnn` before dispatching. This means TTNN dispatch cannot corrupt the PyTorch-side tensor result via in-place mutation — if TTNN mutates the copied tensors during dispatch, the original torch-side tensors (and thus `result`) remain intact. Note: TTNN exceptions are not silently absorbed; if `dispatch_to_ttnn_wrapper` throws, the exception still propagates. Use `DPL_NO_ERROR_PROP` when TTNN dispatch would otherwise mutate shared tensor state and make the comparison meaningless; use `DPL` when tensor-isolation overhead is not needed.
- **CPU**: Dispatches all tensor operations to CPU PyTorch, bypassing TTNN entirely. `CPU.module_run` invokes `_fallback_torch_layer()` directly and never calls `self.forward()` — a custom `forward()` will not execute under this mode.
- **TRACED**: Wraps the forward pass inside a trace-capture context for later replay. Only modules decorated with `@trace_enabled` are traced; others fall back to normal execution with no error or warning.

The run mode is resolved once at process startup because `TENSOR_RUN_IMPLEMENTATION = get_tensor_run_implementation()` is executed at module import time in `module.py`. It cannot be changed after import without reimporting.

---

## `TorchTTNNTensor`

Source file: `models/experimental/tt_symbiote/core/tensor.py`

`TorchTTNNTensor` is a `torch.Tensor` subclass that holds both a CPU-side PyTorch tensor (`.elem`) and an optional device-side TTNN tensor (`.ttnn_tensor`). The dual representation allows user code to treat the object as a normal PyTorch tensor while the framework transparently routes operations to TTNN.

```python
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

class TorchTTNNTensor(torch.Tensor):
    __slots__ = ["elem"]
```

### Key properties

- **`.to_ttnn`**: Returns the TTNN tensor. If `.ttnn_tensor` is `None`, materialises it from `.elem` via `ttnn.from_torch`. The TTNN dtype is derived from `self.elem.dtype` (via `torch_dtype_to_ttnn_dtype`); the `mesh_mapper` is taken from `ttnn_distributed_tensor_config.mesh_mapper` if a `DistributedTensorConfig` is attached, otherwise `None`.
- **`.to_torch`**: Returns the CPU PyTorch tensor. The actual conversion is delegated to `TENSOR_RUN_IMPLEMENTATION.to_torch`, which may call `ttnn.to_torch` on the device-side tensor.
- **`.shape`**: Returns the logical shape, accounting for distributed sharding via `ttnn_distributed_tensor_config.get_logical_shape()` when a `DistributedTensorConfig` is attached.

### Operator dispatch

`TorchTTNNTensor` overrides `__torch_dispatch__`:

```python
@classmethod
def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
    return TENSOR_RUN_IMPLEMENTATION.torch_dispatch(cls, func, types, args, kwargs)
```

This hooks into PyTorch's dispatch mechanism. In `NORMAL` mode, `torch_dispatch` attempts to route operations to TTNN via `DispatchManager.dispatch_to_ttnn_wrapper`; unrecognised operations fall back to `DispatchManager.dispatch_to_torch_wrapper`. In `CPU` mode, every operation goes through the torch path.

### Distributed tensor config

A `DistributedTensorConfig` (from `models/experimental/tt_symbiote/core/run_config.py`) can be attached to a `TorchTTNNTensor` via `set_distributed_tensor_config()`. It carries a `mesh_mapper` (e.g. `ttnn.ShardTensor2dMesh`) and a `mesh_composer` (e.g. `ttnn.ConcatMesh2dToTensor`) used when converting between host and device representations on multi-device meshes.

---

## `register_module_replacement_dict`

Source file: `models/experimental/tt_symbiote/utils/module_replacement.py`

This utility is the entry point for replacing a PyTorch model's submodules with `TTNNModule` equivalents in a single call.

```python
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict

replaced = register_module_replacement_dict(
    model,                          # root nn.Module
    old_class_to_new_class_dict,    # {nn.Linear: TTNNLinear, ...}
    model_config=my_config,         # passed to TTNNModule.set_model_config()
    exclude_replacement={"encoder.layer[0].attention"},  # optional name set
)
```

### How it works

1. `register_module_replacement_dict` first calls `model.named_modules()` to build a `{module_object: name_string}` mapping. This gives every existing `nn.Module` a stable dotted name before any replacements occur.

2. It then calls `register_module_replacement_dict_with_module_names`, which walks the tree recursively. For `nn.Module` nodes it iterates `model._modules` (the canonical `OrderedDict` of named children) and replaces matching entries in-place:

```python
for name, module in model._modules.items():
    if module.__class__ in old_class_to_new_class_dict:
        new_module = initialize_module(module, ...)
        if new_module is not None:
            model._modules[name] = new_module
```

3. It also handles `dict`, `list`, and `tuple` attributes via `dir(model)` inspection, covering cases where submodules are stored outside `_modules`.

4. `initialize_module` calls `new_class.from_torch(old_module)`, assigns `_unique_name` from the pre-built name map, calls `override_children_module_names()`, and calls `set_model_config(model_config)`.

5. The `exclude_replacement` parameter is a `set[str]` of module names to skip. If a module's pre-replacement name appears in this set, `initialize_module` returns `None` and the original `nn.Module` is left in place.

6. The return value is a `Dict[str, TTNNModule]` mapping each replaced module's `module_name` to the new instance.

---

## Weight lifecycle: full sequence

The following sequence summarises the complete lifecycle from a user's perspective:

```python
# 1. Build PyTorch model as normal
torch_model = MyTransformerModel()

# 2. Replace submodules
replaced_modules = register_module_replacement_dict(
    torch_model,
    {nn.Linear: TTNNLinear},
    model_config={"some_key": "some_value"},
)

# 3. Assign device
for m in replaced_modules.values():
    m.to_device(mesh_device)
    m.set_device_state()

# 4. First call triggers preprocess -> move -> forward automatically
# nn.Module.__call__ on root model → dispatches to replaced TTNNModule children,
# each child's __call__ routes through TENSOR_RUN_IMPLEMENTATION.module_run
output = torch_model(input_ids)
```

See Phase 1–3 above for per-phase implementation details.

---

## `DeviceArch` and `@run_on_devices`

Source file: `models/experimental/tt_symbiote/core/module.py`

`DeviceArch` is an `Enum` enumerating all supported Tenstorrent hardware targets:

```python
from models.experimental.tt_symbiote.core.module import DeviceArch

class DeviceArch(Enum):
    N150 = "n150"
    N300 = "n300"
    T3K  = "t3k_wh"
    TG   = "gx_wh"
    P150 = "p150"
    P300 = "p300"
    P150x4 = "p150x4"
    P150x8 = "p150x8"
    BHGLX  = "bhglx"
```

The `@run_on_devices` decorator restricts a `forward` method to a declared subset of architectures. It reads the `MESH_DEVICE` environment variable, resolves it against `MeshShapeToDeviceArch`, and raises `RuntimeError` at call time if the current hardware is not in the allowed set:

```python
from models.experimental.tt_symbiote.core.module import run_on_devices, DeviceArch

class MyTGOnlyModule(TTNNModule):
    @run_on_devices(DeviceArch.TG)
    def forward(self, x):
        ...
```

This is the mechanism by which Symbiote modules can be hardware-gated without branching inside `forward`. The decorator wraps the method; the gate check runs before any TTNN operations are issued.

---

**Next:** [`tt_transformers_internals.md`](./tt_transformers_internals.md)
