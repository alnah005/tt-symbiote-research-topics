# `TTNNModule` — the base class for all TTNN-accelerated layers

`TTNNModule` (defined in `models/experimental/tt_symbiote/core/module.py`) replaces `torch.nn.Module` for every layer that targets Tenstorrent hardware. It provides a structured weight lifecycle, automatic recursion into child modules, and two decorators that cover the most common forward-pass patterns.

## Instance state

Every `TTNNModule` carries these instance attributes, set to their initial values by `__init__`:

| Attribute | Type | Initial value | Purpose |
|---|---|---|---|
| `_device` | device or `None` | `None` | The TTNN device (or mesh device) this module runs on |
| `_preprocessed_weight` | `bool` | `False` | Guards against double preprocessing |
| `_weights_on_device` | `bool` | `False` | Guards against double upload |
| `_fallback_torch_layer` | `nn.Module` or `None` | `None` | PyTorch layer used by `from_torch` |
| `_unique_name` | `str` or `None` | `None` | Human-readable identifier; auto-generated from class name and `id` on first access |
| `_device_state` | `DistributedConfig` or `None` | `None` | Distributed mesh configuration |
| `_model_config` | `dict` | `{}` | Arbitrary per-model configuration dictionary |

## The three-phase weight lifecycle

Weights go through three explicit phases before a forward pass can run. The separation exists because each phase has a different cost and a different point in program flow where it makes sense to execute:

### Phase 1 — `preprocess_weights()`

Performs any CPU-side transformation that must happen before weights reach the device: quantisation, tiling, dtype conversion, etc. `preprocess_weights()` guards against redundant calls via `_preprocessed_weight`, but the flag is set to `True` **before** `preprocess_weights_impl()` executes. If `impl` raises, the flag remains `True` and subsequent calls are silently skipped — making error recovery by retrying non-functional. On error, reset `_preprocessed_weight = False` manually before retrying.

```python
module.preprocess_weights()   # sets _preprocessed_weight = True on first call
```

Subclasses override `preprocess_weights_impl`. The base implementation of `preprocess_weights_impl` iterates every attribute and calls `preprocess_weights()` on any child `TTNNModule` it finds, so a top-level call on the root module cascades automatically through the full hierarchy.

### Phase 2 — `move_weights_to_device()`

Uploads preprocessed weights to the device. Requires that both `_preprocessed_weight` is `True` and that `self.device` is not `None`; violations raise `AssertionError` with the module name in the message.

```python
module.to_device(device)           # sets _device
module.move_weights_to_device()    # sets _weights_on_device = True on first call
```

Subclasses override `move_weights_to_device_impl`. The base implementation recurses into child `TTNNModule` instances in the same way as phase 1.

> **Flag-before-impl trap:** Same pattern as phase 1 — reset `_weights_on_device = False` manually before retrying after a failed `move_weights_to_device_impl()` call.

### Phase 3 — `deallocate_weights()`

Frees device SRAM occupied by the weights. Sets `_weights_on_device = False` so the guards in phase 2 reset and weights can be re-uploaded if needed.

```python
module.deallocate_weights()   # calls deallocate_weights_impl, then resets flag
```

Subclasses override `deallocate_weights_impl`. The base implementation recurses into child `TTNNModule` instances.

### Why the separation matters

Preprocessing is a one-time CPU cost; it can happen at model-load time, well before any device is available. Moving weights to device is a PCIe/fabric transfer; it is expensive and should happen as late as possible (e.g., just before the first batch). Deallocation is critical for models that exceed available on-device SRAM: by deallocating one layer's weights after its forward pass, the next layer can borrow that memory.

## `__call__` and `forward`

`TTNNModule.__call__` does not call `forward` directly. It delegates to `TENSOR_RUN_IMPLEMENTATION.module_run(self, ...)`, which is resolved at import time from the run configuration. This indirection is what allows the same module code to run under different execution modes (default TTNN, CPU fallback, debug trace, etc.) without any change to the module itself.

Subclasses must implement `forward`. Calling a module instance with unimplemented `forward` raises `NotImplementedError`.

## Decorators

### `deallocate_weights_after`

Wraps a `forward` method so that `self.deallocate_weights()` is called automatically after the forward pass returns. Use this when a layer's weights should not remain on device between calls.

```python
from models.experimental.tt_symbiote.core.module import deallocate_weights_after

class MyLayer(TTNNModule):
    @deallocate_weights_after
    def forward(self, x):
        return ttnn.linear(x, self.tt_weight)
```

The decorator uses `functools.wraps`, so the wrapped method's name and docstring are preserved.

### `run_on_devices`

Gates a `forward` method to a specific set of device architectures. At call time, the decorator performs three checks in order, any of which can raise `RuntimeError`:

1. **`self.device is None`** — the module must have a device set via `to_device()` before `forward` is called; if `self.device` is `None`, a `RuntimeError` is raised immediately.
2. **`MESH_DEVICE` absent or unrecognized** — the decorator reads the `MESH_DEVICE` environment variable and looks it up in `MeshShapeToDeviceArch`. If the env var is not set, or its value is not a key in that mapping, `RuntimeError` is raised: "Unable to determine device architecture from MESH_DEVICE environment variable." This check fires before any architecture comparison, so `MESH_DEVICE` must be set to a recognized value before calling any function decorated with `@run_on_devices`.
3. **Architecture not in allowed set** — if the resolved `DeviceArch` is not in the set of architectures passed to `@run_on_devices`, a `RuntimeError` is raised listing the detected and allowed architectures.

```python
from models.experimental.tt_symbiote.core.module import run_on_devices, DeviceArch

class MyLayer(TTNNModule):
    @run_on_devices(DeviceArch.N300, DeviceArch.T3K)
    def forward(self, x):
        return ttnn.linear(x, self.tt_weight)
```

The supported architectures and their `MESH_DEVICE` string values are:

| `DeviceArch` member | `MESH_DEVICE` string |
|---|---|
| `N150` | `"N150"` |
| `N300` | `"N300"` |
| `T3K` | `"T3K"` |
| `TG` | `"TG"` |
| `P150` | `"P150"` |
| `P300` | `"P300"` |
| `P150x4` | `"P150x4"` |
| `P150x8` | `"P150x8"` |
| `BHGLX` | `"BHGLX"` |

Passing zero architectures to `run_on_devices` raises `ValueError` immediately at decoration time.

## Module naming

`module_name` is a property that returns `_unique_name`. If `_unique_name` is `None` (not yet assigned), the property auto-generates a name in the form `ClassName_<id>` and caches it. Names are assigned hierarchically by `set_module_name_recursively` and `override_children_module_names`: a parent module's name becomes the prefix for all descendant names, using dot notation for direct attributes and bracket notation for dict/list members.

```python
# After set_module_name_recursively:
# root._unique_name    == "MyModel"
# root.attn._unique_name  == "MyModel.attn"
# root.layers[0]._unique_name == "MyModel.layers[0]"
```

## `__repr__`

`__repr__` produces a recursive, indented string similar to `torch.nn.Module.__repr__`. It walks `self.__dict__`, skips `_fallback_torch_layer`, and formats every child `TTNNModule` or `torch.nn.Module` as an indented sub-block.

## Module iteration

Two iterators mirror the `torch.nn.Module` interface:

- `named_modules(memo, prefix, remove_duplicate)` — yields `(name, module)` pairs for the module itself and every descendant, depth-first. The `memo` set prevents visiting the same object twice when `remove_duplicate=True` (the default).
- `named_children()` — yields `(name, module)` pairs for immediate children only. It handles direct attributes, dict values, and list/tuple elements. `_fallback_torch_layer` is explicitly excluded.

## `from_torch` class method

```python
layer = MyTTNNLayer.from_torch(existing_torch_module, *constructor_args)
```

Creates a new `TTNNModule` instance and stores `existing_torch_module` in `_fallback_torch_layer`. The `torch_layer` property exposes it. The `train(mode)` method delegates to `torch_layer.train(mode)` when a fallback layer is present.

---

**Next:** [`torch_ttnn_tensor.md`](./torch_ttnn_tensor.md)
