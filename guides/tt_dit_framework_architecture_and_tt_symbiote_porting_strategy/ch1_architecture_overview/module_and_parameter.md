# Module and Parameter Deep Dive

## Prerequisites

Read [`index.md`](./index.md) for the high-level architecture overview.

## Overview

The `Module` and `Parameter` classes in `layers/module.py` are the foundational abstractions of TT-DiT. Every layer, block, and model in the framework inherits from `Module`, and every learnable weight is stored as a `Parameter`. Together they define a self-contained module system that is independent of PyTorch's `nn.Module` -- a deliberate design choice that gives TT-DiT full control over weight lifecycle, device placement, and serialization.

This file also covers two supporting classes: `ModuleList` (an ordered container of modules) and `UnregisteredModule` (a proxy that prevents automatic child registration).

## Module

### Class Definition

```python
# layers/module.py
class Module(ABC):
    def __init__(self) -> None:
        self._children = {}
        self._parameters = {}
        self._is_loaded = False
        self.unload_set = None
```

`Module` is an abstract base class (ABC). Its constructor initializes four pieces of internal state:

- `_children`: an ordered dictionary mapping names to child `Module` instances.
- `_parameters`: an ordered dictionary mapping names to `Parameter` instances.
- `_is_loaded`: a boolean flag tracking whether weights have been loaded.
- `unload_set`: an optional set of modules that must be deallocated before this module can be loaded (used for memory-constrained sharing).

### Automatic Registration via `__setattr__`

TT-DiT uses Python's `__setattr__` mechanism to automatically register child modules and parameters:

```python
# layers/module.py
def __setattr__(self, name: str, value: Any) -> None:
    super().__setattr__(name, value)

    if name in ("_children", "_parameters"):
        return

    children = self.__dict__.get("_children")
    parameters = self.__dict__.get("_parameters")

    if isinstance(value, Module):
        if children is None:
            msg = "cannot assign child module before Module.__init__() call"
            raise AttributeError(msg)
        self._children[name] = value
    elif isinstance(value, Parameter):
        if parameters is None:
            msg = "cannot assign parameter before Module.__init__() call"
            raise AttributeError(msg)
        self._parameters[name] = value
    else:
        if children is not None:
            children.pop(name, None)
        if parameters is not None:
            parameters.pop(name, None)
```

This mirrors PyTorch's `nn.Module.__setattr__` pattern: when you assign a `Module` to an attribute, it is automatically tracked as a child; when you assign a `Parameter`, it is tracked as a parameter. Assigning any other type removes the name from both registries. Similarly, `__delattr__` cleans up both dictionaries.

This means subclass constructors can simply write:

```python
class MyLayer(Module):
    def __init__(self, device, dim):
        super().__init__()
        self.norm = RMSNorm(device, dim)  # auto-registered as child
        self.weight = Parameter(total_shape=[dim, dim], device=device)  # auto-registered as parameter
```

### The Module Lifecycle

A module goes through four phases during its lifetime:

#### Phase 1: Construction (`__init__`)

The constructor creates child modules and declares parameters with their shapes, dtypes, layouts, and mesh distribution. No actual weight data exists yet -- parameters are allocated but empty.

#### Phase 2: Weight Loading (`load_torch_state_dict`)

```python
# layers/module.py
def load_torch_state_dict(
    self, state_dict: Mapping[str, torch.Tensor], *, strict: bool = True, on_host: bool = False
) -> IncompatibleKeys:
```

This is the primary entry point for populating a module with weights from a HuggingFace or PyTorch checkpoint. It accepts a `state_dict` (a flat dictionary mapping dotted key names to `torch.Tensor` values) and recursively distributes entries to child modules and parameters.

The recursive loading is handled by `_load_torch_state_dict_inner`:

1. **Call `_prepare_torch_state(state_dict)`** on the current module. This is an optional override point where modules can reshape, transpose, merge, or rename entries in the state dict before they are passed to children. For example, `Linear._prepare_torch_state` transposes weight matrices; `Attention._prepare_torch_state` merges separate Q/K/V weights into a fused QKV tensor.

2. **Iterate over `named_children()`**: For each child, extract its sub-state using `pop_substate(state_dict, name)` (which strips the `name.` prefix from matching keys) and recursively call `_load_torch_state_dict_inner` on the child.

3. **Iterate over `named_parameters()`**: For each parameter, look up its name in the remaining state dict and call `parameter.load_torch_tensor(tensor, on_host=on_host)`.

4. **Track missing and unexpected keys**: Any parameter name not found in the state dict is added to `missing_keys`; any remaining entries in the state dict after processing all children and parameters are added to `unexpected_keys`.

5. **Set `_is_loaded = True`** and return an `IncompatibleKeys` named tuple. If `strict=True` and there are missing or unexpected keys, a `ValueError` is raised.

#### Phase 3: Forward Pass (`forward` / `__call__`)

```python
# layers/module.py
@abstractmethod
def forward(self, *args: Any, **kwargs: Any) -> Any:
    pass

def __call__(self, *args: Any, **kwargs: Any) -> Any:
    return self.forward(*args, **kwargs)
```

Every module must implement `forward()`. Calling the module directly (via `__call__`) delegates to `forward()`. There are no hooks, no autograd integration, and no dispatch interception -- `forward` directly calls TTNN operations.

#### Phase 4: Deallocation (`deallocate_weights`)

```python
# layers/module.py
def deallocate_weights(self) -> None:
    for _, child in self.named_children():
        child.deallocate_weights()
    for _, parameter in self.named_parameters():
        parameter.deallocate()
    self._is_loaded = False
```

Recursively calls `ttnn.deallocate` on every parameter's data tensor, freeing device memory. Sets `_is_loaded` back to `False` so the module can be reloaded later.

### The `_prepare_torch_state` Hook

```python
# layers/module.py
def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
    """Prepare a PyTorch state_dict in place before loading.

    Override this method to adjust entries before loading them into
    submodules and parameters.
    """
```

This is the key extensibility point for weight transformation. It operates on a mutable dictionary of PyTorch tensors *before* they are converted to TTNN format. Subclasses override it to transform weights before device placement (transposing, merging QKV, padding, chunking, renaming), as described in the weight loading section above. Because it receives PyTorch tensors, all reshaping happens on the host CPU.

### Serialization: `save` and `load`

TT-DiT provides a binary serialization format for caching converted weights:

```python
# layers/module.py
def save(self, directory: str | Path, /, *, prefix: str = "") -> None:
    directory = Path(directory)
    directory.mkdir(exist_ok=True, parents=True)
    for name, child in self.named_children():
        child.save(directory, prefix=f"{prefix}{name}.")
    for name, parameter in self.named_parameters():
        parameter.save(directory / f"{prefix}{name}.tensorbin")
```

Each parameter is saved as a `.tensorbin` file using `ttnn.dump_tensor`. The file stores the tensor data along with its metadata (shape, dtype, layout, memory config). Loading reverses this via `ttnn.load_tensor`:

```python
# layers/module.py
def load(self, directory: str | Path, /, *, prefix: str = "") -> None:
    for name, child in self.named_children():
        child.load(directory, prefix=f"{prefix}{name}.")
    for name, parameter in self.named_parameters():
        path = directory / f"{prefix}{name}.tensorbin"
        parameter.load(path)
    self._is_loaded = True
```

This cache bypasses the expensive `_prepare_torch_state` + `from_torch` conversion on subsequent runs. The pipeline's `utils/cache.py` manages these caches with a `config_id` to ensure parallel-configuration-specific caches are not mixed.

### The `set_unload_set` Mechanism

```python
# layers/module.py
def set_unload_set(self, *args: Module) -> None:
    self.unload_set = set(args)
```

Pipelines use this to declare mutual exclusion groups. For example, in the SD3.5 pipeline, the text encoders and the DiT transformer share a submesh. Before loading the transformer, the pipeline checks if any module in its `unload_set` is loaded and deallocates it first. This enables running models whose total weight size exceeds device memory.

## Parameter

### Class Definition and Initialization

```python
# layers/module.py
class Parameter:
    def __init__(
        self,
        *,
        total_shape: Sequence[int],
        device: ttnn.MeshDevice,
        layout: ttnn.Layout = ttnn.Layout.TILE,
        dtype: ttnn.DataType = ttnn.bfloat16,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
        pad_value: float | None = None,
        mesh_axes: Sequence[int | None] | None = None,
        on_host: bool = False,
    ) -> None:
```

A `Parameter` describes a single weight tensor. Key fields:

| Field | Purpose |
|-------|---------|
| `total_shape` | The global (unsharded) shape of the weight across all mesh devices |
| `local_shape` | Computed automatically: the per-device shape after mesh sharding |
| `device` | The `ttnn.MeshDevice` on which the parameter lives |
| `layout` | `ttnn.Layout.TILE` (default) or `ttnn.Layout.ROW_MAJOR` |
| `dtype` | `ttnn.bfloat16` (default), `ttnn.float32`, etc. |
| `memory_config` | `ttnn.DRAM_MEMORY_CONFIG` (default) or L1 configs |
| `pad_value` | Optional padding value for tile alignment |
| `mesh_axes` | Per-dimension mesh axis assignments for sharding |
| `on_host` | If `True`, keep the tensor in host memory (used for save/load) |
| `_data` | The actual `ttnn.Tensor` (initially `None`) |

### Mesh Distribution via `mesh_axes`

The `mesh_axes` parameter is how TT-DiT expresses tensor parallelism at the parameter level. It is a sequence with one entry per tensor dimension. Each entry is either:

- `None` -- this dimension is replicated across all devices.
- An integer `i` -- this dimension is sharded across mesh axis `i`.

For example, for a weight with `total_shape=[4096, 1024]` on a mesh with `device.shape=(2, 4)`:

- `mesh_axes=[None, None]` -- replicated on all 8 devices. `local_shape = [4096, 1024]`.
- `mesh_axes=[0, None]` -- sharded along dimension 0 across mesh axis 0 (factor 2). `local_shape = [2048, 1024]`.
- `mesh_axes=[None, 1]` -- sharded along dimension 1 across mesh axis 1 (factor 4). `local_shape = [4096, 256]`.
- `mesh_axes=[0, 1]` -- sharded along both dimensions. `local_shape = [2048, 256]`.

During initialization, `Parameter` validates that each tensor dimension is evenly divisible by the mesh size along its assigned axis, and that no mesh axis is assigned to more than one tensor dimension.

### `load_torch_tensor`

```python
# layers/module.py
def load_torch_tensor(self, torch_tensor: torch.Tensor, /, *, on_host: bool = False) -> None:
    shape = tuple(torch_tensor.shape)
    if shape != self.total_shape:
        msg = f"expected tensor shape {self.total_shape}, got {shape}"
        raise LoadingError(msg)

    data = tensor.from_torch(
        torch_tensor,
        device=self.device,
        layout=self.layout,
        dtype=self.dtype,
        memory_config=self.memory_config,
        pad_value=self.pad_value,
        mesh_axes=self.mesh_axes,
        on_host=self.on_host or on_host,
    )
    self._set_data(data, allow_on_host=on_host)
```

This is where the PyTorch-to-TTNN conversion happens. It:

1. Validates the incoming tensor shape matches `total_shape`.
2. Calls `utils.tensor.from_torch`, which constructs a `ttnn.MeshMapper` from the `mesh_axes` specification and calls `ttnn.from_torch` to convert and distribute the tensor.
3. Calls `_set_data` to store the result, which enforces strict validation of dtype, layout, memory config, device, and local shape.

The conversion and device placement happen in a single step -- there is no separate "preprocess on host" and "move to device" phase. This is a fundamental difference from TT-Symbiote's two-phase approach (see [`comparison_with_ttnnmodule.md`](./comparison_with_ttnnmodule.md)).

### Data Property and Validation

```python
# layers/module.py
@property
def data(self) -> ttnn.Tensor:
    if self._data is None:
        msg = "parameter has no data"
        raise RuntimeError(msg)
    return self._data

@data.setter
def data(self, value: ttnn.Tensor) -> None:
    self._set_data(value)
```

The `data` property provides access to the underlying `ttnn.Tensor`. The setter delegates to `_set_data`, which performs comprehensive validation:

- **Device check**: Ensures the tensor is on the expected device (or on host if `on_host=True`).
- **Dtype check**: Validates `value.dtype == self.dtype`.
- **Layout check**: Validates `value.layout == self.layout`.
- **Memory config check**: Validates `value.memory_config() == self.memory_config`.
- **Shape check**: Validates `value.shape == self.local_shape` (the per-device shape, not the global shape).

This strict validation catches misconfigurations early, rather than producing silent corruption during forward passes.

### Deallocation

```python
# layers/module.py
def deallocate(self) -> None:
    if self._data is not None:
        ttnn.deallocate(self._data)
        self._data = None
```

Frees the device memory backing the parameter tensor. After deallocation, accessing `self.data` will raise a `RuntimeError`.

### Save and Load

```python
# layers/module.py
def save(self, path: str | Path, /) -> None:
    ttnn.dump_tensor(path, self.data)

def load(self, path: str | Path, /) -> None:
    tensor = ttnn.load_tensor(path, device=None if self.on_host else self.device)
    self.data = tensor  # triggers _set_data validation
```

Binary serialization uses TTNN's native `dump_tensor`/`load_tensor`. On load, the tensor is placed directly on the target device (or kept on host if `on_host=True`), bypassing the `from_torch` conversion entirely.

## ModuleList

```python
# layers/module.py
class ModuleList(Module):
    def __init__(self, modules: Iterable[Module] = ()) -> None:
        super().__init__()
        for i, m in enumerate(modules):
            self.add_module(str(i), m)
```

`ModuleList` is a container for an ordered sequence of modules, analogous to `torch.nn.ModuleList`. Key properties:

- Children are stored with string index keys (`"0"`, `"1"`, ...).
- Supports `len()`, integer indexing, and slice indexing (returns a new `ModuleList`).
- `forward()` raises a `RuntimeError` -- callers should iterate over the list and call each module individually.
- `append(module)` adds a new module at the next index.

This is used throughout TT-DiT to hold the repeated transformer blocks in each model. For example, `SD35Transformer2DModel` stores its `N` `TransformerBlock` instances in a `ModuleList`.

## UnregisteredModule

```python
# layers/module.py
class UnregisteredModule:
    def __init__(self, module: Module) -> None:
        self.module = module

    def __getattr__(self, name: str) -> Any:
        return getattr(self.module, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)
```

`UnregisteredModule` wraps a `Module` instance to prevent it from being registered as a child when assigned to an attribute. Since it is not an instance of `Module`, the `__setattr__` auto-registration logic ignores it.

This is used when multiple modules need to reference the same underlying module instance without creating duplicate entries in the module tree. For example, some model architectures share the same attention projection weights between spatial and prompt pathways. Wrapping one reference in `UnregisteredModule` ensures the shared weights are only loaded and saved once.

The proxy is transparent: `__getattr__` forwards all attribute access to the wrapped module, and `__call__` forwards to the wrapped module's `forward` method.

## Helper Classes and Utilities

### IncompatibleKeys

```python
# layers/module.py
class IncompatibleKeys(NamedTuple):
    missing_keys: list[str]
    unexpected_keys: list[str]
```

Returned by `load_torch_state_dict` to report any keys that were present in the state dict but not consumed (unexpected), or keys expected by parameters but not found in the state dict (missing).

### LoadingError

```python
# layers/module.py
class LoadingError(Exception):
    pass
```

A dedicated exception for loading failures, providing more specific error messages that include the full module path where the failure occurred. The `_load_torch_state_dict_inner` method catches generic exceptions during child loading and re-raises them as `LoadingError` with the module key prefix prepended.

### The `pop_substate` Utility

From `utils/substate.py`:

```python
# utils/substate.py
def pop_substate(state: MutableMapping[str, torch.Tensor], key: str) -> dict[str, torch.Tensor]:
    prefix = f"{key}."
    return {k.removeprefix(prefix): state.pop(k) for k in list(state) if k.startswith(prefix)}
```

This extracts and removes all entries with a given prefix from a state dict, stripping the prefix from the keys. It is the mechanism by which `_load_torch_state_dict_inner` partitions a flat state dict into per-child substates. Companion utilities `substate` (non-destructive extraction), `has_substate` (existence check), `rename_substate` (key renaming), and `indexed_substates` (extracting numerically indexed sub-states for `ModuleList`) are also available.

## Key Takeaways

- `Module`: standalone ABC with automatic registration, recursive state loading, serialization, and deallocation.
- `Parameter`: declarative shape + mesh + dtype/layout specification with one-step PyTorch conversion.
- `_prepare_torch_state`: centralizes per-module weight transformations before device placement.
- `set_unload_set`: enables memory sharing between pipeline components on constrained devices.
- `ModuleList` and `UnregisteredModule`: container and aliasing patterns for real-world architectures.

---

**Next:** [`comparison_with_ttnnmodule.md`](./comparison_with_ttnnmodule.md)
