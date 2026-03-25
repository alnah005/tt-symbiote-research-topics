# Device Setup

Device setup binds every `TTNNModule` in the model graph to a TTNN device, installs forward-timing hooks on every `nn.Module`, initializes `DistributedConfig` for multi-device meshes, and optionally saves a visualization of the model graph.

**Source file:** `models/experimental/tt_symbiote/utils/device_management.py`

Related sources:
- `models/experimental/tt_symbiote/core/module.py` — `TTNNModule.to_device`, `set_device_state`, `DeviceArch`, `run_on_devices`
- `models/experimental/tt_symbiote/core/run_config.py` — `DistributedConfig`, `DistributedTensorConfig`, `CCLManagerConfig`, `DispatchManager`

---

## Public Entry Point: `set_device`

```python
def set_device(obj, device, device_init=DeviceInit, **kwargs):
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `obj` | `nn.Module` or `TTNNModule` | — | Root model to traverse. Either type is accepted; the traversal behaves differently depending on which branch the root falls into (see Recursive Traversal below). |
| `device` | TTNN device or mesh device | — | Device handle assigned to every `TTNNModule` found during traversal. For multi-device setups this is a mesh device whose `get_num_devices()` returns a value greater than 1. |
| `device_init` | class | `DeviceInit` | Class used to cache and create `DistributedConfig` per device. The default is `DeviceInit` itself. Callers can pass a subclass that overrides `init_state_impl` to control how `DistributedConfig` is constructed. |
| `register_forward_hook` | `bool` (kwarg) | `True` | When `True`, wraps each `nn.Module`'s `forward` method (or `__call__` if `forward` is absent) with the `timed_call` closure that records wall-clock timing in `DispatchManager`. |
| `dump_visualization` | `bool` (kwarg) | `True` | When `True`, calls `draw_model_graph(obj)` after full traversal to save a `model_graph.png` visualization of the model hierarchy. |

### Return value

`set_device` has no return value (`None`). All side effects are applied in-place:
- `module._device` is set on every `TTNNModule` encountered.
- `module._device_state` is set on every multi-device `TTNNModule`.
- `forward` (or `__call__`) is replaced on every `nn.Module` reached.
- `DeviceInit.DEVICE_TO_STATE_DICT` is updated with the `DistributedConfig` for the device.

### Pre-traversal step

If `obj` is an `nn.Module`, `set_device` builds a reverse-lookup map before recursion begins:

```python
module_names = {module: name for name, module in obj.named_modules()}
```

This snapshot is taken once. The inner recursive function uses it to attach the correct dotted name string to each module's timing closure. If `obj` is not an `nn.Module`, `module_names` is left as an empty dict and the name falls back to the `module_name` kwarg passed into the recursion (initially `None`, resolving to `""`).

### Post-traversal step

After `_set_device_recursive(obj)` returns, `set_device` checks:

```python
if kwargs.get("dump_visualization", True):
    draw_model_graph(obj)
```

To suppress visualization:

```python
set_device(model, device, dump_visualization=False)
```

---

## Recursive Traversal: `_set_device_recursive`

`_set_device_recursive` is a closure defined inside `set_device`. It handles the two root types (`nn.Module` and `TTNNModule`) with distinct logic.

### `nn.Module` path

When `current_obj` is an `nn.Module`:

**Step 1 — Name resolution.**
The module's name is looked up from the pre-built `module_names` dict:

```python
name = module_names.get(current_obj, module_name or "")
```

**Step 2 — Forward hook installation** (when `register_forward_hook=True`).
See the [Forward-Hook section](#forward-hook-timed_call) below for the full details of the closure. In brief:
- If `current_obj.forward` exists and does not already have `_is_timed`, it is wrapped and `_is_timed = True` is set.
- If `forward` is absent but `__call__` exists and is not already timed, `__call__` is wrapped instead.

**Step 3 — `_modules` iteration.**
For each `(child_name, module)` in `current_obj._modules.items()`:
- If `module is None`, the entry is skipped.
- If the child is a `TTNNModule`, `_initialize_module_on_device(module, device, device_init)` is called.
- Regardless of type, `_set_device_recursive(module)` is called unconditionally (this allows reaching nested `nn.Module` children inside `TTNNModule` containers).

**Step 4 — Public attribute scan** (`dir` scan, skipping names that start with `_`).
Each non-private attribute is retrieved via `getattr`. Any `AttributeError` or other exception is silently caught and skipped. For each successfully retrieved value, three shapes are handled:
- Bare `TTNNModule` attribute: calls `_initialize_module_on_device`, then `_set_device_recursive`.
- `dict` attribute: for each value `v` in the dict, if `v` is a `TTNNModule` calls `_initialize_module_on_device`; then calls `_set_device_recursive(v)` on all values.
- `list` or `tuple` attribute: for each element `v`, if `v` is a `TTNNModule` calls `_initialize_module_on_device`; then calls `_set_device_recursive(v)` on all elements.

### `TTNNModule` path

When `current_obj` is not an `nn.Module` but is a `TTNNModule`:

1. `_initialize_module_on_device(current_obj, device, device_init)` is called immediately on the node itself.
2. A public attribute scan (similar to Step 4 above, but broader) is applied to `current_obj`. There is no `_modules` step because `TTNNModule` does not maintain a `_modules` dict. Unlike Step 4, bare attributes that are plain `nn.Module` instances (not `TTNNModule`) are also recursed into — the check is `isinstance(value, (nn.Module, TTNNModule))` rather than `isinstance(value, TTNNModule)`.

> **Warning:** Attributes whose names start with `_` are skipped in both paths. Any `TTNNModule` stored in a private attribute (e.g., `self._attn`) is not automatically discovered or bound to the device.

> **Note:** There is no `forward` hook installation in the `TTNNModule` path. Timing hooks are only installed on `nn.Module` instances.

---

## `_initialize_module_on_device`

```python
def _initialize_module_on_device(module: "TTNNModule", device, device_init=DeviceInit):
```

This is the per-`TTNNModule` initialization call made by `_set_device_recursive`. It performs two operations in order:

**1. `module.to_device(device)`**

Defined in `core/module.py`:

```python
def to_device(self, device):
    self._device = device
    return self
```

This simply assigns `device` to `self._device`. The `device` property is a read-through accessor for `self._device`. No data movement happens at this point; weights remain wherever they are.

**2. Conditional `set_device_state` call**

```python
if device.get_num_devices() > 1:
    module.set_device_state(device_init.init_state(device))
```

`set_device_state` is only called when the device is a multi-device mesh. For single-device setups, `module._device_state` is left as `None`.

`set_device_state` is defined in `core/module.py`:

```python
def set_device_state(self, device_state: DistributedConfig = None):
    self._device_state = device_state
    if self._device_state is None:
        self._device_state = DistributedConfig(self.device)
    return self
```

If the argument passed is `None` (which cannot happen via `_initialize_module_on_device` since `init_state` always returns a non-`None` value for multi-device), it falls back to constructing `DistributedConfig(self.device)` directly from the module's device.

---

## `DeviceInit`: Device-State Cache

```python
class DeviceInit:
    DEVICE_TO_STATE_DICT = {}
```

`DeviceInit` is a plain class (not an enum). It acts as a process-level cache for `DistributedConfig` objects, keyed by device handle.

### `init_state`

```python
@classmethod
def init_state(cls, device) -> Optional[DistributedConfig]:
    if device not in cls.DEVICE_TO_STATE_DICT:
        res = cls.init_state_impl(device)
        if res is not None:
            assert isinstance(res, DistributedConfig), f"Expected DistributedConfig, got {type(res)}"
        cls.DEVICE_TO_STATE_DICT[device] = res
    return cls.DEVICE_TO_STATE_DICT[device]
```

The assertion means that a custom `init_state_impl` must return either `None` or a `DistributedConfig` instance; any other return type is a hard error.

### `init_state_impl`

```python
@classmethod
def init_state_impl(cls, device) -> DistributedConfig:
    return DistributedConfig(device)
```

The default implementation constructs a `DistributedConfig` by passing the device handle directly. To customize initialization, subclass `DeviceInit` and override this method, then pass the subclass as `device_init` to `set_device`:

```python
class MyDeviceInit(DeviceInit):
    @classmethod
    def init_state_impl(cls, device) -> DistributedConfig:
        return DistributedConfig(
            mesh_device=device,
            tensor_config=MyCustomTensorConfig(...),
        )

set_device(model, mesh_device, device_init=MyDeviceInit)
```

### Cache lifetime

`DEVICE_TO_STATE_DICT` is a class-level attribute. Its lifetime is the process lifetime unless explicitly cleared. If `set_device` is called multiple times for the same device object (e.g., during re-initialization), the cached `DistributedConfig` from the first call is reused for all subsequent calls.

---

## `DistributedConfig`

```python
@dataclass
class DistributedConfig:
    mesh_device: Any
    tensor_config: Optional[DistributedTensorConfig] = None
    ccl_manager: Optional[Any] = None
```

`DistributedConfig` is defined in `core/run_config.py`. It holds everything a `TTNNModule` needs to operate in a multi-device context.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `mesh_device` | TTNN mesh device | The device handle. Always set; this is the constructor's required positional argument. |
| `tensor_config` | `DistributedTensorConfig` or `None` | Describes how tensors are sharded across devices (mapper) and how they are reassembled (composer). `None` for single-device. |
| `ccl_manager` | `TT_CCL` or `None` | Collective-communication manager for all-reduce and similar operations. `None` for single-device. |

### `__post_init__` behavior

`__post_init__` fires when `DistributedConfig(device)` is constructed (either directly or via `DeviceInit.init_state_impl`). It conditionally populates the two optional fields:

```python
def __post_init__(self):
    if self.tensor_config is None and self.mesh_device.get_num_devices() > 1:
        self.tensor_config = DistributedTensorConfig(
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, self.mesh_device.shape, (0, -1)),
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, self.mesh_device.shape, (0, -1)),
            logical_shape_fn=logical_shape_for_batch_channel_sharding(self.mesh_device.shape),
        )
    if self.ccl_manager is None and self.mesh_device.get_num_devices() > 1:
        self.ccl_manager = TT_CCL(self.mesh_device)
```

Both blocks check `get_num_devices() > 1` independently, so it is possible to construct a `DistributedConfig` that sets `tensor_config` but leaves `ccl_manager` as `None` by passing a custom `ccl_manager=some_value` at construction time that is not `None`.

For single-device setups, `__post_init__` leaves both fields as `None`.

### `get_tensor_config_for_tensor`

```python
def get_tensor_config_for_tensor(self, module_name, tensor):
```

This method is a utility for modules to look up the appropriate `DistributedTensorConfig` for a given output tensor. It checks whether the tensor shape is evenly divisible across the mesh topology:

- If `tensor` is `None`, `self.tensor_config` (the default 2D sharding config) is returned directly without any shape checks.
- If `tensor` is not `None` but the tensor has fewer than 2 dimensions, or its last dimension is not divisible by `mesh_device.shape[-1]`, or `tensor.shape[0]` is not divisible by `mesh_device.shape[0]`, it falls back to a replication config using `ttnn.ReplicateTensorToMesh`.
- Otherwise, it returns `self.tensor_config` (the default 2D sharding config).

A warning is printed to stdout when the fallback is applied, prompting the caller to override `set_output_tensors_config_impl` in their module.

---

## `DistributedTensorConfig`

```python
@dataclass
class DistributedTensorConfig:
    mesh_mapper: Any
    mesh_composer: Any
    logical_shape_fn: Optional[Any] = None
```

`DistributedTensorConfig` is defined in `core/run_config.py`. It describes a single tensor's sharding strategy.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `mesh_mapper` | TTNN mesh mapper | Specifies how a tensor is sharded across devices when sending to the mesh. The default created by `DistributedConfig.__post_init__` uses `ttnn.ShardTensor2dMesh` with axes `(0, -1)`, which shards along the batch and last (channel) dimension simultaneously. |
| `mesh_composer` | TTNN mesh composer | Specifies how per-device tensors are reassembled into a single tensor when reading back from the mesh. The default uses `ttnn.ConcatMesh2dToTensor` with axes `(0, -1)`. |
| `logical_shape_fn` | callable or `None` | Optional function that maps a sharded (per-device) shape to the equivalent logical (full) shape. If `None`, `get_logical_shape` returns the sharded shape unchanged. |

### `get_logical_shape`

```python
def get_logical_shape(self, sharded_shape):
    if self.logical_shape_fn is not None:
        return self.logical_shape_fn(sharded_shape)
    return sharded_shape
```

The default `logical_shape_fn` created by `DistributedConfig.__post_init__` is `logical_shape_for_batch_channel_sharding(mesh_device.shape)`:

```python
def logical_shape_for_batch_channel_sharding(mesh_shape):
    def _logical_shape(shape):
        shape = list(shape)
        logical_shape = [shape[0] * mesh_shape[0]] + shape[1:-1] + [shape[-1] * mesh_shape[1]]
        return tuple(logical_shape)
    return _logical_shape
```

For a mesh with shape `(rows, cols)`, this multiplies `shape[0]` by `rows` and `shape[-1]` by `cols`, leaving intermediate dimensions unchanged.

---

## `CCLManagerConfig`

```python
@dataclass
class CCLManagerConfig:
    mesh_device: Any
    num_links: Optional[int] = None
    topology: Optional[Any] = None

    def __post_init__(self):
        if self.num_links is None:
            self.num_links = 1
        if self.topology is None:
            self.topology = ttnn.Topology.Linear
```

`CCLManagerConfig` is defined in `core/run_config.py` and provides configuration for the CCL (Collective Communication Library) manager. It is not created directly by `DeviceInit` or `DistributedConfig.__post_init__`; the `DistributedConfig` creates a `TT_CCL(mesh_device)` directly. `CCLManagerConfig` exists as a separate dataclass for callers that need to configure the CCL layer explicitly (e.g., setting `num_links` to something other than 1 or using a topology other than `ttnn.Topology.Linear`).

---

## Forward-Hook: `timed_call`

When `register_forward_hook=True` (the default), `_set_device_recursive` installs a timing wrapper on each `nn.Module` it visits. The wrapper is a nested closure:

```python
def timed_call(original_call, module_name, module_class):
    @functools.wraps(original_call)
    def new_call(*args, **kwargs):
        begin = time.time()
        DispatchManager.set_current_module_name(module_name)
        result = original_call(*args, **kwargs)
        DispatchManager.set_current_module_name(None)
        end = time.time()
        DispatchManager.record_timing("TorchModules", module_name, module_class, {}, end - begin)
        return result
    return new_call
```

### Which modules receive the hook

The hook is installed during the `nn.Module` path of `_set_device_recursive`. It is applied to every `nn.Module` node visited, regardless of whether the node is also a `TTNNModule`. `TTNNModule` nodes that are reached via the `TTNNModule` path (the else-branch) do not receive this hook.

### Hook target selection

For each `nn.Module`:

1. If `current_obj.forward` exists and `not hasattr(current_obj.forward, "_is_timed")`: the `forward` attribute is replaced with the wrapped version and `current_obj.forward._is_timed = True` is set.
2. Else if `current_obj.__call__` exists and is not timed: `__call__` is replaced instead.

The `_is_timed` attribute is stored directly on the function object, acting as a double-wrap guard. If `set_device` is called again on the same model, the existing wrappers are not replaced.

### What the hook records

Each call to the wrapped forward records a timing entry via:

```python
DispatchManager.record_timing("TorchModules", module_name, module_class, {}, end - begin)
```

`attrs` is always `{}` in this hook; `duration` is wall-clock elapsed time in seconds from `time.time()`.

### `DispatchManager` name stack

The hook calls `DispatchManager.set_current_module_name(module_name)` before the forward pass and `DispatchManager.set_current_module_name(None)` after. `DispatchManager` uses a stack (`_modules_in_progress`):

- Passing a non-`None` name appends it to the stack and updates `current_module_name` to that name.
- Passing `None` pops the stack; `current_module_name` is updated to the new top, or `None` if the stack becomes empty.

This stack correctly handles nested module calls: while module A's forward is executing and calls module B's forward, both `"A"` and `"B"` are on the stack, and `current_module_name` is `"B"` (the innermost). After B's forward returns, `current_module_name` reverts to `"A"`.

`DispatchManager.current_module_name` is also used by `dispatch_to_ttnn_wrapper` and `dispatch_to_torch_wrapper` to attribute per-operation timing to the correct enclosing module.

### Accessing timing data

After inference, timing entries are available via:

```python
df = DispatchManager.get_timing_entries_stats()  # returns a pandas DataFrame
DispatchManager.save_stats_to_file("timings.csv")  # saves CSV and pivot CSV
```

`save_stats_to_file` prints the top 30 functions (by `func_name`) ranked by total `TorchModules` duration to stdout.

---

## Multi-Device Mesh Setup

### Mesh topology fields

`DistributedConfig.__post_init__` reads the mesh topology through `mesh_device.shape`:

| Field | How topology is used |
|-------|---------------------|
| `tensor_config.mesh_mapper` | `ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, (0, -1))` — shards dim-0 and dim-last across the mesh rows and columns. |
| `tensor_config.mesh_composer` | `ttnn.ConcatMesh2dToTensor(mesh_device, mesh_device.shape, (0, -1))` — concatenates per-device tensors back along dim-0 and dim-last. |
| `tensor_config.logical_shape_fn` | `logical_shape_for_batch_channel_sharding(mesh_device.shape)` — uses `mesh_device.shape[0]` and `mesh_device.shape[1]` to reconstruct the logical tensor dimensions. |
| `ccl_manager` | `TT_CCL(mesh_device)` — receives the full mesh device handle. |

Both `ShardTensor2dMesh` and `ConcatMesh2dToTensor` use `mesh_device.shape` to determine the row and column count of the mesh grid, so the sharding axes align with the physical device topology automatically.

### `DeviceArch` and `run_on_devices`

`core/module.py` defines a `DeviceArch` enum and a `run_on_devices` decorator that allow individual `forward` implementations to restrict execution to specific hardware variants:

```python
class DeviceArch(Enum):
    N150  = "n150"
    N300  = "n300"
    T3K   = "t3k_wh"
    TG    = "gx_wh"
    P150  = "p150"
    P300  = "p300"
    P150x4 = "p150x4"
    P150x8 = "p150x8"
    BHGLX = "bhglx"
```

The `MeshShapeToDeviceArch` dict maps the string values of the `MESH_DEVICE` environment variable to these enum members.

`run_on_devices` is a decorator applied to a `forward` method:

```python
@run_on_devices(DeviceArch.N300, DeviceArch.T3K)
def forward(self, input_tensor):
    ...
```

At call time the decorator:
1. Raises `RuntimeError` if `self.device` is `None` or the module has no `device` attribute.
2. Reads `os.environ.get("MESH_DEVICE")` and resolves it through `MeshShapeToDeviceArch`.
3. Raises `RuntimeError` if the environment variable is unset, unrecognized, or maps to an arch not in the allowed set.

`run_on_devices` is independent of `set_device`. It is a per-method guard, not a device-binding mechanism.

---

## Practical Usage Patterns

### Single-device setup

For a single device, `device.get_num_devices()` returns 1. `_initialize_module_on_device` calls `to_device` but skips `set_device_state`. `_device_state` remains `None` on every module.

```python
import ttnn
from models.experimental.tt_symbiote.utils.device_management import set_device

device = ttnn.open_device(device_id=0)
set_device(model, device)
```

Timing hooks are still installed on all `nn.Module` instances. Visualization is still saved unless suppressed.

### Multi-device mesh setup

```python
import ttnn
from models.experimental.tt_symbiote.utils.device_management import set_device

mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))  # e.g., T3K: 1 row, 8 columns
set_device(model, mesh_device)
```

### Custom `DeviceInit` for non-default sharding

To use a different sharding strategy, override `init_state_impl`:

```python
from models.experimental.tt_symbiote.utils.device_management import DeviceInit, set_device
from models.experimental.tt_symbiote.core.run_config import DistributedConfig, DistributedTensorConfig
import ttnn

class RowOnlyDeviceInit(DeviceInit):
    @classmethod
    def init_state_impl(cls, device) -> DistributedConfig:
        tensor_config = DistributedTensorConfig(
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0),
            mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0),
        )
        return DistributedConfig(mesh_device=device, tensor_config=tensor_config)

set_device(model, mesh_device, device_init=RowOnlyDeviceInit)
```

### Suppressing hooks and visualization

Both optional behaviors can be disabled independently:

```python
# No timing hooks, no graph image
set_device(model, device, register_forward_hook=False, dump_visualization=False)

# Hooks enabled, no graph image
set_device(model, device, dump_visualization=False)
```

### Selective device binding

There is no built-in mechanism in `set_device` to exclude individual modules from device binding (unlike `register_module_replacement_dict`'s `exclude_replacement`). Every `TTNNModule` reachable via public attributes is bound unconditionally. To prevent a module from being bound, store it in a private attribute (prefixed with `_`), which the traversal skips.

---

## Multi-Device Initialization Summary

| Condition | Action |
|-----------|--------|
| `device.get_num_devices() == 1` | `to_device(device)` only; `_device_state` remains `None` |
| `device.get_num_devices() > 1` | `to_device(device)` + `set_device_state(device_init.init_state(device))` |
| First call for this device handle | `DeviceInit.init_state_impl` creates `DistributedConfig(device)`, which builds `ShardTensor2dMesh` mapper/composer and `TT_CCL` |
| Subsequent calls for same device handle | Cached `DistributedConfig` returned from `DeviceInit.DEVICE_TO_STATE_DICT` |
| `init_state_impl` returns `None` | `set_device_state(None)` is called; inside `set_device_state`, the `None` guard fires and `module._device_state` is set to `DistributedConfig(self.device)` — not `None` |

---

## Weight Preprocessing and Upload After `set_device`

After `set_device` returns, the module dict produced by `register_module_replacement_dict` is used to drive weight preprocessing and upload:

```python
# CPU-side transformations: layout changes, quantization, etc.
for module in module_dict.values():
    module.preprocess_weights()

# Upload to device
for module in module_dict.values():
    module.move_weights_to_device()
```

`preprocess_weights` guards against re-execution with `_preprocessed_weight`:

```python
def preprocess_weights(self):
    if not self._preprocessed_weight:
        self._preprocessed_weight = True
    else:
        return
    self.preprocess_weights_impl()
```

`move_weights_to_device` asserts that preprocessing has run (`_preprocessed_weight` must be `True`) and that a device has been set (i.e., `set_device` was called before this). It guards against double-upload with `_weights_on_device`.

The default `preprocess_weights_impl` and `move_weights_to_device_impl` recurse into child `TTNNModule` instances found in `self.__dict__`. Concrete subclasses override these to perform actual tensor operations.

> **Note:** Because `module_dict` contains only the top-level replaced modules, calling `preprocess_weights()` on each entry recursively reaches nested `TTNNModule` children via the default implementation. Any `TTNNModule` not nested under an entry in `module_dict` is not reached and must be handled separately.

---

**Next:** [Chapter 5 — Built-In TTNN Modules](../ch5_builtin_modules/index.md)
