# Comparison: TT-DiT Module vs TT-Symbiote TTNNModule

## Prerequisites

Read [`module_and_parameter.md`](./module_and_parameter.md) for the TT-DiT Module internals.

## Overview

TT-DiT's `Module` and TT-Symbiote's `TTNNModule` both serve as base classes for TTNN-accelerated neural network components, but they take fundamentally different approaches to the same problem. This file provides a detailed side-by-side comparison, identifies equivalent patterns, and highlights architectural gaps that matter for any porting effort.

## Architectural Philosophy

TT-DiT's `Module` is a standalone ABC (detailed in [`module_and_parameter.md`](./module_and_parameter.md)) that calls TTNN directly. TT-Symbiote's `TTNNModule` wraps existing PyTorch layers and routes operations through `__torch_dispatch__` interception. The tables below detail every difference.

## Side-by-Side Comparison

### Base Class Structure

| Feature | TT-DiT `Module` | TT-Symbiote `TTNNModule` |
|---------|-----------------|--------------------------|
| **Inheritance** | `ABC` (no PyTorch) | Standalone class (no `nn.Module` inheritance, but holds `_fallback_torch_layer`) |
| **Child tracking** | `_children` dict, auto-registered via `__setattr__` | Manual iteration over `__dict__` in `named_children()` |
| **Parameter tracking** | `_parameters` dict with `Parameter` class | No dedicated parameter registry; weights stored as instance attributes |
| **`forward()` contract** | Abstract method, must be implemented | Raises `NotImplementedError` by default |
| **Calling convention** | `__call__` -> `forward()` directly | `__call__` -> `call()` -> `TENSOR_RUN_IMPLEMENTATION.module_run()` -> `forward()` |
| **Dispatch integration** | None; calls TTNN ops directly | Routes through `module_run` which manages tracing, fallback, and tensor wrapping |

### Weight Lifecycle

| Phase | TT-DiT `Module` | TT-Symbiote `TTNNModule` |
|-------|-----------------|--------------------------|
| **Source** | HuggingFace `state_dict` (flat `dict[str, torch.Tensor]`) | `torch.nn.Module` instance (via `from_torch`) |
| **Torch reference** | Not stored; state dict consumed and discarded | Stored as `_fallback_torch_layer` for fallback execution |
| **Host preprocessing** | `_prepare_torch_state()` modifies state dict in-place (transpose, merge, rename) | `preprocess_weights_impl()` converts torch weights to host TTNN tensors |
| **Device placement** | Happens inside `Parameter.load_torch_tensor()` during state dict loading (single step) | Separate `move_weights_to_device_impl()` call (calls `ttnn.to_device`) |
| **Idempotency guards** | `_is_loaded` flag on Module | `_preprocessed_weight` and `_weights_on_device` flags |
| **Deallocation** | `deallocate_weights()` -> recursive `parameter.deallocate()` | `deallocate_weights()` -> `deallocate_weights_impl()` -> recursive child deallocation |
| **Reloading** | Load again after `deallocate_weights()` resets `_is_loaded` | Reset `_weights_on_device = False` then call `move_weights_to_device()` again |

The most significant difference is the number of steps. TT-DiT collapses the entire pipeline into one call:

```
state_dict -> _prepare_torch_state -> Parameter.load_torch_tensor -> ttnn.from_torch (with mesh_axes) -> on device
```

TT-Symbiote requires three separate phases:

```
torch_layer -> from_torch (store reference) -> preprocess_weights_impl (host TTNN) -> move_weights_to_device_impl (device)
```

### Distributed Tensor Handling

| Feature | TT-DiT `Module` | TT-Symbiote `TTNNModule` |
|---------|-----------------|--------------------------|
| **Sharding specification** | Per-`Parameter` `mesh_axes` (dimension -> mesh axis mapping) | `DistributedTensorConfig` with `mesh_mapper` + `mesh_composer` |
| **Shard construction** | `utils.tensor.from_torch` builds `ttnn.MeshMapper` from `mesh_axes` | Uses `ttnn.ShardTensor2dMesh` / `ttnn.ReplicateTensorToMesh` via `DistributedConfig` |
| **Composition (device -> host)** | `utils.tensor.to_torch` with `mesh_axes` builds `ttnn.MeshComposer` | `ttnn.ConcatMesh2dToTensor` via `DistributedTensorConfig.mesh_composer` |
| **Config scope** | Each `Parameter` declares its own `mesh_axes` | Global `DistributedConfig` set on the module, applied uniformly |
| **Shape inference** | `Parameter` computes `local_shape` from `total_shape` and `mesh_axes` | `DistributedTensorConfig.logical_shape_fn` maps sharded shape to logical shape |

TT-DiT's per-parameter approach is more granular: within the same module, one parameter might be column-sharded while another is row-sharded. TT-Symbiote's approach is simpler (one config per module) but requires overriding `set_output_tensors_config_impl` when parameters have different distribution patterns.

### Tracing and Traced Execution

| Feature | TT-DiT | TT-Symbiote |
|---------|--------|-------------|
| **Tracing granularity** | Pipeline level (via `utils/tracing.py` `Tracer` class) | Module level (via `@trace_enabled` decorator) |
| **Trace capture** | `Tracer` wraps a function; first call compiles + captures | `module_run` detects trace mode and captures during second forward pass |
| **Trace replay** | `Tracer.__call__` replays via `ttnn.execute_trace` | `module_run` replays via `ttnn.execute_trace` |
| **Input update** | `Tracer._update_input` with strict shape/dtype checking | `_copy_inputs_to_trace_buffer` / `_copy_kwargs_to_trace_buffer` |
| **Layer stacking** | Not needed (traces entire pipeline) | `TTNNLayerStack` groups layers into single trace unit |
| **Hooks** | None | `pre_trace_execute()` / `post_trace_execute()` for custom buffer management |

### Memory Management

| Feature | TT-DiT | TT-Symbiote |
|---------|--------|-------------|
| **Component swapping** | `set_unload_set(*modules)` declares mutual exclusion | No equivalent; components assumed to coexist |
| **Device reference** | `Parameter.device` set at construction | `TTNNModule._device` set via `to_device()` |
| **Device state** | Not applicable | `DistributedConfig` set via `set_device_state()`, includes CCL manager |
| **Architecture guards** | Not applicable | `@run_on_devices(DeviceArch.T3K, ...)` decorator restricts execution to specific device architectures |

## Equivalent Patterns

### Creating a Simple Linear Layer

**TT-DiT:**
```python
# layers/linear.py (simplified)
class Linear(Module):
    def __init__(self, device, in_features, out_features):
        super().__init__()
        self.weight = Parameter(
            total_shape=[in_features, out_features],  # already transposed
            device=device,
            dtype=ttnn.bfloat16,
        )
        self.bias = Parameter(
            total_shape=[1, out_features],
            device=device,
            dtype=ttnn.bfloat16,
        )

    def _prepare_torch_state(self, state):
        # Transpose PyTorch weight from [out, in] to [in, out]
        if "weight" in state:
            state["weight"] = state["weight"].T

    def forward(self, x):
        matmul_config = get_matmul_config(...)
        return ttnn.experimental.minimal_matmul(
            input_tensor=x,
            weight_tensor=self.weight.data,
            bias_tensor=self.bias.data if self.bias is not None else None,
            config=matmul_config,
        )
```

**TT-Symbiote:**
```python
# (typical TTNNLinear pattern, simplified)
class TTNNLinear(TTNNModule):
    def __init__(self):
        super().__init__()
        self.tt_weight = None
        self.tt_bias = None

    @classmethod
    def from_torch(cls, torch_layer):
        new = cls()
        new._fallback_torch_layer = torch_layer
        return new

    def preprocess_weights_impl(self):
        weight = self.torch_layer.weight.data.T  # transpose
        self.tt_weight = ttnn.from_torch(weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        if self.torch_layer.bias is not None:
            self.tt_bias = ttnn.from_torch(self.torch_layer.bias.data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def move_weights_to_device_impl(self):
        self.tt_weight = ttnn.to_device(self.tt_weight, self.device)
        if self.tt_bias is not None:
            self.tt_bias = ttnn.to_device(self.tt_bias, self.device)

    def forward(self, x):
        return ttnn.linear(x, self.tt_weight, bias=self.tt_bias)
```

Key difference: TT-DiT declares the `Parameter` with its full specification upfront (shape, dtype, layout, mesh distribution). TT-Symbiote stores generic TTNN tensors as instance attributes, manually converting and moving them.

### Creating a Module with Tensor-Parallel Weights

**TT-DiT:**
```python
class ColParallelLinear(Module):
    def __init__(self, device, in_features, out_features, tp_mesh_axis):
        super().__init__()
        self.weight = Parameter(
            total_shape=[in_features, out_features],
            device=device,
            mesh_axes=[None, tp_mesh_axis],  # shard output dim across TP axis
        )
```

**TT-Symbiote:**
```python
class TTNNLinearIReplicatedWColSharded(TTNNModule):
    def preprocess_weights_impl(self):
        weight = self.torch_layer.weight.data.T
        # Shard along columns using ShardTensor2dMesh
        mesh_mapper = ttnn.ShardTensor2dMesh(self.device, self.device.shape, dims=(None, -1))
        self.tt_weight = ttnn.from_torch(weight, ..., mesh_mapper=mesh_mapper)
```

TT-DiT's `mesh_axes` parameter is more declarative and composable: the parallelism configuration is part of the parameter specification, not embedded in the preprocessing logic.

### Module Hierarchy Traversal

**TT-DiT:**
```python
# Automatic via _children dict
for name, child in module.named_children():
    ...
for name, param in module.named_parameters():
    ...
```

**TT-Symbiote:**
```python
# Iterates over __dict__, filtering by type
for name, child in module.named_children():  # checks isinstance(child, (torch.nn.Module, TTNNModule))
    ...
# Also handles dicts and lists of modules in __dict__
```

TT-Symbiote's `named_children` is more permissive, scanning the entire `__dict__` and handling nested containers (dicts, lists, tuples). TT-DiT's is more explicit, only returning items in the `_children` registry.

### Serialization

**TT-DiT:**
```python
module.save("/cache/dir/")       # saves all params as .tensorbin files
module.load("/cache/dir/")       # loads .tensorbin files directly to device
```

**TT-Symbiote:**
No built-in serialization. Weight caching is handled externally or not at all.

## Summary of Gaps

The following capabilities exist in TT-DiT but have no TT-Symbiote equivalent:

| Capability | TT-DiT | TT-Symbiote Equivalent |
|-----------|--------|----------------------|
| `_prepare_torch_state` hook | Flexible per-module weight transformation | Must implement in `preprocess_weights_impl` (less structured) |
| `Parameter` class with `mesh_axes` | Declarative, validated parameter specification | Ad-hoc TTNN tensor attributes |
| `set_unload_set` | Component memory swapping | Not available |
| `.tensorbin` weight caching | Built into Module/Parameter | Not available |
| `UnregisteredModule` | Alias without registration | No equivalent |
| `ModuleList` with slice indexing | Container with `__getitem__` | `TTNNLayerStack` (tracing only, no indexing) |
| `IncompatibleKeys` reporting | Structured missing/unexpected key tracking | Not applicable (wraps existing modules) |

Conversely, TT-Symbiote has capabilities that TT-DiT does not:

| Capability | TT-Symbiote | TT-DiT Equivalent |
|-----------|-------------|-------------------|
| `from_torch` class method | Create TTNN module from PyTorch layer | Not applicable (no PyTorch layer wrapper) |
| `_fallback_torch_layer` | Automatic fallback to PyTorch execution | Not available |
| Dispatch interception | `module_run` routes through tensor wrapping and tracing | Not applicable (direct TTNN calls) |
| `@trace_enabled` / `@trace_disabled` | Per-class trace opt-in/opt-out | Not applicable (traces at pipeline level) |
| `TTNNLayerStack` | Trace-enabled layer grouping | Not needed (pipeline-level tracing) |
| `pre_trace_execute` / `post_trace_execute` | Hooks for custom trace buffer management | Not applicable |
| `DeviceArch` and `@run_on_devices` | Architecture-specific execution guards | Not available |
| `DistributedConfig` + `DistributedTensorConfig` | Unified device + tensor distribution config | Parallelism handled via `DiTParallelConfig` + per-parameter `mesh_axes` |
| `set_output_tensors_config` | Automatic output tensor distribution | Not applicable (explicit in forward methods) |

## Implications for Porting

When porting TT-DiT components to TT-Symbiote, the following translation patterns apply:

1. **Each TT-DiT `Module` subclass becomes a `TTNNModule` subclass.** The `_prepare_torch_state` logic moves into `preprocess_weights_impl`, operating on `self.torch_layer` weights rather than a state dict.

2. **`Parameter` declarations become `ttnn.from_torch` calls in `preprocess_weights_impl`.** The `mesh_axes` specification must be translated to `ShardTensor2dMesh` or equivalent mesh mapper configurations.

3. **`set_unload_set` has no direct mapping.** Pipeline-level memory orchestration would need to be handled by a new TT-Symbiote pipeline abstraction or external orchestration code.

4. **Weight caching via `.tensorbin`** would need a TT-Symbiote equivalent, or the ported code would need to retain TT-DiT's `save`/`load` infrastructure.

5. **Direct TTNN calls in `forward()`** are already compatible with TT-Symbiote's forward pass model. The main change is that inputs arrive as `TorchTTNNTensor` wrappers (when dispatch is active) rather than raw `ttnn.Tensor` objects.

---

**Next:** [Chapter 2 -- Parallelism and CCL Infrastructure](../ch2_parallelism_and_ccl/index.md)
