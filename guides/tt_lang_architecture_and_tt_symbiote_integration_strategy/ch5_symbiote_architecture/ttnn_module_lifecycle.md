# TTNNModule Lifecycle

**Source:** `models/experimental/tt_symbiote/core/module.py`

The `TTNNModule` base class is the foundation of TT-Symbiote's module-level acceleration. Every TTNN-accelerated layer --- linear, normalization, attention, MoE, embedding --- inherits from it. Understanding its lifecycle is essential for identifying where TT-Lang can reduce friction.

## The TTNNModule Base Class

`TTNNModule` is *not* a subclass of `torch.nn.Module`. It is a standalone class that manages its own device, weight state, and child modules:

```python
class TTNNModule:
    def __init__(self):
        self._device = None
        self._preprocessed_weight = False
        self._weights_on_device = False
        self._fallback_torch_layer = None
        self._unique_name = None
        self._device_state: Optional[DistributedConfig] = None
        self._model_config = {}
        self._bypass_tensor_wrapping = False
```

Key design decisions:

- **Fallback layer**: Every `TTNNModule` stores its original PyTorch layer in `_fallback_torch_layer`, enabling `from_torch()` factory methods that convert PyTorch modules to TTNN equivalents.
- **State flags**: `_preprocessed_weight` and `_weights_on_device` guard against double-preprocessing and double-movement. These are checked with assertions during traced execution.
- **No `nn.Module` inheritance**: This means no automatic `parameters()`, `state_dict()`, or `named_modules()` from PyTorch. TT-Symbiote re-implements `named_modules()` and `named_children()` by inspecting `__dict__`.

## The 3-Phase Lifecycle

Every `TTNNModule` follows a strict 3-phase lifecycle:

```
Phase 1: preprocess_weights()
    |  Convert PyTorch tensors to TTNN host tensors
    |  (dtype conversion, layout, tiling)
    v
Phase 2: move_weights_to_device()
    |  Transfer host tensors to device DRAM
    |  (requires device to be set)
    v
Phase 3: forward() ... deallocate_weights()
    |  Execute computation on device
    |  Optionally free device memory afterward
```

### Phase 1: `preprocess_weights()`

The public method checks the `_preprocessed_weight` flag, then delegates to `preprocess_weights_impl()`. The default implementation recursively preprocesses children:

```python
def preprocess_weights_impl(self):
    for child in self.__dict__.values():
        if isinstance(child, TTNNModule):
            child.preprocess_weights()
    return self
```

Concrete example from `TTNNLinear`:

```python
def preprocess_weights_impl(self):
    self.tt_weight_host = preprocess_linear_weight(
        self.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    self.tt_bias_host = None
    if self.bias is not None:
        self.tt_bias_host = preprocess_linear_bias(
            self.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
```

This converts `torch.Tensor` weights into TTNN host tensors with the correct dtype (`bfloat16`) and layout (`TILE_LAYOUT`). The host tensors live in CPU memory.

### Phase 2: `move_weights_to_device()`

Asserts that weights are preprocessed and a device is set, then delegates to `move_weights_to_device_impl()`:

```python
def move_weights_to_device_impl(self):
    self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
    self.tt_bias = ttnn.to_device(self.tt_bias_host, self.device) \
        if self.tt_bias_host is not None else None
```

For distributed configurations, this phase becomes more complex. `TTNNLinearInputShardedWeightSharded` must apply mesh mappers during device transfer:

```python
def move_weights_to_device_impl(self):
    if isinstance(self.tt_weight_host, torch.Tensor):
        self.tt_weight_host = preprocess_linear_weight(
            self.tt_weight_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            weights_mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(
                self.device, dim=self.weight_dim
            ),
        )
    # ... similar for bias
    self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
```

### Phase 3: `forward()` and `deallocate_weights()`

`forward()` is abstract --- subclasses must implement it. `deallocate_weights()` frees device memory:

```python
def deallocate_weights(self):
    self.deallocate_weights_impl()
    self._weights_on_device = False
```

The `@deallocate_weights_after` decorator automates cleanup for modules that should free weights after each forward pass (common in LLaMA-optimized variants):

```python
@deallocate_weights_after
def forward(self, input_tensor):
    return super().forward(input_tensor)
```

## The Boilerplate Burden

For every new `TTNNModule` subclass, a developer must implement up to **4 methods**:

1. `preprocess_weights_impl()` --- convert weights to TTNN format
2. `move_weights_to_device_impl()` --- transfer to device
3. `deallocate_weights_impl()` --- free device memory
4. `forward()` --- the actual computation

Plus a `from_torch()` class method for PyTorch conversion. Across the codebase, this pattern repeats identically for dozens of modules, with only the weight names and TTNN ops changing.

**Pain point:** A `TTNNLinear` subclass requires ~30 lines of lifecycle boilerplate before any computation logic. Multiply by the number of sharding variants (`TTNNLinearInputShardedWeightSharded`, `TTNNLinearIColShardedWRowSharded`, `TTNNLinearIColShardedWAllReduced`, `TTNNLinearIReplicatedWColSharded`) and precision variants (`TTNNLinearLLama` with `bfloat8_b`, `TTNNLinearLLamaBFloat16`), and the boilerplate grows rapidly.

## Device Architecture Constraints

The `DeviceArch` enum and `@run_on_devices` decorator restrict module execution to specific hardware:

```python
class DeviceArch(Enum):
    N150 = "n150"
    N300 = "n300"
    T3K = "t3k_wh"
    TG = "gx_wh"
    P150 = "p150"
    P300 = "p300"
    P150x4 = "p150x4"
    P150x8 = "p150x8"
    BHGLX = "bhglx"
```

Usage:

```python
@run_on_devices(DeviceArch.T3K)
def forward(self, input_tensor):
    # Only runs on T3K mesh
    ...
```

The decorator checks the `MESH_DEVICE` environment variable at runtime. This is a runtime check, not a compile-time guarantee.

**Pain point:** Device constraints are scattered across individual `forward()` methods. There is no centralized registry of which modules support which architectures. A TT-Lang approach could encode device constraints declaratively and validate them at compile time.

## Distributed Configuration

**Source:** `core/run_config.py`

Three dataclasses manage distributed execution:

### `DistributedTensorConfig`

```python
@dataclass
class DistributedTensorConfig:
    mesh_mapper: Any          # How to distribute tensor across mesh
    mesh_composer: Any        # How to reassemble from mesh
    logical_shape_fn: Any     # Map sharded shape -> logical shape
```

### `DistributedConfig`

```python
@dataclass
class DistributedConfig:
    mesh_device: Any
    tensor_config: Optional[DistributedTensorConfig] = None
    ccl_manager: Optional[Any] = None
```

`DistributedConfig.__post_init__` auto-creates a `ShardTensor2dMesh` mapper and `ConcatMesh2dToTensor` composer when the mesh has multiple devices, defaulting to batch-channel sharding along dimensions `(0, -1)`.

### `CCLManagerConfig`

```python
@dataclass
class CCLManagerConfig:
    mesh_device: Any
    num_links: Optional[int] = None    # defaults to 1
    topology: Optional[Any] = None     # defaults to ttnn.Topology.Linear
```

These configs flow through `set_device_state()` and `set_output_tensors_config()`, which applies the tensor config to module outputs via `tree_map`.

**Pain point:** Distributed configuration is imperative and threaded manually through module hierarchies. Each module that needs custom sharding must override `set_output_tensors_config_impl()`. A declarative approach (as TT-Lang could provide) would let developers specify sharding strategy at the module level and have the framework propagate it automatically.

## Trace Support

The `@trace_enabled` and `@trace_disabled` decorators (from `run_config.py`) control whether a module participates in TTNN trace capture. During traced execution, `preprocess_weights()` and `move_weights_to_device()` assert that weights are already prepared --- they cannot be called lazily during a trace.

```python
@trace_enabled
class TTNNLinear(TTNNModule):
    ...

@trace_disabled
class TTNNLinearLLama(TTNNLinear):
    ...
```

The LLaMA variants disable tracing because they use `@deallocate_weights_after`, which dynamically frees buffers --- incompatible with trace capture's requirement for stable buffer addresses.

**Pain point:** The interaction between trace capture, weight deallocation, and CCL operations (e.g., `reduce_scatter` + `all_gather` instead of `all_reduce` for trace compatibility) is a source of subtle bugs. TT-Lang could encode these constraints in the type system.

---

**Next:** [`dispatch_system.md`](./dispatch_system.md)
