# TT-Symbiote Weight Loading Pipeline

## Prerequisites

- Familiarity with TT-Symbiote's `TTNNModule` base class ([Chapter 1](../ch1_introduction/index.md)).
- Understanding of the TT-DiT weight pipeline covered in [`tt_dit_weight_pipeline.md`](./tt_dit_weight_pipeline.md).
- Knowledge of TTNN preprocessing utilities (`preprocess_linear_weight`, `preprocess_linear_bias`).

---

## 1. End-to-End Flow

TT-Symbiote uses a three-phase lifecycle for weight management. Unlike TT-DiT's single recursive pass, each phase is a distinct operation that can be invoked independently:

```
PyTorch nn.Module
    |
    v
Phase 1: TTNNModule.from_torch(torch_layer)
    |   - Creates TTNNModule instance
    |   - Stores reference to torch_layer as _fallback_torch_layer
    |   - Copies weight/bias references from torch_layer
    |
    v
Phase 2: module.preprocess_weights()
    |   - Guards against re-preprocessing (_preprocessed_weight flag)
    |   - Calls preprocess_weights_impl()
    |   - Converts torch.Tensor -> ttnn.Tensor on HOST (not device)
    |
    v
Phase 3: module.move_weights_to_device()
    |   - Guards against redundant moves (_weights_on_device flag)
    |   - Calls move_weights_to_device_impl()
    |   - Calls ttnn.to_device() for each weight tensor
    |
    v
Device-resident ttnn.Tensors ready for forward()
```

An optional fourth phase exists for cleanup:

```
Phase 4: module.deallocate_weights()
    |   - Calls deallocate_weights_impl()
    |   - Resets _weights_on_device flag
```

## 2. Phase 1: `from_torch` -- Construction from PyTorch

The `from_torch` class method is the standard entry point. It follows a factory pattern where each concrete module class defines its own conversion logic.

### Base Class Default

```python
class TTNNModule:
    @classmethod
    def from_torch(cls, torch_layer, *args, **kwargs):
        new_layer = cls(*args, **kwargs)
        new_layer._fallback_torch_layer = torch_layer
        return new_layer
```

The base implementation stores the original PyTorch layer for fallback execution and weight access. Subclasses override this to extract specific attributes.

### TTNNLinear

```python
class TTNNLinear(TTNNModule):
    @classmethod
    def from_torch(cls, linear: nn.Linear):
        new_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
        )
        new_linear._fallback_torch_layer = linear
        new_linear.weight = linear.weight    # torch.nn.Parameter reference
        new_linear.bias = linear.bias        # torch.nn.Parameter reference
        return new_linear
```

Key observations:

- The PyTorch `weight` and `bias` are stored as direct attribute references on the TTNNModule, not wrapped in any framework-specific container.
- The `_fallback_torch_layer` reference enables PyTorch-fallback execution when TTNN is unavailable.
- No conversion happens here -- this is purely structural setup.

### TTNNLayerNorm

```python
class TTNNLayerNorm(TTNNModule):
    @classmethod
    def from_torch(cls, layer_norm: nn.LayerNorm):
        if layer_norm.weight is None:
            return layer_norm  # Fall back to PyTorch entirely
        new_layer_norm = cls()
        new_layer_norm._fallback_torch_layer = layer_norm
        return new_layer_norm
```

Normalization layers take a different approach: they do not copy weight references at construction time. Instead, they access weights through `self.torch_layer` during preprocessing. Note the early-return pattern -- if the PyTorch layer lacks a weight, the original `nn.LayerNorm` is returned as-is.

### Alternative: `from_parameters`

Some modules support construction from raw tensors without a PyTorch layer:

```python
class TTNNLinear(TTNNModule):
    @classmethod
    def from_parameters(cls, weight, bias=None):
        new_linear = cls(
            in_features=weight.shape[1],
            out_features=weight.shape[0],
        )
        new_linear.weight = weight
        new_linear.bias = bias
        new_linear.preprocess_weights()
        del new_linear.weight   # Clean up after preprocessing
        del new_linear.bias
        return new_linear
```

This variant immediately triggers preprocessing and then deletes the raw weight references, producing a module ready for device placement.

## 3. Phase 2: `preprocess_weights` -- Host-Side Conversion

The preprocessing phase converts PyTorch tensors into host-resident TTNN tensors. It is guarded by a flag to ensure it runs exactly once:

```python
def preprocess_weights(self):
    if _TRACE_RUNNING:
        assert self._preprocessed_weight, "..."
        return
    if not self._preprocessed_weight:
        self._preprocessed_weight = True
    else:
        return
    self.preprocess_weights_impl()
```

The trace-running guard ensures that during trace replay, no weight preprocessing occurs -- weights must already be on device.

### TTNNLinear Preprocessing

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

The `preprocess_linear_weight` and `preprocess_linear_bias` utilities from `ttnn.model_preprocessing` handle:

- Transposing the weight matrix (PyTorch `[out, in]` to TTNN's expected `[in, out]`)
- Converting to the target dtype
- Applying tile layout padding
- Returning a host-resident `ttnn.Tensor`

The resulting tensors are stored as `tt_weight_host` / `tt_bias_host` -- the `_host` suffix is a naming convention indicating they are not yet on device.

### TTNNLayerNorm Preprocessing

```python
def preprocess_weights_impl(self):
    self.tt_weight = ttnn.from_torch(
        self.torch_layer.weight,
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    self.tt_bias = ttnn.from_torch(
        self.torch_layer.bias,
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
```

LayerNorm accesses weights through the stored `_fallback_torch_layer` reference (exposed as `self.torch_layer`). It calls `ttnn.from_torch` directly rather than using the `preprocess_linear_*` utilities because normalization weights do not need transposition.

### TTNNRMSNorm Preprocessing

```python
def preprocess_weights_impl(self):
    self.tt_weight = ttnn.from_torch(
        self.torch_layer.weight.unsqueeze(0).expand(32, -1),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
```

RMSNorm demonstrates an important pattern: the weight is expanded from `[dim]` to `[32, dim]` during preprocessing. This matches the tile height (32 rows) and avoids runtime broadcasting costs.

### Sharded Linear Preprocessing

For distributed linear layers (`TTNNLinearInputShardedWeightSharded` and its subclasses), preprocessing is deliberately minimal -- it simply stores the raw torch tensors:

```python
class TTNNLinearInputShardedWeightSharded(TTNNLinear):
    def preprocess_weights_impl(self):
        self.tt_bias_host = self.bias
        self.tt_weight_host = self.weight
```

The actual TTNN conversion and sharding happens in Phase 3, because mesh mappers require a device reference.

## 4. Phase 3: `move_weights_to_device` -- Device Placement

The device-placement phase transfers host-resident tensors to the target device. Like preprocessing, it is guarded by a flag:

```python
def move_weights_to_device(self):
    if _TRACE_RUNNING:
        assert self._weights_on_device, "..."
        return
    assert self._preprocessed_weight, "..."
    assert self.device is not None, "..."
    if not self._weights_on_device:
        self._weights_on_device = True
    else:
        return
    self.move_weights_to_device_impl()
```

### TTNNLinear Device Placement

```python
def move_weights_to_device_impl(self):
    self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
    self.tt_bias = (
        ttnn.to_device(self.tt_bias_host, self.device)
        if self.tt_bias_host is not None else None
    )
```

Simple and direct: `ttnn.to_device` places the host tensor onto the target device.

### Sharded Linear Device Placement

For sharded variants, this is where the mesh distribution happens:

```python
class TTNNLinearInputShardedWeightSharded(TTNNLinear):
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
        if isinstance(self.tt_bias_host, torch.Tensor):
            self.tt_bias_host = preprocess_linear_bias(
                self.tt_bias_host,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                weights_mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(
                    self.device, dim=self.input_dim
                ),
            )
        self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
        self.tt_bias = (
            ttnn.to_device(self.tt_bias_host, self.device)
            if self.tt_bias_host is not None else None
        )
```

The `isinstance(self.tt_weight_host, torch.Tensor)` guard handles the case where `move_weights_to_device_impl` might be called after weights have already been converted to TTNN tensors (e.g., after a device reset).

The mesh mapping is **imperative**: each module explicitly creates a `shard_tensor_to_mesh_mapper` with a specific sharding dimension. Compare this with TT-DiT's declarative `mesh_axes` approach.

### TTNNDistributedRMSNorm -- Combined Preprocess and Place

Some modules combine preprocessing and placement in `move_weights_to_device_impl`, skipping `preprocess_weights_impl` entirely:

```python
class TTNNDistributedRMSNorm(TTNNModule):
    def move_weights_to_device_impl(self):
        dim = self.torch_layer.weight.shape[0]
        padded_dim = ((dim + 31) // 32) * 32
        weight = self.torch_layer.weight
        if padded_dim != dim:
            weight = torch.nn.functional.pad(weight, (0, padded_dim - dim), value=1.0)
        self.weight_distributed = ttnn.as_tensor(
            weight.unsqueeze(0).view(1, 1, padded_dim)
                  .reshape([1, 1, padded_dim // 32, 32]).to(torch.bfloat16),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.device, dims=(None, 2),
                mesh_shape=list(self.device.shape)
            ),
        )
        self.weight_distributed = ttnn.to_device(
            self.weight_distributed, self.device
        )
```

This demonstrates a pattern where the three-phase boundary is blurred: all transformation, distribution, and placement occur in a single method call.

## 5. Phase 4: Deallocation

Weight deallocation frees device memory:

```python
class TTNNLinear(TTNNModule):
    def deallocate_weights_impl(self):
        ttnn.deallocate(self.tt_weight)
        if self.tt_bias is not None:
            ttnn.deallocate(self.tt_bias)
        super().deallocate_weights_impl()
```

The `super()` call propagates deallocation to child modules. The `@deallocate_weights_after` decorator automates this for single-use modules:

```python
@deallocate_weights_after
def forward(self, input_tensor):
    return super().forward(input_tensor)
```

## 6. Recursive Propagation

The base `TTNNModule` provides default implementations that propagate each phase to child modules:

```python
def preprocess_weights_impl(self):
    for child in self.__dict__.values():
        if isinstance(child, TTNNModule):
            child.preprocess_weights()
    return self

def move_weights_to_device_impl(self):
    for child in self.__dict__.values():
        if isinstance(child, TTNNModule):
            child.move_weights_to_device()
    return self
```

Unlike TT-DiT's structured `named_children()` iteration, TT-Symbiote iterates over `__dict__.values()` -- any attribute that is a `TTNNModule` instance is treated as a child. This includes items in lists, dicts, and tuples, which are discovered through `named_children()` for traversal but not through the default `__dict__.values()` loop.

The `TTNNLayerStack` container explicitly overrides all three phases to propagate to its `layers` list:

```python
class TTNNLayerStack(TTNNModule):
    def preprocess_weights_impl(self):
        for layer in self.layers:
            if isinstance(layer, TTNNModule):
                layer.preprocess_weights()

    def move_weights_to_device_impl(self):
        for layer in self.layers:
            if isinstance(layer, TTNNModule):
                layer.move_weights_to_device()
```

## 7. Comparative Assessment

### Structural Differences

| Dimension | TT-DiT | TT-Symbiote |
|---|---|---|
| **Construction** | Module tree built independently, state dict loaded after | Module tree built *from* PyTorch modules |
| **Weight references** | `Parameter` objects declared in `__init__` | Raw `torch.Tensor` attributes on `self` |
| **Transformation hook** | `_prepare_torch_state` (dict surgery) | `preprocess_weights_impl` (per-module conversion) |
| **Distribution** | Declarative: `mesh_axes=[None, 0]` on `Parameter` | Imperative: explicit `shard_tensor_to_mesh_mapper` calls |
| **Lifecycle** | Single pass: load = transform + convert + place | Three explicit phases: construct, preprocess, place |
| **Caching** | Built-in `.tensorbin` serialization | None -- must re-convert from PyTorch on every run |
| **Fallback** | No PyTorch fallback | `_fallback_torch_layer` enables PyTorch execution |
| **Validation** | Strict: dtype, layout, shape, device, memory config | Minimal: relies on TTNN runtime errors |

### Advantages of the TT-DiT Approach

1. **Single-phase simplicity**: One call does everything. No risk of calling phases out of order or forgetting a phase.

2. **State dict independence**: The module tree is constructed without a PyTorch model present. Only a `state_dict` (a plain dictionary) is needed. This decouples the TT module definition from the PyTorch module structure.

3. **Caching**: The `.tensorbin` cache provides 10-100x faster loading on subsequent runs, which is critical for production latency.

4. **Declarative distribution**: `mesh_axes` on `Parameter` is self-documenting and validated at construction time, before any weights are loaded.

5. **Strict validation**: Every load checks shape, dtype, layout, memory config, and device, catching mismatches immediately.

### Advantages of the TT-Symbiote Approach

1. **Phase separation for device flexibility**: The three-phase model allows preprocessing to happen once while device placement can be repeated (e.g., after a device reset or when swapping between devices).

2. **PyTorch fallback**: The `_fallback_torch_layer` reference enables transparent fallback, simplifying incremental porting.

3. **Direct PyTorch access**: Accessing `self.torch_layer.weight` directly avoids the need for state dict manipulation utilities like `pop_substate`.

4. **Trace guards**: Built-in assertions prevent weight preprocessing during trace execution, a runtime safety net.

### What Can Be Reused

When porting from TT-Symbiote to TT-DiT style or vice versa:

1. **TTNN preprocessing utilities**: `preprocess_linear_weight` and `preprocess_linear_bias` from `ttnn.model_preprocessing` are framework-agnostic. TT-DiT handles transposition in `_prepare_torch_state` and calls `ttnn.from_torch` directly, but the utility functions perform equivalent operations.

2. **Weight transformation logic**: The actual tensor surgery (transposition, reshaping, padding, merging) is identical regardless of framework. A `_prepare_torch_state` that fuses QKV projections does the same math as a `preprocess_weights_impl` that fuses QKV projections.

3. **Mesh mapping strategies**: The sharding decisions (which dimension to shard, which mesh axis to use) are model-architecture decisions, not framework decisions. A column-parallel linear shards the output dimension in both frameworks.

4. **Deallocation patterns**: Both frameworks support recursive deallocation. TT-DiT's `deallocate_weights` and TT-Symbiote's `deallocate_weights_impl` are structurally equivalent.

### What Must Be Rewritten

1. **Module construction**: TT-DiT modules declare `Parameter` objects in `__init__` with full metadata. TT-Symbiote modules use `from_torch` factories. These patterns are incompatible -- the construction code must be rewritten.

2. **State dict routing**: TT-DiT's recursive `_load_torch_state_dict_inner` with `pop_substate` has no equivalent in TT-Symbiote. Porting to TT-DiT requires implementing `_prepare_torch_state` hooks instead of `from_torch` + `preprocess_weights_impl`.

3. **Serialization**: TT-Symbiote modules have no cache. To gain TT-DiT-style caching, modules must adopt the `Parameter`-based approach with `save`/`load` methods.

4. **Distribution metadata**: TT-DiT's `mesh_axes` declarations must be written from scratch. There is no mechanical translation from TT-Symbiote's imperative `shard_tensor_to_mesh_mapper` calls.

---

## Key Takeaways

1. **TT-Symbiote's three-phase lifecycle (from_torch, preprocess, move_to_device) provides flexibility at the cost of complexity**: Callers must invoke each phase in order, and the framework provides no compile-time guarantee that phases are called correctly -- only runtime assertions.

2. **TT-DiT's declarative `mesh_axes` is superior to imperative mesh mappers for maintainability**: The distribution strategy is visible at `Parameter` declaration time, validated at construction, and does not require the module to know about mesh mapper APIs.

3. **The lack of serialization in TT-Symbiote is a significant production gap**: Every run re-converts from PyTorch, paying the full cost of weight transformation, TTNN conversion, and device transfer. Adding TT-DiT-style `.tensorbin` caching to a TT-Symbiote model would require adopting TT-DiT's `Parameter` abstraction or building an equivalent.

4. **The weight transformation logic (transpositions, QKV fusion, padding, SwiGLU permutation) is framework-independent**: When porting a model, the mathematical operations on weight tensors transfer directly -- only the surrounding framework glue changes.

5. **TT-Symbiote's `from_torch` pattern couples module construction to PyTorch**: This makes it easier to port PyTorch models initially but harder to decouple from PyTorch in production. TT-DiT's state-dict-only approach enables loading from any source that can produce a `dict[str, torch.Tensor]`.

---

**Next:** [Chapter 7 -- Tracing and Performance](../ch7_tracing_and_performance/index.md)
