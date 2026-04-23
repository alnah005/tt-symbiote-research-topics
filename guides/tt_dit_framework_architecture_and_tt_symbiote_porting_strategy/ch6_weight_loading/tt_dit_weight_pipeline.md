# TT-DiT Weight Loading Pipeline

## Prerequisites

- Familiarity with TT-DiT's `Module` and `Parameter` classes ([Chapter 1](../ch1_introduction/index.md)).
- Understanding of TTNN tensor layouts (`TILE_LAYOUT`, `ROW_MAJOR_LAYOUT`) and data types (`bfloat16`, `float32`).
- Basic knowledge of `ttnn.MeshDevice` and multi-device mesh topologies.

---

## 1. End-to-End Flow

The TT-DiT weight loading pipeline converts a HuggingFace PyTorch `state_dict` into device-resident TTNN tensors through a single recursive call:

```
HF state_dict (dict[str, torch.Tensor])
    |
    v
Module.load_torch_state_dict(state_dict)
    |
    v
Module._load_torch_state_dict_inner(state_dict, ...)
    |
    +---> Module._prepare_torch_state(state_dict)   [in-place mutation]
    |
    +---> for each child: child._load_torch_state_dict_inner(child_state)
    |
    +---> for each parameter: Parameter.load_torch_tensor(tensor)
              |
              v
          tensor.from_torch(torch_tensor, device, layout, dtype, mesh_axes, ...)
              |
              v
          ttnn.from_torch(...)  -->  device-resident ttnn.Tensor
```

The entire flow is **single-phase**: one call loads, transforms, distributes, and places all weights. There is no separate "preprocess" or "move to device" step.

## 2. The Entry Point: `load_torch_state_dict`

```python
def load_torch_state_dict(
    self, state_dict: Mapping[str, torch.Tensor],
    *, strict: bool = True, on_host: bool = False
) -> IncompatibleKeys:
```

**Key parameters:**

- **`strict`**: When `True` (the default), raises `ValueError` if any keys in the `state_dict` are missing from the module tree or unexpected. This catches shape mismatches and renamed parameters early.
- **`on_host`**: When `True`, tensors remain in host memory rather than being placed on device. This is used during cache creation -- replicated device tensors would otherwise cause redundant copies when serialized.

**Return value:** An `IncompatibleKeys` named tuple containing `missing_keys` and `unexpected_keys` lists. In non-strict mode, callers can inspect these to handle partial loading.

The method delegates to `_load_torch_state_dict_inner`, which performs the actual recursive descent.

## 3. Recursive Descent: `_load_torch_state_dict_inner`

The inner method performs three operations in order:

### 3.1. State Preparation via `_prepare_torch_state`

```python
state_dict = dict(state_dict)  # shallow copy to allow mutation
self._prepare_torch_state(state_dict)
```

The `_prepare_torch_state` hook is the primary customization point. It receives the local slice of the state dict (only keys relevant to this module and its descendants) and may:

- **Transpose weights** (e.g., Linear transposes `[out, in]` to `[in, out]`)
- **Reshape tensors** (e.g., bias from `[out]` to `[1, out]`)
- **Merge separate keys** (e.g., Attention merges `to_q`, `to_k`, `to_v` into a fused `to_qkv`)
- **Pad tensors** for head-count alignment
- **Rename keys** to match the TT-DiT module structure
- **Permute for fused activations** (e.g., SwiGLU gate reordering in `ColParallelLinear`)

This hook modifies the dict **in place**. The base `Module._prepare_torch_state` is a no-op.

### 3.2. Child Module Recursion

```python
for name, child in self.named_children():
    child_state = pop_substate(state_dict, name)
    child._load_torch_state_dict_inner(child_state, ...)
```

The `pop_substate` utility extracts all keys with a given prefix from the state dict and removes them, ensuring that after all children have been processed, only leaf parameter keys (or unexpected keys) remain:

```python
def pop_substate(state, key):
    prefix = f"{key}."
    return {k.removeprefix(prefix): state.pop(k) for k in list(state) if k.startswith(prefix)}
```

### 3.3. Parameter Loading

```python
for name, parameter in self.named_parameters():
    if name in state_dict:
        parameter.load_torch_tensor(state_dict.pop(name), on_host=on_host)
    else:
        missing_keys.append(f"{module_key_prefix}{name}")
```

Any remaining keys in `state_dict` after both passes are reported as `unexpected_keys`.

## 4. `Parameter.load_torch_tensor` -- From PyTorch to Device

The `Parameter` class encapsulates all the metadata needed to convert a PyTorch tensor into a device-resident TTNN tensor:

```python
class Parameter:
    def __init__(self, *,
        total_shape,      # Global shape across all mesh devices
        device,           # ttnn.MeshDevice
        layout,           # ttnn.Layout (default: TILE)
        dtype,            # ttnn.DataType (default: bfloat16)
        memory_config,    # ttnn.MemoryConfig (default: DRAM)
        pad_value,        # Optional padding value
        mesh_axes,        # Distribution spec: [None, 0, 1] etc.
        on_host,          # Keep on host if True
    ): ...
```

When `load_torch_tensor` is called:

1. **Shape validation**: The incoming PyTorch tensor's shape is compared against `total_shape`. A mismatch raises `LoadingError`.

2. **Conversion**: Delegates to `tensor.from_torch()` in `utils/tensor.py`:

```python
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
```

3. **Validation via `_set_data`**: The resulting TTNN tensor is checked for dtype, layout, memory config, device, and local shape consistency before being stored.

## 5. Mesh Distribution via `mesh_axes`

The `mesh_axes` parameter is TT-DiT's declarative approach to multi-device tensor distribution. It maps tensor dimensions to mesh device axes.

### Semantics

For a tensor with shape `[D0, D1, D2]` and `mesh_axes = [None, 0, 1]`:

- Dimension 0 is **replicated** across all devices.
- Dimension 1 is **sharded** across mesh axis 0.
- Dimension 2 is **sharded** across mesh axis 1.

The local shape on each device becomes:

$$\text{local\_shape}[i] = \begin{cases} \text{total\_shape}[i] & \text{if } \text{mesh\_axes}[i] = \text{None} \\ \text{total\_shape}[i] \;/\; \text{device.shape}[\text{mesh\_axes}[i]] & \text{otherwise} \end{cases}$$

### Internal Implementation

The `from_torch` utility in `utils/tensor.py` converts `mesh_axes` into TTNN placement objects:

```python
placements = _invert_placements(mesh_axes, output_rank=mesh_rank)
placements = [
    ttnn.PlacementShard(p) if p is not None
    else ttnn.PlacementReplicate()
    for p in placements
]
mesh_mapper = ttnn.create_mesh_mapper(device, ttnn.MeshMapperConfig(placements))
```

This inverts the tensor-dimension-to-mesh-axis mapping into a mesh-axis-to-tensor-dimension mapping that TTNN expects.

### Examples from the Codebase

**Replicated weight** (`Linear`):
```python
self.weight = Parameter(
    total_shape=[in_features, out_features],
    device=mesh_device, dtype=dtype
)
# mesh_axes defaults to [None, None] -- replicated on all devices
```

**Column-parallel weight** (`ColParallelLinear`):
```python
self.weight = Parameter(
    total_shape=[in_features, out_features],
    mesh_axes=[fsdp_mesh_axis, mesh_axis],  # e.g., [None, 0]
    device=mesh_device, dtype=dtype,
)
```
Here, the output dimension is sharded across tensor-parallel devices while the input dimension may be sharded across FSDP devices.

**Row-parallel weight** (`RowParallelLinear`):
```python
self.weight = Parameter(
    total_shape=[in_features, out_features],
    mesh_axes=[mesh_axis, fsdp_mesh_axis],  # e.g., [0, None]
    device=mesh_device, dtype=dtype,
)
```
The input dimension is sharded across tensor-parallel devices.

## 6. The `_prepare_torch_state` Pattern in Practice

### Linear Layer

The simplest transformation -- transpose weight from PyTorch's `[out, in]` to TT-DiT's `[in, out]` convention, and reshape bias:

```python
class Linear(Module):
    def _prepare_torch_state(self, state):
        if "weight" in state:
            state["weight"] = state["weight"].transpose(0, 1)
        if "bias" in state:
            state["bias"] = state["bias"].reshape(1, -1)
```

### ColParallelLinear with SwiGLU

A more complex case -- when `activation_fn == "swiglu"`, the output features are doubled and the gate/value projections must be interleaved per device for correct column-fractured execution:

```python
def _prepare_torch_state(self, state):
    weight = state.pop("weight", None)
    bias = state.pop("bias", None)

    def permute_for_swiglu(tensor):
        ndev = self._mesh_axis_size
        tensor = tensor.reshape(-1, 2, ndev, tensor.shape[-1] // 2 // ndev)
        tensor = tensor.permute(0, 2, 1, 3)
        tensor = tensor.reshape(-1, self.out_features)
        return tensor

    if weight is not None:
        weight = weight.transpose(0, 1)
        if self.activation_fn == "swiglu":
            weight = permute_for_swiglu(weight)
        state["weight"] = weight
```

### Attention Block -- QKV Fusion

The most complex example. The HuggingFace state dict has separate `to_q`, `to_k`, `to_v` projections. TT-DiT fuses them into a single `to_qkv` for efficient batched matmul:

```python
class Attention(Module):
    def _prepare_torch_state(self, state):
        weight, bias = self._reshape_and_merge_qkv(
            pop_substate(state, "to_q"),
            pop_substate(state, "to_k"),
            pop_substate(state, "to_v"),
        )
        if weight is not None:
            state["to_qkv.weight"] = weight
```

The merge operation reshapes Q, K, V per tensor-parallel device and head count, pads for alignment, and interleaves them so that column-fracturing naturally distributes heads across devices:

$$W_{\text{QKV}} = \text{cat}(W_Q^{(\text{dev}_0)}, W_K^{(\text{dev}_0)}, W_V^{(\text{dev}_0)}, W_Q^{(\text{dev}_1)}, \ldots)$$

## 7. Serialization: The `.tensorbin` Cache

TT-DiT provides a two-tier loading strategy managed by `utils/cache.py`:

### Cache Hierarchy

```
load_model(tt_model, ...)
    |
    +---> Is cache dir set?  (TT_DIT_CACHE_DIR env var)
    |         |
    |         +---> Cache exists on disk?
    |         |         |
    |         |         +--- YES: tt_model.load(cache_dir)   [fast path]
    |         |         |
    |         |         +--- NO:  tt_model.load_torch_state_dict(state_dict, on_host=True)
    |         |                   tt_model.save(cache_dir)   [create cache]
    |         |                   tt_model.load(cache_dir)   [move to device]
    |         |
    |         +---> Cache dir not set:
    |                   tt_model.load_torch_state_dict(state_dict)  [direct load]
```

### The `save` / `load` Methods

**Saving:** Each `Parameter` is serialized individually via `ttnn.dump_tensor`:

```python
def save(self, directory, /, *, prefix=""):
    for name, child in self.named_children():
        child.save(directory, prefix=f"{prefix}{name}.")
    for name, parameter in self.named_parameters():
        parameter.save(directory / f"{prefix}{name}.tensorbin")
```

The file naming convention produces flat files like:
```
to_qkv.weight.tensorbin
to_qkv.bias.tensorbin
norm_q.weight.tensorbin
```

**Loading:** The reverse operation reads `.tensorbin` files and places them on device:

```python
def load(self, directory, /, *, prefix=""):
    for name, parameter in self.named_parameters():
        path = directory / f"{prefix}{name}.tensorbin"
        parameter.load(path)  # calls ttnn.load_tensor(path, device=self.device)
```

### Cache Key Structure

Cache directories encode the parallelism configuration and mesh shape to avoid loading tensors with incompatible sharding:

```
{TT_DIT_CACHE_DIR}/{model_name}/{subfolder}/{parallel_key}mesh{mesh_shape}_{dtype}
```

For example:
```
/cache/flux1-dev/transformer/TP8_0_SP8_1_mesh8x1_bf16/
```

### The `on_host` Optimization

When creating the cache, `load_torch_state_dict` is called with `on_host=True`. This avoids placing replicated tensors on device where `ttnn.dump_tensor` would write one copy per device shard -- wasting disk space and I/O bandwidth. Instead, host tensors are serialized once, and the subsequent `tt_model.load(cache_dir)` handles device placement.

## 8. Weight Offloading

TT-DiT supports dynamic weight offloading through the `unload_set` mechanism on `Module`:

```python
def set_unload_set(self, *args: Module):
    self.unload_set = set(args)
```

When `load_model` is called, it first deallocates any modules in the target's `unload_set`:

```python
for module in tt_model.unload_set or []:
    module.deallocate_weights()
```

The `deallocate_weights` method recursively calls `ttnn.deallocate` on every `Parameter`, freeing device DRAM. This enables memory-constrained deployments where, for example, a VAE decoder and transformer cannot coexist on device simultaneously.

---

## Key Takeaways

1. **Single-phase design**: TT-DiT's `load_torch_state_dict` combines transformation, conversion, distribution, and device placement into one recursive pass. The `_prepare_torch_state` hook is the sole customization point for weight transformations.

2. **Declarative distribution**: The `mesh_axes` parameter on `Parameter` specifies tensor-to-mesh-axis mapping declaratively. The framework handles the translation to TTNN placement objects internally.

3. **The `.tensorbin` cache eliminates redundant work**: First-run loading converts from PyTorch and writes a cache. Subsequent runs load pre-converted TTNN tensors directly, skipping all PyTorch-side transformations.

4. **`_prepare_torch_state` enables arbitrary state dict surgery**: From simple transpositions (Linear) to complex multi-key fusion with per-device interleaving (Attention QKV merge), this hook pattern handles the full spectrum of HuggingFace-to-TT-DiT transformations without requiring changes to the loading infrastructure.

5. **Shape and metadata validation is strict**: `Parameter._set_data` checks dtype, layout, memory config, device identity, and local shape on every load. Mismatches fail fast with descriptive error messages.

---

**Next:** [`symbiote_weight_pipeline.md`](./symbiote_weight_pipeline.md)
