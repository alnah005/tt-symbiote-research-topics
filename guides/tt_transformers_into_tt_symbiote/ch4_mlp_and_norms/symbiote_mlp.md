# TT Symbiote MLP Modules

Source files:
- `models/experimental/tt_symbiote/modules/linear.py`
- `models/experimental/tt_symbiote/core/module.py`

---

## 1. Overview

TT Symbiote does not provide a ready-made three-weight SwiGLU MLP class. Instead it
provides a library of `TTNNModule`-derived linear layer classes that a user composes to
build an MLP. The classes relevant to transformer MLP layers are:

| Class | Sharding pattern | Activation dtype | Device restriction |
|-------|-----------------|------------------|--------------------|
| `TTNNLinear` | None — single device, interleaved DRAM | `bfloat16` | Any |
| `TTNNLinearLLama` | None — single device, interleaved DRAM | `bfloat8_b` | Any (trace-disabled) |
| `TTNNLinearLLamaBFloat16` | None — single device, interleaved DRAM | `bfloat16` | Any (trace-disabled) |
| `TTNNLinearIColShardedWRowSharded` | Input col-sharded; weight row-sharded; reduce-scatter output | `bfloat16` | T3K only |
| `TTNNLinearLLamaIColShardedWRowSharded` | Same as above | `bfloat8_b` | T3K only (trace-disabled) |
| `TTNNLinearIReplicatedWColSharded` | Input replicated; weight col-sharded | `bfloat16` | T3K only |

All classes inherit from `TTNNModule` (defined in `core/module.py`).

---

## 2. `TTNNModule` — weight lifecycle base class

`TTNNModule` (`core/module.py` lines 60–235) defines three lifecycle stages for weights:

### 2.1 Stage 1 — preprocessing (host-side)

```python
def preprocess_weights(self):
    """Preprocess weights (called once before first use)."""
    if not self._preprocessed_weight:
        self._preprocessed_weight = True
    else:
        return
    self.preprocess_weights_impl()
```
Source: `module.py` lines 79–85.

The default `preprocess_weights_impl` recurses into all `TTNNModule` children
(`module.py` lines 104–110). Leaf classes override it to transform the raw torch weight
tensor into a format ready for device placement (layout conversion, tiling, dtype cast,
mesh mapping).

### 2.2 Stage 2 — moving to device

```python
def move_weights_to_device(self):
    """Move preprocessed weights to device."""
    assert self._preprocessed_weight
    assert self.device is not None
    if not self._weights_on_device:
        self._weights_on_device = True
    else:
        return
    self.move_weights_to_device_impl()
```
Source: `module.py` lines 87–97.

### 2.3 Stage 3 — deallocation

```python
def deallocate_weights(self):
    self.deallocate_weights_impl()
    self._weights_on_device = False
```
Source: `module.py` lines 99–102.

The `@deallocate_weights_after` decorator (`module.py` lines 238–246) wraps a `forward`
method so that `deallocate_weights()` is automatically called after each forward pass.
This is used by `TTNNLinearLLama`, `TTNNLinearLLamaBFloat16`, and
`TTNNLinearLLamaIColShardedWRowSharded` — classes intended for single-pass or
low-memory-budget usage.

### 2.4 Device assignment

```python
def to_device(self, device):
    self._device = device
    return self
```
Source: `module.py` lines 127–130.

`device` must be set before `move_weights_to_device()` is called.

---

## 3. `TTNNLinear` — base single-device linear layer

```python
@trace_enabled
class TTNNLinear(TTNNModule):
    def __init__(self, in_features: int, out_features: int) -> None:
```
Source: `linear.py` lines 16–27.

The `@trace_enabled` decorator means this class participates in TTNN tracing.

### 3.1 Construction paths

| Method | Description |
|--------|-------------|
| `from_parameters(weight, bias=None)` | Constructs from raw weight tensor; calls `preprocess_weights()` immediately and then deletes `self.weight` and `self.bias`. |
| `from_torch(linear: nn.Linear)` | Stores `linear.weight` and `linear.bias` directly; sets `_fallback_torch_layer`; does NOT call `preprocess_weights()`. |

Source: `linear.py` lines 28–52.

### 3.2 `preprocess_weights_impl`

```python
def preprocess_weights_impl(self):
    self.tt_weight_host = preprocess_linear_weight(self.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    self.tt_bias_host = None
    if self.bias is not None:
        self.tt_bias_host = preprocess_linear_bias(self.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
```
Source: `linear.py` lines 58–63.

`preprocess_linear_weight` (from `ttnn.model_preprocessing`) transposes the weight to
match TTNN's convention and converts it to TILE layout at `bfloat16`. The result is a
host-side tensor stored as `self.tt_weight_host`.

### 3.3 `move_weights_to_device_impl`

```python
def move_weights_to_device_impl(self):
    self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
    self.tt_bias = ttnn.to_device(self.tt_bias_host, self.device) if self.tt_bias_host is not None else None
```
Source: `linear.py` lines 65–68.

No sharding is applied. The weight is placed in whatever default memory config
`ttnn.to_device` provides (interleaved DRAM for host-allocated tensors).

### 3.4 `forward`

```python
def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # Pad to 4D if needed
    input_tensor = ttnn.reshape(input_tensor, input_shape)
    tt_output = ttnn.linear(input_tensor, self.tt_weight, bias=self.tt_bias,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_output = ttnn.reshape(tt_output, input_tensor_shape[:-1] + [self.out_features])
    return tt_output
```
Source: `linear.py` lines 77–88. Output memory config is always `DRAM_MEMORY_CONFIG`.

---

## 4. `TTNNLinearLLama` — bfloat8 single-device linear with auto-dealloc

```python
@trace_disabled
class TTNNLinearLLama(TTNNLinear):
    def preprocess_weights_impl(self):
        self.tt_weight_host = preprocess_linear_weight(self.weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        # bias similarly at bfloat8_b
```
Source: `linear.py` lines 179–188.

- Weight dtype: `bfloat8_b` instead of `bfloat16`.
- `@trace_disabled`: cannot be used inside traced execution.
- `@deallocate_weights_after` on `forward`: weights are freed from device after each
  forward call, so the module must call `move_weights_to_device()` again before reuse.

---

## 5. `TTNNLinearLLamaBFloat16` — bfloat16 single-device linear with auto-dealloc

```python
@trace_disabled
class TTNNLinearLLamaBFloat16(TTNNLinear):
    @deallocate_weights_after
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        return super().forward(input_tensor)
```
Source: `linear.py` lines 279–286.

Uses base `TTNNLinear` preprocessing (bfloat16) but adds `@deallocate_weights_after`.

---

## 6. `TTNNLinearIColShardedWRowSharded` — multi-device tensor-parallel linear

```python
class TTNNLinearIColShardedWRowSharded(TTNNLinearInputShardedWeightSharded):
    def __init__(self, in_features, out_features) -> None:
        super().__init__(in_features, out_features, input_dim=-1, weight_dim=-2)
```
Source: `linear.py` lines 126–130.

This is the class most analogous to the `w1`/`w3` projection role in the TT Transformers
MLP on a multi-device (T3K) setup.

The naming convention:
- **I Col Sharded** — the input tensor is expected to already be column-sharded (sharded
  on its last dimension, `dim=-1`).
- **W Row Sharded** — the weight is sharded on its row axis (`dim=-2` of the weight
  matrix).

### 6.1 `preprocess_weights_impl` (inherited from `TTNNLinearInputShardedWeightSharded`)

```python
def preprocess_weights_impl(self):
    self.tt_bias_host = self.bias
    self.tt_weight_host = self.weight  # Stored as raw torch.Tensor; sharding deferred to move_weights_to_device_impl
```
Source: `linear.py` lines 103–105.

### 6.2 `move_weights_to_device_impl` (inherited from `TTNNLinearInputShardedWeightSharded`)

```python
def move_weights_to_device_impl(self):
    if isinstance(self.tt_weight_host, torch.Tensor):
        self.tt_weight_host = preprocess_linear_weight(
            self.tt_weight_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            weights_mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(self.device, dim=self.weight_dim),  # dim=-2
        )
    ...
    self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
```
Source: `linear.py` lines 107–123.

### 6.3 `forward` — reduce-scatter output

```python
@run_on_devices(DeviceArch.T3K)
def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    # Validate input sharding on dim=-1
    ...
    tt_output = ttnn.linear(input_tensor, self.tt_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_output = ttnn.experimental.reduce_scatter_minimal_async(
        tt_output,
        dim=3,
        multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_rs_semaphore_handles(1),
        barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
        num_links=1,
        cluster_axis=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Ring,
        ...
    )
    if self.tt_bias is not None:
        tt_output += self.tt_bias
    ...
    return tt_output
```
Source: `linear.py` lines 133–176.

The `@run_on_devices(DeviceArch.T3K)` guard raises `RuntimeError` at runtime if the
`MESH_DEVICE` environment variable does not resolve to `T3K`. The device architecture
is checked via `os.environ.get("MESH_DEVICE")`, not from the device object directly.
Source: `module.py` lines 277–321.

The reduce-scatter uses `cluster_axis=1` and `ttnn.Topology.Ring`. This is analogous
to the reduce-scatter applied to `w1_out`/`w3_out` on Galaxy in TT Transformers
(`cluster_axis=1`), not to the all-reduce used after the down projection (`w2`), which
uses `tt_all_reduce` on `cluster_axis=0`. In Symbiote, the reduce-scatter is placed
after the single matmul of this class, not after a gate+up pair.

### 6.4 Input sharding validation

The forward method asserts that the input tensor's topology placements satisfy one of:
- Single placement on `dim=-1` (column-sharded activations).
- Two placements: batch dim `0` and then dim `-1`.

Source: `linear.py` lines 135–149.

---

## 7. `TTNNLinearLLamaIColShardedWRowSharded` — bfloat8 version

```python
@trace_disabled
class TTNNLinearLLamaIColShardedWRowSharded(TTNNLinearIColShardedWRowSharded):
    def move_weights_to_device_impl(self):
        if isinstance(self.tt_weight_host, torch.Tensor):
            self.tt_weight_host = preprocess_linear_weight(
                self.tt_weight_host,
                dtype=ttnn.bfloat8_b,     # <-- key difference from base class
                layout=ttnn.TILE_LAYOUT,
                weights_mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(self.device, dim=self.weight_dim),
            )
        ...
    @deallocate_weights_after
    @run_on_devices(DeviceArch.T3K)
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        return super().forward(input_tensor)
```
Source: `linear.py` lines 197–222.

Differences from `TTNNLinearIColShardedWRowSharded`:
1. Weight dtype is `bfloat8_b` instead of `bfloat16`.
2. `@trace_disabled` — cannot be used inside traced execution.
3. `@deallocate_weights_after` — weights are freed after each forward pass.

---

## 8. `TTNNLinearIReplicatedWColSharded` — replicated input, col-sharded weight

```python
class TTNNLinearIReplicatedWColSharded(TTNNLinearInputReplicatedWeightSharded):
    def __init__(self, in_features, out_features) -> None:
        super().__init__(in_features, out_features, weight_dim=-1)
```
Source: `linear.py` lines 256–260.

- Weight sharded on `dim=-1` (column-parallel) at `bfloat16`.
- Input is replicated (no sharding constraint on input tensor).
- No reduce-scatter in `forward` — the output is also col-sharded per device.
- Restricted to T3K via `@run_on_devices(DeviceArch.T3K)`.

This class maps to the role of `w2` (down projection) in a scenario where activations
are not pre-sharded and the output partial sums are left for the caller to reduce.

---

## 9. Activation-wrapping helpers

`linear.py` also provides convenience wrappers that pair a linear with an activation:

| Class | Activation | Built on |
|-------|-----------|----------|
| `TTNNLinearGelu` | `ttnn.gelu` | `TTNNLinearActivation` |
| `TTNNLinearSilu` | `ttnn.silu` | `TTNNLinearActivation` |

These are separate sequential modules (linear then activation), not a fused SwiGLU.
Source: `linear.py` lines 302–352.

There is no Symbiote class that fuses the gating branch and the value branch into a
single `ttnn.mul` with `input_tensor_a_activations=[...]` as TT Transformers does.

---

## 10. Weight lifecycle summary for a Symbiote MLP user

A user building an MLP from Symbiote linear classes would follow this sequence for each
projection:

```python
# 1. Construct the module
proj = TTNNLinearLLamaIColShardedWRowSharded(in_features, out_features)

# 2. Attach the torch weight (from_torch or direct assignment)
proj = TTNNLinearLLamaIColShardedWRowSharded.from_torch(torch_linear)

# 3. Set the target device
proj.to_device(mesh_device)

# 4. Set device state (required for CCL operations inside forward)
proj.set_device_state(distributed_config)

# 5. Preprocess (layout conversion, tile conversion — host side)
proj.preprocess_weights()

# 6. Move to device (sharding and dtype conversion — device placement)
proj.move_weights_to_device()

# 7. Forward pass (if @deallocate_weights_after, weights are freed after this)
output = proj(input_tensor)
```

Steps 5 and 6 are separated to allow host-side preprocessing to happen in advance
(e.g., during model load) and device placement to happen just before inference.

---

## Navigation

- [Index](index.md)
- Previous: [TT Transformers MLP](tt_transformers_mlp.md)
- Next: [Integration Gaps](integration_gaps.md)
