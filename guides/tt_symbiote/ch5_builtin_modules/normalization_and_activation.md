# Chapter 5.2 — Normalization and Activation Modules

All classes documented here are defined in:

- `models/experimental/tt_symbiote/modules/normalization.py` — norm layers
- `models/experimental/tt_symbiote/modules/activation.py` — point-wise activation layers

---

## Normalization Modules

### TTNNLayerNorm

**Parent:** `TTNNModule`
**Source:** `normalization.py`

TTNN-accelerated replacement for `nn.LayerNorm`.

#### `from_torch`

```python
@classmethod
def from_torch(cls, layer_norm: nn.LayerNorm) -> TTNNLayerNorm:
```

If `layer_norm.weight` is `None`, the method prints a warning and returns the original PyTorch `layer_norm` unchanged. Otherwise it creates a new instance, stores `layer_norm` as `_fallback_torch_layer`, and returns it.

#### `preprocess_weights_impl`

```python
def preprocess_weights_impl(self):
    if self.torch_layer is None:
        self._fallback_torch_layer = nn.LayerNorm(normalized_shape=1)
    self.tt_weight = ttnn.from_torch(
        self.torch_layer.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    self.tt_bias = ttnn.from_torch(
        self.torch_layer.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
```

#### `move_weights_to_device_impl`

```python
def move_weights_to_device_impl(self):
    self.tt_weight = ttnn.to_device(self.tt_weight, self.device)
    if self.tt_bias is not None:
        self.tt_bias = ttnn.to_device(self.tt_bias, self.device)
```

#### `forward`

```python
def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
```

1. Converts input to `TILE_LAYOUT` (with `ttnn.DRAM_MEMORY_CONFIG`) if not already tiled.
2. Calls `ttnn.layer_norm(input_tensor, weight=self.tt_weight, bias=self.tt_bias)`.
3. Returns the result directly.

---

### TTNNRMSNorm

**Parent:** `TTNNModule`
**Source:** `normalization.py`

TTNN-accelerated Root Mean Square normalization. The PyTorch counterpart expected by `from_torch` is `DeepseekV2RMSNorm` (also defined in `normalization.py`), which is equivalent to T5 LayerNorm.

#### `from_torch`

```python
@classmethod
def from_torch(cls, rms_norm: DeepseekV2RMSNorm) -> TTNNRMSNorm:
```

Returns the original layer if `rms_norm.weight` is `None` (with a warning). Otherwise stores it as `_fallback_torch_layer`.

#### `preprocess_weights_impl`

```python
def preprocess_weights_impl(self):
    if self.torch_layer is None:
        self._fallback_torch_layer = DeepseekV2RMSNorm(hidden_size=1)
    self.tt_weight = ttnn.from_torch(
        self.torch_layer.weight.unsqueeze(0).expand(32, -1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
```

No bias is stored; `DeepseekV2RMSNorm` has no bias parameter.

#### `move_weights_to_device_impl`

```python
def move_weights_to_device_impl(self):
    self.tt_weight = ttnn.to_device(self.tt_weight, self.device)
```

#### `forward`

```python
def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
```

1. Converts to `TILE_LAYOUT` with `ttnn.DRAM_MEMORY_CONFIG` if needed.
2. Calls `ttnn.rms_norm(x, weight=self.tt_weight, epsilon=self.torch_layer.variance_epsilon)`.
3. Returns the result.

---

### TTNNDistributedRMSNorm

**Parent:** `TTNNModule`
**Decorator:** `@trace_enabled`
**Device constraint:** T3K only (`@run_on_devices(DeviceArch.T3K)`)
**Source:** `normalization.py`

A two-phase distributed RMSNorm that reduces statistics across devices using an async all-gather collective. Designed for multi-device inference where hidden states are sharded across a mesh.

#### `from_torch`

Same guard pattern as `TTNNRMSNorm`: returns the unmodified layer if no weight is present, otherwise stores as `_fallback_torch_layer`. Note: the parameter type annotation in source is `"RMSNorm"` (a forward reference), not `DeepseekV2RMSNorm`.

#### `move_weights_to_device_impl`

Does not use `preprocess_weights_impl`. Directly reshapes and places the weight on the device mesh:

```python
def move_weights_to_device_impl(self):
    dim = self.torch_layer.weight.shape[0]
    self.weight_distributed = ttnn.as_tensor(
        self.torch_layer.weight
            .unsqueeze(0).view(1, 1, dim)
            .reshape([1, 1, dim // 32, 32]),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            self.device, dims=(None, 2), mesh_shape=list(self.device.shape)
        ),
    )
    self.weight_distributed = ttnn.to_device(self.weight_distributed, self.device)
```

#### `forward`

```python
@run_on_devices(DeviceArch.T3K)
def forward(self, inp):
```

1. If `inp` is 3D, unsqueezes to 4D.
2. Calls `ttnn.rms_norm_pre_all_gather(inp, dtype=ttnn.bfloat16)` to compute per-device partial statistics (`tt_stats`).
3. All-gathers `tt_stats` asynchronously using `ttnn.experimental.all_gather_async` with `topology=ttnn.Topology.Linear`.
4. Calls `ttnn.rms_norm_post_all_gather(inp, tt_stats, epsilon=..., weight=self.weight_distributed)` to produce the final normalised output.
5. Deallocates `tt_stats` immediately.

Semaphore handles for both the all-gather and the barrier are obtained from `self.device_state.ccl_manager`.

---

## Activation Modules

All three activation modules follow a common pattern:

- Parent class: `TTNNModule`
- Constructor sets `self._fallback_torch_layer` to the corresponding `torch.nn` activation.
- No weights, no `preprocess_weights_impl` or `move_weights_to_device_impl` overrides.
- `forward` converts input to `TILE_LAYOUT` if needed, then calls the corresponding `ttnn.*` op.

### TTNNSilu

**TTNN op:** `ttnn.silu`

```python
def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        input_tensor = ttnn.to_layout(
            input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    return ttnn.silu(input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

Fallback: `torch.nn.SiLU()`.

---

### TTNNReLU

**TTNN op:** `ttnn.relu`

```python
def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        input_tensor = ttnn.to_layout(
            input_tensor, ttnn.TILE_LAYOUT,
            memory_config=input_tensor.memory_config()
        )
    return ttnn.relu(input_tensor, memory_config=input_tensor.memory_config())
```

Unlike `TTNNSilu` and `TTNNGelu`, `TTNNReLU` preserves the input tensor's original memory config rather than defaulting to `DRAM_MEMORY_CONFIG`.

Fallback: `torch.nn.ReLU()`.

---

### TTNNGelu

**TTNN op:** `ttnn.gelu`

```python
def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        input_tensor = ttnn.to_layout(
            input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    return ttnn.gelu(input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

Fallback: `torch.nn.GELU()`.

---

## Summary Table

| Class | Parent | Has weights | TTNN op | Memory config |
|---|---|---|---|---|
| `TTNNLayerNorm` | `TTNNModule` | Yes (weight + bias, `bfloat16`) | `ttnn.layer_norm` | `DRAM` |
| `TTNNRMSNorm` | `TTNNModule` | Yes (weight, `bfloat16`, expanded 32x) | `ttnn.rms_norm` | `DRAM` |
| `TTNNDistributedRMSNorm` | `TTNNModule` | Yes (weight, `ROW_MAJOR`, mesh-sharded) | `rms_norm_pre_all_gather` + `rms_norm_post_all_gather` | — |
| `TTNNSilu` | `TTNNModule` | No | `ttnn.silu` | `DRAM` |
| `TTNNReLU` | `TTNNModule` | No | `ttnn.relu` | Preserves input config |
| `TTNNGelu` | `TTNNModule` | No | `ttnn.gelu` | `DRAM` |

---

**Next:** [`attention_and_conv.md`](./attention_and_conv.md)
