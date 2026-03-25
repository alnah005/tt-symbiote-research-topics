# Chapter 5.1 — Linear Layers

All linear module classes documented here are defined in:

- `models/experimental/tt_symbiote/modules/linear.py`
- `models/experimental/tt_symbiote/modules/linear_intelligent.py`

---

## TTNNLinear

**Parent:** `TTNNModule`
**Decorator:** `@trace_enabled`
**Weight dtype/layout:** `ttnn.bfloat16` / `ttnn.TILE_LAYOUT`

`TTNNLinear` is the baseline TTNN-accelerated linear layer. It is the root of nearly the entire linear class hierarchy.

### Construction

```python
@classmethod
def from_torch(cls, linear: nn.Linear) -> "TTNNLinear":
    ...

@classmethod
def from_parameters(cls, weight, bias=None) -> "TTNNLinear":
    ...
```

`from_torch` stores a reference to the PyTorch `nn.Linear` as `_fallback_torch_layer`, copies `weight` and `bias` onto the new instance, and returns it without calling `preprocess_weights()`. `from_parameters` accepts raw weight tensors directly and calls `preprocess_weights()` immediately, then deletes the raw `weight` and `bias` attributes.

Constructor parameters: `in_features: int`, `out_features: int`.

### `preprocess_weights_impl`

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

### `move_weights_to_device_impl`

```python
def move_weights_to_device_impl(self):
    self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
    self.tt_bias = ttnn.to_device(self.tt_bias_host, self.device) \
        if self.tt_bias_host is not None else None
```

### `forward`

```python
def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
```

1. If the input is not in `TILE_LAYOUT`, converts it with `ttnn.to_layout(..., ttnn.DRAM_MEMORY_CONFIG)`.
2. Records the original shape, then zero-pads leading dimensions until the tensor is 4D (inserts size-1 dimensions at position 1).
3. Calls `ttnn.linear(input_tensor, self.tt_weight, bias=self.tt_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)`.
4. Reshapes the output back to the original leading dimensions with `out_features` as the last dimension.

---

## TTNNLinearLLama

**Parent:** `TTNNLinear`
**Decorator:** `@trace_disabled`
**Weight dtype/layout:** `ttnn.bfloat8_b` / `ttnn.TILE_LAYOUT`

Optimised for LLaMA models. Differs from `TTNNLinear` in two ways:

1. `preprocess_weights_impl` uses `dtype=ttnn.bfloat8_b` for both weight and bias.
2. `forward` is decorated with `@deallocate_weights_after`, which frees device weight tensors immediately after the forward pass completes.

```python
def preprocess_weights_impl(self):
    self.tt_weight_host = preprocess_linear_weight(
        self.weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
    )
    ...

@deallocate_weights_after
def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    return super().forward(input_tensor)
```

---

## TTNNLinearLLamaBFloat16

**Parent:** `TTNNLinear`
**Decorator:** `@trace_disabled`
**Weight dtype/layout:** `ttnn.bfloat16` / `ttnn.TILE_LAYOUT` (inherited from `TTNNLinear`)

The bfloat16 counterpart to `TTNNLinearLLama`. It does not override `preprocess_weights_impl`, so weights remain in `bfloat16`. It adds only the `@trace_disabled` class decorator and `@deallocate_weights_after` on `forward`.

```python
@deallocate_weights_after
def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    return super().forward(input_tensor)
```

---

## TTNNLinearInputShardedWeightSharded

**Parent:** `TTNNLinear`

This is the base class for the input-sharded / weight-sharded family. It introduces two additional constructor parameters:

| Parameter | Constraint | Meaning |
|---|---|---|
| `input_dim` | Must be `-1` | Shard input tensor on last dimension |
| `weight_dim` | Must be `-2` | Shard weight tensor on second-to-last dimension |

The assertion in `__init__` enforces these constraints hard.

`preprocess_weights_impl` defers work: it merely stores `self.bias` and `self.weight` as `tt_bias_host` and `tt_weight_host` without any conversion.

`move_weights_to_device_impl` performs the actual preprocessing and sharding:

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
    self.tt_bias = ttnn.to_device(self.tt_bias_host, self.device) \
        if self.tt_bias_host is not None else None
```

---

## TTNNLinearIColShardedWRowSharded

**Parent:** `TTNNLinearInputShardedWeightSharded`
**Device constraint:** T3K only (`@run_on_devices(DeviceArch.T3K)`)
**Weight dtype/layout:** `ttnn.bfloat16` / `ttnn.TILE_LAYOUT`

Convenience subclass that fixes `input_dim=-1`, `weight_dim=-2`.

The `forward` method performs topology-aware sharding validation before computing:

1. Asserts that the input tensor's shard placement matches `input_dim` (`-1`). Supports either a single sharding dimension or a batch+feature 2D sharding.
2. Converts to `TILE_LAYOUT` if needed.
3. Zero-pads to 4D.
4. Calls `ttnn.linear(input_tensor, self.tt_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)` — note: **no bias** is passed here.
5. Performs a collective `ttnn.experimental.reduce_scatter_minimal_async` on dimension 3 using `cluster_axis=1`, `ttnn.Topology.Ring`, `chunks_per_sync=10`, `num_workers_per_link=2`, `num_buffers_per_channel=2`. Semaphore handles are retrieved from `self.device_state.ccl_manager`.
6. Adds bias manually after the scatter, if present.
7. Reshapes the output back.

---

## TTNNLinearInputReplicatedWeightSharded

**Parent:** `TTNNLinear`

Base class for the replicated-input / sharded-weight family. Constructor parameter:

| Parameter | Constraint | Meaning |
|---|---|---|
| `weight_dim` | Must be `-1` | Shard weight on last dimension |

`preprocess_weights_impl` defers to raw tensor storage (same pattern as `TTNNLinearInputShardedWeightSharded`).

`move_weights_to_device_impl` preprocesses and shards both weight and bias along `weight_dim` (`-1`):

```python
weights_mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(self.device, dim=self.weight_dim)
```

---

## TTNNLinearIReplicatedWColSharded

**Parent:** `TTNNLinearInputReplicatedWeightSharded`
**Device constraint:** T3K only (`@run_on_devices(DeviceArch.T3K)`)
**Weight dtype/layout:** `ttnn.bfloat16` / `ttnn.TILE_LAYOUT`

Fixes `weight_dim=-1`. The `forward` method does not call `reduce_scatter_minimal_async`. Instead it:

1. Converts to `TILE_LAYOUT` if needed.
2. Zero-pads to 4D.
3. Calls `ttnn.linear(input_tensor, self.tt_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)`.
4. Adds bias manually if present.
5. Reshapes output.

---

## TTNNLinearLLamaIColShardedWRowSharded

**Parent:** `TTNNLinearIColShardedWRowSharded`
**Decorators:** `@trace_disabled`, `@deallocate_weights_after`, `@run_on_devices(DeviceArch.T3K)`
**Weight dtype/layout:** `ttnn.bfloat8_b` / `ttnn.TILE_LAYOUT`

Combines the col/row sharding strategy of `TTNNLinearIColShardedWRowSharded` with `bfloat8_b` weight precision.

Overrides `move_weights_to_device_impl` to use `dtype=ttnn.bfloat8_b` for both weight and bias mesh preprocessing. All other logic (sharding via `ttnn.shard_tensor_to_mesh_mapper`, dimension choices) is the same as the parent class.

---

## TTNNLinearActivation

**Parent:** `TTNNModule`
**Decorator:** `@trace_enabled`

A fused linear + activation module. It is not typically instantiated directly; use `TTNNLinearGelu` or `TTNNLinearSilu` instead.

```python
@classmethod
def from_parameters(cls, weight, linear_class, ttnn_act_fn, nn_act_fn, bias=None):
    ...

@classmethod
def from_torch(cls, linear: nn.Linear, linear_class, ttnn_act_fn, nn_act_fn):
    ...
```

`from_torch` stores a `PytorchLinearActivation(dense=linear, act_fn=nn_act_fn)` as the fallback layer, constructs `self.dense = linear_class.from_torch(linear)`, and stores `ttnn_act_fn` as `self.activation`.

`forward`:

```python
def forward(self, hidden_states):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.activation(hidden_states.to_ttnn)
    return hidden_states
```

---

## TTNNLinearGelu

**Parent:** — (not a `TTNNModule` subclass; a factory class only)

A factory that creates a `TTNNLinearActivation` pre-configured for GELU.

```python
@classmethod
def from_parameters(cls, weight, bias=None, linear_class=TTNNLinear):
    return TTNNLinearActivation.from_parameters(
        weight, linear_class, ttnn.gelu, nn.GELU(), bias
    )

@classmethod
def from_torch(cls, linear: nn.Linear, linear_class=TTNNLinear):
    return TTNNLinearActivation.from_torch(
        linear, linear_class, ttnn.gelu, nn.GELU()
    )
```

The default `linear_class` is `TTNNLinear` but any compatible class may be substituted.

---

## TTNNLinearSilu

**Parent:** — (factory class, same pattern as `TTNNLinearGelu`)

Factory that creates a `TTNNLinearActivation` pre-configured for SiLU.

```python
@classmethod
def from_parameters(cls, weight, bias=None, linear_class=TTNNLinear):
    return TTNNLinearActivation.from_parameters(
        weight, linear_class, ttnn.silu, nn.SiLU(), bias
    )

@classmethod
def from_torch(cls, linear: nn.Linear, linear_class=TTNNLinear):
    return TTNNLinearActivation.from_torch(
        linear, linear_class, ttnn.silu, nn.SiLU()
    )
```

---

## TTNNViTIntermediate

**Parent:** `TTNNLinearGelu`

ViT-specific subclass. `from_torch` accepts a `ViTIntermediate` object, asserts that its activation is `GELUActivation`, sets `_fallback_torch_layer = torch_vit_intermediate`, and builds `self.dense = TTNNLinear.from_torch(torch_vit_intermediate.dense)`. Does not override `from_parameters`.

---

## SmartTTNNLinear

Source: `models/experimental/tt_symbiote/modules/linear_intelligent.py`

**Parent:** `TTNNLinear`
**Weight dtype/layout:** `ttnn.bfloat16` / `ttnn.TILE_LAYOUT` (inherited)

`SmartTTNNLinear` dispatches to one of two code paths at runtime based on input sequence length.

### Constructor

```python
def __init__(self, in_features: int, out_features: int):
```

In addition to calling `super().__init__`, it initialises:

- `self.compute_kernel_config` — a `ttnn.WormholeComputeKernelConfig` (HiFi2, no math approx, no fp32 accumulator, L1 packer accumulation enabled).
- `self.grid_size` — queried from `self.device.compute_with_storage_grid_size()` if a device is already available.
- `self._prefill_pc_cache` — a `Dict[int, Optional[ttnn.MatmulMultiCoreReuseMultiCastProgramConfig]]` keyed by sequence length.

### `forward`

```python
def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    """Dispatch to prefill or decode path based on input sequence length."""
```

1. Converts to `TILE_LAYOUT` if needed.
2. Pads to 4D (handling 2D and 3D inputs).
3. Reads `seq_len = input_shape[-2]`.
4. Sets `mode = "decode"` if `seq_len <= 32`, else `mode = "prefill"`.
5. Calls `self.decode_forward(input_tensor)` or `self.prefill_forward(input_tensor, seq_len)`.
6. Reshapes output back.

### `decode_forward`

```python
def decode_forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    return ttnn.linear(
        input_tensor, self.tt_weight, bias=self.tt_bias,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
```

No program config is applied in decode mode.

### `prefill_forward`

```python
def prefill_forward(self, input_tensor: ttnn.Tensor, seq_len: int) -> ttnn.Tensor:
    return ttnn.linear(
        input_tensor, self.tt_weight, bias=self.tt_bias,
        compute_kernel_config=self.compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=self._get_prefill_pc(seq_len),
    )
```

A `ttnn.MatmulMultiCoreReuseMultiCastProgramConfig` is computed (and cached) per sequence length via `_get_prefill_pc`. For modules named `"lm_head"`, the program config is `None`. The config is built by `get_prefill_pc(M, K, N, grid_size)` using tile-aligned per-core M/N sizing.

---

## SmartTTNNLinearLLama

**Parent:** `SmartTTNNLinear`
**Weight dtype/layout:** `ttnn.bfloat8_b` / `ttnn.TILE_LAYOUT`

Overrides `preprocess_weights_impl` to use `dtype=ttnn.bfloat8_b`. Decorates `forward` with `@deallocate_weights_after` and delegates to `super().forward(input_tensor)`.

---

## SmartTTNNLinearLLamaBFloat16

**Parent:** `SmartTTNNLinear`
**Weight dtype/layout:** `ttnn.bfloat16` / `ttnn.TILE_LAYOUT` (inherited)

Does not override `preprocess_weights_impl`. Decorates `forward` with `@deallocate_weights_after` only.

---

## Summary Table

| Class | Parent | Weight dtype | `@trace_*` | `@deallocate_weights_after` | Multi-device |
|---|---|---|---|---|---|
| `TTNNLinear` | `TTNNModule` | `bfloat16` | `@trace_enabled` | No | No |
| `TTNNLinearLLama` | `TTNNLinear` | `bfloat8_b` | `@trace_disabled` | Yes | No |
| `TTNNLinearLLamaBFloat16` | `TTNNLinear` | `bfloat16` | `@trace_disabled` | Yes | No |
| `TTNNLinearInputShardedWeightSharded` | `TTNNLinear` | `bfloat16` | — | No | Yes (base) |
| `TTNNLinearIColShardedWRowSharded` | `TTNNLinearInputShardedWeightSharded` | `bfloat16` | — | No | T3K |
| `TTNNLinearInputReplicatedWeightSharded` | `TTNNLinear` | `bfloat16` | — | No | Yes (base) |
| `TTNNLinearIReplicatedWColSharded` | `TTNNLinearInputReplicatedWeightSharded` | `bfloat16` | — | No | T3K |
| `TTNNLinearLLamaIColShardedWRowSharded` | `TTNNLinearIColShardedWRowSharded` | `bfloat8_b` | `@trace_disabled` | Yes | T3K |
| `TTNNLinearActivation` | `TTNNModule` | inner class | `@trace_enabled` | No | No |
| `TTNNLinearGelu` | — (factory) | inner class | — | No | No |
| `TTNNLinearSilu` | — (factory) | inner class | — | No | No |
| `TTNNViTIntermediate` | `TTNNLinearGelu` | `bfloat16` | — | No | No |
| `SmartTTNNLinear` | `TTNNLinear` | `bfloat16` | — | No | No |
| `SmartTTNNLinearLLama` | `SmartTTNNLinear` | `bfloat8_b` | — | Yes | No |
| `SmartTTNNLinearLLamaBFloat16` | `SmartTTNNLinear` | `bfloat16` | — | Yes | No |

---

**Navigation:** [Chapter 5 Index](./index.md) | Linear Layers | [Normalization and Activation](./normalization_and_activation.md) | [Attention and Conv](./attention_and_conv.md)
