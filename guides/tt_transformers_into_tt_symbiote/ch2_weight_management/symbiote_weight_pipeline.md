# TT Symbiote Weight Pipeline

TT Symbiote's weight pipeline separates preprocessing (CPU-only tensor conversion) from device placement, and uses class-level decorators to gate whether a module participates in TTNN trace capture.

## The `TTNNModule` weight lifecycle

All weight-bearing modules inherit from `TTNNModule` (`core/module.py`). The lifecycle has three distinct, explicitly guarded phases.

### Phase 1 — `preprocess_weights()`

Called once before any device is available. The base implementation sets `_preprocessed_weight = True` and recursively calls `preprocess_weights()` on child modules (so each child's own idempotency guard is respected). Concrete modules override `preprocess_weights_impl()` to convert raw `torch.Tensor` parameters into host-resident TTNN tensors.

The guard ensures the conversion runs at most once:

```python
def preprocess_weights(self):
    if not self._preprocessed_weight:
        self._preprocessed_weight = True
    else:
        return
    self.preprocess_weights_impl()
```

### Phase 2 — `move_weights_to_device()`

Called after `.to_device(device)` has been called on the module. The base implementation asserts that preprocessing has already run and that a device is set, and is idempotent: it sets `_weights_on_device = True` on the first call and returns immediately on any subsequent call. The per-module overrides (invoked via `move_weights_to_device_impl()`) call `ttnn.to_device()` on the host-resident tensors created in Phase 1.

### Phase 3 — `deallocate_weights()`

Frees device tensors via `ttnn.deallocate()` and resets `_weights_on_device = False`. The `deallocate_weights_after` decorator (also defined in `module.py`) wraps a `forward()` call so that deallocation happens automatically after every forward pass:

```python
def deallocate_weights_after(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self.deallocate_weights()
        return result
    return wrapper
```

---

## `preprocess_linear_weight` and `preprocess_linear_bias`

Both are imported from `ttnn.model_preprocessing`. Every linear module in Symbiote calls them inside `preprocess_weights_impl()`. The key parameters are `dtype` and `layout`:

```python
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight
import ttnn

self.tt_weight_host = preprocess_linear_weight(
    self.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
)
self.tt_bias_host = preprocess_linear_bias(
    self.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
)
```

The result is a host-resident TTNN tensor (the "host-staging" pattern). `tt_weight_host` lives on CPU memory until `move_weights_to_device_impl()` is called, at which point it is pushed to the device via `ttnn.to_device()`.

---

## Linear class hierarchy and dtype choices

### `TTNNLinear` — bfloat16, trace-enabled

```python
@trace_enabled
class TTNNLinear(TTNNModule):
    def preprocess_weights_impl(self):
        self.tt_weight_host = preprocess_linear_weight(
            self.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        ...
    def move_weights_to_device_impl(self):
        self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
        ...
```

`TTNNLinear` is the baseline: bfloat16 precision, no auto-deallocation. The `@trace_enabled` decorator registers the class in `_TRACE_ENABLED_CLASSES` so that `TracedRun` will capture its forward pass inside a TTNN trace.

### `TTNNLinearLLama` — bfloat8_b, trace-disabled, auto-deallocates

```python
@trace_disabled
class TTNNLinearLLama(TTNNLinear):
    def preprocess_weights_impl(self):
        self.tt_weight_host = preprocess_linear_weight(
            self.weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )
        ...
    @deallocate_weights_after
    def forward(self, input_tensor):
        return super().forward(input_tensor)
```

`@trace_disabled` places this class in `_TRACE_DISABLED_CLASSES`. `is_trace_enabled()` returns `False` for any instance even though the parent `TTNNLinear` is trace-enabled. The reason is the `@deallocate_weights_after` decorator on `forward`: auto-deallocation deallocates weight buffers after every forward pass and re-runs `move_weights_to_device()` on the next call. That pattern is incompatible with trace capture, because the trace records the exact buffer addresses, and those addresses are freed between calls.

### `TTNNLinearLLamaBFloat16` — bfloat16, trace-disabled, auto-deallocates

```python
@trace_disabled
class TTNNLinearLLamaBFloat16(TTNNLinear):
    @deallocate_weights_after
    def forward(self, input_tensor):
        return super().forward(input_tensor)
```

Same auto-deallocation behavior as `TTNNLinearLLama`, but uses the bfloat16 preprocessing inherited from `TTNNLinear`. Trace-disabled for the same reason.

### Comparison table

| Class | dtype | `@trace_enabled` | Auto-deallocates |
|---|---|---|---|
| `TTNNLinear` | `bfloat16` | Yes | No |
| `TTNNLinearLLama` | `bfloat8_b` | No (disabled) | Yes |
| `TTNNLinearLLamaBFloat16` | `bfloat16` | No (disabled) | Yes |

---

## Sharded linear variants

Symbiote provides two sharded families for multi-device operation, both constrained to `DeviceArch.T3K`.

### `TTNNLinearIColShardedWRowSharded`

Input is sharded on the column (last) dimension (`input_dim=-1`); weights are sharded on the row (second-to-last) dimension (`weight_dim=-2`). The class name encodes the convention: **I**nput **Col**umn-sharded, **W**eight **Row**-sharded.

The critical property of this class is that it **defers dtype conversion and mesh-sharding until `move_weights_to_device_impl()`** — not during `preprocess_weights_impl()`. In `preprocess_weights_impl()` (inherited from `TTNNLinearInputShardedWeightSharded`), the raw `torch.Tensor` is stored directly as `self.tt_weight_host`:

```python
def preprocess_weights_impl(self):
    self.tt_bias_host = self.bias
    self.tt_weight_host = self.weight  # raw torch.Tensor, no conversion yet
```

The conversion and mesh mapping happen inside `move_weights_to_device_impl()`, once `self.device` (the mesh device) is available:

```python
def move_weights_to_device_impl(self):
    if isinstance(self.tt_weight_host, torch.Tensor):
        self.tt_weight_host = preprocess_linear_weight(
            self.tt_weight_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            weights_mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(self.device, dim=self.weight_dim),
        )
    ...
    self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
```

The reason for deferral: `ttnn.shard_tensor_to_mesh_mapper` requires the mesh device object, which is not available until the module's `to_device()` method is called.

The `forward()` method is decorated with `@run_on_devices(DeviceArch.T3K)`, which checks the `MESH_DEVICE` environment variable and raises a `RuntimeError` if the current device is not T3K. After the `ttnn.linear` call it performs a `reduce_scatter_minimal_async` along `cluster_axis=1`.

### `TTNNLinearIReplicatedWColSharded`

Input is replicated across the mesh; weights are sharded on the last dimension (`weight_dim=-1`). The `forward()` does not include a reduce-scatter, making it suitable for column-parallel linear layers where the all-reduce is done outside.

```python
class TTNNLinearIReplicatedWColSharded(TTNNLinearInputReplicatedWeightSharded):
    def __init__(self, in_features, out_features) -> None:
        super().__init__(in_features, out_features, weight_dim=-1)
```

### `TTNNLinearLLamaIColShardedWRowSharded`

Combines `bfloat8_b` dtype with column/row sharding, plus `@deallocate_weights_after` and `@trace_disabled`. It inherits from `TTNNLinearIColShardedWRowSharded` (not from `TTNNLinearLLama`), overriding `move_weights_to_device_impl()` to call `preprocess_linear_weight` with `dtype=ttnn.bfloat8_b`.

---

## `@trace_enabled` and `@trace_disabled` in detail

Both decorators are defined in `core/run_config.py` and operate on module-registry sets:

```python
_TRACE_ENABLED_CLASSES: Set[Type] = set()
_TRACE_DISABLED_CLASSES: Set[Type] = set()

def trace_enabled(cls: Type) -> Type:
    _TRACE_ENABLED_CLASSES.add(cls)
    return cls

def trace_disabled(cls: Type) -> Type:
    _TRACE_DISABLED_CLASSES.add(cls)
    return cls

def is_trace_enabled(module) -> bool:
    return (
        isinstance(module, tuple(_TRACE_ENABLED_CLASSES))
        and not isinstance(module, tuple(_TRACE_DISABLED_CLASSES))
    )
```

`TracedRun` (also in `run_config.py`) consults `is_trace_enabled()` at call time. A module decorated with `@trace_disabled` is excluded even if an ancestor class is `@trace_enabled`.

The practical rule: any module that calls `deallocate_weights()` after `forward()` must be `@trace_disabled` — see `TTNNLinearLLama` above for the full rationale and the `SmartTTNN*` exception.

---

## `SmartTTNNLinear` (`linear_intelligent.py`)

`SmartTTNNLinear` extends `TTNNLinear` and dispatches to `prefill_forward` or `decode_forward` based on whether the input sequence length is `<= 32` (decode) or longer (prefill). In prefill mode it builds a `ttnn.MatmulMultiCoreReuseMultiCastProgramConfig` keyed by sequence length and caches it in `_prefill_pc_cache`. The module does not override `preprocess_weights_impl()`, so it inherits bfloat16 preprocessing from `TTNNLinear`.

`SmartTTNNLinearLLama` overrides `preprocess_weights_impl()` to use `bfloat8_b` and adds `@deallocate_weights_after` on `forward()`. `SmartTTNNLinearLLamaBFloat16` keeps bfloat16 but also adds `@deallocate_weights_after`.

> **Note:** `SmartTTNNLinear.__init__` calls the module-level `compute_kernel_config()` function at construction time. It also evaluates `self.device.compute_with_storage_grid_size()` for `grid_size`, but this is guarded by an `if self.device` check: because `self.device` is `None` at construction time (set later via `to_device()`), `self.grid_size` is safely initialized to `None` and is lazily populated in `forward()`. The constructor does not raise an error if called before `to_device()`. This is the only Symbiote linear class that queries the device for a compute grid.

---

**Next:** [transformers_weight_pipeline.md](transformers_weight_pipeline.md)
