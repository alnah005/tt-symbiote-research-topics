# Chapter 6 — Implementation Guide: Authoring a New TTNN Module

This guide walks through every step required to produce a working `TTNNModule` subclass,
using `TTNNLinear` from `models/experimental/tt_symbiote/modules/linear.py` as a
concrete reference throughout.

---

## Step 1 — Inherit from `TTNNModule`

```python
from models.experimental.tt_symbiote.core.module import TTNNModule

class TTNNMyLinear(TTNNModule):
    ...
```

`TTNNModule` is defined in
`models/experimental/tt_symbiote/core/module.py`.
It is **not** a `torch.nn.Module` subclass; do not mix the two hierarchies.

### What `TTNNModule.__init__` initialises

Calling `super().__init__()` sets up the following instance attributes that the
lifecycle methods depend on:

| Attribute | Initial value | Purpose |
|---|---|---|
| `_device` | `None` | Set later by `to_device()` |
| `_preprocessed_weight` | `False` | Guards `preprocess_weights()` from running twice |
| `_weights_on_device` | `False` | Guards `move_weights_to_device()` from running twice |
| `_fallback_torch_layer` | `None` | Holds the original `nn.Module` for fallback run modes |
| `_unique_name` | `None` | Set by the replacement machinery; auto-generated if absent |
| `_device_state` | `None` | Distributed configuration, set via `set_device_state()` |
| `_model_config` | `{}` | Arbitrary config dict, set via `set_model_config()` |

Your `__init__` must call `super().__init__()` before storing any attributes, because the
lifecycle machinery reads those sentinel flags.

```python
def __init__(self, in_features: int, out_features: int) -> None:
    super().__init__()           # must come first
    self.in_features = in_features
    self.out_features = out_features
```

---

## Step 2 — Implement `from_torch`

`from_torch` is the factory that the replacement machinery calls. Its signature in
`TTNNModule` is:

```python
@classmethod
def from_torch(cls, torch_layer, *args, **kwargs):
    new_layer = cls(*args, **kwargs)
    new_layer._fallback_torch_layer = torch_layer
    return new_layer
```

You must override it to:

1. Construct your module (`cls(...)`) — which calls your `__init__`.
2. Store `_fallback_torch_layer` — required for all fallback and comparison run modes
   (`NORMAL_WITH_FALLBACK`, `SEL`, `DPL`, `DPL_NO_ERROR_PROP`).
3. Copy the raw PyTorch weight tensors as instance attributes so that
   `preprocess_weights_impl` can find them.

`TTNNLinear.from_torch` shows the canonical pattern:

```python
@classmethod
def from_torch(cls, linear: nn.Linear):
    new_linear = cls(
        in_features=linear.in_features,
        out_features=linear.out_features,
    )
    new_linear._fallback_torch_layer = linear    # (1) store fallback
    new_linear.weight = linear.weight            # (2) copy raw tensors
    new_linear.bias = linear.bias
    return new_linear
```

Note that `from_torch` does **not** call `preprocess_weights()`. The run-mode
`module_run` method calls it lazily on the first forward pass.

---

## Step 3 — `preprocess_weights_impl`

`preprocess_weights_impl` runs once, before the first forward pass, and is responsible
for converting raw PyTorch tensors into TTNN-format host tensors. Weights must **stay on
host** after this method — moving to device happens in the next step.

The base class implementation recurses over `self.__dict__` to call
`preprocess_weights()` on any `TTNNModule` children. If your module composes child
`TTNNModule` instances, call `super().preprocess_weights_impl()` to trigger that
recursion.

### Converting a weight tensor

`TTNNLinear.preprocess_weights_impl` uses `ttnn.model_preprocessing` helpers:

```python
from ttnn.model_preprocessing import preprocess_linear_weight, preprocess_linear_bias

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

For a generic tensor (not a linear weight), use `ttnn.from_torch` directly:

```python
def preprocess_weights_impl(self):
    self.tt_scale_host = ttnn.from_torch(
        self.scale,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
```

The naming convention `*_host` for preprocessed-but-not-yet-on-device tensors is a
convention from the reference implementation; it is not enforced by the framework.

---

## Step 4 — `move_weights_to_device_impl`

This method uploads host-side TTNN tensors to the device. It is called after
`preprocess_weights_impl` and after `self.device` has been set by the run-mode
`module_run` path.

The base class implementation recurses over `self.__dict__` children, just like
`preprocess_weights_impl`. Call `super().move_weights_to_device_impl()` if you have
child `TTNNModule` instances.

### Guard: weights must be preprocessed and `self.device` must be set before this method is ever called

`TTNNModule.move_weights_to_device` (the public method) contains two sequential
assertions:

```python
assert self._preprocessed_weight, (
    f"Weights must be preprocessed for {self.module_name} before moving to device."
)
assert self.device is not None, (
    f"Device must be set for {self.module_name} before moving weights to device."
)
```

The preprocessed-weight assertion fires first, so calling `move_weights_to_device()`
before `preprocess_weights()` has been called will raise `AssertionError` even if the
device is already set. You do not need to re-check either guard inside
`move_weights_to_device_impl`, but doing so defensively is harmless.

### Example from `TTNNLinear`

```python
def move_weights_to_device_impl(self):
    self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
    self.tt_bias = (
        ttnn.to_device(self.tt_bias_host, self.device)
        if self.tt_bias_host is not None
        else None
    )
```

`ttnn.to_device` takes the TTNN host tensor and the device handle stored in
`self.device`. The resulting `self.tt_weight` and `self.tt_bias` are what the `forward`
method consumes.

---

## Step 5 — `deallocate_weights_impl`

This method frees device memory. It is the inverse of `move_weights_to_device_impl`.

### Calling `super()`

The base class `deallocate_weights_impl` recurses over `self.__dict__` children. You
**must** call `super().deallocate_weights_impl()` at the end of your override, or any
child `TTNNModule` instances will not have their weights freed. Forgetting this call is a
common source of device-memory leaks (see [fallback_and_debugging.md](fallback_and_debugging.md)).

```python
def deallocate_weights_impl(self):
    ttnn.deallocate(self.tt_weight)
    if self.tt_bias is not None:
        ttnn.deallocate(self.tt_bias)
    super().deallocate_weights_impl()   # recurse into children
```

This is exactly the pattern in `TTNNLinear.deallocate_weights_impl`.

---

## Step 6 — `forward`

`forward` receives TTNN tensors and must return TTNN tensors (or structures of TTNN
tensors).

### How inputs arrive

The run-mode `module_run` path (in `NormalRun.module_run`) applies a transform pipeline
to every input before calling `forward`:

```
wrap_to_torch_ttnn_tensor -> to_ttnn_wrap -> set_device_wrap(self.device)
```

By the time `forward` is called, each input tensor is a `ttnn.Tensor` already resident on
`self.device`. Your `forward` method can use `ttnn.*` ops directly.

### Inputs that are `TorchTTNNTensor`

When working within the torch-dispatch path (for ops not intercepted at the module
boundary), you may receive a `TorchTTNNTensor`. Extract the underlying `ttnn.Tensor` via
the `.to_ttnn` property:

```python
# Inside a TTNNModule that receives a TorchTTNNTensor:
tt = input_tensor.to_ttnn   # returns the ttnn.Tensor
```

`TTNNLinearActivation.forward` shows this pattern:

```python
def forward(self, hidden_states):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.activation(hidden_states.to_ttnn)
    return hidden_states
```

### Example from `TTNNLinear`

```python
def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        input_tensor = ttnn.to_layout(
            input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    input_tensor_shape = list(input_tensor.shape)
    input_shape = list(input_tensor_shape)
    while len(input_shape) < 4:
        input_shape.insert(1, 1)   # TTNN linear requires 4-D input
    input_tensor = ttnn.reshape(input_tensor, input_shape)
    tt_output = ttnn.linear(
        input_tensor, self.tt_weight, bias=self.tt_bias,
        memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_output = ttnn.reshape(tt_output, input_tensor_shape[:-1] + [self.out_features])
    return tt_output
```

---

## Step 7 — Using `@deallocate_weights_after`

If device memory is scarce, you can free weights immediately after each forward pass by
decorating `forward` with `@deallocate_weights_after`:

```python
from models.experimental.tt_symbiote.core.module import deallocate_weights_after

@deallocate_weights_after
def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    return super().forward(input_tensor)
```

The decorator (from `module.py`) calls `self.deallocate_weights()` after the wrapped
function returns:

```python
def deallocate_weights_after(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self.deallocate_weights()
        return result
    return wrapper
```

`TTNNLinearLLama` and `TTNNLinearLLamaBFloat16` in `linear.py` both use this pattern.
Because `deallocate_weights()` also resets `self._weights_on_device = False`, the next
invocation of `module_run` will call `move_weights_to_device_impl` again, re-uploading
weights from the preprocessed host copies.

---

## Step 8 — Using `@run_on_devices`

When a forward implementation is only correct on a specific set of device architectures,
decorate it with `@run_on_devices`:

```python
from models.experimental.tt_symbiote.core.module import run_on_devices, DeviceArch

@run_on_devices(DeviceArch.T3K)
def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    ...
```

The decorator reads the `MESH_DEVICE` environment variable at call time and maps it to a
`DeviceArch` via `MeshShapeToDeviceArch`. If the active architecture is not in the
allowed set, it raises `RuntimeError` before the forward body runs.

**Supported architectures (from `DeviceArch`):**

| Enum member | String value |
|---|---|
| `DeviceArch.N150` | `"n150"` |
| `DeviceArch.N300` | `"n300"` |
| `DeviceArch.T3K` | `"t3k_wh"` |
| `DeviceArch.TG` | `"gx_wh"` |
| `DeviceArch.P150` | `"p150"` |
| `DeviceArch.P300` | `"p300"` |
| `DeviceArch.P150x4` | `"p150x4"` |
| `DeviceArch.P150x8` | `"p150x8"` |
| `DeviceArch.BHGLX` | `"bhglx"` |

`TTNNLinearIColShardedWRowSharded` in `linear.py` shows `@run_on_devices(DeviceArch.T3K)`
on its `forward`. `TTNNLinearLLamaIColShardedWRowSharded` stacks the decorator with
`@deallocate_weights_after`:

```python
@deallocate_weights_after
@run_on_devices(DeviceArch.T3K)
def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    return super().forward(input_tensor)
```

When stacking, put `@deallocate_weights_after` outermost so that deallocation happens
after the arch check passes and the body runs.

---

## Step 9 — Complete worked example: `TTNNMyLinear`

The following is a complete, minimal custom module that replaces `nn.Linear`. It
deliberately mirrors the structure of `TTNNLinear` so you can see each piece together.

```python
# my_linear.py
import ttnn
import torch
from torch import nn
from ttnn.model_preprocessing import preprocess_linear_weight, preprocess_linear_bias

from models.experimental.tt_symbiote.core.module import TTNNModule


class TTNNMyLinear(TTNNModule):
    """Drop-in replacement for nn.Linear, using TTNN bfloat16 weights."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()                      # initialise all sentinel flags
        self.in_features = in_features
        self.out_features = out_features

    @classmethod
    def from_torch(cls, linear: nn.Linear):
        new_mod = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
        )
        new_mod._fallback_torch_layer = linear  # required for fallback run modes
        new_mod.weight = linear.weight          # raw PyTorch weight; used in preprocess
        new_mod.bias = linear.bias
        return new_mod

    def preprocess_weights_impl(self):
        """Convert PyTorch tensors to TTNN format; keep on host."""
        self.tt_weight_host = preprocess_linear_weight(
            self.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.tt_bias_host = None
        if self.bias is not None:
            self.tt_bias_host = preprocess_linear_bias(
                self.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
        # No children to recurse into, but super() call is harmless.
        super().preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        """Upload host tensors to the device."""
        self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
        self.tt_bias = (
            ttnn.to_device(self.tt_bias_host, self.device)
            if self.tt_bias_host is not None
            else None
        )
        super().move_weights_to_device_impl()

    def deallocate_weights_impl(self):
        """Free device tensors."""
        ttnn.deallocate(self.tt_weight)
        if self.tt_bias is not None:
            ttnn.deallocate(self.tt_bias)
        super().deallocate_weights_impl()       # MUST call super to recurse children

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(
                input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        input_tensor_shape = list(input_tensor.shape)
        input_shape = list(input_tensor_shape)
        while len(input_shape) < 4:
            input_shape.insert(1, 1)
        input_tensor = ttnn.reshape(input_tensor, input_shape)
        tt_output = ttnn.linear(
            input_tensor, self.tt_weight, bias=self.tt_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tt_output = ttnn.reshape(
            tt_output, input_tensor_shape[:-1] + [self.out_features]
        )
        return tt_output
```

---

## Step 10 — Registering the module and running the setup sequence

### Building the replacement dict

```python
from models.experimental.tt_symbiote.utils.module_replacement import (
    register_module_replacement_dict,
)
from torch import nn
from my_linear import TTNNMyLinear

replacement_dict = {
    nn.Linear: TTNNMyLinear,
}
```

### Calling `register_module_replacement_dict`

```python
# torch_model is the original nn.Module you want to accelerate.
ttnn_modules = register_module_replacement_dict(
    model=torch_model,
    old_class_to_new_class_dict=replacement_dict,
    model_config=None,          # optional dict passed to set_model_config()
)
# ttnn_modules is a dict[str, TTNNModule] mapping module_name -> replaced module
```

`register_module_replacement_dict` (in `module_replacement.py`) walks the model tree,
calls `TTNNMyLinear.from_torch(old_nn_linear)` for each matched `nn.Linear`, and splices
the new module back in-place. It also assigns `_unique_name` from the model's
`named_modules()` traversal.

### Setting the device and running

```python
device = ttnn.open_device(device_id=0)

# Set device on every replaced TTNNModule
for name, module in ttnn_modules.items():
    module.to_device(device)

# Now call the model normally.
# The NormalRun.module_run path handles preprocess_weights and move_weights_to_device
# lazily on the first call.
output = torch_model(input_tensor)
```

**Order matters:** `to_device` must be called before the first forward pass. The
`NormalRun.module_run` method asserts `self.device is not None` before calling
`move_weights_to_device`.

### Excluding specific module instances

Pass a set of fully-qualified module names to `exclude_replacement`:

```python
ttnn_modules = register_module_replacement_dict(
    model=torch_model,
    old_class_to_new_class_dict=replacement_dict,
    exclude_replacement={"encoder.layer.0.attention.query"},
)
```

The exclusion check is performed by name using the model's `named_modules()` mapping.

---

**Next:** [`fallback_and_debugging.md`](./fallback_and_debugging.md)
