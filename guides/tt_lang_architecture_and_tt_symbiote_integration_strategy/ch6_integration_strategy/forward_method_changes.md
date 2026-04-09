# Forward Method Changes

This section provides concrete before/after code examples showing how TT-Symbiote `forward()` methods change when replacing TTNN ops with TT-Lang compiled kernels. These are the actual code changes that constitute the integration.

## Pattern Overview

The transformation follows a consistent pattern across all module types:

| Step | Current (TTNN) | Proposed (TT-Lang) |
|------|----------------|---------------------|
| 1. Layout guard | `ttnn.to_layout(...)` | `ttnn.to_layout(...)` (unchanged) |
| 2. Shape setup | `ttnn.reshape(...)` | `ttnn.reshape(...)` (unchanged) |
| 3. Output allocation | Implicit (inside TTNN op) | Explicit `ttnn.empty(...)` |
| 4. Compute | `ttnn.linear(x, w)` | `kernel_fn(x, w, out)` |
| 5. Post-process | `ttnn.reshape(...)` | `ttnn.reshape(...)` (unchanged) |

The only structural changes are in steps 3 and 4: explicit output allocation and a different call syntax.

## Example 1: Activation Function (Simplest Case)

Activation functions are the simplest integration target --- single input, single output, no weights.

### Before: `TTNNSilu` (from `modules/activation.py`)

```python
class TTNNSilu(TTNNModule):
    def __init__(self):
        super().__init__()
        self._fallback_torch_layer = torch.nn.SiLU()

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT,
                                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_output = ttnn.silu(input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return tt_output
```

### After: `TTNNSiluTTLang`

```python
import ttl

# Define the kernel at module level (compiled once, cached)
@ttl.pykernel_gen(grid="auto", num_outs=1)
def silu_kernel(x, output, grid, memory_space, tiled):
    import ttl

    cb_x = ttl.CircularBuffer(x, block_count=2)
    cb_out = ttl.CircularBuffer(output, block_count=2)

    @ttl.compute()
    def compute_silu(grid, memory_space, tiled):
        for block in ttl.TensorBlock(cb_x):
            ttl.copy(cb_x, cb_out, transform=lambda tile: tile.sigmoid() * tile)

    @ttl.datamovement()
    def read_x(grid, memory_space, tiled):
        ttl.copy(x >> cb_x)

    @ttl.datamovement()
    def write_out(grid, memory_space, tiled):
        ttl.copy(cb_out >> output)


class TTNNSiluTTLang(TTNNModule):
    def __init__(self):
        super().__init__()
        self._fallback_torch_layer = torch.nn.SiLU()

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT,
                                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Allocate output (same shape/dtype as input)
        output = ttnn.empty_like(input_tensor,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Execute TT-Lang kernel
        silu_kernel(input_tensor, output)
        return output
```

Key differences:
- Output tensor is explicitly allocated via `ttnn.empty_like`
- The kernel function is defined once at module scope; `pykernel_gen`'s internal cache handles reuse
- The kernel itself contains the compute and data-movement thread definitions

## Example 2: Linear Layer (Weight-Bearing Module)

### Before: `TTNNLinear.forward` (from `modules/linear.py`)

```python
class TTNNLinear(TTNNModule):
    def preprocess_weights_impl(self):
        self.tt_weight_host = preprocess_linear_weight(
            self.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.tt_bias_host = None
        if self.bias is not None:
            self.tt_bias_host = preprocess_linear_bias(
                self.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def move_weights_to_device_impl(self):
        self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
        self.tt_bias = (ttnn.to_device(self.tt_bias_host, self.device)
                        if self.tt_bias_host is not None else None)

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT,
                                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
        input_tensor_shape = list(input_tensor.shape)
        input_shape = list(input_tensor_shape)
        while len(input_shape) < 4:
            input_shape.insert(1, 1)
        input_tensor = ttnn.reshape(input_tensor, input_shape)
        tt_output = ttnn.linear(input_tensor, self.tt_weight, bias=self.tt_bias,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_output = ttnn.reshape(tt_output,
                                  input_tensor_shape[:-1] + [self.out_features])
        return tt_output
```

### After: `TTNNLinearTTLang`

```python
import ttl

@ttl.pykernel_gen(grid="auto", num_outs=1)
def linear_kernel(x, weight, output, grid, memory_space, tiled):
    """Fused matmul kernel: output = x @ weight^T"""
    import ttl

    cb_x = ttl.CircularBuffer(x, block_count=2)
    cb_w = ttl.CircularBuffer(weight, block_count=2)
    cb_out = ttl.CircularBuffer(output, block_count=2)

    @ttl.compute()
    def compute_matmul(grid, memory_space, tiled):
        # Matmul compute logic
        ...

    @ttl.datamovement()
    def read_inputs(grid, memory_space, tiled):
        ttl.copy(x >> cb_x)
        ttl.copy(weight >> cb_w)

    @ttl.datamovement()
    def write_output(grid, memory_space, tiled):
        ttl.copy(cb_out >> output)


class TTNNLinearTTLang(TTNNModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    # preprocess_weights_impl: UNCHANGED from TTNNLinear
    def preprocess_weights_impl(self):
        self.tt_weight_host = preprocess_linear_weight(
            self.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        # Bias handled separately (not fused into kernel in this example)
        self.tt_bias_host = None
        if self.bias is not None:
            self.tt_bias_host = preprocess_linear_bias(
                self.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # move_weights_to_device_impl: UNCHANGED from TTNNLinear
    def move_weights_to_device_impl(self):
        self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
        self.tt_bias = (ttnn.to_device(self.tt_bias_host, self.device)
                        if self.tt_bias_host is not None else None)

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT,
                                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
        input_tensor_shape = list(input_tensor.shape)
        input_shape = list(input_tensor_shape)
        while len(input_shape) < 4:
            input_shape.insert(1, 1)
        input_tensor = ttnn.reshape(input_tensor, input_shape)

        # Allocate output tensor
        output_shape = input_shape[:-1] + [self.out_features]
        output = ttnn.empty(output_shape, dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT,
                             device=self.device,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Execute TT-Lang matmul kernel
        linear_kernel(input_tensor, self.tt_weight, output)

        # Apply bias separately (or fuse into kernel for further optimization)
        if self.tt_bias is not None:
            output = ttnn.add(output, self.tt_bias)

        output = ttnn.reshape(output,
                               input_tensor_shape[:-1] + [self.out_features])
        return output
```

Key observations:
- `preprocess_weights_impl` and `move_weights_to_device_impl` are **completely unchanged**
- `self.tt_weight` is passed as a positional argument to the kernel
- Output allocation is explicit; shape is computed from input shape and `self.out_features`
- Bias addition remains a separate TTNN op (it could be fused into the kernel for a further performance win --- see [Chapter 7](../ch7_fusion_targets/index.md))

## Example 3: Fused Linear + Activation (The High-Value Target)

The real payoff comes from fusing operations that are currently separate TTNN calls. Here is a fused linear + SiLU, replacing what `TTNNLinearActivation` does with two separate ops:

### Before: `TTNNLinearActivation.forward`

```python
class TTNNLinearActivation(TTNNModule):
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)       # ttnn.linear inside
        hidden_states = self.activation(hidden_states)   # ttnn.silu inside
        return hidden_states
```

This executes two separate device programs with an intermediate tensor round-trip through DRAM.

### After: Fused TT-Lang Kernel

```python
@ttl.pykernel_gen(grid="auto", num_outs=1)
def fused_linear_silu(x, weight, output, grid, memory_space, tiled):
    """Fused matmul + SiLU: output = silu(x @ weight^T)"""
    import ttl

    cb_x = ttl.CircularBuffer(x, block_count=2)
    cb_w = ttl.CircularBuffer(weight, block_count=2)
    cb_out = ttl.CircularBuffer(output, block_count=2)
    cb_intermediate = ttl.CircularBuffer.like(cb_out, block_count=1)

    @ttl.compute()
    def compute(grid, memory_space, tiled):
        # Matmul into intermediate, then SiLU into output
        # Intermediate stays in L1 --- never hits DRAM
        ...

    @ttl.datamovement()
    def read(grid, memory_space, tiled):
        ttl.copy(x >> cb_x)
        ttl.copy(weight >> cb_w)

    @ttl.datamovement()
    def write(grid, memory_space, tiled):
        ttl.copy(cb_out >> output)


class TTNNLinearSiluTTLang(TTNNModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    # preprocess/move unchanged from TTNNLinear
    ...

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT,
                                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # ... reshape logic ...

        output = ttnn.empty(output_shape, dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT,
                             device=self.device,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)

        fused_linear_silu(input_tensor, self.tt_weight, output)
        return output
```

The performance benefit: the intermediate matmul result stays in L1 circular buffers and is consumed by the SiLU computation in the same program. No DRAM round-trip for the intermediate tensor.

## Module-Level Kernel Caching

The `pykernel_gen` decorator already maintains a per-kernel cache (a Python `dict` on the wrapper's closure). When kernels are defined at module scope (as shown above), all instances of a module class share the same compiled kernel cache.

This means:
- 32 decoder layers each containing a `TTNNLinearTTLang` share one cache entry (assuming same shapes)
- The first layer compiles, the remaining 31 get cache hits
- Cache keys include tensor shapes, dtypes, and memory spaces, so different layer sizes correctly trigger separate compilations

For kernels that should have per-instance caches (rare, but possible if different instances use different compiler options), define the kernel inside `__init__`:

```python
class TTNNCustomModule(TTNNModule):
    def __init__(self, grid_override=None):
        super().__init__()
        # Per-instance kernel with custom grid
        @ttl.pykernel_gen(grid=grid_override or "auto", num_outs=1)
        def _my_kernel(x, output, grid, memory_space, tiled):
            ...
        self._kernel_fn = _my_kernel
```

## `@deallocate_weights_after` Compatibility

The `@deallocate_weights_after` decorator (from `core/module.py`) wraps `forward()` and calls `self.deallocate_weights()` after it returns:

```python
def deallocate_weights_after(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self.deallocate_weights()   # Runs after forward() completes
        return result
    return wrapper
```

This is fully compatible with TT-Lang kernels. The kernel executes synchronously within `forward()`, consuming the weight tensors. By the time `forward()` returns, the weights have been read and the output computed. Deallocation is safe.

Example with the decorator (mirroring `TTNNLinearLLama`):

```python
class TTNNLinearTTLangLLama(TTNNModule):
    def preprocess_weights_impl(self):
        self.tt_weight_host = preprocess_linear_weight(
            self.weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    def move_weights_to_device_impl(self):
        self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)

    @deallocate_weights_after
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        # ... layout/reshape ...
        output = ttnn.empty(output_shape, dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT,
                             device=self.device,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
        linear_kernel(input_tensor, self.tt_weight, output)
        # After return, deallocate_weights_after calls self.deallocate_weights()
        return output
```

## Device Placement

TT-Lang kernels obtain the device from the first tensor argument (`args[0].device(`) for grid validation. In TT-Symbiote, the device is set on the module via `to_device()` and propagated to weights via `move_weights_to_device_impl`.

No explicit device argument is needed for the kernel. The device is implicit in the tensor arguments. This matches TT-Symbiote's design where `forward()` methods never reference `self.device` for compute --- only for weight management.

However, output tensor allocation does require a device reference:

```python
output = ttnn.empty(shape, dtype=ttnn.bfloat16,
                     layout=ttnn.TILE_LAYOUT,
                     device=self.device,          # From TTNNModule
                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

An alternative is to extract the device from the input tensor:

```python
output = ttnn.empty(shape, dtype=ttnn.bfloat16,
                     layout=ttnn.TILE_LAYOUT,
                     device=input_tensor.device(),  # From activation
                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

Both are valid. Using `self.device` is consistent with existing TT-Symbiote conventions.

## Summary of Required Code Changes

For a module adopting a TT-Lang kernel:

| File / Method | Change Required |
|---------------|----------------|
| Kernel definition (module-level) | **New**: define `@ttl.pykernel_gen` function |
| `__init__` | None (unless per-instance kernel caching needed) |
| `preprocess_weights_impl` | None |
| `move_weights_to_device_impl` | None (optional: add pre-compilation) |
| `deallocate_weights_impl` | None |
| `forward` | **Modified**: add output allocation, replace TTNN op call with kernel call |
| `@deallocate_weights_after` | Compatible, no changes |
| `@run_on_devices` | Compatible, no changes |
| `@trace_enabled` / `@trace_disabled` | Compatible, no changes |

The integration is surgical: define the kernel, modify `forward()`, leave everything else untouched.

---

**Next:** [Chapter 7 --- High-Value Fusion Targets](../ch7_fusion_targets/index.md)
