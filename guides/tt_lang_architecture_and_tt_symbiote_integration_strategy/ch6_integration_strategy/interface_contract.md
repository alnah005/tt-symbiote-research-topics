# Interface Contract: `CompiledTTNNKernel`

This section defines the precise API surface that TT-Symbiote modules must program against when using TT-Lang compiled kernels. Everything here is grounded in the actual implementation in `ttl_api.py` and `kernel_runner.py`.

## `CompiledTTNNKernel.__call__` API

The compiled kernel object is a callable with a strict contract:

```python
class CompiledTTNNKernel:
    def __call__(self, *args):
        """Execute the kernel with the given tensors.

        Args:
            *args: Exactly self.num_tensors ttnn.Tensor arguments.
                   Inputs come first, outputs last.

        Raises:
            ValueError: If len(args) != self.num_tensors
            ValueError: If kernel grid exceeds device compute grid
        """
```

The implementation (from `ttl_api.py` lines 525-561) performs two validations before execution:

1. **Argument count check:** `len(args) != self.num_tensors` raises `ValueError`. The number of tensors is fixed at compile time.

2. **Grid bounds check:** The kernel's `core_ranges` bounding box is compared against `device.compute_with_storage_grid_size()`. If the kernel grid exceeds available cores, execution fails immediately.

After validation, execution flows through `run_kernel_on_device` in `kernel_runner.py`, which:
- Builds `TensorAccessorArgs` for each tensor (compile-time args for C++ address calculation)
- Constructs `KernelDescriptor` objects with kernel source paths, configs, and runtime args
- Constructs `CBDescriptor` objects for circular buffers
- Assembles a `ProgramDescriptor` and dispatches via `ttnn.generic_op`

## Tensor Requirements

Every tensor argument must satisfy three properties, validated in `_compile_ttnn_kernel` (lines 628-640):

### 1. TILE_LAYOUT

```python
if hasattr(arg, "layout") and "TILE" not in str(arg.layout):
    raise ValueError(
        f"TTNN interop requires tilized tensors, but tensor {i} has layout {arg.layout}. "
        f"Use ttnn.to_layout(tensor, ttnn.TILE_LAYOUT) to convert."
    )
```

TT-Symbiote modules already guard for this. The standard pattern seen in `TTNNLinear.forward`, `TTNNSilu.forward`, etc. is:

```python
if input_tensor.layout != ttnn.TILE_LAYOUT:
    input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

### 2. L1 or DRAM Memory Space

```python
mem_space = _detect_memory_space_from_tensor(arg, "unknown")
if mem_space not in ("L1", "DRAM"):
    raise ValueError(...)
```

The memory space is detected from `tensor.memory_config().buffer_type`. TT-Symbiote's `move_weights_to_device_impl` uses `ttnn.to_device()` which places tensors in DRAM by default (via `ttnn.DRAM_MEMORY_CONFIG`).

### 3. Homogeneous Tensor Types

All arguments must be `ttnn.Tensor`. No mixing of `torch.Tensor` and `ttnn.Tensor`:

```python
ttnn_count = sum(1 for arg in args if is_ttnn_tensor(arg))
if ttnn_count > 0 and ttnn_count < len(args):
    raise ValueError("TTNN interop requires all tensors to be the same type...")
```

This is inherently satisfied in TT-Symbiote's `forward()` because activations arrive as `ttnn.Tensor` (via the dispatch system) and weights are placed on-device by `move_weights_to_device_impl`.

## Grid Resolution: `_resolve_grid`

The grid determines how many Tensix cores execute the kernel. TT-Lang's `_resolve_grid` function (line 406) supports three modes:

```python
def _resolve_grid(grid, args, kwargs):
    if callable(grid):
        return grid(*args, **kwargs)     # Dynamic: computed from tensors
    if grid == "auto":
        device = arg.device()
        device_grid = device.compute_with_storage_grid_size()
        return (device_grid.x, device_grid.y)  # Use full device grid
    return grid                           # Static: (cols, rows) tuple
```

For TT-Symbiote integration, the recommended approach depends on the kernel:

| Strategy | When to use | Example |
|----------|-------------|---------|
| `grid="auto"` | Embarrassingly parallel ops (elementwise, activation) | `@ttl.pykernel_gen(grid="auto")` |
| Static tuple | Kernels with specific tiling requirements | `@ttl.pykernel_gen(grid=(8, 8))` |
| Callable | Grid depends on input shape (e.g., matmul) | `@ttl.pykernel_gen(grid=lambda x, w, out: compute_grid(x, w))` |

The grid is resolved at call time, so it can adapt to different devices (N150 vs. T3K) without recompilation --- the cache key includes tensor properties but not the grid directly.

## Compilation Caching: `_make_cache_key`

TT-Lang avoids redundant recompilation through a per-kernel cache. The cache key is built by `_make_cache_key` (line 122):

```python
def _make_cache_key(args, fp32_dest_acc_en, dst_full_sync_en, compiler_options):
    tensor_key = tuple(
        _get_tensor_cache_info(arg) for arg in args if is_ttnn_tensor(arg)
    )
    mesh_key = None
    for arg in args:
        if is_ttnn_tensor(arg) and _is_mesh_tensor(arg):
            mesh_key = tuple(arg.device().shape)
            break
    return (tensor_key, mesh_key, fp32_dest_acc_en, dst_full_sync_en, compiler_options)
```

Each tensor contributes a 4-tuple via `_get_tensor_cache_info`:

```python
def _get_tensor_cache_info(tensor) -> tuple:
    shape = tuple(tensor.shape)
    dtype = str(tensor.dtype)
    mem_config = tensor.memory_config()
    memory_space = str(mem_config.buffer_type) if hasattr(mem_config, "buffer_type") else "unknown"
    layout = str(tensor.layout) if hasattr(tensor, "layout") else "unknown"
    return (shape, dtype, memory_space, layout)
```

**Implications for TT-Symbiote:**

- **Fixed-shape inference** (the common case for LLM serving): the kernel compiles once on the first forward pass and is reused for all subsequent calls. Zero overhead after warmup.

- **Dynamic shapes** (e.g., variable sequence lengths): each new shape triggers a recompilation. This is identical to how TTNN ops behave with `ttnn.generic_op` program caching.

- **Mesh tensors**: the device mesh shape is included in the cache key. A kernel compiled for N150 (single device) will recompile when run on T3K (8 devices). The `_is_mesh_tensor` check detects multi-device tensors by checking `prod(device.shape) > 1`.

## The `pykernel_gen` Decorator: Full Lifecycle

Putting it all together, here is the flow when a TT-Symbiote module calls a TT-Lang kernel:

```
Module.forward(activation)
    |
    v
@ttl.pykernel_gen decorated function called
    |
    +-- _resolve_grid(grid, args, kwargs)
    |       -> (cols, rows) tuple
    |
    +-- _make_cache_key(args, fp32_dest_acc_en, ...)
    |       -> cache lookup
    |
    +-- [CACHE HIT] -> compiled_kernel
    |   [CACHE MISS] -> _compile_kernel(f, args, ...)
    |       |
    |       +-- f(*args) triggers @compute/@datamovement decorators
    |       +-- MLIR pipeline: TTL -> TTKernel -> EmitC -> C++
    |       +-- _compile_ttnn_kernel(module, args, grid, ...)
    |       |       -> validates tensors (TILE_LAYOUT, L1/DRAM, no mixed types)
    |       |       -> builds CompiledTTNNKernel
    |       +-- cache[key] = compiled_kernel
    |
    +-- compiled_kernel(*args)
    |       -> KernelSpec construction
    |       -> run_kernel_on_device(specs, tensors, cb_configs, core_ranges)
    |           -> ttnn.generic_op(io_tensors, program)
    |
    v
Result (output tensor modified in-place)
```

## Output Tensor Convention

A critical difference from TTNN ops: **`CompiledTTNNKernel` does not allocate output tensors.** The caller must pre-allocate the output and pass it as the last argument. In `kernel_runner.py` (line 273):

```python
io_tensors = list(tensors)
if len(io_tensors) < 2:
    io_tensors = [io_tensors[-1]] + io_tensors  # Duplicate for generic_op >= 2 requirement
return ttnn.generic_op(io_tensors, program)
```

For TT-Symbiote, this means the `forward()` method must allocate the output tensor before calling the kernel:

```python
# Allocate output tensor with desired shape, dtype, and memory config
output = ttnn.empty(output_shape, dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
# Execute kernel: inputs first, output last
self._compiled_kernel(input_tensor, self.tt_weight, output)
return output
```

This is the most significant integration detail --- it inverts the allocation model from "op returns a new tensor" to "caller provides the output buffer."

---

**Next:** [`weight_pipeline_interaction.md`](./weight_pipeline_interaction.md)
