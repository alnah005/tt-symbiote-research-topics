# Weight Pipeline Interaction

TT-Symbiote's weight management pipeline --- `preprocess_weights` -> `move_weights_to_device` -> `forward` -> `deallocate_weights` --- is the backbone of its module lifecycle (detailed in [Chapter 5: TTNNModule Lifecycle](../ch5_symbiote_architecture/ttnn_module_lifecycle.md)). This section examines how TT-Lang's JIT compilation model interacts with each stage of this pipeline, and what changes are needed.

## The Weight Pipeline, Revisited

From `core/module.py`, the `TTNNModule` lifecycle enforces a strict ordering:

```python
class TTNNModule:
    def preprocess_weights(self):
        """Called once before first use."""
        if not self._preprocessed_weight:
            self._preprocessed_weight = True
            self.preprocess_weights_impl()  # Subclass hook

    def move_weights_to_device(self):
        """Move preprocessed weights to device."""
        assert self._preprocessed_weight
        assert self.device is not None
        if not self._weights_on_device:
            self._weights_on_device = True
            self.move_weights_to_device_impl()  # Subclass hook

    def forward(self, *args, **kwargs):
        raise NotImplementedError
```

The guarantee is: by the time `forward()` is called, `self._preprocessed_weight == True` and `self._weights_on_device == True`. This means weight tensors stored as `self.tt_weight` etc. are already `ttnn.Tensor` objects on device, in `TILE_LAYOUT`, in L1 or DRAM.

**This guarantee is exactly what TT-Lang needs.** The `CompiledTTNNKernel.__call__` API requires all arguments to be on-device `ttnn.Tensor` in `TILE_LAYOUT`. No changes to the pipeline ordering are needed.

## Weight Tensors as Kernel Arguments

In the current TTNN op pattern, weights are passed as named parameters to TTNN ops:

```python
# Current: TTNNLinear.forward
tt_output = ttnn.linear(input_tensor, self.tt_weight, bias=self.tt_bias,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

In the TT-Lang kernel pattern, weights become positional tensor arguments to the compiled kernel:

```python
# Proposed: weight is a positional arg to the kernel
@ttl.pykernel_gen(grid="auto", num_outs=1)
def fused_linear_silu(x, weight, output, grid, memory_space, tiled):
    # ... compute/datamovement threads defined here ...
    pass

# In forward():
fused_linear_silu(input_tensor, self.tt_weight, output)
```

The kernel sees weights and activations identically --- they are all `ttnn.Tensor` entries in the `tensors` list passed to `run_kernel_on_device`. The distinction between "weight" and "activation" is purely semantic; the kernel indexes them by position.

### Tensor Ordering Convention

TT-Lang kernels use a positional convention:
- **Inputs** (activations, weights, biases) come first
- **Outputs** come last

The `num_outs` parameter to `@ttl.pykernel_gen` tells the compiler how many trailing arguments are outputs. For a linear layer with bias:

```python
@ttl.pykernel_gen(grid="auto", num_outs=1)
def fused_linear(x, weight, bias, output, grid, memory_space, tiled):
    # x:      input activation   (tensor index 0)
    # weight: weight matrix      (tensor index 1)
    # bias:   bias vector        (tensor index 2)
    # output: output activation  (tensor index 3, the single output)
    ...
```

## Lazy Compilation Pattern

TT-Lang kernels are compiled JIT --- on the first call with actual tensor arguments. This creates a natural interaction with TT-Symbiote's lifecycle:

```
Time
  |
  v
  preprocess_weights_impl()     -- Torch tensors -> tilized host tensors
  move_weights_to_device_impl() -- Host tensors -> device tensors
  forward() [first call]        -- TT-Lang compiles kernel (uses tensor shapes/dtypes)
                                   CompiledTTNNKernel cached
  forward() [subsequent calls]  -- Cache hit, no recompilation
```

The JIT compilation happens inside `pykernel_gen._wrapper` (from `ttl_api.py` line 1386):

```python
@functools.wraps(f)
def _wrapper(*args, **kwargs):
    resolved_grid = _resolve_grid(grid, args, kwargs)
    cache_key = _make_cache_key(args, fp32_dest_acc_en=..., ...)

    if cache_key in cache:
        compiled_kernel = cache[cache_key]
    else:
        compiled_kernel = _compile_kernel(f, args, kwargs, resolved_grid, ...)
        if compiled_kernel is not None:
            cache[cache_key] = compiled_kernel

    if compiled_kernel is not None and _should_execute():
        result = compiled_kernel(*args)
        return result
```

### First-Call Overhead

The first `forward()` call incurs the full compilation cost:
1. Python AST parsing of the kernel function
2. MLIR generation and the full TTL pipeline (see [Chapter 2](../ch2_compilation_pipeline/index.md))
3. C++ code generation and writing to `/tmp`
4. `CompiledTTNNKernel` construction

For TT-Symbiote models in production (LLM inference with fixed shapes), this is a one-time warmup cost. The cache key depends on tensor shapes, dtypes, and memory spaces --- all of which are fixed after model initialization.

### Alternative: Eager Pre-Compilation

For latency-sensitive deployments, kernels can be pre-compiled during `move_weights_to_device_impl` by making a dummy forward pass:

```python
def move_weights_to_device_impl(self):
    self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
    # Pre-compile: create dummy input + output with correct shapes
    dummy_input = ttnn.empty([1, 1, self.seq_len, self.in_features],
                              dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT,
                              device=self.device,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
    dummy_output = ttnn.empty([1, 1, self.seq_len, self.out_features],
                               dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT,
                               device=self.device,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # Trigger compilation but skip execution
    import os
    os.environ["TTLANG_COMPILE_ONLY"] = "1"
    self._kernel_fn(dummy_input, self.tt_weight, dummy_output)
    del os.environ["TTLANG_COMPILE_ONLY"]
    ttnn.deallocate(dummy_input)
    ttnn.deallocate(dummy_output)
```

The `TTLANG_COMPILE_ONLY=1` environment variable (checked by `_should_execute()` at line 142) causes `pykernel_gen` to compile and cache the kernel without dispatching to the device.

## Mesh Tensor Handling

TT-Symbiote supports multi-device execution via mesh tensors (see `TTNNLinearInputShardedWeightSharded` in `modules/linear.py`). Weights are sharded across devices using `ttnn.shard_tensor_to_mesh_mapper`:

```python
# From TTNNLinearInputShardedWeightSharded.move_weights_to_device_impl
self.tt_weight_host = preprocess_linear_weight(
    self.tt_weight_host,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    weights_mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(self.device, dim=self.weight_dim),
)
self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
```

TT-Lang handles mesh tensors transparently. The key mechanism is in `_compile_kernel` (line 1011-1013):

```python
is_mesh = has_ttnn_tensors and any(_is_mesh_tensor(arg) for arg in args)
compile_args = args
# For mesh tensors, tensor.shape already returns the per-device shard dimensions
```

And in `_make_cache_key` (line 134):

```python
mesh_key = None
for arg in args:
    if is_ttnn_tensor(arg) and _is_mesh_tensor(arg):
        mesh_key = tuple(arg.device().shape)
        break
return (tensor_key, mesh_key, ...)
```

The mesh device shape is included in the cache key, so a kernel compiled for a single-device N150 will correctly recompile when run on a T3K mesh. However, the kernel itself does not need to be mesh-aware --- `ttnn.generic_op` handles multi-device dispatch internally. Each device executes the same program on its local shard.

### Implications for Sharded Weights

When a TT-Symbiote module shards weights across devices:

1. `preprocess_linear_weight(..., weights_mesh_mapper=...)` creates a mesh tensor on host
2. `ttnn.to_device(mesh_tensor, mesh_device)` places shards on respective devices
3. `tensor.shape` returns per-device shard dimensions (not the global shape)
4. TT-Lang compiles the kernel against the shard dimensions
5. `ttnn.generic_op` executes the kernel on each device

No special handling is needed in the TT-Lang kernel definition. The kernel is written for single-device shapes; mesh distribution is handled by the TTNN runtime.

### Collective Operations

One limitation: TT-Lang kernels currently compile to single-device programs. Collective operations like `reduce_scatter` and `all_gather` (used in `TTNNLinearIColShardedWRowSharded.forward`) must remain as separate TTNN op calls:

```python
# TT-Lang kernel handles the matmul
fused_matmul_kernel(input_tensor, self.tt_weight, partial_output)

# Collective ops remain as TTNN calls
tt_output = ttnn.reduce_scatter(partial_output, dim=3, num_links=1,
                                 cluster_axis=1,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                 topology=ttnn.Topology.Ring)
```

The TT-Lang kernel produces a per-device partial result; the TTNN collective ops handle the cross-device reduction. This is a natural boundary --- fusion across devices requires a different programming model.

## `preprocess_weights_impl` Changes

For most modules, `preprocess_weights_impl` requires **no changes** when adopting TT-Lang kernels. The preprocessing step converts torch tensors to TTNN-compatible host tensors:

```python
# Unchanged
def preprocess_weights_impl(self):
    self.tt_weight_host = preprocess_linear_weight(
        self.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
```

The only scenario requiring changes is when a TT-Lang kernel expects a weight layout that differs from what `preprocess_linear_weight` produces (e.g., a custom tiling scheme). In practice, standard `TILE_LAYOUT` with `bfloat16` or `bfloat8_b` covers all current use cases.

## `move_weights_to_device_impl` Changes

Similarly minimal. The standard pattern remains:

```python
def move_weights_to_device_impl(self):
    self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
```

If pre-compilation is desired (to avoid first-call latency), this is the natural place to trigger it, as shown in the "Eager Pre-Compilation" section above.

## `deallocate_weights_impl` Changes

No changes needed. Weight deallocation is independent of how the weights are consumed in `forward()`:

```python
def deallocate_weights_impl(self):
    ttnn.deallocate(self.tt_weight)
    if self.tt_bias is not None:
        ttnn.deallocate(self.tt_bias)
```

The `@deallocate_weights_after` decorator (from `core/module.py` line 256) wraps `forward()` and calls `self.deallocate_weights()` after it returns. Since TT-Lang kernels execute synchronously within `forward()`, the weights are guaranteed to have been consumed before deallocation runs.

---

**Next:** [`forward_method_changes.md`](./forward_method_changes.md)
