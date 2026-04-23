# Parallel Linear Layers

## Prerequisites

- [Chapter 2 Index](./index.md): understanding of tensor parallelism (TP) and sequence parallelism (SP).
- [CCLManager](./ccl_manager.md): understanding of all-gather, reduce-scatter, and persistent buffer caching.
- [Chapter 1 -- Module and Parameter](../ch1_architecture_overview/module_and_parameter.md): understanding of `Parameter` and its `mesh_axes` mechanism.

---

## Overview

TT-DiT implements three linear layer variants in `models/tt_dit/layers/linear.py`:

1. **`Linear`** -- Standard linear layer with fully replicated weights. No parallelism.
2. **`ColParallelLinear`** -- Column-parallel linear: shards the weight's output dimension across devices. Expects replicated input, produces column-fractured output.
3. **`RowParallelLinear`** -- Row-parallel linear: shards the weight's input dimension across devices. Expects column-fractured input (from a preceding `ColParallelLinear`), produces reduced output via `reduce_scatter`.

Together, `ColParallelLinear` and `RowParallelLinear` implement the **Megatron-LM tensor parallelism** pattern. They are always used as pairs in TT-DiT:
- In attention blocks: `ColParallelLinear` for QKV projections, followed by `ColParallelLinear` for output projections (with all-gather between them for TP).
- In feed-forward blocks: `ColParallelLinear` for the up-projection (ff1), followed by `RowParallelLinear` for the down-projection (ff2), with only one `reduce_scatter` between them.

---

## Linear (Replicated)

The base `Linear` class is straightforward:

```python
# models/tt_dit/layers/linear.py

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True,
                 activation_fn=None, dtype=ttnn.bfloat16, mesh_device=None):
        super().__init__()
        self.weight = Parameter(
            total_shape=[in_features, out_features],  # Already transposed
            device=mesh_device, dtype=dtype)
        self.bias = Parameter(
            total_shape=[1, out_features],
            device=mesh_device, dtype=dtype) if bias else None
```

Key characteristics:
- **Weight shape is `[in_features, out_features]`** (pre-transposed from PyTorch's `[out_features, in_features]`). The `_prepare_torch_state` method handles this transposition during weight loading.
- **No `mesh_axes`** specified on `Parameter`, so weights are fully replicated across all devices.
- **Activation functions** can be fused. `"gelu"` is fused into the matmul as `(ttnn.UnaryOpType.GELU, False)`. Other activations (`"silu"`, `"swiglu"`, `"decomposed_gelu"`, `"gelu_tanh"`, `"quick_gelu"`) are applied after the matmul.
- **SwiGLU**: when `activation_fn="swiglu"`, the output features are doubled (`out_features * 2`) so that the output can be split into two halves -- one for the gate, one for the value -- implementing $\text{SwiGLU}(x) = (\sigma(xW_g)) \odot (xW_v)$.

### Matmul Configuration

All linear layers use `ttnn.experimental.minimal_matmul` with a shape-specific configuration:

```python
def forward(self, x, compute_kernel_config=None, dtype=None,
            default_block_size=None):
    M, K, N = x.padded_shape[-2], x.padded_shape[-1], self.weight.data.padded_shape[-1]
    core_grid = self.mesh_device.compute_with_storage_grid_size()
    matmul_config = get_matmul_config(M, K, N, core_grid, default_block_size)

    output = ttnn.experimental.minimal_matmul(
        input_tensor=x,
        weight_tensor=self.weight.data,
        bias_tensor=self.bias.data if self.bias is not None else None,
        config=matmul_config,
        fused_activation=self.fused_activation_fn,
        compute_kernel_config=compute_kernel_config or self.compute_config,
        dtype=dtype,
    )
    return _apply_activation_fn(output, self.activation_fn)
```

The `get_matmul_config` utility (from `utils/matmul.py`) determines optimal blocking parameters based on the `M`, `K`, `N` dimensions and the available compute grid. The `minimal_matmul` op is a lower-level matmul implementation that gives the framework more direct control over tiling and blocking than `ttnn.linear`.

### Compute Configuration

All linear layers initialize a compute kernel config with:
- **HiFi2 math fidelity** for bfloat16, **HiFi4** for float32
- **Approximate mode disabled** (`math_approx_mode=False`)
- **FP32 destination accumulation** (`fp32_dest_acc_en=True`)
- **Packer L1 accumulation** (`packer_l1_acc=True`)

```python
MATH_FIDELITY = {
    ttnn.bfloat16: ttnn.MathFidelity.HiFi2,
    ttnn.float32: ttnn.MathFidelity.HiFi4,
}

self.compute_config = ttnn.init_device_compute_kernel_config(
    mesh_device.arch(),
    math_fidelity=MATH_FIDELITY[dtype],
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
```

This configuration achieves a balance between numerical accuracy (FP32 accumulation prevents bfloat16 rounding from degrading output quality in deep transformer blocks) and performance (HiFi2 is fast enough for bfloat16 inference).

---

## ColParallelLinear

`ColParallelLinear` implements column-parallel weight sharding following the Megatron-LM pattern:

### Weight Sharding

```python
class ColParallelLinear(Module):
    def __init__(self, in_features, out_features, bias=True,
                 activation_fn=None, dtype=ttnn.bfloat16,
                 mesh_device=None, mesh_axis=0, fsdp_mesh_axis=None,
                 ccl_manager=None, chunks=None):
        ...
        self.weight = Parameter(
            total_shape=[in_features, out_features],
            mesh_axes=[fsdp_mesh_axis, mesh_axis],
            device=mesh_device, dtype=dtype,
        )
        self.bias = Parameter(
            total_shape=[1, out_features],
            mesh_axes=[None, mesh_axis],
            device=mesh_device, dtype=dtype,
        ) if bias else None
```

The `mesh_axes` parameter tells `Parameter` how to shard the tensor across mesh devices:

- **Weight `mesh_axes=[fsdp_mesh_axis, mesh_axis]`**: The first dimension (input features, $K$) is sharded along the FSDP axis (if enabled), and the second dimension (output features, $N$) is sharded along the TP axis. When `fsdp_mesh_axis=None`, the input dimension is replicated.
- **Bias `mesh_axes=[None, mesh_axis]`**: The bias is sharded only along the output dimension (TP axis).

For a concrete example with `in_features=4096`, `out_features=16384`, `mesh_axis=1`, and 4 devices on axis 1:
- Each device stores a weight shard of shape `[4096, 4096]` (16384 / 4 = 4096).
- Each device stores a bias shard of shape `[1, 4096]`.

### FSDP Weight Gathering

When `fsdp_mesh_axis` is set, the weight's input dimension is also sharded -- reducing per-device memory but requiring an all-gather before the matmul:

```python
def forward(self, x, compute_kernel_config=None, default_block_size=None):
    if self.fsdp_mesh_axis is not None and \
       self.mesh_device.shape[self.fsdp_mesh_axis] > 1:
        unsqueezed_weight = ttnn.unsqueeze_to_4D(self.weight.data)
        weight = self.ccl_manager.all_gather_persistent_buffer(
            unsqueezed_weight, dim=2, mesh_axis=self.fsdp_mesh_axis)
        weight = ttnn.reshape(weight, (weight.shape[-2], weight.shape[-1]))
    else:
        weight = self.weight.data

    M, K, N = x.padded_shape[-2], x.padded_shape[-1], weight.padded_shape[-1]
    ...
```

This FSDP pattern is used when the model's weights are too large to fit on each device even with TP sharding. The weight is further partitioned along the SP axis, and gathered just-in-time before the matmul. The `all_gather_persistent_buffer` call uses cached ping-pong buffers to avoid repeated allocation.

### Data Flow

The `ColParallelLinear` forward pass:

1. **Input**: replicated tensor $x$ of shape `[batch, seq, K]` present on all devices.
2. **Weight**: each device $i$ holds $W_i$ of shape `[K, N/T]` (or `[K/S, N/T]` with FSDP, gathered to `[K, N/T]`).
3. **Matmul**: each device computes $y_i = x \cdot W_i$, producing `[batch, seq, N/T]`.
4. **Output**: column-fractured -- each device holds a different slice of the output feature dimension.

No CCL operation is needed in `ColParallelLinear.forward()` itself. The fractured output is consumed directly by the next layer (e.g., a `RowParallelLinear` or an attention computation that operates per-head).

### Chunked Output (minimal_matmul_split)

When `chunks` is set (used for attention QKV projections that produce Q, K, V simultaneously), `ColParallelLinear` returns a list of tensors split along the output dimension:

```python
if self.chunks is not None:
    outputs = ttnn.experimental.minimal_matmul_split(
        x, weight, chunks=self.chunks, dim=-1,
        bias_tensor=self.bias.data if self.bias is not None else None,
        fused_activation=self.fused_activation_fn,
        compute_kernel_config=..., config=matmul_config,
    )
    return [_apply_activation_fn(o, self.activation_fn) for o in outputs]
```

This is more efficient than doing a single large matmul and then splitting, because `minimal_matmul_split` can optimize the split at the kernel level.

### SwiGLU Weight Reshaping

When `activation_fn="swiglu"`, the `_prepare_torch_state` method reshapes the weight to interleave the gate and value projections across devices:

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
        weight = weight.transpose(0, 1)  # Standard transpose
        if self.activation_fn == "swiglu":
            weight = permute_for_swiglu(weight)
        state["weight"] = weight
```

This reshaping ensures that when the weight is column-sharded across $T$ devices, each device gets a balanced mix of gate and value parameters. Without this permutation, devices 0 through $T/2-1$ would get all gate parameters and devices $T/2$ through $T-1$ would get all value parameters, leading to an imbalanced workload.

---

## RowParallelLinear

`RowParallelLinear` implements the complementary row-parallel pattern:

### Weight Sharding

```python
class RowParallelLinear(Module):
    def __init__(self, in_features, out_features, bias=True,
                 dtype=ttnn.bfloat16, mesh_device=None, mesh_axis=0,
                 fsdp_mesh_axis=None, ccl_manager=None):
        ...
        ndev = self.mesh_device.shape[self.mesh_axis] if self.mesh_axis is not None else 1

        self.weight = Parameter(
            total_shape=[in_features, out_features],
            mesh_axes=[mesh_axis, fsdp_mesh_axis],
            device=mesh_device, dtype=dtype,
        )
        self.bias = Parameter(
            total_shape=[1, out_features * ndev],
            mesh_axes=[None, mesh_axis],
            device=mesh_device, dtype=dtype,
        ) if bias else None
```

Key differences from `ColParallelLinear`:
- **`mesh_axes=[mesh_axis, fsdp_mesh_axis]`**: The input dimension ($K$) is sharded along the TP axis, and the output dimension ($N$) is sharded along the FSDP axis (if set). This is the transpose of `ColParallelLinear`'s sharding.
- **Bias total shape is `[1, out_features * ndev]`**: The bias is padded with zeros so that only device 0 holds the actual bias, and the reduce-scatter correctly accumulates it. This avoids double-counting the bias across devices.

### Bias Preparation

```python
def _prepare_torch_state(self, state):
    if "weight" in state:
        state["weight"] = state["weight"].transpose(0, 1)

    bias = state.pop("bias", None)
    if bias is not None:
        bias = bias.reshape(1, -1)
        if self._mesh_axis_size > 1:
            # Pad with zeros so only device 0 contributes bias
            zero_bias = torch.zeros(1, bias.shape[1] * (self._mesh_axis_size - 1))
            bias = torch.cat([bias, zero_bias], dim=-1)
        state["bias"] = bias
```

The bias is expanded to `[1, N * T]` by appending `(T-1)` copies of zero vectors. When this expanded bias is column-sharded across $T$ devices via `mesh_axes=[None, mesh_axis]`, device 0 gets the real bias `[1, N]` and devices 1 through $T-1$ get zeros `[1, N]`. After the reduce-scatter sums partial matmul results, only device 0's bias contributes.

### Data Flow with Reduce-Scatter

```python
def forward(self, x, *, compute_kernel_config=None,
            use_persistent_buffer=True, default_block_size=None):
    # FSDP gather if needed (same pattern as ColParallelLinear)
    if self.fsdp_mesh_axis is not None and ...:
        weight = self.ccl_manager.all_gather_persistent_buffer(...)
    else:
        weight = self.weight.data

    # Local matmul: each device has [K/T, N] weight and [batch, seq, K/T] input
    output = ttnn.experimental.minimal_matmul(
        input_tensor=x, weight_tensor=weight,
        bias_tensor=self.bias.data if self.bias is not None else None,
        config=matmul_config, ...)

    # Reduce-scatter: sum partial results across TP axis
    if self._mesh_axis_size > 1:
        needs_reshape = len(output.shape) <= 3
        if needs_reshape:
            output = ttnn.unsqueeze(output, 0)

        output = self.ccl_manager.reduce_scatter(
            output, dim=3, mesh_axis=self.mesh_axis,
            use_persistent_buffer=use_persistent_buffer)

        if needs_reshape:
            output = ttnn.squeeze(output, 0)

    return output
```

The complete flow:

1. **Input**: column-fractured $x_i$ of shape `[batch, seq, K/T]` on each device $i$ (output from a preceding `ColParallelLinear`).
2. **Weight**: each device holds $W_i$ of shape `[K/T, N]`.
3. **Local matmul**: each device computes partial result $y_i = x_i \cdot W_i$, producing `[batch, seq, N]`.
4. **Bias**: only device 0 adds the real bias; others add zeros.
5. **Reduce-scatter**: sums $\sum_i y_i$ across devices and scatters the result, leaving each device with `[batch, seq, N/T]` (if scattering on dim 3) or `[batch, seq, N]` (if using all-reduce).

The reduce-scatter along `dim=3` (the feature dimension) means each device ends up with a reduced slice of the output. Note that the `dim=3` here operates on the 4D padded tensor (after `unsqueeze` if needed).

---

## Megatron-Style Parallelism in Practice

### Feed-Forward Block

The `ParallelFeedForward` in `models/tt_dit/layers/feedforward.py` demonstrates the canonical pairing:

```python
# models/tt_dit/layers/feedforward.py

class ParallelFeedForward(Module):
    def __init__(self, dim, dim_out=None, mult=4, activation_fn="gelu",
                 inner_dim=None, bias=True, mesh_device=None,
                 mesh_axis=0, fsdp_mesh_axis=None, ccl_manager=None):
        ...
        self.ff1 = ColParallelLinear(
            dim, inner_dim, bias=bias, activation_fn=activation_fn,
            mesh_device=mesh_device, mesh_axis=mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis, ccl_manager=ccl_manager)

        self.ff2 = RowParallelLinear(
            inner_dim, dim_out, bias=bias,
            mesh_device=mesh_device, mesh_axis=mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis, ccl_manager=ccl_manager)

    def forward(self, x, compute_kernel_config=None):
        ff1_out = self.ff1(x, compute_kernel_config=compute_kernel_config)
        return self.ff2(ff1_out, compute_kernel_config=compute_kernel_config)
```

Data flow through `ParallelFeedForward`:

$$
x \xrightarrow{\text{replicated}} \text{ColParallel}(x) \xrightarrow{\text{fractured}} \text{RowParallel}(\cdot) \xrightarrow{\text{reduce-scatter}} y
$$

Only **one CCL operation** (the reduce-scatter inside `RowParallelLinear`) is needed for the entire feed-forward block. The column-fractured intermediate activations flow directly from ff1 to ff2 without any communication.

### Attention Block

In the attention block (`models/tt_dit/blocks/attention.py`), the pattern is slightly different:

```python
# models/tt_dit/blocks/attention.py (simplified)

# QKV projection: ColParallelLinear
self.to_qkv = ColParallelLinear(query_dim, 3 * padded_inner_dim,
                                 mesh_axis=tp_axis, ...)

# Output projection: ColParallelLinear (not RowParallel)
self.to_out = ColParallelLinear(padded_inner_dim, out_dim,
                                mesh_axis=tp_axis, ...)
```

Here, both projections use `ColParallelLinear`. The TP reduction is done via `all_gather_persistent_buffer` after the output projection rather than `reduce_scatter` inside a `RowParallelLinear`:

```python
# In Attention.forward():
spatial = self.ccl_manager.all_gather_persistent_buffer(
    spatial, dim=2, mesh_axis=tp_axis, use_hyperparams=True)
```

This all-gather pattern is used because attention requires the full sequence dimension to be visible on each device for the SDPA computation. The output after SDPA is already split across heads (which are partitioned by TP), so an all-gather on the head dimension reconstitutes the full output.

---

## The prepare_chunked_linear_output Utility

For attention QKV projections that produce interleaved Q, K, V outputs, TT-DiT provides a weight reshaping utility:

```python
# models/tt_dit/layers/linear.py

def prepare_chunked_linear_output(state, *, prefix, device_count, chunks):
    weight_key = f"{prefix}.weight"
    bias_key = f"{prefix}.bias"

    weight = state.get(weight_key)
    if weight is not None:
        _, in_dim = weight.shape
        weight = weight.reshape([chunks, device_count, -1, in_dim])
        weight = weight.transpose(0, 1).reshape([-1, in_dim])
        state[weight_key] = weight
    # Similar for bias
```

This function rearranges the weight tensor so that when it is column-sharded across `device_count` devices, each device gets an interleaved portion of all `chunks` (e.g., Q chunk 0 + K chunk 0 + V chunk 0 on device 0, Q chunk 1 + K chunk 1 + V chunk 1 on device 1, etc.). This is the same interleaving principle as the SwiGLU permutation, generalized to arbitrary chunk counts.

---

## Comparison Table

| Aspect | Linear | ColParallelLinear | RowParallelLinear |
|---|---|---|---|
| Weight sharding | Replicated | Output dim sharded (TP) | Input dim sharded (TP) |
| `mesh_axes` | `None` (default) | `[fsdp, tp]` | `[tp, fsdp]` |
| Input expectation | Replicated | Replicated | Column-fractured |
| Output | Replicated | Column-fractured | Reduced (via reduce-scatter) |
| CCL in forward | None | None (or FSDP all-gather) | reduce_scatter (+ optional FSDP all-gather) |
| Bias handling | Standard | Column-sharded | Zero-padded, column-sharded |
| Activation support | All types | All types | None (applied after reduction) |
| Typical pairing | Standalone | Paired with RowParallel | Paired with ColParallel |

---

## Key Takeaways

1. **`ColParallelLinear` and `RowParallelLinear` implement Megatron-style TP** by sharding weights along complementary dimensions and requiring only one CCL operation per Col-Row pair.
2. **FSDP support** is orthogonal to TP -- weights can be further sharded along the SP axis and gathered just-in-time, trading communication for memory savings.
3. **`mesh_axes` on `Parameter`** is the mechanism that drives weight distribution -- it tells `ttnn.from_torch` how to shard the tensor across the 2D mesh, replacing the need for explicit `ShardTensor2dMesh` calls.
4. **`_prepare_torch_state` handles all weight transformations** -- transposition, SwiGLU interleaving, chunked output reshaping -- at load time, so the forward pass operates on pre-shaped device tensors.
5. **`ttnn.experimental.minimal_matmul`** is used instead of `ttnn.linear` for all linear operations, giving TT-DiT finer control over blocking configuration and kernel selection.

---

**Next:** [`mapping_to_symbiote.md`](./mapping_to_symbiote.md)
