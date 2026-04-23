# Normalization Layers

## Prerequisites

- [Chapter 1 -- Module and Parameter](../ch1_architecture_overview/module_and_parameter.md): understanding of `Parameter` with `mesh_axes` and `_prepare_torch_state`.
- [Chapter 2 -- CCLManager](../ch2_parallelism_and_ccl/ccl_manager.md): understanding of `all_gather_persistent_buffer` and `reduce_scatter_persistent_buffer`.
- [Chapter 1 -- Comparison with TTNNModule](../ch1_architecture_overview/comparison_with_ttnnmodule.md): understanding of `TTNNModule` lifecycle (`preprocess_weights_impl`, `move_weights_to_device_impl`, `forward`).

---

## Overview

TT-DiT implements five normalization layers in `models/tt_dit/layers/normalization.py`. They divide into two tiers:

1. **Single-device norms** (`RMSNorm`, `LayerNorm`, `GroupNorm`) -- operate on activations that are fully present on each device. Wrap stable `ttnn` APIs.
2. **Distributed norms** (`DistributedRMSNorm`, `DistributedLayerNorm`) -- operate on activations that are sharded across devices along the reduction dimension. Require cross-device communication via CCL and use `ttnn.experimental.*` APIs.

TT-Symbiote provides four normalization modules in `models/experimental/tt_symbiote/modules/normalization.py`: `TTNNLayerNorm`, `TTNNRMSNorm`, `TTNNLocalRMSNorm`, and `TTNNDistributedRMSNorm`.

---

## RMSNorm

### TT-DiT Implementation

```python
# models/tt_dit/layers/normalization.py

class RMSNorm(Module):
    def __init__(self, embedding_dim, norm_eps=1e-5,
                 norm_elementwise_affine=True, bias=True,
                 mesh_device=None, dtype=ttnn.bfloat16):
        self.weight = Parameter(total_shape=[1, embedding_dim], ...)
        self.bias = Parameter(total_shape=[1, embedding_dim], ...) if bias else None

    def forward(self, x, compute_kernel_config=None):
        return ttnn.rms_norm(x, weight=..., bias=..., epsilon=self.norm_eps,
                             compute_kernel_config=compute_kernel_config)
```

RMSNorm computes:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma + \beta$$

where $\gamma$ is the learnable weight and $\beta$ is the optional bias.

Key details:
- **Constraint**: `embedding_dim` must be divisible by 32 (tile size).
- **Weight shape**: `[1, embedding_dim]` -- pre-unsqueezed in `_prepare_torch_state` so it broadcasts over the sequence dimension without runtime reshaping.
- **Bias support**: optional, controlled by the `bias` parameter. Most DiT models use bias-free RMSNorm.
- **Compute kernel config**: optionally passed through to control math fidelity and accumulation precision.

### TT-Symbiote Equivalent: `TTNNRMSNorm`

```python
# models/experimental/tt_symbiote/modules/normalization.py

class TTNNRMSNorm(TTNNModule):
    def preprocess_weights_impl(self):
        self.tt_weight = ttnn.from_torch(
            self.torch_layer.weight.unsqueeze(0).expand(32, -1),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def forward(self, x):
        return ttnn.rms_norm(x, weight=self.tt_weight,
                             epsilon=self.torch_layer.variance_epsilon)
```

### Comparison

| Aspect | TT-DiT `RMSNorm` | TT-Symbiote `TTNNRMSNorm` |
|---|---|---|
| Base class | `Module` | `TTNNModule` |
| TTNN op | `ttnn.rms_norm` | `ttnn.rms_norm` |
| Weight shape | `[1, dim]` | `[32, dim]` (expanded) |
| Bias support | Yes (optional) | No |
| Compute kernel config | Configurable per-call | Not configurable |
| Weight management | `Parameter` auto-loads from state dict | Manual `preprocess_weights_impl` + `move_weights_to_device_impl` |
| Epsilon source | Constructor parameter | Reads from `torch_layer.variance_epsilon` at runtime |

**Porting note**: Both call `ttnn.rms_norm`, so the core operation is identical. The weight shape difference (`[1, dim]` vs `[32, dim]`) stems from different broadcasting strategies. TT-Symbiote expands the weight to 32 rows to match tile dimensions; TT-DiT relies on TTNN's internal broadcasting. TT-Symbiote also lacks bias support, which would need to be added for models that use it (e.g., Flux).

---

## LayerNorm

### TT-DiT Implementation

```python
# models/tt_dit/layers/normalization.py

class LayerNorm(Module):
    def __init__(self, embedding_dim, norm_eps=1e-5,
                 norm_elementwise_affine=True, bias=True,
                 mesh_device=None, use_row_major_workaround=False):
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, ...)

        shape = [embedding_dim // 32, 32] if use_row_major_workaround else [1, embedding_dim]
        layout = ttnn.ROW_MAJOR_LAYOUT if use_row_major_workaround else ttnn.TILE_LAYOUT

        self.weight = Parameter(total_shape=shape, layout=layout, ...) if norm_elementwise_affine else None
        self.bias = Parameter(total_shape=shape, layout=layout, ...) if self.use_bias else None

    def forward(self, x):
        return ttnn.layer_norm(x, weight=..., bias=..., epsilon=self.norm_eps,
                               compute_kernel_config=self.compute_kernel_config)
```

LayerNorm computes:

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$

Key details:
- **Row-major workaround** (Issue #20789): when `use_row_major_workaround=True`, weights are stored in `ROW_MAJOR_LAYOUT` with shape `[embedding_dim // 32, 32]` instead of the normal `[1, embedding_dim]` in `TILE_LAYOUT`. This is a hardware workaround for certain shapes.
- **Compute kernel config**: always `HiFi4` with `fp32_dest_acc_en=True` for maximum precision.
- **Dummy weights**: when `use_row_major_workaround=True` and `norm_elementwise_affine=False`, a dummy all-ones weight is created.

### TT-Symbiote Equivalent: `TTNNLayerNorm`

```python
# models/experimental/tt_symbiote/modules/normalization.py

class TTNNLayerNorm(TTNNModule):
    def preprocess_weights_impl(self):
        self.tt_weight = ttnn.from_torch(self.torch_layer.weight,
                                          dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.tt_bias = ttnn.from_torch(self.torch_layer.bias,
                                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def forward(self, input_tensor):
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, ...)
        return ttnn.layer_norm(input_tensor, weight=self.tt_weight, bias=self.tt_bias)
```

### Comparison

| Aspect | TT-DiT `LayerNorm` | TT-Symbiote `TTNNLayerNorm` |
|---|---|---|
| TTNN op | `ttnn.layer_norm` | `ttnn.layer_norm` |
| Compute kernel config | `HiFi4`, `fp32_dest_acc_en` | Default (not specified) |
| Row-major workaround | Supported | Not supported |
| Layout conversion | Not needed (assumed TILE) | Explicit `to_layout` guard |
| Epsilon | Constructor parameter | Not passed (uses TTNN default) |

**Porting note**: The compute kernel config difference could cause numerical divergence. TT-DiT forces `HiFi4` with `fp32_dest_acc_en` while TT-Symbiote uses TTNN defaults. For porting, the TT-Symbiote version should be extended with configurable compute kernel parameters. The row-major workaround is unlikely to be needed for new models but is relevant if porting SD3.5 VAE components.

---

## DistributedRMSNorm

### TT-DiT Implementation

`DistributedRMSNorm` normalizes activations that are sharded along the hidden dimension across multiple devices. It uses a two-phase approach:

**Phase 1 -- Compute local statistics:**
```python
stats = ttnn.experimental.wan_fused_rmsnorm_pre_allgather(
    x, dtype=ttnn.float32,
    compute_kernel_config=self.compute_kernel_config)
```

**Phase 2 -- All-gather statistics and apply norm:**
```python
stats = self.ccl_manager.all_gather_persistent_buffer(
    stats, dim=len(x.shape) - 1, mesh_axis=self.mesh_axis)

x = ttnn.experimental.wan_fused_rmsnorm_post_allgather(
    x, stats, epsilon=self.norm_eps,
    num_heads_per_device=num_heads_per_device,
    weight=self.weight.data,
    transformation_mat=trans_mat,
    rope_cos=rope_cos, rope_sin=rope_sin,
    compute_kernel_config=self.compute_kernel_config)
```

The mathematical computation is:

$$\text{RMS}_\text{global} = \sqrt{\frac{1}{d}\sum_{\text{devices}} \sum_{i} x_i^2 + \epsilon}$$

Each device computes $\sum_i x_i^2$ locally (pre-allgather), then the statistics are gathered across devices, and the normalization is applied using the global RMS value (post-allgather).

Key details:
- **Experimental ops**: uses `ttnn.experimental.wan_fused_rmsnorm_pre_allgather` and `ttnn.experimental.wan_fused_rmsnorm_post_allgather`.
- **Fused RoPE**: the post-allgather op can optionally apply rotary positional embeddings in the same kernel pass, avoiding a separate RoPE call. This is controlled by the `rope_cos`, `rope_sin`, and `trans_mat` parameters.
- **Weight sharding**: the weight parameter is sharded along `mesh_axis` using `mesh_axes=[None, mesh_axis]`, so each device holds the weight slice corresponding to its activation shard.
- **No bias**: bias is explicitly unsupported (`assert not bias`).
- **Statistics dtype**: always `float32` for the statistics computation to preserve precision.

### TT-Symbiote Equivalent: `TTNNDistributedRMSNorm`

```python
# models/experimental/tt_symbiote/modules/normalization.py

class TTNNDistributedRMSNorm(TTNNModule):
    def forward(self, inp):
        tt_stats = ttnn.rms_norm_pre_all_gather(inp, dtype=ttnn.bfloat16)
        tt_stats = ttnn.all_gather(tt_stats, dim=-1, num_links=1,
                                    topology=ttnn.Topology.Ring)
        tt_out = ttnn.rms_norm_post_all_gather(
            inp, tt_stats, epsilon=eps, weight=self.weight_distributed)
        return tt_out
```

### Comparison

| Aspect | TT-DiT `DistributedRMSNorm` | TT-Symbiote `TTNNDistributedRMSNorm` |
|---|---|---|
| Pre-gather op | `ttnn.experimental.wan_fused_rmsnorm_pre_allgather` | `ttnn.rms_norm_pre_all_gather` |
| Post-gather op | `ttnn.experimental.wan_fused_rmsnorm_post_allgather` | `ttnn.rms_norm_post_all_gather` |
| API namespace | `ttnn.experimental` | `ttnn` (stable) |
| Statistics dtype | `float32` | `bfloat16` |
| All-gather | `ccl_manager.all_gather_persistent_buffer` | `ttnn.all_gather` with Ring topology |
| Fused RoPE | Supported via `rope_cos`/`rope_sin` params | Not supported |
| Weight distribution | `Parameter` with `mesh_axes` sharding | Manual `ttnn.as_tensor` + `ShardTensor2dMesh` |
| Bias | Not supported | Not supported |
| Device restriction | None (mesh-aware) | `@run_on_devices(DeviceArch.T3K)` |

**Porting notes**:

1. **Different TTNN APIs**: TT-DiT uses `wan_fused_*` experimental ops while TT-Symbiote uses stable `rms_norm_pre_all_gather` / `rms_norm_post_all_gather`. The `wan_fused_*` variants are optimized for the Wan2.2 model and include fused RoPE support. For porting DiT models that use fused RoPE in the norm pass, the experimental ops would need to be wrapped in a new TT-Symbiote module.

2. **Statistics precision**: TT-DiT computes statistics in `float32`; TT-Symbiote uses `bfloat16`. This may cause numerical differences in large-dimension norms.

3. **All-gather strategy**: TT-DiT uses persistent buffer caching via `ccl_manager` to avoid re-allocating communication buffers across iterations. TT-Symbiote uses plain `ttnn.all_gather` with Ring topology for trace compatibility.

---

## DistributedLayerNorm

### TT-DiT Implementation

`DistributedLayerNorm` follows the same two-phase pattern as `DistributedRMSNorm` but uses Welford's online algorithm for computing mean and variance:

**Phase 1 -- Compute local statistics:**
```python
stats = ttnn.experimental.dit_layernorm_pre_allgather(
    x, self.recip_tensor,
    compute_kernel_config=self.compute_kernel_config)
```

**Phase 2 -- All-gather and apply:**
```python
stats = self.ccl_manager.all_gather_persistent_buffer(
    stats, dim=len(x.shape) - 1, mesh_axis=self.mesh_axis)

x = ttnn.experimental.dit_layernorm_post_allgather(
    x, stats, weight=weight, bias=bias,
    epsilon=self.norm_eps,
    compute_kernel_config=self.compute_kernel_config,
    dtype=dtype)
```

Key details:
- **Welford reciprocal tensor**: the `recip_tensor` is computed once per `(mesh_device_id, width_per_device)` pair using `ttnn.create_layer_norm_reciprocals` and cached in a class-level dictionary (`_recip_tensors`). This avoids recomputing it for every `DistributedLayerNorm` instance with the same configuration.
- **Weight layout**: uses `ROW_MAJOR_LAYOUT` with shape `[embedding_dim // (32 * mesh_width), 32 * mesh_width]`, interleaved by device for correct sharding.
- **Dynamic weight/bias**: the `forward` method accepts optional `dynamic_weight` and `dynamic_bias` parameters, used for adaptive layer norm (adaLN) in transformer blocks. When dynamic parameters are provided, the module must not have static weight/bias.
- **Weight preparation**: `_prepare_torch_state` reshapes weights by interleaving tile-sized chunks across devices:
  ```python
  weight = weight.reshape(mesh_width, -1, TILE_SIZE).permute(1, 0, 2)
                 .reshape(-1, TILE_SIZE * mesh_width)
  ```

### TT-Symbiote Equivalent

**None.** TT-Symbiote does not implement a distributed LayerNorm. The closest module is `TTNNLayerNorm`, but it operates on fully replicated activations with no cross-device reduction.

**Porting requirement**: A new `TTNNDistributedLayerNorm` module would need to be created, wrapping either:
- The `ttnn.experimental.dit_layernorm_pre_allgather` / `dit_layernorm_post_allgather` ops directly (dependency on experimental APIs), or
- A decomposed implementation using `ttnn.layer_norm` parts with manual statistics aggregation.

---

## GroupNorm

### TT-DiT Implementation

```python
# models/tt_dit/layers/normalization.py

class GroupNorm(Module):
    def __init__(self, num_channels, num_groups, *, eps=1e-5,
                 mesh_device, mesh_axis=None, core_grid=None):
        # Adjusts channels/groups for data-parallel sharding
        self.num_channels = num_channels // self.num_devices
        self.num_groups = num_groups // self.num_devices
        self.num_virtual_cols = ttnn.operations.normalization.dram_group_norm_virtual_columns(...)

        self.weight = Parameter(total_shape=[...], layout=ttnn.ROW_MAJOR_LAYOUT, ...)
        self.bias = Parameter(total_shape=[...], layout=ttnn.ROW_MAJOR_LAYOUT, ...)
        self.mask = Parameter(total_shape=[1, num_groups, 32, 32 * block_wt], ...)

    def forward(self, x, num_out_blocks=-1):
        x = x.reshape([batch_size, 1, width * height, channels])
        x = ttnn.group_norm(x, weight=..., bias=..., input_mask=self.mask.data,
                             num_groups=self.num_groups, epsilon=self.eps,
                             core_grid=self.core_grid, inplace=False,
                             num_out_blocks=num_out_blocks,
                             output_layout=ttnn.TILE_LAYOUT)
        x = x.reshape([batch_size, height, width, channels])
        return x
```

GroupNorm partitions channels into groups and normalizes within each group:

$$\text{GroupNorm}(x_{c}) = \frac{x_{c} - \mu_{g(c)}}{\sqrt{\sigma^2_{g(c)} + \epsilon}} \cdot \gamma_{c} + \beta_{c}$$

where $g(c)$ is the group assignment for channel $c$.

Key details:
- **Data-parallel support**: when `mesh_axis` is set, `num_channels` and `num_groups` are divided by the number of devices along that axis.
- **Input mask**: `ttnn.group_norm` requires an explicit `input_mask` parameter generated by `ttnn.create_group_norm_input_mask`. This mask defines the channel-to-group mapping.
- **Weight preparation**: weights are reformatted using `ttnn.create_group_norm_weight_bias_rm` for row-major compatibility with the DRAM group norm kernel.
- **Reshape pattern**: input is reshaped from `[B, H, W, C]` to `[B, 1, H*W, C]` before the group norm call, then reshaped back.
- **from_torch factory**: provides a `from_torch(torch.nn.GroupNorm, ...)` class method for direct conversion.

### TT-Symbiote Equivalent

**None.** TT-Symbiote does not implement a GroupNorm module. The `ttnn.group_norm` op is available in TTNN but is not wrapped by any TT-Symbiote module.

**Porting requirement**: A new `TTNNGroupNorm` module would need to wrap `ttnn.group_norm` with the mask generation and weight preparation logic. The mask and weight preparation are the main complexity -- they require TTNN utility functions (`create_group_norm_input_mask`, `create_group_norm_weight_bias_rm`, `dram_group_norm_virtual_columns`, `find_max_tile_span`) that are part of `ttnn.operations.normalization`.

---

## Summary Table

| Layer | TT-DiT | TT-Symbiote | Core TTNN Op | Status |
|---|---|---|---|---|
| RMSNorm | `RMSNorm` | `TTNNRMSNorm` | `ttnn.rms_norm` | Compatible (minor weight shape difference) |
| LayerNorm | `LayerNorm` | `TTNNLayerNorm` | `ttnn.layer_norm` | Compatible (missing compute config in Symbiote) |
| DistributedRMSNorm | `DistributedRMSNorm` | `TTNNDistributedRMSNorm` | `ttnn.experimental.wan_fused_rmsnorm_*` vs `ttnn.rms_norm_pre/post_all_gather` | Different APIs; both functional |
| DistributedLayerNorm | `DistributedLayerNorm` | -- | `ttnn.experimental.dit_layernorm_*` | No TT-Symbiote equivalent |
| GroupNorm | `GroupNorm` | -- | `ttnn.group_norm` | No TT-Symbiote equivalent |
| Local RMSNorm | -- | `TTNNLocalRMSNorm` | `ttnn.rms_norm` | TT-Symbiote only (per-head norms for Gemma4) |

---

## Key Takeaways

1. **Single-device norms are directly portable**: `RMSNorm` and `LayerNorm` both map to `ttnn.rms_norm` / `ttnn.layer_norm` in both frameworks. The differences are in weight management and compute configuration, not the underlying TTNN ops.

2. **Distributed norms use different TTNN APIs**: TT-DiT's `DistributedRMSNorm` uses `wan_fused_*` experimental ops; TT-Symbiote's `TTNNDistributedRMSNorm` uses stable `rms_norm_pre/post_all_gather`. The fused RoPE support in TT-DiT's variant is a performance optimization not available in TT-Symbiote.

3. **DistributedLayerNorm and GroupNorm are gaps**: These two layers have no TT-Symbiote equivalent and would require new module implementations for porting. DistributedLayerNorm is used in SD3.5 and Flux transformer blocks; GroupNorm is used in all VAE models.

4. **Compute precision differs**: TT-DiT consistently uses `HiFi4` with `fp32_dest_acc_en=True` for normalization. TT-Symbiote uses TTNN defaults. For numerical fidelity during porting, the compute kernel config should be explicitly set.

5. **Weight preparation patterns diverge**: TT-DiT uses `_prepare_torch_state` to transform PyTorch state dict tensors before loading. TT-Symbiote uses `preprocess_weights_impl` and `move_weights_to_device_impl`. Both achieve the same end (TTNN-compatible weight tensors on device) but the lifecycle hooks differ.

---

**Next:** [`ttnn_experimental_ops.md`](./ttnn_experimental_ops.md)
