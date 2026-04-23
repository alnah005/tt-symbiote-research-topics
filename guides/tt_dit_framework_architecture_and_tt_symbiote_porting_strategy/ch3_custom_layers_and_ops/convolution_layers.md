# Convolution Layers

## Prerequisites

- [Chapter 1 -- Module and Parameter](../ch1_architecture_overview/module_and_parameter.md): understanding of `Parameter` with `mesh_axes` for tensor-parallel weight sharding.
- [Chapter 2 -- CCLManager](../ch2_parallelism_and_ccl/ccl_manager.md): understanding of `reduce_scatter_persistent_buffer` and `all_gather_persistent_buffer`.
- [TTNN Experimental Ops](./ttnn_experimental_ops.md): understanding of `ttnn.experimental.conv3d` and `ttnn.experimental.neighbor_pad_async`.

---

## Overview

Convolution layers are used exclusively in VAE (Variational Autoencoder) components of diffusion models. TT-DiT implements two convolution layers:

1. **`Conv2d`** (`layers/conv2d.py`) -- wraps `ttnn.conv2d` with support for input-channel and output-channel tensor parallelism. Used in image VAEs (SD3.5, Flux).
2. **`ContextParallelConv3d`** (`layers/conv3d.py`) -- wraps `ttnn.experimental.conv3d` with context parallelism over the temporal dimension. Used in video VAEs (Mochi, Wan2.1).

TT-Symbiote provides `TTNNConv2dNHWC` (`modules/conv.py`) and several fused variants (`TTNNConv2dBNNHWC`, `TTNNConv2dBNActivationNHWC`). There is no TT-Symbiote Conv3d.

---

## TT-DiT Conv2d

### Architecture

```python
# models/tt_dit/layers/conv2d.py

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, *,
                 kernel_size, stride=1, padding=0, dilation=1,
                 mesh_device, in_mesh_axis=None, out_mesh_axis=None,
                 ccl_manager=None):
```

The layer supports three parallelism modes:

| Mode | `in_mesh_axis` | `out_mesh_axis` | Input | Output |
|---|---|---|---|---|
| No parallelism | `None` | `None` | Replicated | Replicated |
| Input-channel parallel | Set | `None` | Sharded on channels | Sharded on channels |
| Output-channel parallel | `None` | Set | Replicated or sharded | Sharded on channels |

### Weight Layout

Weights are stored with shape `[out_channels, in_channels, kernel_h, kernel_w]` in `ROW_MAJOR_LAYOUT` on the host, distributed across devices via `mesh_axes`:

```python
self.weight = Parameter(
    total_shape=[out_channels, in_channels, *kernel_size],
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=mesh_device,
    mesh_axes=[out_mesh_axis, in_mesh_axis, None, None],
    on_host=True,  # Kept on host, transferred per forward call
)
```

The bias has a special multi-device layout:
```python
self.bias = Parameter(
    total_shape=[in_mesh_axis_size, 1, 1, out_channels],
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=mesh_device,
    mesh_axes=[in_mesh_axis, None, None, out_mesh_axis],
    on_host=True,
)
```

The bias shape `[in_mesh_axis_size, 1, 1, out_channels]` is structured so that only the first device along `in_mesh_axis` holds the actual bias values; all other devices hold zeros. This ensures that after `reduce_scatter`, the bias is applied exactly once.

### Slice Configuration

TT-DiT's `Conv2d` uses a `ttnn.Conv2dSliceConfig` to control how the convolution is tiled across DRAM:

```python
slice_config = ttnn.Conv2dSliceConfig(
    num_slices=self.slice_params[(h, w, in_channels, out_channels)],
    slice_type=ttnn.Conv2dDRAMSliceWidth,
)
```

The `slice_params` dictionary maps `(height, width, in_channels, out_channels)` tuples to optimal slice counts for different mesh configurations. This is a performance tuning mechanism -- incorrect slice counts cause out-of-memory errors for large convolutions.

### Forward Pass

```python
def forward(self, x):
    b, h, w, c = x.shape  # NHWC format

    # Optional all-gather for output-channel parallelism
    if self.out_mesh_axis_size != 1 and c == self.in_channels // self.out_mesh_axis_size:
        x = vae_all_gather(self.ccl_manager, x, cluster_axis=self.out_mesh_axis)

    x, [out_height, out_width] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=self.weight.data,
        bias_tensor=self.bias.data,
        in_channels=self.in_channels // self.in_mesh_axis_size,
        out_channels=self.weight.data.shape[0],
        device=self.mesh_device,
        kernel_size=self.kernel_size,
        stride=self.stride,
        padding=self.padding,
        batch_size=b,
        input_height=h,
        input_width=w,
        conv_config=ttnn.Conv2dConfig(act_block_h_override=32),
        compute_config=self.compute_config,
        slice_config=slice_config,
        return_output_dim=True,
    )

    x = ttnn.reshape(x, (b, out_height, out_width, -1))

    # Reduce-scatter for input-channel parallelism
    if self.in_mesh_axis is not None:
        x = self.ccl_manager.reduce_scatter_persistent_buffer(
            x, dim=-1, mesh_axis=self.in_mesh_axis)

    return x
```

Key observations:
1. **Input format**: NHWC (batch, height, width, channels) -- this is TTNN's native format.
2. **Compute config**: `HiFi4` with `fp32_dest_acc_en=True` for accumulation precision.
3. **act_block_h_override=32**: forces the activation block height to 32, which is a tile-alignment requirement.
4. **reduce_scatter after conv**: when using input-channel parallelism, each device computes a partial convolution result. The `reduce_scatter` sums partial results across devices.

### Error Handling

The forward pass includes a specific OOM error handler that re-raises `RuntimeError` with the convolution shape parameters, making debugging easier:

```python
except RuntimeError as e:
    m = re.search(r"Out of Memory: (.*)", str(e))
    if m:
        raise RuntimeError(
            f"conv2d out of memory with (h, w, in_c, out_c) = "
            f"{(h, w, self.in_channels, self.out_channels)} ...")
```

---

## TT-Symbiote TTNNConv2dNHWC

### Architecture

```python
# models/experimental/tt_symbiote/modules/conv.py

class TTNNConv2dNHWC(TTNNModule):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation,
                 groups=1, slice_config=None):
```

### Weight Preprocessing

TT-Symbiote's conv uses the `tt_cnn` builder abstraction to prepare weights:

```python
def preprocess_weights_impl(self):
    self.tt_weight, self.tt_bias = Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(
        self.torch_layer.conv.weight, self.torch_layer.conv.bias)
```

The `Conv2dConfiguration` class (from `models/tt_cnn/tt/builder.py`) handles weight layout transformation, including channel padding and tiling to match TTNN's expected format.

### Forward Pass

```python
def forward(self, input_tensor, reshape_output=True):
    batch_size, input_height, input_width, _ = input_tensor.shape

    config = Conv2dConfiguration(
        input_height=input_height, input_width=input_width,
        in_channels=self.in_channels, out_channels=self.out_channels,
        batch_size=batch_size, kernel_size=self.kernel_size,
        stride=self.stride, padding=self.padding,
        groups=self.groups, dilation=self.dilation,
        weight=self.tt_weight, bias=self.tt_bias,
        slice_strategy=self.slice_config,
    )
    layer = TtConv2d(config, input_tensor.device())

    out, h_w = layer(input_tensor, return_output_dim=reshape_output)
    out = self.reshape(out, [batch_size, h_w[0], h_w[1], -1])
    return out
```

Key observations:
1. **Lazy construction**: the `TtConv2d` object is created on each forward call (with caching via `CACHED_TTCNN` dict).
2. **Builder pattern**: uses `Conv2dConfiguration` + `TtConv2d` instead of calling `ttnn.conv2d` directly.
3. **No tensor parallelism**: there is no `mesh_axis` support. Weights are replicated across all devices.

### Fused Variants

TT-Symbiote provides additional fused convolution modules:

- **`TTNNConv2dBNNHWC`**: fuses Conv2d + BatchNorm2d by folding BN parameters into convolution weights at preprocessing time:
  ```python
  weight, bias = fold_batch_norm2d_into_conv2d(
      conv.weight, conv.bias, bn.weight, bn.bias,
      bn.running_mean, bn.running_var, bn.eps)
  ```
- **`TTNNConv2dBNActivationNHWC`**: fuses Conv2d + BatchNorm2d + ReLU. The ReLU is applied as a fused activation via `ttnn.UnaryOpType.RELU` in the `Conv2dConfiguration`.

TT-DiT has no fused Conv+BN or Conv+BN+ReLU variants because diffusion models do not use BatchNorm.

---

## Comparison: Conv2d

| Aspect | TT-DiT `Conv2d` | TT-Symbiote `TTNNConv2dNHWC` |
|---|---|---|
| Underlying TTNN op | `ttnn.conv2d` (direct) | `ttnn.conv2d` (via `TtConv2d` builder) |
| Tensor parallelism | Input-channel and output-channel | None |
| CCL operations | `reduce_scatter`, `all_gather` | None |
| Weight location | Host (transferred per call) | Device (cached) |
| Slice configuration | Per-shape lookup table | Optional `slice_config` parameter |
| Fused BN | Not supported | `TTNNConv2dBNNHWC`, `TTNNConv2dBNActivationNHWC` |
| Dilation support | Asserts `dilation == (1, 1)` | Supported (passthrough to TTNN) |
| Groups support | Not supported (implicit `groups=1`) | Supported |
| `from_torch` | `Conv2d.from_torch(nn.Conv2d, ...)` | `TTNNConv2dNHWC.from_torch(nn.Conv2d, ...)` |
| Caching | No caching | `CACHED_TTCNN` hash-based cache |

**Porting considerations**:

1. **Tensor parallelism is the main gap**: TT-Symbiote's Conv2d has no concept of distributing channels across devices. For VAE porting, this limits execution to single-device or requires implementing TP manually.

2. **Weight location**: TT-DiT keeps weights on host (`on_host=True`) and transfers them per forward call. This is a deliberate choice for VAE models where the encoder and decoder share device memory and cannot both have weights resident simultaneously. TT-Symbiote caches weights on device.

3. **The builder abstraction adds indirection**: TT-Symbiote's `Conv2dConfiguration` + `TtConv2d` pattern adds a layer of abstraction over `ttnn.conv2d`. This is convenient for ResNet-style models but may obscure performance tuning opportunities that TT-DiT's direct `ttnn.conv2d` call exposes.

---

## TT-DiT ContextParallelConv3d

### Architecture

```python
# models/tt_dit/layers/conv3d.py

class ContextParallelConv3d(Module):
    def __init__(self, in_channels, out_channels, *,
                 kernel_size, stride=(1,1,1), bias=True,
                 causal=True, context_parallel=True, groups=1,
                 padding_mode, mesh_device, parallel_config, ccl_manager):
```

This layer performs 3D convolution on video data with the temporal dimension distributed across devices (context parallelism).

### Causal Temporal Padding

Video diffusion models require causal convolution -- the output at time $t$ depends only on inputs at times $\leq t$. This is implemented by front-padding the temporal dimension:

```python
def _causal_pad_input(self, x_NTHWC, pad_front, pad_back=0):
    if self.padding_mode == "zeros":
        x_pad = ttnn.pad(x_NTHWC, (0, 0, 0, 0, pad_front, pad_back), value=0.0)
    elif self.padding_mode == "replicate":
        front_slice = x_NTHWC[:, 0:1, :, :, :]
        x_pad = ttnn.concat([front_slice] * pad_front + [x_NTHWC], dim=1)
```

For a kernel of temporal size $k_t$, the front padding is $k_t - 1$ frames.

### Multi-Device Temporal Padding

When context parallelism is active (`parallel_config.time_parallel.factor > 1`), temporal frames are distributed across devices. Padding requires exchanging halo frames between adjacent devices using `ttnn.experimental.neighbor_pad_async` (wrapped by `vae_neighbor_pad`):

```python
if self.parallel_config.time_parallel.factor > 1:
    halo_tensor = vae_neighbor_pad(
        self.ccl_manager, halo_tensor,
        cluster_axis=self.parallel_config.time_parallel.mesh_axis,
        dim=0, padding_left=2, padding_right=0,
        padding_mode="replicate")
```

### Weight Layout

Conv3d weights are preprocessed into a specific layout for the `ttnn.experimental.conv3d` kernel:

```python
def _prepare_torch_state(self, state):
    weight = state["weight"]
    # Original: [out_c, in_c, kd, kh, kw]
    weight = weight.permute(2, 3, 4, 1, 0)  # -> [kd, kh, kw, in_c, out_c]

    # Pad in_channels to alignment boundary (16)
    if c_in % 16 != 0:
        weight = F.pad(weight, (0, 0, 0, 16 - c_in % 16))

    # Reshape for C_in_block striding
    weight = weight.reshape(kd, hk, kw, num_c_in_blocks, c_in_block, out_c)
    weight = weight.permute(3, 0, 1, 2, 4, 5)  # num_c_in_blocks first
    weight = weight.reshape(-1, out_c)

    state["weight"] = weight
```

The final weight shape is `[kd * kh * kw * in_channels_padded, out_channels]`, flattened for the `conv3d` kernel's expected layout. The `C_in_block` dimension is placed first so the kernel can stride over input channel blocks.

### Blocking Configuration

The `Conv3dConfig` controls spatial and channel blocking:

```python
ttnn.Conv3dConfig(
    weights_dtype=ttnn.bfloat16,
    output_layout=ttnn.ROW_MAJOR_LAYOUT,
    T_out_block=1,       # Temporal output blocking
    W_out_block=16,      # Width output blocking
    H_out_block=2,       # Height output blocking
    C_out_block=96,      # Output channel blocking
    C_in_block=128,      # Input channel blocking
    compute_with_storage_grid_size=grid_size,
)
```

These are tuned per input channel count:

| `in_channels` | `C_in_block` | `C_out_block` | `T_out_block` | `H_out_block` | `W_out_block` |
|---|---|---|---|---|---|
| 768 | 128 | 96 | 1 | 2 | 16 |
| 512 | 128 | 128 | 1 | 8 | 4 |
| 256 | 128 | 128 | 4 | 4 | 2 |
| 128 | 128 | 128 | 1 | 2 | 16 |

### Forward Pass

```python
def forward(self, x_NTHWC):
    # 1. Temporal padding (single-device or multi-device)
    x_pad = self._causal_pad_input(x_NTHWC, pad_front=kernel_size[0]-1)

    # 2. Conv3d
    out = ttnn.experimental.conv3d(
        input_tensor=x_pad,
        weight_tensor=self.weight.data,
        bias_tensor=self.bias.data,
        config=self.conv_config,
        output_channels=self.out_channels,
        kernel_size=self.kernel_size,
        stride=self.stride,
        padding=self.padding,
        padding_mode=self.padding_mode,
        dtype=ttnn.bfloat16,
        groups=self.groups,
        compute_kernel_config=self.compute_kernel_config,
    )
    return out
```

---

## TT-Symbiote Conv3d

**There is no Conv3d in TT-Symbiote.** The `ttnn.experimental.conv3d` op exists in the TTNN library but no TT-Symbiote module wraps it.

### Impact on Porting

Video model VAEs (Mochi, Wan2.1/2.2) are built entirely around Conv3d layers. Porting these to TT-Symbiote requires:

1. **A new `TTNNConv3d` module**: wrapping `ttnn.experimental.conv3d` with the `TTNNModule` lifecycle (`preprocess_weights_impl`, `move_weights_to_device_impl`, `forward`).
2. **Weight preprocessing**: the complex weight permutation and padding logic from TT-DiT's `_prepare_torch_state` would need to be replicated in `preprocess_weights_impl`.
3. **Blocking configuration**: the per-shape blocking table from `get_conv3d_config` would need to be integrated.
4. **Context parallelism (optional)**: if multi-device video inference is required, the halo exchange via `neighbor_pad_async` would also need to be wrapped.

Without Conv3d support, only the transformer backbone of video models can be ported -- the VAE encoder/decoder (which converts between pixel space and latent space) remains blocked.

---

## Additional Convolution Modules in TT-Symbiote

TT-Symbiote provides several specialized convolution wrappers beyond the base `TTNNConv2dNHWC`:

### TTNNConv2dBNNHWC

Fuses `Conv2d + BatchNorm2d` by folding BN parameters into convolution weights at load time:

$$w' = w \cdot \frac{\gamma}{\sqrt{\sigma^2_\text{running} + \epsilon}}, \quad b' = (b - \mu_\text{running}) \cdot \frac{\gamma}{\sqrt{\sigma^2_\text{running} + \epsilon}} + \beta$$

This eliminates a separate BN pass during inference. Irrelevant for diffusion models (no BN) but useful for vision backbones.

### TTNNConv2dBNActivationNHWC

Extends `TTNNConv2dBNNHWC` with a fused ReLU activation via `ttnn.UnaryOpType.RELU` in the `Conv2dConfiguration`. Currently only supports ReLU (`assert isinstance(activation, nn.ReLU)`).

### TTNNConv2dNHWCInputMultipleOf16

Handles Conv2d layers where `in_channels < 16` or `in_channels` is not a multiple of 16 by zero-padding the weight tensor's input channel dimension:

```python
if conv.in_channels > 16 or conv.in_channels % 16 == 0:
    return TTNNConv2dNHWC.from_torch(conv, slice_config)
# else pad to 16 channels
conv.weight = nn.Parameter(F.pad(conv.weight, (0, 0, 0, 0, 0, (16 - in_c % 16) % 16)))
```

### TTNNBottleneck

A ResNet Bottleneck block composed of three convolutions (1x1, 3x3, 1x1) with BN and ReLU, plus a skip connection:

```python
def forward(self, x):
    identity = x
    out = self.conv1(x)   # TTNNConv2dBNActivationNHWC (1x1)
    out = self.conv2(out)  # TTNNConv2dBNActivationNHWC (3x3)
    out = self.conv3(out)  # TTNNConv2dBNNHWC (1x1)
    out = out + identity
    out = self.relu(out)
    return out
```

### TTNNPatchEmbedding

Implements ViT patch embedding using `ttnn.fold` + `ttnn.linear` instead of `ttnn.conv2d`. This is conceptually a convolution with `kernel_size == stride == patch_size`, decomposed into a fold (extracting patches) and a linear projection:

```python
def forward(self, pixel_values):
    pixel_values = ttnn.reshape(pixel_values, (...))
    folded = ttnn.fold(pixel_values, stride_h, stride_w)
    output = ttnn.linear(folded, self.ttnn_weight, bias=self.ttnn_bias)
    return output
```

TT-DiT's `PatchEmbed` uses a similar decomposition (`_unfold_conv2d`) but with `ttnn.reshape` + `ttnn.permute` instead of `ttnn.fold`.

---

## Key Takeaways

1. **Conv2d is portable with limitations**: Both frameworks wrap `ttnn.conv2d`, but TT-DiT adds tensor parallelism (input-channel and output-channel sharding) that TT-Symbiote lacks. Single-device VAE porting is straightforward; multi-device requires implementing TP in TT-Symbiote's conv module.

2. **Conv3d is a hard gap**: There is no TT-Symbiote Conv3d module. Video model VAEs (Mochi, Wan) are blocked on this. The required weight preprocessing, blocking configuration, and causal padding logic from TT-DiT would need to be ported to a new `TTNNConv3d` module.

3. **Patch embedding uses linear, not conv**: Both frameworks implement patch embedding as an unfolded linear projection rather than a true convolution. This makes patch embedding portable despite the conv layer differences.

4. **TT-Symbiote has fused Conv+BN that TT-DiT lacks**: The `TTNNConv2dBNNHWC` and `TTNNConv2dBNActivationNHWC` classes fuse BatchNorm into convolution weights. This is irrelevant for diffusion models but demonstrates TT-Symbiote's strength in vision backbone acceleration.

5. **Weight management diverges significantly**: TT-DiT keeps conv weights on host (`on_host=True`) for dynamic loading/unloading. TT-Symbiote caches conv configurations and weights on device. For memory-constrained VAE inference where encoder and decoder alternate, TT-DiT's approach is more memory-efficient.

---

**Next:** [Chapter 4 -- Joint Attention and Transformer Blocks](../ch4_attention_and_transformer_blocks/index.md)
