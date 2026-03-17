# Weight Conversion

Weight conversion is the process of transforming checkpoint tensors from their stored precision (typically bfloat16 or float32) into the quantized formats (bfloat8_b or bfloat4_b) that TTNN uses for reduced-precision matmul kernels. This page covers the API call, when to do it, shape requirements, transposition, T3K sharding, and the dequantization path that operates transparently during inference.

## The Core API Call

```python
import torch
import ttnn

weight_tt = ttnn.as_tensor(
    weight,                                # torch.Tensor, bfloat16, on CPU
    dtype=ttnn.bfloat8_b,                 # or ttnn.bfloat4_b
    layout=ttnn.TILE_LAYOUT,              # mandatory for quantized dtypes
    device=device,                         # ttnn.Device handle
    memory_config=ttnn.DRAM_MEMORY_CONFIG, # place in DRAM
)
```

`ttnn.as_tensor` performs the quantization, transfers the packed tile data to device DRAM, and returns a `ttnn.Tensor` handle. The original `weight` tensor is not modified.

### Required Arguments

| Argument | Value | Reason |
|---|---|---|
| `dtype` | `ttnn.bfloat8_b` or `ttnn.bfloat4_b` | Selects the packed tile format |
| `layout` | `ttnn.TILE_LAYOUT` | bfloat8_b and bfloat4_b packing is defined only over 32x32 tiles |
| `device` | a `ttnn.Device` | Triggers host-to-device transfer after quantization |
| `memory_config` | `ttnn.DRAM_MEMORY_CONFIG` | Expert weight matrices are too large for L1 |

> **Warning:** Passing `layout=ttnn.ROW_MAJOR_LAYOUT` with a quantized dtype will raise an error. The block floating-point encoding in bfloat8_b and bfloat4_b is defined per 32x32 tile, so row-major layout is not a valid combination.

## Load-Time vs On-the-Fly Conversion

There are two points at which you could call `ttnn.as_tensor` for quantized weights:

**Load-time conversion** (recommended for inference): Convert all expert weights once when the model is loaded, before the inference loop begins. Each `ttnn.as_tensor` call is a one-time startup cost. During inference, the device-resident quantized tensor is reused across every forward pass.

**On-the-fly conversion**: Call `ttnn.as_tensor` inside the forward pass, just before the matmul. This adds the full host-to-device transfer and quantization overhead to every forward pass.

For inference serving, on-the-fly conversion is almost never the right choice. The latency added per forward pass accumulates directly into time-to-first-token and throughput measurements.

```python
# Load-time conversion (correct pattern for inference)
def load_expert_weights(checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    expert_weights = {}
    for layer_idx in range(num_layers):
        for expert_idx in range(num_experts):
            key = f"layers.{layer_idx}.experts.{expert_idx}.w1.weight"
            w = state_dict[key].to(torch.bfloat16)  # normalize to bfloat16 first
            # Ensure shape is 32-aligned on both dims before conversion
            assert w.shape[0] % 32 == 0 and w.shape[1] % 32 == 0, (
                f"Weight {key} shape {w.shape} is not 32-aligned"
            )
            expert_weights[(layer_idx, expert_idx, "w1")] = ttnn.as_tensor(
                w,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
    return expert_weights
```

## Shape Requirements and Transposition

TTNN's TILE_LAYOUT requires that both spatial dimensions of the tensor are multiples of 32. For expert weight matrices, you must verify (and if necessary pad) the shape before calling `ttnn.as_tensor`.

Expert weight matrices are frequently stored in checkpoints as `[d_model, d_ff]` (input features by hidden features). For column-major matmul on Wormhole hardware, these weights are typically transposed to `[d_ff, d_model]` so that the activation vector `[batch, d_model]` multiplies against the weight's second dimension.

The important rule: **perform transposition on the CPU torch tensor before calling `ttnn.as_tensor`.** Transposing a tile-packed bfloat4_b tensor after the fact requires repacking every tile and is not a zero-cost operation.

```python
import torch
import ttnn

def prepare_expert_weight(raw_weight: torch.Tensor, dtype: ttnn.DataType, device) -> ttnn.Tensor:
    """
    Converts a raw checkpoint weight to a device-resident quantized tensor.

    Args:
        raw_weight: shape [d_model, d_ff] as loaded from checkpoint (bfloat16 or float32)
        dtype: ttnn.bfloat8_b or ttnn.bfloat4_b
        device: ttnn.Device handle

    Returns:
        ttnn.Tensor of shape [d_ff, d_model] in TILE_LAYOUT on device DRAM
    """
    # Step 1: Normalize to bfloat16 on CPU
    w = raw_weight.to(torch.bfloat16)

    # Step 2: Transpose to [d_ff, d_model] for column-major matmul
    w = w.t().contiguous()  # contiguous() ensures no strided-view issues

    # Step 3: Validate 32-alignment after transposition
    d_ff, d_model = w.shape
    assert d_ff % 32 == 0, f"d_ff={d_ff} is not divisible by 32"
    assert d_model % 32 == 0, f"d_model={d_model} is not divisible by 32"

    # Step 4: Quantize and transfer to device
    w_tt = ttnn.as_tensor(
        w,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return w_tt
```

> **Tip:** If `d_model` or `d_ff` is not a multiple of 32, use `torch.nn.functional.pad` to pad the tensor to the nearest multiple of 32 before transposition. Track the original unpadded dimensions so you can slice the output after the matmul.

## T3K Distributed Expert Weights with `mesh_mapper`

On T3K (Tenstorrent 3000-series multi-chip) systems, expert weight tensors are sharded across multiple devices. The `mesh_mapper` argument to `ttnn.as_tensor` controls how the tensor is distributed across the mesh.

For Mixture-of-Experts (MoE) expert weight tensors stacked along an expert dimension, a typical sharding strategy uses `mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0)`, which shards along dimension 0 so that experts are distributed across devices.

```python
# T3K multi-chip weight conversion with mesh device
# Assumes weights are stacked: shape [num_experts, d_ff, d_model]
# mesh_device is a ttnn.MeshDevice covering all chips in the T3K node

stacked_experts = torch.stack([
    expert_weights[i] for i in range(num_experts)
], dim=0).to(torch.bfloat16)  # shape: [num_experts, d_ff, d_model]

stacked_tt = ttnn.as_tensor(
    stacked_experts,
    dtype=ttnn.bfloat8_b,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),  # shard experts across chips
)
```

> **Warning:** `ttnn.ShardTensorToMesh` behavior depends on the mesh topology configured for your T3K node. Verify that the number of experts is evenly divisible by the number of mesh devices before applying expert-dimension sharding. Uneven sharding will result in some devices holding more experts than others, which can cause load imbalance or runtime errors.

## The Dequantization Path: No User Call Needed

TTNN automatically dequantizes quantized weights during the matmul kernel; see `dtype_in_linear_and_matmul.md` for the full dequantization path.

---

**Next:** [`compute_kernel_config.md`](./compute_kernel_config.md)
