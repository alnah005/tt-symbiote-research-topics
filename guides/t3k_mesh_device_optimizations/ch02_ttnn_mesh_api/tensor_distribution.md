# Distributing Tensors Across the 8-Device Mesh

> **Quick Reference — TTNN API Symbols Introduced in This File**
>
> | Symbol | Module | Description |
> |---|---|---|
> | `ttnn.TensorSpec` | `ttnn` | Descriptor combining shape, data type, and memory placement for a tensor, used to specify how a tensor is distributed across a mesh. |
> | `ttnn.ShardSpec` | `ttnn` | Descriptor specifying the size, orientation, and memory layout of a single shard within a larger sharded tensor. |
> | `ttnn.ShardStrategy` | `ttnn` | Enum with values `ROW_WISE`, `COL_WISE`, and `BLOCK_WISE` controlling the axis along which a tensor is divided. |
> | `ttnn.ReplicateTensorToMesh` | `ttnn` | Mesh mapper that replicates a host tensor identically onto every device in the mesh. |
> | `ttnn.ShardTensorToMesh` | `ttnn` | Mesh mapper that divides a host tensor into N slices and places one slice per device. |
> | `ttnn.from_torch` | `ttnn` | Converts a PyTorch tensor to a TTNN tensor, optionally distributing it across a mesh according to a mesh mapper. |
> | `ttnn.to_device` | `ttnn` | Moves an already-created TTNN tensor to a target device or mesh, with optional memory configuration. |
> | `ttnn.from_device` | `ttnn` | Moves a TTNN tensor from device memory back to host memory, gathering shards if the source is a sharded mesh tensor. |

Before you can dispatch operations on a `MeshDevice`, the tensors those operations consume must be placed on device memory. This file describes the abstractions TTNN provides for specifying tensor distribution — `TensorSpec` and `ShardSpec` — and the API calls that move data from host to mesh.

---

## TensorSpec and ShardSpec

`TensorSpec` and `ShardSpec` describe the intended layout of a tensor before it is actually created or transferred. They are declarative descriptors: they tell TTNN what you want, not how to achieve it. TTNN uses them to allocate memory, configure DMA transfers, and set up the metadata that operations read to locate tensor data.

### TensorSpec

A `TensorSpec` combines four properties:

| Property | Description |
|---|---|
| `shape` | The logical shape of the full tensor (e.g., `(8192, 4096)` for a weight matrix). This is the shape from the model's perspective, independent of how it is sharded. |
| `dtype` | The element data type: `ttnn.bfloat16`, `ttnn.float32`, `ttnn.int8`, etc. |
| `layout` | The on-device storage layout: `ttnn.TILE_LAYOUT` (tiled, required for matmul and most compute kernels) or `ttnn.ROW_MAJOR_LAYOUT` (contiguous row-major, used for routing tables and small metadata tensors). |
| `memory_config` | A `ttnn.MemoryConfig` object specifying whether the tensor lives in L1 or DRAM, and if sharded, how shards are distributed within each device. Memory configuration is covered in detail in Chapter 4; for this chapter, the examples use `ttnn.DRAM_MEMORY_CONFIG` and `ttnn.L1_MEMORY_CONFIG` as shorthand. |

You construct a `TensorSpec` as:

```python
spec = ttnn.TensorSpec(
    shape=ttnn.Shape([8192, 4096]),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

`TensorSpec` is most commonly used implicitly — you pass its components directly to `ttnn.from_torch` or `ttnn.empty` rather than constructing a `TensorSpec` object explicitly. However, understanding its four properties is necessary for understanding how `ShardSpec` extends it.

### ShardSpec

`ShardSpec` describes how a single device's portion of a sharded tensor is laid out in that device's memory. It is not a standalone object; it is embedded inside the `ttnn.MemoryConfig` you pass to operations or to `ttnn.from_torch`.

The key properties of `ShardSpec` are:

| Property | Description |
|---|---|
| `shard_grid` | A `ttnn.CoreRangeSet` specifying which Tensix cores on the device store this shard. For DRAM-resident tensors, the shard is stored in a single DRAM bank rather than distributed across cores. |
| `shard_shape` | The shape of the local shard: `(shard_height, shard_width)` in elements, for both `TILE_LAYOUT` and `ROW_MAJOR_LAYOUT`. When using `TILE_LAYOUT`, each dimension must be a multiple of 32 (the tile size in elements), but the values themselves are always element counts, not tile counts. For example, `shard_shape=(32, 32)` means 32 elements × 32 elements — exactly 1 tile per dimension with `TILE_LAYOUT` — not 32 tiles × 32 tiles. |
| `shard_orientation` | `ttnn.ShardOrientation.ROW_MAJOR` or `ttnn.ShardOrientation.COL_MAJOR`, controlling the order in which tile rows are laid out in memory within the shard. |
| `halo` | Whether the shard includes halo (overlap) elements from adjacent shards. For MoE workloads, halo is not used. |

In practice, you specify sharding behavior through higher-level helpers (`ttnn.ShardTensorToMesh`, discussed below) rather than building `ShardSpec` objects manually.

---

## Row-Wise vs. Column-Wise Sharding

Sharding a tensor across eight devices divides the tensor along one of its dimensions. TTNN calls this the shard strategy. The two most common strategies for 2D tensors are row-wise (sharding along the first dimension, i.e., rows) and column-wise (sharding along the second dimension, i.e., columns). A third option, block-wise sharding, divides both tensor dimensions simultaneously. Two distinct uses of "block-wise sharding" must be kept separate: (1) block-wise **device distribution** — routing different 2D blocks to different devices across two mesh axes — requires a mesh with more than one row and is not directly applicable to the `(1, 8)` T3K mesh, which has only a single row axis; (2) block-wise **L1 sharding within a single device** — partitioning a tensor into 2D blocks mapped to Tensix core grids within one device — is independent of the mesh topology and is fully available on T3K regardless of mesh shape.

### Row-Wise Sharding

Row-wise sharding divides the tensor's first dimension into N equal slices, one per device. For a weight tensor of shape `(8192, 4096)` sharded row-wise across eight devices, each device holds a `(1024, 4096)` slice — 1024 rows of the full 8192-row matrix.

Row-wise sharding is appropriate when:

- The downstream operation (typically a matrix multiplication) is parallelized along the "input" dimension of the weight. In a standard feed-forward network, if the input activation is `(batch, seq_len, 8192)` and the weight is `(8192, 4096)`, sharding the weight row-wise means each device multiplies a different slice of the hidden dimension against the full input, then contributes a partial sum that must be reduced across devices via `ttnn.all_reduce` or `ttnn.reduce_scatter` to produce the final output.
- The tensor is large in its first dimension and small in its second, and you want to minimize inter-device communication volume.

In MoE expert parallelism, row-wise sharding is commonly used for the expert down-projection weights (the `W_out` matrix of each expert), where the hidden dimension is split across devices and each device contributes a partial output that is later summed.

### Column-Wise Sharding

Column-wise sharding divides the tensor's second (or last) dimension into N equal slices, one per device. For a weight tensor of shape `(4096, 8192)` sharded column-wise across eight devices, each device holds a `(4096, 1024)` slice — 1024 columns of the full 8192-column output dimension.

Column-wise sharding is appropriate when:

- The downstream operation is parallelized along the "output" dimension of the weight. In a feed-forward network, if the input activation is `(batch, seq_len, 4096)` and the weight is `(4096, 8192)`, sharding the weight column-wise means each device computes a different slice of the 8192-element output independently, requiring no intermediate reduction — only a gather at the end if the full output is needed.
- Each device's output shard can be consumed directly by the next layer without reconstructing the full tensor (for example, in fused expert dispatch where each device routes its output slice directly into the all-to-all send buffer).

Column-wise sharding for expert up-projection weights (`W_in` of each expert) is a common pattern in MoE inference on T3K, since it allows each device to produce its portion of the expert's internal representation without synchronizing with other devices until the combine phase.

### Comparison

| Dimension | Row-Wise | Column-Wise |
|---|---|---|
| Divided axis | First (rows, input dim) | Last (columns, output dim) |
| Each device receives | Full output width, partial input depth | Partial output width, full input depth |
| Communication required after matmul | All-reduce or reduce-scatter (to sum partial products) | All-gather (to assemble full output, if needed) |
| Suitable for | Down-projection, partial-sum patterns | Up-projection, output-parallel patterns |
| Memory per device (8-device split) | `(full_rows/8, full_cols)` | `(full_rows, full_cols/8)` |

For both strategies, tensors must have their divided dimension divisible by the number of devices (8 for a full T3K mesh). If the dimension is not divisible by 8, you must pad before sharding and unpad after gathering.

---

## Replicated vs. Sharded Tensors

Not every tensor benefits from sharding. Some tensors are small enough, or used identically by every device, that replicating them on all eight devices is the right choice.

### When to Replicate

Replicate a tensor when:

- Its size is small relative to per-device DRAM capacity (for example, a bias vector of shape `(4096,)` is 8 KiB in bfloat16 — negligible per-device cost even when replicated eight times).
- Every device's kernel reads the same data from the tensor (for example, a layer norm scale and bias, or the expert router's weight matrix `W_router` of shape `(4096, num_experts)` when the full routing computation runs on every device before dispatching tokens).
- The tensor is written once at model load time and never updated during inference (static replication has no ongoing synchronization cost; the replication cost is paid only once during weight loading).

### When to Shard

Shard a tensor when:

- Its size is large enough that per-device DRAM capacity would be exceeded by replication (for example, the expert weight matrices in a MoE model with 128 experts and `hidden_dim=4096` are large enough that distributing them — 16 experts per device — is necessary, not optional).
- Different devices' kernels process different slices of the tensor (for example, each device holds only the expert weights for the experts assigned to it, and never needs the other experts' weights).
- You want to exploit inter-device parallelism on a shared weight (for example, sharding a large shared attention projection across devices).

### Memory and Bandwidth Trade-Offs

| Factor | Replicated | Sharded |
|---|---|---|
| Per-device memory cost | Full tensor size | `1/N × full tensor size` |
| Load-time bandwidth cost (host → devices) | N copies transferred | 1 copy total, split |
| Operation communication cost | None (data is local) | Depends on operation (may require reduce, gather, or all-to-all) |
| Access pattern | Fully local; no synchronization | Must reconstruct full tensor for operations that require it |
| Typical use case | Small weights, biases, router weights | Expert weights, large activations, KV cache |

The cross-over point between replication and sharding depends on the per-device DRAM budget and the communication overhead of the operations that consume the tensor. Chapter 4 provides concrete memory budget estimates for the Qwen3MoE configuration; Chapter 5 applies those estimates to expert weight placement decisions.

---

## How MoE Expert Weights Are Placed Across Devices

In a Mixture-of-Experts model, each expert is an independent feed-forward sub-network. A Qwen3MoE configuration may have, for example, 64 or 128 routed experts. On a T3K system, the conventional strategy is to distribute experts uniformly: with 8 devices and 64 experts, each device holds 8 experts; with 128 experts, each device holds 16.

For each device, the expert weights it holds are loaded into its DRAM as independent tensors (one tensor per weight matrix per expert), not as a single contiguous block. This allows TTNN's memory allocator to place them independently and the dispatch scheduler to stream them into L1 independently as each expert's compute kernel is invoked.

The placement convention is that experts are assigned to devices in contiguous groups: device 0 holds experts 0–(K-1), device 1 holds experts K–(2K-1), and so on, where K = num_experts / 8. This is the "naive uniform" placement; Chapter 5 discusses load-aware and locality-aware variants that deviate from this pattern.

From an API perspective, expert weights are loaded device-by-device at model initialization time. For each device, you slice the full weight tensor for the experts assigned to that device, place the slice in device DRAM, and associate it with the device's expert compute kernels. The host never sends all expert weights to all devices; it sends only each device's assigned slice.

---

## API Patterns

### `ttnn.from_torch` with Multi-Device Placement

`ttnn.from_torch` is the primary entry point for transferring host (PyTorch) tensors to device memory. For multi-device placement, it accepts a `mesh_mapper` argument that specifies how to distribute the tensor across the mesh.

**Replicated placement:**

```python
import torch
import ttnn

# A router weight that every device needs in full.
router_weight_host = torch.randn(4096, 64, dtype=torch.bfloat16)

router_weight_mesh = ttnn.from_torch(
    router_weight_host,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
# router_weight_mesh is now resident on all 8 devices with identical data.
```

**Column-wise sharded placement (expert up-projection weights):**

```python
# An expert weight matrix, column-wise sharded across all 8 devices.
# Full shape: (4096, 8192) — 4096 input dim, 8192 output dim.
# Each device holds columns: (4096, 1024).
expert_up_weight_host = torch.randn(4096, 8192, dtype=torch.bfloat16)

expert_up_weight_mesh = ttnn.from_torch(
    expert_up_weight_host,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
# Device at (row=0, col=N) holds columns [N*1024 : (N+1)*1024].
```

**Row-wise sharded placement (expert down-projection weights):**

```python
# An expert weight matrix, row-wise sharded across all 8 devices.
# Full shape: (8192, 4096) — 8192 input dim, 4096 output dim.
# Each device holds rows: (1024, 4096).
expert_down_weight_host = torch.randn(8192, 4096, dtype=torch.bfloat16)

expert_down_weight_mesh = ttnn.from_torch(
    expert_down_weight_host,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
# Device at (row=0, col=N) holds rows [N*1024 : (N+1)*1024].
```

The `dim` argument to `ttnn.ShardTensorToMesh` specifies which dimension of the host tensor is divided. `dim=0` produces row-wise sharding; `dim=1` produces column-wise sharding for a 2D tensor.

### `ttnn.to_device` for Mesh Targets

`ttnn.to_device` moves a TTNN tensor that was previously created on the host (or on a single device) to a `MeshDevice`. Unlike `ttnn.from_torch`, it operates on an already-existing TTNN tensor and accepts an optional `memory_config` to control where on the device the data lands.

```python
# NOTE: Do NOT use ttnn.to_device with a MeshDevice for mesh-wide replication.
# For replication, use ttnn.from_torch with mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device).
single_device = mesh_device.get_device(0)
host_tensor = ttnn.from_torch(
    torch.randn(4096, 4096, dtype=torch.bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
)
device_tensor = ttnn.to_device(
    host_tensor,
    device=single_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

> **Warning — do not use `ttnn.to_device` for mesh-wide replication.** Calling `ttnn.to_device(tensor, device=mesh_device)` without a `mesh_mapper` does not reliably replicate the tensor across all 8 devices. The correct API for replication across a `MeshDevice` is `ttnn.from_torch` with `mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)`, as shown in the "Replicated placement" example above.

### Retrieving Results with `ttnn.from_device`

After inference, you retrieve results from the mesh using `ttnn.from_device`. For sharded tensors, this gathers the shards from all devices and concatenates them back into a single host tensor along the sharded dimension.

```python
# Gather a column-wise sharded result back to host.
# mesh_output has shape (batch, seq_len, 8192) distributed as 8 × (batch, seq_len, 1024).
host_output = ttnn.from_device(mesh_output)  # host-memory ttnn.Tensor of shape (batch, seq_len, 8192)
torch_output = ttnn.to_torch(host_output)    # convert to torch.Tensor for use with PyTorch ops
```

For replicated tensors, `ttnn.from_device` returns the data from one device (typically the first in the mesh) without any gather operation, since all devices hold identical data.

---

## Shape Constraints and Common Errors

Sharding a tensor across eight devices imposes divisibility constraints:

- The divided dimension must be evenly divisible by 8. For `dim=1` (column-wise), `tensor.shape[1]` must be divisible by 8; for `dim=0` (row-wise), `tensor.shape[0]` must be divisible by 8.
- When using `TILE_LAYOUT`, each shard must be at least one tile in each dimension. TTNN's tile size is 32×32 elements. Therefore, the shard size in each dimension must be at least 32 elements and must be a multiple of 32. For a column-wise shard of a `(4096, 8192)` tensor across 8 devices, each device's shard is `(4096, 1024)` — 1024 is a multiple of 32, satisfying the tile layout constraint. Note that any multiple of 32 is valid (e.g., 96 = 3×32 also satisfies the constraint); the requirement is divisibility by 32, not equality to a power of 32.
- **Combined constraint (important):** Because the divided dimension must satisfy both constraints simultaneously, the necessary combined condition for an 8-device T3K mesh with `TILE_LAYOUT` is that the divided dimension must be divisible by `num_devices × 32 = 8 × 32 = 256`. A dimension divisible by 8 but not by 256 — for example 40, 48, 64, 96, 128, or 192 — produces shards that are not multiples of 32 and will fail with a tile-alignment error at runtime. Implementing a padding check as `dim % 8 == 0` is therefore insufficient; the correct check is `dim % 256 == 0`.
- If your tensor's divided dimension is not divisible by 256 (for an 8-device mesh with `TILE_LAYOUT`), you must pad it to the next multiple of 256 before calling `ttnn.from_torch`. The padding can be arbitrary (zeros are conventional for weight padding); you must unpad the gathered output if the padded dimension affects your result.
- **Non-divided dimensions must also be multiples of 32 for `TILE_LAYOUT`.** The constraints above apply to the divided dimension, but `TILE_LAYOUT` requires that *every* dimension of the per-device shard be a multiple of 32 — including dimensions that sharding does not divide. For example, a tensor of shape `(4097, 8192)` sharded column-wise across 8 devices passes the divided-dimension check (`8192 % 256 == 0`), but each device's shard has shape `(4097, 1024)` — and `4097 % 32 != 0`. The `ttnn.from_torch` call will fail with a tile-alignment error on the non-divided dimension. The correct approach is to pad **all** tensor dimensions to multiples of 32 first, then shard.

Violations of these constraints produce errors from the TTNN memory allocator or the sharding mapper, typically at the point of the `ttnn.from_torch` call rather than during operation dispatch.

---

**Next:** [collective_primitives.md](./collective_primitives.md)
