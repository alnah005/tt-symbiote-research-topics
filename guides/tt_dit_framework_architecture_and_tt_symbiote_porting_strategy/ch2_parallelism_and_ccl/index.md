# Chapter 2: Parallelism and CCL Infrastructure

## Prerequisites

- [Chapter 1 -- Architecture Overview](../ch1_architecture_overview/index.md): understanding of TT-DiT's `Module`, `Parameter`, and the overall codebase layout.
- Familiarity with TT mesh devices (`ttnn.MeshDevice`, `ttnn.MeshShape`) and multi-device programming.
- Basic understanding of Megatron-style tensor and sequence parallelism concepts (explained from first principles below).

---

## Overview

TT-DiT implements a **3-axis parallelism model** to distribute diffusion transformer workloads across multi-device meshes (e.g., 2x4 T3K or 4x8 TG systems). The three axes are:

1. **CFG Parallel (CFG-P)** -- Classifier-Free Guidance parallelism that duplicates the denoising computation across submeshes to run conditional and unconditional forward passes simultaneously.
2. **Sequence Parallel (SP)** -- Shards the spatial token sequence across devices along one mesh axis, reducing per-device memory for activations.
3. **Tensor Parallel (TP)** -- Shards weight matrices column-wise or row-wise across devices along another mesh axis, reducing per-device weight memory and distributing GEMM computation.

These three axes are orthogonal and composable. A `DiTParallelConfig` object describes how they are mapped onto the physical mesh device, and pipeline code uses this configuration to create submeshes and instantiate one `CCLManager` per submesh for collective communication.

---

## The Three Parallelism Axes

### CFG Parallel

Classifier-Free Guidance (CFG) is a standard technique in diffusion models where each denoising step runs the model twice: once with the text prompt (conditional) and once without (unconditional). The two outputs are then blended with a guidance scale:

$$\epsilon_\text{guided} = \epsilon_\text{uncond} + s \cdot (\epsilon_\text{cond} - \epsilon_\text{uncond})$$

where $s$ is the guidance scale (typically 3.5--7.5).

CFG-P exploits this by **running each branch on a separate submesh**. On a 2x4 mesh with `cfg_parallel = ParallelFactor(factor=2, mesh_axis=0)`, the mesh is split into two 1x4 submeshes: one for the conditional pass, one for the unconditional pass. Both execute in parallel, and results are combined on the host.

Not all models use CFG. Flux1, for example, uses guidance embeddings instead of dual-pass CFG, so its `cfg_parallel` factor is always 1.

### Sequence Parallel (SP)

Sequence parallelism shards the spatial token dimension across devices. In a DiT model, images are patchified into spatial tokens of shape `[batch, seq_len, hidden_dim]`. With SP factor $N$ along mesh axis 0, each device holds `seq_len / N` tokens.

SP requires collective communication at specific points:
- **All-gather** before attention (so each device sees the full sequence for key/value computation).
- **Reduce-scatter** after attention outputs are produced.

The ring joint SDPA kernel (`ttnn.transformer.ring_joint_scaled_dot_product_attention`) fuses the all-gather into the attention computation itself, performing the gather incrementally as chunks arrive.

### Tensor Parallel (TP)

Tensor parallelism partitions weight matrices across devices. TT-DiT uses the Megatron-style approach with two complementary patterns:

- **Column Parallel (ColParallelLinear)**: The weight matrix $W$ of shape `[K, N]` is split along the $N$ (output) dimension, so each device holds `[K, N/T]`. The input $x$ is replicated; each device computes a slice of the output.
- **Row Parallel (RowParallelLinear)**: The weight matrix of shape `[K, N]` is split along the $K$ (input) dimension, so each device holds `[K/T, N]`. Each device receives a slice of the input (the column-fractured output from a preceding ColParallel layer) and produces a partial result. A **reduce-scatter** (or all-reduce) sums the partial results.

These two layer types are always paired in TT-DiT: a `ColParallelLinear` feeds into a `RowParallelLinear` (or vice versa), with at most one collective communication operation between them. This is the key insight from the Megatron-LM parallelism strategy.

For a detailed walkthrough of these layers, see [`parallel_linear_layers.md`](./parallel_linear_layers.md).

---

## DiTParallelConfig

The parallelism configuration is defined in `models/tt_dit/parallel/config.py` using two `NamedTuple` types:

```python
# models/tt_dit/parallel/config.py

class ParallelFactor(NamedTuple):
    factor: int       # Number of devices along this parallelism axis
    mesh_axis: int    # Which mesh dimension (0 or 1) this axis maps to

class DiTParallelConfig(NamedTuple):
    cfg_parallel: ParallelFactor
    tensor_parallel: ParallelFactor
    sequence_parallel: ParallelFactor
```

Each `ParallelFactor` binds a **parallelism degree** (how many ways to split) to a **physical mesh axis** (which dimension of the 2D mesh to split along). The constraint is that `cfg_parallel.factor * sequence_parallel.factor * tensor_parallel.factor` must equal the total number of devices, and the mesh axes must be assigned consistently.

### Concrete Configurations

The pipeline factory methods define default configurations per mesh shape. From the Motif pipeline:

```python
# models/tt_dit/pipelines/motif/pipeline_motif.py (default_config)

{
    (2, 4): {
        "cfg_config": (2, 0),   # CFG-P: factor=2 on mesh axis 0
        "sp": (1, 0),           # SP: factor=1 (disabled)
        "tp": (4, 1),           # TP: factor=4 on mesh axis 1
    },
    (4, 8): {
        "cfg_config": (2, 1),   # CFG-P: factor=2 on mesh axis 1
        "sp": (4, 0),           # SP: factor=4 on mesh axis 0
        "tp": (4, 1),           # TP: factor=4 on mesh axis 1
    },
}
```

On a **2x4 T3K mesh** (8 devices), Motif uses:
- CFG-P = 2 on axis 0: splits the 2 rows into 2 submeshes of shape 1x4.
- TP = 4 on axis 1: each row of 4 devices does tensor parallelism.
- SP = 1: no sequence parallelism (each submesh sees the full sequence).

On a **4x8 TG mesh** (32 devices), Motif uses:
- CFG-P = 2 on axis 1: splits into 2 submeshes of shape 4x4.
- SP = 4 on axis 0: each submesh's 4 rows shard the sequence.
- TP = 4 on axis 1: each row of 4 devices does tensor parallelism within the submesh.

For Flux1 (no CFG), the config always sets `cfg_parallel = ParallelFactor(factor=1, mesh_axis=0)` and creates only one submesh:

```python
# models/tt_dit/pipelines/flux1/pipeline_flux1.py

{
    (2, 4): {"sp": (2, 0), "tp": (4, 1), ...},
    (4, 4): {"sp": (4, 0), "tp": (4, 1), ...},
    (4, 8): {"sp": (4, 0), "tp": (8, 1), ...},
}
```

### Additional Parallel Config Types

Besides `DiTParallelConfig`, the module defines specialized configs for other pipeline components:

| Config Class | Fields | Used By |
|---|---|---|
| `DiTParallelConfig` | `cfg_parallel`, `tensor_parallel`, `sequence_parallel` | DiT transformer blocks |
| `EncoderParallelConfig` | `tensor_parallel` | Text encoders (CLIP, T5) |
| `VAEParallelConfig` | `tensor_parallel` | Standard VAE decoder |
| `VaeHWParallelConfig` | `height_parallel`, `width_parallel` | VAE with spatial parallelism |
| `MochiVAEParallelConfig` | `time_parallel`, `h_parallel`, `w_parallel` | Mochi video VAE |

Encoders and VAEs typically reuse one of the submesh's axes for their own tensor parallelism, defaulting to the SP axis when TP is assigned to a different axis.

---

## Submesh Creation and CCLManager Instantiation

Pipeline constructors orchestrate the mesh setup. The general pattern (from `MotifPipeline.__init__`):

```python
# models/tt_dit/pipelines/motif/pipeline_motif.py (simplified)

# 1. Compute the submesh shape from SP and TP factors
submesh_shape = list(mesh_device.shape)
submesh_shape[parallel_config.sequence_parallel.mesh_axis] = parallel_config.sequence_parallel.factor
submesh_shape[parallel_config.tensor_parallel.mesh_axis] = parallel_config.tensor_parallel.factor

# 2. Create submeshes -- one per CFG parallel factor
self._submesh_devices = mesh_device.create_submeshes(
    ttnn.MeshShape(*submesh_shape)
)[0 : parallel_config.cfg_parallel.factor]

# 3. Create a CCLManager per submesh
self._ccl_managers = [
    CCLManager(submesh_device, num_links=num_links, topology=topology)
    for submesh_device in self._submesh_devices
]
```

**Step 1** determines how large each submesh should be by setting the SP and TP dimensions of the submesh shape. For a 2x4 mesh with SP=1 on axis 0 and TP=4 on axis 1, the submesh shape is `[1, 4]`.

**Step 2** calls `mesh_device.create_submeshes()` which partitions the physical mesh into as many submeshes of the given shape as will fit. For a 2x4 mesh split into `[1, 4]` submeshes, this yields 2 submeshes. We take the first `cfg_parallel.factor` of these.

**Step 3** creates a `CCLManager` for each submesh. The CCLManager manages all collective communication within that submesh -- semaphores, ping-pong buffers, and the helper methods for `all_gather` and `reduce_scatter` operations.

The resulting architecture for a 2x4 Motif deployment looks like:

```
Physical Mesh (2x4):
+---+---+---+---+
| 0 | 1 | 2 | 3 |  <- Submesh 0 (conditional, CCLManager 0)
+---+---+---+---+
| 4 | 5 | 6 | 7 |  <- Submesh 1 (unconditional, CCLManager 1)
+---+---+---+---+
       TP=4 (axis 1) within each submesh
```

Encoders and VAEs typically run on `submesh_devices[0]` only (they do not need CFG duplication).

---

## Chapter Files

This chapter is organized into the following sections:

- [`ccl_manager.md`](./ccl_manager.md) -- Detailed walkthrough of the CCLManager: SubDevice setup, semaphore initialization with ping-pong indexing, persistent buffer caching, helper methods, hyperparameter tuning, and VAE-specific CCL ops.
- [`parallel_linear_layers.md`](./parallel_linear_layers.md) -- ColParallelLinear and RowParallelLinear: Megatron-style parallelism, weight sharding via `mesh_axes`, FSDP weight gathering, `_prepare_torch_state` reshaping, and `minimal_matmul` usage.
- [`mapping_to_symbiote.md`](./mapping_to_symbiote.md) -- Comparison with TT-Symbiote's `DistributedConfig`/`DistributedTensorConfig`, distributed linear variants, gaps in TT-Symbiote's CCL infrastructure, and recommendations.

---

## Key Takeaways

1. **TT-DiT uses 3-axis parallelism** (CFG, sequence, tensor) composed via `DiTParallelConfig`, with each axis mapped to a physical mesh dimension through `ParallelFactor(factor, mesh_axis)` tuples.
2. **CFG parallelism is implemented via submeshes** -- the physical mesh is partitioned, and each submesh gets its own `CCLManager` for independent collective communication.
3. **Tensor and sequence parallelism are orthogonal** -- TP shards weights (column/row), SP shards activations (sequence tokens), and they operate along different mesh axes.
4. **The CCLManager is the central coordination point** for all collective communication within a submesh, managing semaphores, ping-pong buffers, and async CCL operations.
5. **Default configurations scale across hardware** -- the same pipeline code adapts from T3K (8 devices) to TG (32 devices) by adjusting the parallelism factors in the config dictionaries.

---

**Next:** [`ccl_manager.md`](./ccl_manager.md)
