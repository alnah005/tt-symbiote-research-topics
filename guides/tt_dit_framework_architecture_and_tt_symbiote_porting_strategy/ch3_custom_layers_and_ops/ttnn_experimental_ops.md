# TTNN Experimental Operations Catalog

## Prerequisites

- [Chapter 2 -- Parallel Linear Layers](../ch2_parallelism_and_ccl/parallel_linear_layers.md): understanding of how TT-DiT linear layers use `ttnn.experimental.minimal_matmul`.
- [Chapter 2 -- CCLManager](../ch2_parallelism_and_ccl/ccl_manager.md): understanding of `all_gather_persistent_buffer` and the async collective communication model.
- [Normalization Layers](./normalization_layers.md): understanding of how distributed norms use experimental pre/post-allgather ops.

---

## Overview

TT-DiT makes heavy use of the `ttnn.experimental` namespace -- operations that are functional but not yet promoted to the stable `ttnn` API. These ops are subject to change and are generally not available in TT-Symbiote's module library.

This file catalogs every `ttnn.experimental.*` operation found in the TT-DiT codebase, organized by functional category. For each op, we document:

1. **Purpose**: what the operation does.
2. **Where used**: which TT-DiT layers/models call it.
3. **Key parameters**: the most important arguments.
4. **TT-Symbiote equivalent**: whether TT-Symbiote uses the same op, a different op for the same purpose, or has no equivalent.

---

## 1. Matrix Multiplication

### `ttnn.experimental.minimal_matmul`

**Purpose**: General matrix multiplication with explicit block-size control via `ttnn.MinimalMatmulConfig`. This is TT-DiT's universal matmul primitive -- every `Linear`, `ColParallelLinear`, and `RowParallelLinear` uses it.

**Where used**:
- `layers/linear.py` -- `Linear.forward`, `ColParallelLinear.forward`, `RowParallelLinear.forward`

**Key parameters**:
```python
ttnn.experimental.minimal_matmul(
    input_tensor,           # Activation tensor
    weight_tensor,          # Weight tensor (already transposed in _prepare_torch_state)
    bias_tensor=None,       # Optional fused bias
    config=matmul_config,   # ttnn.MinimalMatmulConfig with M/K/N block sizes
    fused_activation=None,  # Optional fused activation (e.g., GELU)
    compute_kernel_config=...,  # Math fidelity, fp32 accumulation
    dtype=None,             # Optional output dtype override
)
```

The `config` parameter is a `ttnn.MinimalMatmulConfig` computed by `utils/matmul.py:get_matmul_config()`, which looks up optimal block sizes for known `(M, K, N)` shape combinations on the device's core grid (8x8, 8x9, 12x10, or 13x9):

```python
ttnn.MinimalMatmulConfig(
    M_block_size=8,    # Tiles per block along M dimension
    K_block_size=8,    # Tiles per block along K dimension
    N_block_size=8,    # Tiles per block along N dimension
    subblock_h=2,      # Subblock height for inner-loop tiling
    subblock_w=2,      # Subblock width for inner-loop tiling
    compute_with_storage_grid_size=core_grid,
)
```

**TT-Symbiote equivalent**: `ttnn.linear` (stable API). TT-Symbiote's `TTNNLinear` calls:
```python
ttnn.linear(input_tensor, weight, bias=bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

**Difference**: `ttnn.linear` uses TTNN's internal heuristics to choose block sizes and tiling. `ttnn.experimental.minimal_matmul` gives the caller explicit control over blocking, which is critical for performance tuning. If a ported layer's performance is poor with `ttnn.linear`, switching to `minimal_matmul` with tuned configs is the likely fix.

---

### `ttnn.experimental.minimal_matmul_split`

**Purpose**: Performs a single matrix multiply but splits the output along the last dimension into `chunks` separate tensors. Used by `ColParallelLinear` when `chunks` is set to produce separate Q, K, V projections from a single fused QKV weight matrix.

**Where used**:
- `layers/linear.py` -- `ColParallelLinear.forward` (when `self.chunks is not None`)

**Key parameters**:
```python
ttnn.experimental.minimal_matmul_split(
    x, weight,
    chunks=3,           # Number of output splits (e.g., 3 for Q/K/V)
    dim=-1,             # Split dimension
    bias_tensor=None,
    fused_activation=None,
    compute_kernel_config=...,
    config=matmul_config,
)
```

**TT-Symbiote equivalent**: **None.** TT-Symbiote would perform the matmul and then call `ttnn.chunk` separately, which requires materializing the full output before splitting.

---

## 2. Normalization (Pre/Post All-gather)

### `ttnn.experimental.wan_fused_rmsnorm_pre_allgather`

**Purpose**: Computes local RMS statistics on a device-local shard of the hidden dimension. Returns a small statistics tensor (sum of squares per element) that is then all-gathered across devices.

**Where used**:
- `layers/normalization.py` -- `DistributedRMSNorm.forward`

**Key parameters**:
```python
ttnn.experimental.wan_fused_rmsnorm_pre_allgather(
    x,                          # Input activation (device-local shard)
    dtype=ttnn.float32,         # Statistics dtype (float32 for precision)
    compute_kernel_config=...,
)
```

**TT-Symbiote equivalent**: `ttnn.rms_norm_pre_all_gather` (stable API). The stable version is used by `TTNNDistributedRMSNorm`:
```python
tt_stats = ttnn.rms_norm_pre_all_gather(inp, dtype=ttnn.bfloat16)
```

**Difference**: The `wan_fused_*` variant computes statistics in `float32` by default; the stable version in TT-Symbiote uses `bfloat16`. The `wan_fused_*` variant is also part of a kernel-fused pipeline that can optionally apply RoPE in the post-allgather step.

---

### `ttnn.experimental.wan_fused_rmsnorm_post_allgather`

**Purpose**: Applies RMSNorm using the globally-gathered statistics. Optionally fuses rotary positional embedding (RoPE) application in the same kernel pass.

**Where used**:
- `layers/normalization.py` -- `DistributedRMSNorm.forward`

**Key parameters**:
```python
ttnn.experimental.wan_fused_rmsnorm_post_allgather(
    x,                              # Input activation
    stats,                          # Gathered statistics from pre_allgather
    epsilon=1e-5,
    num_heads_per_device=1,         # For multi-head RoPE fusion
    weight=...,                     # Learnable scale
    compute_kernel_config=...,
    transformation_mat=None,        # Optional RoPE transformation matrix
    rope_cos=None,                  # Optional RoPE cosine embeddings
    rope_sin=None,                  # Optional RoPE sine embeddings
)
```

**TT-Symbiote equivalent**: `ttnn.rms_norm_post_all_gather` (stable API). No RoPE fusion support:
```python
tt_out = ttnn.rms_norm_post_all_gather(inp, tt_stats, epsilon=eps, weight=weight)
```

**Difference**: The fused RoPE parameters (`rope_cos`, `rope_sin`, `transformation_mat`, `num_heads_per_device`) are unique to the experimental variant. This fusion eliminates a separate RoPE kernel launch, which is a significant performance win for Wan2.2 models where RoPE immediately follows normalization.

---

### `ttnn.experimental.dit_layernorm_pre_allgather`

**Purpose**: Computes local Welford statistics (running mean and M2) for distributed LayerNorm. Uses a precomputed reciprocal tensor for efficient online variance computation.

**Where used**:
- `layers/normalization.py` -- `DistributedLayerNorm.forward`

**Key parameters**:
```python
ttnn.experimental.dit_layernorm_pre_allgather(
    x,                          # Input activation (device-local shard)
    recip_tensor,               # Precomputed reciprocals for Welford algorithm
    compute_kernel_config=...,
)
```

**TT-Symbiote equivalent**: **None.** No distributed LayerNorm exists in TT-Symbiote.

---

### `ttnn.experimental.dit_layernorm_post_allgather`

**Purpose**: Applies LayerNorm using globally-gathered Welford statistics. Supports dynamic weight/bias for adaptive layer normalization (adaLN).

**Where used**:
- `layers/normalization.py` -- `DistributedLayerNorm.forward`

**Key parameters**:
```python
ttnn.experimental.dit_layernorm_post_allgather(
    x, stats,
    weight=..., bias=...,
    epsilon=1e-5,
    compute_kernel_config=...,
    dtype=None,                 # Optional output dtype
)
```

**TT-Symbiote equivalent**: **None.**

---

## 3. Convolution

### `ttnn.experimental.conv3d`

**Purpose**: 3D convolution for video models. Processes spatio-temporal data in NTHWC (batch, time, height, width, channels) format.

**Where used**:
- `layers/conv3d.py` -- `ContextParallelConv3d.forward`
- `models/vae/vae_wan2_1.py` -- Wan VAE encoder/decoder blocks

**Key parameters**:
```python
ttnn.experimental.conv3d(
    input_tensor=x_pad_NTHWC,
    weight_tensor=...,
    bias_tensor=...,
    config=ttnn.Conv3dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=1, W_out_block=16, H_out_block=2,
        C_out_block=96, C_in_block=128,
        compute_with_storage_grid_size=grid_size,
    ),
    output_channels=...,
    kernel_size=(3, 3, 3),
    stride=(1, 1, 1),
    padding=(0, 1, 1),
    padding_mode="replicate",
    dtype=ttnn.bfloat16,
    groups=1,
    compute_kernel_config=...,
)
```

The `ttnn.Conv3dConfig` provides detailed control over blocking:
- `C_in_block` / `C_out_block`: channel-dimension blocking for L1 tiling.
- `T_out_block` / `H_out_block` / `W_out_block`: spatial-dimension output blocking.
- `compute_with_storage_grid_size`: core grid for the operation.

**TT-Symbiote equivalent**: **None.** TT-Symbiote has no Conv3d support. Video model VAEs cannot be ported without either:
1. Implementing a `TTNNConv3d` module wrapping `ttnn.experimental.conv3d`, or
2. Decomposing Conv3d into a sequence of Conv2d operations (lossy and slow).

---

## 4. Attention Utilities

### `ttnn.experimental.nlp_create_qkv_heads`

**Purpose**: Reshapes a concatenated QKV tensor into separate Q, K, V head tensors with the layout expected by `ttnn.transformer.scaled_dot_product_attention`. Handles the permutation from `[B, S, 3*H*D]` to three tensors of shape `[B, H, S, D]`.

**Where used**:
- `encoders/qwen25vl/model_qwen25vl.py` -- Qwen2.5-VL encoder attention
- `models/transformers/wan2_2/attention_wan.py` -- Wan2.2 attention

**Key parameters**:
```python
q, k, v = ttnn.experimental.nlp_create_qkv_heads(
    qkv_tensor,
    num_heads=num_heads,
    num_kv_heads=num_kv_heads,  # For GQA/MQA
    transpose_k=True,           # Whether to transpose K for attention
)
```

**TT-Symbiote equivalent**: **Yes.** TT-Symbiote uses the same op in its attention modules:
```python
queries, keys, values = ttnn.experimental.nlp_create_qkv_heads(
    qkv_tensor, num_heads=..., num_kv_heads=..., transpose_k=True)
```

This is one of the few `ttnn.experimental` ops used by both frameworks.

---

### `ttnn.experimental.rotary_embedding_llama`

**Purpose**: Applies rotary positional embeddings (RoPE) to query and key tensors using the LLaMA-style implementation. Operates on tensors in `[B, H, S, D]` format.

**Where used**:
- `models/transformers/attention_mochi.py` -- Mochi attention
- `models/transformers/wan2_2/attention_wan.py` -- (via post-allgather RoPE fusion in DistributedRMSNorm)

**Key parameters**:
```python
q = ttnn.experimental.rotary_embedding_llama(
    q, cos, sin, trans_mat, is_decode_mode=False)
```

**TT-Symbiote equivalent**: **Yes.** TT-Symbiote uses the same op extensively:
```python
query_states = ttnn.experimental.rotary_embedding_llama(
    query_states, cos, sin, trans_mat, is_decode_mode=False)
```

Both frameworks call the identical underlying TTNN kernel.

---

### `ttnn.experimental.nlp_concat_heads`

**Purpose**: Reverses `nlp_create_qkv_heads` by concatenating multi-head attention output back from `[B, H, S, D]` to `[B, S, H*D]`.

**Where used**: Not directly in TT-DiT's core layers (done manually via reshape), but used in TT-Symbiote's attention modules.

**TT-Symbiote equivalent**: **Yes.** Used in `TTNNSelfAttention`:
```python
context_layer = ttnn.experimental.nlp_concat_heads(context_layer)
```

---

## 5. Fused Operations

### `ttnn.experimental.dit_minimal_matmul_addcmul_fused`

**Purpose**: A fused kernel that performs matmul + elementwise `add-cmul` (add and component-wise multiply) in a single pass. Used in Wan2.2 attention to apply adaptive modulation after the output projection.

**Where used**:
- `models/transformers/wan2_2/attention_wan.py`

**Key parameters**:
```python
output = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
    input_tensor, weight_tensor,
    residual,           # Tensor to add
    modulation,         # Tensor for component-wise multiply
    config=matmul_config,
    compute_kernel_config=...,
)
```

The fused operation computes: $\text{output} = \text{residual} + (\text{input} \times \text{weight}) \odot \text{modulation}$

**TT-Symbiote equivalent**: **None.** This would need to be decomposed into separate matmul, multiply, and add operations:
```python
mm = ttnn.linear(input_tensor, weight)
scaled = ttnn.multiply(mm, modulation)
output = ttnn.add(residual, scaled)
```

The decomposed version requires three kernel launches instead of one.

---

## 6. Collective Communication

### `ttnn.experimental.all_gather_async`

**Purpose**: Asynchronous all-gather operation used throughout TT-DiT's transformer and attention layers. Unlike the synchronous `ttnn.all_gather`, this variant returns immediately and overlaps communication with computation.

**Where used**:
- `parallel/manager.py` -- `CCLManager.all_gather_persistent_buffer`
- `models/transformers/transformer_sd35.py` -- SD3.5 transformer blocks
- `models/transformers/transformer_mochi.py` -- Mochi transformer blocks
- `models/transformers/attention_sd35.py` -- SD3.5 attention
- `models/transformers/attention_mochi.py` -- Mochi attention
- `encoders/clip/model_clip.py` -- CLIP encoder

**Key parameters**:
```python
tensor = ttnn.experimental.all_gather_async(
    tensor,
    dim=dim,
    mesh_device=self.mesh_device,
    cluster_axis=mesh_axis,
    num_links=1,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    topology=ttnn.Topology.Ring,
    num_workers=1,
    num_buffers_per_channel=2,
    persistent_output_tensor=cached_buffer,  # Reuse pre-allocated output
)
```

**TT-Symbiote equivalent**: `ttnn.all_gather` (synchronous, stable API):
```python
ttnn.all_gather(tt_stats, dim=-1, num_links=1, topology=ttnn.Topology.Ring)
```

**Difference**: The async variant enables communication-computation overlap, which is critical for TT-DiT's pipelined execution. TT-Symbiote uses synchronous collectives. The `persistent_output_tensor` parameter in the async variant enables buffer reuse across iterations (trace-compatible).

---

### `ttnn.experimental.reduce_scatter_minimal_async`

**Purpose**: Asynchronous reduce-scatter used by `CCLManager.reduce_scatter_persistent_buffer`. Reduces partial results across devices and scatters the output.

**Where used**:
- `parallel/manager.py` -- `CCLManager.reduce_scatter_persistent_buffer`

**Key parameters**:
```python
tensor = ttnn.experimental.reduce_scatter_minimal_async(
    tensor,
    dim=dim,
    mesh_device=self.mesh_device,
    cluster_axis=mesh_axis,
    num_links=1,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    topology=ttnn.Topology.Ring,
    num_workers=1,
    num_buffers_per_channel=2,
    persistent_output_tensor=cached_buffer,
)
```

**TT-Symbiote equivalent**: `ttnn.reduce_scatter` (synchronous, stable API):
```python
ttnn.reduce_scatter(tt_output, dim=3, num_links=1,
                     cluster_axis=1, topology=ttnn.Topology.Ring)
```

---

### `ttnn.experimental.neighbor_pad_async`

**Purpose**: Asynchronous halo exchange for context-parallel convolution. Pads a tensor by receiving neighbor data from adjacent devices in the mesh. Used for temporal padding in Conv3d context parallelism.

**Where used**:
- `models/vae/vae_wan2_1.py` -- Wan VAE temporal halo exchange
- `parallel/config.py` -- `vae_neighbor_pad` utility

**Key parameters**:
```python
x_pad = ttnn.experimental.neighbor_pad_async(
    x,
    dim=dim,
    cluster_axis=mesh_axis,
    padding_left=2,
    padding_right=0,
    padding_mode="replicate",
    mesh_device=self.mesh_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    topology=ttnn.Topology.Ring,
)
```

**TT-Symbiote equivalent**: **None.** Context parallelism and halo exchange are TT-DiT-exclusive features.

---

### `ttnn.experimental.slice_reshard_async`

**Purpose**: Asynchronous slice and reshard operation for redistributing tensor slices across devices. Used for spatial parallelism in VAE models.

**Where used**:
- `parallel/config.py` -- `vae_reshard` utility

**TT-Symbiote equivalent**: **None.**

---

## Summary Table

| Experimental Op | Category | TT-Symbiote Equivalent | Porting Impact |
|---|---|---|---|
| `minimal_matmul` | Matmul | `ttnn.linear` | Low (functional, perf difference) |
| `minimal_matmul_split` | Matmul | None (matmul + chunk) | Low |
| `wan_fused_rmsnorm_pre_allgather` | Norm | `ttnn.rms_norm_pre_all_gather` | Medium (different API) |
| `wan_fused_rmsnorm_post_allgather` | Norm | `ttnn.rms_norm_post_all_gather` | Medium (no RoPE fusion) |
| `dit_layernorm_pre_allgather` | Norm | None | High (no equivalent) |
| `dit_layernorm_post_allgather` | Norm | None | High (no equivalent) |
| `conv3d` | Convolution | None | High (video models only) |
| `nlp_create_qkv_heads` | Attention | Same op (shared) | None |
| `rotary_embedding_llama` | Attention | Same op (shared) | None |
| `nlp_concat_heads` | Attention | Same op (shared) | None |
| `dit_minimal_matmul_addcmul_fused` | Fused | None (decompose) | Medium |
| `all_gather_async` | CCL | `ttnn.all_gather` (sync) | Medium (no async overlap) |
| `reduce_scatter_minimal_async` | CCL | `ttnn.reduce_scatter` (sync) | Medium (no async overlap) |
| `neighbor_pad_async` | CCL | None | High (context parallelism) |
| `slice_reshard_async` | CCL | None | High (spatial parallelism) |

---

## Key Takeaways

1. **`minimal_matmul` is the most pervasive experimental op**: Every linear layer in TT-DiT uses it. Replacing it with `ttnn.linear` is functionally correct but may degrade performance because `ttnn.linear` cannot accept explicit block-size configurations. Performance-sensitive porting may require keeping `minimal_matmul`.

2. **Three attention ops are shared**: `nlp_create_qkv_heads`, `rotary_embedding_llama`, and `nlp_concat_heads` are used by both frameworks. These require no porting effort.

3. **Distributed norm ops are the hardest gap**: The `dit_layernorm_*` ops have no equivalent anywhere in TT-Symbiote. Porting models that use `DistributedLayerNorm` (SD3.5, Flux) requires either wrapping the experimental ops directly or implementing a decomposed distributed LayerNorm from scratch.

4. **Async CCL ops trade portability for performance**: TT-DiT's `all_gather_async` and `reduce_scatter_minimal_async` enable communication-computation overlap. TT-Symbiote's synchronous collectives are functional equivalents but cannot achieve the same throughput. This is a performance gap, not a correctness gap.

5. **Video model ops have no path in TT-Symbiote**: `conv3d`, `neighbor_pad_async`, and `slice_reshard_async` are all required for Mochi/Wan video VAEs and have no TT-Symbiote equivalent. Porting video models is blocked on these ops.

---

**Next:** [`convolution_layers.md`](./convolution_layers.md)
