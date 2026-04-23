# Joint Attention

## Prerequisites

- [Chapter 1 -- Module and Parameter](../ch1_architecture_overview/module_and_parameter.md): `Module`, `Parameter`, `UnregisteredModule`, and `_prepare_torch_state` lifecycle.
- [Chapter 2 -- Parallel Linear Layers](../ch2_parallelism_and_ccl/parallel_linear_layers.md): `ColParallelLinear` and tensor-parallel column fracturing.
- [Chapter 3 -- Normalization Layers](../ch3_custom_layers_and_ops/normalization_layers.md): `RMSNorm` (single-device, per-head variant).
- [Chapter 4 -- Index](./index.md): overview of how DiT joint attention differs from LLM attention.

---

## Overview

The `Attention` class in `blocks/attention.py` implements the joint spatial-prompt attention mechanism that is the core innovation of Diffusion Transformer models. It is adapted from the HuggingFace Diffusers `AttentionProcessor` but rewritten entirely against TTNN operations for Tenstorrent hardware.

The class handles:

1. **Fused QKV projection** for both spatial and prompt streams.
2. **Per-head RMSNorm** on Q and K tensors.
3. **Head padding** to align with hardware tile boundaries.
4. **RoPE** application on spatial and prompt sequences independently.
5. **Joint SDPA** via two execution paths (standard and ring/SP).
6. **Post-attention output projections** for both streams.
7. **Weight sharing** via `UnregisteredModule` for models that reuse spatial weights for the prompt path.

Source: `models/tt_dit/blocks/attention.py`

---

## Constructor Parameters

```python
class Attention(Module):
    def __init__(
        self,
        *,
        query_dim: int,           # Hidden dimension of input spatial/prompt tokens
        head_dim: int,            # Dimension per attention head
        heads: int,               # Number of attention heads (before padding)
        out_dim: int,             # Output dimension
        added_kv_proj_dim: int,   # Hidden dim for prompt QKV (0 = no prompt path)
        context_pre_only: bool,   # If True, prompt has no output projection
        pre_only: bool,           # If True, spatial has no output projection
        use_spatial_weights_for_prompt: bool,  # Share weights via UnregisteredModule
        context_head_scaling: bool,            # Per-head scaling on prompt Q
        eps: float,               # RMSNorm epsilon
        mesh_device, ccl_manager, parallel_config, padding_config,
        k_chunk_size: int = 512,  # SDPA K chunk size
        q_chunk_size: int = 128,  # SDPA Q chunk size
        is_fsdp: bool = False,    # Full Sharded Data Parallel mode
    )
```

Key observations:

- **`query_dim` equals `out_dim`** in all current models -- the attention block preserves dimensionality.
- **`added_kv_proj_dim`** controls whether a prompt path exists. When set to 0, the class creates zero-sized placeholder tensors for the prompt side.
- **`pre_only`** and **`context_pre_only`** allow the final transformer block to skip the output projections that would normally project the attention output back to the hidden dimension.
- **`use_spatial_weights_for_prompt`** enables weight sharing (see the [UnregisteredModule section](#unregisteredmodule-weight-sharing) below).

---

## Fused QKV Projection

Instead of three separate linear layers (`to_q`, `to_k`, `to_v`), TT-DiT fuses them into a single `ColParallelLinear`:

```python
self.to_qkv = ColParallelLinear(query_dim, 3 * padded_inner_dim, mesh_axis=tp_axis, **common_args)
```

where `padded_inner_dim = head_dim * padded_heads`. The output has shape `[batch, seq_len, 3 * n_local_heads * head_dim]` after tensor-parallel sharding.

The prompt stream has its own fused projection:

```python
self.add_qkv_proj = ColParallelLinear(added_kv_proj_dim, 3 * padded_inner_dim, ...)
```

After the linear projection, `ttnn.transformer.split_query_key_value_and_split_heads` splits the fused output into separate Q, K, V tensors and reshapes them into multi-head format:

```python
qkv = self.to_qkv(spatial)
# shape: [batch, spatial_seq_len / sp_factor, 3 * n_local_heads * head_dim]

q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
    qkv, num_heads=local_heads, transpose_key=False
)
# each shape: [batch, n_local_heads, spatial_seq_len / sp_factor, head_dim]
```

Note `transpose_key=False` -- the key is not transposed at split time. The SDPA kernel handles the transpose internally.

---

## Per-Head RMSNorm

After splitting Q, K, V, TT-DiT applies RMSNorm independently to Q and K:

```python
q = self.norm_q(q)
k = self.norm_k(k)
```

Each `norm_q` and `norm_k` is an instance of `RMSNorm(embedding_dim=head_dim, ...)`. Since the input tensor at this point has shape `[batch, n_local_heads, seq_len, head_dim]`, the norm operates on the last dimension (`head_dim`) independently for each head.

This per-head normalization serves two purposes:

1. **Training stability**: it prevents attention logits from growing unboundedly across heads, which is especially important in DiT models where the spatial and prompt sequences can have very different magnitudes.
2. **Complements $1/\sqrt{d}$ scaling**: by normalizing Q and K to unit RMS, the dot products are naturally bounded. Note that both mechanisms are active simultaneously -- TT-DiT passes no explicit `scale` to `joint_scaled_dot_product_attention`, so the kernel applies the default $1/\sqrt{d_k}$ scaling in addition to the per-head RMSNorm.

The same normalization is applied to the prompt stream:

```python
add_q = self.norm_added_q(add_q)
add_k = self.norm_added_k(add_k)
```

---

## Head Padding

TT hardware operates on tiles of 32x32 elements. When the number of attention heads is not a multiple of the tile size or the tensor-parallel factor, `PaddingConfig` pads the head count:

```python
self.padded_heads = padding_config.target_heads if padding_config is not None else heads
self.n_local_heads = self.padded_heads // self.parallel_config.tensor_parallel.factor
```

For example, if a model has 24 heads and TP factor 8, we need `padded_heads` to be divisible by 8 and by the tile width. The padding config may round this up to 32, giving `n_local_heads = 4` per device.

The padding is applied during weight loading (see `_reshape_and_merge_qkv` below) and also to `context_head_factors`:

```python
if self.padding_config is not None:
    pad = (0, self.padding_config.head_padding)
    factors = torch.nn.functional.pad(factors, pad)
```

---

## Weight Preparation: `_prepare_torch_state` and `_reshape_and_merge_qkv`

HuggingFace Diffusers stores separate `to_q.weight`, `to_k.weight`, `to_v.weight` parameters. TT-DiT's fused `to_qkv` expects a single merged weight. The `_prepare_torch_state` method bridges this gap:

```python
def _prepare_torch_state(self, state):
    weight, bias = self._reshape_and_merge_qkv(
        pop_substate(state, "to_q"),
        pop_substate(state, "to_k"),
        pop_substate(state, "to_v"),
    )
    state["to_qkv.weight"] = weight
    state["to_qkv.bias"] = bias
```

The `_reshape_and_merge_qkv` method performs a non-trivial interleaving. The goal is to arrange the fused QKV weight so that column-fracturing (the TP sharding strategy of `ColParallelLinear`) cleanly splits along the head dimension. The algorithm:

1. **Transpose** each of Q, K, V weights to `[input_dim, output_dim]`.
2. **Pad** the output dimension to match `padded_heads * head_dim` using `pad_weight_tensor`.
3. **Reshape** to `[input_dim, n_devices, n_local_heads, head_dim]`.
4. **Concatenate** Q, K, V along the `n_local_heads` axis: `[input_dim, n_devices, 3 * n_local_heads, head_dim]`.
5. **Flatten** back to `[input_dim, 3 * padded_heads * head_dim]`.
6. **Transpose** back to `[3 * padded_heads * head_dim, input_dim]`.

This interleaving ensures that when `ColParallelLinear` fractures the output dimension across devices, each device gets a contiguous block of `[3 * n_local_heads * head_dim]` -- its own slice of Q, K, V heads. The `split_query_key_value_and_split_heads` kernel then separates these cleanly.

The output projection weights (`to_out`, `to_add_out`) also require padding on the input dimension when `PaddingConfig` is active, since they receive the concatenated multi-head output which has padded head count.

---

## RoPE Application

RoPE is applied **after** QKV split and per-head normalization, but **before** the joint SDPA. Crucially, spatial and prompt sequences receive different RoPE embeddings:

```python
if spatial_rope is not None:
    q = _apply_rope(q, spatial_rope)
    k = _apply_rope(k, spatial_rope)

# ... prompt path ...
if prompt_rope is not None:
    add_q = _apply_rope(add_q, prompt_rope)
    add_k = _apply_rope(add_k, prompt_rope)
```

The `_apply_rope` function implements the standard rotary embedding formula using TTNN operations:

```python
def _apply_rope(x, freqs_cis):
    cos, sin = freqs_cis
    cos = cos.reshape([1, 1, *cos.shape])
    sin = sin.reshape([1, 1, *sin.shape])
    return x * cos + ttnn.alt_complex_rotate90(x) * sin
```

This computes:

$$\text{RoPE}(x) = x \cdot \cos(\theta) + \text{rotate90}(x) \cdot \sin(\theta)$$

where `ttnn.alt_complex_rotate90` performs the complex-number rotation that swaps and negates alternating pairs of elements in the head dimension. The `cos` and `sin` tensors have shape `[seq_len, head_dim]` and are broadcast over `[batch, heads, seq_len, head_dim]`.

The spatial RoPE typically encodes 2D or 3D positional information (row, column, and optionally frame index for video), while the prompt RoPE encodes 1D sequential position. These are completely independent position embedding spaces.

---

## Context Head Factors

For models with `context_head_scaling=True`, the prompt query is multiplied by learned per-head scaling factors after normalization and RoPE:

```python
if self.context_head_factors is not None:
    add_q = add_q * self.context_head_factors.data
```

The `context_head_factors` parameter has shape `[padded_heads, 1, 1]` (sharded across TP on the head axis) and is broadcast over the `[batch, n_local_heads, prompt_seq_len, head_dim]` tensor.

This mechanism allows the model to learn per-head importance weighting for the prompt conditioning. Some heads may attend more strongly to text conditioning while others focus on spatial self-attention. This parameter is used in SD3.5 and Motif models.

---

## UnregisteredModule: Weight Sharing

When `use_spatial_weights_for_prompt=True`, the prompt path reuses the exact same weights as the spatial path:

```python
if use_spatial_weights_for_prompt:
    self.add_qkv_proj = UnregisteredModule(self.to_qkv)
    self.norm_added_q = UnregisteredModule(self.norm_q)
    self.norm_added_k = UnregisteredModule(self.norm_k)
    self.to_add_out = UnregisteredModule(self.to_out) if self.to_out is not None else None
```

`UnregisteredModule` (from `layers/module.py`) wraps a `Module` without registering it in the parent's `_children` dictionary. This means:

- **Forward pass**: `self.add_qkv_proj(prompt)` calls `self.to_qkv(prompt)` -- the same weights are used.
- **Weight loading**: `_load_torch_state_dict_inner` does not recurse into `UnregisteredModule` children, so no duplicate weights are loaded. The prompt path picks up the spatial weights automatically.
- **Memory**: only one copy of the weights exists in device memory.

Without `UnregisteredModule`, a naive `self.add_qkv_proj = self.to_qkv` would register `to_qkv` under both names in the children dictionary, causing double-loading attempts.

---

## SDPA Execution Paths

The forward method has two SDPA execution paths depending on whether sequence parallelism (SP) is active.

### Path 1: Standard Joint SDPA (SP=1)

When running without sequence parallelism:

```python
spatial, prompt = ttnn.transformer.joint_scaled_dot_product_attention(
    q, k, v,           # spatial Q, K, V
    add_q, add_k, add_v,  # prompt Q, K, V
    joint_strategy="rear",
    program_config=self.sdpa_program_config,
    compute_kernel_config=self.sdpa_compute_kernel_config,
)
```

This TTNN operation internally concatenates the spatial and prompt K/V along the sequence dimension, then computes attention for the spatial Q against the combined K/V, and separately for the prompt Q against the combined K/V. The `joint_strategy="rear"` means the prompt K/V are appended **after** the spatial K/V.

Mathematically, for spatial query $Q_s$ and prompt query $Q_p$, with combined keys $K = [K_s; K_p]$ and values $V = [V_s; V_p]$:

$$\text{Attn}_s = \text{softmax}\left(\frac{Q_s K^T}{\sqrt{d}}\right) V$$

$$\text{Attn}_p = \text{softmax}\left(\frac{Q_p K^T}{\sqrt{d}}\right) V$$

Both spatial and prompt tokens attend to the full combined sequence, enabling bidirectional information flow between the image and text modalities.

### Path 2: Ring Joint SDPA (SP > 1)

When sequence parallelism is active, the spatial sequence is sharded across devices along the sequence dimension. The ring attention algorithm performs SDPA without materializing the full sequence on any single device:

```python
spatial, prompt, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
    q, k, v,
    add_q, add_k, add_v,
    persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(...),
    persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(...),
    joint_strategy="rear",
    logical_n=spatial_sequence_length,
    ...
    cluster_axis=self.parallel_config.sequence_parallel.mesh_axis,
    mesh_device=self.mesh_device,
    topology=self.ccl_manager.topology,
    subdevice_id=self.ccl_manager.ccl_sub_device_id,
    ccl_core_grid_offset=(0, self.sdpa_worker_grid[1]),
)
```

Key differences from the standard path:

- **`logical_n`**: the full spatial sequence length before sharding. Each device only holds `spatial_sequence_length / sp_factor` tokens.
- **Persistent ping-pong buffers**: pre-allocated device memory for K and V chunks that are rotated around the ring.
- **CCL parameters**: cluster axis, semaphores, sub-device ID, and core grid offset for inter-device communication.
- **Returns LSE**: the log-sum-exp values from the chunked softmax computation (used for numerical stability across ring steps).

The prompt sequence is **not** sharded across SP devices -- it is replicated on all devices. Only the spatial sequence benefits from sequence parallelism.

---

## SDPA Configuration

The SDPA kernels use a specific program configuration:

```python
self.sdpa_program_config = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=self.sdpa_worker_grid,
    q_chunk_size=q_chunk_size,   # default 128
    k_chunk_size=k_chunk_size,   # default 512
    exp_approx_mode=False,
)
```

- **`sdpa_worker_grid`**: uses all columns of the compute grid, but reserves the last row (`y - 1`) -- this row is used for CCL communication in ring attention.
- **`q_chunk_size=128`**: the query is processed in chunks of 128 tokens.
- **`k_chunk_size=512`**: the key/value are processed in chunks of 512 tokens. This asymmetry (Q chunks smaller than K chunks) reflects the memory access pattern: K/V are streamed from DRAM, while Q stays in L1.
- **`exp_approx_mode=False`**: uses exact exponential computation rather than a polynomial approximation.

The compute kernel is configured for Wormhole hardware:

```python
self.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
)
```

Note `HiFi2` (not `HiFi4`) and `fp32_dest_acc_en=False`. This trades some numerical precision for throughput. The comment in the source notes that `fp32_dest_acc_en=True` should be tried if correctness issues arise.

---

## Post-Attention Projections

After SDPA, the multi-head outputs are concatenated back into the hidden dimension:

```python
spatial = ttnn.transformer.concatenate_heads(spatial)
prompt = ttnn.transformer.concatenate_heads(prompt)
```

This reverses the head split: `[batch, n_local_heads, seq_len, head_dim]` becomes `[batch, seq_len, n_local_heads * head_dim]`.

The output projections then map back to the model's hidden dimension. Because these are `ColParallelLinear` layers (fractured on the output dimension), an all-gather is needed first to reconstruct the full input:

```python
if self.to_out is not None:
    spatial = self.ccl_manager.all_gather_persistent_buffer(
        spatial, dim=2, mesh_axis=tp_axis, use_hyperparams=True
    )
    spatial = self.to_out(spatial)
```

The same pattern applies to `to_add_out` for the prompt stream. When `pre_only=True` (spatial) or `context_pre_only=True` (prompt), the corresponding output projection is `None` and skipped.

---

## Spatial Sequence Padding Utility

The class provides static methods for padding the spatial sequence to meet SP and SDPA chunk alignment requirements:

```python
@classmethod
def spatial_sequence_padding_length(cls, *, length, sp_factor, k_chunk_size=512):
    if sp_factor == 1:
        return 0
    divisor = k_chunk_size * sp_factor
    return -length % divisor
```

The spatial sequence must be padded to a multiple of `k_chunk_size * sp_factor`. For example, with `sp_factor=4` and `k_chunk_size=512`, the sequence must be a multiple of 2048 tokens. This ensures that when the sequence is split across 4 devices, each shard is a multiple of the K chunk size.

---

## Key Takeaways

1. **Fused QKV with interleaved weight layout**: the `_reshape_and_merge_qkv` method carefully interleaves Q, K, V weights so that column-parallel fracturing produces correct per-device head slices. This is a non-trivial weight transformation that must be replicated exactly for any port.

2. **Per-head RMSNorm complements $1/\sqrt{d}$ scaling**: unlike LLM attention, DiT attention normalizes Q and K per head in addition to the standard $1/\sqrt{d_k}$ scaling (both are active simultaneously). This changes the attention distribution and must be preserved during porting.

3. **Two SDPA kernels serve different parallelism modes**: `joint_scaled_dot_product_attention` handles the single-device/TP-only case, while `ring_joint_scaled_dot_product_attention` adds sequence parallelism with ring-style K/V rotation and persistent ping-pong buffers.

4. **UnregisteredModule enables zero-cost weight sharing**: models that use the same projections for spatial and prompt paths avoid memory duplication and double-loading without any special handling in the weight loading code.

5. **Prompt sequence is never sharded for SP**: sequence parallelism only applies to the spatial dimension. The prompt sequence is replicated across all SP devices and participates in attention on every device.

---

**Next:** [`transformer_block.md`](./transformer_block.md)
