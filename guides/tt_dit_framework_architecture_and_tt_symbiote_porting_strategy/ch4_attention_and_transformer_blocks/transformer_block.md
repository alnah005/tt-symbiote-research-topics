# Transformer Block

## Prerequisites

- [Chapter 4 -- Joint Attention](./joint_attention.md): understanding of the `Attention` class, joint SDPA, and per-head RMSNorm.
- [Chapter 3 -- Normalization Layers](../ch3_custom_layers_and_ops/normalization_layers.md): understanding of `DistributedLayerNorm` and its two-phase all-gather pattern.
- [Chapter 2 -- Parallel Linear Layers](../ch2_parallelism_and_ccl/parallel_linear_layers.md): `ColParallelLinear` for the modulation projections.
- [Chapter 2 -- CCL Manager](../ch2_parallelism_and_ccl/ccl_manager.md): `CCLManager.all_gather_persistent_buffer` for gathering distributed activations before attention and feedforward.

---

## Overview

The `TransformerBlock` class in `blocks/transformer_block.py` is the fundamental repeating unit of every DiT model in TT-DiT. It combines:

1. **Adaptive Layer Normalization (adaLN)** -- time-conditioned modulation that shifts, scales, and gates the hidden states based on the diffusion timestep embedding.
2. **Joint Attention** -- the `Attention` class described in [joint_attention.md](./joint_attention.md).
3. **Feedforward Network** -- a `ParallelFeedForward` with megatron-style parallelism.
4. **Dual-stream processing** -- the spatial and prompt sequences are processed in parallel through symmetric sub-blocks, with interaction only at the attention stage.

Source: `models/tt_dit/blocks/transformer_block.py`

---

## Constructor: Building the Block

```python
class TransformerBlock(Module):
    def __init__(
        self,
        *,
        dim: int,                    # Hidden dimension
        modulation_dim: int | None,  # Dimension of time embedding (defaults to dim)
        num_heads: int,              # Number of attention heads
        head_dim: int,               # Dimension per head
        context_pre_only: bool,      # Final block: no prompt output path
        add_attention_to_output: bool = True,  # Whether attention is residual
        context_head_scaling: bool = False,     # Per-head scaling on prompt Q
        ff_activation_fn: str = "gelu",        # Feedforward activation
        mesh_device, ccl_manager, parallel_config, padding_config,
        attention_k_chunk_size: int = 512,
        attention_q_chunk_size: int = 128,
        is_fsdp: bool = False,
    )
```

The `modulation_dim` parameter defaults to `dim` when not specified. This allows models where the time embedding has a different width from the hidden dimension to use a different projection size.

The `add_attention_to_output` parameter controls whether the attention output is added as a residual to the hidden state before the feedforward sub-block. When `True` (the default), the spatial hidden state used by the feedforward norm is the post-attention residual. When `False`, the original pre-attention hidden state is used for the feedforward norm input, and only the final feedforward residual updates the output. This is an architecture-specific knob used by some model variants.

---

## Adaptive Layer Normalization (adaLN)

The core innovation of DiT models is **adaptive layer normalization** -- the normalization parameters are not learned constants but are dynamically generated from the diffusion timestep embedding. This conditions every layer of the transformer on the current noise level.

### Modulation Projection

The time embedding is projected to produce six modulation parameters for the spatial stream and six (or two) for the prompt stream:

```python
self.norm1_linear = ColParallelLinear(modulation_dim, 6 * dim, ...)
self.norm1_context_linear = ColParallelLinear(
    modulation_dim,
    6 * dim if not context_pre_only else 2 * dim,
    ...
)
```

The `norm1_linear` projects the time embedding from `modulation_dim` to `6 * dim`, producing six sets of modulation parameters, each of width `dim`. These are used by `prepare_chunked_linear_output` in `_prepare_torch_state` to ensure correct TP sharding.

For the prompt stream, when `context_pre_only=True` (the final transformer block), only two modulation parameters are needed (shift and scale for the pre-attention norm -- there is no gate, feedforward, or output path).

### Forward: Time Embedding Processing

```python
if not skip_time_embed_activation_fn:
    time_embed = ttnn.silu(time_embed, memory_config=ttnn.DRAM_MEMORY_CONFIG)

spatial_time = self.norm1_linear(time_embed)
prompt_time = self.norm1_context_linear(time_embed)
```

The time embedding first passes through SiLU activation, then through the modulation linear layer. The `skip_time_embed_activation_fn` flag allows layers to share a pre-activated time embedding when multiple blocks use the same activation.

### Chunking into Shift, Scale, Gate

The projected time embedding is split into six equal chunks:

```python
(
    spatial_shift_attn,    # shift for attention sub-block
    spatial_scale_attn,    # scale for attention sub-block
    spatial_gate_attn,     # gate for attention output
    spatial_shift_ff,      # shift for feedforward sub-block
    spatial_scale_ff,      # scale for feedforward sub-block
    spatial_gate_ff,       # gate for feedforward output
) = _chunk_time3d(spatial_time, 6)
```

The helper `_chunk_time3d` performs a simple slice along the last dimension:

```python
def _chunk_time3d(t, count):
    size = t.shape[-1] // count
    return [t[:, :, i * size : (i + 1) * size] for i in range(count)]
```

Each chunk has shape `[batch, 1, dim / tp_factor]` (the sequence dimension is 1 because the time embedding is the same for all tokens).

---

## Spatial Attention Sub-Block

The attention sub-block follows this sequence:

### Step 1: Adaptive LayerNorm

```python
spatial_normed = ttnn.squeeze(
    self.norm1_norm(
        ttnn.unsqueeze(spatial, 0),
        dynamic_weight=(1 + spatial_scale_attn),
        dynamic_bias=spatial_shift_attn,
    ),
    0,
)
```

The `DistributedLayerNorm.forward()` accepts `dynamic_weight` and `dynamic_bias` parameters (see [Chapter 3 -- Normalization Layers](../ch3_custom_layers_and_ops/normalization_layers.md)). The modulation formula is:

$$\text{adaLN}(x) = (1 + \gamma) \cdot \text{LayerNorm}(x) + \beta$$

where $\gamma = \text{spatial\_scale\_attn}$ and $\beta = \text{spatial\_shift\_attn}$. Note the `(1 + scale)` formulation: at initialization (when the scale is near zero), this approximates standard LayerNorm with no modulation.

The `unsqueeze/squeeze` around the norm call is required because `DistributedLayerNorm` expects a 4D input `[1, batch, seq_len, dim]` while the rest of the block works with 3D tensors `[batch, seq_len, dim]`.

The `norm1_norm` instance is a `DistributedLayerNorm` with `norm_elementwise_affine=False` and `bias=False` -- it has no learned weight or bias parameters, because these are replaced by the dynamic modulation.

### Step 2: All-Gather Before Attention

```python
spatial_normed = self.ccl_manager.all_gather_persistent_buffer(
    spatial_normed, dim=2, mesh_axis=tp_axis, use_hyperparams=True
)
```

Because the LayerNorm output is TP-sharded (each device holds `dim / tp_factor`), we must all-gather to reconstruct the full hidden dimension before feeding into `Attention`. The attention's `to_qkv` is a `ColParallelLinear` that expects the full input dimension.

### Step 3: Joint Attention

```python
spatial_attn, prompt_attn = self.attn.forward(
    spatial=spatial_normed,
    prompt=prompt_normed,
    spatial_rope=spatial_rope,
    prompt_rope=prompt_rope,
    spatial_sequence_length=spatial_sequence_length,
)
```

See [joint_attention.md](./joint_attention.md) for the full details.

### Step 4: Gating and Residual

```python
spatial_attn = spatial_attn * spatial_gate_attn
spatial_plus_attn = spatial + spatial_attn
if self.add_attention_to_output:
    spatial = spatial_plus_attn
```

The gate is element-wise multiplication with the time-conditioned gate parameter. This allows the model to dynamically control how much of the attention output contributes to the residual, conditioned on the diffusion timestep.

---

## Spatial Feedforward Sub-Block

### Step 1: Adaptive LayerNorm (FF)

```python
spatial_normed = ttnn.squeeze(
    self.norm2(
        ttnn.unsqueeze(spatial_plus_attn, 0),
        dynamic_weight=(1 + spatial_scale_ff),
        dynamic_bias=spatial_shift_ff,
    ),
    0,
)
```

Note that the feedforward norm is applied to `spatial_plus_attn` (the post-attention residual), regardless of the `add_attention_to_output` flag. This means the feedforward always sees the attention-updated representation.

### Step 2: All-Gather and Feedforward

```python
spatial_normed = self.ccl_manager.all_gather_persistent_buffer(
    spatial_normed, dim=2, mesh_axis=tp_axis, use_hyperparams=True
)
spatial_ff = ttnn.squeeze(self.ff(ttnn.unsqueeze(spatial_normed, 0)), 0)
spatial_ff = spatial_ff * spatial_gate_ff
spatial = spatial + spatial_ff
```

The `ParallelFeedForward` (`self.ff`) implements a megatron-style parallel MLP with `ColParallelLinear` for the up-projection and `RowParallelLinear` for the down-projection. See [Chapter 2 -- Parallel Linear Layers](../ch2_parallelism_and_ccl/parallel_linear_layers.md) for details. The `unsqueeze/squeeze` is again for dimension compatibility (the feedforward expects 4D input).

---

## Prompt Stream Processing

The prompt stream is processed symmetrically to the spatial stream, but with its own set of modulation parameters:

```python
prompt_normed = ttnn.squeeze(
    self.norm1_context_norm(
        ttnn.unsqueeze(prompt, 0),
        dynamic_weight=(1 + prompt_scale_attn),
        dynamic_bias=prompt_shift_attn,
    ),
    0,
)
```

The prompt side uses `norm1_context_norm` (its own `DistributedLayerNorm` instance) and `norm1_context_linear` (its own modulation projection).

After attention, if `context_pre_only=False`:

```python
prompt_plus_attn = prompt + prompt_attn
prompt_normed = self.norm2_context(...)
prompt_ff = self.ff_context(...)
prompt_ff = prompt_ff * prompt_gate_ff
prompt = prompt + prompt_ff
```

The prompt has its own separate feedforward network (`self.ff_context`) and second norm (`self.norm2_context`). This means the spatial and prompt paths share no feedforward weights -- they interact **only** through the joint attention.

### `context_pre_only` Mode

When `context_pre_only=True` (used for the final transformer block in models like SD3.5 and Flux):

- Only 2 modulation parameters are produced for the prompt (shift and scale for the pre-attention norm).
- No gate is applied to the prompt attention output (`prompt_gate_attn = None`).
- No feedforward is applied to the prompt.
- No output projection is applied in the attention (`to_add_out = None`).
- The forward method returns `(spatial, None)`.

This is because the final block only needs to produce the spatial output (the denoised image tokens). The prompt representation is not needed after the last block.

---

## Weight Loading: `_prepare_torch_state`

The `_prepare_torch_state` method handles the mapping from HuggingFace Diffusers' state dict key naming to TT-DiT's internal naming:

```python
def _prepare_torch_state(self, state):
    rename_substate(state, "norm1.linear", "norm1_linear")
    rename_substate(state, "norm1.norm", "norm1_norm")
    rename_substate(state, "norm1_context.linear", "norm1_context_linear")
    rename_substate(state, "norm1_context.norm", "norm1_context_norm")
    rename_substate(state, "ff.net.0.proj", "ff.ff1")
    rename_substate(state, "ff.net.2", "ff.ff2")
    rename_substate(state, "ff_context.net.0.proj", "ff_context.ff1")
    rename_substate(state, "ff_context.net.2", "ff_context.ff2")
```

Diffusers nests the norm under `norm1.linear` and `norm1.norm`; TT-DiT flattens these to `norm1_linear` and `norm1_norm` because `Module.__setattr__` would otherwise try to register `norm1` as a child with sub-children.

The feedforward renaming maps Diffusers' `ff.net.0.proj` (the gated activation linear) and `ff.net.2` (the output linear) to TT-DiT's `ff.ff1` and `ff.ff2`.

Additionally, `prepare_chunked_linear_output` is called for the modulation projections:

```python
prepare_chunked_linear_output(
    state,
    prefix="norm1_linear",
    device_count=self.parallel_config.tensor_parallel.factor,
    chunks=6,
)
```

This rearranges the output weight of `norm1_linear` so that TP column-fracturing produces chunks that are aligned with the 6-way split. Without this, slicing the fractured output into 6 equal parts would not yield correct shift/scale/gate values.

---

## Complete Forward Flow

Summarizing the full forward pass of `TransformerBlock`:

```
Input: spatial [B, S/SP, D/TP], prompt [B, P, D/TP], time_embed [B, 1, D]

1. SiLU(time_embed) unless skip_time_embed_activation_fn
2. spatial_time = norm1_linear(time_embed)  -> [B, 1, 6*D/TP]
3. prompt_time = norm1_context_linear(time_embed) -> [B, 1, 6*D/TP] or [B, 1, 2*D/TP]
4. Split spatial_time into 6 chunks: shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff
5. Split prompt_time into 6 (or 2) chunks

6. spatial_normed = adaLN(spatial, shift_attn, scale_attn)
7. prompt_normed = adaLN(prompt, prompt_shift_attn, prompt_scale_attn)
8. All-gather spatial_normed and prompt_normed across TP

9. spatial_attn, prompt_attn = Attention(spatial_normed, prompt_normed, ropes)
10. spatial_attn *= gate_attn
11. spatial = spatial + spatial_attn  (residual)

12. spatial_normed = adaLN(spatial_plus_attn, shift_ff, scale_ff)
13. All-gather spatial_normed across TP
14. spatial_ff = FF(spatial_normed)
15. spatial_ff *= gate_ff
16. spatial = spatial + spatial_ff  (residual)

17. (symmetric steps 10-16 for prompt, unless context_pre_only)

Output: spatial [B, S/SP, D/TP], prompt [B, P, D/TP] or None
```

---

## All-Gather Placement Analysis

The block performs **four** all-gather operations in total (six if the prompt is not pre-only):

| All-Gather | Location | Dimension | Purpose |
|---|---|---|---|
| 1 | Before attention (spatial) | `dim=2` (hidden) | Reconstruct full hidden dim for `to_qkv` |
| 2 | Before attention (prompt) | `dim=2` (hidden) | Reconstruct full hidden dim for `add_qkv_proj` |
| 3 | Inside attention (spatial out) | `dim=2` (hidden) | Reconstruct for `to_out` projection |
| 4 | Inside attention (prompt out) | `dim=2` (hidden) | Reconstruct for `to_add_out` projection |
| 5 | Before FF (spatial) | `dim=2` (hidden) | Reconstruct full hidden dim for `ff1` |
| 6 | Before FF (prompt) | `dim=2` (hidden) | Reconstruct full hidden dim for `ff_context.ff1` |

All gathers are on the TP axis along the hidden dimension (`dim=2` in the 3D tensor). The SP axis is separate and handled by the ring attention kernel. This means each transformer block incurs significant cross-device communication -- the all-gathers dominate the communication cost for the block.

---

## Key Takeaways

1. **Adaptive LayerNorm replaces static normalization**: every norm in the block uses dynamically-generated shift, scale, and gate parameters derived from the diffusion timestep. This is the mechanism by which the model conditions on the noise level at each denoising step.

2. **Six modulation parameters per stream**: each stream (spatial and prompt) has shift/scale for the attention norm, gate for the attention output, shift/scale for the FF norm, and gate for the FF output. These are produced by a single `ColParallelLinear` projection of the time embedding.

3. **Dual-stream with interaction only at attention**: the spatial and prompt paths have completely independent normalization, modulation, and feedforward layers. They interact only through the joint attention mechanism, which concatenates their K/V sequences.

4. **`context_pre_only` reduces the final block**: in the last transformer block, the prompt stream is truncated after the attention norm -- no gate, no feedforward, no output. This saves compute since the prompt representation is not needed after the final denoising layer.

5. **Weight renaming is essential for interoperability**: the `_prepare_torch_state` method bridges the naming gap between HuggingFace Diffusers checkpoints and TT-DiT's internal structure, including non-trivial rearrangements for TP-compatible chunking of the modulation projection weights.

---

**Next:** [`comparison_with_symbiote_attention.md`](./comparison_with_symbiote_attention.md)
