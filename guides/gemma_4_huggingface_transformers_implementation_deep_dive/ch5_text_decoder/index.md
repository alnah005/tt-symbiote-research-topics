# Chapter 5: Text Decoder

This chapter covers the text decoder stack in Gemma 4, which is the most architecturally complex subsystem in the model. It features dual attention types (sliding window and global), Mixture-of-Experts (MoE) parallel paths, per-layer input embeddings, KV sharing across layers, and per-layer scaling. All text decoder classes live in [`modeling_gemma4.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py). Refer to [Chapter 2](../ch2_configuration_hierarchy/index.md) for `Gemma4TextConfig` parameter defaults.

## Module Tree

```
Gemma4TextModel (extends Gemma3TextModel)
  |
  +-- embed_tokens: Gemma4TextScaledWordEmbedding
  |     num_embeddings=vocab_size, embedding_dim=hidden_size
  |     embed_scale = sqrt(hidden_size)
  |
  +-- [if hidden_size_per_layer_input]
  |     +-- embed_tokens_per_layer: Gemma4TextScaledWordEmbedding
  |     |     num_embeddings=vocab_size_per_layer_input
  |     |     embedding_dim=num_hidden_layers * hidden_size_per_layer_input
  |     |     embed_scale = sqrt(hidden_size_per_layer_input)
  |     +-- per_layer_model_projection: nn.Linear(hidden_size, num_hidden_layers * hidden_size_per_layer_input, bias=False)
  |     +-- per_layer_projection_norm: Gemma4RMSNorm(hidden_size_per_layer_input)
  |     +-- per_layer_input_scale: 2^(-0.5) = 0.7071
  |     +-- per_layer_model_projection_scale: hidden_size^(-0.5)
  |
  +-- layers: ModuleList of Gemma4TextDecoderLayer x num_hidden_layers (30)
  |     (see Section 5.5 below for full decoder layer tree)
  |
  +-- rotary_emb: Gemma4TextRotaryEmbedding
  |     Maintains separate inv_freq buffers per layer type:
  |       sliding_attention_inv_freq  (rope_type="default", theta=10000, dim=head_dim)
  |       full_attention_inv_freq     (rope_type="proportional", theta=1M, dim=global_head_dim)
  |
  +-- norm: Gemma4RMSNorm(hidden_size)
```

---

## 5.1 Gemma4TextScaledWordEmbedding

```python
class Gemma4TextScaledWordEmbedding(Gemma3TextScaledWordEmbedding):
    pass
```

A trivial subclass (identity pass-through) of `Gemma3TextScaledWordEmbedding`, which itself extends `nn.Embedding`. On forward, the raw embedding lookup result is multiplied by `embed_scale`, which is stored as a persistent-False buffer initialized to `sqrt(hidden_size)` for the main embedding and `sqrt(hidden_size_per_layer_input)` for the per-layer embedding.

**Forward:** `output = embedding_lookup(input_ids) * embed_scale`

---

## 5.2 Gemma4TextRotaryEmbedding

```python
class Gemma4TextRotaryEmbedding(Gemma3RotaryEmbedding):
```

Unlike models with a single RoPE configuration, Gemma 4 maintains **separate inverse-frequency buffers per attention type**, because sliding and global attention layers use different RoPE parameters.

### Initialization

The constructor iterates over `config.layer_types` (the unique set, e.g. `{"sliding_attention", "full_attention"}`) and for each type:

1. Reads `config.rope_parameters[layer_type]` to get `rope_type` and `rope_theta`.
2. For `"sliding_attention"`: uses the default RoPE init with `rope_theta=10000` and `dim=head_dim` (256).
3. For `"full_attention"`: uses the `"proportional"` RoPE init with `rope_theta=1000000` and `dim=global_head_dim` (512). The proportional init function is passed `head_dim_key="global_head_dim"` so it reads the correct dimension.

Each layer type gets two registered buffers:
- `{layer_type}_inv_freq` -- the active inverse frequencies `[dim/2]`
- `{layer_type}_original_inv_freq` -- a clone for dynamic RoPE reset

And one attribute:
- `{layer_type}_attention_scaling` -- a float scaling factor (1.0 for default, configured for proportional)

### Forward

```python
def forward(self, x, position_ids, layer_type=None):
    inv_freq = getattr(self, f"{layer_type}_inv_freq")
    attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

    inv_freq_expanded = inv_freq[None, :, None].float().expand(B, -1, 1)  # [B, dim/2, 1]
    position_ids_expanded = position_ids[:, None, :].float()               # [B, 1, S]

    freqs = inv_freq_expanded @ position_ids_expanded  # [B, dim/2, S]
    freqs = freqs.transpose(1, 2)                      # [B, S, dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)            # [B, S, dim]
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    return cos, sin  # both [B, S, dim], cast to x.dtype
```

The model's `forward` pre-computes position embeddings for each unique layer type once:

```python
position_embeddings = {}
for layer_type in self.unique_layer_types:
    position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)
```

Each decoder layer then receives the embeddings corresponding to its own type.

---

## 5.3 Gemma4TextAttention

```python
@use_kernelized_func(apply_rotary_pos_emb)
class Gemma4TextAttention(nn.Module):
```

This is the most structurally nuanced attention class in the Gemma 4 codebase. Each layer is either `"sliding_attention"` or `"full_attention"`, determined by `Gemma4TextConfig.layer_types[layer_idx]`.

### Dual Head Dimensions

| Property | Sliding Attention | Global (Full) Attention |
|---|---|---|
| `head_dim` | `config.head_dim` (256) | `config.global_head_dim` (512) |
| `num_key_value_heads` | `config.num_key_value_heads` (4) | `config.num_global_key_value_heads` (4) |
| `sliding_window` | `config.sliding_window` (512) | `None` |
| `use_alternative_attention` (k=v) | `False` | `True` (when `config.attention_k_eq_v=True`) |
| RoPE theta | 10,000 | 1,000,000 |
| RoPE partial rotary factor | 1.0 (full) | 0.25 (partial) |

### K-Equals-V Mode

When `config.attention_k_eq_v` is `True` (the default) and the layer is global attention, `self.v_proj` is set to `None`. During forward:

```python
value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
```

This means value states are a copy of key states at the raw `k_proj` output (pre-k_norm, pre-RoPE). This halves the KV parameters for global layers.

### Normalization

All three QKV paths have RMSNorm:
- `q_norm`: `Gemma4RMSNorm(head_dim, with_scale=True)` -- standard RMSNorm with learnable scale
- `k_norm`: `Gemma4RMSNorm(head_dim, with_scale=True)` -- standard RMSNorm with learnable scale
- `v_norm`: `Gemma4RMSNorm(head_dim, with_scale=False)` -- RMSNorm **without** learnable scale

The `scaling` factor is hardcoded to `1.0` (no `1/sqrt(d_k)` scaling -- the normalization handles magnitude control).

### KV Sharing Across Layers

The final `Gemma4TextConfig.num_kv_shared_layers` decoder layers do not compute their own KV states. Instead, they reuse KV from the last non-shared layer of the same attention type.

**Initialization logic:**

```python
first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0

if is_kv_shared_layer:
    # Find last non-shared layer of same type (sliding or full)
    prev_layers = config.layer_types[:first_kv_shared_layer_idx]
    kv_shared_layer_index = len(prev_layers) - 1 - prev_layers[::-1].index(config.layer_types[layer_idx])
```

**Forward logic for shared layers:**

```python
if self.is_kv_shared_layer and past_key_values is not None:
    key_states, value_states = past_key_values.shared_layers[self.kv_shared_layer_index]
```

**Storage logic for source layers:** The last non-shared layer of each attention type sets `store_full_length_kv = True` and saves its KV into `past_key_values.shared_layers[self.layer_idx]` for downstream shared layers to read.

Shared layers still have `k_proj` and `v_proj` parameters (they are not pruned), but these are only used during prefill when no cache exists. During cached generation, the projections are bypassed entirely.

### Projection Dimensions

For **sliding attention** (head_dim=256, num_heads=8, num_kv_heads=4):
- `q_proj`: `nn.Linear(hidden_size, 8 * 256 = 2048, bias=False)`
- `k_proj`: `nn.Linear(hidden_size, 4 * 256 = 1024, bias=False)`
- `v_proj`: `nn.Linear(hidden_size, 4 * 256 = 1024, bias=False)`
- `o_proj`: `nn.Linear(2048, hidden_size, bias=False)`

For **global attention** with k=v (global_head_dim=512, num_heads=8, num_global_kv_heads=4):
- `q_proj`: `nn.Linear(hidden_size, 8 * 512 = 4096, bias=False)`
- `k_proj`: `nn.Linear(hidden_size, 4 * 512 = 2048, bias=False)`
- `v_proj`: `None` (k=v mode)
- `o_proj`: `nn.Linear(4096, hidden_size, bias=False)`

### Forward Data Flow

```
hidden_states [B, S, hidden_size]
    |
    +--> q_proj -> view [B, S, num_heads, head_dim] -> q_norm -> apply_rotary_pos_emb -> transpose [B, num_heads, S, head_dim]
    |
    +--> k_proj -> view [B, S, num_kv_heads, head_dim] -> k_norm -> apply_rotary_pos_emb -> transpose [B, num_kv_heads, S, head_dim]
    |
    +--> v_proj (or copy of key_states if k=v) -> view -> v_norm -> transpose [B, num_kv_heads, S, head_dim]
    |
    +--> [if not shared: update past_key_values cache]
    +--> [if shared: read from past_key_values.shared_layers]
    +--> [if store_full_length_kv: save to past_key_values.shared_layers]
    |
    +--> attention_interface(Q, K, V, mask, scaling=1.0, sliding_window=...) -> [B, num_heads, S, head_dim]
    |
    +--> reshape [B, S, num_heads * head_dim] -> o_proj -> [B, S, hidden_size]
```

---

## 5.4 Gemma4TextMLP

```python
class Gemma4TextMLP(Gemma3MLP):
```

A gated MLP (SwiGLU-style) that conditionally doubles its intermediate size for KV-shared layers.

### Double-Width Logic

```python
first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
use_double_wide_mlp = config.use_double_wide_mlp and is_kv_shared_layer
self.intermediate_size = config.intermediate_size * (2 if use_double_wide_mlp else 1)
```

When `config.use_double_wide_mlp` is `True` (the default) and the layer is in the KV-sharing region, the MLP uses `2 * intermediate_size` instead of `intermediate_size`. This compensates for the reduced representational capacity from sharing KV states.

### Layers

- `gate_proj`: `nn.Linear(hidden_size, intermediate_size, bias=False)`
- `up_proj`: `nn.Linear(hidden_size, intermediate_size, bias=False)`
- `down_proj`: `nn.Linear(intermediate_size, hidden_size, bias=False)`
- `act_fn`: `ACT2FN[config.hidden_activation]` (gelu_pytorch_tanh)

### Forward

```python
def forward(self, x):
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

Shape: `[B, S, hidden_size]` -> `[B, S, hidden_size]` with intermediate expansion to `[B, S, intermediate_size]`.

---

## 5.5 Gemma4TextDecoderLayer

```python
class Gemma4TextDecoderLayer(Gemma3DecoderLayer):
```

Each decoder layer is a pre-norm transformer block with attention, MLP, optional parallel MoE, optional per-layer input injection, and a learnable layer scalar.

### Full Sub-Module Tree

```
Gemma4TextDecoderLayer (extends Gemma3DecoderLayer)
  |
  +-- input_layernorm: Gemma4RMSNorm(hidden_size)
  +-- self_attn: Gemma4TextAttention
  +-- post_attention_layernorm: Gemma4RMSNorm(hidden_size)
  |
  +-- pre_feedforward_layernorm: Gemma4RMSNorm(hidden_size)
  +-- mlp: Gemma4TextMLP
  +-- post_feedforward_layernorm: Gemma4RMSNorm(hidden_size)
  |
  +-- [if enable_moe_block]
  |     +-- router: Gemma4TextRouter
  |     +-- experts: Gemma4TextExperts
  |     +-- post_feedforward_layernorm_1: Gemma4RMSNorm(hidden_size)  (for MLP output)
  |     +-- pre_feedforward_layernorm_2: Gemma4RMSNorm(hidden_size)   (for MoE input)
  |     +-- post_feedforward_layernorm_2: Gemma4RMSNorm(hidden_size)  (for MoE output)
  |
  +-- [if hidden_size_per_layer_input]
  |     +-- per_layer_input_gate: nn.Linear(hidden_size, hidden_size_per_layer_input, bias=False)
  |     +-- act_fn: gelu_pytorch_tanh
  |     +-- per_layer_projection: nn.Linear(hidden_size_per_layer_input, hidden_size, bias=False)
  |     +-- post_per_layer_input_norm: Gemma4RMSNorm(hidden_size)
  |
  +-- layer_scalar: Buffer (scalar, initialized to 1.0)
```

### Forward Data Flow

The forward pass has four sequential stages:

**Stage 1 -- Self-Attention:**
```
residual = hidden_states
hidden_states = input_layernorm(hidden_states)
hidden_states = self_attn(hidden_states, position_embeddings, attention_mask, ...)
hidden_states = post_attention_layernorm(hidden_states)
hidden_states = residual + hidden_states
```

**Stage 2 -- Feedforward (MLP + optional MoE):**
```
residual = hidden_states
hidden_states = pre_feedforward_layernorm(hidden_states)
hidden_states = mlp(hidden_states)   # MLP path

if enable_moe_block:
    hidden_states_1 = post_feedforward_layernorm_1(hidden_states)  # post-norm MLP output

    # MoE path operates on the residual (pre-MLP hidden states)
    hidden_states_flat = residual.reshape(-1, hidden_size)
    _, top_k_weights, top_k_index = router(hidden_states_flat)
    hidden_states_2 = pre_feedforward_layernorm_2(hidden_states_flat)
    hidden_states_2 = experts(hidden_states_2, top_k_index, top_k_weights)
    hidden_states_2 = post_feedforward_layernorm_2(hidden_states_2)

    hidden_states = hidden_states_1 + hidden_states_2  # sum parallel paths

hidden_states = post_feedforward_layernorm(hidden_states)
hidden_states = residual + hidden_states
```

Key detail: When MoE is enabled, the MLP and MoE paths run in parallel from the same residual stream. The MLP operates on the pre-feedforward-normed input, while the MoE router and experts operate on the raw residual. The two outputs are combined by element-wise addition before the final post-feedforward norm. See [MoE Details](moe_details.md) for a deep dive.

**Stage 3 -- Per-Layer Input Injection (optional):**
```
if hidden_size_per_layer_input:
    residual = hidden_states
    hidden_states = per_layer_input_gate(hidden_states)       # project down
    hidden_states = act_fn(hidden_states)                     # gelu
    hidden_states = hidden_states * per_layer_input           # element-wise with per-layer embedding
    hidden_states = per_layer_projection(hidden_states)       # project back up
    hidden_states = post_per_layer_input_norm(hidden_states)
    hidden_states = residual + hidden_states
```

The `per_layer_input` tensor is a slice `[B, S, hidden_size_per_layer_input]` from the per-layer embedding pipeline described in Section 5.7.

**Stage 4 -- Layer Scalar:**
```
hidden_states *= self.layer_scalar
```

A learnable scalar (initialized to 1.0) that gates the entire layer's output.

---

## 5.6 Gemma4RMSNorm

```python
class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, with_scale=True):
```

Gemma 4 uses its own RMSNorm implementation (distinct from the standard `nn.RMSNorm`). It supports an optional `with_scale` flag:

- `with_scale=True` (default): includes a learnable `weight` parameter `[dim]`, applied after normalization.
- `with_scale=False`: pure normalization without any learnable parameter. Used by `v_norm` in attention and `norm` in the router.

The normalization uses `torch.pow(mean_squared, -0.5)` rather than `torch.rsqrt()` to address numerical differences between PyTorch and JAX compiler backends.

---

## 5.7 Per-Layer Input Embeddings Pipeline

Gemma 4 introduces a per-layer input embedding mechanism that provides each decoder layer with a unique, token-dependent conditioning signal. This is controlled by `Gemma4TextConfig.hidden_size_per_layer_input`.

### Model-Level Infrastructure (`Gemma4TextModel`)

When `hidden_size_per_layer_input > 0`, the model creates:

1. **`embed_tokens_per_layer`**: A second `Gemma4TextScaledWordEmbedding` with:
   - `num_embeddings = config.vocab_size_per_layer_input`
   - `embedding_dim = num_hidden_layers * hidden_size_per_layer_input`
   - Scaled by `sqrt(hidden_size_per_layer_input)`

2. **`per_layer_model_projection`**: `nn.Linear(hidden_size, num_hidden_layers * hidden_size_per_layer_input, bias=False)` -- projects the main embeddings into per-layer space.

3. **`per_layer_projection_norm`**: `Gemma4RMSNorm(hidden_size_per_layer_input)` -- normalizes each per-layer slice.

### Forward Pipeline

```
Step 1: get_per_layer_inputs(input_ids, inputs_embeds)
    per_layer_inputs = embed_tokens_per_layer(input_ids)
    per_layer_inputs = reshape to [B, S, num_hidden_layers, hidden_size_per_layer_input]

Step 2: project_per_layer_inputs(inputs_embeds, per_layer_inputs)
    per_layer_projection = per_layer_model_projection(inputs_embeds) * (hidden_size^-0.5)
    per_layer_projection = reshape to [B, S, num_hidden_layers, hidden_size_per_layer_input]
    per_layer_projection = per_layer_projection_norm(per_layer_projection)
    combined = (per_layer_projection + per_layer_inputs) * 2^(-0.5)

Step 3: Per decoder layer i:
    per_layer_input = combined[:, :, i, :]  # [B, S, hidden_size_per_layer_input]
    # Used in Stage 3 of decoder layer forward (Section 5.5)
```

When `input_ids` are unavailable (e.g., `generate` from `inputs_embeds`), the model reverses the main embedding by brute-force comparison against the embedding weight matrix to recover `input_ids`.

---

## 5.8 Gemma4TextModel.forward

The top-level text model forward orchestrates mask creation, position embedding computation, per-layer input preparation, and the decoder loop.

### Causal Mask Creation

The model creates **separate causal masks** for each attention type:

```python
causal_mask_mapping = {
    "full_attention": create_causal_mask(**mask_kwargs),
    "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
}
```

At the multimodal level, `create_causal_mask_mapping()` extends this with bidirectional attention support for vision tokens. When `mm_token_type_ids` is provided (indicating vision token positions), sliding attention layers receive an `or_mask_function` that allows vision tokens within the same vision group to attend to each other bidirectionally, overriding the causal constraint.

### Decoder Loop

```python
for i, decoder_layer in enumerate(self.layers):
    per_layer_input = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None

    hidden_states = decoder_layer(
        hidden_states,
        per_layer_input,
        position_embeddings=position_embeddings[config.layer_types[i]],
        attention_mask=causal_mask_mapping[config.layer_types[i]],
        ...
    )

hidden_states = self.norm(hidden_states)  # final RMSNorm
```

Each layer receives the position embeddings and attention mask corresponding to its own type (`"sliding_attention"` or `"full_attention"`).

---

## TTNN Porting Considerations

1. **Dual head dimensions**: Sliding layers use head_dim=256, global layers use global_head_dim=512. The TTNN attention kernel must handle both dimensions, potentially requiring two separate attention op configurations or a parameterized kernel.

2. **K-equals-V mode**: Global attention layers skip `v_proj` and copy key states as value states. On TTNN, this eliminates one matmul per global layer and requires the value path to share the key buffer (pre-k_norm, pre-RoPE).

3. **KV sharing**: The last `num_kv_shared_layers` layers read KV from earlier layers. On device, this means the KV cache tensors for source layers must remain accessible (not freed) and may need to be on the same device or copied. This is a cache management concern for the TTNN runtime.

4. **Double-width MLP**: KV-shared layers double the MLP intermediate dimension. Weight tensors for these layers are 2x wider, affecting memory layout and potentially requiring different shard configurations.

5. **Per-layer input embeddings**: The `embed_tokens_per_layer` produces a `[B, S, num_hidden_layers * hidden_size_per_layer_input]` tensor that is sliced per-layer. On TTNN, consider whether to keep this as one large tensor with slice views or pre-split into per-layer tensors.

6. **Separate RoPE buffers**: Two sets of inverse frequencies must be stored and used conditionally per layer type. Pre-computing cos/sin for both types (as the HuggingFace code does) avoids per-layer branching.

7. **MoE parallel path**: The MLP and MoE run in parallel from the same residual. On TTNN, these could be scheduled concurrently if device resources allow. See [MoE Details](moe_details.md) for MoE-specific porting notes.

8. **Layer scalar**: A per-layer multiplicative scalar (initialized to 1.0) is applied to the final output. This is a trivial element-wise multiply but must not be fused away during optimization.

9. **Attention scaling = 1.0**: Unlike most transformer implementations, Gemma 4 does not use `1/sqrt(d_k)` scaling. The TTNN attention kernel must set scaling to 1.0 explicitly or disable the default scaling.

10. **Bidirectional vision masking**: The sliding window mask for vision tokens uses an `or_mask_function` to create bidirectional attention within vision groups. This custom mask logic must be replicated in the TTNN mask generation pipeline.

---

**Next:** [MoE Details](moe_details.md)
