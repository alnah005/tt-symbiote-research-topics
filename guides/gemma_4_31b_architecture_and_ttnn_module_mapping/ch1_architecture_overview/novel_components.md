# Novel Components

This file describes the architectural innovations in Gemma 4 that are not found
in standard LLaMA-style decoders. Some of these features are configured to be
active in the 31B variant; others exist in the architecture but are disabled by
the 31B config.

## K=V Sharing

**Status in 31B:** Active in all 10 global layers (`attention_k_eq_v=true`).

In global attention layers, a single `k_proj` linear projection serves as the
basis for both keys and values; the V projection weight is not instantiated.
See [`heterogeneous_attention_configs.md`](./heterogeneous_attention_configs.md#kv-sharing-in-global-layers)
for the full dataflow and ASCII diagram. Sharing the projection saves one
[5376, 2048] matrix per global layer (~22 MB at BF16 each, ~220 MB total
across 10 layers).

## V-Norm

**Status in 31B:** Active in all 60 layers (both sliding and global).

Every attention layer applies RMSNorm to value vectors with `with_scale=False`.
This means the normalization divides by the root-mean-square magnitude but does
**not** multiply by a learned scale parameter.

### Definition

```math
\text{v-norm}(v) = \frac{v}{\sqrt{\text{mean}(v^2) + \epsilon}}
```

where $\epsilon = 10^{-6}$.

This is implemented by `Gemma4RMSNorm(dim=head_dim, eps=1e-6, with_scale=False)`.
The class conditionally omits the `self.weight` parameter when `with_scale=False`.

### Contrast With Standard RMSNorm

Standard RMSNorm (used for `input_layernorm`, `q_norm`, `k_norm`, etc.) includes
a learned per-element scale:

```math
\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma
```

V-norm omits the $\gamma$ term entirely. There is no trainable parameter
associated with V-norm --- it is a pure normalization operation.

### Why It Matters for TTNN

The `TTNNDistributedRMSNorm` module in tt-symbiote expects a weight tensor.
Implementing V-norm requires one of three strategies:

1. Pass an all-ones weight tensor (functionally correct but wastes memory for a
   dummy parameter).
2. Use a `with_scale=False` code path in the TTNN RMSNorm implementation if
   one exists.
3. Implement manually:
   `ttnn.mul(v, ttnn.rsqrt(ttnn.mean(ttnn.square(v), dim=-1, keepdim=True) + eps))`

## Per-Layer Embeddings (PLE)

**Status in 31B:** Disabled (`hidden_size_per_layer_input=0`).

PLE is an architectural feature present in the `Gemma4TextDecoderLayer` class
but gated by the `hidden_size_per_layer_input` config parameter. When this
value is greater than zero, PLE is active.

### How PLE Works (In Variants Where It Is Enabled)

PLE introduces a **second embedding table** (`embed_tokens_per_layer`) that
maps each token to a per-layer residual signal. The embedding table has shape:

$$
[\text{vocab size per layer input},\; \text{num hidden layers} \times \text{hidden size per layer input}]
$$

For each input token, this table produces a vector that is reshaped into
`[num_hidden_layers, hidden_size_per_layer_input]` --- one small vector per
decoder layer.

Additionally, a model-level projection (`per_layer_model_projection`) maps the
main embeddings to a parallel per-layer signal:

$$
\text{per layer projection} = \text{linear}(\text{input embeds}) \times \frac{1}{\sqrt{\text{hidden size}}}
$$

The two signals (embedding-based and projection-based) are combined:

$$
\text{per layer input} = (\text{per layer projection} + \text{per layer embedding}) \times 2^{-0.5}
$$

Within each decoder layer, the PLE injection occurs **after** the main residual
path (after both attention and FFN). The injection submodules are:

1. `per_layer_input_gate`: Linear [hidden_size, hidden_size_per_layer_input]
2. Activation: `gelu_pytorch_tanh`
3. Element-wise multiply with the per-layer input signal
4. `per_layer_projection`: Linear [hidden_size_per_layer_input, hidden_size]
5. `post_per_layer_input_norm`: RMSNorm(hidden_size)
6. Residual add

### Multimodal Handling

For multimodal inputs (images, video, audio), PLE uses the pad token ID for
non-text token positions. The per-layer embeddings are computed before the soft
token merge that replaces vision/audio token positions with encoder outputs.

### Relevance to 31B

Since `hidden_size_per_layer_input=0` in the 31B config, none of the PLE
submodules are instantiated. The decoder layer forward pass skips the PLE block
entirely. However, PLE is relevant for understanding the full Gemma 4
architecture family and may be active in other Gemma 4 variants.

## Logit Softcapping

**Status in 31B:** Active (`final_logit_softcapping=30.0`).

The output logits are soft-capped before being returned. This bounds the logit
magnitudes to the range $(-30, +30)$ using a tanh-based squashing function:

```math
\text{logits capped} = 30.0 \cdot \tanh\!\left(\frac{\text{logits}}{30.0}\right)
```

The implementation is:

1. Divide logits by 30.0.
2. Apply `tanh`.
3. Multiply by 30.0.

This prevents extreme logit values from destabilizing sampling or loss
computation. The `tanh` function smoothly compresses values that exceed the cap
rather than hard-clipping them.

### TTNN Mapping

This maps to a sequence of TTNN ops: `ttnn.multiply` (by $1/30$), `ttnn.tanh`,
`ttnn.multiply` (by $30$). Alternatively, a fused softcap kernel could combine
all three into a single operation.

## GeGLU Activation

**Status in 31B:** Active in all 60 layers (`hidden_activation="gelu_pytorch_tanh"`).

The feed-forward network in every decoder layer uses a Gated Linear Unit with
GELU activation (GeGLU). The `Gemma4TextMLP` implements:

```math
\text{FFN}(x) = W_{down} \cdot \left(\text{GELU}_{\tanh}(W_{gate} \cdot x) \odot W_{up} \cdot x\right)
```

where:

- $W_{gate}$: [5376, 21504] --- gate projection
- $W_{up}$: [5376, 21504] --- up projection
- $W_{down}$: [21504, 5376] --- down projection
- $\text{GELU}_{\tanh}$: GELU with tanh approximation
- $\odot$: element-wise multiplication

The gate and up projections can potentially be fused into a single
[5376, 43008] matrix for a single TTNN matmul, with the output split into gate
and up halves before applying the activation and element-wise multiply.

### Activation Function

The `gelu_pytorch_tanh` activation is the tanh-approximated GELU:

```math
\text{GELU}_{\tanh}(x) = 0.5 \cdot x \cdot \left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}} \cdot (x + 0.044715 \cdot x^3)\right)\right)
```

In TTNN this maps to `ttnn.gelu` with the appropriate approximation mode.

## Q-Norm and K-Norm

**Status in 31B:** Active in all 60 layers.

In addition to the layer-level norms, every attention module applies RMSNorm to
the query and key vectors after projection and before RoPE:

- `q_norm`: `Gemma4RMSNorm(head_dim, eps=1e-6, with_scale=True)`
- `k_norm`: `Gemma4RMSNorm(head_dim, eps=1e-6, with_scale=True)`

These are applied per-head (the norm dimension is `head_dim`, not
`hidden_size`). The `head_dim` varies by layer type: 256 for sliding, 512 for
global.

## KV Sharing Across Layers

**Status in 31B:** Disabled (`num_kv_shared_layers=0`).

The architecture supports an optimization where the last `num_kv_shared_layers`
layers reuse the KV tensors computed by an earlier layer of the same type,
eliminating the K and V projections entirely for those layers. In the 31B
config, this is set to 0, so every layer computes its own KV tensors.

## Layer Scalar

**Status in 31B:** Active (initialized to 1.0).

Each decoder layer has a `layer_scalar` buffer (registered as a non-trainable
buffer) that multiplies the final output. In the released 31B checkpoint this
is 1.0 for all layers, making it a no-op. It exists as a hook for potential
per-layer scaling during training or fine-tuning.

## Summary of Feature Activation in 31B

| Feature | Active? | Config Parameter |
|---------|---------|------------------|
| K=V sharing (global layers) | Yes | `attention_k_eq_v=true` |
| V-norm (all layers) | Yes | Always on in `Gemma4TextAttention` |
| Q-norm / K-norm (all layers) | Yes | Always on in `Gemma4TextAttention` |
| 4x RMSNorm per layer | Yes | Always on — see [`layer_organization.md`](./layer_organization.md) |
| GeGLU FFN | Yes | `hidden_activation="gelu_pytorch_tanh"` |
| Logit softcapping | Yes | `final_logit_softcapping=30.0` |
| Tied embeddings | Yes | `tie_word_embeddings=true` |
| PLE | No | `hidden_size_per_layer_input=0` |
| MoE block | No | `enable_moe_block=false` |
| KV sharing across layers | No | `num_kv_shared_layers=0` |
| Double-wide MLP | No | `use_double_wide_mlp=false` |
| Layer scalar (non-trivial) | No | Buffer = 1.0 |

---

**Next:** [Chapter 2 --- Projection Weights and Tensor Shapes](../ch2_projection_shapes/index.md)
