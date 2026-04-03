# TTNNGemma4DecoderLayer

The decoder layer is the fundamental repeating unit of the Gemma 4 31B model.
All 60 layers share the same `TTNNGemma4DecoderLayer` class, but each layer's
constructor uses its `layer_idx` to determine whether to instantiate a sliding
or global attention submodule. The FFN, norms, and PLE injection are identical
across all layers.

## Forward Pass Dataflow

The following diagram shows the complete data path through a single decoder
layer during decode (batch=1, single token). Each box is a TTNN operation or
submodule call.

```text
hidden_states [1, 1, 5376]  (replicated on all 8 devices)
      |
      +--- (save for residual_1)
      |
      v
 PLE injection (no-op in 31B; see ple_module.md)
      |
      v
 input_layernorm: TTNNDistributedRMSNorm [5376]
      |
      v
 self_attn: TTNNGemma4SlidingAttention or TTNNGemma4GlobalAttention
      |        (dispatched by layer_idx; see Chapter 5)
      |        output: [1, 1, 5376] (after O proj + all-reduce)
      |
      v
 residual_1 = hidden_states + attn_output     ttnn.add
      |
      +--- (save for residual_2)
      |
      v
 post_attention_layernorm: TTNNDistributedRMSNorm [5376]
      |
      v
 mlp: TTNNGemma4FFN (see ffn_module.md)
      |        output: [1, 1, 5376] (after down proj + all-reduce)
      |
      v
 post_feedforward_layernorm: TTNNDistributedRMSNorm [5376]
      |
      v
 residual_2 = residual_1 + ffn_output         ttnn.add
      |
      v
 layer_scalar * residual_2                    (1.0 in 31B, no-op)
      |
      v
 output hidden_states [1, 1, 5376]
```

## Constructor

The constructor receives the full model config and the layer index. The layer
index determines the attention type via the 5:1 pattern: layers at indices
{5, 11, 17, 23, 29, 35, 41, 47, 53, 59} are global; all others are sliding.

```python
class TTNNGemma4DecoderLayer(TTNNModule):
    def __init__(self, layer_idx: int, config: Gemma4Config, mesh_device):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size  # 5376

        # Determine layer type from the 5:1 pattern
        self.is_global = (layer_idx % 6 == 5)

        # --- Norms ---
        self.input_layernorm = TTNNDistributedRMSNorm(
            config.hidden_size,       # 5376
            eps=config.rms_norm_eps,   # 1e-6
        )
        self.post_attention_layernorm = TTNNDistributedRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_feedforward_layernorm = TTNNDistributedRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # --- Attention (dispatched by layer type) ---
        if self.is_global:
            self.self_attn = TTNNGemma4GlobalAttention(layer_idx, config, mesh_device)
        else:
            self.self_attn = TTNNGemma4SlidingAttention(layer_idx, config, mesh_device)

        # --- FFN (identical across all layers) ---
        self.mlp = TTNNGemma4FFN(config, mesh_device)

        # --- PLE (no-op in 31B) ---
        self.ple_injection = TTNNGemma4PLE(layer_idx, config, mesh_device)

        # --- Layer scalar (1.0 in 31B, registered as buffer) ---
        self.layer_scalar = 1.0  # loaded from checkpoint
```

### Layer Type Dispatch

The `is_global` flag is computed as `layer_idx % 6 == 5`. This produces the
correct global layer indices:

| layer_idx | layer_idx % 6 | is_global |
|-----------|---------------|-----------|
| 0 | 0 | No (sliding) |
| 1 | 1 | No (sliding) |
| 2 | 2 | No (sliding) |
| 3 | 3 | No (sliding) |
| 4 | 4 | No (sliding) |
| **5** | **5** | **Yes (global)** |
| 6 | 0 | No (sliding) |
| ... | ... | ... |
| **59** | **5** | **Yes (global)** |

This matches the `layer_types` array from `config.json` (see
[Chapter 1 --- Layer Organization](../ch1_architecture_overview/layer_organization.md)).

### Attention Module Types

The attention submodule is one of two concrete classes, both subclasses of
`TTNNGemma4AttentionBase` (see
[Chapter 5 --- Design Options](../ch5_attention_module_design/design_options.md)
for the recommendation):

| Layer Type | Attention Class | KV Heads | head_dim | RoPE | Window |
|------------|----------------|----------|----------|------|--------|
| Sliding | `TTNNGemma4SlidingAttention` | 16 | 256 | Full, theta=10K | 1024 |
| Global | `TTNNGemma4GlobalAttention` | 4 | 512 | Partial, theta=1M | Full causal |

The decoder layer does not need to know the internal details of either attention
type. It calls `self.self_attn.forward(...)` with a uniform interface and
receives `[1, 1, 5376]` output in both cases.

## Forward Method

```python
def forward(
    self,
    hidden_states,          # [B, 1, 5376] on all devices (replicated)
    cos_sliding,            # precomputed cos table for sliding RoPE
    sin_sliding,            # precomputed sin table for sliding RoPE
    cos_global,             # precomputed cos table for global p-RoPE
    sin_global,             # precomputed sin table for global p-RoPE
    kv_cache,               # PagedKVCache managing all 60 layers
    current_pos,            # current decode position (scalar)
    page_table,             # page table for paged attention
    ple_signals=None,       # per-layer PLE signals (None in 31B)
):
    # --- PLE injection (no-op in 31B) ---
    hidden_states = self.ple_injection(hidden_states, ple_signals)

    # --- Save input for first residual connection ---
    residual = hidden_states

    # --- Pre-attention norm ---
    hidden_states = self.input_layernorm(hidden_states)

    # --- Attention (dispatch is implicit via polymorphism) ---
    if self.is_global:
        cos, sin = cos_global, sin_global
    else:
        cos, sin = cos_sliding, sin_sliding

    attn_output = self.self_attn(
        hidden_states,
        cos=cos,
        sin=sin,
        kv_cache=kv_cache,
        current_pos=current_pos,
        page_table=page_table,
    )

    # --- First residual add ---
    hidden_states = ttnn.add(residual, attn_output)

    # --- Save for second residual connection ---
    residual = hidden_states

    # --- Pre-FFN norm ---
    hidden_states = self.post_attention_layernorm(hidden_states)

    # --- GeGLU FFN ---
    ffn_output = self.mlp(hidden_states)

    # --- Post-FFN norm ---
    ffn_output = self.post_feedforward_layernorm(ffn_output)

    # --- Second residual add ---
    hidden_states = ttnn.add(residual, ffn_output)

    # --- Layer scalar (1.0 in 31B, no-op) ---
    # hidden_states = ttnn.multiply(hidden_states, self.layer_scalar)

    return hidden_states
```

## Norm Placement

Gemma 4 uses a **pre-norm** architecture with an additional post-FFN norm. This
is different from the standard pre-norm-only pattern used in LLaMA:

| Norm | Position | Applied To |
|------|----------|------------|
| `input_layernorm` | Before attention | Hidden states entering Q/K/V projections |
| `post_attention_layernorm` | Before FFN (after first residual) | Hidden states entering gate/up projections |
| `post_feedforward_layernorm` | After FFN (before second residual add) | FFN output before it is added to the residual |

The `post_feedforward_layernorm` is applied to the FFN output **before** the
residual add, not after. This is a Gemma-specific choice. The residual
connection adds the **normalized** FFN output to the saved hidden state.

All three norms are `TTNNDistributedRMSNorm` with `hidden_size=5376` and
`eps=1e-6`. They have learned scale parameters (`with_scale=True`), unlike the
V-norm in the attention module which has `with_scale=False`.

## Residual Connections

There are exactly two residual additions per decoder layer:

1. **Post-attention residual:** `residual_1 = input + attn_output`. The input
   is the hidden state before `input_layernorm` (but after PLE injection). The
   attention output has already been projected through O and all-reduced across
   devices.

2. **Post-FFN residual:** `residual_2 = residual_1 + post_ffn_norm(ffn_output)`.
   Note that `residual_1` (the post-attention sum) is the skip target, not the
   pre-norm hidden state. The FFN output passes through
   `post_feedforward_layernorm` before the add.

Both residual additions use `ttnn.add` on replicated tensors of shape
`[B, 1, 5376]`. Since the O and down projections are row-parallel with
all-reduce (see
[Chapter 6 --- Weight Sharding](../ch6_tp_sharding/weight_sharding.md)), the
tensors entering the residual add are already replicated across all devices.

## Tensor Shapes Through the Layer

All shapes are for B=1 single-token decode, after TP=8 sharding where
applicable.

| Step | Tensor | Shape (Per Device) | Notes |
|------|--------|-------------------|-------|
| Input | `hidden_states` | `[1, 1, 5376]` | Replicated |
| After PLE | `hidden_states` | `[1, 1, 5376]` | No-op in 31B |
| After input_layernorm | `hidden_states` | `[1, 1, 5376]` | Replicated |
| Attention output | `attn_output` | `[1, 1, 5376]` | After O proj + all-reduce |
| After residual_1 | `hidden_states` | `[1, 1, 5376]` | Replicated |
| After post_attn_norm | `hidden_states` | `[1, 1, 5376]` | Replicated |
| FFN output | `ffn_output` | `[1, 1, 5376]` | After down proj + all-reduce |
| After post_ffn_norm | `ffn_output` | `[1, 1, 5376]` | Replicated |
| After residual_2 | `hidden_states` | `[1, 1, 5376]` | Replicated |

The hidden_size dimension (5376) is never sharded across devices --- it is
always replicated. Sharding occurs only within the attention projections (Q/K/V
head dimensions) and FFN projections (intermediate_size dimension).

## Memory Lifetime of Residual Tensors

During the forward pass, two tensors must be held simultaneously:

1. The **residual tensor** saved before the norm.
2. The **working tensor** being processed through the submodule (attention or FFN).

Each residual tensor is `[1, 1, 5376]` at BF16 = 10,752 bytes. This is trivial
relative to the weight memory and KV cache. The residual can be stored in L1 on
each device.

After the residual add, the saved tensor can be deallocated. The peak activation
memory per layer is dominated by the attention and FFN intermediate tensors (see
[Chapter 8](../ch8_performance/index.md) for the full memory analysis).

## from_torch Integration

The `from_torch` class method creates a `TTNNGemma4DecoderLayer` from the
corresponding HuggingFace `Gemma4TextDecoderLayer`:

```python
@classmethod
def from_torch(cls, hf_layer, layer_idx, config, mesh_device):
    ttnn_layer = cls(layer_idx, config, mesh_device)

    # Norms (direct weight copy)
    ttnn_layer.input_layernorm = TTNNDistributedRMSNorm.from_torch(
        hf_layer.input_layernorm
    )
    ttnn_layer.post_attention_layernorm = TTNNDistributedRMSNorm.from_torch(
        hf_layer.post_attention_layernorm
    )
    ttnn_layer.post_feedforward_layernorm = TTNNDistributedRMSNorm.from_torch(
        hf_layer.post_feedforward_layernorm
    )

    # Attention (dispatch by layer type)
    if ttnn_layer.is_global:
        ttnn_layer.self_attn = TTNNGemma4GlobalAttention.from_torch(
            hf_layer.self_attn, layer_idx, config, mesh_device
        )
    else:
        ttnn_layer.self_attn = TTNNGemma4SlidingAttention.from_torch(
            hf_layer.self_attn, layer_idx, config, mesh_device
        )

    # FFN
    ttnn_layer.mlp = TTNNGemma4FFN.from_torch(
        hf_layer.mlp, config, mesh_device
    )

    # Layer scalar
    ttnn_layer.layer_scalar = hf_layer.layer_scalar.item()

    return ttnn_layer
```

The key dispatch point is the attention module: `from_torch` inspects
`layer_idx` to determine whether to create a `TTNNGemma4SlidingAttention` or
`TTNNGemma4GlobalAttention`, as recommended in
[Chapter 5 --- Design Options](../ch5_attention_module_design/design_options.md).

---

**Next:** [`ffn_module.md`](./ffn_module.md)
