# TTNNGemma4Model

The top-level model module orchestrates the complete Gemma 4 31B inference
pipeline: token embedding, optional PLE precomputation, the 60-layer decoder
loop, final normalization, the LM head with tied weights, and logit
softcapping. This file covers the full module structure, weight loading from
HuggingFace checkpoints, KV cache initialization, and decode loop design.

## Top-Level Forward Pass

```text
input_ids [B, 1]   (token IDs for decode step)
      |
      v
 embed_tokens [262144, 5376]
      |
      v
 input_embeds [B, 1, 5376]
      |
      v
 scale by sqrt(5376)       hidden_states = input_embeds * sqrt(hidden_size)
      |
      v
 (PLE precomputation -- no-op in 31B)
      |
      v
 +------------------------------------------------------------------+
 | Decoder Loop: layers[0] through layers[59]                       |
 |                                                                  |
 |   for i in range(60):                                            |
 |       hidden_states = layers[i](                                 |
 |           hidden_states, cos_sliding, sin_sliding,               |
 |           cos_global, sin_global, kv_cache, current_pos,         |
 |           page_table, ple_signals=None                           |
 |       )                                                          |
 |                                                                  |
 |   Layer types by index:                                          |
 |   [S S S S S G S S S S S G S S S S S G ... S S S S S G]         |
 |    0 1 2 3 4 5 6 7 8 9 ...                         58 59        |
 +------------------------------------------------------------------+
      |
      v
 norm: TTNNDistributedRMSNorm [5376]       (final layer norm)
      |
      v
 hidden_states [B, 1, 5376]
      |
      v
 lm_head: matmul with embed_tokens.weight^T   (tied weights)
      |
      v
 logits [B, 1, 262144]
      |
      v
 logit_softcapping(logits, cap=30.0)
      |
      v
 capped_logits [B, 1, 262144]
```

## Constructor

```python
class TTNNGemma4Model(TTNNModule):
    def __init__(self, config: Gemma4Config, mesh_device):
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.hidden_size = config.hidden_size          # 5376
        self.num_layers = config.num_hidden_layers     # 60
        self.vocab_size = config.vocab_size            # 262144

        # --- Token embedding ---
        self.embed_tokens = TTNNEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size,
        )

        # --- Embedding scale factor ---
        self.embed_scale = math.sqrt(self.hidden_size)  # sqrt(5376) ~ 73.32

        # --- PLE (disabled in 31B) ---
        self.ple_enabled = config.hidden_size_per_layer_input > 0
        # embed_tokens_per_layer and per_layer_model_projection
        # are NOT instantiated when ple_enabled is False

        # --- RoPE cos/sin precomputation ---
        # Two sets of tables for the two RoPE configurations
        self.sliding_cos, self.sliding_sin = precompute_rope_tables(
            max_seq_len=config.max_position_embeddings,
            head_dim=config.head_dim,                   # 256
            theta=10000.0,
            partial_rotary_factor=1.0,
        )
        self.global_cos, self.global_sin = precompute_rope_tables(
            max_seq_len=config.max_position_embeddings,
            head_dim=config.global_head_dim,            # 512
            theta=1000000.0,
            partial_rotary_factor=0.25,                 # 128 rotary dims
        )

        # --- Decoder layers ---
        self.layers = [
            TTNNGemma4DecoderLayer(layer_idx=i, config=config, mesh_device=mesh_device)
            for i in range(self.num_layers)
        ]

        # --- Final norm ---
        self.norm = TTNNDistributedRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )

        # --- LM head: tied with embed_tokens ---
        # No separate lm_head weight; reuse embed_tokens.weight
        self.final_logit_softcapping = config.final_logit_softcapping  # 30.0

        # --- KV cache ---
        self.kv_cache = None  # initialized in init_kv_cache()
```

## Embedding and Scaling

The token embedding lookup is followed by a scale factor of $\sqrt{5376}$:

```math
\text{hidden states} = \text{embed tokens}(\text{input ids}) \times \sqrt{5376}
```

This scaling is a Gemma convention (also present in Gemma 1 and Gemma 2) that
replaces the more common $1/\sqrt{d_{model}}$ scaling applied inside attention.
The scaling is applied once at the model input rather than per-layer.

```python
# In forward()
hidden_states = self.embed_tokens(input_ids)       # [B, 1, 5376]
hidden_states = ttnn.multiply(hidden_states, self.embed_scale)
```

The embed_scale is a scalar constant (~73.32), so this is a simple
`ttnn.multiply` with a scalar.

## Decoder Loop

The 60-layer decoder loop iterates through all layers sequentially. Each layer
is a `TTNNGemma4DecoderLayer` that internally dispatches to the correct
attention type based on its `layer_idx`.

```python
# In forward()
for i in range(self.num_layers):
    hidden_states = self.layers[i](
        hidden_states,
        cos_sliding=self.sliding_cos,
        sin_sliding=self.sliding_sin,
        cos_global=self.global_cos,
        sin_global=self.global_sin,
        kv_cache=self.kv_cache,
        current_pos=current_pos,
        page_table=page_table,
        ple_signals=None,           # PLE disabled in 31B
    )
```

### Layer Sequence

The 60 layers follow the 5:1 sliding/global pattern:

```text
Layer:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 ... 54 55 56 57 58 59
Type:   S  S  S  S  S  G  S  S  S  S  S  G  S  S  ... S  G  S  S  S  G
```

Within the loop, each layer receives the same set of cos/sin tables (both
sliding and global). The decoder layer selects the appropriate tables based on
its `is_global` flag (see
[`decoder_layer_module.md`](./decoder_layer_module.md)).

### Metal Trace Considerations

For production decode, the 60-layer loop is captured as a single Metal Trace.
Because each layer's forward pass is deterministic (no runtime branching inside
the attention subclasses, thanks to the polymorphic design from
[Chapter 5](../ch5_attention_module_design/design_options.md)), the entire trace
is a fixed sequence of ops.

The trace captures:

1. 50 sliding-layer forward passes (each with the same op sequence).
2. 10 global-layer forward passes (each with their own op sequence).
3. Two all-reduce operations per layer (120 total).
4. The final norm and LM head matmul.

The trace is recorded once and replayed for every subsequent decode step,
eliminating host dispatch overhead.

## Final Norm and LM Head

After the decoder loop, the hidden states pass through a final RMSNorm and then
the LM head projection.

```python
# In forward()
hidden_states = self.norm(hidden_states)           # [B, 1, 5376]

# LM head with tied weights
logits = ttnn.matmul(
    hidden_states,
    self.embed_tokens.weight,                      # [5376, 262144] (transposed)
)                                                  # [B, 1, 262144]
```

### Tied Weights

The `tie_word_embeddings=true` config means the LM head reuses the token
embedding weight matrix. The embedding weight has shape `[262144, 5376]`
(vocab_size x hidden_size). For the LM head matmul, it is used as
`hidden_states @ weight.T`, effectively computing
`[B, 1, 5376] x [5376, 262144] = [B, 1, 262144]`.

In TTNN, the embedding weight is stored on device as part of `embed_tokens`.
The LM head matmul reads the same weight tensor, transposed. No separate weight
tensor is allocated.

### Weight Sharding for the LM Head

The LM head matmul `[5376, 262144]` is the largest single matmul in the model
by output dimension. Sharding options:

- **Column-parallel:** Shard the output dim (vocab) across 8 devices:
  262144 / 8 = 32768 per device. Weight shape per device: `[5376, 32768]` =
  ~352 MB at BF16. This is the natural choice since the embedding table can be
  row-sharded across devices (each device holds 32768 vocab entries), and the
  same shard serves both embedding lookup and the LM head.

- **Output gathering:** After the column-parallel LM head matmul, each device
  holds logits for 32768 vocab entries. For sampling, either gather to a single
  device or perform distributed top-k.

## Logit Softcapping

After the LM head, logits are soft-capped to the range $(-30, +30)$:

```math
\text{logits capped} = 30.0 \cdot \tanh\!\left(\frac{\text{logits}}{30.0}\right)
```

```python
# In forward()
logits = ttnn.multiply(logits, 1.0 / self.final_logit_softcapping)  # / 30.0
logits = ttnn.tanh(logits)
logits = ttnn.multiply(logits, self.final_logit_softcapping)         # * 30.0
```

This maps to three TTNN ops: `ttnn.multiply`, `ttnn.tanh`, `ttnn.multiply`.
A fused softcap kernel could combine these into one op, but the unfused version
is correct and the ops are elementwise on a single tensor, so the overhead is
small.

The softcap prevents extreme logit values from destabilizing sampling. The tanh
function smoothly compresses values beyond the cap rather than hard-clipping.

## Weight Loading

### HuggingFace to TTNN Weight Mapping

The following table maps HuggingFace checkpoint parameter names to the
corresponding TTNN module attributes:

| HuggingFace Name | TTNN Module | Shape | Sharding |
|-------------------|-------------|-------|----------|
| `model.embed_tokens.weight` | `embed_tokens.weight` | `[262144, 5376]` | Row-sharded (vocab dim) |
| `model.norm.weight` | `norm.weight` | `[5376]` | Replicated |
| `model.layers.{i}.input_layernorm.weight` | `layers[i].input_layernorm.weight` | `[5376]` | Replicated |
| `model.layers.{i}.post_attention_layernorm.weight` | `layers[i].post_attention_layernorm.weight` | `[5376]` | Replicated |
| `model.layers.{i}.post_feedforward_layernorm.weight` | `layers[i].post_feedforward_layernorm.weight` | `[5376]` | Replicated |
| `model.layers.{i}.self_attn.q_proj.weight` | `layers[i].self_attn.q_proj.weight` | Varies by type | Col-sharded |
| `model.layers.{i}.self_attn.k_proj.weight` | `layers[i].self_attn.k_proj.weight` | Varies by type | Col-sharded (sliding) or Replicated (global) |
| `model.layers.{i}.self_attn.v_proj.weight` | `layers[i].self_attn.v_proj.weight` | `[5376, 4096]` | Col-sharded (sliding only) |
| `model.layers.{i}.self_attn.o_proj.weight` | `layers[i].self_attn.o_proj.weight` | Varies by type | Row-sharded |
| `model.layers.{i}.self_attn.q_norm.weight` | `layers[i].self_attn.q_norm.weight` | `[head_dim]` | Replicated |
| `model.layers.{i}.self_attn.k_norm.weight` | `layers[i].self_attn.k_norm.weight` | `[head_dim]` | Replicated |
| `model.layers.{i}.mlp.gate_proj.weight` | `layers[i].mlp.gate_proj.weight` | `[5376, 21504]` | Col-sharded |
| `model.layers.{i}.mlp.up_proj.weight` | `layers[i].mlp.up_proj.weight` | `[5376, 21504]` | Col-sharded |
| `model.layers.{i}.mlp.down_proj.weight` | `layers[i].mlp.down_proj.weight` | `[21504, 5376]` | Row-sharded |

### Layer-Type-Dependent Weight Shapes

Attention weights vary by layer type. The weight loading logic must inspect
`layer_idx` to determine the correct source shapes:

| Weight | Sliding (50 layers) | Global (10 layers) |
|--------|--------------------|--------------------|
| `q_proj.weight` | `[8192, 5376]` | `[16384, 5376]` |
| `k_proj.weight` | `[4096, 5376]` | `[2048, 5376]` |
| `v_proj.weight` | `[4096, 5376]` | N/A (K=V sharing) |
| `o_proj.weight` | `[5376, 8192]` | `[5376, 16384]` |
| `q_norm.weight` | `[256]` | `[512]` |
| `k_norm.weight` | `[256]` | `[512]` |

Note: HuggingFace stores linear weights as `[out_features, in_features]`
(transposed from the convention used elsewhere in this guide). The TTNN
`from_torch` methods handle the transposition.

### V-Norm Handling

V-norm has `with_scale=False` --- there is no learned weight in the checkpoint.
The TTNN module must handle this by either:

1. Not loading any weight (if the TTNN RMSNorm supports `with_scale=False`).
2. Creating an all-ones weight tensor (if using the standard
   `TTNNDistributedRMSNorm` that expects a weight).

See [Chapter 3 --- V-Norm Implementation](../ch3_kv_sharing_and_vnorm/vnorm_implementation.md)
for the full analysis.

### Global Layer V Projection

For global layers, the HuggingFace checkpoint contains no `v_proj.weight`
because K=V sharing is active. The weight loading code must skip the V
projection for layers where `layer_idx % 6 == 5`. The `TTNNGemma4GlobalAttention`
class does not have a `v_proj` attribute, so no weight needs to be loaded.

## KV Cache Initialization

The KV cache must accommodate two different geometries for sliding and global
layers.

### Paged KV Cache Configuration

```python
def init_kv_cache(self, max_batch_size, max_seq_len, block_size=64):
    """Initialize paged KV cache for all 60 layers."""

    # Sliding layers: 50 layers, 16 KV heads, head_dim=256, window=1024
    # Global layers:  10 layers, 4 KV heads (replicated), head_dim=512, full causal

    sliding_config = PagedAttentionConfig(
        num_kv_heads=16,          # per-device: 2 after TP=8 sharding
        head_dim=256,
        max_seq_len=1024,         # window-bounded
        block_size=block_size,
        max_batch_size=max_batch_size,
    )

    global_config = PagedAttentionConfig(
        num_kv_heads=4,           # replicated on all devices (not sharded)
        head_dim=512,
        max_seq_len=max_seq_len,  # full causal, not window-bounded
        block_size=block_size,
        max_batch_size=max_batch_size,
    )

    self.kv_cache = PagedKVCache(
        layer_configs=[
            sliding_config if (i % 6 != 5) else global_config
            for i in range(self.num_layers)
        ],
        mesh_device=self.mesh_device,
    )
```

### Per-Device KV Cache Memory

From [Chapter 6 --- KV Cache Sharding](../ch6_tp_sharding/kv_cache_sharding.md):

| Component | Per-Device Memory (BF16, B=1) |
|-----------|------------------------------|
| 50 sliding layers (2 heads/dev, 1024 window) | 100.0 MB (constant) |
| 10 global layers (4 heads/dev replicated, S tokens) | S-dependent |
| **Total at S=8,192** | 100.0 + 640.0 = **740.0 MB** |
| **Total at S=32,768** | 100.0 + 2,560.0 = **2,660.0 MB** |

At BFP8 KV cache (1 byte/element), these numbers halve.

### Page Table Management

Each layer type requires a different page table:

- **Sliding layers:** The page table references at most
  `ceil(1024 / block_size)` pages. At `block_size=64`, this is 16 pages per
  sequence. Old pages are recycled as the window slides.

- **Global layers:** The page table grows with sequence length. At
  `block_size=64` and S=8192, this is 128 pages per sequence.

The `PagedKVCache` object manages separate page tables for each layer or
layer type. During decode, the page table is passed to `paged_sdpa_decode`
along with the current position.

## Decode Orchestration

### Single-Step Decode

```python
def decode_step(self, input_ids, current_pos, page_table):
    """Execute a single decode step (one new token per sequence)."""

    # 1. Embedding + scale
    hidden_states = self.embed_tokens(input_ids)           # [B, 1, 5376]
    hidden_states = ttnn.multiply(hidden_states, self.embed_scale)

    # 2. Slice cos/sin tables for current position
    cos_sliding = self.sliding_cos[current_pos]            # [1, 256]
    sin_sliding = self.sliding_sin[current_pos]            # [1, 256]
    cos_global = self.global_cos[current_pos]              # [1, 128]
    sin_global = self.global_sin[current_pos]              # [1, 128]

    # 3. Decoder loop
    for i in range(self.num_layers):
        hidden_states = self.layers[i](
            hidden_states,
            cos_sliding, sin_sliding,
            cos_global, sin_global,
            self.kv_cache, current_pos, page_table,
        )

    # 4. Final norm
    hidden_states = self.norm(hidden_states)               # [B, 1, 5376]

    # 5. LM head (tied weights)
    logits = ttnn.matmul(hidden_states, self.embed_tokens.weight)
                                                           # [B, 1, 262144]

    # 6. Logit softcapping
    logits = ttnn.multiply(logits, 1.0 / 30.0)
    logits = ttnn.tanh(logits)
    logits = ttnn.multiply(logits, 30.0)

    return logits
```

### Metal Trace Integration

For production performance, the entire `decode_step` is captured as a Metal
Trace:

1. **Trace capture:** Run `decode_step` once with tracing enabled. TTNN records
   every device op (matmuls, norms, RoPE, SDPA, all-reduce) into a trace
   buffer.

2. **Trace replay:** On subsequent decode steps, replay the trace. Only the
   input tensors (`input_ids`, `current_pos`, `page_table`) are updated; the
   op sequence and program configs are fixed.

3. **Requirements for traceability:**
   - No Python-level conditionals that change the op sequence between steps.
     The `is_global` dispatch is resolved at construction time (polymorphism),
     so the trace sees a fixed op sequence.
   - All tensor shapes must be constant across decode steps. Since B and S=1
     are fixed for decode, and the cos/sin table slices are always shape
     `[1, dim]`, this holds.
   - KV cache updates use in-place operations that do not change tensor shapes.

### Autoregressive Generation Loop

```python
def generate(self, prompt_ids, max_new_tokens):
    """Autoregressive generation after prefill."""

    # Assume prefill has already populated the KV cache for prompt_ids
    current_pos = len(prompt_ids)

    for step in range(max_new_tokens):
        # Get the last generated token
        input_ids = last_token_ids                         # [B, 1]
        page_table = self.kv_cache.get_page_table(current_pos)

        # Run one decode step
        logits = self.decode_step(input_ids, current_pos, page_table)

        # Sample next token
        next_token = sample(logits[:, -1, :])              # [B]

        # Update state
        last_token_ids = next_token.unsqueeze(1)           # [B, 1]
        current_pos += 1
```

## Complete Per-Device Memory Summary

The following table summarizes the per-device memory budget for the full model
at BF16, B=1, S=8192:

| Component | Per-Device Memory |
|-----------|------------------|
| Embedding (sharded, 32768 entries) | ~352 MB |
| 50 sliding layers: attention weights | ~1,650 MB |
| 50 sliding layers: FFN weights | ~4,335 MB |
| 10 global layers: attention weights | ~660 MB |
| 10 global layers: FFN weights | ~867 MB |
| Norm weights (all layers + final) | ~1 MB |
| Sliding KV cache (50 layers, window=1024) | 100 MB |
| Global KV cache (10 layers, S=8192) | 640 MB |
| RoPE cos/sin tables | ~2 MB |
| **Total** | **~8,607 MB** |

This leaves approximately 3.4 GB of headroom per device (out of 12 GB DRAM)
for activations, page tables, and the Metal Trace buffer. At BFP8 weight
quantization, the weight memory roughly halves, providing substantially more
headroom for longer sequences or larger batch sizes.

See [Chapter 8 --- Performance Analysis](../ch8_performance/index.md) for the
detailed memory budget breakdown and optimization roadmap.

---

**Next:** [Chapter 8 --- Performance Analysis and Optimization Roadmap](../ch8_performance/index.md)
