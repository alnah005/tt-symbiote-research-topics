# PLE Shapes

This file documents the Per-Layer Embedding (PLE) tensor shapes in the Gemma 4
architecture. While PLE is **disabled in the 31B config**
(`hidden_size_per_layer_input=0`), understanding its shapes is important for
two reasons: (1) other Gemma 4 variants may enable PLE, and (2) the PLE
injection point in the decoder layer forward pass exists in the reference code
even when disabled.

## PLE Status in 31B

| Config Parameter | Value | Effect |
|-----------------|-------|--------|
| `hidden_size_per_layer_input` | 0 | PLE disabled; no submodules instantiated |
| `vocab_size_per_layer_input` | 262144 | Would be PLE vocab size if enabled |

When `hidden_size_per_layer_input=0`, the following are true:

- No `embed_tokens_per_layer` embedding table is created.
- No `per_layer_model_projection` linear layer is created.
- No per-layer `per_layer_input_gate`, `per_layer_projection`, or
  `post_per_layer_input_norm` submodules are created.
- The decoder layer forward pass skips the PLE injection block entirely.
- **There are zero PLE parameters in the 31B model.**

## PLE Architecture (For Variants Where Enabled)

For completeness and forward compatibility, the following describes the PLE
shapes when `hidden_size_per_layer_input > 0`. Let:

- $d_{\text{ple}}$ = `hidden_size_per_layer_input` (the PLE hidden dimension)
- $V_{\text{ple}}$ = `vocab_size_per_layer_input` (typically equals `vocab_size`)
- $L$ = `num_hidden_layers` = 60
- $d_{\text{model}}$ = `hidden_size` = 5376

### Model-Level PLE Components

| Component | Shape | Notes |
|-----------|-------|-------|
| `embed_tokens_per_layer` | [$V_{\text{ple}}$, $L \times d_{\text{ple}}$] | Second embedding table |
| `per_layer_model_projection` | [$d_{\text{model}}$, $L \times d_{\text{ple}}$] | Projects main embeddings to PLE space |

The `embed_tokens_per_layer` table maps each token ID to a vector of size
$L \times d_{\text{ple}}$, which is then reshaped to $[L, d_{\text{ple}}]$ ---
one small vector per decoder layer. This is conceptually similar to having 60
separate small embedding tables, but packed into a single lookup for efficiency.

### Per-Layer PLE Submodules

Each decoder layer (when PLE is enabled) contains:

| Submodule | Shape | Notes |
|-----------|-------|-------|
| `per_layer_input_gate` | [$d_{\text{model}}$, $d_{\text{ple}}$] | Gates the PLE signal |
| `per_layer_projection` | [$d_{\text{ple}}$, $d_{\text{model}}$] | Projects PLE back to model dim |
| `post_per_layer_input_norm` | [$d_{\text{model}}$] | RMSNorm after PLE injection |

### PLE Injection Dataflow

The PLE injection occurs at the end of each decoder layer, after both the
attention and FFN blocks:

```text
hidden_states  [B, S, d_model]
      |
      v
per_layer_input_gate linear  [d_model, d_ple]
      |
      v
gelu_pytorch_tanh activation
      |
      v
element-wise multiply with per_layer_input  [B, S, d_ple]
      |
      v
per_layer_projection linear  [d_ple, d_model]
      |
      v
post_per_layer_input_norm (RMSNorm)
      |
      v
residual add with hidden_states
      |
      v
output hidden_states  [B, S, d_model]
```

### PLE Combination Formula

The per-layer input signal is computed once at the model level and passed to
each decoder layer:

```math
\text{per layer embed} = \text{embed tokens per layer}(\text{input ids}).\text{reshape}(B, S, L, d_{\text{ple}})
```

```math
\text{per layer proj} = \text{per layer model projection}(\text{input embeds}).\text{reshape}(B, S, L, d_{\text{ple}})
```

```math
\text{per layer input} = (\text{per layer proj} + \text{per layer embed}) \times 2^{-0.5}
```

Each decoder layer $i$ receives slice $[:, :, i, :]$ of shape $[B, S, d_{\text{ple}}]$.

### Multimodal Handling

For multimodal inputs (images, video, audio), the PLE embedding lookup uses the
**pad token ID** for all non-text token positions. This is because:

1. PLE embeddings are computed **before** the soft token merge that replaces
   vision/audio positions with encoder outputs.
2. Non-text tokens do not have meaningful token IDs for the text vocabulary.
3. Using the pad token ensures a neutral PLE contribution for non-text positions.

## Implications for 31B TTNN Implementation

Since PLE is disabled in the 31B config, the TTNN implementation should:

1. **Not instantiate any PLE submodules.** The `TTNNGemma4DecoderLayer` should
   check `hidden_size_per_layer_input` and skip PLE module creation when it is 0.
2. **Not allocate PLE weight memory.** There are no PLE weights to load or shard.
3. **Skip the PLE injection in the forward pass.** The conditional check
   `if self.per_layer_input is not None` (or equivalent) short-circuits the
   PLE block.
4. **Leave a hook for future variants.** The code structure should accommodate
   PLE activation if a future Gemma 4 variant enables it, but should not
   introduce any overhead for the 31B case.

The net effect is that PLE adds zero compute, zero memory, and zero latency to
the 31B inference path.

---

**Next:** [Chapter 3 --- K=V Sharing and V-Norm Implementation](../ch3_kv_sharing_and_vnorm/index.md)
