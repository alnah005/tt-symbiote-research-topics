# TTNNGemma4PLE

Per-Layer Embeddings (PLE) is an architectural feature in the Gemma 4 decoder
that injects a per-layer residual signal into each decoder layer's input. In the
31B config, PLE is **disabled** (`hidden_size_per_layer_input=0`), making the
PLE module a no-op. This file documents the PLE mechanism for completeness and
for future Gemma 4 variants where PLE may be active.

## PLE Status in 31B

| Config Parameter | Value | Effect |
|------------------|-------|--------|
| `hidden_size_per_layer_input` | 0 | PLE disabled; no submodules instantiated |
| `vocab_size_per_layer_input` | 262144 | Unused when `hidden_size_per_layer_input=0` |

When `hidden_size_per_layer_input=0`:

- The `embed_tokens_per_layer` embedding table is not created.
- The `per_layer_model_projection` linear layer is not created.
- The per-layer `per_layer_input_gate`, `per_layer_projection`, and
  `post_per_layer_input_norm` submodules inside each decoder layer are not
  created.
- The PLE injection in the decoder layer forward pass is a no-op (returns
  `hidden_states` unchanged).

## Implementation as No-Op

```python
class TTNNGemma4PLE(TTNNModule):
    def __init__(self, layer_idx: int, config: Gemma4Config, mesh_device):
        super().__init__()
        self.layer_idx = layer_idx
        self.ple_dim = config.hidden_size_per_layer_input  # 0 in 31B
        self.enabled = self.ple_dim > 0

        if self.enabled:
            # These are NOT instantiated for 31B
            self.per_layer_input_gate = TTNNLinear(
                config.hidden_size, self.ple_dim
            )
            self.per_layer_projection = TTNNLinear(
                self.ple_dim, config.hidden_size
            )
            self.post_per_layer_input_norm = TTNNDistributedRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

    def forward(self, hidden_states, ple_signals=None):
        if not self.enabled:
            return hidden_states  # no-op in 31B

        # --- PLE injection (for variants where PLE is active) ---
        # ple_signals shape: [B, 1, ple_dim] (precomputed for this layer)
        gate_out = self.per_layer_input_gate(hidden_states)   # [B, 1, ple_dim]
        gate_out = ttnn.gelu(gate_out, fast_and_approximate_mode=True)
        gate_out = ttnn.mul(gate_out, ple_signals)            # [B, 1, ple_dim]
        projection = self.per_layer_projection(gate_out)      # [B, 1, hidden_size]
        projection = self.post_per_layer_input_norm(projection)
        hidden_states = ttnn.add(hidden_states, projection)   # residual add
        return hidden_states
```

## PLE Mechanism (When Enabled)

For documentation purposes and to support future Gemma 4 variants, this section
describes the full PLE mechanism as implemented in the HuggingFace reference.

### Model-Level Precomputation

At the model level (inside `TTNNGemma4Model`), PLE precomputes per-layer
signals from the input tokens before the decoder loop begins:

```text
input_ids [B, S]
      |
      +---> embed_tokens [262144, 5376] ---------> input_embeds [B, S, 5376]
      |                                                   |
      |                                         per_layer_model_projection
      |                                         [5376, num_layers * ple_dim]
      |                                                   |
      |                                         scale by 1/sqrt(5376)
      |                                                   |
      |                                         reshape [B, S, 60, ple_dim]
      |                                                   |
      +---> embed_tokens_per_layer [262144, 60*ple_dim]   |
                        |                                  |
                 reshape [B, S, 60, ple_dim]               |
                        |                                  |
                        +------------- add ----------------+
                                       |
                                  scale by 2^(-0.5)
                                       |
                                  ple_signals [B, S, 60, ple_dim]
```

The result is a tensor of shape `[B, S, 60, ple_dim]` containing one PLE
signal vector per layer per token. During the decoder loop, layer $i$ receives
slice `ple_signals[:, :, i, :]` of shape `[B, S, ple_dim]`.

### Per-Layer Injection

Within each decoder layer, the PLE injection occurs **before** the
`input_layernorm` and attention:

1. **Gate:** `per_layer_input_gate` projects `hidden_states` from
   `hidden_size` to `ple_dim`, followed by `gelu_pytorch_tanh` activation.
2. **Multiply:** Element-wise multiply of the gated projection with the
   precomputed PLE signal for this layer.
3. **Project back:** `per_layer_projection` maps from `ple_dim` back to
   `hidden_size`.
4. **Norm:** `post_per_layer_input_norm` applies RMSNorm to the projection.
5. **Residual add:** The normalized projection is added to `hidden_states`.

### Injection Point in Decoder Layer

```text
hidden_states [B, S, 5376]
      |
      +--- PLE injection (gate -> activate -> multiply -> project -> norm -> add)
      |
      v
 input_layernorm
      |
      v
 self_attn
      |
 ... (rest of decoder layer)
```

PLE modifies the hidden states **before** the pre-attention norm. This allows
the per-layer embedding signal to influence both the attention and FFN
computations in each layer.

## Host vs Device Decision

If PLE were active, the key implementation question would be where to execute
the PLE precomputation and per-layer injection:

### PLE Precomputation (Model Level)

| Approach | Pros | Cons |
|----------|------|------|
| **Host** | Simple embedding lookup; small matmul `[5376, 60*ple_dim]` | Requires host-to-device transfer of `ple_signals` before decode loop |
| **Device** | No transfer; PLE signals already on device | Embedding lookup on device requires `ttnn.embedding`; may not be worth the complexity for a one-time precompute |

**Recommendation (if PLE were active):** Perform PLE precomputation on host.
The embedding lookup is a simple table index, and the projection matmul is
small. The resulting `ple_signals` tensor is transferred to device once before
the decode loop. At B=1, S=1 decode, the transfer is `60 * ple_dim * 2` bytes
--- negligible.

### Per-Layer Injection

| Approach | Pros | Cons |
|----------|------|------|
| **Device** | Stays in the device compute pipeline; no host roundtrip | Requires two small `ttnn.linear` calls and a `ttnn.gelu` per layer |
| **Host** | Could precompute the full injection for all layers | Breaks the device pipeline; host-device sync per layer |

**Recommendation (if PLE were active):** Execute per-layer injection on device.
The PLE submodule operations (two small linear projections, one activation, one
norm, one add) are lightweight and should run within the traced decode step. The
weights for `per_layer_input_gate` and `per_layer_projection` would be stored on
device alongside the other layer weights.

## Multimodal Pad-Token Handling

For multimodal inputs (images, video, audio), non-text token positions in the
input sequence are handled specially by PLE:

1. **Before soft-token merge:** PLE embeddings are computed for all positions
   using the original `input_ids`. Non-text positions (vision/audio tokens) use
   the **pad token ID** for their PLE lookup, since these positions do not have
   meaningful text token IDs.

2. **After soft-token merge:** The main `input_embeds` at vision/audio positions
   are replaced with encoder outputs (e.g., SigLIP vision encoder outputs). But
   the PLE signals computed in step 1 remain unchanged --- they still carry the
   pad-token-derived PLE vectors for those positions.

3. **During decoding:** Each decoder layer injects the PLE signal regardless of
   whether the token at that position is a text token or a replaced
   vision/audio token.

This design means PLE for non-text tokens is effectively a constant bias derived
from the pad token embedding. The model learns during training to handle this
gracefully.

### Relevance to 31B

Since PLE is disabled in 31B, multimodal pad-token handling is moot for this
variant. However, any future Gemma 4 variant with `hidden_size_per_layer_input > 0`
will need this logic. The implementation should:

1. Accept a `pad_token_id` parameter (default: 0).
2. Before PLE precomputation, replace non-text token IDs in the input with
   `pad_token_id`.
3. Compute PLE signals using the modified input IDs.

## Weight Shapes (When PLE Is Active)

For reference, the weight shapes in a PLE-enabled variant with
`hidden_size_per_layer_input = D_ple`:

| Weight | Shape | Location | Count |
|--------|-------|----------|-------|
| `embed_tokens_per_layer` | `[vocab_size, 60 * D_ple]` | Model level | 1 |
| `per_layer_model_projection` | `[5376, 60 * D_ple]` | Model level | 1 |
| `per_layer_input_gate` | `[5376, D_ple]` | Per layer | 60 |
| `per_layer_projection` | `[D_ple, 5376]` | Per layer | 60 |
| `post_per_layer_input_norm` | `[5376]` (scale) | Per layer | 60 |

In the 31B config (`D_ple = 0`), none of these weights exist.

---

**Next:** [`full_model_module.md`](./full_model_module.md)
