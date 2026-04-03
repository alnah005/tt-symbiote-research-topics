# Layer Organization

## Layer Count and Types

The Gemma 4 31B text decoder contains **60 decoder layers** (indices 0 through
59). These layers are divided into two types:

- **50 sliding-window attention layers** --- local attention limited to a
  1024-token window.
- **10 global (full) attention layers** --- full causal attention over the
  entire context.

Every layer is an instance of the same class (`Gemma4TextDecoderLayer`), but
the attention submodule within each layer is configured differently depending
on the layer type.

## The 5:1 Pattern

The layer schedule follows a strict repeating pattern: **five sliding layers
followed by one global layer**, for 10 complete groups of 6.

```text
Group  0:  layers  0- 5   [S, S, S, S, S, G]
Group  1:  layers  6-11   [S, S, S, S, S, G]
Group  2:  layers 12-17   [S, S, S, S, S, G]
Group  3:  layers 18-23   [S, S, S, S, S, G]
Group  4:  layers 24-29   [S, S, S, S, S, G]
Group  5:  layers 30-35   [S, S, S, S, S, G]
Group  6:  layers 36-41   [S, S, S, S, S, G]
Group  7:  layers 42-47   [S, S, S, S, S, G]
Group  8:  layers 48-53   [S, S, S, S, S, G]
Group  9:  layers 54-59   [S, S, S, S, S, G]
```

Where `S` = sliding attention, `G` = global (full) attention.

The **global layer indices** are: **5, 11, 17, 23, 29, 35, 41, 47, 53, 59**.

Two invariants hold:

1. Every 6th layer (counting from 0) at position `6k + 5` is global.
2. The final layer (index 59) is always global.

This pattern is stored explicitly in the `layer_types` array in `config.json`.
At model construction time, the layer type is determined by
`config.layer_types[layer_idx]`, which returns either `"sliding_attention"` or
`"full_attention"`.

## Anatomy of a Single Decoder Layer

Every `Gemma4TextDecoderLayer` contains the following submodules, regardless of
whether it is a sliding or global layer:

| Submodule | Class | Purpose |
|-----------|-------|---------|
| `input_layernorm` | `Gemma4RMSNorm(5376)` | Pre-attention normalization |
| `self_attn` | `Gemma4TextAttention` | Attention (sliding or global) |
| `post_attention_layernorm` | `Gemma4RMSNorm(5376)` | Post-attention normalization |
| `pre_feedforward_layernorm` | `Gemma4RMSNorm(5376)` | Pre-FFN normalization |
| `mlp` | `Gemma4TextMLP` | GeGLU feed-forward network |
| `post_feedforward_layernorm` | `Gemma4RMSNorm(5376)` | Post-FFN normalization |
| `layer_scalar` | Buffer (scalar=1.0) | Per-layer output scaling |

Note that Gemma 4 uses **four** RMSNorm layers per decoder layer, not the two
found in standard LLaMA-style architectures. The extra pair
(`pre_feedforward_layernorm` and `post_feedforward_layernorm`) wraps the FFN
block separately from the attention post-norm.

### PLE Submodules (Conditional)

When `hidden_size_per_layer_input > 0`, the decoder layer also contains PLE
injection submodules. In the 31B config this value is **0**, so PLE submodules
are **not instantiated**. See [`novel_components.md`](./novel_components.md)
for a description of how PLE works in variants where it is enabled.

## Block Diagram

```text
                          input hidden_states
                                  |
                                  v
                    +----------------------------+
                    |     input_layernorm         |
                    |   RMSNorm(5376, eps=1e-6)  |
                    +----------------------------+
                                  |
                                  v
                    +----------------------------+
                    |        self_attn            |
                    |  Gemma4TextAttention        |
                    |  (sliding OR global config) |
                    +----------------------------+
                                  |
                                  v
                    +----------------------------+
                    | post_attention_layernorm    |
                    |   RMSNorm(5376, eps=1e-6)  |
                    +----------------------------+
                                  |
                            +-----+-----+
                            |  residual  |
                            |    add     |<--- input hidden_states
                            +-----+-----+
                                  |
                                  v
                    +----------------------------+
                    | pre_feedforward_layernorm   |
                    |   RMSNorm(5376, eps=1e-6)  |
                    +----------------------------+
                                  |
                                  v
                    +----------------------------+
                    |          mlp                |
                    |   Gemma4TextMLP (GeGLU)     |
                    |   gate: [5376, 21504]       |
                    |   up:   [5376, 21504]       |
                    |   down: [21504, 5376]       |
                    +----------------------------+
                                  |
                                  v
                    +----------------------------+
                    | post_feedforward_layernorm  |
                    |   RMSNorm(5376, eps=1e-6)  |
                    +----------------------------+
                                  |
                            +-----+-----+
                            |  residual  |
                            |    add     |<--- pre-FFN residual
                            +-----+-----+
                                  |
                                  v
                    +----------------------------+
                    |   * layer_scalar (1.0)      |
                    +----------------------------+
                                  |
                                  v
                          output hidden_states
```

**Note:** The post-norm-then-residual pattern in this diagram differs from the
more common pre-norm-only design. In Gemma 4, the attention and FFN outputs are
each normalized **before** being added back to the residual stream, providing
additional training stability.

---

**Next:** [`heterogeneous_attention_configs.md`](./heterogeneous_attention_configs.md)
