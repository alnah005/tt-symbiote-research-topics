# The Window Size Parameter

This file defines the window size parameter `w` precisely, surveys its typical
values in production models, analyses how it relates to context length and
effective receptive field, introduces the sink-token extension that some models
use alongside windowed attention, and explains where `w` lives in model
configuration files and how tt-transformers surfaces it.

## Definition of w

`w` is a positive integer that specifies the number of tokens, counting
backward from and including the current token, that each query may attend to.

Formally, for a query at position t in a sequence:

$$\mathcal{A}_{\text{win}}(t) = \bigl\{\,s \in \mathbb{Z} \;\big|\; \max(0,\, t - w + 1) \leq s \leq t\,\bigr\}$$

The attended window spans from `t − w + 1` (inclusive) to `t` (inclusive),
giving at most `w` positions. When `t < w − 1`, the left boundary is clamped
to 0 and the window contains only `t + 1` positions.

`w` is a per-model constant. It is fixed at training time and must match exactly
at inference time. Using a smaller window at inference than at training is
theoretically possible with degraded quality; using a larger window at inference
provides no benefit and wastes compute because the model's weights were not
trained to make use of the extra context.

`w` applies uniformly to all layers and all heads unless the model architecture
specifies per-layer or per-head window sizes (uncommon in current models but
architecturally valid).

## Typical Values in Production Models

| Model family          | Window size w | Max context length | w / context ratio |
|-----------------------|---------------|--------------------|-------------------|
| Mistral 7B v0.1       | 4 096         | 8 192              | 0.50              |
| Mistral 7B v0.2       | 4 096         | 32 768             | 0.13              |
| Mixtral 8×7B          | 4 096         | 32 768             | 0.13              |
| Qwen2 7B              | 4 096         | 131 072            | 0.03              |
| Qwen2 72B             | 4 096         | 131 072            | 0.03              |
| Qwen2.5 7B            | 4 096         | 131 072            | 0.03              |
| Qwen2.5 72B           | 4 096         | 131 072            | 0.03              |
| Qwen2.5-Coder 32B     | 4 096         | 131 072            | 0.03              |

The dominant value in the table is 4 096. This is not coincidental: 4 096
tokens cover roughly 3 000–4 000 words of dense prose or around 200–400 lines
of code, which empirically captures the vast majority of local syntactic and
semantic dependencies in these tasks. The value also aligns conveniently with
standard tile sizes and power-of-two memory granularity on accelerator hardware.

A value of 8 192 is used by some larger models or newer variants when slightly
longer-range reasoning is needed without committing to full attention across the
entire context.

## Relationship Between w, Context Length, and Effective Receptive Field

### Context length vs window size

A model's advertised context length (e.g., 131 072 for Qwen2) describes the
maximum total sequence length that can be processed by the model's position
encoding system — typically Rotary Position Embedding (RoPE) in these models.
This is orthogonal to `w`. A model with context length 131 072 and w = 4 096
can accept a 131 072-token input, but each query only directly attends to the
most recent 4 096 tokens of that input.

### Effective receptive field

The effective receptive field of a single attention layer is exactly `w` tokens:
the query has no direct path to any token beyond position `t − w`. Across L
stacked layers the receptive field grows. In a windowed-attention-only
transformer, the theoretical receptive field after L layers is:

$$\text{RF}(L) = w + (L - 1) \cdot (w - 1) = 1 + L \cdot (w - 1)$$

In practice this bound is loose. Each layer's output is a weighted average of
the window, so information from distant tokens diminishes exponentially with
layer depth. For L = 32 layers and w = 4 096 the theoretical receptive field
is over 130 000 tokens, meaning that windowed models can in principle propagate
information from the full context through the layer stack. However the strength
of that signal decreases with distance; tasks requiring strong long-range recall
(e.g., retrieval from early-document facts) will show quality degradation
compared with full attention. This per-layer bound applies identically during prefill and decode; the circular-buffer KV cache layout that enforces it at decode time is described in Chapter 2.

## Sink Tokens and the Extended Formulation

### Motivation

Empirical work on streaming inference (StreamingLLM, Mistral's attention sink
approach) observed that attention heads tend to assign disproportionately high
weight to the very first token(s) of the sequence even when those tokens carry
no semantic content. This phenomenon is attributed to the softmax normalisation:
when no nearby key is a strong match for the query, the model "parks" attention
mass on the first token as a stable attractor, because the first token was
always present during training and its key vector was trained to absorb residual
attention.

When sliding-window attention evicts the first token from the KV cache (as
happens once the context exceeds `w`), models relying on this attention sink
behaviour can degrade noticeably.

### Formulation with sink tokens

The sink-token extension reserves k_sink ≥ 1 positions at the start of the
KV cache for the first k_sink tokens, which are never evicted. The attended set
for query at position t becomes:

$$\mathcal{A}_{\text{sink}}(t) =
  \{0, 1, \ldots, k_{\text{sink}} - 1\}
  \;\cup\;
  \{\max(k_{\text{sink}},\, t - w + 1),\; \ldots,\; t\}$$

The two components are:
- **Global positions**: the first k_sink tokens, always attended regardless of t.
- **Local window**: the most recent w tokens counted from t, but starting no
  earlier than position k_sink to avoid double-counting.

In practice k_sink = 1 (the single first token) is the most common choice.
Mistral v0.1 uses k_sink = 4 (four "rolling buffer" tokens at position 0 are
kept as global attention sinks) in some implementations.

The mask diagram for T = 8, w = 3, k_sink = 1 is:

```text
Key position →  0 1 2 3 4 5 6 7
               ─────────────────
Query pos 0  │  1 . . . . . . .    (only self; sink = self)
Query pos 1  │  1 1 . . . . . .
Query pos 2  │  1 1 1 . . . . .
Query pos 3  │  1 1 1 1 . . . .    (sink at 0 + window [1,3])
Query pos 4  │  1 . 1 1 1 . . .    (sink at 0 + window [2,4])
Query pos 5  │  1 . . 1 1 1 . .    (sink at 0 + window [3,5])
Query pos 6  │  1 . . . 1 1 1 .    (sink at 0 + window [4,6])
Query pos 7  │  1 . . . . 1 1 1    (sink at 0 + window [5,7])
```

The non-contiguous attention pattern (a gap between sink tokens and the local window) first appears at t = w + k_sink. For w=3, k_sink=1, this is t=4: sink={0}, local window={2,3,4}, leaving position 1 unattended. At t=3 the attended set is still contiguous: {0,1,2,3}.
The KV cache in this regime must hold both the k_sink sink-token entries and the
w entries of the rolling window — a total of k_sink + w entries.

For the purposes of this guide, k_sink = 0 (no sink tokens) is the default
case unless explicitly stated. When sink tokens are relevant, the formulation
above applies.

## How w Is Stored in Model Config

### HuggingFace `config.json`

Windowed attention models distributed via HuggingFace store the window size as
a field in `config.json` inside the model checkpoint directory. The field name
varies by model family:

```text
Mistral / Mixtral:   "sliding_window": 4096
Qwen2 / Qwen2.5:     "sliding_window": 4096
```

Both fields carry the same semantic: the window size `w` as defined above. A
value of `null` or the absence of the field indicates full attention. Some
configs include both a `sliding_window` field and a `max_window_layers` field;
the latter specifies the number of layers (counting from layer 0) that use
windowed attention, with the remaining layers using full attention. Qwen2 models
use this pattern: only layers 0 through `max_window_layers − 1` apply the window
constraint; deeper layers use full attention.

### tt-transformers model configuration

Within the tt-transformers codebase, model configurations are represented as
Python dataclasses (typically named `ModelArgs` or similar). The window size is
stored as an integer attribute, for example:

```python
sliding_window: int = 4096
```

The field is read from the HuggingFace `config.json` during model loading and
stored on the `ModelArgs` instance. Inference code that builds the KV cache or
constructs attention masks reads this field to determine:

- The fixed KV cache allocation size `[B, H, w, d]` for windowed layers.
- The boundary condition for the circular-buffer write pointer (see Chapter 2).
- The mask shape for prefill: a band of width `w` along the diagonal of the
  T × T score matrix (see Chapter 3).

When `max_window_layers` is present in the config, the inference engine applies
windowed attention only to the designated layers and instantiates separate KV
cache tensors of different shapes for windowed vs non-windowed layers.

For tt-transformers specifically, the relevant paths are the model's
`ModelArgs` dataclass (populated from `config.json`) and the attention module's
forward method, which consults `args.sliding_window` to decide whether to apply
the window mask and which KV cache shape to target.

---

**Next:** [Chapter 2 — KV Cache Management During Decode](../ch2_kv_cache_management/index.md)
