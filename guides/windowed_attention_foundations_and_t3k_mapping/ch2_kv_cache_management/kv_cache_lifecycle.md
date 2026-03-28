# KV Cache Lifecycle During Decode

This file traces the lifecycle of a KV cache from the first generated token
through the steady state of long-form generation, derives the exact size
formulae for windowed and full-attention caches, and quantifies the memory
saving factor as a function of generation length T and window size w.

## What the KV Cache Is and Why It Exists

Autoregressive decode generates one token per forward pass. At each step, the
model must compute attention between the current query and all prior keys and
values. Without a cache, every decode step would recompute every key and value
vector from scratch — O(T²) total work across T generated tokens. The KV cache
stores the key and value tensors produced at each previous step so that they can
be read directly from DRAM on subsequent steps, reducing per-step compute to
O(1) (or O(w) for windowed attention).

For a single attention layer with H heads, batch size B, head dimension d, and
dtype of `dtype_bytes` bytes per element, the KV cache at step T contains:

- **Key cache**: one d-dimensional vector per head per batch item per prior
  position → shape `[B, H, T, d]`
- **Value cache**: identical shape `[B, H, T, d]`

Together: `2 * B * H * T * d * dtype_bytes` bytes per layer.

This file is concerned with a single layer. The full-model KV cache multiplies
by the number of layers L; that factor is identical for full and windowed
attention and cancels in any ratio, so it is omitted here.

## Token-by-Token Growth: The Fill Phase

Decode begins after the prefill pass has processed the prompt tokens. Let the
prompt have length `P`. The KV cache after prefill holds P entries per head
(positions 0 through P−1). During decode the model generates new tokens one at
a time; call the generation step index `g`, starting at `g = 0` for the first
generated token. The absolute position of the current token is `t = P + g`.

### Full-Attention Cache Growth

Under full attention, every new key and value is appended to the cache. The
cache size after generating `g` tokens is:

$$\text{size}_{\text{full}}(g) = 2 \cdot B \cdot H \cdot (P + g) \cdot d \cdot \text{dtype\_bytes}$$

This grows without bound as `g` increases. For a 7B model with H = 32 heads, d = 128, B = 1, bfloat16, after generating
32 768 tokens: cache size ≈ 2 × 1 × 32 × 32 768 × 128 × 2 = 536 870 912
bytes ≈ 512 MiB per layer. With 32 layers the full-model KV cache is 16 GiB —
consuming the majority of Wormhole DRAM and growing further with each new token.

### Windowed Cache: The Fill Phase (g < w − P)

When the total context size (prompt + generated tokens) is still smaller than w,
the windowed attention set A_win(t) = [0, t] and the window has not yet filled
up. In this phase the windowed cache behaves identically to the full-attention
cache: it grows with each new token.

Define the **fill threshold** as the generation step at which the window first
reaches capacity:

$$g_{\text{fill}} = \max(0,\, w - P)$$

(When P ≥ w the fill phase is skipped entirely: the cache is already full from
prefill and eviction begins immediately at g = 0.)

During the fill phase (`0 ≤ g < g_fill`):

$$\text{size}_{\text{win}}(g) = 2 \cdot B \cdot H \cdot (P + g) \cdot d \cdot \text{dtype\_bytes}$$

This is identical to the full-attention case during this phase.

## Windowed Eviction: Reaching Steady State

Once `P + g ≥ w`, the window is full. Query at absolute position t = P + g
attends to positions [t − w + 1, t]. Position `t − w` (the oldest entry
currently in the cache) is outside the window and will never again be attended
to by any future query, because future queries have t' > t and their windows
shift further right (formally: the evicted position `t − w` is strictly below every future window's left boundary `t' − w + 1` since `t' > t`). It is therefore safe — and necessary for bounded memory —
to evict that entry.

The **eviction rule** is:

> When appending the KV vectors for the new token at position t, discard the
> KV vectors for position `t − w` (the entry that just fell out of the window).

After each decode step in steady state, the number of entries in the cache
remains exactly w. The cache size is constant:

$$\text{size}_{\text{win,steady}} = 2 \cdot B \cdot H \cdot w \cdot d \cdot \text{dtype\_bytes}$$

This is independent of `g` (and therefore of T = P + g). No matter how long
generation continues, the windowed cache does not grow.

### Invariant

At the start of decode step g (with `g ≥ g_fill`), the cache holds exactly the
KV vectors for the w positions [t − w + 1, t − 1] (the w−1 entries from prior
steps) plus the new entry being written for position t. After the write and
eviction, the cache again holds exactly w entries covering [t − w + 1, t].

Formally, for all `g ≥ g_fill`:

$$\text{cache contents after step } g = \{(k_s, v_s) \;|\; t - w + 1 \leq s \leq t\},
\quad t = P + g$$

This invariant matches the window definition `A_win(t') = [t' − w + 1, t']`
for any future step t' > t: the cache always contains every key and value that
the next query could need.

## Steady-State Size Formulae

The two formulae below are the key quantitative results of this chapter. They
hold per attention layer; multiply by the number of layers L for the full model.

**Windowed attention (steady state):**

$$\text{KV cache size}_{\text{win}} = 2 \cdot B \cdot H \cdot w \cdot d \cdot \text{dtype\_bytes}$$

**Full attention at generation step T = P + g:**

$$\text{KV cache size}_{\text{full}}(T) = 2 \cdot B \cdot H \cdot T \cdot d \cdot \text{dtype\_bytes}$$

Both formulae share the factor `2 * B * H * d * dtype_bytes`; the distinction
is the cache length dimension: constant `w` for windowed, growing `T` for full.

### Numeric Example

Parameters: B = 1, H = 32, d = 128, dtype = bfloat16 (2 bytes), w = 4 096,
T = 32 768. Per layer:

| Variant         | Cache length | Size per layer | 32-layer model |
|-----------------|-------------|----------------|----------------|
| Windowed        | w = 4 096   | 64 MiB         | 2 GiB          |
| Full at T=32768 | T = 32 768  | 512 MiB        | 16 GiB         |
| Full at T=131072| T = 131 072 | 2 048 MiB      | 64 GiB         |

For T = 131 072 (Qwen2's max context length) the full-attention KV cache would
exceed the combined DRAM of an 8-chip T3K mesh (8 × 12 GiB = 96 GiB); the
windowed cache at w = 4 096 is 2 GiB per model, leaving ample headroom for
weights.

## Memory Saving Factor

Define the memory saving factor R(T, w) as the ratio of full-attention cache
size to windowed cache size at generation length T:

$$R(T, w) = \frac{2 \cdot B \cdot H \cdot T \cdot d \cdot \text{dtype\_bytes}}
               {2 \cdot B \cdot H \cdot w \cdot d \cdot \text{dtype\_bytes}}
         = \frac{T}{w}$$

The saving factor is simply T / w, growing linearly with generation length and
inversely with window size.

Representative values for w = 4 096:

| T (total sequence length) | R = T / w  |
|---------------------------|-----------|
| 4 096 (= w)               | 1× (none) |
| 8 192                     | 2×        |
| 16 384                    | 4×        |
| 32 768                    | 8×        |
| 65 536                    | 16×       |
| 131 072                   | 32×       |

The same factor R applies to DRAM read bandwidth per decode step, since the
number of bytes read from the KV cache is proportional to the number of cache
entries accessed. This is the primary performance benefit on Wormhole hardware
where DRAM bandwidth, not compute, is the decode bottleneck.


## Summary of the Lifecycle

```text
Phase            | Cache size         | Eviction?
-----------------|--------------------|----------
Prefill          | grows to P entries | No
Fill (g < g_fill)| grows to w entries | No
Steady state     | fixed at w entries | Yes: oldest entry evicted each step
```

The transition from fill phase to steady state happens exactly once, at step
`g_fill = max(0, w − P)`. After that, the cache size never changes. The
circular buffer layout that implements this steady-state eviction without
moving data in DRAM is described in
[`circular_buffer_layout.md`](./circular_buffer_layout.md).

---

**Next:** [`circular_buffer_layout.md`](./circular_buffer_layout.md)
