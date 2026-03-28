# Full Causal Attention vs Windowed Attention

This file gives precise mathematical definitions of both attention variants,
illustrates the structural difference with mask diagrams, and summarises the
complexity implications that motivate windowed attention on memory-bandwidth-
limited hardware.

## Full Causal Attention — Formal Definition

Let the input sequence have T tokens. For a single attention head, each token
position t (0-indexed, 0 ≤ t < T) produces a query vector q_t ∈ ℝ^d, a key
vector k_t ∈ ℝ^d, and a value vector v_t ∈ ℝ^d, where d is the head dimension.

The causal constraint requires that position t cannot attend to any future
position s > t. Under full causal attention, the set of positions that query t
may attend to is:

$$\mathcal{A}_{\text{full}}(t) = \{0, 1, \ldots, t\}$$

The attention output for position t is:

$$o_t = \sum_{s \in \mathcal{A}_{\text{full}}(t)} \alpha_{t,s} \, v_s$$

where the attention weights α are derived from a masked softmax over raw scores:

$$e_{t,s} = \frac{q_t \cdot k_s}{\sqrt{d}}$$

$$\alpha_{t,s} = \frac{\exp(e_{t,s})}{\sum_{s' \in \mathcal{A}_{\text{full}}(t)} \exp(e_{t,s'})}$$

In matrix form, processing all T positions simultaneously:

- Q ∈ ℝ^{T×d}, K ∈ ℝ^{T×d}, V ∈ ℝ^{T×d}
- Score matrix S = Q K^T / √d ∈ ℝ^{T×T}
- Causal mask M_full ∈ {0, −∞}^{T×T}: M_full[t, s] = 0 if s ≤ t, else −∞
- Output O = softmax(S + M_full) · V ∈ ℝ^{T×d}

The resulting attention pattern is a lower-triangular matrix: every query
attends to all tokens from position 0 through its own position.

## Windowed Attention — Formal Definition

Windowed (sliding-window) attention introduces a scalar hyperparameter w called
the window size (w ≥ 1). The query at position t is restricted to the w most
recent tokens, including itself:

$$\mathcal{A}_{\text{win}}(t) = \{\max(0,\, t - w + 1),\; \ldots,\; t\}$$

When t ≥ w − 1 the window is exactly w tokens wide: positions [t − w + 1, t].
When t < w − 1 (early positions) the window is naturally bounded by the
sequence start, giving a window of size t + 1.

The attention output formula is identical to full attention with the smaller
attendee set substituted:

$$o_t = \sum_{s \in \mathcal{A}_{\text{win}}(t)} \alpha_{t,s} \, v_s$$

$$\alpha_{t,s} = \frac{\exp\!\left(\dfrac{q_t \cdot k_s}{\sqrt{d}}\right)}
               {\displaystyle\sum_{s' \in \mathcal{A}_{\text{win}}(t)}
                \exp\!\left(\dfrac{q_t \cdot k_{s'}}{\sqrt{d}}\right)}$$

(For numerical stability the standard max-subtraction trick applies, with the max taken over the same window 𝒜_win(t).)

In matrix form, the window constraint replaces the causal mask with a
band-diagonal mask:

$$M_{\text{win}}[t, s] =
\begin{cases}
0      & \text{if } \max(0, t-w+1) \leq s \leq t \\
-\infty & \text{otherwise}
\end{cases}$$

The output is then O = softmax(S + M_win) · V, where S = Q K^T / √d as before.
Outside the band, the −∞ mask entries reduce to zero probability after softmax,
so those V vectors contribute nothing to the output.

## Side-by-Side Attention Mask Diagrams

The diagrams below show the allowed attention positions (marked `1`) for a
sequence of T = 8 tokens. Rows are query positions (0 = top), columns are key
positions (0 = left).

**Full causal attention mask (T = 8):**

```text
Key position →  0 1 2 3 4 5 6 7
               ─────────────────
Query pos 0  │  1 . . . . . . .
Query pos 1  │  1 1 . . . . . .
Query pos 2  │  1 1 1 . . . . .
Query pos 3  │  1 1 1 1 . . . .
Query pos 4  │  1 1 1 1 1 . . .
Query pos 5  │  1 1 1 1 1 1 . .
Query pos 6  │  1 1 1 1 1 1 1 .
Query pos 7  │  1 1 1 1 1 1 1 1
```

Every query attends to all tokens from position 0 through its own position.
The number of attended positions grows linearly: query t attends to t + 1 tokens.

**Windowed attention mask (T = 8, w = 3):**

```text
Key position →  0 1 2 3 4 5 6 7
               ─────────────────
Query pos 0  │  1 . . . . . . .
Query pos 1  │  1 1 . . . . . .
Query pos 2  │  1 1 1 . . . . .
Query pos 3  │  . 1 1 1 . . . .
Query pos 4  │  . . 1 1 1 . . .
Query pos 5  │  . . . 1 1 1 . .
Query pos 6  │  . . . . 1 1 1 .
Query pos 7  │  . . . . . 1 1 1
```

## Complexity Comparison

The table below gives asymptotic complexity for the two attention variants as a
function of total sequence length T, window size w, number of heads H, batch
size B, and head dimension d. For clarity the B and H factors are omitted from
the O() expressions; they multiply through identically for both variants.

### Prefill (processing a full sequence of length T in one pass)

| Operation          | Full attention       | Windowed attention  | Notes                                      |
|--------------------|----------------------|---------------------|--------------------------------------------|
| QK score matrix    | O(T² · d)            | O(T · w · d)        | Band-diagonal vs full lower triangle       |
| Softmax            | O(T²)                | O(T · w)            | Denominator sums over w entries per row    |
| AV weighted sum    | O(T² · d)            | O(T · w · d)        | Same structure as QK                       |
| KV cache write     | O(T · d)             | O(T · d)            | All K/V must still be computed and stored  |
| **Total FLOPs**    | **O(T² · d)**        | **O(T · w · d)**    | Saving factor: T / w                       |

For T = 32 768 and w = 4 096 the saving factor is 8×. For w = 8 192 it is 4×.

### Decode (single new token attending to prior context)

| Resource               | Full attention at step T | Windowed attention | Notes                                         |
|------------------------|--------------------------|---------------------|-----------------------------------------------|
| KV reads from DRAM     | O(T · d)                 | O(w · d)            | Dominant bandwidth cost; window caps it at w  |
| QK FLOPs               | O(T · d)                 | O(w · d)            | Dot products with T or w key vectors          |
| AV FLOPs               | O(T · d)                 | O(w · d)            | Weighted sum over T or w value vectors        |
| KV cache size (bytes)  | 2 · B · H · T · d · dtype_bytes  | 2 · B · H · w · d · dtype_bytes | Windowed cache has fixed size regardless of T |
| **DRAM BW per step**   | **grows with T**         | **constant in T**   | Critical for long-generation scenarios        |

The decode row labelled "KV reads from DRAM" is the primary performance driver
on Wormhole hardware, which is DRAM-bandwidth-limited at batch = 1 (see
Chapter 7 roofline analysis). Full attention read volume grows without bound;
windowed attention read volume is capped at w regardless of generation length.

---

**Next:** [`window_size_parameter.md`](./window_size_parameter.md)
