# Linear Attention and Its RNN Equivalence

## Kernel Substitution

The softmax in standard attention is not an algebraic necessity — it is one choice of similarity function. Linear attention replaces it with a positive kernel function φ: R^{d_k} → R^{d_k'} that satisfies φ(x)^T φ(y) ≈ sim(x, y) for some similarity measure. A common minimal choice is φ(x) = elu(x) + 1 or simply φ(x) = x (identity), which gives an unbiased inner-product kernel.

The query-at-position-t output under kernel substitution becomes:

```
o_t = Attn(q_t, K, V)
    = φ(q_t)^T (sum_{j<=t} φ(k_j) v_j^T) / φ(q_t)^T (sum_{j<=t} φ(k_j))
```

The key observation is that the denominator and the numerator both factor into a dot product between φ(q_t) (which depends only on the current query) and a cumulative sum that depends only on the past context. Neither sum requires materializing a T×T matrix.

## State Matrix Formulation

Define the **state matrix** at position t as the outer-product accumulation of all past key-value pairs:

```
S_t = sum_{j<=t} φ(k_j) v_j^T   ∈ R^{d_k × d_v}
```

This is a fixed-size matrix regardless of sequence length T. It can be maintained as a running sum with the recurrence:

```
S_t = S_{t-1} + k_t v_t^T
```

(This recurrence uses the identity kernel φ(x) = x; for the general φ substitution rule, see the Kernel note in the RNN Recurrence section below.)

The denominator normalization term is the analogous scalar:

```
z_t = z_{t-1} + φ(k_t)   ∈ R^{d_k}
```

The output at position t is then:

```
o_t = S_t^T φ(q_t) / z_t^T φ(q_t)
```

All of these operations are O(d_k × d_v) per step — independent of T.

## RNN Recurrence

The resulting system is a true RNN with hidden state S_t ∈ R^{d_k × d_v}:

| Symbol | Meaning | Size |
|--------|---------|------|
| S_t | Hidden state (state matrix) | R^{d_k × d_v} = R^{128 × 128} |
| z_t | Normalization accumulator | R^{d_k} |
| k_t, v_t | Input-derived key and value | R^{d_k}, R^{d_v} |
| q_t | Input-derived query | R^{d_k} |
| o_t | Output at step t | R^{d_v} |

The recurrence at each step is:

```
S_t = S_{t-1} + k_t v_t^T          (state update)
z_t = z_{t-1} + k_t                (normalization accumulator)
o_t = S_t^T q_t / (z_t^T q_t)     (normalized readout)
```

> **Kernel note:** The bare forms `k_t v_t^T`, `S_t^T q_t`, `z_t^T q_t`, and `z_t = z_{t-1} + k_t` shown here all assume the identity kernel φ(x) = x (or L2-normalization, as used by DeltaNet). In particular, the normalization accumulator written as `z_t = z_{t-1} + k_t` is the identity-kernel specialization of the canonical form `z_t = z_{t-1} + φ(k_t)` defined in the State Matrix Formulation section above. In practice, many modern linear attention variants — including DeltaNet and most gated extensions — adopt the identity kernel or L2-normalization rather than a nontrivial φ, so all of these bare forms are valid for those variants without further substitution. For a non-identity kernel, replace every bare `k_t` and `q_t` in the state update, readout, and normalization accumulator with φ(k_t) and φ(q_t) respectively.

> **Denominator note:** The normalization denominator `z_t^T q_t` prevents `o_t` from growing unboundedly as S accumulates outer products over long sequences. Without it, the output magnitude scales with T. In practice, DeltaNet uses L2-normalized keys (`k̃_t` with `‖k̃_t‖ = 1`) and L2-normalized queries, which makes `z_t^T q_t ≈ constant` (approximately uniform across positions), so the denominator is typically omitted in DeltaNet's implementation. The general linear attention form requires the denominator.

There is no dependency on T in either equation. Decode is O(d_k × d_v) FLOPs and O(d_k × d_v) memory, independent of how long the sequence has grown. This is the fundamental advantage of linear attention over softmax attention for autoregressive generation.

For a typical configuration with d_k × d_v = 128 × 128, each head's state matrix holds 16,384 entries × 2 bytes = 32 KB — a fixed memory footprint per head regardless of sequence length T. The number of heads in specific model configurations (e.g., Gated Delta Net interleaved layers) is covered in Chapter 2.

## The Forgetting Problem

The additive recurrence `S_t = S_{t-1} + k_t v_t^T` accumulates all past tokens with equal weight. There is no mechanism to reduce the influence of older tokens or to remove stale associations. Over a long sequence, S becomes a superposition of all past writes:

```
S_T = sum_{t=1}^{T} k_t v_t^T
```

When a query q arrives, the retrieval `S_T^T q` computes a dot product against every past key simultaneously. For large T, the signal from any specific past token is diluted by the aggregate of all others. In the extreme, S approaches a constant matrix (by the law of large numbers over random keys), and all position-specific information is lost.

This is not a problem for short sequences or highly structured inputs where keys are nearly orthogonal. But for long-context tasks — where a model must retrieve a specific fact written thousands of tokens ago while ignoring intervening content — vanilla linear attention degrades badly. Empirically, perplexity diverges from that of softmax attention at context lengths where state saturation sets in.

The variants discussed in the next section each address this limitation differently, by introducing a forgetting gate G_t that modulates how much of the previous state S_{t-1} is retained before the new write k_t v_t^T is added.

---

**Next:** [`linear_attention_variants_comparison.md`](./linear_attention_variants_comparison.md)
