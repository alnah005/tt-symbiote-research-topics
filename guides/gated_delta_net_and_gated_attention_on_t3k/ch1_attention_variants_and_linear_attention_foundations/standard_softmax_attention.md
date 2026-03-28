# Standard Softmax Attention

## SDPA Formula

Causal scaled dot-product attention for a single head is:

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V$$

where:
- **$Q \in \mathbb{R}^{T \times d_k}$**, **$K \in \mathbb{R}^{T \times d_k}$**, **$V \in \mathbb{R}^{T \times d_v}$** are the query, key, and value projections for one head over a sequence of length T.
- **$M \in \mathbb{R}^{T \times T}$** is the causal mask: $M_{ij} = 0$ if $j \leq i$, $-\infty$ otherwise, ensuring position t can only attend to positions $j \leq t$.
- **$d_k$** is the per-head key/query dimension; the $1/\sqrt{d_k}$ factor prevents dot-product magnitude from growing with $d_k$ and stabilizes softmax gradients.
- The output shape is **[T, d_v]** per head; across all heads it is **[T, H]** where H is the model dimension (H=2048 in the target configuration).

The softmax normalizes each row of the T×T logit matrix independently, producing an attention pattern: a probability distribution over past positions for each query position. This pattern is what makes standard attention a general-purpose associative memory — any subset of past positions can receive non-negligible weight on any given query.

## Interpretation as Memory Retrieval

The attention pattern $a_t = \text{softmax}(q_t K^\top / \sqrt{d_k})$ computes a soft selection over all past value vectors. The output at position t is a weighted mixture of past values:

$$o_t = \sum_{j \leq t} a_{t,j} \cdot v_j$$

The retrieval quality is unconstrained: two very similar queries that should retrieve different memories can do so as long as the corresponding keys differ. This is the core strength of softmax attention — its expressiveness comes from the full T×T interaction matrix, which can represent arbitrary retrieval patterns.

The price is the **KV cache**. During autoregressive decode, every previously seen key and value vector must be stored so that attention over the full prefix can be computed. For a single layer and single batch element, the KV cache holds:

- Keys: **[T, d_k]** per head × n_kv_heads — total T × d_k × n_kv_heads entries
- Values: **[T, d_v]** per head × n_kv_heads — total T × d_v × n_kv_heads entries

The cache grows linearly with sequence length T and cannot be bounded independently of T without losing information. For long sequences (T in the tens of thousands) across many layers and batch elements, KV cache memory becomes the dominant constraint on serving throughput.

## Complexity

| Phase | FLOPs (per head) | Memory |
|-------|-----------------|--------|
| Prefill (full sequence) | $O(T^2 \cdot d_k)$ | $O(T \cdot d_k)$ KV cache + $O(T^2)$ peak activation (see note) |
| Decode (one step, full prefix) | $O(T \cdot d_k)$ | $O(T \cdot d_k)$ KV cache (grows each step) |

The $O(T^2)$ prefill cost comes from materializing the full T×T attention matrix; the $O(T)$ decode cost per step comes from computing dot products between the new query and all T cached keys. Neither cost can be reduced without approximation as long as the full softmax over all positions is computed.

> **Prefill peak memory note:** The $O(T \cdot d_k)$ figure in the table reflects only the KV cache that persists between steps. During prefill, the full T×T logit matrix must also be materialized before the softmax and the weighted sum with V can be computed — this is an additional $O(T^2)$ peak activation cost. For large T this dominates: at T=8192 and d_k=128 the attention matrix is 64× larger than the KV cache entries for that layer. This is the primary reason FlashAttention-style tiling is necessary: by processing the T×T matrix in blocks that fit in SRAM, it avoids ever allocating the full $O(T^2)$ buffer in HBM. A reader planning memory budgets for prefill should treat $O(T^2)$ as the binding constraint, not $O(T \cdot d_k)$.

## Grouped Query Attention (GQA)

Standard multi-head attention (MHA) assigns one independent K and V projection per query head. With n_q_heads query heads, there are n_q_heads key heads and n_q_heads value heads, producing a KV cache that scales with n_q_heads.

GQA reduces this by using **n_kv_heads < n_q_heads** key and value heads, where each KV head is shared by a group of query heads. Formally, if query head h belongs to group g(h), then:

$$\text{Attn}_h(Q_h, K, V) = \text{softmax}\!\left(\frac{Q_h K_{g(h)}^\top}{\sqrt{d_k}} + M\right) V_{g(h)}$$

All query heads within group g(h) attend to the same K and V matrices. This reduces KV cache size by a factor of n_q_heads / n_kv_heads with negligible quality loss at large model scale, as established empirically in the GQA paper (Ainslie et al. 2023).

**Gated Attention layers in Qwen3.5 use GQA with n_q_h=16 query heads and n_kv_h=2 KV heads**, a compression factor of 8×. Per-head dimension is d_h=256. The KV cache for a single Gated Attention layer at sequence length T is:

```
2 × T × d_h × n_kv_h = 2 × T × 256 × 2 = 1024 × T entries per layer per batch element
```

This is the baseline memory profile that linear attention variants aim to replace with a constant-size state.

> **Note on head dimensions across variants:** The d_h=256 figure above is specific to Gated Attention heads. DeltaNet and vanilla linear attention use d_k=128 (so their state matrix $S \in \mathbb{R}^{128 \times 128}$ = 16,384 entries per head). These two dimensions are **not** directly comparable; do not divide 1024 × T by 16,384 to derive a relative memory figure without accounting for the per-head dimension difference. Cross-variant memory comparisons using concrete numbers are deferred to Chapter 2, where both architectures are fully specified.

---

**Next:** [`linear_attention_rnn_equivalence.md`](./linear_attention_rnn_equivalence.md)
