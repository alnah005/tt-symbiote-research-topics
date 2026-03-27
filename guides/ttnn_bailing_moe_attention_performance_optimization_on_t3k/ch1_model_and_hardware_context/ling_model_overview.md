# Ling Model Overview

Ling is a Mixture-of-Experts (MoE) language model built on the BailingMoeV2 architecture. Its attention layers implement Grouped Query Attention (GQA) with a partial rotary position embedding and optional QK normalization. This file defines the exact hyperparameters and structural choices that determine what the attention layer must compute on every forward pass, and explains why the prefill and decode execution paths have fundamentally different performance characters.

## BailingMoeV2 Architecture Overview

BailingMoeV2 is a decoder-only transformer. Each decoder layer contains:

1. An attention sublayer (`TTNNBailingMoEAttention`).
2. A Mixture-of-Experts feed-forward sublayer, consisting of a router and a set of expert MLPs where only a sparse subset of experts is activated per token.

This guide focuses exclusively on the attention sublayer. The MoE FFN is a separate performance domain and is not discussed here.

### Decoder Stack Structure

At a high level, each decoder block follows the standard pre-norm pattern:

```
input
  └─ RMSNorm
       └─ TTNNBailingMoEAttention   ← this guide's subject
            └─ residual add
                 └─ RMSNorm
                      └─ MoE FFN
                           └─ residual add
                                └─ output
```

`TTNNBailingMoEAttention` receives the normalized hidden state, projects it to Q, K, and V, applies rotary position embeddings (with optional QK normalization), performs paged scaled dot-product attention, and projects the result back to the hidden dimension.

## Attention Configuration

### Core Hyperparameters

The following hyperparameters are fixed by the Ling model checkpoint and are referenced throughout every chapter of this guide.

Attention hyperparameters for the Ling (BailingMoeV2) model.

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| `hidden_size` | 4096 | Dimension of the residual stream entering and leaving each attention layer |
| `num_heads` | 16 | Number of query (Q) attention heads |
| `num_kv_heads` | 4 | Number of key (K) and value (V) attention heads |
| `head_dim` | 128 | Dimension of each individual attention head; model hyperparameter (not derived from `hidden_size / num_heads`) — see note below |
| `partial_rotary_factor` | 0.5 | Fraction of `head_dim` to which rotary embeddings are applied |
| `use_qk_norm` | True | Whether per-head RMSNorm is applied to Q and K after projection |

Note on `head_dim`: The value 128 is a model configuration constant. It is not derived from `hidden_size / num_heads` in all configurations; treat it as an independent parameter. In Ling's case this gives a Q projection width of 2048 and KV projection width of 512 — see the GQA section below.

### Grouped Query Attention (16/4 GQA)

Ling uses **16/4 GQA**: 16 query heads share 4 key/value head pairs. Each KV head is shared by `num_heads / num_kv_heads = 16 / 4 = 4` query heads. This ratio is called the GQA grouping factor.

The practical consequences are:

- **KV cache is 4× smaller** than full multi-head attention (MHA) would require for the same number of Q heads. With `head_dim=128` and `bfloat16` dtype, each KV head occupies `128 * 2 = 256` bytes per token per layer, for a total KV footprint of `4 * 256 * 2 = 2048` bytes per token per layer (factor of 2 for K and V).
- **The attention kernel must broadcast** each KV head to the 4 Q heads that reference it. How this broadcast is performed is specific to the paged SDPA kernel and is discussed in (see Chapter 5, `paged_sdpa_chunk_sizes.md`).
- **QKV projection sizes are asymmetric.** The weight matrix for Q is `(hidden_size, num_heads * head_dim) = (4096, 2048)`, while K and V are each `(hidden_size, num_kv_heads * head_dim) = (4096, 512)`. When fused into a single projection, the combined output width is `2048 + 512 + 512 = 3072` columns. This asymmetry matters for column-sharding across 8 chips (see Chapter 2, `fusion_mechanics.md`).

### Partial Rotary Position Embedding

`partial_rotary_factor = 0.5` means that rotary embeddings are applied to only the **first half** of each head's `head_dim` dimensions. Concretely:

- The rotary-embedded portion covers dimensions `[0, head_dim * partial_rotary_factor) = [0, 64)` of each head.
- Dimensions `[64, 128)` are left unchanged (no rotation applied).

This is a design choice that reduces the computational cost of RoPE while retaining positional information in the lower-frequency components of the representation. The critical performance consequence of `partial_rotary_factor < 1.0` is examined in (see Chapter 6, `partial_rotary_rope.md`): it disables the distributed RoPE kernel (`TTNNDistributedRotaryPositionEmbedding`) and forces a non-distributed fallback (`TTNNRotaryPositionEmbedding`), which cannot exploit T3K's 8-chip parallelism for this operation.

### QK Normalization

When `use_qk_norm=True`, a per-head RMSNorm is applied to both the Q and K tensors after RoPE. This improves training stability in large MoE models but adds a reshape-normalize-reshape sequence to every decode step.

The specific operation sequence when `use_qk_norm=True` is:

1. Apply `TTNNRotaryPositionEmbedding` to Q and K (RoPE runs first).
2. Reshape Q from 3D `(batch, num_heads, head_dim)` to 2D to satisfy `TTNNRMSNorm`'s input shape requirement.
3. Apply `TTNNRMSNorm` to Q.
4. Reshape Q back to 3D.
5. Repeat steps 2–4 for K (using a separate norm weight).

Q and K go through RoPE first, then through `TTNNRMSNorm` if `use_qk_norm=True`.

This adds latency proportional to two RMSNorm kernel invocations plus four reshape operations per decode step. The cost is analyzed in (see Chapter 6, `qk_norm_latency.md`).

## TTNNBailingMoEAttention in the Decoder Stack

`TTNNBailingMoEAttention` is the TTNN-compiled attention module that replaces the standard PyTorch attention implementation for T3K execution. Its forward method, at a high level, performs:

```python
# Pseudocode — not exact TTNN API
def forward(hidden_states, cos, sin, page_table, kv_cache):
    # 1. Fused QKV projection (column-sharded across 8 chips, followed by all-reduce)
    qkv = TTNNLinearIColShardedWAllReduced(hidden_states, W_qkv)

    # 2. Host round-trip to convert from column-sharded to replicated layout
    qkv_replicated = _to_replicated(qkv)

    # 3. Split into Q, K, V
    q, k, v = split(qkv_replicated)

    # 4. Rotary position embedding (non-distributed due to partial_rotary_factor)
    q, k = TTNNRotaryPositionEmbedding(q, k, cos, sin)

    # 5. Optional QK norm — applied after RoPE
    if use_qk_norm:
        q = TTNNRMSNorm(reshape(q))  # with reshape pre/post
        k = TTNNRMSNorm(reshape(k))  # with reshape pre/post

    # 6. Update paged KV cache
    paged_update_on_device(k, v, page_table, kv_cache)

    # 7. Paged scaled dot-product attention
    attn_out = paged_sdpa_decode(q, kv_cache, page_table)

    # 8. Output projection
    out = TTNNLinear(attn_out, W_o)
    return out
```

Each of these steps involves specific tensor memory configurations and CCL operations that are the subject of the chapters that follow. The sequence above is the decode path; the prefill path differs substantially and is described in the next section.

## Prefill vs. Decode Execution Paths

The attention layer has two distinct execution modes that differ in input shape, parallelism strategy, and which operations dominate latency.

### Prefill

In the prefill phase, the model processes the full input prompt in a single forward pass. The input tensor shape is `(batch, seq_len, hidden_size)` where `seq_len` is the prompt length, which may range from tens to thousands of tokens.

Prefill characteristics:

- **Compute-bound.** With a long sequence, the QK^T matmul (`seq_len × seq_len × head_dim`) dominates, and the hardware's matrix-multiply engines (Tensix cores) are the bottleneck.
- **Sharding across sequence length is effective.** A long sequence can be split across chips in the sequence dimension, giving each chip a contiguous chunk to process.
- **Attention mask is causal.** The attention pattern is lower-triangular, which the SDPA kernel must account for with a causal mask.
- **KV cache is written but not read** for most of the sequence (read only for the last token's attention).

### Decode

In the decode phase, the model generates one new token per step. The input tensor shape is `(batch=1, seq_len=1, hidden_size)` — a single token per device step. This is the performance-critical regime for interactive inference.

Decode characteristics:

- **Memory-bandwidth-bound on the KV cache.** Each decode step reads the full accumulated KV cache (all prior tokens) for the attention computation, but computes only one new output token. The ratio of data read to arithmetic performed is high, making memory bandwidth the bottleneck rather than compute.
- **CCL and host transfer overhead is proportionally large.** Because the arithmetic work per step is tiny (one token), fixed overheads such as all-reduce communication, host round-trips, and kernel launch latency represent a significant fraction of total step time. This is the primary reason this guide focuses on decode: overheads that are negligible during prefill can dominate during decode.
- **Paged KV cache is required.** Ling uses a paged KV cache (analogous to virtual memory for KV storage) to handle variable-length sequences efficiently. The `paged_sdpa_decode` kernel and `paged_update_on_device` operation are decode-specific.

The table below summarizes the key differences that drive architectural choices:

Comparison of prefill and decode execution profiles for Ling's attention layer on T3K.

| Property | Prefill | Decode Step |
|----------|---------|-------------|
| Input shape | `(1, seq_len, 4096)`, `seq_len` ≥ 128 | `(1, 1, 4096)` |
| Dominant cost | QK^T matmul (compute) | KV cache read (memory bandwidth) |
| CCL overhead share | Small (diluted by long compute) | Large (dominates short compute) |
| Host round-trip cost | Negligible | Significant |
| SDPA kernel variant | Flash attention (chunked prefill) | Paged SDPA decode |
| KV cache role | Write-heavy | Read-heavy |

The remaining chapters of this guide analyze the decode path exclusively, except where prefill is explicitly mentioned for contrast.

---

**Next:** [`t3k_topology_primer.md`](./t3k_topology_primer.md)
