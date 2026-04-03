# RoPE Precomputation and Storage

This file covers the precomputation strategy for the two separate sets of
cos/sin embedding tables required by Gemma 4 31B, their memory footprint,
and the per-step slicing approach for decode inference on the T3K mesh.

## Two Sets of Cos/Sin Tables

Gemma 4 31B requires two distinct cos/sin table pairs --- one for each RoPE
configuration:

| Table Set | Layer Type | $\theta$ | Reference Table Shape | Rotary Dims |
|-----------|-----------|----------|----------------------|-------------|
| Sliding | 50 sliding layers | 10,000 | `[max_seq_len, 256]` | 256 (all) |
| Global | 10 global layers | 1,000,000 | `[max_seq_len, 128]` | 128 (first 25%) |

The global table shape of `[max_seq_len, 128]` matches the HuggingFace
reference implementation (Strategy B). If using the full-width TTNN
optimization (Strategy A from
[`global_proportional_rope.md`](./global_proportional_rope.md)), the global
table shape expands to `[max_seq_len, 512]` with identity values in the
non-rotated columns.

### Sliding Table Computation

```python
# theta = 10,000, head_dim = 256, full rotation
inv_freq_sliding = 1.0 / (10_000.0 ** (torch.arange(0, 256, 2).float() / 256.0))
# Shape: [128]

positions = torch.arange(0, max_seq_len).float()
freqs_sliding = torch.outer(positions, inv_freq_sliding)   # [max_seq_len, 128]
emb_sliding = torch.cat([freqs_sliding, freqs_sliding], dim=-1)  # [max_seq_len, 256]

cos_sliding = emb_sliding.cos()   # [max_seq_len, 256]
sin_sliding = emb_sliding.sin()   # [max_seq_len, 256]
```

### Global Table Computation (Strategy A --- Full-Width, TTNN Optimization)

This approach does NOT match the HuggingFace reference. It is a TTNN
optimization that encodes identity values into the non-rotated columns.

```python
# theta = 1,000,000, head_dim = 512, partial_rotary_factor = 0.25
# dim = int(head_dim * partial_rotary_factor) = 128
dim = int(512 * 0.25)  # = 128
rope_angles = dim // 2  # = 64
nope_angles = 512 // 2 - rope_angles  # = 192

inv_freq_rotated = 1.0 / (1_000_000.0 ** (torch.arange(0, dim, 2).float() / dim))
# Shape: [64], denominator is dim=128 (NOT head_dim=512)

inv_freq_global = torch.cat([inv_freq_rotated, torch.zeros(nope_angles)])
# Shape: [256]

freqs_global = torch.outer(positions, inv_freq_global)     # [max_seq_len, 256]
emb_global = torch.cat([freqs_global, freqs_global], dim=-1)  # [max_seq_len, 512]

cos_global = emb_global.cos()     # [max_seq_len, 512]
sin_global = emb_global.sin()     # [max_seq_len, 512]
```

### Global Table Computation (Strategy B --- Narrow, HuggingFace Reference)

```python
# theta = 1,000,000, dim = 128 (= head_dim * partial_rotary_factor)
inv_freq_narrow = 1.0 / (1_000_000.0 ** (torch.arange(0, 128, 2).float() / 128.0))
# Shape: [64], denominator is dim=128 (NOT head_dim=512)

freqs_narrow = torch.outer(positions, inv_freq_narrow)     # [max_seq_len, 64]
emb_narrow = torch.cat([freqs_narrow, freqs_narrow], dim=-1)  # [max_seq_len, 128]

cos_global_narrow = emb_narrow.cos()   # [max_seq_len, 128]
sin_global_narrow = emb_narrow.sin()   # [max_seq_len, 128]
```

## Memory Footprint

### Per-Table Memory at BF16

Each table element is 2 bytes (BF16). The memory for a single table (cos or
sin) is:

```math
\text{table\_bytes} = \texttt{max\_seq\_len} \times \texttt{width} \times 2
```

### At 256K Context (`max_seq_len = 262144`)

| Table | Shape | Bytes | MB |
|-------|-------|-------|----|
| `cos_sliding` | [262144, 256] | 134,217,728 | 128.0 |
| `sin_sliding` | [262144, 256] | 134,217,728 | 128.0 |
| `cos_global` (full-width) | [262144, 512] | 268,435,456 | 256.0 |
| `sin_global` (full-width) | [262144, 512] | 268,435,456 | 256.0 |
| **Total (Strategy A)** | | **805,306,368** | **768.0** |
| `cos_global` (narrow) | [262144, 128] | 67,108,864 | 64.0 |
| `sin_global` (narrow) | [262144, 128] | 67,108,864 | 64.0 |
| **Total (Strategy B)** | | **402,653,184** | **384.0** |

Strategy B (narrow tables) matches the HuggingFace reference implementation
and saves 384 MB of DRAM per device compared to Strategy A (full-width TTNN
optimization). On a 12 GB DRAM budget per Wormhole chip, this is a
significant saving (6.4% to 3.2% of total DRAM).

### At Shorter Context Lengths

| `max_seq_len` | Strategy A Total | Strategy B Total |
|---------------|------------------|------------------|
| 8,192 (8K) | 24.0 MB | 12.0 MB |
| 32,768 (32K) | 96.0 MB | 48.0 MB |
| 131,072 (128K) | 384.0 MB | 192.0 MB |
| 262,144 (256K) | 768.0 MB | 384.0 MB |

For long-context deployments, the cos/sin tables become a non-trivial fraction
of the DRAM budget, making Strategy B increasingly attractive.

### Memory Optimization: BFP8 Tables

Cos/sin values are bounded in $[-1, 1]$ and vary smoothly. Storing the tables
in BFP8 (`bfloat8_b`, 1 byte per element) would halve the memory:

| Strategy | BF16 Total (256K) | BFP8 Total (256K) |
|----------|-------------------|-------------------|
| A | 768 MB | 384 MB |
| B | 384 MB | 192 MB |

The precision loss from BFP8 quantization of cos/sin values is small for
low-frequency dimensions (where values change slowly) but may introduce
noticeable error for the highest-frequency pairs. This should be validated
empirically before deployment.

## Storage Strategy: Precompute at Init, Store in DRAM

### Lifecycle

1. **Model initialization (host-side):** Compute all four tables (cos/sin for
   sliding and global) on the CPU using float32 arithmetic, then quantize to
   BF16.

2. **Transfer to device:** Convert each table to a `ttnn.Tensor` with
   `TILE_LAYOUT` and write to each device's DRAM. Under TP=8, replicate
   all tables on all 8 devices (see
   [`sliding_rope.md`](./sliding_rope.md) for the replication rationale).

3. **Per-step slicing (decode):** At each decode step, slice the row
   corresponding to the current position from each table. For prefill, slice
   the range of rows corresponding to the input sequence.

4. **No recomputation:** The tables are static and do not change during
   inference. There is no need to recompute or update them after
   initialization.

### TTNN Tensor Properties

| Property | Value |
|----------|-------|
| Dtype | `ttnn.bfloat16` |
| Layout | `TILE_LAYOUT` |
| Memory config | `ttnn.DRAM_MEMORY_CONFIG` |
| Persistent | Yes (allocated at init, freed at model teardown) |

### Per-Step Slice Operation

During decode at position $p$:

```python
# Sliding layers
cos_s = ttnn.slice(cos_sliding, [p, 0], [p + 1, 256])   # [1, 256]
sin_s = ttnn.slice(sin_sliding, [p, 0], [p + 1, 256])   # [1, 256]

# Global layers (Strategy B --- HuggingFace reference, narrow tables)
cos_g = ttnn.slice(cos_global, [p, 0], [p + 1, 128])    # [1, 128]
sin_g = ttnn.slice(sin_global, [p, 0], [p + 1, 128])    # [1, 128]

# Global layers (Strategy A --- full-width TTNN optimization)
# cos_g = ttnn.slice(cos_global, [p, 0], [p + 1, 512])  # [1, 512]
# sin_g = ttnn.slice(sin_global, [p, 0], [p + 1, 512])  # [1, 512]
```

The sliced tensors are small (256 bytes for sliding, 256 bytes for narrow
global, or 1024 bytes for full-width global, at BF16) and can reside in L1
SRAM during the RoPE kernel execution.

During prefill with a sequence of length $S$:

```python
cos_s = ttnn.slice(cos_sliding, [0, 0], [S, 256])       # [S, 256]
sin_s = ttnn.slice(sin_sliding, [0, 0], [S, 256])       # [S, 256]
```

## Position Indexing

### Decode Mode

In autoregressive decode, each step processes a single new token. The position
index increments by 1 at each step. Both table sets are indexed with the same
position (the absolute sequence position of the new token).

### Prefill Mode

During prefill, a batch of $S$ tokens is processed simultaneously. The
position indices are `[0, 1, 2, ..., S-1]` for the initial prompt. The
cos/sin slice spans these rows contiguously.

### Continuation After Prefill

After prefilling $S$ tokens, the first decode step uses position $p = S$.
Subsequent steps use $p = S + 1, S + 2, \ldots$ This is identical for both
sliding and global tables --- both use absolute positions.

## Interaction With the Layer Loop

The 60-layer decoder loop must select the correct cos/sin tables for each
layer. A clean implementation precomputes both sets of position embeddings
before entering the layer loop:

```python
# Before the 60-layer loop
cos_sin_sliding = (cos_s, sin_s)   # precomputed for this step
cos_sin_global = (cos_g, sin_g)    # precomputed for this step

for layer_idx in range(60):
    layer_type = config.layer_types[layer_idx]
    if layer_type == "sliding_attention":
        position_embeddings = cos_sin_sliding
    else:  # "full_attention"
        position_embeddings = cos_sin_global

    hidden_states = decoder_layers[layer_idx](
        hidden_states,
        position_embeddings=position_embeddings,
        ...
    )
```

This avoids re-slicing the tables inside the loop. The HuggingFace
implementation follows the same pattern: `Gemma4TextRotaryEmbedding` computes
position embeddings for all layer types in a single call, and each layer
receives its type-appropriate `(cos, sin)` tuple.

## Summary of Decisions

| Decision | Recommended Choice | Rationale |
|----------|-------------------|-----------|
| Table width (global) | Narrow (Strategy B) for bringup; full-width (Strategy A) if dispatch overhead dominates | Matches HF reference first, then optimize |
| Dtype | BF16 | Standard precision; BFP8 requires validation |
| Storage location | DRAM on all 8 devices | Avoids per-step CCL for table access |
| Lifecycle | Precompute at init | Tables are static |
| Position indexing | Absolute positions | Both layer types use the same position counter |
| Per-step slicing | `ttnn.slice` from DRAM | Small output fits in L1 |

---

**Next:** [Chapter 5 --- Heterogeneous Attention Module Design](../ch5_attention_module_design/index.md)
