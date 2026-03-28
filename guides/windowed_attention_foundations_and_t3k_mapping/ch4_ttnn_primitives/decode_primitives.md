# Decode TTNN Primitives

During decode the model generates one new token per forward pass. At generation
step T the attention layer receives a single query vector per head, updates the
circular-buffer KV cache with the new key and value vectors, and computes the
attended output. This document specifies every TTNN operation involved in this
pipeline, the tensor shapes at each op boundary, the program configuration knobs
relevant to performance, and the mechanism by which the window constraint is
enforced.

## Tensor Shape Conventions for Decode

Throughout this file the following notation is used:

- `B` — batch size (number of concurrent sequences)
- `H_q` — number of query heads
- `H_kv` — number of key/value heads; `H_kv ≤ H_q`, `H_q / H_kv` is an integer
- `w` — window size (circular-buffer capacity)
- `d` — head dimension
- `T` — absolute token position (0-indexed from prompt start)

Grouped query attention (GQA) is supported whenever `H_q > H_kv`. Multi-head
attention (MHA) is the special case `H_q = H_kv`. Multi-query attention (MQA)
is the special case `H_kv = 1`.

## Step-by-Step Decode Pipeline

### Step 1 — Project the New Token

The embedding of the new token is projected through `W_q`, `W_k`, `W_v` weight
matrices. This produces three vectors with shapes:

```text
q_T :  [B, H_q,  1, d]    — query vector for position T
k_T :  [B, H_kv, 1, d]    — key vector for position T
v_T :  [B, H_kv, 1, d]    — value vector for position T
```

These projections are standard `ttnn.matmul` calls against the weight tensors
and are not specific to windowed attention; they are included for completeness.
RoPE rotations are applied to `q_T` and `k_T` at this point using T (the
absolute token position) as the rotation index.

### Step 2 — Write New KV into the Circular Buffer

Before attention computation, the new key and value vectors are inserted into
the circular-buffer KV cache at slot `T mod w`:

```python
update_index = T % w
ttnn.update_cache(k_cache, k_T, update_index)   # in-place write
ttnn.update_cache(v_cache, v_T, update_index)   # in-place write
pos_offset = max(0, T - w + 1)
```

**Tensor shapes:**

```text
k_cache :  [B, H_kv, w, d]   — full circular-buffer key cache (DRAM, in-place)
v_cache :  [B, H_kv, w, d]   — full circular-buffer value cache (DRAM, in-place)
k_T     :  [B, H_kv, 1, d]   — new key vector (input to update_cache)
v_T     :  [B, H_kv, 1, d]   — new value vector (input to update_cache)
```

`ttnn.update_cache` performs an in-place scatter write to slice
`[:, :, update_index, :]` of the cache tensor. The cache tensor shape is never
modified; only the data at the target slot changes. After this write, the cache
holds `min(T+1, w)` valid entries. The write pointer is implicit in
`update_index`; `pos_offset` is maintained by the decode loop as a scalar.

**Write-before-read ordering:** `ttnn.update_cache` must complete before the
attention call that reads back `k_cache` and `v_cache`. In TTNN's asynchronous
command queue model the ordering is guaranteed by issuing the two `update_cache`
calls before the attention call in the same command queue. There is no explicit
synchronisation primitive required as long as all ops run on the same device
command queue.

### Step 3 — Compute Scaled Dot-Product Attention

After the cache update the attention output is computed. Two implementation
paths are available: the explicit three-op sequence using `ttnn.matmul` and
`ttnn.softmax`, and the preferred fused op
`ttnn.scaled_dot_product_attention_decode`.

#### Path A — Explicit Three-Op Sequence

For clarity the windowed decode attention can be expressed as three separate
TTNN calls:

```python
# QK dot-product
# q_T:  [B, H_q,  1, d]
# k_T_win: [B, H_kv, w, d]  (or [B, H_q, w, d] after GQA expansion)
scale = 1.0 / (d ** 0.5)
scores = ttnn.matmul(q_T, ttnn.transpose(k_win, -1, -2))   # [B, H_q, 1, w]
scores = scores * scale

# Optional: apply position mask for fill phase (T <= w-1)
# scores += position_mask    # [B, 1, 1, w], -inf for invalid slots

# Softmax over the w key positions
attn_weights = ttnn.softmax(scores, dim=-1)   # [B, H_q, 1, w]

# AV multiply
# v_win: [B, H_kv, w, d]  (or [B, H_q, w, d] after GQA expansion)
output = ttnn.matmul(attn_weights, v_win)     # [B, H_q, 1, d]
```

**Tensor shapes at each op:**

```text
q_T           :  [B, H_q,  1, d]    — query (DRAM or L1)
k_win         :  [B, H_kv, w, d]    — window-limited K slice (DRAM)
scores (QK^T) :  [B, H_q,  1, w]    — after GQA broadcast of k_win
attn_weights  :  [B, H_q,  1, w]    — after softmax
v_win         :  [B, H_kv, w, d]    — window-limited V slice (DRAM)
output        :  [B, H_q,  1, d]    — attended output
```

GQA expansion (`H_q > H_kv`) is implicit in `ttnn.matmul`'s broadcast semantics
when K/V heads are broadcast across the query-head group, or can be made explicit
by repeating K and V heads `H_q / H_kv` times before the matmul. The fused op
in Path B handles this natively.

#### Path B — `ttnn.scaled_dot_product_attention_decode`

The preferred implementation for production decode is the fused op:

```python
output = ttnn.scaled_dot_product_attention_decode(
    q_T,          # [B, H_q,  1, d]
    k_cache,      # [B, H_kv, w, d]
    v_cache,      # [B, H_kv, w, d]
    attn_mask=position_mask,   # [B, 1, 1, w] or None; -inf for invalid slots
    scale=1.0 / (d ** 0.5),
    program_config=sdpa_decode_cfg,
)
# output: [B, H_q, 1, d]
```

This op computes Q × K^T, scales, applies the optional mask, runs softmax, and
multiplies by V in a single fused kernel. GQA is handled natively: when
`H_q > H_kv` the op broadcasts K and V heads across each query-head group
without materialising the expanded tensors.

**Interface summary:**

| Argument           | Shape                   | Notes                                        |
|--------------------|-------------------------|----------------------------------------------|
| `input` (Q)        | `[B, H_q, 1, d]`        | Single query vector per head                 |
| `key` (K cache)    | `[B, H_kv, w, d]`       | Full circular buffer; GQA if H_q > H_kv     |
| `value` (V cache)  | `[B, H_kv, w, d]`       | Full circular buffer; same GQA semantics     |
| `attn_mask`        | `[B, 1, 1, w]` or None  | Optional per-sequence mask; -inf = exclude   |
| `scale`            | scalar float            | Typically `1/sqrt(d)`                        |
| output             | `[B, H_q, 1, d]`        | Attended output                              |

The op accepts the full `[B, H_kv, w, d]` cache tensor and reads all `w` slots.
The window constraint (excluding slots written to positions beyond the window, or
padding slots in the fill phase T <= w-1) is communicated via `attn_mask`.

## Program Configuration Knobs

`ttnn.scaled_dot_product_attention_decode` accepts a
`ScaledDotProductAttentionDecodeOpSeqlenqChunkSize`-style program config
(named `SDPADecodeProgramConfig` in the TTNN API). The relevant fields are:

### `compute_with_storage_grid_size`

Specifies the 2-D core grid dimensions on Wormhole (e.g., `(8, 8)` for a 64-core
grid). The decode op tiles the batch and head dimensions across the available
cores. For a T3K single-device run, the full Wormhole grid is `8 × 8 = 64` cores.
Smaller grids reduce resource usage but lower parallelism.

```text
Tile assignment in decode (illustrative, B=8, H_q=32, grid=(8,4)=32 cores):

  Each core handles one (batch_idx, head_group) pair.
  With H_q / H_kv = 4 (GQA group size 4), each core handles 4 query heads
  against 1 KV head, computing a [4, 1, w] score matrix.
```

### `q_chunk_size`

Number of query heads processed per compute tile within a single core. Smaller
values reduce L1 SRAM pressure (fewer Q vectors resident simultaneously) at the
cost of more compute iterations. In decode, because there is only 1 query
sequence position (`S_q = 1`), `q_chunk_size` typically applies to chunking
along the head dimension. The default is often 32 (process all H_q / cores heads
at once if they fit).

### `k_chunk_size`

Number of key positions processed per compute tile within a single core. The
kernel loads a tile of `k_chunk_size` K vectors and a corresponding V tile into
L1, computes the partial QK^T scores for those positions, accumulates into a
running softmax denominator (online softmax), then loads the next K tile. Larger
`k_chunk_size` values improve DRAM burst efficiency (larger DMA requests) but
require more L1. Smaller values allow finer-grained overlap of DRAM loads with
compute.

For windowed decode with `w` positions the kernel iterates
`ceil(w / k_chunk_size)` times per head. A typical setting is 256 or 512; for
`w = 4096` and `k_chunk_size = 512` this yields 8 iterations.

```text
L1 working set per core per iteration (k_chunk_size = 512, d = 128, BF16):

  K tile:  512 × 128 × 2 =  131,072 bytes  (128 KiB)
  V tile:  512 × 128 × 2 =  131,072 bytes  (128 KiB)
  Q tile:    1 × 128 × 2 =      256 bytes  (per head; multiply by heads_per_core
                                             for total Q L1)
  Scores:    1 × 512 × 2 =    1,024 bytes
  Total:                    ≈ 260 KiB per head

  Note: the 260 KiB figure assumes heads are processed one at a time
  (as scaled_dot_product_attention_decode typically operates). When
  heads_per_core > 1 the Q tile and score buffer scale proportionally,
  but K and V tiles are shared across heads in the same group and do not
  multiply by heads_per_core.

  Wormhole L1 SRAM per core: 1536 KiB — ample headroom.
```

### `packer_l1_acc`

Boolean flag that enables packing intermediate results directly into L1 accumulators
rather than writing back to DRAM between tiles. Recommended `True` for decode.

## Window Constraint Enforcement

There are two strategies for ensuring that only valid window positions contribute
to the attention output. Both are correct; they differ in implementation
complexity and memory requirements.

### Strategy 1 — Window-Limited K/V Slice

Pass only the `min(T+1, w)` valid K and V vectors to the attention op by
slicing the circular buffer down to its valid region before calling SDPA decode.

```python
n_valid = min(T + 1, w)
k_win = k_cache[:, :, :n_valid, :]   # [B, H_kv, n_valid, d]
v_win = v_cache[:, :, :n_valid, :]   # [B, H_kv, n_valid, d]
output = ttnn.scaled_dot_product_attention_decode(q_T, k_win, v_win, ...)
```

This is correct only during the **fill phase** (T < w), where the buffer is not
yet full and the valid entries happen to be physically contiguous starting at
slot 0. In steady state (T ≥ w) the valid entries span the full `w` slots but
are not necessarily in temporal order (the wrap-around layout from Chapter 2),
so slicing `[:, :, :w, :]` gives all `w` slots correctly and no further slicing
is needed. In steady state this strategy degenerates to passing the full cache.

The key advantage is that the attention op sees a tensor with exactly `n_valid`
slots; no masking is required. The disadvantage is that `n_valid` changes on
every step during the fill phase, potentially requiring dynamic shape handling or
padding to a fixed-size tensor for the full-buffer steady-state case.

### Strategy 2 — Full Buffer with Position Mask

Pass the full `[B, H_kv, w, d]` cache buffer to the attention op at all times
and supply a per-step mask that sets invalid slots to `-inf`:

```python
# Build mask: shape [B, 1, 1, w]
# Slot s is valid if its absolute position is in [pos_offset, T]
# Equivalently: slot s holds position (s - pos_offset) mod w + pos_offset
# Valid iff that position >= pos_offset (always true) and <= T
# During fill phase (T < w): slots [T+1, w-1] are uninitialised → mask out
# In steady state: all w slots are valid → mask is all-zeros
n_valid = min(T + 1, w)
position_mask = torch.zeros(B, 1, 1, w)
if n_valid < w:
    position_mask[:, :, :, n_valid:] = float('-inf')
position_mask_tt = ttnn.from_torch(position_mask, dtype=ttnn.bfloat16, ...)

output = ttnn.scaled_dot_product_attention_decode(
    q_T, k_cache, v_cache, attn_mask=position_mask_tt, ...
)
```

```text
Position mask at step T=5, w=8 (fill phase, 6 valid slots):

  Slot:   0     1     2     3     4     5     6     7
         ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
  Mask:  │  0  │  0  │  0  │  0  │  0  │  0  │-inf │-inf │
         └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
          valid (slots 0–5 hold positions 0–5)  ← invalid

Position mask at T=11, w=8 (steady state, all 8 slots valid):

  Slot:   0     1     2     3     4     5     6     7
         ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
  Mask:  │  0  │  0  │  0  │  0  │  0  │  0  │  0  │  0  │
         └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
          all valid (zero mask = no masking needed)
```

Note that in steady state the mask encodes no band constraint — the circular
buffer already guarantees that only the `w` most recent tokens are present. The
windowed constraint is enforced implicitly by the fixed buffer capacity rather
than explicitly by the mask.

## Memory Configurations

TTNN exposes memory config choices that affect whether tensors live in DRAM or
L1 and how they are sharded across cores. The relevant choices for decode are:

### Q Tensor: `[B, H_q, 1, d]`

The query tensor is small (one vector per head per batch item) and is freshly
computed each step. Two configurations are practical:

| Config                | Description                                        | When to use                                   |
|-----------------------|----------------------------------------------------|-----------------------------------------------|
| `DRAM_MEMORY_CONFIG`  | Tensor lives in DRAM; cores DMA-fetch as needed    | Default; minimal L1 pressure                  |
| `L1_MEMORY_CONFIG`    | Tensor lives in L1; zero DMA cost for the Q read   | Beneficial if Q projection and SDPA share cores |
| `HEIGHT_SHARDED`      | Q rows (batch × heads) sharded across core grid    | Large-batch scenarios where Q is a bottleneck |

For batch 1 decode, placing Q in L1 avoids a DRAM round-trip and is recommended
when the Q tensor fits (B × H_q × d × 2 bytes; for B=1, H_q=32, d=128 this is
8 KiB — trivially small).

### K and V Tensors: `[B, H_kv, w, d]`

The KV cache is the dominant DRAM consumer. Options:

| Config                   | Description                                          | Notes                                             |
|--------------------------|------------------------------------------------------|---------------------------------------------------|
| `DRAM_MEMORY_CONFIG`     | Full cache in DRAM; loaded tile-by-tile during SDPA  | Required when cache exceeds L1; the common case   |
| `HEIGHT_SHARDED` (L1)    | Cache shards across cores; each core holds its slice | Only viable for very small `w` or many cores       |
| Interleaved DRAM         | Default DRAM interleaving across DRAM channels       | Maximises aggregate DRAM bandwidth                |

For all practical window sizes (`w ≥ 512`) the KV cache tensors reside in
DRAM. At `w = 4096`, `H_kv = 8`, `d = 128`, bfloat16, one KV cache tensor is:

$$B \times 8 \times 4096 \times 128 \times 2 = B \times 8 \text{ MiB}$$

Well above L1 capacity per core (1.5 MiB). DRAM interleaved is the correct and
only feasible config. The `scaled_dot_product_attention_decode` kernel handles
the tiled DRAM→L1 streaming internally; the caller need not manage tiles
manually.

### Output Tensor: `[B, H_q, 1, d]`

The output is the same shape as Q. It is written to DRAM by default
(`DRAM_MEMORY_CONFIG`) and subsequently used as input to the output projection
`W_o`. For high-throughput decode, placing the output in L1 and feeding it
directly to the `W_o` matmul in the same L1 buffer can save one DRAM write and
one DRAM read. Whether this is beneficial depends on whether the projection
matmul is scheduled on the same core grid.

## Complete Decode Step: Op Sequence Summary

```text
Input: token embedding [B, 1, model_dim]

  ┌──────────────────────────────────────────────────────────────────┐
  │  ttnn.matmul × 3   — project to Q, K, V                         │
  │  shapes out: Q [B, H_q, 1, d]  K [B, H_kv, 1, d]  V [B, H_kv, 1, d] │
  └──────────────────────────┬───────────────────────────────────────┘
                             │
  ┌──────────────────────────▼───────────────────────────────────────┐
  │  RoPE rotation (custom kernel or ttnn elementwise)               │
  │  uses T (the absolute token position) to derive rotation angles  │
  └──────────────────────────┬───────────────────────────────────────┘
                             │
  ┌──────────────────────────▼───────────────────────────────────────┐
  │  ttnn.update_cache(k_cache, k_T, T % w)                          │
  │  ttnn.update_cache(v_cache, v_T, T % w)                          │
  │  cache shapes: [B, H_kv, w, d]  (in-place DRAM write, 1 slot)   │
  └──────────────────────────┬───────────────────────────────────────┘
                             │
  ┌──────────────────────────▼───────────────────────────────────────┐
  │  ttnn.scaled_dot_product_attention_decode(                        │
  │      Q [B,H_q,1,d], K [B,H_kv,w,d], V [B,H_kv,w,d],            │
  │      attn_mask [B,1,1,w], scale, program_config)                 │
  │  output: [B, H_q, 1, d]                                          │
  └──────────────────────────┬───────────────────────────────────────┘
                             │
  ┌──────────────────────────▼───────────────────────────────────────┐
  │  ttnn.matmul — output projection W_o                             │
  │  output: [B, 1, model_dim]                                       │
  └──────────────────────────────────────────────────────────────────┘
```

The five op groups above constitute one complete windowed-attention decode step.
All KV state is maintained in the fixed-shape `[B, H_kv, w, d]` tensors between
steps; no tensor allocation or deallocation occurs per step in steady state.

---

**Next:** [`prefill_primitives.md`](./prefill_primitives.md)
