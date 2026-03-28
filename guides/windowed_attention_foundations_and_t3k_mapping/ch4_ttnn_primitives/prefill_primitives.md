# Prefill TTNN Primitives

During prefill the model processes the full prompt sequence of length T in a
single forward pass. All T query, key, and value vectors are computed simultaneously,
attention scores form a `[T, T]` matrix, and the windowed constraint is expressed
as a band-diagonal mask on that matrix. This document specifies the TTNN ops and
tensor shapes for the prefill path, explains how to construct the band mask as a
TTNN tensor, analyses whether `ttnn.scaled_dot_product_attention` with a mask
argument is sufficient, describes the side-effect of prefill on the
circular-buffer KV tensor, and weighs the trade-offs between a full-size masked
kernel and a chunked windowed kernel.

## Tensor Shape Conventions for Prefill

Notation follows [Chapter 4 — Decode Primitives](./decode_primitives.md#tensor-shape-conventions-for-decode); `T` here denotes prompt sequence length, not absolute token position.

All prompt tokens are processed simultaneously, so the sequence-length dimension
of Q, K, and V is `T` (not 1 as in decode).

## Prefill Op Sequence

### Step 1 — Project All Tokens to Q, K, V

The full prompt embedding `[B, T, model_dim]` is projected to Q, K, V in one
batched `ttnn.matmul` per weight matrix:

```python
Q = ttnn.matmul(x, W_q)   # [B, T, H_q * d]  → reshape to [B, H_q, T, d]
K = ttnn.matmul(x, W_k)   # [B, T, H_kv * d] → reshape to [B, H_kv, T, d]
V = ttnn.matmul(x, W_v)   # [B, T, H_kv * d] → reshape to [B, H_kv, T, d]
```

After reshape and head-split:

```text
Q :  [B, H_q,  T, d]
K :  [B, H_kv, T, d]
V :  [B, H_kv, T, d]
```

RoPE rotations are applied to Q and K using absolute position indices 0 through
T-1.

### Step 2 — Construct the Band-Diagonal Attention Mask

The prefill window constraint requires a `[T, T]` (or broadcastable `[1, 1, T,
T]`) mask where entry `[t, s]` is 0 (allow) when `max(0, t - w + 1) ≤ s ≤ t`
and `-inf` (suppress) otherwise. The mask encodes both causal masking (s > t is
always suppressed) and the sliding-window constraint (s < t - w + 1 is
suppressed for t ≥ w).

#### Band-Mask Construction: Torch Route

The most reliable construction path converts a CPU/host-side torch tensor to a
TTNN tensor. The construction exploits the decomposition of the band mask into
a lower-triangular mask minus a strictly-lower-than-(t-w) mask:

```python
import torch
import ttnn

# Causal lower-triangular: allows s <= t
causal = torch.ones(T, T).tril()

# Lower bound: suppress s < t - w + 1, i.e., suppress the rows below the band
# triu(-w+1) keeps elements where column >= row - (w-1), i.e., s >= t - w + 1
band = causal.triu(-(w - 1))   # [T, T], 1 = allowed, 0 = suppressed

# Convert to additive mask: 0 where allowed, -inf where suppressed
attn_bias = torch.zeros(T, T)
attn_bias[band == 0] = float('-inf')
attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)   # [1, 1, T, T]

# Transfer to TTNN device tensor
band_mask_tt = ttnn.from_torch(
    attn_bias,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
# band_mask_tt shape: [1, 1, T, T]
```

The `tril()` call enforces causality (future positions are suppressed). The
`triu(-(w-1))` call enforces the lower window bound (positions more than w-1
steps in the past are suppressed). Their intersection is exactly the
band-diagonal mask from
[`../ch3_data_dependencies/prefill_access_patterns.md`](../ch3_data_dependencies/prefill_access_patterns.md).

Worked example for T = 8, w = 3 (1 = allowed, 0 = suppressed):

```text
causal (tril):             triu(-(w-1)) = triu(-2):    band = intersection:

  1 . . . . . . .            1 1 1 1 1 1 1 1            1 . . . . . . .
  1 1 . . . . . .            1 1 1 1 1 1 1 1            1 1 . . . . . .
  1 1 1 . . . . .            1 1 1 1 1 1 1 1            1 1 1 . . . . .
  1 1 1 1 . . . .            . 1 1 1 1 1 1 1            . 1 1 1 . . . .
  1 1 1 1 1 . . .            . . 1 1 1 1 1 1            . . 1 1 1 . . .
  1 1 1 1 1 1 . .            . . . 1 1 1 1 1            . . . 1 1 1 . .
  1 1 1 1 1 1 1 .            . . . . 1 1 1 1            . . . . 1 1 1 .
  1 1 1 1 1 1 1 1            . . . . . 1 1 1            . . . . . 1 1 1
```

`triu(-2)` keeps entry `[row, col]` when `col >= row - 2`. Rows 0–2 satisfy
`col >= row - 2` for all columns (since the minimum column index is 0 and
`0 >= row - 2` holds for row ≤ 2), so all 8 entries are kept in those rows.
For row 3 col 0 is zeroed (0 < 3-2=1); for row 4 cols 0–1 are zeroed; and so
on. The result is a matrix of mostly 1s in the upper-left with a descending
staircase of zeros in the lower-left corner. The intersection with `tril`
(which keeps only the lower-left triangle) produces the w=3 band diagonal.

This matches the mask diagram in
[`../ch3_data_dependencies/prefill_access_patterns.md`](../ch3_data_dependencies/prefill_access_patterns.md).

#### Band-Mask Construction: TTNN Arange Route

For large T values it may be preferable to construct the mask on-device to avoid
transferring a `T × T` host tensor over PCIe. The construction uses
`ttnn.arange` to produce row and column index tensors, then elementwise
comparisons:

```python
# Row indices: [T, 1] broadcast to [T, T]
row_idx = ttnn.arange(0, T, 1, device=device).reshape(T, 1)   # [T, 1]
# Col indices: [1, T] broadcast to [T, T]
col_idx = ttnn.arange(0, T, 1, device=device).reshape(1, T)   # [1, T]

# Causal: col <= row
causal_mask = ttnn.le(col_idx, row_idx)                        # [T, T], bool

# Window lower bound: col >= row - (w - 1)
lower_mask = ttnn.ge(col_idx, row_idx - (w - 1))               # [T, T], bool

# Combined: allowed = causal AND lower_bound
allowed = ttnn.logical_and(causal_mask, lower_mask)            # [T, T], bool

# Convert to additive bias: 0.0 where True, -inf where False
NEG_INF = -1e9   # or float('-inf') depending on dtype precision
band_mask_tt = ttnn.where(allowed, 0.0, NEG_INF)               # [T, T], bfloat16
band_mask_tt = ttnn.reshape(band_mask_tt, [1, 1, T, T])        # [1, 1, T, T]
```

This avoids host-device transfer for the mask at the cost of three on-device
elementwise ops. For T = 4096 the mask is `4096 × 4096 × 2 = 32 MiB` — note
that materialising this full mask in DRAM is necessary regardless of which
construction route is used. For very large T (T ≥ 16384) the 32 MiB – 512 MiB
mask tensor becomes a non-trivial DRAM consumer and the chunked windowed approach
(see below) is preferable precisely because it avoids materialising the full
`[T, T]` mask.

### Step 3 — Scaled Dot-Product Attention

#### Option A — `ttnn.scaled_dot_product_attention` with Mask Argument

TTNN provides a Flash-Attention style fused SDPA op for prefill:

```python
output = ttnn.scaled_dot_product_attention(
    Q,               # [B, H_q,  T, d]
    K,               # [B, H_kv, T, d]
    V,               # [B, H_kv, T, d]
    attn_mask=band_mask_tt,   # [1, 1, T, T]
    scale=1.0 / (d ** 0.5),
    is_causal=False,          # mask already encodes causality
    program_config=sdpa_prefill_cfg,
)
# output: [B, H_q, T, d]
```

This op fuses QK^T, scale, mask add, softmax, and V-weighted sum into a single
tiled kernel. The `attn_mask` argument is an additive bias added to the raw
scores before softmax: positions set to `-inf` in `band_mask_tt` are effectively
excluded from the softmax.

**Does the mask argument cover the band/window constraint case?**

Yes — the mask-argument interface is general enough to express any per-position
bias including the band-diagonal mask. The op does not validate the mask
structure; it simply adds the provided tensor element-wise to the score matrix.
The band-diagonal mask is a valid input.

However, two important constraints apply:

1. **Shape restriction:** `ttnn.scaled_dot_product_attention` requires the Q, K,
   and V sequence lengths to be known at compile time and typically requires T to
   be a multiple of the tile size (32 for Wormhole's matmul tiles). If T is not
   a multiple of 32 it must be padded.

2. **Kernel tiling and masked-out work:** See `kernel_or_op_gap_analysis.md` for the full gap analysis.

**GQA support:** When `H_q > H_kv`, `ttnn.scaled_dot_product_attention` handles
GQA natively by broadcasting K and V heads across each query-head group,
provided the head counts satisfy `H_q % H_kv == 0`.

#### Option B — Chunked Windowed Kernel

A chunked windowed kernel processes query tiles of height `q_chunk` and, for
each query tile covering rows `[t0, t0 + q_chunk)`, loads only the key tiles
that intersect the valid band `[max(0, t0 - w + 1), t0 + q_chunk)`. Key tiles
entirely outside this range are skipped.

In TTNN this can be implemented by calling `ttnn.scaled_dot_product_attention`
repeatedly on query chunks, passing a per-chunk K/V slice and a per-chunk mask:

```python
outputs = []
for t0 in range(0, T, q_chunk):
    t1 = min(t0 + q_chunk, T)
    k_start = max(0, t0 - w + 1)
    k_end   = t1                    # k positions up to t1-1 are needed

    Q_chunk    = Q[:, :, t0:t1, :]          # [B, H_q,  q_chunk, d]
    K_chunk    = K[:, :, k_start:k_end, :]  # [B, H_kv, k_win,   d]
    V_chunk    = V[:, :, k_start:k_end, :]  # [B, H_kv, k_win,   d]
    mask_chunk = band_mask_tt[:, :, t0:t1, k_start:k_end]  # [1,1,q_chunk,k_win]

    out_chunk = ttnn.scaled_dot_product_attention(
        Q_chunk, K_chunk, V_chunk,
        attn_mask=mask_chunk,
        scale=1.0 / (d ** 0.5),
    )
    outputs.append(out_chunk)          # [B, H_q, q_chunk, d]

output = ttnn.concat(outputs, dim=2)   # [B, H_q, T, d]
```

The window size of the K/V chunk is `k_win = t1 - k_start ≤ q_chunk + w - 1`.
In steady state (t0 ≥ w) this is `q_chunk + w - 1 ≈ w` for small `q_chunk`.
The total K/V data loaded is therefore `T × w × d × 2` bytes, matching the
ideal tiled streaming cost from the arithmetic intensity analysis in Chapter 3.

**Trade-offs: Full-Size Masked Kernel vs Chunked Windowed Kernel:**

| Property                   | Full-size masked kernel          | Chunked windowed kernel               |
|----------------------------|----------------------------------|---------------------------------------|
| TTNN op used               | Single `ttnn.scaled_dot_product_attention` call | Multiple `ttnn.scaled_dot_product_attention` calls |
| Score matrix materialised  | Full `[T, T]` (in tiles)         | Per-chunk `[q_chunk, k_win]` at a time |
| K/V DRAM reads             | Full `[T, d]` per K/V head       | `~w × d` per query chunk (only band)  |
| Compute                    | O(T²) (tiles, masked but not skipped) | O(T × w) (only band tiles computed)  |
| Band mask tensor in DRAM   | Required: `[1, 1, T, T]`         | Per-chunk slice; no full mask needed  |
| Implementation complexity  | Low (one op call)                | Moderate (loop + dynamic shapes)      |
| Recommended for            | `w/T ≥ 0.5` or small T          | `w/T < 0.5` and large T              |

For a production model with T = 32768 and w = 4096 (`w/T ≈ 0.125`), the
full-size masked kernel performs roughly 8× more FLOPS and DRAM reads than
necessary. The chunked windowed kernel is the correct choice at this ratio.

## Prefill Side-Effect: Populating the Circular-Buffer KV Tensor

After prefill, the decode phase begins. The decode phase uses the circular-buffer
KV cache `[B, H_kv, w, d]`, not the full `[B, H_kv, T, d]` K/V tensors
computed during prefill. The transition requires writing the last `w` K and V
vectors from the prefill sequence into the circular buffer before the first
decode step.

### What Is Retained

Only the last `min(T, w)` key and value vectors from the prompt are needed for
decode. If `T ≤ w` all prompt K/V vectors are retained. If `T > w` only the
last `w` vectors (positions `T - w` through `T - 1`) are retained; earlier
prompt K/V vectors are outside the window at all subsequent decode positions and
will never be attended to again.

### Writing the Circular Buffer from Prefill Output

At the end of prefill:

```python
# K and V from prefill:  [B, H_kv, T, d]
# Number of tokens to keep: n_keep = min(T, w)
n_keep = min(T, w)
k_tail = K[:, :, T - n_keep : T, :]   # [B, H_kv, n_keep, d]
v_tail = V[:, :, T - n_keep : T, :]   # [B, H_kv, n_keep, d]

# Write into circular buffer starting at slot (T - n_keep) mod w
for i in range(n_keep):
    slot = (T - n_keep + i) % w
    ttnn.update_cache(k_cache, k_tail[:, :, i:i+1, :], slot)
    ttnn.update_cache(v_cache, v_tail[:, :, i:i+1, :], slot)

# Set pos_offset to reflect that oldest retained position is T - n_keep
pos_offset = T - n_keep   # = T - w if T >= w, else 0
```

If `T ≤ w` (the entire prompt fits within the window) all `T` tokens are
written starting at slot 0. The write pointer after prefill is `T mod w = T`
(since T < w there is no wrap). The buffer has `T` valid entries in slots 0
through `T - 1`.

If `T > w` (the prompt exceeds the window) the last `w` tokens are written into
slots `(T - w) mod w` through `(T - 1) mod w`. Depending on whether `T` is a
multiple of `w`, these writes may wrap around within the buffer.

In practice, rather than iterating slot by slot, the prefill-to-decode handoff
is implemented as a bulk copy of the `[B, H_kv, n_keep, d]` tail into the
appropriate slice of the `[B, H_kv, w, d]` cache buffer using a single
`ttnn.matmul`-free copy or a dedicated `ttnn.update_cache` call that accepts a
slice argument.

### Shape Diagram: Prefill Handoff (T = 12, w = 8)

```text
Prefill K tensor (full):              Circular-buffer KV cache after handoff:

  [B, H_kv, 12, d]                     [B, H_kv, 8, d]

  position index:                       slot index:
    0  1  2  3  4  5  6  7  8  9 10 11    0   1   2   3   4   5   6   7
  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐ ┌───┬───┬───┬───┬───┬───┬───┬───┐
  │t0│t1│t2│t3│t4│t5│t6│t7│t8│t9│tA│tB│ │ t8│ t9│ tA│ tB│ t4│ t5│ t6│ t7│
  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘ └───┴───┴───┴───┴───┴───┴───┴───┘
   ←── discarded (outside window) ───→
                        ←──── retained (last w=8 positions) ────────────→

  pos_offset = 12 - 8 = 4 (position t4 = oldest retained)
  wp (next write slot) = 12 % 8 = 4  (slot 4 will receive position t12 next)
```

The slot assignment for the retained tokens follows the circular formula:
`slot(T - w + i) = (T - w + i) mod w`. For T = 12, w = 8:

```text
position t4  → slot 4 % 8 = 4
position t5  → slot 5 % 8 = 5
position t6  → slot 6 % 8 = 6
position t7  → slot 7 % 8 = 7
position t8  → slot 8 % 8 = 0
position t9  → slot 9 % 8 = 1
position t10 → slot 10 % 8 = 2
position t11 → slot 11 % 8 = 3
```

Hence the circular buffer layout after handoff: slots [0,1,2,3] hold
[t8, t9, t10, t11] and slots [4,5,6,7] hold [t4, t5, t6, t7] — matching the
diagram above.

## Prefill Op Summary

```text
Input: prompt embedding [B, T, model_dim]

  ┌──────────────────────────────────────────────────────────────────┐
  │  ttnn.matmul × 3   — project to Q, K, V                         │
  │  shapes: Q [B, H_q, T, d]  K [B, H_kv, T, d]  V [B, H_kv, T, d] │
  └──────────────────────────┬───────────────────────────────────────┘
                             │
  ┌──────────────────────────▼───────────────────────────────────────┐
  │  RoPE rotations — positions 0..T-1                               │
  └──────────────────────────┬───────────────────────────────────────┘
                             │
  ┌──────────────────────────▼───────────────────────────────────────┐
  │  Band mask construction                                           │
  │  via torch.ones(T,T).tril().triu(-(w-1))  or  ttnn.arange route │
  │  mask shape: [1, 1, T, T], dtype bfloat16, DRAM                  │
  └──────────────────────────┬───────────────────────────────────────┘
                             │
  ┌──────────────────────────▼───────────────────────────────────────┐
  │  ttnn.scaled_dot_product_attention(Q, K, V, attn_mask=band_mask) │
  │  OR chunked windowed loop if w/T < 0.5                           │
  │  output: [B, H_q, T, d]                                          │
  └──────────────────────────┬───────────────────────────────────────┘
                             │
  ┌──────────────────────────▼───────────────────────────────────────┐
  │  ttnn.matmul — output projection W_o                             │
  │  output: [B, T, model_dim]                                       │
  └──────────────────────────┬───────────────────────────────────────┘
                             │
  ┌──────────────────────────▼───────────────────────────────────────┐
  │  KV cache population (prefill handoff)                           │
  │  ttnn.update_cache × min(T,w) — writes last w K/V into [B,H_kv,w,d] │
  │  pos_offset ← max(0, T - w)                                      │
  └──────────────────────────────────────────────────────────────────┘
```

---

**Next:** [`kernel_or_op_gap_analysis.md`](./kernel_or_op_gap_analysis.md)
