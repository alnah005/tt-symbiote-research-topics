# Kernel and Op Gap Analysis

This document surveys every TTNN and tt-transformers operation that is relevant
to implementing windowed attention, evaluates each against the windowed
attention requirements identified in this chapter, and produces a gap summary
with recommended paths forward. It directly addresses research question Q8:
where does the current TTNN/tt-transformers kernel surface fall short of full
windowed attention support?

## Scope of the Survey

The survey covers four families of ops:

1. `ttnn.scaled_dot_product_attention` — the standard Flash-Attention style
   prefill SDPA op
2. `ttnn.scaled_dot_product_attention_decode` — the decode-oriented single-query
   SDPA op with GQA support
3. `paged_sdpa` — paged variant of SDPA for prefill in tt-transformers
4. `paged_sdpa_decode` — paged variant of SDPA for decode in tt-transformers

For each op the analysis covers five dimensions:

- **Mask support:** does the op accept an additive attention mask or `is_causal`
  flag that can encode the band/window constraint?
- **Window expressibility:** can the mask (if accepted) express the full
  band-diagonal windowed constraint including the lower bound `s ≥ t - w + 1`?
- **GQA support:** is grouped query attention (H_q > H_kv) handled natively?
- **Circular-buffer input:** can the op accept the fixed-shape `[B, H_kv, w, d]`
  circular buffer directly without requiring the caller to re-order slots
  into temporal order?
- **Tensor shape restrictions:** what T, w, d, B, H constraints must be satisfied?

## Op 1 — `ttnn.scaled_dot_product_attention`

### Interface

```python
ttnn.scaled_dot_product_attention(
    query,                    # [B, H_q, S_q, d]
    key,                      # [B, H_kv, S_kv, d]
    value,                    # [B, H_kv, S_kv, d]
    attn_mask=None,           # [B or 1, H or 1, S_q, S_kv] additive bias; optional
    is_causal=False,          # if True, uses internal causal mask; overrides attn_mask
    scale=None,               # float or None (defaults to 1/sqrt(d))
    program_config=None,      # SDPAMultiCoreProgramConfig
    valid_seq_len=None,       # int or None; number of valid kv positions
)
# returns: [B, H_q, S_q, d]
```

### Mask Support

The `attn_mask` argument accepts an additive bias tensor that is added to the
raw QK^T scores before softmax. This is the standard Flash-Attention masking
interface. The op does not restrict the mask to causal or band-diagonal forms;
any `[B_or_1, H_or_1, S_q, S_kv]` tensor is accepted.

`is_causal=True` generates an internal causal (lower-triangular) mask and is
mutually exclusive with `attn_mask`. Setting `is_causal=True` does NOT encode
the windowed lower bound; it produces a standard full causal mask.

### Window Expressibility

**Yes, via `attn_mask`.** The band-diagonal mask constructed in
[`prefill_primitives.md`](./prefill_primitives.md) (a `[1, 1, T, T]` tensor
with 0 in the band and -inf outside) can be passed directly as `attn_mask`.
The op adds it correctly before softmax.

**Limitation:** The op does not skip compute on fully-masked tiles. The kernel
processes the full S_q × S_kv score matrix in tiles. For `w << T` this means
O(T²) compute is performed even though only O(T·w) scores are non-masked. The
band constraint is applied correctly at the numeric level but is not exploited
to reduce FLOP count.

### GQA Support

Yes. When `H_q > H_kv` the op broadcasts K and V heads across query-head groups.
The ratio `H_q / H_kv` must be an integer.

### Circular-Buffer Input

**Partially.** The op accepts arbitrary K and V shapes `[B, H_kv, S_kv, d]`.
Passing the circular buffer `[B, H_kv, w, d]` as K and V is valid for the
decode path (where S_kv = w). For the prefill path the K and V are the full
prompt K/V tensors `[B, H_kv, T, d]`, not the circular buffer.

The circular buffer is populated as a side-effect of prefill (see
[`prefill_primitives.md`](./prefill_primitives.md)) and becomes the direct input
only for decode. The op has no awareness of the wrap-around slot ordering within
the circular buffer; it treats the input as a plain `[S_kv, d]` sequence. If
the caller passes the buffer with wrap-boundary ordering (oldest at the start,
newest contiguous), the mask must account for the physical slot order rather
than the logical temporal order.

### Tensor Shape Restrictions

| Restriction                       | Detail                                                                 |
|-----------------------------------|------------------------------------------------------------------------|
| S_q and S_kv must be multiples of 32 | TTNN's tile size on Wormhole is 32×32; non-multiple lengths require padding |
| Maximum S_q, S_kv                 | Limited by on-chip tile SRAM; very large T requires further chunking   |
| `d` must be a supported head dim  | Common values 64, 128 are supported; verify for non-standard d         |
| `B × H_q` must fit core grid       | At most `grid_w × grid_h` cores available; overflow requires serialisation |

### Gap Summary for `ttnn.scaled_dot_product_attention`

| Requirement                        | Status    | Notes                                                      |
|------------------------------------|-----------|-------------------------------------------------------------|
| Band-diagonal mask (additive)      | Supported | Via `attn_mask`; O(T²) compute regardless of band width    |
| Skip fully-masked tiles (O(T·w))   | Gap       | Kernel does not skip zero-contribution tiles                |
| GQA (H_q > H_kv)                  | Supported | Native broadcast                                            |
| Circular-buffer slot reordering    | Gap       | Caller must linearise the buffer into temporal order        |
| Chunked windowed kernel mode       | Gap       | No built-in chunked mode; caller must loop externally       |

---

## Op 2 — `ttnn.scaled_dot_product_attention_decode`

### Interface

```python
ttnn.scaled_dot_product_attention_decode(
    input_tensor,             # [B, H_q, 1, d]  (the query)
    key,                      # [B, H_kv, S_kv, d]
    value,                    # [B, H_kv, S_kv, d]
    attn_mask=None,           # [B or 1, 1, 1, S_kv] additive bias; optional
    scale=None,
    program_config=None,      # SDPADecodeProgramConfig
    valid_seq_len=None,       # int scalar or [B] tensor; valid kv positions per seq
    compute_kernel_config=None,
)
# returns: [B, H_q, 1, d]
```

### Mask Support

`attn_mask` is supported and has the expected decode shape `[B_or_1, 1, 1,
S_kv]` — one mask value per KV position, applied uniformly across all query
heads. This is sufficient to express the fill-phase validity mask (setting
uninitialised slots to -inf) but does NOT express a per-head or per-query-position
varying mask.

`valid_seq_len` provides an alternative mechanism: it tells the kernel how many
KV positions are valid (the rest are implicitly masked). This is equivalent to
the fill-phase position mask for the case where valid positions occupy a
contiguous prefix `[0, valid_seq_len)` of the KV buffer. For steady-state
decode (all `w` slots populated) `valid_seq_len = w` and no masking is needed.

### Window Expressibility

**Yes for fill phase, implicit for steady state.** The fill-phase constraint
(exclude slots beyond `T`) is expressible via `attn_mask` or `valid_seq_len`.

The steady-state window constraint requires no mask because the circular buffer
holds exactly `w` entries — it is structurally bounded to the window. The op
with `S_kv = w` and no masking is the correct steady-state decode implementation.

**Gap:** The op has no `window_size` parameter. If the caller were to pass a
larger KV buffer (e.g., for paged attention where the buffer is allocated to
`max_T` but only `w` recent entries should be attended) the op cannot internally
enforce the window boundary. The caller must pre-slice the buffer to `w` entries
or pass an explicit mask.

### GQA Support

Yes. This is the primary production op for GQA decode; H_q, H_kv, and group
size `H_q / H_kv` must be integers. The internal kernel tiles across
`(B, H_kv)` pairs and handles the H_q expansion per KV head.

### Circular-Buffer Input

**Yes, with slot-order caveat.** The op accepts `[B, H_kv, w, d]` directly.
In steady state the op reads all `w` KV slots, which is exactly the full
circular buffer content. The softmax is computed over all `w` scores, weighting
each slot by its dot-product score with Q. The temporal ordering of slots within
the buffer is irrelevant to the softmax — the op does not assume slots are in
temporal order.

However, RoPE requires that the positional encodings applied to K vectors match
the absolute token positions of the entries in each slot. If the circular buffer
stores K vectors that already have RoPE applied at write time, then the slot
order within the buffer is irrelevant to the dot-product computation; the
rotation is baked into the K values. This is the standard approach in TTNN-based
transformers and means the circular-buffer layout is fully compatible with this
op.

### Tensor Shape Restrictions

| Restriction                          | Detail                                                               |
|--------------------------------------|----------------------------------------------------------------------|
| S_q = 1 required                     | Op is decode-only; prefill (S_q > 1) is not supported               |
| S_kv (= w) should be multiple of 32  | Tile alignment; non-multiple S_kv requires padding                   |
| `d` must be a power of 2 ≤ 128       | Hardware limitation on matmul tile inner dimension                   |
| `B × H_q` must map to core grid       | Same constraint as the prefill op                                    |

### Gap Summary for `ttnn.scaled_dot_product_attention_decode`

| Requirement                            | Status    | Notes                                                        |
|----------------------------------------|-----------|--------------------------------------------------------------|
| Fill-phase validity mask               | Supported | Via `attn_mask` or `valid_seq_len`                           |
| Steady-state windowed decode           | Supported | Pass `[B, H_kv, w, d]` cache directly; no mask needed       |
| GQA (H_q > H_kv)                      | Supported | Native; primary use case                                     |
| Window boundary in paged/grow-in-place | Gap       | No built-in windowing for over-allocated buffers; requires caller slicing or explicit mask |
| Chunked/blocked K streaming            | Supported | Internally tiles K in `k_chunk_size` blocks                  |
| Multi-position decode (S_q > 1)        | Gap       | Not supported; would require the prefill op                  |

---

## Op 3 — `paged_sdpa` (tt-transformers)

### Interface

`paged_sdpa` is a tt-transformers abstraction (not a raw TTNN op) that wraps
`ttnn.scaled_dot_product_attention` with a page-table indirection layer. It
accepts a physical KV block tensor `[total_blocks, H_kv, block_size, d]` and a
page table `[B, max_seq_len / block_size]` that maps logical KV positions to
physical blocks.

```python
output = paged_sdpa(
    query,          # [B, H_q, T, d]
    key_block_store,   # [total_blocks, H_kv, block_size, d]
    value_block_store, # [total_blocks, H_kv, block_size, d]
    page_table,     # [B, num_pages_per_seq]
    attn_mask=None,    # [1, 1, T, S_kv] additive bias; optional
    scale=None,
    program_config=None,
)
# returns: [B, H_q, T, d]
```

### Mask Support

`paged_sdpa` delegates mask application to the underlying
`ttnn.scaled_dot_product_attention` call. An `attn_mask` argument is forwarded
with shape `[1, 1, S_q, S_kv]`. The same band-diagonal mask used in the
non-paged prefill path can be constructed and passed; the page-table layer does
not interfere with mask application.

### Window Expressibility

**Via `attn_mask` only.** The page-table layer itself has no concept of a window
constraint. The page table maps every logical position to a physical block; the
op fetches all pages listed in the page table. To enforce a window constraint:

1. The page table must only list pages containing positions within the window
   `[t - w + 1, t]` (page-aware windowing strategy from Chapter 5).
2. OR: an additive band mask covering the full logical KV sequence must be passed.

Neither mechanism is built into `paged_sdpa`; both require external coordination
by the caller.

### GQA Support — Inherited from `ttnn.scaled_dot_product_attention`

### Circular-Buffer Input

**Incompatible by design.** The circular buffer is a fixed-shape DRAM tensor
with implicit wrap-around slot ordering. The paged system uses an explicit page
table to map logical positions to physical blocks. These two memory management
strategies are mutually exclusive for a given sequence. To use `paged_sdpa` with
windowed attention the paged layer must implement the circular-buffer-as-pages
strategy (allocating `ceil(w / block_size)` physical pages per sequence and
reusing them in round-robin order), as discussed in Chapter 5.

### Tensor Shape Restrictions

Inherits all restrictions from `ttnn.scaled_dot_product_attention`, plus:

| Restriction                         | Detail                                                                |
|-------------------------------------|-----------------------------------------------------------------------|
| S_kv must equal num_pages × block_size | Page table covers the full logical sequence; partial-page sequences require masking |
| block_size must be a multiple of 32  | Tile alignment for the underlying matmul                              |
| page_table dtype                    | Must be uint32 (physical block indices)                               |

### Gap Summary for `paged_sdpa`

| Requirement                            | Status    | Notes                                                          |
|----------------------------------------|-----------|----------------------------------------------------------------|
| Band-diagonal mask (additive)          | Supported | Forwarded to underlying `ttnn.scaled_dot_product_attention`    |
| Skip fully-masked tiles                | Gap       | Inherited gap from `ttnn.scaled_dot_product_attention`         |
| Window constraint in page table        | Gap       | Page table must encode recency; not built in                   |
| Circular-buffer compatibility          | Gap       | Requires circular-buffer-as-pages strategy; not natively supported |
| GQA                                    | Supported | Inherited                                                      |

---

## Op 4 — `paged_sdpa_decode` (tt-transformers)

### Interface

`paged_sdpa_decode` is the decode-phase counterpart to `paged_sdpa`. It accepts
a single query vector and fetches the KV blocks indicated by the page table for
the current sequence, delegating to `ttnn.scaled_dot_product_attention_decode`.

```python
output = paged_sdpa_decode(
    query,             # [B, H_q, 1, d]
    key_block_store,   # [total_blocks, H_kv, block_size, d]
    value_block_store, # [total_blocks, H_kv, block_size, d]
    page_table,        # [B, num_pages_per_seq]
    attn_mask=None,    # [B or 1, 1, 1, S_kv] optional
    scale=None,
    valid_seq_len=None,    # int or [B]; valid KV positions
    program_config=None,
)
# returns: [B, H_q, 1, d]
```

### Mask Support

`attn_mask` is forwarded to `ttnn.scaled_dot_product_attention_decode` with the
decode mask shape `[B_or_1, 1, 1, S_kv]`. The fill-phase validity mask is
expressible. `valid_seq_len` is also forwarded and provides a scalar upper bound
on valid KV positions.

### Window Expressibility

**Via `valid_seq_len` or `attn_mask`, not natively.** Like `paged_sdpa`,
`paged_sdpa_decode` has no built-in window-size parameter. For windowed decode:

- If the page table lists exactly `ceil(w / block_size)` pages (one per active
  window block), `valid_seq_len = w` (or `min(T+1, w)` during fill) correctly
  limits the attention to the window. This requires the page table to be kept
  up to date as tokens are generated, adding `ceil(w / block_size)` page-table
  entries per active sequence rather than `ceil(T / block_size)`.
- Alternatively, an explicit mask can suppress out-of-window blocks, but this
  requires the mask to track which physical pages contain out-of-window tokens —
  complex for an arbitrary page allocation.

### GQA Support — Inherited from `ttnn.scaled_dot_product_attention_decode`

### Circular-Buffer Input

**Not compatible without adaptation.** The same incompatibility as `paged_sdpa`:
the circular-buffer model and the paged model have separate bookkeeping systems.
The circular-buffer-as-pages strategy (Chapter 5) is required to bridge them.

### Tensor Shape Restrictions

Inherits restrictions from `ttnn.scaled_dot_product_attention_decode`, plus
page-table alignment constraints.

### Gap Summary for `paged_sdpa_decode`

| Requirement                            | Status    | Notes                                                          |
|----------------------------------------|-----------|----------------------------------------------------------------|
| Fill-phase validity mask               | Supported | Via `valid_seq_len` or `attn_mask`                             |
| Steady-state windowed decode           | Supported (conditional) | Requires page table limited to w-token window      |
| Window boundary enforcement            | Gap       | No `window_size` parameter; caller must maintain page table    |
| Circular-buffer compatibility          | Gap       | Requires circular-buffer-as-pages strategy                     |
| GQA                                    | Supported | Inherited                                                      |
| Out-of-window page eviction            | Gap       | No built-in eviction; caller must update page table per step   |

---

## Consolidated Gap Table

The table below lists every gap identified across all four ops and rates its
severity for a production windowed attention implementation.

| Gap                                             | Ops Affected                                          | Severity  | Description                                                        |
|-------------------------------------------------|-------------------------------------------------------|-----------|--------------------------------------------------------------------|
| G1: O(T²) compute for masked prefill            | `ttnn.scaled_dot_product_attention`, `paged_sdpa`     | High      | Kernel does not skip fully-masked tiles; O(T·w) ideal not achieved |
| G2: No native `window_size` parameter for decode | `ttnn.scaled_dot_product_attention_decode`, `paged_sdpa_decode` | Medium | Caller must pre-slice buffer or supply explicit mask per step |
| G3: No circular-buffer slot reordering          | All four ops                                          | Low–Medium | Temporal order mismatch between circular-buffer layout and logical KV order; mitigated if RoPE is applied at write time |
| G4: No chunked windowed kernel mode for prefill | `ttnn.scaled_dot_product_attention`                   | High      | Caller must loop externally; each loop iteration recompiles if shapes change |
| G5: No page-table windowing (paged ops only)    | `paged_sdpa`, `paged_sdpa_decode`                     | High      | Window enforcement via paging requires per-step page table management not currently automated |
| G6: `d` restricted to powers of 2 ≤ 128 (decode) | `ttnn.scaled_dot_product_attention_decode`            | Low       | Non-standard head dims (e.g., 96, 256) require padding or op extension |
| G7: S_q > 1 not supported in decode op          | `ttnn.scaled_dot_product_attention_decode`            | Low       | Speculative decode (multiple draft tokens) cannot reuse the decode op |

---

## Recommended Paths Forward

### Closing G1: O(T²) Prefill Compute

**Recommended path: implement a chunked windowed kernel as an external loop with
static chunk sizes.**

Rather than modifying the Flash-Attention kernel to skip tiles internally (which
requires kernel surgery), expose windowed prefill as a fixed-chunk-size loop
over `ttnn.scaled_dot_product_attention` calls where each call operates on a
query chunk of size `q_chunk` and a K/V chunk of size `q_chunk + w - 1`. Both
shapes are determined by `q_chunk` and `w`, which are static at model config
time. TTNN does not recompile when the same shapes repeat; the per-call overhead
is amortised across `T / q_chunk` iterations.

The loop body is:

```python
# q_chunk and w are compile-time constants for a given model config
for t0 in range(0, T, q_chunk):
    k_start = max(0, t0 - w + 1)
    k_end   = min(t0 + q_chunk, T)
    # Shapes: Q_chunk=[B,H_q,q_chunk,d], K_chunk=[B,H_kv,k_win,d]
    # k_win = k_end - k_start <= q_chunk + w - 1  (constant in steady state)
    ...
```

This eliminates G1 at the cost of O(T / q_chunk) kernel launches per layer per
forward pass, which is acceptable for typical values (T = 32768, q_chunk = 256
→ 128 launches).

### Closing G2: Window Enforcement in Decode

**Recommended path: use Strategy 2 (fixed-shape buffer + pre-computed fill-phase
mask) from [`decode_primitives.md`](./decode_primitives.md).**

Pass the full `[B, H_kv, w, d]` cache to `ttnn.scaled_dot_product_attention_decode`
at every step with a constant-shape `attn_mask` tensor. During the fill phase
(T < w) the mask has `-inf` in slots `[T+1, w-1]`; in steady state the mask is
all-zeros. A zero mask can be cached as a pre-allocated TTNN tensor and reused.
No `window_size` parameter is needed in the op itself.

This closes G2 without any op modification. The only overhead is a mask tensor
of `B × 1 × 1 × w × 2` bytes (for B=8, w=4096 this is 64 KiB — negligible).

### Closing G3: Circular-Buffer Slot Reordering

**Recommended path: apply RoPE at write time (current standard practice) and
accept physical slot order.**

If K vectors are stored in the circular buffer with RoPE already applied at
the absolute positions of their tokens, the dot product `Q · K^T` is correct
regardless of the physical slot ordering within the buffer. The softmax is
over raw dot products; slot order does not affect the result. RoPE-at-write-time
is the standard approach in tt-transformers and eliminates the slot-reordering
gap entirely with no op changes.

G3 becomes a non-issue given this practice. If RoPE-at-decode-time (applying
rotation during the attention call based on runtime position) is required, a
gather-reorder step must precede the SDPA call; this is an O(w) copy, not a
kernel change.

### Closing G4: Chunked Windowed Prefill Mode

Addressed jointly with G1 (see above). The external loop IS the chunked windowed
kernel mode. A helper function `windowed_sdpa_prefill(Q, K, V, w, q_chunk)`
wraps the loop and can be contributed to the tt-transformers model utilities
without requiring any TTNN kernel changes.

### Closing G5: Page-Table Windowing

**Recommended path: implement the circular-buffer-as-pages strategy (Chapter 5)
with a thin page-table management layer in the tt-transformers decode loop.**

Allocate `ceil(w / block_size)` physical pages per sequence slot. On each decode
step, the page-table entry for the evicted position is overwritten with the
physical block containing the new token. This is O(1) per step and requires only
a small integer update to the page table (one entry per `block_size` new tokens).
Chapter 5 analyses this strategy in depth.

### Op Extension vs New Kernel vs New Program Config

| Gap  | Approach                 | Justification                                                   |
|------|--------------------------|------------------------------------------------------------------|
| G1   | External loop (Python)   | No kernel change; static shapes avoid recompile                 |
| G2   | Caller convention        | Fixed-shape buffer + pre-allocated mask; no TTNN change needed  |
| G3   | RoPE-at-write convention | Standard practice already in tt-transformers                    |
| G4   | External loop (Python)   | Same as G1                                                      |
| G5   | Page-table management layer | New code in decode loop; no kernel change                    |
| G6   | Padding in model adapter | Pad `d` to next power of 2; minimal perf impact for small delta |
| G7   | Use prefill op for S_q>1 | `ttnn.scaled_dot_product_attention` supports S_q > 1            |

---

**Next:** [Chapter 5 — Paged KV Cache Interaction](../ch5_paged_kv_cache/index.md)
