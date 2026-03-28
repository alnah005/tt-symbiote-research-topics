# Existing Kernel Survey for Windowed Attention

This document surveys the four TTNN and tt-transformers operations that are
relevant to windowed attention, evaluates each against the windowed attention
requirements identified across this guide, consolidates the gaps into a summary
table, and provides a recommended implementation path for each gap.

This chapter is a high-level synthesis. For the detailed interface specifications
of each op, the per-op gap tables, and the full reasoning behind each gap
classification, see
[`../ch4_ttnn_primitives/kernel_or_op_gap_analysis.md`](../ch4_ttnn_primitives/kernel_or_op_gap_analysis.md).

## Ops Under Survey

The survey covers four op families:

1. `ttnn.scaled_dot_product_attention` — Flash-Attention style prefill SDPA
2. `ttnn.scaled_dot_product_attention_decode` — decode-oriented single-query SDPA
3. `paged_sdpa` — paged-KV prefill SDPA wrapper in tt-transformers
4. `paged_sdpa_decode` — paged-KV decode SDPA wrapper in tt-transformers

## Evaluation Dimensions

Each op is assessed on five dimensions relevant to windowed attention:

- **Mask support:** does the op accept an additive attention mask or equivalent
  mechanism that can encode the band/window constraint?
- **Window expressibility:** can that mechanism express the full band-diagonal
  windowed constraint, including the lower bound `s >= t - w + 1`?
- **GQA support:** is grouped query attention (`H_q > H_kv`) handled natively?
- **Circular-buffer compatibility:** can the op accept the fixed-shape
  `[B, H_kv, w, d]` circular buffer directly without the caller reordering
  slots into temporal order?
- **Shape restrictions:** what constraints on T, w, d, B, H must be satisfied?

## Op 1 — `ttnn.scaled_dot_product_attention`

This is the primary prefill op implementing Flash-Attention style tiled QK^T computation. The key caveat for windowed attention is that `is_causal=True` does not encode the window lower bound; a full band-diagonal `attn_mask` of shape `[1, 1, T, T]` with 0 in the band `[t - w + 1, t]` and `-inf` outside must be passed explicitly. The kernel applies the mask correctly but does not skip compute on fully-masked tiles, so for `w << T` approximately O(T² − T·w) FLOPs are wasted — this is the G1 performance gap.

## Op 2 — `ttnn.scaled_dot_product_attention_decode`

This is the primary decode op, processing S_q = 1 against an arbitrary-length KV cache. The `valid_seq_len` parameter handles fill-phase masking correctly by implicitly masking positions beyond the specified valid count, and in steady state (T ≥ w) the circular buffer is structurally bounded so no mask is needed. There is no `window_size` parameter; if the caller passes a KV buffer larger than w, an explicit mask or pre-sliced buffer is required.

## Op 3 — `paged_sdpa`

`paged_sdpa` wraps `ttnn.scaled_dot_product_attention` with a page-table indirection layer over a physical KV block store. The `attn_mask` is forwarded to the underlying kernel, but a full `[T, T]` tensor is required even when most entries are masked; Strategy B (circular-buffer-as-pages from [`../ch5_paged_kv_cache/paged_sdpa_and_windowing.md`](../ch5_paged_kv_cache/paged_sdpa_and_windowing.md)) is recommended for circular-buffer paging because neither window enforcement mechanism is built into the op itself.

## Op 4 — `paged_sdpa_decode`

`paged_sdpa_decode` is the decode-phase counterpart to `paged_sdpa`, delegating to `ttnn.scaled_dot_product_attention_decode` with S_q = 1. Page-table management is entirely the caller's responsibility for window enforcement: the page table must list exactly `ceil(w / block_size)` pages (the w most recent tokens, i.e., Strategy B pages) and must be updated O(1) per step as the oldest block falls out of the window.

## Gap Summary Table

The table below consolidates the capability assessment across all four ops and
five dimensions. Cells are marked as follows:

- **Yes** — fully supported; no caller workaround required
- **Partial** — supported with caveats (noted below)
- **Via caller** — achievable but requires explicit caller-side convention
- **No** — not supported; a gap exists

| Dimension                      | `ttnn.sdpa` (prefill) | `ttnn.sdpa_decode` | `paged_sdpa` | `paged_sdpa_decode` |
|--------------------------------|-----------------------|--------------------|--------------|---------------------|
| Mask support                   | Yes                   | Yes                | Yes          | Yes                 |
| Window expressibility          | Via caller            | Via caller         | Via caller   | Via caller          |
| GQA support                    | Yes                   | Yes                | Yes (inh.)   | Yes (inh.)          |
| Circular-buffer compatibility  | Partial               | Yes                | No           | No                  |
| Shape restrictions satisfied   | Partial               | Partial            | Partial      | Partial             |

```text
Legend:
  sdpa         = ttnn.scaled_dot_product_attention
  sdpa_decode  = ttnn.scaled_dot_product_attention_decode
  paged_sdpa   = paged_sdpa in tt-transformers
  inh.         = inherited from underlying TTNN op

Partial for circular-buffer compatibility (ttnn.sdpa):
  Accepts [B, H_kv, w, d] for decode path; not applicable for prefill path.

Partial for shape restrictions:
  All ops require tile-aligned dimensions (multiples of 32 for S_kv, block_size).
  Decode ops additionally require d to be a power of 2 ≤ 128.

Via caller (window expressibility across all ops):
  No op has a native window_size parameter.
  Window constraint is enforced through attn_mask, valid_seq_len,
  page-table management, or fixed-shape buffer — all caller-side conventions.
```

### Identified Gaps

The full G1–G7 gap inventory with per-op attribution and interface details is in [`kernel_or_op_gap_analysis.md`](../ch4_ttnn_primitives/kernel_or_op_gap_analysis.md). No gap blocks a correct initial implementation; G1 (O(T²) wasted prefill FLOPs) and G4 (chunked prefill shape restriction) are the only high-severity performance gaps.

## Recommended Implementation Path

### Correct Initial Implementation — No New Kernels Required

The gap severity column shows that no gap blocks a correct initial
implementation. The following caller-side conventions resolve all blocking
issues:

**Prefill (non-paged path):**

1. Construct a band-diagonal `attn_mask` of shape `[1, 1, T, T]` with 0 in
   band `[t - w + 1, t]` and `-inf` outside. Pass to `ttnn.scaled_dot_product_attention`.
2. Accept the O(T²) compute cost; for T ≤ 8192 this is tolerable.
3. Write the last `w` rows of the computed K and V tensors into the circular
   buffer as a side-effect of prefill (see
   [`../ch4_ttnn_primitives/prefill_primitives.md`](../ch4_ttnn_primitives/prefill_primitives.md)).

**Decode (non-paged path):**

1. Maintain the circular buffer `[B, H_kv, w, d]` with `ttnn.update_cache`
   at slot `T % w` before each attention call.
2. In fill phase (T < w): pass `attn_mask` with `-inf` at slots `[T+1, w-1]`,
   or use `valid_seq_len = T + 1`.
3. In steady state (T >= w): pass the full buffer with no mask.
4. Apply RoPE at write time so that slot ordering does not affect dot-product
   correctness.

**Decode (paged path):**

1. Implement the circular-buffer-as-pages strategy from Chapter 5: allocate
   `ceil(w / block_size)` physical pages per sequence, reuse them round-robin.
2. Update the page table entry for the evicted block on each step where
   `T % block_size == 0` (O(1) per step).
3. Pass `valid_seq_len = min(T+1, w)` to `paged_sdpa_decode`.

These conventions require no TTNN op modifications and no new kernel code.
The implementation lives entirely in the decode loop and model adapter layer
in tt-transformers Python code.

### Performance-Optimised Implementation — One New Kernel Needed

The only gap that materially limits performance at scale is **G1** (wasted
O(T²) − O(T·w) FLOPs during prefill for large T and small w). Closing G1
requires a band-mask-aware Flash Attention kernel for prefill that skips
fully-masked tiles.

Two approaches exist for closing G1:

**Approach A — External chunked loop (no kernel change):**

Expose windowed prefill as a Python-level loop over
`ttnn.scaled_dot_product_attention` calls, each operating on a query chunk of
size `q_chunk` and a KV window of size `q_chunk + w - 1`. Because both shapes
are determined by `q_chunk` and `w` (model config constants), TTNN does not
recompile across iterations. The loop structure is:

```python
for t0 in range(0, T, q_chunk):
    k_start  = max(0, t0 - w + 1)
    k_end    = min(t0 + q_chunk, T)
    kv_len   = k_end - k_start
    q_len    = min(q_chunk, T - t0)

    # Q_chunk: [B, H_q, q_len, d]
    # K_chunk: [B, H_kv, kv_len, d]  (≤ q_chunk + w - 1)

    # Build an explicit attention mask in absolute token-position space.
    # is_causal=True must NOT be used here: it applies a lower-triangular
    # mask in local chunk index space, which is incorrect whenever
    # k_start < t0 (i.e., every chunk after the first once T ≥ w).
    # In that case, valid KV tokens with local index j > local Q index i
    # but absolute KV position k_start+j ≤ absolute Q position t0+i
    # would be incorrectly masked out, silently corrupting attention weights.
    #
    # Correct mask: entry (i, j) is valid (0.0) when
    #   k_start + j <= t0 + i   (causal: KV not in the future of Q)
    #   k_start + j >= t0 + i - w + 1  (window: KV not too far in the past)
    # and -inf otherwise.
    q_idx  = torch.arange(t0, t0 + q_len).unsqueeze(1)   # [q_len, 1]
    kv_idx = torch.arange(k_start, k_end).unsqueeze(0)    # [1, kv_len]
    causal_ok = kv_idx <= q_idx                            # causal constraint
    window_ok = kv_idx >= q_idx - w + 1                   # window constraint
    mask = torch.where(causal_ok & window_ok,
                       torch.zeros(q_len, kv_len),
                       torch.full((q_len, kv_len), float("-inf")))
    # Expand to [B, 1, q_len, kv_len] for broadcast over batch and heads
    mask = mask.unsqueeze(0).unsqueeze(0)

    output_chunk = ttnn.scaled_dot_product_attention(
        Q_chunk, K_chunk, V_chunk, attn_mask=mask, ...
    )
```

This eliminates wasted FLOPS at the cost of `T / q_chunk` kernel launches per
layer, which is acceptable (for T = 32768, q_chunk = 256: 128 launches).

**Approach B — New program config with tile-skip logic:**

Add a `SDPAWindowedProgramConfig` to `ttnn.scaled_dot_product_attention` that
accepts a `window_size` parameter and instructs the kernel to skip tile
computations where the entire tile is outside the band. This requires kernel
surgery but achieves the ideal O(T·w) FLOPs in a single kernel launch.

Approach A is the recommended path for an initial performance-optimised
implementation because it requires no kernel changes and achieves the same FLOP
reduction. Approach B is the long-term target for production deployments with
large T.

### Op Extension vs New Program Config vs New Kernel

| Gap  | Recommended Approach                        | Scope                                | Kernel Change? |
|------|---------------------------------------------|--------------------------------------|----------------|
| G1   | External chunked loop (Approach A)          | Python decode-loop helper function   | No             |
| G2   | Caller convention (fixed buffer + mask)     | Model adapter layer                  | No             |
| G3   | RoPE-at-write-time convention               | Already standard in tt-transformers  | No             |
| G4   | Same as G1 (external loop IS chunked mode)  | Python decode-loop helper function   | No             |
| G5   | Per-step page table management layer        | Thin Python wrapper in decode loop   | No             |
| G6   | Pad `d` to next supported value             | Model adapter                        | No             |
| G7   | Use `ttnn.sdpa` for S_q > 1 cases           | Model adapter                        | No             |

## Overall Conclusion

The caller-side conventions described above (circular-buffer KV layout, band-diagonal mask, fill-phase validity mask, and page-table management for paged sequences) are sufficient to close all seven identified gaps.

A performance-optimised implementation adds one enhancement: a chunked windowed
prefill loop (or, as a longer-term item, a band-mask-aware Flash Attention kernel)
to reduce prefill FLOPs from O(T²) to O(T·w).

---

**End of guide.** Return to [Guide Index](../index.md)
