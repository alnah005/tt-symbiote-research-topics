# Paged SDPA Sliding Window Investigation

This file investigates whether `ttnn.transformer.scaled_dot_product_attention_decode`
(hereafter `paged_sdpa_decode`) natively supports a `sliding_window_size`
parameter, how it interacts with the paged KV cache, and what fallback
strategies are available if the support is incomplete.

## Does `paged_sdpa_decode` Accept `sliding_window_size`?

**Yes.** The function signature for
`ttnn.transformer.scaled_dot_product_attention_decode` includes a
`sliding_window_size` keyword parameter:

```python
ttnn.transformer.scaled_dot_product_attention_decode(
    input_tensor_q,
    input_tensor_k,
    input_tensor_v,
    *,
    ...
    sliding_window_size=None,    # <-- native sliding window support
    page_table_tensor=None,
    ...
)
```

When `sliding_window_size` is set to a positive integer (e.g., 1024), the
kernel restricts each query to attend only to the most recent
`sliding_window_size` KV positions. When `None` (the default), full causal
attention is used with no window restriction.

This parameter is documented in the
[paged SDPA decode function signature guide](../../paged_sdpa_decode_for_gqa/ch2_ttnn_api/function_signature.md),
which describes it as: "When set, restricts each query to attend only to the
most recent `sliding_window_size` KV positions."

## How `sliding_window_size` Interacts With Paged KV Cache

### Window Masking Within the Kernel

When both `page_table_tensor` and `sliding_window_size` are provided, the
kernel applies the window constraint as an attention mask over the paged KV
cache. The mechanism is:

1. The kernel reads `cur_pos_tensor[b]` to determine the current position $T$
   for each batch element $b$.
2. The valid window range for batch element $b$ is
   $[\max(0, T - W + 1), T]$ where $W$ is `sliding_window_size`.
3. For each KV position in the page table, the kernel computes the absolute
   token position. If the position falls outside the window range, its
   attention logit is set to $-\infty$ before softmax.
4. Pages that contain **exclusively** out-of-window tokens may be skipped
   entirely by the kernel's gather logic, saving DRAM bandwidth.

### Page Loading Optimization

The degree to which the kernel optimizes page loading for windowed attention
depends on the implementation:

**Optimistic case:** The kernel computes the first and last virtual pages
that overlap with the window and only gathers those pages from the block
pool. For a 1024-token window with `block_size=64`, this means loading at
most $\lceil 1024/64 \rceil + 1 = 17$ pages (the +1 accounts for
block-boundary misalignment), regardless of total sequence length. This
provides $O(W)$ DRAM bandwidth cost per decode step, independent of $T$.

**Conservative case:** The kernel gathers all pages from 0 to
$\lfloor T / \text{block\_size} \rfloor$ and applies the window mask
post-gather. In this case, DRAM bandwidth scales with $T$ even though only
$W$ tokens contribute to the attention output. The extra bandwidth is wasted
on loading pages whose contributions are zeroed by the mask.

The exact behavior depends on the tt-metal kernel implementation version.
For Gemma 4 31B with 50 sliding layers and a 1024-token window, the
distinction matters significantly at long sequence lengths:

| Sequence Length | Pages Loaded (Optimistic) | Pages Loaded (Conservative) | Ratio |
|----------------|--------------------------|----------------------------|-------|
| 1K | 16 | 16 | 1.0x |
| 8K | 17 | 128 | 7.5x |
| 32K | 17 | 512 | 30x |
| 256K | 17 | 4096 | 240x |

**Recommendation:** Profile the actual page loading behavior at long sequence
lengths during bringup. If the kernel uses the conservative approach, the
circular-buffer-as-pages fallback (Strategy 2 below) may be necessary for
production performance.

## Interaction With `cur_pos_tensor`

The `sliding_window_size` parameter works in conjunction with `cur_pos_tensor`
to determine the window boundaries. The kernel derives:

```text
window_start[b] = max(0, cur_pos_tensor[b] - sliding_window_size + 1)
window_end[b]   = cur_pos_tensor[b]
```

For each batch element, only KV positions in `[window_start, window_end]`
receive finite attention logits. All other positions are masked to $-\infty$.

This means `cur_pos_tensor` must accurately reflect the absolute token
position (0-indexed from the start of the sequence). If `cur_pos_tensor`
reports incorrect values, the window boundaries will be wrong, leading to:

- **Too-small `cur_pos`:** The window starts earlier than intended, causing
  the query to attend to stale or out-of-window tokens.
- **Too-large `cur_pos`:** The window starts later than intended, causing the
  query to miss valid recent tokens.

## Fallback Strategy 1 --- Manual Page Table Truncation

If the kernel does not optimize page loading for windowed attention (the
conservative case above), the host can manually truncate the page table to
reference only the pages that overlap with the window.

### Mechanism

At decode step $T$ with window $W$ and `block_size` $B_s$:

1. Compute the first and last virtual pages in the window:

   ```python
   t_low = max(0, T - W + 1)
   p_low = t_low // block_size
   p_high = T // block_size
   ```

2. Extract the relevant page table slice:

   ```python
   # Original page table: [batch, max_pages_per_seq]
   # Truncated page table: [batch, p_high - p_low + 1]
   truncated_page_table = page_table[:, p_low:p_high + 1]
   ```

3. Pass the truncated page table and adjust `cur_pos_tensor`:

   ```python
   # Adjust cur_pos to be relative to the truncated page table
   adjusted_cur_pos = cur_pos_tensor - p_low * block_size

   attn_output = paged_sdpa_decode(
       q, k_cache, v_cache,
       cur_pos_tensor=adjusted_cur_pos,
       sliding_window_size=W,            # still needed for partial-block masking
       page_table_tensor=truncated_page_table,
   )
   ```

### Pros

- Forces the kernel to load only the window-relevant pages, achieving $O(W)$
  DRAM bandwidth.
- No kernel modification required.
- Works with the existing `paged_sdpa_decode` interface.

### Cons

- Requires per-step page table manipulation on the host, adding Python
  overhead and preventing the page table from being a static device tensor.
- The `cur_pos_tensor` adjustment introduces complexity and potential for
  off-by-one errors.
- Dynamic page table shapes may conflict with Metal Trace capture, which
  requires fixed tensor shapes across captured steps.

### Compatibility With Metal Trace

Metal Trace requires that all tensor shapes and op arguments remain constant
across captured steps. A dynamically truncated page table violates this
requirement because `p_low` and `p_high` change on every step.

**Workaround:** Pad the truncated page table to a fixed size of
$\lceil W / B_s \rceil + 1$ entries on every step. Use sentinel values
(-1 or duplicate entries pointing to a "dummy" page) for positions outside
the window. This requires the kernel to handle sentinel page entries
gracefully.

## Fallback Strategy 2 --- Circular KV Buffer for Sliding Layers

Instead of using the paged KV cache for sliding layers, allocate a fixed-size
circular buffer of exactly 1024 tokens per layer and bypass the paging
mechanism entirely. This is the approach described in the
[windowed attention guide](../../windowed_attention_foundations_and_t3k_mapping/ch2_kv_cache_management/circular_buffer_layout.md).

### Mechanism

For each sliding layer, allocate:

```text
K cache: [B, 16, 1024, 256]   -- 16 KV heads, 1024-token circular buffer, head_dim=256
V cache: [B, 16, 1024, 256]
```

Use `ttnn.update_cache` to write to slot `T % 1024` on each decode step:

```python
update_index = T % 1024
ttnn.update_cache(k_cache, key_states, update_index)
ttnn.update_cache(v_cache, value_states, update_index)
```

For SDPA, pass the full circular buffer to
`ttnn.scaled_dot_product_attention_decode` (the non-paged variant) with a
position mask that handles the fill phase:

```python
n_valid = min(T + 1, 1024)
if n_valid < 1024:
    # Fill phase: mask out unwritten slots
    position_mask = build_fill_mask(n_valid, 1024)
else:
    # Steady state: all 1024 slots valid, no mask needed
    position_mask = None

attn_output = ttnn.scaled_dot_product_attention_decode(
    query_states,
    k_cache,          # [B, 16, 1024, 256]
    v_cache,          # [B, 16, 1024, 256]
    cur_pos_tensor=current_pos,
    attn_mask=position_mask,
    scale=self.scale,
)
```

### Circular-Buffer-as-Pages Alternative

The circular buffer can also be overlaid on the paged infrastructure by
allocating exactly $\lceil 1024 / \text{block\_size} \rceil$ pages per
sequence and writing in circular fashion, as described in
[paged SDPA and windowing](../../windowed_attention_foundations_and_t3k_mapping/ch5_paged_kv_cache/paged_sdpa_and_windowing.md).
With `block_size=64`, this is 16 pages. The page table entries are fixed
(never change after allocation), and the host reorders the page table at
each step to present blocks in chronological order. This approach combines
the memory efficiency of the circular buffer with the paged cache
infrastructure.

### Pros of Circular Buffer Approach

- **Bounded memory.** Exactly 1024 tokens per layer per sequence, regardless
  of total sequence length. At BF16:
  $2 \times 16 \times 1024 \times 256 \times 2 = 16{,}777{,}216$ bytes
  $= 16$ MB per layer. Across 50 sliding layers: 800 MB per device (before
  TP sharding).
- **Constant DRAM bandwidth.** The SDPA kernel always reads exactly 1024 KV
  entries, never more. No wasted bandwidth on out-of-window pages.
- **Metal Trace compatible.** All tensor shapes are fixed. The only
  per-step variable is the `update_index` scalar and (optionally) the mask,
  which has a fixed shape.
- **Simple implementation.** No page table manipulation, no per-step
  truncation.

### Cons of Circular Buffer Approach

- **Two KV cache systems.** The model must maintain paged KV caches for
  global layers (which need the paged infrastructure for variable-length
  full-causal attention) and circular buffers for sliding layers. This
  increases implementation complexity.
- **No memory sharing.** Paged caches allow different sequences to share a
  block pool, enabling efficient multi-sequence serving. Circular buffers
  are pre-allocated per sequence and cannot be shared.
- **Fill phase handling.** During the first 1024 tokens, the circular buffer
  is not yet full. The SDPA call needs a position mask to exclude unwritten
  slots. This mask transitions from dynamic (during fill) to static (at
  steady state), which may require two code paths or a mask that is always
  applied but becomes all-zeros at steady state.

## Recommendation for Gemma 4 31B

### Phase 1: Initial Bringup

Use the **native `sliding_window_size` parameter** of `paged_sdpa_decode`.
This is the simplest approach:

```python
attn_output = paged_sdpa_decode(
    q, k_cache, v_cache,
    cur_pos_tensor=current_pos,
    scale=self.scale,
    sliding_window_size=1024,
    page_table_tensor=page_table,
    program_config=config,
)
```

This requires no page table manipulation, no circular buffer setup, and no
custom masking. It works immediately if the kernel correctly handles the
`sliding_window_size` parameter with paged mode.

### Phase 2: Performance Evaluation

Profile the decode latency at long sequence lengths (32K+) for sliding
layers. Measure:

1. **DRAM bandwidth utilization.** Is the kernel loading only ~16 pages
   (1024 tokens) or all pages up to `cur_pos`?
2. **Per-layer sliding attention latency.** Does it remain constant as
   sequence length grows, or does it increase?
3. **Comparison with global layers.** Sliding layer SDPA should be
   significantly faster than global layer SDPA at long sequences due to the
   window constraint.

### Phase 3: Optimization (If Needed)

If profiling reveals that the kernel does not optimize page loading for
windowed attention:

1. **First try Strategy 2 (circular-buffer-as-pages)** from the windowed
   attention guide. Allocate 16 fixed pages per sequence per sliding layer,
   write circularly, reorder the page table on the host. This is the
   recommended fallback because it bounds memory and bandwidth while staying
   within the paged infrastructure.

2. **If circular-buffer-as-pages is insufficient** (e.g., incompatible with
   the serving framework's page allocator), fall back to **Strategy 1
   (manual page table truncation)** with a fixed-size padded page table for
   Metal Trace compatibility.

3. **Long-term:** Request a kernel optimization to `paged_sdpa_decode` that
   implements page-aware windowing (Strategy A from the windowed attention
   guide): the kernel computes `p_low` and `p_high` from `cur_pos` and
   `sliding_window_size`, and only gathers pages in that range.

## Summary Table

| Approach | Memory Bounded? | DRAM BW Bounded? | Kernel Mod? | Trace Compatible? | Complexity |
|----------|----------------|------------------|-------------|-------------------|------------|
| Native `sliding_window_size` | No (pages accumulate) | Depends on kernel | No | Yes | Low |
| Manual page table truncation | Yes (truncated) | Yes (forced) | No | Needs padding workaround | Medium |
| Circular buffer (non-paged) | Yes (1024 fixed) | Yes (always 1024) | No | Yes | Medium |
| Circular-buffer-as-pages | Yes (16 pages fixed) | Yes (always 16 pages) | No | Yes | Medium |
| Kernel-native windowed paging | Yes | Yes | Yes (new `start_page`) | Yes | High (kernel dev) |

For Gemma 4 31B, the native `sliding_window_size` parameter provides the
fastest path to a working implementation. The circular-buffer-as-pages
approach is the recommended optimization if native windowing does not optimize
page loading.

## Cross-Reference to Windowed Attention Guide

The following files in the
[windowed attention guide](../../windowed_attention_foundations_and_t3k_mapping/index.md)
provide detailed background for the strategies discussed above:

- [Circular Buffer Layout](../../windowed_attention_foundations_and_t3k_mapping/ch2_kv_cache_management/circular_buffer_layout.md):
  slot assignment, write pointer arithmetic, and TTNN tensor shape.
- [Paged SDPA and Windowing](../../windowed_attention_foundations_and_t3k_mapping/ch5_paged_kv_cache/paged_sdpa_and_windowing.md):
  page-aware windowing (Strategy A) and circular-buffer-as-pages (Strategy B)
  with host-side page table reordering.
- [Decode Primitives](../../windowed_attention_foundations_and_t3k_mapping/ch4_ttnn_primitives/decode_primitives.md):
  the step-by-step decode pipeline for windowed attention, including
  `ttnn.update_cache` and `ttnn.scaled_dot_product_attention_decode`.

---

**Next:** [Chapter 6 --- Tensor-Parallel Sharding on T3K](../ch6_tp_sharding/index.md)
