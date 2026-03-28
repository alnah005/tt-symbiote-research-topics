# Paged SDPA and Windowing

This file analyses how windowed attention can be integrated with the paged KV
cache in tt-transformers. It begins with a self-contained recap of the paged
KV cache model, explains how `paged_sdpa_decode` selects pages for a given
sequence position, then develops and compares two strategies for enforcing a
window constraint within the paging layer.

## Recap: The Paged KV Cache Model

### Motivation

The non-paged circular-buffer design from Chapter 2 allocates a fixed
`[B, H_kv, w, d]` tensor per layer up front, with each batch slot occupying
a dedicated, contiguous region of DRAM. This works well when `B` is fixed and
all sequences are expected to reach steady-state window length. In a serving
system, however, sequences arrive and finish at unpredictable times, their
generation lengths vary widely, and the maximum simultaneous batch size is not
known at allocation time. Pre-allocating `B_max * w` slots per layer wastes
DRAM when many slots are empty and cannot be reclaimed by other sequences.

Paged KV caches solve the fragmentation problem with the same mechanism
operating systems use for process memory: virtual addresses (logical token
positions within a sequence) are separated from physical addresses (actual DRAM
locations) through a page table. Physical memory is allocated in fixed-size
blocks called **pages** or **blocks**, and a per-sequence **page table** records
which physical block holds each virtual page of that sequence's KV history.

### Definitions

| Term | Definition |
|------|------------|
| `block_size` | Number of tokens (`S_blk`) whose KV vectors fit in one physical page. Must be a positive integer; common values are 32, 64, 128. |
| Physical block | Contiguous DRAM region holding `block_size` key vectors and `block_size` value vectors for one layer, one head group. Shape: `[H_kv, block_size, d]` for one layer. |
| Block pool | The set of all physical blocks available for allocation. Total capacity: `N_blocks * block_size` token-slots per layer. |
| Virtual page index | `p = floor(t / block_size)` — the logical page number within a sequence for absolute token position `t`. |
| Page table | Per-sequence array of length `ceil(T_max / block_size)` mapping virtual page indices to physical block indices. Entry value `-1` or a sentinel indicates an unallocated page. |
| Page table tensor | In tt-transformers: an integer tensor of shape `[B, max_pages_per_seq]` that lives in DRAM and is passed to `paged_sdpa_decode`. |

### Physical Layout

The paged KV cache is stored as a single pre-allocated tensor (the **block
pool**) per layer, with shape:

```text
block_pool_K : [N_blocks, H_kv, block_size, d]
block_pool_V : [N_blocks, H_kv, block_size, d]
```

where `N_blocks` is the total number of physical blocks. At decode step `T`
(0-based position index of the token being written, so `T+1` tokens have been
stored), the sequence occupies `floor(T / block_size) + 1` physical blocks.
Different sequences can occupy non-contiguous blocks from the same pool.

The page table for a batch of `B` sequences is an integer matrix:

```text
page_table : [B, max_pages_per_seq]   (dtype: int32)
```

Entry `page_table[b, p]` is the physical block index for virtual page `p` of
sequence `b`. When `paged_sdpa_decode` runs, it uses this table to gather the
relevant blocks from the pool before computing attention.

### How `paged_sdpa_decode` Selects Pages

At decode step `T` for sequence `b`, the set of virtual pages covering the
KV history is:

$$\text{pages required} = \bigl\{0, 1, \ldots, \bigl\lfloor T / \text{block\_size} \bigr\rfloor\bigr\}$$

The op resolves each virtual page to a physical block via `page_table[b, :]`
and loads those blocks from the block pool. Specifically, the kernel:

1. Reads `page_table[b, 0 .. floor(T / block_size)]` to obtain a list of
   physical block indices.
2. Gathers those blocks from `block_pool_K` and `block_pool_V`, assembling a
   logical KV tensor of shape `[1, H_kv, T_rounded, d]` where
   `T_rounded = (floor(T / block_size) + 1) * block_size` (padded to block boundary).
3. Applies an attention mask that zeroes out the `T_rounded - T` padding
   positions at the end.
4. Computes SDPA over the gathered KV tensor with the single query vector.

The critical observation is that in the default (non-windowed) implementation,
**all** pages from virtual page 0 up to and including the page containing
token `T` are loaded. The page table is an ordered sequence of block
allocations from the beginning of the sequence, and the op's gather covers
that entire prefix.

```text
Non-windowed decode at T=200, block_size=64:

  Virtual pages:  [  0   ] [  1   ] [  2   ] [  3   ]
  Token range:   0–63     64–127   128–191  192–200 (partial)

  Page table: [b, 0..3] = [phys_7, phys_2, phys_15, phys_0]

  paged_sdpa_decode gathers all 4 blocks → assembles [1, H_kv, 256, d]
  → masks out positions 201–255 → computes attention over 201 tokens
```

## Strategy A: Page-Aware Windowing

### Core Idea

Rather than loading all `floor(T / block_size) + 1` pages, the op loads only the
pages that cover the current window — between `ceil(w / block_size)` and
`ceil(w / block_size) + 1` pages depending on block-boundary alignment. Pages
older than the window boundary are not gathered, so they contribute no bandwidth
cost and do not enter the attention computation.

The window boundary falls at absolute position `t_low = T - w + 1`. The first
page that is at least partially within the window is virtual page:

$$p_{\text{low}} = \Bigl\lfloor \frac{T - w + 1}{\text{block\_size}} \Bigr\rfloor$$

and the last page is:

$$p_{\text{high}} = \Bigl\lfloor \frac{T}{\text{block\_size}} \Bigr\rfloor$$

giving `p_high - p_low + 1` pages to gather. In steady state this count
equals exactly `ceil(w / block_size)` (plus at most 1 extra page if the window
boundary does not align to a block boundary).

```text
Page-aware windowing at T=200, w=128, block_size=64:

  t_low = 200 - 128 + 1 = 73   →  p_low  = floor(73/64)  = 1
  t_high = 200             →  p_high = floor(200/64) = 3

  Pages to gather: virtual [1, 2, 3] → physical [phys_2, phys_15, phys_0]

  Assembled tensor: [1, H_kv, 192, d]  (3 blocks × 64)
  Mask: zero positions 73–200 as valid, -inf for 64–72 (partial first block)
        and -inf for 201–255 (padding at end)

  Positions NOT loaded: virtual page 0 (tokens 0–63) — outside window
```

### Page Table Recency Requirement

For this strategy to work, `page_table[b, p_low .. p_high]` must correctly map
the most recent `ceil(w / block_size)` virtual pages to valid physical blocks.
This is satisfied automatically as long as the page table entries for `p_low`
through `p_high` are populated. Since the standard paged allocator fills the
table in order (page 0, then 1, then 2, ...), the entries are always present
once the sequence has reached token `T`.

However, a windowed serving system may wish to **free** physical blocks for
virtual pages older than `p_low` once they fall outside the window, returning
them to the block pool for other sequences. If it does so, it must update
`page_table[b, 0 .. p_low - 1]` to the sentinel value (-1) or otherwise mark
them invalid. The kernel must be aware that `-1` entries for pages before
`p_low` are expected and should not trigger a bounds-check failure.

The standard `paged_sdpa_decode` gathers pages `0 .. floor(T / block_size)`,
which includes the now-freed pages. This breaks page-aware windowing without
modification.

### Required Interface Changes for Strategy A

To support page-aware windowing in `paged_sdpa_decode`, the following changes
are required:

1. **`start_page` argument** — An integer (or `[B]` tensor for per-sequence
   control) that tells the kernel to begin gathering at virtual page
   `p_low` rather than page 0. This avoids loading freed blocks and allows
   the kernel to short-circuit the gather loop for old pages.

   ```text
   paged_sdpa_decode(q, block_pool_K, block_pool_V,
                     page_table,
                     seq_len=T+1,
                     start_page=p_low,     ← new argument
                     attn_mask=window_mask)
   ```

2. **Window mask adjustment** — The attention mask must cover the gathered
   region `[p_low * block_size, (p_high + 1) * block_size)` rather than
   `[0, T_rounded)`. The first `(T - w + 1) % block_size` positions within
   the first gathered block fall before `t_low` and must be masked to `-inf`
   to exclude partial-block tokens outside the window boundary.

3. **Page table sentinel handling** — The kernel must tolerate sentinel entries
   (`-1`) in `page_table[b, 0 .. p_low - 1]` without dereferenceing them,
   because the host may have freed those blocks for reuse by other sequences.

4. **`SDPADecodeProgramConfig` extension** — A new boolean field
   `windowed_paging` (or equivalent) that activates the modified gather
   path. When false, behaviour is identical to the current implementation
   (backward-compatible).

### Block-Boundary Misalignment

When the window start `t_low` does not align to a block boundary, the first
gathered block contains `(t_low % block_size)` tokens from before the window
that must be masked out. This is handled by the window mask and imposes no
correctness risk, but it does mean that up to `block_size - 1` tokens of KV
data from outside the window are loaded from DRAM and then discarded after
masking. For `block_size = 64` and `w` not a multiple of 64, up to 63 extra
key/value vectors are loaded per sequence per decode step. In most cases this
overhead is negligible.

```text
Block-boundary misalignment example (w=100, block_size=64, T=200):

  t_low = 101
  p_low = floor(101/64) = 1  →  block covers tokens 64–127
  Tokens 64–100 are outside the window (t < t_low)

  These 37 tokens are loaded from DRAM (part of block 1) but masked to -inf.
  Wasted bandwidth: 37 × H_kv × d × 2 bytes per decode step per layer.
```

## Strategy B: Circular-Buffer-as-Pages

### Core Idea

Instead of treating the page table as an append-only sequence of block
allocations, this strategy maps the circular-buffer eviction policy directly
onto the paging layer. The sequence is allocated exactly `N_win` physical
blocks at the start, where:

$$N_{\text{win}} = \Bigl\lceil \frac{w}{\text{block\_size}} \Bigr\rceil$$

These `N_win` blocks are the sequence's permanent, fixed allocation for the
duration of its lifetime. New tokens are written into blocks in round-robin
order, overwriting the oldest block once all `N_win` blocks are full —
mirroring exactly the slot-level circular buffer from Chapter 2 but at
block granularity.

The page table for such a sequence has exactly `N_win` entries, and those
entries never change after initial allocation (the physical block indices are
constant). What changes on each decode step is which block holds the current
write target, tracked by a **block write pointer** `bwp`:

$$\text{bwp}(T) = \Bigl\lfloor \frac{T}{\text{block\_size}} \Bigr\rfloor \bmod N_{\text{win}}$$

The within-block write offset is:

$$\text{offset\_in\_block}(T) = T \bmod \text{block\_size}$$

```text
Circular-buffer-as-pages example: w=8, block_size=4, N_win=2

  Two physical blocks allocated: phys_A (tokens 0–3 initially), phys_B (tokens 4–7)
  page_table[b, :] = [phys_A, phys_B]   (fixed, never changes)

  T=0:  write to phys_A[0]   (bwp=0, offset=0)
  T=1:  write to phys_A[1]
  T=2:  write to phys_A[2]
  T=3:  write to phys_A[3]
  T=4:  write to phys_B[0]   (bwp=1, offset=0)
  T=5:  write to phys_B[1]
  T=6:  write to phys_B[2]
  T=7:  write to phys_B[3]
  T=8:  write to phys_A[0]   (bwp=0, offset=0) — phys_A tokens 0–3 overwritten
  T=9:  write to phys_A[1]
  ...
```

### Reading Back the Window

At decode step `T`, `paged_sdpa_decode` must read all `N_win` blocks in the
correct temporal order (oldest block first, wrapping around). The oldest block
is at virtual page:

$$p_{\text{oldest}} = \Bigl(\Bigl\lfloor \frac{T}{\text{block\_size}} \Bigr\rfloor + 1\Bigr) \bmod N_{\text{win}}$$

The blocks must be gathered in the order:

$$p_{\text{oldest}},\; (p_{\text{oldest}} + 1) \bmod N_{\text{win}},\; \ldots,\; (p_{\text{oldest}} + N_{\text{win}} - 1) \bmod N_{\text{win}}$$

to assemble a logically ordered `[1, H_kv, N_win * block_size, d]` KV tensor.
The wrap-around ordering must be communicated to the kernel, either by passing
a reordered page list or by providing `p_oldest` and letting the kernel compute
the modular sequence internally.

```text
Circular-buffer gather at T=9, w=8, block_size=4, N_win=2:

  bwp = floor(9/4) mod 2 = 2 mod 2 = 0  (currently writing into page 0 = phys_A)
  p_oldest = (floor(9/4) + 1) mod 2 = (2+1) mod 2 = 1  →  phys_B

  Gather order: [phys_B, phys_A]
    phys_B: tokens 4–7 (positions t=4,5,6,7 — replaced at T=4..7, still current)
    phys_A: tokens 8–9 in slots [0,1]; slots [2,3] still contain old data (T=2,3)

  Mask: positions 2–9 → phys_B slots 0–3 = positions 4–7 (valid)
                          phys_A slots 0–1 = positions 8–9 (valid)
                          phys_A slots 2–3 = positions 2–3 (valid, inside window [2,9])
  Note: tokens 0 and 1 are absent from the assembled tensor — their slots in
  phys_A were overwritten with tokens 8 and 9. The assembled tensor covers
  exactly token positions 2–9 (the current window [T-w+1, T] = [2, 9]).
```

In the divisible case (`w % block_size == 0`), the unwritten slots of the
write block contain tokens from exactly one full window rotation ago — these
are still within the current window and must NOT be masked. Only the
currently-being-written slot is new; the remaining slots are valid. See the
algebraic proof below that `n_stale = 0` at steady state in the divisible
case.

In the non-divisible case only (`w % block_size ≠ 0`), some tail slots of
the write block may fall outside the window and require masking to `-inf`.

### Compatibility with the Existing Interface

Strategy B is significantly more compatible with the current `paged_sdpa_decode`
interface than Strategy A. Specifically:

- The page table has a fixed, sequence-independent length `N_win` (no growing
  table, no sentinel entries for freed pages).
- The `N_win` physical blocks are always valid and always present in the page
  table. No `start_page` argument is needed.
- The kernel still gathers `N_win` blocks — the same count on every decode step
  after the fill phase — so the gather loop count is constant.
- The only new information the kernel needs is `p_oldest` (or equivalently
  `bwp`) to construct the correct gather order. A single integer scalar per
  sequence suffices.

The existing interface accepts `seq_len` to know how many total token positions
are valid (for masking). Under **Option 2** (kernel-native circular gather,
where `seq_len = T+1` is the full token count), `seq_len` encodes both the
total token count and the write pointer: `bwp = ((seq_len - 1) // block_size) % N_win`.
This means the kernel can derive `p_oldest` from `seq_len` and `N_win` without
any new argument, provided the page table is interpreted as a circular array
rather than a linear one.

> **Under Option 1 (`seq_len = min(T+1, w)`), this derivation does not work.**
> At steady state `seq_len` is pinned at `w` regardless of `T`, so
> `((seq_len - 1) // block_size) % N_win` returns a constant rather than the
> actual write-block pointer. The host must either pass `bwp` as a separate
> kernel argument, or absorb the circular reordering into the page table
> directly (which is exactly what the host-side reordering in Option 1 does —
> the reordered page table makes the kernel's knowledge of `bwp` unnecessary).

## Comparative Analysis

### Interface Compatibility

| Property | Strategy A (Page-Aware) | Strategy B (Circular-as-Pages) |
|---|---|---|
| Page table length | Grows with T (up to `ceil(T_max/block_size)`) | Fixed at `N_win` = `ceil(w/block_size)` |
| Physical blocks per sequence | Grows to `ceil(w/block_size)` – `ceil(w/block_size) + 1`; worst-case `ceil(w/block_size) + 1` | Fixed at `N_win` from the start |
| Kernel gather range | `p_low .. p_high` (sliding window over virtual pages) | All `N_win` pages (always) |
| New kernel argument | `start_page` (per-sequence scalar or vector) | `block_write_pointer` or derivable from `seq_len` |
| Sentinel/freed page handling | Kernel must skip pages with index `-1` | Not required |
| Block-boundary partial masking | Up to `block_size - 1` wasted loads | Up to `block_size - 1` stale slots masked |
| Compatible with current `paged_sdpa_decode`? | No — requires `start_page` and sentinel skipping | Mostly yes — only gather ordering changes |

### Memory Accounting

Under Strategy A, a sequence that has generated `T` tokens has `floor(T / block_size) + 1`
allocated physical blocks before old ones are freed. The maximum simultaneous
allocation before any blocks are returned is `floor(T / block_size) + 1` blocks,
peaking at `floor(T_max / block_size) + 1` for the longest sequence. After the window
stabilises, the host frees older blocks, leaving between `ceil(w / block_size)` and
`ceil(w / block_size) + 1` blocks in use depending on block-boundary alignment
(precisely `floor(T/bs) - floor((T-w+1)/bs) + 1` where `bs = block_size`).
Worst-case pool reservation must use `ceil(w / block_size) + 1`.

Under Strategy B, `N_win` blocks are allocated immediately and held for the
entire sequence lifetime. There is no temporary over-allocation. Total DRAM
reserved per sequence per layer is:

$$2 \times N_{\text{win}} \times H_{\text{kv}} \times \text{block\_size} \times d \times \text{dtype\_bytes}$$

The factor of 2 is for keys and values. This is essentially identical to the
non-paged circular buffer size from Chapter 2; the paging layer adds only the
overhead of the page table itself (a small integer array).

### Recommendation

**Strategy B (circular-buffer-as-pages) is the preferred approach** for
integrating windowed attention with `paged_sdpa_decode`. The key reasons are:

1. The page table is fixed-length, which simplifies the host-side allocator and
   avoids the state machine required for progressive block freeing under
   Strategy A.
2. The kernel interface change is minimal: only the gather ordering within the
   fixed `N_win` pages changes. This can be expressed as a reordering of the
   page table slice before passing it to the kernel, handled entirely in Python
   host code without any kernel modification.
3. There are no sentinel entries, no `start_page` argument, and no kernel-level
   bounds checking for freed blocks — all of which would add complexity and
   potential correctness hazards.
4. The memory footprint is identical to Strategy A at steady state and avoids
   the temporary over-allocation of Strategy A during the early generation steps.

## Required Interface Changes for Strategy B

### Option 1: Host-Side Page Table Reordering

The lightest-weight integration requires no kernel changes at all. The host
Python code reorders the page table slice for each sequence before calling
`paged_sdpa_decode`, placing pages in `[p_oldest, p_oldest+1, ..., (p_oldest + N_win - 1) % N_win]`
order:

```python
def reorder_page_table_for_windowing(page_table, seq_lens, block_size, N_win):
    """
    page_table: [B, N_win]  (original fixed allocation)
    seq_lens:   [B]         (current token count per sequence)
    Returns a reordered page table [B, N_win] with oldest block first.
    """
    B = page_table.shape[0]
    reordered = page_table.clone()
    for b in range(B):
        T = seq_lens[b] - 1  # 0-indexed position of most recent token
        if T < N_win * block_size:
            # Fill phase: the buffer is not yet full. Blocks were written in
            # natural order (phys_A first, then phys_B, ...), so the page table
            # is already "oldest first". Reordering is not needed and would be
            # harmful: applying p_oldest = (T // block_size + 1) % N_win places
            # a not-yet-written block at assembled-tensor index 0, which falls
            # below seq_len = T+1 and is therefore not suppressed by the
            # kernel's trailing-padding mask.
            pass  # keep natural order; seq_len = T+1 correctly masks unwritten tail
        else:
            # Steady state: the buffer has been fully written at least once.
            # Circular reorder so the oldest block (the one about to be
            # overwritten next) is placed at assembled-tensor index 0.
            p_oldest = (T // block_size + 1) % N_win
            indices = [(p_oldest + i) % N_win for i in range(N_win)]
            reordered[b] = page_table[b, indices]
    return reordered
```

The reordered page table is then passed as-is to `paged_sdpa_decode`. The kernel
sees a linearly-ordered set of blocks (as it normally would) and behaves
identically to the non-windowed case, except that the blocks are now in
chronological window order. The `seq_len` argument is set to `min(T+1, w)` so
the kernel's trailing-padding mask suppresses the `N_win * block_size - min(T+1, w)`
trailing slots during the fill phase.

This option has zero kernel development cost and works with the existing
`paged_sdpa_decode` without modification.

### Option 2: Kernel-Native Circular Gather

A more principled integration adds a `circular_block_offset` field to
`SDPADecodeProgramConfig`:

```text
SDPADecodeProgramConfig:
  ...existing fields...
  windowed: bool = False
    When True, interprets the page table as a circular array of N_win blocks.
  circular_block_offset: int = 0
    Index of the oldest block within the page table (p_oldest).
    Ignored when windowed = False.
```

With this config, the kernel computes:

$$\text{physical\_block}(i) = \text{page\_table}\bigl[b,\; (i + \text{circular\_block\_offset}) \bmod N_{\text{win}}\bigr]$$

for `i = 0, 1, ..., N_win - 1`. This avoids the per-step page table
reallocation on the host but requires a modest kernel modification to
replace the linear gather loop with a modular one.

`circular_block_offset` can be communicated as a per-sequence value via a
`[B]` integer tensor if different sequences in the same batch have different
window positions (the common case in serving).

### Mask Construction for Windowed Paging

#### Common case: `block_size` divides `w` exactly

The analysis below applies when `N_win * block_size == w`, i.e., when
`block_size` divides `w` exactly. This is the standard deployable
configuration (e.g., `w = 4096`, `block_size = 64`, `N_win = 64`). Non-
divisible cases introduce partially-stale blocks that require additional
masking analysis beyond this guide's scope; a note on that appears at the end
of this subsection.

**Definitions (steady state, `T >= w - 1`):**

```text
t_low            = T - w + 1           (first valid token position)
phys_block(t)    = (t // block_size) % N_win
bwp              = phys_block(T)       (block currently being written)
p_oldest         = (bwp + 1) % N_win  (oldest block, placed first after reorder)
oldest_block_start = ((T // block_size) - N_win + 1) * block_size
```

**Step 1 — Count stale slots.**

The assembled tensor (after page-table reordering for Option 1) spans token
positions `oldest_block_start` through `T + (block_size - 1 - T % block_size)`
(padded to the block boundary). The number of stale tokens at the head of the
assembled tensor is:

```
n_stale = max(0, t_low - oldest_block_start)
        = max(0, (T - w + 1) - ((T // block_size) - N_win + 1) * block_size)
```

When `block_size` divides `w`, this simplifies to `n_stale = 0` at every
step `T >= w - 1`:

```
oldest_block_start = ((T // block_size) - N_win + 1) * block_size
t_low              = T - w + 1

t_low - oldest_block_start
  = (T - w + 1) - ((T // block_size) - N_win + 1) * block_size
```

Because `N_win = w / block_size` and `T = q * block_size + r` for some
integers `q, r` with `0 <= r < block_size`:

```
= (q*bs + r - w + 1) - (q - w/bs + 1) * bs
= (q*bs + r - w + 1) - q*bs + (w/bs - 1)*bs
= r - w + 1 + w - bs
= r - bs + 1
<= 0   (since r < bs, so r - bs + 1 <= 0)
```

Thus `n_stale = max(0, r - bs + 1) = 0` always. The assembled tensor
contains no stale tokens.

**Step 2 — Mask application.**

With `n_stale = 0`, no stale-prefix masking is required at steady state.
The only masking needed is:

- **Fill phase (`T < w - 1`):** trailing slots `[T + 1, N_win * block_size - 1]`
  that have never been written must be set to `−∞`. This is handled by
  passing `seq_len = T + 1` to `paged_sdpa_decode`; the kernel's trailing-
  padding mask covers these positions automatically.
- **Steady state (`T >= w - 1`):** no masking needed. Pass `seq_len = w`
  (or equivalently `N_win * block_size` when a caller-supplied `attn_mask`
  is used that already encodes all validity).

The mask tensor shape is `[B, 1, 1, N_win * block_size]`.

**Worked example — T=9, w=8, block_size=4, N_win=2:**

```text
bwp  = (9 // 4) % 2 = 2 % 2 = 0        → phys_A is the write block
p_oldest = (0 + 1) % 2 = 1             → phys_B is the oldest block

Physical block contents at T=9:
  phys_B (page 1): tokens 4, 5, 6, 7   (written at T=4,5,6,7; not yet overwritten)
  phys_A (page 0): token 8 (slot 0), token 9 (slot 1),
                   token 2 (slot 2, from prev rotation), token 3 (slot 3)

Assembled tensor after reordering (p_oldest=1 first):
  slots 0–3: phys_B = [t4, t5, t6, t7]
  slots 4–7: phys_A = [t8, t9, t2, t3]

oldest_block_start = ((9//4) - 2 + 1) * 4 = (2 - 1) * 4 = 4
t_low              = 9 - 8 + 1 = 2
n_stale            = max(0, 2 - 4) = 0

Window [t_low, T] = [2, 9]. The assembled tensor covers:
  token positions {4,5,6,7,8,9,2,3} = all of [2,9]. ✓

No stale-prefix masking needed. All 8 assembled slots are valid.
seq_len = w = 8 (no trailing padding at steady state when N_win*bs == w).
```

> **Note on non-divisible cases:** When `block_size` does not divide `w`
> (i.e., `N_win * block_size > w`), the assembled tensor is wider than `w`
> and `n_stale` can be positive at certain steps. Additionally, the write
> block contains tail slots whose previous-rotation token positions fall
> before `t_low`, requiring partial-block masking. The interaction between
> the stale prefix and these tail slots introduces accounting complexity that
> is outside the scope of this guide. In production, choose `block_size` to
> divide `w` exactly to avoid this class of masking errors entirely.

---

**Next:** [`eviction_and_page_reuse.md`](./eviction_and_page_reuse.md)
