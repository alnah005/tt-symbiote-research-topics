# Eviction and Page Reuse

This file examines the correctness and efficiency properties of the paged KV
cache once a windowed sequence enters steady state — that is, once it has
generated more than `w` tokens and the window eviction policy begins discarding
old KV entries. The analysis applies to both strategies introduced in
`paged_sdpa_and_windowing.md`, with emphasis on the correctness invariants that
the paging layer must maintain and the memory fragmentation characteristics of
fixed-size windowed page pools.

## How Window Eviction Maps to Page Eviction

### Token-Level Eviction

At the token level, windowed attention evicts the KV vectors for position
`T - w` when the new token at position `T` is written. This was established in
Chapter 2: the circular buffer slot `(T - w) mod w` is overwritten in place.
One slot is evicted per decode step, and the eviction is implicit in the
overwrite — there is no separate deletion operation at the token level.

### Block-Level Eviction

In the paged setting, token-level eviction does not correspond directly to
block-level eviction. A physical block holds `block_size` token slots. An
entire block becomes free for reuse only when **all** `block_size` of its token
slots have been evicted — i.e., when all the tokens it holds have fallen outside
the window.

For Strategy B (circular-buffer-as-pages, the recommended approach), blocks are
never individually freed and reallocated. The fixed `N_win` blocks are
overwritten in round-robin order, so block-level "eviction" is the overwrite
of the oldest block's contents with the new block-worth of tokens. This happens
every `block_size` decode steps:

```text
Block eviction cadence (w=256, block_size=64, N_win=4):

  T=0..63:   Writing into block 0
  T=64..127: Writing into block 1
  T=128..191: Writing into block 2
  T=192..255: Writing into block 3   (buffer full after T=255)
  T=256..319: Writing into block 0   ← block 0 overwritten; tokens 0–63 evicted
  T=320..383: Writing into block 1   ← block 1 overwritten; tokens 64–127 evicted
  ...

  Block eviction frequency: one block evicted per block_size steps = every 64 steps
  Tokens evicted per event: block_size = 64 tokens simultaneously
```

For Strategy A (page-aware windowing), a block becomes eligible for return to
the pool when its highest-indexed token's position falls below `t_low = T - w + 1`.
The host reclaims this block, updates the page table entry to the sentinel
value, and returns the physical block to the allocator's free list. This happens
in the same `block_size`-step cadence but requires an explicit deallocation step.

## Risk of Stale Page Table Entries

### Strategy A Staleness Hazard

Under Strategy A, the page table for a long sequence can accumulate sentinel
entries for freed blocks alongside valid entries for current-window blocks. The
risk of a stale entry arises in two scenarios:

**Hazard 1 — Premature freeing.** If the host frees a block whose tokens are
still inside the window (e.g., `p_low` is miscalculated), the kernel will read
the `-1` sentinel and either dereference an invalid block index (undefined
behaviour) or silently compute attention with wrong data if the physical block
has since been allocated to a different sequence and overwritten with its KV
vectors.

**Hazard 2 — Late freeing.** If the host does not free old blocks promptly
(delayed reclamation), the block pool fills up unnecessarily and sequence
allocations stall. While this is a performance issue rather than a correctness
issue, it can manifest as a hard allocation failure in a saturated serving
system.

**Hazard 3 — Concurrent freeing and re-allocation.** In a multi-threaded host
runtime where one thread is running decode and another is managing the allocator,
a block freed by the allocator thread may be immediately assigned to a new
sequence while the decode thread is still holding a reference in the page table.
The window of vulnerability is the interval between the host updating
`page_table[b, p_old]` to `-1` and the kernel actually reading that entry. If
the kernel has already fetched the old (non-sentinel) value from a cached copy
of the page table, it may read stale data. This race must be prevented by
ensuring the page table update is committed to device memory before the next
decode kernel launch.

Strategy B eliminates Hazards 1 and 3 entirely because no blocks are ever freed.
The fixed `N_win` blocks are permanently assigned to the sequence, and the page
table never contains sentinel entries after initial allocation. Hazard 2 does
not apply because block pool pressure is predictable and static per sequence.

### Correctness Invariants

Regardless of which strategy is used, the paging layer must maintain the
following invariants at all times:

**Invariant 1 — Page table validity.** For every active sequence `b` and every
virtual page `p` within the current window (`p_low(T) <= p <= p_high(T)`), the
page table entry `page_table[b, p]` must hold a valid, in-bounds physical block
index pointing to a block that contains the KV vectors for the corresponding
token range.

Formally:

$$\forall\, b \in [0, B-1],\; \forall\, p \in [p_{\text{low}}(T),\, p_{\text{high}}(T)]: \quad
  \text{page\_table}[b, p] \in [0, N_{\text{blocks}} - 1]$$

**Invariant 2 — Block ownership uniqueness.** No physical block may appear in
the page tables of two different active sequences simultaneously, except through
deliberate copy-on-write sharing (which windowed serving does not use). This
invariant prevents cross-sequence contamination.

$$\forall\, b_1 \neq b_2,\; \forall\, p_1, p_2:\quad
  \text{page\_table}[b_1, p_1] \neq \text{page\_table}[b_2, p_2]
  \quad \text{(when both entries are valid)}$$

**Invariant 3 — Write-before-read ordering.** The KV write for token `T`
(via `ttnn.update_cache` or the paged equivalent) must complete before the
attention read for the same step. This is a TTNN command queue ordering
constraint: both writes must be enqueued before the `paged_sdpa_decode` call on
the same queue.

**Invariant 4 — Block freshness.** The data in each physical block must
correspond exactly to the token range indicated by the page table. This invariant
is violated if a block is freed, reallocated to a new sequence, partially
overwritten by the new sequence's KV vectors, and then incorrectly read by the
original sequence's attention op because of a stale page table entry.

Under Strategy B, Invariants 1 and 4 reduce to: the host correctly computes
`bwp = (T // block_size) % N_win` and writes new KV vectors to the block at
`page_table[b, bwp]`. Since the page table is static, the risk is limited to
the write address computation.

### Stale Entry Detection

A defensive implementation can add a lightweight staleness check at the
host level. Before each decode step, the host verifies:

```python
def check_window_invariants(page_table, seq_lens, block_size, N_win, N_blocks):
    for b in range(len(seq_lens)):
        T = seq_lens[b] - 1
        if T < block_size:
            # Strategy B pre-allocates all N_win blocks, so invariants hold from T=0.
            # This guard is only needed for lazy-allocation variants where blocks are
            # allocated on demand and page-table entries may be uninitialized before
            # the first block_size steps.  Under Strategy B as written, the guard is
            # unnecessary and can be removed without affecting correctness.
            continue
        p_oldest = (T // block_size + 1) % N_win
        # In Strategy B: all N_win entries must be valid block indices
        for i in range(N_win):
            p = (p_oldest + i) % N_win
            assert 0 <= page_table[b, p] < N_blocks, (
                f"Sequence {b}: page_table[{p}] = {page_table[b, p]} is invalid "
                f"at T={T}"
            )
```

This check is O(B * N_win) per step and is recommended in debug/validation
builds. In production it should be compiled out.

## Memory Fragmentation Implications of Fixed-Size Windowed Page Pools

### Baseline: No Fragmentation Under Strategy B

A fixed-size windowed page pool under Strategy B has a remarkably clean
fragmentation profile. Each active sequence permanently holds exactly `N_win`
physical blocks. The pool has `N_pool` total blocks. The maximum number of
simultaneously active sequences is:

$$B_{\max} = \Bigl\lfloor \frac{N_{\text{pool}}}{N_{\text{win}}} \Bigr\rfloor$$

This is exact — there is no internal fragmentation (blocks are fully utilised
once steady state is reached) and no external fragmentation (all blocks are the
same size and perfectly interchangeable). The pool can be managed as a simple
free list of block indices with O(1) allocate and free operations.

```text
Pool state example: N_pool=20, N_win=4, three active sequences

  Blocks  0– 3: allocated to sequence 0
  Blocks  4– 7: allocated to sequence 1
  Blocks  8–11: allocated to sequence 2
  Blocks 12–19: free  (can accommodate two more sequences: 20 - 12 = 8 ≥ 4)

  External fragmentation = 0  (contiguous free region)
  Internal fragmentation = 0  (each block fully used at steady state)
  Maximum simultaneous sequences = floor(20/4) = 5
```

During the fill phase (before `T` reaches `w`), some token slots within
allocated blocks are uninitialised (never written). These are masked to `-inf`
in the attention mask. The physical memory is allocated but logically unused,
which is technically internal fragmentation, but it is bounded by `N_win * block_size`
slots per sequence and resolves automatically as the sequence reaches steady state.

### Complications from Mixed Window Sizes

If different sequences in the same batch have different window sizes `w_b`, their
`N_win` values differ:

$$N_{\text{win},b} = \Bigl\lceil \frac{w_b}{\text{block\_size}} \Bigr\rceil$$

A pool of uniform-size blocks can still serve all sequences, but the simple
`B_max = floor(N_pool / N_win)` formula no longer applies. The allocator must
pack allocations of varying sizes into the pool, which can produce external
fragmentation in the general case.

For example, if `N_pool = 10` and sequence sizes alternate between `N_win = 3`
and `N_win = 4`:

```text
Allocation order: seq A (3), seq B (4), seq C (3)

  Blocks 0–2:  seq A
  Blocks 3–6:  seq B
  Blocks 7–9:  seq C

  Remaining free: 0 blocks
  If seq B finishes and is deallocated, blocks 3–6 are freed (4 contiguous).
  Next allocation request for N_win=3 uses blocks 3–5; block 6 is left isolated.
  Next allocation request for N_win=4 cannot be satisfied (only 1 free block).

  This is classic external fragmentation.
```

Mitigations include:

- **Uniform window size policy**: use the same `w` for all sequences in a given
  serving deployment. Model configs (Qwen, Mistral) typically specify a single
  `w` so this is achievable without loss of generality.
- **Power-of-two block pool partitioning**: maintain separate free lists for
  different `N_win` values and allocate from the matching list. Memory
  efficiency decreases but allocation is fragmentation-free within each class.
- **Buddy allocator**: not well-suited here because `N_win` values are not
  necessarily powers of two.

### Strategy A Fragmentation Characteristics

Under Strategy A, fragmentation behaviour is more complex because block
allocations grow then shrink as sequences advance through their generation.
See `paged_sdpa_and_windowing.md` Memory Accounting for the derivation.
The freed blocks return to the pool in the same order they were allocated
(oldest first), which is ideal for a FIFO-ordered free list and minimises
fragmentation.

However, if multiple sequences have different current positions T_b, their block
freeing events are interleaved, and the freed blocks do not form contiguous
regions in the pool unless the allocator was careful to allocate blocks for
different sequences from contiguous regions. In practice, allocators use simple
linear scans or free lists that tolerate non-contiguous free regions, so this
is not typically a problem.

### Quantitative Memory Comparison

For a representative serving configuration (one layer, one device):

| Parameter | Value |
|---|---|
| `block_size` | 64 |
| `w` | 4096 |
| `H_kv` | 8 |
| `d` | 128 |
| dtype | BF16 (2 bytes) |

Per-sequence block count under Strategy B:

$$N_{\text{win}} = \lceil 4096 / 64 \rceil = 64 \text{ blocks}$$

Memory per block:

$$2 \times H_{\text{kv}} \times \text{block\_size} \times d \times 2 \text{ bytes}
= 2 \times 8 \times 64 \times 128 \times 2 = 262{,}144 \text{ bytes} = 256 \text{ KiB}$$

(factor of 2 for K and V)

Total per-sequence windowed KV cache:

$$64 \times 256 \text{ KiB} = 16 \text{ MiB per sequence per layer}$$

For a 32-layer model with 8 GiB of DRAM per Wormhole chip:

$$B_{\max} = \Bigl\lfloor \frac{8 \text{ GiB}}{32 \times 16 \text{ MiB}} \Bigr\rfloor = \Bigl\lfloor \frac{8192}{512} \Bigr\rfloor = 16 \text{ sequences}$$

This is a theoretical ceiling assuming all DRAM is used for KV cache. In
practice, weight tensors and activations consume the majority of DRAM, leaving
1–2 GiB for KV cache and supporting 2–4 simultaneous windowed sequences at
`w = 4096`. Reducing `w` proportionally increases `B_max`.

### Block Pool Sizing Guidance

The block pool size `N_pool` should be chosen to support the target batch size
`B_target` with margin:

$$N_{\text{pool}} \geq B_{\text{target}} \times N_{\text{win}} \times (1 + \epsilon)$$

where `epsilon` is a headroom factor (e.g., 0.1 to 0.2) to accommodate the
fill-phase over-allocation in Strategy A or to allow in-flight sequence starts
before prior sequences have fully completed in Strategy B. For Strategy B, no
headroom is strictly required because allocations are perfectly predictable;
the margin is a safety buffer against implementation errors.

---

**Next:** [Chapter 6 — T3K Mesh Sharding and CCL Implications](../ch6_t3k_sharding/index.md)
