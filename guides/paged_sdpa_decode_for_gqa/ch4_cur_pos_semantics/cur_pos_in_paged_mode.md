# `cur_pos` in Paged KV Mode

## Dual Role: Masking and Block Selection

In dense KV mode `cur_pos[i]` only controls the causal mask.  In paged mode
it does two things:

1. **Causal masking** — key positions `k >= cur_pos[i]` are masked out.
2. **Physical block selection** — the kernel uses `cur_pos[i]` to determine
   how many blocks from the page table are live for batch element `i`.

The number of active blocks for element `i` is:

```
num_active_blocks[i] = ceil(cur_pos[i] / block_size)
```

The kernel reads exactly those blocks from the page table; it does not read
blocks beyond `num_active_blocks[i]`, so stale entries in the page table at
higher indices do not affect correctness as long as `cur_pos[i]` is accurate.

---

## Block-Boundary vs. Mid-Block Behavior

### Exactly on a boundary (`cur_pos[i]` divisible by `block_size`)

All active blocks are **fully valid** — every slot in every active block
contains real KV data.  No intra-block masking is needed; the causal mask
falls precisely on the block boundary.

```
block_size = 32
cur_pos[0] = 64         → num_active_blocks = 2
Active blocks: block[0] (slots 0-31) and block[1] (slots 32-63) — both full.
```

### Mid-block (`cur_pos[i]` not divisible by `block_size`)

The last active block is **partially valid**.  Within that block, only the
first `cur_pos[i] % block_size` slots hold real data; the remaining slots
are masked out by the causal mask before softmax.

```
block_size = 32
cur_pos[0] = 65         → num_active_blocks = ceil(65/32) = 3
Active blocks: block[0] (slots 0-31, full),
               block[1] (slots 32-63, full),
               block[2] (slot  64 only — 1 valid slot; slots 65-95 masked).
```

The partial-block masking is applied by the kernel based on the local offset
`cur_pos[i] % block_size`.

---

## Worked Example

```python
block_size = 32

# Element 0: exactly 2 full blocks
cur_pos_0 = 64
num_blocks_0 = 64 // 32          # = 2 (no remainder)
# block[page_table[0][0]] → slots 0-31, all valid
# block[page_table[0][1]] → slots 32-63, all valid

# Element 1: 2 full blocks + 1 partial block
cur_pos_1 = 65
num_blocks_1 = (65 + 31) // 32   # = ceil(65/32) = 3
# block[page_table[1][0]] → slots 0-31,  all valid
# block[page_table[1][1]] → slots 32-63, all valid
# block[page_table[1][2]] → slot  64 valid; slots 65-95 MASKED

# Element 2: boundary at 0 — nothing to attend to
cur_pos_2 = 0
num_blocks_2 = 0                  # no blocks read; output is all-masked (or undefined)
```

---

## GitHub Issue #30362 — Sporadic PCC Failures at Certain `cur_pos` Values

**Status: open as of early 2026.**

Sporadic PCC (Pearson Correlation Coefficient) failures have been observed in
paged-mode SDPA when `cur_pos[i]` lands at specific values.  The failures are
non-deterministic across runs and appear more frequently near block boundaries
(values near multiples of `block_size`) but have also been seen mid-block.

Likely root cause: an off-by-one in the block-boundary arithmetic inside the
kernel — either the wrong number of blocks is fetched, or the intra-block
offset is computed incorrectly, producing a shifted mask.

**Workaround for validation and testing:**

Test at dense position coverage to surface failures reliably:

```python
# Option A: test every position (thorough, slow for large seq lens)
for pos in range(1, max_seq_len + 1):
    cur_pos = [pos] * b
    output = sdpa_decode(..., cur_pos=cur_pos)
    validate_pcc(output, reference)

# Option B: test all multiples of block_size plus offsets ±1 (faster)
block_size = 32
positions_to_test = []
for k in range(1, max_seq_len // block_size + 2):
    for delta in range(-1, 2):           # boundary-1, boundary, boundary+1
        p = k * block_size + delta
        if 1 <= p <= max_seq_len:
            positions_to_test.append(p)

for pos in positions_to_test:
    cur_pos = [pos] * b
    output = sdpa_decode(..., cur_pos=cur_pos)
    validate_pcc(output, reference)
```

Until the issue is resolved, do not assume correctness for untested `cur_pos`
values in paged mode.  Production deployments should validate across the full
position range during bringup on new hardware or after kernel updates.
