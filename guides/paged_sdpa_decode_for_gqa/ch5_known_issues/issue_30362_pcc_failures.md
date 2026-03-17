# GitHub Issue #30362 — Sporadic PCC Failures in Paged SDPA Decode

**Status: open as of early 2026. No upstream fix merged.**

---

## Reproduction Configuration

Failures have been confirmed under this exact configuration:

```
b          = 1
nh         = 8
nkv        = 1          (MQA; group_size = 8)
s          = 128 * 1024 (128K token sequence length)
block_size = 128
grid       = (8, 4)     (32 cores)
```

At sporadic positions in the range `0–16384` (`cur_pos` values), the output PCC
against a reference softmax implementation falls below the accepted threshold.
The failures are not uniformly distributed; they cluster near block boundaries
(multiples of 128) but also appear mid-block at unpredictable offsets.

---

## Why CI Did Not Catch It

The continuous integration test suite increments `cur_pos` by fixed strides:

- **Stride 71**: covers positions 0, 71, 142, 213, … — none of which align with
  the specific affected values.
- **Stride 3001**: covers positions 0, 3001, 6002, … — similarly avoids the
  affected positions by coincidence.

Because the failing positions are sparse and the strides step over them, all CI
runs pass even though the bug is present.  This is a coverage gap, not a
resolution.

---

## Suspected Root Cause

The most likely cause is a block-boundary arithmetic error in the paged address
computation inside the Flash-Decode kernel.  Two candidate sites:

1. **`num_active_blocks` computation**: an off-by-one in `ceil(cur_pos / block_size)`
   causes either one too many or one too few blocks to be fetched from the page
   table, shifting all subsequent KV reads.

2. **Intra-block offset**: the partial-block mask for the last active block uses
   `cur_pos % block_size`; if `cur_pos` is exactly a multiple of `block_size`
   (boundary case), a mishandled zero remainder can cause the last full block to
   be partially masked, dropping valid KV entries from the softmax.

Neither site has been conclusively identified as of early 2026.

---

## Impact on GQA Configurations

Issue #30362 was reproduced with `nkv = 1` (MQA), but the same block-boundary
arithmetic is shared across all GQA configurations.  When `nkv > 1`:

- The physical block address for KV head `h` is computed as
  `block_base + h * block_size * dh`.
- If the active-block count is wrong by one, the head offset is applied to the
  wrong physical block, reading KV data from the wrong position in memory.

For a 4-KV-head GQA model (`nkv = 4`, `nh = 32`, `group_size = 8`) using
`block_size = 128`, the boundary positions occur every 128 tokens.  At each
such boundary there is a risk of reading the wrong physical block for all 4 KV
heads simultaneously, producing a correlated error across all query heads in
the affected group.

---

## Recommended Workaround

Do not rely on stride-based position coverage in tests.  Use one of the
following strategies:

```python
# Option A: test every position (thorough; slow for large seq_len)
for pos in range(1, seq_len + 1):
    output = sdpa_decode(..., cur_pos=[pos] * b)
    assert pcc(output, reference) > threshold

# Option B: boundary-focused sweep (faster; covers the known-bad region)
block_size = 128
bad_positions = []
for k in range(1, seq_len // block_size + 2):
    for delta in range(-2, 3):       # ±2 around each boundary
        p = k * block_size + delta
        if 1 <= p <= seq_len:
            bad_positions.append(p)

for pos in bad_positions:
    output = sdpa_decode(..., cur_pos=[pos] * b)
    assert pcc(output, reference) > threshold
```

Until the issue is resolved, validate across the full position range during
initial bringup on new hardware or after any kernel update that touches
block-address arithmetic.
