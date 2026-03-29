# Final Cross-Chapter Consistency Review

## Issue 1: Source filename inconsistency -- `gated_delta_net.py` vs `gdn.py`

The top-level `index.md` (line 72) lists the GDN implementation file as `gated_delta_net.py`. Every other chapter that references this file -- Ch3 index (line 5), `gdn_decode_flow.md` (line 3), `batched_projections.md` (line 42), `gdn_prefill_strategy.md` (lines 5, 9), `state_replication.md` (line 9) -- calls it `gdn.py`. One of these names is wrong. Pick one and fix the other.

**Fix:** Update the `index.md` source code tree to use `gdn.py` (or vice versa), matching the actual filename in the repository.

## Issue 2: Contradictory total state across 48 GDN layers -- 576 MB vs 605 MB

- Ch1 `hybrid_architecture.md` (line 59): "576 MB across 48 GDN layers per device"
- Ch3 `recurrence_math.md` (line 169): "48 * 12 MB ~= 576 MB per device"
- Ch3 `conv1d_shift_register.md` (line 101): "~605 MB per device for 48 layers at 12.6 MB each"
- Ch6 `l1_state_design.md` (line 3): "The total state footprint of 605 MB"

The 576 MB figure comes from 48 * 12.0 MB (raw bytes). The 605 MB figure comes from 48 * 12.6 MB (tile-aligned bytes). Both calculations are internally valid, but using two different numbers for the same quantity across chapters is confusing.

**Fix:** Pick one consistent number with an explicit basis. Recommended: use 12.0 MB / 576 MB everywhere for raw state size, and note tile overhead parenthetically in the one place it matters (Ch7 `performance_summary.md` where the tile arithmetic is derived). Alternatively, use 12.6 MB / 605 MB everywhere and note it includes tile alignment.

## Issue 3: Per-layer state size stated as both "12 MB" and "12.6 MB"

Related to Issue 2 but appearing at the per-layer level:

- Ch3 `recurrence_math.md` (line 168): "12,288 KB = 12 MB (no tile padding -- both dimensions are exact multiples of 32)"
- Ch7 `performance_summary.md` (line 38): "12,582,912 bytes = 12.6 MB including tile alignment overhead"

Ch3 explicitly states there is no tile padding (128 and 128 are multiples of 32), yet Ch7 computes 12.6 MB "including tile alignment overhead" from the same dimensions. If both dimensions are exact multiples of tile size, there should be no alignment overhead, and both numbers should be 12.0 MB. The Ch7 calculation `384 * 16 * 2048 = 12,582,912` is correct arithmetic but equals 12.0 MiB, not 12.6 MB -- the discrepancy arises from mixing MiB and MB units without stating so.

**Fix:** In Ch7 `performance_summary.md`, note that `12,582,912 bytes = 12.0 MiB = 12.6 MB (decimal)` to resolve the apparent contradiction, or simply use 12 MB consistently since the tile padding claim is unfounded for these dimensions.

No other cross-chapter factual or consistency errors found. Navigation footer chains are complete and correct across all content files. All cross-chapter references resolve to valid files.
