## Pass 1

1. **`window_size_parameter.md`, lines 139–151 — Sink-token mask diagram is wrong for multiple rows (wrong answer / wrong implementation)**

   The diagram for T=8, w=3, k_sink=1 uses the formula
   `A_sink(t) = {0,...,k_sink−1} ∪ {max(k_sink, t−w+1), ..., t}`.
   Applying the formula correctly:

   | Query pos t | Global sink | Local window = {max(1, t−2), ..., t} | Union |
   |-------------|-------------|---------------------------------------|-------|
   | 3 | {0} | {max(1,1)=1, 2, 3} | {0,1,2,3} |
   | 4 | {0} | {max(1,2)=2, 3, 4} | {0,2,3,4} |
   | 5 | {0} | {max(1,3)=3, 4, 5} | {0,3,4,5} |
   | 6 | {0} | {max(1,4)=4, 5, 6} | {0,4,5,6} |
   | 7 | {0} | {max(1,5)=5, 6, 7} | {0,5,6,7} |

   The diagram as written shows only 2 local-window entries per row from pos 3 onward
   (e.g., row 3: `1 . 1 1 . . . .` = {0,2,3}; row 4: `1 . . 1 1 . . .` = {0,3,4}).
   Every row from pos 3 onward is missing one attended position (the entry just after
   the sink). The diagram must be corrected to:

   ```text
   Query pos 3  │  1 1 1 1 . . . .    (sink at 0 + window [1,3])
   Query pos 4  │  1 . 1 1 1 . . .    (sink at 0 + window [2,4])
   Query pos 5  │  1 . . 1 1 1 . .    (sink at 0 + window [3,5])
   Query pos 6  │  1 . . . 1 1 1 .    (sink at 0 + window [4,6])
   Query pos 7  │  1 . . . . 1 1 1    (sink at 0 + window [5,7])
   ```

2. **`full_vs_windowed_attention.md`, line 187 — KV cache size byte formula omits H and B (wrong numerical answer)**

   The "KV cache size (bytes)" row in the decode complexity table gives concrete byte
   expressions: `T · d · 2 · dtype_bytes` (full) and `w · d · 2 · dtype_bytes`
   (windowed). Unlike the rows above it, this row is presented as a concrete formula,
   not an O() expression where B and H are explicitly stated to be omitted. The
   complete formula (consistent with `plan.md` line 74) must include all dimensions:

   - Full attention: `B · H · T · d · 2 · dtype_bytes`
   - Windowed attention: `B · H · w · d · 2 · dtype_bytes`

   A reader computing cache memory requirements directly from this table would
   underestimate actual DRAM consumption by a factor of B·H.

3. **`window_size_parameter.md`, line 153 — Threshold for non-contiguous attended set is stated incorrectly**

   The text states: "The attended set is no longer contiguous once position t exceeds
   `w + k_sink − 1`."

   For k_sink=1, w=3: the threshold given is 3+1−1 = 3. But at t=3 the attended set
   is {0,1,2,3} (corrected per issue 1 above), which IS contiguous. The set first
   becomes non-contiguous at t=4: {0,2,3,4}. The correct threshold is
   `t > w + k_sink − 1`, i.e., `t ≥ w + k_sink`, or equivalently `t ≥ w + 1` for
   k_sink=1. The condition should read: "once position t strictly exceeds
   `w + k_sink − 1`" (i.e., `t ≥ w + k_sink`), not "exceeds" in the inclusive sense.
   Fix: change "once position t exceeds `w + k_sink − 1`" to "once position t exceeds
   `w + k_sink − 1` (i.e., t ≥ w + k_sink)".

   Note: this issue is contingent on the diagram fix in item 1; once the diagram is
   corrected the prose threshold should be reconciled with the corrected pattern.

## Pass 2

**No feedback — chapter approved.**

All three Pass 1 issues were verified as correctly fixed:

1. Sink-token mask diagram rows for t=3–7 now match `A_sink(t) = {0,...,k_sink−1} ∪ {max(k_sink, t−w+1), ..., t}` with w=3, k_sink=1.
2. KV cache size formula includes B and H: `2 · B · H · T · d · dtype_bytes` / `2 · B · H · w · d · dtype_bytes`.
3. Non-contiguity threshold prose correctly states the gap first appears at t = w + k_sink (t=4 for w=3, k_sink=1).

No new correctness issues were found. All formulas, mask diagrams, complexity claims, and receptive-field derivations checked out. Navigation footers are present and correct on both content files. All links in `index.md` are clickable markdown links.

---

## Pass 3

**No feedback — chapter approved.**

All formulas, diagrams, complexity claims, and receptive-field derivations independently verified:

- Receptive-field formula `RF(L) = w + (L−1)·(w−1) = 1 + L·(w−1)` expands correctly.
- Prefill saving factors (8× for w=4096, T=32768; 4× for w=8192) are arithmetically correct.
- KV cache size formula includes B and H: `2 · B · H · T · d · dtype_bytes` / `2 · B · H · w · d · dtype_bytes`.
- Sink-token mask diagram rows t=3–7 match `A_sink(t) = {0,...,k_sink−1} ∪ {max(k_sink, t−w+1), ..., t}` with w=3, k_sink=1.
- Non-contiguity threshold (t = w + k_sink = 4) is consistent with diagram and prose.
- Steady-state sink-extended KV cache size of k_sink + w entries is correct (no double-counting since local window starts at max(k_sink, t−w+1)).
- Navigation footers present and correct on both content files.
- All links in `index.md` are clickable markdown links.

---

## Change Log (Pass 1)

All three issues identified in Pass 1 were applied to the Chapter 1 files on 2026-03-27.

1. **`window_size_parameter.md` — sink-token mask diagram corrected.** Rows for query positions 3–7 were redrawn to reflect the correct attended sets under the formula `A_sink(t) = {0,...,k_sink−1} ∪ {max(k_sink, t−w+1), ..., t}` with w=3, k_sink=1. Row t=3 now shows {0,1,2,3} (was {0,2,3}); rows t=4 through t=7 each gained one additional attended position (the first position of the local window, which had been omitted).

2. **`full_vs_windowed_attention.md` — KV cache size formula updated to include B and H.** The "KV cache size (bytes)" row in the decode complexity table was changed from `T · d · 2 · dtype_bytes` / `w · d · 2 · dtype_bytes` to `2 · B · H · T · d · dtype_bytes` / `2 · B · H · w · d · dtype_bytes`.

3. **`window_size_parameter.md` — non-contiguity threshold prose corrected.** The sentence "The attended set is no longer contiguous once position t exceeds w + k_sink − 1" was replaced with a precise statement: the non-contiguous pattern first appears at t = w + k_sink (t=4 for w=3, k_sink=1), with t=3 explicitly noted as still contiguous.
