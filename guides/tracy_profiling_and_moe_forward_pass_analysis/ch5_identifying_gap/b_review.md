# B Review — Chapter 5: Identifying the 16ms Gap — Pass 1

## Verdict

4 errors found.

---

### Error 1

**File:** `index.md`, line 57 (Hypothesis 3 — CCL collective latency)

**Stated:** "Latency scales with message size (tokens × d_model / num_chips)."

**Correct:** Per authoritative facts, CCL gap scales with `num_active_tokens × d_model` (total message size, linear). The `/num_chips` factor is a per-chip partition detail, not part of the scaling relationship. The chapter's own Method 3 code in `gap_attribution.md` and `common_gap_patterns.md` correctly computes a per-chip byte count, but the stated scaling law in the hypothesis description incorrectly embeds `/num_chips` into the formula that defines how the gap scales.

---

### Error 2

**File:** `gap_attribution.md`, line 175 vs. `common_gap_patterns.md`, line 234 — internal inconsistency on effective T3K ethernet bandwidth

**Stated in `gap_attribution.md`:** `effective_bandwidth_bytes_per_ns = 100e9 / 1e9  # 100 bytes/ns` (= 100 GB/s effective)

**Stated in `common_gap_patterns.md`:** `effective_bw_bytes_per_s = 7e9  # 7 GB/s`

**Correct:** These two values are irreconcilable. The chapter uses two contradictory effective bandwidth figures for the same T3K all-to-all operation, producing vastly different latency estimates (~0.15 ms vs. ~2.1 ms at seq_len=1024). One value must be authoritative; the chapter does not signal that the two code blocks are modeling different scenarios. The value used in `common_gap_patterns.md` (7 GB/s) is more consistent with known T3K ethernet link effective throughput for large all-to-all collectives. The 100 GB/s figure in `gap_attribution.md` should be corrected to match.

---

### Error 3

**File:** `common_gap_patterns.md`, line 217 — incorrect byte count for total CCL message volume

**Stated:** "For DeepSeek-V3 at seq_len=1024: `1024 × 8 × 7168 × 2 = 117,964,800 bytes ≈ 113 MB`."

**Correct:** 1024 × 8 × 7168 × 2 = 117,440,512 bytes ≈ 112 MB. The stated product (117,964,800) is wrong; the correct value is 117,440,512. The MB approximation should be ~112 MB, not 113 MB.

---

### Error 4

**File:** `gap_attribution.md`, line 175 comment and line 179 — latency estimate derived from the incorrect 100 GB/s bandwidth figure

**Stated:** "At seq_len=1024: ~0.15 ms (well below 16ms; see Pattern C for when this is larger)"

**Correct:** The 0.15 ms estimate is arithmetically consistent with the 100 GB/s assumption used in that code block, but that bandwidth assumption is contradicted by the 7 GB/s figure used in `common_gap_patterns.md` (Error 2 above). Using the 7 GB/s figure that Pattern C is built around, the same calculation yields ~2.1 ms — still well below 16ms, but the claim of ~0.15 ms propagates a false bandwidth value and misleads the reader about the expected CCL magnitude before they reach Pattern C.

---

## Agent A Change Log — B Feedback Pass 1

1. **`index.md` line 57:** Remove `/num_chips` from the scaling formula in the Hypothesis 3 description. Change to: "Latency scales with message size (num_active_tokens × d_model bytes)."

2. **`gap_attribution.md` Method 3 bandwidth and estimate:** Replace `effective_bandwidth_bytes_per_ns = 100e9 / 1e9  # 100 bytes/ns` with `effective_bw_bytes_per_s = 7e9  # 7 GB/s` (matching `common_gap_patterns.md`). Recompute and update the comment on line 179 to reflect the corrected estimate (~2.1 ms at seq_len=1024, not ~0.15 ms).

3. **`common_gap_patterns.md` line 217:** Correct the byte count from `117,964,800` to `117,440,512` and the MB approximation from `≈ 113 MB` to `≈ 112 MB`.

4. **`gap_attribution.md` line 179 comment:** After fixing Error 2 and recomputing, update the inline comment to state the corrected estimated latency (~2.1 ms) rather than ~0.15 ms.

## Agent A Change Log — B Feedback Pass 1
- index.md: Removed /num_chips from CCL scaling formula in Hypothesis 3 description
- gap_attribution.md: Changed bandwidth from 100 GB/s to 7 GB/s; updated latency comment from ~0.15 ms to ~2.1 ms
- common_gap_patterns.md: Corrected byte count from 117,964,800 to 117,440,512; updated MB approximation from 113 MB to 112 MB

# B Review — Chapter 5: Identifying the 16ms Gap — Pass 2

## Pass 1 Fix Verification

1. `index.md` line 57: Formula now reads "num_active_tokens × d_model bytes" — fix confirmed.
2. `gap_attribution.md` bandwidth: Now `effective_bw_bytes_per_s = 7e9  # 7 GB/s` — fix confirmed.
3. `gap_attribution.md` line 179 comment: Now reads `# At seq_len=1024: ~2.1 ms` — fix confirmed.
4. `common_gap_patterns.md` line 217: Now reads `117,440,512 bytes ≈ 112 MB` — fix confirmed.

## Remaining Error

### Error 1

**File:** `common_gap_patterns.md`, Pattern C code block, comment on the `bytes_per_chip` line

**Stated:** `# = 117_964_800 / 8 = 14_745_600 bytes ≈ 14 MB`

**Correct:** The numerator `117_964_800` is the old incorrect total byte count that was fixed in Pass 1 (corrected to `117,440,512`). The stale comment was not updated. The correct comment should read `# = 117_440_512 / 8 = 14_680_064 bytes ≈ 14 MB`. The code itself (`(seq_len * top_k * d_model * bytes_per_element) / num_chips`) computes the right value, but the inline comment still propagates the old wrong number.

## Agent A Change Log — B Feedback Pass 2
- common_gap_patterns.md: Fixed stale comment in Pattern C bytes_per_chip line from "117_964_800 / 8 = 14_745_600" to "117_440_512 / 8 = 14_680_064"

# B Review — Chapter 5: Identifying the 16ms Gap — Pass 3

## Pass 2 Fix Verification

`common_gap_patterns.md` Pattern C code block, `bytes_per_chip` comment: now reads `# = 117_440_512 / 8 = 14_680_064 bytes ≈ 14 MB` — fix confirmed.

## Verdict

No feedback — chapter approved.
