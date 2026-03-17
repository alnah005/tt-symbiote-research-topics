# B Review — Chapter 3: GQA Tensor Layout Requirements — Pass 1

## Verdict

One factual error found. All other claims verified correct against the supplied ground truth.

---

## Error 1 — `gqa_grouping_in_kernel.md`, Comparison Table, Row 3: Wrong nkv_padded and effective_group_size values

**Location:** `gqa_grouping_in_kernel.md`, section "Comparison Table", third data row.

**Claim in the file:**

| Scenario | pnh | nkv_padded | effective_group_size | Correct? |
|----------|-----|------------|---------------------|----------|
| nh=32, nkv=4, padded correctly | 32 | 8 | 4 | Yes |

**Why this is wrong:**

For `nh=32`, `nkv=4`:

- `original_group_size = nh // nkv = 32 // 4 = 8`
- `pnh = 32` (already a multiple of 32, no change)
- `nkv_padded = pnh // original_group_size = 32 // 8 = 4` — not 8

The table states `nkv_padded = 8`, but the correct derivation yields `nkv_padded = 4`.

Consequently, `effective_group_size = pnh // nkv_padded = 32 // 4 = 8`, which matches the original `group_size = 8`. The table instead states `effective_group_size = 4`, which is the group size from the first two rows (nh=16, nkv=4) — it appears the row was copied from row 1 without recomputing for nh=32.

**Correct row:**

| Scenario | pnh | nkv_padded | effective_group_size | Correct? |
|----------|-----|------------|---------------------|----------|
| nh=32, nkv=4, padded correctly | 32 | 4 | 8 | Yes |

Both values in the row (nkv_padded and effective_group_size) are wrong. The effective_group_size of 4 shown in the table does not match the original group_size of 8, which would mean the "padded correctly" scenario is actually shown producing an incorrect group_size — directly contradicting the "Yes" in the Correct? column.

---

## All Other Claims Verified Correct

The following facts were checked against the ground truth and found to be accurately stated across all four files:

- Q decode tensor shape: `[1 x b x nh x dh]` — correct in `head_axis_conventions.md`.
- K/V non-paged shape: `[b x nkv x s x dh]` — correct in `head_axis_conventions.md`.
- Paged K/V shape: `[max_num_blocks x nkv x block_size x dh]` — correct in `paged_layout_for_gqa.md`.
- `paged_update_cache` input shape: `[b x nkv x 1 x dh]` — correct in `paged_layout_for_gqa.md`.
- GQA kernel formula: `kv_head_idx = q_head_idx // group_size` — correct in `gqa_grouping_in_kernel.md`.
- Kernel does NOT check `nh % nkv == 0`; violation produces wrong output silently — correctly stated.
- `nkv_padded = pnh // original_group_size` padding rule — correctly stated and demonstrated.
- Independent padding failure example (nkv=4→32, pnh=32, effective_group_size=1, MHA collapse) — correct.
- Correct padding example (nkv=4, nh=16 → pnh=32, nkv_padded=8, effective_group_size=4) — correct.
- Old KV cache layout `[nkv x b x S x dh]` vs. new `[b x nkv x S x dh]`, issue #12330 — correctly stated.
- Each paged block stores `block_size` token positions for all `nkv` KV heads simultaneously — correct.
- Writing `[b x nh x 1 x dh]` into paged cache identified as wrong — correct.

## Agent A Change Log — B Feedback Pass 1
- gqa_grouping_in_kernel.md: Fixed comparison table row for nh=32, nkv=4 — corrected nkv_padded=4 (was 8), effective_group_size=8 (was 4)

---

# B Review — Chapter 3: GQA Tensor Layout Requirements — Pass 2

Pass 1 fix verified. No feedback — chapter approved.
