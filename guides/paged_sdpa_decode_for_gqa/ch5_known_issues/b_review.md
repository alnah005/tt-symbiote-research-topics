# B Review — Chapter 5: Known Issues and Correctness Pitfalls — Pass 1

## Finding 1 — Inconsistent "old cache layout" notation between `index.md` and `silent_shape_violations.md`

`index.md` (Issue Summary Table, row "Silent: cache layout mismatch") describes the old layout as `[nkv x b x S x d]` — dense-cache notation with a batch dimension `b` and a flat sequence dimension `S`.

`silent_shape_violations.md` Section 2 describes the old paged KV cache layout as `[nkv x max_num_blocks x block_size x dh]` — paged notation with no `b` dimension and `max_num_blocks` in position 2.

These two descriptions use different dimension sets and cannot both be correct for the same layout. The validated fact is that the old dense cache was `[nkv x b x S x d]` and the current paged cache is `[max_num_blocks x nkv x block_size x dh]`. The detail file should label its "old" paged layout as `[nkv x max_num_blocks x block_size x dh]` (which it does), but the index should not represent that same concept using dense-cache notation `[nkv x b x S x d]` as though they describe the same thing. One of the two representations is misapplied to the wrong cache type.

**Location**: `index.md` line 30 vs. `silent_shape_violations.md` lines 59–62.

---

No additional factual errors found. All other validated facts — Issue #30362 reproduction config (b=1, nh=8, nkv=1, s=128K, block_size=128, grid=(8,4)), CI stride values (71 and 3001), padding formula (`nkv_padded = nh_padded / original_group_size`), `paged_update_cache` input shape (`[b x nkv x 1 x dh]`), `page_table_tensor` requirements (int32, row-major, on device), Q/K/V tensor shapes, GQA head mapping formula, issue #12330 native GQA addition, post-#12330 `repeat_interleave` collapse to MQA, issue #21534 BFP8/BF16 cache miss, issue #16674 Blackhole hang, and Wormhole B0 DRAM controller count (6, not 8) with 12 GDDR6 banks — are all stated correctly across the chapter files.

## Agent A Change Log — B Feedback Pass 1
- silent_shape_violations.md: Clarified cache layout mismatch section — the [nkv x b x S x d] → [b x nkv x S x d] change applies to dense K/V; paged KV cache has its own shape [max_num_blocks x nkv x block_size x dh]
- index.md: Made issue table notation consistent with silent_shape_violations.md fix

---

# B Review — Chapter 5: Known Issues — Pass 2

## Pass 1 Fix Verification

`silent_shape_violations.md` Section 2 now correctly scopes the layout change to the dense K/V path only. The old layout `[nkv x b x S x dh]` and current layout `[b x nkv x S x dh]` are clearly labeled as dense-cache conventions. A dedicated "Paged KV cache note" block explicitly states the paged shape is always `[max_num_blocks x nkv x block_size x dh]` and is unaffected by the dense-layout change. Fix is correct.

`index.md` Issue Summary Table row "Silent: cache layout mismatch" now reads the old dense layout as `[nkv x b x S x dh]`, the current as `[b x nkv x S x dh]`, and includes a parenthetical `(dense cache only; paged cache shape is [max_num_blocks x nkv x block_size x dh])`. This is consistent with the detail file and with the key facts. Fix is correct.

## Scan of Remaining Files

`gqa_workaround_history.md`: No errors. Pre-#12330 workaround, post-#12330 native GQA behavior, and the `repeat_interleave`-left-in → `group_size=1` → silent MQA path are all stated correctly. Dense K/V shapes in the summary table match `[b x nkv x s x dh]` as required.

`issue_30362_pcc_failures.md`: No errors. Status correctly marked open as of early 2026. Reproduction config (b=1, nh=8, nkv=1, s=128K, block_size=128, grid=(8,4)) and CI stride values (71 and 3001) match key facts exactly.

`program_cache_issues.md`: No errors. Issue #21534 (BFP8/BF16 cache miss) correctly marked fixed. Issue #16674 (Blackhole hang) correctly marked open. Wormhole B0 DRAM controller count correctly stated as 6 (not 8) with 12 GDDR6 banks.

Pass 1 fix verified. No feedback — chapter approved.
