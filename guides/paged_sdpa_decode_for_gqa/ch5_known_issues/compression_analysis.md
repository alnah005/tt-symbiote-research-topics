# Compression Analysis: Chapter 5 Known Issues and Correctness Pitfalls

## Summary
- Files analyzed: `index.md`, `issue_30362_pcc_failures.md`, `silent_shape_violations.md`, `gqa_workaround_history.md`, `program_cache_issues.md`
- Estimated current line count: 64 + 106 + 157 + 94 + 125 = 546
- Estimated post-compression line count: ~510
- Estimated reduction: ~7%

---

## CRUCIAL Suggestions

None.

No verbatim or near-verbatim multi-line blocks (code blocks, tables, derivations, or explanatory paragraphs) appear in two or more files. Each file covers a distinct failure category, and cross-file references to shared facts (shapes, formulas) are either single-line reminders in the index or contextually distinct uses of the same formula in different explanatory settings.

---

## MINOR Suggestions

### M1 — `kv_head_idx = q_head_idx // group_size` stated twice in distinct files

The formula appears in:
- `silent_shape_violations.md` line 31 — in the context of padding-induced GQA collapse showing that a wrong `group_size` produces wrong KV head lookups.
- `gqa_workaround_history.md` line 38 — in the context of native GQA kernel behavior after #12330.

Both uses are contextually necessary; the formula is a single line each time. No consolidation is required, but a cross-reference note ("see also `silent_shape_violations.md` §1") in one of the two locations would reduce reader confusion about why the same formula appears in two places.

### M2 — `index.md` "Tensor Shape Reminder" partially restates detail covered in sub-files

`index.md` lines 52–63 contain a consolidated shape block (Q, dense K/V, paged K/V, `page_table_tensor`, `paged_update_cache` input, GQA mapping, correct padding formula). The same information appears at greater depth in:
- `silent_shape_violations.md` sections 1–4 (padding formula, dense layout, paged layout, `page_table_tensor` spec, `paged_update_cache` shape).

This is mild redundancy appropriate for a chapter index (readers use the index block as a quick reference without opening sub-files). No removal is recommended, but the index block could carry a note: "Full specifications and failure modes are in `silent_shape_violations.md`."

### M3 — `paged_update_cache` input shape stated in two files

- `index.md` line 58: `paged_update_cache input: [b x nkv x 1 x dh]` (one line in the shape reminder).
- `silent_shape_violations.md` line 101: same shape stated in the prose of section 3.

This is the same single-line fact appearing in an index summary and in a detail file. This is acceptable chapter structure. No action required.

### M4 — `[max_num_blocks x nkv x block_size x dh]` stated in two files

- `index.md` line 56: in the shape reminder block.
- `silent_shape_violations.md` line 83: in the paged KV cache note under section 2.

Same as M3 — index summary vs. detail explanation. No action required.

---

## Load-Bearing Evidence

The following facts appear across the chapter and must not be removed from any file that currently contains them:

- Issue #30362 reproduction config: `b=1, nh=8, nkv=1, s=128K, block_size=128, grid=(8,4)`; CI strides 71 and 3001 step over affected positions; status open as of early 2026. (`issue_30362_pcc_failures.md`)
- Dense K/V current layout: `[b x nkv x S x dh]`; old (pre-#12330) layout: `[nkv x b x S x dh]`. (`silent_shape_violations.md`, `index.md`)
- Paged KV cache shape: `[max_num_blocks x nkv x block_size x dh]`. (`silent_shape_violations.md`, `index.md`)
- `paged_update_cache` input shape: `[b x nkv x 1 x dh]`. (`silent_shape_violations.md`, `index.md`)
- `repeat_interleave` left in after #12330 causes `effective_group_size = 1`, silently degrading to MQA. (`gqa_workaround_history.md`, `index.md`)
- `page_table_tensor` must be int32, row-major layout, on device. (`silent_shape_violations.md`, `index.md`)
- GQA mapping: `kv_head_idx = q_head_idx // group_size`. (`silent_shape_violations.md`, `gqa_workaround_history.md`)
- Correct padding invariant: `nkv_padded = nh_padded / original_group_size`. (`silent_shape_violations.md`, `index.md`)

---

## VERDICT: Crucial updates: no
