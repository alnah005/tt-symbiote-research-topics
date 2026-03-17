# Chapter 5: Known Issues and Correctness Pitfalls

## Prerequisites

- **Chapter 1** — GQA and paged attention concepts; `nh`, `nkv`, `group_size`, `block_size`.
- **Chapter 2** — `ttnn.transformer.scaled_dot_product_attention_decode` API; `page_table_tensor` shape and dtype requirements.
- **Chapter 3** — Tensor layout conventions; padding to tile multiples; `nkv_padded / nh_padded` invariant.
- **Chapter 4** — `cur_pos` semantics; block selection arithmetic; Issue #30362 overview.

---

## Learning Objectives

By the end of this chapter you should be able to:

1. Identify the specific `cur_pos` values at which Issue #30362 produces PCC failures and explain why CI did not catch them.
2. List the four silent failure modes that produce wrong output without raising an error.
3. Describe the pre-#12330 `repeat_interleave` workaround, why it was necessary, and why applying it after #12330 is a correctness bug.
4. Explain the program cache issues tracked in #21534 and #12330, and how `page_table_tensor` shape affects cache correctness.
5. Apply the recommended workarounds for each issue category.

---

## Issue Summary Table

| Issue ID | Description | Status | Workaround File |
|---|---|---|---|
| #30362 | Sporadic PCC failures at certain `cur_pos` values (0–16K range); block boundary arithmetic | Open (early 2026) | `issue_30362_pcc_failures.md` |
| Silent: padding collapse | `nkv_padded / nh_padded != nkv / nh` changes effective `group_size`; no error raised | No upstream fix | `silent_shape_violations.md` |
| Silent: cache layout mismatch | Old dense K/V layout `[nkv x b x S x dh]` instead of current `[b x nkv x S x dh]`; wrong head indexing (dense cache only; paged cache shape is `[max_num_blocks x nkv x block_size x dh]`) | No upstream fix | `silent_shape_violations.md` |
| Silent: cache head count | `paged_update_cache` writing `nh` heads instead of `nkv` heads corrupts GQA KV data | No upstream fix | `silent_shape_violations.md` |
| Silent: page table dtype | `page_table_tensor` not int32 row-major on device; silent wrong-block reads | No upstream fix | `silent_shape_violations.md` |
| #12330 / pre-#12330 | `repeat_interleave` on K/V still applied after native GQA landed; collapses to MQA | Fixed in kernel; user code risk | `gqa_workaround_history.md` |
| #21534 | Program cache miss / wrong cache count with BFP8 K/V and BF16 Q | Fixed | `program_cache_issues.md` |
| Page table shape in cache key | Early TTNN reused compiled kernel when block count changed between calls | Fixed in #12330 work | `program_cache_issues.md` |
| #16674 | `paged_update_cache` hangs on Blackhole hardware | Open | `program_cache_issues.md` |

---

## File Map

| File | Content |
|---|---|
| [`issue_30362_pcc_failures.md`](./issue_30362_pcc_failures.md) | Full description of #30362: affected positions, CI gap, GQA impact, workaround |
| [`silent_shape_violations.md`](./silent_shape_violations.md) | Four silent failure modes: padding collapse, layout mismatch, head count, page table dtype |
| [`gqa_workaround_history.md`](./gqa_workaround_history.md) | `repeat_interleave` history; risk of mixing old workaround with native GQA kernel |
| [`program_cache_issues.md`](./program_cache_issues.md) | Issues #21534, #16674; page table shape cache key; BFP8/BF16 cache miss |

---

## Tensor Shape Reminder

```
Q:               [1  x b  x nh  x dh]
K / V (dense):   [b  x nkv x s   x dh]
K / V (paged):   [max_num_blocks x nkv x block_size x dh]
page_table_tensor: [b x max_num_blocks_per_seq]  — int32, row-major, on device
paged_update_cache input: [b x nkv x 1 x dh]
```

GQA mapping: `kv_head_idx = q_head_idx // group_size`

Correct padding: `nkv_padded = nh_padded / original_group_size`
