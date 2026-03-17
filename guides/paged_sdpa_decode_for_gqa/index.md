# Guide: Paged SDPA Decode for GQA on Tenstorrent Hardware

This guide is a systematic reference for ML engineers debugging correctness failures in
`ttnn.transformer.scaled_dot_product_attention_decode` when using Grouped Query Attention
(GQA) with a paged KV cache. Its immediate motivation is the Ling model, which produces
incorrect text during decode due to a suspected mismatch between how the caller constructs
tensors and how TTNN's Flash-Decode kernel interprets them for a 4 KV head / 16 Q head
GQA configuration. The guide covers the conceptual model (GQA + paged blocks), the
complete API contract, tensor layout requirements, `cur_pos` semantics, a catalog of
known open bugs and silent failure modes, and a step-by-step debugging workflow that
moves from cheap shape checks to numerical PCC comparison to root-cause isolation.

---

## Audience

**Who should read this guide:**

ML engineers integrating or porting transformer models (especially LLMs with GQA) to
Tenstorrent hardware, and engineers triaging incorrect decode output — wrong tokens,
degraded text quality, or low PCC against a PyTorch reference — when using
`ttnn.transformer.scaled_dot_product_attention_decode`.

**What you are assumed to know:**

- Standard scaled dot-product attention math: Q, K, V projections, softmax, output projection.
- MHA, MQA, and GQA at the conceptual level.
- What a KV cache is and why it exists in autoregressive decoding.
- Basic TTNN tensor operations: creating tensors, reading `.shape`, applying memory configs.
- Python-level LLM inference loops (prefill vs. decode distinction).

**What this guide teaches:**

- How TTNN's Flash-Decode kernel differs from standard SDPA in tensor shape requirements.
- The exact layout TTNN expects for GQA KV tensors (`[b x nkv x s x dh]`) vs. what naive
  ports commonly produce.
- How the paged KV cache block structure works in TTNN and what `page_table_tensor` maps.
- The precise semantics of `cur_pos` / `cur_pos_tensor` and the off-by-one traps.
- Known open correctness bugs (Issue #30362) and silent failure modes with no upstream fix.
- A structured debugging workflow (shape audit → `cur_pos` validation → PCC comparison →
  root-cause isolation).

---

## Chapter Navigation

| Chapter | Description | Key files |
|---------|-------------|-----------|
| [Ch 1 — GQA Paged Fundamentals](ch1_gqa_paged_fundamentals/index.md) | Establishes vocabulary: GQA head-count invariant (`nh = nkv * group_size`), paged KV block model, `page_table_tensor` semantics, and why GQA expansion must never be applied before writing to a paged cache. | [`gqa_concept.md`](ch1_gqa_paged_fundamentals/gqa_concept.md), [`paged_kv_cache_concept.md`](ch1_gqa_paged_fundamentals/paged_kv_cache_concept.md), [`gqa_plus_paging_interaction.md`](ch1_gqa_paged_fundamentals/gqa_plus_paging_interaction.md) |
| [Ch 2 — TTNN API](ch2_ttnn_api/index.md) | Documents every parameter of `ttnn.transformer.scaled_dot_product_attention_decode`: tensor shapes for paged and non-paged modes, layout and dtype constraints, `SDPAProgramConfig` fields, and the `num_cores >= b * nkv` grid-size constraint. | [`function_signature.md`](ch2_ttnn_api/function_signature.md), [`tensor_shape_reference.md`](ch2_ttnn_api/tensor_shape_reference.md), [`sdpa_program_config.md`](ch2_ttnn_api/sdpa_program_config.md) |
| [Ch 3 — GQA Tensor Layout](ch3_gqa_tensor_layout/index.md) | Specifies exactly how TTNN expects Q, K, V in memory for GQA decode, including the 32-padding silent bug that collapses GQA to MQA when padding changes the Q/KV head ratio, and the old vs. current dense KV cache axis ordering. | [`head_axis_conventions.md`](ch3_gqa_tensor_layout/head_axis_conventions.md), [`gqa_grouping_in_kernel.md`](ch3_gqa_tensor_layout/gqa_grouping_in_kernel.md), [`paged_layout_for_gqa.md`](ch3_gqa_tensor_layout/paged_layout_for_gqa.md) |
| [Ch 4 — `cur_pos` Semantics](ch4_cur_pos_semantics/index.md) | Defines `cur_pos[i]` precisely (pre-write current context length; equals the 0-indexed position of the next token to be written), covers the Python-list vs. device-tensor trade-off, documents three common off-by-one and shape mistakes, and explains how `cur_pos` drives both causal masking and physical block selection in paged mode. | [`cur_pos_definition.md`](ch4_cur_pos_semantics/cur_pos_definition.md), [`per_user_vs_shared.md`](ch4_cur_pos_semantics/per_user_vs_shared.md), [`cur_pos_in_paged_mode.md`](ch4_cur_pos_semantics/cur_pos_in_paged_mode.md) |
| [Ch 5 — Known Issues](ch5_known_issues/index.md) | Catalogs all documented bugs and silent failure modes: Issue #30362 (sporadic PCC failures at block boundaries, open as of early 2026), four silent shape violations, the `repeat_interleave` / native GQA mixing bug, and program cache issues #21534 and #16674. | [`issue_30362_pcc_failures.md`](ch5_known_issues/issue_30362_pcc_failures.md), [`silent_shape_violations.md`](ch5_known_issues/silent_shape_violations.md), [`gqa_workaround_history.md`](ch5_known_issues/gqa_workaround_history.md), [`program_cache_issues.md`](ch5_known_issues/program_cache_issues.md) |
| [Ch 6 — Debugging](ch6_debugging/index.md) | Provides a four-step debug ladder (shape audit → `cur_pos` validation → PCC comparison → root-cause isolation) with a PyTorch reference implementation, PCC thresholds, binary search strategy for Issue #30362 boundary failures, and an escalation flowchart. | [`shape_validation_checklist.md`](ch6_debugging/shape_validation_checklist.md), [`cur_pos_validation.md`](ch6_debugging/cur_pos_validation.md), [`pcc_comparison_workflow.md`](ch6_debugging/pcc_comparison_workflow.md), [`root_cause_isolation.md`](ch6_debugging/root_cause_isolation.md) |

---

## Quick-Start Paths

**If you are debugging PCC failures or incorrect generated tokens:**
Start at Ch5 (`ch5_known_issues/index.md`) to check whether your failure matches a known
issue. Then follow the debug ladder in Ch6. If you hit unfamiliar terms, backfill from
Ch3 (tensor layout) and Ch4 (`cur_pos`).

**If you are setting up paged GQA decode for the first time:**
Read Ch1 → Ch2 → Ch3 in order. This establishes all vocabulary before you touch the API
and before the shape requirements make sense.

**If your shapes look correct but output is still wrong:**
Go directly to Ch4 (`cur_pos` semantics). Off-by-one errors in `cur_pos` produce silent
wrong output that passes shape checks. After Ch4, run the PCC comparison workflow in Ch6.

**If you suspect a padding/tile-alignment bug:**
Read Ch3 (`gqa_grouping_in_kernel.md`) for the padding-collapse silent bug, then
Ch5 (`silent_shape_violations.md`) for the full catalog. Use the shape audit checklist
in Ch6 Step 1 to confirm.

**If you are auditing an existing codebase for the `repeat_interleave` / native GQA mixing bug:**
Read Ch5 (`gqa_workaround_history.md`) for the history, then Ch6
(`root_cause_isolation.md`) for the diagnostic procedure.

**If you hit a hang on Blackhole hardware:**
See Ch5 (`program_cache_issues.md`) for Issue #16674 (`paged_update_cache` hang on
Blackhole). This is unrelated to GQA but affects end-to-end decode stability.

---

## Key Authoritative Facts

### Tensor shapes (paged mode, GQA)

| Tensor | Shape | Layout | Dtype |
|--------|-------|--------|-------|
| Q | `[1 x b x nh x dh]` | Tile | BF16 |
| K / V | `[max_num_blocks x nkv x block_size x dh]` | Tile | BFP8 or BF16 |
| `paged_update_cache` input | `[b x nkv x 1 x dh]` | Tile | matches K/V |
| `page_table_tensor` | `[b x max_num_blocks_per_seq]` | Row-major | int32, on device |
| `cur_pos_tensor` | `[b]` | Row-major | int32, on device |
| Output | `[1 x b x pnh x dh]` | Tile | BF16 |

`pnh` = `nh` padded to nearest 32 for tile alignment.

### GQA constraints

- `nh % nkv == 0` is required. The kernel does **not** check this — violation produces wrong
  output silently.
- Correct padding invariant: `nkv_padded = nh_padded / original_group_size`. When padding
  changes this ratio, the kernel sees the wrong `group_size` and attention is silently wrong.
- KV head count in the paged cache is always `nkv` (the unexpanded model value), never `nh`.

### `cur_pos` definition

`cur_pos[i]` is the current context length — the number of valid tokens already in the KV
cache for batch element `i` before the current decode write. It is the 0-indexed position
of the next token to be written: for a sequence with 5 tokens already cached, `cur_pos[i]
= 5` and the next write goes at position 5. It is an input to `paged_update_cache`, not a
result of it. After the first decode step, `cur_pos[i] = 1`. Passing a scalar instead of a
length-`b` list produces undefined or silently wrong behavior.

### Supported block sizes

32, 64, or 128 tokens per block. `max_num_blocks_per_seq = ceil(s / block_size)`.

### Grid-size constraint

`num_cores >= b * nkv`. Violating this produces wrong or zero output for some batch
elements or KV heads without raising an error.

### Known open bug: Issue #30362 (early 2026 status)

Sporadic PCC failures at certain `cur_pos` values in the range 0–16K when using paged
SDPA decode. CI did not catch this because position increments of 71 or 3001 skip the
affected positions. Root cause is suspected block-boundary arithmetic in paged address
computation. No upstream fix merged. Workaround: test at dense position coverage or at
multiples of `block_size`.

---

## Cross-Chapter Dependency Note

Chapters are cumulative. Each chapter builds on vocabulary and shape definitions
established by prior chapters:

| Chapter | Depends on |
|---------|-----------|
| Ch2: TTNN API | Ch1 — tensor shape descriptions assume GQA and paging vocabulary |
| Ch3: GQA Layout | Ch1, Ch2 — uses head-count vocabulary and API shapes |
| Ch4: `cur_pos` | Ch1, Ch2, Ch3 — paged-mode block selection requires blocks (Ch1) and K/V shapes (Ch3) |
| Ch5: Known Issues | Ch2, Ch3, Ch4 — each issue is explained in terms of API params, layout, and `cur_pos` |
| Ch6: Debugging | Ch2, Ch3, Ch4, Ch5 — the checklist and PCC workflow reference all prior chapters |

**Key forward references flagged in earlier chapters:**

- Ch1 → Ch3: `gqa_plus_paging_interaction.md` notes that `nkv` in the paged cache is the
  unexpanded model value — full shape specification is in Ch3 (`paged_layout_for_gqa.md`).
- Ch2 → Ch4: `tensor_shape_reference.md` introduces `cur_pos` without full semantics —
  complete definition is in Ch4.
- Ch3 → Ch5: `gqa_grouping_in_kernel.md` notes the padding-collapse silent bug — full
  catalog is in Ch5 (`silent_shape_violations.md`).
- Ch4 → Ch5: `cur_pos_in_paged_mode.md` references Issue #30362 — full details are in Ch5.
- Ch5 → Ch6: `gqa_workaround_history.md` warns about `repeat_interleave` mixing — the
  diagnostic for detecting it in a live codebase is in Ch6 (`root_cause_isolation.md`).

---

## Terminology Quick Reference

| Term | Definition |
|------|-----------|
| `nh` | Number of query heads (e.g., 16 in Ling) |
| `nkv` | Number of KV heads (e.g., 4 in Ling) |
| `group_size` | `nh / nkv`; Q heads sharing one KV head |
| `dh` | Head dimension |
| `b` | Batch size |
| `s` | Full KV cache sequence capacity (max storable tokens) |
| `block_size` | Tokens per paged KV block; one of 32, 64, 128 |
| `max_num_blocks_per_seq` | `ceil(s / block_size)` |
| `max_num_blocks` | `b * max_num_blocks_per_seq` (total physical blocks) |
| `cur_pos[i]` | Current context length (0-indexed next-write position); for a seq with N tokens cached, cur_pos[i]=N |
| `cur_pos_tensor` | Device-side `[b]` int32 row-major encoding of `cur_pos` |
| `page_table_tensor` | `[b x max_num_blocks_per_seq]` int32 row-major on device |
| `paged_update_cache` | TTNN op writing one decode step's K/V into physical blocks |
| Flash-Decode | TTNN's decode-time SDPA kernel; parallelizes over batch × KV sequence |
| PCC | Pearson Correlation Coefficient; > 0.99 is good, < 0.98 indicates systematic error |
| BFP8 | `bfloat8_b`; 8-bit format used for KV cache to reduce memory |
| BF16 | BFloat16; standard compute dtype for Q and activations |
| Tile layout | TTNN native 32×32 tile storage; requires dims to be multiples of 32 |
| Row-major | Non-tiled storage; required for `page_table_tensor` and `cur_pos_tensor` |
| SDPAProgramConfig | TTNN config specifying grid size and chunk sizes for SDPA kernel |

Full conventions, notation rules, and formatting standards are in [`plan.md`](plan.md#3-conventions).
