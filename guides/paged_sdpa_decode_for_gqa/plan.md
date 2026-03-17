# Plan: Paged SDPA Decode for GQA (Group Query Attention)

**Why this guide exists:** The Ling model generates incorrect text during decode. The root cause is suspected to be a mismatch between how the caller constructs tensors and how `ttnn.transformer.scaled_dot_product_attention_decode` (the paged Flash-Decode kernel) interprets them for GQA with 4 KV heads and 16 Q heads. This guide provides a systematic reference for ML engineers debugging such correctness failures.

---

## 1. Audience

**Target reader:** ML engineers debugging correctness failures in TTNN attention kernels who are integrating or porting transformer models (such as Ling) to Tenstorrent hardware.

**What they already know:**
- Standard scaled dot-product attention math (Q, K, V projections, softmax, output projection)
- Multi-head attention (MHA), multi-query attention (MQA), and grouped query attention (GQA) at the conceptual level
- Basic TTNN tensor operations: creating tensors, memory configs, layout types
- Python-level LLM inference loops (prefill/decode distinction)
- What a KV cache is and why it exists

**What they do NOT know (and this guide teaches):**
- How TTNN's Flash-Decode kernel differs from standard SDPA in tensor shape requirements
- The specific layout TTNN expects for GQA KV tensors (`[b x nkv x s x dh]`) vs. what naive ports produce
- How the paged KV cache block structure works in TTNN and what `page_table_tensor` maps
- The exact semantics of `cur_pos` / `cur_pos_tensor` and common misinterpretations
- Known correctness bugs in paged SDPA decode (issue #30362) and their workaround status
- A systematic debugging workflow for incorrect decode output

---

## 2. Chapter List

### Chapter 1: GQA and Paged Attention Fundamentals

**Directory:** `ch1_gqa_paged_fundamentals/`

**Description:** Establishes the conceptual vocabulary — what GQA is, why Q and KV head counts differ, what a paged KV cache is, and why these two ideas interact in a non-trivial way during autoregressive decoding.

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Prerequisite checklist and forward references to Ch2 API details

- `gqa_concept.md`
  - Definition of MHA, MQA, and GQA with the head-count relationship: `n_kv_heads * group_size = n_q_heads`
  - Concrete example: 16 Q heads / 4 KV heads → 4 groups of 4, each group shares one KV head
  - Why GQA reduces KV cache memory footprint by 4x in this configuration
  - Distinction between "broadcast" (expand K/V before attention) vs. "native group" implementations
  - How TTNN historically used `repeat_interleave` as a workaround before native GQA support landed in Flash-Decode (issue #12330)

- `paged_kv_cache_concept.md`
  - What paged KV cache is: fixed-size logical blocks mapped to physical storage via a page table
  - Supported block sizes in TTNN: 32, 64, 128 tokens per block
  - How a page table tensor `[b x max_num_blocks_per_seq]` maps each batch element's sequence positions to physical block indices
  - How paged storage changes the K/V tensor shape: from `[b x nkv x s x dh]` (contiguous) to `[max_num_blocks x nkv x block_size x dh]` (paged)
  - Why paged attention is essential for long-context decode (e.g., 128K tokens) and dynamic batching

- `gqa_plus_paging_interaction.md`
  - How paged storage and GQA interact: the page table indexes physical blocks agnostic of head count
  - Key invariant: `max_num_blocks = batch * max_num_blocks_per_seq` for the physical K/V tensor
  - Illustrative diagram: page table → physical block → KV head slice
  - Common conceptual mistake: applying GQA expansion (repeat_interleave) before writing to paged cache vs. keeping 4 KV heads in the cache

---

### Chapter 2: TTNN paged_sdpa_decode API

**Directory:** `ch2_ttnn_api/`

**Description:** Documents the complete TTNN `scaled_dot_product_attention_decode` function — every parameter, every tensor shape, paged vs. non-paged modes, and the constraints the kernel silently assumes.

**Files:**

- `index.md`
  - Chapter overview; links to official TTNN docs at `docs.tenstorrent.com`
  - Summary table of paged vs. non-paged call modes

- `function_signature.md`
  - Full Python signature (as of v0.55+):
    ```python
    ttnn.transformer.scaled_dot_product_attention_decode(
        input_tensor_q,   # [1 x b x nh x dh]
        input_tensor_k,   # [b x nkv x s x dh]  (non-paged)
                          # or [max_num_blocks x nkv x block_size x dh]  (paged)
        input_tensor_v,   # same shape as K
        *,
        is_causal=True,
        attn_mask=None,   # [b x 1 x s x s]
        cur_pos=None,     # List[int], length b
        cur_pos_tensor=None,  # ttnn.Tensor shape [b], row-major
        scale=None,
        program_config=None,  # SDPAProgramConfig
        compute_kernel_config=None,
        sliding_window_size=None,
        page_table_tensor=None,  # [b x max_num_blocks_per_seq]  (paged mode)
        memory_config=None,
    ) -> ttnn.Tensor  # [1 x b x nh x dh]
    ```
  - Explanation of every parameter: type, shape, default, and purpose
  - Output tensor shape: `[1 x b x pnh x dh]` where `pnh` is padded to nearest 32 for tile alignment
  - Note on `queue_id` parameter present in some versions

- `tensor_shape_reference.md`
  - Complete shape table for both non-paged and paged modes for Q, K, V, output
  - Distinction between `nh` (query head count) and `nkv` (KV head count) in the kernel's view
  - Layout requirements: TTNN tile layout vs. row-major; which tensors must be in which layout
  - Data type constraints: Q in BFloat16, K/V can be BFP8 (bfloat8_b) for memory savings; interaction with compute precision
  - Constraint: `num_cores >= batch * nkv` — what happens when this is violated, and the fix added in issue #12330
  - Page table tensor shape: `[b x max_num_blocks_per_seq]`, must be row-major int32 on device

- `sdpa_program_config.md`
  - `SDPAProgramConfig` fields: `compute_with_storage_grid_size`, `q_chunk_size`, `k_chunk_size`
  - How grid size affects parallelization: the kernel parallelizes over `b` and the K/V sequence dimension
  - How `q_chunk_size` and `k_chunk_size` interact with GQA group size
  - Example config for a 4-KV-head, 16-Q-head model on an 8x4 grid

---

### Chapter 3: GQA Tensor Layout Requirements

**Directory:** `ch3_gqa_tensor_layout/`

**Description:** Explains precisely how TTNN expects 4 KV heads to relate to 16 Q heads in tensor memory — the head axis positions, the grouping convention, and the difference between pre-expanded and native GQA layouts.

**Files:**

- `index.md`
  - Chapter overview
  - Diagram: conceptual head grouping (4 KV groups × 4 Q heads each = 16 Q heads total)

- `head_axis_conventions.md`
  - TTNN's tensor dimension ordering for decode: `[1 x b x nh x dh]` for Q; `[b x nkv x s x dh]` for K/V
  - Why Q has a leading `1` dimension (single decode step, batch padded into head dim)
  - How `nlp_create_qkv_heads_decode` produces these shapes and pads `nh` / `nkv` to multiples of 32
  - Cache layout change history: old `[nkv x b x S x d]` vs. current `[b x nkv x S x d]` (from issue #12330) — and why passing the old layout to a newer kernel silently produces wrong results

- `gqa_grouping_in_kernel.md`
  - How the Flash-Decode kernel computes the GQA group index: `kv_head_idx = q_head_idx // group_size`
  - The requirement that `nh % nkv == 0` — the kernel does NOT check this and will produce incorrect output (not an error) if violated
  - Effect of TTNN's 32-padding: if `nkv=4` is padded to `nkv_padded=32` but `nh_padded=32` as well, then effective `group_size=1`, collapsing GQA to MQA — a silent correctness bug
  - Correct approach: ensure `nh_padded / nkv_padded == intended_group_size`; if padding changes the ratio, the KV tensor must be pre-padded to match

- `paged_layout_for_gqa.md`
  - Paged KV tensor shape with GQA: `[max_num_blocks x nkv x block_size x dh]` — KV head count is the actual model `nkv`, not expanded
  - Physical block indexing: each block stores `block_size` token positions for all `nkv` KV heads simultaneously
  - `paged_update_cache` expected input shape: `[b x nkv x 1 x dh]` per decode step (one token per batch element)
  - Common mistake: writing `[b x nh x 1 x dh]` (expanded Q-head count) into the paged cache — causes the wrong KV data to be read back during attention

---

### Chapter 4: cur_pos Interpretation

**Directory:** `ch4_cur_pos_semantics/`

**Description:** Defines exactly what `cur_pos` means in the Flash-Decode kernel, distinguishes per-user vs. shared interpretations, explains the position `-1` skip behavior, and catalogs the most common misinterpretations that cause incorrect decode output.

**Files:**

- `index.md`
  - Chapter overview
  - Quick reference: `cur_pos[i]` = number of valid tokens in the KV cache for batch element `i`

- `cur_pos_definition.md`
  - Precise definition: `cur_pos[i]` is the length of the valid prefix in the KV cache for batch element `i`, i.e., how many tokens have been written at indices `[0, cur_pos[i])` — the kernel masks out attention beyond this point
  - Important: `cur_pos` is 0-indexed and refers to the number of *existing* KV tokens, not the index of the *token being generated*. After the first decode step, `cur_pos[i]=1`, not `cur_pos[i]=0`
  - Two passing modes: Python list `cur_pos` (CPU, re-compiled per unique value) vs. `cur_pos_tensor` (device tensor, avoids recompilation)
  - Special value `-1`: skips all computation for batch index `i`; output for that index is undefined — intended for padding in variable-length batches

- `per_user_vs_shared.md`
  - Per-user semantics: each batch element has an independent `cur_pos[i]`, enabling batches with different sequence lengths (e.g., continuous batching)
  - Common mistake 1: passing a scalar instead of a length-`b` list (e.g., `cur_pos=512` instead of `cur_pos=[512]*b`) — behavior is undefined or silently wrong depending on TTNN version
  - Common mistake 2: using the *token index being written* rather than the *cache length after writing*: off-by-one error that shifts the causal mask by one position, causing the current token to attend to a future position
  - Common mistake 3: sharing one `cur_pos` for all batch elements when sequences have diverged in length — leads to over-masking or under-masking

- `cur_pos_in_paged_mode.md`
  - How `cur_pos` interacts with paged KV: the kernel uses `cur_pos[i]` to determine which physical blocks to read via the page table, not just for masking
  - Relationship between `cur_pos[i]`, `block_size`, and active blocks: `num_active_blocks[i] = ceil(cur_pos[i] / block_size)`
  - What happens when `cur_pos[i]` falls on a block boundary vs. mid-block: partial block masking behavior
  - Issue #30362: sporadic PCC failures at certain `cur_pos` values in paged mode — likely related to block boundary arithmetic; workaround status as of early 2026

---

### Chapter 5: Known Issues and Correctness Pitfalls

**Directory:** `ch5_known_issues/`

**Description:** Catalogs documented bugs, shape constraint violations, silent failure modes, and GQA-specific pitfalls in TTNN paged attention as observed in the tt-metal repository.

**Files:**

- `index.md`
  - Chapter overview
  - Summary table: Issue ID → Description → Status → Workaround

- `issue_30362_pcc_failures.md`
  - Full description of GitHub issue #30362: paged SDPA decode fails PCC check at sporadic positions between 0–16K when using configuration `[b=1, nh=8, nkv=1, s=128K, block_size=128, grid=(8,4)]`
  - Why CI did not catch it: position increments of 71 or 3001 skip the affected positions
  - Suspected root cause: block boundary arithmetic error in the paged address computation
  - Impact on GQA: if the same boundary bug exists when `nkv > 1`, GQA models (4 KV heads) may hit wrong physical block addresses at specific `cur_pos` values
  - Current status: open as of early 2026; no upstream fix merged
  - Recommended workaround: test at dense position coverage (every position, or at known-bad boundaries like multiples of `block_size`)

- `silent_shape_violations.md`
  - Padding-induced GQA collapse: when `nkv_padded / nh_padded != nkv / nh`, the group size seen by the kernel is wrong — no error is raised
  - Cache layout mismatch: passing `[nkv x b x S x d]` (old layout) instead of `[b x nkv x S x d]` (current layout) causes wrong head indexing; symptoms look like all heads attending to the same K/V
  - `paged_update_cache` head count mismatch: writing `nh` heads instead of `nkv` heads to the paged cache corrupts KV data for GQA models
  - `page_table_tensor` dtype/layout errors: must be int32 row-major on device; passing wrong dtype causes silent read from wrong block addresses

- `gqa_workaround_history.md`
  - Pre-issue-#12330 era: `repeat_interleave` on K/V to match Q head count before calling SDPA decode; this worked but doubled/quadrupled KV memory usage per decode step
  - Issue #12330 Round 3: native GQA support added to Flash-Decode kernel, allowing `nkv < nh` directly; `repeat_interleave` workaround became unnecessary but some codebases still use it
  - Risk of mixing old and new: if `repeat_interleave` is applied AND the kernel natively handles GQA, the effective `nkv` seen by the kernel equals `nh`, changing the group size to 1 (MQA behavior) — a correctness bug with no error
  - Recommendation: audit your pipeline for any `repeat_interleave` or `expand` on K/V before the SDPA decode call

- `program_cache_issues.md`
  - Issue #21534: program cache miss / incorrect cache count when using paged attention with BFP8 K/V and BF16 Q; caused test assertion failures
  - Issue with `page_table_tensor` and program caching: early TTNN versions did not include page table shape in the cache key, causing reuse of a wrong compiled kernel when block count changed between calls
  - Fix: enable program caching of page table tensors (landed in TTNN as part of issue #12330 work)
  - Blackhole-specific: `paged_update_cache` hanging (issue #16674) on Blackhole hardware — unrelated to GQA but relevant to end-to-end decode stability

---

### Chapter 6: Debugging Incorrect Decode Output

**Directory:** `ch6_debugging/`

**Description:** Provides a systematic, step-by-step diagnostic process for engineers whose model produces incorrect text during decode — from shape validation through numerical comparison to root cause isolation.

**Files:**

- `index.md`
  - Chapter overview and "debug ladder" summary (ordered from cheapest to most expensive checks)
  - Prerequisites: a reference PyTorch implementation of the same attention that is known-correct

- `shape_validation_checklist.md`
  - Step-by-step shape audit for Q, K, V, output, page_table_tensor
  - How to extract and log TTNN tensor shapes: `tensor.shape`, `tensor.get_legacy_shape()`
  - Checking that `nh % nkv == 0` and that `nh_padded / nkv_padded == nh / nkv`
  - Verifying page table shape is `[b x max_num_blocks_per_seq]` and that `max_num_blocks_per_seq * block_size >= max_seq_len`
  - Verifying K/V paged shape: `[max_num_blocks x nkv x block_size x dh]`
  - Checklist item for cache layout: confirm `b` is axis 0 (not axis 1) in K/V tensor

- `cur_pos_validation.md`
  - How to validate `cur_pos` values: log them before every decode call and confirm they match expected sequence lengths
  - Off-by-one test: run two decode steps and verify `cur_pos[0]` increments by exactly 1 each step
  - Batch consistency test: for a single-sequence batch (`b=1`), verify `cur_pos` is a length-1 list, not a scalar
  - Paged mode test: verify that `floor(cur_pos[i] / block_size)` equals the number of fully-written blocks in the page table for batch element `i`
  - Testing position `-1` skip: if any batch element uses `-1`, confirm output for that element is explicitly ignored downstream

- `pcc_comparison_workflow.md`
  - Setting up a reference SDPA in PyTorch matching the exact Q/K/V values used in TTNN (including padding zeros, matching `scale`, matching `is_causal`)
  - Computing PCC (Pearson Correlation Coefficient) between TTNN output and PyTorch reference using `tt_lib.tensor.pearson_correlation_coefficient` or a manual NumPy equivalent
  - PCC thresholds: > 0.99 is good; < 0.98 suggests a systematic error; < 0.9 suggests a layout/shape mismatch
  - Strategy for narrowing position range: binary search over `cur_pos` values to find the first failing position (relevant to issue #30362 boundary bugs)
  - Isolating the KV cache write path: compare `paged_update_cache` output tensor against a reference to confirm KV values are correct before the attention computation

- `root_cause_isolation.md`
  - Flowchart: shape error → layout error → cur_pos error → paged block boundary error → kernel bug
  - How to disable paging (use contiguous K/V) to determine whether the bug is in the paging logic or the attention computation itself
  - How to test non-GQA (set `nkv = nh`) to determine whether the bug is in the GQA group-size logic
  - How to test with `cur_pos_tensor` vs. `cur_pos` list to rule out compilation/caching artifacts
  - Escalation path: if the bug reproduces with contiguous K/V, correct shapes, and `nkv = nh`, it is likely a kernel correctness bug — file an issue with the minimal reproducer configuration used in issue #30362

---

## 3. Conventions

### Terminology Table

| Term | Definition |
|------|-----------|
| `nh` | Number of query heads (e.g., 16 in Ling) |
| `nkv` | Number of KV heads (e.g., 4 in Ling) |
| `group_size` | `nh / nkv`; number of Q heads that share one KV head (e.g., 4) |
| `dh` | Head dimension (size of each individual attention head vector) |
| `b` | Batch size |
| `s` | Full KV cache sequence length (max tokens storable) |
| `block_size` | Number of tokens per paged KV block; one of 32, 64, 128 |
| `max_num_blocks` | Total physical KV blocks allocated: `b * max_num_blocks_per_seq` |
| `max_num_blocks_per_seq` | Maximum blocks per sequence: `ceil(s / block_size)` |
| `cur_pos[i]` | Length of valid KV prefix for batch element `i` (not the index being written — the count after writing) |
| `cur_pos_tensor` | Device-side row-major int32 tensor of shape `[b]` encoding the same info as `cur_pos` list |
| `page_table_tensor` | Device-side int32 tensor `[b x max_num_blocks_per_seq]` mapping logical blocks to physical block indices |
| `paged_update_cache` | TTNN op that writes one decode step's K/V into the correct physical blocks of the paged KV cache |
| Flash-Decode | TTNN's decode-time SDPA kernel; parallelizes over batch and KV sequence dimension |
| GQA | Grouped Query Attention: multiple Q heads share a single KV head; `nh > nkv` with `nh % nkv == 0` |
| MQA | Multi-Query Attention: all Q heads share a single KV head; `nkv = 1` |
| MHA | Multi-Head Attention: `nh == nkv` |
| PCC | Pearson Correlation Coefficient; used in TTNN tests to measure output correctness vs. reference |
| BFP8 | BFloat Point 8-bit format (`bfloat8_b`); used for KV cache to reduce memory |
| BF16 | BFloat16; standard TTNN compute dtype for Q and activations |
| Tile layout | TTNN's native 32×32 tile storage format; requires tensor dims to be multiples of 32 |
| Row-major | TTNN's non-tiled storage format; used for integer tensors like `page_table_tensor` and `cur_pos_tensor` |
| SDPAProgramConfig | TTNN config object specifying compute grid size and chunk sizes for the SDPA kernel |

### Notation Conventions

- Tensor shapes are written as `[dim0 x dim1 x ...]` using `x` as the separator.
- Python-level calls are shown as `ttnn.transformer.scaled_dot_product_attention_decode(...)`.
- Issue references use the GitHub format: `tenstorrent/tt-metal#NNNNN`.
- Version-specific behavior is tagged with `[v0.55+]` or `[pre-v0.50]` inline.
- "Kernel" refers specifically to the Metalium device kernel, not the Python op wrapper.
- "Caller" refers to Python-level model code that invokes the TTNN op.

### Formatting Rules

- Code blocks use Python syntax highlighting unless showing tensor shapes, which use plain text.
- Shape tables use Markdown tables with `Tensor | Shape | Layout | Dtype | Notes` columns.
- Warnings about silent failure modes (no error raised, wrong output) are prefixed with **[SILENT FAILURE]** in bold.
- Known-open bugs are prefixed with **[OPEN BUG #NNNNN]** and link to the GitHub issue.
- Fixed bugs are prefixed with **[FIXED in #NNNNN]**.
- Steps in checklists and workflows are numbered, not bulleted, to make them easy to reference in bug reports.

---

## 4. Cross-Chapter Dependencies

### Dependency Table

| Chapter | Depends On | Reason |
|---------|-----------|--------|
| Ch2: TTNN API | Ch1 | Tensor shape descriptions assume familiarity with GQA head counts and paged block structure |
| Ch3: GQA Layout | Ch1, Ch2 | Uses head-count vocabulary from Ch1; references API shapes defined in Ch2 |
| Ch4: cur_pos | Ch1, Ch2, Ch3 | `cur_pos` in paged mode (Ch4 §4) requires understanding of blocks (Ch1) and K/V shapes (Ch3) |
| Ch5: Known Issues | Ch2, Ch3, Ch4 | Each issue is explained in terms of API parameters, layout conventions, and cur_pos semantics |
| Ch6: Debugging | Ch2, Ch3, Ch4, Ch5 | The debug checklist and PCC workflow reference all prior chapters |

### Forward References to Flag

The following forward references appear in earlier chapters and should be explicitly linked to the target section:

- **Ch1 → Ch3:** `gqa_plus_paging_interaction.md` mentions that `nkv` in the paged cache is the unexpanded model value — readers should be flagged to see Ch3 (`paged_layout_for_gqa.md`) for the full shape specification.
- **Ch2 → Ch4:** `tensor_shape_reference.md` introduces `cur_pos` and `cur_pos_tensor` parameters without full semantics — readers should be flagged to Ch4 for the complete definition.
- **Ch2 → Ch3:** `sdpa_program_config.md` references `nkv` in the context of the `num_cores >= batch * nkv` constraint — readers need Ch3 to understand what `nkv` means after padding.
- **Ch3 → Ch5:** `gqa_grouping_in_kernel.md` mentions that violating `nh_padded / nkv_padded == nh / nkv` causes silent wrong output — readers should be flagged to Ch5 (`silent_shape_violations.md`) for the full catalog.
- **Ch4 → Ch5:** `cur_pos_in_paged_mode.md` references issue #30362 for block boundary PCC failures — full details are in Ch5 (`issue_30362_pcc_failures.md`).
- **Ch5 → Ch6:** `gqa_workaround_history.md` warns about mixing `repeat_interleave` with native GQA — the diagnostic for detecting this in a live codebase is in Ch6 (`root_cause_isolation.md`).
