# B Review — Chapter 1: GQA and Paged Attention Fundamentals — Pass 1

1. **`gqa_concept.md`, Section 5 Warning (~line 121)** — The warning claims that calling the native-GQA kernel with a `repeat_interleave`-expanded K/V tensor (i.e., `nkv_apparent = nh` heads) "silently collapses GQA to MHA behavior using only the first copy of each KV head" and produces wrong results. This is factually incorrect. When the kernel infers `group_size_apparent = nh / nh = 1`, it maps Q head `i` to KV head `i` in the expanded tensor. Because `repeat_interleave` lays out the expanded tensor as `[kv0, kv0, …, kv1, kv1, …]` (each original KV head repeated `group_size` times consecutively), Q head `i` addresses expanded KV head `i`, which is a copy of original KV head `i // group_size`. That is the correct GQA mapping. The result is numerically identical to native GQA — just wasteful. The reader who follows this warning's logic will believe the approach is silently wrong when it is silently correct. Fix: replace the claim of incorrect output with the accurate statement that the approach produces correct results but wastes bandwidth and memory (and defeats paged-cache size savings), making it a performance bug rather than a correctness bug.

2. **`gqa_plus_paging_interaction.md`, Section 5 Scenario B (~line 121)** — The same factual error appears here. Scenario B states: "When the Flash-Decode kernel is called with `nkv=16` and `nh=16`, it computes `group_size = 1` and collapses to MHA behavior … if the model's weights were trained for GQA grouping, the attention patterns are semantically different from what the model expects." As explained above, `group_size = 1` applied to a `repeat_interleave`-expanded cache recovers the correct GQA grouping numerically. Calling it "semantically different from what the model expects" misleads implementers into thinking they must avoid this path for correctness reasons, when the only real cost is wasted memory and bandwidth. Fix: state that Scenario B produces correct attention scores but stores `nh` KV heads instead of `nkv`, discarding the memory reduction that motivated GQA.

3. **`gqa_concept.md`, line 23** — The baseline description of MHA states `nkv / nh = 1.0 (no KV reduction)`. The reduction factor used throughout the rest of the chapter (table on line 151, Section 4 table) is expressed as `nh / nkv` (how many times smaller the GQA cache is versus MHA). Using `nkv / nh` here inverts the framing without explanation. A reader computing the reduction factor will get `1 / group_size` instead of `group_size` if they follow this formula. Fix: state the baseline as `nh / nkv = 1` (reduction factor = 1×, i.e., no reduction), consistent with the `group_size`× convention used everywhere else.

4. **`paged_kv_cache_concept.md`, Section 6, `paged_update_cache` input shape (~line 148)** — The file specifies the per-step input as `[b x nkv x 1 x dh]`. This represents one new token per batch element for all KV heads. However, the file never states what the `1` dimension corresponds to (it is the sequence/token dimension, value = 1 for a single decode step). A downstream implementer who uses this shape for prefill (where multiple tokens are processed) or who confuses the `1` with a batch-inner dimension will produce an incorrectly shaped write. Fix: add a brief note that the `1` is the single-token decode dimension and that `paged_update_cache` in prefill mode takes `s_chunk` tokens, not `1`.

No further correctness issues found beyond the four items above.

# B Review — Chapter 1: GQA and Paged Attention Fundamentals — Pass 2

All four Pass 1 fixes are confirmed present and correct:
- `gqa_concept.md` line 121: repeat_interleave warning now correctly labels the issue as a performance bug, not a correctness bug.
- `gqa_concept.md` line 23: MHA baseline now reads `nh / nkv = 1` (reduction factor = 1×, no reduction), consistent with the rest of the chapter.
- `gqa_plus_paging_interaction.md` Scenario B: now states the attention output is numerically correct, framing the issue as performance/memory waste only.
- `paged_kv_cache_concept.md` Section 6: now includes a note explaining the `1` as the single-token decode dimension.

**Remaining correctness issue:**

1. **`gqa_plus_paging_interaction.md`, Section 7 (~line 172)** — The padding example concludes: "Effective `group_size = 32 / 32 = 1` — kernel sees MQA, not GQA." This label is wrong. `group_size = 1` means each query head maps to a distinct KV head (1:1), which is MHA behavior, not MQA. MQA is the opposite extreme: `group_size = nh`, meaning all query heads share a single KV head. A reader who internalizes "group_size=1 means MQA" will have an inverted mental model of the MHA/MQA/GQA spectrum and will misdiagnose or mislabel padding-induced regressions. Fix: replace "kernel sees MQA, not GQA" with "kernel sees MHA behavior (group_size=1: each Q head gets its own KV head), not GQA grouping."

# B Review — Chapter 1: GQA and Paged Attention Fundamentals — Pass 3

Pass 2 fix confirmed: `gqa_plus_paging_interaction.md` Section 7 (~line 172) now reads "kernel sees MHA behavior (group_size=1: each Q head gets its own KV head), not GQA grouping." This is correct and replaces the previously wrong "MQA" label.

No further correctness issues found across all four chapter files (`index.md`, `gqa_concept.md`, `paged_kv_cache_concept.md`, `gqa_plus_paging_interaction.md`).

**No feedback — chapter approved.**
