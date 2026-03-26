# B Review — Chapter 7: Profiling and Roadmap — Pass 1

## Finding 1 — Wrong line number for Bailing Q projection (`comparison_to_other_implementations.md`, side-by-side table)

The table entry for `TTNNBailingMoEAttention` Q projection type reads:

> `TTNNLinearIColShardedWRowSharded` (line 2363 of `attention.py`)

The actual assignment is at line **2374** of `attention.py`:

```python
new_attn.q_proj = TTNNLinearIColShardedWRowSharded.from_torch(q_linear)  # line 2374
new_attn.k_proj = TTNNLinearIReplicatedWColSharded.from_torch(k_linear)  # line 2375
```

Line 2363 is inside an `nn.Linear` constructor call and has nothing to do with `TTNNLinearIColShardedWRowSharded`. The K/V reference to line 2375 is correct and unchanged.

---

## Finding 2 — `_to_replicated` line range wrongly attributed to `TTNNQwen3FullAttention` (`comparison_to_other_implementations.md`, "`_to_replicated` Elimination" section)

The prose states:

> `TTNNQwen3FullAttention` and `TTNNGlm4MoeLiteAttention` still call `_to_replicated` after the all-gather (lines 1895–1897 of `attention.py`).

Lines 1895–1897 of `attention.py` are inside `TTNNGlm4MoeLiteAttention._forward_decode_paged` (class defined at line 1493; `_forward_decode_paged` begins at line 1828). `TTNNQwen3FullAttention` is defined in `qwen_attention.py`, not `attention.py`. Its `_to_replicated` calls are at lines **766–768 of `qwen_attention.py`** (inside `TTNNQwen3FullAttention._forward_decode_paged`).

The claim that lines 1895–1897 of `attention.py` cover both Qwen3 and GLM4 is incorrect: those lines belong to GLM4 only.

---

## Finding 3 — Qwen3 collective count "4 … no separate Q gather" is inaccurate (`comparison_to_other_implementations.md`, side-by-side table and "Q Projection Strategy" section)

The table states Qwen3 has:

> 4 (1 hidden all-gather + 3 post-proj all-gathers; no separate Q gather)

In `qwen_attention.py` `_project_qkv`, the Q projection output is explicitly all-gathered at line 522:

```python
q_proj_output = self._maybe_all_gather(q_proj_output)   # line 522 — Q all-gather
key_states    = self._maybe_all_gather(key_states)       # line 523
value_states  = self._maybe_all_gather(value_states)     # line 524
```

There IS a Q all-gather in Qwen3. What is absent is a *reduce-scatter* inside Q projection (because `TTNNLinearIReplicatedWColSharded` does not perform one). The phrase "no separate Q gather" is factually wrong. The Q projection strategy section in `bottleneck_ranking.md` is internally contradictory — it correctly says switching Bailing to `TTNNLinearIReplicatedWColSharded` "eliminates the reduce-scatter in Q projection and the subsequent `_maybe_all_gather(query_states)` call," yet the comparison table implies Qwen3 has no Q all-gather.

Additionally, `_project_qkv` conditionally all-gathers cos and sin at lines 582–583 on every distributed decode step. If those are counted, Qwen3 has 6 gathers (1 hidden + 3 post-proj + 2 cos/sin), not 4. The basis for the "4" figure (excluding cos/sin) is not stated.

---

## Finding 4 — `TTNNBailingMoEAttention._to_replicated` exists in source; prose implies it was removed

The table row for `_to_replicated host round-trip` says for Bailing:

> **Not called** (eliminated; line 2642–2646 comment)

This is correct for `_forward_decode_paged`. However, `_to_replicated` is still defined in `TTNNBailingMoEAttention` at lines 2288–2310 of `attention.py`. The "Key Technique" section recommends that Qwen3 and GLM4 "adopt the reshape-based approach," which implies they should remove `_to_replicated` calls. Any reader checking Bailing's source will find the method present, which will appear to contradict the chapter's claim. The chapter should clarify that `_to_replicated` exists but is bypassed in the decode path, rather than suggesting it has been deleted.

---

## Finding 5 — Qwen3 Q projection line reference is the class variable definition, not the q_proj assignment (`comparison_to_other_implementations.md`, side-by-side table and `bottleneck_ranking.md`)

Both files cite Qwen3 Q projection as "line 250 of `qwen_attention.py`." Line 250 in the source is:

```python
LinearClsIn = TTNNLinearIReplicatedWColSharded if distributed else TTNNLinear
```

This is the class-selector variable assignment; the actual `q_proj` attribute assignment is at line **255**:

```python
new_attn.q_proj = LinearClsIn.from_torch(torch_attn.q_proj)  # line 255
```

A reader looking up line 250 finds only a conditional variable, not the q_proj assignment itself. The cite should be line 255 (or the range 250–255 if the intent is to show both the class selection and the assignment).

## Agent A Change Log — Pass 1

- item 1: `comparison_to_other_implementations.md` — corrected Q projection line from 2363 to 2374. The `new_attn.q_proj = TTNNLinearIColShardedWRowSharded.from_torch(...)` assignment is at line 2374.
- item 2: `comparison_to_other_implementations.md` — corrected `_to_replicated` attribution. Qwen3's calls are at lines 766–768 of `qwen_attention.py` (not "line 1895–1897 equivalent"). GLM4's calls are at lines 1895–1897 of `attention.py`.
- item 3: `comparison_to_other_implementations.md` and `bottleneck_ranking.md` — corrected Qwen3 collective count. Qwen3 does have a Q all-gather (line 522 of `qwen_attention.py`). The correct count for Qwen3 is 6 (1 hidden + 3 post-proj all-gathers including Q + 2 cos/sin all-gathers; no reduce-scatter). Updated "no separate Q gather" to "no reduce-scatter." Updated bottleneck_ranking.md Rank 1 to clarify: reduce-scatter eliminated, Q all-gather at line 2631 remains.
- item 4: `comparison_to_other_implementations.md` — changed "Not called (eliminated)" to "Not called in decode path (bypassed; lines 2642–2646 reshape approach)" to reflect that `_to_replicated` still exists in the code (lines 2288–2310) but is not invoked.
- item 5: `comparison_to_other_implementations.md` and `bottleneck_ranking.md` — corrected Qwen3 Q projection assignment from line 250 to line 255. Line 250 is the `LinearClsIn` class-selector variable; line 255 is the actual `new_attn.q_proj = LinearClsIn.from_torch(...)` call.

---

# B Review — Chapter 7: Profiling and Roadmap — Pass 2

## Finding 1 — Wrong line numbers for step 8a/8b `to_memory_config` calls (`profiling_methodology.md`, "Measuring `to_memory_config` transitions" section)

The document states:

> **Step 8a (line 2655)**: `ttnn.to_memory_config(query_states, L1_MEMORY_CONFIG)` — 131,072 bytes at B=32
> **Step 8b (line 2656)**: `ttnn.to_memory_config(key_states, L1_MEMORY_CONFIG)` — 32,768 bytes at B=32

In the actual source, line 2655 is the comment `# Move to L1 for QK norm (reshape doesn't work on sharded tensors)`. The assignments are:

- Line 2656: `query_states = ttnn.to_memory_config(query_states, ttnn.L1_MEMORY_CONFIG)` (step 8a)
- Line 2657: `key_states = ttnn.to_memory_config(key_states, ttnn.L1_MEMORY_CONFIG)` (step 8b)

Both cited line numbers are off by one. Step 8a is at line 2656 and step 8b is at line 2657.

---

## Finding 2 — `bottleneck_ranking.md` Rank 5 wrongly attributes direct `ttnn.rms_norm` calls to GLM4

Rank 5 states:

> `TTNNGlm4MoeLiteAttention` uses direct `ttnn.rms_norm(tensor, weight=self._weight, epsilon=...)` calls instead, eliminating module dispatch overhead.

The source does not support this. `TTNNGlm4MoeLiteAttention.from_torch` (line 1540 of `attention.py`) assigns:

```python
NormCls = TTNNDistributedRMSNorm if distributed else TTNNRMSNorm
```

and uses `NormCls.from_torch(...)` for all its layer norms (lines 1544, 1550). These are module objects, not direct `ttnn.rms_norm` calls. The `comparison_to_other_implementations.md` table correctly states GLM4 uses "`TTNNRMSNorm` or `TTNNDistributedRMSNorm` (configurable via `distributed` flag)." The claim in Rank 5 directly contradicts both the source and the comparison table.

The implementation that uses direct `ttnn.rms_norm` calls with pre-loaded weights is `TTNNQwen3FullAttention` (lines 556 and 565 of `qwen_attention.py`), not GLM4. Rank 5's Proposed Fix should reference Qwen3, not GLM4, as the validated precedent for the direct `ttnn.rms_norm` pattern.

---

## Agent A Change Log — Pass 1 (C compression)

- C1: Removed byte-volume specifics from `index.md` "Relationship to Prior Chapters" table. Stripped "528 KB/step; 91% avoidable" and "163,840 bytes/step" from the table — these numbers belong in `bottleneck_ranking.md`, not the index navigation table.
- C2: Collapsed the "Key Technique: Q Projection Strategy" prose section in `comparison_to_other_implementations.md` from 2 paragraphs (~8 lines) to a 2-sentence pointer to `bottleneck_ranking.md` Rank 1.
- C3: Collapsed the "Key Technique: SDPA Compute Config" prose section in `comparison_to_other_implementations.md` from 2 paragraphs to one paragraph, removing restatement of Chapter 6 content and keeping only the unique DeepSeek V3 attribution detail.

---

# B Review — Chapter 7: Profiling and Roadmap — Pass 3

## Finding 1 — Lead sentence for collective ops cites wrong set of lines and wrong API name (`profiling_methodology.md`, "Measuring all_gather collectives" section)

**File:** `profiling_methodology.md`

**Claim:** The sentence at line 46 reads:

> Each synchronous `ttnn.all_gather` call in the decode path (lines 2626, 2631, 2632, 2633 of `attention.py`) can be timed with the same bracket pattern.

**What is wrong:** Two errors, both verifiable from the known facts:

1. Lines 2631, 2632, and 2633 are `_maybe_all_gather` calls, not `ttnn.all_gather` calls. The known facts explicitly distinguish: "attention.py line 2626: ttnn.all_gather call" and "attention.py lines 2631, 2632, 2633: _maybe_all_gather calls." Labeling all four lines as `ttnn.all_gather` is factually incorrect.

2. The parenthetical "(lines 2626, 2631, 2632, 2633)" omits line 2624, which the numbered list immediately below correctly identifies as the first of the five collective positions. The known facts confirm "attention.py line 2624: q_proj(hidden_states) call" as a distinct collective position. The lead sentence lists 4 lines but then says "The five collective positions," with item 1 being line 2624 — an internal contradiction that also misrepresents the source.

**Suggested fix:** Change the sentence to:

> The five collective positions in the decode path (lines 2624, 2626, 2631, 2632, 2633 of `attention.py`) can each be timed with the bracket pattern. Note that line 2624 contains an implicit reduce-scatter inside `q_proj`, line 2626 is an explicit `ttnn.all_gather`, and lines 2631–2633 call `_maybe_all_gather`:

## APPROVED (no further issues found)

All other verifiable claims in the three submitted files are consistent with the known source facts:

- `profiling_methodology.md`: step 8a at line 2656 (`query_states` to L1) and step 8b at line 2657 (`key_states` to L1) are confirmed by known facts. The nine `to_memory_config` steps (8a, 8b, 12a–12d, 16a, 16b, 20) are confirmed by Chapter 1 `op_sequence.md`. The three host-device round-trip locations are not contradicted by available sources.
- `bottleneck_ranking.md` Rank 5: The attribution of direct `ttnn.rms_norm` calls to Qwen3 at lines 556 and 565 of `qwen_attention.py` is confirmed by known facts. No factual errors remain in the submitted Rank 5 text.
- `comparison_to_other_implementations.md`: All line number citations for verified facts (line 2374, 2375, 255, 766–768, 341–346, 2288–2310, 2642–2646) are correct per known facts.
