# B Review — Chapter 1: Decode Path Architecture and Op Sequence — Pass 1

1. [index.md, ~line 57]: The summary table header states "The following table lists all **24 ops** in `_forward_decode_paged` in execution order," but the table contains **32 rows** (steps 4a/4b, 5a/5b/5c, 8a/8b, 12a/12b/12c/12d, and 16a/16b each contribute multiple rows). The step numbering 1–24 is internally consistent, but a developer counting ops from the header claim would expect 24 discrete operations and find 32. Fix: change "all 24 ops" to "all ops across 24 numbered steps" or "24 steps (32 sub-operations)" so the row count is not misread as the step count.

2. [index.md, ~line 63–64]: The summary table row for step 2 labels the output as "TILE, DRAM, col-sharded (reduce-scatter output)" and the row for step 3 labels the input as "TILE, DRAM, col-sharded" — but step 3 is an `all_gather` of `hidden_states`, not `query_states`. Step 3's input is the original col-sharded `hidden_states` (not the reduce-scatter output of step 2). These are two different tensors sharing the same layout descriptor in adjacent rows, which could cause a reader to mistake step 3 as operating on `query_states`. Fix: explicitly name the tensor in the "Input layout / memory" column for step 3 (e.g., "`hidden_states`, col-sharded") to distinguish it from the step-2 output.

3. [tensor_layouts.md, ~line 95]: The `kv_vol` formula annotation reads "= 4 elements **at batch=1**," implying the value changes with batch size. The derivation `(1 * B * 4 * 128) / 128 / B` simplifies to `4` for any value of B — the quantity is batch-invariant. A reader who tries to generalize the formula to larger batch sizes and expects a different value would compute the wrong shard height. Fix: remove the "at batch=1" qualifier, or note explicitly that the result is batch-independent: "= 4 elements (batch-invariant)".

4. [op_sequence.md, ~line 931 — Critical Path Summary table]: The collective communication row lists steps "3, 5a, 5b, 5c" and labels them "Four synchronous `all_gather` calls." This is correct for all-gathers only, but step 2 contains an additional collective (the `reduce_scatter_minimal_async` inside `TTNNLinearIColShardedWRowSharded`) that is not listed in this summary. The plan (plan.md, bottleneck_ranking.md) counts 5 total collectives including the reduce-scatter. A reader using this table as an inventory for profiling would instrument only 4 collectives and miss the reduce-scatter in the Q projection. Fix: add step 2 (reduce-scatter) as a row in the Critical Path Summary table, or add a footnote clarifying that step 2 contains an additional collective not counted here.

5. [op_sequence.md, ~line 233–235]: The note at the end of step 5c states "Combined with step 3, there are **four all-gather barriers** in the projection phase alone." This count is correct (steps 3, 5a, 5b, 5c). However, two lines earlier the note says the four all-gathers compare to "`TTNNQwen3FullAttention` (three all-gathers)." The plan's comparison table (plan.md) lists Qwen3 as having "3 (1 input + 3 post-proj; no separate Q gather because Q proj is replicated)" = 4, not 3. If Qwen3 uses replicated input for Q, it still performs one input all-gather + 3 post-projection all-gathers = 4. Fix: verify the Qwen3 all-gather count and correct the comparison, or qualify it as "Chapter 2 provides the authoritative comparison."

---

# B Review — Chapter 1: Decode Path Architecture and Op Sequence — Pass 2

**No feedback — chapter approved.**

---

# B Review — Chapter 1: Decode Path Architecture and Op Sequence — Pass 3

1. [index.md, summary table, step 11, "Touches host?" column]: The table marks step 11 as "**Yes (conditional)**", but `op_sequence.md` (lines 466–468) correctly states that `ttnn.from_torch` inside `get_cos_sin_for_decode` "is always called, uploading the `[1, B]` uint32 position index tensor to device on every decode step." The only conditional part is the additional `ttnn.to_torch` device-drain that occurs when `position_ids` arrives as a `ttnn.Tensor`. A reader who consults only the summary table to identify unavoidable host touches per decode step would classify step 11 as sometimes avoidable — but the from_torch PCIe upload is not avoidable by passing a different input type. The column entry should read "**Yes (always)**" (with a note that an additional to_torch device-drain is conditional), matching the correct statement already present in `op_sequence.md` and in the prose below the table (index.md lines 92–97).

---

## Agent A Change Log — Pass 1

- [item 1]: In `index.md` line 57, changed "all 24 ops" to "all ops across 24 numbered steps" so the table row count is not misread as the step count.
- [item 2]: In `index.md` step 3 row, changed the "Input layout / memory" column from "TILE, DRAM, col-sharded" to "`hidden_states`, col-sharded" to explicitly identify the tensor and distinguish it from the step-2 `query_states` reduce-scatter output.
- [item 3]: In `tensor_layouts.md` line 95, replaced "= 4 elements at batch=1" with "= 4 elements (batch-invariant)" to remove the misleading batch=1 qualifier and note that the result holds for any B.
- [item 4]: In `op_sequence.md` Critical Path Summary table, split the single "Collective communication" row into two rows — one for step 2 (reduce-scatter inside Q projection) and one for steps 3, 5a, 5b, 5c (four all-gathers) — so the reduce-scatter is no longer omitted from the collective inventory.
- [item 5]: In `op_sequence.md` step 5c note, replaced the claim "compares to `TTNNQwen3FullAttention` (three all-gathers)" with "Chapter 2 provides the authoritative comparison of this count against `TTNNQwen3FullAttention`". Verification against `qwen_attention.py` `_project_qkv` (lines 509–524, 582–583) confirmed the Qwen3 decode path performs at least 4 post-projection all-gathers (hidden_states input + q_proj + k_proj + v_proj) plus cos/sin gathers, making the original "three" count incorrect; deferral to Chapter 2 is the appropriate fix.

## Agent A Change Log — Pass 3

- item 1: Changed step 11 "Touches host?" from "Yes (conditional)" to "Yes (always)" with note about conditional device-drain.

---

# B Review — Chapter 1 — Pass 4

1. [op_sequence.md, line 462 — Step 11 "Host touch" field]: The field still reads "**Conditional**", directly contradicting the correction made to `index.md` in Pass 3 (which now correctly reads "**Yes (always)**"). The `ttnn.from_torch` inside `get_cos_sin_for_decode` executes unconditionally on every decode step; only the additional `ttnn.to_torch` device-drain is conditional. A reader implementing based on `op_sequence.md` alone would conclude the Step 11 host touch is avoidable by controlling the input type, and would not account for it as a hard per-step PCIe cost. Fix: change the Step 11 "Host touch" field in `op_sequence.md` to "**Yes (always)**; additionally a `ttnn.to_torch` device-drain occurs if `position_ids` arrives as a `ttnn.Tensor`", matching `index.md`.

---

## Agent A Change Log — Pass 4
- item 1: op_sequence.md Step 11 Host touch field changed from "Conditional" to "Yes (always)" with clarifying note.

---

# B Review — Chapter 1 — Pass 5

1. [index.md line 22 and op_sequence.md line 932 — "seven distinct memory layout transitions"]: Both files claim there are **seven** distinct `ttnn.to_memory_config` calls in this function. Counting from the summary table in `index.md` and from the code (lines 2656–2657, 2708–2709, 2718, 2727, 2753–2754, 2783): the actual calls are steps 8a, 8b, 12a, 12b, 12c, 12d, 16a, 16b, and 20 — **nine** distinct `to_memory_config` calls. A reader using the "seven" figure to inventory optimization targets in Chapter 3 would miss two calls (the pair at steps 12a/12b for cos/sin, or some other pair). Fix: change "seven" to "nine" in both locations, or if the count is intentionally excluding cos/sin (12a/12b) as non-tensor-state transitions, add an explicit note explaining what is excluded and why.

No further issues found. The op sequence structure, tensor shapes, line number references, host-touch characterizations, and the `kv_vol` batch-invariance correction from Pass 1 are all verified correct against `attention.py` lines 2610–2799.

## Agent A Change Log — Pass 5
- item 1: Changed `to_memory_config` call count from "seven" to "nine" in index.md and op_sequence.md.

---

# B Review — Chapter 1 — Pass 6

No feedback — chapter approved.
