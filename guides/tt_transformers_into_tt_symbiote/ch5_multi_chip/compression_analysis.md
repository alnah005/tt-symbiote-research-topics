# Compression Analysis: Chapter 5 — Multi-Chip Parallelism — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~687 lines
- Estimated post-compression line count: ~654 lines
- Estimated reduction: ~5%

---

## CRUCIAL Suggestions

### index.md ~lines 41–47
**Issue:** The "Why Multi-Chip Parallelism Matters" section explains column-parallel and row-parallel tensor parallelism with definitions that are then repeated in detail in `tt_transformers_parallelism.md` Section 4 (lines 136–168) and again summarized in `integration_strategy.md` Section 1.1 table. The index prose adds no constraint or numeric value not found in the dedicated file — it merely previews content that follows.
**Suggestion:** Remove lines 41–47 (the full "Why Multi-Chip Parallelism Matters" section, including its heading and two bullet points). The horizontal rule delimiter on line 48 should also be removed. The chapter navigation table immediately following provides the forward pointer; readers who need the definition will find it in `tt_transformers_parallelism.md`.

### tt_transformers_parallelism.md ~lines 196–211 (Section 6 Galaxy-Specific Topology Differences table)
**Issue:** The table in Section 6 is a near-verbatim re-aggregation of information already scattered across earlier sections of the same file:
- `w1_dims`, `w2_dims` are documented at lines 145–153 (Section 4, MLP patterns).
- `wqkv` and `wo` shard dims are documented at lines 160–167 (Section 4, Attention patterns).
- The `tt_all_reduce` implementation difference (Path A vs Path B/C) is documented at lines 107–132 (Section 3).
- MLP inter-projection CCL (reduce-scatter/all-gather around element-wise multiply) is documented with line references in Section 7 (lines 214–229).
- `slice_mat`, decode output gather axis, and KV cache rows appear also in Section 7.

The table provides no additional facts beyond what those sections document; it is purely a re-statement.
**Suggestion:** Remove Section 6 entirely (lines 196–212, including the heading, table, and trailing `---` delimiter). Add a one-sentence cross-reference in Section 3's Galaxy path (Path B) note pointing readers to Section 7 for a full enumeration of Galaxy-specific call sites. This preserves all load-bearing values while eliminating the duplication.

### symbiote_distributed_primitives.md ~line 57
**Issue:** The sentence "The `reduce_scatter_minimal_async` call is the key difference from the single-device `TTNNLinear.forward`: it performs the partial-result reduction across the T3K cluster axis after the matmul." restates what the four preceding numbered steps (1–5 in `forward`) already show explicitly in code. The code block makes the reduce-scatter unmistakable; the prose sentence adds nothing the code does not show.
**Suggestion:** Remove that trailing explanatory sentence (line 57).

---

## MINOR Suggestions

### index.md ~lines 21–38 (topology sub-sections N300/T3K/TG prose)
**Issue:** The prose descriptions under "N300 (2-chip)", "T3K (8-chip)", and "TG / Galaxy (32-chip)" partly re-explain the table on lines 13–19 and partly forward-reference `tt_transformers_parallelism.md` details (e.g., which branch of `tt_all_reduce` is taken, the `reduce_scatter_minimal_async` semantics). The N300 and T3K sub-sections in particular hedge ("so the only collective axis with more than one device is axis 1") in ways that duplicate the mesh shape already visible in the table.
**Suggestion:** Trim the N300 and T3K paragraphs to a single sentence each noting the code-path branch, removing re-statements of mesh shape and chip count.

### integration_strategy.md ~lines 47–49 (Section 2.2 recommendation sentence)
**Issue:** "The topology should become a parameter of the Symbiote module (or read from `CCLManagerConfig.topology`) rather than a compile-time constant." The `CCLManagerConfig.topology` field is documented in `symbiote_distributed_primitives.md` Section 6 — this sentence hedges with an alternative that is already implied by that documented field.
**Suggestion:** Remove the parenthetical "(or read from `CCLManagerConfig.topology`)" to simplify.

### integration_strategy.md ~lines 54–58 (Section 2.3 last two sentences)
**Issue:** "This is equivalent when `self.device` is the correct mesh device, but TT Transformers' approach is more explicit and does not depend on `self.device` being set before weight loading. The Symbiote approach works as long as `move_weights_to_device_impl` is always called after device assignment — which the `TTNNModule` lifecycle guarantees." This hedging note defends the status quo with reasoning that is only relevant if changing the approach. The conclusion is that no change is needed, which the section heading already signals.
**Suggestion:** Remove the last two sentences; retain only the statement of equivalence.

---

## Load-Bearing Evidence

- `index.md` line ~15–19: topology table with exact chip counts, mesh shapes, Ethernet link counts, and TT Transformers identifiers — load-bearing because these numeric values do not appear together in any other single location and are the source-of-truth reference for `link_dict`.
- `index.md` lines ~33–37: Galaxy-specific behavioral differences list (weight sharding dim swap, all-reduce path, MLP intermediate CCL, `slice_mat`, link counts 4/3) — load-bearing as the index-level summary of cross-cutting Galaxy behavior; however these are all documented individually in `tt_transformers_parallelism.md` so the index bullets are redundant only in part. The link counts "4 links on axis 0 / 3 links on axis 1" appear in `tt_transformers_parallelism.md` Section 1 table and are not at risk.
- `tt_transformers_parallelism.md` line ~111: "Callers on these topologies therefore receive a scatter-reduced (not fully replicated) result" — load-bearing behavioral constraint; not inferable from the function name alone.
- `tt_transformers_parallelism.md` line ~132: "`use_composite=True` when `self.dim == 8192`" — load-bearing specific numeric trigger condition.
- `tt_transformers_parallelism.md` lines ~220–221: "hard codes to 2 links… to avoid any performance regressions" — load-bearing operational note about a hard-coded value and its rationale.
- `tt_transformers_parallelism.md` lines ~225–228: four specific conditions for the `w2` all-reduce parameters (`dim`, `sharded`, `use_composite`, `dtype`) — load-bearing parameter values.
- `symbiote_distributed_primitives.md` line ~59: "`@run_on_devices(DeviceArch.T3K)` … raises `RuntimeError`" — load-bearing behavioral constraint (arch enforcement).
- `symbiote_distributed_primitives.md` lines ~131–134: `ShardTensor2dMesh` with `dims=(None, 2)` and reshape to `[1, 1, dim//32, 32]` — load-bearing non-obvious weight layout detail.
- `symbiote_distributed_primitives.md` lines ~195–199: `logical_shape_for_batch_channel_sharding` formula — load-bearing because the shape reconstruction formula is non-obvious.
- `symbiote_distributed_primitives.md` line ~213–214: `DistributedConfig.__post_init__` auto-creates `TT_CCL` and `DistributedTensorConfig` when `num_devices > 1` — load-bearing behavioral default.
- `symbiote_distributed_primitives.md` line ~248: "No TG / Galaxy variant exists yet in Symbiote" — singleton fact; only stated once across the files.
- `integration_strategy.md` lines ~75–81: Prefetcher CCL coupling (must pass `subdevice_id` through all CCL calls) — load-bearing non-obvious constraint not documented in the primitive files.
- `integration_strategy.md` line ~99: DRAM-sharded vs interleaved DRAM distinction — load-bearing because it explains kernel selection differences (correctness vs performance distinction).

---

## VERDICT
- Crucial updates: yes

## Change Log — Pass 1 CRUCIAL fixes applied

- Change 1: Removed `index.md` lines 41–48 — the "Why Multi-Chip Parallelism Matters" section (heading, two bullet-point definitions of column-parallel and row-parallel, and trailing `---` delimiter). Content is fully covered in `tt_transformers_parallelism.md` Section 4 and `integration_strategy.md` Section 1.1.
- Change 2: Removed `tt_transformers_parallelism.md` Section 6 "Galaxy-Specific Topology Differences" (heading, table, and trailing `---` delimiter, lines 196–212). All table rows duplicate facts already documented in Sections 3, 4, and 7 of the same file. Added a one-sentence forward-reference to Section 7 at the end of the Path B description in Section 3.
- Change 3: Removed the trailing explanatory sentence from `symbiote_distributed_primitives.md` Section 1 forward method description (the sentence beginning "The `reduce_scatter_minimal_async` call is the key difference…"). It paraphrases the numbered code steps that precede it.

---

# Compression Analysis: Chapter 5 — Multi-Chip Parallelism — Pass 2

## Summary
- Current line count: ~657 lines (after Pass 1: index.md 52, tt_transformers_parallelism.md 222, symbiote_distributed_primitives.md 256, integration_strategy.md 127)
- Estimated post-compression: ~657 lines (no CRUCIAL fixes to apply)
- Estimated reduction: 0%

## Pass 1 Fix Verification

1. **index.md "Why Multi-Chip Parallelism Matters" section removed** — confirmed. index.md is 52 lines; the file jumps directly from the TG/Galaxy prose block to the Chapter Navigation table. No "Why Multi-Chip Parallelism Matters" heading or bullet points are present.

2. **tt_transformers_parallelism.md Section 6 "Galaxy-Specific Topology Differences" table removed** — confirmed. The file is 222 lines; the old Section 6 (summary table of shard dims, all-reduce paths) is gone. A forward-reference sentence ("For a full enumeration of all Galaxy-specific call sites…see Section 6 below") was added at line 125 inside the Path B description in Section 3. The new Section 6 is "All-Reduce Placement in the MLP Forward Pass" — the original Section 7 renumbered.

3. **symbiote_distributed_primitives.md trailing explanatory sentence removed** — confirmed. After the five-step `forward` numbered list in Section 1, line 57 is the `---` section delimiter. The sentence beginning "The `reduce_scatter_minimal_async` call is the key difference…" is absent.

## CRUCIAL Suggestions

None.

Detailed scan rationale:

- **integration_strategy.md Section 3.2 TG gap list (lines 84–91) vs tt_transformers_parallelism.md Section 3 Path B/C descriptions**: The four bullet points in Section 3.2 name patterns that are absent in Symbiote. The same pattern names appear in tt_transformers_parallelism.md as documentation of what TT Transformers does. The two occurrences carry different information (one is "here is how it works", the other is "Symbiote lacks this"); not a zero-information restatement.

- **integration_strategy.md Section 4 summary table (lines 104–118)**: Aggregates status and action-required columns that do not appear together elsewhere. Not purely redundant.

- **symbiote_distributed_primitives.md Section 3 forward closing sentence (lines 113–113)**: "No CCL operation is performed in `forward`. The caller is responsible for any subsequent all-gather if the full output dimension is needed. This matches the column-parallel contract." The code block (steps 1–4) does make the absence of CCL visible, but the sentence adds the explicit label "column-parallel contract" and the caller responsibility note. Below CRUCIAL threshold; flagged as MINOR below.

- **No verbatim or near-verbatim cross-file copy found** in any other location not already addressed in Pass 1.

- **No warning/callout blocks** whose first sentence repeats the heading are present in any of the four files.

## MINOR Suggestions

### symbiote_distributed_primitives.md Section 3, closing prose (line 113)

**Issue:** "No CCL operation is performed in `forward`. The caller is responsible for any subsequent all-gather if the full output dimension is needed. This matches the column-parallel contract." The first two sentences are directly observable from the code (steps 1–5 contain no CCL call). The third sentence adds a conceptual label but is already implied by the class name `TTNNLinearIReplicatedWColSharded` and Section 3's opening description ("Each device computes a partial output along its slice of the output columns. No reduction is needed after the matmul.").
**Suggestion:** Remove the two observable sentences; retain only "This matches the column-parallel contract." as a one-sentence closing note — or remove all three sentences since the class-level description already establishes the contract. Below CRUCIAL threshold; do not apply in this pass.

### integration_strategy.md Section 2.3, last two sentences (line 58)

**Issue:** Already flagged in Pass 1 as MINOR. "This is equivalent when `self.device` is the correct mesh device, but TT Transformers' approach is more explicit and does not depend on `self.device` being set before weight loading. The Symbiote approach works as long as `move_weights_to_device_impl` is always called after device assignment — which the `TTNNModule` lifecycle guarantees." These sentences defend the Symbiote status quo against a concern that the section does not raise as an action item; the conclusion (no change needed) is already implied by the section structure.
**Suggestion:** Remove the last two sentences (retain only the equivalence statement at the start of the paragraph). Below CRUCIAL threshold; do not apply in this pass.

## VERDICT
- Crucial updates: no
