# Compression Analysis: Mixture of Experts — Pass 1

## Summary
- Total files analyzed: 5
- Estimated current line count: ~571 lines
- Estimated post-compression line count: ~460 lines
- Estimated reduction: ~19%

## CRUCIAL Suggestions

### 1. DMA transfer size (512 bytes) stated three times across two files
`architecture_overview.md` lines 83–87 gives a full paragraph computing 256 × 2 = 512 bytes
and noting the `.float()` widening to 1024 bytes on CPU. `router_and_routing.md` lines 79–88
repeats this calculation almost word-for-word with the same LaTeX formula AND the same
prose about `.float()` widening. One of the two occurrences is entirely redundant.

**Recommendation:** Keep the detailed explanation in `router_and_routing.md` (it is the file
dedicated to routing mechanics). In `architecture_overview.md` reduce lines 82–87 to a single
sentence: "The DMA transfer is 512 bytes (256 × 2 bf16 bytes); see `router_and_routing.md`
for the full breakdown." Saves ~6 lines.

---

### 2. `params per expert = 3,145,728` computed twice with identical decomposition
`architecture_overview.md` lines 16–19 computes:
`2048 × 1024 + 512 × 2048 = 3,145,728`.
`dram_budget.md` lines 34–38 re-derives the same number element-by-element with the same
LaTeX markup and same intermediate values.

**Recommendation:** In `dram_budget.md` replace the full three-equation derivation with a
one-line reference: "Each expert has 3,145,728 elements (derived in `architecture_overview.md`)."
Saves ~7 lines.

---

### 3. bfp4 storage formula computed twice in the same file (`dram_budget.md`)
`dram_budget.md` line 42 computes: `256 × 3,145,728 × 0.5 × 40 ≈ 15.0 GB` (in the
"Expert Weight Size Derivation" section).
`dram_budget.md` line 83 repeats **the exact same formula and result** in the "Why bfp4 for
Routed Expert Weights" section.

**Recommendation:** In the "Why bfp4" section replace the repeated formula at line 83 with
"At bfp4, storage is approximately 15.0 GB (see derivation above)." Saves ~4 lines.

---

### 4. Fused gate+up matmul explanation duplicated across files
`architecture_overview.md` lines 42–47 explains the fused `[d × 2m]` layout, the single
matmul dispatch, and the `ttnn.split` to recover gate and up outputs.
`expert_computation.md` lines 63–86 re-explains the same concept with virtually the same
mathematical framing, the same `[B × 2m]` result shape, and the same split shape diagram.

**Recommendation:** Trim `expert_computation.md` lines 63–86 to 2–3 sentences that point
back to `architecture_overview.md` for the conceptual rationale and focus only on the
specific code call (the `ttnn.linear` + `ttnn.split` block). The "saves one matmul dispatch"
sentence at lines 85–86 is already in `architecture_overview.md` line 46 ("one matmul
dispatch instead of two"). Saves ~8 lines.

---

### 5. "Key properties of the accumulation" bullet 1 restates the obvious (`expert_computation.md` lines 178–182)
The bullet beginning "`result = shared_out` sets the initial accumulator…" re-explains in
three sentences what the immediately preceding one-line code comment `# Routed experts
(top-k, fused gate+up, L1 intermediates)` and the line `result = shared_out` already make
self-evident. The "no separate add is needed" clause is implicit in the code.

**Recommendation:** Delete lines 179–182 entirely (the first bullet). Saves ~4 lines.

---

## MINOR Suggestions

### A. Over-long inline explanation of `.float()` widening (`router_and_routing.md` lines 83–88)
A 6-line paragraph explains that `.float()` converts bf16 to float32 after the DMA transfer,
with the parenthetical "(1024 bytes) on the CPU." This is self-evident from the `.float()`
call name and the preceding comment. The clause about tile-padding also hedges unnecessarily.

**Recommendation:** Collapse lines 83–88 to: "The DMA transfer is 512 bytes (256 bf16
values). `.float()` widens to float32 on CPU after the transfer." Saves ~4 lines.

---

### B. `w != 1.0` guard over-explained (`expert_computation.md` lines 136–138)
Lines 136–138 devote a full paragraph to explaining that `w != 1.0` is "virtually never
true" for the 8-expert case and calling it a "no-cost micro-optimization." This is a single
guard condition that takes 3 words to read in code.

**Recommendation:** Replace with a one-line comment in the code block itself (e.g.,
`# skip if weight == 1 (only for num_experts_per_tok=1)`) and remove the prose paragraph.
Saves ~3 lines.

---

### C. Batched decode assumption restated in `router_and_routing.md` (`architecture_overview.md` lines 99–112 vs. `router_and_routing.md` lines 66–67)
`architecture_overview.md` lines 99–112 fully explains the batch=32 same-prompt assumption
and why row-0 is representative. `router_and_routing.md` line 66 repeats "row 0 representative
for all batch items" in a comment, and the surrounding prose at lines 65–67 restates the
same assumption without adding new information.

**Recommendation:** Remove the redundant clause from `router_and_routing.md` line 66 and
the single-sentence surrounding prose. Saves ~2 lines.

---

### D. "Key Facts at a Glance" in `index.md` partially duplicates each file's opening sentence
`index.md` lines 35–41 list six facts that are each a verbatim or near-verbatim restatement
of the first sentence of the corresponding section in each file (e.g., "256 routed experts +
1 shared expert per layer" appears almost identically in `architecture_overview.md` line 5).

**Recommendation:** This section is valuable as a chapter-level summary and should be kept.
However, the opening sentences of `architecture_overview.md` and `dram_budget.md` could be
shortened slightly to avoid the near-verbatim echo. Low priority — the index is an expected
navigational aid.

---

### E. Shared expert gate code block repeated (`architecture_overview.md` lines 67–68 vs. `router_and_routing.md` lines 129–158)
`architecture_overview.md` lines 67–68 mentions `shared_gate_weight_tt` shape and that no
sync is needed. `router_and_routing.md` lines 124–158 repeats the entire load code block
(14 lines) and re-explains the `[1, 1, 2048, 1]` shape and broadcast. The load code block is
already in `__init__`; listing it in both files is redundant.

**Recommendation:** In `router_and_routing.md`, condense the gate load code block (lines
138–147) to a single comment referencing `__init__` and focus the prose only on the forward-pass
`sigmoid` + `mul` calls. Saves ~10 lines.

---

## Load-Bearing Evidence

- `architecture_overview.md` line ~14: `$$\text{active experts} = 8\ (\text{routed}) + 1\ (\text{shared})$$` — load-bearing because it establishes the fundamental 8+1 activation count that all DRAM calculations and dispatch counts downstream depend on.
- `architecture_overview.md` line ~59: `$$g = \sigma(\mathbf{x}\, W_{\text{shared\_gate}})$$` — load-bearing because it defines the shared expert gating formula which is unique to Qwen3.5 and not replicated elsewhere in the chapter.
- `router_and_routing.md` line ~66: `logits_cpu = ttnn.to_torch(router_logits).float()[0, 0, 0, : self.num_experts]` — load-bearing because this is the exact sync-point line; the slice `[0, 0, 0, :]` and the `.float()` call are both semantically meaningful and must be preserved verbatim.
- `expert_computation.md` line ~103: `input_tensor_a_activations=[ttnn.UnaryOpType.SILU]` — load-bearing because this documents the non-obvious TTNN API for fused SiLU+multiply; it is not derivable from first principles and is the key fact of the SwiGLU fusion section.
- `dram_budget.md` line ~76: `256 × 3,145,728 × 1 byte × 40 ≈ 30.0 GB` — load-bearing because this is the falsification argument for bfp8: the model literally cannot fit without bfp4; the number must remain to make the argument.
- `dram_budget.md` lines ~160–166 (Summary of Precision Choices table): load-bearing because it consolidates all dtype decisions in one place with rationale; the table is the chapter's primary reference artifact and should not be compressed.
- `index.md` line ~37: `One host-device sync per MoE layer per token (router logits readback: 256 floats, ~1 KB).` — load-bearing because it is the chapter-level performance claim; the "~1 KB" figure (slightly rounded from 512 bytes) must be consistent with the body files.

## VERDICT
- Crucial updates: yes

---
## Change Log (Agent A — Pass 1 CRUCIAL fixes)
- architecture_overview.md: Replaced duplicate 512-byte DMA paragraph with cross-reference to router_and_routing.md
- dram_budget.md: Replaced duplicate params-per-expert derivation with one-line cross-reference
- dram_budget.md: Replaced duplicate bfp4 formula in "Why bfp4" section with back-reference
- expert_computation.md: Trimmed fused gate+up re-explanation to 2-3 sentences; retained code block
- expert_computation.md: Removed redundant "key properties" result=shared_out prose

# Compression Analysis: Mixture of Experts — Pass 2

## Summary
- Total files analyzed: 5
- Estimated current line count: ~708 lines (index: 42, architecture_overview: 134, router_and_routing: 163, expert_computation: 206, dram_budget: 163)
- Estimated post-compression line count: ~689 lines
- Estimated reduction: ~3% (Pass 1 CRUCIAL fixes already applied; remaining gains are MINOR)

## CRUCIAL Suggestions
None — all 5 Pass 1 CRUCIAL items confirmed applied:

1. `architecture_overview.md` line 81: duplicate 512-byte DMA paragraph collapsed to a single cross-reference sentence pointing to `router_and_routing.md`. CONFIRMED.
2. `dram_budget.md` line 32: full params-per-expert re-derivation replaced with `"Per-expert parameter count = 3,145,728 (derivation in \`architecture_overview.md\`)."` CONFIRMED.
3. `dram_budget.md` line 75: duplicate bfp4 formula in "Why bfp4" section replaced with `"At bfp4, storage is approximately 15.0 GB (see derivation above)."` CONFIRMED.
4. `expert_computation.md` lines 63–65: fused gate+up re-explanation trimmed to pointer sentence plus retained code block; conceptual rationale no longer re-stated. CONFIRMED.
5. `expert_computation.md` accumulation loop: opening "result = shared_out sets the initial accumulator" prose bullet removed; bullets now start with "All intermediate tensors…". CONFIRMED.

No new crucial issues found in this pass.

## MINOR Suggestions

### A. `.float()` widening paragraph still present in full (`router_and_routing.md` lines 83–88)
Pass 1 MINOR suggestion A was not applied. The paragraph reads: "The router logits are produced by `ttnn.linear` against a `ttnn.bfloat16` weight, so the tensor on device is bf16. `ttnn.to_torch` DMA-copies 512 bytes of bf16 data from device to host; the subsequent `.float()` call widens those values to float32 (1024 bytes) on the CPU, but that conversion happens after the transfer. In practice, `ttnn.to_torch` syncs the device command queue and DMA-copies the entire tile-padded tensor before slicing, but the logically relevant data is just those 256 bf16 values." The byte-widening detail (512 → 1024 bytes CPU-side) and the tile-padding hedge are non-load-bearing.

**Recommendation:** Replace lines 83–88 with: "The DMA transfer is 512 bytes (256 bf16 values). `.float()` widens to float32 on CPU after the transfer; tile-padding beyond the first 256 columns is discarded by the subsequent slice." Saves ~4 lines.

---

### B. `w != 1.0` guard over-explained (`expert_computation.md` lines 121–123)
Pass 1 MINOR suggestion B was not applied. The three-sentence paragraph calling the guard a "no-cost micro-optimization" that is "virtually never true" for 8-expert decode adds no implementer value.

**Recommendation:** Replace the three prose sentences with an inline code comment on the `if w != 1.0:` line: `# guard only triggers when num_experts_per_tok == 1`. Remove the surrounding paragraph. Saves ~3 lines.

---

### C. Shared expert gate init code block repeated (`router_and_routing.md` lines 135–147)
Pass 1 MINOR suggestion E was not applied. The 14-line `load_shared`-equivalent code block reproducing the `ttnn.as_tensor` call for the gate weight, with `cache_file_name`, `DRAM_MEMORY_CONFIG`, etc., is already covered by the `load_shared` pattern in `dram_budget.md` lines 131–146 and `__init__`. The shape `[1, 1, 2048, 1]` is the only unique piece of information; the full block is boilerplate.

**Recommendation:** In `router_and_routing.md`, replace lines 135–147 (the full init code block) with a single sentence: "The weight is loaded in `__init__` as a bf16 DRAM tensor of shape `[1, 1, 2048, 1]` (see `dram_budget.md` — Summary of Precision Choices)." Saves ~10 lines.

---

### D. Dispatch count self-correction in same sentence (`expert_computation.md` lines 169–171)
The sentence "The loop dispatches 4 TTNN ops per expert (linear, split, fused-mul, linear) plus a scale and an add — `ttnn.split` and `ttnn.mul` are two separate kernel dispatches — for a total of approximately 6 device operations × 8 experts = ~48 dispatches" corrects its own count mid-sentence (starts with 4, arrives at 6). A reader reading quickly will pick up "4 ops" and miscount.

**Recommendation:** Restate as: "Each expert requires approximately 6 device operations (gate+up linear → split → fused SiLU-mul → down linear → scale → add), for ~48 total dispatches per MoE layer. All are enqueued before any Python synchronization." Saves 0 lines but eliminates the misleading "4 ops" opener.

---

### E. "Saves ~X lines" accounting
The remaining MINOR savings (A: ~4, B: ~3, C: ~10) total approximately 17 lines, moving estimated line count from ~708 to ~691. Suggestion D is a rewording with no line reduction.

## Load-Bearing Evidence
- `index.md` line ~37: `"One host-device sync per MoE layer per token (router logits readback: 256 floats, ~1 KB)."` — load-bearing because it is the chapter-level performance headline; any edit to the sync mechanism must keep this fact consistent.
- `architecture_overview.md` line ~14: `$$\text{active experts} = 8\ (\text{routed}) + 1\ (\text{shared})$$` — load-bearing because it anchors the 8+1 activation count that all downstream DRAM and dispatch calculations depend on.
- `router_and_routing.md` line ~66: `logits_cpu = ttnn.to_torch(router_logits).float()[0, 0, 0, : self.num_experts]` — load-bearing because the slice `[0, 0, 0, :]`, the `.float()` cast, and the `self.num_experts` bound are all semantically meaningful and constitute the exact sync-point line; must not be paraphrased away.
- `expert_computation.md` line ~88: `input_tensor_a_activations=[ttnn.UnaryOpType.SILU]` — load-bearing because this is the non-obvious TTNN API parameter for fused SiLU+multiply; it cannot be derived from first principles and is the key fact of the SwiGLU fusion section.
- `dram_budget.md` line ~70: `256 × 3,145,728 × 1 byte × 40 ≈ 30.0 GB` — load-bearing because this is the falsification argument for bfp8 (model cannot fit); the number must remain to make the capacity argument concrete.
- `dram_budget.md` lines ~150–158 (Summary of Precision Choices table): load-bearing because it is the chapter's primary consolidated reference for all dtype decisions with rationale; must not be compressed.

## VERDICT
- Crucial updates: no
