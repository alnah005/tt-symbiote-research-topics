# Agent B Review — Chapter 4: MLP Layers and Norms — Pass 1

## Issues

### Issue 1 — `tt_transformers_mlp.md` section 1.3, ~line 93
**What the guide said:** "On Galaxy (TG, 32-device), `w1` and `w3` always use
`ttnn.bfloat8_b` regardless of the optimization setting (see `mlp.py` line 148:
`dtype=ttnn.bfloat8_b if TG else ...`)."

**What the source actually says:** `mlp.py` line 148 is the `dtype=` argument of the
`ttnn.linear` call that produces `w1_out`. It controls the **output activation dtype**,
not the weight dtype. The `w1` and `w3` weights are loaded via `as_sharded_tensor` with
`ff1_3_dtype` (resolved from `decoders_optimizations.get_tensor_dtype(...)`) on all
devices including Galaxy. There is no Galaxy-specific override for the weight dtype in
the weight-loading code.

**Fix applied:** Corrected the description to distinguish output activation dtype (always
`bfloat8_b` on TG) from weight dtype (resolved via `decoders_optimizations` on all
devices).

---

### Issue 2 — `integration_gaps.md` section 2.7, line 130
**What the guide said:** The TT Transformers Galaxy path includes "`all_gather_async`
calls between w3/w1 outputs and the element-wise multiply."

**What the source actually says:** In `mlp.py` lines 251–266, `all_gather_async` is
applied to `w2_in` — the result of the `ttnn.mul` SwiGLU step — before the `w2`
down-projection matmul. The gather does not occur between the w1/w3 outputs and the
multiply; it occurs after the multiply.

**Fix applied:** Corrected the description to state that `all_gather_async` operates on
`w2_in` after `ttnn.mul`, before the `w2` matmul.

---

### Issue 3 — `symbiote_mlp.md` section 6.3, line 273
**What the guide said:** "Source: `module.py` lines 296–316." (for `run_on_devices`)

**What the source actually says:** The `run_on_devices` function is defined at
`module.py` lines 277–321. Lines 296–316 are the inner `wrapper` function body only;
they do not include the outer function definition or the `decorator` closure.

**Fix applied:** Corrected the source citation to `module.py` lines 277–321.

---

### Issue 4 — `tt_transformers_mlp.md` section 2.4, line 240
**What the guide said:** "After the multiply both `w1_out` and `w3_out` are
deallocated." — immediately following the `ttnn.mul` code block, implying deallocation
occurs right after the multiply.

**What the source actually says:** `ttnn.deallocate(w3_out)` and
`ttnn.deallocate(w1_out)` are at `mlp.py` lines 248–249, after an intervening optional
`ttnn.to_memory_config` call at line 246. The deallocations also precede the Galaxy
`all_gather_async` block (lines 251–266) which operates on `w2_in`. Deallocation is not
immediate after `ttnn.mul`.

**Fix applied:** Added the missing line references and clarified the execution context
around the deallocations.

---

## Change Log — Pass 1 fixes applied

1. **`tt_transformers_mlp.md` section 1.3**: Replaced incorrect claim that Galaxy `w1`/`w3`
   weights use `bfloat8_b` with accurate description that `mlp.py` line 148 sets the
   output activation dtype on TG, while weights use `ff1_3_dtype` from
   `decoders_optimizations` on all devices.

2. **`integration_gaps.md` section 2.7**: Corrected "`all_gather_async` calls between
   w3/w1 outputs and the element-wise multiply" to "`all_gather_async` calls on `w2_in`
   (after the `ttnn.mul` SwiGLU step, before the `w2` down-projection matmul)."

3. **`symbiote_mlp.md` section 6.3**: Corrected `run_on_devices` source line citation
   from "module.py lines 296–316" to "module.py lines 277–321."

4. **`tt_transformers_mlp.md` section 2.4**: Clarified that `w1_out` and `w3_out`
   deallocations happen at lines 248–249 (after an optional `to_memory_config` at
   line 246), not immediately after `ttnn.mul`, and added note that the Galaxy
   `all_gather_async` block operates on `w2_in`.

---

# Agent B Review — Chapter 4: MLP Layers and Norms — Pass 2

## Pass 1 Fix Verification

1. **Fix 1 (w1/w3 dtype, section 1.3):** Confirmed. The guide now correctly states that `mlp.py` line 148 controls the output activation dtype (always `bfloat8_b` on TG) and that the weight dtype is resolved via `decoders_optimizations.get_tensor_dtype` (`ff1_3_dtype`) on all devices including Galaxy. The corrected text is present in the current file.

2. **Fix 2 (Galaxy all_gather_async placement, integration_gaps.md section 2.7):** Confirmed. The guide now correctly states that `all_gather_async` is applied to `w2_in` (after the `ttnn.mul` SwiGLU step, before the `w2` down-projection matmul), not between w1/w3 outputs and the multiply.

3. **Fix 3 (run_on_devices source line range, symbiote_mlp.md section 6.3):** Confirmed. The citation now reads "module.py lines 277–321" which matches the actual function definition in source.

4. **Fix 4 (w1_out/w3_out deallocation context, tt_transformers_mlp.md section 2.4):** Confirmed. The guide now explicitly notes that deallocation happens at lines 248–249 after the optional `to_memory_config` at line 246, and before the Galaxy `all_gather_async` block at lines 251–266 that operates on `w2_in`.

## New issues found

### Issue 1 — `tt_transformers_mlp.md` section 1.5, non-Galaxy sharding dimension descriptions

**What the guide said:**
> "On non-Galaxy multi-device setups (e.g., T3K, N300):
> - `w1` / `w3` are column-parallel: sharded on `dim=-2` (the output/column axis).
> - `w2` is row-parallel: sharded on `dim=-1` (the input/row axis)."

**What the source actually says:** `mlp.py` line 80: `w1_dims = (-2, -1)` for non-Galaxy. With `ShardTensor2dMesh(device, dims=(-2,-1), mesh_shape=[1, N])`, the row-axis of the mesh (1 device) maps to `dim=-2` and the col-axis (N devices) maps to `dim=-1`. The actual device-splitting therefore occurs on `dim=-1`, which is the output-features axis after transpose — that is the column-parallel dimension. For `w2_dims = (-1, -2)`, col-axis (N devices) maps to `dim=-2`, so w2 is actually split on `dim=-2` (the input-features axis = row-parallel). The guide had `dim=-2` and `dim=-1` swapped in both bullet points.

**Fix applied:** Corrected to `dim=-1` for w1/w3 and `dim=-2` for w2 in the non-Galaxy description.

---

### Issue 2 — `symbiote_mlp.md` section 6.3, incorrect comparison of Symbiote reduce-scatter to TT Transformers down-projection all-reduce

**What the guide said:**
> "This is the same collective operation used in the TT Transformers MLP for down-projection all-reduce, but it is placed here after the gate/up projection matmuls, not after the down projection."

**What the source actually says:** The TT Transformers down-projection all-reduce (`mlp.py` lines 290–311) uses `tt_all_reduce` on `cluster_axis=0`, a full all-reduce, not a reduce-scatter. The Symbiote `TTNNLinearIColShardedWRowSharded` reduce-scatter (`linear.py` lines 158–172) uses `cluster_axis=1` and `ttnn.Topology.Ring`. The Symbiote operation is analogous to the TT Transformers Galaxy reduce-scatter on `w1_out`/`w3_out` (also `cluster_axis=1`), not to the down-projection all-reduce.

**Fix applied:** Replaced the incorrect comparison with an accurate one: the Symbiote reduce-scatter (cluster_axis=1) is analogous to the TT Transformers w1/w3 Galaxy reduce-scatter (cluster_axis=1), and is distinct from the w2 all-reduce (tt_all_reduce on cluster_axis=0).

## Verdict

Two factual errors found and fixed. Pass 1 fixes all confirmed present and correct. Chapter approved after Pass 2 corrections.

## Change Log — Pass 2 fixes applied

1. **`tt_transformers_mlp.md` section 1.5**: Corrected non-Galaxy sharding dimension descriptions: w1/w3 changed from "sharded on `dim=-2`" to "sharded on `dim=-1`" (the actual col-axis split dimension); w2 changed from "sharded on `dim=-1`" to "sharded on `dim=-2`" (the actual col-axis split dimension for the down projection). Source: `mlp.py` lines 79–80 with `ShardTensor2dMesh` semantics and `cluster_shape=[1, N]`.

2. **`symbiote_mlp.md` section 6.3**: Corrected the comparison of the Symbiote reduce-scatter to TT Transformers collectives. Replaced the claim that it is "the same collective operation used in the TT Transformers MLP for down-projection all-reduce" with an accurate statement that it is analogous to the TT Transformers Galaxy reduce-scatter on w1_out/w3_out (cluster_axis=1), not to the w2 all-reduce (tt_all_reduce on cluster_axis=0).

---

# Agent B Review — Chapter 4: MLP Layers and Norms — Pass 3

## Pass 2 Fix Verification

1. **Fix 1 (w1/w3 sharding dim=-1 for non-Galaxy, `tt_transformers_mlp.md` section 1.5):** Confirmed. The guide now reads "sharded on `dim=-1` (the output/column axis)" for w1/w3 and "sharded on `dim=-2` (the input/row axis)" for w2 on non-Galaxy devices. This matches `mlp.py` lines 79–80 with `ShardTensor2dMesh` and `cluster_shape=[1, N]`.

2. **Fix 2 (reduce-scatter vs. all-reduce comparison, `symbiote_mlp.md` section 6.3):** Confirmed. The guide now correctly states that the Symbiote reduce-scatter (`cluster_axis=1`) is analogous to the TT Transformers Galaxy reduce-scatter on `w1_out`/`w3_out` (`cluster_axis=1`), and is distinct from the `w2` all-reduce (`tt_all_reduce` on `cluster_axis=0`).

## New issues found

No feedback — chapter approved.

## Verdict

Pass 2 fixes confirmed present and correct. Full fresh check of all four guide files against all five source files found no new factual errors. Chapter approved.

## Change Log — Pass 3 fixes applied

No fixes required.

---

# Agent B Review — Chapter 4: MLP and Norms — Pass 4 (Post-Compression + New Files)

## Pass 3 fix verification (existing files)

All four existing files (`tt_transformers_mlp.md`, `symbiote_mlp.md`, `integration_gaps.md`, `index.md`) were re-checked against source after the C compression pass. No new factual errors were introduced by compression. Pass 1–3 fixes remain intact.

## New files review (normalization_comparison.md, decoder_block_assembly.md)

### Issue 1 — `attn_out.deallocate()` placement stated as "after the first residual add" (Severity: Medium)

**File:** `decoder_block_assembly.md` line 84

**What guide said:**
> "`attn_out.deallocate()` is called explicitly after the first residual add."

**What source says:** `decoder.py` line 279 — `ttnn.deallocate(attn_out)` is an unconditional call that sits after `self.ff_norm(...)` (line 260) and the entire `pre_ff_norm` conditional block (lines 262–277). It does not run immediately after the first residual add at lines 250–253 or 272–275.

**Fix:** Updated the bullet to state that `attn_out.deallocate()` is called after `ff_norm` and the optional `pre_ff_norm` residual-add block (at `decoder.py` line 279), not immediately after the first residual add.

---

### Issue 2 — "Every `ttnn.add` call uses `activation_dtype` for non-TG" overclaims (Severity: Medium)

**File:** `decoder_block_assembly.md` lines 81–83

**What guide said:**
> "Every `ttnn.add` call explicitly passes `memory_config=skip_mem_cfg`. The dtype is explicitly controlled (`bfloat16` for TG, or `activation_dtype` from `DecodersPrecision` for non-TG)."

**What source says:**
- `decoder.py` line 251: `dtype=ttnn.bfloat16 if TG else None` (first residual add — non-TG dtype is `None`)
- `decoder.py` line 273: `dtype=ttnn.bfloat16 if TG else None` (pre_ff_norm residual add — non-TG dtype is `None`)
- `decoder.py` lines 306–308: `dtype=self.args.ccl_dtype if TG and not self.args.is_distributed_norm(mode) else activation_dtype or ttnn.bfloat16` (final residual add — only here is `activation_dtype` used for non-TG)

The intermediate residual adds pass `dtype=None` for non-TG, not `activation_dtype`. Only the final `ttnn.add` uses `activation_dtype`.

**Fix:** Updated the dtype description to distinguish the intermediate residual adds (non-TG: `dtype=None`) from the final residual add (non-TG: `activation_dtype or ttnn.bfloat16`).

---

## Verdict

Approved with fixes — 2 issues found and fixed in `decoder_block_assembly.md`. The existing 4 files are factually clean post-compression. `normalization_comparison.md` is factually accurate throughout.

## Change Log — Pass 4 fixes applied

1. **`decoder_block_assembly.md` line 84**: Corrected "`attn_out.deallocate()` is called explicitly after the first residual add" to state it is called after `ff_norm` and the optional `pre_ff_norm` block (at `decoder.py` line 279).

2. **`decoder_block_assembly.md` lines 81–83**: Replaced the blanket claim that all non-TG `ttnn.add` calls use `activation_dtype` with an accurate per-site description: intermediate adds pass `dtype=None` for non-TG; only the final residual add uses `activation_dtype or ttnn.bfloat16`.

---

# Agent B Review — Chapter 4: MLP and Norms — Pass 5

## Pass 4 Fix Verification

1. **Fix 1 (`attn_out.deallocate()` placement, `decoder_block_assembly.md` lines 86–88):** Confirmed. The guide now correctly states that `attn_out.deallocate()` is called at `decoder.py` line 279 — after `ff_norm` and after the conditional `pre_ff_norm` residual-add block — not immediately after the first residual add.

2. **Fix 2 (`ttnn.add` dtype description, `decoder_block_assembly.md` lines 81–85):** Confirmed. The guide now correctly distinguishes the intermediate residual adds (non-TG: `dtype=None`, `decoder.py` lines 250–252 and 272–274) from the final residual add (non-TG: `activation_dtype or ttnn.bfloat16`, `decoder.py` lines 302–309).

## New issues found

### Issue 1 — `integration_gaps.md` section 1 table, lines 26–27: RMSNorm and Distributed RMSNorm reported as absent (Severity: High)

**File:** `integration_gaps.md` lines 26–27 (high-level comparison table)

**What the guide said:**
> "RMSNorm module (pre/post MLP) | … | No norm module exists in `modules/linear.py` or anywhere else in the current `tt_symbiote` module tree"
> "Distributed RMSNorm (pre-all-gather / post-all-gather split) | … | No"

**What the source actually says:** `models/experimental/tt_symbiote/modules/normalization.py` (added as a key source file in Pass 4 context) contains three normalization classes: `TTNNLayerNorm`, `TTNNRMSNorm`, and `TTNNDistributedRMSNorm`. These classes do exist; they have functional gaps versus TT Transformers (documented in `normalization_comparison.md`), but they are not absent.

**Fix applied:** Both table rows corrected to "Partial" with accurate gap descriptions referencing `normalization.py`.

---

### Issue 2 — `integration_gaps.md` section 2.8: states "No normalization module exists in the `tt_symbiote` module tree" (Severity: High)

**File:** `integration_gaps.md` section 2.8, line 138

**What the guide said:**
> "TT Symbiote: No normalization module exists in the `tt_symbiote` module tree — no `TTNNRMSNorm`, `TTNNLayerNorm`, or equivalent class."

**What the source actually says:** `normalization.py` provides `TTNNRMSNorm`, `TTNNDistributedRMSNorm`, and `TTNNLayerNorm`. The accurate situation is that these classes exist but have functional gaps (missing `HiFi2` compute config, `TILE_LAYOUT` vs. `ROW_MAJOR_LAYOUT` weight, T3K restriction on `TTNNDistributedRMSNorm`, hardcoded `Linear` topology, no `add_unit_offset` support, no `DistributedNorm` wrapper).

**Fix applied:** Section 2.8 rewritten to accurately describe the existing classes and their gaps, with a pointer to `normalization_comparison.md` sections 4.1–4.6 for the full gap inventory.

## Verdict

Issues found requiring this pass. Two factual errors corrected in `integration_gaps.md` (table entries and section 2.8 body text). Pass 4 fixes confirmed intact. No new errors found in the other five guide files.

## Change Log — Pass 5 fixes applied

1. **`integration_gaps.md` table line 26**: Corrected "No norm module exists in `modules/linear.py` or anywhere else in the current `tt_symbiote` module tree" to "Partial — `TTNNRMSNorm` exists in `modules/normalization.py` but uses `TILE_LAYOUT` weight (not `ROW_MAJOR_LAYOUT`) and omits `HiFi2` compute kernel config; no `DistributedNorm` wrapper."

2. **`integration_gaps.md` table line 27**: Corrected "No" for Distributed RMSNorm to "Partial — `TTNNDistributedRMSNorm` exists in `modules/normalization.py`; T3K only; CCL params differ (topology hardcoded to `Linear`; no chunks_per_sync/num_workers/num_buffers_per_channel)."

3. **`integration_gaps.md` section 2.8**: Replaced the incorrect "No normalization module exists" claim with an accurate description of `TTNNRMSNorm`, `TTNNDistributedRMSNorm`, and their functional gaps relative to TT Transformers.

---

# Agent B Review — Chapter 4: MLP and Norms — Pass 6

## Pass 5 Fix Verification

1. **Fix 1 (`integration_gaps.md` table line 26 — RMSNorm "No" → "Partial"):** Confirmed. The table now reads "Partial — `TTNNRMSNorm` exists in `modules/normalization.py` but uses `TILE_LAYOUT` weight (not `ROW_MAJOR_LAYOUT`) and omits `HiFi2` compute kernel config; no `DistributedNorm` wrapper." This matches `normalization.py` lines 84–85 (`layout=ttnn.TILE_LAYOUT`) and line 95 (no `compute_kernel_config` argument to `ttnn.rms_norm`).

2. **Fix 2 (`integration_gaps.md` table line 27 — Distributed RMSNorm "No" → "Partial"):** Confirmed. The table now reads "Partial — `TTNNDistributedRMSNorm` exists in `modules/normalization.py`; T3K only; CCL params differ (topology hardcoded to `Linear`; no chunks_per_sync/num_workers/num_buffers_per_channel)." This matches `normalization.py` line 126 (`@run_on_devices(DeviceArch.T3K)`) and line 140 (`topology=ttnn.Topology.Linear`).

3. **Fix 3 (`integration_gaps.md` section 2.8 rewrite):** Confirmed. Section 2.8 now accurately describes `TTNNRMSNorm`, `TTNNDistributedRMSNorm`, and `TTNNLayerNorm` as existing classes with specific functional gaps. All gap claims checked against `normalization.py` source.

## New issues found

### Issue 1 — `integration_gaps.md` section 3 summary item 6: "must be implemented from scratch" contradicts Pass 5 fixes (Severity: Medium)

**File:** `integration_gaps.md` section 3, lines 204–205 (before fix)

**What the guide said:**
> "**RMSNorm module** — required for any complete transformer layer; must be implemented from scratch in TT Symbiote."

**What the source actually says:** Pass 5 correctly established (and `normalization.py` confirms) that `TTNNRMSNorm` and `TTNNDistributedRMSNorm` already exist in `modules/normalization.py`. The "from scratch" language was left over from before Pass 5 and directly contradicts the corrected table and section 2.8.

**Fix applied:** Updated summary item 6 to state that partial implementations exist with functional gaps, consistent with the now-correct table and section 2.8.

## Verdict

Approved — one residual inconsistency found and fixed in `integration_gaps.md` section 3. All three Pass 5 fixes confirmed intact and accurate against source. No new factual errors found in `decoder_block_assembly.md`.

## Change Log — Pass 6 fixes applied

1. **`integration_gaps.md` section 3, item 6**: Replaced "must be implemented from scratch in TT Symbiote" with accurate language acknowledging that `TTNNRMSNorm` and `TTNNDistributedRMSNorm` exist in `modules/normalization.py` but retain functional gaps (weight layout, missing `HiFi2` config, T3K restriction, no `add_unit_offset`), with a pointer to section 2.8.

---

# Agent B Review — Chapter 4: MLP and Norms — Pass 7

## Pass 6 Fix Verification

**Fix 1 (`integration_gaps.md` section 3, item 6 — "from scratch" → partial implementations acknowledged):** Confirmed. Lines 204–207 now read: "partial implementations exist (`TTNNRMSNorm`, `TTNNDistributedRMSNorm` in `modules/normalization.py`) but have functional gaps (weight layout, missing `HiFi2` config, T3K restriction, no `add_unit_offset`). See section 2.8 for details." This is consistent with the corrected table (lines 26–27) and section 2.8, and is supported directly by `normalization.py` — `TTNNRMSNorm` at line 85 (`layout=ttnn.TILE_LAYOUT`), `TTNNDistributedRMSNorm` at line 126 (`@run_on_devices(DeviceArch.T3K)`). All three sections (table, section 2.8, summary item 6) are now internally consistent.

## New issues found

### Issue 1 — `integration_gaps.md` section 3, item 7: "collectives between gate/up outputs and the multiply" contradicts section 2.7 (Severity: Medium)

**File:** `integration_gaps.md` section 3, lines 209–211 (before fix)

**What the guide said:**
> "**Galaxy support** — required for 32-device deployments; involves different sharding dimensions, additional collectives between gate/up outputs and the multiply, and TG memory configs."

**What the source actually says:** This description of the collective placement was corrected in Pass 1 (section 2.7 fix). `mlp.py` lines 251–266 show that `all_gather_async` is applied to `w2_in` — the result of the `ttnn.mul` SwiGLU step — **after** the multiply, not between the gate/up outputs and the multiply. Section 2.7 of the same file already correctly states: "after the `ttnn.mul` SwiGLU step, before the `w2` down-projection matmul." The summary item 7 retained the pre-Pass-1 wording, creating an internal inconsistency.

**Fix applied:** Updated summary item 7 to state "an `all_gather_async` on `w2_in` (after the SwiGLU multiply, before the `w2` matmul)" — consistent with section 2.7 and the source.

## Verdict

Approved — one residual inconsistency found and fixed in `integration_gaps.md` section 3, item 7. Pass 6 fix confirmed intact and accurate against `normalization.py`. All sections of `integration_gaps.md` are now internally consistent. `decoder_block_assembly.md` spot-check found no new issues.

## Change Log — Pass 7 fixes applied

1. **`integration_gaps.md` section 3, item 7**: Replaced "additional collectives between gate/up outputs and the multiply" with "an `all_gather_async` on `w2_in` (after the SwiGLU multiply, before the `w2` matmul)" — consistent with the corrected section 2.7 and `mlp.py` lines 251–266.

---

# Agent B Review — Chapter 4: MLP and Norms — Pass 8

## Pass 7 Fix Verification

**Fix 1 (`integration_gaps.md` section 3, item 7 — Galaxy CCL placement):** Confirmed. Lines 209–211 now read: "an `all_gather_async` on `w2_in` (after the SwiGLU multiply, before the `w2` matmul), and TG-specific memory configs." This is verified against `mlp.py` lines 248–266: `ttnn.deallocate(w3_out)` and `ttnn.deallocate(w1_out)` occur at lines 248–249 (after the `ttnn.mul` SwiGLU step), followed by the `all_gather_async` block on `w2_in` at lines 251–266, which precedes `ttnn.linear(..., self.w2, ...)` at line 275. The fix is correctly applied and internally consistent with section 2.7 (lines 117–129).

## New issues found

None. Approved.

## Verdict

Approved — Pass 7 fix confirmed present and accurate against source. Full check of `integration_gaps.md` found no remaining inconsistencies across table (lines 26–27), section 2.7, section 2.8, and section 3 items 6–7. All prior pass fixes (Passes 1–7) remain intact.

## Change Log — Pass 8 fixes applied

None — approved.

---

# Agent B Review — Chapter 4: MLP and Norms — Post-Compression Review (Pass 9)

## Review scope

Full post-compression review of all six chapter files after compression Passes 1, 2, and 3 (cuts C1–C11, NC1, NC2). Verified: (1) all compression cut locations have accurate surrounding prose and no dangling references; (2) all prior b-review fixes from Passes 1–8 remain intact; (3) key code blocks spot-checked for internal consistency (source files `mlp.py` and `decoder.py` are not present in this repository, so code block verification was limited to cross-file consistency and internal consistency within each guide file); (4) a full read of every guide file for new factual errors.

## Sources checked

- `/localdev/salnahari/testing_dir/tt-symbiote-research-topics/guides/tt_transformers_into_tt_symbiote/ch4_mlp_and_norms/compression_analysis.md` — full compression history (Passes 1–3)
- `/localdev/salnahari/testing_dir/tt-symbiote-research-topics/guides/tt_transformers_into_tt_symbiote/ch4_mlp_and_norms/tt_transformers_mlp.md`
- `/localdev/salnahari/testing_dir/tt-symbiote-research-topics/guides/tt_transformers_into_tt_symbiote/ch4_mlp_and_norms/symbiote_mlp.md`
- `/localdev/salnahari/testing_dir/tt-symbiote-research-topics/guides/tt_transformers_into_tt_symbiote/ch4_mlp_and_norms/integration_gaps.md`
- `/localdev/salnahari/testing_dir/tt-symbiote-research-topics/guides/tt_transformers_into_tt_symbiote/ch4_mlp_and_norms/normalization_comparison.md`
- `/localdev/salnahari/testing_dir/tt-symbiote-research-topics/guides/tt_transformers_into_tt_symbiote/ch4_mlp_and_norms/decoder_block_assembly.md`
- `/localdev/salnahari/testing_dir/tt-symbiote-research-topics/guides/tt_transformers_into_tt_symbiote/ch4_mlp_and_norms/index.md`
- Source files (`sources/mlp.py`, `sources/decoder.py`) — not present in this repository; code block verification relied on cross-file consistency only.

## Issues found

None — no new factual errors found in any guide file.

**Note on b_review Pass 3 phantom confirmation:** Pass 3 of b_review claimed the text "sharded on `dim=-1` (the output/column axis)" for w1/w3 was present in `tt_transformers_mlp.md` §1.5 after the Pass 2 fix. This text is absent from the current file — §1.5 contains only the code block and source citation, with no prose description of the non-Galaxy case. This is consistent with compression cut C2, which removed that prose block as redundant. The current file state is factually accurate (the code block itself is correct), and no prose error is present. The b_review Pass 3 verification statement was inaccurate about the text being present, but this does not require any change to the guide file.

## Detailed per-location findings

**C1 (`tt_transformers_mlp.md` §1.2 — docstring block):** Clean. The weight-mapping table is present; the removed docstring block is absent. No dangling reference or incomplete sentence.

**C2 (`tt_transformers_mlp.md` §1.5 — sharding prose paragraph):** Clean. §1.5 contains only the code block and source citation. No dangling reference. The prior b_review correction target (incorrect dim values in prose) is moot — the prose was removed by C2 and the code block is correct.

**C3 (`tt_transformers_mlp.md` §2.3 — "no fused matmul path" sentence):** Clean. §2.3 leads directly with the code block. No dangling reference.

**C4 (`tt_transformers_mlp.md` §2.4 — "fused op restates SwiGLU formula" sentence):** Clean. The post-`ttnn.mul` prose retained is the deallocation/memory-config ordering note (lines 222–224), which is the load-bearing content. No dangling reference.

**C5 (`symbiote_mlp.md` §2.1 — "guard ensures called at most once" sentence):** Clean. The code block is present; the removed sentence is absent. No dangling reference.

**C6 (`symbiote_mlp.md` §2.2 — "Again guarded for idempotency" sentence):** Clean. Code block present; sentence absent. No dangling reference.

**C7 (`symbiote_mlp.md` §6.1 — "Unlike TTNNLinear, no layout conversion" prose block):** Clean. The code comment `# Stored as raw torch.Tensor; sharding deferred to move_weights_to_device_impl` remains in the code block and carries the same information. No dangling reference.

**C8 (`symbiote_mlp.md` §6.2 — "sharded on dim=-2" / "Dtype is bfloat16" sentences):** Clean. Both facts are shown in the retained `move_weights_to_device_impl` code block (`dtype=ttnn.bfloat16` and `dim=self.weight_dim`). No dangling reference.

**C9 (`integration_gaps.md` §2.1 — TT Transformers paragraph):** Clean. Section opens directly with "**TT Symbiote:**" then "**What to implement:**". No dangling reference.

**C10 (`integration_gaps.md` §2.2 — TT Transformers paragraph and duplicate code block):** Clean. Section opens directly with "**TT Symbiote:**" then "**What to implement:**". No dangling reference.

**C11 (`integration_gaps.md` §2.8 — two-sentence merge):** Clean. The merged sentence is no longer applicable — §2.8 was subsequently rewritten by b_review Pass 5 to accurately describe the existing partial implementations. The compression-merged sentence is not present (replaced by the longer accurate description). No issue.

**NC1 (`normalization_comparison.md` §1.1 — "Compute kernel config" sentence):** Clean. §1.1 goes directly from the `forward` signature block to the distributed path description. The HiFi2 fact is retained in `tt_transformers_mlp.md` §3.2 (code block, lines 351–358) and `normalization_comparison.md` §4.5. No dangling reference.

**NC2 (`normalization_comparison.md` §4.1 — TT Transformers paragraph):** Clean. §4.1 opens with "**TT Symbiote:**" then "**What to add:**". No TT Transformers paragraph present. Full context for the `add_unit_offset` flag is available in §1.3. No dangling reference.

## Verdict

Approved

## Change Log

None.
