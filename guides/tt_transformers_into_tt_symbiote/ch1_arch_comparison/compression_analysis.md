# Compression Analysis: Chapter 1 — Architectural Comparison — Pass 1

## Summary
- Total files analyzed: 2
- Estimated current line count: ~597 lines
- Estimated post-compression line count: ~470 lines
- Estimated reduction: ~21%

---

## CRUCIAL Suggestions

### [tt_symbiote_internals.md] ~lines 127–134
**Issue:** The run-mode bullet list is severely bloated. Each entry contains deeply nested multi-sentence technical notes, caveats, repetition of the same `set_device()` warning (restated nearly verbatim across NORMAL_WITH_FALLBACK, SEL, and DPL bullets — three separate times), and hedging prose like "Do not use DPL to validate TTNN correctness before `set_device(model, ttnn_device)` ... has been called." This same warning — that the TTNN path only runs when `self.device is not None` — is copy-pasted across lines 129, 130, and 131. Each bullet also embeds parenthetical class-hierarchy notes mid-sentence (e.g., `issubclass` reflexivity on LIGHTWEIGHT, CPU subclassing NormalRun) that belong in a separate reference table, not inline in a user-facing bullet list.
**Suggestion:** Extract the repeated `set_device()` device-guard warning into a single shared callout block above the list. Reduce each bullet to a 2–3 sentence max: name, behavior, and one distinguishing trait. Move class-hierarchy / `issubclass` minutiae into a small reference table or footnote.

### [tt_symbiote_internals.md] ~lines 218–244
**Issue:** The "Weight lifecycle: full sequence" code block is almost entirely redundant with the three-phase weight lifecycle already documented in detail at lines 48–86 of the same file. The prose at line 244 ("Steps 4's preprocessing and device movement happen lazily… Subsequent calls skip the preprocessing and device-movement phases because `_preprocessed_weight` and `_weights_on_device` are already `True`") restates exactly what lines 53–58 and 68–73 already show structurally in code.
**Suggestion:** Collapse the "full sequence" section to the code block only (removing the prose summary below it), and add a one-sentence cross-reference: "See Phase 1–3 above for the per-phase implementation details." The code block itself is load-bearing as a usage example and should be kept.

### [tt_transformers_internals.md] ~lines 98–122
**Issue:** The six `get_math_fidelity(...)` call assignments (lines 101–118) are presented as a verbatim copy of the constructor body. This 18-line block only demonstrates that the same `get_math_fidelity` API is called six times with different `OpGroup` variants — a pattern the reader can extrapolate after seeing two examples. The explanatory prose at lines 121–122 restates what `MathFidelitySetting` values mean, which is then repeated again in the `ModelOptimizations` enums table at line 286.
**Suggestion:** Truncate the block to show two representative calls (one decode, one prefill), then write "… (pattern repeats for all six OpGroup variants: LI_QKV_DECODE, SDPA_DECODE, LI_O_DECODE, LI_QKV_PREFILL, SDPA_PREFILL, LI_O_PREFILL)." Remove the inline re-explanation of HIFI2/HIFI4 here and point the reader to the `ModelOptimizations` enums table in the same file.

### [tt_transformers_internals.md] ~lines 179
**Issue:** Line 179 is a single run-on sentence of ~90 words embedded inside the `MLP` prose, explaining the padding axis selection logic ("The padding axis is `dims[-1]` on non-TG topologies and `dims[0]` on TG topology..."). It is structurally disrupted — it is mid-paragraph, orphaned from the code block it annotates (which starts on line 181), and duplicates information already partially stated in the sentence that precedes it (about `as_sharded_tensor` behavior). The sentence also restates the `dims` tuples that appear explicitly in the code example at lines 186–187.
**Suggestion:** Delete the standalone run-on sentence. Attach the TG/non-TG padding-axis distinction as a code comment on the relevant line inside the block, or as a concise two-row table directly beneath the block.

---

## MINOR Suggestions

### [tt_symbiote_internals.md] ~lines 9
**Issue:** The opening paragraph of `TTNNModule` ends with "Symbiote avoids it by maintaining its own child-traversal logic." The phrase "It deliberately does **not** subclass `torch.nn.Module`" is then partially re-explained at the same level of detail in `tt_transformers_internals.md` lines 17–23 (the `LightweightModule` section), where the same three trade-off points (no autograd, no `_modules`/`_parameters`/`_buffers`, direct `__dict__` access) are enumerated. The cross-file repetition is minor since they describe different classes, but the motivation sentence ("attribute access and method dispatch on `nn.Module` carry measurable host overhead") is architecturally important context duplicated across both files without cross-reference.
**Suggestion:** Add a one-line cross-reference in `tt_symbiote_internals.md` pointing to `tt_transformers_internals.md`'s `LightweightModule` section, noting the parallel design rationale, rather than restating the motivation independently.

### [tt_symbiote_internals.md] ~lines 86–95
**Issue:** The `@deallocate_weights_after` prose explanation ("This is useful when a module's weights are large and should not occupy device memory between invocations") is a hedging rationale sentence that adds no implementation information beyond what the decorator name itself already communicates.
**Suggestion:** Cut the explanation sentence; the decorator name and the code example are self-documenting.

### [tt_symbiote_internals.md] ~lines 196–208
**Issue:** Steps 1–6 in the `register_module_replacement_dict` "How it works" section contain verbose narrative ("This gives every existing `nn.Module` a stable dotted name before any replacements occur") that paraphrases what the adjacent code already shows. Step 3 and Step 5 in particular are pure prose restatements of the logic visible in the code block at lines 201–205.
**Suggestion:** Trim the prose in steps 3 and 5 to single-line notes. The code block carries the meaning; the prose should only add what the code cannot show (e.g., the `exclude_replacement` semantics).

### [tt_transformers_internals.md] ~lines 253–263
**Issue:** The `warmup_model_prefill` section has a trailing sentence — "The warmup also sweeps sampling parameters (`can_sample_on_device`, `non_greedy_decoding_on_device`) to ensure all kernel variants are compiled." — that is isolated from the surrounding code example and reads as an afterthought. It names two variables without explaining their relationship to `_supports_on_device_sampling` (already described at lines 54–56 of the same file).
**Suggestion:** Either integrate this note into the `Generator` construction section near line 54 where sampling support is first explained, or cut it as it adds little without that context.

### [tt_transformers_internals.md] ~lines 295–307
**Issue:** The `ModelOptimizations.accuracy()` factory method note (lines 297–298) contains a parenthetical aside — "Note: `FF2` (`w2`, the down-projection) is **not** set to BFP4 in this profile — it inherits `BFP8` from `_default_settings()`." — that exists to pre-empt a potential misreading. This is hedging prose inserted to forestall a confusion that the reader may not have. The same `FF2` default (`BFP8`) is already stated in the `MLP` per-layer dtype section at line 208.
**Suggestion:** Remove the parenthetical note from the `accuracy()` description and rely on the cross-reference to the MLP dtype section already present in the document.

---

## Load-Bearing Evidence

- `tt_symbiote_internals.md` line ~110: `"TENSOR_RUN_IMPLEMENTATION is a class object selected once at module import time by get_tensor_run_implementation() from models/experimental/tt_symbiote/core/run_config.py. It reads the TT_SYMBIOTE_RUN_MODE environment variable..."` — load-bearing because this is the only place the environment variable name, the registry structure, and the startup-time resolution are co-located. Removing or compressing this paragraph would break the reader's ability to configure run modes.

- `tt_symbiote_internals.md` line ~136: `"The run mode is resolved once at process startup because TENSOR_RUN_IMPLEMENTATION = get_tensor_run_implementation() is executed at module import time in module.py. It cannot be changed after import without reimporting."` — load-bearing because the "cannot be changed after import" constraint is a non-obvious operational fact that directly affects how users configure the framework in multi-test environments. This must not be cut.

- `tt_symbiote_internals.md` line ~173: `"A DistributedTensorConfig ... can be attached to a TorchTTNNTensor via set_distributed_tensor_config(). It carries a mesh_mapper (e.g. ttnn.ShardTensor2dMesh) and a mesh_composer (e.g. ttnn.ConcatMesh2dToTensor) used when converting between host and device representations on multi-device meshes."` — load-bearing because this is the only documentation of the `mesh_composer` field and its role in device-to-host reconstruction for multi-device tensors.

- `tt_transformers_internals.md` line ~54: `"self.sampling is only created when _supports_on_device_sampling is True. That flag requires both conditions to hold: (1) prefetcher is None, AND (2) vocab_size // sampling_splits <= 64 * 1024."` — load-bearing because both conditions are non-obvious and the AND relationship is important; omitting either condition would cause silent bugs.

- `tt_transformers_internals.md` line ~56: `"sampling_splits equals num_devices on multi-device meshes ... but is hardcoded to 2 on single-device (1×1) meshes."` — load-bearing; the hardcoded `2` on single-device is a non-obvious special case with a concrete threshold impact (`vocab_size <= 131,072`) that must remain documented.

- `tt_transformers_internals.md` line ~300: `"Per-decoder-layer overrides can be loaded from JSON files named performance_decoder_config.json or accuracy_decoder_config.json placed alongside the model weights."` — load-bearing because this is the only documentation of the JSON-driven override mechanism and its file naming convention. Production users need this to tune individual layers without source changes.

---

## VERDICT
- Crucial updates: yes

## Change Log — Pass 1 CRUCIAL fixes applied

- Change 1: Extracted repeated device-guard warning into shared callout above run-mode list; removed from NORMAL_WITH_FALLBACK, SEL, DPL bullets individually. Class-hierarchy minutiae moved to reference table.
- Change 2: Collapsed redundant lifecycle prose after full-sequence code block; replaced with one-sentence cross-reference to Phase 1-3.
- Change 3: Truncated 6-call math fidelity block to 2 examples + comment; removed HIFI2/HIFI4 re-explanation, added cross-reference to enums table.
- Change 4: Removed orphaned run-on padding-axis sentence; preserved TG/non-TG distinction as code comment or table near the code block.

---

# Compression Analysis: Chapter 1 — Architectural Comparison — Pass 2

## Summary
- Total files analyzed: 2
- Estimated current line count: ~610 lines (tt_symbiote_internals.md ~302 + tt_transformers_internals.md ~308)
- Estimated post-compression line count: ~590 lines
- Estimated reduction: ~3%

## Pass 1 Fix Verification

- **Change 1 (device-guard warning extraction + class-hierarchy table):** CORRECTLY APPLIED. The `> **Note:**` callout block appears at line 127–128 of `tt_symbiote_internals.md` as a single shared warning. The class-hierarchy reference table is present at lines 130–141. Individual run-mode bullets (NORMAL_WITH_FALLBACK, SEL, DPL) no longer repeat the warning verbatim.
- **Change 2 (collapse lifecycle prose to code block + cross-reference):** CORRECTLY APPLIED. The "Weight lifecycle: full sequence" section retains only the code block and ends with "See Phase 1–3 above for per-phase implementation details." at line 261. Redundant prose summary is gone.
- **Change 3 (truncate 6-call math fidelity block to 2 examples):** CORRECTLY APPLIED. Lines 101–108 of `tt_transformers_internals.md` show two representative calls followed by a comment `# … pattern repeats for LI_O_DECODE, LI_QKV_PREFILL, SDPA_PREFILL, LI_O_PREFILL`. The inline HIFI2/HIFI4 re-explanation is removed.
- **Change 4 (remove orphaned padding-axis sentence, add table):** CORRECTLY APPLIED. The run-on sentence is gone. A two-column table at lines 183–188 captures the TG/non-TG topology distinction for `dims` and padding axis. Line 189 adds a concise follow-on note.

## CRUCIAL Suggestions

None. All Pass 1 CRUCIAL fixes are applied and no new CRUCIAL-level redundancy is present in either file.

## MINOR Suggestions

### [tt_symbiote_internals.md] ~lines 86
**Issue:** The `@deallocate_weights_after` explanation sentence — "This is useful when a module's weights are large and should not occupy device memory between invocations." — restates nothing beyond what the decorator name already communicates. This was flagged in Pass 1 and not yet actioned.
**Suggestion:** Delete this sentence. The decorator name, import, and code example are self-documenting.

### [tt_symbiote_internals.md] ~lines 144–151 (LIGHTWEIGHT bullet)
**Issue:** The LIGHTWEIGHT bullet says "Only per-op dispatch is bypassed" and also that "`module_run` is fully inherited from `NormalRun`". The inheritance is already visible in the reference table at line 133 (`LIGHTWEIGHT` superclass = `NormalRun`). The inline restatement of class inheritance inside the bullet duplicates the table.
**Suggestion:** Remove the parenthetical "fully inherited from `NormalRun`" clause from the LIGHTWEIGHT bullet; the table carries that information.

### [tt_symbiote_internals.md] ~lines 213–229 (register_module_replacement_dict steps 3 and 5)
**Issue:** Step 3 prose ("It also handles `dict`, `list`, and `tuple` attributes via `dir(model)` inspection, covering cases where submodules are stored outside `_modules`.") and Step 5 prose ("The `exclude_replacement` parameter is a `set[str]` of module names to skip. If a module's pre-replacement name appears in this set, `initialize_module` returns `None` and the original `nn.Module` is left in place.") are verbose paraphrases of the surrounding code. This was flagged in Pass 1 and not yet actioned.
**Suggestion:** Trim Step 3 to: "Also handles `dict`/`list`/`tuple` attributes outside `_modules`." Trim Step 5 to: "Names in `exclude_replacement` cause `initialize_module` to return `None`, leaving the original in place."

### [tt_transformers_internals.md] ~line 259 (warmup sampling parameters note)
**Issue:** The trailing sentence "The warmup also sweeps sampling parameters (`can_sample_on_device`, `non_greedy_decoding_on_device`) to ensure all kernel variants are compiled." is isolated from the `_supports_on_device_sampling` context already established at lines 54–56. The two variable names are introduced here with no surrounding explanation. This was flagged in Pass 1 and not yet actioned.
**Suggestion:** Cut this sentence. The variables are not defined or anchored in this section, and the warmup's purpose (compile all kernel variants) is already implied by the preceding sweep description.

### [tt_transformers_internals.md] ~line 293 (FF2 parenthetical in accuracy() note)
**Issue:** The note "Note: `FF2` (`w2`, the down-projection) is **not** set to BFP4 in this profile — it inherits `BFP8` from `_default_settings()`." pre-empts a confusion the reader likely does not have, and the FF2 BFP8 default is already stated in the MLP per-layer dtype section at line 204. This was flagged in Pass 1 and not yet actioned.
**Suggestion:** Remove this parenthetical sentence from the `accuracy()` factory method description.

## Load-Bearing Evidence

- `tt_symbiote_internals.md` line ~110–125: The `_RUN_MODE_REGISTRY` dict and the prose explaining `TT_SYMBIOTE_RUN_MODE` environment variable resolution — load-bearing as the sole co-located reference for run mode configuration; cutting would leave users unable to set or understand run modes.

- `tt_symbiote_internals.md` line ~153: `"The run mode is resolved once at process startup ... It cannot be changed after import without reimporting."` — load-bearing; the import-time resolution constraint is non-obvious and operationally critical for test harness configuration.

- `tt_symbiote_internals.md` lines ~188–191: The `DistributedTensorConfig` description including `mesh_mapper` and `mesh_composer` examples — load-bearing as the only documentation of the `mesh_composer` field and its role in host reconstruction on multi-device meshes.

- `tt_symbiote_internals.md` lines ~215–229: Steps 1 and 4 of `register_module_replacement_dict` — load-bearing because the two-phase name-snapshotting + replacement logic (step 1 snapshots names before any mutation; step 4 assigns `_unique_name` from that snapshot) is non-obvious behavior that cannot be inferred from the code alone.

- `tt_transformers_internals.md` lines ~54–56: Both `_supports_on_device_sampling` conditions (prefetcher is None AND vocab_size threshold) plus the single-device `sampling_splits=2` special case — load-bearing; the AND relationship and hardcoded value are non-obvious and omitting either leads to silent misconfiguration.

- `tt_transformers_internals.md` lines ~161–168: The `as_sharded_tensor` helper behavior (transpose, pad, force 4D shape `[1,1,H,W]`, then `ttnn.as_tensor`) — load-bearing; the mandatory 4D reshape for DRAM prefetcher compatibility is a non-obvious constraint.

- `tt_transformers_internals.md` lines ~296–303: The JSON override file names (`performance_decoder_config.json`, `accuracy_decoder_config.json`) and their placement alongside weights — load-bearing; this is the only documentation of the production layer-tuning mechanism.

## VERDICT
- Crucial updates: no
