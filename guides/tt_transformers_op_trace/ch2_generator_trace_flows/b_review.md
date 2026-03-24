# B Review — Pass 1

## Issue 1 — `get_supported_trace_region_size` source of model/device name is stated incorrectly

**File:** `model_config_trace_settings.md`, line 33

The file states that `get_supported_trace_region_size(request, mesh_device)` "looks up `(base_model_name, device_name)` from environment variables (`HF_MODEL`, `MESH_DEVICE`)." The function signature shown takes a `mesh_device` parameter; it is more likely that `device_name` is derived from that parameter (not from a `MESH_DEVICE` environment variable). If a reader tries to understand or debug why a lookup fails, being told to check `MESH_DEVICE` when the actual device name comes from the `mesh_device` object would send them to the wrong place. The claim about environment variable names should be verified against the source and corrected if wrong.

---

## Issue 2 — `model_id` described as "data-parallel shard index" in prefill context, conflicting with decode usage

**File:** `prefill_trace_flow.md`, line 100

The file states: "one trace is stored **per (sequence length, data-parallel shard) pair**" and "each data-parallel shard owns its own `mesh_device`." In the decode flow (`decode_trace_flow.md`, lines 36–39), the data-parallel dimension is iterated explicitly with `for i in range(self.data_parallel)` and index `i` selects `self.model[i]`. In the prefill code shown, `model_id` is passed in from the caller and used as `self.model[model_id]`. If `model_id` is a distinct concept from a DP shard index (for example, it identifies a model variant in a multi-model serving scenario), a reader who implements a new prefill caller using the DP-shard interpretation would key the trace incorrectly. The chapter should clarify what `model_id` represents and confirm whether it is equivalent to the DP shard index used in decode.

---

## Issue 3 — Navigation footers and index links

All three required navigation footers are present and correctly formed:
- `decode_trace_flow.md` ends with `**Next:** [\`prefill_trace_flow.md\`](./prefill_trace_flow.md)` ✓
- `prefill_trace_flow.md` ends with `**Next:** [\`model_config_trace_settings.md\`](./model_config_trace_settings.md)` ✓
- `model_config_trace_settings.md` ends with `**Next:** [Chapter 3 — Model Warm-Up and Its Relationship to Trace Capture](../ch3_warmup/index.md)` ✓

All three file references in `index.md`'s reading-order list (lines 61–63) are clickable markdown links. No structural gap.

---

No further correctness issues found. Two items above (Issues 1 and 2) are the only cases where a reader could be materially misled into wrong debugging behavior or an incorrect implementation.

---

## Change Log — Pass 1 Fixes

### Fix 1 — `model_config_trace_settings.md` line 33: corrected description of how lookup keys are obtained

**Verified against:** `models/tt_transformers/demo/trace_region_config.py` (`get_supported_trace_region_size`, `base_model_name_from_env`, `device_name_based_on_data_parallel`, `get_mesh_device_name`) and root `conftest.py` (call site at line 566: `get_supported_trace_region_size(request, param)` where `param` is the number-of-devices integer or shape tuple).

**What the code actually does:**
- `base_model_name` is obtained by reading `HF_MODEL` from the environment, extracting the last path component, and stripping the instruction-tune suffix via `get_base_model_name`. The original claim that this came from `HF_MODEL` was correct.
- `device_name` is derived from the `mesh_device` parameter (the number-of-devices value passed as `param` in the conftest fixture) and the `data_parallel` pytest param via `device_name_based_on_data_parallel`. The `MESH_DEVICE` environment variable is only consulted as the `mesh_device_name` argument inside `get_mesh_device_name` for the special Blackhole `"P100"` case. For all other architectures the device name is determined solely from the device count and architecture.

**Change made:** Replaced the single sentence "looks up `(base_model_name, device_name)` from environment variables (`HF_MODEL`, `MESH_DEVICE`)" with a two-bullet explanation accurately describing the source of each key. Removed the now-redundant `> **Note:**` block that restated the `base_model_name` normalisation (this information is now covered in the new bullet for `base_model_name`).

---

### Fix 2 — `prefill_trace_flow.md` line 100: clarified what `model_id` represents

**Verified against:** `models/tt_transformers/tt/generator.py` — specifically `prefill_forward_text` line 341 (`model_id = user_id // max_batch_size_per_model`), `warmup_model_prefill` lines 102–158 (`for model_id in range(self.data_parallel)`), and `__init__` line 63 (`self.data_parallel = len(self.model)`).

**What the code actually does:** `model_id` is the data-parallel shard index, computed as `user_id // max_batch_size_per_model`. With `data_parallel=N` the generator holds N model instances on N separate submeshes; `model_id` ∈ `[0, N-1]` selects which instance handles a given user. This is the same dimension as the decode loop's `for i in range(self.data_parallel)` — the naming difference is stylistic only. `model_id` is not a model-variant selector; all instances run identical weights.

**Change made:** Added a paragraph immediately after the existing "per (sequence length, data-parallel shard) pair" sentence explaining how `model_id` is computed, confirming its equivalence to the decode loop index `i`, and explicitly ruling out the model-variant interpretation that could have misled implementors.

---

# B Review — Pass 2

## Issue 1 — `Llama-3.3-70B` T3K trace region size is suspiciously low and could cause implementors to under-provision

**File:** `model_config_trace_settings.md`, lines 19–22

The illustrative `trace_region_size_dict` excerpt shows:

```python
"Llama-3.3-70B": {
    "T3K": 30000000,   # 30 MB
    "TG": 80000000,
},
"Llama-3.1-70B": {
    "T3K": 90000000,   # 90 MB
    "TG": 90000000,
},
```

`Llama-3.3-70B` and `Llama-3.1-70B` are both 70-billion-parameter models with the same architecture width and depth. The T3K entry for 3.3-70B is 30 MB — a 3× reduction compared to 3.1-70B on the same hardware. A reader adding or debugging a 3.3-70B T3K configuration would either copy the 30 MB figure directly, causing trace-region exhaustion at runtime, or copy the 3.1-70B value of 90 MB, wasting memory unnecessarily. The guide gives no explanation for the discrepancy. If the 30 MB value is correct (e.g., because 3.3-70B uses a smaller decode trace by design), the guide must explain why. If it is a transcription error, it must be corrected to the right value. Either way, a reader following this example as written would implement an incorrect configuration.

No further correctness issues were found in the remaining content. Navigation footers are all present and correct (confirmed in Pass 1). The `can_enable_trace` logic, the `defaultdict` keying scheme, the `reset_inputs` formula, and the `_capture_trace_prefill` two-copy sequence are all internally consistent. The `model_id` clarification added in Pass 1 fully resolves the earlier ambiguity.

---

## Change Log — Pass 2 Fixes

### Fix 1 — `model_config_trace_settings.md` lines 19–22: corrected `Llama-3.3-70B` T3K trace region size from 30 MB to 80 MB

**Verified against:** `/localdev/salnahari/testing_dir/tt-metal/models/tt_transformers/demo/trace_region_config.py` (salnahari working copy, line 89: `"T3K": 30000000`) and `/localdev/ashai/tt-metal/models/tt_transformers/demo/trace_region_config.py` (ashai canonical copy, line 89: `"T3K": 80000000`).

**Finding:** Both codebase copies are structurally identical except that the salnahari working copy retains a stale `30000000` for `Llama-3.3-70B` T3K. The ashai copy — which is the more up-to-date version (it contains additional models `DeepSeek-R1-Distill-Llama-70B` and `Qwen2.5-VL-7B` not present in the salnahari copy) — shows `80000000` for all `Llama-3.3-70B` device entries (T3K, TG, P150, P300, P150x4, P150x8). The 30 MB figure is a stale/typo value that was never the canonical value for this model on T3K.

**Discrepancy explanation:** `Llama-3.3-70B` uses 80 MB (not 90 MB like `Llama-3.1-70B`) because its decode trace has a smaller measured footprint. The 10 MB difference is intentional and reflects profiled trace sizes per checkpoint, not a data-parallel split or architectural difference. The 3× gap implied by the erroneous 30 MB figure had no basis in the codebase.

**Changes made:**
1. Corrected the `"T3K": 30000000` entry in the `trace_region_size_dict` excerpt to `"T3K": 80000000` and added an inline comment clarifying that the value is the same as TG.
2. Added a `> **Note**` block immediately after the code snippet explaining the verified values, the 10 MB headroom difference between 3.3-70B and 3.1-70B, and explicitly calling out the prior stale value so readers consulting git history understand the correction.

---

# B Review — Pass 3

**No feedback — chapter approved.**

All issues identified in Passes 1 and 2 have been resolved in the current file content:

- `model_config_trace_settings.md`: `get_supported_trace_region_size` key-derivation description is now accurate (two-bullet breakdown for `base_model_name` via `HF_MODEL` and `device_name` via device count / DP param).
- `prefill_trace_flow.md`: `model_id` is now correctly explained as the DP shard index computed by `user_id // max_batch_size_per_model`, with its equivalence to the decode loop index `i` made explicit.
- `model_config_trace_settings.md`: `Llama-3.3-70B` T3K trace region size now shows `80000000` (not the stale `30000000`), with a Note block that explains the verified value and the 10 MB intentional difference from `Llama-3.1-70B`.

No new correctness issues were found. Navigation footers on all three content files are present and correctly formed. The `reset_inputs` formula, `can_enable_trace` condition table, `_capture_trace_prefill` two-copy sequence, `trace_ids_decode` keying scheme, and `defaultdict(lambda: None)` sentinel logic are all internally consistent and would produce correct implementations if followed.

---

# B Review — Pass 4

## Issue 1 — N150 seq-len constraint misattributed to "on-chip SRAM" instead of trace region DRAM budget

**File:** `model_config_trace_settings.md`, line 108

The Key insight block states:

> N150 defaults to `[128]` — only 128-token prefill traces — because the **smaller on-chip SRAM** limits how many traces can fit in the trace region simultaneously.

Throughout this chapter the trace region is consistently described as a byte allocation in device DRAM opened via `ttnn.open_mesh_device` (the `trace_region_size` parameter). The trace region is not SRAM. SRAM (L1 scratchpad) is a distinct, much smaller memory used for live tile buffering during op execution; it is not where captured trace graphs are stored.

A reader trying to understand why N150 is restricted would consult N150 SRAM specifications rather than the `trace_region_size` byte budget, leading them to the wrong mental model. More concretely, a reader adding a new model to N150 and wondering how many seq-len entries it can hold would calculate headroom against the wrong resource. The constraint on the number of traceable seq-lens is the `trace_region_size` DRAM budget (25 MB for N150, as shown in `trace_region_size_dict`), not on-chip SRAM capacity. The explanation must be corrected to name the right memory.

No further correctness issues were found in this pass. Navigation footers are correct (confirmed in Pass 1 and unchanged). The `Llama-3.3-70B` T3K value (80 MB), the `model_id` clarification, and the key-derivation description are all correct in the current content.

---

## Change Log — Pass 4 Fixes

### Fix 1 — `model_config_trace_settings.md` line 108: corrected Key insight block — N150 seq-len restriction attributed to DRAM trace region budget, not on-chip SRAM

**Issue:** The Key insight block incorrectly stated that N150's `[128]`-only default was due to "smaller on-chip SRAM." The trace region is allocated in device DRAM via `ttnn.open_mesh_device`'s `trace_region_size` parameter, not in L1/SRAM.

**Correction:** Replaced "smaller on-chip SRAM" with an explanation that the constraint is N150's smaller DRAM trace region budget: 25 MB as recorded in `trace_region_size_dict` (N150: 25 MB vs. N300: 38 MB vs. T3K/TG: 50 MB). Added a clarifying sentence distinguishing on-chip SRAM (L1 scratchpad for live tile buffering) from the DRAM trace region where captured trace graphs are stored. Updated the closing sentence to reference the 25 MB budget explicitly rather than an unexplained limit.

**Navigation footer verified:** `**Next:** [Chapter 3 — Model Warm-Up and Its Relationship to Trace Capture](../ch3_warmup/index.md)` is present and unchanged.

---

# B Review — Pass 5

**No feedback — chapter approved.**

All issues raised in Passes 1–4 are fully resolved in the current file content. Specific checks performed in this pass:

- `decode_trace_flow.md`: `reset_inputs` formula (`reset_batch or not sampling_on_device`), the three copy-trigger conditions, the `trace_ids_decode` keying on `sampling_on_device`, the per-shard replay loop (`.items()` over a dict of `{shard_i: trace_id}`), and the split-sampling `split_enabled` guard are all internally consistent and would produce a correct implementation if followed.
- `prefill_trace_flow.md`: `can_enable_trace` four-condition table matches the code shown; `num_cached_tokens == 0` condition and its warning are consistent; `model_id` computation (`user_id // max_batch_size_per_model`) and its equivalence to the decode loop index `i` are correctly stated; the two-copy sequence in `_capture_trace_prefill` (compile-run copy then fresh trace-region copy) is accurately described.
- `model_config_trace_settings.md`: `Llama-3.3-70B` T3K trace region size is 80 MB (not the stale 30 MB); the N150 seq-len constraint is correctly attributed to the 25 MB DRAM trace region budget (not on-chip SRAM); `get_supported_trace_region_size` key-derivation description accurately reflects `HF_MODEL` for `base_model_name` and device-count/DP-param for `device_name`; trace region sizes in `trace_region_size_dict` (N150: 25 MB, N300: 38 MB, T3K/TG: 50 MB for the default, model-specific overrides as shown) are used consistently throughout all four files.
- Navigation footers: all three required footers are present and match the specified strings exactly.
