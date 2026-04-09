# Agent B Review: Chapter 5 — Pass 1

**Verdict: 2 issues found.**

## Issue 1 — module_catalog.md claims "complete inventory" but omits `linear_intelligent.py` (structural gap)

**File:** `module_catalog.md`, line 5 and `index.md`, line 66

The catalog states it "inventories every TTNNModule subclass in TT-Symbiote" and the index describes it as a "Complete inventory." However, the source directory `modules/linear_intelligent.py` contains three TTNNModule subclasses not mentioned anywhere in the catalog:

- `SmartTTNNLinear` — adaptive prefill/decode linear with sequence-length-dependent program config selection
- `SmartTTNNLinearLLama` — bfloat8_b variant with `@deallocate_weights_after`
- `SmartTTNNLinearLLamaBFloat16` — bfloat16 variant with `@deallocate_weights_after`

These belong in the Linear Modules table and the summary count. The total class count (~50) is also understated.

**Fix:** Add a "Smart Linear Modules" subsection (or extend the existing Linear table) covering all three classes from `modules/linear_intelligent.py`. Update the Linear module count and overall total accordingly. Adjust the "Complete inventory" language if the catalog is intentionally scoped.

## Issue 2 — module_catalog.md navigation footer links to non-existent Chapter 6

**File:** `module_catalog.md`, line 222

The footer reads:

> **Next:** [Chapter 6 --- Integration Strategy](../ch6_integration_strategy/index.md)

The directory `ch6_integration_strategy/` does not exist in the guide. This is a broken link.

**Fix:** Either remove the "Next" footer until Chapter 6 is written, or change it to point back to the chapter index (e.g., `**Next:** [Back to Chapter 5 index](./index.md)`).

---

No other factual errors, wrong API names, wrong method signatures, coherence issues, or missing navigation elements found. All class names, function signatures, dataclass fields, enum values, and code samples verified against source.

---

# Agent B Review: Chapter 5 --- Pass 2

**Verdict: 3 issues found.**

Pass 1 Issue 1 (missing SmartTTNN classes) has been resolved --- all three classes now appear in the Linear Modules table with accurate descriptions. Pass 1 Issue 2 (broken ch6 link) remains open and is carried forward below.

## Issue 1 (carried from Pass 1) --- module_catalog.md navigation footer still links to non-existent Chapter 6

**File:** `module_catalog.md`, line 232

The footer still reads:

> **Next:** [Chapter 6 --- Integration Strategy](../ch6_integration_strategy/index.md)

The directory `ch6_integration_strategy/` does not exist. This is a broken link.

**Fix:** Remove the "Next" footer or replace it with a link back to the chapter index until Chapter 6 is created.

## Issue 2 --- module_catalog.md Linear table omits `TTNNLinearInputReplicatedWeightSharded`

**File:** `module_catalog.md`, Linear Modules table

The source file `modules/linear.py` defines 13 classes. The catalog table lists 12 of them (plus 3 Smart variants from `linear_intelligent.py`). The missing class is `TTNNLinearInputReplicatedWeightSharded` (line 252 of `linear.py`), which is the parent class of `TTNNLinearIReplicatedWColSharded`. Since the catalog claims to be a "complete inventory," this omission should be corrected. The summary row "Linear | 15" should become 16.

**Fix:** Add a row for `TTNNLinearInputReplicatedWeightSharded` to the Linear Modules table (bfloat16, Input:replicated + Weight:sharded, no CCL, Trace Enabled). Update the summary count from 15 to 16 and the total from ~53 to ~54.

## Issue 3 --- ttnn_module_lifecycle.md attributes distributed config dataclasses to wrong source file

**File:** `ttnn_module_lifecycle.md`, sections "DistributedTensorConfig", "DistributedConfig", "CCLManagerConfig" (lines 169--201)

The page header declares **Source:** `core/module.py`, and the three dataclasses (`DistributedTensorConfig`, `DistributedConfig`, `CCLManagerConfig`) are presented under that umbrella. However, all three are defined in `core/run_config.py`, not `core/module.py`. The `TTNNModule` class itself is indeed in `module.py`, but the distributed config types are not.

**Fix:** Add a source annotation before the "Distributed Configuration" heading, e.g., **Source:** `core/run_config.py`, so readers know to look in the correct file.

---

All other claims verified against source: `DeviceArch` enum values match, `SmartTTNNLinear` threshold (32 tokens) is accurate, `@deallocate_weights_after` usage on LLama variants is correct, handler anatomy and dispatch table structure match `default_dispatcher.py`, and sub-page navigation footers (`ttnn_module_lifecycle.md` -> `dispatch_system.md` -> `module_catalog.md`) are correct and functional.

---

# Agent B Review: Chapter 5 --- Pass 3

**Verdict: 1 issue found.**

Pass 2 Issue 2 (`TTNNLinearInputReplicatedWeightSharded` missing from catalog) has been resolved --- the class now appears in the Linear Modules table at line 31 with accurate sharding description. Pass 2 Issue 3 (distributed config source attribution) has been resolved --- `ttnn_module_lifecycle.md` line 167 now correctly attributes `DistributedTensorConfig`, `DistributedConfig`, and `CCLManagerConfig` to `core/run_config.py`.

## Issue 1 (carried from Pass 1 and Pass 2) --- module_catalog.md navigation footer still links to non-existent Chapter 6

**File:** `module_catalog.md`, line 233

The footer still reads:

> **Next:** [Chapter 6 --- Integration Strategy](../ch6_integration_strategy/index.md)

The directory `ch6_integration_strategy/` does not exist in the guide. This is a broken link that has persisted across all three passes.

**Fix:** Remove the "Next" footer or replace it with a link back to the chapter index until Chapter 6 is created. Alternatively, if Chapter 6 is planned as the next writing task, keep the footer but add a comment noting it is a forward reference.

---

All other claims verified against source for Pass 3:

- `TTNNLinearInputReplicatedWeightSharded` correctly listed as bfloat16, I:replicated W:sharded (dim=-1), no CCL, Trace Enabled --- matches `linear.py`.
- Distributed config source annotation (`core/run_config.py`) is accurate --- all three dataclasses confirmed in that file.
- `DeviceArch` enum values (9 variants) match `core/module.py` exactly.
- Handler count "~80" is reasonable (source has 77 `handle_` functions in `default_dispatcher.py`).
- Linear module count of 16 matches: 12 from `linear.py` + 1 `TTNNViTIntermediate` + 3 Smart variants from `linear_intelligent.py`.
- Total class count "~54" is reasonable (actual count approximately 55).
- Navigation footers present on all three content files; absent from `index.md` (correct).
- All index.md links are clickable relative links and resolve to existing files.
