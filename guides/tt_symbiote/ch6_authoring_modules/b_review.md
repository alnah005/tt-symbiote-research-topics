# Agent B Review — Chapter 6: Authoring a New TTNN Module — Pass 1

## Issues

### Issue 1 — `implementation_guide.md` ~line 86 (Step 2, `from_torch` note): "replacement machinery" claimed to call `preprocess_weights()` lazily

**What the guide said:**
> "The replacement machinery and the run-mode `module_run` method call it lazily on the first forward pass."

**What the source says:**
`register_module_replacement_dict` and its inner helper `register_module_replacement_dict_with_module_names` (both in `utils/module_replacement.py`) never call `preprocess_weights()`. The function only walks the model tree, replaces modules, assigns `_unique_name`, and calls `set_model_config`. Lazy weight preprocessing is performed exclusively by `NormalRun.module_run` (and the other `module_run` implementations in `run_config.py`), which calls `self.preprocess_weights()` at the start of each invocation.

**Fix applied:** Removed "The replacement machinery and" from the sentence; it now reads "The run-mode `module_run` method calls it lazily on the first forward pass."

---

### Issue 2 — `implementation_guide.md` lines 146–153 (Step 4, `move_weights_to_device` guard): first assertion omitted

**What the guide said:**
The subsection presented only a single assertion guarding `move_weights_to_device`:
```python
assert self.device is not None, (
    f"Device must be set for {self.module_name} before moving weights to device."
)
```

**What the source says:**
`TTNNModule.move_weights_to_device` (`core/module.py` lines 87–92) contains **two** sequential assertions:
```python
assert (
    self._preprocessed_weight
), f"Weights must be preprocessed for {self.module_name} before moving to device."
assert self.device is not None, f"Device must be set for {self.module_name} before moving weights to device."
```
The preprocessed-weight assertion fires **before** the device assertion. A developer who calls `move_weights_to_device()` without having first called `preprocess_weights()` will receive an `AssertionError` about the preprocessing flag, not about the device — contradicting the guide's description.

**Fix applied:** The subsection heading and prose were updated to mention both assertions; the code block now shows both `assert` statements with the `_preprocessed_weight` check first, and a note was added explaining that the preprocessed-weight assertion fires first.

---

### Issue 3 — `fallback_and_debugging.md` line 35: DPL run mode incorrectly described as "Same as SEL at module level"

**What the guide said:**
```
| DPL | DPLRun | Same as SEL at module level; propagates TTNN tensors downstream |
```

**What the source says:**
`DPLRun.module_run` and `SELRun.module_run` (`run_config.py`) differ in how they combine outputs after comparing them:

- `SELRun.module_run` (line 685): `create_new_ttnn_tensors_using_torch_output(torch_output, ttnn_output)` — uses the default `assign_ttnn_to_torch=False`, which sets `torch_output.elem = None`, so downstream modules receive a wrapper whose only live data is the TTNN tensor embedded in a torch-shaped shell.
- `DPLRun.module_run` (line 735): `create_new_ttnn_tensors_using_torch_output(torch_output, ttnn_output, assign_ttnn_to_torch=True)` — keeps `torch_output.elem` live alongside the TTNN tensor, so both paths remain available to downstream modules.

They are structurally similar (both run both paths and call `compare_fn_outputs`) but produce tensors with different downstream properties; calling them "the same at module level" is incorrect.

**Fix applied:** The DPL table cell was rewritten to describe the DPL-specific behavior explicitly, distinguishing it from SEL.

---

## Change Log — Pass 1 fixes applied

1. **`implementation_guide.md` ~line 85–86** — Removed incorrect claim that the replacement machinery calls `preprocess_weights()`; the sentence now attributes lazy preprocessing solely to the run-mode `module_run` path.

2. **`implementation_guide.md` lines 146–163** — Expanded the "Guard" subsection under Step 4 to show both assertions (`assert self._preprocessed_weight` and `assert self.device is not None`) in source order, updated the subsection heading, and added a prose note that the preprocessed-weight check fires first.

3. **`fallback_and_debugging.md` line 35** — Replaced the inaccurate "Same as SEL at module level" DPL description with an accurate description that notes DPL calls `create_new_ttnn_tensors_using_torch_output` with `assign_ttnn_to_torch=True`, keeping the torch elem live alongside the TTNN tensor, unlike SEL which clears elem.

---

# Agent B Review — Chapter 6: Authoring TTNN Modules — Pass 2

## Pass 1 Fix Verification

1. Removed false claim that "replacement machinery" calls `preprocess_weights()` — CORRECTLY APPLIED. `implementation_guide.md` line 85–86 now reads "The run-mode `module_run` method calls it lazily on the first forward pass." with no reference to the replacement machinery.

2. Expanded `move_weights_to_device` guard description to show both assertions in source order — CORRECTLY APPLIED. `implementation_guide.md` lines 145–162 now show `assert self._preprocessed_weight` first (matching `module.py` lines 89–91), followed by `assert self.device is not None` (matching `module.py` line 92), with prose noting the preprocessed-weight check fires first.

3. Fixed DPL run-mode table row — CORRECTLY APPLIED. `fallback_and_debugging.md` line 35 now accurately describes DPL using `assign_ttnn_to_torch=True`, distinguishing it from SEL which uses the default `False`.

## New issues found

### Issue 1 — `fallback_and_debugging.md` line 121: wrong file path for `dispatcher_config.py` (Severity: Medium)
**File:** fallback_and_debugging.md line 121
**What the guide said:** `"The \`DEBUG\` dispatcher (registered in \`dispatcher_config.py\` as \`"DEBUG"\`)"`
**What the source says:** The only `dispatcher_config.py` in the codebase is at `core/dispatchers/dispatcher_config.py` (`_auto_register_dispatchers()` registers `"DEBUG"` at line 99). There is no `core/dispatcher_config.py`.
**Fix:** Changed `dispatcher_config.py` to `core/dispatchers/dispatcher_config.py`.

## Verdict
Issues found and fixed — one new error corrected. No further pass required; all known factual errors are now resolved.

## Change Log — Pass 2 fixes applied

1. **`fallback_and_debugging.md` line 121** — Corrected wrong file path from `dispatcher_config.py` to `core/dispatchers/dispatcher_config.py`, which is the actual location of the `DEBUG` dispatcher registration.

---

# Agent B Review — Chapter 6: Authoring TTNN Modules — Pass 3

## Pass 2 Fix Verification

**`fallback_and_debugging.md` line 121** — CORRECTLY APPLIED. The line now reads `"The \`DEBUG\` dispatcher (registered in \`core/dispatchers/dispatcher_config.py\` as \`"DEBUG"\`)"`, matching the actual registration at `core/dispatchers/dispatcher_config.py` line 99 (`register_dispatcher("DEBUG", debug_dispatcher)`).

## New issues found

None. All previous fixes confirmed. No new factual errors found.

Verification summary of all claims checked against source:

- `NormalRun.module_run` transform pipeline (`wrap_to_torch_ttnn_tensor -> to_ttnn_wrap -> set_device_wrap`) — matches `run_config.py` line 561.
- `move_weights_to_device` two-assertion guard (preprocessed check first, device check second) — matches `module.py` lines 89–92.
- `DPLRun.module_run` calls `create_new_ttnn_tensors_using_torch_output(..., assign_ttnn_to_torch=True)` — matches `run_config.py` line 735.
- `SELRun.module_run` calls `create_new_ttnn_tensors_using_torch_output(torch_output, ttnn_output)` with default `assign_ttnn_to_torch=False` — matches `run_config.py` line 685.
- `DPLRunNoErrorProp.module_run` uses `copy_to_ttnn` and `compose_transforms(wrap_to_torch_ttnn_tensor, to_ttnn_wrap, set_device_wrap(...))` — matches `run_config.py` lines 775–779.
- `NormalRunWithFallback.module_run` fallback uses `self.torch_layer(*args, **kwds)` (original args, not transformed) — matches `run_config.py` lines 636, 642.
- `deallocate_weights_after` decorator body — matches `module.py` lines 238–247 exactly.
- `DeviceArch` enum string values (`n150`, `n300`, `t3k_wh`, `gx_wh`, `p150`, `p300`, `p150x4`, `p150x8`, `bhglx`) — all match `module.py` lines 253–261.
- `DispatchManager.save_stats_to_file` writes `timings.csv` and `timings_pivot.csv` — matches `run_config.py` lines 280–304.
- Backend labels (`TTNN`, `Torch`, `TorchModules`) in `timings.csv` — match `run_config.py` lines 206, 233, 589.
- `DEBUG` dispatcher wraps `DEFAULT` dispatcher (delegates to `default_dispatcher`) — matches `debug_dispatcher.py` lines 29–36, 50–51.
- `register_module_replacement_dict` assigns `_unique_name` from `model.named_modules()` traversal — matches `module_replacement.py` line 145.
- `run_on_devices` reads `MESH_DEVICE` env var via `MeshShapeToDeviceArch` and raises `RuntimeError` on mismatch — matches `module.py` lines 302–315.

## Verdict

Approved

## Change Log — Pass 3 fixes applied

None — approved.

---

# Agent B Review — Chapter 6: Authoring TTNN Modules — Post-Compression Review

## Issues found

None. Compression edits verified. Chapter 6 is factually accurate.

Verification details:

**Compression fix 1 — `implementation_guide.md` Step 6 (removed "Key points:" paragraph and 3 bullets after the TTNNLinear.forward code block):**

- The code block for `TTNNLinear.forward` ends at the closing fence on line 261. After removal, the next content is the horizontal rule (`---`) and Step 7 heading. The transition is coherent: Step 6's purpose was to show the `forward` pattern, which the code block alone conveys. No prose was left mid-sentence or referencing the removed bullets.
- No other location in `implementation_guide.md`, `fallback_and_debugging.md`, or `index.md` references the removed bullets or contains a dangling cross-reference to them.

**Compression fix 2 — `fallback_and_debugging.md` Common Mistake #1 (replaced verbatim "Correct:" code block with cross-reference sentence):**

- The cross-reference sentence reads: "**Correct:** See the canonical pattern in [implementation_guide.md — Step 5](implementation_guide.md#step-5--deallocate_weights_impl)."
- The anchor target `#step-5--deallocate_weights_impl` corresponds to `## Step 5 — \`deallocate_weights_impl\`` in `implementation_guide.md` (line 182), which is present and contains the canonical `super().deallocate_weights_impl()` pattern (lines 193–199). The cross-reference is accurate.
- The surrounding mistake description (the "Wrong:" block and explanatory prose) remains internally consistent: it describes the consequence of omitting `super()` and directs the reader to the correct pattern. The `module.py` source confirms the base class `deallocate_weights_impl` (lines 120–125) recurses over `self.__dict__` values, matching the prose.
- No other content in the chapter references or depends on the removed verbatim code block.

## Verdict

Approved

## Change Log

None.
