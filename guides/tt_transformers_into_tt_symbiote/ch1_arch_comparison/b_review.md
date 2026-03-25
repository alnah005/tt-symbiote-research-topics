# Agent B Review — Chapter 1: Architectural Comparison — Pass 1

1. **File: `tt_symbiote_internals.md`, ~line 17**
   **Error:** The guide states "The `__init__` method initialises **five** instance attributes that the framework relies on." The actual `TTNNModule.__init__` in `module.py` (lines 64–70) initialises **seven** attributes: `_device`, `_preprocessed_weight`, `_weights_on_device`, `_fallback_torch_layer`, `_unique_name`, `_device_state`, and `_model_config`. A reader implementing the class from this description would miss `_device_state` and `_model_config`, both of which are non-trivially used (`_device_state` is required by `set_output_tensors_config` and `_model_config` is exposed as a property and set via `set_model_config`).
   **Fix:** Change "five" to "seven" and ensure `_device_state` and `_model_config` are included in both the prose and the code block.

2. **File: `tt_symbiote_internals.md`, ~line 132**
   **Error:** The guide describes the `CPU` run mode as: "Dispatches all tensor operations to CPU PyTorch, bypassing TTNN entirely. Requires the `TT_SYMBIOTE_DISPATCHER=CPU` environment variable." This is misleading in direction of causality. Per `run_config.py` lines 1158–1163, the `TT_SYMBIOTE_DISPATCHER=CPU` variable is not what activates the `CPU` run mode — the run mode itself is selected via `TT_SYMBIOTE_RUN_MODE=CPU`. The `TT_SYMBIOTE_DISPATCHER=CPU` check is an assertion inside `get_tensor_run_implementation` that fires *after* the mode is selected, confirming the CPU dispatcher is already active. A reader would be misled into thinking setting only `TT_SYMBIOTE_DISPATCHER=CPU` activates CPU dispatch; they also need `TT_SYMBIOTE_RUN_MODE=CPU`.
   **Fix:** Clarify that `TT_SYMBIOTE_RUN_MODE=CPU` selects the run mode, and `TT_SYMBIOTE_DISPATCHER=CPU` is a required prerequisite that is asserted at startup — both variables must be set.

3. **File: `tt_symbiote_internals.md`, ~line 129**
   **Error:** The guide states the `SEL` run mode is "Single-Engine Lightweight" and "Inherits `NormalRun`; intended for single-engine execution paths." The class definition in `run_config.py` line 645 confirms it inherits `NormalRun`. However, the guide describes `LIGHTWEIGHT` as an unnamed entry in the registry code block (line 113 of the guide) without any prose description, even though it is a distinct class (`LightweightRun` at line 593 of `run_config.py`) from which `SELRun` does **not** inherit. A reader reconciling the registry table with the mode descriptions will find no entry for `LIGHTWEIGHT` in the bulleted list, creating a gap: one registry entry is documented, one is silently omitted.
   **Fix:** Add a bullet for `LIGHTWEIGHT` (maps to `LightweightRun`) in the run-mode descriptions list so all eight registry entries are documented.

4. **File: `tt_transformers_internals.md`, ~line 51 (Transformer key-members block)**
   **Error:** The guide's constructor summary includes `self.sampling = SamplingGenerator(...) # conditionally, if on-device sampling is supported` as if it is always present. The actual code in `model.py` lines 143–150 creates `self.sampling` only when `self._supports_on_device_sampling` is `True`, which requires `prefetcher is None` AND `vocab_size // sampling_splits <= 64 * 1024`. The comment in the guide says "conditionally" but the code block does not show the condition, and the `_supports_on_device_sampling` flag that gates it is not mentioned anywhere in the guide. A reader implementing a `Transformer`-like class would not know to guard `self.sampling` accesses.
   **Fix:** Show the guard condition or at minimum note that `self.sampling` is only set when `_supports_on_device_sampling` is `True`, and describe the two conditions that determine that flag.

5. **File: `tt_transformers_internals.md`, ~line 90**
   **Error:** The guide states "`TensorGroup` is an `Enum` from `models/tt_transformers/tt/model_config.py` with values `FF1_FF3`, `FF2`, `WQKV`, `WO`, `KV_CACHE`, and `ACTIVATION`." It also documents this in the enum table at line 278. The source code in `attention.py` (lines 117–125) directly uses `TensorGroup.WQKV`, `TensorGroup.WO`, `TensorGroup.KV_CACHE`, and `TensorGroup.ACTIVATION`, which all match. However, `mlp.py` uses `TensorGroup.FF1_FF3` and `TensorGroup.FF2` (lines 90–94), and `attention.py` uses `TensorGroup.ACTIVATION`. The values listed are correct and complete for the usages shown. No error here — this item is withdrawn; see the count note below.

   *Replacing item 5 with the actual issue found:*

5. **File: `tt_symbiote_internals.md`, ~line 238 (weight lifecycle sequence, step 4 comment)**
   **Error:** The guide's "full sequence" code block (line 238) contains the comment `# nn.Module.__call__ -> TTNNModule.__call__` on `output = torch_model(input_ids)`. This is incorrect. `torch_model` is an `nn.Module` (the original PyTorch model), and `torch_model(input_ids)` invokes `nn.Module.__call__`, which in turn calls child modules. The replaced submodules are `TTNNModule` instances stored in `torch_model._modules`, so it is the *submodule* calls — not `torch_model.__call__` — that route through `TTNNModule.__call__`. The comment implies `torch_model.__call__` directly dispatches to `TTNNModule.__call__`, which is wrong and would mislead a reader trying to trace the call chain.
   **Fix:** Change the comment to reflect the actual path: `nn.Module.__call__` on the root model eventually reaches the replaced `TTNNModule` children, each of whose `__call__` routes to `TENSOR_RUN_IMPLEMENTATION.module_run`.

## Change Log — Pass 1 fixes applied

- Fix 1: Changed "five" to "seven" instance attributes in TTNNModule.__init__ description; added _device_state and _model_config.
- Fix 2: Clarified CPU mode activation: both TT_SYMBIOTE_RUN_MODE=CPU and TT_SYMBIOTE_DISPATCHER=CPU must be set.
- Fix 3: Added LIGHTWEIGHT (LightweightRun) mode description; clarified SELRun inherits NormalRun, not LightweightRun.
- Fix 4: Expanded self.sampling conditionality: described _supports_on_device_sampling flag and its two guard conditions.
- Fix 5: Corrected call-chain comment: root nn.Module.__call__ dispatches to TTNNModule children, not directly to TTNNModule.__call__.

---

# Agent B Review — Chapter 1: Architectural Comparison — Pass 2

1. **File: `tt_symbiote_internals.md`, SEL run-mode bullet (~line 130)**
   **Error:** The guide describes the `SEL` mode as "intended for single-engine execution paths" and implies it is a single-path executor like `NORMAL`. The actual `SELRun.module_run` in `run_config.py` (lines 665–685) runs **both** the reference `torch_layer` path and the TTNN `forward` path, then calls `compare_fn_outputs` on both inputs and outputs before returning. This is a dual-path comparison mode, functionally identical in structure to `DPLRun`. A reader who reads this description and selects `SEL` expecting a lightweight single-path executor will be surprised to find it requires `_fallback_torch_layer` to be set and incurs double the forward-pass cost. The current description would cause an incorrect implementation decision.
   **Fix:** Correct the SEL description to state that `SELRun` runs both the PyTorch reference path and the TTNN path in parallel and compares their outputs, analogous to DPL. Note that `_fallback_torch_layer` must be set for SEL to function.

2. **File: `tt_symbiote_internals.md`, LIGHTWEIGHT run-mode bullet (~line 127)**
   **Error:** The guide says LIGHTWEIGHT is "A minimal execution path that bypasses the full dispatcher overhead." The actual `LightweightRun` class (lines 593–599 of `run_config.py`) inherits `NormalRun` and only overrides `torch_dispatch` — its `module_run` is fully inherited from `NormalRun`, which still calls `preprocess_weights()`, `move_weights_to_device()`, wraps all inputs as `TorchTTNNTensor`, converts them to TTNN, and records per-operator timings. The only difference from `NORMAL` is that `torch_dispatch` routes all tensor-level operations directly to the PyTorch backend without attempting TTNN dispatch. Describing it as bypassing "the full dispatcher overhead" is incorrect: the module-level overhead (preprocess, move, tensor wrapping) is identical to NORMAL. A reader relying on this description to choose between LIGHTWEIGHT and NORMAL for a host-latency-sensitive path would make the wrong choice.
   **Fix:** Clarify that LIGHTWEIGHT differs from NORMAL only in `torch_dispatch` behavior (tensor ops go directly to PyTorch rather than attempting TTNN routing); the `module_run` path — including `preprocess_weights`, `move_weights_to_device`, and input tensor wrapping — is identical to NORMAL.

3. **File: `tt_transformers_internals.md`, navigation footer (last line)**
   **Error:** The footer reads `**Next:** [Chapter 2 — Weight Management and Precision](../ch2_weight_management/index.md)`. The file `../ch2_weight_management/index.md` does not exist in the repository. A reader following the chapter sequence will reach a dead link and have no path to the next chapter.
   **Fix:** Either create the target file or update the footer to point to the correct next document. If Chapter 2 is not yet written, replace the link with a plain-text note such as "Chapter 2 — Weight Management and Precision (forthcoming)."

## Change Log — Pass 2 fixes applied

- Fix 1: Corrected SEL mode description — it is a dual-path validation mode (runs both TTNN and PyTorch, compares with PCC), not a single-engine execution path.
- Fix 2: Corrected LIGHTWEIGHT mode description — only torch_dispatch is overridden; module_run is inherited from NormalRun and still runs weight preprocessing, device movement, and timing instrumentation.
- Note: Issue 3 (broken link to ch2_weight_management/index.md) is expected — Chapter 2 has not been written yet. The link path is correct per plan.

---

# Agent B Review — Chapter 1: Architectural Comparison — Pass 3

1. **File: `tt_symbiote_internals.md`, CPU run-mode bullet (~line 133)**
   **Error:** The guide states: "`TT_SYMBIOTE_DISPATCHER=CPU` is a separate required prerequisite that is asserted at startup inside `get_tensor_run_implementation` — **both** variables must be set for CPU mode to work." This is factually wrong. The guard in `get_tensor_run_implementation` (run_config.py lines 1158–1163) is `if issubclass(result, LightweightRun)`, which fires only when the selected run mode is `LightweightRun` or a subclass of it (i.e., `LIGHTWEIGHT` and `TRACED`). `CPU` is defined as `class CPU(NormalRun)`, so it does **not** subclass `LightweightRun` and the assertion never fires for it. A reader setting up `CPU` mode who believes both variables are required will add an unnecessary env variable; more critically, a reader setting up `LIGHTWEIGHT` or `TRACED` mode has no indication from the current text that `TT_SYMBIOTE_DISPATCHER=CPU` is required for those modes.
   **Fix:** Move the `TT_SYMBIOTE_DISPATCHER=CPU` prerequisite note from the `CPU` bullet to the `LIGHTWEIGHT` (and `TRACED`) bullet. For `CPU` mode, only `TT_SYMBIOTE_RUN_MODE=CPU` is needed. For `LIGHTWEIGHT` and `TRACED`, both `TT_SYMBIOTE_RUN_MODE` and `TT_SYMBIOTE_DISPATCHER=CPU` are required (asserted at import time).

2. **File: `tt_symbiote_internals.md`, NORMAL_WITH_FALLBACK run-mode bullet (~line 129)**
   **Error:** The guide says `NORMAL_WITH_FALLBACK` "catches any exception from `forward()` and re-runs the original `torch_layer` instead." This describes only the module-level fallback in `module_run`. The actual `NormalRunWithFallback` class also overrides `torch_dispatch` (run_config.py lines 603–616), where individual per-op exceptions are caught and that single operation is retried on the torch path before raising. A reader who reads only the guide description and selects `NORMAL_WITH_FALLBACK` expecting a module-level fallback will be surprised to find that single operators also silently fall back, making it harder to diagnose partial TTNN dispatch failures. This misrepresents the scope of the fallback and can cause incorrect debugging conclusions.
   **Fix:** Note that `NORMAL_WITH_FALLBACK` applies fallback at two levels: (1) per-op in `torch_dispatch` — individual ops that throw are retried on the torch path, and (2) module-level in `module_run` — if `forward()` throws, the entire module re-runs via `torch_layer`.

## Change Log — Pass 3 fixes applied

- Fix 1: Moved TT_SYMBIOTE_DISPATCHER=CPU requirement from CPU bullet to LIGHTWEIGHT/TRACED bullets. CPU subclasses NormalRun (no assertion); LIGHTWEIGHT and TRACED subclass LightweightRun (assertion fires in get_tensor_run_implementation).
- Fix 2: Expanded NORMAL_WITH_FALLBACK description to document two-level fallback: per-op retry in torch_dispatch AND module-level fallback in module_run.

---

# Agent B Review — Chapter 1: Architectural Comparison — Pass 4

1. **File: `tt_symbiote_internals.md`, LIGHTWEIGHT run-mode bullet (~line 127)**
   **Error:** The sentence reads "Because `LightweightRun` is a subclass of `LightweightRun`, `get_tensor_run_implementation` asserts that the CPU dispatcher is active." This is a tautological self-reference: a class cannot be a subclass of itself in any meaningful explanatory sense. The actual reason the assertion fires is that Python's `issubclass(X, Y)` returns `True` when `X is Y` (reflexive) — so `issubclass(LightweightRun, LightweightRun)` is `True` not because of an inheritance relationship to a parent named `LightweightRun`, but simply because `LightweightRun` is the exact class being tested. As written, a reader trying to understand the class hierarchy will be confused or misled into thinking there are two distinct `LightweightRun` classes in a parent-child relationship. The sentence needs to be corrected to give an accurate explanation: `get_tensor_run_implementation` checks `issubclass(result, LightweightRun)`, and since `LightweightRun` satisfies its own `issubclass` check (Python's `issubclass` is reflexive), the assertion fires when `LIGHTWEIGHT` itself is selected.
   **Fix:** Replace the broken self-reference with: "Because `get_tensor_run_implementation` checks `issubclass(result, LightweightRun)` and Python's `issubclass` is reflexive, this assertion fires for `LightweightRun` itself (LIGHTWEIGHT mode) as well as for any subclass of it (TRACED mode)."

2. **File: `tt_symbiote_internals.md`, `self.sampling` conditionality (~line 54)**
   **Error:** The guide states condition (2) as `vocab_size // sampling_splits <= 64 * 1024` but never defines `sampling_splits`. The actual code in `model.py` line 143 is:
   ```python
   sampling_splits = self.args.num_devices if list(self.mesh_device.shape) != [1, 1] else 2
   ```
   On a single-device mesh (`shape == [1, 1]`), `sampling_splits` is hardcoded to `2`, not `num_devices` (which would be `1`). Using `1` instead of `2` doubles the effective threshold, so a reader computing whether on-device sampling is supported on a 1x1 mesh will get the wrong boolean. For example, a model with `vocab_size = 100000` would be incorrectly assessed as supported (`100000 // 1 = 100000 > 65536` is False) vs. actually not supported (`100000 // 2 = 50000 <= 65536` is True). A reader who copies the formula without defining `sampling_splits` will compute wrong results on single-device setups.
   **Fix:** Add the definition: "`sampling_splits` equals `args.num_devices` when the mesh shape is not `[1, 1]`, and `2` otherwise."

3. **File: `tt_symbiote_internals.md`, CPU run-mode bullet (~line 133)**
   **Error:** The guide says CPU mode "Dispatches all tensor operations to CPU PyTorch, bypassing TTNN entirely." This description omits a critical behavioral distinction: `CPU.module_run` (run_config.py lines 806–812) calls `self.torch_layer(*func_args, **func_kwargs)` directly — it never calls `self.forward()`. Every other run mode (`NORMAL`, `LIGHTWEIGHT`, `NORMAL_WITH_FALLBACK`, `SEL`) calls `self.forward()` inside `module_run`. A reader who writes a TTNNModule subclass with a custom `forward()` method and then runs it under `CPU` mode will find that `forward()` is silently never executed; instead, the original `_fallback_torch_layer` (i.e., the original PyTorch module passed to `from_torch()`) is called. This is a materially different execution path from what the guide implies and would cause incorrect behavior if `forward()` contains side effects or assertions the reader expects to run in CPU mode.
   **Fix:** Add: "Unlike other run modes, `CPU.module_run` does not call `self.forward()`. It calls `self.torch_layer()` (the `_fallback_torch_layer`) directly. If `_fallback_torch_layer` is not set, CPU mode will raise an `AttributeError`."
## Change Log — Pass 4 fixes applied

- Fix 1: Replaced tautological LIGHTWEIGHT sentence with correct explanation: issubclass is reflexive, so the LightweightRun guard triggers for LightweightRun and its subclasses.
- Fix 2: Defined sampling_splits: equals num_devices on multi-device meshes, hardcoded to 2 on single-device (1x1) meshes. Effective threshold on single chip: vocab_size <= 131,072.
- Fix 3: Added note to CPU mode: module_run calls _fallback_torch_layer() directly, never calls self.forward().

---

# Agent B Review — Chapter 1: Architectural Comparison — Pass 5

1. **File: `tt_symbiote_internals.md`, TRACED run-mode bullet (~line 134)**
   **Error:** The guide says TRACED "Wraps the forward pass inside a trace-capture context for later replay." This is only true for modules decorated with `@trace_enabled`. From `TracedRun.module_run` (run_config.py lines 1036–1053), when `is_trace_enabled(self)` is `False`, the method prints "[Not Trace-Enabled, Running Normally]" and falls back to the exact same non-traced execution path as `NormalRun`. A reader who sets `TT_SYMBIOTE_RUN_MODE=TRACED` on a module that is not decorated with `@trace_enabled` will observe no tracing at all, yet the guide gives no indication that tracing is opt-in per module class. Equally, a reader implementing a new module who wants it traced will not know they must apply the `@trace_enabled` decorator from `models/experimental/tt_symbiote/core/run_config.py`.
   **Fix:** Add that tracing is selective: only module classes decorated with `@trace_enabled` are actually traced. Modules without this decorator run the normal `LightweightRun` forward path even under `TRACED` mode. Include the import path `from models.experimental.tt_symbiote.core.run_config import trace_enabled`.

2. **File: `tt_symbiote_internals.md`, SEL run-mode bullet (~line 130)**
   **Error:** The guide states SEL "always executes both paths and validates agreement between them." From `SELRun.module_run` (run_config.py lines 664–685), the TTNN `self.forward()` path is only executed when `self.device is not None` (line 673). When no device has been set, only the `torch_layer` reference path runs and `compare_fn_outputs` is never called. A reader who instantiates a SEL-mode module before calling `to_device()` will believe they have validated TTNN correctness when the TTNN path was never reached.
   **Fix:** Change "always executes both paths" to "executes the PyTorch reference path unconditionally, and additionally executes the TTNN path and compares outputs only when `self.device is not None`."

3. **File: `tt_symbiote_internals.md`, NORMAL_WITH_FALLBACK run-mode bullet (~line 129)**
   **Error:** The guide describes the module-level fallback as firing "if the entire `forward()` throws." This is only one of two fallback conditions. The actual `NormalRunWithFallback.module_run` (run_config.py lines 619–641) has a second, unconditional branch: when `self.device is None`, it skips TTNN entirely and calls `self.torch_layer(*args, **kwds)` directly without attempting `forward()` at all. A reader who does not set a device on their TTNNModule in `NORMAL_WITH_FALLBACK` mode will silently receive torch reference output with no warning that the TTNN path was never exercised.
   **Fix:** Document the device=None branch: "If `self.device` is not set, `NORMAL_WITH_FALLBACK` also falls back silently to `torch_layer` without attempting `forward()` at all."

4. **File: `tt_transformers_internals.md`, Attention compute kernel configs (~line 123)**
   **Error:** The guide says: "The `configuration` object also carries two pre-built compute kernel configs used as defaults: `configuration.compute_kernel_config_hifi2` and `configuration.compute_kernel_config_hifi4`." The actual `Attention.__init__` (attention.py lines 94–97) copies **three** configs from `configuration`: `compute_kernel_config_hifi2`, `compute_kernel_config_hifi2_fp16`, AND `compute_kernel_config_hifi4`. The omission of `compute_kernel_config_hifi2_fp16` is material: `model_config.py` line 4091 maps `MathFidelitySetting.HIFI2_FP16` to `configuration.compute_kernel_config_hifi2_fp16`, and this is the default config for `LI_FF1_FF3` and `LI_FF2` in the standard `ModelOptimizations` performance profile (model_config.py lines 266–267). A reader implementing a comparable attention module who copies "two configs" from the guide will lack `hifi2_fp16` and obtain incorrect kernel configuration for MLP linear operators.
   **Fix:** Change "two pre-built compute kernel configs" to "at least three": `compute_kernel_config_hifi2`, `compute_kernel_config_hifi2_fp16`, and `compute_kernel_config_hifi4`. Note that `hifi2_fp16` is the default for FF1/FF3 and FF2 linear operators.

5. **File: `tt_transformers_internals.md`, MLP `as_sharded_tensor` padding dimension (~line 178)**
   **Error:** The guide says `as_sharded_tensor` "optionally pads the hidden dimension." The actual padding call in `mlp.py` line 61 is `pad_hidden_dim(raw_weight, dims[0] if args.is_galaxy else dims[-1])`. On non-TG, `dims[-1]` is used. For `w1`/`w3`, `dims = (-2, -1)`, so `dims[-1] = -1` (the column/output dimension of the transposed weight). For `w2`, `dims = (-1, -2)`, so `dims[-1] = -2` (the row/input dimension). The padding is therefore applied to different axes for `w1`/`w3` vs `w2` on non-TG hardware — it is not uniformly "the hidden dimension." On TG, `dims[0]` is used: for `w1`/`w3`, `dims = (-1, -2)`, so `dims[0] = -1`; for `w2`, `dims = (-2, -1)`, so `dims[0] = -2`. A reader implementing the padding logic who applies it uniformly to "the hidden dimension" will pad the wrong axis for `w2` and produce incorrect weight shapes.
   **Fix:** State that the padding dimension is axis `dims[-1]` on non-TG and `dims[0]` on TG (Galaxy), which differs between `w1`/`w3` and `w2` due to the inverted `dims` tuples.

## Change Log — Pass 5 fixes applied

- Fix 1: TRACED mode: added @trace_enabled requirement — modules without this decorator fall back to non-traced execution silently.
- Fix 2: SEL mode: TTNN path only runs when self.device is not None; validated before set_device() gives false approval.
- Fix 3: NORMAL_WITH_FALLBACK: documented silent torch fallback when device=None (no exception raised).
- Fix 4: Attention compute kernel configs: corrected count from 2 to 3; added hifi2_fp16 as the MLP FF1/FF3 and FF2 default.
- Fix 5: MLP padding axis: corrected to per-weight dims tuple (dims[-1] non-TG, dims[0] TG); not a uniform "hidden dimension" axis.

---

# Agent B Review — Chapter 1: Architectural Comparison — Pass 6

1. **File: `tt_symbiote_internals.md`, NORMAL_WITH_FALLBACK run-mode bullet (~line 129)**
   **Error:** The bullet ends with "Always call `set_device()` before inference; otherwise all modules silently route to PyTorch with no indication." The method `set_device()` does not exist on `TTNNModule`. The correct method name is `to_device(device)` (module.py line 127). A reader who follows this instruction verbatim will call a non-existent method and receive an `AttributeError`. The correct method is shown correctly elsewhere in the same file (e.g., the weight lifecycle code block at line 235 uses `m.to_device(mesh_device)`), making this inconsistency particularly likely to cause confusion.
   **Fix:** Replace `set_device()` with `to_device(device)` in the prose.

2. **File: `tt_symbiote_internals.md`, DPL run-mode bullet (~line 131)**
   **Error:** The guide states DPL "asserts that the outputs match." The actual `DPLRun.module_run` and `DPLRun.torch_dispatch` in `run_config.py` (lines 688–734) call `compare_fn_outputs`, which is defined in `utils.py` (lines 56–112). `compare_fn_outputs` never raises an exception or asserts — on PCC < 0.999 it prints a warning and sets `passed = False`, then continues. A reader who implements a validation harness expecting `DPLRun` to raise on mismatches will receive no exception even when TTNN and PyTorch outputs diverge significantly. The claim "asserts" is factually wrong and would cause an incorrect integration: a caller checking "no exception raised => outputs match" will get false confidence.
   **Fix:** Replace "asserts that the outputs match" with "logs a warning when outputs diverge (PCC < 0.999) but does not raise an exception." Note that `DPL_NO_ERROR_PROP` also only logs — neither DPL variant raises on mismatch.

## Change Log — Pass 6 fixes applied

- Fix 1: NORMAL_WITH_FALLBACK device note: corrected method name from set_device() instance method to set_device(model, ttnn_device) utility function from utils/device_management.py.
- Fix 2: DPL output comparison: corrected "asserts match" to "prints PCC warning only" — compare_fn_outputs never raises; divergent outputs do not halt execution.

---

# Agent B Review — Chapter 1: Architectural Comparison — Pass 7

1. **File: `tt_symbiote_internals.md`, DPL run-mode bullet (~line 131)**
   **Error:** The guide states DPL "Runs both the TTNN path and the reference `torch_layer`." This claim is unconditional, but `DPLRun.module_run` (run_config.py lines 706–735) gates the TTNN forward pass and `compare_fn_outputs` call behind `if self.device is not None`. When no device has been set on the module, `DPLRun.module_run` returns only the `torch_layer` output — the TTNN path is never entered and no comparison is performed. This is the exact same device-gating pattern that was already caught and fixed for SEL mode in Pass 5. A reader who uses DPL to validate TTNN correctness before calling `set_device(model, ttnn_device)` will receive only the PyTorch reference output with no comparison and no warning that the TTNN path was skipped. The guide's statement "Runs both the TTNN path and the reference `torch_layer`" is therefore only true when a device is set.
   **Fix:** Add the same device-gating note to DPL that was added for SEL in Pass 5: "Note: `DPLRun.module_run` only runs `self.forward()` (the TTNN path) when `self.device is not None`. When no device has been set, only the torch reference path runs and no comparison is performed. Do not use DPL for validating TTNN correctness before calling `set_device()`."

No further issues found in this pass.

## Change Log — Pass 7 fixes applied

- Fix 1: DPL mode: added device guard note — DPLRun.module_run only runs TTNN forward() and compare_fn_outputs when self.device is not None. Call set_device(model, ttnn_device) before using DPL for TTNN validation.

---

# Agent B Review — Chapter 1: Architectural Comparison — Pass 8

1. **File: `tt_transformers_internals.md`, `ModelOptimizations.accuracy()` description (~line 297)**
   **Error:** The guide states: "Models larger than 70B still use `BFP4` MLPs and `BFP8` attention because they are empirically insensitive to precision at that scale." The phrase "BFP4 MLPs" is incorrect. The actual `accuracy()` branch for 70B+ models (`model_config.py` lines 100–104) only sets `{TensorGroup.FF1_FF3: PrecisionSetting.BFP4}` explicitly. `FF2` (the down-projection / `w2` weight) is not set to BFP4 in this branch; it retains its value from `_default_settings()`, which is `BFP8` (line 256). A reader implementing a 70B+ accuracy configuration who reads "BFP4 MLPs" will incorrectly also set `FF2` to BFP4, introducing additional quantisation error on the down-projection that the actual codebase deliberately avoids.
   **Fix:** Replace "BFP4 MLPs" with "BFP4 FF1/FF3 weights (`w1`/`w3`)" and note that `FF2` (`w2`) stays at `BFP8` (from the default) in this profile.

No further issues found in this pass.

## Change Log — Pass 8 fixes applied

- Fix 1: ModelOptimizations.accuracy() 70B+ description: replaced "BFP4 MLPs" with "BFP4 FF1/FF3 (w1/w3) only"; noted FF2 (w2) remains BFP8 from _default_settings().

---

# Agent B Review — Chapter 1: Architectural Comparison — Pass 9

1. **File: `tt_transformers_internals.md`, MLP per-layer dtype section (~line 208)**
   **Error:** The guide states: "Typical values are `ttnn.bfloat4_b` for `FF1_FF3` and `ttnn.bfloat8_b` for `FF2`." This is incorrect. The `_default_settings()` method in `model_config.py` (lines 254–256) sets `TensorGroup.FF1_FF3` to `PrecisionSetting.BFP8`, not `BFP4`. `BFP4` for `FF1_FF3` is only used in performance mode (and the 70B+ accuracy-mode special case). Describing `BFP4` as the typical value for `FF1_FF3` reverses the default: the default for both `FF1_FF3` and `FF2` is `BFP8`. A reader implementing per-layer dtype selection who follows this description will load `w1`/`w3` at `BFP4` by default, introducing more quantization error than the actual default configuration and making their results inconsistent with baseline benchmarks reported against the real defaults.
   **Fix:** Change to "Default values are `ttnn.bfloat8_b` for both `FF1_FF3` and `FF2`. `FF1_FF3` is lowered to `ttnn.bfloat4_b` only in performance mode (and the 70B+ accuracy-mode exception), while accuracy mode for most models leaves both at `BFP8` or higher."

No further issues found in this pass.

## Change Log — Pass 9 fixes applied

- Fix 1: MLP per-layer dtype defaults corrected: _default_settings() sets both FF1_FF3 and FF2 to BFP8. BFP4 for FF1_FF3 is performance mode only, not the typical/default.

---

# Agent B Review — Chapter 1: Architectural Comparison — Pass 10

No feedback — chapter approved.

---

# Agent B Review — Chapter 1: Architectural Comparison — Post-Compression Pass

## Post-compression check (4 changed areas)

1. **`tt_symbiote_internals.md`: Device-guard warning extracted into shared callout; per-bullet repetitions removed; class-hierarchy details moved to reference table.**
   The shared callout (Note block above the run-mode list) is present and correct. The per-bullet repetitions are gone. The reference table was introduced cleanly — with one exception: the `DPL_NO_ERROR_PROP` row lists `DPLRun` as the superclass, but `run_config.py` line 738 defines `class DPLRunNoErrorProp(NormalRun)`. The table entry is factually wrong and was not present in prior passes (the error was introduced by compression). Fixed.

2. **`tt_symbiote_internals.md`: Redundant lifecycle prose after "full sequence" code block collapsed to one-sentence cross-reference.**
   The cross-reference ("See Phase 1–3 above for per-phase implementation details.") is present and accurate. No load-bearing information was lost. Clean.

3. **`tt_transformers_internals.md`: 6-call math fidelity block truncated to 2 representative examples + "pattern repeats" comment.**
   The two retained calls (`LI_QKV_DECODE`, `SDPA_DECODE`) are correct. The comment enumerates the omitted four (`LI_O_DECODE`, `LI_QKV_PREFILL`, `SDPA_PREFILL`, `LI_O_PREFILL`), which matches the source (attention.py lines 132–143). No factual errors introduced. Clean.

4. **`tt_transformers_internals.md`: Orphaned padding-axis sentence removed; replaced with 2-row topology table.**
   The topology table (`Non-TG` / `TG` rows with `w1`/`w3` dims, `w2` dims, and padding axis expression) is consistent with `mlp.py` lines 61, 79–80. The note "Padding is applied per weight using its own `dims` tuple" correctly captures that the padding axis differs between `w1`/`w3` and `w2`. Clean.

## New issues found

**File:** `tt_symbiote_internals.md`, class hierarchy quick reference table (line 138)
**Problem:** The `DPL_NO_ERROR_PROP` row lists `DPLRun` as its superclass. The actual definition in `run_config.py` (line 738) is `class DPLRunNoErrorProp(NormalRun)`. A reader relying on this table to understand the class hierarchy — e.g., to determine which methods `DPL_NO_ERROR_PROP` inherits, or to write an `issubclass` guard — will get the wrong answer. `DPLRunNoErrorProp` does not inherit `DPLRun`'s `module_run`; it defines its own.
**Correct statement:** `DPL_NO_ERROR_PROP` maps to `DPLRunNoErrorProp(NormalRun)`, not `DPLRunNoErrorProp(DPLRun)`.
**Fix:** Change the `DPL_NO_ERROR_PROP` superclass cell from `DPLRun` to `NormalRun`.

## Verdict

One issue found.

## Change Log — Post-Compression fixes applied

- `tt_symbiote_internals.md`, class hierarchy table: corrected `DPL_NO_ERROR_PROP` superclass from `DPLRun` to `NormalRun` (source: `run_config.py` line 738, `class DPLRunNoErrorProp(NormalRun)`).

---

# Agent B Review — Chapter 1: Architectural Comparison — Final Pass

## Re-check of post-compression fix

The fix is correctly applied. `tt_symbiote_internals.md` line 138 shows `DPL_NO_ERROR_PROP | NormalRun | No`, matching `run_config.py` line 738: `class DPLRunNoErrorProp(NormalRun)`.

## New issues found

**File:** `tt_symbiote_internals.md`, DPL_NO_ERROR_PROP bullet (line 149)

**Problem:** The guide states `DPL_NO_ERROR_PROP` "suppresses assertion failures, logging mismatches instead of raising." This description is wrong on two counts. First, `DPLRun` (the implied contrast) already does not raise on mismatches — `compare_fn_outputs` only prints a warning — so "suppresses assertion failures" describes no real behavioral difference. Second, the actual distinction between DPL and DPL_NO_ERROR_PROP is in `torch_dispatch`: `DPLRun.torch_dispatch` reuses the incoming `args` tensors for TTNN dispatch, whereas `DPLRunNoErrorProp.torch_dispatch` first creates separate copies via `copy_to_ttnn` before dispatching to TTNN. This means that when TTNN dispatch throws an exception, the exception is absorbed and the original torch-side tensor result is returned uncorrupted. In DPL, a TTNN dispatch error can corrupt the shared tensor objects and produce a meaningless comparison. A reader choosing between DPL and DPL_NO_ERROR_PROP based on the current description will select the wrong mode: they will not know that the real risk DPL_NO_ERROR_PROP protects against is tensor corruption from TTNN exceptions, not suppression of assertions.

**Correct statement:** `DPL_NO_ERROR_PROP` is structurally identical to DPL at the module level (both run PyTorch and TTNN paths and compare with PCC without raising). The difference is in `torch_dispatch`: DPL_NO_ERROR_PROP creates independent TTNN tensor copies before dispatching so that TTNN exceptions cannot corrupt the PyTorch-side tensors. Use DPL_NO_ERROR_PROP when TTNN dispatch errors would otherwise contaminate the comparison; use DPL otherwise.

**Fix:** Replaced the bullet with an accurate description of the tensor-copy isolation mechanism. See Change Log below.

## Verdict

One issue found.

## Change Log — Final Pass fixes applied

- `tt_symbiote_internals.md`, DPL_NO_ERROR_PROP bullet: replaced "suppresses assertion failures, logging mismatches instead of raising" with an accurate description — both DPL and DPL_NO_ERROR_PROP log-only on mismatch; the real distinction is that DPL_NO_ERROR_PROP uses `copy_to_ttnn` in `torch_dispatch` to create isolated tensor copies for TTNN dispatch, preventing TTNN exceptions from corrupting the torch-side result tensors.

---

# Agent B Review — Chapter 1: Architectural Comparison — Final Pass 2

## Re-check of Final Pass fix

The DPL_NO_ERROR_PROP fix was applied to `tt_symbiote_internals.md` line 149. The `copy_to_ttnn` mechanism and the `torch_dispatch` isolation are correctly identified as the distinguishing feature of `DPL_NO_ERROR_PROP`. However, the fix introduced one factually wrong claim: "errors are silently absorbed and only the torch result is returned."

Inspection of `run_config.py` lines 749–756 shows that `DispatchManager.dispatch_to_ttnn_wrapper` is called with no surrounding try/except in `DPLRunNoErrorProp.torch_dispatch`. If TTNN dispatch raises an exception, it propagates — it is not absorbed. The actual protection provided by `copy_to_ttnn` is isolation against **in-place mutation**: TTNN dispatch operates on copies of the tensors, so if TTNN mutates them during a failed dispatch, the original torch-side tensors used to compute `result` remain intact. This is a mutation-isolation guarantee, not an exception-absorption guarantee.

The fix's description is therefore partially incorrect: the `copy_to_ttnn` mechanism and its purpose are correctly identified, but the claim that TTNN errors are "silently absorbed" is not supported by the source code.

## New issues found

**File:** `tt_symbiote_internals.md`, line 149 (DPL_NO_ERROR_PROP bullet)

**Problem:** The Final Pass fix rewrote the DPL_NO_ERROR_PROP bullet to say "errors are silently absorbed and only the torch result is returned." This is factually incorrect. `DPLRunNoErrorProp.torch_dispatch` (run_config.py lines 740–759) calls `DispatchManager.dispatch_to_ttnn_wrapper` with no try/except. If that call raises, the exception propagates normally. The real protection is mutation isolation: `copy_to_ttnn` (run_config.py lines 156–169) creates new `TorchTTNNTensor` objects from `e.elem.clone()`, so TTNN operates on independent copies. If TTNN mutates those copies in-place during a failing dispatch, the original `args` tensors — and the `result` already computed from the torch path — are unaffected. This is a mutation-isolation guarantee, not exception silencing.

**Correct statement:** `DPL_NO_ERROR_PROP` creates isolated TTNN tensor copies via `copy_to_ttnn` to prevent in-place TTNN mutations from corrupting the torch-side result. TTNN exceptions are not absorbed; they propagate as normal. Use `DPL_NO_ERROR_PROP` when TTNN dispatch may mutate shared tensor state in ways that would corrupt the torch result and make the comparison meaningless.

**Fix:** Replace "errors are silently absorbed and only the torch result is returned" with a description of the mutation-isolation guarantee and a note that exceptions still propagate.

## Verdict

One issue found.

## Change Log — Final Pass 2 fixes applied

- `tt_symbiote_internals.md`, DPL_NO_ERROR_PROP bullet (line 149): corrected "errors are silently absorbed" to "TTNN dispatch cannot corrupt the PyTorch-side tensor result via in-place mutation." Added explicit note that TTNN exceptions are not silently absorbed — they still propagate if `dispatch_to_ttnn_wrapper` throws. Updated use-case guidance accordingly.

---

# Agent B Review — Chapter 1: Architectural Comparison — Final Pass 3

## Re-check of Final Pass 2 fix

The Final Pass 2 fix is correctly applied and accurate. `tt_symbiote_internals.md` line 149 now reads: "creates separate TTNN tensor copies via `copy_to_ttnn` before dispatching. This means TTNN dispatch cannot corrupt the PyTorch-side tensor result via in-place mutation — if TTNN mutates the copied tensors during dispatch, the original torch-side tensors (and thus `result`) remain intact. Note: TTNN exceptions are not silently absorbed; if `dispatch_to_ttnn_wrapper` throws, the exception still propagates."

Verified against `run_config.py`:
- `copy_to_ttnn` (lines 156–169): creates `TorchTTNNTensor(ttnn.from_torch(e.elem.clone()))` copies when `e.ttnn_tensor is not None`, providing mutation isolation. Correct.
- `DPLRunNoErrorProp.torch_dispatch` (lines 738–759): calls `dispatch_to_ttnn_wrapper` with no surrounding try/except, so TTNN exceptions propagate normally. Correct.

## New issues found

**File:** `tt_symbiote_internals.md`, line 172 (`.to_ttnn` property description under `TorchTTNNTensor`)

**Problem:** The guide states `.to_ttnn` materialises from `.elem` "using the dtype and mesh mapper from `ttnn_distributed_tensor_config`." The source (`NormalRun.to_ttnn`, run_config.py lines 534–553) derives the TTNN dtype from `torch_dtype_to_ttnn_dtype(self.elem.dtype)` — i.e., from the tensor's own `.elem.dtype` — not from `ttnn_distributed_tensor_config`. Only the `mesh_mapper` is sourced from `ttnn_distributed_tensor_config` (and only when one is attached; otherwise `None`). A reader implementing a custom `to_ttnn` conversion from this description would incorrectly look up dtype in `ttnn_distributed_tensor_config` instead of in `elem.dtype`, producing wrong dtype selection and potentially an `AttributeError` on configs that carry no dtype field.

**Correct statement:** The TTNN dtype comes from `torch_dtype_to_ttnn_dtype(self.elem.dtype)`; only the `mesh_mapper` comes from `ttnn_distributed_tensor_config`.

**Fix:** Replace "using the dtype and mesh mapper from `ttnn_distributed_tensor_config`" with "The TTNN dtype is derived from `self.elem.dtype` (via `torch_dtype_to_ttnn_dtype`); the `mesh_mapper` is taken from `ttnn_distributed_tensor_config.mesh_mapper` if a `DistributedTensorConfig` is attached, otherwise `None`."

## Verdict

One issue found.

## Change Log — Final Pass 3 fixes applied

- `tt_symbiote_internals.md`, line 172 (`.to_ttnn` property): corrected dtype source from "from `ttnn_distributed_tensor_config`" to "from `self.elem.dtype` via `torch_dtype_to_ttnn_dtype`"; clarified that only the `mesh_mapper` comes from `ttnn_distributed_tensor_config`.

---

# Agent B Review — Chapter 1: Architectural Comparison — Final Pass 4

## Re-check of Final Pass 3 fix

The fix is correctly applied. `tt_symbiote_internals.md` line 172 now reads:

> "The TTNN dtype is derived from `self.elem.dtype` (via `torch_dtype_to_ttnn_dtype`); the `mesh_mapper` is taken from `ttnn_distributed_tensor_config.mesh_mapper` if a `DistributedTensorConfig` is attached, otherwise `None`."

Verified against `run_config.py` `NormalRun.to_ttnn` (lines 534–553):
- `dtype=torch_dtype_to_ttnn_dtype(self.elem.dtype)` — dtype sourced from `self.elem.dtype`, not from `ttnn_distributed_tensor_config`. Correct.
- `mesh_mapper=self.ttnn_distributed_tensor_config.mesh_mapper if self.ttnn_distributed_tensor_config else None` — mesh_mapper conditionally from `ttnn_distributed_tensor_config`, otherwise `None`. Correct.

The fix accurately reflects the source code.

## New issues found

No feedback — chapter approved.

## Verdict

No feedback — chapter approved.
