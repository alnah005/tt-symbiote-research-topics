# Agent B Review — Chapter 7: Use Cases and Benchmarking — Pass 1

## Issues

### Issue 1 — `test_llama_intelligent` output filename wrong (Severity: High)
**File:** llm_acceleration.md line 169
**What guide said:** `DispatchManager.save_stats_to_file("llama_timing_stats.csv")` — and the surrounding prose says "Both test functions follow the same five-step sequence"
**What source says:** `test_llama` (line 80) saves to `"llama_timing_stats.csv"`. `test_llama_intelligent` (line 122) saves to `"llama_intelligent_timing_stats.csv"`.
```
# test_llama.py line 122
DispatchManager.save_stats_to_file("llama_intelligent_timing_stats.csv")
```
**Fix:** The shared setup block applies to `test_llama` only. The `test_llama_intelligent` function uses a different output filename. The block at line 169 must use `"llama_timing_stats.csv"` for `test_llama` and note that `test_llama_intelligent` writes to `"llama_intelligent_timing_stats.csv"`.

## Verdict
Approved with fixes

## Change Log — Pass 1 fixes applied

1. **llm_acceleration.md**: Updated the "Setup sequence" section that claimed "Both test functions follow the same five-step sequence" to clarify that the two functions differ in their output filename. `test_llama` saves to `"llama_timing_stats.csv"` and `test_llama_intelligent` saves to `"llama_intelligent_timing_stats.csv"`. The unified code block has been split into a note clarifying the difference.

---

# Agent B Review — Chapter 7: Use Cases and Benchmarking — Pass 2

## Pass 1 Fix Verification

Confirmed. `llm_acceleration.md` lines 169–172 now show the active `DispatchManager.save_stats_to_file("llama_timing_stats.csv")` for `test_llama` and a commented line annotating that `test_llama_intelligent` saves to `"llama_intelligent_timing_stats.csv"`. The surrounding prose at line 147 correctly reads "differing only in the output filename passed to `save_stats_to_file`". The fix is correctly and completely applied.

## New issues found

### Issue 1 — Run-mode table and `DPL`/`DPL_NO_ERROR_PROP`/`SEL` propagation descriptions are inaccurate (Severity: Medium)

**File:** `speech_and_debugging.md` lines 127–130 (table) and lines 135, 143–145 (prose)

**What guide said:**
- Table: `DPL` — "propagates TTNN results"; `DPL_NO_ERROR_PROP` — "propagates PyTorch results"; `SEL` — "propagates PyTorch results, does not propagate TTNN errors"
- `DPL` prose: "propagates the TTNN result to subsequent operations. Because TTNN errors propagate forward, a single bad layer's output will corrupt all downstream layers."
- `DPL_NO_ERROR_PROP` prose: "runs both paths but propagates the PyTorch result instead of the TTNN result."

**What source says:**
Both `DPLRun.module_run` (`run_config.py` line 735) and `DPLRunNoErrorProp.module_run` (line 792) call `create_new_ttnn_tensors_using_torch_output(torch_output, ttnn_output, assign_ttnn_to_torch=True)`. This function returns `torch_output` with `.ttnn_tensor` set to the TTNN result and `.elem` (the PyTorch tensor) retained. Neither mode "propagates TTNN results only" or "propagates PyTorch results only" — both propagate a `TorchTTNNTensor` wrapper with both values populated.

The actual isolation mechanism in `DPL_NO_ERROR_PROP` is at the ATen dispatch level: `DPLRunNoErrorProp.torch_dispatch` (lines 748–758) uses `copy_to_ttnn` to create independent tensor copies for the TTNN path, so TTNN errors cannot corrupt the PyTorch-path tensors. `DPLRun.torch_dispatch` (lines 695–704) passes the original `args` tensor buffers to TTNN, so TTNN errors in one layer do reach downstream inputs.

`SELRun` (lines 661–685) uses `assign_ttnn_to_torch=False` (default), so `.elem` is set to `None` after execution — the torch tensor is not retained, only the TTNN tensor carries forward. The guide's "propagates PyTorch results" description for `SEL` is therefore incorrect.

**Fix applied:** Updated the run-mode table and the `DPL`, `DPL_NO_ERROR_PROP`, and `SEL` prose descriptions to accurately describe the propagation mechanism and isolation strategy.

## Verdict

Approved — Issue 1 found and fixed. No further pass required; all factual claims now match source.

## Change Log — Pass 2 fixes applied

1. **speech_and_debugging.md**: Corrected the run-mode table entries for `DPL`, `DPL_NO_ERROR_PROP`, and `SEL` to accurately describe error propagation behaviour rather than using the inaccurate "propagates TTNN results" / "propagates PyTorch results" shorthand.
2. **speech_and_debugging.md**: Updated `DPL` mode prose to describe the combined `TorchTTNNTensor` propagation and clarify that error propagation occurs because original tensor buffers are shared across layers.
3. **speech_and_debugging.md**: Updated `DPL_NO_ERROR_PROP` mode prose to describe the actual isolation mechanism: independently copied tensors are passed to the TTNN path so TTNN errors do not corrupt the PyTorch-path tensors fed to subsequent layers.

---

# Agent B Review — Chapter 7: Use Cases and Benchmarking — Pass 3

## Pass 2 Fix Verification

Confirmed. `speech_and_debugging.md` now accurately describes all three run modes:

- **DPL table entry** (line 127): "Dual path: runs both PyTorch and TTNN, propagates a combined tensor (PyTorch elem + TTNN tensor), compares outputs; TTNN errors in one layer affect downstream inputs" — matches `DPLRun.module_run` (`run_config.py` line 735) calling `create_new_ttnn_tensors_using_torch_output(..., assign_ttnn_to_torch=True)` while sharing the original `args` tensor buffers.
- **DPL_NO_ERROR_PROP table entry** (line 128): "Dual path: runs both with independently copied tensors for the TTNN path; TTNN errors do not corrupt the PyTorch-path tensors that downstream layers receive, compares outputs" — matches `DPLRunNoErrorProp.torch_dispatch` (`run_config.py` lines 748–758) using `copy_to_ttnn` to isolate the TTNN path.
- **SEL table entry** (line 129): "Single evaluation log: runs both, compares outputs, does not propagate TTNN errors" — matches `SELRun` (`run_config.py` lines 646–686) calling `create_new_ttnn_tensors_using_torch_output` without `assign_ttnn_to_torch=True` (defaulting to `False`), which clears `.elem` after each call.
- DPL mode prose (line 135) and DPL_NO_ERROR_PROP prose (lines 144–149) are also consistent with the source.

The fix is correctly and completely applied.

## New issues found

None. All previous fixes confirmed. No new factual errors found.

Specific checks performed for Pass 3:

1. **`llm_acceleration.md` — `SmartTTNNLinear` decode threshold** (line 128): Guide states `seq_len <= 32` triggers the decode path. Source (`linear_intelligent.py` line 32) confirms `mode = "decode" if seq_len <= 32 else "prefill"`. Correct.
2. **`llm_acceleration.md` — decode path memory config** (line 136): Guide states the decode path calls `ttnn.linear` with `memory_config=ttnn.DRAM_MEMORY_CONFIG` and no program config. Source (`linear_intelligent.py` lines 74–79) confirms. Correct.
3. **`llm_acceleration.md` — `lm_head` hard-coded `None` program config** (line 118): Guide states `SmartTTNNLinear._get_prefill_pc` returns `None` for `module_name == "lm_head"`. Source (`linear_intelligent.py` lines 51–53) confirms. Correct.
4. **`vision_and_multimodal.md` — ViT input tensor shape** (line 123): Guide states the test passes `(1, 224, 224, 4)` and notes this is a test limitation. Source (`test_vit.py` line 149) confirms `torch.randn(1, 224, 224, 4)`. Correct.
5. **`vision_and_multimodal.md` — `TTNNViTIntermediate` module location** (line 45): Guide states it is in `modules/linear.py`. Source (`test_vit.py` line 15) imports it from `modules.linear`. Correct.
6. **`speech_and_debugging.md` — `DPL_NO_ERROR_PROP` completion message format** (line 154): Guide shows `"DPLNoErrorPropRun: Done Executing {class_name} from {module_name} on device {device}"`. Source (`run_config.py` line 793–795) prints `f"DPLNoErrorPropRun: Done Executing {self.__class__.__name__} from {self.module_name} on device {self.device}"`. The guide's format string matches; "This message appears for every `TTNNModule.forward` call" is accurate because the print is in `module_run`, which handles `TTNNModule` forward calls. Correct.

## Verdict

Approved

## Change Log — Pass 3 fixes applied

None — approved.

---

# Agent B Review — Chapter 7: Use Cases and Benchmarking — Post-Compression Review

## Issues found

None. Compression edits verified. Chapter 7 is factually accurate.

Specific checks performed:

1. **`llm_acceleration.md` — `backend` table cell inline descriptions (compression fix 1)**
   The `backend` column description (line 193) states: `TTNN` = a TTNNModule forward/weight call dispatched to device; `Torch` = an individual ATen op through the Python dispatch key; `TorchModules` = the outer `nn.Module.forward` timer injected by `set_device`'s `timed_call` wrapper.
   Source confirms: `run_config.py` line 206 records `"TTNN"` for TTNN ATen ops; line 234 records `"Torch"` for torch ATen ops; `device_management.py` line 67 records `"TorchModules"` from the `timed_call` closure inside `set_device`. Descriptions are accurate and complete.

2. **`llm_acceleration.md` — idempotency sentence after setup code block (compression fix 2)**
   The single retained sentence at line 175 reads: "Both `preprocess_weights` and `move_weights_to_device` are idempotent; they no-op on subsequent calls."
   Source confirms: `module.py` lines 79–84 show `preprocess_weights` checks `_preprocessed_weight` and returns early if already set; lines 87–96 show `move_weights_to_device` checks `_weights_on_device` and returns early if already set. The sentence is accurate.

3. **`speech_and_debugging.md` — `DPL` mode section after heading-restatement removal (compression fix 3)**
   The `DPL` section (lines 133–141) opens with the error-propagation mechanism sentence and the `export` command. No factual content was altered; only a sentence restating the section heading was removed. The section remains coherent and accurate against `run_config.py` `DPLRun.module_run` and `DPLRun.torch_dispatch`.

4. **`speech_and_debugging.md` — `DPL_NO_ERROR_PROP` mode section after heading-restatement removal (compression fix 4)**
   The `DPL_NO_ERROR_PROP` section (lines 143–157) opens with the isolation-mechanism sentence. No factual content was altered. The description of independently copied tensors (via `copy_to_ttnn`) and the completion-message format remain accurate against `run_config.py` `DPLRunNoErrorProp.torch_dispatch` (lines 748–758) and `DPLRunNoErrorProp.module_run` (lines 793–795).

5. **`speech_and_debugging.md` — timing-section cross-reference sentence (compression fix 5)**
   Lines 205–207 read: "The pivot table produced by `DispatchManager.save_stats_to_file("whisper_timing_stats.csv")` has the same structure as the LLaMA pivot table. The fallback-layer identification and TTNN vs CPU column guidance in `llm_acceleration.md` ("Identifying fallback layers") applies equally here."
   Source confirms: both test files call the same `DispatchManager.save_stats_to_file` method, which always writes the same pivot structure (run_config.py lines 281–304). The "Identifying fallback layers" subsection exists at lines 222–225 of `llm_acceleration.md`. The cross-reference is accurate and points to the correct location.

## Verdict

Approved

## Change Log

None.
