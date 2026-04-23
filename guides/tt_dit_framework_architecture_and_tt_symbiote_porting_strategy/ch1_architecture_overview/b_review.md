# Agent B Review: Chapter 1 — Architecture Overview — Pass 1

1. **File:** `module_and_parameter.md`, line 27. **Error:** The text states "Its constructor initializes three pieces of internal state" but then lists four items (`_children`, `_parameters`, `_is_loaded`, `unload_set`). The actual source code (`layers/module.py` lines 36-39) confirms four attributes are set in `__init__`. **Fix:** Change "three pieces of internal state" to "four pieces of internal state."

2. **File:** `comparison_with_ttnnmodule.md`, line 31. **Error:** The text refers to TT-Symbiote intercepting `torch.__dispatch__`. The attribute `torch.__dispatch__` does not exist. The actual dispatch mechanism is `__torch_dispatch__`, a dunder protocol method defined on the `TorchTTNNTensor` subclass (see `core/tensor.py` line 31: `def __torch_dispatch__(cls, func, types, args=(), kwargs=None)`). A reader searching the codebase for `torch.__dispatch__` would find nothing. **Fix:** Change `torch.__dispatch__` to `__torch_dispatch__` (the tensor subclass protocol method).

3. **File:** `index.md`, line 131. **Error:** The encoder stack table lists Motif as "CLIP only (Motif)." The Motif pipeline (`pipelines/motif/pipeline_motif.py`) imports both `CLIPTokenizerEncoderPair` and `T5TokenizerEncoderPair`, and its constructor defaults to `enable_t5_text_encoder: bool = True`. Motif uses CLIP + T5, not CLIP alone. A reader relying on this table would underestimate Motif's encoder requirements and misunderstand its architecture. **Fix:** Change the Motif encoder entry from "CLIP only (Motif)" to "CLIP + T5 (Motif)" in the encoder stack bullet list.

---

# Agent B Review: Chapter 1 — Architecture Overview — Pass 2

Previous Pass 1 findings (three items) have been verified as fixed:
- `module_and_parameter.md` now correctly says "four pieces of internal state."
- `comparison_with_ttnnmodule.md` now correctly references `__torch_dispatch__`.
- `index.md` now correctly lists Motif as "CLIP + T5."

## New Finding

1. **File:** `index.md`, line 143 and `comparison_with_ttnnmodule.md`, lines 111-132. **Error:** The text in `index.md` states that "`Linear` wraps `ttnn.linear` or `ttnn.experimental.minimal_matmul`." The actual source (`layers/linear.py` lines 61-75) shows `Linear.forward()` exclusively uses `ttnn.experimental.minimal_matmul`. The string `ttnn.linear` does not appear anywhere in `layers/linear.py`. The "or" framing incorrectly implies `ttnn.linear` is a valid code path for the `Linear` layer. Additionally, the simplified TT-DiT code example in `comparison_with_ttnnmodule.md` (the "Creating a Simple Linear Layer" section) shows `return ttnn.linear(x, self.weight.data, bias=self.bias.data)`, reinforcing the incorrect impression. A reader implementing or modifying a Linear-equivalent layer based on these examples would use the wrong TTNN operation. **Fix:** In `index.md`, change the `Linear` bullet to "`Linear` wraps `ttnn.experimental.minimal_matmul`." In `comparison_with_ttnnmodule.md`, update the simplified TT-DiT example to use `ttnn.experimental.minimal_matmul` or add a clear note that the real implementation uses `minimal_matmul`, not `ttnn.linear`.

---

# Agent B Review: Chapter 1 — Architecture Overview — Pass 3

Previous Pass 1 findings (three items) and Pass 2 finding (one item) have been verified as fixed:
- `module_and_parameter.md` correctly says "four pieces of internal state."
- `comparison_with_ttnnmodule.md` correctly references `__torch_dispatch__`.
- `index.md` correctly lists Motif as "CLIP + T5."
- `index.md` now correctly says `Linear` wraps `ttnn.experimental.minimal_matmul` (no mention of `ttnn.linear`).
- `comparison_with_ttnnmodule.md` TT-DiT code example now correctly uses `ttnn.experimental.minimal_matmul`.

## Pass 3 Verification

All factual claims in the three chapter files were verified against source code:
- `Module.__init__` four attributes: confirmed (`layers/module.py` lines 35-39).
- `__setattr__` auto-registration logic: confirmed (lines 50-73).
- `_load_torch_state_dict_inner` ordering (prepare_torch_state, children, parameters, unexpected keys): confirmed (lines 96-136).
- `__call__` -> `forward()` directly with no hooks: confirmed (lines 229-230).
- `Parameter` fields and `load_torch_tensor` validation: confirmed (lines 318-454).
- `ModuleList.forward()` raises `RuntimeError`: confirmed (lines 241-242).
- `UnregisteredModule` proxy pattern: confirmed (lines 277-315).
- TT-Symbiote `TTNNModule.__call__` -> `call()` -> `module_run()` chain: confirmed (`core/module.py` lines 78-82).
- TT-Symbiote `__torch_dispatch__` on `TorchTTNNTensor`: confirmed (`core/tensor.py` line 31).
- TT-Symbiote `named_children` iterates `__dict__`: confirmed (`core/module.py` lines 254-268).
- Motif encoder stack CLIP + T5: confirmed (`pipelines/motif/pipeline_motif.py`).
- Supported models table (six models, pipeline classes, file locations): confirmed against directory listing.
- `QwenImage.md` annotation "not present as separate file": confirmed.
- Navigation links within chapter (index.md, module_and_parameter.md, comparison_with_ttnnmodule.md): all point to existing files.

**No feedback — chapter approved.**

---

# Agent B Review: Chapter 1 — Architecture Overview — Pass 4

Previous passes 1-3 findings (four items total) verified as fixed against source code:
- `module_and_parameter.md` correctly says "four pieces of internal state" (confirmed: `layers/module.py` lines 35-39).
- `comparison_with_ttnnmodule.md` correctly references `__torch_dispatch__` (confirmed: `core/tensor.py` line 31).
- `index.md` correctly lists Motif as "CLIP + T5" (confirmed: `pipelines/motif/pipeline_motif.py` imports both `CLIPTokenizerEncoderPair` and `T5TokenizerEncoderPair`).
- `index.md` and `comparison_with_ttnnmodule.md` correctly reference `ttnn.experimental.minimal_matmul` for `Linear` (confirmed: `layers/linear.py` lines 65-72, no `ttnn.linear` usage).

## Pass 4 Independent Verification

Verified the following claims against source code with no issues found:
- `Module.__init__` four attributes, `__setattr__` auto-registration, `__delattr__` cleanup: confirmed (`layers/module.py` lines 34-84).
- `_load_torch_state_dict_inner` ordering (prepare_torch_state -> children -> parameters -> unexpected keys): confirmed (lines 96-136).
- `load_torch_state_dict` sets `_is_loaded = True` after inner call, raises `ValueError` if strict and keys mismatch: confirmed (lines 138-173).
- `__call__` delegates to `forward()` with no hooks or dispatch: confirmed (lines 229-230).
- `Parameter.__init__` validates mesh_axes divisibility and no duplicate mesh axis assignment: confirmed (lines 318-377).
- `Parameter.load_torch_tensor` shape validation, `from_torch` call, `_set_data` validation (device, dtype, layout, memory_config, shape): confirmed (lines 379-454).
- `ModuleList.forward()` raises `RuntimeError`: confirmed (lines 241-242).
- `UnregisteredModule` proxy forwards `__getattr__` and `__call__`: confirmed (lines 277-315).
- TT-Symbiote `TTNNModule.__call__` -> `call()` -> `TENSOR_RUN_IMPLEMENTATION.module_run()` -> `forward()`: confirmed (`core/module.py` lines 78-82, `core/run_config.py` line 1172ff).
- TT-Symbiote `named_children()` iterates `__dict__`, handles dicts/lists/tuples, skips `_fallback_torch_layer`: confirmed (`core/module.py` lines 254-268).
- TT-DiT Tracer first call: compile run then capture run (two invocations): confirmed (`utils/tracing.py` lines 62-85).
- TT-Symbiote TracedRun three-phase lifecycle: warm-up (run 1) -> capture (run 2) -> replay (run 3+): confirmed (`core/run_config.py` lines 1270-1288).
- Supported models table: all six pipeline directories exist, all transformer files exist, QwenImage.md absent as noted: confirmed via directory listing.
- Directory tree: `reference/motif/`, `encoders/qwen25vl/`, `encoders/umt5/`, `pipelines/wan/pipeline_wan_i2v.py`, `models/transformers/wan2_2/`: all confirmed present.
- Mochi "824x480, 168 frames": confirmed from `models/Mochi_1.md`.
- Encoder stacks: SD3.5 and Flux1 use CLIP + T5 (confirmed from pipeline imports), Mochi uses T5 (confirmed), Wan2.2 uses UMT5 (confirmed from `encoders/umt5/`), Qwen-Image uses Qwen2.5-VL (confirmed from `encoders/qwen25vl/`).
- Attention patterns: joint spatial+prompt for SD3.5/Flux1/Motif (confirmed: `blocks/attention.py` uses `joint_scaled_dot_product_attention` and `ring_joint_scaled_dot_product_attention`); Wan2.2 uses self-attention + cross-attention (confirmed: `attention_wan.py` lines 269-270).
- Navigation links within chapter (index.md -> module_and_parameter.md, module_and_parameter.md -> comparison_with_ttnnmodule.md): all point to existing files within the chapter.

## New Finding

1. **File:** `comparison_with_ttnnmodule.md`, line 254. **Issue:** The "Next" navigation footer links to `../ch2_parallelism_and_ccl/index.md`, but the directory `ch2_parallelism_and_ccl/` does not exist yet (only `ch1_architecture_overview/` and `plan.md` exist under the guide root). This is a broken link. A reader clicking it will get a 404 or file-not-found error. **Fix:** Either remove the "Next" footer until Chapter 2 is created, or change it to a non-linked placeholder such as "**Next:** Chapter 2 -- Parallelism and CCL Infrastructure (coming soon)."
