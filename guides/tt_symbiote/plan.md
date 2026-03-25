# Research Guide Plan: TT Symbiote

## Audience

This guide targets ML engineers and researchers who:

- Are already familiar with PyTorch and standard deep-learning concepts (modules, tensors, forward passes, weight management).
- Have a basic understanding of what Tenstorrent hardware is and that it uses TTNN as its low-level kernel/operation library, but do not yet know the TT-Symbiote framework.
- Want to accelerate existing PyTorch LLM or vision models on Tenstorrent devices with minimal code changes.
- May also want to extend the framework by authoring new TTNN-backed modules or custom dispatchers.

Readers are **not** expected to know TTNN internals, TT-Symbiote source code, or distributed mesh-device programming before starting.

---

## Chapter List

### Chapter 1 — What Is TT Symbiote?
**Description:** Introduces TT-Symbiote as a transparent PyTorch-to-TTNN acceleration framework, explains the problem it solves, and maps its source-tree layout.

**Chapter directory:** `ch1_introduction`

**Files:**
- `index.md` — Chapter overview and reading guide.
- `motivation.md`
  - Why running PyTorch models on Tenstorrent hardware requires a bridging framework.
  - The fundamental tension between PyTorch's Python dispatch mechanism and TTNN's hardware-specific kernels.
  - How TT-Symbiote provides transparency: existing model code requires only a handful of setup calls, not a rewrite.
  - High-level positioning relative to tt-transformers (TT-Symbiote is model-agnostic; tt-transformers provides pre-written Tenstorrent-native models).
- `source_layout.md`
  - Walk through the repository tree: `core/`, `modules/`, `utils/`, `tests/`.
  - Purpose of each top-level subdirectory.
  - Key environment variables that control runtime behaviour (`TT_SYMBIOTE_RUN_MODE`, `TT_SYMBIOTE_DISPATCHER`, `MESH_DEVICE`).
  - List of supported device architectures: N150, N300, T3K, TG, P150, P300, P150x4, P150x8, BHGLX (from `DeviceArch` enum in `core/module.py`).

---

### Chapter 2 — Core Abstractions
**Description:** Covers the three foundational building blocks — `TTNNModule`, `TorchTTNNTensor`, and the dispatcher system — that every other part of the framework depends on.

**Chapter directory:** `ch2_core_abstractions`

**Files:**
- `index.md` — Chapter overview and how the three abstractions fit together.
- `ttnn_module.md`
  - `TTNNModule` as the base class for all TTNN-accelerated layers (`core/module.py`).
  - Lifecycle state tracked per instance: `_preprocessed_weight`, `_weights_on_device`, `_device`, `_fallback_torch_layer`, `_unique_name`, `_model_config`.
  - The three-phase weight lifecycle: `preprocess_weights()` → `move_weights_to_device()` → `deallocate_weights()`, and why the separation matters for memory-constrained hardware.
  - How the base implementations of `preprocess_weights_impl`, `move_weights_to_device_impl`, and `deallocate_weights_impl` recurse into child `TTNNModule` instances automatically.
  - The `deallocate_weights_after` decorator for automatic post-forward deallocation (used in `TTNNLinearLLama`).
  - The `run_on_devices` decorator for architecture-gated forward methods (checks `MESH_DEVICE` env var against `DeviceArch` enum).
  - `module_name` uniqueness, `__repr__` format (`ClassName(module_name=...)`), and `named_modules` / `named_children` iteration.
- `torch_ttnn_tensor.md`
  - `TorchTTNNTensor` as a `torch.Tensor` subclass (`core/tensor.py`).
  - Dual backing: `elem` (a plain `torch.Tensor`) and `ttnn_tensor` (a `ttnn.Tensor`); only one is expected to carry live data at a time.
  - `__torch_dispatch__` as the hook that intercepts every PyTorch ATen operation and routes it through the active dispatcher.
  - Properties `to_torch` and `to_ttnn` for explicit conversion.
  - `DistributedTensorConfig` attachment for mesh-sharded tensors: `mesh_mapper`, `mesh_composer`, and `logical_shape_fn`.
  - How `shape` is reported for distributed tensors vs. single-device tensors.
- `dispatcher_system.md`
  - The dispatcher as the decision engine between TTNN and PyTorch execution paths (`core/dispatcher.py`, `core/dispatchers/dispatcher_config.py`).
  - Dispatcher registry: `register_dispatcher`, `set_dispatcher`, `get_active_dispatcher`, `list_available_dispatchers`.
  - Auto-registration at import time for DEFAULT, DEBUG, CPU, and TENSOR_OPS dispatchers.
  - `TT_SYMBIOTE_DISPATCHER` environment variable taking precedence over programmatic selection; default fallback to CPU when neither is set.
  - Public interface: `can_dispatch_to_ttnn(func_name, args, kwargs)` and `dispatch_to_ttnn(func_name, args, kwargs)`.
  - How to register a custom dispatcher at runtime.

---

### Chapter 3 — Run Modes and the Dispatch Manager
**Description:** Explains the six execution modes that govern how TTNN and PyTorch paths are combined during inference, and how the `DispatchManager` tracks and records timing data.

**Chapter directory:** `ch3_run_modes_and_timing`

**Files:**
- `index.md` — Chapter overview; when to use which run mode.
- `run_modes.md`
  - Six modes controlled by `TT_SYMBIOTE_RUN_MODE` (from `core/run_config.py`):
    - `NORMAL` — full TTNN execution.
    - `NORMAL_WITH_FALLBACK` — TTNN with automatic PyTorch fallback on errors.
    - `SEL` (Segment Each Layer) — PyTorch receives TTNN tensors and results are compared with PCC.
    - `DPL` (Debug Per Layer) — TTNN and PyTorch run in parallel; outputs compared with PCC.
    - `DPL_NO_ERROR_PROP` — DPL variant where TTNN takes PyTorch tensors as input to prevent error propagation.
    - `CPU` — CPU-only execution for baseline validation.
  - `@trace_enabled` and `@trace_disabled` class decorators on `TTNNLinear` variants and their effect on tracing.
  - `no_dispatch()` context manager for preventing infinite recursion when falling back to native PyTorch ops.
  - Helper transforms: `unwrap_to_torch`, `wrap_from_torch`, `copy_to_torch`, `copy_to_ttnn`, `set_device_wrap`, `create_new_ttnn_tensors_using_torch_output`.
- `dispatch_manager.md`
  - `DispatchManager` as a class-level (singleton-style) timing recorder.
  - `set_current_module_name` stack mechanism for attributing ATen ops to their enclosing module.
  - `dispatch_to_ttnn_wrapper` and `dispatch_to_torch_wrapper` and how they record backend, module name, function name, and elapsed time.
  - Saving timing data: `DispatchManager.save_stats_to_file(filename.csv)` produces a flat CSV and a pivot table CSV.
  - How to read timing output to identify bottleneck layers.

---

### Chapter 4 — Module Replacement and Device Setup
**Description:** Walks through the two setup functions that transform a stock PyTorch model into a TTNN-accelerated one: `register_module_replacement_dict` and `set_device`.

**Chapter directory:** `ch4_module_replacement_and_device`

**Files:**
- `index.md` — Chapter overview; typical setup sequence.
- `module_replacement.md`
  - `register_module_replacement_dict(model, nn_to_ttnn_dict, model_config, exclude_replacement)` as the entry point (`utils/module_replacement.py`).
  - Internal recursive traversal: handles `nn.Module._modules`, dict attributes, and list/tuple attributes.
  - `initialize_module`: calls `from_torch(old_module)`, sets `_unique_name` from the global name map, and calls `set_model_config`.
  - `exclude_replacement` set of module-name strings; how to discover names before excluding them.
  - Return value: a dict mapping module names to the freshly created `TTNNModule` instances (used to drive weight preprocessing).
  - Selective replacement patterns: replace all except one problematic layer; mix PyTorch and TTNN layers in the same model.
- `device_setup.md`
  - `set_device(model, ttnn_device, device_init, **kwargs)` recursive traversal (`utils/device_management.py`).
  - `DeviceInit` class: `init_state` caches a `DistributedConfig` per device, calling `DistributedConfig(device)` which auto-creates `ShardTensor2dMesh` mapper/composer and `TT_CCL` for multi-device meshes.
  - Forward-hook registration: every `nn.Module`'s `forward` is wrapped with `timed_call` that calls `DispatchManager.set_current_module_name` for attribution.
  - Enabling/disabling the visualization dump via `dump_visualization` kwarg (calls `draw_model_graph`).
  - Multi-device initialization: `set_device_state` is called on each `TTNNModule` when `device.get_num_devices() > 1`.
  - Weight preprocessing after `set_device`: iterating the returned module dict, calling `preprocess_weights()` then `move_weights_to_device()` on each module.

---

### Chapter 5 — Built-In TTNN Modules
**Description:** Surveys every production-ready TTNN module that ships with the framework, grouped by operation type, with emphasis on the linear variants used for LLM acceleration.

**Chapter directory:** `ch5_builtin_modules`

**Files:**
- `index.md` — Chapter overview and full catalogue table.
- `linear_layers.md`
  - `TTNNLinear` — bfloat16 weights; uses `preprocess_linear_weight`/`preprocess_linear_bias`; DRAM memory config; auto-pads input to 4-D before `ttnn.linear`.
  - `TTNNLinearLLama` — bfloat8_b weights; `@trace_disabled`; `@deallocate_weights_after` on `forward` for LLM memory efficiency.
  - `TTNNLinearLLamaBFloat16` — bfloat16 LLaMA variant with weight auto-deallocation.
  - `TTNNLinearGelu` / `TTNNLinearSilu` / `TTNNLinearActivation` — fused linear + activation patterns.
  - Sharded variants: `TTNNLinearInputShardedWeightSharded`, `TTNNLinearIColShardedWRowSharded`, `TTNNLinearInputReplicatedWeightSharded`, `TTNNLinearIReplicatedWColSharded` — T3K multi-device column/row tensor parallelism with `reduce_scatter_minimal_async`.
  - `TTNNLinearLLamaIColShardedWRowSharded` — combined bfloat8 + sharding + auto-deallocation.
- `normalization_and_activation.md`
  - `TTNNLayerNorm` and `TTNNRMSNorm` — normalisation layers and their weight preprocessing.
  - `TTNNSilu`, `TTNNReLU`, `TTNNGelu` — elementwise activations dispatching to `ttnn.silu`, `ttnn.relu`, `ttnn.gelu`.
- `attention_and_conv.md`
  - Attention modules: `TTNNViTSelfAttention` (deprecated), `TTNNSelfAttention`, `TTNNSDPAAttention`, `TTNNFusedQKVSelfAttention`, `TTNNWhisperAttention`, `LlamaAttention`.
  - Convolution modules: `TTNNConv2dNHWC`, `TTNNConv2dBNNHWC`, `TTNNConv2dBNActivationNHWC`, `TTNNBottleneck`, `TTNNMaxPool2dNHWC`, `TTNNUpsampleNHWC`.
  - Embedding modules: `TTNNPatchEmbedding`, `TTNNViTEmbeddings`.
  - Tensor-op modules: `TTNNPermute`, `TTNNReshape`, `TTNNAdd`.

---

### Chapter 6 — Authoring a New TTNN Module
**Description:** Provides a step-by-step guide to implementing a custom `TTNNModule`, covering the factory pattern, the weight lifecycle, fallback semantics, and device-arch gating.

**Chapter directory:** `ch6_authoring_modules`

**Files:**
- `index.md` — Chapter overview; when to write a custom module vs. composing existing ones.
- `implementation_guide.md`
  - Inherit from `TTNNModule`; import from `core/module.py`.
  - Implement `from_torch(cls, torch_layer)`: create instance, store `_fallback_torch_layer`, copy any configuration parameters.
  - Implement `preprocess_weights_impl()`: convert PyTorch tensors to TTNN format with appropriate dtype and layout; store on host as `*_host` attributes.
  - Implement `move_weights_to_device_impl()`: call `ttnn.to_device` with `self.device`; store result as device-side attributes.
  - Implement `deallocate_weights_impl()`: call `ttnn.deallocate` on all device-side weight tensors; call `super().deallocate_weights_impl()` to recurse into children.
  - Implement `forward(*args, **kwargs)`: operate directly on TTNN tensors received from `TorchTTNNTensor.to_ttnn`; return a TTNN tensor or `TorchTTNNTensor`.
  - Using `@deallocate_weights_after` when device memory is limited.
  - Using `@run_on_devices(DeviceArch.T3K, ...)` to guard architecturally-specific forward implementations.
  - Registering the new module in a replacement dict and running the standard setup sequence.
- `fallback_and_debugging.md`
  - How `_fallback_torch_layer` is used in `NORMAL_WITH_FALLBACK` and `DPL` modes.
  - Using `TT_SYMBIOTE_RUN_MODE=DPL_NO_ERROR_PROP` to isolate a single module's error.
  - Using `TT_SYMBIOTE_DISPATCHER=DEBUG` for verbose per-operation logging.
  - PCC (Pearson Correlation Coefficient) comparison between TTNN and PyTorch outputs in SEL/DPL modes.
  - Reading `DispatchManager` timing output to confirm the new module's operations are being dispatched to TTNN rather than falling back to PyTorch.

---

### Chapter 7 — End-to-End Use Cases and Performance Benchmarking
**Description:** Demonstrates complete integration workflows for several model families (LLM, vision, speech), explains how to read timing CSV output, and summarises observed performance characteristics.

**Chapter directory:** `ch7_use_cases_and_benchmarking`

**Files:**
- `index.md` — Chapter overview; how to choose a reference test as a starting point.
- `llm_acceleration.md`
  - LLaMA-3.2-1B-Instruct end-to-end walkthrough based on `tests/test_llama.py`.
  - Pre-replacement step: wrapping `LlamaMLP` to expose individual gate/up/down projections as distinct `nn.Linear` layers before TTNN replacement.
  - Replacement mapping: `nn.Linear` → `TTNNLinear`, `nn.SiLU` → `TTNNSilu`, RMSNorm → `TTNNRMSNorm`, self-attention → `LlamaAttention`.
  - `exclude_replacement={"lm_head"}` pattern for keeping the language model head on CPU.
  - `SmartTTNNLinear` (`linear_intelligent.py`) as an adaptive linear that selects an optimal TTNN variant at runtime.
  - Saving and interpreting `llama_timing_stats.csv` and `llama_timing_stats_pivot.csv`.
  - GPT-OSS-20B and GLM-4.5-Air as examples of larger-scale LLM acceleration.
- `vision_and_multimodal.md`
  - ViT (Vision Transformer) with `TTNNLinear` + `TTNNLayerNorm` replacement; `TTNNPatchEmbedding` and `TTNNViTEmbeddings`.
  - ResNet50 with `TTNNConv2dBNActivationNHWC` and `TTNNBottleneck`; NHWC layout implications.
  - OWL-ViT object detection and YUNet face detection as specialised vision models.
  - OpenVLA-7B (vision-language-action) as an example of multimodal acceleration.
  - HunyuanVideo 1.5 (text-to-video) as a diffusion/video generation use case.
- `speech_and_debugging.md`
  - SpeechT5 and Whisper-large-v3 as speech model examples; `TTNNWhisperAttention`.
  - Using `TT_SYMBIOTE_DISPATCHER=CPU && TT_SYMBIOTE_RUN_MODE=DPL_NO_ERROR_PROP` combination for speech model debugging.
  - `tests/test_dpl.py` as the canonical Debug Per Layer reference test.
  - Interpreting timing pivot tables: comparing TTNN vs. Torch column totals, identifying layers that still fall back to PyTorch.

---

## Conventions

**Terminology:**

| Term | Meaning in this guide |
|---|---|
| TTNN | Tenstorrent's low-level neural-network operation library |
| TT-Symbiote / TT Symbiote | The PyTorch-to-TTNN acceleration framework documented here |
| tt-transformers | Separate Tenstorrent library of pre-written, hardware-native transformer model implementations |
| `TTNNModule` | The abstract base class all TTNN-accelerated layers inherit from |
| `TorchTTNNTensor` | The `torch.Tensor` subclass that carries both a PyTorch and a TTNN backing tensor |
| dispatcher | A module that implements `can_dispatch_to_ttnn` and `dispatch_to_ttnn`; selected via `TT_SYMBIOTE_DISPATCHER` |
| run mode | The execution strategy selected via `TT_SYMBIOTE_RUN_MODE` |
| PCC | Pearson Correlation Coefficient; used to compare TTNN vs PyTorch numerical outputs |
| mesh device | A multi-chip Tenstorrent device abstracted as a 2D mesh (e.g., T3K = 8 chips) |
| host / device | "host" = CPU-side memory; "device" = on-chip SRAM/DRAM on the Tenstorrent card |
| bfloat8_b | Tenstorrent's 8-bit bfloat format; lower memory than bfloat16, used in LLM linear layers |

**Code notation:**

- All source paths are given relative to the repository root `tt-metal/` (e.g., `models/experimental/tt_symbiote/core/module.py`).
- Python class names use `CamelCase`; environment variable names use `SCREAMING_SNAKE_CASE`.
- Shell command examples use `export VAR=VALUE && pytest ...` to show variable and command together on one line.
- Method signatures are written in Python style with type annotations where they exist in the source.

**Formatting rules:**

- Each file begins with a one-sentence summary of its purpose.
- Code blocks use the `python` or `bash` language tag.
- Information boxes use blockquote syntax (`>`) prefixed with **Note:** or **Warning:** for callouts.
- Tables are used for catalogues of modules, environment variables, or run modes; prose is used for explanations.
- Section headers within a file use `##` (H2) or `###` (H3); the file title itself is `#` (H1).
- No emojis.

---

## Cross-Chapter Dependencies

| Chapter | Directly depends on concepts introduced in |
|---|---|
| Ch 2 — Core Abstractions | Ch 1 (source layout, device architecture enum, env-var names) |
| Ch 3 — Run Modes and Dispatch Manager | Ch 2 (`TorchTTNNTensor.__torch_dispatch__`, dispatcher system, `TTNNModule.__call__`) |
| Ch 4 — Module Replacement and Device Setup | Ch 2 (`TTNNModule` lifecycle, `from_torch` factory), Ch 3 (`DispatchManager` forward hooks and timing) |
| Ch 5 — Built-In TTNN Modules | Ch 2 (`TTNNModule` base, weight lifecycle methods), Ch 4 (`register_module_replacement_dict`, `set_device`) |
| Ch 6 — Authoring a New TTNN Module | Ch 2 (full `TTNNModule` contract), Ch 3 (run-mode fallback semantics), Ch 4 (replacement dict registration) |
| Ch 7 — Use Cases and Benchmarking | Ch 4 (setup sequence), Ch 5 (which built-in modules to use per task), Ch 6 (custom modules like `LlamaMLP` wrapper), Ch 3 (`DispatchManager.save_stats_to_file`) |
