# Plan: Deploying a TT Symbiote Model on tt-inference-server

## Audience

**Target readers:** ML engineers and systems engineers who have written a model using the TT Symbiote framework (`TTNNModule` subclasses, TTNN-backed forward passes) and now want to expose it as a production inference endpoint through `tt-inference-server`.

**Assumed knowledge:**
- Familiarity with PyTorch `nn.Module` and the HuggingFace Transformers API
- Basic knowledge of TT Symbiote internals: `TTNNModule`, `TorchTTNNTensor`, the dispatcher system, and the weight lifecycle (`preprocess_weights` → `move_weights_to_device`)
- Awareness of TTNN mesh devices (`ttnn.open_mesh_device`) and multi-chip device types (N150, N300, T3K, Galaxy)
- Basic understanding of vLLM as an LLM serving framework (continuous batching, OpenAI-compatible API)

**Not assumed:**
- Prior knowledge of `tt-inference-server`'s internal code, `ModelSpec` system, or Docker-based deployment workflow
- Experience writing custom vLLM model backends or extending vLLM's `Platform` abstraction
- Familiarity with the `tt-vllm-plugin` or `generator_vllm.py` integration layer in `tt-metal`

---

## Chapter List

### Chapter 1 — tt-inference-server Architecture Overview

**Description:** Maps the full `tt-inference-server` stack from top-level CLI entry point down to the TT-Metal model execution layer, establishing the mental model needed before touching any integration code.

**Directory:** `ch1_architecture_overview/`

**Files:**

- `index.md`
  - Chapter overview and reading order
  - One-paragraph answer to: "what is tt-inference-server?" — it is a workflow orchestrator and serving wrapper that runs vLLM (or SGLang) as the core inference engine, behind Docker containers, with a `ModelSpec`-based configuration system and an OpenAI-compatible HTTP API
  - Summary table of the six major subsystems: CLI (`run.py`), `ModelSpec` registry (`workflows/model_spec.py`), workflow orchestrator (`workflows/run_workflows.py`), `tt-vllm-plugin` (Tenstorrent's vLLM platform plugin), `TTPlatform` (hardware abstraction registered with vLLM), and `TTModelLoader` (model instantiation gate)
  - What's next: Chapter 2 covers the `ModelSpec` configuration schema

- `serving_stack.md`
  - The three-layer serving stack: (1) vLLM OpenAI API server (`vllm.entrypoints.openai.api_server`), (2) `TTPlatform` vLLM platform plugin, (3) the model implementation (e.g., `LlamaForCausalLM` in `generator_vllm.py`)
  - How vLLM is launched: `runpy.run_module("vllm.entrypoints.openai.api_server")` after prepending device-specific vLLM arguments from `DeviceModelSpec.vllm_args` to `sys.argv`
  - `VLLM_USE_V1=1` requirement: the tt-vllm-plugin only supports vLLM v1 architecture; the `TTPlatform.supports_v1()` method enforces this
  - What `TTPlatform` does: inherits from vLLM's `Platform` base class; renames model architecture strings from `"LlamaForCausalLM"` to `"TTLlamaForCausalLM"` to force lookup of TT-specific model classes; disables chunked prefill, prefix caching, and speculative decoding (not yet supported); enforces `n=1` sampling, no `best_of`, and no `prompt_logprobs`
  - `TTModelLoader`: the vLLM model loader that replaces the standard `DefaultModelLoader`; validates that the resolved model class implements `initialize_vllm_model()`; calls it with `hf_config`, `mesh_device`, `max_batch_size`, `max_seq_len`, `tt_data_parallel`, and `optimizations`
  - How device setup flows: `open_mesh_device()` in `tt_worker.py` reads the `MESH_DEVICE` environment variable, maps it to a grid shape (e.g., `"T3K"` → `(1, 8)`), configures fabric topology, and calls `ttnn.open_mesh_device()`; this mesh device handle is what gets passed to `initialize_vllm_model`
  - Tokenization ownership: vLLM owns tokenization entirely via HuggingFace `AutoTokenizer` loaded from the `--model` path; the model implementation never handles raw text

- `request_lifecycle.md`
  - End-to-end flow of a single completion request: HTTP POST → vLLM OpenAI API server → scheduler → `TTWorker` → `TTModelRunner` → model's `prefill_forward()` / `decode_forward()` → sampled token → HTTP response
  - Role of the vLLM KV cache block manager: allocates fixed-size blocks (default block size = 16 or 64 tokens) from pre-allocated TTNN KV cache tensors; passes a `block_table` (page table) to each forward call
  - How `TTModelInput` carries per-step state: `input_ids`, `input_positions`, `prompt_lens`, `block_tables`, `batch_size`, `max_seq_len`, and `TTSamplingParams` (temperature, top_k, top_p)
  - Prefill vs. decode distinction: prefill processes the full prompt in one or more forward calls; decode generates one token per step with a page table pointing into the cached KV blocks
  - On-device sampling: when enabled, logits computation and token sampling are captured as separate TTNN traces to eliminate Python overhead during decode

---

### Chapter 2 — The ModelSpec Configuration System

**Description:** Explains the `ModelSpec`/`ModelSpecTemplate` schema that controls which models are available, which hardware devices they run on, and how vLLM server parameters are derived from that schema.

**Directory:** `ch2_model_spec/`

**Files:**

- `index.md`
  - Chapter overview
  - Why this chapter matters: every model deployed through `tt-inference-server`'s workflow automation must have a `ModelSpec` entry; understanding the schema is the first concrete integration step
  - What's next: Chapter 3 covers the `initialize_vllm_model` contract that the model class must satisfy

- `schema_reference.md`
  - `ImplSpec` dataclass: `impl_id` (string identifier, e.g., `"tt_transformers"`), `impl_name`, `repo_url`, `code_path` — for a TT Symbiote model this would point to the symbiote repo and the module path
  - `DeviceModelSpec` dataclass fields: `device` (a `DeviceTypes` enum value: N150, N300, T3K, Galaxy, P100, P150, P150X4, P150X8, P300, etc.), `max_concurrency` (max batch size), `max_context` (max sequence length in tokens), `vllm_args` (dict of extra CLI args appended to vLLM launch), `override_tt_config` (dict passed to model init for precision/performance tuning), `env_vars` (per-device environment variable overrides)
  - `ModelSpecTemplate` fields: `weights` (list of HuggingFace model repo strings from which `ModelSpec` instances are expanded), `impl` (an `ImplSpec`), `tt_metal_commit`, `vllm_commit`, `inference_engine` (`InferenceEngine.VLLM`), `device_model_specs` (list of `DeviceModelSpec`)
  - `ModelSpec` (the expanded, per-weight-per-device instance): `model_id`, `hf_model_repo`, `model_name`, `device_type`, `device_model_spec`, `env_vars`, `uses_tensor_model_cache`, `has_builtin_warmup`, `docker_image`, `status`
  - `ModelStatusTypes`: Experimental, Functional, Complete — set automatically when benchmark targets are met
  - How `TT_MODEL_SPEC_JSON_PATH` is used: the vLLM server startup script reads this environment variable to locate the JSON-serialized `ModelSpec` for the current deployment; it extracts `vllm_args`, `override_tt_config`, and hardware mesh settings

- `adding_a_model_spec.md`
  - Step-by-step instructions for registering a new TT Symbiote model: define an `ImplSpec` pointing to the symbiote codebase, create a `ModelSpecTemplate` with the target HF model repo, define a `DeviceModelSpec` per supported device type with appropriate `max_concurrency` and `max_context` values
  - How `vllm_args` maps to vLLM CLI flags: `"max-model-len"`, `"max-num-seqs"`, `"block-size"`, `"max-num-batched-tokens"`, `"enable-chunked-prefill"` (must be `"false"` for TT hardware)
  - How `override_tt_config` keys are consumed: the `TTModelLoader` extracts `"optimizations"` (valid values: `"performance"` or `"accuracy"`); the `tt_worker.py` functions extract `"is_embedding_model"`, `"trace_region_size"`, `"num_command_queues"`, `"worker_l1_size_bytes"`
  - Naming conventions for `model_id`: lowercase, hyphen-separated, include device suffix (e.g., `"my-symbiote-model-8b-n300"`)
  - Required environment variables for first deployment: `HF_TOKEN` (for weight download), `JWT_SECRET` (for API key generation), `SERVICE_PORT` (defaults to 8000)

- `workflow_commands.md`
  - How `run.py` uses the `ModelSpec`: `python3 run.py --model <model_name> --device <device> --workflow server` resolves the `ModelSpec` by name, generates the `TT_MODEL_SPEC_JSON_PATH` file, and launches the Docker container with the correct image and environment
  - Alternative: direct plugin invocation without Docker for development — install `tt-vllm-plugin` into the tt-metal Python environment, set `VLLM_USE_V1=1` and `HF_MODEL=<hf_repo>`, then run `vllm serve <hf_repo> --max-model-len <N> --max-num-seqs <B> --block-size 64`
  - Workflow types: `server` (deploy only), `benchmarks` (perf sweep), `evals` (accuracy), `reports` (generate markdown summary)
  - Output directories: `docker_server/`, `benchmarks_output/`, `evals_output/`, `run_logs/` — all timestamped and config-tagged

---

### Chapter 3 — The Model Implementation Contract

**Description:** Defines the exact Python interface a model class must satisfy to be loadable by `TTModelLoader`, covering class structure, required methods, their signatures, and return types.

**Directory:** `ch3_model_contract/`

**Files:**

- `index.md`
  - Chapter overview
  - The two-sentence summary of the contract: the model class must (1) be registered in vLLM's `ModelRegistry` under a `"TT<ArchName>"` key and (2) implement a `@classmethod initialize_vllm_model()` that constructs and returns a ready-to-run model instance
  - What's next: Chapter 4 covers weight loading and tokenization

- `class_registration.md`
  - How `TTPlatform.get_model_cls()` transforms the HuggingFace architecture name: it prepends `"TT"` to the string found in `config.json`'s `"architectures"` field (e.g., `"LlamaForCausalLM"` → `"TTLlamaForCausalLM"`)
  - How to register the class: call `ModelRegistry.register_model("TTMySymbioteModel", MySymbioteModelClass)` from within the `register_tt_models()` function that the server startup script invokes before launching vLLM
  - Where `register_tt_models()` lives: `tt-vllm-plugin/tt_vllm_plugin/model_loader/tt_loader.py`; it is the appropriate extension point for adding a new model family
  - Why the `"TT"` prefix: it prevents collision with vLLM's built-in model registry entries and signals to `TTModelLoader` that the class was explicitly provided for TT hardware

- `initialize_vllm_model.md`
  - Full method signature:
    ```python
    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,         # transformers.PretrainedConfig loaded from the HF model repo
        mesh_device,       # ttnn.MeshDevice already opened by tt_worker.py
        max_batch_size,    # int; derived from DeviceModelSpec.max_concurrency × tt_data_parallel
        max_seq_len,       # int; derived from DeviceModelSpec.max_context
        n_layers=None,     # int or None; allows partial-layer loading for testing
        tt_data_parallel=1,# int; data parallel degree
        optimizations: str = "performance",  # "performance" or "accuracy"
    ) -> "cls":
    ```
  - What the method must do: initialize all TTNN tensors and model weights on the provided `mesh_device`, and return a fully initialized instance of `cls` ready to receive `prefill_forward()` and `decode_forward()` calls
  - The mesh device is already open: `open_mesh_device()` has been called by the worker before `initialize_vllm_model`; the model must NOT call `ttnn.open_mesh_device()` or `ttnn.close_mesh_device()` — device lifecycle is owned by the worker
  - The `optimizations` parameter maps to the `override_tt_config["optimizations"]` value from `DeviceModelSpec`; use it to select between bfloat16/bfloat8 precision, HiFi4/HiFi2 math fidelity, and trace capture settings
  - For TT Symbiote models: `initialize_vllm_model` should call `model.preprocess_weights()`, `model.move_weights_to_device(mesh_device)`, and any warmup sweeps, then return the model instance

- `forward_interface.md`
  - `prefill_forward(tokens, prompt_lens, page_table, kv_cache, ...)` — processes the prompt; `tokens` is a 1D CPU tensor of token IDs, `prompt_lens` is a list of ints, `page_table` is an optional 2D CPU tensor mapping sequence positions to KV cache block indices; returns logits tensor
  - `decode_forward(tokens, page_table, kv_cache, ...)` — processes one token per active sequence; `tokens` is shape `(batch_size, 1)` on CPU, `page_table` is `(batch_size, max_blocks_per_seq)` on CPU; returns logits tensor
  - `allocate_kv_cache(num_blocks, block_size, num_kv_heads, head_dim, dtype, ...)` — allocates and returns TTNN KV cache tensors; called once during worker initialization; the returned tensors are held by the worker and passed to each forward call
  - KV cache tensor shape conventions: each cache is typically `(num_blocks, block_size, num_kv_heads, head_dim)` or transposed variant depending on the paged attention kernel; must match the kernel's expectations
  - Return types: `prefill_forward` and `decode_forward` must return logits as a CPU or device tensor with shape `(batch_size, vocab_size)`; the worker samples the next token from these logits

- `constraints.md`
  - Chunked prefill is disabled by `TTPlatform.check_and_update_config()`; the model does not need to handle chunked prefill
  - Prefix caching is disabled; the model does not need to implement prefix KV cache reuse
  - Sampling constraints: only `n=1` is supported; no `best_of`; no `prompt_logprobs`
  - Tensor parallelism: `TTPlatform` enforces single-process execution (no distributed vLLM worker groups); multi-chip parallelism is handled entirely inside the model via TTNN mesh operations, not by vLLM's tensor parallel infrastructure
  - Block size: defaults to 16 in vLLM but `TTPlatform` overrides it to 64 for TT hardware; the paged attention kernel in TTNN expects this block size; models must not hardcode a different block size
  - Pinned memory: `TTPlatform.is_pin_memory_available()` returns `False`; do not request pinned memory in any model initialization path
  - Data-parallel batch gathering: `TTPlatform.requires_gathered_batch_dp()` returns `True`; vLLM will gather batches across data-parallel ranks before calling the model

---

### Chapter 4 — Weight Loading and Tokenization

**Description:** Explains how `tt-inference-server` discovers and delivers model weights to the model implementation, and how tokenization is handled so the model author does not need to provide a custom tokenizer.

**Directory:** `ch4_weights_and_tokenization/`

**Files:**

- `index.md`
  - Chapter overview
  - Key insight: tt-inference-server delegates weight loading to the model's own `initialize_vllm_model()` method; the model is responsible for reading its own weights; tokenization is fully owned by vLLM via HuggingFace
  - What's next: Chapter 5 covers hardware initialization and device setup

- `weight_discovery.md`
  - Default weight source: HuggingFace Hub; `setup_host.py` calls `huggingface-cli download <hf_repo>` using `HF_TOKEN` and places files in a default directory structure under `/home/<user>/`
  - How the model receives the weight path: `hf_config` passed to `initialize_vllm_model` is a `transformers.PretrainedConfig` object loaded from the downloaded model directory; the path to the directory can be recovered via `hf_config._name_or_path`
  - HuggingFace checkpoint format: tt-inference-server always downloads standard HuggingFace-format checkpoints (`.safetensors` or `.bin` shards); the model implementation is responsible for loading them — via `torch.load`, `safetensors.torch.load_file`, or `ttnn.as_tensor` with a cache file
  - Custom weight formats: if the TT Symbiote model requires pre-converted weights (e.g., bfloat8 tiles, DRAM-sharded layout), the conversion must happen inside `initialize_vllm_model` or be pre-computed and stored alongside the HF checkpoint; `tt-inference-server` has no built-in conversion pipeline
  - `uses_tensor_model_cache` flag in `ModelSpec`: when `True`, the server expects a side-channel cache directory for pre-converted TTNN weight tensors; `setup_host.py` creates this directory and passes it via an environment variable; use this flag for models that cache converted weights to disk
  - `has_builtin_warmup` flag: when `True`, the server does not inject warmup HTTP requests before opening the API; set this when `initialize_vllm_model` already runs warmup trace captures internally
  - `TTModelLoader.download_model()` and `load_weights()` raise `NotImplementedError` — the TT loader deliberately does not use vLLM's standard weight-loading pipeline; all weight management is inside `initialize_vllm_model`

- `tokenization.md`
  - vLLM owns tokenization: the vLLM server loads `AutoTokenizer.from_pretrained(hf_model_path)` and tokenizes all input text before dispatching to the model; the model only ever sees integer token IDs
  - No tokenizer code required in the model implementation: unlike some serving frameworks, the TT model class does not need to implement or configure a tokenizer
  - HuggingFace tokenizer discovery: vLLM resolves the tokenizer from the same `--model` path used to load `config.json`; for TT Symbiote models backed by a standard HF model repo, the tokenizer is already present in the checkpoint
  - Custom tokenizer scenario: if the TT Symbiote model uses a vocabulary different from any HuggingFace model, a custom `tokenizer_config.json` and vocabulary files must be placed in the checkpoint directory before deployment; pass `--tokenizer <path>` in `DeviceModelSpec.vllm_args` to override the default path
  - Special tokens: `eos_token_id`, `pad_token_id`, and `bos_token_id` are read from `config.json` or `tokenizer_config.json`; ensure these are set correctly to avoid generation truncation errors

---

### Chapter 5 — Hardware Initialization and Device Ownership

**Description:** Clarifies the boundary between what `tt-inference-server` sets up on behalf of the model (mesh device, fabric, environment) and what the model implementation is responsible for inside `initialize_vllm_model`.

**Directory:** `ch5_hardware_init/`

**Files:**

- `index.md`
  - Chapter overview
  - The single most important rule: the mesh device is opened by the server worker before `initialize_vllm_model` is called, and closed by the worker after the server shuts down; the model must not open or close the device
  - What's next: Chapter 6 covers the full integration checklist

- `device_lifecycle.md`
  - Who calls `ttnn.open_mesh_device()`: `open_mesh_device()` in `tt-vllm-plugin/tt_vllm_plugin/worker/tt_worker.py`, called during `TTWorker` initialization
  - How the mesh grid is determined: the `MESH_DEVICE` environment variable is read by `get_mesh_grid()` in `tt_worker.py`; accepted values are device type strings (`"N150"`, `"N300"`, `"T3K"`) or tuple strings (`"(1,8)"`); this maps to `ttnn.MeshShape(rows, cols)` passed to `ttnn.open_mesh_device()`
  - Fabric configuration: `set_fabric()` is called before device open to configure the inter-chip fabric topology (1D ring for 6U Galaxy clusters, standard 1D otherwise); `reset_fabric()` is called on shutdown
  - Dispatch core axis: controlled by `override_tt_config["dispatch_core_axis"]` (values: `"row"` or `"col"`); passed as `DispatchCoreConfig` to `ttnn.open_mesh_device()`
  - Trace region size: controlled by `override_tt_config["trace_region_size"]`; passed to `ttnn.open_mesh_device()` to pre-allocate trace buffer memory
  - `TT_VISIBLE_DEVICES` environment variable: restricts which physical Tenstorrent devices are visible to the process; set by the server workflow; the model must not modify it

- `model_init_responsibilities.md`
  - What the model MUST do inside `initialize_vllm_model`: load weights onto the provided `mesh_device`, initialize any TTNN tensor constants (cosine/sine matrices for RoPE, etc.), run any compilation warmup passes, and return the initialized model instance
  - What the model MUST NOT do: call `ttnn.open_mesh_device()`, call `ttnn.close_mesh_device()`, call `ttnn.synchronize_device()` (unless required for correctness during init), modify `TT_VISIBLE_DEVICES`, or import and call `set_fabric()` directly
  - For TT Symbiote models specifically: call `model.preprocess_weights()` (host-side conversion), then `model.move_weights_to_device(mesh_device)` (upload to device), then any warmup sweeps; `deallocate_weights` should not be called during init — deallocation is deferred to the per-forward decorator pattern
  - Multi-chip TT Symbiote models: the `mesh_device` is already a multi-chip mesh (e.g., 1×8 for T3K); pass it to `TTNNModule.set_device_state()` (or the equivalent `DeviceState` / `DistributedConfig` setup) before calling `move_weights_to_device`
  - KV cache allocation: `allocate_kv_cache()` is called by the worker separately from `initialize_vllm_model`; do not pre-allocate KV cache inside `initialize_vllm_model` — let the worker call `allocate_kv_cache` with the exact block count it has negotiated with vLLM's block manager

- `environment_variables_reference.md`
  - Complete reference table of all environment variables relevant to a TT Symbiote model deployment:

  | Variable | Set by | Purpose |
  |---|---|---|
  | `VLLM_USE_V1` | operator / serve.sh | Must be `"1"` for tt-vllm-plugin |
  | `HF_MODEL` | operator / serve.sh | HuggingFace model repo used as `--model` arg |
  | `HF_TOKEN` | operator | HuggingFace auth token for gated model downloads |
  | `JWT_SECRET` | operator | Seeds the vLLM API key |
  | `SERVICE_PORT` | operator | HTTP port (default 8000) |
  | `MESH_DEVICE` | workflow / operator | Device type string: `"N150"`, `"N300"`, `"T3K"`, `"(1,8)"`, etc. |
  | `TT_VISIBLE_DEVICES` | workflow | Comma-separated physical device IDs |
  | `TT_MODEL_SPEC_JSON_PATH` | workflow | Path to JSON-serialized `ModelSpec` for current deployment |
  | `VLLM_API_KEY` | workflow | Derived from `JWT_SECRET` if not set |
  | `TT_LLAMA_TEXT_VER` | server startup | Selects Llama implementation variant (set by `register_tt_models`) |
  | `TT_QWEN3_TEXT_VER` | server startup | Selects Qwen3 implementation variant |
  | `TT_SYMBIOTE_RUN_MODE` | model / operator | TT Symbiote run mode (NORMAL, DPL, SEL, CPU) |
  | `TT_SYMBIOTE_DISPATCHER` | model / operator | Active dispatcher selection |

---

### Chapter 6 — Integration Checklist and Worked Example

**Description:** Provides a concrete, step-by-step recipe for integrating a TT Symbiote model into `tt-inference-server`, followed by a minimal worked example showing all required pieces in one place.

**Directory:** `ch6_integration_checklist/`

**Files:**

- `index.md`
  - Chapter overview and how to use the checklist
  - Prerequisites before starting: a TT Symbiote model that passes correctness tests in `NORMAL` mode on the target hardware; a HuggingFace model repo (or local directory) that contains `config.json` and tokenizer files; a tt-metal Python environment with `tt-vllm-plugin` installed
  - What's next: Chapter 7 covers debugging and performance tuning

- `integration_steps.md`
  - **Step 1 — Verify the architecture string:** check `config.json` for the `"architectures"` field (e.g., `["LlamaForCausalLM"]`); the TT Symbiote class must be registered under `"TT" + architecture_name`
  - **Step 2 — Implement `initialize_vllm_model`:** add a `@classmethod initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len, ...)` to the TT Symbiote model class (or to a thin vLLM adapter class wrapping it); load weights, move to device, run warmup, return instance
  - **Step 3 — Implement `prefill_forward` and `decode_forward`:** add these methods; they receive CPU tensors for `tokens` and `page_table`, and TTNN KV cache tensors; they must return logits as a tensor (CPU or TTNN-on-device)
  - **Step 4 — Implement `allocate_kv_cache`:** allocate and return per-layer KV cache tensors with shape matching the paged attention kernel's expectations; use `block_size=64` as the default for TT hardware
  - **Step 5 — Register the class:** call `ModelRegistry.register_model("TTMyModel", MyModelClass)` inside `register_tt_models()` in `tt_loader.py`; add a conditional branch keyed on an `impl_id` or environment variable
  - **Step 6 — Define a `ModelSpec`:** add a `ModelSpecTemplate` to `workflows/model_spec.py` with the HF repo, `ImplSpec` pointing to the symbiote codebase, and at least one `DeviceModelSpec`
  - **Step 7 — Test with direct plugin invocation:** set `VLLM_USE_V1=1` and `HF_MODEL=<repo>`, run `vllm serve <repo> --max-model-len <N> --max-num-seqs <B> --block-size 64` in the tt-metal Python environment; test the `/health` endpoint and a `POST /v1/completions` request
  - **Step 8 — Test with workflow automation:** run `python3 run.py --model <model_name> --device <device> --workflow server` from the tt-inference-server repo root; verify Docker container starts and the health check passes

- `worked_example.md`
  - Minimal worked example: a hypothetical `TTSymbioteTextModel` wrapping a small causal LM built with TT Symbiote modules
  - The `config.json` assumption: architecture field is `["SymbioteTextModel"]`; the registration key is `"TTSymbioteTextModel"`
  - Complete skeleton of the vLLM adapter class:
    - `@classmethod initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len, ...)`: constructs a `TTSymbioteTextModel`, calls `preprocess_weights()`, `move_weights_to_device(mesh_device)`, returns the instance
    - `prefill_forward(self, tokens, prompt_lens, page_table, kv_cache)`: calls into the model's `forward()` with `mode=PREFILL`, page table broadcast; returns logits
    - `decode_forward(self, tokens, page_table, kv_cache)`: calls into the model's `forward()` with `mode=DECODE`; returns logits
    - `allocate_kv_cache(self, num_blocks, block_size, ...)`: allocates `ttnn.Tensor` KV blocks per layer; returns dict keyed by layer index
  - Registration snippet inside `register_tt_models()` in `tt_loader.py`
  - `ModelSpecTemplate` definition for the example model targeting N150 and N300
  - Common mistakes: forgetting `VLLM_USE_V1=1`; registering under the wrong key (missing `"TT"` prefix); returning TTNN-device tensors instead of CPU tensors from `prefill_forward` when the worker expects CPU; calling `ttnn.open_mesh_device` inside `initialize_vllm_model`

---

### Chapter 7 — Debugging, Constraints, and Performance Tuning

**Description:** Covers the debugging workflow for integration failures, enumerates the platform constraints imposed by `TTPlatform`, and explains which `DeviceModelSpec` levers tune performance and throughput.

**Directory:** `ch7_debugging_and_tuning/`

**Files:**

- `index.md`
  - Chapter overview
  - When to read this chapter: after successfully completing the integration checklist in Chapter 6 and running into correctness or performance issues
  - Key principle: isolate the model from the server before debugging — run `initialize_vllm_model` in a standalone script with a manually opened mesh device to confirm weight loading and a single forward pass before plugging into vLLM

- `debugging_integration.md`
  - Using the mock vLLM API server: `tt-inference-server` includes a mock server for testing the API surface without Tenstorrent hardware; use it to validate the request/response format before running on real hardware
  - Checking `initialize_vllm_model` in isolation: write a standalone Python script that calls `open_mesh_device(...)` from `tt_worker.py`, then calls `MyModelClass.initialize_vllm_model(hf_config, mesh_device, max_batch_size=1, max_seq_len=2048)`, then calls `prefill_forward` with a dummy input
  - Common `ValueError` from `TTModelLoader`: "model must be registered in vLLM's ModelRegistry with a TT-prefixed architecture name" — means either the class is not registered or the architecture string in `config.json` does not match the registration key
  - `TTPlatform` constraint errors: chunked prefill, prefix caching, and speculative decoding requests will be rejected at config validation time with clear error messages
  - KV cache shape mismatch: if `allocate_kv_cache` returns tensors with the wrong shape, the paged attention kernel will produce incorrect outputs; verify block size, num_kv_heads, and head_dim match the model's actual GQA configuration
  - Run mode for debugging with TT Symbiote: set `TT_SYMBIOTE_RUN_MODE=DPL` (dual-path logging) to run both the TTNN and PyTorch paths in parallel and compare outputs layer-by-layer; check PCC values against the reference (>0.99 for attention, >0.98 for MLP)

- `platform_constraints_reference.md`
  - Complete table of `TTPlatform` constraints and their rationale:

  | Constraint | Value | Reason |
  |---|---|---|
  | Chunked prefill | Disabled | Not yet supported by TT paged attention kernel |
  | Prefix caching | Disabled | No TTNN-side prefix cache reuse implementation |
  | Speculative decoding | Disabled | Not yet supported |
  | `n` sampling parameter | Must be 1 | Only single-sequence sampling per request |
  | `best_of` | Not supported | Requires multiple samples per request |
  | `prompt_logprobs` | Not supported | Requires per-token log probability computation |
  | Tensor parallelism | Single process | Multi-chip handled inside model via TTNN mesh ops |
  | Pinned memory | Unavailable | `TTPlatform.is_pin_memory_available()` returns False |
  | Block size | 64 (default) | TT paged attention kernel expects 64-token blocks |
  | `VLLM_USE_V1` | Must be `"1"` | Plugin only supports vLLM v1 architecture |

- `performance_tuning.md`
  - `DeviceModelSpec.max_concurrency`: sets `--max-num-seqs` (vLLM's batch size cap); for decode-bound models on T3K, typical values are 32; reduce if DRAM capacity is the bottleneck
  - `DeviceModelSpec.max_context`: sets `--max-model-len`; determines KV cache depth; larger values require more TTNN KV cache memory; compute `max_num_blocks = max_context / block_size` to estimate memory
  - `override_tt_config["optimizations"]`: `"performance"` selects bfloat8 weights and HiFi2 math fidelity; `"accuracy"` selects bfloat16 and HiFi4; start with `"accuracy"` to validate correctness, switch to `"performance"` for throughput benchmarks
  - `override_tt_config["trace_region_size"]`: pre-allocates trace buffer in L1; required for TTNN trace capture during decode; typical values from existing models are in the range of 8–23 MB
  - `has_builtin_warmup = True`: set this when `initialize_vllm_model` itself performs warmup sweeps (calls `prefill_forward` and `decode_forward` with dummy inputs to JIT-compile all ops); avoids the server sending HTTP warmup requests that may time out
  - Benchmarking: after integration, run `python3 run.py --model <name> --device <device> --workflow benchmarks` to measure TTFT, throughput, and per-token latency against the performance targets defined in `DeviceModelSpec.perf_targets_map`

---

## Conventions

### Terminology

| Term | Definition used throughout this guide |
|---|---|
| **tt-inference-server** | The orchestration and serving repository at `tenstorrent/tt-inference-server`; provides workflow automation, Docker deployment, and the `tt-vllm-plugin` |
| **tt-vllm-plugin** | The pip-installable Tenstorrent vLLM platform plugin located at `tt-inference-server/tt-vllm-plugin/`; registers `TTPlatform` and `TTModelLoader` with vLLM |
| **TTPlatform** | vLLM `Platform` subclass in `tt_vllm_plugin/platform.py`; controls model class name rewriting, constraint enforcement, and worker class selection |
| **TTModelLoader** | vLLM `BaseModelLoader` subclass in `tt_vllm_plugin/model_loader/tt_loader.py`; validates and invokes `initialize_vllm_model` |
| **ModelSpec** | A fully-expanded per-weight-per-device configuration object from `workflows/model_spec.py` |
| **ModelSpecTemplate** | A template that expands into multiple `ModelSpec` instances via the cross-product of `weights` × `device_model_specs` |
| **DeviceModelSpec** | The per-device sub-configuration inside a `ModelSpecTemplate`: batch size, context length, vLLM args, override_tt_config |
| **ImplSpec** | A descriptor that identifies the model implementation codebase (repo URL, code path, impl ID) |
| **initialize_vllm_model** | The required `@classmethod` that constructs, initializes, and returns a model instance; the primary integration contract |
| **prefill_forward** | The method called by the vLLM worker during prompt processing; receives full prompt token IDs and a page table |
| **decode_forward** | The method called by the vLLM worker during autoregressive generation; receives one token per active sequence |
| **allocate_kv_cache** | The method called once during worker startup to allocate per-layer TTNN KV cache tensors |
| **mesh device** | A `ttnn.MeshDevice` spanning one or more Tenstorrent chips; opened by `open_mesh_device()` in `tt_worker.py` |
| **block size** | The KV cache block size in tokens; defaults to 64 on TT hardware; must match what the paged attention kernel expects |
| **override_tt_config** | A dict field in `DeviceModelSpec` passed to `initialize_vllm_model` and `tt_worker.py` for hardware-specific tuning |
| **TT Symbiote** | The module-replacement acceleration framework (`TTNNModule`, `TorchTTNNTensor`, dispatcher system) |
| **TTNN** | Tenstorrent's C++/Python operator library; the hardware abstraction layer |
| **vLLM v1** | The vLLM v1 architecture (`VLLM_USE_V1=1`); the only mode supported by `tt-vllm-plugin` |
| **OpenAI-compatible API** | The HTTP API surface exposed by the vLLM OpenAI server at `/v1/completions`, `/v1/chat/completions`, `/v1/embeddings` |

### Notation

- File paths within `tt-inference-server` are given relative to the repository root (e.g., `tt-vllm-plugin/tt_vllm_plugin/platform.py`).
- File paths within `tt-metal` are given relative to the `tt-metal` repository root (e.g., `models/tt_transformers/tt/generator_vllm.py`).
- Environment variable names are written in `UPPER_SNAKE_CASE` inline code style (e.g., `VLLM_USE_V1`).
- Method and class names are written in inline code style (e.g., `initialize_vllm_model`, `TTPlatform`).
- When showing method signatures, Python type hints are included where known; `...` indicates omitted body.
- The `"TT"` prefix convention (as in `"TTLlamaForCausalLM"`) is always written with the exact two capital letters followed by the original HuggingFace architecture name.

### Formatting Rules

- Each chapter `index.md` must begin with a one-paragraph chapter summary and end with a "What's next" sentence pointing to the following chapter (Chapter 7 ends with a reference to the integration checklist in Chapter 6 as a return path).
- Integration steps in `integration_steps.md` are numbered and start with a bold label (e.g., **Step 1 — ...**).
- Constraint and environment variable tables use three columns with consistent header rows.
- Code examples include the full import path comment above the snippet (e.g., `# tt-vllm-plugin/tt_vllm_plugin/model_loader/tt_loader.py`).
- All mentions of numeric defaults (block size 64, warmup timeout, port 8000) are called out as "defaults" and noted as overridable.

---

## Cross-Chapter Dependencies

| Chapter | Depends on concepts from |
|---|---|
| Ch 2 (ModelSpec System) | Ch 1 — readers must understand the overall serving stack before the ModelSpec schema makes sense as a configuration layer |
| Ch 3 (Model Implementation Contract) | Ch 1, Ch 2 — the `initialize_vllm_model` signature only makes sense once the reader knows how `TTModelLoader` invokes it (Ch 1) and how `DeviceModelSpec` parameters flow in (Ch 2) |
| Ch 4 (Weight Loading and Tokenization) | Ch 3 — weight loading happens inside `initialize_vllm_model`; understanding the method contract (Ch 3) is prerequisite |
| Ch 5 (Hardware Initialization) | Ch 1, Ch 3 — device lifecycle is owned by the worker layer (Ch 1); knowing what the model must not do inside `initialize_vllm_model` presupposes understanding the method contract (Ch 3) |
| Ch 6 (Integration Checklist) | Ch 2, Ch 3, Ch 4, Ch 5 — the step-by-step recipe references ModelSpec definition (Ch 2), method signatures (Ch 3), weight path handling (Ch 4), and device ownership rules (Ch 5) |
| Ch 7 (Debugging and Tuning) | Ch 3, Ch 5, Ch 6 — debugging integration failures requires knowing the contract (Ch 3) and device boundary (Ch 5); tuning levers refer to `DeviceModelSpec` fields introduced in Ch 6's worked example |
