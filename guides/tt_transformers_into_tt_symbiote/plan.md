# Plan: Integrating TT Transformers into TT Symbiote

## Audience

**Target readers:** ML engineers and systems engineers who want to run LLMs on Tenstorrent hardware using TT Symbiote and need to understand how to port or reuse optimizations from `tt-transformers`.

**Assumed knowledge:**
- Familiarity with PyTorch `nn.Module` and the HuggingFace Transformers API
- Basic understanding of TTNN (Tenstorrent's low-level operator library)
- Awareness of what TT Symbiote is at a high level (module-replacement framework for transparent TTNN acceleration)
- Exposure to `tt-transformers` at a high level (hand-written, production-grade TTNN LLM inference stack)

**Not assumed:**
- Deep knowledge of TTNN tensor layouts, memory configs, or collective communication ops
- Prior experience writing TTNN kernel configurations or sharding strategies
- Familiarity with TT Symbiote internals (`TTNNModule`, `TorchTTNNTensor`, dispatcher system)

---

## Chapter List

### Chapter 1 — Architectural Comparison: TT Symbiote vs TT Transformers

**Description:** Establishes a side-by-side mental model of the two codebases — their design philosophies, abstraction layers, and how each handles the path from a PyTorch model to TTNN execution.

**Directory:** `ch1_arch_comparison/`

**Files:**
- `index.md`
  - Chapter overview and reading order
  - Summary table: TT Symbiote vs TT Transformers on five axes (abstraction level, model scope, weight management, device placement, debugging)
  - Defines the integration problem: Symbiote provides easy onboarding; TT Transformers provides hand-tuned LLM performance; the goal is to bring TT Transformers optimizations into Symbiote's framework

- `tt_symbiote_internals.md`
  - The `TTNNModule` base class: `from_torch()`, `preprocess_weights_impl()`, `move_weights_to_device_impl()`, `deallocate_weights_impl()`, and `forward()`
  - How `__call__` delegates to `TENSOR_RUN_IMPLEMENTATION.module_run` and how run modes (NORMAL, DPL, SEL, CPU) alter dispatch behavior
  - The `TorchTTNNTensor` wrapper: dual representation (`.to_torch`, `.to_ttnn`), transparent operator dispatch via `TT_SYMBIOTE_DISPATCHER`
  - The `register_module_replacement_dict` utility: how it walks `nn.Module._modules`, replaces classes, assigns unique names, and supports `exclude_replacement`
  - Weight lifecycle: preprocessed-on-host → moved-to-device → optionally auto-deallocated with `@deallocate_weights_after`
  - `DeviceArch` enum and the `@run_on_devices` decorator for hardware-gating forward passes

- `tt_transformers_internals.md`
  - `LightweightModule` base class (vs `nn.Module`) and why TT Transformers avoids PyTorch autograd overhead
  - `Transformer` (model.py): embedding, `RotarySetup`, stacked `TransformerBlock` layers, `LMHead`, `TT_CCL`
  - `Attention` (attention.py): per-layer dtype selection via `decoders_optimizations`, `TensorGroup`/`OpGroup` enums, compute kernel configs (hifi2, hifi4), sliding window, TG-specific slice matrices
  - `MLP` (mlp.py): DRAM-sharded weight loading, `ShardTensor2dMesh`, per-layer dtype (`FF1_FF3`, `FF2`), hidden-dim padding
  - `Generator` (generator.py): prefill/decode loop, trace capture (`ttnn.begin_trace_capture`), paged attention, warmup sweeps
  - `ModelOptimizations` / `model_config.py`: `TensorGroup`, `OpGroup`, `PrecisionSetting`, `MathFidelitySetting`, per-decoder-layer override JSON files

---

### Chapter 2 — Weight Management and Precision

**Description:** Compares how the two stacks load, convert, shard, and cache model weights, and identifies what must change when bringing TT Transformers weight handling into TT Symbiote.

**Directory:** `ch2_weight_management/`

**Files:**
- `index.md`
  - Chapter overview
  - Motivation: precision choices (bfloat16, bfloat8, bfloat4) and sharding strategies are the primary source of TT Transformers' performance advantage; understanding them is a prerequisite to porting

- `symbiote_weight_pipeline.md`
  - `preprocess_linear_weight` / `preprocess_linear_bias` from `ttnn.model_preprocessing`
  - `TTNNLinear` (bfloat16), `TTNNLinearLLama` (bfloat8_b + auto-deallocation), `TTNNLinearLLamaBFloat16`
  - Sharded linear variants: `TTNNLinearIColShardedWRowSharded`, `TTNNLinearIReplicatedWColSharded`; how they defer conversion until `move_weights_to_device_impl` (when the mesh device is available)
  - The host-staging pattern: `tt_weight_host` stored on CPU first, then `ttnn.to_device()`
  - Trace-mode gating: `@trace_enabled` vs `@trace_disabled` decorators and why LLaMA variants are trace-disabled (auto-deallocation incompatible with trace capture)

- `transformers_weight_pipeline.md`
  - `ttnn.as_tensor` with `cache_file_name`: on-disk weight caching to avoid repeated conversion
  - `ShardTensor2dMesh` and `ShardTensorToMesh`: how weights are sharded across a mesh at load time rather than at runtime
  - `load_checkpoints.py` utilities: `convert_hf_to_meta`, `reverse_permute` (RoPE key/query weight reorder), `standardize_hf_keys`
  - `w1_w3_mem_config` / `w2_mem_config`: `create_dram_sharded_mem_config` for placing weight tiles in DRAM across chips
  - Dummy weights mode for compile-time testing without real checkpoints

- `reuse_vs_rewrite.md`
  - Features that can be **reused directly**: `ttnn.model_preprocessing` preprocessing functions, bfloat8/bfloat16 dtype choices, the host-staging pattern
  - Features that require **adaptation**: TT Transformers weight-cache file path system (`weight_cache_path / prefix`) — Symbiote has no equivalent and one must be added or the cache skipped
  - Features that must be **rewritten from scratch**: `ShardTensor2dMesh` loading inside `TTNNModule.preprocess_weights_impl()` (the sharding dimensions depend on `args.cluster_shape` which Symbiote does not carry), per-layer dtype selection system (`ModelOptimizations` / `TensorGroup` / `OpGroup` pipeline)

---

### Chapter 3 — Attention: RoPE, KV Cache, and SDPA

**Description:** Deep-dives into the attention subsystem — the most complex component — comparing TT Transformers' hand-optimized multi-chip attention with TT Symbiote's current attention modules, and maps out what needs to be ported.

**Directory:** `ch3_attention/`

**Files:**
- `index.md`
  - Chapter overview
  - Why attention is the hardest subsystem to integrate: it combines RoPE, paged KV cache, grouped query attention (GQA), two execution modes (prefill / decode), collective communication (all-gather, reduce-scatter), and multi-chip layout concerns

- `symbiote_attention_overview.md`
  - `TTNNViTSelfAttention` (deprecated), `TTNNSDPAAttention`, `TTNNFusedQKVSelfAttention`, `TTNNSelfAttention`, `TTNNWhisperAttention` — current module inventory
  - `LlamaAttention` in `test_llama.py`: the existing TT Symbiote LLaMA attention wrapper (imported from `modules/attention.py`)
  - `TTNNRotaryPositionEmbedding` and `TTNNDistributedRotaryPositionEmbedding` from `modules/rope.py`
  - `TTNNPagedAttentionKVCache` (Cache subclass in attention.py): block-size, page-table layout, `_tt_key_cache` / `_tt_value_cache` lists
  - `PagedAttentionConfig`: `block_size`, `max_num_blocks`, `batch_size`, derived `max_seq_length` and `blocks_per_sequence`

- `transformers_attention_overview.md`
  - `Attention.__init__` parameters: `mesh_device`, `tt_ccl`, `args`, `state_dict`, `weight_cache_path`, `layer_num`, `dtype`, `transformation_mats`, `configuration`, `paged_attention_config`, `use_paged_kv_cache`, `prefetcher`
  - Per-layer dtype selection: `wqkv_dtype`, `wo_dtype`, `activation_dtype` from `decoders_optimizations.get_tensor_dtype`
  - Compute kernel configs: `compute_kernel_config_hifi2`, `hifi2_fp16`, `hifi4` — how they are selected and applied to QKV / SDPA / output matmuls
  - Sliding window attention: `configuration.layer_types[layer_num] == "sliding_attention"` and `sliding_window` parameter
  - TG (Galaxy) specific logic: `slice_mat`, `user_selection_matrix`, `n_local_heads` and `n_local_kv_heads` calculation for 32-device topology
  - `RotarySetup` in `rope.py`: pre-computed cosine/sine matrices, `get_both_trans_mats()`, prefetcher integration

- `integration_gaps.md`
  - Gap 1: TT Transformers attention requires explicit `state_dict` and `weight_cache_path`; Symbiote attention is initialized via `from_torch(torch_layer)` — bridging these requires either adapting `from_torch` to accept a state dict path or loading weights eagerly from the HF model
  - Gap 2: Prefill vs decode mode switching — TT Transformers uses a `Mode` enum and changes tensor layouts and kernel configs per call; Symbiote has no equivalent mode concept and a forward-pass convention must be established
  - Gap 3: `TT_CCL` (collective communication library) is a TT Transformers concept; Symbiote's sharded linear layers use `ttnn.experimental.reduce_scatter_minimal_async` directly — a unified CCL interface needs to be defined or the Symbiote approach extended
  - Gap 4: Paged KV cache in Symbiote (`TTNNPagedAttentionKVCache`) exists but is not wired into the HF `generate()` loop in the same way TT Transformers' `Generator` does it — integration requires either adapting `Generator` or writing a Symbiote-native decode loop

---

### Chapter 4 — MLP, Normalization, and the Full Decoder Block

**Description:** Covers the feed-forward network and normalization layers, comparing implementations and identifying which Symbiote modules can serve as drop-in replacements and which must be extended.

**Directory:** `ch4_mlp_and_norms/`

**Files:**
- `index.md`
  - Chapter overview
  - Decoder block composition in both stacks: norm → attention → norm → MLP (plus residual connections and CCL all-reduce in TT Transformers)

- `mlp_comparison.md`
  - TT Symbiote MLP path: `TTNNLinear` / `TTNNLinearLLama` + `TTNNSilu` replaced via `register_module_replacement_dict`; the `LlamaMLP` wrapper in `test_llama.py` that restructures the HF MLP to expose separate gate/up/down projections
  - TT Transformers `MLP`: `w1`, `w2`, `w3` as DRAM-sharded `ttnn.as_tensor` objects, `FF1_FF3` / `FF2` dtypes, hidden-dim padding via `pad_to_size`, `tt_all_reduce` after `w2`
  - What can be reused: `TTNNLinearLLama` (bfloat8 + auto-deallocation) maps directly onto `w1`/`w3` use case; `TTNNSilu` maps onto the gate activation
  - What must be added: a Symbiote-native MLP module that performs `tt_all_reduce` after the down-projection when running on multi-chip — this does not exist today

- `normalization_comparison.md`
  - TT Symbiote: `TTNNLayerNorm`, `TTNNRMSNorm`, `TTNNDistributedRMSNorm`
  - TT Transformers: `RMSNorm` (from `models.common.rmsnorm`) and `DistributedNorm` — the latter shards the norm across devices and uses an all-gather to reconstruct
  - Reuse assessment: `TTNNRMSNorm` is a direct replacement for single-device `RMSNorm`; `TTNNDistributedRMSNorm` maps onto `DistributedNorm` but needs to verify the all-gather strategy matches TT Transformers' `distributed_norm.py`
  - `rms_norm_add_unit_offset` flag in TT Transformers: controls whether the norm weight is treated as a multiplicative offset from 1 (Gemma-style); Symbiote's `TTNNRMSNorm` does not expose this flag — it must be added

- `decoder_block_assembly.md`
  - How TT Transformers `TransformerBlock` (decoder.py) wires together attention, MLP, norms, and residual addition on a mesh device
  - How the equivalent in Symbiote is assembled today for LLaMA (via `test_llama.py` and `register_module_replacement_dict`)
  - Step-by-step recipe for assembling a Symbiote decoder block that uses TT Transformers-grade ops: define a `TTNNLlamaDecoder` module, populate it with `LlamaAttention`, a multi-chip MLP wrapper, `TTNNRMSNorm`, and wire residual adds
  - Performance consideration: residual additions in TT Transformers are done on TTNN tensors with explicit memory configs; in Symbiote they fall through to the PyTorch dispatcher unless explicitly handled

---

### Chapter 5 — Multi-Chip Parallelism and Collective Communication

**Description:** Explains how TT Transformers distributes LLM inference across multiple Tenstorrent chips (tensor parallelism) and how those patterns map — or do not yet map — onto TT Symbiote's current distributed primitives.

**Directory:** `ch5_multi_chip/`

**Files:**
- `index.md`
  - Chapter overview
  - Scope: N300 (2-chip), T3K (8-chip Wormhole cluster), TG/Galaxy (32-chip) topologies
  - Why this chapter matters: most performance gains from TT Transformers come from multi-chip parallelism; without it Symbiote on a single chip is only a fraction of achievable throughput

- `tt_transformers_parallelism.md`
  - `TT_CCL`: wrapper around `ttnn.experimental` collective ops; `tt_all_gather` and `tt_all_reduce` used in attention and MLP
  - Column-parallel vs row-parallel linear: QKV and FFN gate/up are column-parallel (output sharded), output projection and FFN down are row-parallel (input sharded) + all-reduce
  - `ShardTensor2dMesh`: how a weight matrix is cut across a 2D mesh of chips (row-axis and col-axis dims)
  - TG-specific topology: `ReplicateTensorToMesh` for certain weights, `ShardTensorToMesh(dim=1)` for the slice matrix
  - Prefetcher (`prefetcher.py`): asynchronous weight pre-loading into L1 before the compute kernel runs; requires 4D weight tensors `[1, 1, H, W]`

- `symbiote_distributed_primitives.md`
  - Existing distributed support: `TTNNLinearIColShardedWRowSharded` (column-input, row-weight sharding + `reduce_scatter_minimal_async`), `TTNNLinearIReplicatedWColSharded` (replicated input, column-weight)
  - `TTNNDistributedRotaryPositionEmbedding` and `TTNNDistributedRMSNorm`: existing multi-chip norm and RoPE
  - `DistributedTensorConfig` / `DistributedConfig` / `CCLManager` in `core/run_config.py`: device-state object passed through `TTNNModule.set_device_state()`
  - `@run_on_devices(DeviceArch.T3K)` decorator: how hardware gating is enforced at the module level

- `integration_strategy.md`
  - What can be reused: `TTNNLinearIColShardedWRowSharded` covers the column-parallel linear pattern; `TTNNDistributedRMSNorm` covers distributed norm; `reduce_scatter_minimal_async` covers the all-reduce after row-parallel linear
  - What must be adapted: the `TT_CCL` wrapper pattern — Symbiote uses raw `ttnn.experimental` calls inside module `forward()`, which is harder to maintain and tune; recommend introducing a thin CCL helper consistent with TT Transformers' `ccl.py`
  - What must be built from scratch: the prefetcher integration — `Prefetcher` in TT Transformers requires weights to be 4D and loaded with specific DRAM-sharded memory configs; Symbiote's `preprocess_weights_impl` / `move_weights_to_device_impl` pipeline would need a parallel path for prefetcher-aware weight loading
  - Mesh device shape configuration: TT Transformers reads `args.cluster_shape`; Symbiote relies on the `MESH_DEVICE` environment variable and `DeviceArch`; a unified config adapter is needed

---

### Chapter 6 — The Decode Loop: From `Generator` to Symbiote Inference

**Description:** Traces the full decode-time inference path in TT Transformers (prefill, KV cache, decode, trace capture) and maps each piece to its nearest Symbiote equivalent, identifying what a Symbiote-native LLM inference loop requires.

**Directory:** `ch6_decode_loop/`

**Files:**
- `index.md`
  - Chapter overview
  - The two phases of LLM inference: prefill (process prompt, fill KV cache) and decode (autoregressive token generation)
  - Why trace capture matters: `ttnn.begin_trace_capture` / `ttnn.execute_trace` eliminates host-side Python overhead during decode, which is the dominant bottleneck at small batch sizes

- `tt_transformers_generator.md`
  - `Generator.__init__`: holds list of `model` instances (data parallel), `trace_id_*` / `trace_inputs_*` / `trace_output_*` dicts for prefill and decode traces
  - `warmup_model_prefill`: sweeps supported sequence lengths and sampling parameters to JIT-compile all ops before serving
  - `_capture_trace_prefill`: calls `prepare_prefill_inputs_trace`, runs one forward pass to compile, then captures the trace with `ttnn.begin_trace_capture`
  - Decode trace capture: split-sampling mode (`enable_split_sampling`) — the decode step is split into (a) forward pass to logits and (b) sampling, each captured as a separate trace
  - Paged attention integration: `page_table` is passed into the model forward; `TTNNPagedAttentionKVCache` manages block allocation
  - vLLM and SGLang backends: `generator_vllm.py` and `generator_sglang.py` extend `Generator` with serving-framework-specific scheduling

- `symbiote_inference_path.md`
  - Current Symbiote inference: calls HuggingFace `model.generate()` with TTNN-replaced modules — no custom decode loop, no trace capture, no paged attention wiring
  - Advantages of the current approach: zero code changes to the HF model class; the KV cache is managed by HF's built-in `DynamicCache` or `StaticCache`
  - Limitations: every decode step incurs Python overhead from the HF generation loop; the `TorchTTNNTensor` dispatcher adds wrapping/unwrapping cost; no trace capture possible with the current `@trace_disabled` pattern on LLaMA linears
  - The role of `TT_SYMBIOTE_RUN_MODE`: SEL and DPL modes exist for debugging but do not accelerate; NORMAL mode is the production path and is where trace capture would be inserted

- `integration_roadmap.md`
  - Option A (minimal): keep HF `generate()`, add trace capture inside `TTNNModule.__call__` using a context manager that wraps a forward pass in `ttnn.begin_trace_capture` / `ttnn.execute_trace` — feasible but requires all modules in a trace to avoid deallocating weights mid-trace
  - Option B (full port): write a Symbiote-native `Generator` class modeled on TT Transformers' `Generator` that drives the decode loop, manages paged KV cache, and handles prefill/decode mode switching — higher effort but enables full performance parity
  - Recommended path: start with Option A for rapid prototyping and performance measurement, then migrate toward Option B once the attention and MLP modules are validated
  - Concrete milestones: (1) replace LLaMA linears with TT Transformers-grade bfloat8 sharded linears in Symbiote; (2) validate PCC vs HF reference; (3) add CCL all-reduce to MLP; (4) add trace capture; (5) integrate paged attention; (6) benchmark decode throughput vs TT Transformers baseline

---

### Chapter 7 — Worked Example: LLaMA 3.2-1B on Symbiote with TT Transformers Optimizations

**Description:** Walks through a concrete, end-to-end integration example for LLaMA 3.2-1B, applying lessons from all prior chapters to produce a Symbiote test that uses TT Transformers-grade precision and sharding.

**Directory:** `ch7_llama_worked_example/`

**Files:**
- `index.md`
  - Chapter overview and prerequisites (chapters 1–6)
  - Hardware target: N150 (single chip) for initial validation, N300 (2-chip) for distributed path
  - Reference: existing `test_llama.py` in TT Symbiote and `demo/simple_text_demo.py` in TT Transformers

- `step1_module_map.md`
  - Concrete mapping table: every HuggingFace `LlamaForCausalLM` sub-module class → its Symbiote replacement → the TT Transformers equivalent it corresponds to
  - `LlamaRMSNorm` → `TTNNRMSNorm` ↔ `RMSNorm`
  - `LlamaMLP` → custom `TTNNLlamaMLP` ↔ `MLP`
  - `LlamaAttention` → `LlamaAttention` (already in Symbiote attention.py) ↔ `Attention`
  - `nn.Linear` (lm_head) → excluded via `exclude_replacement={"lm_head"}`
  - Embedding layer: `nn.Embedding` → not currently replaced in Symbiote; TT Transformers uses a custom `Embedding` module with row-major layout

- `step2_precision_config.md`
  - How to select dtypes for each linear group in the Symbiote context: use `TTNNLinearLLama` (bfloat8) for QKV, output, gate, up projections; use `TTNNLinearLLamaBFloat16` for down projection (matches TT Transformers' `FF2` = bfloat8 default but allows experimenting with bfloat16)
  - How to verify output quality: run in DPL mode (`TT_SYMBIOTE_RUN_MODE=DPL`) and compare PCC between TTNN and PyTorch outputs per layer
  - Expected PCC thresholds per component based on TT Transformers' own tests: >0.99 for attention output, >0.98 for MLP output, >0.99 for full decoder block

- `step3_validation_and_benchmarking.md`
  - How to run the existing `test_llama.py` as a baseline
  - How to measure tokens-per-second before and after applying TT Transformers precision settings
  - How to use `DispatchManager.save_stats_to_file()` (already in the test) to identify per-layer latency bottlenecks
  - Expected performance characteristics: bfloat8 weights reduce DRAM bandwidth by 2× vs bfloat16; auto-deallocation with `@deallocate_weights_after` reduces peak device memory usage; multi-chip sharding (N300) should yield near-2× throughput for compute-bound layers
  - Known limitations and open issues: trace capture not yet integrated; paged attention not wired through HF generate; lm_head must be excluded to avoid host-device data movement on every token

---

## Conventions

### Terminology

| Term | Definition used throughout this guide |
|---|---|
| **TT Symbiote** | The module-replacement framework at `models/experimental/tt_symbiote/` |
| **TT Transformers** (TTT) | The hand-written LLM inference stack at `models/tt_transformers/` |
| **TTNN** | Tenstorrent's C++/Python operator library; the hardware abstraction layer both stacks use |
| **TTNNModule** | TT Symbiote base class; do not confuse with `LightweightModule` (TT Transformers base class) |
| **LightweightModule** | TT Transformers base class; a thin wrapper over Python with no PyTorch autograd |
| **mesh device** | A logical device spanning one or more Tenstorrent chips, created via `ttnn.open_mesh_device` |
| **bfloatX_b** | TTNN block-floating-point formats: `bfloat8_b` = 8-bit, `bfloat4_b` = 4-bit |
| **bfloat16** | Standard 16-bit brain float; `ttnn.bfloat16` |
| **DRAM_MEMORY_CONFIG** | `ttnn.DRAM_MEMORY_CONFIG`; default off-chip memory placement |
| **TILE_LAYOUT** | `ttnn.TILE_LAYOUT`; 32×32 tile format required for most TTNN matmul ops |
| **PCC** | Pearson Correlation Coefficient; used as the numerical accuracy metric throughout |
| **prefill** | The phase where the full prompt is processed in one (or chunked) forward pass |
| **decode** | The autoregressive phase where one token is generated per step |
| **GQA** | Grouped Query Attention; `n_kv_heads < n_heads` |
| **CCL** | Collective Communication Library; ops like all-gather and reduce-scatter across chips |
| **trace capture** | `ttnn.begin_trace_capture` / `ttnn.execute_trace`; records and replays ops to eliminate Python overhead |

### Notation

- File paths are always given as absolute paths starting from the repo root (e.g., `models/experimental/tt_symbiote/core/module.py`).
- Code snippets use Python syntax highlighting.
- When comparing the two stacks, TT Symbiote is on the left/first and TT Transformers is on the right/second.
- "Reuse", "adapt", and "rewrite" are used with strict meaning: **reuse** = take as-is with no code changes; **adapt** = take with modifications; **rewrite** = implement from scratch in the Symbiote style.

### Formatting Rules

- Each chapter `index.md` must start with a one-paragraph summary of what the chapter covers and end with a "What's next" sentence pointing to the following chapter.
- Tables comparing the two stacks should have three columns: Feature, TT Symbiote, TT Transformers.
- Code examples must include the full import path for every referenced symbol.
- Gap analysis sections use a consistent heading pattern: "Gap N: \<short title\>".
- PCC values, dtype names (`bfloat8_b`, `bfloat16`), and environment variable names (`TT_SYMBIOTE_RUN_MODE`) are always written in inline code style.

---

## Cross-Chapter Dependencies

| Chapter | Depends on concepts from |
|---|---|
| Ch 2 (Weight Management) | Ch 1 — readers must understand `TTNNModule.preprocess_weights_impl()` and `LightweightModule` before the weight pipeline comparison makes sense |
| Ch 3 (Attention) | Ch 1, Ch 2 — attention modules use the same weight-staging pattern and dtype selection system described in Ch 2 |
| Ch 4 (MLP and Norms) | Ch 1, Ch 2 — MLP sharded linears are a specialization of the weight sharding patterns in Ch 2; decoder block assembly requires Ch 3's attention module |
| Ch 5 (Multi-Chip) | Ch 2, Ch 3, Ch 4 — distributed linears extend the weight-sharding concepts in Ch 2; CCL is used in both attention (Ch 3) and MLP (Ch 4) |
| Ch 6 (Decode Loop) | Ch 3, Ch 5 — the Generator's trace capture depends on paged KV cache (Ch 3) and collective communication (Ch 5) being in place |
| Ch 7 (Worked Example) | All prior chapters — the LLaMA example applies the module map (Ch 1), precision choices (Ch 2), attention port (Ch 3), MLP assembly (Ch 4), multi-chip config (Ch 5), and inference loop (Ch 6) |
