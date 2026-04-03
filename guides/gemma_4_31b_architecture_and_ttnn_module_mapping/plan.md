# Plan: Gemma 4 31B Architecture and TTNN Module Mapping

## Audience

This guide targets ML systems engineers and kernel developers working on the
TT-NN / tt-symbiote stack who need to implement Gemma 4 31B inference on the
T3K 1x8 Wormhole mesh. The reader will use this guide as the definitive
reference when writing `TTNNModule` subclasses for every submodule in the
Gemma 4 31B decoder.

**Assumed knowledge:**
- Familiarity with transformer decoder architectures (MHA, GQA, RMSNorm, FFN)
- Working knowledge of TTNN tensor operations, memory configs, and program configs
- Experience with `TTNNModule` authoring in tt-symbiote (module replacement, `forward` signatures)
- Basic understanding of T3K device topology (8 Wormhole chips, 1x8 mesh, Ethernet links)
- Exposure to paged KV cache concepts and `ttnn.transformer.scaled_dot_product_attention_decode`

**Not assumed:**
- Knowledge of Gemma 4 specific architectural innovations (heterogeneous attention, K=V sharing, V-norm, PLE, dual RoPE)
- Prior work with models that have structurally different layer types sharing a single decoder stack
- Experience with partial rotary position embeddings
- Understanding of how to shard tensors when KV head counts differ across layer types

---

## Chapter List

### Chapter 1 --- Gemma 4 31B Architecture Overview

**Description:** Provides a complete reference of the Gemma 4 31B model
architecture, layer-by-layer, covering every submodule, its configuration
parameters, and how the 60 layers are organized into sliding and global types.

**Directory:** `ch1_architecture_overview/`

**Files:**

- `index.md`
  - Chapter overview and reading order
  - Quick-reference table: all config.json parameters relevant to the text decoder

- `layer_organization.md`
  - 60 total layers: 48 sliding-window layers and 12 global (full) attention layers
  - Pattern: every 6th layer is global (indices 5, 11, 17, 23, 29, 35, 41, 47, 53, 59)
  - Final layer is always global
  - Each layer is a `Gemma4TextDecoderLayer` containing: pre-attention RMSNorm, attention, post-attention RMSNorm, GeGLU FFN, and PLE injection point
  - Block diagram of a single decoder layer with all submodules labelled

- `heterogeneous_attention_configs.md`
  - Two structurally different attention configurations coexisting in the same model
  - Sliding attention: 32 query heads, 16 KV heads, head_dim=256, window=1024 tokens, standard RoPE (theta=10000), full rotary (all dims)
  - Global attention: 32 query heads, 4 KV heads, head_dim=512, full causal, proportional RoPE (theta=1000000), partial_rotary_factor=0.25
  - Table comparing every parameter side-by-side between the two types
  - Implications: projection weight shapes differ between layer types (K/V projections are `[hidden_size, num_kv_heads * head_dim]` which is `[5376, 4096]` for sliding and `[5376, 2048]` for global)
  - `attention_k_eq_v=True` for global layers: the V projection is eliminated; K weights are reused for V, with divergent post-processing paths

- `novel_components.md`
  - K=V sharing: in global layers, a single projection produces both K and V; K receives scaled RMSNorm + RoPE; V receives unscaled RMSNorm (no RoPE)
  - V-norm: RMSNorm applied to value vectors with `with_scale=False` (no learned scale parameter) across all layer types
  - Per-Layer Embeddings (PLE): a second embedding table that injects a small residual signal into every decoder layer, computed before soft-token merge for multimodal inputs
  - `final_logit_softcapping=30.0`: output logit soft-capping before the final projection
  - `hidden_activation="gelu_pytorch_tanh"`: GeGLU activation in all FFN layers
  - `num_kv_shared_layers=0` in the 31B config (no KV sharing between layers in this variant)

---

### Chapter 2 --- Projection Weights and Tensor Shapes

**Description:** Derives the exact weight tensor shapes and activation tensor
shapes for every linear projection in both sliding and global layer types,
providing the foundation for TTNN matmul program configs.

**Directory:** `ch2_projection_shapes/`

**Files:**

- `index.md`
  - Chapter overview
  - Master shape table covering all projections across both layer types

- `qkv_projections.md`
  - Q projection: `[hidden_size, num_heads * head_dim]`
    - Sliding: `[5376, 32 * 256] = [5376, 8192]`
    - Global: `[5376, 32 * 512] = [5376, 16384]`
  - K projection: `[hidden_size, num_kv_heads * head_dim]`
    - Sliding: `[5376, 16 * 256] = [5376, 4096]`
    - Global: `[5376, 4 * 512] = [5376, 2048]`
  - V projection:
    - Sliding: `[5376, 16 * 256] = [5376, 4096]` (separate V projection)
    - Global: eliminated (K=V sharing; K projection reused)
  - O projection: `[num_heads * head_dim, hidden_size]`
    - Sliding: `[8192, 5376]`
    - Global: `[16384, 5376]`
  - Decode activation shapes at batch=1: Q `[1, 1, 32, head_dim]`, K `[1, 1, num_kv_heads, head_dim]`, V same as K

- `ffn_projections.md`
  - GeGLU structure: gate projection + up projection (fused or separate), then down projection
  - Gate: `[5376, 21504]`
  - Up: `[5376, 21504]`
  - Down: `[21504, 5376]`
  - Identical across all 60 layers (no per-layer-type FFN variation)
  - Activation function: `gelu_pytorch_tanh` (GELU with tanh approximation)

- `ple_shapes.md`
  - PLE embedding table shape: `[vocab_size, ple_dim]` (second embedding table)
  - Per-layer projection: projects PLE vector to `hidden_size` with a lightweight residual block
  - Injection: added to the residual stream at the start of each decoder layer
  - Multimodal handling: PLE uses pad token ID for vision/audio token positions

---

### Chapter 3 --- K=V Sharing and V-Norm Implementation

**Description:** Deep-dives into the two most novel attention features in
Gemma 4 --- K=V weight sharing in global layers and V-norm without learned
scale --- and analyzes how each maps to TTNN operations.

**Directory:** `ch3_kv_sharing_and_vnorm/`

**Files:**

- `index.md`
  - Chapter overview
  - Why these features matter for TTNN: K=V sharing changes the projection count and V-norm requires a scale-free RMSNorm variant

- `k_eq_v_mechanism.md`
  - Detailed dataflow: single weight matrix produces a shared tensor; this tensor is cloned into K and V paths
  - K path: scaled RMSNorm (with learned scale) followed by partial RoPE (only 25% of dims rotated)
  - V path: unscaled RMSNorm (`with_scale=False`) with no RoPE applied
  - Why this works: V vectors carry semantic content that should not be position-dependent; sharing the projection reduces parameters by eliminating one `[5376, 2048]` matrix per global layer
  - TTNN mapping: single `ttnn.linear` for the shared K/V projection, followed by a `ttnn.clone` or slice to create two tensors, then divergent norm + RoPE paths
  - Impact on fused QKV: standard fused QKV matmul packs Q, K, V projections into one weight; with K=V sharing, the fused weight for global layers packs Q and K only (V reuses K's slice)

- `vnorm_implementation.md`
  - V-norm definition: `RMSNorm(v, eps=1e-6, with_scale=False)` --- normalizes by RMS magnitude without multiplying by a learned gamma
  - Present in all 60 layers (both sliding and global), not just global layers
  - Mathematical expression: `v_normed = v / sqrt(mean(v^2) + eps)`
  - TTNN mapping options:
    (a) `TTNNDistributedRMSNorm` with scale weights set to all-ones (functionally equivalent but wastes a parameter tensor)
    (b) Custom `ttnn.rms_norm` call with `weight=None` or a `with_scale=False` code path if supported
    (c) Manual implementation: `ttnn.mul(v, ttnn.rsqrt(ttnn.mean(ttnn.square(v)) + eps))`
  - Analysis of whether `TTNNDistributedRMSNorm` supports the `with_scale=False` variant: what changes are needed if it does not
  - Performance comparison of the three options

---

### Chapter 4 --- Dual RoPE and Partial Rotary Embedding

**Description:** Covers the two distinct RoPE configurations used by sliding
and global layers, including partial rotary embedding where only 25% of
dimensions are rotated.

**Directory:** `ch4_dual_rope/`

**Files:**

- `index.md`
  - Chapter overview
  - Quick reference: sliding RoPE vs global p-RoPE parameter table

- `sliding_rope.md`
  - Standard RoPE: theta=10000, full rotary (all `head_dim=256` dimensions rotated)
  - Applied to both Q and K in sliding layers
  - TTNN mapping: `TTNNRotaryPositionEmbedding` or `TTNNDistributedRotaryPositionEmbedding` with standard cos/sin tables of shape `[max_seq_len, 256]`
  - Cos/sin table precomputation and device placement

- `global_proportional_rope.md`
  - Proportional RoPE (p-RoPE): theta=1000000, `partial_rotary_factor=0.25`
  - Only the first 128 of 512 head dimensions receive rotary encoding; the remaining 384 dimensions pass through unchanged
  - Why this design: high theta + partial rotation provides better long-context extrapolation to 256K tokens by leaving 75% of dimensions as pure semantic channels
  - Applied to Q and K in global layers (V receives no RoPE, as described in Chapter 3)
  - TTNN mapping: cos/sin tables of shape `[max_seq_len, 128]` (not 512)
  - Implementation: split head tensor into rotary dims `[:128]` and pass-through dims `[128:]`, apply RoPE to the rotary slice, concatenate
  - Compatibility with `TTNNRotaryPositionEmbedding`: the `partial_rotary_factor < 1.0` forces non-distributed RoPE in current tt-symbiote (cross-reference to existing findings in the TTNNBailingMoEAttention guide)
  - Performance cost of non-distributed partial RoPE on T3K and potential workarounds

- `rope_precomputation.md`
  - Two separate sets of cos/sin tables must be precomputed and stored on device
  - Sliding: `[max_seq_len, 256]` with theta=10000
  - Global: `[max_seq_len, 128]` with theta=1000000
  - Memory footprint of cos/sin tables at 256K context length
  - Strategy: precompute at model init, store in DRAM, slice per decode step

---

### Chapter 5 --- Heterogeneous Attention Module Design

**Description:** Addresses the core design question: how to structure the
TTNNModule classes for the two attention types, analyzing single-class vs
dual-class approaches and providing a recommended architecture.

**Directory:** `ch5_attention_module_design/`

**Files:**

- `index.md`
  - Chapter overview
  - Central question: one `TTNNGemma4Attention` class with config-driven branching, or separate `TTNNGemma4SlidingAttention` and `TTNNGemma4GlobalAttention` classes?

- `design_options.md`
  - Option A --- Single unified class:
    - Constructor takes `layer_type: Literal["sliding", "global"]` and configures head counts, head dims, RoPE, K=V sharing accordingly
    - Pros: single code path to maintain, consistent interface for the decoder layer, mirrors HuggingFace implementation
    - Cons: many conditional branches in `forward`, two different KV cache shapes, harder to optimize per-type
  - Option B --- Two separate classes:
    - `TTNNGemma4SlidingAttention` and `TTNNGemma4GlobalAttention` each with specialized `forward` logic
    - Pros: cleaner per-type optimization, no runtime branching, easier to profile independently
    - Cons: code duplication, two sets of program configs to maintain
  - Option C --- Base class with specialized subclasses:
    - `TTNNGemma4AttentionBase` with shared logic (Q projection, output projection, post-attention norm), subclassed for attention-type-specific logic (KV projection, RoPE, SDPA call)
    - Pros: best of both worlds --- shared code in base, specialized code in subclasses
    - Cons: slightly more complex class hierarchy
  - Recommendation with rationale

- `sliding_attention_forward.md`
  - Decode forward pass for sliding attention:
    1. Q, K, V projections (separate K and V weights)
    2. V-norm (unscaled RMSNorm on V)
    3. Standard RoPE on Q and K (full rotation, theta=10000)
    4. KV cache update (paged, window-bounded to 1024 tokens)
    5. `paged_sdpa_decode` with sliding window constraint
    6. Output projection
  - Tensor shapes at each step for batch=1 decode
  - Whether `paged_sdpa_decode` supports `sliding_window_size` parameter natively
  - If not natively supported: manual KV cache truncation strategy (only store last 1024 tokens, circular buffer approach from windowed attention guide)

- `global_attention_forward.md`
  - Decode forward pass for global attention:
    1. Q projection and shared K/V projection (single weight matrix)
    2. Clone/split shared projection into K and V tensors
    3. K path: scaled RMSNorm + partial p-RoPE (128/512 dims)
    4. V path: unscaled RMSNorm (no RoPE)
    5. KV cache update (paged, full causal --- no window)
    6. `paged_sdpa_decode` with full causal attention
    7. Output projection
  - Tensor shapes at each step for batch=1 decode
  - How the fused QKV optimization changes when V shares K's projection

- `paged_sdpa_sliding_window.md`
  - Investigation: does `ttnn.transformer.scaled_dot_product_attention_decode` accept a `sliding_window_size` parameter?
  - If yes: how it interacts with paged KV cache (does it limit which pages are loaded?)
  - If no: two fallback strategies:
    (a) Manually truncate the page table to only reference the last `ceil(1024 / block_size)` pages
    (b) Use a fixed-size circular KV cache of 1024 tokens and bypass paging for sliding layers
  - Trade-offs: paged sliding vs circular buffer for 1024-token window
  - Cross-reference to existing windowed attention guide findings

---

### Chapter 6 --- Tensor-Parallel Sharding on T3K

**Description:** Defines the optimal sharding strategy for Gemma 4 31B across
the T3K 1x8 mesh, addressing the challenge of two different KV head counts
(16 for sliding, 4 for global) and two different head dimensions.

**Directory:** `ch6_tp_sharding/`

**Files:**

- `index.md`
  - Chapter overview
  - Central challenge: TP=8 works cleanly for 32 Q heads (4 per device) but creates asymmetry for KV heads (16/8=2 per device for sliding vs 4/8=0.5 per device for global)

- `sharding_strategy_analysis.md`
  - Q heads: 32 heads / 8 devices = 4 Q heads per device (clean split for both layer types)
  - Sliding KV heads: 16 heads / 8 devices = 2 KV heads per device (clean split, GQA group_size=2 per device)
  - Global KV heads: 4 heads / 8 devices = 0.5 heads per device (CANNOT split cleanly)
  - Options for global layer KV sharding:
    (a) Replicate all 4 KV heads on every device --- wastes memory but avoids cross-device KV communication
    (b) Shard 4 KV heads across 4 devices, leave 4 devices idle for KV --- underutilizes mesh
    (c) Shard by head_dim instead of head count: each device holds all 4 KV heads but a slice of the 512-dim head --- requires gather before SDPA
    (d) Use TP=4 for global layers only (pairs of devices share KV heads)
  - Memory analysis for each option: KV cache size per device at 256K context
  - CCL overhead analysis for each option
  - Recommendation with rationale

- `weight_sharding.md`
  - Column-parallel sharding for Q, K, V, gate, up projections (shard output dim across devices)
  - Row-parallel sharding for O, down projections (shard input dim across devices)
  - Sliding layer weight shapes per device after TP=8 split
  - Global layer weight shapes per device (depends on KV sharding strategy chosen)
  - All-reduce after row-parallel matmuls: `ttnn.all_reduce` on hidden_size dim
  - Compatibility with `TTNNLinearIColShardedWAllReduced` pattern

- `kv_cache_sharding.md`
  - Sliding layers: each device holds KV cache for its 2 local KV heads, window=1024
  - Global layers: KV cache sharding depends on strategy from `sharding_strategy_analysis.md`
  - Per-device KV cache memory budget at various sequence lengths
  - Page table configuration for paged KV cache under each sharding strategy
  - Total DRAM budget for KV caches across all 60 layers on each device

---

### Chapter 7 --- Decoder Layer and Full Model Assembly

**Description:** Assembles all prior components into the complete decoder layer
and full model TTNNModule structure, including PLE injection, the 60-layer
loop, and prefill considerations.

**Directory:** `ch7_model_assembly/`

**Files:**

- `index.md`
  - Chapter overview
  - Module hierarchy diagram: `TTNNGemma4Model` > `TTNNGemma4DecoderLayer` x 60 > attention + FFN + norms

- `decoder_layer_module.md`
  - `TTNNGemma4DecoderLayer` structure:
    1. PLE injection: add per-layer embedding residual to input hidden states
    2. Pre-attention RMSNorm
    3. Attention (dispatches to sliding or global submodule based on layer index)
    4. Residual add
    5. Pre-FFN RMSNorm
    6. GeGLU FFN (gate + up projections, GELU activation, elementwise multiply, down projection)
    7. Residual add
  - Constructor: receives `layer_idx` to determine attention type from the 5:1 pattern
  - Forward signature and tensor flow

- `ffn_module.md`
  - `TTNNGemma4FFN` implementation:
    - Fused gate+up projection or separate projections
    - `gelu_pytorch_tanh` activation on gate output
    - Elementwise multiply of activated gate with up projection output
    - Down projection
  - TTNN ops: `ttnn.linear` for projections, `ttnn.gelu` (with tanh approximation), `ttnn.mul`, `ttnn.all_reduce`
  - Program config recommendations for `[5376, 21504]` and `[21504, 5376]` matmuls on T3K

- `ple_module.md`
  - `TTNNGemma4PLE` implementation:
    - Second embedding lookup (on host or device)
    - Per-layer linear projection to hidden_size
    - Residual addition into the decoder layer input
  - Whether PLE should run on host (simple lookup + small matmul) or device
  - Multimodal PLE handling: pad token positions for vision/audio tokens

- `full_model_module.md`
  - `TTNNGemma4Model` top-level structure:
    - Token embedding lookup
    - PLE precomputation for all layers
    - 60x decoder layer loop (alternating sliding/global attention)
    - Final RMSNorm
    - LM head (tied with embedding weights)
    - Logit soft-capping (scale by 30.0, tanh, scale back)
  - Weight loading: HuggingFace checkpoint to TTNN weight mapping
  - KV cache initialization for both layer types
  - Decode loop orchestration

---

### Chapter 8 --- Performance Analysis and Optimization Roadmap

**Description:** Analyzes the expected performance characteristics of the
Gemma 4 31B implementation on T3K and identifies the key optimization
opportunities.

**Directory:** `ch8_performance/`

**Files:**

- `index.md`
  - Chapter overview
  - Summary of key performance metrics and bottlenecks

- `memory_budget.md`
  - Weight memory: 60 layers x (attention weights + FFN weights) at BF16 and with quantization options
  - KV cache memory: 48 sliding layers x 1024 window + 12 global layers x full context, per device after TP sharding
  - Activation memory: peak per-layer activation footprint
  - Total DRAM budget per device and whether the model fits on T3K with 12 GB DRAM per chip
  - Quantization requirements: which weights must be quantized (BFP8, BFP4) to fit

- `decode_latency_analysis.md`
  - Per-layer latency breakdown: attention (QKV proj + RoPE + SDPA + O proj), FFN (gate/up + activation + down), norms, PLE
  - Sliding layer vs global layer latency comparison (sliding has smaller SDPA cost due to window, but more KV heads)
  - CCL overhead: all-reduce after each row-parallel matmul
  - Expected total decode latency for 60 layers
  - Comparison with similar-sized models already running on T3K

- `optimization_roadmap.md`
  - Metal Trace capture for the decode loop
  - Multi-CQ overlap opportunities (CCL pipelining with compute)
  - Fused QKV optimization for sliding layers; fused Q+K optimization for global layers
  - DRAM-sharded weight storage for decode matmuls
  - BFP8 KV cache to halve KV memory
  - Potential for KV sharing across layers (`num_kv_shared_layers > 0` in future variants)
  - V-norm fusion with the KV cache write path
  - Partial RoPE optimization: avoiding the split-apply-concat overhead

---

## Conventions

### Terminology

| Term | Definition |
|------|------------|
| Sliding layer | A decoder layer using sliding-window attention with 16 KV heads, head_dim=256, window=1024 |
| Global layer | A decoder layer using full causal attention with 4 KV heads, head_dim=512 |
| K=V sharing | Global-layer optimization where a single projection weight produces both K and V tensors |
| V-norm | RMSNorm applied to value vectors without a learned scale parameter (`with_scale=False`) |
| p-RoPE | Proportional RoPE with high theta (1M) and partial rotary factor (0.25) used in global layers |
| PLE | Per-Layer Embeddings --- a second embedding table injecting residual signals per decoder layer |
| GeGLU | Gated activation FFN using GELU with tanh approximation as the gate function |
| `head_dim` | Dimension of each attention head; 256 for sliding, 512 for global |
| `num_kv_heads` | Number of key-value heads; 16 for sliding, 4 for global |
| `hidden_size` | Model hidden dimension: 5376 |
| `intermediate_size` | FFN intermediate dimension: 21504 |
| TP | Tensor parallelism --- sharding weights and activations across devices |
| T3K | Tenstorrent Galaxy board with 8 Wormhole chips in a 1x8 mesh |
| CCL | Collective Communication Library (TTNN multi-device communication) |
| `paged_sdpa_decode` | `ttnn.transformer.scaled_dot_product_attention_decode` with page table |
| BF16 | BFloat16 (2 bytes/element) |
| BFP8 | Block floating point 8-bit (`bfloat8_b`, 1 byte/element) |
| BFP4 | Block floating point 4-bit (`bfloat4_b`, 0.5 bytes/element) |

### Notation

- Tensor shapes use square brackets with named dimensions: `[B, H, S, D]`.
- Weight shapes use `[in_features, out_features]` (PyTorch convention).
- Complexity expressions use big-O: O(B * H * S * D).
- All memory sizes are in bytes unless stated otherwise, with dtype specified.
- TTNN op names use code font: `ttnn.linear`, `ttnn.all_reduce`.
- Layer indices are 0-based: layer 0 through layer 59.
- Device indices are 0-based: device 0 through device 7.

### Formatting Rules

- Every file begins with a `# Title` H1 header.
- Section headers use `##` (H2) and `###` (H3); no deeper nesting.
- Diagrams are ASCII art in code blocks with `text` language tag.
- Tables use GitHub-flavored Markdown pipe syntax.
- Equations use LaTeX math fences (` ```math ``` `).
- Cross-chapter references use relative paths: `../ch1_architecture_overview/layer_organization.md`.
- Cross-guide references use the guide directory name: `windowed_attention_foundations_and_t3k_mapping`.
- No external URLs --- all findings are self-contained.

---

## Cross-Chapter Dependencies

```
Ch1 (Architecture Overview)
  ├── Ch2 (Projection Shapes)         — uses config parameters and layer type definitions from Ch1
  ├── Ch3 (K=V Sharing & V-Norm)      — uses attention config details from Ch1
  └── Ch4 (Dual RoPE)                 — uses RoPE parameters from Ch1
        │
Ch2 ────┤
Ch3 ────┤
Ch4 ────┤
        ▼
Ch5 (Attention Module Design)          — uses shapes from Ch2, K=V logic from Ch3, RoPE from Ch4
        │
        ▼
Ch6 (TP Sharding)                      — uses head counts from Ch1, tensor shapes from Ch2, module design from Ch5
        │
        ▼
Ch7 (Model Assembly)                   — uses all prior chapters to compose the full model
        │
        ▼
Ch8 (Performance)                      — uses shapes from Ch2, sharding from Ch6, module structure from Ch7
```

**Explicit dependencies by chapter:**

- **Chapter 2** requires: config parameters (hidden_size, head counts, head dims, intermediate_size) from Chapter 1.
- **Chapter 3** requires: attention configuration details (K=V flag, V-norm behavior, per-layer-type differences) from Chapter 1; projection shapes from Chapter 2.
- **Chapter 4** requires: RoPE parameters (theta, partial_rotary_factor) per layer type from Chapter 1; head_dim values from Chapter 2.
- **Chapter 5** requires: all projection shapes from Chapter 2; K=V sharing dataflow from Chapter 3; RoPE application logic from Chapter 4; sliding window and paged SDPA concepts from the windowed attention guide.
- **Chapter 6** requires: head counts and hidden_size from Chapter 1; weight and activation shapes from Chapter 2; the attention module interface from Chapter 5.
- **Chapter 7** requires: layer organization from Chapter 1; all submodule designs from Chapters 3, 4, 5; sharding decisions from Chapter 6.
- **Chapter 8** requires: weight and activation sizes from Chapter 2; sharding strategy from Chapter 6; full module structure from Chapter 7.
