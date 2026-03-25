# Guide Plan: TT Transformers Key Optimizations

**Topic:** Key optimizations implemented in tt-transformers for LLM inference on Tenstorrent hardware
**Source repository:** `tenstorrent/tt-metal` — the `models/tt_transformers/` directory is what the community refers to as "tt-transformers"
**Created:** 2026-03-24

---

## 1. Audience

**Primary reader:** A software engineer who:
- Is familiar with transformer architecture (attention, MLP, layer norm, KV cache) and LLM inference concepts (prefill vs. decode, GQA, paged attention)
- Writes PyTorch and understands GPU programming concepts (memory bandwidth, tile-level compute, kernel fusion)
- Is new to Tenstorrent hardware (Wormhole/Blackhole) and TTNN — they have never written a TTNN op or configured a matmul program config before

**What they already know:**
- FlashAttention-2 algorithm and why it matters (online softmax, tiling over KV blocks)
- How to profile PyTorch models (torch.profiler, CUDA events)
- Block floating point and quantization basics (INT8, FP16)
- The idea of tensor parallelism and data parallelism

**What they do NOT know:**
- Tenstorrent Tensix architecture (matrix engine, SFPU, Packer/Unpacker, NoC, L1 vs DRAM topology)
- TTNN programming model (TensorMemoryLayout, BufferType, TilizedLayout, ProgramConfig variants)
- Math fidelity (LoFi / HiFi2 / HiFi3 / HiFi4) and its throughput implications
- Tenstorrent-specific data formats (BFP4, BFP8, block-shared exponent encoding)
- How collective communication (all-gather, reduce-scatter) maps onto a device mesh

---

## 2. Chapter List

### Chapter 1 — Tenstorrent Hardware and TTNN Foundations
**Directory:** `ch1_hardware_and_ttnn_foundations/`

**Description:** Establishes the mental model every subsequent chapter builds on — Tensix core internals, memory hierarchy, data formats, and how TTNN expresses tensors and operations.

**Files:**

#### `index.md`
- One-paragraph chapter overview and reading order
- Prerequisites and what the reader will be able to do after this chapter

#### `tensix_architecture.md`
- Tensix core components: matrix FPU (8×16 × 16×16 primitive), SFPU, Packer, Unpacker
- Register file (Dst): capacity in fp16 mode (8 tiles) vs fp32 mode (4 tiles), implication for output subblock sizing
- NoC (Network-on-Chip): intra-chip bandwidth, how tiles are routed between cores
- L1 SRAM (120 KB per core, fast) vs DRAM (large, shared, higher latency) — fundamental reason sharding to L1 is a performance goal
- Wormhole peak throughput numbers: LoFi 262 TOPS, HiFi2 148 TOPS, HiFi4 74 TOPS per device

#### `ttnn_tensor_model.md`
- TilizedLayout vs RowMajorLayout: why 32×32 tiles are the native computation unit; 16×16 "faces" inside each tile align with the matrix engine primitive
- BufferType: `DRAM` (interleaved across banks, default) vs `L1` (local to cores, used for sharding)
- TensorMemoryLayout: `INTERLEAVED` vs `HEIGHT_SHARDED` / `WIDTH_SHARDED` / `BLOCK_SHARDED`
- How to create a sharded memory config (`ttnn.create_sharded_memory_config`)
- Data types: BF16, FP32, BFP8_B (16-datum block, 7-bit mantissa + shared exponent), BFP4_B (4-bit mantissa + shared exponent); memory footprint comparison table

#### `math_fidelity_and_data_formats.md`
- Anatomy of a fidelity stage: the 5-bit × 7-bit hardware multiplier; how LoFi/HiFi2/HiFi3/HiFi4 consume mantissa bits across 1–4 passes
- Throughput vs accuracy tradeoff table (LoFi = 4× faster than HiFi4 for matmul)
- `WormholeComputeKernelConfig` fields: `math_fidelity`, `math_approx_mode`, `fp32_dest_acc_en`, `packer_l1_acc`
- When `packer_l1_acc` helps: multi-tile accumulation sequences; L1 += Dst semantics
- Decision guide: which fidelity to use for QKV projections, attention score computation, MLP layers

---

### Chapter 2 — Attention Optimizations: FlashAttention, GQA, and Paged Decode
**Directory:** `ch2_attention_optimizations/`

**Description:** Covers the full attention kernel stack in tt-transformers, from the prefill FlashAttention-2 implementation to Flash-Decode for single-token generation, GQA head grouping, paged KV cache, and sliding window attention.

**Files:**

#### `index.md`
- Chapter overview and map of prefill path vs decode path
- TTNN API entry points: `ttnn.transformer.scaled_dot_product_attention` (prefill), `ttnn.transformer.scaled_dot_product_attention_decode` (decode), `ttnn.transformer.paged_scaled_dot_product_attention_decode` (paged decode)

#### `flash_attention_prefill.md`
- Why naive attention is memory-bandwidth-bound: O(S²) intermediate writes to DRAM
- FlashAttention-2 tiling: parallelizing Q chunks across Tensix cores with online softmax on KV blocks; intermediates stay in L1 (120 KB), never go to DRAM
- Causality-aware load balancing on fixed core grids: pairing Q_low and Q_high chunks per core achieves 1.6× speedup vs naive sequential Q assignment
- Sparse causal mask: applying the triangular mask only to diagonal KV blocks, skipping fully-masked blocks entirely
- Double-buffering with circular buffers: Reader/Writer/Compute kernels run asynchronously; Q, K, V tiles held in dual-slot L1 buffers to overlap data movement with compute
- Achieved speedup: 20× over naive DRAM-based attention baseline (9×–44× range across sequence lengths 512–16 K and head dims 64/128/256)
- `SDPAProgramConfig` parameters: `compute_with_storage_grid_size`, `q_chunk_size`, `kv_chunk_size`
- Sliding window attention: `sliding_window_size` restricts attention to last N tokens, avoiding loading irrelevant KV blocks
- Windowed SDPA for vision attention modules

#### `flash_decode_and_gqa.md`
- Flash-Decode algorithm: single token Q attending to full KV cache, parallelized over the KV sequence dimension (instead of Q dimension as in prefill)
- Input shapes: Q `[1 × batch × n_heads × head_dim]`, K/V `[batch × n_kv_heads × seq × head_dim]`
- GQA / MQA: multiple Q heads per KV group — how head grouping maps onto core parallelization
- `SDPAMultiCoreProgramConfig`: parallelization over batch and KV sequence (not Q sequence)
- `cur_pos_tensor`: per-batch position tracking enables selective decode (skip finished sequences with pos = -1)
- Compute kernel config for decode: typically HiFi2 for attention score accumulation, HiFi4 rarely needed
- Ring-distributed SDPA: `ttnn.transformer.ring_distributed_scaled_dot_product_attention` reduces redundant computation from causal masking across multi-device ring topology

#### `paged_attention_kv_cache.md`
- Memory fragmentation problem with contiguous KV caches: fixed-length allocation wastes memory for variable-length sequences
- Paged KV cache: KV blocks allocated in fixed-size pages; a page table maps sequence positions to physical page indices
- TTNN API: `ttnn.transformer.paged_scaled_dot_product_attention_decode` and `paged_flash_multi_latent_attention_decode`
- Page table tensor type and how it participates in the attention kernel (DRAM reads keyed by page index)
- Program caching for page table tensors: avoiding recompilation across decode steps when page table shape is stable
- `paged_update_cache` operation: fusing K-cache and V-cache updates by sharding K on cores [0–8] and V on cores [8–16] for parallel DRAM writes
- Multi-Latent Attention (MLA) support: `flash_mla_prefill` and `chunked_flash_mla_prefill` for DeepSeek-style compressed KV projection

---

### Chapter 3 — MatMul Optimizations: Program Configs, Sharding, and Kernel Tuning
**Directory:** `ch3_matmul_optimizations/`

**Description:** Covers how `ttnn.matmul` is configured for peak throughput in LLM linear layers — from choosing the right program config and sharding strategy, to tuning output subblocks, math fidelity, and weight data formats.

**Files:**

#### `index.md`
- Chapter overview: the four matmul program configs and when to use each
- Connection to LLM layer types: QKV projection (decode, small M), MLP FF1/FF2 (large N), output projection

#### `matmul_program_configs.md`
- Why `ttnn.matmul` requires explicit program config for high performance (auto-heuristic covers common cases but cannot always match hand-tuned configs)
- `MatmulMultiCoreReuseProgramConfig`: standard 2D tiled matmul; weight reuse across a core grid; parameters: `compute_with_storage_grid_size`, `in0_block_w`, `out_subblock_h`, `out_subblock_w`, `per_core_M`, `per_core_N`
- `MatmulMultiCoreReuseMultiCastProgramConfig`: multicast weights to multiple cores — preferred for tall-M (prefill) workloads; reduces redundant DRAM reads
- `MatmulMultiCoreReuseMultiCast1DProgramConfig`: for width-sharded or height-sharded 1D operands (common in fused MLP); activation shard fed directly, output shard computed in-place
- `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`: weight matrix sharded directly across DRAM banks; each core pulls its own shard, enabling bandwidth-bound decode to use maximum DRAM parallelism; activation interleaved in L1
- Output subblock dimensions (`out_subblock_h` × `out_subblock_w`): must fit in Dst register (8 tiles fp16, 4 tiles fp32); larger subblocks improve register reuse, reduce Packer overhead
- `in0_block_w`: controls K-dimension tile block size for input reuse; larger values increase L1 pressure but reduce NoC transactions

#### `weight_layout_and_quantization.md`
- Default weight layout for LLM inference: `TILE` layout, `DRAM` interleaved at load time, pre-converted to target dtype before device execution
- BFP8 vs BFP4 for MLP weights: practical impact — BFP4 FF1/FF3 weights gave 8B Llama model +22% tokens/s/user (23 → 28 t/s/u); halved bandwidth per weight element
- BFP8 for attention weights: lower sensitivity for QKV projections in most models; Llama 3 series tested as insensitive to attention precision at BFP8
- Layer-specific precision table from PERF.md: `ff1_3` (BFP4), `ff2` (BFP4/BFP8), `wqkv` (BFP8), `wo` (BFP8); KV-cache quantized to BFP8
- `math_fidelity` pairing with dtype: BFP4 weights → LoFi sufficient; BFP8 weights → HiFi2; BF16 activations with BFP8 weights → HiFi2 or HiFi4 for accuracy-sensitive layers
- Transposed weight layout: using `transpose_b=True` vs pre-transposing on host; effect on program config selection

#### `l1_sharding_for_matmul.md`
- When to shard activations to L1: decode phase (small M, large N) benefits most; prefill (large M) can also benefit from activation sharding
- Height sharding activations for 1D matmul (batch dimension along cores): each core gets a slice of the batch, weight broadcast via NoC multicast
- Block sharding for 2D matmul: both activation rows and weight columns distributed; requires Reduce-Scatter / All-Gather if followed by non-local ops
- DRAM-sharded matmul (decode): weight columns stored across DRAM banks; activation rows in L1; no weight broadcast needed — each core fetches its own weight shard
- Throughput comparison: interleaved DRAM matmul vs L1-sharded vs DRAM-sharded for decode (single token, large batch)

---

### Chapter 4 — Memory Access Patterns and Tensor Sharding Strategies
**Directory:** `ch4_memory_and_sharding/`

**Description:** Explains Tenstorrent's memory hierarchy in depth, how tensor sharding maps computation to cores, and the specific sharding patterns used for each transformer sublayer in tt-transformers.

**Files:**

#### `index.md`
- Chapter overview: memory hierarchy diagram (L1 per-core → DRAM → PCIe host)
- Why memory access is the primary performance bottleneck in decode vs compute in prefill

#### `memory_hierarchy.md`
- L1 SRAM: 120 KB per Tensix core, local, low-latency — used for active tiles during computation
- DRAM: shared across cores via NoC; interleaved pages distributed round-robin across DRAM banks; bandwidth ~300 GB/s aggregate on Wormhole
- Host memory (DRAM on host side): weights uploaded once, remain on device across inferences (weight caching)
- Buffer management: circular buffers as L1 staging areas; double-buffering hides DRAM latency by prefetching the next tile while compute runs on the current one
- Interleaved vs sharded allocation: interleaved distributes pages uniformly (flexible, but each core must fetch across all banks); sharded concentrates a core's data locally (better locality, less NoC traffic)

#### `sharding_patterns_in_tt_transformers.md`
- Attention QKV projection (decode): activation width-sharded across head groups; weight DRAM-sharded for bandwidth saturation; result height-sharded for next matmul
- Attention output projection (`wo`): activation height-sharded (head results from all cores); weight DRAM-sharded or block-sharded
- MLP FF1/FF2/FF3 (SwiGLU): FF1 and FF3 computed in parallel, each width-sharded; element-wise SiLU × gate computed in L1 without DRAM round-trip; FF2 follows with its own shard config
- Layer norm / RMSNorm: element-wise ops run on height-sharded activations without resharding; fused into preceding matmul output where possible
- KV cache update (`update_cache`): K and V sharded on disjoint core sets (cores [0–8] for K, cores [8–16] for V) for parallel DRAM write

#### `double_buffering_and_pipelining.md`
- Circular buffer mechanics in TT-Metalium: Reader kernel writes to CB slots; Compute kernel consumes from the same CB; Writer kernel reads from output CB
- Double-buffering: 2 slots per CB means Reader can prefetch tile N+1 while Compute works on tile N; eliminates stalls for memory-bandwidth-bound ops
- Async kernel execution: Reader, Compute, Writer kernels run on independent RISC-V cores within one Tensix core; true pipeline overlap
- When double-buffering helps most: DRAM-bandwidth-bound ops (large weight reads in decode); less useful when compute-bound (large-batch prefill matmuls)

---

### Chapter 5 — Multi-Device Scaling: Tensor Parallelism and Data Parallelism
**Directory:** `ch5_multi_device_scaling/`

**Description:** Covers how tt-transformers distributes large LLMs across multiple Tenstorrent chips using tensor parallelism (weight sharding + collective communication) and data parallelism (batch replication), and the hybrid strategies used for production deployments.

**Files:**

#### `index.md`
- Chapter overview: device topologies supported (N300 2-chip, T3K 8-chip, Galaxy 32-chip)
- SPMD programming model: MeshDevice as virtual device, single dispatch dispatches to all chips

#### `tensor_parallelism.md`
- Column parallelism (split output features): QKV projection weight sharded along output dim; each device computes partial Q/K/V; no inter-device comm until attention output
- Row parallelism (split input features): output projection (`wo`) weight sharded along input (head) dim; each device holds all heads for its heads, needs Reduce-Scatter after local matmul
- `ShardTensor2dMesh` mapper: how attention weights (QKV) and MLP weights are sliced across the mesh for T3K (1×8, dim=(3,2)) vs Galaxy (8×4, dims=(2,3))
- All-Gather after column-parallel: reconstructs full activation for the next layer
- Reduce-Scatter after row-parallel: aggregates partial sums across devices
- CCL operations in TTNN: `ttnn.all_gather`, `ttnn.reduce_scatter`; line vs ring topology selection
- Head distribution: `n_heads / n_devices` attention heads per device; GQA head grouping preserved

#### `data_parallelism.md`
- Data parallelism for throughput: replicate full model across submeshes (each submesh = 1 TP group), each submesh serves a different user batch
- Hybrid TP+DP example: T3K (8 chips) as 1 TP group at batch 1 vs Galaxy (32 chips) split into 4 TP groups × 8 chips each with DP=4 at batch 4
- `tt-run` YAML-based rank bindings for multi-process SPMD launch; `TT_VISIBLE_DEVICES` isolation
- Throughput scaling results: 65 tokens/s/user decode on Galaxy with Llama 3.3-70B at batch 32

#### `collective_communication_optimization.md`
- Why CCL latency matters in decode: each decode step requires All-Gather (before or after attention) and Reduce-Scatter (after MLP); CCL time can dominate decode latency at small batch sizes
- Ring topology: each device sends/receives from one neighbor; low bandwidth requirements, O(n) steps
- Line topology: better for rectangular meshes; selected based on mesh geometry
- Pipelining matmul with collective: overlapping the all-gather of the next layer with the compute of the current layer where buffer space allows
- Future directions: fused matmul+all-reduce kernels being investigated

---

### Chapter 6 — LLM-Specific Optimizations: Quantization, Fused Ops, and Inference Pipeline
**Directory:** `ch6_llm_specific_optimizations/`

**Description:** Covers the cross-cutting optimizations that wire all of the above together for end-to-end LLM inference: mixed-precision quantization strategies, fused operator implementations (RoPE, RMSNorm, SwiGLU), program/weight caching, prefill chunking, and profiling.

**Files:**

#### `index.md`
- Chapter overview: how the pieces assemble into a complete decode loop
- Performance numbers context: Llama 3.1 8B on N150 (~28 t/s/u), 70B on T3K, 70B on Galaxy batch 32

#### `mixed_precision_strategy.md`
- Block floating point formats: BFP8 (7-bit mantissa, 1 shared exponent per 16 values, ~50% of BF16 bandwidth) and BFP4 (3-bit mantissa, ~25% of BF16 bandwidth)
- Default precision policy in tt-transformers:
  - MLP weights (FF1, FF3): BFP4 → 2× lower bandwidth than BFP8, +22% decode throughput on 8B model
  - MLP weights (FF2): BFP4 or BFP8 depending on model sensitivity
  - Attention weights (QKV, output): BFP8 — more sensitive to quantization noise
  - KV cache: BFP8 — halves cache memory, allows longer context at same DRAM capacity
  - Activations: BF16 (intermediate) or BFP8 where accuracy budget permits
- Math fidelity pairing table: BFP4 weight → LoFi; BFP8 weight × BF16 act → HiFi2; BF16 × BF16 → HiFi4
- Accuracy mode vs performance mode: PERF.md configurations, accuracy evaluated on 512-token prefill + 511 decode tokens
- Model-specific exceptions: Qwen-2.5-7B uses BFP8 MLP (sensitivity to BFP4); Llama-3.1-8B uses BFP8 in only the 32nd decoder layer
- Impact on context length: BFP8 KV cache enables 64 K and 131 K context window tests on smaller devices via chunked prefill

#### `fused_ops.md`
- RoPE (Rotary Position Embedding): `fused_rotary_embedding` op applies RoPE to Q and K in parallel; Q height-sharded on cores [0–31], K height-sharded on cores [32–63]; avoids sequential Q-then-K processing
- RMSNorm / LayerNorm: elementwise op fused with preceding matmul output in L1; avoids DRAM round-trip; `math_approx_mode=True` in compute kernel config reduces rsqrt cost
- SwiGLU MLP: FF1 and FF3 weight projections computed simultaneously (both column-parallel on separate device subsets or separate core sets); element-wise `silu(FF1(x)) × FF3(x)` fused in L1 before FF2
- Fused activation in matmul: `fused_activation` field in `MatmulMultiCoreReuseMultiCast1DProgramConfig` applies an activation function in the Packer unit without writing intermediate to L1/DRAM

#### `program_and_weight_caching.md`
- JIT compilation overhead: first inference compiles all kernels to RISC-V binaries; subsequent identical shapes reuse cached binaries from `TT_METAL_CACHE` directory
- Program caching: compiled programs stored persistently; decode loop reuses the same program every step when input shapes are stable (batch size, head dim, KV length unchanged)
- Weight caching: model weights remain resident on device DRAM across requests; no PCIe transfer per inference; weight upload happens once at model load
- Page table caching: paged SDPA decode supports program caching for page table tensors when page table shape does not change between steps
- Implications for latency: first-token latency (TTFT) includes compilation overhead on cold start; subsequent requests hit cache

#### `prefill_chunking_and_context_length.md`
- Chunked prefill: for very long contexts (131 K tokens) that exceed L1 working set, prefill is split into fixed-size chunks (e.g., 512 or 2 K tokens); each chunk updates KV cache incrementally
- KV cache growth during chunked prefill: pages allocated as chunks are processed; paged KV cache natural fit for chunked prefill
- Prefill math fidelity: HiFi4 for attention score computation (full precision needed for long-context accuracy); MLP can use HiFi2
- Decode vs prefill configuration switching: different `WormholeComputeKernelConfig` objects passed to the same ops at runtime

#### `profiling_and_tuning.md`
- Tracy profiler integration: `ttnn.tracer` wraps op dispatch; per-op start/stop timestamps from device; identifying bottleneck ops
- Performance metrics: T/S/U (tokens/second/user = 1/inter-token latency), T/S (total throughput = T/S/U × batch), TTFT (time-to-first-token)
- Common bottlenecks and fixes:
  - Low T/S/U: CCL latency dominates → check TP degree, reduce device count or increase batch
  - Low T/S: matmul underutilization → check `out_subblock` sizing, math fidelity, weight dtype
  - High TTFT: chunked prefill overhead or cold JIT cache → pre-warm cache, tune chunk size
- PERF.md as configuration reference: decode fidelity settings per model, data parallel configs

---

### Chapter 7 — End-to-End Model Walkthrough: Llama 3 on Tenstorrent
**Directory:** `ch7_llama3_walkthrough/`

**Description:** Ties all previous chapters together through a concrete walkthrough of Llama 3 inference in tt-transformers, tracing each optimization choice layer by layer from token embedding through final logits.

**Files:**

#### `index.md`
- Chapter overview: why Llama 3 is the reference model in tt-transformers
- Mapping of Llama 3 architecture (RMSNorm, GQA, RoPE, SwiGLU, BF16 residuals) to TTNN optimization choices

#### `llama3_decode_path.md`
- Token embedding lookup: DRAM-interleaved embedding table, no sharding needed
- QKV projection: `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`, BFP8 weight, HiFi2, activation BF16 in L1
- Fused RoPE on Q and K: `fused_rotary_embedding`, Q on cores [0–31], K on cores [32–63]
- Paged decode attention: `paged_scaled_dot_product_attention_decode`, HiFi2, GQA head grouping, per-batch `cur_pos_tensor`
- KV cache update: `paged_update_cache`, parallel K/V DRAM write on disjoint core sets
- Output projection: `wo` matmul, DRAM-sharded weights, Reduce-Scatter across TP devices
- All-Gather activation before MLP
- SwiGLU MLP: FF1/FF3 parallel BFP4 matmuls, fused SiLU×gate in L1, FF2 BFP4 matmul, Reduce-Scatter
- RMSNorm: fused with approx mode, output in BF16 for residual add
- Residual add in BF16 in L1

#### `llama3_prefill_path.md`
- How prefill differs from decode: large M dimension, fully compute-bound, no paged decode
- QKV: `MatmulMultiCoreReuseMultiCastProgramConfig` with multicast weights, large per_core_M
- FlashAttention-2 prefill: `scaled_dot_product_attention`, HiFi4 for score computation, SDPAProgramConfig chunk tuning
- Chunked prefill for 131 K context: chunk size selection, KV page allocation per chunk
- T3K multi-device prefill: attention heads split across 8 devices; FlashAttention runs per-device on local Q/K/V shards; All-Gather before output projection

#### `performance_numbers_and_tuning_checklist.md`
- Reference performance table: N150 (8B, ~28 t/s/u decode), N300 (8B), T3K (70B), Galaxy (70B batch 32, ~65 t/s/u)
- Step-by-step tuning checklist for a new model bring-up:
  1. Choose data types for each weight matrix (BFP4 for MLP, BFP8 for attention as starting point)
  2. Select math fidelity per op (LoFi for BFP4, HiFi2 for BFP8, HiFi4 for attention prefill)
  3. Pick program config per matmul (DRAM-sharded for decode, multicast for prefill)
  4. Set output subblock dims to maximize Dst utilization
  5. Enable `packer_l1_acc` for multi-step accumulations
  6. Profile with Tracy, identify CCL or compute bottleneck, adjust TP degree or batch size

---

## 3. Conventions

### Terminology

| Term | Definition used in this guide |
|---|---|
| **tt-transformers** | The `models/tt_transformers/` directory inside `tenstorrent/tt-metal`; the reference LLM implementation for Tenstorrent hardware |
| **TTNN** | The Python/C++ operator library (`ttnn` package); the primary programming surface above TT-Metalium |
| **TT-Metalium** | The low-level kernel programming model; not used directly in this guide except when explaining circular buffers and kernel dispatch |
| **Tensix core** | One compute tile on a Tenstorrent chip; contains matrix FPU, SFPU, Packer, Unpacker, and two RISC-V control cores |
| **Wormhole** | The current Tenstorrent chip generation (Wormhole B0); used in N150, N300, T3K, Galaxy configurations |
| **Blackhole** | Next-generation Tenstorrent chip; optimizations described here apply to Wormhole unless stated otherwise |
| **L1** | Per-core SRAM (120 KB); fast, local to one Tensix core |
| **DRAM** | Off-chip shared memory; higher capacity, higher latency than L1 |
| **BFP8** | Block Floating Point 8: 16 values share one 8-bit exponent, each value has 7-bit mantissa; roughly half the memory of BF16 |
| **BFP4** | Block Floating Point 4: same block structure as BFP8 but 3-bit mantissa; roughly quarter the memory of BF16 |
| **Math fidelity** | Hardware precision mode: LoFi (1 pass, fastest), HiFi2 (2 passes), HiFi3 (3 passes), HiFi4 (4 passes, most accurate) |
| **packer_l1_acc** | Compute kernel config flag enabling L1 += Dst accumulation in the Packer unit |
| **Program config** | A TTNN configuration object (e.g., `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`) that controls core count, tile blocking, and data movement for an op |
| **SDPA** | Scaled Dot-Product Attention; the underlying attention kernel |
| **GQA** | Grouped Query Attention; multiple Q heads per KV head group |
| **MQA** | Multi-Query Attention; extreme case of GQA with one KV head |
| **MLA** | Multi-Latent Attention; DeepSeek compressed KV projection variant |
| **Prefill** | Processing the full input prompt in one (or chunked) forward pass; compute-bound, large sequence lengths |
| **Decode** | Generating one new token per step; memory-bandwidth-bound, small M dimension |
| **TP** | Tensor Parallelism; weights sharded across devices with collective communication |
| **DP** | Data Parallelism; full model replicated across device groups, different batches processed independently |
| **CCL** | Collective Communication Library; TTNN's All-Gather, Reduce-Scatter, All-Reduce implementations |
| **T/S/U** | Tokens per Second per User = 1 / inter-token latency |
| **TTFT** | Time to First Token; latency of the prefill phase |

### Notation

- Code identifiers (class names, function names, config fields) are always written in `monospace`.
- TTNN API calls are shown as `ttnn.<namespace>.<function>()` matching the actual import path.
- Tensor shapes are written as `[dim0 × dim1 × dim2 × dim3]` with `×` (multiplication sign), matching TTNN documentation convention.
- Performance numbers are cited with the source (PERF.md, tech report, or issue) in a parenthetical footnote.
- When a behavior differs between prefill and decode, a callout box labeled **Prefill** or **Decode** is used.
- Hardware generation qualifiers: "on Wormhole" or "on Blackhole" when the behavior is generation-specific.

### Formatting Rules

- All chapters use H2 (`##`) for top-level sections and H3 (`###`) for subsections; H4 is permitted only for deeply nested detail.
- Code examples use fenced blocks with `python` or `cpp` language tags.
- Tables are used for: data type comparisons, math fidelity throughput, precision decision guides, and performance number summaries.
- Cross-references use relative Markdown links (`../ch1_hardware_and_ttnn_foundations/tensix_architecture.md`).
- Each file ends with a **Key Takeaways** section (3–5 bullet points) and a **Further Reading** section linking to official docs or tech reports.

---

## 4. Cross-Chapter Dependencies

The chapters are ordered from foundational (Ch1) to applied (Ch7). Each chapter may reference concepts introduced in prior chapters as noted below.

| Chapter | Depends On |
|---|---|
| **Ch2 — Attention Optimizations** | Ch1 (`tensix_architecture.md` for L1 size justifying tile-in-L1 approach; `math_fidelity_and_data_formats.md` for HiFi2/HiFi4 choices; `ttnn_tensor_model.md` for sharded tensor shapes) |
| **Ch3 — MatMul Optimizations** | Ch1 (`tensix_architecture.md` for Dst register capacity → output subblock sizing; `math_fidelity_and_data_formats.md` for fidelity × dtype pairing; `ttnn_tensor_model.md` for DRAM interleaved vs L1 sharded) |
| **Ch4 — Memory and Sharding** | Ch1 (`ttnn_tensor_model.md` for sharding API; `tensix_architecture.md` for L1 vs DRAM hierarchy); Ch3 (`matmul_program_configs.md` for DRAM-sharded matmul; `l1_sharding_for_matmul.md` for L1 sharding patterns) |
| **Ch5 — Multi-Device Scaling** | Ch3 (`matmul_program_configs.md` — column/row parallel matmuls are sharded matmuls); Ch4 (`sharding_patterns_in_tt_transformers.md` — per-layer shard layout assumed known) |
| **Ch6 — LLM-Specific Optimizations** | Ch1 (`math_fidelity_and_data_formats.md` for BFP formats); Ch2 (paged attention, prefill chunking); Ch3 (matmul weight formats); Ch4 (double buffering); Ch5 (CCL for TP); all prior chapters are assumed complete |
| **Ch7 — Llama 3 Walkthrough** | All chapters; Ch7 synthesizes every optimization into a concrete model and should be read last |

### Specific Concept Dependencies

- `packer_l1_acc`: introduced in Ch1 (`math_fidelity_and_data_formats.md`), applied in Ch3 (`matmul_program_configs.md`) and Ch6 (`fused_ops.md`)
- Circular buffers / double-buffering: introduced in Ch2 (`flash_attention_prefill.md`), generalized in Ch4 (`double_buffering_and_pipelining.md`)
- DRAM-sharded matmul config: introduced in Ch3, referenced in Ch4 and Ch7
- Page table / paged KV cache: introduced in Ch2 (`paged_attention_kv_cache.md`), referenced in Ch6 (`program_and_weight_caching.md`, `prefill_chunking_and_context_length.md`) and Ch7
- Math fidelity × dtype table: introduced in Ch1, referenced in Ch2, Ch3, Ch6, and Ch7
- `ShardTensor2dMesh`: introduced in Ch5, referenced in Ch7

---

*End of plan.md*
