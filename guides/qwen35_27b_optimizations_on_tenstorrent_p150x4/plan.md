# Qwen3.5-27B Optimizations on Tenstorrent P150x4 -- Guide Plan

## Audience

This guide is for Tenstorrent engineers and advanced users who want to understand and reproduce the optimizations used to deploy Qwen3.5-27B on the P150x4 (4-chip Blackhole) platform. Readers should already be familiar with:

- Tensor parallelism (TP) concepts and multi-device programming with `ttnn`
- The tt-metal stack: circular buffers, NOC data movement, DRAM vs L1 memory hierarchy, TILE_LAYOUT
- Transformer model architecture fundamentals (attention, MLP, residual connections)
- Basic familiarity with linear attention / state-space model concepts (recurrence states vs KV caches)

Readers do NOT need prior knowledge of the Qwen3.5 architecture, DeltaNet recurrence math, or the specific custom kernel APIs used here.

---

## Chapter List

### Chapter 1: Qwen3.5-27B Architecture and Hardware Mapping

**Description:** Introduces the hybrid GDN/attention architecture of Qwen3.5-27B and explains how it maps to TP=4 across four Blackhole chips.

**Files:**
- `index.md` -- Chapter overview and learning objectives
- `hybrid_architecture.md` -- The 48 GDN + 16 full attention layer structure
- `tp_sharding_strategy.md` -- How weights and activations are sharded across 4 devices

**Content:**

`hybrid_architecture.md`:
- The 64-layer structure: 3 GDN + 1 Attention repeating 16 times (`layer_types` list in `Qwen35ModelArgs`)
- Key model dimensions: hidden=5120, head_dim=256, 24 Q heads, 4 KV heads, MLP intermediate=17408
- GDN-specific dimensions: Nk=16 key heads, Dk=128, Nv=48 value heads, Dv=128 (from `model_config.py` constants)
- How GDN layers replace KV caches with recurrence state tensors of shape `[B*Nv_TP, Dk, Dv]`
- The `Transformer` class in `model.py`: builds with `Qwen35Attention` then swaps GDN layers via `TtGatedDeltaNet`

`tp_sharding_strategy.md`:
- TP=4 dimension splits: `gdn_nk_tp=4`, `gdn_nv_tp=12`, `gdn_qkv_dim_tp=2560`, `gdn_qkvz_dim_tp=4096`
- Column-parallel projections (QKVZ, Q+gate, K, V, MLP w1/w3) sharded along output dim
- Row-parallel projections (GDN out, attention wo, MLP w2) sharded along input dim, followed by all-reduce
- KV head replication when `n_kv_heads < num_devices` (the `replicate_kv_weight` mechanism)
- Weight preparation helpers: `prepare_gdn_qkv` interleaving Q/K/V for clean TP sharding; `prepare_attn_qg` for fused Q+gate; `prepare_conv_taps` for per-TP conv weights
- CCL Ring topology for all-reduce and sampling all-gather on P150x4

---

### Chapter 2: Full Attention Layer Optimizations

**Description:** Covers the Qwen3.5-specific full attention layer, including partial RoPE, QK L2 norms, sigmoid gating, and DRAM-sharded decode matmuls.

**Files:**
- `index.md` -- Chapter overview
- `attention_architecture.md` -- Qwen3.5 attention differences from standard transformers
- `dram_sharded_decode.md` -- DRAM-sharded matmul configuration for decode projections
- `flash_attention_prefill.md` -- Flash SDPA for prefill with 2D matmul configs

**Content:**

`attention_architecture.md`:
- Five differences from standard attention (per `attention.py` docstring): partial RoPE (64/256 dims), QK L2 norms with learned scale, sigmoid output gating, fused Q+gate projection, separate K/V projections
- The `Qwen35PartialRopeSetup` class: precomputes cos/sin tables for 64 dims, HuggingFace split-halves format
- `apply_partial_rope_decode` and `apply_partial_rope_prefill`: slice first 64 dims, apply rotation, concat passthrough dims
- QK RMSNorm: `ttnn.rms_norm` followed by multiply with learned `q_norm`/`k_norm` scale weights

`dram_sharded_decode.md`:
- `create_dram_sharded_mem_config`: WIDTH_SHARDED across 8 DRAM cores, padded to `TILE_SIZE * DRAM_CORES`
- `create_dram_sharded_matmul_program_config`: `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` with M=1 for decode
- The `_shard_linear` pattern: activation sharded to L1 WIDTH_SHARDED, weight in DRAM sharded, output to L1 WIDTH_SHARDED then unsharded to DRAM
- Per-head KV cache update via `paged_update_cache` with HEIGHT_SHARDED shard config (32x256 per core on 8x4 grid)
- BFP8 weights with HiFi2 compute kernel (`WormholeComputeKernelConfig` with `math_fidelity=HiFi2`)

`flash_attention_prefill.md`:
- 2D matmul for prefill projections: `MatmulMultiCoreReuseMultiCastProgramConfig` on 8x8 grid, M parallelized over grid_y, N over grid_x
- `create_prefill_matmul_program_config`: compute `per_core_M`, `per_core_N`, `out_subblock_w` respecting FP32 DST limit (h*w <= 4)
- Flash SDPA: `ttnn.transformer.scaled_dot_product_attention(is_causal=True)` with `SDPAProgramConfig` on 8x8 grid
- Dynamic chunk sizing: `q_chunk=k_chunk=256` for seq_len >= 2048, else 64
- Typecast to bfloat8_b before SDPA for memory efficiency

---

### Chapter 3: GDN Layer Decode Pipeline

**Description:** Details the Gated DeltaNet decode forward pass -- from projections through conv1d, recurrence, and output gating -- covering both fused and unfused paths.

**Files:**
- `index.md` -- Chapter overview
- `gdn_decode_flow.md` -- End-to-end decode dataflow for a single GDN layer
- `conv1d_shift_register.md` -- The 4-tap causal conv1d implemented as a trace-compatible shift register
- `recurrence_math.md` -- DeltaNet recurrence equations and their mapping to tensor operations

**Content:**

`gdn_decode_flow.md`:
- The two paths: `_forward_decode_fused` (full fused kernel) vs `_forward_decode_unfused` (fallback)
- Fused QKVZ projection: single DRAM-sharded matmul producing `[1, B, qkvz_dim_tp]`, split into QKV and Z
- AB projection: separate matmul producing `[1, B, Nv_TP*2]`, split into a and b gate inputs
- Post-recurrence: RMS norm via `ttnn.rms_norm`, SiLU gating with Z, output projection, all-reduce
- Tensor lifecycle and `ttnn.deallocate` discipline for memory management

`conv1d_shift_register.md`:
- 4-element shift register: `states[0..3]` with `ttnn.copy` chain implementing causal convolution
- Weighted sum via `ttnn.multiply` + `ttnn.mac` with precomputed `conv_taps[0..3]`
- SiLU activation on conv output
- Why shift register is trace-compatible (no dynamic indexing, fixed copy pattern)

`recurrence_math.md`:
- DeltaNet recurrence equations:
  1. `state *= exp(g)` -- exponential decay
  2. `kv_mem = k_row @ state` -- [1,K] x [K,V] -> [1,V]
  3. `delta = beta * (v - kv_mem)` -- scaled innovation
  4. `state += outer(k_col, delta)` -- rank-1 state update
  5. `output = q @ state` -- readout
- Gate computation: `beta = sigmoid(b)`, `g = -exp(A_log) * softplus(a + dt_bias)`
- L2 normalization of Q and K before recurrence, Q scaled by `Dk^(-0.5)`
- Head expansion: `repeat_interleave` from Nk_TP=4 key heads to Nv_TP=12 value heads (repeat_factor=3)
- Recurrence state shape: `[B*Nv_TP, Dk, Dv]` = `[32*12, 128, 128]` per device = 12.6 MB per layer in bfloat16

---

### Chapter 4: Custom Fused GDN Kernel

**Description:** Deep dive into the custom C++ kernel that fuses L2 norm, gate computation, and DeltaNet recurrence into a single dispatch, covering the reader/compute/writer architecture.

**Files:**
- `index.md` -- Chapter overview and kernel architecture diagram
- `kernel_dispatch.md` -- Python-side dispatch via `gdn_kernel_op.py` and `ttnn.generic_op`
- `reader_kernel.md` -- The reader dataflow kernel: batched NOC reads and sub-tile extraction
- `compute_kernel.md` -- The compute kernel: L2 norm, gates, recurrence phases
- `writer_kernel.md` -- The writer kernel: output and state writeback

**Content:**

`kernel_dispatch.md`:
- `gdn_full_fused_inplace` API: takes conv_out, a/b scalars, constants, state, output; dispatches single kernel
- `_build_full_fused_device_program`: constructs `ProgramDescriptor` with `KernelDescriptor` for reader/compute/writer
- Multi-device support via `MeshProgramDescriptor`: per-device programs with correct buffer addresses
- Core assignment: pairs distributed across cores with `pairs_per_core + remainder` pattern
- 26 circular buffer descriptors: purpose of each CB index (c_0 through c_31)
- Compile-time args: `Kt=4`, `Vt=4`, `STATE_IN_L1`, `STATE_IS_SHARDED`, `Nv_TP`, `Nk_TP`, `repeat_factor`, tile offsets
- Runtime args: buffer addresses + pair_offset + num_pairs per core
- Kernel content hashing for program cache invalidation

`reader_kernel.md`:
- Batched NOC read strategy: all 44 reads per pair issued before a SINGLE `noc_async_read_barrier()` (vs 17 barriers in naive approach)
- Scratch layout within 1 tile (2048 bytes): Q (512B) + K (512B) + V (512B) + scalars (256B) = 1792B
- Sub-tile row extraction: `issue_row_reads` reads 2 face-halves (cols 0-15 and 16-31) per tile row
- Sub-tile scalar extraction: `issue_scalar_read` reads one 64-byte aligned row containing the target scalar
- Local copy phase (post-barrier): `copy_row_to_tile` and `copy_scalar_to_tile` move data from scratch to CB tiles
- K tiles zeroed before copy (required for correct outer product in compute kernel)
- Pair-to-head mapping: `batch_idx = p / Nv_TP`, `v_head = p % Nv_TP`, `k_head = v_head / repeat_factor`
- Persistent constants: norm_w, scale, rms_scale, rms_eps read once with single barrier
- `generate_reduce_scaler` for the all-ones tile used in reduce operations
- HEIGHT_SHARDED state path: direct L1 memcpy (no NOC) when `STATE_IS_SHARDED=1`

`compute_kernel.md`:
- Phase 1 (L2 Norm Q): transpose -> matmul dot product (avoids reduce_row) -> rsqrt -> multiply by scale -> normalize all Kt tiles
- Phase 2 (L2 Norm K): identical to Phase 1 but without scale multiplication
- Phase 3 (K Transpose): `transpose_wh_tile` for outer product in recurrence
- Phase 4 (Gates): `sigmoid(b)` for beta; `softplus(a + dt_bias)` via `exp` + `log1p`; multiply by `neg_exp_A`
- Phase 5 (Recurrence): exp(g) decay, state_b broadcast multiply, kv_mem matmul, delta subtraction, beta-scaled delta, outer product state update via copy+matmul accumulate, q@state readout
- CB flow: reader pushes per-pair inputs; compute consumes and produces cb_out + cb_state_out; writer drains
- FP32 dest accumulation enabled (`fp32_dest_acc_en=True`) for numerical precision
- HiFi4 math fidelity for recurrence (vs HiFi2 for matmul projections)

`writer_kernel.md`:
- Output written as `[num_pairs, 1, Dv]` in sequential per-pair tile layout
- State writeback: 16 tiles per pair to DRAM (interleaved) or L1 (HEIGHT_SHARDED)
- HEIGHT_SHARDED path: direct L1 memcpy using `volatile tt_l1_ptr` pointers (no NOC write)
- Single `noc_async_write_barrier()` per pair covering both output and state writes

---

### Chapter 5: Prefill TTFT Optimization

**Description:** Explains the optimizations that achieve 5.3x TTFT speedup (498ms/tok to 94ms/tok), covering batched projections, flash attention, and B=1 GDN prefill.

**Files:**
- `index.md` -- Chapter overview with performance numbers
- `batched_projections.md` -- 2D matmul for compute-bound prefill projections
- `gdn_prefill_strategy.md` -- Batched QKVZ + sequential per-token recurrence with B=1 states
- `state_replication.md` -- Post-prefill B=1 to B=32 state replication for KV cache and GDN states

**Content:**

`batched_projections.md`:
- Attention prefill: QKV projections via 2D matmul on full `[1, 1, seq_len, dim]` input
- GDN prefill: QKVZ and AB projections computed once for full sequence using `create_prefill_matmul_program_config`
- Output projection also batched: `[1, 1, seq_len, value_dim_tp]` via 2D matmul after stacking per-token outputs
- Comparison: decode uses DRAM-sharded M=1 matmuls (bandwidth-bound); prefill uses 2D multicast matmuls (compute-bound)

`gdn_prefill_strategy.md`:
- Why GDN prefill cannot be fully parallelized: recurrence state has sequential dependency across tokens
- Hybrid approach: batch projections, then iterate per-token for conv1d + recurrence
- `_init_prefill_states`: creates separate B=1 conv and rec states (not shared with B=32 decode states)
- Per-token loop: slice from pre-computed `qkvz_all` and `ab_all`, run conv1d shift register, run full fused kernel with `num_pairs = 1 * Nv_TP = 12`
- Outputs collected in list and concatenated with `ttnn.concat` before batched output projection
- The 5.3x speedup breakdown: batched projections eliminate per-token matmul dispatch overhead; 2D matmuls utilize compute more efficiently

`state_replication.md`:
- `replicate_prefill_state_to_batch` in `TtGatedDeltaNet`: B=1 prefill rec_states `[Nv_TP, Dk, Dv]` replicated to `[B*Nv_TP, Dk, Dv]` per device via `torch.repeat` + `ShardTensorToMesh`
- `replicate_kv_cache_to_batch` in `Qwen35Attention`: per-device KV cache user 0 expanded to all B=32 slots
- Conv states replicated similarly: `[1, 1, qkv_dim_tp]` expanded to `[1, B, qkv_dim_tp]`
- Cleanup: prefill-specific state tensors deallocated after replication

---

### Chapter 6: L1 State Management and Rolling Window (WIP)

**Description:** Covers the work-in-progress optimization to move GDN recurrence states from DRAM to L1 for reduced bandwidth bottleneck, including the rolling window strategy.

**Files:**
- `index.md` -- Chapter overview and motivation (GDN = 85% of decode time)
- `l1_state_design.md` -- The rolling window L1 state approach and its implementation
- `height_sharded_kernel.md` -- HEIGHT_SHARDED L1 state support in the custom kernel
- `sdpa_l1_conflict.md` -- The SDPA circular buffer conflict challenge

**Content:**

`l1_state_design.md`:
- Bottleneck: GDN recurrence state is 12.6 MB per layer, read+written every decode step; 48 layers = ~1.2 GB DRAM bandwidth per step
- Profiler breakdown: GDN 469.6ms (85%), Attention 69.2ms (12%), Overhead 15.7ms (3%)
- `enable_l1_state()`: loads first 3 GDN layers' states to L1 via `ttnn.to_memory_config(state, ttnn.L1_MEMORY_CONFIG)`
- Rolling window of 3 (matches 3-GDN + 1-Attention pattern): `_swap_l1_state` saves old group to DRAM, loads new group
- Forward pass hooks: wraps each GDN layer's forward to inject swap logic based on `needed_block = gdn_i // W`
- DRAM backup: `_dram_state` reference preserved for each GDN layer; L1 data copied back via `ttnn.to_memory_config(..., output_tensor=gdn._dram_state)`

`height_sharded_kernel.md`:
- HEIGHT_SHARDED state: tiles local to compute cores, no NOC needed for read/write
- Reader kernel: when `STATE_IS_SHARDED=1`, uses `volatile tt_l1_ptr` direct memcpy instead of `noc_async_read_tile`
- Writer kernel: same direct L1 memcpy path for state writeback
- `InterleavedAddrGenFast<false>` (L1 interleaved) vs direct shard offset addressing
- Validation: HEIGHT_SHARDED works for 1-2 layers with correct "Paris" output

`sdpa_l1_conflict.md`:
- SDPA circular buffers temporarily expand into L1 during attention layers
- CB region ends at approximately 1,264 KB/core; GDN state must be placed above this
- Challenge: SDPA CBs overlap with HEIGHT_SHARDED GDN state addresses during the 1-in-4 attention layers
- Potential solutions: reduce SDPA CB footprint, use zero-copy pre-allocated L1 buffers, or coordinate L1 address space between GDN state and SDPA CBs
- Current status: L1 INTERLEAVED works with 4 layers; HEIGHT_SHARDED validated for 1-2 layers

---

### Chapter 7: Performance Analysis and Remaining Bottlenecks

**Description:** Summarizes measured performance, profiler breakdowns, and identifies the remaining optimization opportunities.

**Files:**
- `index.md` -- Chapter overview
- `performance_summary.md` -- Current numbers and comparison to baseline
- `bottleneck_analysis.md` -- Where time is spent and what can be improved

**Content:**

`performance_summary.md`:
- Decode throughput: 14.6 tok/s/user (batch=32, traced)
- TTFT per-token: 94 ms/token (down from 498 ms/token, 5.3x improvement)
- TTFT for 96-token prompt: 9.1s (down from 47.8s)
- Test commands: `test_e2e_generate.py::test_e2e_generate_traced`, `test_ttft.py::test_ttft_batched_prefill`, `test_profile_breakdown.py`

`bottleneck_analysis.md`:
- GDN layers dominate decode: 85% of time (9.78 ms/layer average vs 4.33 ms/layer for attention)
- Root cause: DRAM bandwidth for 12.6 MB state read+write per GDN layer per step
- Expected impact of L1 state: eliminates NOC reads/writes for state, replacing with local L1 memcpy
- Other potential optimizations: further kernel fusion (RMS norm + SiLU into the compute kernel), conv1d fusion, reducing pair dispatch overhead
- Prefill GDN remains sequential (recurrence dependency); potential for chunked/parallel recurrence algorithms
- FP8 weight precision: already using BFP8 with HiFi2 for projections; HiFi4 used for recurrence precision

---

## Conventions

### Terminology
- **GDN**: Gated DeltaNet -- the linear attention variant used in 48 of 64 layers
- **Full Attention**: Standard multi-head attention with KV cache, used in 16 of 64 layers
- **TP**: Tensor Parallelism (TP=4 means sharded across 4 devices)
- **P150x4**: Tenstorrent Blackhole platform with 4 chips
- **Recurrence state** (or **rec_state**): The `[B*Nv_TP, Dk, Dv]` tensor maintained across decode steps for GDN layers (analogous to KV cache for attention)
- **Pair**: A (batch, value_head) combination; `num_pairs = B * Nv_TP` is the unit of work for the fused kernel
- **DRAM-sharded**: Weight matrix stored WIDTH_SHARDED across 8 DRAM cores; matmul uses `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`
- **2D matmul**: Prefill matmul using `MatmulMultiCoreReuseMultiCastProgramConfig` on an 8x8 compute grid
- **CB**: Circular Buffer -- on-chip SRAM buffer used for data movement between reader/compute/writer kernels
- **NOC**: Network-on-Chip -- the data movement fabric connecting cores, DRAM, and L1

### Notation
- Dimensions: `B` = batch size (32 for decode, 1 for prefill), `Dk` = key dim (128), `Dv` = value dim (128), `Nk` = key heads (16), `Nv` = value heads (48), `HD` = attention head dim (256)
- TP-local dimensions use `_TP` suffix: `Nk_TP = Nk / TP = 4`, `Nv_TP = Nv / TP = 12`
- Tile dimensions use `t` suffix: `Kt = Dk / 32 = 4`, `Vt = Dv / 32 = 4`
- State size: `state_tiles = Kt * Vt = 16` tiles per pair

### Formatting Rules
- All code references use backtick formatting: `function_name()`, `variable_name`, `ClassName`
- File paths are relative to the model root (`qwen35_27b/tt/`) unless otherwise noted
- Tensor shapes are written in bracket notation: `[B, Nv_TP, Dv]`
- Memory sizes use standard units: KB, MB; tile size is always 2048 bytes (32x32 bfloat16)
- Performance numbers always specify the measurement context (batch size, traced/untraced, prompt length)

---

## Cross-Chapter Dependencies

| Chapter | Depends On | Concepts Referenced |
|---------|-----------|-------------------|
| Chapter 2 (Attention) | Chapter 1 | TP sharding strategy, DRAM-sharded config builders, partial RoPE dim=64 |
| Chapter 3 (GDN Decode) | Chapter 1 | TP dimensions (Nk_TP, Nv_TP), recurrence state shape, head expansion (repeat_factor=3) |
| Chapter 4 (Fused Kernel) | Chapter 3 | DeltaNet recurrence math, L2 norm + gate equations, pair concept |
| Chapter 5 (Prefill) | Chapters 2, 3 | Flash SDPA from Ch2; GDN recurrence from Ch3; 2D matmul config from Ch2 |
| Chapter 6 (L1 State) | Chapters 3, 4 | Recurrence state size (12.6 MB/layer) from Ch3; HEIGHT_SHARDED kernel paths from Ch4; SDPA CB footprint from Ch2 |
| Chapter 7 (Performance) | All prior | Profiler breakdown references all layer types; optimization opportunities reference specific mechanisms from Chs 2-6 |
