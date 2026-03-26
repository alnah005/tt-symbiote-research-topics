# Plan: TTNNBailingMoEAttention Performance Optimization Guide

## Audience

**Target readers:** ML systems engineers and hardware-aware ML engineers working on tt-symbiote or tt-transformers inference stacks. Readers are expected to understand:
- Python-level TTNN op dispatch and multi-device mesh abstractions (`MeshDevice`, `ReplicateTensorToMesh`, `ShardTensorToMesh`)
- Grouped Query Attention (GQA) mechanics — the distinction between Q heads and KV heads, and why the GQA ratio matters for memory and compute
- Basic Tenstorrent hardware concepts: L1 vs DRAM buffering, tile layout (32×32), core grids, and how sharding strategies map to physical cores
- The decode vs prefill distinction and why they have different performance characteristics

Readers do **not** need prior experience profiling Wormhole hardware or detailed knowledge of the CCL (collective communication layer) internals.

---

## Chapter List

### Chapter 1 — Decode Path Architecture and Op Sequence

**Description:** Walks through every operation in `_forward_decode_paged` in execution order, explaining what each op does, what tensor layout it expects, and what layout it produces.

**Directory:** `chapter_01_decode_path_architecture/`

**Files:**

#### `index.md`
- Overview of what `_forward_decode_paged` is responsible for and why understanding its full op sequence is a prerequisite for every later chapter
- Summary table: op name, input layout, output layout, memory location, and whether it touches the host

#### `op_sequence.md`
- Step-by-step annotation of every operation in `_forward_decode_paged` (lines 2610–2799 of `attention.py`):
  1. `ttnn.to_layout` → TILE_LAYOUT if needed
  2. `q_proj(hidden_states)` via `TTNNLinearIColShardedWRowSharded`: sharded input → reduce-scatter output (col-sharded per device)
  3. `ttnn.all_gather(hidden_states)` → replicated hidden states for K/V input
  4. `k_proj(hidden_states_replicated)` and `v_proj(hidden_states_replicated)` via `TTNNLinearIReplicatedWColSharded`: replicated input → col-sharded output
  5. `_maybe_all_gather` on Q → full Q replicated across devices; `_maybe_all_gather` on K/V → full K/V replicated
  6. Reshape Q/K/V to `[1, batch, heads, head_dim]` (S B H D decode format)
  7. `ttnn.typecast` to bfloat16 (Q, K, V)
  8. `ttnn.to_memory_config(q/k, L1_MEMORY_CONFIG)` — move to L1 before QK norm
  9. `_apply_qk_norm`: reshape to `[batch*heads, head_dim]`, `ttnn.rms_norm` (non-distributed), reshape back
  10. Construct `cache_position_tensor` on host via `torch.tensor`; transfer to device via `ttnn.from_torch` as `cur_pos_tt`
  11. `get_cos_sin_for_decode(cache_position_tensor)` — on-device embedding lookup using pre-cached row-major tables; reshape/transpose to `[1, batch, 1, head_dim]`
  12. Create HEIGHT_SHARDED memory configs for cos/sin and Q/K; move with `ttnn.to_memory_config`
  13. `get_trans_mat_decode_sharded(batch_size)` — lazily created and cached HEIGHT_SHARDED transformation matrix
  14. `ttnn.experimental.rotary_embedding_llama` for Q (is_decode_mode=True)
  15. `ttnn.experimental.rotary_embedding_llama` for K
  16. Re-shard K and V to `HEIGHT_SHARDED` layout for paged KV cache update
  17. `paged_update_on_device(key_states, value_states, layer_idx, cur_pos_tt)` — writes into on-device KV cache
  18. `ttnn.deallocate` K and V
  19. `paged_sdpa_decode(query_states, layer_idx, ...)` — reads from on-device KV cache and computes attention
  20. Move SDPA output to HEIGHT_SHARDED for `nlp_concat_heads_decode`
  21. `ttnn.experimental.nlp_concat_heads_decode` — concatenates heads into `[1, 1, batch, hidden_size]`
  22. `dense(attn_output)` via `TTNNLinearIReplicatedWColSharded`
  23. `ttnn.slice` to trim batch padding (if batch < 32)
  24. `ttnn.reshape` to `[batch, 1, hidden_size]`
- Note which ops are on the critical path for decode latency vs which are setup-only or negligible-volume

#### `tensor_layouts.md`
- Reference table of tensor shapes and layouts at each step:
  - Shape (logical, e.g. `[1, B, num_heads, head_dim]`)
  - Memory location (DRAM or L1)
  - Sharding strategy (replicated, HEIGHT_SHARDED, or col-sharded)
  - Data type (bfloat16, int32, etc.)
- Diagram showing how memory location transitions across the decode path (DRAM → L1 → HEIGHT_SHARDED → DRAM for KV cache)
- Explanation of why the S B H D layout `[1, batch, heads, head_dim]` is required for paged attention kernels

---

### Chapter 2 — Collective Communication Costs and Sharding Strategy

**Description:** Analyzes the all_gather and reduce_scatter ops in the decode path, explains why two separate all_gathers are performed (one for K/V input, one for Q output), and evaluates whether a more efficient sharding scheme is possible on T3K's 1×8 mesh.

**Directory:** `chapter_02_collective_communication/`

**Files:**

#### `index.md`
- Introduction to tensor parallelism on T3K: what the 1×8 mesh topology means, how `cluster_axis=1` maps to the row of 8 devices in the mesh, and what bandwidth is available per link
- Summary of collective ops present in the decode path and their total data volume

#### `all_gather_topology.md`
- Detailed breakdown of each collective in `_forward_decode_paged`:
  1. `ttnn.all_gather(hidden_states, dim=-1)` before K/V projections — why this is needed: `TTNNLinearIReplicatedWColSharded` expects fully-replicated input because its weights are sharded on the output (column) dimension; the input hidden states arriving from the MoE layer are col-sharded, so they must be gathered first
  2. `_maybe_all_gather(query_states)` after Q projection — Q is produced by `TTNNLinearIColShardedWRowSharded` via reduce-scatter, so each device holds `hidden_size / num_devices` of Q; the all-gather reconstitutes the full Q across all devices
  3. `_maybe_all_gather(key_states)` and `_maybe_all_gather(value_states)` — K/V from `TTNNLinearIReplicatedWColSharded` are already col-sharded (output sharded); the all-gather makes them fully replicated so all devices have complete K/V for the paged attention kernel
- Comparison of synchronous `ttnn.all_gather` used in `TTNNBailingMoEAttention` vs the async `ttnn.experimental.all_gather_async` used in `TTNNQwen3FullAttention` and `TTNNDistributedRMSNorm`; latency implications of synchronization barriers
- Ring topology vs Linear topology for all_gather on an 8-device ring: what `ttnn.Topology.Ring` vs `ttnn.Topology.Linear` means for latency and bandwidth utilization

#### `sharding_alternatives.md`
- Analysis of the current Q sharding scheme: `TTNNLinearIColShardedWRowSharded` shards the Q projection weight on the row dimension (output), shards input on the last dimension, and uses reduce-scatter to produce col-sharded output; the subsequent all-gather then reconstitutes full Q
- Why the hidden all-gather cost before K/V is structurally redundant with the Q reduce-scatter: both operations handle the same `hidden_states` tensor but in opposite directions
- Alternative 1: Use `TTNNLinearIReplicatedWColSharded` for Q as well (same as K/V), eliminating the reduce-scatter + all-gather pair for Q; trade-off: Q projection requires replicated hidden states, which means the initial all-gather of hidden states covers all three projections, removing the second all-gather for Q
- Alternative 2: Fused QKV projection: a single matmul on replicated hidden states producing col-sharded `[Q|K|V]`, followed by one all-gather (or no all-gather if downstream kernels can accept col-sharded input); reference: how the original `BailingMoeV2Attention` uses a fused `query_key_value` projection
- Alternative 3: Tensor-parallel decode where Q heads are permanently sharded (num_heads / num_devices heads per device), avoiding any gather of Q; feasibility analysis for `num_heads=16`, `num_kv_heads=4` on 8 devices (2 Q heads and 0.5 KV heads per device — KV heads do not divide evenly, making this non-trivial)
- Expected latency impact of removing one all_gather per decode step

---

### Chapter 3 — Memory Layout Transitions and L1 Pressure

**Description:** Maps every `ttnn.to_memory_config` call in the decode path, explains why each transition is required by downstream kernels, and identifies which transitions are avoidable.

**Directory:** `chapter_03_memory_layout_transitions/`

**Files:**

#### `index.md`
- Background: what it costs to move a tensor from DRAM to L1 and from interleaved to sharded layout on Wormhole; why redundant transitions waste NoC bandwidth
- Inventory of all `ttnn.to_memory_config` calls in the decode path

#### `transition_analysis.md`
- Per-transition analysis:
  1. `ttnn.to_memory_config(query_states, L1_MEMORY_CONFIG)` — required before QK norm because `ttnn.rms_norm` on sharded tensors (post-all-gather) is not supported; cost: DRAM-to-L1 copy for the full Q tensor `[1, B, num_heads, head_dim]`
  2. `ttnn.to_memory_config(key_states, L1_MEMORY_CONFIG)` — same reason as Q
  3. `ttnn.to_memory_config(cos_ttnn, rope_shard_mem)` and `ttnn.to_memory_config(sin_ttnn, rope_shard_mem)` — cos/sin embedding lookup result lives in DRAM; RoPE decode kernel requires HEIGHT_SHARDED L1 input
  4. `ttnn.to_memory_config(query_states, q_shard_mem)` — transition from L1 interleaved (post-QK norm) to HEIGHT_SHARDED L1 required by `rotary_embedding_llama` in decode mode
  5. `ttnn.to_memory_config(key_states, k_shard_mem)` — same as Q for RoPE
  6. `ttnn.to_memory_config(key_states, kv_mem)` and `ttnn.to_memory_config(value_states, kv_mem)` — re-shard to a different HEIGHT_SHARDED config for `paged_update_cache`; the RoPE output shard config uses `shape=(TILE_SIZE, head_dim)` per core while paged_update expects a different shard height; this is a second HEIGHT_SHARDED config for the same K tensor
  7. `ttnn.to_memory_config(attn_output, sdpa_output_memcfg)` — SDPA decode output to HEIGHT_SHARDED for `nlp_concat_heads_decode`
- Identify which transitions can be fused or eliminated:
  - The DRAM→L1 transition before QK norm and the L1→HEIGHT_SHARDED transition for RoPE could be merged if QK norm could operate directly on HEIGHT_SHARDED input, or if QK norm were reordered before the all-gather so the input is small and sharded
  - The two separate HEIGHT_SHARDED configs for K (one for RoPE, one for paged_update) could potentially be unified if the shard spec is made compatible across both kernels

#### `avoidable_transitions.md`
- Deep-dive on the QK-norm-induced DRAM→L1 transition as the most costly unnecessary data movement
- Current implementation path: `all_gather → DRAM → L1_MEMORY_CONFIG → rms_norm → HEIGHT_SHARDED`
- Proposed alternative: apply QK norm before the all_gather, while Q and K are still per-device col-sharded; this requires a distributed RMSNorm (`TTNNDistributedRMSNorm` using `rms_norm_pre_all_gather` + `all_gather_async` + `rms_norm_post_all_gather`) rather than the current non-distributed `TTNNRMSNorm`; the comment in `from_torch` states "head_dim too small to shard across devices" — this chapter evaluates whether that constraint is truly binding for `head_dim=128`
- The two-step K shard reconfiguration after RoPE: why the RoPE shard spec `(TILE_SIZE, head_dim)` differs from the paged_update shard spec `(kv_vol, padded_head_dim)`, and whether the paged_update kernel's shard spec requirement can be relaxed or matched at RoPE output time

---

### Chapter 4 — Host-Device Round-Trips and On-Device Alternatives

**Description:** Identifies all operations that synchronize with or transfer data through the host CPU in the decode path, explains their latency cost, and proposes fully on-device alternatives.

**Directory:** `chapter_04_host_device_roundtrips/`

**Files:**

#### `index.md`
- Why host-device round-trips are latency killers in decode: the host must stall waiting for the device to drain its command queue before a synchronous `ttnn.to_torch` can complete; on a T3K PCIe setup, a single round-trip introduces several hundred microseconds of overhead that blocks the next decode step
- Inventory of all host-device touches in `_forward_decode_paged`

#### `cur_pos_roundtrip.md`
- The `cur_pos_tt` creation pattern (lines 2663–2685):
  - If `cache_position` is a `ttnn.Tensor`, it must be transferred to host via `ttnn.to_torch` and then sent back via `ttnn.from_torch`
  - If `cache_position` is `None`, a new `torch.tensor([cur_pos])` is constructed and then sent to device via `ttnn.from_torch`
  - Either path involves a host-side Python object and a `ttnn.from_torch` call each decode step
- The same pattern appears identically in `TTNNQwen3FullAttention._forward_decode_paged` and `TTNNGlm4MoeLiteAttention._forward_decode_paged`, so this is a codebase-wide issue
- On-device alternative: if `cache_position` arrives as an already-resident `ttnn.Tensor` with `ReplicateTensorToMesh` topology, `ttnn.from_torch` can be skipped entirely; requires callers to maintain a persistent device-resident position tensor that is updated on-device each step (e.g., via `ttnn.add` or `ttnn.assign`)
- Cost estimation: `ttnn.from_torch` for a 1-D int32 tensor of length `batch_size` on T3K including PCIe transfer and topology metadata setup

#### `to_replicated_analysis.md`
- The `_to_replicated` method (lines 2288–2313): performs a full device→host→device round-trip to convert a post-all-gather tensor from AllGatherMesh topology to `ReplicateTensorToMesh` topology; the docstring notes this is "negligible overhead" for tiny decode tensors
- `_to_replicated` is **not called** in `_forward_decode_paged` (it is commented out in earlier revisions) but **is called** in `TTNNGlm4MoeLiteAttention._forward_decode_paged` (lines 1894–1897 of `attention.py`) and in `TTNNQwen3FullAttention._to_replicated` — documenting the discrepancy and why `TTNNBailingMoEAttention` eliminated it
- Why the paged attention kernels require `ReplicateTensorToMesh` topology: the `paged_update_on_device` kernel inspects tensor topology metadata to decide which device slice to write to; if the topology is AllGatherMesh rather than ReplicateTensorToMesh, the mapping is incorrect
- How `TTNNBailingMoEAttention` avoids `_to_replicated` for Q/K/V: by constructing Q/K/V directly with `ttnn.reshape` from the all-gathered tensor (which already has full data on each device), then passing it to `paged_update_on_device`; whether the topology metadata difference actually causes correctness issues in the current implementation should be verified
- On-device topology conversion: `ttnn.experimental.ttnn_to_device_mesh_mapper` or equivalent API that can update topology metadata without a host round-trip

#### `get_cos_sin_host_ops.md`
- `get_cos_sin_for_decode` in `BailingRotarySetup` (lines 420–472 of `rope.py`): if `position_ids` is a `ttnn.Tensor`, it must be extracted to host via `ttnn.to_torch`, then a new `ttnn.from_torch` is called to create `pos_ttnn` for the `ttnn.embedding` lookup; this is a round-trip every decode step
- The subsequent `ttnn.embedding` call is on-device, but the position tensor re-upload is not
- The `cache_position_tensor` created earlier in `_forward_decode_paged` is already a `torch.Tensor` (int32) so it avoids the round-trip from device; however, `get_cos_sin_for_decode` defensively handles both torch and ttnn inputs, adding a device→host copy when the input is already on device
- Proposed refactor: pass `cache_position_tensor` (already a host tensor) directly to `get_cos_sin_for_decode` to skip the conditional device→host copy; alternatively, maintain a persistent device-resident position embedding lookup index and update it in-place each step

---

### Chapter 5 — QK Normalization: Cost Analysis and Distributed Alternatives

**Description:** Measures the latency contribution of QK normalization in the decode path, explains why it currently uses non-distributed `TTNNRMSNorm`, and evaluates whether `TTNNDistributedRMSNorm` would be faster or whether QK norm can be fused with the projection.

**Directory:** `chapter_05_qk_normalization/`

**Files:**

#### `index.md`
- Role of QK normalization in the Bailing MoE model: prevents attention logit explosion in deep MoE architectures; applied after Q/K projection and before RoPE (lines 2659, 2559)
- Why QK norm is performance-relevant: at batch=1 decode with `num_heads=16`, `num_kv_heads=4`, `head_dim=128`, the norm tensors are small but the memory layout transition into L1 that enables them adds overhead

#### `current_implementation.md`
- Walk through `_apply_qk_norm` (lines 2454–2493): reshapes Q from `[1, B, H, D]` to `[B*H, D]`, calls `TTNNRMSNorm.forward` (which calls `ttnn.rms_norm`), reshapes back; same for K
- `TTNNRMSNorm` is **non-distributed**: weights are stored replicated on all devices (via `move_weights_to_device_impl`), and the RMSNorm computation runs independently on each device; this works because after the all-gather, each device holds the full Q and K tensors
- Why "head_dim too small to shard across devices" (comment in `from_torch`, line 2380): for `head_dim=128`, sharding across 8 devices would give 16 elements per device — below the minimum tile width of 32; `TTNNDistributedRMSNorm` requires the reduction dimension to be divisible across devices at tile granularity
- The required DRAM→L1 transition before QK norm (from chapter 3): cost analysis specific to Q `[1, B, 16, 128]` and K `[1, B, 4, 128]` tensors
- Whether the typecast ops before and after QK norm (lines 2481–2484) add latency

#### `distributed_alternative.md`
- `TTNNDistributedRMSNorm` implementation (lines 99–151 of `normalization.py`): uses `ttnn.rms_norm_pre_all_gather` → `all_gather_async` → `ttnn.rms_norm_post_all_gather`; this is designed for normalizing tensors where the reduction dimension is split across devices (e.g., the full hidden state `[B, S, hidden_size/num_devices]`)
- Why this does **not** apply to QK norm: after the all-gather in the decode path, Q and K are already fully replicated on each device (each device has the complete `[1, B, heads, head_dim]` tensor); distributing the norm would require re-splitting what was just gathered, which is counterproductive
- Alternative: apply QK norm **before** the all-gather, while Q is still col-sharded (`[B, seq, num_heads * head_dim / num_devices]`); at this point the reduction is over `head_dim` within a single device (no cross-device reduction needed), so `TTNNRMSNorm` would work correctly without distributed norm; the shapes would be `[B * (num_heads / num_devices), head_dim]` per device — feasible for RMSNorm
- Trade-off: moving QK norm before the all-gather changes the kernel's input topology from AllGather to ReduceScatter output; need to verify that the L1 transition can be avoided or that the shard layout is already compatible with `ttnn.rms_norm`
- Fusion opportunity: in some implementations (e.g., `TTNNGlm4MoeLiteAttention`), the normalization after a projection is fused inline using the device-resident weight; `TTNNBailingMoEAttention` wraps it in a separate `TTNNRMSNorm` module which adds overhead from module dispatch; direct `ttnn.rms_norm` call with pre-loaded weights would reduce Python overhead

---

### Chapter 6 — Math Fidelity, Compute Kernel Settings, and SDPA Configuration

**Description:** Explains the HiFi4 / HiFi2 / LoFi math fidelity modes on Wormhole, documents the current SDPA compute kernel config, and evaluates whether HiFi2 is sufficient for correctness in `TTNNBailingMoEAttention`.

**Directory:** `chapter_06_math_fidelity/`

**Files:**

#### `index.md`
- Overview of compute kernel configurations in TTNN: `math_fidelity`, `math_approx_mode`, `fp32_dest_acc_en`, `packer_l1_acc`; how they interact to control the accuracy vs throughput trade-off
- Current config in `TTNNBailingMoEAttention.move_weights_to_device_impl` (lines 2434–2440): `HiFi4`, `fp32_dest_acc_en=True`, `packer_l1_acc=True`

#### `hifi4_vs_hifi2.md`
- What HiFi4 means on Wormhole: uses 4 mantissa bits for BFP8 intermediate products and 4 exponent bits for the accumulator; provides near-FP32 accuracy at roughly 2× the throughput of FP32
- What HiFi2 means: uses 2 mantissa bits; approximately 2× faster than HiFi4 but with higher rounding error
- What LoFi means: 1 mantissa bit; fastest but largest accuracy degradation
- How `fp32_dest_acc_en=True` interacts with fidelity: forces the destination register to FP32 precision regardless of math fidelity; combined with HiFi4 this is the highest-accuracy configuration; combined with HiFi2 it preserves accumulator precision while reducing product precision
- How `packer_l1_acc=True` interacts: enables the packer to accumulate into L1 before writing to DRAM, reducing DRAM write traffic for accumulation-heavy ops like SDPA's QK^T matmul
- Comparison to `TTNNQwen3FullAttention`'s config: `HiFi4`, `fp32_dest_acc_en=False`, `packer_l1_acc=False`; the rationale in the comment (lines 338–347 of `qwen_attention.py`) is that `fp32_dest_acc_en=False` increases `dst_size` from 4 to 8 tiles, enabling larger chunks in the inner loop; this is a throughput vs accuracy trade that `TTNNBailingMoEAttention` has not adopted

#### `accuracy_throughput_tradeoff.md`
- Methodology for evaluating HiFi2 correctness: compare attention output logits between HiFi4 and HiFi2 configurations on representative Bailing MoE inputs; measure max absolute error and relative error on the SDPA output `attn_output`
- Expected throughput gain from switching to HiFi2: approximately 1.5–2× for the SDPA kernel alone; total decode step speedup will be smaller because SDPA is one of several contributors
- Expected throughput gain from switching `fp32_dest_acc_en` to `False`: enables larger tile batching in the inner loop; potentially 10–20% gain for the SDPA matmul
- Combined config to benchmark: `HiFi2`, `fp32_dest_acc_en=False`, `packer_l1_acc=False` — matches Qwen3 style; measures both compute throughput and accuracy degradation
- `q_chunk_size=256`, `k_chunk_size=256` in the decode config are currently set to 0 (meaning kernel picks defaults); the prefill config uses 256×256; investigate what chunk sizes are optimal for decode (typically smaller because Q is 1 token)
- `exp_approx_mode=False` in both configs: controls whether the exponential in softmax uses a polynomial approximation; setting to `True` adds throughput at the cost of softmax precision; analysis of whether approximate exp is safe given that attention is followed by a dense projection

---

### Chapter 7 — Profiling Methodology and Optimization Roadmap

**Description:** Describes how to collect and interpret T3K performance profiles for the decode path, provides a ranked list of bottlenecks based on the analysis in previous chapters, and gives expected speedup estimates for each optimization.

**Directory:** `chapter_07_profiling_and_roadmap/`

**Files:**

#### `index.md`
- Purpose of this chapter: translate analysis from chapters 1–6 into actionable profiling steps and a prioritized optimization backlog
- Prerequisites: a working Bailing MoE decode loop running on T3K at batch=1

#### `profiling_methodology.md`
- How to use `ttnn.tracy` or `ttnn.perf_device_event_handler` to measure op-level latency for a single decode step
- How to isolate the attention forward pass from the rest of the model: wrap `_forward_decode_paged` in a manual timer using `ttnn.synchronize_device` before and after; compare this total to the sum of individual op timings to detect scheduling gaps
- Key metrics to collect per op: kernel wall time, number of data movement bytes, NoC utilization, L1 occupancy
- Specific ops to instrument:
  - `ttnn.all_gather` calls (two hidden_states gathers + three post-projection gathers = 5 total collective ops)
  - Each `ttnn.to_memory_config` call (7 calls identified in chapter 3)
  - `ttnn.rms_norm` for Q and K
  - `ttnn.experimental.rotary_embedding_llama` for Q and K
  - `paged_update_on_device`
  - `paged_sdpa_decode`
  - `ttnn.experimental.nlp_concat_heads_decode`
  - `ttnn.linear` inside `dense`
  - `ttnn.from_torch` for `cur_pos_tt` (host-device transfer)
- How to interpret tracy traces: identifying back-to-back dispatches with no compute overlap (serialized memory moves), stalls waiting for collective completion, and outlier kernel durations

#### `bottleneck_ranking.md`
- Ranked list of bottlenecks by estimated latency contribution at batch=1 decode on T3K, based on op sequence analysis (to be validated with profiling):

  1. **Redundant all_gather for Q** (chapter 2): Q is gathered after reduce-scatter; switching Q to `TTNNLinearIReplicatedWColSharded` or using a fused QKV projection would eliminate one full all_gather of `[B, seq, num_heads * head_dim]`; estimated latency saving: 1 all_gather × ~hidden_size / 8 bytes per link × num_links
  2. **Two-step K memory re-sharding** (chapter 3): K moves from DRAM→L1 (for QK norm), then L1→HEIGHT_SHARDED (for RoPE), then HEIGHT_SHARDED→different-HEIGHT_SHARDED (for paged_update); reducing to one transition saves 2 NoC round-trips
  3. **`cur_pos_tt` host round-trip** (chapter 4): `ttnn.from_torch` for a tiny int32 tensor each decode step; host stall and PCIe overhead; eliminate by maintaining a persistent device tensor
  4. **QK norm DRAM→L1 transition** (chapter 3, chapter 5): moving Q and K to L1 before norm when they could remain in HEIGHT_SHARDED if norm were reordered or fused; small data volume but adds pipeline stall
  5. **HiFi4 vs HiFi2 for SDPA** (chapter 6): if accuracy permits, switching to HiFi2 with `fp32_dest_acc_en=False` can improve SDPA throughput by 1.5–2×; SDPA is compute-bound in decode for long contexts
  6. **`get_cos_sin_for_decode` partial host op** (chapter 4): position tensor re-upload in `BailingRotarySetup`; minor but eliminable

- For each bottleneck: description, root cause file and line number, proposed fix, expected speedup magnitude (rough), risk level (correctness or API compatibility)

#### `comparison_to_other_implementations.md`
- Side-by-side comparison of `TTNNBailingMoEAttention._forward_decode_paged` vs `TTNNQwen3FullAttention._forward_decode_paged` and `TTNNGlm4MoeLiteAttention._forward_decode_paged` on the key dimensions:

  | Dimension | TTNNBailingMoEAttention | TTNNQwen3FullAttention | TTNNGlm4MoeLiteAttention |
  |---|---|---|---|
  | Q projection type | TTNNLinearIColShardedWRowSharded (reduce-scatter + 2nd all_gather) | TTNNLinearIReplicatedWColSharded (1 all_gather covers all) | TTNNLinearIColShardedWRowSharded |
  | All-gathers per decode step | 5 (1 input + 3 post-proj + 1 implicit in Q proj) | 3 (1 input + 3 post-proj; no separate Q gather because Q proj is replicated) | 4 |
  | all_gather API | `ttnn.all_gather` (synchronous) | `ttnn.experimental.all_gather_async` + synchronize | `ttnn.all_gather` (synchronous) |
  | QK norm | TTNNRMSNorm (non-distributed, after all_gather) | ttnn.rms_norm direct (non-distributed, after all_gather) | TTNNRMSNorm / TTNNDistributedRMSNorm |
  | `_to_replicated` host round-trip | Not called (eliminated) | Called for all Q/K/V | Called for all Q/K/V |
  | cur_pos host round-trip | `ttnn.from_torch` each step | `ttnn.from_torch` each step | `ttnn.from_torch` each step |
  | SDPA fidelity | HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True | HiFi4, fp32_dest_acc_en=False, packer_l1_acc=False | HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True |
  | Async collective for RMSNorm | No | No | Yes (TTNNDistributedRMSNorm uses all_gather_async) |

- Key techniques in `TTNNQwen3FullAttention` that could be adopted:
  1. Using `TTNNLinearIReplicatedWColSharded` for all projections (Q, K, V) — eliminates the reduce-scatter in Q proj and the second all-gather for Q; applies when `num_heads >= num_devices` is not a strict requirement (both models have 16 Q heads and 8 devices, so 2 Q heads per device is feasible with replicated projection)
  2. Using `all_gather_async` instead of synchronous `all_gather` to overlap communication with subsequent compute; Qwen3 already has the `ccl_manager` semaphore infrastructure wired up
  3. `fp32_dest_acc_en=False` for SDPA to increase dst_size and enable larger inner-loop batching

- Key advantage of `TTNNBailingMoEAttention` vs `TTNNQwen3FullAttention`: the elimination of `_to_replicated` host round-trips; this is the most impactful single optimization already made; Qwen3 and GLM4 should adopt this pattern

---

## Conventions

**Tensor shape notation:** dimensions are listed in the order they appear in TTNN (outermost first). Four-dimensional tensors use the format `[dim0, dim1, dim2, dim3]`. The symbolic names used throughout this guide are:
- `B` = batch size
- `S` = sequence length (=1 for decode)
- `H` = number of query attention heads (= `num_heads` = 16 for Bailing MoE)
- `Hkv` = number of KV heads (= `num_kv_heads` = 4 for Bailing MoE)
- `D` = head dimension (= `head_dim` = 128 for Bailing MoE)
- `d_model` = hidden size (= `hidden_size` = `H * D` = 2048 for Bailing MoE)
- `N` = number of devices (= 8 for T3K 1×8 mesh)

**Decode layout name:** `S B H D` refers to the layout `[1, batch, heads, head_dim]` required by paged attention kernels; this is the layout in which tensors exist after the `ttnn.reshape` on line 2644–2646 of `attention.py`.

**Source file references:** all line number references are to the state of the codebase at the time of writing. Primary source files:
- `attention.py` = `/localdev/salnahari/testing_dir/tt-metal/models/experimental/tt_symbiote/modules/attention.py`
- `linear.py` = `/localdev/salnahari/testing_dir/tt-metal/models/experimental/tt_symbiote/modules/linear.py`
- `normalization.py` = `/localdev/salnahari/testing_dir/tt-metal/models/experimental/tt_symbiote/modules/normalization.py`
- `rope.py` = `/localdev/salnahari/testing_dir/tt-metal/models/experimental/tt_symbiote/modules/rope.py`
- `qwen_attention.py` = `/localdev/salnahari/testing_dir/tt-metal/models/experimental/tt_symbiote/modules/qwen_attention.py`

**Op naming:** TTNN ops are referenced by their Python API name exactly as they appear in source (e.g., `ttnn.all_gather`, `ttnn.experimental.rotary_embedding_llama`). When a method of a module class is referenced, it is named as `ClassName.method_name` (e.g., `BailingRotarySetup.get_cos_sin_for_decode`).

**Topology terminology:** "col-sharded" means each device holds `output_dim / N` columns of the output tensor (sharded on the last dimension); "replicated" means each device holds a full copy of the tensor. `TTNNLinearIColShardedWRowSharded` = input col-sharded, weight row-sharded (produces col-sharded output via reduce-scatter). `TTNNLinearIReplicatedWColSharded` = input replicated, weight col-sharded (produces col-sharded output via local matmul with no inter-device communication).

**Quantitative claims:** throughput and latency estimates that have not been measured by profiling are marked as "rough estimate" or "to be validated"; they are order-of-magnitude guidance only. Profiling methodology to produce actual numbers is given in chapter 7.

---

## Cross-Chapter Dependencies

- **Chapter 2** (collective communication) depends on the op sequence established in **Chapter 1**; it references the specific positions in the decode path where all_gather ops appear.
- **Chapter 3** (memory layout transitions) depends on **Chapter 1** for the full list of `ttnn.to_memory_config` calls and their positions in the op sequence.
- **Chapter 4** (host-device round-trips) depends on **Chapter 1** for the identification of `ttnn.from_torch` call sites. The `cur_pos_tt` analysis in chapter 4 references the position in the op sequence between QK norm and cos/sin lookup.
- **Chapter 5** (QK normalization) depends on **Chapter 3** for the analysis of the DRAM→L1 transition that is caused by QK norm's layout requirement; the proposed fix in chapter 5 (moving QK norm before the all-gather) directly resolves the layout transition identified in chapter 3.
- **Chapter 6** (math fidelity) is largely self-contained but references the SDPA op introduced in **Chapter 1** and uses the comparison table from **Chapter 7** to contrast configs across implementations.
- **Chapter 7** (profiling and roadmap) depends on all prior chapters: the bottleneck ranking in `bottleneck_ranking.md` references the root causes identified in chapters 2–6 by file and line number; the comparison table in `comparison_to_other_implementations.md` synthesizes the per-chapter analysis.
- The distinction between synchronous `ttnn.all_gather` and async `ttnn.experimental.all_gather_async` is introduced in **Chapter 2** and referenced again in **Chapter 7**'s comparison table.
- The `TTNNDistributedRMSNorm` implementation is introduced in **Chapter 5** and referenced in **Chapter 7**'s comparison table entry for `TTNNGlm4MoeLiteAttention`.
