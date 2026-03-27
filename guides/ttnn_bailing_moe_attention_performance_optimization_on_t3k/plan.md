# Plan: TTNNBailingMoEAttention Performance Optimization on T3K

## Audience

This guide targets ML systems engineers and kernel developers who are already familiar with:
- The Tenstorrent T3K hardware topology (1×8 mesh of Wormhole chips, mesh interconnect via Ethernet links)
- Core TTNN programming model: tensor memory configs, sharding strategies (WIDTH, HEIGHT, BLOCK), and multi-device tensor types
- Basic transformer attention mechanics (multi-head attention, GQA, RoPE, paged KV-cache)
- Python-level model code in the `tt-transformers` / `tt-metal` stacks

Readers do not need prior exposure to the Ling (BailingMoeV2) model specifically, but should be comfortable reading TTNN op invocations and interpreting profiler traces.

The guide is actionable: every chapter ends with concrete measurement steps or code changes a reader can apply to a T3K system.

---

## Chapter List

### Chapter 1 — `ch1_model_and_hardware_context`
**Description:** Establishes the Ling model's attention layer structure and the T3K hardware topology that constrains every optimization decision.

Files:
- `index.md`
  - Overview of the chapter's scope and reading order
  - Summary table mapping each research question to the chapter that answers it
- `ling_model_overview.md`
  - BailingMoeV2 / Ling architecture: MoE layout, attention config (16 Q heads, 4 KV heads, head_dim=128, hidden_size), and how `TTNNBailingMoEAttention` fits into the decoder stack
  - Key hyperparameters referenced throughout the guide: `num_heads`, `num_kv_heads`, `head_dim`, `partial_rotary_factor`, `use_qk_norm`
  - Distinction between prefill and decode execution paths and why their performance profiles differ
- `t3k_topology_primer.md`
  - T3K physical layout: 8 Wormhole chips on a 1×8 logical mesh, Ethernet CCL bandwidth and latency characteristics
  - Relevant TTNN sharding primitives used by the attention layer: `TensorMemoryLayout` variants (HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED, INTERLEAVED), `ShardSpec`, `MemoryConfig`
  - How `num_links` in CCL operations maps to physical Ethernet connections and why it matters for all-reduce/all-gather latency

---

### Chapter 2 — `ch2_fused_qkv_projection`
**Description:** Analyzes the fused QKV projection (`TTNNLinearIColShardedWAllReduced`) that replaces three separate matmuls and five CCL operations, and evaluates whether `num_links=1` is optimal.

Files:
- `index.md`
  - Chapter scope and prerequisites (Chapter 1 concepts: mesh topology, sharding)
- `fusion_mechanics.md`
  - How `TTNNLinearIColShardedWAllReduced` fuses the Q, K, V weight columns into a single matmul with column-sharding across the 8 chips
  - The all-reduce step that follows: which CCL primitive is used, what tensor shape is communicated, and how `num_links` controls Ethernet utilization
  - Theoretical bandwidth model: compute time of the fused matmul vs. all-reduce transfer time as a function of hidden size
- `latency_savings_analysis.md`
  - Baseline: estimated latency of 3 separate matmuls + 5 CCL ops (all-gather + scatter pattern) on T3K for Ling's hidden size
  - Fused path: measured or estimated latency of 1 matmul + 1 all-reduce
  - Expected savings and where the remaining bottleneck lies (compute-bound vs. communication-bound)
  - How to measure this split using TTNN op timers (preview of Chapter 7 tooling)
- `num_links_tuning.md`
  - What `num_links=1` means physically: one Ethernet path used for CCL traffic
  - Sensitivity analysis: expected latency change when `num_links` is increased to 2 or 4 for the all-reduce at Ling's hidden dimension
  - Recommendation: the value of `num_links` to use and the conditions under which it should be revisited (batch size, hidden size changes)

*Answers question 1.*

---

### Chapter 3 — `ch3_host_roundtrip_replication`
**Description:** Explains the `_to_replicated` host round-trip that converts an all-gathered QKV tensor from device to CPU and back, quantifies its latency at decode batch=1, and surveys device-side alternatives.

Files:
- `index.md`
  - Chapter scope and prerequisites (Chapter 2: output tensor layout after fused QKV)
- `roundtrip_mechanics.md`
  - Step-by-step trace of `_to_replicated`: `ConcatMeshToTensor` pulling data from all 8 chips to host, `torch` tensor construction, `from_torch` with `ReplicateTensorToMesh` pushing back to all chips
  - Why this is required: the paged-attention kernel's topology constraint that forces a replicated (not sharded) QKV layout on each chip
  - Data volume transferred: tensor shape at decode batch=1, dtype, and resulting bytes moved over PCIe per direction
- `host_transfer_overhead.md`
  - PCIe throughput model for T3K host transfers (bidirectional, per chip vs. aggregate)
  - Estimated round-trip latency at decode batch=1 given tensor dimensions
  - How to measure actual latency using Tracy or TTNN op timers (cross-reference Chapter 7)
  - Sensitivity to batch size: how overhead scales and at what batch size it becomes negligible relative to compute
- `device_side_alternatives.md`
  - Survey of TTNN primitives that could perform tensor replication without host involvement: `ttnn.all_gather` with `ReplicateTensorToMesh` output config, device-side reshape/concat, or a custom CCL
  - Feasibility analysis for each alternative given the paged-attention kernel's input constraints
  - Recommended path forward: whether a device-side approach is immediately actionable or requires kernel changes

*Answers question 2.*

---

### Chapter 4 — `ch4_memory_config_transitions`
**Description:** Catalogs every memory-configuration transition in the decode path (HEIGHT_SHARDED RoPE, K/V re-sharding before paged update), counts their total per-step cost, and identifies the dominant overhead.

Files:
- `index.md`
  - Chapter scope and prerequisites (Chapter 1: sharding primitives; Chapter 3: post-replication tensor layout)
- `decode_tensor_lifecycle.md`
  - Annotated diagram of tensor memory configs at each stage of a single decode step: QKV after replication → HEIGHT_SHARDED split for RoPE → post-RoPE Q, K, V layouts → re-shard before `paged_update_on_device` → SDPA inputs
  - Definition of `rope_shard_mem` (`(TILE_SIZE, head_dim)` shard shape) and why that specific shape is chosen
  - Exact sequence of `ttnn.to_memory_config` or equivalent calls, with source and destination `MemoryConfig` for each
- `transition_cost_model.md`
  - How TTNN executes a memory-config transition: data movement path (L1 → DRAM → L1, or L1 → L1), expected cycles, and how tensor size determines cost
  - Cost estimate for each identified transition in the decode step
  - Which transition dominates and why (data volume, distance, or kernel launch overhead)
- `optimization_opportunities.md`
  - Transitions that may be eliminable by adjusting upstream output memory configs (e.g., producing output already in HEIGHT_SHARDED format)
  - Transitions that are kernel-mandated and cannot be removed without changing the paged-attention or RoPE kernel
  - Concrete code locations in `TTNNBailingMoEAttention` where changes would be made

*Answers question 3.*

---

### Chapter 5 — `ch5_sdpa_and_compute_config`
**Description:** Covers the paged SDPA decode kernel configuration (`q_chunk_size=0`, `k_chunk_size=0`, GQA layout) and the math-fidelity trade-off (`HiFi4` vs. `HiFi2`) for the Ling model's attention correctness requirements.

Files:
- `index.md`
  - Chapter scope and prerequisites (Chapter 1: GQA head counts; Chapter 4: tensor layout entering SDPA)
- `paged_sdpa_chunk_sizes.md`
  - What `q_chunk_size` and `k_chunk_size` control in the paged SDPA kernel: tiling strategy over sequence length in the Q and KV dimensions
  - Semantics of chunk size 0: how the kernel interprets it (auto-selection, full sequence, or a specific default), backed by kernel source or documentation
  - Correctness and performance implications for Ling's GQA configuration: 16 Q heads grouped over 4 KV heads, `head_dim=128`, and typical decode sequence lengths
  - Recommended values: whether 0 is correct, or whether explicit chunk sizes would improve throughput or reduce register pressure
- `math_fidelity_tradeoff.md`
  - Explanation of TTNN math fidelity levels (`LoFi`, `HiFi2`, `HiFi4`) and how they map to hardware compute modes on Wormhole
  - Role of `fp32_dest_acc_en=True` and `packer_l1_acc=True` in accumulation precision for softmax and QK dot products
  - Accuracy requirements for attention in a BFloat16 GQA model: theoretical analysis of precision loss at `HiFi2` vs. `HiFi4`
  - Performance delta: expected throughput improvement when dropping from `HiFi4` to `HiFi2`, based on Wormhole FPU characterization
  - Recommendation: whether `HiFi2` is safe to use and how to validate with a numeric accuracy test

*Answers questions 4 and 5.*

---

### Chapter 6 — `ch6_rope_and_qk_norm`
**Description:** Examines the QK normalization detour and the non-distributed RoPE forced by `partial_rotary_factor < 1.0`, quantifying their decode-step latency contributions and identifying avoidance strategies.

Files:
- `index.md`
  - Chapter scope and prerequisites (Chapter 4: memory-config transition framework; Chapter 5: SDPA input layout)
- `qk_norm_latency.md`
  - Code path when `use_qk_norm=True`: L1 move, 3D→2D reshape, `TTNNRMSNorm` call, 2D→3D reshape, repeated for Q and K each decode step
  - Latency breakdown: reshape kernel cost, RMSNorm kernel cost, and L1 move cost vs. fused QKV matmul latency (cross-reference Chapter 2)
  - Whether the L1 move is required by `TTNNRMSNorm`'s input constraints or is a precaution
  - Options for reducing overhead: fusing reshape into the norm kernel, using an in-place norm that avoids the L1 move, or accepting cost as negligible relative to SDPA
- `partial_rotary_rope.md`
  - Why `partial_rotary_factor < 1.0` disables `TTNNDistributedRotaryPositionEmbedding` and forces `TTNNRotaryPositionEmbedding` (non-distributed)
  - Performance cost of non-distributed RoPE on T3K: the kernel runs on a single chip or is replicated without exploiting mesh parallelism, compared to the distributed variant
  - Analysis of whether cos/sin tables can be padded to full `head_dim` to enable the distributed kernel while keeping the partial-rotary mask applied post-embedding
  - Alternative: a device-side slice op after distributed RoPE to discard padded dimensions, and whether this is cheaper than non-distributed execution
  - Recommendation with implementation sketch

*Answers questions 6 and 7.*

---

### Chapter 7 — `ch7_profiling_and_bottleneck_identification`
**Description:** Provides a practical, step-by-step guide to profiling the full `TTNNBailingMoEAttention` forward at op-level granularity on T3K using Tracy and TTNN op timers, enabling identification of the single largest decode bottleneck.

Files:
- `index.md`
  - Chapter scope and prerequisites (all prior chapters: each op discussed in context)
  - Summary of profiling tools available and when to use each
- `ttnn_op_timers.md`
  - How to enable TTNN op-level timing: environment variables, `ttnn.tracer` context manager, or compile-time flags
  - Reading a TTNN op timer report: output format, how to map op names to source-code call sites in `TTNNBailingMoEAttention`
  - Example invocation for a decode batch=1 forward pass and expected output structure
  - Interpreting results: identifying the ops that account for >80% of decode latency
- `tracy_profiling.md`
  - Tracy setup for T3K: build flags, server startup, and connecting to a multi-device host
  - Annotating `TTNNBailingMoEAttention` forward with Tracy zones to get per-op spans in the timeline view
  - Capturing a single decode step trace: recommended capture window and how to isolate the attention layer from the rest of the decoder
  - Reading the Tracy timeline: correlating device-side op durations with host-side CCL and data-movement zones
  - Common pitfalls: warm-up steps, JIT compilation artifacts, and PCIe transfer inflation in first-run captures
- `bottleneck_decision_tree.md`
  - Decision tree: given a profiling result, which chapter's optimization to apply first
  - Example scenarios: if host round-trip (Chapter 3) dominates vs. if CCL all-reduce (Chapter 2) dominates vs. if SDPA kernel (Chapter 5) dominates
  - Recommended iteration loop: profile → change one variable → re-profile → compare

*Answers question 8.*

---

## Conventions

**Terminology:**
- "decode step" always refers to a single autoregressive forward pass with batch=1 and sequence length=1 (the token-generation phase), unless explicitly noted otherwise.
- "prefill" refers to the prompt-processing forward pass where the full input sequence is processed in parallel.
- "T3K" refers specifically to the Tenstorrent T3K system with 8 Wormhole n300 chips arranged in a 1×8 logical mesh.
- "chip" and "device" are used interchangeably to mean one Wormhole ASIC in the mesh.
- "host" means the x86 CPU and its DRAM, connected to the T3K via PCIe.
- Op names are written in `code font` exactly as they appear in TTNN Python source (e.g., `TTNNLinearIColShardedWAllReduced`, `paged_sdpa_decode`).
- Sharding strategy names match the `TensorMemoryLayout` enum values: `HEIGHT_SHARDED`, `WIDTH_SHARDED`, `BLOCK_SHARDED`, `INTERLEAVED`.

**Notation:**
- Tensor shapes are written as `(dim0, dim1, ...)` with dimension names spelled out on first use (e.g., `(batch, seq_len, num_heads, head_dim)`).
- Latency estimates are given in microseconds (µs) unless the value exceeds 1 ms, in which case milliseconds (ms) are used.
- Bandwidth is in GB/s (gigabytes per second, base-10).
- When referencing TTNN source files, use paths relative to the `tt-metal` repository root (e.g., `ttnn/cpp/ttnn/operations/...`).
- GQA configuration is always stated as `Q_heads / KV_heads` (e.g., `16/4 GQA`).

**Formatting:**
- Each file begins with a single `#` H1 heading matching the file's topic.
- Code snippets use fenced code blocks with explicit language tags (`python`, `cpp`, `bash`).
- Every table has a caption immediately above it (plain text, no heading marker).
- Cross-references to other chapters use the format: `(see Chapter N, filename.md)`.
- Measurements that are estimates or theoretical (not yet empirically validated on T3K) are marked with the tag `[ESTIMATE]` inline.
- Measurements that have been empirically verified on T3K are marked with `[MEASURED]` inline.

---

## Cross-Chapter Dependencies

- **Chapter 2** depends on **Chapter 1** for the T3K mesh topology description and the definition of column-sharded weight layout.
- **Chapter 3** depends on **Chapter 2** for the output tensor shape and memory config produced by the fused QKV projection (the input to `_to_replicated`).
- **Chapter 4** depends on **Chapter 1** for sharding primitive definitions and depends on **Chapter 3** for the post-replication tensor layout that begins the decode memory-config chain.
- **Chapter 5** depends on **Chapter 1** for GQA head configuration and depends on **Chapter 4** for the tensor layout entering `paged_sdpa_decode`.
- **Chapter 6** depends on **Chapter 4** for the memory-config transition cost model (used to assess the L1 move in QK norm) and depends on **Chapter 5** for the SDPA input layout (context for partial RoPE output format).
- **Chapter 7** depends on all prior chapters: its bottleneck decision tree maps profiling outcomes to the optimization strategies developed in Chapters 2–6. It also provides the measurement methodology referenced by `[MEASURED]` tags in Chapters 2, 3, 4, and 5.
