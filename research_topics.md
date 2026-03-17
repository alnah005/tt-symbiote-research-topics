# Research Topics

This file tracks research topics that the Architect needs to investigate for making informed decisions.

---

## Format

```
## [Topic Name]
**Date:** YYYY-MM-DD
**Status:** Pending | In Progress | Completed
**Why Needed:** [Reason this research is necessary]
**Questions:**
- Question 1
- Question 2

**Findings:**
[Results of research go here]

---
```

## Topics

---

## MoE Optimization Techniques for TTNN
**Date:** 2026-03-16
**Status:** Completed
**Why Needed:** Need to understand best practices for optimizing Mixture of Experts models on Tenstorrent hardware, specifically comparing batched matmul vs sparse_matmul approaches.
**Questions:**
- What are the performance characteristics of sparse_matmul vs batched matmul for MoE?
- How should sparsity tensors be constructed for optimal performance?
- What program configs are recommended for different batch/sequence sizes?

**Findings:**
Full guide written: `guides/moe_optimization_techniques_for_ttnn/` (ch01–ch08).

- **sparse_matmul vs batched matmul:** Use `sparse_matmul` when token utilization < ~50% (C/32 < 0.5, i.e., C < 16). At decode with B≤32 and C=2, token utilization is 6.25% — sparse_matmul skips 93.75% of zero tile rows. Switch to dense batched matmul at prefill when C grows past 16.
- **Sparsity tensor construction:** Build a `[E, C]` sparsity tensor from router top-k indices after capacity-aware assignment. Place in L1 for decode (small), DRAM for prefill (large). Tile-align to 32-element boundaries; pad with sentinel value (e.g., -1) for empty capacity slots.
- **Program configs:** `per_core_M = ceil(C/32)` — equals 1 for all decode batch sizes (C < 32). Use a (2, 32) core grid (2 cores/expert × 32 local experts = 64 active cores) on T3K. `out_subblock_h = 1` at decode; `in0_block_w = K_t` only if L1 budget permits (K_t=224 at H=7168 is large — may need blocking).

---

## T3K Mesh Device Optimizations
**Date:** 2026-03-16
**Status:** Completed
**Why Needed:** TTNNQwen3MoE runs on T3K (1x8 mesh) and needs device-specific optimizations for expert parallelism.
**Questions:**
- What are the optimal num_links settings for all_to_all operations on T3K?
- How should memory configs (L1 vs DRAM) be chosen for decode vs prefill?
- What are the bandwidth characteristics between T3K devices?

**Findings:**
Full guide written: `guides/t3k_mesh_device_optimizations/` (ch01–ch07).

- **num_links:** Use `num_links=1` for decode (dispatch volume ≤ 6.4 MB at B=32). Link setup overhead outweighs throughput gain below ~20 MB payload. Use `num_links=2` for prefill (dispatch volume ~245 MB+ at B=1,S=2048). The threshold is empirically ~20 MB; confirm with `ch03_all_to_all_num_links/num_links_parameter.md`.
- **Memory config (decode):** All activations and A2A buffers in `L1_MEMORY_CONFIG`. Expert weights always in `DRAM_MEMORY_CONFIG` (too large for L1). KV cache always DRAM. Set memory configs once before the decode loop — do not reset per step (avoids program cache invalidation).
- **Memory config (prefill):** All tensors (activations, A2A buffers, expert weights) in `DRAM_MEMORY_CONFIG`; prefill A2A buffers are too large for L1 (~29–940 MB depending on B and S).
- **T3K bandwidth:** ~12.5 GB/s per Ethernet link; 1×8 linear mesh with average hop count 3.0. Interior devices (IDs 1–6) have 2 active ports; endpoint devices (0, 7) have 1. Wormhole B0: 80 Tensix cores, ~1.5 MB L1/core, ~120 MB aggregate L1, ~300 GB/s DRAM bandwidth [UNVERIFIED].

---

## Expert Parallelism Strategies
**Date:** 2026-03-16
**Status:** Completed
**Why Needed:** Qwen3.5-35B has 256 experts with top-8 routing. Need optimal dispatch/combine strategies.
**Questions:**
- How does all_to_all_dispatch/combine compare to alternative expert routing schemes?
- What is the optimal expert-to-device assignment for 256 experts on 8 devices?
- How should routing weights be processed to minimize overhead?

**Findings:**
Full guide written: `guides/expert_parallelism_strategies/` (ch01–ch08).

- **Dispatch/combine scheme:** `ttnn.all_to_all` (dispatch + combine) is the preferred scheme for Qwen3.5-35B on T3K at decode batch sizes B=1–32. All-gather-based expert sharding is only better below ~4 tokens/step where dispatch volume drops below link-setup overhead. At B=32, dispatch volume ≈ 6.4 MB, latency ≈ 0.51 ms per direction — the workload is communication-bound.
- **Expert assignment:** Uniform 32-experts-per-device (256/8) as default. Apply load-aware bin-packing rebalancing only when any expert's activation frequency exceeds 4× average (f_e > 12.5%). Replicate hot experts onto additional devices when f_e > 1/N = 12.5% and DRAM budget allows; replication factor r_e = max(1, ceil(f_e × N)).
- **Routing weight processing:** Fuse router projection with top-k selection. Defer weight renormalization to the combine kernel (avoid a separate pass). Use BF16 for W_r (3.67 MB); INT8 (1.84 MB) only under DRAM pressure. Capacity factor CF=1.25 → C=2 at B=32; Poisson overflow ≈ 8% per expert, total drops ≈ 10.6% of expert slots. Use hard-drop with renormalization of surviving top-k weights.

---

## Weight Quantization for MoE Experts
**Date:** 2026-03-17
**Status:** Completed
**Why Needed:** DeepSeek-V3 uses bfloat4_b/bfloat8_b weight quantization for experts, but Qwen uses full bfloat16. Need to evaluate quantization trade-offs.
**Questions:**
- What accuracy loss is expected from bfloat4_b vs bfloat8_b vs bfloat16 for expert weights?
- How does weight quantization affect compute throughput on Wormhole?
- Which projections (gate/up/down) are most sensitive to quantization?

**Findings:**
Mixed layout (bfloat4_b gate/up + bfloat8_b down) reduces per-expert memory from 84.0 MB to 28.0 MB (3x reduction). T3K (8 chips): 16 experts/chip; BF16 1,344 MB/chip → mixed 448 MB/chip. PCC thresholds: gate/up ≥ 0.96, down ≥ 0.975, full layer ≥ 0.97, production baseline > 0.999. Both LOFI and HIFI2 use fp32_dest_acc_en=False.

---

## Compute Kernel Configuration for MoE
**Date:** 2026-03-17
**Status:** Completed
**Why Needed:** DeepSeek-V3 uses COMPUTE_KERNEL_CONFIG_LOFI with packer_l1_acc, but Qwen MoE doesn't specify compute kernel configs. Need to optimize.
**Questions:**
- What is the performance difference between LoFi, HiFi2, and HiFi4 for MoE expert matmuls?
- How does packer_l1_acc affect throughput for expert computations?
- What is the accuracy trade-off for using math_approx_mode?

**Findings:**
Two canonical configs for Wormhole B0 MoE: LOFI (gate/up projections) and HIFI2 (down projection). BOTH have fp32_dest_acc_en=False. packer_l1_acc savings formula = (K_t−1)/K_t. PCC standard threshold > 0.999; strict tier > 0.9995. For PCC ≤ 0.9995 use HIFI2; for > 0.9995 use HIFI4 with fp32_dest_acc_en=True. Benchmark: 20 timed iterations, median + p95, warm-up ≥ 3.

---

## Expert Weight Memory Layout Optimization
**Date:** 2026-03-17
**Status:** Completed
**Why Needed:** Current implementation stores expert weights in DRAM with standard interleaved config. DRAM-sharded layouts may improve memory bandwidth.
**Questions:**
- What is the performance gain from DRAM-sharded weight storage?
- How should expert weights be laid out for optimal prefetch patterns?
- What are the tile size constraints for expert weight sharding?

**Findings:**
Per-expert BF16 memory: 84.0 MB (3 × 7168 × 2048 × 2 bytes for Qwen/DeepSeek-V3). Decode regime (batch_size × top_k ≤ 16) benefits from DRAM-sharded layout; prefill (effective_M > 256) benefits from interleaved. Qwen crossover effective_M ≈ 556; Mixtral ≈ 451. L1 weight double-buffer = 2 × in0_block_w × per_core_N_t × tile_size_bytes (no M_t). Reshard overhead ~2.3 s for Mixtral 8x7B (768 tensors × 3 ms/tensor).

---

## Paged SDPA Decode for GQA (Group Query Attention)
**Date:** 2026-03-17
**Status:** Completed
**Why Needed:** Ling model generates incorrect text during decode. Need to understand paged_sdpa_decode kernel expectations for GQA with 4 KV heads and 16 Q heads.
**Questions:**
1. What does paged_sdpa_decode kernel expect for GQA (4 KV heads to 16 Q heads)?
2. Is there a mismatch in how cur_pos is interpreted?
3. Are there any known issues with TTNN paged attention?

**Findings:**
cur_pos semantics: 0-indexed position of NEXT token to write = current context length BEFORE write (not post-write). For N tokens cached, cur_pos[i]=N. ShardSpec takes element counts not bytes. shard_H % 32 == 0, shard_W % 32 == 0, shard bytes % 32 == 0. Decode-regime GQA uses paged SDPA with KV cache sharded across cores.

---

## Tracy Profiling and MoE Forward Pass Analysis
**Date:** 2026-03-17
**Status:** Completed
**Why Needed:** Need op-level breakdown of MoE forward pass to identify bottlenecks and understand where the 16ms gap occurs.
**Questions:**
1. Have you captured a Tracy trace or op-level breakdown of the MoE forward pass?
2. What operations occur between expert dispatch and combine?
3. Does the 16ms gap scale with sequence length?

**Findings:**
Pattern A (fused SwiGLU): 3 dispatches; Pattern B (unfused): 4 dispatches; Pattern C (with CCL): adds allgather + reduce_scatter. CCL scaling law: O(num_active_tokens × d_model) — no /num_chips term. Memory-bound condition: expert_capacity/32 < num_cores (for Qwen, expert_capacity = seq_len/16, threshold seq_len < ~40,960). T3K ethernet ~7 GB/s/link. Tracy profiling identifies dispatch-to-matmul gaps (Pattern C only) vs compute gaps (all patterns).

---

## SiLU Activation Latency Measurement
**Date:** 2026-03-17
**Status:** Completed
**Why Needed:** Need to understand SiLU activation contribution to overall MoE latency.
**Questions:**
1. What is the current measured latency of the SiLU activation in MoE expert computation?
2. How does SiLU latency compare to the matmul operations?
3. Would fusing SiLU with matmul improve performance?

**Findings:**
SiLU runs on the SFPU as a sequential pass; arithmetic intensity = 0.5 FLOP/byte (2 FLOPs / 4 bytes read+write); SiLU latency is 4–8% of gate_proj matmul at 128 tokens. Fused Pattern A (3 dispatches) vs unfused Pattern B (4 dispatches); fusion beneficial when num_tokens ≥ 16. Only gate_proj SiLU is fusible in a single ttnn.matmul call; up_proj SiLU requires a separate dispatch.

