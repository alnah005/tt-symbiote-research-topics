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
**Status:** Pending
**Why Needed:** DeepSeek-V3 uses bfloat4_b/bfloat8_b weight quantization for experts, but Qwen uses full bfloat16. Need to evaluate quantization trade-offs.
**Questions:**
- What accuracy loss is expected from bfloat4_b vs bfloat8_b vs bfloat16 for expert weights?
- How does weight quantization affect compute throughput on Wormhole?
- Which projections (gate/up/down) are most sensitive to quantization?

**Findings:**
[Pending research]

---

## Compute Kernel Configuration for MoE
**Date:** 2026-03-17
**Status:** Pending
**Why Needed:** DeepSeek-V3 uses COMPUTE_KERNEL_CONFIG_LOFI with packer_l1_acc, but Qwen MoE doesn't specify compute kernel configs. Need to optimize.
**Questions:**
- What is the performance difference between LoFi, HiFi2, and HiFi4 for MoE expert matmuls?
- How does packer_l1_acc affect throughput for expert computations?
- What is the accuracy trade-off for using math_approx_mode?

**Findings:**
[Pending research]

---

## Expert Weight Memory Layout Optimization
**Date:** 2026-03-17
**Status:** Pending
**Why Needed:** Current implementation stores expert weights in DRAM with standard interleaved config. DRAM-sharded layouts may improve memory bandwidth.
**Questions:**
- What is the performance gain from DRAM-sharded weight storage?
- How should expert weights be laid out for optimal prefetch patterns?
- What are the tile size constraints for expert weight sharding?

**Findings:**
[Pending research]

---

## Paged SDPA Decode for GQA (Group Query Attention)
**Date:** 2026-03-17
**Status:** Pending
**Why Needed:** Ling model generates incorrect text during decode. Need to understand paged_sdpa_decode kernel expectations for GQA with 4 KV heads and 16 Q heads.
**Questions:**
1. What does paged_sdpa_decode kernel expect for GQA (4 KV heads to 16 Q heads)?
2. Is there a mismatch in how cur_pos is interpreted?
3. Are there any known issues with TTNN paged attention?

Findings: [Pending research]

---

## Tracy Profiling and MoE Forward Pass Analysis
**Date:** 2026-03-17
**Status:** Pending
**Why Needed:** Need op-level breakdown of MoE forward pass to identify bottlenecks and understand where the 16ms gap occurs.
**Questions:**
1. Have you captured a Tracy trace or op-level breakdown of the MoE forward pass?
2. What operations occur between expert dispatch and combine?
3. Does the 16ms gap scale with sequence length?

**Findings:**
[Pending research - requires running Tracy profiler on actual hardware]

---

## SiLU Activation Latency Measurement
**Date:** 2026-03-17
**Status:** Pending
**Why Needed:** Need to understand SiLU activation contribution to overall MoE latency.
**Questions:**
1. What is the current measured latency of the SiLU activation in MoE expert computation?
2. How does SiLU latency compare to the matmul operations?
3. Would fusing SiLU with matmul improve performance?

**Findings:**
[Pending research - requires profiling on hardware]

---

## Sparse Matmul and W1/W3 Fusion Cleanup for Qwen MoE
**Date:** 2026-03-17
**Status:** Completed
**Why Needed:** The Qwen MoE implementation has environment variable toggles for sparse_matmul vs batched matmul and fused vs unfused w1/w3 projections. These toggles add code complexity and dead code paths that need to be removed.
**Questions:**
1. What environment variables control the implementation paths?
2. Which code paths need to be removed?
3. What is the impact on tests and behavior?

**Findings:**
Full cleanup plan written: `PLAN_qwen_moe_cleanup.md`

- **Environment variables to remove:** `TT_QWEN_USE_SPARSE_MATMUL` and `TT_QWEN_FUSED_GATE_UP` (both default to "1" enabled)
- **File to modify:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/qwen_moe.py`
- **Code paths to remove:**
  - Batched matmul path (~25 lines in forward())
  - Unfused sparse_matmul path (~27 lines in forward())
  - Conditional weight creation (`tt_w1_proj`, `tt_w3_proj`)
  - Conditional program config creation (`_gate_up_program_config`)
- **Net reduction:** ~100+ lines of dead code removed
- **Behavior change:** None - the default path (sparse_matmul + fused w1/w3) becomes the only path
- **Testing:** All existing tests should pass unchanged since they already test the default path

