# Chapter 3 — MatMul Optimizations: Program Configs, Sharding, and Kernel Tuning

`ttnn.matmul` performance depends heavily on the program config. The TTNN runtime exposes four distinct program configs for matrix multiplication, each optimized for a different regime of tensor shapes and memory layouts. Choosing the right config — and tuning parameters like `in0_block_w`, `out_subblock_h`/`out_subblock_w`, and core grid size — is what separates a 2× speedup from a 10× speedup over the default auto-heuristic. LLM workloads run two fundamentally different matmul regimes: decode (small M, large K and N, bandwidth-bound) and prefill (large M, large K and N, compute-bound). Each regime calls for a different config. This chapter covers all four program configs, the weight layout and quantization decisions that make each fast, and the sharding strategies that eliminate unnecessary DRAM traffic.

## Learning Objectives

- Understand all four `ttnn.matmul` program configs and the workload regime each targets
- Know when to choose DRAM-sharded vs multicast vs 1D vs standard 2D configs based on M, K, N shape and phase (decode vs prefill)
- Understand the output subblock sizing constraint imposed by the Dst register and how to maximize subblock size for register reuse
- Apply BFP4 or BFP8 weight quantization appropriately per layer type, and pair each dtype with the correct math fidelity
- Design L1 sharding layouts that eliminate activation DRAM reads and chain cleanly across ops without unnecessary re-sharding

## The Four Matmul Program Configs

| Config | Best for | Key characteristic |
|---|---|---|
| `MatmulMultiCoreReuseProgramConfig` | Standard 2D tiled matmul | Independent per-core DRAM fetch; no cross-core weight sharing |
| `MatmulMultiCoreReuseMultiCastProgramConfig` | Large-M prefill workloads | Weight multicast to many cores |
| `MatmulMultiCoreReuseMultiCast1DProgramConfig` | 1D-sharded activations (fused MLP) | Activation shard fed directly to core |
| `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` | Decode (small M, large N) | Each core fetches its own weight shard from DRAM |

## Files in This Chapter

Read in the following order:

1. **`matmul_program_configs.md`** — The four program configs explained in depth: parameters, use cases, and the Dst register constraint that governs output subblock sizing.
2. **`weight_layout_and_quantization.md`** — How weights are loaded, laid out, and quantized (BFP4/BFP8/BF16); math fidelity pairing; pre-transposed weight layout.
3. **`l1_sharding_for_matmul.md`** — Height sharding, block sharding, and DRAM-sharded decode; when each eliminates DRAM traffic and how to chain sharded ops cleanly.

---

Previous: [Chapter 2 — Attention Optimizations](../ch2_attention_optimizations/index.md)
