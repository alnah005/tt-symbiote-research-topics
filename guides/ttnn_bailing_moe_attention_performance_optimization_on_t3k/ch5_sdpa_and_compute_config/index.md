# Chapter 5 — SDPA Kernel Configuration and Compute Fidelity

## Scope

This chapter examines the two compute-side configuration decisions that govern the accuracy and throughput of `paged_sdpa_decode` in the Ling decode path: the chunk-size parameters (`q_chunk_size` and `k_chunk_size`) that control the kernel's internal tiling strategy, and the math-fidelity setting (`HiFi4` vs. `HiFi2`) that determines the precision of the FPU operations inside the SDPA kernel.

The Ling model invokes `paged_sdpa_decode` with `q_chunk_size=0` and `k_chunk_size=0`. These zero values are not obvious: they do not mean "zero tiles" — they encode a specific kernel behaviour. Understanding what that behaviour is, why it is correct for Ling's 16/4 GQA layout, and whether a different setting would improve throughput is the subject of [`paged_sdpa_chunk_sizes.md`](./paged_sdpa_chunk_sizes.md).

The kernel is additionally configured with:

```python
compute_kernel_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
```

`HiFi4` is the highest fidelity mode available on Wormhole and is the safe default for BFloat16 attention. Whether it is *necessary* — as opposed to the lower-cost `HiFi2` — depends on the precision requirements of the softmax and QK dot product in Ling's specific GQA layout. This question is analysed in [`math_fidelity_tradeoff.md`](./math_fidelity_tradeoff.md).

This chapter answers **Questions 4 and 5** of the guide:

- **Question 4:** What do `q_chunk_size=0` and `k_chunk_size=0` mean in `paged_sdpa_decode`, and are they the correct settings for Ling's GQA configuration?
- **Question 5:** Can `HiFi4` be safely replaced by `HiFi2` in the SDPA compute kernel config, and what is the expected throughput gain?

## Prerequisites

Readers should be familiar with the following material from earlier chapters:

- **GQA head configuration** — Ling uses 16/4 GQA: `num_heads=16` Q heads, `num_kv_heads=4` KV heads, `head_dim=128`. The grouping factor is 4 (each KV head serves 4 Q heads). GQA head counts determine how the paged SDPA kernel partitions work across Q and KV tiles (see Chapter 1, [`ling_model_overview.md`](../ch1_model_and_hardware_context/ling_model_overview.md)).
- **Tensor layout entering SDPA** — After the memory-config transitions catalogued in Chapter 4, Q arrives at `paged_sdpa_decode` in DRAM INTERLEAVED layout, shape `(1, 16, 32, 128)` (tile-padded), BF16 TILE_LAYOUT. The paged KV-cache pool lives in DRAM and is accessed through a block table. Q enters `paged_sdpa_decode` in DRAM_INTERLEAVED layout after the T2a post-RoPE eviction (Ch4); no further memory-config transition is applied between RoPE output and SDPA input (see Chapter 4, [`decode_tensor_lifecycle.md`](../ch4_memory_config_transitions/decode_tensor_lifecycle.md)). *Shape derivation:* After `_to_replicated`, the fused QKV output is reshaped and split. For Q: the 16-head, 128-dim configuration (from Ch1) with TILE_LAYOUT padding gives shape `(1, 16, 32, 128)` — see Chapter 4 for the full split sequence.
- **TTNN compute kernel config structure** — `ttnn.WormholeComputeKernelConfig` is the hardware-specific compute config object accepted by SDPA and other Wormhole kernels. Its three fields (`math_fidelity`, `fp32_dest_acc_en`, `packer_l1_acc`) are covered in this chapter; readers who need background on TTNN op dispatch should review Chapter 1, [`t3k_topology_primer.md`](../ch1_model_and_hardware_context/t3k_topology_primer.md).

## Reading Order

Work through the files in this order:

1. [`paged_sdpa_chunk_sizes.md`](./paged_sdpa_chunk_sizes.md) — What `q_chunk_size` and `k_chunk_size` control, why zero means "process all heads as a single chunk," and whether explicit values would help or hurt throughput for Ling's GQA layout.
2. [`math_fidelity_tradeoff.md`](./math_fidelity_tradeoff.md) — How Wormhole fidelity levels map to hardware FPU modes, the precision impact of `HiFi2` vs. `HiFi4` on BFloat16 attention, and a concrete recommendation for whether to lower fidelity.

## Key Symbols Used in This Chapter

| Symbol | Meaning |
|---|---|
| `N_q` | Number of Q heads = 16 |
| `N_kv` | Number of KV heads = 4 |
| `G` | GQA grouping factor = `N_q / N_kv` = 4 |
| `H` | `head_dim` = 128 elements |
| `T` | `TILE_SIZE` = 32 elements |
| `S` | KV sequence length (total cached tokens at decode step) |
| `q_chunk_size` | Tiling stride over Q heads inside the SDPA kernel |
| `k_chunk_size` | Tiling stride over KV sequence length inside the SDPA kernel |
| `MathFidelity` | TTNN enum: `LoFi`, `HiFi2`, `HiFi4` |
| `fp32_dest_acc_en` | Enable FP32 destination accumulator in Wormhole FPU |
| `packer_l1_acc` | Enable L1 accumulation during packing (reduces register spills) |

**Start reading:** [Paged SDPA Chunk Sizes](paged_sdpa_chunk_sizes.md)
