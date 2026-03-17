# Chapter 7: T3K Multi-Chip MoE Optimization

## Prerequisites

This chapter builds directly on Chapters 1 through 6. Readers should be comfortable with:

- **Chapter 1** ([routing_and_sparsity.md](../ch01_moe_architecture_fundamentals/routing_and_sparsity.md)): sparsity ratio $\rho = k/E$ and its effect on effective compute
- **Chapter 2** ([ttnn_programming_model.md](../ch02_ttnn_wormhole_primer/ttnn_programming_model.md)): `DRAM_MEMORY_CONFIG`, `L1_MEMORY_CONFIG`, and Wormhole B0 memory hierarchy
- **Chapter 3** ([program_configs_batched.md](../ch03_batched_matmul_for_moe/program_configs_batched.md)): `per_core_M`, `out_subblock_h`, and batched matmul program config construction
- **Chapter 4** ([when_sparse_matmul_wins.md](../ch04_sparse_matmul_for_moe/when_sparse_matmul_wins.md)): the sparsity threshold $\rho^*$ below which `sparse_matmul` outperforms batched matmul
- **Chapter 5** ([sparsity_tensor_placement.md](../ch05_sparsity_tensor_construction/sparsity_tensor_placement.md)): sparsity tensor construction and placement for multi-device contexts
- **Chapter 6** ([decision_guide.md](../ch06_comparative_analysis/decision_guide.md)): hybrid strategy selection

You should also be familiar with the T3K guide ([ch02_ttnn_mesh_api/collective_primitives.md](../../t3k_guide/ch02_ttnn_mesh_api/collective_primitives.md)) for `ttnn.all_to_all` usage.

---

## Overview

Chapters 1–6 treated MoE optimization on a single Wormhole B0 chip. This chapter extends those strategies to the **T3K eight-chip mesh**, the hardware configuration used for large MoE inference deployments such as Qwen3.5-35B.

The T3K is a 1×8 linear mesh of Wormhole B0 devices connected by Ethernet links (~12.5 GB/s each). Expert Parallelism (EP) maps naturally onto this topology: with $E = 256$ experts and $N = 8$ devices, each chip holds $E_d = 32$ experts. The communication primitives that tie them together are `ttnn.all_to_all` dispatch (routing tokens to expert-holding devices) and `ttnn.all_to_all` combine (returning weighted expert outputs to originating devices).

The single-chip strategies from earlier chapters remain valid per-device, but two new constraints reshape the optimization problem:

1. **Communication overhead.** Each MoE layer incurs two all-to-all operations. At decode batch sizes ($B = 1$–$32$), this communication cost dominates the expert FFN compute time. The per-chip local batch is small, so arithmetic intensity is low and all-to-all latency becomes the primary bottleneck.

2. **Reduced per-chip local batch.** Under uniform routing each device receives $T_d \approx B \cdot k / N \cdot E_d / E = B$ tokens (the expected number stays $B$ due to the token-to-expert structure, but each individual expert sees $C$ tokens where $C$ is the expert capacity — see Section 3 of `expert_parallelism_on_t3k.md`). With $C = 2$ at $B = 32$, core utilization per expert is $2/32 = 6.25\%$.

**Key insight:** on T3K, `sparse_matmul` is even more critical than on a single chip. With only 32 local experts and small $C$ at decode, tile-skipping avoids wasting cycles on zero-padded rows for experts that received no tokens. The sparsity ratio $\rho = k/E = 3.1\%$ is extreme; at $B = 1$ only 8 of 256 experts fire at all, meaning most devices' experts are entirely idle on any given decode step.

---

## Key Technical Parameters

| Parameter | Value | Notes |
|---|---|---|
| Model | Qwen3.5-35B | Primary reference model |
| $E$ | 256 | Total experts |
| $k$ | 8 | Top-k routing |
| $N$ | 8 | T3K devices |
| $E_d$ | 32 | Experts per device ($= E/N$) |
| $H$ | 7168 | Hidden dimension |
| $\rho$ | 3.1% | Active fraction at $B = 1$ |
| Mesh shape | (1, 8) | `ttnn.MeshDevice` |
| Topology | `ttnn.Topology.Linear` | 1×8 linear |
| `cluster_axis` | 1 | Column axis for all-to-all |
| Ethernet bandwidth | ~12.5 GB/s | Per Ethernet link |
| Cores per chip | 80 Tensix | Wormhole B0 |
| L1 per core | 1.5 MB | Wormhole B0 |
| Tile size | 32×32 | TTNN tile |

---

## Chapter Contents

| File | Description |
|---|---|
| `expert_parallelism_on_t3k.md` | EP degree, all-to-all dispatch/combine API, latency model, when EP adds overhead vs. when it is forced by memory |
| `sharding_strategies.md` | Activation tensor distribution, expert weight placement in DRAM, sparsity tensor construction per chip, replicated vs. sharded tensors |
| `program_configs_t3k.md` | Per-chip program config derivation under EP, grid utilization analysis, worked examples for Qwen3.5-35B and Mixtral 8x7B |

---

## Navigation

Read the files in order:

1. [expert_parallelism_on_t3k.md](./expert_parallelism_on_t3k.md) — start here for the communication model
2. [sharding_strategies.md](./sharding_strategies.md) — tensor placement decisions
3. [program_configs_t3k.md](./program_configs_t3k.md) — concrete config derivations and worked examples

---

## References

- Qwen3.5-35B model card and architecture notes
- T3K hardware specifications (Tenstorrent internal)
- Chapter 1: [routing_and_sparsity.md](../ch01_moe_architecture_fundamentals/routing_and_sparsity.md)
- Chapter 4: [when_sparse_matmul_wins.md](../ch04_sparse_matmul_for_moe/when_sparse_matmul_wins.md)
- Chapter 5: [sparsity_tensor_placement.md](../ch05_sparsity_tensor_construction/sparsity_tensor_placement.md)
- T3K guide: [collective_primitives.md](../../t3k_guide/ch02_ttnn_mesh_api/collective_primitives.md)

## Next Steps

After completing this chapter, proceed to **Chapter 8** for end-to-end prefill/decode pipeline integration on T3K, including memory budget planning and layer scheduling across multiple MoE layers.
