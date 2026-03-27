# Chapter 3 — Host Round-Trip Replication

## Scope

This chapter examines one of the most consequential latency contributions in the Ling decode path that is not a compute kernel: the **host round-trip** that converts the all-reduced QKV tensor from a device-resident TTNN multi-device tensor into a `ReplicateTensorToMesh`-distributed tensor before `paged_sdpa_decode` can consume it.

The operation responsible is called `_to_replicated` in the attention layer implementation. Despite its compact name, it involves three distinct sub-steps: pulling data from all 8 T3K chips to host DRAM over PCIe, constructing a new `torch.Tensor` on the CPU, and pushing the result back to all 8 chips simultaneously. At decode batch=1, this round-trip is a significant fraction of the total attention latency because the PCIe transaction overhead is paid regardless of payload size. In ROW_MAJOR layout the tensor is only 6 KB, but in TILE_LAYOUT (the actual on-device layout) the transfer is approximately 192 KB per chip; see [`roundtrip_mechanics.md`](./roundtrip_mechanics.md) for the exact size accounting.

This chapter answers **Question 2** of the guide: *Why is the `_to_replicated` round-trip required, how much latency does it add, and can it be eliminated by a device-side primitive?*

## Prerequisites

Readers should have worked through Chapters 1 and 2 before proceeding. The specific concepts required are:

- **T3K topology and PCIe connectivity** — the physical path data travels when a tensor is moved between chip DRAM and host DRAM (see Chapter 1, [`t3k_topology_primer.md`](../ch1_model_and_hardware_context/t3k_topology_primer.md))
- **TTNN multi-device tensor types** — the distinction between a sharded distribution (e.g., WIDTH_SHARDED with `ShardTensorToMesh`) and a replicated distribution (`ReplicateTensorToMesh`), and why these are different TTNN tensor types even when all chips hold numerically identical values (see Chapter 1, [`t3k_topology_primer.md`](../ch1_model_and_hardware_context/t3k_topology_primer.md))
- **Fused QKV output layout** — the tensor that enters `_to_replicated` is a `(1, 1, 1, 3072)` BF16 tensor produced by the all-reduce at the end of `TTNNLinearIColShardedWAllReduced`; all chips hold numerically identical values, but the TTNN tensor object has a sharded distribution type (see Chapter 2, [`fusion_mechanics.md`](../ch2_fused_qkv_projection/fusion_mechanics.md)). Note: this all-reduce output tensor resides in DRAM (`DRAM_MEMORY_CONFIG`) on each chip when it enters `_to_replicated`

No additional prerequisites are required.

## Why the Round-Trip Exists

After the fused QKV all-reduce, each chip holds an identical copy of the `(1, 1, 1, 3072)` output, but the TTNN runtime tracks the tensor as a **sharded multi-device tensor** (specifically, a `ShardTensorToMesh`-distributed type) rather than as a replicated one. The `paged_sdpa_decode` kernel has a hard constraint on its input tensor type: it requires Q, K, and V to be in **`ReplicateTensorToMesh` distribution** so that each chip can independently execute the attention computation against its local KV-cache pages without any chip-to-chip coordination during the SDPA kernel itself.

TTNN currently lacks a device-side primitive that can reinterpret or re-tag an existing sharded tensor as replicated when the shard values are identical — it has no "in-place reinterpret-distribution" operation. The only supported path is to materialise the full tensor on the host via `ConcatMeshToTensor` and then re-distribute it via `from_torch` with a `ReplicateTensorToMesh` mapper. This is what `_to_replicated` does.

## Reading Order

Work through the files in this order:

1. [`roundtrip_mechanics.md`](./roundtrip_mechanics.md) — The step-by-step trace of `_to_replicated`: which TTNN and PyTorch APIs are called, in what order, and what data volumes are transferred.
2. [`host_transfer_overhead.md`](./host_transfer_overhead.md) — Quantitative PCIe latency model, estimated round-trip cost at batch=1, measurement methodology, and batch-size sensitivity.
3. [`device_side_alternatives.md`](./device_side_alternatives.md) — Survey of TTNN primitives that could replace the round-trip without involving the host, feasibility analysis for each, and a recommended path forward.

## Key Symbols Used in This Chapter

| Symbol | Meaning |
|---|---|
| `C` | Total output columns of fused QKV weight = 3072 |
| `P` | Number of chips = 8 |
| `B` | PCIe transfer bandwidth per chip (practical) |
| `t_RT` | Full host round-trip latency (device→host→device) |
| `V_tensor` | Byte size of the QKV tensor per direction of transfer |

**Start reading:** [Roundtrip Mechanics](roundtrip_mechanics.md)
