# Chapter 5: Sparsity Tensor Construction

## Prerequisites

Before reading this chapter, you should have completed:

- **Chapter 1** (`ch01_moe_architecture_fundamentals/`) — MoE routing, expert capacity, and token dispatch
- **Chapter 2** (`ch02_ttnn_wormhole_primer/`) — TTNN tensor model, memory configs, and layout conventions
- **Chapter 3** (`ch03_batched_matmul_for_moe/`) — Batched matmul patterns and activation tensor shapes
- **Chapter 4** (`ch04_sparse_matmul_for_moe/sparse_matmul_internals.md`) — **Required.** The sparsity tensor format accepted by `ttnn.sparse_matmul`, tile-level skip semantics, and program caching constraints

In particular, Chapter 4 `sparse_matmul_internals.md` defines the exact contract between the caller and `ttnn.sparse_matmul`: the mask shape, value encoding, and the condition under which tile rows are skipped. Chapter 5 is the implementation counterpart — it explains how to build a correct mask from router output and place it efficiently in device memory.

---

## Learning Objectives

By the end of this chapter you will be able to:

- Describe the required shape, dtype, and layout of the sparsity tensor accepted by `ttnn.sparse_matmul` and explain why each constraint exists
- Convert top-k router output into a correctly shaped TTNN sparsity tensor, handling partial tiles and capacity overflow correctly
- Choose between L1 and DRAM placement for the sparsity tensor and justify the choice for decode vs. prefill regimes
- Identify the six most common construction mistakes, detect them from symptoms, and apply the correct fix
- Integrate sparsity tensor construction into a TTNN Trace-based decode loop without triggering recompilation

---

## Quick-Reference Checklist

Use this checklist before every call to `ttnn.sparse_matmul`. Each item corresponds to a pitfall documented in `common_pitfalls.md`.

| # | Check | Common mistake |
|---|-------|----------------|
| 1 | `mask.shape[0] == E_d * ceil(C / 32)` and `mask.shape[1] == ceil(H / 32)` | Shape mismatch — M_t or K_t derived from wrong value (see P1) |
| 2 | Every tile row that contains at least one real token has mask value 1 | Last partial tile row incorrectly zeroed when `C % 32 != 0` (see P2) |
| 3 | Mask is recomputed from fresh router output on every decode step | Stale mask from previous step reused in KV-cache loop (see P3) |
| 4 | `mask.memory_config == ttnn.L1_MEMORY_CONFIG` (decode regime) | Mask placed in DRAM, adding bandwidth overhead per tile row (see P4) |
| 5 | `mask.dtype == ttnn.uint8` | Mask constructed with `torch.float32` or `ttnn.bfloat16` (see P5) |
| 6 | Activation and mask tensors are padded to a canonical (B, S, C) shape | Dynamic M_t from variable B/S forces recompilation on every call (see P6) |

> **Warning:** Items 2 and 3 are silent correctness bugs — TTNN will not raise an error. The model will produce wrong outputs. Always validate mask content programmatically during development.

---

## Reading Order

Read the files in the following order:

1. **[`sparsity_tensor_format.md`](./sparsity_tensor_format.md)** — Start here. Establishes the exact contract: shape, dtype, layout, memory placement, and the meaning of each mask entry. All other files assume this contract.

2. **[`constructing_from_router_output.md`](./constructing_from_router_output.md)** — Step-by-step construction pipeline from routing indices to a `ttnn.uint8` TILE_LAYOUT tensor on device. Includes complete Python pseudocode with shapes annotated at every step.

3. **[`sparsity_tensor_placement.md`](./sparsity_tensor_placement.md)** — Memory placement decisions: L1 vs. DRAM, sharding for multi-device expert parallelism, and integration with `ttnn.Trace`.

4. **[`common_pitfalls.md`](./common_pitfalls.md)** — Reference section for debugging. Six failure modes with symptoms, root causes, and fixes. Consult this when something is wrong.

---

## Chapter Notation

The following symbols are used consistently across all files in this chapter. They are consistent with notation introduced in Chapter 1 and Chapter 4.

| Symbol | Definition | Qwen3.5-35B value |
|--------|------------|-------------------|
| $M_t$ | Number of tile rows in the activation tensor: $M_t = \lceil C / 32 \rceil$ | 1 (decode, B=1); 8 (prefill, B=4, S=2048) |
| $K_t$ | Number of tile columns in the activation tensor: $K_t = \lceil H / 32 \rceil$ | 224 |
| $E$ | Total number of experts across all devices | 256 |
| $E_d$ | Number of local experts per device: $E_d = E / N$ | 32 |
| $k$ | Number of experts selected per token (top-k routing) | 8 |
| $B$ | Batch size (number of sequences) | varies |
| $S$ | Sequence length | 1 (decode); 2048 (prefill) |
| $\rho$ | Sparsity ratio: fraction of expert slots (and tile rows) that are active: $\rho = \text{active\_experts} / E_d$ (consistent with Chapter 4 convention). Fraction of tile rows skipped = $1 - \rho$. | $\approx 0.031$ for B=1 decode |
| $C$ | Expert capacity: maximum number of tokens routed to a single expert: $C = \lceil k \times B \times S / E \rceil$ | 1 (B=1, S=1); 256 (B=4, S=2048) |

> **Tip:** For Qwen3.5-35B decode with B=1, $\rho \approx 0.031$: only 1 of the 32 local experts is active on average. That means $1 - \rho = 31/32 \approx 0.969$ of tile rows are skipped — approximately 97% of expert FFN compute is avoided by `sparse_matmul`.

---

## References

- Chapter 4, `sparse_matmul_internals.md` — sparsity tensor contract and tile-skip semantics
- Chapter 1, MoE architecture fundamentals — routing, top-k selection, expert capacity
- Chapter 2, TTNN Wormhole primer — tensor layout, memory configs, tile model
- Qwen3.5-35B model card — architecture parameters (E=256, k=8, H=7168, N=8)
