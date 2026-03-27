# Transition Cost Model

## Overview

This file explains how TTNN executes a memory-config transition at the hardware level, derives a cost formula as a function of tensor size and data path, and then applies that formula to each of the transitions catalogued in `decode_tensor_lifecycle.md`. It concludes by identifying the dominant cost contributor and explaining why.

All cost estimates in this file are marked `[ESTIMATE]` because they are derived from known hardware parameters and theoretical models rather than direct measurement on a live T3K system. Measurement methodology is provided in Chapter 7, `ttnn_op_timers.md`.

## How TTNN Executes a Memory-Config Transition

A call to `ttnn.to_memory_config(tensor, target_config)` always results in a **data-movement kernel** being dispatched to one or more Tensix cores. The kernel reads the source tensor from its current location and writes the result to the new location. No arithmetic is performed — this is a pure copy with possible re-tiling.

### Data Movement Paths

There are three physically distinct data movement paths on a Wormhole chip:

**Path A: DRAM → L1**
Data is read from DRAM banks through the chip's NOC (Network-on-Chip) and written into Tensix core L1 SRAM. This is the path for transitions T1a, T1b, T3a, T3b, and the conditional T4. DRAM bandwidth on Wormhole is approximately **288 GB/s** aggregate (all DRAM banks, all NOC ports) [ESTIMATE], but for a single-core or small-grid data movement kernel the practical bandwidth to L1 is limited by the NOC link rate and L1 write bandwidth. For a single Tensix core, the effective L1 write bandwidth from DRAM is approximately **8–16 GB/s** [ESTIMATE].

**Path B: L1 → DRAM**
Data is read from Tensix core L1 and written to DRAM banks through the NOC. This is the path for transitions T2a and T2b. The effective bandwidth is similar to Path A in the reverse direction: approximately **8–16 GB/s** per core [ESTIMATE].

**Path C: L1 → L1 (intra-chip)**
Data is read from one set of core L1 banks and written directly to another set, routing through the NOC without touching DRAM. This path is faster than A or B because DRAM access latency (tens of clock cycles per access) is avoided. Effective bandwidth for L1→L1 transfers is approximately **32–48 GB/s** [ESTIMATE] for small multi-core transfers, limited by NOC link saturation.

### Kernel Dispatch Overhead

Every `ttnn.to_memory_config` call incurs a **fixed overhead** independent of data size. This overhead includes:

1. **Python-to-C++ dispatch**: The Python call crosses the pybind11 layer to reach the TTNN C++ runtime. Estimated cost: **1–3 µs** [ESTIMATE].
2. **Op dispatch and enqueue**: The TTNN runtime identifies the appropriate kernel, validates tensor metadata, and enqueues the command into the device's command queue. Estimated cost: **2–5 µs** [ESTIMATE].
3. **Kernel launch**: The Tensix cores receive the dispatch command and begin execution. For small tensors, this launch latency can dominate over the actual data transfer time. Estimated cost: **2–5 µs** [ESTIMATE].

**Total fixed overhead per transition: approximately 5–13 µs [ESTIMATE].**

At decode batch=1, the tensors involved in memory-config transitions are very small (a few KB). This means that for most transitions in the Ling decode path, the **fixed overhead dominates** over the bandwidth-limited transfer time.

### Transfer Time Formula

For a tensor of `B` bytes over a path with effective bandwidth `BW` (GB/s), the transfer time is:

```
t_transfer = B / BW

where B is in bytes, BW in bytes/s.
```

The total transition latency is:
```
t_trans = t_overhead + t_transfer = t_overhead + (B / BW)
```

For the transitions in this chapter, `B` is small enough that `t_transfer << t_overhead` for all cases, as shown in the per-transition estimates below.

### Re-Tiling Cost

If the source tensor is in TILE_LAYOUT and the destination requires a different tile alignment (e.g., the shard boundaries fall at positions that are not multiples of 32), the transition kernel must re-tile the data. For the transitions in the Ling decode path, the shard shape `(32, 128)` is tile-aligned (32 = TILE_SIZE, 128 = 4×TILE_SIZE), so re-tiling adds no cost — but this claim is only valid once tensors are in the expected shape.

**T3b requires special attention.** `v_raw` arrives at Stage 6 with shape `(1, 1, 32, 512)`, which does not match the `(32, 128)` shard boundary of `kv_update_mem`. A `ttnn.reshape` to `(1, 4, 32, 128)` must be applied before the `ttnn.to_memory_config` call; `to_memory_config` alone cannot perform a shape change. Once the reshape has been applied and V has shape `(1, 4, 32, 128)`, the shard alignment holds and no re-tiling is required. The no-re-tiling claim for T3b is therefore valid only after this reshape prerequisite is satisfied.

## Tensor Size Reference

Before computing per-transition costs, the byte size of each tensor involved is established.

Table: Tensor sizes for Ling decode step transitions

| Tensor | Tile-padded shape | Elements | BF16 bytes |
|---|---|---|---|
| `q_heads` / `q_rope_in` | `(1,16,32,128)` | 65,536 | 131,072 (128 KB) |
| `k_heads` / `k_rope_in` | `(1, 4,32,128)` | 16,384 |  32,768  (32 KB) |
| `v_raw` / `v_update_in` | `(1, 1,32,512)` → reshaped `(1,4,32,128)` | 16,384 |  32,768  (32 KB) |
| `q_post_rope` | `(1,16,32,128)` | 65,536 | 131,072 (128 KB) |
| `k_post_rope` / `k_update_in` | `(1, 4,32,128)` | 16,384 |  32,768  (32 KB) |

Note: The 16-head Q tensor at 128 KB is the largest tensor traversing any transition in this decode step.

## Per-Transition Cost Estimates

The following estimates use:
- Fixed overhead per transition: **8 µs** [ESTIMATE] (mid-point of 5–13 µs range)
- Effective DRAM↔L1 bandwidth (single kernel, small tensor): **10 GB/s** [ESTIMATE]
- Effective L1→L1 bandwidth: **40 GB/s** [ESTIMATE]

Table: Per-transition cost estimates applying `t_trans = t_overhead + (B / BW)` [all ESTIMATE]

| Transition | Description | Tensor | Size | Path | t_transfer | t_overhead | Total |
|---|---|---|---|---|---|---|---|
| T1a | Q DRAM→L1 (RoPE in) | `q_heads` | 128 KB | DRAM→L1 | 13.1 µs | 8 µs | ≈ 21 µs |
| T1b | K DRAM→L1 (RoPE in) | `k_heads` | 32 KB | DRAM→L1 | 3.3 µs | 8 µs | ≈ 11 µs |
| T2a | Q L1→DRAM (post-RoPE) | `q_rope_out` | 128 KB | L1→DRAM | 13.1 µs | 8 µs | ≈ 21 µs |
| T2b | K L1→DRAM (post-RoPE) | `k_rope_out` | 32 KB | L1→DRAM | 3.3 µs | 8 µs | ≈ 11 µs |
| T3a | K DRAM→L1 (`paged_update_on_device`) | `k_post_rope` | 32 KB | DRAM→L1 | 3.3 µs | 8 µs | ≈ 11 µs |
| T3b | V DRAM→L1 (`paged_update_on_device`) | `v_raw` | 32 KB | DRAM→L1 | 3.3 µs | 8 µs | ≈ 11 µs |
| T4 | Q DRAM→L1 (SDPA in, conditional) | `q_post_rope` | 128 KB | DRAM→L1 | 13.1 µs | 8 µs | ≈ 21 µs |
| T_norm_in | QK norm Q+K DRAM→L1 (in, 2 transitions) | `q`+`k` 128+32 KB | DRAM→L1 | — | — | ≈ 32 µs |
| T_norm_out | QK norm Q+K L1→DRAM (out, 2 transitions) | `q`+`k` 128+32 KB | L1→DRAM | — | — | ≈ 32 µs |

Note: T4 cost applies only when `paged_sdpa_decode` requires a fresh DRAM→L1 load of `q_post_rope`. T_norm_in and T_norm_out each apply only when `use_qk_norm=True`. T_norm_in bundles the two DRAM→L1 moves before norm (Q-in 21 µs + K-in 11 µs = 32 µs); T_norm_out bundles the two L1→DRAM moves after norm (Q-out 21 µs + K-out 11 µs = 32 µs). They are listed separately because Priority 1 eliminates T_norm_in (and T2a/T2b) while T_norm_out remains non-eliminable without a kernel output config change. Full T_norm breakdown is in Chapter 6, `qk_norm_latency.md`.

## Total Transition Cost per Decode Step

Table: Aggregated transition latency for Ling decode step (all estimates)

| ID | Description | Tensor size | Path | Estimated latency [ESTIMATE] |
|---|---|---|---|---|
| T1a | Q DRAM→L1 (RoPE in) | 128 KB | DRAM→L1 | 21 µs |
| T1b | K DRAM→L1 (RoPE in) |  32 KB | DRAM→L1 | 11 µs |
| T2a | Q L1→DRAM (RoPE out) | 128 KB | L1→DRAM | 21 µs |
| T2b | K L1→DRAM (RoPE out) |  32 KB | L1→DRAM | 11 µs |
| T3a | K DRAM→L1 (`paged_update_on_device` in) | 32 KB | DRAM→L1 | 11 µs |
| T3b | V DRAM→L1 (`paged_update_on_device` in) | 32 KB | DRAM→L1 | 11 µs |
| T4 | Q DRAM→L1 (SDPA in, conditional) | 128 KB | DRAM→L1 | 21 µs |
| T_norm_in | QK norm Q+K DRAM→L1 (in, eliminable) | 128+32 KB | DRAM→L1 | 32 µs |
| T_norm_out | QK norm Q+K L1→DRAM (out, non-eliminable) | 128+32 KB | L1→DRAM | 32 µs |
| **Total (without T4)** | | | | **≈ 150 µs** |
| **Total (with T4)** | | | | **≈ 171 µs** |

These estimates are per-chip (all chips execute the same transitions in parallel, so the wall-clock time is not multiplied by 8).

## Dominant Transition and Why

The transitions split into two groups by cost:

- **Q transitions (T1a, T2a, T_norm_in_Q, T_norm_out_Q, conditional T4):** each costs ~21 µs because Q is 128 KB — the tile-padded (1,16,32,128) tensor. The 16-head structure inflates Q to 4× the size of K or V, making every Q-touching transition more expensive.
- **K and V transitions (T1b, T2b, T3a, T3b, T_norm_in_K, T_norm_out_K):** each costs ~11 µs because K and V are 32 KB at 4 KV heads.

The **dominant transition by individual cost** is any transition that involves the full Q tensor: T1a, T2a, T4 (if present), and the Q side of QK norm. Each costs approximately 21 µs [ESTIMATE].

However, by **aggregate contribution**, the QK norm transitions are the dominant overhead block. When `use_qk_norm=True`, the norm introduces four DRAM↔L1 transitions (2 for Q, 2 for K) totalling approximately **64 µs** [ESTIMATE] — more than any single named transition group. These four transitions divide into two distinct sets: T_norm_in (Q-in 21 µs + K-in 11 µs = 32 µs), which moves Q and K from DRAM into L1 before the norm kernel, and T_norm_out (Q-out 21 µs + K-out 11 µs = 32 µs), which evicts Q and K back to DRAM after the norm kernel. This distinction matters for optimisation: T_norm_in is eliminable (Priority 1 targets it together with T2a and T2b), while T_norm_out is non-eliminable without changing the norm kernel's output memory config. The norm kernel cannot fuse its data-movement with the RoPE path: the two kernels have different input layout requirements, so a separate round-trip to DRAM is mandatory between them.

The root cause of the dominance is **kernel launch overhead, not bandwidth**. At decode batch=1, the data volumes are small enough that the actual bytes-per-second limit of the DRAM↔L1 path accounts for only 3–13 µs of each transition's estimated 11–21 µs total cost. The remaining 8 µs (the fixed overhead estimate) represents command dispatch, kernel startup, and host-device synchronisation. Halving the tensor size would not halve the transition cost; it would merely reduce the bandwidth component, which is already the minority contributor.

**Implication:** Optimisation strategies that reduce tensor sizes (e.g., reducing head count) offer limited benefit for transition cost. Strategies that eliminate transitions entirely (fusing kernels, changing output memory configs to avoid round-trips) are far more effective, as they eliminate the full ~8–21 µs of both components. This is the subject of `optimization_opportunities.md`.

---

**Next:** [Optimization Opportunities](./optimization_opportunities.md)
