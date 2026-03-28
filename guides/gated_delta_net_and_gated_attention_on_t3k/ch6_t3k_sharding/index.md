# Chapter 6 — T3K Sharding Strategy for Gated Delta Net State

## Overview

This chapter answers **Q7: How should the Gated Delta Net state matrix and associated tensors be sharded across the 8 Wormhole devices of a T3K system?**

The T3K 1×8 mesh topology and per-chip hardware parameters were established in Chapter 5. Here they are used to derive per-device memory and bandwidth figures for two classes of tensors: the recurrent state matrices of the Gated Delta Net layers, and the KV caches of the Gated Attention layers.

The recommended strategy is **head-parallel sharding** for the Gated Delta Net state matrix combined with **tensor-parallel column sharding** for all input and output projections. This design places no CCL operations on the critical path of the recurrent state update: each device computes its share of heads independently, and a single small all-gather of the output projection completes the layer.

## Sections

1. [`t3k_mesh_topology.md`](./t3k_mesh_topology.md) — T3K 1×8 mesh configuration, per-chip memory budget, and CCL bandwidth figures.

2. [`head_parallel_state_sharding.md`](./head_parallel_state_sharding.md) — Head-parallel sharding of the Gated Delta Net state matrix: per-device tensor shapes, memory footprint, and CCL communication pattern for decode and prefill.

3. [`alternative_sharding_strategies.md`](./alternative_sharding_strategies.md) — Analysis of replicated-state and other alternatives; recommendation comparison table.

4. [`kv_cache_sharding_for_gated_attention.md`](./kv_cache_sharding_for_gated_attention.md) — Sharding the Gated Attention KV cache across 8 T3K devices, memory budget implications, and the n_kv_h=2 constraint.

## Key Take-Away

Head-parallel sharding distributes the recurrent state across devices with zero CCL overhead on the state itself. The only collective communication per Gated Delta Net layer is a single all-gather of 4 KB (the final output projection partial sum), which takes approximately 0.16 µs at 25 GB/s — less than 2.5% of the 7.36 µs per-layer state I/O time. This makes T3K scaling of Gated Delta Net essentially overhead-free for decode.

The Gated Attention KV cache is the dominant memory consumer at long context: at T=262,144, all 10 Gated Attention layers together require approximately 5.12 GiB per device (with KV heads replicated), limiting batch size to B=1 at full context on current hardware.

---

**Previous:** [Chapter 5 — Roofline Analysis](../ch5_roofline_analysis/index.md)
**Next:** [Chapter 7 — Kernel Gaps and Development Roadmap](../ch7_kernel_gaps_and_roadmap/index.md)
