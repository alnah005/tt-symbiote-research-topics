# T3K Mesh Topology and CCL Bandwidth Budget

> **Source note:** T3K = 4 × Wormhole n300 cards; each n300 card = 2 Wormhole chips; 8 chips total in a 1×8 linear mesh.

## Physical Configuration

A T3K system assembles four Wormhole n300 dual-chip cards into a single chassis:

| Component | Count | Notes |
|---|---|---|
| n300 cards | 4 | Each card = 2 Wormhole chips on one PCB |
| Wormhole chips | 8 | Unit of TP sharding; half of one n300 |
| Tensix cores per chip | 64 | 128 per n300, split evenly |
| DRAM per chip | 12 GB GDDR6 | 6 controllers × 2 GB |
| DRAM bandwidth per chip | 288 GB/s | Half of n300's 576 GB/s |
| BF16 compute per chip | 131 TFLOPS | Half of n300's 262 TFLOPS |

The 8 chips are arranged in a **1×8 linear mesh** for CCL operations. TTNN uses this topology for `ttnn.experimental.all_gather_async`, `ttnn.reduce_scatter`, and `ttnn.all_reduce`.

## Inter-Chip Connectivity

| Link type | Bandwidth | Latency |
|---|---|---|
| Die-to-die (same n300 card) | 200 Gbps = 25 GB/s | ~1 µs |
| QSFP-DD Ethernet (inter-card) | 200 Gbps = 25 GB/s per port | ~2–5 µs |
| Effective all-gather BW (8 devices, ring) | ~25 GB/s | — |

**Why the effective bandwidth equals the link rate:** In a ring all-gather over 8 devices, each device participates in 7 sequential transfer steps. The bottleneck is the slowest link traversed. With homogeneous 25 GB/s links, the effective bandwidth per device is approximately 25 GB/s (empirical; host overhead and routing reduce peak somewhat).

The DRAM-to-CCL bandwidth ratio for T3K:

```
DRAM bandwidth per chip   = 288 GB/s
Effective CCL bandwidth   ≈  25 GB/s
Ratio                     = 288 / 25 ≈ 11.5×
```

Inter-chip communication is 11.5× slower than DRAM access. Any sharding strategy that places CCL on the critical path of the recurrent step will degrade decode performance substantially.

## Per-Chip Memory Budget

Qwen3.5-35B-A3B in BF16: approximately 70 GB total parameters (including all MoE expert weights). Across 8 chips:

```
Model weights per chip ≈ 70 GB / 8 = 8.75 GB
DRAM per chip          = 12 GB
Remaining per chip     = 12 − 8.75 = 3.25 GB
```

The 3.25 GB remainder must accommodate:
- Gated Delta Net recurrent states (per active layer)
- Gated Delta Net conv states (per active layer)
- Gated Attention KV caches (for 10 attention layers)
- Activation buffers (layer input/output tensors)

As shown in the following sections, the recurrent states and conv states are small (a few MB total). The Gated Attention KV cache at long context is the budget-limiting factor.

## Available CCL Operations

The following TTNN CCL primitives are available on T3K:

| Operation | TTNN API | Use in hybrid model |
|---|---|---|
| All-gather | `ttnn.experimental.all_gather_async` | Restore replicated hidden state after col-sharded projection |
| Reduce-scatter | `ttnn.reduce_scatter` | Alternative to all-gather for row-parallel projection output |
| All-reduce | `ttnn.all_reduce` | Full sum-then-broadcast; higher cost than all-gather for TP |

For the decode-step sharding strategy, only all-gather is required, and only once per layer (after the output projection).

---

**Next:** [`head_parallel_state_sharding.md`](./head_parallel_state_sharding.md)
