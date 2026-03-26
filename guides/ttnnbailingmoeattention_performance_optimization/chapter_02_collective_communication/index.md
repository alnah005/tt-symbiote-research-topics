# Chapter 2 — Collective Communication Costs and Sharding Strategy

## Tensor Parallelism on T3K

The Tenstorrent T3K hosts eight Wormhole devices arranged in a **1×8 mesh**. In the tt-symbiote stack this is represented as a `MeshDevice` with shape `(1, 8)`. The axis that matters for tensor parallelism is `cluster_axis=1` — the axis spanning all eight devices in the single row. Operations such as `ttnn.all_gather` and `ttnn.experimental.reduce_scatter_minimal_async` that carry a `cluster_axis` argument target this axis.

Each adjacent pair of Wormhole devices is connected by a high-speed Ethernet link with a peak bidirectional bandwidth of approximately **100 GB/s** per direction per link at the hardware level; effective application-level bandwidth is somewhat lower due to protocol overhead. With `num_links=1`, each collective uses one of these inter-device links.

**Ring topology.** The eight devices are wired in a ring at the physical layer. The `ttnn.Topology.Ring` setting for all-gather allows the CCL runtime to route traffic around the ring in both directions simultaneously, halving the effective distance for every message. `ttnn.Topology.Linear` treats the ring as a chain and routes traffic in one direction only; this is used by `TTNNQwen3FullAttention` (see [all_gather_topology.md](all_gather_topology.md)).

On a ring of N=8 devices, an all-gather of a tensor of total size V bytes requires each device to receive (N−1)/N × V bytes, spread across N−1 steps. Each step moves V/N bytes across one link. The minimum latency is therefore dominated by V/(8 × link_bandwidth) per hop.

## Collective Ops in the Decode Path

`TTNNBailingMoEAttention._forward_decode_paged` (lines 2610–2799 of `attention.py`) executes **five collective operations** per decode step. Four are explicit `ttnn.all_gather` calls; the fifth is an implicit `reduce_scatter` buried inside `q_proj`'s `TTNNLinearIColShardedWRowSharded.forward`.

The table below uses the symbolic notation defined in the guide conventions:

| Step | Op | TTNN call | Direction | Data volume per link | Sync? |
|------|-----|-----------|-----------|----------------------|-------|
| 1 | Reduce-scatter inside Q projection | `ttnn.experimental.reduce_scatter_minimal_async(…, dim=3, cluster_axis=1, topology=Ring)` (inside `TTNNLinearIColShardedWRowSharded.forward`, line 158–172 of `linear.py`) dispatched at line 2624 of `attention.py` | full Q matmul output → col-sharded | B × 1 × H×D → B × 1 × (H×D / N) = B × d\_model / N bytes per device | Async (minimal-async variant) |
| 2 | All-gather hidden states for K/V input | `ttnn.all_gather(hidden_states, dim=-1, num_links=1)` at line 2626 of `attention.py` | col-sharded → replicated | B × 1 × (d\_model / N) bytes per step × (N−1) steps = B × d\_model × (N−1)/N bytes total | Synchronous |
| 3 | All-gather Q after reduce-scatter | `_maybe_all_gather(query_states)` → `ttnn.all_gather(q, dim=-1, num_links=1)` at line 2631 of `attention.py` | col-sharded → replicated | B × 1 × (d\_model / N) per step × (N−1) steps | Synchronous |
| 4 | All-gather K after K projection | `_maybe_all_gather(key_states)` → `ttnn.all_gather(k, dim=-1, num_links=1)` at line 2632 of `attention.py` | col-sharded → replicated | B × 1 × (Hkv×D / N) per step × (N−1) steps | Synchronous |
| 5 | All-gather V after V projection | `_maybe_all_gather(value_states)` → `ttnn.all_gather(v, dim=-1, num_links=1)` at line 2633 of `attention.py` | col-sharded → replicated | B × 1 × (Hkv×D / N) per step × (N−1) steps | Synchronous |

**Total link traffic per decode step** (at S=1):

- Op 1 (reduce-scatter): moves B × d\_model / N × 2 bytes = B × 512 bytes per device received (half the all-gather volume for equal-size tensors)
- Ops 2, 3: each moves B × d\_model × (N−1)/N × 2 bytes = B × 2048 × 7/8 × 2 ≈ B × 3584 bytes per device received
- Ops 4, 5: each moves B × Hkv×D × (N−1)/N × 2 bytes = B × 512 × 7/8 × 2 = B × 896 bytes per device received

At B=32 these resolve to Op 1 ≈ 16 KB, Ops 2–3 ≈ 112 KB each, Ops 4–5 ≈ 28 KB each, for a total of ≈ 296 KB received per device per step (rough estimate; to be validated with profiling per chapter 7 methodology).

Note that Op 3 (Q all-gather) and Op 2 (hidden-states all-gather) move the same volume. This is a key structural inefficiency: by choosing `TTNNLinearIColShardedWRowSharded` for Q projection instead of `TTNNLinearIReplicatedWColSharded`, the implementation adds one full hidden-size all-gather that `TTNNQwen3FullAttention` does not pay. This is analyzed in detail in [sharding_alternatives.md](sharding_alternatives.md).

## Navigation

**Next:** [All-Gather Topology and Detailed Breakdown](all_gather_topology.md)
