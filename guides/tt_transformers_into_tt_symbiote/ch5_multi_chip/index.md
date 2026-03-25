# Chapter 5 — Multi-Chip Parallelism and Collective Communication

## Overview

This chapter documents how TT Transformers implements tensor parallelism across Tenstorrent multi-chip topologies, what distributed primitives TT Symbiote already provides, and the integration strategy for bringing the two together.

---

## Supported Topologies

Tenstorrent multi-chip systems come in three production configurations. The distinction matters because the collective communication primitives take different code paths for each.

| Topology | Chip count | Mesh shape | Ethernet links (axis 0 / axis 1) | TT Transformers identifier |
|----------|-----------|------------|----------------------------------|---------------------------|
| N300     | 2          | 1 x 2      | 1 / 1                            | `"N300"` in `determine_device_name` |
| T3K      | 8          | 1 x 8      | 1 / 1                            | `"T3K"` |
| TG / Galaxy | 32      | 4 x 8      | 4 / 3                            | `"TG"` or `"BHGLX"` |

Source: `ccl.py` `link_dict` dictionary, lines 34-46.

### N300 (2-chip)

Two chips connected by a single Ethernet link on each axis. The mesh is 1x2, so the only collective axis with more than one device is axis 1. `tt_all_reduce` on a 1xN mesh reaches into the `1 in list(mesh_device.shape)` branch and uses `reduce_scatter_minimal_async` directly (no subsequent all-gather, because the scatter already delivers a partial result to each device; a second pass with `all_gather_async` would be required for full all-reduce semantics — see `tt_transformers_parallelism.md` for the complete call sequence).

### T3K (8-chip)

Eight chips in a linear 1x8 mesh. Same code path as N300 for tensor parallelism — the "1 in shape" branch applies — but with eight participants. `get_num_links` returns 1 link for both axes.

### TG / Galaxy (32-chip)

Thirty-two chips arranged as a 4x8 2-D mesh. The Galaxy path (`args.is_galaxy == True`) takes qualitatively different branches:

- Weight sharding dimensions are swapped relative to T3K (e.g. `w1_dims = (-1, -2)` for Galaxy vs `(-2, -1)` for T3K — `mlp.py` lines 79-80).
- `tt_all_reduce` uses `all_gather_async` followed by `fast_reduce_nc` rather than `reduce_scatter_minimal_async`.
- The MLP applies an intermediate reduce-scatter along `cluster_axis=1` (the 8-chip axis) before the `w2` matmul, then an all-gather along the same axis after (see `mlp.py` lines 181-266).
- Attention uses a `slice_mat` matmul to select the per-device group's batch slice after QKV all-reduce (see `attention.py` lines 64-88, 526-533).
- Ethernet link counts differ: 4 links on axis 0 (row axis) and 3 links on axis 1 (column axis).

---

## Chapter Navigation

| File | Contents |
|------|----------|
| [tt_transformers_parallelism.md](tt_transformers_parallelism.md) | TT_CCL, tt_all_gather, tt_all_reduce, weight sharding, MLP/attention CCL patterns |
| [symbiote_distributed_primitives.md](symbiote_distributed_primitives.md) | Symbiote's distributed linears, distributed RMSNorm, distributed RoPE, config dataclasses |
| [integration_strategy.md](integration_strategy.md) | What maps directly, what must be adapted, what is missing |

---

**Previous chapter:** [Chapter 4 — MLP and Normalization](../ch4_mlp_and_norms/)
**Next chapter:** *(pending)*
