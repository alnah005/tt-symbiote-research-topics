# T3K Submesh and Env Vars

## Overview

T3K is a 1×8 mesh of Wormhole N300 devices connected through the Galaxy interconnect. Because Galaxy requires the host to claim all 8 devices before sub-allocating them, dots.ocr always opens the full parent mesh first and then carves out a logical 1×2 or 1×1 submesh to satisfy the TP≤2 constraint derived in [gqa_tp_constraint.md](gqa_tp_constraint.md). This file describes the physical topology, the submesh lifecycle, all relevant environment variables, and the scheduling implications for shared-server deployments.

### T3K Physical Topology

T3K is composed of 8 Wormhole N300 devices arranged as a 1×8 mesh. The N300 is a dual-chip card; each card contributes one device to the mesh. All 8 devices are linked by the Galaxy interconnect, which provides the high-bandwidth, low-latency fabric used for tensor-parallel collective operations (all-reduce, all-gather).

The physical topology is fixed: there is no way to isolate a subset of devices from the Galaxy fabric without claiming the full mesh from the host OS.

### Submesh Approach

The entry point for mesh management is `open_dots_mesh_device()` in `tt/mesh.py`. The function follows a two-phase open protocol:

1. **Open the full 1×8 parent mesh.** When `DOTS_T3K_OPEN_FULL_MESH=1` (the default), the function calls the TTNN mesh device API to claim all 8 devices. This gives the host exclusive control of the entire Galaxy fabric.

2. **Carve a logical submesh.** Based on the value of `DOTS_T3K_TP`, the function calls `create_submesh` to allocate a contiguous sub-device group from the parent:
   - `DOTS_T3K_TP=2` → `create_submesh(shape=(1, 2))` → a 1×2 submesh (2 devices)
   - `DOTS_T3K_TP=1` → `create_submesh(shape=(1, 1))` → a 1×1 submesh (1 device)

3. **Return the submesh handle.** The model receives the submesh device handle, not the parent mesh handle. All TTNN tensor operations, KV cache allocations, and collective ops run through the submesh. The remaining 6 devices (at TP=2) or 7 devices (at TP=1) are held by the parent mesh but receive no work.

Why open the full mesh first? The Galaxy interconnect requires the host to register all 8 devices as a single logical unit before any sub-allocation is possible. A partial open — claiming only 2 of the 8 N300 cards — leaves the remaining 6 in an undefined ownership state that can cause initialization failures for any subsequent process attempting to claim them. Opening the full mesh atomically prevents this race condition.

### Env Var Reference

| Env var | Default | Effect |
|---------|---------|--------|
| `DOTS_T3K_OPEN_FULL_MESH` | `1` | When `1`, opens the full 1×8 parent mesh before creating a submesh. Set to `0` only in single-device development environments where T3K Galaxy fabric is absent. |
| `DOTS_T3K_TP` | `2` | Submesh width. `1` creates a 1×1 submesh; `2` creates a 1×2 submesh. Values above 2 violate the GQA constraint and will cause a shape assertion failure at model initialization. |
| `DOTS_LM_HEAD_MAX_COLUMNS_PER_DEVICE` | `2048` | Maximum number of vocabulary columns handled per device per TTNN op in the LM head. Prevents L1 circular buffer overflow at `vocab_size=151936`. |
| `DOTS_MAX_SEQ_LEN` | (none) | Absolute maximum sequence length. Inputs longer than this value are truncated or rejected before the forward pass. |
| `DOTS_MAX_SEQ_LEN_WH_LB` | (none) | Minimum chunk size for the chunked prefill loop. Prevents degenerate single-token chunks on Wormhole devices where kernel launch overhead dominates at very small tile sizes. |

> **Warning:** Values above 2 cause an immediate shape assertion failure at model initialization. See [gqa_tp_constraint.md](gqa_tp_constraint.md).

### Mesh Teardown

`close_dots_mesh_device()` reverses the two-phase open in strict order:

1. Release the submesh first by destroying the submesh device handle.
2. Release the parent mesh after the submesh is fully torn down.

This ordering is required because the submesh holds references into the parent mesh's device registry. Releasing the parent mesh while the submesh handle is still live leaves dangling device references and can corrupt the Galaxy fabric state for other processes. Any wrapper code that calls `close_dots_mesh_device()` must not retain the submesh handle after the call returns.

### Scheduling Implications

At TP=2, dots.ocr holds all 8 T3K devices for the duration of its session because the parent mesh was opened unconditionally. The 6 devices not assigned to the submesh are idle — they perform no computation and receive no tensors — but they are registered to the dots.ocr process and are unavailable to any other model server running on the same machine.

This has a concrete implication for tt_symbiote deployments on shared T3K servers:

- A Llama-3 70B server configured for TP=8 requires exclusive access to all 8 devices. It cannot start while dots.ocr holds the parent mesh.
- Two separate dots.ocr instances cannot co-run on the same T3K, even though each uses only 2 of 8 devices, because the first instance's parent mesh claim blocks the second's.
- Scheduling must treat dots.ocr as an 8-device workload for resource accounting purposes, regardless of `DOTS_T3K_TP`.

> **Note:** This constraint is a consequence of the Galaxy interconnect's ownership model, not a software limitation of tt_symbiote. Changing it would require modifications at the TTNN mesh device layer.

### LM Head Memory Budget

The LM head projects from `hidden_size=1536` to `vocab_size=151936`. This is a large matrix and, without chunking, would require a circular buffer allocation proportional to the full vocabulary dimension on each device.

`DOTS_LM_HEAD_MAX_COLUMNS_PER_DEVICE=2048` caps the column width of each TTNN matrix multiply op in the LM head. At each TP level:

**At TP=1:** the full 151,936 vocabulary columns land on one device.

```
ceil(151936 / 2048) = 75 ops per device per decode step
```

**At TP=2:** the vocabulary is split across 2 devices.

```
columns per device = ceil(151936 / 2) = 75968
ops per device     = ceil(75968 / 2048) = 38 ops per device per decode step
```

Without this cap, a single TTNN op would need to materialize the full 75,968-column output slice in L1 simultaneously. The 2048-column budget keeps each op's output activation within the L1 SRAM capacity of a Wormhole N300 core grid, trading a modest increase in kernel launch count (38 launches at TP=2) for reliable execution at all sequence lengths.

**Next:** [Chunked Prefill](chunked_prefill.md)
