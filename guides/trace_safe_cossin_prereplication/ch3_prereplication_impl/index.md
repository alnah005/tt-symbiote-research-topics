# Pre-Allocating Replicated cos/sin Buffers

By the end of this chapter you will know the required shape, dtype, layout, memory config, and mesh mapping for the pre-allocated cos/sin buffers that replace `_ensure_replicated`, and the concrete changes needed to `move_weights_to_device_impl` and `forward` to apply the pre-allocation pattern to cos/sin in `TTNNQwen3FullAttention`.

---

## Quick-Reference: Pre-Allocated cos/sin Buffer Attributes

| Attribute | Value | Rationale |
|---|---|---|
| Shape | `[1, 1, 1, rotary_dim]` (decode; seq_len=1) | `ttnn.copy` requires identical source and destination shapes |
| dtype | `ttnn.bfloat16` | compute dtype; op requirement |
| Layout | `ttnn.TILE_LAYOUT` | trace-safety; op requirement |
| Memory config | `ttnn.DRAM_MEMORY_CONFIG` | persistent buffer; L1 reserved for activations |
| Mesh mapping | `ReplicateTensorToMesh(mesh_device)` | TP replication requirement |

Full derivation for each attribute is in [`downstream_op_constraints.md`](./downstream_op_constraints.md) (shape, layout, dtype, memory config) and [`replicated_mesh_mapping.md`](./replicated_mesh_mapping.md) (mesh mapping).

---

## Chapter 2 Prerequisites (Brief Recap)

Chapter 2 established that any tensor updated at each decode step must be pre-allocated as a stable device buffer before trace capture and updated via `ttnn.copy` inside the traced region — this chapter applies that pattern to cos/sin.

---

## Lifecycle Diagram: `_cos_replicated`

```
Phase 1 — move_weights_to_device_impl  (before any trace capture)
────────────────────────────────────────────────────────────────────
  move_weights_to_device_impl() called once during model init
  │
  └─ self._cos_replicated = ttnn.from_torch(
         torch.zeros(1, 1, 1, rotary_dim, dtype=torch.bfloat16),
         dtype=ttnn.bfloat16,
         layout=ttnn.TILE_LAYOUT,
         device=self.mesh_device,
         mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
         memory_config=ttnn.DRAM_MEMORY_CONFIG,
     )
     │
     └─ stable DRAM address A allocated on every T3K device
        A is valid for the entire decode session

Phase 2 — Capture Run
────────────────────────────────────────────────────────────────────
  ttnn.begin_trace_capture(mesh_device, cq_id=0)
  │
  │  ttnn.copy(cos, self._cos_replicated)
  │    └─ DMA from incoming cos buffer into address A recorded
  │       in command buffer
  │
  │  ttnn.experimental.rotary_embedding(..., cos=self._cos_replicated)
  │    └─ compute kernel dispatch with address A recorded;
  │       kernel reads updated cos values from address A
  │
  ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

Phase 3 — Each Replay  (every subsequent decode step)
────────────────────────────────────────────────────────────────────
  ttnn.execute_trace(mesh_device, trace_id, cq_id=0)
  │
  ├─ recorded DMA re-issued: current step's cos written into A
  │    (ttnn.copy was inside the bracket; replay re-executes it
  │     with the updated incoming cos tensor for this step)
  │
  └─ recorded kernel dispatch reads rotary values from address A
     address A is valid — buffer is still alive, content is current
```

---

## What's Next

Read the files in this chapter in the following order:

1. [`downstream_op_constraints.md`](./downstream_op_constraints.md) — Derives the required shape, layout, dtype, and memory config for the pre-allocated buffer by working backwards from `ttnn.experimental.rotary_embedding`.
2. [`replicated_mesh_mapping.md`](./replicated_mesh_mapping.md) — Explains what replication means on a T3K mesh, why cos/sin must be replicated rather than sharded, and how to verify replication at runtime.
3. [`move_weights_impl_changes.md`](./move_weights_impl_changes.md) — Provides the concrete, annotated code changes to `move_weights_to_device_impl` and `TTNNQwen3FullAttention.forward`.
