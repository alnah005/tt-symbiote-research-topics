# Chapter 2 — The _decode_cur_pos Pre-Allocation Pattern

By the end of this chapter you will understand the complete lifecycle of `_decode_cur_pos` — from allocation in `move_weights_to_device_impl` before any trace capture begins, through per-step update via `ttnn.copy` inside the traced region, to stable read access on every replay. You will also understand why this same three-phase lifecycle is the correct template for cos/sin pre-allocation, and what makes cos/sin structurally different from a scalar position index.

---

## Chapter 1 Prerequisites (Brief Recap)

Chapter 1 established that device buffer addresses must be stable before `ttnn.begin_trace_capture`, that `ttnn.from_torch` is unsafe inside the capture bracket, and that `ttnn.copy` into a pre-allocated buffer is safe — these three facts underpin everything in this chapter.

---

## Lifecycle Diagram: `_decode_cur_pos`

```
Before trace capture
────────────────────────────────────────────────────────────────────
  move_weights_to_device_impl() called once during model init
  │
  └─ self._decode_cur_pos = ttnn.from_torch(
         torch.tensor([0], dtype=torch.int32),
         dtype=ttnn.int32,
         layout=ttnn.ROW_MAJOR_LAYOUT,
         device=self.mesh_device,
         mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
         memory_config=ttnn.DRAM_MEMORY_CONFIG,
     )
     │
     └─ device buffer allocated at address A
        A is stable for the lifetime of the model instance

Phase 1 — Compile Run  (no trace active)
────────────────────────────────────────────────────────────────────
  forward(step=0) called normally.
  ttnn.copy(cur_pos_host, self._decode_cur_pos)  # writes 0 into A
  All ops use self._decode_cur_pos — address A observed by kernels.

Phase 2 — Capture Run
────────────────────────────────────────────────────────────────────
  ttnn.begin_trace_capture(mesh_device, cq_id=0)
  │
  │  ttnn.copy(cur_pos_host, self._decode_cur_pos)
  │    └─ DMA into address A recorded in command buffer
  │
  │  kernel_op(..., cur_pos=self._decode_cur_pos)
  │    └─ kernel dispatch with address A recorded
  │
  ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

Phase 3 — Replay (every decode step)
────────────────────────────────────────────────────────────────────
  Update cur_pos_host with current step integer (host side only).
  ttnn.execute_trace(mesh_device, trace_id, cq_id=0)
  │
  ├─ recorded DMA writes new value into address A  <- ttnn.copy is
  │    (copy was recorded inside the bracket;          inside the
  │     replay re-executes it with current value)      capture bracket
  │
  └─ recorded kernel dispatch reads from address A
     address A is valid — buffer is still alive
```

---

## Learning Objectives

By the end of this chapter you will be able to:

1. Identify the exact `ttnn.from_torch` call in `move_weights_to_device_impl` that allocates `_decode_cur_pos` and explain the dtype, shape, layout, and memory config choices.
2. Locate the `ttnn.copy` call that updates `_decode_cur_pos` at each decode step and explain why it is trace-safe while `ttnn.from_torch` is not.
3. State the three properties that make the `_decode_cur_pos` pattern work and explain why removing any one of them would break trace replay.
4. Extract the generalizable four-step pre-allocation pattern and apply it to any decode-step tensor that changes value between steps.
5. Identify what makes cos/sin structurally different from `_decode_cur_pos` and state the design implications for cos/sin pre-allocation.

---

## Files in Reading Order

1. [`decode_cur_pos_walkthrough.md`](./decode_cur_pos_walkthrough.md) — Annotated walkthrough of the `_decode_cur_pos` allocation code, the per-step update pattern, and the three properties that make it work.
2. [`pattern_generalization.md`](./pattern_generalization.md) — The generalizable four-step pre-allocation pattern extracted from `_decode_cur_pos`, what makes cos/sin different, and the design decision for Chapter 3.
3. [`traced_run_alloc_kwarg_tensor.md`](./traced_run_alloc_kwarg_tensor.md) — Whether `TracedRun._alloc_kwarg_tensor` handles cos/sin keyword arguments, its limitations, and why `move_weights_to_device_impl` is the preferred location for cos/sin pre-allocation.

---

**Next:** [`decode_cur_pos_walkthrough.md`](./decode_cur_pos_walkthrough.md)
