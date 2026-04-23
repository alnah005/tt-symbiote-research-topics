# Chapter 2 — The Async CCL Pattern in tt-transformers for Traced Decode

Chapter 1 established that `_maybe_all_gather` calls `ttnn.synchronize_device()`, that this call is a host-blocking barrier incompatible with Metal Trace capture, and that CQ0 FIFO ordering already provides the sequencing guarantee that synchronize_device was meant to supply. This chapter documents the exact pattern used by `models/tt_transformers/tt/attention.py` and `models/tt_transformers/tt/ccl.py` to call `ttnn.experimental.all_gather_async` inside traced decode regions without any `synchronize_device` call anywhere in the path. By the end of this chapter you will understand cycling semaphores, persistent output buffers, and the absence of host-side barriers in the working reference model — everything needed to adapt `_maybe_all_gather` to the same pattern.

---

## Chapter 1 Prerequisites

- Ch1 established that `synchronize_device` is a host-blocking barrier incompatible with Metal Trace capture — see [`synchronize_device_semantics.md`](../ch1_maybe_all_gather_anatomy/synchronize_device_semantics.md) for the full argument.
- CQ0 FIFO ordering already provides the sequencing guarantee that `synchronize_device` was meant to supply, making the call unnecessary in single-CQ dispatch.
- The Finish token IS a CQ0 device command (recorded and replayed); the host-side blocking wait is NOT recorded — so the barrier is silently absent on every replay.

---

## Decode Trace Loop: Where `all_gather_async` Appears

The diagram below shows the three-phase decode trace lifecycle in tt-transformers. Annotations mark where `all_gather_async` is enqueued and explicitly confirm where `synchronize_device` does NOT appear.

```
Phase 1 — Compile Run (not recorded)
┌───────────────────────────────────────────────────────────────────┐
│  model.forward(x, pos, mode="decode")                            │
│    └─ Attention.forward_decode(...)                              │
│         ├─ ttnn.linear(x, wqkv)           ← compute op → CQ0    │
│         ├─ tt_all_reduce(...)                                    │
│         │    └─ all_gather_async(...)      ← CCL op → CQ0       │
│         │       [NO synchronize_device]                          │
│         ├─ nlp_create_qkv_heads_decode()  ← compute op → CQ0    │
│         ├─ rotary_embedding_llama()       ← compute op → CQ0    │
│         ├─ paged_update_cache()           ← compute op → CQ0    │
│         ├─ sdpa_decode()                 ← compute op → CQ0    │
│         ├─ all_gather_async(...)          ← CCL op → CQ0       │
│         │  [NO synchronize_device]                               │
│         └─ ttnn.linear(attn_out, wo)      ← compute op → CQ0    │
│  [output buffers allocated, program cache populated]             │
└───────────────────────────────────────────────────────────────────┘

Phase 2 — Trace Capture (begin → end bracket)
┌───────────────────────────────────────────────────────────────────┐
│  ttnn.begin_trace_capture(device, cq_id=0)                       │
│  model.forward(x, pos, mode="decode")   ← SAME call as above    │
│    └─ [all ops re-enqueued; device commands written to trace buf] │
│       [all_gather_async calls appear in trace]                   │
│       [NO synchronize_device — host never blocks]                │
│  ttnn.end_trace_capture(device, trace_id, cq_id=0)              │
│  [cycling semaphore indices snapshotted for pre-replay reset]    │
└───────────────────────────────────────────────────────────────────┘

Phase 3 — Repeated execute_trace (per decode step)
┌───────────────────────────────────────────────────────────────────┐
│  loop:                                                            │
│    [reset GlobalSemaphore handles to 0]                          │
│    [restore TT_CCL cycling indices to capture-time values]       │
│    ttnn.execute_trace(device, trace_id, cq_id=0)                 │
│    [device replays baked-in command buffer; no host dispatch]    │
│    [all_gather_async baked addresses remain valid each replay]   │
└───────────────────────────────────────────────────────────────────┘
```

Key observation from the diagram:

Before each `execute_trace`, the GlobalSemaphore handles are reset to 0 and the `TT_CCL` cycling indices are restored to their capture-time values. This ensures the same handle addresses that were baked into the trace are selected on every replay.

---

## Learning Objectives

1. Identify the exact `ttnn.experimental.all_gather_async` call signature used in `Attention.forward_decode` at `models/tt_transformers/tt/attention.py`, including every argument and its purpose.
2. Confirm that the `Attention.forward_decode` path contains no `ttnn.synchronize_device()` call and explain why this is safe under CQ0 ordering.
3. Describe the double-buffer design of `TT_CCL`'s semaphore pools and explain how `get_and_cycle_ag_semaphore_handles` and `get_and_cycle_barrier_semaphore_handle` advance the slot index on each call.
4. Explain why cycling semaphores are necessary inside a trace: what aliasing failure they prevent between consecutive iterations.
5. State the persistent output buffer contract for trace-safe ops and explain how `persistent_output_buffer=None` satisfies it via the program cache.

---

## Files in Reading Order

1. [`all_gather_async_in_traced_attention.md`](./all_gather_async_in_traced_attention.md) — The `all_gather_async` call signature from `Attention.forward_decode`, confirmation that no `synchronize_device` is present, and an explanation of why CQ0 ordering makes synchronize_device unnecessary.
2. [`cycling_semaphore_mechanics.md`](./cycling_semaphore_mechanics.md) — The double-buffer initialization in `TT_CCL.__init__`, the `get_and_cycle_*` methods, and why cycling is required inside a trace to prevent semaphore aliasing.
3. [`persistent_output_buffer_contract.md`](./persistent_output_buffer_contract.md) — The contract that trace-safe ops must satisfy regarding output buffer address stability, and how `all_gather_async` with `persistent_output_buffer=None` satisfies it through program caching.
