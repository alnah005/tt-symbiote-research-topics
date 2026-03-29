# Chapter 5 — Implementing Trace Support: A Step-by-Step Guide

This chapter synthesizes everything established in Chapters 1–4 into a concrete implementation plan. By the end of this chapter you will understand the full lifecycle of `TT_CCL` semaphore handles across model initialization, the compile run, trace capture, and repeated trace replay; you will know exactly which device-side semaphore values must be reset before each replay and why; you will understand why host-side index fields must be restored to their captured values before each `execute_trace` call; and you will know which files in the tt-transformers codebase require changes to make trace-backed decode loops safe for async CCL operations.

## Prerequisites

This chapter assumes you have read and internalized:

- **Chapter 1** — `GlobalSemaphore` internals, L1 address stability, the double-buffered handle layout, and the `semaphore_index` mapping from `cluster_axis`
- **Chapter 2** — How semaphore addresses travel as RTAs through `override_runtime_arguments`, and why the DRAM command buffer assembled during `end_trace_capture` is immutable after assembly
- **Chapter 3** — The handle-cycling protocol, what gets baked at capture time, and the two failure modes (deadlock and silent corruption)
- **Chapter 4** — The `use_composite=True` four-group reset requirement, the role of `reset_global_semaphore_value` in the CQ FIFO ordering model, and the four trace paths in the codebase that currently lack semaphore management

---

## The Full Lifecycle in One Diagram

The diagram below shows the correct event ordering from model initialization through N trace replays. Read it top to bottom; each indented block is a sub-step of the block above it.

```
┌─────────────────────────────────────────────────────────────────────┐
│  MODEL INITIALIZATION                                               │
│                                                                     │
│    TT_CCL.__init__()                                                │
│      allocate ag_semaphore_handles[si][0..1]  (2 per slot)         │
│      allocate rs_semaphore_handles[si][0..1]  (3 per slot)         │
│      allocate barrier_semaphore_handles[si][0..1] (1 per slot)     │
│      set ag_semaphores_idx    = [0, 0, 0]  (one entry per si)      │
│      set rs_semaphores_idx    = [0, 0, 0]                          │
│      set barrier_semaphore_idx = [0, 0, 0]                         │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  COMPILE RUN  (populates program cache, not traced)                 │
│                                                                     │
│    [optional but strongly recommended]                              │
│    reset all semaphore handles to 0 on device                       │
│                                                                     │
│    call traced ops (e.g. all_gather_async, reduce_scatter, ...)     │
│      get_and_cycle_ag_semaphore_handles(si) → uses slot N, idx→N'  │
│      get_and_cycle_barrier_semaphore_handles(si) → uses M, idx→M'  │
│      (for use_composite=True: RS handles also consumed)             │
│                                                                     │
│    after compile run: indices have advanced                         │
│      ag_semaphores_idx[si]    = N'  (= (N+1) % 2)                 │
│      barrier_semaphore_idx[si] = M'  (= (M+1) % 2)                │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PRE-CAPTURE SETUP                                                  │
│                                                                     │
│    1. snapshot current host indices                                 │
│         captured_ag_idx      = list(tt_ccl.ag_semaphores_idx)      │
│         captured_rs_idx      = list(tt_ccl.rs_semaphores_idx)      │
│         captured_barrier_idx = list(tt_ccl.barrier_semaphore_idx)  │
│                                                                     │
│    2. reset device semaphore values for handles at captured indices │
│         (same reset code that precedes every replay)               │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TRACE CAPTURE                                                      │
│                                                                     │
│    ttnn.begin_trace_capture(mesh_device, cq_id=0, tid=trace_id)    │
│                                                                     │
│    call traced ops                                                  │
│      get_and_cycle_ag_semaphore_handles(si)                         │
│        → returns handles[si][captured_ag_idx[si]]                  │
│        → bakes those L1 addresses into DRAM command buffer          │
│        → advances ag_semaphores_idx[si] to (captured_ag_idx[si]+1) % 2 │
│      (RS and barrier handles advanced similarly)                    │
│                                                                     │
│    ttnn.end_trace_capture(mesh_device, cq_id=0, tid=trace_id)      │
│      assembles DRAM command buffer — now IMMUTABLE                  │
│                                                                     │
│    record capture-time indices (already snapshotted above)         │
│    NOTE: host indices have now advanced past capture-time values    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  DECODE LOOP  (repeated N times)                                    │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  PER-REPLAY SETUP  (before every execute_trace call)         │  │
│  │                                                               │  │
│  │  1. reset device semaphore values for captured handles        │  │
│  │       (enqueued via CQ FIFO before execute_trace)            │  │
│  │                                                               │  │
│  │  2. restore host indices to capture-time values               │  │
│  │       tt_ccl.ag_semaphores_idx      = list(captured_ag_idx)  │  │
│  │       tt_ccl.rs_semaphores_idx      = list(captured_rs_idx)  │  │
│  │       tt_ccl.barrier_semaphore_idx  = list(captured_barrier_idx) │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  ttnn.execute_trace(mesh_device, cq_id=0, tid=trace_id,      │  │
│  │                     blocking=True)                            │  │
│  │    replays DRAM command buffer                                │  │
│  │    kernels use the L1 addresses baked at capture time         │  │
│  │    kernels self-reset their semaphores on completion          │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                   loop back for next token                          │
└─────────────────────────────────────────────────────────────────────┘
```

### Reading the diagram

The key constraint the diagram encodes is:

- The DRAM command buffer assembled during `end_trace_capture` permanently encodes the L1 addresses of the handles that were active at capture time. Those addresses never change.
- Between every two `execute_trace` calls the device-side semaphore counters at those fixed addresses must be back at zero. If the previous replay's kernels have not yet self-reset them (possible when `blocking=False`), or if they were never reset at all, the next replay sees stale values and either deadlocks or skips through.
- The host-side index fields in `TT_CCL` must be restored to capture-time values before each `execute_trace` because `get_and_cycle_*` advanced them during capture. If they are not restored, any non-traced CCL call that runs between replays will consume the wrong handle slot and collide with the trace's baked addresses.

---

## Files Covered in This Chapter

| File | Purpose |
|---|---|
| `code_changes_required.md` | Concrete Python code for index snapshot/restore helpers, device semaphore reset loops for both `use_composite=False` and `use_composite=True`, complete capture wrapper checklist, complete replay wrapper checklist, and guidance on identifying which handles a trace uses |
| `verifying_correctness.md` | Test strategies for numerical comparison, deadlock detection, silent-corruption detection, host-counter verification, device-side reset verification, and a common-mistake checklist |

---

## What's Next

Read the files in this order:

1. **`code_changes_required.md`** — Start here. This file contains all the code snippets and numbered checklists you need to implement trace-compatible async CCL in a tt-transformers model. Work through it sequentially; each section builds on the previous one.

2. **`verifying_correctness.md`** — After implementing the changes, use this file to design your test plan. It covers the numerical comparison approach, how to distinguish deadlock from silent corruption, and how to verify both the host-side index state and the device-side semaphore values at runtime.
