# Chapter 6 — Implementation Plan: Removing synchronize_device and Adopting async CCL

This chapter provides the concrete, step-by-step code changes needed to remove `ttnn.synchronize_device()` from `_maybe_all_gather` and — depending on dispatch intent — replace the underlying `ttnn.all_gather` with `ttnn.experimental.all_gather_async` using cycling semaphores from `TT_CCL`. By the end of this chapter you will have a complete implementation checklist covering both the structural changes to the attention modules and the trace capture wrapper changes required for correct semaphore index management during trace replay.

---

## Prerequisites

Before implementing the changes in this chapter, ensure you have read and understood:

- [Chapter 2 — The Async CCL Pattern](../ch2_async_ccl_pattern/index.md): the cycling semaphore mechanics (`TT_CCL`, `get_and_cycle_ag_semaphore_handles`, `get_and_cycle_barrier_semaphore_handle`), the `all_gather_async` argument list, and the persistent output buffer contract.
- [Chapter 3 — Root Cause Analysis](../ch3_root_cause_analysis/index.md): which `all_gather` variant `_maybe_all_gather` currently calls, why CQ0 FIFO ordering makes `synchronize_device` unnecessary, and the definitive removability verdict.
- [Chapter 4 — Symbiote-Wide Audit](../ch4_symbiote_audit/index.md): which modules are affected and whether `_maybe_all_gather` is a shared base-class method (single fix) or duplicated across two classes (two fixes).

---

## Two-Path Decision

Before writing any code, confirm dispatch intent by checking the call sites identified in [Chapter 4](../ch4_symbiote_audit/audit_results.md):

**Type A — Delete `synchronize_device` only (synchronous all_gather, CQ0 ordering suffices):**
- Choose this path if: `_maybe_all_gather` currently calls synchronous `ttnn.all_gather` AND the intent is to keep it synchronous (no latency optimization objective for this PR, only trace enablement).
- What changes: one-line deletion of `ttnn.synchronize_device(self.mesh_device)`.
- What does not change: the `ttnn.all_gather` call, module constructors, `TT_CCL` wiring, or trace capture wrapper.
- Risk: low — CQ0 FIFO ordering already guarantees the output is valid before the next enqueued op reads it.

**Type B2 — Replace `ttnn.all_gather` with `ttnn.experimental.all_gather_async` + cycling semaphores AND delete `synchronize_device`:**
- Choose this path if: async dispatch is the intent (latency optimization or parity with the `models/tt_transformers/tt/attention.py` pattern), or if the all_gather will be inside a trace bracket and a persistent output buffer contract must be satisfied.
- What changes: module constructors (add `TT_CCL` parameter), `_maybe_all_gather` implementation (replace `ttnn.all_gather` with `ttnn.experimental.all_gather_async`, delete `ttnn.synchronize_device`), and the trace capture wrapper (add semaphore index snapshot/restore).
- Risk: moderate — requires correct semaphore initialization, cycling, and reset-before-replay; validate with the tests described in [Chapter 7](../ch7_validation/index.md).

---

## Before/After Overview

**Before (current implementation):**

```python
def _maybe_all_gather(self, x, cluster_axis, ...):
    if self.num_devices > 1:
        x = ttnn.all_gather(          # synchronous; enqueued to CQ0
            x,
            dim=...,
            num_links=self.num_links,
            cluster_axis=cluster_axis,
            memory_config=self.all_gather_memory_config,
        )
        ttnn.synchronize_device(self.mesh_device)  # <-- trace-blocking; unnecessary
    return x
```

**After (Type A — one-line deletion):**

```python
def _maybe_all_gather(self, x, cluster_axis, ...):
    if self.num_devices > 1:
        x = ttnn.all_gather(          # synchronous; CQ0 ordering suffices
            x,
            dim=...,
            num_links=self.num_links,
            cluster_axis=cluster_axis,
            memory_config=self.all_gather_memory_config,
        )
        # synchronize_device removed — CQ0 FIFO ordering guarantees output validity
    return x
```

**After (Type B2 — async CCL with cycling semaphores):**

```python
def _maybe_all_gather(self, x, cluster_axis, ...):
    if self.num_devices > 1:
        x = ttnn.experimental.all_gather_async(
            x,
            dim=...,
            persistent_output_buffer=None,      # program cache provides buffer stability
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(
                cluster_axis                    # cycling semaphore handle slot 0 or 1
            ),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(
                cluster_axis                    # barrier semaphore for completion signal
            ),
            num_links=self.num_links,
            topology=self.ccl_topology,
            memory_config=self.all_gather_memory_config,
        )
        # synchronize_device removed — completion signaled via GlobalSemaphore to device
    return x
```

---

## What's Next

Read the following files in order:

1. [`structural_changes.md`](./structural_changes.md) — Detailed description of each code change: `TT_CCL` wiring in module constructors, `_maybe_all_gather` signature and body changes for both Type A and Type B2, and the one-line `synchronize_device` deletion.

2. [`trace_capture_wrapper_changes.md`](./trace_capture_wrapper_changes.md) — Changes to the `TracedRun` capture and replay logic required when Type B2 is adopted: semaphore index snapshot before capture, `reset_global_semaphore_value` calls before each replay, and the numbered checklist for code review.
