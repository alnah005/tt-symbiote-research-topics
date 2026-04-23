# The Persistent Output Buffer Contract

This file explains the contract that any op inside a Metal Trace must satisfy regarding the address of its output buffer. It shows how `ttnn.experimental.all_gather_async` with `persistent_output_buffer=None` satisfies this contract through the program cache mechanism, contrasts this with an op that breaks the contract, and explains what `_maybe_all_gather` must do to satisfy it. By the end you will understand why the `persistent_output_buffer` argument exists, what "address stability" means in the context of trace replay, and what investigation is needed for the all_gather variant currently used in `_maybe_all_gather`.

---

## The Contract

When `ttnn.execute_trace` replays a captured command buffer, it re-issues the exact sequence of device commands recorded during capture. Each command contains baked-in device addresses: the addresses of its input tensors and of its output tensor, as they existed at capture time.

**The contract:** The output tensor of every op inside a trace must be allocated at the same device address on every replay. If the address changes between replays, the baked-in address in the trace command becomes a dangling pointer — it points to either an invalid memory region or a buffer now owned by a different op.

This is not a concern for persistent tensors such as weight matrices, KV caches, or tensors pre-allocated before the trace lifecycle begins — those are pinned at a fixed address throughout. The concern is for intermediate tensors that are the *output* of ops inside the trace: their addresses must also be stable across replays.

---

## How `all_gather_async` with `persistent_output_buffer=None` Satisfies the Contract

Passing `persistent_output_buffer=None` does not mean no output buffer exists. It means the caller is delegating buffer address management to the op's program cache mechanism. The behavior across the three phases of the trace lifecycle is:

**Phase 1 — Compile run (not recorded):**

```python
output = ttnn.experimental.all_gather_async(
    input_tensor,
    persistent_output_buffer=None,   # why: no pre-allocated buffer provided;
                                      #      runtime will allocate one
    dim=3,
    multi_device_global_semaphore=...,
    ...
)
# Runtime allocates a new output buffer at device address 0xABCD_0000 (example).
# The program cache entry for this (op, input shape, memory config) tuple records:
#   output_address = 0xABCD_0000
# The op completes; the output tensor is available at 0xABCD_0000.
```

**Phase 2 — Capture run (recorded into trace):**

```python
output = ttnn.experimental.all_gather_async(
    input_tensor,
    persistent_output_buffer=None,
    ...
)
# Program cache HIT: this (op, input shape, memory config) tuple was seen in Phase 1.
# The cached entry provides output_address = 0xABCD_0000.
# The override_runtime_arguments mechanism writes 0xABCD_0000 into the device command.
# The trace command buffer records: "all_gather_async → output at 0xABCD_0000"
```

**Phase 3 — execute_trace replay:**

```python
ttnn.execute_trace(device, trace_id, cq_id=0)
# The trace replays: "all_gather_async → output at 0xABCD_0000"
# The buffer at 0xABCD_0000 was allocated in Phase 1 and never freed.
# The address is valid; the op writes its output there correctly.
# Every subsequent replay also finds the buffer at 0xABCD_0000.
```

The buffer allocated in Phase 1 is kept alive by the TTNN program cache for the lifetime of the model instance (not freed between calls as long as the program cache entry is valid). The program cache ensures the same address is used on every call after the first. This is what makes `persistent_output_buffer=None` trace-safe.

> **Note:** The `persistent_output_buffer` parameter also supports an alternative usage: the caller can pass an explicitly pre-allocated tensor (e.g., `persistent_output_buffer=my_preallocated_tensor`). In that case, the op writes its output directly to the caller-provided buffer's address on every call. This is an equally valid approach and may be preferable when the caller needs precise control over the output buffer's memory configuration. The `None` form (program-cache-driven) and the explicit-buffer form are both trace-safe; the `None` form is the pattern used throughout `models/tt_transformers/tt/ccl.py` and `models/tt_transformers/tt/attention.py`.

---

## Contrast: An Op That Breaks the Contract

Consider a hypothetical op that allocates a fresh output tensor on every call with no program caching:

```python
# Hypothetical trace-unsafe pattern

def my_gather(input_tensor):
    output = ttnn.Tensor(shape=..., dtype=..., layout=..., device=device)
    # ^^^ allocates a NEW buffer at a NEW address on every call
    # ... perform gather into output ...
    return output

# Phase 1 (compile run):  output allocated at 0xABCD_0000
# Phase 2 (capture run):  output allocated at 0xDEAD_0000  ← DIFFERENT address
#   → trace records: "my_gather → output at 0xDEAD_0000"
# Phase 3 (replay 1):     output allocated at 0xCAFE_0000  ← DIFFERENT again
#   → trace attempts to write to 0xDEAD_0000, but my_gather put its output at 0xCAFE_0000
#   → the downstream op reads 0xDEAD_0000, which may be uninitialized or belong to another tensor
#   → silent data corruption or segfault on device
```

This is the trace-safety requirement described in Chapter 1: if any op inside a trace allocates a new output buffer on each call without providing address stability, the trace's baked-in address becomes invalid on every replay after the capture.

---

## What `_maybe_all_gather` Must Do

To satisfy the persistent output buffer contract, `_maybe_all_gather` must use an op that provides address stability. There are two paths:

**Path A — Use `ttnn.experimental.all_gather_async` with `persistent_output_buffer=None`:**

The program cache mechanism described above provides automatic address stability. This is the recommended path: it is the same mechanism used by all async CCL ops in `models/tt_transformers/tt/attention.py` and `models/tt_transformers/tt/ccl.py`. No explicit buffer pre-allocation is needed.

**Path B — Use any all_gather variant with an explicit pre-allocated `persistent_output_buffer`:**

The caller allocates a tensor in `__init__` (before the trace lifecycle begins, at a stable device address) and passes it to the all_gather op on every call. The op writes its output to that fixed address. This works but requires the caller to manage the buffer's lifecycle and memory configuration.

> **Warning:** If `_maybe_all_gather` currently calls the synchronous `ttnn.all_gather` (not `all_gather_async`), it is necessary to determine whether `ttnn.all_gather` satisfies the persistent output buffer contract through its own program caching. If it does not — that is, if each call allocates a new output buffer with no program-cache address stability — then switching to `all_gather_async` with `persistent_output_buffer=None` is required for trace safety, independent of the `synchronize_device` removal. Both changes (removing `synchronize_device` and switching to `all_gather_async`) may be necessary together.

---

## Investigation Required for `_maybe_all_gather`

Before implementing the async CCL pattern, the following question must be answered:

**Does `_maybe_all_gather` currently call `ttnn.all_gather` (synchronous) or `ttnn.experimental.all_gather_async`?**

- If it calls `ttnn.all_gather`: the persistent output buffer contract must be verified for that op. The synchronous form may or may not use program caching for output address stability. This is investigated in [`../ch3_root_cause_analysis/what_all_gather_variant_is_used.md`](../ch3_root_cause_analysis/what_all_gather_variant_is_used.md).
- If it already calls `ttnn.experimental.all_gather_async` but without cycling semaphores: the contract is likely satisfied for the output buffer, but the semaphore cycling (and removal of `synchronize_device`) is still required.

> **Key finding:** The persistent output buffer contract is a separate concern from the `synchronize_device` removal, but both must be satisfied for `_maybe_all_gather` to become fully trace-compatible. The `all_gather_async` + `persistent_output_buffer=None` pattern, as used in `models/tt_transformers/tt/attention.py`, satisfies both simultaneously: it is async (no host blocking), uses cycling semaphores (no aliasing), and relies on program caching for output address stability.
