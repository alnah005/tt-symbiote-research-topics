# Chapter 1: GlobalSemaphore Internals and the Double-Buffer Design

This chapter establishes the foundational vocabulary and data structures that all subsequent chapters depend on. By the end of this chapter you will understand what a `GlobalSemaphore` object is at the tt-metal level — how it maps to a stable L1 address on each device core, how `ttnn.create_global_semaphore` allocates it, and why the `TT_CCL` class maintains double-buffered semaphore handles with host-side cycling counters to permit pipelining of back-to-back async CCL operations.

---

## Learning objectives

After reading this chapter you will be able to:

- Explain what a `GlobalSemaphore` object is, where it lives in device memory, and why its L1 address is fixed for the lifetime of the object.
- Describe the three handle arrays (`barrier_semaphore_handles`, `ag_semaphore_handles`, `rs_semaphore_handles`) and the three index arrays (`barrier_semaphore_idx`, `ag_semaphores_idx`, `rs_semaphores_idx`) that `TT_CCL.__init__` creates.
- Trace exactly which handle a `get_and_cycle_*` call returns and what side-effect it leaves on the index array.
- State the key invariant that must hold before any CCL invocation: the host counter and the device semaphore value must be consistent.

---

## Relationship diagram

The diagram below shows how the Python-level handle, the underlying L1 buffer, and the device-side semaphore word relate to each other. Read it top-to-bottom as the allocation flows from the Python call site down to the device.

```
Python (host)
─────────────────────────────────────────────────────────────────────────────
  ttnn.create_global_semaphore(mesh_device, core_range_set, initial_value=0)
        │
        │  returns
        ▼
  GlobalSemaphore object
  ┌─────────────────────────────────────────────────────────────────────┐
  │  buffer_  : AnyBuffer (wraps a HEIGHT_SHARDED MeshBuffer in L1)    │
  │  device_  : IDevice* (or MeshDevice*)                              │
  │  cores_   : CoreRangeSet  (every Tensix core on the mesh device)   │
  │                                                                     │
  │  .address() → DeviceAddr  ──────────────────────────────────┐      │
  └──────────────────────────────────────────────────────────── │ ─────┘
                                                                │
        same uint32 address on every core in the CoreRangeSet  │
        (sharded buffer: one 4-byte page per core)             │
                                                                ▼
Device L1 (one Tensix core, repeated for every core in the set)
  ┌─────────────────────────────────────────────────────────────┐
  │  address()  →  [ uint32_t semaphore_word ]                 │
  │                  ↑ written 0 at object creation             │
  │                  ↑ written by async CCL kernel on completion│
  │                  ↑ reset to 0 by reset_global_semaphore_value│
  └─────────────────────────────────────────────────────────────┘

Host-side index arrays in TT_CCL (one slot per cluster-axis variant)
  barrier_semaphore_idx[3]  →  cycling counter, mod 2
  ag_semaphores_idx[3]      →  cycling counter, mod 2
  rs_semaphores_idx[3]      →  cycling counter, mod 2

  array slot 0 = cluster_axis=0
  array slot 1 = cluster_axis=1
  array slot 2 = cluster_axis=None
```

The critical property: `.address()` reads `buffer_.get_buffer()->address()` (see `global_semaphore.cpp` line 79), which is the buffer's allocation address in L1. This address does not change after the object is created — buffer allocation is performed once in `setup_buffer` and the result is stable for the lifetime of the object.

---

## Glossary

| Term | Definition in this guide |
|---|---|
| `GlobalSemaphore` | A tt-metal C++ object (`tt::tt_metal::GlobalSemaphore`) that wraps a HEIGHT_SHARDED L1 `MeshBuffer`; each core in the `CoreRangeSet` holds a 4-byte semaphore word at the same L1 address. |
| handle | A single `GlobalSemaphore` object as returned by `ttnn.create_global_semaphore`; identified uniquely by its L1 address. |
| cycling counter | An integer stored in `barrier_semaphore_idx`, `ag_semaphores_idx`, or `rs_semaphores_idx` at a given array position; it advances modulo 2 on each `get_and_cycle_*` call to alternate between double-buffer slots. |
| double-buffer slot | One of the two alternating `GlobalSemaphore` objects (or lists of objects) that `TT_CCL.__init__` creates for a given cluster-axis variant; indexed 0 or 1 by the host-side cycling counter modulo 2. |

Terms introduced in later chapters (`capture-time handle`, `program cache hit`, `TraceNode`, `RTA`) are defined where they first appear.

---

## Files in this chapter (reading order)

1. **[global_semaphore_api.md](./global_semaphore_api.md)** — How `ttnn.create_global_semaphore` works, what `GlobalSemaphore::address()` returns, what `ttnn.reset_global_semaphore_value` does, and where `TT_CCL.__init__` allocates all the handles.

2. **[double_buffer_design.md](./double_buffer_design.md)** — Why two alternating handles are needed, the three host-side index arrays, the `get_and_cycle_*` methods in full, a 4-call sequence illustration, and the key invariant about host-counter / device-state consistency.

## What's next

After reading this chapter in order, continue to:

**[Chapter 2 — How Semaphore Addresses Flow into Kernel Runtime Arguments](../ch2_semaphore_rta_path/index.md)**
