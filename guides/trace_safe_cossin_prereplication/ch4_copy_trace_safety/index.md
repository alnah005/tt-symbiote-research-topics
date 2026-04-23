# Trace Safety of `ttnn.copy` to a Replicated Destination

This chapter answers one concrete question: is `ttnn.copy` from a replicated source tensor to a pre-allocated replicated destination tensor trace-safe inside a Metal Trace capture bracket? The answer is YES. `ttnn.copy` records a DMA transfer command that references the source and destination device buffer addresses baked at capture time. Because no new buffer is allocated, those addresses are valid and identical on every replay iteration. The command can be re-issued verbatim by the trace engine without any Python re-execution or buffer reallocation.

---

## Prerequisite: Chapter 1 — Host Ops vs. Device DMA Ops

Chapter 1 established the foundational distinction: Metal Trace records device-side commands (kernel dispatches, DMA transfers, semaphore operations) with concrete device addresses baked in. On replay, those commands are re-issued verbatim — no Python code runs, no buffers are reallocated. This means any operation that allocates a new device buffer inside the trace bracket invalidates the trace, because the new buffer's address was not known at capture time. Operations that merely write into a pre-existing device buffer at a known address are safe; they produce the same DMA command with the same addresses on every replay.

---

## Prerequisite: Chapter 3 — Pre-Allocation of `_cos_replicated`

Chapter 3 established the concrete implementation pattern. `_cos_replicated` (and `_sin_replicated`) are pre-allocated in `move_weights_to_device_impl` before any trace capture begins. They are initialized with zeros and replicated across every device in the mesh using `ReplicateTensorToMesh`. At the top of the traced `forward`, `ttnn.copy(cos, self._cos_replicated)` writes the current decode step's cos values into this pre-existing buffer. Because the buffer already existed with a stable device address before `begin_trace_capture` was called, the DMA command recorded in the trace is valid on every replay.

---

## What's Next

Read the following files in order:

1. [`what_copy_records.md`](what_copy_records.md) — What DMA command `ttnn.copy` enqueues and why it does not allocate a new device buffer.
2. [`source_tensor_stability.md`](source_tensor_stability.md) — Where cos/sin come from at decode time and whether that source is trace-safe as the input to `ttnn.copy`.
3. [`replay_correctness_verification.md`](replay_correctness_verification.md) — How to verify that the copy correctly updates cos/sin values across consecutive replay steps and detect the stale-value failure mode.
