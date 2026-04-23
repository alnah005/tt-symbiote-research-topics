# Chapter 5 — Warm-Up Guard Preservation

This chapter analyzes whether the `rotary_dim % 64 != 0` guard in `TTNNRotaryPositionEmbedding.forward` continues to function correctly after the pre-replication change introduced in Chapters 3 and 4. The answer is YES. The guard fires during the warm-up compile run, which executes before the trace capture bracket is opened; after the change, the guard inspects the pre-allocated replicated buffer rather than the raw sharded input, and the column count it sees is still correct. The guard remains an effective sentinel against misconfigured replication.

---

## Prerequisite: Chapter 3 — Pre-Allocation Plan

Chapter 3 established that `_cos_replicated` and `_sin_replicated` are pre-allocated before any trace capture begins, using `ReplicateTensorToMesh(mesh_device)`. Each device therefore holds a local buffer with shape `[1, 1, 1, rotary_dim]` — all `rotary_dim` columns present on every device, with no sharding across the column axis.

---

## Prerequisite: Chapter 4 — `ttnn.copy` Trace Safety

Chapter 4 established that `ttnn.copy(cos, self._cos_replicated)` writes the current step's cos values into the pre-existing replicated buffer without allocating a new device buffer. This is the mechanism that keeps the guard's inspected tensor stable: the guard checks the buffer that was pre-allocated before the trace, not a freshly created tensor.

---

## What's Next

Read the following files in order:

1. [`guard_mechanism_analysis.md`](guard_mechanism_analysis.md) — How the `rotary_dim % 64 != 0` guard works and when it fires.
2. [`guard_adequacy_after_change.md`](guard_adequacy_after_change.md) — Whether the guard remains adequate after switching from `_ensure_replicated` to `ttnn.copy` into a pre-allocated buffer.
3. [`non_tile_aligned_rotary_dim_interaction.md`](non_tile_aligned_rotary_dim_interaction.md) — Interaction between the pre-replication change and non-tile-aligned `rotary_dim` values, and why the Qwen3 case is clean.
