# Integration Checklist

This checklist provides a sequenced set of pre-conditions, implementation steps, and post-implementation checks for introducing `_cos_replicated` and `_sin_replicated` into `TTNNQwen3FullAttention`. Complete the pre-conditions before writing any code. Execute the implementation steps in order without skipping. Run all post-implementation checks before opening a review.

---

## Pre-Conditions to Verify Before Beginning the Change

- [ ] Confirm `TTNNQwen3FullAttention` has `move_weights_to_device_impl` or an equivalent pre-capture hook that runs before any trace capture bracket is opened. If no such hook exists, it must be added before the pre-allocation code can be placed correctly.

- [ ] Confirm that `_decode_cur_pos` pre-allocation in `move_weights_to_device_impl` uses `TILE_LAYOUT` and `DRAM_MEMORY_CONFIG` with `ReplicateTensorToMesh`. This is the existing canonical pattern; `_cos_replicated` and `_sin_replicated` must be pre-allocated using the same layout, memory config, and mapper. Using a different config risks introducing hard-to-diagnose layout mismatches at the `ttnn.copy` call site.

- [ ] Confirm the shape of cos/sin tensors in `TTNNQwen3FullAttention.forward` at decode time (seq_len = 1). The expected shape is `[1, 1, 1, rotary_dim]`. If the shape is `[1, 1, rotary_dim]` (three dimensions), ensure the unsqueeze that adds the batch dimension happens outside the trace bracket before any `ttnn.copy` call. See Chapter 5 (`non_tile_aligned_rotary_dim_interaction.md`) for the trace-safety implications of in-trace unsqueeze.

- [ ] Confirm that `ttnn.experimental.rotary_embedding` does not allocate new output buffers that depend on the shape of its cos/sin inputs at trace time. If the kernel allocates based on cos/sin shape, then changing cos/sin from a dynamically produced tensor to a pre-allocated buffer may change the allocation behavior inside the trace bracket. Inspect the kernel signature and any pre-allocated output buffers passed to it.

- [ ] Confirm that the device has been opened with a `trace_region_size` that can accommodate two additional `ttnn.copy` DMA commands per decode step (one for cos, one for sin). Each `ttnn.copy` records a small number of DMA commands in the trace command buffer (one per device in the mesh, so eight on T3K). The additional size required is negligible relative to typical `trace_region_size` values, but it should be confirmed rather than assumed.

---

## Implementation Steps in Order

**Step 1 — Pre-allocate `_cos_replicated` and `_sin_replicated` in `move_weights_to_device_impl`.**

Add the two pre-allocations immediately after the `_decode_cur_pos` pre-allocation. Both buffers use the canonical `ttnn.zeros` pattern from Chapter 5 ([`../ch5_warmup_guard/non_tile_aligned_rotary_dim_interaction.md`](../ch5_warmup_guard/non_tile_aligned_rotary_dim_interaction.md), "Tile-Aligned Case") — same shape, dtype, layout, memory config, and mapper. If the parameters listed here ever diverge from that section, the Chapter 5 version is authoritative.

```python
# In move_weights_to_device_impl, after _decode_cur_pos allocation:
# why: pre-allocate stable device buffers for replicated cos/sin before
#      any trace capture bracket is opened. TILE_LAYOUT + DRAM_MEMORY_CONFIG
#      matches the _decode_cur_pos canonical pattern.
# Canonical source: ch5_warmup_guard/non_tile_aligned_rotary_dim_interaction.md,
# "Tile-Aligned Case".
self._cos_replicated = ttnn.zeros(
    shape=[1, 1, 1, self.rotary_dim],
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=self.mesh_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
)
self._sin_replicated = ttnn.zeros(
    shape=[1, 1, 1, self.rotary_dim],  # identical parameters to _cos_replicated above
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=self.mesh_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
)
```

Also add `self._cos_replicated = None` and `self._sin_replicated = None` to `__init__` so the attributes exist even if `move_weights_to_device_impl` has not been called.

**Step 2 — Add a debug assertion in `move_weights_to_device_impl` to verify replication.**

Immediately after the pre-allocations, assert that every per-device tensor has the full `rotary_dim` columns:

```python
# why: catch a misconfigured mapper (e.g., ShardTensorToMesh instead of
#      ReplicateTensorToMesh) at device-load time, before any forward pass.
for name, buf in [("_cos_replicated", self._cos_replicated),
                  ("_sin_replicated", self._sin_replicated)]:
    for i, t in enumerate(ttnn.get_device_tensors(buf)):
        assert t.shape[-1] == self.rotary_dim, (
            f"{name} device {i}: expected {self.rotary_dim} cols, "
            f"got {t.shape[-1]}. Check mesh_mapper."
        )
```

**Step 3 — Replace `_ensure_replicated` calls in `forward` with `ttnn.copy`.**

Inside `TTNNQwen3FullAttention.forward`, replace:

```python
# Before (trace-unsafe):
cos = self._ensure_replicated(cos)
sin = self._ensure_replicated(sin)
```

with:

```python
# After (trace-safe):
# why: ttnn.copy writes into the pre-existing _cos_replicated buffer at its
#      stable device address; no new buffer is allocated; the DMA command
#      recorded in the trace is valid on every replay.
ttnn.copy(cos, self._cos_replicated)
ttnn.copy(sin, self._sin_replicated)
```

**Step 4 — Update all downstream references within `forward`.**

After the `ttnn.copy` calls, replace every use of the local variables `cos` and `sin` with `self._cos_replicated` and `self._sin_replicated`, respectively. This typically means updating the argument passed to `TTNNRotaryPositionEmbedding.forward` and any intermediate variable that holds a reference to the replicated tensor.

```python
# Before:
output = self.rotary_emb(q, k, cos, sin, ...)

# After:
# why: downstream code must consume the stable pre-allocated buffer,
#      not the original incoming cos/sin which may be sharded.
output = self.rotary_emb(q, k, self._cos_replicated, self._sin_replicated, ...)
```

Search the full body of `forward` for any remaining reference to the bare `cos` and `sin` local variables that should be replaced.

**Step 5 — Retain `_ensure_replicated` as a private helper but remove it from the traced call path.**

Do not delete `_ensure_replicated`. It remains useful for:

- Warm-up-only diagnostic calls (guarded by `TTNN_DEBUG`).
- Prefill path, where dynamic replication is still acceptable (see `prefill_scope_note.md`).

If `_ensure_replicated` is currently called unconditionally for both decode and prefill, introduce a branch:

```python
# Separate the decode and prefill paths:
# why: prefill uses a dynamically produced replicated tensor (trace-unsafe but
#      prefill is not yet traced); decode uses the pre-allocated buffer (trace-safe).
if is_decode:
    ttnn.copy(cos, self._cos_replicated)
    ttnn.copy(sin, self._sin_replicated)
    cos_out = self._cos_replicated
    sin_out = self._sin_replicated
else:
    cos_out = self._ensure_replicated(cos)
    sin_out = self._ensure_replicated(sin)
```

---

## Post-Implementation Checks

- [ ] Run the existing non-traced forward pass test. Confirm PCC > 0.999 versus the pre-change baseline. A regression here indicates that the `ttnn.copy` target or the downstream variable update was incorrect.

- [ ] Run the warm-up guard test by directly calling `TTNNRotaryPositionEmbedding.forward` with a sharded cos tensor (shape `[1, 1, 1, rotary_dim / 8]` per device) as described in Test 4 of `test_plan.md`. Do not attempt to trigger the guard via `TTNNQwen3FullAttention.forward`: after the change, that path calls `ttnn.copy(cos, self._cos_replicated)` before passing `self._cos_replicated` downstream. `ttnn.copy` writes data into `self._cos_replicated` without changing its memory layout or mapper; `self._cos_replicated` retains full `rotary_dim` columns regardless of what sharded data was written. The guard will therefore never fire via the outer `forward` path. Do not use a sharded cos passed to `TTNNQwen3FullAttention.forward` as a guard test vector: the per-device shard has `rotary_dim / 8` columns while `self._cos_replicated` has `rotary_dim` columns — shape-incompatible for `ttnn.copy` — so a shape mismatch error will be raised at the `ttnn.copy` call site, not at the guard. All guard testing must be done by calling `TTNNRotaryPositionEmbedding.forward` directly with a sharded tensor, as described in Test 4 of `test_plan.md`.

- [ ] Run two consecutive traced decode steps and compare each step's output to the corresponding non-traced reference output. Confirm PCC > 0.999 for both steps. This is the minimum test to confirm that `ttnn.copy` inside the trace correctly updates the cos/sin buffer between replay calls.
