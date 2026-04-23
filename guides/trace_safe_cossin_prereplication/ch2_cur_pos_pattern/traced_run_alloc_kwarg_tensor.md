# TracedRun._alloc_kwarg_tensor and cos/sin

This file investigates whether `TracedRun._alloc_kwarg_tensor` exists, what it pre-allocates, and whether it can be used to handle `cos` and `sin` keyword argument tensors passed to `TTNNQwen3FullAttention.forward`. By the end you will have a clear answer on the gap, an explanation of why `move_weights_to_device_impl` is preferable regardless, and a forward reference to the Chapter 3 code changes.

> **Note:** The tt-symbiote source repository was not accessible at a local path during the writing of this guide. The analysis below is based on the documented behavior of `TracedRun._capture_trace` as described in related research guides (`ttnn_all_reduce_trace_compatibility`, `tt_transformers_into_tt_symbiote`, `tt_symbiote`), the plan specification for this guide, and the research question posed in `research_topics.md`. A TODO marker is placed at any point where direct source confirmation was not possible.

---

## The Documented Behavior of `TracedRun._capture_trace`

Based on the source reconstruction available in `guides/ttnn_all_reduce_trace_compatibility/ch1_trace_mechanics/buffer_address_stability.md` (lines 26–43), the `_capture_trace` method in `models/experimental/tt_symbiote/core/run_config.py` pre-allocates persistent input buffers using the following pattern:

```python
# From run_config.py — _capture_trace (simplified)
# Source: guides/ttnn_all_reduce_trace_compatibility/ch1_trace_mechanics/
#         buffer_address_stability.md, lines 26–43

mem_config = TracedRun._input_memory_config or ttnn.DRAM_MEMORY_CONFIG

trace_inputs = []
trace_func_args = []

for arg in func_args:          # <-- iterates over POSITIONAL arguments only
    if isinstance(arg, ttnn.Tensor):
        host_tensor = arg.cpu() if arg.storage_type() != ttnn.StorageType.HOST else arg
        trace_input = ttnn.to_device(host_tensor, device, memory_config=mem_config)
        # why: ttnn.to_device allocates a new device buffer at a stable address
        #      before the capture bracket opens; this address is then baked into
        #      the command buffer when forward is called inside begin_trace_capture.
        trace_inputs.append(trace_input)
        trace_func_args.append(trace_input)
    # ... (non-Tensor args passed through unchanged)
```

**Key observation:** the loop iterates over `func_args` — the positional arguments to `module.forward`. There is no corresponding loop over `func_kwargs`.

The related `_copy_inputs_to_trace_buffer` method also operates only over the positional `trace_inputs` list:

```python
# From run_config.py — _copy_inputs_to_trace_buffer (simplified)
# Source: guides/ttnn_all_reduce_trace_compatibility/ch1_trace_mechanics/
#         buffer_address_stability.md, lines 50–58

for arg, trace_input in zip(new_args, entry.trace_inputs):
    if isinstance(arg, ttnn.Tensor):
        ttnn.copy(arg, trace_input)
        # why: writes new data into the stable device buffer;
        #      no reallocation; the address recorded in the command buffer
        #      remains valid.
```

Again, `entry.trace_inputs` contains only the pre-allocated buffers for **positional** arguments.

---

## Does `_alloc_kwarg_tensor` Exist?

> **TODO:** Direct confirmation requires reading `run_config.py` from the tt-symbiote source. The method name `_alloc_kwarg_tensor` was identified as a research question in `research_topics.md` ("Does `TracedRun._alloc_kwarg_tensor` already pre-allocate cos/sin buffers...?"), indicating it may or may not exist.

Based on the documented behavior of `_capture_trace` and `_copy_inputs_to_trace_buffer` (both of which operate only over positional arguments), the current evidence indicates:

> **Key Finding:** `TracedRun._alloc_kwarg_tensor` does not appear in the documented implementation of `TracedRun._capture_trace`. The `_capture_trace` method pre-allocates persistent input buffers only for the module's positional `func_args`, not for keyword arguments (`func_kwargs`). Keyword argument tensors such as `cos` and `sin` — passed as `cos=...` and `sin=...` in a `forward(self, hidden_states, cos=..., sin=...)` signature — are not pre-allocated by `_capture_trace` under the documented implementation.

This finding is consistent with the framing of the research question in `research_topics.md`, which asks whether `_alloc_kwarg_tensor` "already pre-allocates cos/sin buffers" rather than asserting that it does.

---

## What `_capture_trace` Does Handle for Keyword Arguments

While `_capture_trace` does not pre-allocate buffers for kwarg tensors, it does include kwargs in the cache key computation (from `run_modes.md`):

```
_make_cache_key(module_name, func_args, func_kwargs)
```

Both positional and keyword tensor arguments are included in the key. This means a change in the shape, dtype, or layout of a kwarg tensor will produce a different cache key and trigger a new trace capture — but the kwarg tensor's buffer itself is not pre-allocated as a stable persistent buffer. The kwarg tensor is passed through to `module.forward` as-is, with its original (possibly unstable) device buffer address.

For `cos` and `sin` tensors passed as kwargs, this means:

1. If they arrive as freshly computed device tensors (allocated outside the capture bracket and passed in), their addresses are stable for that particular call but may change between decode steps if the caller re-allocates them.
2. If `_ensure_replicated` re-allocates them inside the capture bracket (the current bug), the new buffer's address is baked into the command buffer — but the allocation is not pre-allocated by `TracedRun`, so it is not protected against address changes on replay.

Neither case satisfies the buffer address stability invariant.

---

## The Limitation of `_alloc_kwarg_tensor` for cos/sin

Even if `_alloc_kwarg_tensor` existed and pre-allocated buffers for `cos` and `sin` kwargs, it would face two structural limitations for this use case:

### Limitation 1 — No control over mesh mapping

`_capture_trace` allocates positional-arg buffers using `ttnn.to_device(host_tensor, device, memory_config=mem_config)`. The `memory_config` is a class-level setting (`TracedRun._input_memory_config`). There is no mechanism to specify a `mesh_mapper` for individual kwargs. A generalized `_alloc_kwarg_tensor` method would face the same constraint: to pre-allocate cos/sin as replicated tensors, it would need to use `ReplicateTensorToMesh`, but `_capture_trace`'s allocation path does not expose a per-argument mesh mapper.

Without `ReplicateTensorToMesh`, the pre-allocated cos/sin buffer would be allocated with whatever default placement `ttnn.to_device` uses — which may be sharded or single-device, depending on the global `TracedRun._input_memory_config`. A sharded buffer would re-introduce the original crash that `_ensure_replicated` was designed to fix.

### Limitation 2 — No control over layout

`ttnn.experimental.rotary_embedding` requires `TILE_LAYOUT` for its cos/sin inputs. The `TracedRun._capture_trace` allocation path uses `memory_config` but does not accept a `layout` argument distinct from the source tensor's layout. If the source `cos` tensor arrives in `ROW_MAJOR_LAYOUT`, the pre-allocated buffer would also be in `ROW_MAJOR_LAYOUT`, and a layout conversion would be triggered inside the trace — which may allocate an intermediate transient buffer.

`move_weights_to_device_impl`, by contrast, accepts explicit `layout=ttnn.TILE_LAYOUT` in the `ttnn.from_torch` call and can guarantee the correct layout from the start.

---

## Conclusion

`TracedRun._alloc_kwarg_tensor` does not appear to exist in the current implementation, and even if added, it would face structural limitations for cos/sin pre-allocation: no control over the `mesh_mapper` (required for `ReplicateTensorToMesh`) and no control over the `layout` (required for `TILE_LAYOUT` compatibility with `ttnn.experimental.rotary_embedding`).

The recommended location for cos/sin pre-allocation is `move_weights_to_device_impl`, following the `_decode_cur_pos` model:

- `move_weights_to_device_impl` has access to `self.mesh_device` and can explicitly specify `mesh_mapper=ReplicateTensorToMesh(self.mesh_device)`.
- `move_weights_to_device_impl` can explicitly specify `layout=ttnn.TILE_LAYOUT`, guaranteeing compatibility with the downstream rotary embedding op without any in-trace layout conversion.
- The allocation is performed before `TracedRun._capture_trace` is ever called, satisfying the buffer address stability invariant unconditionally.
- The `ttnn.copy` update inside `forward` is independent of `TracedRun` internals — it is a plain TTNN op that any caller can reason about without understanding `TracedRun`'s buffer management.

> **Key Finding:** Pre-allocating cos/sin buffers in `move_weights_to_device_impl` (following the `_decode_cur_pos` model) is preferable to any `_alloc_kwarg_tensor` approach because it provides explicit control over mesh mapping and layout — the two properties that are non-negotiable for cos/sin replicated tensor compatibility with `ttnn.experimental.rotary_embedding`.

---

## Forward Reference

For the concrete code changes to `move_weights_to_device_impl` and `TTNNQwen3FullAttention.forward` that implement this recommendation, see [`../ch3_prereplication_impl/move_weights_impl_changes.md`](../ch3_prereplication_impl/move_weights_impl_changes.md).
