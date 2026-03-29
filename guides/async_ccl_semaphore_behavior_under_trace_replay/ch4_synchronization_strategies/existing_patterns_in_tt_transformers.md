# Existing Patterns in tt-transformers

By the end of this file you will know precisely which traced paths in the tt-transformers codebase currently call async CCL ops, whether any of them perform the semaphore management described in this chapter, and which paths would require changes before async CCL semaphores can be used safely with trace replay.

---

## Scope of the audit

This audit covers the following files, all under `/tt-metal/models/`:

- `tt_transformers/tt/generator.py` — the primary `Generator` class used by text and multimodal models
- `demos/llama3_70b_galaxy/tt/generator.py` — the Galaxy-specific generator
- `tt_transformers/tt/ccl.py` — the older `TT_CCL` class and CCL helper functions (`tt_all_reduce`, `tt_all_gather`)
- `common/modules/tt_ccl.py` — the newer `TT_CCL` class shared across TTTv2 modules
- `demos/llama3_70b_galaxy/tt/llama_ccl.py` — the Galaxy-specific `TT_CCL`-like class
- `demos/deepseek_v3/tt/ccl.py` — the DeepSeek V3 `CCL` class

---

## generator.py (tt_transformers): no semaphore management around trace calls

The main generator (`models/tt_transformers/tt/generator.py`) contains three trace capture sites and corresponding replay paths:

**Prefill trace** (`_capture_trace_prefill` / `_prefill_forward_trace`):
The capture calls `ttnn.begin_trace_capture`, runs `model.ttnn_prefill_forward`, then calls `ttnn.end_trace_capture`. The replay calls `ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)`. There are no calls to `ttnn.reset_global_semaphore_value` and no modifications to `TT_CCL` index fields anywhere in the capture or replay paths.

**Decode trace** (`_capture_decode_trace_text` / `_decode_forward_trace_text`):
The capture (lines 866–877) calls `ttnn.begin_trace_capture`, runs `model.ttnn_decode_forward`, then calls `ttnn.end_trace_capture`. The replay (line 920) calls `ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)`. No semaphore management is present.

**Multimodal decode trace** (lines 1550–1584):
Same pattern: `begin_trace_capture`, multimodal forward, `end_trace_capture`. Replay at line 1671: `ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)`. No semaphore management.

---

## generator.py (llama3_70b_galaxy): no semaphore management around trace calls

The Galaxy-specific generator (`models/demos/llama3_70b_galaxy/tt/generator.py`) has two trace capture sites:

**Prefill trace** (lines 508–513): `begin_trace_capture` → prefill forward → `end_trace_capture`. Replay at line 538. No semaphore management.

**Decode trace** (lines 694–707): `begin_trace_capture` → decode forward → `end_trace_capture`. Replay at line 724. No semaphore management.

The Galaxy generator also does not call `tt_ccl.reset_gather_and_buffer_idx()` (a method defined in `llama_ccl.py` that resets `gather_idx` and `barrier_semaphore_idx` to zero) in any path adjacent to `execute_trace`.

---

## Which CCL paths are active in the traced models

Async CCL ops are used through two mechanisms in the traced decode forward:

**Via `tt_all_reduce` in `ccl.py`:** The `tt_all_reduce` helper in `models/tt_transformers/tt/ccl.py` calls `get_and_cycle_ag_semaphore_handles(cluster_axis)` and `get_and_cycle_barrier_semaphore_handle(cluster_axis)` on the `use_composite=False` path (`all_gather_async` + `fast_reduce_nc`). On the `use_composite=True` path (`reduce_scatter_minimal_async` + `all_gather_async`), the function makes **four** cycling calls in sequence (lines 264–286):

1. `get_and_cycle_rs_semaphore_handles(cluster_axis)` — for `reduce_scatter_minimal_async`
2. `get_and_cycle_barrier_semaphore_handle(cluster_axis)` — barrier for `reduce_scatter_minimal_async`
3. `get_and_cycle_ag_semaphore_handles(cluster_axis)` — for the subsequent `all_gather_async`
4. `get_and_cycle_barrier_semaphore_handle(cluster_axis)` — barrier for `all_gather_async`

All four handle groups are baked into the trace at capture time and must be reset to 0 before each replay. Pre-replay resets that cover only the RS handles and one barrier handle (two of the four groups) leave the AG handles and the second barrier handle unreset — which produces the silent corruption failure mode described in Chapter 3. Both paths run during decode and are traced when the model uses `tt_all_reduce`.

**Directly in attention and MLP layers:** `models/tt_transformers/tt/attention.py` calls `self.tt_ccl.get_and_cycle_ag_semaphore_handles()` (with `cluster_axis=None`) and `self.tt_ccl.get_and_cycle_barrier_semaphore_handle()` at lines 672–674 and 690–694 (the T3K ring path). `models/tt_transformers/tt/mlp.py` calls `get_and_cycle_rs_semaphore_handles(cluster_axis)` and `get_and_cycle_barrier_semaphore_handle(cluster_axis)` at lines 185–186 and 201–202, and `get_and_cycle_ag_semaphore_handles(cluster_axis)` with `get_and_cycle_barrier_semaphore_handle(cluster_axis)` at lines 257, 262.

In all cases, the `get_and_cycle_*` calls happen inside the model forward pass, which is called inside the `begin_trace_capture` / `end_trace_capture` bracket.

---

## Why no semaphore management exists today

The existing decode trace in the current codebase targets hardware configurations and model paths that use the composite `reduce_scatter_minimal_async` + `all_gather_async` route or the ring `all_gather_matmul_async` route. These are present in the code, but the trace infrastructure was built when the primary tested configuration either did not use global semaphores (e.g., smaller devices) or used them in a sequential non-traced context where the self-reset behavior is sufficient.

No `snapshot_semaphore_indices` or `restore_semaphore_indices` method exists in any `TT_CCL` class in the codebase. No call to `ttnn.reset_global_semaphore_value` appears in any of the generator files adjacent to `execute_trace`. This is confirmed by a search of `models/tt_transformers/tt/generator.py` and `models/demos/llama3_70b_galaxy/tt/generator.py`.

The only existing `reset_global_semaphore_value` calls in model code appear in:

- `models/demos/gpt_oss/tt/ccl.py` (line 93–95): a `reset_global_semaphores` method that resets all RS and AG ping-pong semaphores, but this is called in a non-traced context
- `models/tt_dit/parallel/manager.py` (lines 242–248): a `reset_global_semaphores` method resetting all semaphores, also in a non-traced context

Neither of these patterns is applied to the tt-transformers or Galaxy generator trace paths.

---

## llama_ccl.py and deepseek_v3/tt/ccl.py: same pattern, same gaps

`models/demos/llama3_70b_galaxy/tt/llama_ccl.py` defines its own `TT_CCL`-like class with `gather_idx`, `barrier_semaphore_idx`, `barrier_semaphore_handles`, `gather_semaphore_handles`, and `reduce_semaphore_handles`. It has `get_and_cycle_barrier_semaphore_handle(cluster_axis)` and uses `gather_idx` directly (without a `get_and_cycle_*` wrapper) for the AG semaphore selection. It defines `reset_gather_and_buffer_idx()` (lines 120–123) which resets `gather_idx`, `reduce_scatter_buffer_idx`, and `barrier_semaphore_idx` to zero — a partial host-counter reset. However, this method is not called adjacent to `execute_trace` in the Galaxy generator.

`models/demos/deepseek_v3/tt/ccl.py` defines a `CCL` class with `gather_sems`, `reduce_scatter_sems`, `barrier_sems`, and per-type counters (`gather_sem_cnt`, `reduce_scatter_sem_cnt`, `barrier_sem_cnt`). Its structure follows the same double-buffer pattern: `sems_per_axis = 2`, a counter per axis per semaphore type, a `_get_sem_and_update_counter` helper. It has no reset method and no existing trace integration.

Both classes would need the same two-part fix — host counter restore before replay, device semaphore reset before replay — before being used in a traced async CCL context.

---

## Summary: what needs to be added

| Path | Async CCL ops in traced forward? | Host counter reset before replay? | Device semaphore reset before replay? |
|---|---|---|---|
| `tt_transformers/tt/generator.py` decode trace | Yes (via `attention.py`, `mlp.py`, `ccl.py`) | No | No |
| `tt_transformers/tt/generator.py` prefill trace | Yes (same model ops) | No | No |
| `demos/llama3_70b_galaxy/tt/generator.py` decode trace | Yes (via `llama_ccl.py`) | No (`reset_gather_and_buffer_idx` not called at replay) | No |
| `demos/llama3_70b_galaxy/tt/generator.py` prefill trace | Yes | No | No |

All four paths require the synchronization strategies described in this chapter before trace replay with async CCL semaphores is correct.

---

**Next chapter:** [Chapter 5 — Implementing Trace Support: A Step-by-Step Guide](../ch5_implementation_guide/index.md)
