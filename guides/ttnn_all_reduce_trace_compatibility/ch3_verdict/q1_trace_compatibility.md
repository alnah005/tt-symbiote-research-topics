# Q1 — Is `ttnn.all_reduce` Compatible With Trace Capture and Replay?

This file answers the first research question: whether `ttnn.all_reduce` called with a
synchronous Ring topology on a T3K (8-device, 1×8 logical ring) is compatible with
`ttnn.begin_trace_capture` / `ttnn.execute_trace` semantics. The answer is conditional:
compatibility holds when the non-composite path is taken and when the intermediate
`scattered_tensor` is pre-allocated as a persistent buffer. The composite path is
trace-incompatible without explicit persistent buffer management.

---

## Conclusion

`ttnn.all_reduce` as called by `TTNNLinearIColShardedWAllReduced` does not hold any
persistent semaphore state between invocations. However, both execution paths through
the composite `all_reduce_async` implementation allocate device tensors whose addresses
must be stable across all trace replays. The trace capture records the buffer addresses
that were live at the moment `ttnn.begin_trace_capture` was called; any buffer that is
freshly allocated on each call will produce a different address on each forward pass,
causing replays to read or write stale or incorrect memory.

The specific conditions are:

- **Non-composite path (reduce-scatter + all-gather):** one dynamic intermediate is
  allocated — the `scattered_tensor` produced by `ttnn.reduce_scatter`. This tensor
  must be pre-allocated and its address must be stable before capture begins. The
  path uses only local (per-program) semaphores, which introduces no additional state
  risk.
- **Composite path (`composite_all_gather` + local reduce):** the composite branch
  calls `composite_common::composite_all_gather`, which internally uses
  `ttnn::prim::all_broadcast` followed by `ttnn::concat` to produce the gathered
  tensor. It then reduces locally using `local_sum` (which calls `ttnn::moreh_sum` for
  non-float32 dtypes) or `local_sum_float32` (which calls `ttnn::sum` via
  `ttnn::transpose` for float32 inputs). Two dynamic intermediates are allocated for
  `TTNNLinearIColShardedWAllReduced` — the reshaped tensor and the `gather_tensor`
  passed to the local reduce. Both are created and deallocated inside the composite
  branch on every call. This makes the composite path unconditionally
  trace-incompatible without custom persistent buffer injection at each allocation site.
  A third intermediate (the interleaved conversion tensor from `sharded_to_interleaved`)
  is only allocated when `change_mem_config` is `true`, i.e., when the input is sharded
  (`all_reduce_async.cpp` lines 279–284). Because `TTNNLinearIColShardedWAllReduced`
  passes `memory_config=ttnn.DRAM_MEMORY_CONFIG` to `ttnn.linear`, the input to
  `ttnn.all_reduce` is always DRAM interleaved and the conversion tensor is never
  allocated. If a future version changes the `ttnn.linear` output to an L1-sharded
  memory config, this conversion tensor will also require pre-allocation as a persistent
  buffer before trace capture.

---

## Path selection: composite vs non-composite

The non-composite (reduce-scatter + all-gather) path is taken when all four predicates
evaluated in `ExecuteAllReduceAsync::invoke` are `false`: `use_composite_all_gather`,
`use_composite_reduce_scatter`, `dim != composite_dim`, and `composite_for_2d_mesh`.
On T3K with `FABRIC_1D_RING`, a `TILE_LAYOUT` DRAM tensor whose last dimension is
divisible by $8 \times 32 = 256$ satisfies all conditions for the non-composite path.
See [`call_chain.md` §4](../ch2_all_reduce_internals/call_chain.md) for the full
sequential predicate definitions, source citations, and the T3K evaluation table.

---

## T3K path assessment for `TTNNLinearIColShardedWAllReduced`

`TTNNLinearIColShardedWAllReduced.forward` in
`models/experimental/tt_symbiote/modules/linear.py` calls:

```python
# linear.py lines 217-224
tt_output = ttnn.linear(input_tensor, self.tt_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
tt_output = ttnn.all_reduce(
    tt_output,
    num_links=1,
    topology=ttnn.Topology.Ring,
    cluster_axis=1,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

Key properties of `tt_output` entering `ttnn.all_reduce`:

| Property | Value | Implication |
|----------|-------|-------------|
| Memory config | `DRAM_MEMORY_CONFIG` (interleaved) | Not sharded; `sharded_to_interleaved` is skipped |
| Layout | `TILE_LAYOUT` | Row-major composite path not triggered |
| Fabric config on T3K | `FABRIC_1D_RING` | `composite_for_2d_mesh` guard is `false` |
| Ring size (T3K cluster axis 1) | 8 devices | $8 \bmod 2 = 0$, satisfies even-ring requirement |
| Typical decode shape | `[1, 1, 1, out_features]` after `ttnn.linear` | Last dimension must be divisible by 8 after tile-rounding |

For the non-composite path to be taken, the scatter dimension (last dimension) of
`tt_output` must satisfy:

$$\frac{\text{out features}}{8} \bmod \text{tile width} = 0$$

Standard hidden dimensions used with T3K deployments (e.g., 4096, 8192, 14336, 28672)
are all divisible by $8 \times 32 = 256$, so this condition is satisfied in practice.

> **Key finding:** On T3K with `FABRIC_1D_RING` and standard hidden dimensions,
> `TTNNLinearIColShardedWAllReduced` routes `ttnn.all_reduce` to the non-composite
> reduce-scatter + all-gather path. The composite path is not reached during normal
> decode operation.

> **Warning:** If the hidden dimension after sharding is not tile-aligned per device
> (e.g., due to an unusual model size or padding), `use_composite_reduce_scatter` will
> return `true` and the composite path will be taken. The composite path is
> trace-incompatible. Always verify that `out_features / 8` is a multiple of 32 before
> enabling `TracedRun` for a new model configuration.

---

## Evidence from the existing codebase

> **Important:** The two test files cited below exercise `ttnn.experimental.all_reduce_async`
> (the async variant, with explicit `GlobalSemaphore` handles) and a custom kernel-based
> all-reduce — **not** `ttnn.all_reduce` (the synchronous path, no semaphore arguments).
> They establish that CCL-class operations are traceable in principle and demonstrate the
> required persistent-buffer pattern, but they are **analogous patterns**, not direct
> validation of `ttnn.all_reduce` trace compatibility. Direct trace validation of
> `ttnn.all_reduce` (the synchronous path) on T3K is an **open test gap**.

**`tests/ttnn/unit_tests/operations/ccl/test_new_all_reduce.py`**

Parameterised with `@pytest.mark.parametrize("trace_mode", [True])` and
`"trace_region_size": 23887872`. The `run_all_reduce_impl` helper performs a full
compile run, then captures a trace via `ttnn.begin_trace_capture` / `ttnn.end_trace_capture`,
and finally calls `ttnn.execute_trace`. The input and output tensors are pre-allocated
before capture. The test calls `ttnn.experimental.all_reduce_async` — the async
variant with explicit `GlobalSemaphore` handles and a pre-allocated `tt_intermediate_tensor`
pool — not the synchronous `ttnn.all_reduce` path. It confirms that CCL trace is
viable when all buffers are stable, but does not validate the synchronous path.

**`models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py`**

Parameterised with `"trace_region_size": 573440`. The test pre-allocates
`output_tensor` and `intermediate_tensor` before the compile run, then captures a
warmup trace and a main trace, both of which call `DeepseekMinimalAllReduce.op` — a
custom kernel-based all-reduce that passes explicit `semaphores` handles — with the
same persistent buffers. This is again an async-semaphore pattern, not a test of the
synchronous `ttnn.all_reduce` invocation used by `TTNNLinearIColShardedWAllReduced`.
The pattern establishes that intermediate tensors must be created and brought to device
before `ttnn.begin_trace_capture`:

```python
# test_ccl_all_reduce.py lines 200-210
trace_id_warmup = ttnn.begin_trace_capture(submesh, cq_id=0)
for i in range(num_warmup_iter):
    ttnn_result = DeepseekMinimalAllReduce.op(
        input_tensor_mesh,
        intermediate_tensor,          # pre-allocated persistent buffer
        cluster_axis=cluster_axis,
        persistent_output_tensor=output_tensor,  # pre-allocated persistent buffer
        residual_tensor_mesh=residual_tensor_mesh,
        semaphores=semaphores,
    )
ttnn.end_trace_capture(submesh, trace_id_warmup, cq_id=0)
```

The key structural lesson from both tests is that every tensor whose device address
must be stable — input, output, and all intermediates — must exist on device before
the first call to `ttnn.begin_trace_capture`. This lesson applies equally to the
synchronous `ttnn.all_reduce` path, but the synchronous path itself has not been
exercised under trace capture in T3K CI. Closing this test gap — running
`ttnn.all_reduce` with `FABRIC_1D_RING` inside a `begin_trace_capture` /
`execute_trace` cycle on a T3K submesh — would provide direct validation of the
analysis in this chapter.

---

**Next:** [`q2_semaphore_state.md`](./q2_semaphore_state.md)
