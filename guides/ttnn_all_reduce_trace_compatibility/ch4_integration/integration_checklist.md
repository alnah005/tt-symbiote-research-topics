# Integration Checklist — `TTNNLinearIColShardedWAllReduced` Under Trace

This file provides a sequential checklist for enabling `TracedRun` on
`TTNNLinearIColShardedWAllReduced`. Each item is derived directly from the requirements
identified in Chapter 3. Items are grouped into pre-conditions (verified before the
first trace capture) and post-capture validation steps (run after `ttnn.end_trace_capture`
to confirm correctness before promotion to production).

---

## Pre-conditions

### Device setup

- [ ] **`FABRIC_1D_RING` fabric config is active before `ttnn.open_mesh_device`.**
  The fabric must be set by calling `ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)`
  before the mesh device is opened. In the pytest conftest fixture, this is
  accomplished by passing `"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING` inside
  the `device_params` dict. If `FABRIC_2D` is active, `ttnn.all_reduce` routes to the
  composite path (`composite_all_gather` + local reduce), which is trace-incompatible.

  > **Warning:** `ttnn.FabricConfig.FABRIC_1D` and `ttnn.FabricConfig.FABRIC_1D_RING`
  > are distinct enum values. The ring topology required for T3K cluster axis 1 is
  > `FABRIC_1D_RING`. Verify the exact constant via `ttnn.FabricConfig.__members__`
  > at the Python REPL if there is any doubt.

- [ ] **`trace_region_size` is non-zero in `ttnn.open_mesh_device`.**
  Pass `trace_region_size` as a keyword argument to `ttnn.open_mesh_device`. In the
  conftest-based pattern, include `"trace_region_size": <bytes>` in `device_params`.
  A value of `131072` bytes is sufficient for a single-layer unit test; production
  models typically require 50,000,000 or more. A zero or absent trace region causes
  `ttnn.begin_trace_capture` to fail at runtime.

- [ ] **Mesh shape is `(1, 8)` with cluster axis 1 as the collective axis.**
  `TTNNLinearIColShardedWAllReduced` hard-codes `cluster_axis=1` in its `ttnn.all_reduce`
  call. The ring size seen by the collective is the length of axis 1 of the mesh,
  which must be 8 for T3K. Confirm with `mesh_device.shape()`.

### Shape and routing

- [ ] **`out_features` is divisible by 256.**
  The non-composite path requires $\text{out features} / 8 \bmod 32 = 0$, i.e.,
  $\text{out features} \bmod 256 = 0$. Standard T3K hidden dimensions (4096, 8192,
  14336, 28672) satisfy this condition. A violation forces the composite path, which
  allocates dynamic intermediate tensors during the local reduce step and is
  trace-incompatible.

  > **Key finding:** `in_features` does not affect path selection; only `out_features`,
  > the fabric config, and the mesh shape determine whether the composite path is taken.

- [ ] **`memory_config=ttnn.DRAM_MEMORY_CONFIG` is passed to `ttnn.all_reduce`.**
  The existing `TTNNLinearIColShardedWAllReduced.forward` implementation already passes
  this argument. Confirm it has not been changed. Sharded memory configs can trigger
  a `sharded_to_interleaved` conversion inside `all_reduce_async`, which introduces
  an additional dynamic allocation.

### Module and decorator state

- [ ] **`TTNNLinearIColShardedWAllReduced` is recognised as trace-enabled by `TracedRun`.**
  `TTNNLinear` (the root of the inheritance chain) carries the `@trace_enabled`
  decorator. `TTNNLinearIColShardedWAllReduced` has no `@trace_disabled` override, so
  `is_trace_enabled(module)` returns `True` for any instance of this class. Confirm
  by calling `is_trace_enabled(layer)` in a debug session before the first traced run.

- [ ] **`TracedRun.configure(device=mesh_device)` is called before the first forward pass.**
  `TracedRun` reads `TracedRun._device` during execution. If `configure` is not called,
  the class-level `_device` attribute is `None` and the trace cache key will be computed
  against `None`. In practice the module's own `self.device` attribute drives execution,
  but `TracedRun.configure` must still be called to initialise `_trace_cache` to a
  clean state between test runs.

### Warm-up and buffer pre-allocation

- [ ] **A warm-up run completes successfully outside trace capture before `begin_trace_capture`.**
  `TracedRun._capture_trace` calls `module.forward(*func_args, **func_kwargs)` once
  before `ttnn.begin_trace_capture`. This warm-up compiles all kernels (matmul,
  reduce-scatter, all-gather) and allocates the output tensor at a stable DRAM address.
  The resulting `trace_output` tensor is stored in `TraceEntry.trace_output` and
  serves as the persistent output buffer for all subsequent replays.

  > **Note:** The framework handles warm-up automatically inside `_capture_trace`. The
  > caller does not need to run a separate warm-up before invoking `TracedRun`. However,
  > any caller-managed warm-up (e.g., for kernel compilation outside the trace path)
  > must complete and `ttnn.synchronize_device` must return before `begin_trace_capture`
  > is called.

- [ ] **`scattered_tensor` is pre-allocated and passed to `ttnn.reduce_scatter` as `output_tensor=` before trace capture.**
  The current `TTNNLinearIColShardedWAllReduced.forward` calls `ttnn.all_reduce`, which
  internally allocates `scattered_tensor` (shape $[\ldots,\ \text{out features} / 8]$)
  on every invocation. This dynamic allocation is not handled by `TracedRun._capture_trace`.

  > **Required code change:** The code pattern below implies a refactor of
  > `TTNNLinearIColShardedWAllReduced.forward`. The `ttnn.all_reduce` call in the
  > current forward method **must be replaced** with direct `ttnn.reduce_scatter` +
  > `ttnn.all_gather` calls — there is no configuration option or keyword argument that
  > allows a pre-allocated buffer to be injected through the `ttnn.all_reduce` API.
  > `all_reduce_async.cpp` hardcodes `std::nullopt` for `optional_output_tensor` in the
  > internal `ttnn::reduce_scatter` call (cluster\_axis overload, lines 344–358), making
  > pre-allocation impossible without bypassing `ttnn.all_reduce` entirely.

  The recommended remediation is to split `ttnn.all_reduce` into explicit
  `ttnn.reduce_scatter` + `ttnn.all_gather` calls, passing the pre-allocated buffer:

  ```python
  # Allocate once, before ttnn.begin_trace_capture
  scattered_shape = list(tt_output.shape)
  scattered_shape[-1] = out_features // 8   # ring_size = 8 on T3K
  scattered_tensor = ttnn.zeros(
      scattered_shape,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=mesh_device,
      memory_config=ttnn.DRAM_MEMORY_CONFIG,
  )

  # Inside forward (replaces ttnn.all_reduce):
  scattered_tensor = ttnn.reduce_scatter(
      tt_output,
      dim=-1,
      cluster_axis=1,
      memory_config=ttnn.DRAM_MEMORY_CONFIG,
      output_tensor=scattered_tensor,  # stable pre-allocated intermediate
      num_links=1,
      topology=ttnn.Topology.Ring,
  )
  tt_output = ttnn.all_gather(
      scattered_tensor,
      dim=-1,
      cluster_axis=1,
      memory_config=ttnn.DRAM_MEMORY_CONFIG,
      num_links=1,
      topology=ttnn.Topology.Ring,
  )
  ```

  Until this split is implemented, `ttnn.all_reduce` must be treated as potentially
  trace-unsafe for any shape where address stability of the intermediate cannot be
  empirically confirmed.

- [ ] **`TraceEntry.trace_output` (warm-up buffer) is distinct from the capture-window output buffer that `ttnn.execute_trace` writes into during replays.**
  Confirm that the warm-up output buffer and the capture-window output buffer are
  treated as separate allocations. `TracedRun.module_run` returns `entry.trace_output`
  (warm-up buffer, never updated by replays); callers invoking `ttnn.execute_trace`
  directly must read from `capture_output` (the capture-window buffer) to observe
  replay results. This is a known limitation of the current `TracedRun` implementation.
  See `minimal_test_pattern.md` Steps 4b, 5, and 6 for the full two-phase sequence and
  buffer allocation context.

- [ ] **`self.tt_bias` is pre-loaded at a fixed device address before trace capture (bias buffer stability).**
  `TTNNLinearIColShardedWAllReduced.forward` executes `tt_output += self.tt_bias` after
  `ttnn.all_reduce`. This in-place add is a separate buffer stability concern (see
  `ch3_verdict/q3_requirements_and_limitations.md`, Limitation 3). Two conditions must
  hold before trace capture:

  1. `self.tt_bias` must already be resident on device — it is a persistent weight
     tensor and its device address must be fixed for the lifetime of the trace. Loading
     or re-allocating the bias tensor after trace capture has begun will invalidate the
     recorded address.

  2. The in-place `+=` operation may or may not reuse `tt_output`'s buffer for its
     output. The resulting buffer address is recorded into the trace. Verify empirically
     (or by inspection of the op's allocation behaviour) that the bias-add output buffer
     is stable across replays. If the op allocates a new output buffer on each call, the
     bias step must be refactored to write into a pre-allocated persistent output tensor
     before trace capture is started.

  Until both conditions are confirmed, treat the bias step as a potential source of
  trace buffer instability independent of the `ttnn.all_reduce` intermediate buffers.

- [ ] **No intermediate tensor from inside `ttnn.all_reduce` is referenced after trace capture completes.**
  The trace records absolute device addresses. Any tensor whose device-side buffer is
  freed after capture and reallocated at a different address will cause the trace to
  read or write stale memory. Intermediates that escape the forward function (e.g.,
  via Python closure or module attribute assignment) must be identified and kept alive
  at fixed addresses for the lifetime of the trace.

---

## Post-capture validation steps

### Aliasing detection

After calling `ttnn.end_trace_capture` and `ttnn.synchronize_device`, run two
consecutive replays and compare their outputs:

`capture_output` is the output tensor returned by `layer.forward` inside the
`begin_trace_capture` / `end_trace_capture` window — it must be retained (not
discarded with `_ = ...`) so that replay results can be read from it. See
`minimal_test_pattern.md` Step 5 for the full definition and allocation context.

```python
# NOTE: ttnn.execute_trace writes replay results into the CAPTURE-WINDOW output buffer —
# the buffer that was live (and returned by layer.forward) inside the
# begin_trace_capture / end_trace_capture window. This buffer must be retained as
# capture_output (rather than discarded with _ = ...) so that replay results can be
# read from it.
#
# entry.trace_output (the warm-up buffer) is what TracedRun.module_run returns to
# callers (run_config.py line 1064), but it is NEVER written by ttnn.execute_trace.
# Reading entry.trace_output after replay always reflects stale warm-up data and
# cannot detect aliasing. Use capture_output instead.

ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
output_1 = ttnn.to_torch(capture_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))

ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
output_2 = ttnn.to_torch(capture_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))

assert torch.allclose(output_1, output_2), (
    "Outputs differ across replays: semaphore aliasing or buffer overlap present"
)
```

If the two outputs differ, the trace contains a buffer conflict — most commonly a
semaphore or intermediate tensor that is incremented rather than reset between replays.
See Chapter 3 (`q2_semaphore_state.md`) for the semaphore audit; if `ttnn.all_reduce`
with `std::nullopt` semaphores is used as documented, this check should pass.

### Numerical accuracy against non-traced reference

Compare the traced output against a non-traced reference produced by running the module
under `NormalRun` or by calling `forward` directly outside any trace context:

```python
from models.common.utility_functions import comp_pcc

pcc_passed, pcc_value = comp_pcc(reference_output, traced_output, pcc=0.999)
assert pcc_passed, f"PCC below threshold: {pcc_value:.6f}"
```

> **Note:** `comp_pcc` accepts two PyTorch tensors and a PCC threshold. A threshold of
> 0.999 is standard for bfloat16 collective operations where rounding order may differ
> between the traced and non-traced paths. Lower thresholds indicate numerical
> instability and should be investigated rather than accepted.

---

**Next:** [`minimal_test_pattern.md`](./minimal_test_pattern.md)
