# Minimal Test Pattern for Traced `TTNNLinearIColShardedWAllReduced`

This file presents an annotated code skeleton for a pytest unit test that validates
`TTNNLinearIColShardedWAllReduced` under trace capture and replay on a T3K mesh. The
skeleton is not a runnable script — it omits imports and helper scaffolding that vary by
environment — but every step maps to a concrete `TracedRun` internal or a checklist item
from `integration_checklist.md`, making it directly translatable into a production test.

---

## Parametrize axes

```python
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 131072, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [(1, 8)],   # T3K: 1 row, 8 columns
    indirect=True,
)
@pytest.mark.parametrize("in_features", [4096, 8192])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("cluster_axis", [1])
def test_ttnn_linear_icol_sharded_w_all_reduced_trace(
    mesh_device, device_params, in_features, dtype, cluster_axis
):
    ...
```

`device_params` flows to `ttnn.open_mesh_device` via the conftest `mesh_device` fixture.
`trace_region_size` is passed directly as a keyword argument; `fabric_config` is
extracted from `device_params` and applied via `ttnn.set_fabric_config` before the
device is opened. Both are conftest responsibilities — the test body sees only the
already-configured `mesh_device`.

`out_features` is not parametrized here because the test only needs to confirm trace
mechanics for one output shape. Use `out_features = in_features` as a representative
value (satisfies $\text{out features} \bmod 256 = 0$ for all parametrized `in_features`).

---

## Step 1 — Instantiate module and move weights to device

```python
    # Corresponds to: integration_checklist.md "Module and decorator state"
    out_features = in_features

    layer = TTNNLinearIColShardedWAllReduced(
        in_features=in_features,
        out_features=out_features,
    )

    # Synthesise random weights; in a real test load from a checkpoint
    import torch
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)
    layer.weight = weight
    layer.bias = None
    layer.preprocess_weights()
    layer.move_weights_to_device()
    set_device(layer, mesh_device)   # from models.experimental.tt_symbiote.utils.device_management
```

> **Note:** `set_device` propagates `mesh_device` to `layer.device` and registers the
> module so that `run_on_devices(DeviceArch.T3K)` guards pass.

---

## Step 2 — Create column-sharded input tensor

```python
    # Input shape: [1, 1, 32, in_features]
    # Each device receives [1, 1, 32, in_features // 8] (sharded on dim -1 across 8 devices)
    input_torch = torch.randn(1, 1, 32, in_features, dtype=torch.bfloat16)

    input_tt = ttnn.from_torch(
        input_torch,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=-1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
```

The column sharding on `dim=-1` satisfies the `input_dim=-1` assertion inside
`TTNNLinearIColShardedWAllReduced.forward`. Each device holds a `[1, 1, 32, in_features // 8]`
slice.

---

## Step 3 — Non-traced reference run (kernel compilation)

```python
    # Corresponds to: TracedRun._capture_trace warm-up, and
    # integration_checklist.md "Warm-up and buffer pre-allocation"

    # Run once outside any trace to compile kernels and obtain a reference output.
    # This mirrors the warm-up that _capture_trace performs internally before
    # ttnn.begin_trace_capture.
    reference_output_tt = layer.forward(input_tt)
    ttnn.synchronize_device(mesh_device)

    reference_output = ttnn.to_torch(
        reference_output_tt,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1),
    )
```

> **Note:** If the module is called via `TracedRun.module_run` rather than `layer.forward`
> directly, the first call triggers `_capture_trace` automatically, performing its own
> warm-up followed by trace capture. The pattern here separates the two concerns for
> clarity: (a) obtain a non-traced reference, then (b) capture a fresh trace.

---

## Step 4 — Pre-allocate `persistent_input` and `scattered_tensor` before capture

```python
    # Corresponds to: TracedRun._capture_trace lines 969–974 (persistent input allocation)
    # and integration_checklist.md "scattered_tensor is pre-allocated ... before trace capture"
    #
    # persistent_input is the stable device-side input buffer that every replay copies
    # fresh data into before calling ttnn.execute_trace. It is allocated here (before
    # the warm-up) so that both the warm-up run (Step 4b) and the capture window (Step 5)
    # use the same persistent buffer — matching the behaviour of _capture_trace, which
    # allocates trace_inputs before the warm-up call at line 990.

    cq_id = 0

    # Allocate a persistent input buffer — mirrors _capture_trace's trace_input
    # allocation (lines 972–974). This buffer is copied into before every replay.
    persistent_input = ttnn.to_device(
        input_tt.cpu(),
        mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Pre-allocate scattered_tensor (required only if forward has been refactored to use
    # explicit ttnn.reduce_scatter + ttnn.all_gather instead of ttnn.all_reduce).
    # If the unmodified ttnn.all_reduce call is still present, this pre-allocation cannot
    # be injected through the public API and the scatter intermediate address stability
    # is unguaranteed (see integration_checklist.md).

    num_devices = mesh_device.shape()[1]   # 8 for T3K
    scattered_shape = [1, 1, 32, out_features // num_devices]

    scattered_tensor = ttnn.zeros(
        scattered_shape,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Pass scattered_tensor to the module or hold it at module scope so that the
    # modified forward can reference it. The exact mechanism depends on how the
    # forward method is refactored (e.g., module attribute, closure, or argument).
```

---

## Step 4b — Warm-up run (before trace capture)

```python
    # Mirrors _capture_trace line 990 — warm-up before begin_trace_capture; see integration_checklist.md pre-conditions.

    warm_up_output = layer.forward(persistent_input)
    ttnn.synchronize_device(mesh_device)
```

---

## Step 5 — Trace capture

```python
    # Corresponds to: TracedRun._capture_trace lines 992–995
    #
    # If using TracedRun.module_run, capture happens automatically on the first call.
    # The manual pattern below shows the equivalent low-level sequence for testing.

    # Open the capture window and run the forward pass inside it.
    # Unlike _capture_trace (which discards the capture-window output with _ = ...),
    # this test retains the return value as capture_output — this is the CAPTURE-WINDOW
    # output buffer. ttnn.execute_trace writes replay results into THIS buffer during
    # every replay. warm_up_output (from Step 4b) is never updated by replays.
    #
    # _capture_trace line 993: _ = module.forward(*trace_func_args, **func_kwargs)
    # We retain the return value here to enable reading replay results in Steps 6 and 7.
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=cq_id)
    capture_output = layer.forward(persistent_input)   # retain instead of discarding
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=cq_id)
    ttnn.synchronize_device(mesh_device)
```

The two-phase structure above — warm-up outside the window, capture inside the window —
mirrors what `TracedRun._capture_trace` performs (lines 989–994 of `run_config.py`).
The key difference is that `_capture_trace` discards the capture-window return value
(`_ = module.forward(...)`), while this test retains it as `capture_output` so that
replay results can be read directly from the correct buffer.
In the `TracedRun` path the caller never calls `begin_trace_capture` directly; it is
shown here explicitly so the test can verify the low-level behaviour in isolation.

> **Warning:** `ttnn.synchronize_device` must be called after `ttnn.end_trace_capture`
> before reading any output. The end-of-capture call is asynchronous; without
> synchronisation, the capture-window buffer may contain undefined values from the
> capture bookkeeping phase rather than from an actual forward pass.

> **Which buffer do replays write into?** `ttnn.execute_trace` writes into the
> capture-window output buffer — the buffer returned by `layer.forward(persistent_input)`
> inside `begin_trace_capture` / `end_trace_capture`, stored here as `capture_output`.
> The warm-up output (`warm_up_output` from Step 4b) is stored in `TraceEntry.trace_output`
> by `TracedRun` but is **never updated** by replays. Reading from `capture_output`
> after `execute_trace` gives the replay result; reading from `warm_up_output` after
> `execute_trace` gives stale warm-up data.

---

## Step 6 — Dual replay and aliasing check

```python
    # Corresponds to: integration_checklist.md "Aliasing detection"

    # First replay
    ttnn.copy(input_tt, persistent_input)  # write new input into stable buffer
    ttnn.execute_trace(mesh_device, trace_id, cq_id=cq_id, blocking=True)
    output_1 = ttnn.to_torch(
        capture_output,   # execute_trace writes into the capture-window buffer
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1),
    )

    # Second replay (same input, should produce identical result)
    ttnn.copy(input_tt, persistent_input)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=cq_id, blocking=True)
    output_2 = ttnn.to_torch(
        capture_output,   # read again from the same capture-window buffer after second replay
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1),
    )

    assert torch.allclose(output_1, output_2), (
        "Outputs differ between replay 1 and replay 2. "
        "Semaphore aliasing or buffer overlap is present in the trace."
    )
```

Both reads use `capture_output` — the capture-window buffer — because `execute_trace` never writes into `warm_up_output`.

---

## Step 7 — Numerical comparison against non-traced reference

```python
    # Corresponds to: integration_checklist.md "Numerical accuracy against non-traced reference"
    # Compares output_1 (from execute_trace via capture_output, Step 6) against reference_output
    # (non-traced forward, Step 3). Do NOT compare against warm_up_output — trivially high PCC.
    from models.common.utility_functions import comp_pcc

    # output_1 is already on CPU (from Step 6 ttnn.to_torch call after execute_trace)
    # reference_output is already on CPU (from Step 3 ttnn.to_torch call)
    pcc_passed, pcc_value = comp_pcc(reference_output, output_1, pcc=0.999)
    assert pcc_passed, f"PCC below threshold against non-traced reference: {pcc_value:.6f}"
```

The 0.999 threshold accounts for floating-point reordering under bfloat16 arithmetic between the traced and non-traced paths.

---

## Step 8 — Release trace

```python
    # Always release the trace to free the trace region on the device.
    # In TracedRun this is done via TracedRun.release_all() or TracedRun.release(module_name).
    ttnn.release_trace(mesh_device, trace_id)
```

If the test is structured as a pytest fixture with a `yield`, `ttnn.release_trace` should
be placed in the teardown block (after `yield`) to ensure it runs even if an assertion
fails.

---

## Mapping to `TracedRun` internals

| Test step | `TracedRun` equivalent |
|-----------|------------------------|
| Step 3 — non-traced reference | `NormalRun.module_run` or direct `forward` before `TracedRun.configure` is called |
| Step 4 — `persistent_input` allocation | `_capture_trace` lines 969–974: allocates `trace_inputs` list via `ttnn.to_device` per input argument — done before the warm-up call |
| Step 4 — `scattered_tensor` pre-allocation | Caller responsibility; not handled by `_capture_trace` |
| Step 4b — warm-up (`warm_up_output = layer.forward(persistent_input)`) | `_capture_trace` line 990: `trace_output = module.forward(*func_args, **func_kwargs)` — runs **before** `begin_trace_capture`; result stored as `TraceEntry.trace_output` (warm-up buffer, never updated by replays) |
| Step 5 — `ttnn.begin_trace_capture` / `capture_output = layer.forward(persistent_input)` / `ttnn.end_trace_capture` | `_capture_trace` lines 992–994: capture window opened, `_ = module.forward(*trace_func_args, ...)` run inside it (return value discarded in `_capture_trace`; retained here as `capture_output`), capture window closed; `capture_output` is the **capture-window output buffer** that `execute_trace` writes into on every replay |
| Step 5 — `ttnn.synchronize_device` | `_capture_trace` line 995 |
| Step 6 — `ttnn.copy` | `_copy_inputs_to_trace_buffer`: `ttnn.copy(arg, trace_input)` |
| Step 6 — `ttnn.execute_trace` | `module_run` line 1063: `ttnn.execute_trace(entry.device, entry.trace_id, ...)` — writes into the capture-window output buffer (`capture_output`), **not** into `entry.trace_output` |
| Step 6 — reading replay output from `capture_output` | Low-level test reads `capture_output` directly; `module_run` line 1064 instead returns `entry.trace_output` (warm-up buffer — stale), which is the `TracedRun`-level abstraction's known limitation |
| Step 7 — PCC between `output_1` and `reference_output` | Validates numerically that the traced path (replay output from `capture_output`) matches the non-traced path (`reference_output` from Step 3) |
| Step 8 — `ttnn.release_trace` | `TracedRun.release_all()` or `TracedRun.release(module_name)` |

---

**End of guide.** Return to [Guide Index](../index.md)
