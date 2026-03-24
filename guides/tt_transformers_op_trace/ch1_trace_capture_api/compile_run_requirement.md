# Compile Run Requirement

Every trace capture in tt-transformers is preceded by a non-traced forward pass called the
compile run. This is not optional. The section below explains what the compile run accomplishes,
what happens if it is skipped, and how the trace region must be sized to avoid out-of-memory
errors at capture time.

---

## What the Compile Run Does

When a TTNN op is dispatched for the first time on a given device, TTNN must:

1. **Compile the kernel binary** for the specific op configuration (data type, tile layout,
   grid shape, shard spec). Compilation happens on the host CPU and typically takes tens to
   hundreds of milliseconds.
2. **Cache the compiled program** in the TTNN program cache, keyed by op configuration. On the
   next call with the same configuration the cached binary is reused immediately.
3. **Upload the kernel binary to the device**. The compiled binary must reside in device L1
   (Tensix local memory) or be staged for dispatch before the op can run.

The compile run performs all of this work during a standard, untraced execution. By the time the
compile run finishes, every kernel binary needed by the forward pass is compiled, cached, and
available on the device.

---

## What Happens If the Compile Run Is Skipped

If `ttnn.begin_trace_capture` is called before any untraced forward pass has run, then the very
first time each op is dispatched — which is now inside the capture window — TTNN triggers kernel
compilation during the recording phase.

The consequences are significant:

- **Compilation latency is embedded in the trace.** Compilation triggers transient DRAM
  allocations during the capture window. Because `mark_allocations_safe()` was already called
  by `begin_trace_capture`, these transient allocations are held live (prevented from being
  freed or reallocated) for the entire duration of recording. This inflates the trace
  footprint with allocations that belong to compilation overhead rather than to the model's
  steady-state execution, and may also embed kernel paths that become invalid once those
  transient allocations are eventually released.
- **Non-deterministic recording.** The transient DRAM allocations created during compilation
  inside the capture window cause the effective memory footprint of the recorded trace to
  exceed what is actually needed for replay. Because `mark_allocations_safe()` holds them
  live, the trace region must accommodate these extra allocations, potentially causing OOM at
  `end_trace_capture` time.
- **Trace invalidation on device restart.** A device restart clears the trace region in DRAM
  entirely. Any `trace_id` handle obtained before the restart becomes invalid; a subsequent
  `execute_trace` call with that handle fails because the trace buffer no longer exists in the
  device's DRAM. This failure is unrelated to compile-run state — the trace is gone regardless
  of whether a compile run was performed. After a restart, both the compile run and the full
  capture sequence must be re-executed from scratch.

> **Warning:** Always execute at least one complete, untraced forward pass before calling
> `ttnn.begin_trace_capture`. The compile run is not a performance optimization — it is a
> correctness requirement.

---

## The Pattern in `generator.py`

Both `_capture_decode_trace_text` and `_capture_trace_prefill` in
`models/tt_transformers/tt/generator.py` follow the same structure:

### `_capture_decode_trace_text` (lines 823–883)

```python
def _capture_decode_trace_text(self, tokens, current_pos, ...):
    # Compile run — untraced dispatch to warm kernels and program cache
    self._decode_forward_no_trace_text(
        tokens, current_pos, ...
    )
    logger.info("Done Compiling Model")

    # Prepare device inputs at fresh addresses for the capture run
    device_inputs_i = copy_host_to_device(host_inputs, mesh_device=...)

    # Capture run
    trace_id = ttnn.begin_trace_capture(self.model_args[i].mesh_device, cq_id=0)
    self.model[i].ttnn_decode_forward(*device_inputs[i], ...)
    ttnn.end_trace_capture(self.model_args[i].mesh_device, trace_id, cq_id=0)

    return trace_ids, tt_out_trace, *device_inputs
```

The `_decode_forward_no_trace_text` call is the compile run. It executes the full decode forward
pass — including all attention, feedforward, and normalization ops — without any capture bracket.
Only after that call completes and the `"Done Compiling Model"` log is emitted does the capture
bracket open.

### `_capture_trace_prefill` (lines 167–209)

```python
def _capture_trace_prefill(self, prefill_ids, ...):
    # Compile run
    device_inputs = copy_host_to_device(host_inputs, mesh_device=...)
    transformed_inputs = self.model[model_id].transform_and_embed_prefill_inputs_device(...)
    tt_out_trace = self.model[model_id].ttnn_prefill_forward(...)
    logger.info("Done Compiling Model")

    # Fresh device inputs for the capture run
    device_inputs = copy_host_to_device(host_inputs, mesh_device=...)

    # Capture run
    trace_id = ttnn.begin_trace_capture(self.model_args[model_id].mesh_device, cq_id=0)
    transformed_inputs = self.model[model_id].transform_and_embed_prefill_inputs_device(...)
    tt_out_trace = self.model[model_id].ttnn_prefill_forward(...)
    ttnn.end_trace_capture(self.model_args[model_id].mesh_device, trace_id, cq_id=0)

    logger.info("Done Capturing Prefill Trace")
    return trace_id, tt_out_trace, *device_inputs
```

Notice that `copy_host_to_device` is called twice: once before the compile run (line 184) and
again before the capture run (line 196). The second call allocates a fresh set of device buffers
whose addresses will be recorded by the capture. These are the buffers that replay must write to
via `copy_host_to_device(device_tensors=...)` on every subsequent step.

> **Key insight:** The device buffers allocated just before `begin_trace_capture` are the
> canonical aliased buffers for that trace. Their addresses are baked into the command buffer.
> The compile-run buffers can be discarded — they serve only to trigger compilation.

---

## `trace_region_size` in `device_params`

The trace region is a statically reserved DRAM allocation that holds command buffers. It is
configured via the `trace_region_size` key in the `device_params` dict, which is passed to the
pytest fixture that opens the mesh device. The reservation happens at device initialization time,
before any model weights are loaded.

Example from `models/tt_transformers/demo/simple_text_demo.py`:

```python
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
)
```

This reserves 50 MB (50,000,000 bytes) of DRAM for trace storage before the mesh device is
opened.

### Why OOM occurs if `trace_region_size` is too small

`MeshTrace::populate_mesh_buffer` copies the recorded command sequence into the trace region at
`end_trace_capture` time. If the encoded command buffer is larger than the reserved region, the
DRAM allocator cannot satisfy the allocation and raises an out-of-memory error. The trace capture
is aborted and no `trace_id` is returned.

The encoded size of a command buffer grows with the number of ops in the forward pass, the
number of layers, and the number of devices in the mesh. Larger models on more devices produce
larger command buffers.

### Reference values from `trace_region_config.py`

`models/tt_transformers/demo/trace_region_config.py` provides per-model, per-device size
recommendations from the `get_supported_trace_region_size` function:

| Model | Device | `trace_region_size` (bytes) |
|---|---|---|
| Llama-3.2-3B | N150 | 10,000,000 |
| Llama-3.1-8B | N150 | 25,000,000 |
| Llama-3.1-8B | N300 | 38,000,000 |
| Llama-3.1-8B | T3K | 50,000,000 |
| Llama-3.3-70B | T3K | 30,000,000 *(see note below)* |
| Llama-3.3-70B | TG | 80,000,000 |
| Llama-3.1-70B | T3K | 90,000,000 |
| Llama-3.1-70B | TG | 90,000,000 |
| Qwen3-32B | T3K | 90,000,000 |
| Qwen3-32B | TG | 96,000,000 |
| Qwen2.5-72B | T3K | 70,000,000 |
| Qwen2.5-72B | TG | 70,000,000 |

> **Note (Llama-3.3-70B T3K — 30 MB value):** The `30,000,000` value for Llama-3.3-70B on T3K
> is significantly smaller than the `90,000,000` value for Llama-3.1-70B on the same hardware.
> If this figure was sourced from a data-parallel (DP) configuration in which the T3K is split
> into multiple N150 submeshes, it represents the per-submesh trace footprint rather than the
> full single-mesh T3K footprint. Users running Llama-3.3-70B on a T3K as a single mesh (no DP
> split) should treat this value with caution: setting `trace_region_size=30000000` on a
> single-mesh T3K run may be insufficient and cause an OOM at `end_trace_capture` time. Verify
> against `trace_region_config.py` for your specific deployment mode before committing to this
> value.

> **Note:** When data-parallel (DP) mode splits a T3K into multiple independent N150 submeshes,
> each submesh independently opens a mesh device with the configured `trace_region_size`. Setting
> a very large value in DP mode multiplies the reservation across all submeshes. The comment in
> `trace_region_config.py` warns: a T3K with DP=8 effectively has 8 N150s, and a
> `trace_region_size` sized for a T3K would be dangerously large per submesh.

### Dynamic trace region allocation

If `trace_region_size` is set to `0`, the runtime switches to dynamic allocation mode.
Concretely, `begin_mesh_trace` calls `begin_dram_high_water_mark_tracking()` at the start of the
capture window, and `end_mesh_trace` calls `end_dram_high_water_mark_tracking()` at the close.
The runtime then reads the recorded allocation and deletion high-water marks to determine the
exact DRAM footprint required for the trace, and reserves only that amount. This avoids
over-reservation but adds measurement overhead at capture time and is not recommended for
production deployments where capture latency must be deterministic.

> **Note:** `trace_region_size=0` is a recognized sentinel for this dynamic mode. It does not
> mean "zero bytes reserved" — the runtime uses the high-water mark result to size the actual
> reservation after recording ends. Using explicit sizes from `trace_region_config.py` is still
> preferred for production use.

---

**Next:** [`replay_mechanics.md`](./replay_mechanics.md)
