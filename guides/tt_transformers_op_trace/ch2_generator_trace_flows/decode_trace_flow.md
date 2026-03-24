# Decode Trace Flow

This file traces the execution path for a single decode step through `models/tt_transformers/tt/generator.py`, starting at the public `decode_forward` entry point and following the code into the trace capture and replay helpers. By the end you will understand how the `enable_trace` flag selects the trace path, why `trace_ids_decode` is keyed on the `sampling_on_device` boolean, how `reset_batch` and page-table changes trigger a full `copy_host_to_device` before replay, and how the split-sampling variant captures a second trace for the sampling step.

---

## Routing at `decode_forward`

`decode_forward` in `generator.py` is the entry point called by vLLM and internal tests. Near the end of the function body it branches on `enable_trace`:

```python
if enable_trace:
    tt_decode_output = self._decode_forward_trace_text(**decode_kwargs, reset_batch=mode_switched)
else:
    tt_decode_output = self._decode_forward_no_trace_text(**decode_kwargs)
```

The `mode_switched` flag — set to `True` whenever the `Generator` transitions from prefill mode to decode mode — is forwarded as `reset_batch`. This forces a full input copy on the first decode step after a prefill, because the device-side token and page-table tensors may have changed.

Before the branch, `decode_forward` also computes two important booleans:

```python
sampling_on_device = sampling_params is not None
split_sampling_enabled = bool(self.enable_split_sampling and sampling_on_device)
self._set_sampling_trace_mode(split_sampling_enabled)
```

`sampling_on_device` becomes the dictionary key used to select which captured trace to replay. `split_sampling_enabled` tells the capture helper whether to break the trace at the logits boundary and capture a second sampling trace separately.

---

## Capturing the Trace: `_capture_decode_trace_text`

```
Python host     ┌────────────────────────────────────────────────────────────┐
                │  _capture_decode_trace_text                                │
                │  1. compile run: _decode_forward_no_trace_text             │
                │  2. for each DP shard i:                                   │
                │     a. prepare_decode_inputs_host → device_inputs[i]       │
                │     b. begin_trace_capture(mesh_device, cq_id=0)           │
                │     c. ttnn_decode_forward(*device_inputs[i], ...)         │
                │     d. end_trace_capture(mesh_device, trace_id, cq_id=0)   │
                │     e. if split_enabled:                                    │
                │          sampling_module.capture_trace(logits, tt_out_tok) │
                │  return trace_ids, tt_out_trace, *device_inputs            │
                └────────────────────────────────────────────────────────────┘

TTNN runtime    ┌────────────────────────────────────────────────────────────┐
                │  Records op graph between begin/end brackets               │
                │  Allocates output buffers in trace region                  │
                └────────────────────────────────────────────────────────────┘

Device          ┌────────────────────────────────────────────────────────────┐
                │  device_inputs[i] become the "live" buffers for replay     │
                └────────────────────────────────────────────────────────────┘
```

Key details:

- `ttnn.begin_trace_capture` and `ttnn.end_trace_capture` are called per shard with `cq_id=0`. This is why `num_command_queues: 1` is required in the device fixture — trace capture and replay must use command queue 0 exclusively.
- The `device_inputs` list returned from the capture function contains the **same on-device tensor objects** that the recorded graph reads from. All future replays must write updated values into these exact buffers.
- The `split_enabled` flag is computed per shard as:

```python
split_enabled = (
    sampling_on_device
    and sampling_module is not None
    and getattr(sampling_module, "enable_internal_trace", False)
)
```

When `split_enabled` is `True`, `capture_sampling_trace=True` is passed into `ttnn_decode_forward`, which routes the sampling step through its own internal trace, and `sampling_module.capture_trace(logits=..., tt_out_tok=...)` captures the sampling trace after the main trace bracket closes.

---

## First Call vs. Subsequent Calls: `_decode_forward_trace_text`

```python
def _decode_forward_trace_text(
    self, tokens, current_pos, page_table=None, kv_cache=None, sampling_on_device=False, reset_batch=False
):
    if not self.trace_ids_decode[sampling_on_device]:
        trace_ids, tt_out_trace, *device_inputs = self._capture_decode_trace_text(...)
        self.trace_ids_decode[sampling_on_device] = trace_ids
        self.trace_inputs_decode[sampling_on_device] = device_inputs
        self.trace_output_decode[sampling_on_device] = tt_out_trace
    ...
```

`self.trace_ids_decode` is a `defaultdict(lambda: None)` initialised on the `Generator` instance. Its key is the `sampling_on_device` boolean, so up to two separate traces may be held:

| Key (`sampling_on_device`) | Scenario |
|---|---|
| `False` | Greedy decode — logits returned to host, token selected there |
| `True` | On-device sampling — sampling step runs on device, token tensor returned |

> **Key insight:** The two traces differ in which TTNN ops are included (the sampling ops are only present when `sampling_on_device=True`), so they must be captured and stored separately. Replaying the wrong trace would produce incorrect output.

---

## Input Reset Before Replay: `reset_inputs`

Before calling `ttnn.execute_trace`, the function decides whether to re-copy host inputs to the device buffers:

```python
reset_inputs = reset_batch or not sampling_on_device
if self.prev_page_table is None or any(
    not torch.equal(prev, curr) for prev, curr in zip(self.prev_page_table, page_table)
):
    reset_inputs = True
    if page_table is not None:
        self.prev_page_table = tuple(pt.clone() for pt in page_table)

if reset_inputs:
    for i in range(self.data_parallel):
        host_inputs_i = self.model[i].prepare_decode_inputs_host(tokens[i], current_pos[i], user_page_table)
        copy_host_to_device(
            host_tensors=host_inputs_i,
            device_tensors=self.trace_inputs_decode[sampling_on_device][i],
        )
```

Three conditions independently trigger a copy:

1. **`reset_batch=True`** — the caller requested a fresh batch (e.g., mode switch from prefill to decode).
2. **`not sampling_on_device`** — in greedy mode the token tensor must always be refreshed because the host selects the next token and needs to write it into the device buffer before replay.
3. **Page table changed** — new pages have been allocated or the KV-cache layout has been shuffled; the page table tensor on device must be updated.

> **Note:** When `sampling_on_device=True` and neither `reset_batch` is set nor the page table has changed, the copy is skipped entirely. The sampling result from the previous replay already sits in the device buffer as the input for the next step, so no host-to-device transfer is needed for the token tensor.

---

## Replay

After the conditional copy, every shard's trace is dispatched in a tight loop:

```python
for i, trace_id in self.trace_ids_decode[sampling_on_device].items():
    ttnn.execute_trace(self.model_args[i].mesh_device, trace_id, cq_id=0, blocking=False)
```

`blocking=False` allows the host to continue to the next iteration while the device executes. The output buffers (`self.trace_output_decode[sampling_on_device]`) remain valid for reading because they are pre-allocated inside the trace region and reused in place across all replays.

---

## Split-Sampling Replay

When `sampling_on_device=True` and the split-sampling trace was captured, `_decode_forward_trace_text` calls an additional sampling step after `execute_trace`:

```python
if sampling_on_device:
    for i in range(self.data_parallel):
        sampling_module = getattr(self.model[i], "sampling", None)
        if sampling_module is None or not getattr(sampling_module, "enable_internal_trace", False):
            new_outputs.append(outputs[i])
            continue
        new_outputs.append(
            sampling_module.sample(
                logits=outputs[i],
                tt_out_tok=self.trace_inputs_decode[sampling_on_device][i][0],
            )
        )
```

`sampling_module.sample` replays the sampling trace that was captured by `sampling_module.capture_trace` during `_capture_decode_trace_text`. This keeps the sampling ops out of the main decode trace, allowing the decode trace to be reused with or without sampling.

> **Warning:** `self.enable_split_sampling` defaults to `True` on every new `Generator` instance. Disabling it (`generator.enable_split_sampling = False`) merges the sampling step back into the main decode trace, which means the full trace must be re-captured if the sampling mode changes.

---

**Next:** [`prefill_trace_flow.md`](./prefill_trace_flow.md)
