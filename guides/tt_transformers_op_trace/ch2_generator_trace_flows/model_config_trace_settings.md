# Model Configuration and Trace Settings

This file covers the configuration layer that governs trace capture across the tt-transformers stack: the `get_supported_trace_region_size` lookup table in `models/tt_transformers/demo/trace_region_config.py`, the `get_trace_prefill_supported_seq_lens` per-device and per-model dictionaries in `models/tt_transformers/tt/model_config.py`, the `num_command_queues: 1` device parameter in `models/tt_transformers/demo/simple_text_demo.py`, and the path through which `trace_region_size` reaches the device at runtime. By the end you will know where to look when a model runs out of trace region memory and how to add support for a new model or device configuration.

---

## `get_supported_trace_region_size` in `trace_region_config.py`

`models/tt_transformers/demo/trace_region_config.py` contains a nested dictionary (`trace_region_size_dict`) that maps `(base_model_name, device_name)` to a byte size:

```python
trace_region_size_dict = {
    "Llama-3.1-8B": {
        "N150": 25000000,
        "N300": 38000000,
        "T3K": 50000000,
        "TG": 50000000,
    },
    "Llama-3.3-70B": {
        "T3K": 80000000,   # same as TG; 3.3-70B uses a smaller decode trace than 3.1-70B
        "TG": 80000000,
        ...
    },
    "Llama-3.1-70B": {
        "T3K": 90000000,
        "TG": 90000000,
        ...
    },
    ...
}
```

> **Note — why Llama-3.3-70B T3K is 80 MB, not 90 MB:** `Llama-3.3-70B` and `Llama-3.1-70B` share the same parameter count and architecture width, but their decode-graph footprints differ slightly. Verified against `models/tt_transformers/demo/trace_region_config.py`: the canonical entry for `Llama-3.3-70B` is **80 MB** uniformly across all device types (T3K, TG, and all Blackhole variants), while `Llama-3.1-70B` uses **90 MB**. The 10 MB headroom difference reflects profiled trace sizes for each model checkpoint; both values are intentional. An earlier draft of this guide contained a stale 30 MB value for `Llama-3.3-70B` T3K — that figure has been corrected here to 80 MB.

The `get_supported_trace_region_size(request, mesh_device)` function looks up `(base_model_name, device_name)` and returns the byte value, or `None` if no entry exists. The two keys are obtained as follows:

- `base_model_name` is derived from the `HF_MODEL` environment variable: the last path component is extracted and then passed through `get_base_model_name` to strip the instruction-tune suffix (e.g. `"Llama-3.1-8B-Instruct"` → `"Llama-3.1-8B"`).
- `device_name` is derived from the `mesh_device` parameter (the number of devices, passed as an integer or `(rows, cols)` tuple from the pytest fixture's `param`) together with the `data_parallel` value read from the test's `callspec.params`. The helper `device_name_based_on_data_parallel` divides total device count by `data_parallel` and calls `get_mesh_device_name` on the result to produce a string such as `"N150"` or `"T3K"`. The `MESH_DEVICE` environment variable is only consulted inside `get_mesh_device_name` for the special Blackhole `"P100"` case; for all other configurations the device name comes purely from the device count and architecture.

### How it flows to the device fixture

The top-level `conftest.py` at the tt-metal repository root imports `get_supported_trace_region_size` and calls it inside the `mesh_device` pytest fixture:

```python
override_trace_region_size = get_supported_trace_region_size(request, param)
if override_trace_region_size:
    device_params["trace_region_size"] = override_trace_region_size
    logger.info(f"Overriding trace region size to {override_trace_region_size}")
```

`device_params` is then passed to `ttnn.open_mesh_device`, which allocates the trace region on each Tensix chip during device initialisation. Because the region is allocated at open time, its size cannot be changed while the device is open.

### What happens when the region is exhausted

If the accumulated byte size of all captured traces exceeds the allocated region, the TTNN runtime raises an error at the point where `ttnn.begin_trace_capture` tries to record the next graph. The error message reports the required size. The README for `tt_transformers` notes:

> For models not listed in `get_supported_trace_region_size`, the default trace region size will be used as set in `simple_text_demo.py`. The default trace region size may not be sufficient for such a model and there will be a helpful error message that informs the required trace region size, which can be overridden by setting the `trace_region_size` argument in the demo.

To resolve the error: add the model and device to `trace_region_size_dict` in `trace_region_config.py` with a byte value large enough to hold all decode and prefill traces for that configuration.

> **Warning:** When using data-parallel configurations (e.g., T3K DP=8 where each submesh is an N150), the `trace_region_size` is applied to **each submesh independently**. Setting an oversized value for the aggregate configuration risks out-of-memory failures on each individual N150. This is why per-device entries in `trace_region_size_dict` exist (e.g., the `"N150": 25000000` entry for Llama-3.1-8B is separate from `"T3K": 50000000`).

---

## `get_trace_prefill_supported_seq_lens` in `model_config.py`

`get_trace_prefill_supported_seq_lens` in `models/tt_transformers/tt/model_config.py` returns the list of padded sequence lengths for which prefill tracing is enabled on the current device. It is structured as a two-level fallback:

```python
default_supported_seq_lens = {
    "N150": [128],
    "N300": [128, 1024],
    "T3K": [128, 1024],
    "TG": [128, 1024],
    "P150": [128, 1024],
    "P300": [128, 1024],
    ...
}

model_specific_supported_seq_lens = {
    "Llama-3.1-8B": {
        "N150": [128, 1024],          # overrides default N150 = [128]
        "N300": [128, 1024, 2048, 4096, 8192],
        "T3K": [128, 1024, 2048, 4096, 8192],
        "TG":  [128, 1024, 2048, 4096, 8192],
    },
    "Llama-3.1-70B": {
        "T3K": [128, 1024, 2048, 4096, 8192],
        "TG":  [128, 1024, 2048, 4096, 8192],
    },
    "Llama-3.2-3B": {
        "N150": [],                   # tracing disabled on N150 for this model
    },
    ...
}
```

The lookup logic is:

```python
result = model_specific_supported_seq_lens.get(model_name, {}).get(
    device_name, default_supported_seq_lens.get(device_name)
)
```

If a model-specific entry exists for the `(model_name, device_name)` pair it takes precedence; otherwise the default for the device is used.

> **Key insight:** N150 defaults to `[128]` — only 128-token prefill traces — because its smaller DRAM trace region budget limits how many seq-len traces can fit simultaneously. The `trace_region_size_dict` entry for N150 is **25 MB** (compared to 38 MB for N300 and 50 MB for T3K/TG). That 25 MB budget is what constrains the number of traceable seq-len entries, not on-chip SRAM (L1), which is a separate, smaller scratchpad used for live tile buffering during op execution and is not where captured trace graphs are stored. Llama-3.1-8B on N150 is explicitly given `[128, 1024]` because it has been validated to fit within that 25 MB budget. Conversely, N300 and multi-chip configurations have larger trace region budgets and can afford the larger lists.

The returned list is then capped by `cap_seq_lens_to_max_prefill_chunk_size` to remove lengths that exceed the model's `max_prefill_chunk_size`, which prevents capturing a trace for a length the model cannot actually process.

The result is stored at `ModelArgs.__init__` time as `self.trace_prefill_supported_seq_lens` and consulted by `can_enable_trace` on every prefill call.

---

## `num_command_queues: 1` in `simple_text_demo.py`

The pytest fixture parameters for the main demo test are:

```python
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
)
```

`num_command_queues: 1` opens the device with a single command queue (`cq_id=0`). All three trace API calls (`begin_trace_capture`, `end_trace_capture`, `execute_trace`) are issued with `cq_id=0`.

> **Warning:** If the device is opened with `num_command_queues: 2` (as is done in `simple_vision_demo.py`), a second command queue becomes available for host-device transfers overlapped with compute. However, TTNN trace capture currently records a graph tied to a specific command queue. If you mix `cq_id=0` and `cq_id=1` operations in a traced region the results are undefined. The text-only demo therefore uses a single queue to keep the trace semantics well-defined.

---

## Summary of Configuration Touchpoints

| Configuration | File | Consumed by |
|---|---|---|
| `trace_region_size` byte budget | `trace_region_config.py::get_supported_trace_region_size` | Root `conftest.py` → `ttnn.open_mesh_device` |
| Default `trace_region_size` | `simple_text_demo.py` `device_params` parametrize | Root `conftest.py` (overridden per model) |
| `num_command_queues: 1` | `simple_text_demo.py` `device_params` parametrize | `ttnn.open_mesh_device` |
| Allowed prefill seq lens | `model_config.py::get_trace_prefill_supported_seq_lens` | `ModelArgs.trace_prefill_supported_seq_lens` |
| Per-call gate | `model_config.py::can_enable_trace` | `prefill_forward_text` per user |

---

**Next:** [Chapter 3 — Model Warm-Up and Its Relationship to Trace Capture](../ch3_warmup/index.md)
