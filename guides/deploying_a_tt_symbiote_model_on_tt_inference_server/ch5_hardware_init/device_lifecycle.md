# Device Lifecycle

This file traces the exact sequence of calls the server worker performs to open and close the Tenstorrent mesh device, and explains every knob that controls how the device is configured.

## Who Opens the Device

`ttnn.open_mesh_device()` is called inside `open_mesh_device()` in `tt-vllm-plugin/tt_vllm_plugin/worker/tt_worker.py`. That function is invoked during `TTWorker` initialization, before the worker signals readiness to the vLLM engine and before `initialize_vllm_model` is ever called.

The high-level call sequence inside `TTWorker.__init__` looks like this:

```python
# Inside TTWorker.__init__ (simplified)
self._configure_visible_devices()   # reads TT_VISIBLE_DEVICES from env and applies device filter
self._apply_fabric_config()          # calls set_fabric() if needed
self.mesh_device = open_mesh_device(
    mesh_shape=get_mesh_grid(),
    dispatch_core_config=self._dispatch_core_config(),
    trace_region_size=self._trace_region_size(),
)
```

The model never sees this code path. By the time `initialize_vllm_model(mesh_device, ...)` is called, `self.mesh_device` is already fully initialized and passed in as an argument.

## How the Mesh Grid Is Determined

`get_mesh_grid()` in `tt_worker.py` reads the `MESH_DEVICE` environment variable and converts it to a `ttnn.MeshShape`. The accepted values and their translations are:

| `MESH_DEVICE` value | Resulting `ttnn.MeshShape` |
|---|---|
| `"N150"` | `ttnn.MeshShape(1, 1)` |
| `"N300"` | `ttnn.MeshShape(1, 2)` |
| `"T3K"` | `ttnn.MeshShape(1, 8)` |
| `"(1,8)"` | `ttnn.MeshShape(1, 8)` |
| `"(2,4)"` | `ttnn.MeshShape(2, 4)` |
| `"(8,4)"` | `ttnn.MeshShape(8, 4)` (Galaxy) |

When `MESH_DEVICE` is a device-type string such as `"T3K"`, `get_mesh_grid()` looks up a hardcoded mapping to find the canonical mesh shape for that hardware. When `MESH_DEVICE` is already a tuple string such as `"(1,8)"`, it is parsed directly. The resulting shape is passed as the `mesh_shape` argument to `ttnn.open_mesh_device()`:

```python
rows, cols = get_mesh_grid()  # e.g., (1, 8) for T3K
mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(rows, cols),
    ...
)
```

## Fabric Configuration

Before the device is opened, `set_fabric()` is called to configure the inter-chip fabric topology. The fabric topology controls how DRAM and CCL (Collective Communication Library) operations route data across chips:

- **6U Galaxy clusters** (32-chip, `(8,4)` mesh): `set_fabric()` configures a **2D mesh ring** topology. The Galaxy board's physical Ethernet links form a torus-style interconnect across both mesh dimensions. CCL operations route data along the column dimension (the ring within each column of 8 chips) while the dispatch layer routes across rows of 4 chips. Do not assume a flat 1D ring when writing CCL patterns for Galaxy — the fabric is aware of both dimensions.
- **All other configurations** (N300, T3K, standalone N150): `set_fabric()` configures **standard 1D** fabric.

On worker shutdown, `reset_fabric()` is called to tear down fabric configuration before `ttnn.close_mesh_device()` completes the teardown sequence.

```python
# Simplified teardown in TTWorker.shutdown()
reset_fabric()
ttnn.close_mesh_device(self.mesh_device)
self.mesh_device = None
```

The model must not call `set_fabric()` or `reset_fabric()` directly. These calls modify global process state and must bracket the full device lifetime, not individual inference calls.

## Dispatch Core Axis

The dispatch core axis controls which dimension of the Tensix grid is reserved for command dispatch. It is read from `override_tt_config["dispatch_core_axis"]` and accepts two string values:

| Value | Meaning |
|---|---|
| `"row"` | Reserve an entire row of Tensix cores for dispatch |
| `"col"` | Reserve an entire column of Tensix cores for dispatch |

The string is translated to a `ttnn.DispatchCoreConfig` object and passed to `ttnn.open_mesh_device()`:

```python
axis = override_tt_config.get("dispatch_core_axis", "row")
dispatch_core_config = ttnn.DispatchCoreConfig(
    ttnn.DispatchCoreAxis.ROW if axis == "row" else ttnn.DispatchCoreAxis.COL
)
mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(rows, cols),
    dispatch_core_config=dispatch_core_config,
    ...
)
```

Choosing `"col"` frees up an extra row of compute cores, which can improve throughput for compute-bound models. The default `"row"` setting is safer for all hardware configurations. The model should never read or write `override_tt_config["dispatch_core_axis"]` — it is a worker-level concern.

## Trace Region Size

`ttnn` supports execution traces: pre-compiled, replayed command sequences that amortize host-side dispatch overhead across many inference steps. The trace region is a chunk of L1 memory reserved exclusively for storing compiled traces.

`override_tt_config["trace_region_size"]` (an integer, in bytes) is passed directly to `ttnn.open_mesh_device()` as `trace_region_size`:

```python
trace_region_size = override_tt_config.get("trace_region_size", 1_500_000)
mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(rows, cols),
    dispatch_core_config=dispatch_core_config,
    trace_region_size=trace_region_size,
)
```

If the model attempts to capture a trace that requires more L1 than `trace_region_size` bytes, `ttnn` will raise an out-of-memory error at capture time. The correct fix is to increase `trace_region_size` in the deployment configuration, not to call `ttnn.open_mesh_device()` again from inside the model.

## `TT_VISIBLE_DEVICES`

`TT_VISIBLE_DEVICES` is a comma-separated list of physical Tenstorrent device IDs (as enumerated by the `tt-smi` tool) that are visible to the current process. For example, `TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` makes all eight device IDs of a T3K visible. T3K consists of 4 N300 cards (2 Wormhole chips per card = 8 Wormhole chips total, 8 device IDs), and each Wormhole chip presents as a single device ID — so `tt-smi` reports 8 device IDs (0, 1, 2, 3, 4, 5, 6, 7) for a T3K host.

The operator sets `TT_VISIBLE_DEVICES` in the process environment before the worker process is launched; the server workflow's container or process launch configuration is responsible for placing it there. `TTWorker.__init__` then calls `_configure_visible_devices()` to read the existing value and apply the device filter — the worker does not override or re-set the variable, it reads it. This ordering is what allows multiple independent worker processes to partition a multi-chip host without interfering with each other. The model must not modify `TT_VISIBLE_DEVICES` — changing it after the worker has already read it during init has no effect on `ttnn`'s view of available devices and may confuse other subprocesses.

---

**Next:** [`model_init_responsibilities.md`](./model_init_responsibilities.md)
