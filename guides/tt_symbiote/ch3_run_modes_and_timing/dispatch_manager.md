# Dispatch Manager — a class-level timing recorder that attributes every ATen operation and every module forward pass to a named module and backend.

`DispatchManager` is defined in `core/run_config.py` (line 179). It is not instantiated; all of its state is stored as class variables and all of its methods are `@staticmethod`. It therefore behaves as a process-wide singleton.

---

## Class-level state

```python
class DispatchManager:
    timings: Dict[str, Any] = {}
    _modules_in_progress: List[str] = []
    current_module_name: Optional[str] = None
```

| Attribute | Type | Purpose |
|-----------|------|---------|
| `timings` | `dict` | Top-level container. `record_timing` creates a key for each distinct `backend` value (e.g., `"TTNN"`, `"Torch"`, `"TorchModules"`) as an empty sentinel dict `{}`, but these per-backend keys are never populated with data. All actual timing data is stored under the `"TimingEntries"` key as a flat list of timing record dicts. Do not read `DispatchManager.timings["TTNN"]` expecting timing records; use `DispatchManager.timings["TimingEntries"]` or `get_timing_entries_stats()` instead. |
| `_modules_in_progress` | `list[str]` | Stack of module names currently executing. The top (last element, `[-1]`) of the list is the innermost active module. |
| `current_module_name` | `str or None` | Alias for the top of `_modules_in_progress`; updated on every push and pop. |

---

## Module-name stack: `set_current_module_name`

```python
@staticmethod
def set_current_module_name(module_name: Optional[str]) -> None:
    if module_name is None:
        assert DispatchManager._modules_in_progress, "No module name to pop"
        DispatchManager._modules_in_progress.pop()
        if DispatchManager._modules_in_progress:
            DispatchManager.current_module_name = DispatchManager._modules_in_progress[-1]
        else:
            DispatchManager.current_module_name = None
    else:
        DispatchManager._modules_in_progress.append(module_name)
        DispatchManager.current_module_name = module_name
```

Passing a string pushes onto the stack; passing `None` pops. Both `NormalRun.module_run` and `TracedRun.module_run` bracket the forward pass with push/pop on **all** execution paths (traced, non-traced, and early-return). The push happens **after** `preprocess_weights()` has already been called and timed. Specifically, in `TracedRun.module_run`, `set_current_module_name(self.module_name)` is called unconditionally before the `is_trace_enabled` check, so ATen ops dispatched during `self.forward()` on the non-trace-enabled early-return path are attributed to this module's name — not the parent's.

> **Important:** `SELRun.module_run`, `DPLRun.module_run`, and `DPLRunNoErrorProp.module_run` do **not** call `set_current_module_name` at all. ATen ops dispatched during `self.forward(...)` in those modes will see `current_module_name` from whatever parent module is already on the stack (or `None` at the top level). Do not rely on `current_module_name` for op attribution when using SEL or DPL run modes. Consequently, no module-level timing entries are written for these modes.

```python
self.preprocess_weights()           # runs before push; timed with self.module_name directly
DispatchManager.set_current_module_name(self.module_name)  # push — now current_module_name is set
self.move_weights_to_device()       # timed with current_module_name active
self.forward(...)                   # timed with current_module_name active
DispatchManager.set_current_module_name(None)              # pop
```

Because modules can be nested (a transformer block calling attention calling a linear layer), the stack ensures that intermediate ATen ops are attributed to the correct innermost module. Note: `preprocess_weights` runs outside the stack push, so any ATen ops it triggers will see `current_module_name` as the parent module (or `None` at top level), not this module's name.

> **Note:** If a module raises an exception before the `set_current_module_name(None)` pop, the stack will be left in a dirty state. The calling code does not use `try/finally` to guarantee cleanup. This is a potential source of incorrect attribution on error paths.

---

## Dispatch wrappers

### `dispatch_to_ttnn_wrapper`

```python
@staticmethod
def dispatch_to_ttnn_wrapper(func, ttnn_args, ttnn_kwargs):
    from models.experimental.tt_symbiote.core.dispatcher import dispatch_to_ttnn

    begin = time.time()
    res = dispatch_to_ttnn(func.name(), ttnn_args, ttnn_kwargs)
    end = time.time()
    func_name = f"{func.name().replace('aten::', 'TTNN::')}"
    DispatchManager.record_timing(
        "TTNN",
        "" if DispatchManager.current_module_name is None
        else DispatchManager.current_module_name + f".{func_name}",
        func_name,
        {},
        end - begin,
    )
    return res
```

Records one entry per TTNN-dispatched ATen op:

| Field | Value |
|-------|-------|
| `backend` | `"TTNN"` |
| `module_name` | `"{current_module_name}.TTNN::{op}"` or `""` if no module is active |
| `func_name` | ATen name with `aten::` replaced by `TTNN::` (e.g. `TTNN::mm.default`) |
| `duration` | Wall-clock seconds for the `dispatch_to_ttnn` call |

### `dispatch_to_torch_wrapper`

```python
@staticmethod
def dispatch_to_torch_wrapper(func, torch_args, torch_kwargs, wrap=True):
    with no_dispatch():
        func_args = tree_map(unwrap_to_torch(func), torch_args)
        func_kwargs = tree_map(unwrap_to_torch(func), torch_kwargs)
        begin = time.time()
        if can_dispatch_to_torch(func.name(), func_args, func_kwargs):
            func_res = dispatch_to_torch(func.name(), func_args, func_kwargs)
        else:
            func_res = func(*func_args, **func_kwargs)
        end = time.time()
        DispatchManager.record_timing(
            "Torch",
            "" if DispatchManager.current_module_name is None
            else DispatchManager.current_module_name + f".{func.name()}",
            func.name(),
            {},
            end - begin,
        )
        rs = tree_map(wrap_from_torch, func_res) if wrap else func_res
    return rs
```

Records one entry per PyTorch-fallback ATen op:

| Field | Value |
|-------|-------|
| `backend` | `"Torch"` |
| `module_name` | `"{current_module_name}.{aten_op}"` or `""` |
| `func_name` | Raw ATen name (e.g. `aten::mm.default`) |
| `duration` | Wall-clock seconds |

See [`no_dispatch()` in run_modes.md](run_modes.md#the-no_dispatch-context-manager) for why this prevents re-entry into `__torch_dispatch__`.

The `wrap` parameter (default `True`) controls whether the result is re-wrapped as `TorchTTNNTensor`. `LightweightRun` passes `wrap=False`.

### Hidden per-op overhead entry: `can_dispatch_to_ttnn`

`NormalRun.torch_dispatch` records an **additional** timing entry for every intercepted ATen operation, before either wrapper is called:

| Field | Value |
|-------|-------|
| `backend` | `"TTNN"` |
| `module_name` | `"{current_module_name}.can_dispatch_to_ttnn"` or `""` |
| `func_name` | `"can_dispatch_to_ttnn"` |
| `duration` | Wall-clock seconds for the `can_dispatch_to_ttnn` check |

This entry is **only** produced by `NormalRun.torch_dispatch`. `NormalRunWithFallback` overrides `torch_dispatch` entirely and does **not** record this entry — it calls `can_dispatch_to_ttnn` inline without any timing instrumentation. The entry is also absent in `SELRun`, `DPLRun`, `DPLRunNoErrorProp`, `LightweightRun`, `CPU`, and `TracedRun` (which inherits `LightweightRun.torch_dispatch`). When summing `backend == "TTNN"` entries to estimate total TTNN time under `NORMAL` mode, filter out rows where `func_name == "can_dispatch_to_ttnn"` to avoid inflating the result. No such filtering is needed for `NORMAL_WITH_FALLBACK`.

### Additional timing recorded by `NormalRun.module_run`

Beyond ATen-level wrappers, `NormalRun.module_run` records four timing entries per module call. `TracedRun.module_run` records the same four entries **only on the traced execution path** (cache hit or cache miss). On the non-trace-enabled early-return path (`not is_trace_enabled(self)` or `_TRACE_RUNNING` is already `True`), `TracedRun` records the `_preprocess_weights`, `_move_weights_to_device`, and `_forward` entries but **does not** record the `TorchModules` entry; it returns early. For `SELRun`, `DPLRun`, and `DPLRunNoErrorProp`, see the [Important note in the `set_current_module_name` section above](#module-name-stack-set_current_module_name).

> **Important:** `NormalRunWithFallback` overrides `module_run` with its own implementation that contains no `record_timing` calls and no `set_current_module_name` calls. Modules running under `NORMAL_WITH_FALLBACK` mode produce **no module-level timing entries** — not `_preprocess_weights`, not `_move_weights_to_device`, not `_forward`, and not `TorchModules`. Only ATen-level entries from `dispatch_to_ttnn_wrapper` and `dispatch_to_torch_wrapper` (called within `self.forward`) will be present.

| `func_name` | `backend` | What it measures | Paths recorded |
|-------------|-----------|-----------------|----------------|
| `{ClassName}_preprocess_weights` | `"TTNN"` | Time to call `self.preprocess_weights()` | `NormalRun` and `TracedRun` only |
| `{ClassName}_move_weights_to_device` | `"TTNN"` | Time to call `self.move_weights_to_device()` | `NormalRun` and `TracedRun` only |
| `{ClassName}_forward` | `"TTNN"` | Time from just before dispatch decision to after trace execute or normal `self.forward()` | `NormalRun` and `TracedRun` only |
| `{ClassName}_capture_trace` | `"TTNN"` | Time for the full trace capture (warm-up + `begin/end_trace_capture`) | `TracedRun` cache-miss path only |
| `{ClassName}` | `"TorchModules"` | Total wall-clock time for the entire `module_run` call | `NormalRun` always; `TracedRun` traced-path only (not non-trace-enabled early-return); never for `NormalRunWithFallback` |

The `"TorchModules"` entry is the primary signal for identifying slow modules. On a `TracedRun` cache miss, an additional `{ClassName}_capture_trace` entry appears in the flat CSV. However, the `{ClassName}_forward` timing window (`begin` is set before the `_TRACE_RUNNING` lock check; `end` is set after the cache-miss branch completes) **already encompasses the full `_capture_trace` call**. Therefore, `_forward` on a cache-miss run is not comparable to `_forward` on a cache-hit run even after filtering the `_capture_trace` entry. To compare steady-state forward times, use only rows where no `_capture_trace` entry exists for the same `(module_name, func_name)` pair (i.e., cache-hit runs only).

---

## `record_timing`

```python
@staticmethod
def record_timing(backend: str, module_name: str, func_name: str, attrs: dict, duration: float) -> None:
    if backend not in DispatchManager.timings:
        DispatchManager.timings[backend] = {}
    if "TimingEntries" not in DispatchManager.timings:
        DispatchManager.timings["TimingEntries"] = []
    DispatchManager.timings["TimingEntries"].append({
        "attrs": attrs,
        "module_name": module_name,
        "func_name": func_name,
        "duration": duration,
        "backend": backend,
    })
```

Every call appends one dict to `DispatchManager.timings["TimingEntries"]`. The `attrs` field is always `{}` in current call sites; it is reserved for future use.

### Resetting

```python
DispatchManager.clear_timings()  # sets DispatchManager.timings = {}
```

Call this at the start of a measurement window to avoid accumulating data from warm-up runs.

---

## Saving timing data: `save_stats_to_file`

```python
DispatchManager.save_stats_to_file("run_timing.csv")
```

The method:

1. Asserts the filename ends with `.csv`.
2. Builds a `pandas.DataFrame` from `timings["TimingEntries"]` and writes it to `{filename}` with `index=True`.
3. Builds a pivot table indexed by `(func_name, module_name)` with `backend` as columns and `sum(duration)` as values. Adds derived columns:

| Column | Source |
|--------|--------|
| `Total_Duration` | Sum of all backend durations for that `(func_name, module_name)` pair |
| `Min_Duration` | Minimum single-backend duration across all backends |
| `Max_Duration` | Maximum single-backend duration across all backends |
| `Row_Count` | Total number of timing entries merged into this row |

4. Writes the pivot table to `{filename_without_extension}_pivot.csv`.
5. If the pivot table has a `"TorchModules"` column, prints the top 30 **class names** (`func_name`) sorted by total `TorchModules` duration in descending order. Note: this groups by `func_name` (class name, e.g. `TransformerBlock`), not by `module_name` (instance path, e.g. `model.transformer.h.0`). To find the slowest individual module instance use `get_timing_entries_stats()` as shown in Step 5.

### Output file summary

| File | Contents |
|------|---------|
| `run_timing.csv` | Flat log: one row per recorded operation, with `attrs`, `module_name`, `func_name`, `duration`, `backend` columns. |
| `run_timing_pivot.csv` | Aggregated view: rows are `(func_name, module_name)` pairs; columns are backends plus `Total_Duration`, `Min_Duration`, `Max_Duration`, `Row_Count`. |

---

## Reading timing output to identify bottleneck layers

### Step 1: look at the flat CSV for raw data

The flat `run_timing.csv` shows every dispatch event in execution order. Filter on `backend == "TorchModules"` to see module-level durations. Sort by `duration` descending to find the slowest individual module invocations.

### Step 2: use the pivot CSV for aggregated bottleneck analysis

In `run_timing_pivot.csv`:

- The `TorchModules` column (if present) holds the total wall-clock time for each module across all invocations.
- The `TTNN` column holds the sum of all individual TTNN ATen-op times for that `(func_name, module_name)`.
- The `Torch` column holds the sum of all PyTorch-fallback times.

Sort by `Total_Duration` descending to find the highest-cost layers.

### Step 3: interpret the module name format

`module_name` values have the pattern:

```
{module_name}.{backend_prefix}::{aten_op}
```

For example: `model.transformer.h.0.attn.TTNN::mm.default`. The module name passed to `DispatchManager.set_current_module_name` is everything before the final `.<backend_prefix>::<op>` suffix — in this example `model.transformer.h.0.attn`. Module names themselves may contain dots (e.g. `model.transformer.h.0`), so splitting on the first `.` would give only the top-level component; always split on the last occurrence of `.TTNN::` or `.aten::` to recover the module name. The suffix is the specific ATen operation.

Entries recorded at the module-run level (preprocess, weights, forward, TorchModules) use plain `module_name` without an op suffix.

### Step 4: identify dispatch imbalance

Compare the `TTNN` and `Torch` backend durations for the same `func_name`. A high `Torch` duration relative to `TTNN` for a dispatchable op may indicate a conversion overhead or a fallback being triggered unexpectedly. A high `TTNN` duration for a non-dispatchable op is expected.

### Step 5: use `get_timing_entries_stats()` for interactive analysis

```python
df = DispatchManager.get_timing_entries_stats()
# df is a pandas DataFrame with columns: attrs, module_name, func_name, duration, backend
# NOTE: For TorchModules entries, func_name = the class name (e.g. "TransformerLayer"),
# while module_name = the instance path (e.g. "model.transformer.h.0").
# Group by module_name (not func_name) to find the slowest individual module instances.
slow_modules = (
    df[df["backend"] == "TorchModules"]
    .groupby("module_name")["duration"]
    .sum()
    .sort_values(ascending=False)
    .head(20)
)
print(slow_modules)
```

---

## Summary of `DispatchManager` API

| Method | Description |
|--------|-------------|
| `set_current_module_name(name)` | Push `name` onto the module stack; pass `None` to pop. |
| `dispatch_to_ttnn_wrapper(func, args, kwargs)` | Invoke TTNN dispatch and record timing. |
| `dispatch_to_torch_wrapper(func, args, kwargs, wrap)` | Invoke PyTorch fallback and record timing. |
| `record_timing(backend, module_name, func_name, attrs, duration)` | Append one timing entry. |
| `clear_timings()` | Reset all recorded data. |
| `get_timing_entries_stats()` | Return all entries as a `pandas.DataFrame`. |
| `save_stats_to_file(filename.csv)` | Write flat CSV and pivot CSV; print top-30 module summary. |

---

Next: Chapter 4
