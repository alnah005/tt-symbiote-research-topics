# Signpost Profile: User-Defined Profiling Zones

**Env var:** `TTLANG_SIGNPOST_PROFILE=1`
**Module:** `python/ttl/_src/signpost_profile.py`
**Pipeline hook:** `_run_signpost_profile()` in `ttl_api.py`

Where auto-profiling instruments every source line automatically, signpost profiling lets the programmer mark specific regions of interest. This is useful for measuring coarse-grained phases (e.g., "load weights", "compute attention", "write output") without the noise of per-line instrumentation.

## The `ttl.signpost` Operator

Defined in `python/ttl/operators.py` with the `@syntax("signpost")` decorator:

```python
@syntax("signpost")
def signpost(name: str):
    """
    Mark a profiling scope visible in Tracy.

    Use as a context manager to wrap a region of interest:

        with ttl.signpost("my_region"):
            ...

    Generates a DeviceZoneScopedN in the emitted C++ code. Enable
    TTLANG_SIGNPOST_PROFILE=1 to collect per-region cycle counts.
    """
    return ttl.signpost(name)
```

Usage in a TT-Lang kernel:

```python
@ttl.pykernel_gen(grid=(8, 8))
def my_kernel(input_tensor, output_tensor):
    with ttl.signpost("load_phase"):
        block = cb_in.wait()
        ttl.copy(input_tensor[r, c], block).wait()

    with ttl.signpost("compute_phase"):
        result = block * block
        out_block.store(result)

    with ttl.signpost("store_phase"):
        ttl.copy(out_block, output_tensor[r, c]).wait()
```

The `with` statement is handled by the TT-Lang AST compiler (see [Chapter 2](../ch2_compilation_pipeline/index.md)), which emits a `ttl.signpost` MLIR op with `is_end=False` at entry and `is_end=True` at exit.

## Lowering Through MLIR

The signpost op follows the same lowering path as auto-profile signposts:

1. **AST to MLIR:** The `@syntax("signpost")` decorator causes the AST compiler to emit a `ttl.signpost` op.
2. **`ttl-lower-signpost-to-emitc`:** This MLIR pass (part of the standard compilation pipeline, listed in [Chapter 2](../ch2_compilation_pipeline/index.md)) converts `ttl.signpost` ops into `DeviceZoneScopedN` C++ macro calls.
3. **Device execution:** The macro calls produce `ZONE_START`/`ZONE_END` entries in the device profiler CSV.

User-defined signposts are prefixed with `ttl_` to distinguish them from tt-metal internal zones. For example, `ttl.signpost("load_phase")` becomes zone name `ttl_load_phase` in the CSV.

## Runtime: `_run_signpost_profile()`

After kernel execution, `_run_signpost_profile()` in `ttl_api.py`:

1. Checks `TT_METAL_PROFILER_MID_RUN_DUMP=1` and warns if unset (profiler data may be stale).
2. Calls `ttnn.ReadDeviceProfiler(device)` to flush data.
3. Resolves the logs path: `$TT_METAL_HOME/generated/profiler/.logs/`.
4. Delegates to `signpost_profile.run(logs_path)`.

## `parse_signpost_zones()`

The core parsing function reads the device profiler CSV and extracts only user-defined zones:

```python
def parse_signpost_zones(csv_path: Path) -> List[Tuple[str, str, int]]:
    """Returns: List of (display_name, thread, cycles) tuples."""
```

Key details:

- **Filter:** Only zones whose name starts with the `_USER_PREFIX = "ttl_"` are processed. All tt-metal internal zones and auto-profile signposts are skipped.
- **Matching:** `ZONE_START`/`ZONE_END` pairs are matched using the composite key `"{thread}_{zone_name}"`.
- **Display name:** The `ttl_` prefix is stripped for display (e.g., `ttl_load_phase` becomes `load_phase`).
- **Duration:** Computed as `ZONE_END.timestamp - ZONE_START.timestamp` in device cycles.

Returns a list of `(display_name, thread, cycles)` tuples in CSV encounter order.

## `format_report()`

Aggregates results by `(name, thread)` pair and prints a tabular summary:

```python
def format_report(zones: List[Tuple[str, str, int]]) -> str:
```

The report columns are:

| Column | Description |
|--------|-------------|
| `NAME` | Signpost display name (prefix stripped) |
| `THREAD` | RISC thread (NCRISC, BRISC, TRISC_0, etc.) |
| `COUNT` | Number of times this zone was entered |
| `TOTAL` | Sum of all durations in cycles |
| `AVG` | Integer average: `total // count` |
| `MIN` | Minimum single-entry duration |
| `MAX` | Maximum single-entry duration |

Example output:

```
================================================================================
SIGNPOST PROFILE
================================================================================

  NAME              THREAD        COUNT        TOTAL        AVG        MIN        MAX
  ----------------- ------------ ------ ------------ ---------- ---------- ----------
  load_phase        NCRISC            1       12,345     12,345     12,345     12,345
  compute_phase     TRISC_0           1       45,678     45,678     45,678     45,678
  store_phase       BRISC             1        8,901      8,901      8,901      8,901

================================================================================
```

When a signpost zone is entered multiple times (e.g., inside a loop), `COUNT` will be greater than 1 and the `MIN`/`MAX` spread reveals iteration variance.

## Relationship to Auto-Profile

Signpost profiling and auto-profiling are independent modes that can be enabled simultaneously. The key differences:

| Aspect | Auto-Profile | Signpost Profile |
|--------|-------------|-----------------|
| Granularity | Per source line | User-defined regions |
| Instrumentation | Automatic (every line) | Manual (`with ttl.signpost(...)`) |
| Signpost prefix | `<kernel>_L<lineno>` | `ttl_<name>` |
| Report format | Annotated source listing | Aggregated table |
| CB attribution | Yes (via `cb_flow_graph.json`) | No |
| Roofline analysis | Yes | No |

Both modes filter the same CSV. Auto-profile signposts lack the `ttl_` prefix, so `parse_signpost_zones()` naturally ignores them. Conversely, `parse_device_profile_csv()` only processes signposts registered with the `SourceLineMapper`, so user signposts are ignored there.

## Example Usage

```bash
export TT_METAL_DEVICE_PROFILER=1
export TTLANG_SIGNPOST_PROFILE=1
python my_kernel.py
```

For combined analysis with Perfetto visualization:

```bash
export TT_METAL_DEVICE_PROFILER=1
export TTLANG_SIGNPOST_PROFILE=1
export TTLANG_PERF_SERV=1
python my_kernel.py
```

This prints the signpost table to stdout and then launches the Perfetto trace server.

---

**Next:** [`perf_dump_and_perfetto.md`](./perf_dump_and_perfetto.md)
