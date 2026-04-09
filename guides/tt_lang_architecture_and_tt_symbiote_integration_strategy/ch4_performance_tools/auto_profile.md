# Auto-Profile: Per-Source-Line Cycle Counts

**Env var:** `TTLANG_AUTO_PROFILE=1`
**Module:** `python/ttl/_src/auto_profile.py`
**Pipeline hook:** `_run_profiling_pipeline()` in `ttl_api.py`

Auto-profiling is TT-Lang's most detailed profiling mode. It automatically instruments every source line in a kernel with signpost pairs, then maps device cycle counts back to the original Python source. The result is a terminal report that shows, for each line, how many cycles it consumed and what fraction of the thread's total time it represents.

## How It Works

### 1. Compile-Time Instrumentation

During AST-to-MLIR lowering (see [Chapter 2](../ch2_compilation_pipeline/index.md)), the `TTLCompiler` in `ttl_ast.py` checks `is_auto_profile_enabled()`. When enabled, it calls `_emit_line_signpost_if_needed(node)` at every AST node. This method:

1. Extracts the file line number from the AST node.
2. If the line has changed since the last signpost, closes the previous signpost with an `is_end=True` call and opens a new one.
3. Registers the signpost name with the global `SourceLineMapper` via `register_signpost(name, lineno, source)`.

Signpost names are constructed directly in `TTLCompiler._emit_line_signpost_if_needed()` and `_emit_op_signposts()` within `ttl_ast.py`. There are two levels:

**Line-level signposts** follow the pattern:

```
<kernel_name>_L<lineno>
```

For example, `compute_L52` marks line 52 of the compute kernel. These are emitted at line boundaries — when the AST walker moves to a new source line, it closes the previous signpost and opens a new one. The construction is `f"{self.name}_L{file_lineno}"` (`ttl_ast.py` line 232).

**Op-level signposts** instrument individual CB operations within a line:

```
<kernel_name>_L<lineno>_<op_name>
<kernel_name>_L<lineno>_implicit_<op_name>
```

For example, `dm_read_L52_cb_wait` or `dm_write_L52_implicit_cb_pop`. The `implicit_` prefix marks ops inserted by the compiler rather than written by the user. The construction is `f"{self.name}_L{file_lineno}_{prefix}{op_name}"` (`ttl_ast.py` line 264).

> **Note:** `auto_profile.py` defines a `generate_signpost_name()` function (line 116) that would produce a `_C{col}` column suffix, but this function is dead code — it is never called anywhere in the codebase. The actual signpost names never include a column offset.

At the end of each kernel function body, `_close_final_signpost()` emits the closing `is_end=True` for the last open signpost.

### 2. MLIR Pass: Signpost Lowering

The compilation pipeline includes `ttl-lower-signpost-to-emitc`, which converts `ttl.signpost` MLIR ops into `DeviceZoneScopedN` C++ macro calls. These are the instrumentation points that the tt-metal device profiler recognizes, recording `ZONE_START` and `ZONE_END` timestamps in the CSV.

Additionally, when auto-profiling is enabled, the pipeline injects `ttl-dump-cb-flow-graph` to produce a `cb_flow_graph.json` file. This JSON describes the circular buffer topology — which CBs connect which kernels, and where DMA barriers and consumer waits occur. The output path is either `$TT_METAL_HOME/generated/profiler/.logs/cb_flow_graph.json` or the directory of `$TTLANG_PROFILE_CSV`.

### 3. Runtime: CSV Parsing and Report Generation

After kernel execution, `_run_profiling_pipeline()` in `ttl_api.py`:

1. Calls `ttnn.ReadDeviceProfiler(device)` to flush profiler data.
2. Locates `profile_log_device.csv` at `$TT_METAL_HOME/generated/profiler/.logs/` (or `$TTLANG_PROFILE_CSV`).
3. Calls `parse_device_profile_csv(csv_path, line_mapper)`.
4. Loads the CB flow graph and builds attribution maps.
5. Calls `print_profile_report(...)`.

## `SourceLineMapper`

The `SourceLineMapper` class maintains the mapping between signpost names and source locations:

```python
class SourceLineMapper:
    def __init__(self):
        self.signpost_to_line: Dict[str, Tuple[int, str]] = {}
        self.source_lines: List[str] = []
        self.line_offset: int = 0

    def register_signpost(self, signpost_name: str, lineno: int, source: str): ...
    def get_line_info(self, signpost_name: str) -> Optional[Tuple[int, str]]: ...
```

A global instance is maintained via `_global_line_mapper` and accessed through `get_line_mapper()`. The AST compiler populates it during lowering; the CSV parser queries it during report generation.

## `parse_device_profile_csv()`

This function reads the device profiler CSV and extracts timing data from signpost zones:

```python
def parse_device_profile_csv(
    csv_path: Path, line_mapper: SourceLineMapper
) -> List[ProfileResult]:
```

The CSV format has columns including:
- Column 3: RISC thread (`NCRISC`, `BRISC`, `TRISC_0`, `TRISC_1`, `TRISC_2`)
- Column 5: timestamp (cycle count)
- Column 10: signpost/zone name
- Column 11: zone type (`ZONE_START` or `ZONE_END`)

The parser matches `ZONE_START`/`ZONE_END` pairs keyed by `"{thread}_{signpost}"`, computes the duration, and looks up the source line via the `SourceLineMapper`. Results are returned as `ProfileResult` objects sorted by line number:

```python
class ProfileResult:
    signpost: str        # e.g., "compute_L52_cb_wait"
    thread: str          # e.g., "TRISC_0"
    cycles: int          # Duration in device cycles
    lineno: int          # Source line number
    source: str          # Source line text
    op_name: str | None  # Parsed op name (e.g., "cb_wait"), None for line-only
    implicit: bool       # True if compiler-inserted (e.g., implicit_cb_pop)
```

The `parse_signpost_name()` function extracts the op name and implicit flag from the signpost string using regex:

```
"compute_L52"              -> (None, False)       # line-only
"dm_read_L52_cb_wait"      -> ("cb_wait", False)  # explicit op
"dm_write_L52_implicit_cb_pop" -> ("cb_pop", True) # compiler-inserted
```

## CB Flow Graph Attribution

The auto-profiler goes beyond simple per-line cycle counts by attributing CB synchronization stalls to their root causes. This uses two maps built from `cb_flow_graph.json`:

### `build_cb_wait_to_dma_map()`

Maps consumer `cb_wait` locations to the DMA barrier or compute producer they are waiting on:

- **Read direction** (DMA reads into CB, compute consumes): compute's `cb_wait` maps to the DMA read barrier, labeled `"DMA"`.
- **Write direction** (compute produces, DMA writes from CB): DM write's `cb_wait` maps to the compute producer, labeled `"compute"`.

Returns: `Dict[(kernel, line) -> (source_kernel, source_line, cb_index, label)]`

### `build_dma_producer_to_cb_map()`

Maps DMA read barrier locations to their CB index, enabling color-coded source line highlighting.

Returns: `Dict[(kernel, line) -> cb_index]`

In the report, these show up as indented remarks below the source line:

```
52    23.4%  12,345     block = cb_in.wait()
                                              ├─ 8,200 cb_wait
                                              ╰─ waiting for DMA @ line 34 (dm_read)
```

## `print_profile_report()`

The report is organized by RISC thread in a fixed order: `NCRISC`, `BRISC`, `TRISC_0`, `TRISC_1`, `TRISC_2`. For each thread, it displays the full kernel source with cycle annotations.

### Color coding

- **Red:** The hottest line (most cycles) in the thread.
- **Yellow:** The second-hottest line.
- **CB background colors:** Eight pastel ANSI backgrounds (light steel blue, pale turquoise, lavender, etc.) distinguish circular buffers. DMA producer lines get a colored background; consumer lines get a colored remark.

### Report sections

1. **CB color key** — legend mapping background colors to CB indices.
2. **Per-thread source listing** — every source line with `LINE`, `%TIME`, `CYCLES`, and `SOURCE` columns. Lines with multiple ops show a breakdown with tree-drawing characters.
3. **Thread summary** — total cycles and op count per thread.
4. **Roofline analysis** — separates sync waits (`cb_wait`, `cb_reserve`) from actual work, then compares memory threads (NCRISC, BRISC) against compute threads (TRISC_0/1/2) to determine whether the kernel is memory-bound or compute-bound:

```
ROOFLINE ANALYSIS
==================================================
  Thread       Total   - Sync Waits   = Work
  NCRISC      50,000   -     12,000   =    38,000
  TRISC_0     45,000   -      5,000   =    40,000

  5% compute bound
  Compute |────────────────────●───────────────────| Memory
          40,000 cycles                     38,000 cycles
```

The roofline indicator position is $\text{pos} = \lfloor \frac{\text{memory\_cycles}}{\text{memory\_cycles} + \text{compute\_cycles}} \times (W - 1) \rfloor$ where $W$ is the bar width. The bound percentage is $\frac{|\text{compute} - \text{memory}|}{\max(\text{compute}, \text{memory})} \times 100$.

## Example Usage

```bash
export TT_METAL_DEVICE_PROFILER=1
export TTLANG_AUTO_PROFILE=1
python my_kernel.py
```

The report prints to stdout after kernel execution. No additional setup is needed — the instrumentation is fully automatic.

---

**Next:** [`signpost_profile.md`](./signpost_profile.md)
