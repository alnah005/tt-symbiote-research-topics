# Perf Dump and Perfetto Trace Server

This section covers two related but distinct features: the NOC profiler summary (`TTLANG_PERF_DUMP=1`) and the Perfetto trace server (`TTLANG_PERF_SERV=1`).

## Part 1: Perf Dump — NOC Profiler Summary

**Env var:** `TTLANG_PERF_DUMP=1`
**Module:** `python/ttl/_src/perf_summary.py`
**Pipeline hook:** `_run_perf_dump()` in `ttl_api.py`

### What It Shows

Perf dump provides a hardware-level summary of NOC (Network-on-Chip) traffic and kernel timing. Where auto-profile and signpost profiling focus on cycle counts per code region, perf dump answers questions about memory bandwidth, transfer patterns, and data movement topology.

### Additional Prerequisites

Beyond the standard `TT_METAL_DEVICE_PROFILER=1` and `TT_METAL_HOME`, perf dump requires:

- **`TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1`** — enables NOC event tracing, which produces `noc_trace_*.json` files alongside the device profiler CSV.
- **`TT_METAL_PROFILER_MID_RUN_DUMP=1`** — recommended; ensures profiler data is flushed before reading.

### Data Sources

`_run_perf_dump()` reads three artifacts:

1. **`noc_trace_*.json`** — NOC event traces (one per program dispatch), located in `$TT_METAL_HOME/generated/profiler/.logs/`.
2. **`/tmp/ttlang_cb_flow_graph.json`** — CB flow graph produced by the `ttl-dump-cb-flow-graph` MLIR pass at compile time.
3. **`/tmp/ttlang_pipe_graph.json`** — pipe graph copied from the compiler temp directory.

### `ProgramSummary` Data Class

The `parse_noc_trace()` function processes a single NOC trace JSON into a `ProgramSummary`:

```python
@dataclass
class ProgramSummary:
    program_id: int
    source_cores: Set[Tuple[int, int]]     # Active cores
    dram_bytes_read: int                    # Total DRAM read bytes
    dram_bytes_written: int                 # Total DRAM write bytes
    l1_bytes_read: int                      # Total L1 read bytes
    l1_bytes_written: int                   # Total L1 write bytes
    multicast_count: int                    # L1 pipe multicast events
    multicast_bytes: int                    # L1 pipe multicast bytes
    semaphore_count: int                    # Pipe synchronization events
    read_barrier_count: int                 # NOC read barriers
    write_barrier_count: int                # NOC write barriers
    transfer_sizes: Dict[int, int]          # Size -> count histogram
    brisc_durations: List[int]              # Per-core BRISC kernel times
    ncrisc_durations: List[int]             # Per-core NCRISC kernel times
    trisc0_durations: List[int]             # Per-core TRISC_0 kernel times
    # ... trisc1, trisc2
    min_timestamp: Optional[int]
    max_timestamp: Optional[int]
```

### NOC Event Classification

The parser classifies each NOC event by type:

| Event Type | Classification Logic |
|---|---|
| DRAM read | `READ` event where destination is not a compute core |
| DRAM write | `WRITE` event where destination is not a compute core |
| L1 read | `READ` event where destination is a compute core |
| L1 write | `WRITE` event where destination is a compute core |
| L1 multicast | `WRITE` with `mcast_start_x`/`mcast_start_y` targeting compute cores |
| Barrier | `BARRIER` in event type (read or write start) |
| Semaphore | `SEMAPHORE` in event type |

Compute cores are identified from zone events in the same trace — any core that has a `zone` field is considered a compute core. Non-compute destinations are assumed to be DRAM banks.

### Kernel Duration Merging

`parse_kernel_durations()` reads `profile_log_device.csv` separately to extract per-thread kernel durations (from `*-KERNEL` zone pairs). These are merged into the `ProgramSummary` by `run_host_id`. For DM threads (BRISC, NCRISC), durations from the CSV take precedence over NOC JSON durations as the CSV provides per-core granularity. TRISC durations (TRISC_0, TRISC_1, TRISC_2) come exclusively from the CSV.

### Report Format

`format_summary()` produces human-readable text:

```
--- Program 0 (my_kernel) ---
grid: 8x8 (64 cores)
duration: 125,000 cycles (125.0 us)
  DRAM read:        2.0 MB  (512 transfers)
  DRAM write:       2.0 MB  (512 transfers)
  L1 multicast:   128.0 KB  (64 transfers, pipe)
  effective BW:   29.3 GB/s (total payload / duration)
  transfer size:  4.0 KB (uniform)
  barriers:       64 read (1 per 8 reads), 64 write (1 per 8 writes)
  semaphores:     128 events
  noc reads:      NOC0=256, NOC1=256
  noc writes:     NOC0=320, NOC1=256
  DRAM channels:  8
  kernel time:
    BRISC    12,500 cycles (12.5 us)
    NCRISC   15,000 cycles (15.0 us)
    TRISC_0  45,000 cycles (45.0 us)
```

Effective bandwidth is computed as $\text{BW} = \frac{\text{total\_bytes}}{\text{duration\_cycles} / (\text{freq\_MHz} \times 10^6)}$.

Machine-readable JSON output is available via `--json` or `output_json=True`.

### Chip Info Parsing

`parse_chip_info()` reads architecture metadata from the CSV header line:

```python
def parse_chip_info(logs_path: Path) -> Tuple[str, int, int]:
    """Returns (arch_name, freq_mhz, max_compute_cores)."""
```

The header contains fields like `ARCH: wormhole_b0`, `CHIP_FREQ[MHz]: 1000`, `Max Compute Cores: 64`.

## Part 2: Perfetto Trace Server

**Env var:** `TTLANG_PERF_SERV=1`
**Module:** `python/ttl/_src/perf_trace_server.py`
**Pipeline hook:** Runs last in the profiling sequence in `ttl_api.py`

### What It Does

The Perfetto trace server converts the device profiler CSV into Chrome Trace Event format and serves it over HTTP. A landing page auto-opens Perfetto UI in the browser and pushes the trace data via `postMessage`, avoiding HTTPS/mixed-content issues.

### CSV to Trace Events

`csv_to_trace_events()` reads the device profiler CSV and produces Chrome Trace Event format:

```python
def csv_to_trace_events(csv_path: Path) -> List[dict]:
```

Each `ZONE_START`/`ZONE_END` pair becomes an "X" (complete) event:

```json
{
    "name": "compute_L52",
    "cat": "TRISC_0",
    "ph": "X",
    "ts": 12.345,
    "dur": 5.678,
    "pid": "Core (1,2)",
    "tid": "TRISC_0",
    "args": {"source": "kernel.cpp:52"}
}
```

Key details:

- **Timestamps** are converted from cycles to microseconds: $t_{\mu s} = \frac{\text{cycles}}{\text{freq\_MHz}}$.
- **Wrapper zones** (`BRISC-FW`, `BRISC-KERNEL`, `NCRISC-FW`, `NCRISC-KERNEL`, `TRISC-FW`, `TRISC-KERNEL`) are filtered out as they obscure the actual trace.
- **All timestamps are normalized** to start at 0 by subtracting `min(ts)`.
- **Process ID** is set to the core coordinate string (e.g., `Core (1,2)`), so Perfetto groups traces by core.
- **Thread ID** is the RISC processor name, so within each core the five RISC threads appear as separate swim lanes.

The chip frequency is parsed from the CSV header via `_parse_chip_freq()`.

### HTTP Server

`serve_trace()` starts a local HTTP server:

```python
def serve_trace(csv_path: Path, port: Optional[int] = None):
```

The server handles two routes:

| Path | Response |
|------|----------|
| `/` (or any non-`/trace.json`) | HTML landing page |
| `/trace.json` | Trace event JSON |

The landing page JavaScript:
1. Fetches `/trace.json` from the same origin.
2. Opens `https://ui.perfetto.dev/` in a new window.
3. Pings the Perfetto window with `"PING"` every 50ms until it replies `"PONG"`.
4. Sends the trace buffer via `postMessage` with the `perfetto` payload format.

This approach avoids the need for HTTPS on the local server — the trace data is fetched from the same HTTP origin and then pushed to Perfetto's HTTPS page via the `postMessage` API.

### Port Selection and Docker Support

- If no port is specified, `_find_free_port()` binds to port 0 and reads the assigned port.
- `_get_container_ip()` detects Docker containers (checks for `/.dockerenv`) and resolves the container's IP via `socket.gethostbyname(socket.gethostname())`.
- The server prints SSH tunnel instructions for remote access:

```
TTLANG PERFETTO TRACE SERVER
======================================================================
  42 trace events ready
  Serving on port 8234

  From your local machine, run:
    ssh -N -L 8234:172.17.0.2:8234 user@<server>

  Then open:
    http://localhost:8234

  Press Enter to stop the server...
======================================================================
```

### Integration with Other Profiling Modes

`TTLANG_PERF_SERV=1` runs **after** all other profiling hooks in the execution sequence. It can be combined with any other profiling flag:

```bash
# Signpost profile table + Perfetto visualization
TTLANG_SIGNPOST_PROFILE=1 TTLANG_PERF_SERV=1 python my_kernel.py

# Auto-profile report + Perfetto visualization
TTLANG_AUTO_PROFILE=1 TTLANG_PERF_SERV=1 python my_kernel.py

# Full dump: NOC summary + CB/pipe graphs + Perfetto
TTLANG_PERF_DUMP=1 TTLANG_PERF_SERV=1 python my_kernel.py
```

After serving, the runtime deletes `TTLANG_PERF_SERV` from `os.environ` to prevent the server from launching again on subsequent kernel calls within the same process.

### Standalone Usage

The trace server can also run standalone against a previously-collected CSV:

```bash
python -m ttl._src.perf_trace_server --path /path/to/profiler/.logs/
python -m ttl._src.perf_trace_server --path /path/to/profile_log_device.csv --port 9000
```

Similarly, the perf summary tool has a standalone mode:

```bash
python -m ttl._src.perf_summary --path /path/to/profiler/.logs/
python -m ttl._src.perf_summary --path /path/to/profiler/.logs/ --json
python -m ttl._src.perf_summary --names "my_kernel,ttnn.multiply"
```

---

**Next:** [Chapter 5 — TT-Symbiote Architecture and Pain Points](../ch5_symbiote_architecture/index.md)
