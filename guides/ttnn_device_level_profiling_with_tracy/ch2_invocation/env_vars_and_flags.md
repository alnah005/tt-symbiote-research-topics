# Environment Variables and Flags

## Complete Reference Table

| Variable | Value | Effect |
|---|---|---|
| `TT_METAL_DEVICE_PROFILER` | `1` | Enables on-device CSV profiler; writes `ops_perf_results.csv` |
| `TT_METAL_DEVICE_PROFILER_DISPATCH_CORES` | `0` or `1` | Includes dispatch core timing in device CSV; anchors device time to host time |
| `TRACY_NO_EXIT` | `1` | Prevents the profiled process from exiting before Tracy finishes flushing all events |
| `TT_METAL_CLEAR_L1` | `1` | Clears L1 between runs; ensures deterministic starting state for profiling |
| `TT_METAL_PROFILER_SYNC` | `1` | Adds a host–device sync point after each op; increases timing accuracy at the cost of throughput |

## Variable Details

### `TT_METAL_DEVICE_PROFILER=1`

Setting this variable to `1` activates the on-device cycle-counter profiler. At the end of each kernel dispatch cycle, tt-metal reads the per-core hardware cycle counters from all Tensix cores that participated in the op and writes a row to `ops_perf_results.csv`. Each row records the `DEVICE KERNEL DURATION` — the wall-clock span from the first core's start cycle to the last core's end cycle — along with per-core breakdowns.

> **Warning:** This variable is necessary but not sufficient. If the tt-metal binary was **not** built with `ENABLE_PROFILER=ON`, setting `TT_METAL_DEVICE_PROFILER=1` will still produce a `ops_perf_results.csv`, but every kernel duration column will be zero. The cycle-counter instrumentation in `kernel_profiler.hpp` is compiled out of kernel binaries when `ENABLE_PROFILER` is absent. See `build_requirements.md` for how to verify the build.

### `TT_METAL_DEVICE_PROFILER_DISPATCH_CORES=1`

When set to `1`, the device profiler additionally captures timing data from the dispatch cores (the Tensix cores that handle command dispatch to the compute fabric). This serves two purposes:

1. **Broader coverage** — Dispatch latency becomes visible in the CSV, making it possible to attribute delays to dispatch stalls rather than compute kernel execution.
2. **Host–device time anchoring** — Because dispatch cores interact with both host-initiated commands and device execution, their timestamps provide a reference point that can be used to correlate device cycle counts with host wall-clock time recorded by Tracy.

Set to `0` (or leave unset) when you only need compute kernel timing and want to minimize the volume of CSV output. The default behavior when the variable is unset is equivalent to `0`.

### `TRACY_NO_EXIT=1`

Tracy's client-side ring buffer is drained asynchronously by a background thread that streams events to the `tracy-capture` server over a local TCP connection. When a profiled process exits, the ring buffer may not be fully drained yet — particularly after a short-running test that produces a burst of zones immediately before exit.

Setting `TRACY_NO_EXIT=1` instructs the Tracy client to block in its shutdown path until the background drain thread confirms the server has acknowledged all outstanding events. This is **required** whenever using `tracy-capture` in non-interactive (batch) mode, because there is no human operator watching for the process to finish; without it, the final events of the capture are silently dropped.

> **Tip:** In interactive mode (Tracy GUI open and connected), the GUI itself provides backpressure that prevents event loss, so `TRACY_NO_EXIT=1` is less critical. In CI or scripted capture sessions, always set it.

### `TT_METAL_CLEAR_L1=1`

Clears all L1 SRAM on each Tensix core between dispatch cycles. Without this, leftover data or stale cycle-counter state from a previous kernel can bleed into the next measurement. The cost is an additional DMA clear operation per dispatch cycle, which adds a small fixed overhead to every `host_dispatch_time` measurement.

This variable is most important when:
- Running back-to-back kernels in the same process without a device reset in between.
- Comparing two different kernel implementations in the same pytest session.
- Investigating anomalous first-run vs. subsequent-run timing differences (the classic warm-cache effect).

### `TT_METAL_PROFILER_SYNC=1`

Inserts an explicit host–device synchronization barrier after each op completes. Normally, tt-metal pipelines host dispatch and device execution to maximize throughput: the host may have queued several ops ahead while earlier ops are still running on device. This pipelining means that `device_kernel_time` measured from the CSV for op N may overlap with `host_dispatch_time` for op N+1 from Tracy's perspective.

Setting `TT_METAL_PROFILER_SYNC=1` serializes this pipeline: the host waits for device completion before dispatching the next op. This guarantees that the Tracy zone for op N ends after — not during — the device execution of op N, producing accurate per-op latency numbers at the cost of eliminating dispatch-execution overlap.

## Variable Interaction Rules

### `TT_METAL_DEVICE_PROFILER=1` is required for the CSV

The device profiler and the Tracy profiler are controlled independently. Tracy is activated solely by the build-time `TRACY_ENABLE` define; it does not check `TT_METAL_DEVICE_PROFILER` at runtime. The converse is also true: the CSV profiler does not know or care whether Tracy is running. You can run either profiler alone or both together:

| Goal | Required variables |
|---|---|
| Tracy capture only (`.tracy` file, no CSV) | `TRACY_NO_EXIT=1` (and `TRACY_ENABLE` build) |
| Device CSV only (no Tracy capture) | `TT_METAL_DEVICE_PROFILER=1` |
| Both outputs simultaneously | `TT_METAL_DEVICE_PROFILER=1` + `TRACY_NO_EXIT=1` |

> **Important:** For any workflow that involves Tracy capture, `tracy-capture` must be started **before** the profiled process (pytest) is launched. The Tracy client attempts to connect to the capture server during process initialization. If `tracy-capture` is not already listening at that point, the connection attempt fails silently — `TRACY_NO_EXIT=1` will still cause the process to block on exit waiting for a server that never connected, and no `.tracy` output will be produced.

### `TRACY_NO_EXIT=1` is required in non-interactive mode

If you start `tracy-capture -o profile.tracy -f` and then run pytest without `TRACY_NO_EXIT=1`, the pytest process will exit as soon as its last test completes. Tracy's ring buffer background thread has typically 10–500 ms of events still queued. The resulting `profile.tracy` file will be truncated — it opens successfully in the Tracy GUI but the final portion of the trace is missing.

> **Warning:** A truncated `.tracy` file is not always obviously incomplete. The GUI will display the events that were flushed before exit and show nothing for the period after truncation. If your trace appears to end abruptly before the test should have finished, `TRACY_NO_EXIT=1` is missing.

## Latency Measurement vs. Throughput Measurement

The choice of `TT_METAL_PROFILER_SYNC=1` is the key decision point when configuring a capture session.

**Use `TT_METAL_PROFILER_SYNC=1` for latency measurement:**

When your goal is to measure the true end-to-end latency of a single op — how long from "host submits op" to "device has finished executing" — sync mode is essential.

**Leave `TT_METAL_PROFILER_SYNC=1` unset for throughput measurement:**

When your goal is to measure ops/second or tokens/second across a long workload, adding a sync barrier after every single op serializes the dispatch pipeline and artificially reduces throughput.

## pytest-Specific Considerations

### Setting variables in the shell before running pytest (recommended for most cases)

```bash
export TT_METAL_DEVICE_PROFILER=1
export TT_METAL_DEVICE_PROFILER_DISPATCH_CORES=1
export TRACY_NO_EXIT=1
export TT_METAL_CLEAR_L1=1
export TT_METAL_PROFILER_SYNC=1

pytest tests/ttnn/my_test.py -s -v
```

This approach applies the variables to every test in the session. It is the most straightforward method and the one used in the `capture_workflow.md` examples.

### Setting variables via `os.environ` in a `conftest.py` fixture

For tighter control — for example, when only one test in a multi-test file should enable profiling — you can set environment variables in a pytest fixture:

```python
# conftest.py
import os
import pytest

@pytest.fixture(scope="function")
def profiler_env():
    """Enable both profilers for the duration of one test function."""
    original = {}
    env_vars = {
        "TT_METAL_DEVICE_PROFILER": "1",
        "TT_METAL_DEVICE_PROFILER_DISPATCH_CORES": "1",
        "TRACY_NO_EXIT": "1",
        "TT_METAL_CLEAR_L1": "1",
    }
    for key, value in env_vars.items():
        original[key] = os.environ.get(key)
        os.environ[key] = value
    yield
    for key, original_value in original.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value
```

Then in the test file:

```python
def test_matmul_with_profiling(profiler_env, device):
    # TT_METAL_DEVICE_PROFILER=1 is active for this test only
    ...
```

> **Warning:** `os.environ` modifications in a Python fixture only affect the current process and its future subprocesses. They do **not** retroactively affect the tt-metal runtime if the device has already been initialized in a previous test. For environment variables that are read once at device initialization time (such as `TT_METAL_DEVICE_PROFILER` and `TT_METAL_CLEAR_L1`), the fixture approach works correctly only if the device is initialized after the fixture runs — that is, device initialization must happen inside the test function body or in a function-scoped fixture that executes after `profiler_env`. A session-scoped device fixture initialized before `profiler_env` will not see the new variable values. `TT_METAL_DEVICE_PROFILER` is particularly critical: if it is not set before device initialization, the CSV will not be generated and no error message is produced.

---

**Next:** [`capture_workflow.md`](./capture_workflow.md)
