# Resource Limits

The simulator enforces hardware resource limits at kernel launch time so that over-provisioned kernels fail fast in simulation rather than silently misbehaving on device. All configuration lives in `SimulatorConfig` (defined in `python/sim/context_types.py`) and is manipulated through the API in `python/sim/program.py`.

## DFB Count Limit: `set_max_dfbs()`

Each Tenstorrent core has a fixed number of circular-buffer (CB) slots in hardware. The simulator mirrors this constraint:

```python
from python.sim.program import set_max_dfbs, get_max_dfbs

set_max_dfbs(64)        # raise the limit
print(get_max_dfbs())   # 64
```

**Default:** 32 DFBs per core (`SimulatorConfig.max_dfbs = 32`).

When `Program._run_cooperative()` executes, it reads `get_context().kernel_dfb_count` -- a counter incremented by each `make_dataflow_buffer_like()` call during the kernel body -- and compares it against the limit:

```python
dfb_count = get_context().kernel_dfb_count
max_dfbs = get_max_dfbs()
if dfb_count > max_dfbs:
    warnings.warn(
        f"Kernel defines {dfb_count} dataflow buffers, "
        f"but the hardware limit is {max_dfbs}. "
        f"Reduce the number of ttl.make_dataflow_buffer_like() calls.",
        stacklevel=2,
    )
```

This emits a Python `UserWarning` rather than a hard error, because the simulator can still execute the kernel -- the warning alerts the developer that the kernel would fail hardware allocation.

## L1 Memory Limit: `set_max_l1_bytes()`

Every DFB occupies L1 SRAM on-core. The simulator tracks the aggregate L1 footprint and warns when it exceeds the hardware budget:

```python
from python.sim.program import set_max_l1_bytes, get_max_l1_bytes

set_max_l1_bytes(1_572_864)       # 1.5 MiB
print(get_max_l1_bytes())         # 1572864
```

**Default:** 1336 KiB (1,368,064 bytes), defined in `context_types.py`:

```python
DEFAULT_MAX_L1_BYTES = (1464 - 128) * 1024   # 1336 KiB = 1_368_064 bytes
```

The value comes from the Blackhole/Wormhole L1 size of 1464 KiB minus 128 KiB reserved for program binary and stack space.

The check mirrors the DFB-count check:

```python
total_l1_bytes = get_context().kernel_l1_bytes
max_l1 = get_max_l1_bytes()
if total_l1_bytes > max_l1:
    warnings.warn(
        f"Total DataflowBuffer capacity per core ({total_l1_bytes} bytes) "
        f"exceeds the L1 memory limit of {max_l1} bytes.",
        stacklevel=2,
    )
```

Both `kernel_dfb_count` and `kernel_l1_bytes` are reset to zero at the start of every `operation()` call, so each kernel is checked independently.

### Computing L1 Usage

Each `DataflowBuffer` contributes:

$$
\text{capacity\_bytes} = \text{block\_count} \times \prod(\text{shape}) \times \text{bytes\_per\_tile}
$$

where `shape` is the tile-grid shape (e.g., `(2, 3)` means 6 tiles per block) and `bytes_per_tile` depends on the element dtype and tile dimensions. The `kernel_l1_bytes` accumulator sums these across all DFBs created in the kernel body.

## DFBStats: Runtime Snapshots

`DFBStats` (defined in `python/sim/dfb.py`) is a `NamedTuple` that captures the instantaneous state of a `DataflowBuffer`'s ring buffer:

```python
class DFBStats(NamedTuple):
    capacity: int           # total slots (= block_count)
    visible: int            # slots ready to consume (via wait())
    reserved: int           # slots reserved for writing (via reserve())
    free: int               # slots available for reservation
    head: int               # current read slot index
    slots: List[Optional[Tensor]]  # per-slot contents (None = empty)
```

The relationship between the counters is:

$$
\text{free} = \text{capacity} - \text{visible} - \text{reserved}
$$

This is maintained by the internal `DFBState` object (in `python/sim/dfbstate.py`), which manages the ring buffer's `head`, `visible`, and `reserved` counters. `DFBState.free()` computes the above formula, and `DFBState.back_slot()` returns `(head + visible) % capacity` -- the next slot where a reservation will be placed.

`DFBStats` snapshots are useful for:

- **Debugging** -- inspecting ring-buffer occupancy at a breakpoint.
- **Statistics collection** -- the `SimulatorStats` system (in `stats.py`) can record per-DFB reserve/wait counts and tile throughput when `enable_stats()` is called.

## SimulatorConfig Summary

The full set of simulator-level configuration knobs lives in `SimulatorConfig`:

| Field | Type | Default | Description |
|---|---|---|---|
| `max_dfbs` | `int` | `32` | Maximum DataflowBuffers per core |
| `scheduler_algorithm` | `str` | `"fair"` | Scheduling algorithm (`"greedy"` or `"fair"`) |
| `default_auto_grid` | `Shape` | `(8, 8)` | Default grid when `grid="auto"` |
| `max_l1_bytes` | `int` | `1_368_064` | Maximum L1 bytes per core (1336 KiB) |
| `num_devices` | `int` | `4` | Number of simulated devices |

All fields are accessed through the `SimulatorContext` retrieved by `get_context()`, which uses greenlet-local storage so that concurrent simulations (if any) do not interfere.

---

**Next:** [Chapter 4 -- Performance Analysis Tools](../ch4_performance_tools/index.md)
