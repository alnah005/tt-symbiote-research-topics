# Chapter 3 -- Functional Simulator

## Purpose

TT-Lang ships a pure-Python **functional simulator** that validates kernel correctness without requiring Tenstorrent hardware. The simulator faithfully emulates the dataflow-buffer (DFB) state machine, multi-core cooperative scheduling, and resource limits so that a kernel that passes simulation will obey the same contracts on silicon.

This chapter examines the simulator's internals. For a high-level overview of DFBs and the Block state machine, see [Chapter 1 -- Programming Model](../ch1_programming_model/index.md). For the compilation pipeline that eventually produces device binaries, see [Chapter 2 -- Compilation Pipeline](../ch2_compilation_pipeline/index.md).

## Simulator Entry Point: `operation()`

Every TT-Lang kernel begins with the `@ttl.operation()` decorator defined in `sim/operation.py`. The decorator:

1. **Resolves the grid.** If the caller passes `grid="auto"`, the decorator reads the configurable default (set via `set_default_grid()`; factory default is `(8, 8)`).
2. **Injects `grid` into the function's globals** so the kernel body can reference `grid` as a bare name.
3. **Executes the kernel body**, which registers exactly three thread templates -- one `@compute()` and two `@datamovement()` -- via the thread registry in `sim/decorators.py`.
4. **Validates the thread set.** The wrapper enforces exactly 1 compute thread and 2 data-movement threads, raising `ValueError` otherwise.
5. **Constructs and runs a `Program`**, passing the ordered thread templates and the resolved grid.

```python
@ttl.operation(grid=(4, 4))
def my_kernel(inp, out):
    dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_thread():
        ...

    @ttl.datamovement()
    def dm0():
        ...

    @ttl.datamovement()
    def dm1():
        ...
```

When `my_kernel(inp, out)` is called, the `operation` wrapper runs the body (registering the three threads), builds a `Program`, and invokes cooperative simulation across all `4 * 4 = 16` cores.

## Sim vs. On-Device: When to Use Which

| Criterion | Functional Simulator | On-Device |
|---|---|---|
| **Hardware required** | No | Yes (Wormhole / Blackhole) |
| **What it validates** | DFB contracts, state-machine transitions, deadlocks, resource limits | End-to-end performance, real NoC latency, numerical precision |
| **Speed** | Fast iteration; seconds per kernel | Compile + flash + run; minutes |
| **Debugging** | Full Python tracebacks with source locations | Device-side logs, profiling traces |
| **Typical use** | Development, CI, correctness regression | Performance tuning, final validation |

**Rule of thumb:** develop and test kernel logic in the simulator first. Move to on-device only after the simulator run is clean.

## Chapter Contents

| File | Topic |
|---|---|
| [`dfb_state_machine.md`](./dfb_state_machine.md) | `BlockStateMachine`, `STATE_TRANSITIONS` table, `AccessState` lifecycle, `DFBContractError`, per-thread type enforcement |
| [`multicore_scheduling.md`](./multicore_scheduling.md) | `GreenletScheduler`: cooperative scheduling with greenlets, thread binding, scheduling algorithms (greedy / fair), `Program` execution |
| [`resource_limits.md`](./resource_limits.md) | `set_max_dfbs`, `set_max_l1_bytes`, default L1 limit (1336 KiB), `DFBStats` snapshots |

## Key Takeaways

- The `@ttl.operation()` decorator is the single entry point for every simulated kernel. It resolves the grid, collects thread templates, and hands them to `Program`.
- The simulator catches entire classes of bugs -- state-machine violations, deadlocks, resource overflows -- that would otherwise manifest as silent data corruption or hangs on hardware.
- Simulation and on-device execution share the same kernel source; the simulator is not a separate language or tool but the same Python code running in a controlled environment.
