# Chapter 1 — TT-Lang Programming Model

TT-Lang is a Python-embedded DSL that targets Tenstorrent hardware through a decorator-based programming model. It occupies the layer between user-level PyTorch/TTNN code and the TT-Metal runtime, providing a single `import ttl` namespace that exposes kernel authoring, dataflow buffer management, asynchronous data movement, and multi-node grid execution.

## Position in the Software Stack

```
PyTorch / TTNN  (host tensors, high-level ops)
       │
       ▼
    TT-Lang     (kernel DSL: decorators, DFBs, copy, grid intrinsics)
       │
       ▼
  TT-Metal / TT-Metalium  (device runtime, circular buffers, NOC)
       │
       ▼
  Tenstorrent Hardware  (Tensix cores, RISC-V processors)
```

A TT-Lang kernel is a decorated Python function that declares dataflow buffers and spawns exactly three threads (one compute, two data-movement) across a grid of Tensix cores. The DSL has two execution backends:

1. **Compiler path** (`ttl/ttl_api.py`) — Lowers decorated Python to MLIR via the TTL dialect, then to C++ kernels that run on device.
2. **Simulator path** (`sim/`) — Executes the same Python source with cooperative greenlet scheduling for functional verification without hardware.

Both paths share the same user-facing API so that `import ttl` works identically in either mode.

## The Unified `ttl` Namespace

The package-level `__init__.py` re-exports every symbol a kernel author needs:

```python
# From ttl/__init__.py (abridged)
from ttl.ttl import (
    operation,         # @ttl.operation — top-level kernel decorator
    compute,           # @ttl.compute  — compute thread decorator
    datamovement,      # @ttl.datamovement — data-movement thread decorator
    Program,           # Execution harness (internal)
    make_dataflow_buffer_like,  # DFB factory
    copy,              # Asynchronous data transfer
    node,              # Current core coordinates
    grid_size,         # Grid dimensions
    math,              # Element-wise math (sqrt, exp, ...)
)

from ttl.ttl_api import (
    CircularBuffer,       # Compiled-path CB type
    CopyTransferHandler,  # Transfer handle with .wait()
    TensorBlock,          # Operator-overloaded tensor block
)
```

This means a kernel file needs only:

```python
import ttl
import ttnn
```

All decorator, buffer, copy, and grid APIs are then available as `ttl.<name>`.

## Chapter Contents

This chapter covers the four pillars of the TT-Lang programming model:

| File | Topic |
|------|-------|
| [`decorators_and_threads.md`](./decorators_and_threads.md) | The three decorator tiers (`@ttl.operation`, `@ttl.compute`, `@ttl.datamovement`), thread registration, and the `BindableTemplate` protocol |
| [`dataflow_buffers.md`](./dataflow_buffers.md) | `make_dataflow_buffer_like`, `DFBState` ring-buffer internals, block acquisition via `wait()`/`reserve()`, and the `BlockStateMachine` lifecycle |
| [`tensor_blocks_and_grid.md`](./tensor_blocks_and_grid.md) | `TensorBlock` operator overloading, `CopyTransferHandler`, `ttl.copy()`, grid intrinsics (`ttl.node`, `ttl.grid_size`), and a full walkthrough of `eltwise_add.py` |

## Key Takeaways

- **Decorator + DFB = implicit synchronization:** The three-thread structure and ring-buffered DFBs eliminate explicit locks or barriers; all inter-thread coordination is encoded in `reserve`/`push`/`wait`/`pop` state transitions.
- **Write once, run two ways:** Because the simulator and compiler share the same `ttl` API surface, a kernel can be functionally verified on CPU before any hardware is available.
- **Grid-aware data partitioning:** Each core independently computes its tile range from `ttl.node()` and `ttl.grid_size()`, so scaling to larger grids requires no kernel code changes.
