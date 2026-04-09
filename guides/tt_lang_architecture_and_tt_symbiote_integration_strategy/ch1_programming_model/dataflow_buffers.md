# Dataflow Buffers

Dataflow buffers (DFBs) are the central communication primitive in TT-Lang. They implement a producer-consumer ring buffer that synchronizes data movement threads (which read from / write to device memory) with compute threads (which perform arithmetic). Each DFB maps to a hardware circular buffer (CB) on the Tensix core.

## `make_dataflow_buffer_like`

The factory function creates a `DataflowBuffer` (simulator) or `CircularBuffer` (compiler) with properties derived from a tensor:

```python
a_dfb = ttl.make_dataflow_buffer_like(a_in, shape=(GRANULARITY, 1), block_count=2)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `tensor` | `ttnn.Tensor` | Likeness tensor — provides dtype and element shape |
| `shape` | `Tuple[int, ...]` | Tile-grid shape per operation. E.g., `(2, 1)` means each `wait()`/`reserve()` acquires a 2-row by 1-column tile block |
| `block_count` | `int` | Ring buffer capacity (default 2 for double buffering) |

The `block_count` parameter controls how many independent blocks can coexist in the ring buffer. With `block_count=2`, the producer can write to one slot while the consumer reads from another — classic double buffering that hides data transfer latency.

**Source (simulator):** `python/sim/dfb.py`, class `DataflowBuffer.__init__`

```python
# From sim/dfb.py — DataflowBuffer constructor (abridged)
class DataflowBuffer:
    def __init__(self, likeness_tensor, shape, block_count=2):
        self.likeness_tensor = likeness_tensor
        self._shape = shape
        self._block_count = block_count
        # Derive element shape from tile shape and TILE_SIZE (32)
        self._element_shape = tuple(
            1 if edim == 1 else tdim * TILE_SIZE
            for edim, tdim in zip(likeness_tensor.shape, shape)
        )
        # Initialize ring buffer state
        self._state = DFBState()
        self._state.cap = block_count
        self._state.shape = shape
        self._state.buf = [None] * block_count
        self._state.reset()
```

**Source (compiler):** `python/ttl/circular_buffer.py`, class `CircularBuffer`

In the compiler path, each CB is assigned a monotonically increasing index at creation time, used later during MLIR lowering:

```python
# From ttl/circular_buffer.py
class CircularBuffer:
    def __init__(self, tensor, shape, block_count):
        self.tensor = tensor
        self.shape = shape
        self.block_count = block_count
        self._cb_index = _next_cb_index()  # Auto-assigned CB index
```

## `DFBState` — Ring Buffer Internals

The `DFBState` class (defined in `sim/dfbstate.py`) holds the raw ring buffer counters. All counters are in units of "operations" (i.e., blocks), not individual tiles.

```python
class DFBState:
    __slots__ = ("cap", "buf", "head", "visible", "reserved", "shape")

    def __init__(self):
        self.cap: Size = 1          # capacity in operations (= block_count)
        self.buf: List[Optional[Tensor]] = []  # ring buffer slots
        self.head: Index = 0        # current read position
        self.visible: Size = 0      # operations ready to consume
        self.reserved: Size = 0     # operations reserved for writing
        self.shape: Shape           # tile-grid shape

    def free(self) -> Size:
        """Slots available for reservation."""
        return self.cap - self.visible - self.reserved

    def back_slot(self) -> Index:
        """Next reservation slot index."""
        return (self.head + self.visible) % self.cap
        # NOTE: This formula assumes at most one outstanding reservation
        # at a time (i.e., reserved is 0 when back_slot() is called).
        # Each DFB supports at most one outstanding reservation per
        # producer; a second reserve() call will block until the first
        # reservation is pushed.
```

The invariant maintained at all times is:

$$\text{visible} + \text{reserved} + \text{free} = \text{cap}$$

where $\text{free} = \text{cap} - \text{visible} - \text{reserved}$.

### Ring Buffer State Diagram

```
                   ┌────────────────────────┐
                   │    Ring Buffer (cap=2)  │
                   ├──────────┬─────────────┤
         head ───► │  slot 0  │   slot 1    │
                   └──────────┴─────────────┘
                     visible     reserved
                   ◄──────────► ◄───────────►

  reserve() claims a free slot ──► reserved++
  push()    makes it visible   ──► reserved--, visible++
  wait()    reads a visible slot
  pop()     frees the head     ──► visible--, head = (head+1) % cap
```

## Block Acquisition: `wait()` and `reserve()`

`DataflowBuffer` exposes two acquisition methods that return `Block` objects. Both support Python's context manager protocol (`with` statement), which automatically calls `pop()` or `push()` on exit.

### `reserve()` — Producer Acquires a Write Slot

```python
# From sim/dfb.py
def reserve(self) -> Block:
    block_if_needed(self, "reserve")   # Yield if no free slots
    state = self._state
    slot_idx = state.back_slot()
    slot = Tensor(torch.zeros(self._element_shape, ...))
    state.buf[slot_idx] = slot
    state.reserved += 1
    block = Block(
        tensor=slot, shape=state.shape,
        acquisition=BlockAcquisition.RESERVE,
        thread_type=thread_type,
    )
    block.dfb = self
    return block
```

The returned `Block` starts in state `MW` (Must Write). For a DM thread, the expected operation is `COPY_DST`; for a compute thread, it is `STORE`.

### `wait()` — Consumer Acquires a Read Slot

```python
# From sim/dfb.py
def wait(self) -> Block:
    block_if_needed(self, "wait")      # Yield if no visible slots
    state = self._state
    slot = state.buf[state.head]       # Read from head position
    block = Block(
        tensor=slot, shape=state.shape,
        acquisition=BlockAcquisition.WAIT,
        thread_type=thread_type,
    )
    block.dfb = self
    return block
```

The returned `Block` starts in state `MR` (Must Read). For a DM thread, the expected operation is `COPY_SRC`; for a compute thread, it is `STORE_SRC`.

### Context Manager Usage

Blocks implement `__enter__` and `__exit__` so they can be used with `with`:

```python
with a_dfb.wait() as a_blk, out_dfb.reserve() as out_blk:
    out_blk.store(a_blk + a_blk)
# __exit__ automatically calls a_blk.pop() and out_blk.push()
```

From `sim/dfb.py`, the `Block.__exit__` method:

```python
def __exit__(self, exc_type, exc_val, exc_tb):
    if exc_type is None and self.dfb is not None:
        if self._acquisition == BlockAcquisition.RESERVE:
            self.push()
        elif self._acquisition == BlockAcquisition.WAIT:
            self.pop()
```

## `BlockStateMachine` and `AccessState` Lifecycle

Every `Block` owns a `BlockStateMachine` instance that validates the sequence of operations performed on it. This catches programming errors such as reading from a block before it has been written, or forgetting to call `tx.wait()`.

**Source:** `python/sim/blockstate.py`

### Access States

```python
class AccessState(Enum):
    MW  = auto()  # Must Write: reserved block, contains garbage
    MR  = auto()  # Must Read: waited block or written block, must be consumed
    RW  = auto()  # Read-Write: has been read, can be read again or overwritten
    ROR = auto()  # Read-Only while Reading: async copy in flight
    NAW = auto()  # No Access while Writing: async copy writing to block
    OS  = auto()  # Out of Scope: block has been pushed/popped
```

### State Transition Table

Transitions are keyed by `(BlockAcquisition, ThreadType)` context and `(operation, current_state)`:

**DM thread, RESERVE acquisition (producer read path):**

| Operation | From State | To State | Next Expected Ops |
|-----------|-----------|----------|-------------------|
| `copy_dst` | MW | NAW | `{TX_WAIT}` |
| `tx_wait` | NAW | MR | `{PUSH, COPY_SRC}` |
| `copy_src` | MR | ROR | `{TX_WAIT, COPY_SRC}` |
| `tx_wait` | ROR (N=1) | RW | `{COPY_DST, COPY_SRC, PUSH}` |

**COMPUTE thread, WAIT acquisition (consumer compute path):**

| Operation | From State | To State | Next Expected Ops |
|-----------|-----------|----------|-------------------|
| `assign_src` | MR | RW | `{STORE_SRC, STORE, POP}` |
| `store_src` | MR/RW | RW | `{STORE_SRC, STORE, POP}` |
| `store_dst` | RW | MR | `{STORE_SRC}` |

### ROR(N) — Multiple In-Flight Copies

When a DM thread issues multiple `copy()` calls from the same block, the state machine tracks the number of in-flight transfers with an `_ror_count` counter:

```python
# From sim/blockstate.py — BlockStateMachine.transition()
if self._access_state == AccessState.ROR:
    if operation_key == "copy_src":
        self._ror_count += 1      # Another copy in flight
        return
    if operation_key == "tx_wait" and self._ror_count > 1:
        self._ror_count -= 1      # One copy completed, others still in flight
        return
    # When _ror_count == 1 and tx_wait fires, fall through to table lookup
    # which transitions ROR -> RW
```

### Initialization Based on Context

The initial state is set by `BlockStateMachine.initialize()`:

| Acquisition | Thread | Initial State | Expected Ops |
|-------------|--------|---------------|-------------|
| RESERVE | DM | MW | `{COPY_DST}` |
| RESERVE | COMPUTE | MW | `{STORE}` |
| WAIT | DM | MR | `{COPY_SRC}` |
| WAIT | COMPUTE | MR | `{STORE_SRC}` |

### Lifecycle Example — Compute Thread, Elementwise Add

```
              reserve() ──► MW {STORE}
                              │
                     store(a_blk + b_blk)
                              │
              store_dst  ──► MR {STORE_SRC, PUSH}
                              │
               push()   ──► OS {} (out of scope)
```

For the `wait()` blocks (`a_blk`, `b_blk`):

```
              wait()    ──► MR {STORE_SRC}
                              │
                  used in (a_blk + b_blk)
                   assign_src fires
                              │
                         ──► RW {STORE_SRC, STORE, POP}
                              │
            store() on result marks store_src
                              │
                         ──► RW {STORE_SRC, STORE, POP}
                              │
               pop()    ──► OS {} (out of scope)
```

## Hardware Resource Limits

The `Program` execution checks two hardware constraints before launching the scheduler:

1. **DFB count.** Warns if the number of `make_dataflow_buffer_like` calls exceeds the hardware limit (configurable via `set_max_dfbs()`).
2. **L1 memory.** Warns if total DFB capacity in bytes exceeds the per-core L1 limit (defaults to 1336 KiB for Blackhole/Wormhole, configurable via `set_max_l1_bytes()`).

```python
# From sim/program.py
dfb_count = get_context().kernel_dfb_count
if dfb_count > max_dfbs:
    warnings.warn(
        f"Kernel defines {dfb_count} dataflow buffers, "
        f"but the hardware limit is {max_dfbs}."
    )
```

---

**Next:** [`tensor_blocks_and_grid.md`](./tensor_blocks_and_grid.md)
