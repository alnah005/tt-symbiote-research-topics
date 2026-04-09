# DFB State Machine

This section dives into the implementation of the Block state machine that enforces correct DFB usage at simulation time. For the conceptual introduction to DFBs and the reserve/wait protocol, see [Chapter 1 -- Programming Model](../ch1_programming_model/index.md).

All source references point into `python/sim/blockstate.py` and `python/sim/dfb.py`.

## AccessState Enum

Every `Block` tracks an `AccessState` that governs what operations are legal at any given moment. The enum is defined in `blockstate.py`:

| State | Meaning |
|---|---|
| `MW` (Must Write) | Block was reserved and contains garbage -- must be written to before anything else. |
| `MR` (Must Read) | Block was waited-on or freshly written -- must be read from or pushed before reuse. |
| `RW` (Read-Write) | Block has been both written and read -- can be read again or overwritten. |
| `ROR` (Read-Only while Reading) | One or more async copies are in flight from this block. Tracked with a reference count $N$. |
| `NAW` (No Access while Writing) | An async copy is writing into this block -- no other access until `tx.wait()` completes. |
| `OS` (Out of Scope) | Block has been pushed or popped -- no further access is legal. |

The lifecycle of a typical **producer** block (DM thread, `reserve` acquisition):

$$
\text{MW} \xrightarrow{\text{copy\_dst}} \text{NAW} \xrightarrow{\text{tx\_wait}} \text{MR} \xrightarrow{\text{push}} \text{OS}
$$

And a typical **consumer** block (DM thread, `wait` acquisition):

$$
\text{MR} \xrightarrow{\text{copy\_src}} \text{ROR}(1) \xrightarrow{\text{tx\_wait}} \text{RW} \xrightarrow{\text{pop}} \text{OS}
$$

## BlockStateMachine

`BlockStateMachine` (in `blockstate.py`) encapsulates all state-tracking logic. Each `Block` instance holds exactly one `BlockStateMachine` in its `_sm` slot.

### Construction and Initialization

```python
class BlockStateMachine:
    __slots__ = (
        "_acquisition",    # BlockAcquisition.RESERVE or .WAIT
        "_thread_type",    # ThreadType.COMPUTE or .DM
        "_access_state",   # current AccessState
        "_expected_ops",   # set[ExpectedOp] -- what's legal next
        "_ror_count",      # in-flight copy count while in ROR
    )
```

On `initialize()`, the machine sets the starting state:

| Acquisition | Thread | Initial State | Initial Expected Ops |
|---|---|---|---|
| `RESERVE` | `DM` | `MW` | `{COPY_DST}` |
| `RESERVE` | `COMPUTE` | `MW` | `{STORE}` |
| `WAIT` | `DM` | `MR` | `{COPY_SRC}` |
| `WAIT` | `COMPUTE` | `MR` | `{STORE_SRC}` |

Temporary blocks (those created by arithmetic operations, not backed by a DFB) skip `initialize()` and instead call `set_unrestricted()`, which sets the state to `RW` with an empty expected-ops set, allowing unrestricted access.

### The Transition Method

The core of the machine is `transition()`:

```python
def transition(self, operation_key, operation_display, expected_op):
    self.validate(operation_display, expected_op)
    # ROR(N) in-state logic ...
    context_key = (self._acquisition, self._thread_type)
    transition_key = (operation_key, self._access_state)
    new_access_state, new_expected_ops = STATE_TRANSITIONS[context_key][transition_key]
    ...
```

The method:

1. **Validates** that `expected_op` is in the current `_expected_ops` set. If not, it raises a `RuntimeError` with a detailed diagnostic including the current acquisition, thread type, and access state.
2. **Handles ROR(N) in-state transitions** for `copy_src` (increments $N$) and `tx_wait` (decrements $N$). Only when $N$ reaches 1 does `tx_wait` fall through to the table.
3. **Looks up the table** using the composite key `(acquisition, thread_type)` then `(operation_key, access_state)`.
4. **Applies the new state** and expected-ops set.

## STATE_TRANSITIONS Table

The full transition table is a nested dictionary keyed by `(BlockAcquisition, ThreadType)` at the outer level and `(operation_key, AccessState)` at the inner level. Each entry maps to `(new_AccessState, set[ExpectedOp])`.

### DM Thread Transitions

**WAIT acquisition (consumer path):**

| Operation | From State | To State | Next Expected Ops |
|---|---|---|---|
| `copy_src` | `MR` | `ROR` | `{TX_WAIT, COPY_SRC}` |
| `copy_src` | `RW` | `ROR` | `{TX_WAIT, COPY_SRC}` |
| `copy_dst` | `RW` | `NAW` | `{TX_WAIT}` |
| `tx_wait` | `ROR` | `RW` | `{COPY_DST, COPY_SRC, POP}` |
| `tx_wait` | `NAW` | `MR` | `{COPY_SRC}` |

**RESERVE acquisition (producer path):**

| Operation | From State | To State | Next Expected Ops |
|---|---|---|---|
| `copy_src` | `MR` | `ROR` | `{TX_WAIT, COPY_SRC}` |
| `copy_src` | `RW` | `ROR` | `{TX_WAIT, COPY_SRC}` |
| `copy_dst` | `MW` | `NAW` | `{TX_WAIT}` |
| `copy_dst` | `RW` | `NAW` | `{TX_WAIT}` |
| `tx_wait` | `NAW` | `MR` | `{PUSH, COPY_SRC}` |
| `tx_wait` | `ROR` | `RW` | `{COPY_DST, COPY_SRC, PUSH}` |

### Compute Thread Transitions

**WAIT acquisition:**

| Operation | From State | To State | Next Expected Ops |
|---|---|---|---|
| `assign_src` | `MR` | `RW` | `{STORE_SRC, STORE, POP}` |
| `assign_src` | `RW` | `RW` | `{STORE_SRC, STORE, POP}` |
| `store_src` | `MR` | `RW` | `{STORE_SRC, STORE, POP}` |
| `store_src` | `RW` | `RW` | `{STORE_SRC, STORE, POP}` |
| `store_dst` | `RW` | `MR` | `{STORE_SRC}` |

**RESERVE acquisition:**

| Operation | From State | To State | Next Expected Ops |
|---|---|---|---|
| `store_src` | `MR` | `RW` | `{STORE_SRC, STORE, PUSH}` |
| `store_src` | `RW` | `RW` | `{STORE_SRC, STORE, PUSH}` |
| `store_dst` | `MW` | `MR` | `{STORE_SRC, PUSH}` |
| `store_dst` | `RW` | `MR` | `{STORE_SRC, PUSH}` |

### Terminal Transitions: push and pop

`push()` and `pop()` are handled by dedicated methods rather than the table:

- **`transition_push()`** -- only valid for `RESERVE` blocks. Moves state to `OS` and clears expected ops.
- **`transition_pop()`** -- only valid for `WAIT` blocks in `MR` or `RW` state. Moves state to `OS` and clears expected ops.

Both raise `RuntimeError` if called on the wrong acquisition type or in an invalid state.

## ROR(N): In-Flight Copy Tracking

When a DM thread issues multiple `copy_src` operations before calling `tx.wait()`, the state machine tracks the number of in-flight copies with `_ror_count`:

1. First `copy_src` from `MR` or `RW`: state transitions to `ROR`, `_ror_count` set to 1 (via the table).
2. Subsequent `copy_src` while in `ROR`: `_ror_count` incremented, state stays `ROR` (in-state transition, does not hit the table).
3. Each `tx_wait` while `_ror_count > 1`: `_ror_count` decremented, state stays `ROR`.
4. Final `tx_wait` when `_ror_count == 1`: falls through to the table, which transitions `ROR` to `RW`.

This ensures the block remains read-only until every outstanding copy has completed.

## DFBContractError

The `errors.py` module defines a small hierarchy:

```
DFBError (RuntimeError)
  +-- DFBContractError
  +-- DFBOutOfRange
  +-- DFBTimeoutError
```

`DFBContractError` is raised by `DataflowBuffer` methods (in `dfb.py`) when higher-level DFB invariants are violated -- for example, calling `reserve()` when no free slots exist and the caller incorrectly bypassed blocking. The `BlockStateMachine` itself raises plain `RuntimeError` for state-transition violations, keeping the two error surfaces distinct.

## Per-Thread Type Enforcement

The thread type (`ThreadType.DM` or `ThreadType.COMPUTE`) is baked into each `BlockStateMachine` at construction time and determines which quadrant of the `STATE_TRANSITIONS` table applies. This means:

- A **DM thread** can only perform `copy_src`, `copy_dst`, and `tx_wait` operations on its blocks.
- A **COMPUTE thread** can only perform `assign_src`, `store_src`, and `store_dst` operations on its blocks.

Attempting a DM operation on a compute-thread block (or vice versa) will fail at the `validate()` step because the corresponding `ExpectedOp` will never appear in the compute-thread's expected-ops set. The separation is structural, not a runtime flag check -- the table simply has no entries that would allow cross-thread operations.

The current thread type is tracked globally via `context.set_current_thread_type()` / `get_current_thread_type()`, which the `GreenletScheduler` sets before switching into each greenlet and clears afterward.

---

**Next:** [`multicore_scheduling.md`](./multicore_scheduling.md)
