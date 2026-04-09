# Agent B Review: Chapter 1 — Pass 1

1. [tensor_blocks_and_grid.md] ~line 186: The coordinate decomposition formula is wrong. The guide states `node_col = k mod C, node_row = floor(k / C)` for a grid of shape `(C, R)`. The actual source (`sim/corecontext.py`, `node()`) iterates `reversed(grid)`, producing `node_col = floor(k / R)` and `node_row = k mod R`. These are materially different and would cause incorrect tile partitioning if a reader implemented the formula as written. → Fix the formula to `node_col = floor(k / R), node_row = k mod R`, or equivalently restate in terms that match the source's `reversed(grid)` decomposition.

2. [index.md] ~line 77: No navigation footer. Every content file in the chapter must end with a navigation footer. The other three files have `**Next:** ...` links, but `index.md` has none. → Add a footer such as `**Next:** [decorators_and_threads.md](./decorators_and_threads.md)`.

3. [decorators_and_threads.md] ~line 200: Navigation footer has only a "Next" link but no "Previous" link back to the index. The same issue affects `dataflow_buffers.md` (~line 288) and `tensor_blocks_and_grid.md` (~line 371). If the convention is forward-only navigation this is fine, but if footers are expected to be bidirectional, all three content files are missing their "Previous" links. → Clarify convention or add `**Previous:** [index.md](./index.md)` (and analogous links) to each file.

4. [dataflow_buffers.md] ~line 80: The `back_slot()` formula shown is `(self.head + self.visible) % self.cap`. This matches the source (`dfbstate.py:46`) verbatim, so it is not a code-vs-guide mismatch. However, the prose on line 79 labels it "Next reservation slot index", which is only correct because the source enforces at most one outstanding reservation at a time (via `_pending_reserved_block`). A reader unaware of that single-reservation invariant could incorrectly believe `back_slot()` accounts for multiple concurrent reservations. → Add a brief note that `back_slot()` is valid because `DataflowBuffer` enforces at most one pending reservation; calling `reserve()` twice without an intervening `push()` raises a `RuntimeError`.

5. [tensor_blocks_and_grid.md] ~line 164: The guide shows `node_col, node_row = ttl.node(dims=2)  # Returns (x, y) tuple`. The comment labels the return as `(x, y)`, implying column-first ordering. The actual source returns `(coords[0], coords[1])` where `coords[0]` corresponds to the first grid dimension. Whether this is "x" or "y" depends on convention, but the eltwise_add example (`examples/eltwise_add.py:17-18`) uses `grid_cols, grid_rows = ttl.grid_size(dims=2)` and `node_col, node_row = ttl.node(dims=2)`, meaning the first grid dimension is columns and the second is rows. The `(x, y)` comment is acceptable only if x=col and y=row, which is the standard interpretation. This is borderline but not materially misleading given the variable names in context. → No change required, but consider dropping the `(x, y)` comment to avoid ambiguity.

---

# Agent B Review: Chapter 1 — Pass 2

Pass 1 issues #1 (coordinate formula) and #4 (back_slot caveat) have been fixed correctly. Issue #5 was advisory and remains acceptable. Issues #2 and #3 (navigation) are addressed below if still present.

1. [dataflow_buffers.md] ~line 248-254: The "Lifecycle Example — Compute Thread, Elementwise Add" shows `store_dst` transitioning from MW to `MR {STORE_SRC}` for a RESERVE/COMPUTE block. The actual source (`blockstate.py`, lines 170-173) defines this transition as `("store_dst", AccessState.MW): (AccessState.MR, {ExpectedOp.STORE_SRC, ExpectedOp.PUSH})`. The expected-ops set is missing `PUSH`. A reader implementing the state machine from this diagram would incorrectly reject `push()` after `store_dst` on a reserved compute block. Fix: change `MR {STORE_SRC}` to `MR {STORE_SRC, PUSH}` in the lifecycle diagram.

2. [index.md]: Still has no navigation footer. The three content files all end with `**Next:** [...]` links, but index.md ends abruptly after "Key Takeaways." For consistency and discoverability, add a footer such as `**Next:** [decorators_and_threads.md](./decorators_and_threads.md)` at the bottom.

No other materially misleading issues found. Remaining items from Pass 1 (bidirectional nav, `(x, y)` comment) are stylistic preferences, not correctness problems.

---

# Agent B Review: Chapter 1 — Pass 3

Verified all four files against the tt-lang source code. Checked:

- `node()` coordinate decomposition formula (tensor_blocks_and_grid.md ~line 186) against `sim/corecontext.py` -- correct after Pass 1 fix.
- `back_slot()` formula and invariant note (dataflow_buffers.md ~line 80) against `sim/dfbstate.py` -- correct, caveat added in Pass 1.
- DM/RESERVE state transition table (dataflow_buffers.md ~line 200) against `sim/blockstate.py` STATE_TRANSITIONS -- all four rows match source.
- COMPUTE/WAIT state transition table (dataflow_buffers.md ~line 209) against `sim/blockstate.py` -- all rows match source.
- Lifecycle diagram expected-ops sets (dataflow_buffers.md ~line 252) -- `{STORE_SRC, PUSH}` now correct after Pass 2 fix.
- `BlockStateMachine.initialize()` table (dataflow_buffers.md ~line 238) against `sim/blockstate.py` lines 240-251 -- all four rows match.
- `__init__.py` re-exports (index.md ~line 29) against `python/ttl/__init__.py` -- all symbols present.
- `eltwise_add.py` walkthrough code (tensor_blocks_and_grid.md) against `examples/eltwise_add.py` -- matches verbatim.
- Navigation footers on all three content files -- present (decorators_and_threads.md, dataflow_buffers.md, tensor_blocks_and_grid.md).
- Clickable links in index.md chapter contents table -- all three links present and correctly formed.

**No feedback -- chapter approved.**
