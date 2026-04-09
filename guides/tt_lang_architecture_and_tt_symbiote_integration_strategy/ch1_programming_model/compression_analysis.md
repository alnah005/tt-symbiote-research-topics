# Chapter 1 — Change Log

## 2026-04-09

1. **tensor_blocks_and_grid.md (line ~186):** Fixed the `node()` coordinate decomposition formula. The previous formula (`node_col = k mod C, node_row = floor(k/C)`) was incorrect. Updated to match the actual source code in `sim/corecontext.py`, which iterates `reversed(grid)` and produces `node_col = floor(k/R), node_row = k mod R` for a grid `(C, R)`. Added an explanatory sentence about the reverse-iteration logic.

2. **dataflow_buffers.md (`back_slot()`):** Added a clarifying note that the `back_slot()` formula `(head + visible) % cap` relies on a single-reservation invariant — each DFB supports at most one outstanding reservation at a time. Without this invariant, a second concurrent reservation would receive the same slot index, since `reserved` is not included in the offset calculation.

3. **dataflow_buffers.md (line ~252):** Fixed the expected ops set in the Compute thread lifecycle diagram for the `store_dst` transition from MW to MR. Changed `{STORE_SRC}` to `{STORE_SRC, PUSH}` to match the source code in `blockstate.py` (line 171-173), which defines the MW `store_dst` transition as `{ExpectedOp.STORE_SRC, ExpectedOp.PUSH}`.

---

# Compression Analysis: Chapter 1 — TT-Lang Programming Model — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~948 lines
- Estimated post-compression line count: ~820 lines
- Estimated reduction: ~13%

## CRUCIAL Suggestions

### [tensor_blocks_and_grid.md] ~lines 256
**Issue:** Double-buffering explanation restates what `dataflow_buffers.md` line 21 already covers. The sentence "With `block_count=2`, double buffering is enabled: one block can be filled by DM while compute processes the other" is nearly identical to `dataflow_buffers.md` line 21: "With `block_count=2`, the producer can write to one slot while the consumer reads from another — classic double buffering that hides data transfer latency."
**Suggestion:** Replace the sentence at line 256 with a cross-reference: "Double buffering behavior is explained in [`dataflow_buffers.md`](./dataflow_buffers.md#make_dataflow_buffer_like)."

### [tensor_blocks_and_grid.md] ~lines 263-327 (read/write DM threads)
**Issue:** The tiling loop pattern (`node_col, node_row = ttl.node(dims=2)` / `for local_row in range(rows_per_node)` / bounds check / `for local_col ...`) is shown three times in the walkthrough: once for compute (lines 263-276), once for read (lines 289-303), and once for write (lines 315-327). The loop structure is identical in all three; only the inner body differs.
**Suggestion:** Show the full loop once (in the compute thread), then for read and write threads show only the inner body with a note like "Using the same tiling loop as compute above:". This removes ~20 duplicate lines.

### [index.md] ~lines 71-77 (Key Takeaways)
**Issue:** The five bullet points largely restate what the chapter table (lines 63-69), the opening paragraph (line 3), and the per-file introductions already say. For example, "Decorator-driven structure: Every kernel is a @ttl.operation containing exactly one @ttl.compute and two @ttl.datamovement closures, mirroring the Tensix core's three RISC-V processors" duplicates `decorators_and_threads.md` line 3 almost verbatim, and is also stated at `index.md` line 20.
**Suggestion:** Remove the Key Takeaways section entirely. The chapter introduction (lines 1-25) and table of contents (lines 63-69) already provide sufficient orientation. Alternatively, reduce to 2-3 bullets that add genuinely new synthesis rather than restating what each sub-file's header already says.

### [decorators_and_threads.md] ~lines 171-182 (Simulator vs Compiler table)
**Issue:** The table partially duplicates the dual-backend explanation already given in `index.md` lines 20-25. Specifically, the row for `@ttl.operation(grid=...)` restates "Resolves grid, executes body, builds Program, runs GreenletScheduler" vs "compiles threads to MLIR, generates C++ kernel sources" which is the same information as index.md's compiler/simulator path description.
**Suggestion:** Trim the table to only the decorator-specific rows (`@ttl.compute` and `@ttl.datamovement`). Replace the `@ttl.operation` row and the closing paragraph (lines 182) with a forward-reference to `index.md`'s dual-backend description.

## MINOR Suggestions

### [dataflow_buffers.md] ~lines 82-86 (back_slot NOTE comment)
**Issue:** The 5-line NOTE comment inside the `back_slot()` code block ("NOTE: This formula assumes at most one outstanding reservation...") restates what the prose added as a correction (per the change log entry #2) and what the invariant formula at line 91-93 already implies. Having it in both the code comment and the change log and the invariant section is triple coverage.
**Suggestion:** Shorten the code comment to a single line: `# Assumes at most one outstanding reservation at a time.` The detailed explanation is already in the surrounding prose.

### [dataflow_buffers.md] ~lines 89-93
**Issue:** The invariant "$\text{visible} + \text{reserved} + \text{free} = \text{cap}$" is stated, and then immediately restated as "$\text{free} = \text{cap} - \text{visible} - \text{reserved}$". The second form is just an algebraic rearrangement of the first and is also the literal code of the `free()` method shown 10 lines above.
**Suggestion:** Remove line 93 ("where $\text{free} = \text{cap} - \text{visible} - \text{reserved}$"). The formula and the code already convey this.

### [decorators_and_threads.md] ~lines 47-55 and [dataflow_buffers.md] ~lines 162-165
**Issue:** The `with` statement pattern for `wait()`/`reserve()` is shown in `decorators_and_threads.md` (compute example) and again in `dataflow_buffers.md` (context manager usage section). Both show essentially the same pattern.
**Suggestion:** In `decorators_and_threads.md`, keep the brief example but remove the inline explanation of what `with` does (it is already the subject of `dataflow_buffers.md` lines 157-176). Add a forward-reference to the DFB file.

### [tensor_blocks_and_grid.md] ~lines 278-283 (with statement explanation)
**Issue:** The 5-step numbered explanation of the `with` statement (calls `a_dfb.wait()`, blocks until pushed, creates temporary Block, writes result, `__exit__` fires pop/push) restates the context manager lifecycle already covered in `dataflow_buffers.md` lines 157-176.
**Suggestion:** Replace with a brief note: "The `with` block acquires input blocks via `wait()` and an output slot via `reserve()`; on exit, `pop()` and `push()` fire automatically (see [dataflow_buffers.md](./dataflow_buffers.md#context-manager-usage))."

### [tensor_blocks_and_grid.md] ~lines 246
**Issue:** The sentence "The `-(-a // b)` idiom computes $\lceil a/b \rceil$ in Python" is a general Python idiom explanation unlikely to be needed by the target audience (kernel developers).
**Suggestion:** Remove the explanatory sentence; the code comment `# Ceiling division` is sufficient.

### [index.md] ~lines 31-50 (abridged __init__.py listing)
**Issue:** The code block shows import symbols with inline comments that duplicate the table at lines 63-69 (which maps files to topics covering these same symbols).
**Suggestion:** Remove the inline comments from the import listing (e.g., `# @ttl.operation — top-level kernel decorator`). The chapter contents table and the sub-files themselves explain each symbol.

## VERDICT
- Crucial updates: yes

---

## 2026-04-09 — Compression Pass 1: CRUCIAL Suggestions Applied

All four CRUCIAL compression suggestions were applied:

1. **tensor_blocks_and_grid.md ~line 256:** Replaced the duplicate double-buffering explanation with a cross-reference to `dataflow_buffers.md`.

2. **tensor_blocks_and_grid.md ~lines 263-327:** Extracted the shared tiling loop (node coordinates, local_row/local_col iteration, bounds checks) into a single "Shared Tiling Loop" section shown before the three thread walkthroughs. Each thread section now shows only its differing inner body with a comment referencing the shared loop.

3. **index.md ~lines 71-77:** Reduced Key Takeaways from 5 restated bullets to 3 genuinely synthetic bullets covering implicit synchronization, write-once-run-two-ways, and grid-aware scaling.

4. **decorators_and_threads.md ~lines 171-182:** Removed the redundant dual-backend prose paragraph and closing sentence. Replaced with a cross-reference to `index.md` for the architecture overview, keeping only the decorator-specific comparison table.

---

# Compression Analysis: Chapter 1 — TT-Lang Programming Model — Pass 2

## Summary
- Pass 2 re-checks the 4 CRUCIAL items identified in Pass 1
- All 4 CRUCIAL items were applied in the intervening edit pass
- No remaining CRUCIAL-level duplication detected

## Re-Check of Pass 1 CRUCIAL Items

### 1. [tensor_blocks_and_grid.md] Double-buffering duplication
**Status:** RESOLVED. Line 256 now uses a cross-reference — `(see [dataflow_buffers.md](./dataflow_buffers.md#make_dataflow_buffer_like) for details)` — instead of restating the double-buffering explanation.

### 2. [tensor_blocks_and_grid.md] Tiling loop repeated across three threads
**Status:** RESOLVED. Lines 258-274 extract the shared tiling loop into a single "Shared Tiling Loop" section. The compute, read, and write thread sections now show only their differing inner bodies with `# ... shared tiling loop ...` placeholder comments.

### 3. [index.md] Key Takeaways restating earlier content
**Status:** RESOLVED. The Key Takeaways section (lines 71-75) was reduced from 5 restated bullets to 3 synthetic bullets that add genuine cross-cutting observations (implicit synchronization, dual-backend portability, grid-aware scaling) not stated verbatim elsewhere.

### 4. [decorators_and_threads.md] Simulator vs Compiler table duplicating index.md
**Status:** PARTIALLY RESOLVED. Line 173 adds a cross-reference to `index.md` for the architecture overview. The `@ttl.operation` row (line 179) still contains "Resolves grid, executes body, builds Program, runs GreenletScheduler" vs "Resolves grid, compiles threads to MLIR, generates C++ kernel sources" — information already in `index.md` lines 21-23. However, this remaining overlap is minor: the table row provides a side-by-side comparison format that `index.md`'s prose does not, and removing it would leave a gap in the table. Downgraded from CRUCIAL to MINOR.

## Load-Bearing Evidence

- **index.md line 73:** `"Decorator + DFB = implicit synchronization:" The three-thread structure and ring-buffered DFBs eliminate explicit locks or barriers; all inter-thread coordination is encoded in reserve/push/wait/pop state transitions.` — This Key Takeaway bullet synthesizes information from both `decorators_and_threads.md` and `dataflow_buffers.md` into a single cross-cutting insight not found in either sub-file. Cutting it would lose the only place where the reader sees the decorator-DFB interaction principle stated concisely.

- **tensor_blocks_and_grid.md line 256:** `The block_count=2 enables double buffering (see [dataflow_buffers.md](./dataflow_buffers.md#make_dataflow_buffer_like) for details).` — The cross-reference is the only navigational link from the walkthrough's buffer creation to the full DFB explanation. Removing it would leave the reader without a pointer to the ring-buffer mechanics that explain why `block_count=2` matters.

- **decorators_and_threads.md line 173:** `For the dual-backend architecture overview, see [index.md](./index.md#position-in-the-software-stack).` — This cross-reference is the only explicit link between the decorator-level comparison table and the high-level architecture description. Without it, a reader encountering the Simulator vs Compiler table would not know that `index.md` provides the broader context.

- **tensor_blocks_and_grid.md lines 258-274 (Shared Tiling Loop section):** This factored-out loop shows the `ttl.node(dims=2)` / `rows_per_node` / bounds-check pattern once for all three threads. Cutting it would force the reader to mentally diff three identical loop structures to find the per-thread differences.

## MINOR Suggestions

### [decorators_and_threads.md] ~line 179 — `@ttl.operation` row in Simulator vs Compiler table
**Issue:** The `@ttl.operation` row still partially restates information from `index.md` lines 21-23. Downgraded from the Pass 1 CRUCIAL finding because the cross-reference at line 173 now provides context, and the tabular format adds value that the prose does not.
**Suggestion:** Shorten the `@ttl.operation` row cells to focus on what differs from the decorator-specific rows. For example: Simulator cell could say "Builds `Program`, runs `GreenletScheduler`" (removing "Resolves grid, executes body" which applies to both paths). Compiler cell could say "Compiles to MLIR, generates C++ sources" (removing "Resolves grid" which is shared). Saves ~15 words.

### [tensor_blocks_and_grid.md] ~lines 343-362 — Data Flow Summary diagram
**Issue:** The ASCII diagram at the end of the walkthrough visually restates the DM0/compute/DM1 data flow that the preceding three thread sections (lines 280-324) already describe in code. It is useful as a quick reference but adds ~20 lines.
**Suggestion:** Keep the diagram but consider moving it to immediately after the "Shared Tiling Loop" section (before the per-thread details) so it serves as an orientation map rather than a summary that repeats what was just read.

## VERDICT
- Crucial updates: no
