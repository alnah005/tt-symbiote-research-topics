# B Review — Chapter 1: Pass 1

1. **File:** `index.md`, ~line 87
   **Issue:** The glossary definition of **dispatch overhead** states it covers "phases 1–3 of the dispatch path (argument validation, kernel selection, command encoding)" — explicitly excluding phase 4 (CQ submission). However, the summary table in `host_dispatch_path.md` (line 123) includes CQ submission in the **Total dispatch overhead** row (adding 1–5 us to reach the stated ~17–63 us total). A reader using the glossary definition to calculate or bound dispatch overhead would arrive at 16–58 us (phases 1–3 only), not the 17–63 us stated in the table. This creates a direct numerical inconsistency.
   **Fix:** Either (a) broaden the glossary definition to say "phases 1–4" so it matches the table's calculation, or (b) relabel the table's bottom row as "Total host overhead (including CQ submission)" and keep the glossary covering only phases 1–3. Pick one definition and apply it consistently everywhere the term appears.

2. **File:** `command_queues.md`, ~line 13
   **Issue:** The FIFO correctness claim states: "the FIFO order ensures A completes before B starts." FIFO ordering on a command queue guarantees ordering of *dispatch* (the order in which commands are issued to device cores), not necessarily that every memory write from A is globally visible before B's first read. Presenting the FIFO guarantee as ensuring full *completion* of A before *start* of B — without qualification — could lead a reader to incorrectly omit explicit memory barriers or synchronization in cases where the hardware only guarantees ordered issuance, not ordered completion visibility.
   **Fix:** Qualify the statement: "the FIFO order ensures A is *dispatched and completes* before B is dispatched" only if the hardware enforces completion ordering (not just issuance ordering) between sequential commands on the same CQ. If the guarantee is issuance ordering only, rewrite to: "the FIFO order ensures A is issued to the device before B is issued; because the device executes commands sequentially on the same queue, A's writes are visible to B." This makes the chain of reasoning explicit and correct.

# B Review — Chapter 1: Pass 2

1. **File:** `command_queues.md`, ~line 19 and line 34 (diagram)
   **Issue:** The file states (line 19): "the firmware advances the read pointer *after each command is dispatched to a device core for execution*." Dispatched-to-core means the command has been sent for execution and the read pointer has already moved. Yet the diagram (lines 25–32) shows the read pointer sitting *under* cmd C while the caption (line 34) says "the device … is currently processing C." If C has been dispatched (per the definition on line 19), the read pointer must already be past C — at the first empty slot — not aligned with C. As drawn, a reader building a CQ consumer would place the pointer one position too early: they would not advance past a command until execution completes, rather than until dispatch to the core completes.
   **Fix:** Either (a) move the diagram's read pointer to the first empty slot (past C) and update the caption to "the device has dispatched A, B, and C; C is executing on the cores," or (b) redefine line 19 to say the read pointer advances only after execution completes, and update the caption accordingly. Pick one consistent model and apply it to both the diagram and the prose definition.

---

## Change Log — Pass 1 Fixes

- `index.md`: Updated "dispatch overhead" glossary entry to include phase 4 (CQ submission, 1–5 us), aligning with the 17–63 us total in host_dispatch_path.md.
- `command_queues.md`: Clarified FIFO ordering guarantee — added explicit chain: ordered issuance + sequential per-CQ execution implies A completes before B starts.

## Change Log — Pass 2 Fixes

- `command_queues.md`: Updated the CQ diagram so the read pointer is shown past cmd C (pointing to next slot) when C is "currently processing", consistent with the prose definition that the read pointer advances on dispatch-to-core.

# B Review — Chapter 1: Pass 3

1. **File:** `command_queues.md`, ~line 173
   **Issue:** The summary table row "Used for data movement in dual-CQ mode" marks CQ0 as "No (compute only)." This implies CQ0 is *restricted* to compute operations when dual-CQ mode is enabled, which is false. CQ0 is a general-purpose queue; the typical dual-CQ pattern dedicates CQ0 to compute and CQ1 to transfers, but nothing prevents enqueuing a data movement command to CQ0 in dual-CQ mode. A reader implementing a dual-CQ program would incorrectly believe all data movement must go through CQ1, potentially misstructuring synchronization for cases where CQ0 data movement is intentional or necessary.
   **Fix:** Change the CQ0 cell for "Used for data movement in dual-CQ mode" from "No (compute only)" to "Yes, but typically compute only — CQ1 handles transfers by convention" (or equivalent phrasing that marks it as a recommended pattern, not a hard restriction).

## Change Log — Pass 3 Fixes

- `command_queues.md`: Corrected CQ0 data movement entry in summary table — CQ0 is not hardware-restricted to compute only; the compute/data-movement split between CQ0 and CQ1 is a convention. Updated table and surrounding prose accordingly.

# B Review — Chapter 1: Pass 4

No feedback — chapter approved.

# B Review — Chapter 1: Pass 5

1. **File:** `command_queues.md`, ~line 167
   **Issue:** The navigation footer links to `../ch2_async_ops/index.md`, but that file does not exist in the repository. A reader who finishes Chapter 1 and follows the "Next" link will get a broken reference with no path forward through the guide.
   **Fix:** Create `ch2_async_ops/index.md` (or at minimum a stub), or update the footer to point to whatever file actually exists as the next entry point until Chapter 2 is written.

# B Review — Chapter 1: Pass 6

1. **File:** `host_dispatch_path.md`, ~line 16
   **Issue:** The parenthetical about async op mode states: "the four phases still execute before the command is enqueued." Phase 4 *is* CQ submission — i.e., enqueuing the command. The sentence therefore claims that phase 4 (enqueuing) completes before the command is enqueued, which is self-contradictory. In async mode, the host typically returns early by handing work to a background thread; it does not guarantee all four phases complete before the call returns. Stating that all four phases including enqueuing happen "before the command is enqueued" is internally inconsistent and would mislead a reader trying to understand what async mode actually defers.
   **Fix:** Correct the parenthetical to accurately describe what async mode changes. For example: "Async op mode, covered in Chapter 2, returns control to the caller before the four phases complete — the phases are executed by a background dispatch thread — but the command still reaches the CQ through the same four-phase path." Alternatively, if the intent is that phases 1–3 still complete on the calling thread and only CQ submission is deferred, state that explicitly: "phases 1–3 still execute on the calling thread; only phase 4 (CQ submission) is handed off to a background thread."

## Change Log — Pass 6 Fixes

- `host_dispatch_path.md`: Corrected async op mode description — all four dispatch phases run on a background thread; the Python caller returns before phases begin. Removed self-contradictory claim that phases execute "before the command is enqueued."

# B Review — Chapter 1: Pass 7

No feedback — chapter approved.
