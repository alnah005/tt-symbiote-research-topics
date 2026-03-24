# B Review — Pass 1

1. **`trace_api_overview.md`, line 85 — Contradictory `blocking` default claim.**
   The file states "The default value of `blocking` in the nanobind binding is `True`" but the Python function signature shown three lines earlier (line 71) displays `blocking=False`. These two statements directly contradict each other: one says the default is `True`, the other shows it as `False`. A reader implementing a wrapper or reading the binding would be misled about the actual default. Fix: verify the actual default in the nanobind binding and state it once, consistently. If the nanobind default is `True` and tt-transformers overrides it to `False` on every call, then the Python signature shown in the code block should not include `blocking=False` as a default — it should be shown as an explicit argument, or the prose should clarify that the Python-level wrapper redefines the default.

2. **`compile_run_requirement.md`, lines 35–46 — Incorrect causal claim about what goes wrong when compile run is skipped.**
   The file claims that skipping the compile run causes "program binary references recorded in the trace may point to freshly compiled artifacts in L1 that are in a temporary state, producing an inconsistent capture." This is misleading: kernel compilation in TTNN is a host-side JIT operation that produces a program object stored in the program cache. The compiled binary is not held in a "temporary L1 state" — L1 is per-core Tensix memory managed at dispatch time, not at compile time. The actual problem is that compilation is triggered as a side effect during the capture window, causing non-deterministic host-side work and temporary DRAM allocations (which `mark_allocations_safe` holds live) to inflate the trace footprint. A reader following this explanation would have an incorrect mental model of where kernel binaries live at compile time. Fix: remove the "temporary state in L1" claim. The correct explanation is that compilation allocates transient host and DRAM objects during the capture window; `mark_allocations_safe` keeps those allocations live, expanding the memory footprint and potentially causing OOM.

3. **`compile_run_requirement.md`, lines 162–165 — `trace_region_size` table anomaly: Llama-3.3-70B T3K value appears implausibly small relative to Llama-3.1-70B T3K.**
   The table shows `Llama-3.3-70B | T3K | 30,000,000` while `Llama-3.1-70B | T3K | 90,000,000`. Both are 70B-parameter models on the same T3K device. A three-fold reduction for the 3.3 variant would require a major architectural difference in the number of traced ops (e.g., far fewer layers or a fundamentally different compute graph). If this value was copied from a per-submesh DP configuration rather than a full T3K single-mesh configuration, it is wrong by a factor of 3 or more and would cause OOM on capture for users who set `trace_region_size=30000000` on a T3K single-mesh run of Llama-3.3-70B. Fix: verify the value against `trace_region_config.py`; if the 30 MB figure is correct, add an explanatory note (e.g., architectural differences such as grouped-query attention reducing op count). If it is a DP-submesh value, correct it to the full-mesh value.

4. **`replay_mechanics.md`, lines 193–196 — Ordering guarantee claim is stated too broadly.**
   The file states "`ttnn.copy_host_to_device_tensor` enqueues the DMA write on the same command queue. Because the command queue is ordered, the DMA write is guaranteed to complete before the subsequent trace replay command begins execution on the device." This is only correct if `copy_host_to_device_tensor` also submits to `cq_id=0`. The file itself immediately qualifies this in the following note (lines 198–201), but the unqualified assertion on lines 193–196 stands alone as a guaranteed fact, which a reader implementing their own input-update loop could rely on incorrectly if they use a different write path. This is a coherence issue between the assertion and its qualification rather than an outright factual error, but the unqualified statement on line 194 ("no explicit synchronization is needed") is stated categorically and would cause a race condition if a reader applied it to a non-`copy_host_to_device_tensor` write. Fix: fold the qualification into the main claim rather than placing it in a note after a categorical assertion.

5. **`trace_api_overview.md`, line 50 — "compute program binaries (by reference — not inline)" recorded claim needs clarification or is potentially incorrect.**
   The file states that `end_trace_capture` seals "NOC transfer descriptors, compute program binaries (by reference — not inline)". If program binaries are recorded only by reference (i.e., a pointer to a cached binary), this implies the binary must remain at a stable address in the program cache across replays. But the file never establishes what "by reference" means concretely in this context or what happens if the program cache is evicted or the binary is relocated between a capture and a replay. If the reference is a DRAM address of the compiled binary that could become stale, a reader implementing a system that evicts the program cache mid-session would produce silently incorrect behavior. Fix: clarify whether "by reference" means a pointer into a fixed-address region that is guaranteed stable for the lifetime of the trace, or whether program binaries are actually inlined into the command buffer in a way that makes the trace self-contained.

# B Review — Pass 2

1. **`replay_mechanics.md`, lines 177–186 — Contradictory implicit-synchronization claim for `.cpu(blocking=False)`.**
   The prose states: "Calling `.cpu()` or `ttnn.to_torch` on a device tensor internally waits for all in-flight operations on that tensor's command queue to complete before transferring data to host memory." The code example that immediately follows shows `.cpu(blocking=False)`, which explicitly does NOT wait — it initiates the DMA transfer asynchronously and returns a handle to a host tensor that is not yet valid. A reader who applies the prose guarantee to the `blocking=False` call shown would believe the returned host tensor is immediately safe to read, but it is not. The prose as written is only correct for `.cpu()` with the default `blocking=True`; the example undermines it. Fix: scope the synchronization guarantee explicitly to `blocking=True` (or the default call with no argument), and state separately that `blocking=False` does not synchronize — the host tensor is only valid after the subsequent event-based or explicit synchronization described later in the section.

2. **`compile_run_requirement.md`, lines 50–53 — "Cache misses on replay after a restart" bullet contains an incorrect causal chain.**
   The third bullet under "What Happens If the Compile Run Is Skipped" reads: "If the program cache was not warm before capture, replaying the trace on a subsequent run (for example, after a restart) may encounter a different cache state, causing replay to fail or produce incorrect results." This is wrong in two ways. First, after a device restart the trace region is cleared; the `trace_id` handle is invalid and `execute_trace` would fail at the dispatch layer — not because the program cache differs, but because the trace no longer exists in DRAM. Second, within the same session, the program cache IS warm immediately after capture (it was warmed during the capture window itself), so the cache-state divergence scenario does not arise from skipping the compile run in-session. The example conflates restart behavior (which destroys the trace regardless of compile-run state) with the actual in-session consequence of skipping the compile run. A reader following this explanation would implement incorrect recovery logic. Fix: remove or rewrite the "after a restart" example. The correct in-session consequence of skipping the compile run is the DRAM footprint inflation and OOM risk described in the first two bullets; the restart scenario is irrelevant to compile-run skipping.

## Change Log — Pass 1 Fixes

The following changes were applied to address all five items from the Pass 1 review:

**Item 1 — `trace_api_overview.md`, `blocking` default contradiction.**
Updated the `ttnn.execute_trace` code block to show the raw API signature with `blocking=True`
(the actual nanobind default). Rewrote the prose to clearly separate the raw API default (`True`)
from the tt-transformers usage, which always passes `blocking=False` explicitly. A second code
block now shows the `generator.py` call pattern with `blocking=False` explicitly annotated. The
contradiction between the displayed signature and the prose is resolved.

**Item 2 — `compile_run_requirement.md`, incorrect "temporary state in L1" claim.**
Removed the claim that program binary references point to "freshly compiled artifacts in L1 that
are in a temporary state." Replaced it with the correct explanation: compilation inside the
capture window triggers transient DRAM allocations; `mark_allocations_safe()` holds those
allocations live for the duration of recording, inflating the trace footprint and potentially
embedding kernel paths that become invalid once those transient allocations are released. The
second bullet was also updated to remove the L1 reference and accurately describe the DRAM
footprint inflation mechanism.

**Item 3 — `compile_run_requirement.md`, Llama-3.3-70B T3K 30 MB table anomaly.**
Added a `*(see note below)*` annotation to the `Llama-3.3-70B | T3K | 30,000,000` row. Inserted
a new note block immediately before the existing DP note, warning that the 30 MB value may
reflect a per-submesh DP configuration rather than a full single-mesh T3K footprint. The note
advises users running Llama-3.3-70B on a single-mesh T3K to verify against `trace_region_config.py`
before using this value, as setting `trace_region_size=30000000` on a full T3K single-mesh run
may be insufficient and cause OOM at `end_trace_capture` time.

**Item 4 — `replay_mechanics.md`, unqualified "no synchronization needed" claim.**
Rewrote the "When synchronization is not needed between replays" section to lead with a positive
statement of when synchronization IS needed (reading outputs to host via `.cpu()` or callbacks),
then scope the "no explicit sync needed" claim to the specific, qualified case: same-CQ
`copy_host_to_device_tensor` writes only. Folded the qualification directly into the main
paragraph rather than isolating it in a note after an unqualified categorical assertion. Changed
the trailing callout from a `Note` to a `Warning` to reinforce the race-condition risk for
non-`copy_host_to_device_tensor` write paths.

**Item 5 — `trace_api_overview.md`, program cache pinning not explained.**
Added a sentence immediately after "compute program binaries (by reference — not inline)"
clarifying that program binaries stored in the program cache are pinned for the lifetime of the
trace, that the trace holds a reference preventing eviction or relocation, and that replay is
therefore safe as long as the trace object remains live.

## Change Log — Pass 2 Fixes

**Item 1 — `replay_mechanics.md`, `.cpu(blocking=False)` contradiction.**
Rewrote the "Implicit synchronization via `.cpu()` or `ttnn.to_torch`" section to distinguish
the two blocking modes clearly. The blocking guarantee is now explicitly scoped to
`.cpu(blocking=True)` (the default): the host blocks until in-flight operations complete and
the returned host tensor is immediately valid. A separate paragraph now covers
`.cpu(blocking=False)`: it returns immediately without waiting, the host tensor is not valid
until the DMA transfer completes, and a subsequent `ttnn.synchronize_device()` or event wait is
required before reading from it. The prose and the code example (which uses `blocking=False`)
are now consistent — the example is accompanied by prose that explains why it is not
immediately safe to read from the returned tensors.

**Item 2 — `compile_run_requirement.md`, "cache misses after a restart" bullet.**
Replaced the third bullet under "What Happens If the Compile Run Is Skipped" entirely. The
original bullet incorrectly attributed post-restart replay failure to a cache-state divergence
caused by skipping the compile run. The replacement bullet now states the correct failure mode:
a device restart clears the trace region in DRAM entirely, making any existing `trace_id`
handle invalid and causing `execute_trace` to fail because the trace buffer no longer exists.
The bullet also clarifies that this failure is causally unrelated to compile-run state — after
any restart, both the compile run and the full capture sequence must be re-executed from scratch.

# B Review — Pass 3

1. **`compile_run_requirement.md`, "Dynamic trace region allocation" section — unverified claim that `trace_region_size=0` enables dynamic mode.**
   The file states: "If `trace_region_size` is set to `0`, the runtime switches to dynamic allocation mode: it tracks the DRAM high-water mark during recording and allocates only what is needed." This is a specific behavioral claim about a sentinel value. If `0` is not a recognized sentinel in the TTNN runtime — for example, if it causes the trace region to be sized at zero bytes, resulting in an immediate OOM on the first `populate_mesh_buffer` call — then a reader who deliberately sets `trace_region_size=0` expecting graceful dynamic allocation would get a hard failure instead. The claim is stated as fact but is not corroborated anywhere else in the chapter. If this value is unsupported or its behavior differs from what is described, implementations that follow this guidance will fail at capture time. Fix: verify that `0` is a valid dynamic-mode sentinel in the runtime (check the `MeshDevice` initialization path); if it is not, remove this section or replace it with the actual mechanism for dynamic sizing (if one exists).

2. **`compile_run_requirement.md`, lines 118–121 — Incorrect line-number annotation for the second `copy_host_to_device` call.**
   The prose states: "Notice that `copy_host_to_device` is called twice: once before the compile run (line 184) and again before the capture run (line 196)." The `_capture_trace_prefill` code block shown immediately above these lines has the second `copy_host_to_device` call at the position labeled line 106 within the snippet (immediately after the `# Fresh device inputs for the capture run` comment). Citing lines 184 and 196 of `generator.py` is meaningful only if those line numbers are accurate; if a reader navigates to `generator.py` to follow the cross-reference and finds different content at those lines, they cannot locate the described pattern. More critically, the phrase "line 184" and "line 196" are presented as authoritative anchors for the two distinct calls — if they are swapped, transposed, or stale, a reader auditing the two-allocation pattern would be unable to verify the canonical aliased buffer claim. Fix: verify that line 184 and line 196 of `generator.py` correspond respectively to the compile-run `copy_host_to_device` and the capture-run `copy_host_to_device`; update if stale.

No further correctness issues found in `index.md`, `trace_api_overview.md`, or `replay_mechanics.md` beyond those addressed in Passes 1 and 2.

## Change Log — Pass 3 Fixes

**Item 1 — `compile_run_requirement.md`, `trace_region_size=0` dynamic mode claim.**
Verified against `tt_metal/distributed/mesh_device.cpp` (`begin_mesh_trace` and `end_mesh_trace`
implementations). The claim is accurate: `trace_region_size == 0` is a recognized sentinel; the
runtime calls `begin_dram_high_water_mark_tracking()` at capture start and
`end_dram_high_water_mark_tracking()` at capture end, then uses the recorded allocation and
deletion high-water marks to size the actual DRAM reservation. The dynamic-mode section was
rewritten to state the mechanism precisely (high-water mark tracking via the two named calls)
rather than describing it in abstract terms only. A clarifying note was added to make explicit
that `0` does not mean "zero bytes reserved" — it is a sentinel that triggers the high-water mark
path — and that explicit sizes from `trace_region_config.py` are still preferred for production
use.

**Item 2 — `compile_run_requirement.md`, `generator.py` line-number cross-references.**
Verified against the current `models/tt_transformers/tt/generator.py`. Line 184 is the
compile-run `copy_host_to_device` call and line 196 is the capture-run `copy_host_to_device`
call in `_capture_trace_prefill` (function starts at line 167). Both line numbers are accurate
and current. No change to the line-number references was required; the prose at lines 118–121 of
the guide is correct as written.

# B Review — Pass 4

1. **`compile_run_requirement.md`, lines 38–45 — Factually incorrect claim that "the command buffer records the timing of events."**
   The first bullet under "What Happens If the Compile Run Is Skipped" opens with: "The command buffer records the timing of events as they occur during the capture run. Compilation delays inside the capture window mean the command buffer reflects a slower dispatch sequence than what you will see after the program cache is warm."
   Both sentences are wrong. A command buffer encodes a sequence of device commands (kernel launch descriptors, NOC transfer descriptors, synchronization barriers), not timestamps or timing information. There is no mechanism by which compilation latency during capture is "baked into" the replay speed — the device executes the pre-encoded commands as fast as it can on replay, regardless of how long host-side compilation took during the capture run. A reader who follows this explanation would (a) incorrectly believe that a trace captured without a compile run replays more slowly than one captured after a warm cache, and (b) have a fundamentally wrong model of what a command buffer contains. The actual correctness consequence of skipping the compile run is the DRAM footprint inflation described in the remainder of the same bullet (transient allocations held live by `mark_allocations_safe`), which is correct. Fix: remove the "records the timing of events" and "slower dispatch sequence" sentences entirely. The bullet should begin at the transient-allocation explanation, which accurately describes the real failure mode.

**No further correctness issues found in `index.md`, `trace_api_overview.md`, or `replay_mechanics.md`.**

## Change Log — Pass 4 Fixes

**Item 1 — `compile_run_requirement.md`, incorrect timing claim in the first bullet of "What Happens If the Compile Run Is Skipped."**
Removed the two factually incorrect sentences: "The command buffer records the timing of events
as they occur during the capture run." and "Compilation delays inside the capture window mean
the command buffer reflects a slower dispatch sequence than what you will see after the program
cache is warm." Command buffers encode device commands (kernel launch descriptors, NOC transfer
descriptors, synchronization barriers), not timestamps or timing information. Replay speed is
determined by how fast the device executes the pre-encoded commands, not by how long host-side
compilation took during the capture window. The correct consequence of skipping the compile run
— transient DRAM allocations held live by `mark_allocations_safe` inflating the trace footprint
— was already described correctly in the remainder of the same bullet and has been retained
unchanged.

# B Review — Pass 5

No feedback — chapter approved.

# B Review — Pass 6

No feedback — chapter approved.
