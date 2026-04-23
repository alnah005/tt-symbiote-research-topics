# Why `synchronize_device` Blocks Trace Capture

This file explains the Metal Trace capture contract, what happens when `ttnn.synchronize_device()` is called inside a trace bracket, whether the call is silently dropped or raises an error, and the full scope of the problem across the hybrid decoder layer stack. By the end you will understand precisely why the presence of `ttnn.synchronize_device` in `_maybe_all_gather` makes the entire Qwen3.6-35B-A3B hybrid attention stack non-traceable.

---

## The Metal Trace Capture Contract

Metal Trace capture works by intercepting all op submissions to CQ0 during a special recording session. Between `ttnn.begin_trace_capture(device, cq_id=0)` and `ttnn.end_trace_capture(device, cq_id=0)`, every encoded kernel dispatch command that arrives at CQ0 is written into a trace recording buffer in device DRAM at the same time it is executed live. The result is a verbatim binary copy of the command stream that can be replayed later via `ttnn.execute_trace`.

The capture contract imposes one strict requirement:

**Within a capture bracket, the host may only submit device-side commands to CQ0. Any operation that causes the host to pause and wait for the device — rather than simply enqueuing a command and continuing — violates the recording invariant.**

The reason for this requirement is structural: the trace buffer records only device-side commands, not host-side execution state. A host-blocking call is not a command submitted to CQ0; it is a pause in the host's progress, waiting for the device to acknowledge something. That pause is invisible to the trace recorder. The recorded command stream contains no representation of the fact that the host waited. On replay, the trace replays the recorded commands at maximum speed, with no pause at the point where the host-blocking call was issued during capture.

---

## What Happens When `synchronize_device` Is Called Inside a Trace Bracket

When `ttnn.synchronize_device(mesh_device)` is called between `ttnn.begin_trace_capture` and `ttnn.end_trace_capture`, the following sequence occurs:

1. The call executes normally: the host enqueues a finish command to CQ0, enters a blocking wait, and returns when the device's queue is empty.
2. The trace recorder observes the finish command being enqueued and records it into the trace buffer.

> **Note — Correct mechanism (reconciled):** `ttnn.synchronize_device` enqueues a Finish token to CQ0 (a real device-side command) and then the **host blocks** waiting for the device to acknowledge that Finish.
>
> - The **Finish token IS a CQ0 command** and IS recorded by the trace recorder into the trace buffer during capture.
> - The **host-side blocking wait is NOT a device command** and is NOT recorded by the trace recorder.
> - On trace replay, the Finish token is re-issued to the device (providing a device-level fence), but the host **never blocks** — trace replay is a fire-and-forget command re-issue with no host waits.
>
> The core problem for trace capture is that `synchronize_device` introduces a host-blocking wait inside the capture bracket, violating the contract that trace capture must be a fully async device command stream. The CQ0 FIFO ordering guarantee already ensures that the all_gather output is available to the next enqueued op; the `synchronize_device` call is therefore unnecessary for correctness and must be removed so the capture bracket contains only device-side commands.

The practical consequence:

- **During capture:** The host blocks as usual, paying the full PCIe round-trip cost. The op sequence that follows `synchronize_device` does not start executing on the host until the queue drains. The capture proceeds correctly, but the host-blocking latency is incurred on every capture run and every non-traced execution path.
- **On replay:** The Finish token is re-issued to the device (providing a device-level fence), but the host-side blocking wait is absent. The ops that preceded the synchronize point have been pre-encoded in the trace buffer in the order they were submitted during capture, so they will execute in the same order on replay. The sequencing guarantee that the synchronize call was intended to provide is already provided by CQ0 FIFO ordering.

The implication is subtle: **for sequencing purposes, `synchronize_device` inside a trace bracket is redundant even on the first capture replay, because trace replay preserves the op submission order and the re-issued Finish token provides a device-level fence.** The actual harm is not that replay produces wrong outputs due to a missing barrier; the harm is the structural incompatibility with the trace-first design requirement — the host-blocking wait violates the contract that the capture bracket must contain only async device commands.

> **Warning:** The behavior described above — where `ttnn.synchronize_device` inside a trace bracket is recorded as a finish command and replayed without a host-side block — is the behavior observed in practice. It is not a documented guarantee. Future tt-metal versions may handle this differently. The safe and correct approach is to ensure that `ttnn.synchronize_device` is never called inside a trace bracket: not during the capture run, and not during any execution that is intended to be captured.

---

## Does `synchronize_device` Inside a Trace Bracket Raise an Error?

In the current tt-metal runtime (as of the time this guide was written), `ttnn.synchronize_device` called inside a `ttnn.begin_trace_capture` / `ttnn.end_trace_capture` bracket does **not** raise a Python exception and does **not** produce a visible error message. The call executes silently: the host blocks, the queue drains, and execution continues. The trace recorder silently drops the host-side blocking behavior and records only the device-side finish command.

This silent behavior is the most dangerous aspect of the interaction. A developer who inserts a `ttnn.synchronize_device` call inside a module's forward method and then runs that module under `TracedRun` may observe correct outputs (because the capture run executed correctly, with the synchronize call doing its usual work), never realizing that replay would behave differently if the synchronize had been providing an ordering guarantee that is absent from the trace.

In the specific case of `_maybe_all_gather`, the model appears to work correctly under `TracedRun` on the capture run. Failures, if any, would appear on replay — but because the TTNN CQ0 ordering guarantee already provides the same sequencing as `synchronize_device`, replay typically also produces correct outputs. The visible problem is not correctness but traceability: the presence of `synchronize_device` adds host-blocking latency that the trace does not eliminate during capture runs and non-traced execution paths, and it signals an incorrect design.

Regarding `all_gather_async` and cycling semaphores: since `synchronize_device` IS recorded (as a Finish token) and IS replayed (as a device fence), but the host-side wait is absent on replay, the Finish token on replay does not interact with any host-readable semaphore state — that would only be a concern if code explicitly reads semaphore values from the host side. The more important point is that `synchronize_device` is **unnecessary** and must be removed before replacing the underlying all_gather with `all_gather_async`, which requires proper cycling semaphore management. A residual `synchronize_device` call alongside `all_gather_async` would not correctly manage the cycling semaphore state and would defeat the double-buffering design — but that is a separate concern from whether the Finish token is recorded.

> **Key finding:** `ttnn.synchronize_device` inside a trace bracket neither raises an error nor silently corrupts outputs in the current runtime. It executes during capture (with its full host-blocking cost); the Finish token is recorded in the trace and replayed as a device fence, but the host-side blocking wait is not. The core problem is structural: the host-blocking wait violates the capture contract (fully async device command stream), and the call indicates a design assumption (that host-side barriers are needed for ordering) that is false in single-CQ mode. Removing it is also a prerequisite for replacing the underlying all_gather with `all_gather_async`, which requires proper cycling semaphore management rather than a blunt host sync.

---

## Scope of the Problem Across the Hybrid Decoder Layer Stack

The Qwen3.6-35B-A3B decoder stack is a hybrid architecture that alternates between DeltaNet linear attention layers (`TTNNQwen3LinearAttention`) and standard full-attention layers (`TTNNQwen3FullAttention`). Both module types call `_maybe_all_gather` in their decode-mode forward passes, as documented in [`call_sites_and_control_flow.md`](./call_sites_and_control_flow.md).

In tt-symbiote's `TRACED` run mode, the `LayerStack` that iterates over the decoder layers is intended to be captured as a single trace: one `ttnn.begin_trace_capture` / `ttnn.end_trace_capture` bracket enclosing the forward passes of all layers in the stack. This is the design that enables end-to-end trace execution with maximum throughput — the per-step decode latency is reduced by eliminating the host-side dispatch overhead for every op in every layer.

For this to work, every module in the stack must satisfy the trace capture contract: no host-blocking calls inside the bracket. Currently, `_maybe_all_gather` violates this contract on every call on multi-device deployments. Because `_maybe_all_gather` is called at least twice per layer (once after QKV projection, once before the output projection), a hybrid stack with $H$ attention layers produces at least $2H$ `ttnn.synchronize_device` calls per decode step inside the trace bracket.

The consequences:

1. **Latency:** Each `ttnn.synchronize_device` call adds a PCIe round-trip (10–30 µs) plus any remaining device execution time. For $H = 28$ hybrid layers and 2 calls per layer, the total host-blocking overhead is at least 56 synchronize calls per decode step. At 20 µs per call (a conservative estimate for a nearly-idle device queue), this is 56 × 20 µs = 1.12 ms of pure synchronization overhead per decode step.

2. **Prerequisite for `all_gather_async`:** The current implementation may use synchronous `ttnn.all_gather` inside `_maybe_all_gather`. If the goal is to replace it with `ttnn.experimental.all_gather_async` — which is required for true trace compatibility with cycling semaphores — then `ttnn.synchronize_device` must be removed as part of the same change. `all_gather_async` requires proper cycling semaphore management (device-side, coordinated by the CCL op itself); using a blunt `synchronize_device` alongside it does not correctly fulfill that management role. The Finish token that `synchronize_device` enqueues provides a device-level fence but does not interact with the semaphore cycling protocol that `all_gather_async` depends on for double-buffering correctness.

3. **Host-blocking latency on every non-replay execution path:** The host-blocking cost of `synchronize_device` is paid in full during every capture run and during any non-traced execution path (e.g., eager mode, prefill). On replay via `ttnn.execute_trace`, the Finish token is re-issued to the device without a host wait — so the synchronization overhead it imposed during capture is absent on replay. This asymmetry means the trace correctly omits the host wait, but the unnecessary latency is still paid on every capture and non-traced decode step. The design intent of trace execution — a single `ttnn.execute_trace` call issuing all ops with no per-layer host involvement — is also compromised: the capture bracket is not a purely async device command stream as the contract requires, which may affect runtime behavior or future runtime enforcement.

To make the hybrid decoder stack fully traceable, `ttnn.synchronize_device` must be removed from `_maybe_all_gather` across both `TTNNQwen3FullAttention` and `TTNNQwen3LinearAttention` (or their shared base class). The removal also requires replacing the underlying all_gather op with a trace-compatible variant, as described in subsequent chapters.

---

**Next:** [Chapter 2 — The Async CCL Pattern in tt-transformers for Traced Decode](../ch2_async_ccl_pattern/index.md)

---

## Change Log (B Review Pass 1)
- Reconciled contradiction: Finish token IS recorded by trace (device CQ0 command); host-side blocking wait is NOT (item 1)
- Corrected consequence 3: host is not required on every replay; Finish token is replayed without host wait (item 2)
- Clarified all_gather_async semaphore concern relative to resolved item 1 (item 3)
