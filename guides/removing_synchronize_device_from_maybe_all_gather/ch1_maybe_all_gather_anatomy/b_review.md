# B Review — Pass 1

1. [`why_this_blocks_trace_capture.md`, ~26, **Factual contradiction: finish command recorded vs. not recorded**]
   `index.md` (line 12) states: "host-side wait points like `ttnn.synchronize_device` are not device-side commands and are not recorded in the trace buffer." `why_this_blocks_trace_capture.md` (~line 26) states the opposite: "The trace recorder observes the finish command being enqueued and records it into the trace buffer." These two claims directly contradict each other on the core technical fact — whether the finish command appears in the trace or not. A reader who reads both will receive mutually exclusive accounts of the recording mechanism. One of the two statements must be corrected to match the actual tt-metal runtime behavior, and the rationale for why `synchronize_device` is problematic for trace must be grounded in whichever account is accurate.

2. [`why_this_blocks_trace_capture.md`, ~65, **Factual contradiction: "host must dispatch each layer's ops" vs. traced-replay model**]
   Consequence item 3 in the "Scope" section claims: "the host must be involved in dispatching each layer's ops rather than issuing a single `ttnn.execute_trace` command." This directly contradicts the replay model described earlier in the same file, which establishes that the trace (including the recorded finish command) is replayed via `ttnn.execute_trace` without per-layer host dispatch. If the trace is captured and replayed as described, the host does issue a single `ttnn.execute_trace`. Fix: restate consequence 3 accurately — e.g., that the host-blocking latency is paid on every capture run and on any non-traced execution path, not that host dispatch is required on every replay.

3. [`why_this_blocks_trace_capture.md`, ~45, **Potential factual error: claim that `synchronize_device` "signals an incorrect design that would become a problem if all_gather_async were used"**]
   The file states: "it signals an incorrect design that would become actively harmful if the all_gather were replaced with an async variant requiring cycling semaphore management." However, if `ttnn.synchronize_device` actually IS recorded into the trace buffer (as claimed in the Note block), then on replay it would reset the semaphore cycling counter on every replay step, not just during capture — which would be an active correctness failure, not merely structural unsoundness. If it is NOT recorded (as claimed in `index.md`), then on replay it would be absent and would not reset the semaphore. The severity of the consequence depends entirely on resolving issue #1 above. As written, the severity claim is ambiguous and a reader implementing `all_gather_async` could underestimate the risk.

# B Review — Pass 2

No feedback — chapter approved.
