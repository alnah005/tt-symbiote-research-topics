## Pass 1

1. **`what_is_tracy.md`, line 53 — wrong env var used to illustrate Tracy capture.**
   The bash example in the Two-Process Model section uses `TT_METAL_DEVICE_PROFILER=1` when launching the workload. That env var enables the *device CSV profiler*, not Tracy. Tracy is enabled at build time via `TRACY_ENABLE` / `-DENABLE_PROFILER=ON`; no env var is needed at runtime to activate Tracy instrumentation. A reader following this example would think `TT_METAL_DEVICE_PROFILER=1` is required to get Tracy data, and would also inadvertently run the device profiler while believing they are running a Tracy-only capture. Fix: remove `TT_METAL_DEVICE_PROFILER=1` from the Tracy example command (or replace it with a note that the device profiler is a separate tool activated separately).

2. **`what_is_tracy.md`, line 44 — capture server does not timestamp events.**
   The text states that `tracy-capture` "timestamps it against its own clock." In Tracy's architecture, the *client* (profiled process) timestamps every zone event using its own monotonic clock before placing it in the ring buffer. The server receives those already-timestamped events over the socket; it does not assign or re-assign timestamps. The server's role is to receive, compress, and write to disk. Fix: change "timestamps it against its own clock" to something like "receives the already-timestamped event stream, compresses it, and writes it to disk."

3. **`two_profilers_compared.md`, line 69 — program cache misses do not inflate `DEVICE KERNEL DURATION`.**
   The Known Blind Spots section states: "a first-call kernel recompilation inflates `DEVICE KERNEL DURATION` for that call." Kernel compilation (JIT program creation) is a host-side activity. The device profiler's cycle counters start only once the kernel is actually executing on Tensix cores; they do not run during host-side compilation. A recompile would inflate Tracy's `host_dispatch_time` zone, not the device-profiler CSV column. Fix: correct the statement to say that a program cache miss inflates the *host dispatch time* (visible in Tracy), not `DEVICE KERNEL DURATION`, and that the CSV gives no indication of whether a long preceding dispatch was caused by recompilation.

## Pass 2

**No feedback — chapter approved.**

All three issues raised in Pass 1 have been resolved in the current files: the bash example in `what_is_tracy.md` correctly omits any runtime env var for Tracy activation; the two-process model description correctly attributes timestamping to the client; and the program-cache-miss blind spot correctly attributes the inflation to `host_dispatch_time` (Tracy) rather than `DEVICE KERNEL DURATION`. Navigation footers are present on both content files. The `index.md` uses clickable markdown links for all file references. No new factual, coherence, or structural correctness issues were found.

## Pass 3

**No feedback — chapter approved.**

All Pass 1 corrections remain intact. Each claim checked against strict scope criteria:

- `what_is_tracy.md`: two-process model correctly attributes timestamping to the client (not the capture server); bash example correctly carries no runtime env var for Tracy; `uint32` thread ID type is accurate per Tracy's data model; zero-cost-when-disabled claim is factually correct (macros expand to nothing without `TRACY_ENABLE`).
- `two_profilers_compared.md`: latency decomposition equation (`total_op_latency = host_dispatch_time + device_kernel_time + sync_overhead`) is correct and each term is assigned to the right tool; `DEVICE KERNEL DURATION` correctly defined as the wall-clock span from first core start to last core end; program-cache-miss blind spot correctly attributes inflation to Tracy's `host_dispatch_time`, not the CSV column.
- Navigation footers: `what_is_tracy.md` links forward to `two_profilers_compared.md`; `two_profilers_compared.md` links forward to `../ch2_invocation/index.md`. Both present and structurally correct.
- `index.md`: all file references use clickable markdown links; no non-linked plain-text file references found.
