# Capture Workflow

This file describes the step-by-step process for capturing a Tracy trace from a tt-metal workload. It covers launch order, required environment variables, a complete two-terminal example, and the failure modes you are most likely to encounter.

This file assumes you have already built tt-metal with `ENABLE_TRACY=ON` and built `tracy-capture` from the `tt_metal/third_party/tracy` submodule. If you have not done both, return to `build_flags.md`.

---

## The Three Components

Every Tracy capture session involves three components. Launch them in the order listed here.

### 1. `tracy-capture` (the capture server)

Start this first, before the profiled process. The capture server listens on TCP port 8086 and waits for a client to connect.

```bash
./tracy-capture -o output.tracy -f
```

- `-o output.tracy`: path where the binary trace database will be written.
- `-f`: force-overwrite the output file if it already exists. Without this flag, `tracy-capture` will refuse to start if the file already exists.

The server continues running and accumulates events until the profiled process exits (or until you interrupt the server with Ctrl-C). Do not interrupt the server before the profiled process has fully exited, or the trace file may be truncated.

### 2. The profiled tt-metal process

Launch your workload after `tracy-capture` is already listening. Without `TRACY_ON_DEMAND=1`, the Tracy client in the process will block at startup until the capture server accepts its connection — this is desirable for profiling sessions where you want complete data from process start.

```bash
TT_METAL_DEVICE_PROFILER=1 TRACY_NO_EXIT=1 python3 my_moe_benchmark.py
```

Environment variables are described in detail in the next section.

### 3. Tracy GUI (`tracy-profiler`) — optional

You may optionally open the Tracy GUI while the capture is running to observe the live trace timeline. The GUI connects to the same capture server or opens an existing `.tracy` file. For most MoE analysis workflows, it is simpler to let the capture complete and then open the resulting file for post-hoc analysis.

---

## Environment Variables for the Profiled Process

Set these in the shell that launches your tt-metal process (Terminal 2 in the example below).

| Variable | Effect |
|---|---|
| `TT_METAL_DEVICE_PROFILER=1` | Activates the on-device CSV profiler. Writes `profile_log_device.csv` to the working directory at process exit. Required if you want device-side kernel timing data alongside the Tracy CPU trace. |
| `TRACY_NO_EXIT=1` | Keeps the process alive after `main()` or script completion until Tracy has finished flushing all buffered events to the capture server. Without this, the process exits before the ring buffer drains. |
| `TRACY_ON_DEMAND=1` | This is a compile-time define, not a runtime env var. Non-blocking behavior must be compiled in with `-DTRACY_ON_DEMAND=1` at build time. Setting it as a shell variable at launch has no effect. See `build_flags.md`. |

> **Warning:** Do not build with `TRACY_ON_DEMAND=1` and also set `TRACY_NO_EXIT=1` for a deliberate profiling session. A binary compiled with `TRACY_ON_DEMAND` discards events when no server is reachable, which can silently produce an empty or partial trace; the flush wait from `TRACY_NO_EXIT=1` will then wait for nothing. Pick one based on your intent.

---

## Complete Two-Terminal Example

```bash
# Terminal 1: start the capture server first
./tracy-capture -o moe_trace.tracy -f

# Terminal 2: run the profiled workload
TT_METAL_DEVICE_PROFILER=1 TRACY_NO_EXIT=1 python3 my_moe_benchmark.py
```

The expected sequence of events:

1. `tracy-capture` prints a message indicating it is listening.
2. The Python script starts; the Tracy client connects to the server and blocks briefly to confirm the handshake.
3. The workload runs; `tracy-capture` reports bytes received in real time.
4. The Python script finishes; `TRACY_NO_EXIT=1` holds the process open while Tracy drains its buffer.
5. The Tracy client disconnects; `tracy-capture` writes and closes `moe_trace.tracy`.
6. The process exits; `profile_log_device.csv` is written to the working directory.

---

## `TRACY_NO_EXIT` in Detail

Tracy buffers recorded events in a memory ring buffer and transmits them to the capture server asynchronously. In long-running C++ processes, the process typically lives long enough for the buffer to drain naturally before exit. Python scripts often do not: the Python interpreter exits quickly after the last line of user code runs, and the Tracy client's background thread may not have finished flushing.

`TRACY_NO_EXIT=1` inserts a blocking wait into the Tracy client's shutdown path. The process does not proceed past the exit hook until the ring buffer is empty and the capture server has acknowledged the final batch of events. The cost is a pause of a few seconds at the end of each run; the benefit is a complete trace file.

> **Tip:** For Python MoE benchmarks, always set `TRACY_NO_EXIT=1`. The missing-tail problem — where the last few seconds of your trace are absent from the `.tracy` file — is almost always caused by omitting this variable.

---

## Common Failure Mode: Client/Server Version Mismatch

Tracy embeds a protocol version number in the handshake. If the client library compiled into your tt-metal binary and the `tracy-capture` binary were built from different Tracy commits, the handshake fails.

**Symptom:** `tracy-capture` prints a version error immediately after the client connects, then either disconnects or produces a `.tracy` file that the GUI refuses to open.

**Fix:** Rebuild `tracy-capture` from the same `tt_metal/third_party/tracy` submodule commit that was used when building tt-metal.

Build `tracy-capture` first as described in `build_flags.md`; the binary will be at `tt_metal/third_party/tracy/capture/build/unix/capture`.

Do not use a `tracy-capture` binary installed from a system package manager or downloaded separately. Pin everything to the submodule.

---

For a full description of the `.tracy` binary and `tracy-csvexport` output format, see `output_format.md`.

---

## Next Steps

Proceed to [`output_format.md`](output_format.md) to learn how to read and post-process both output files, compute inter-zone gaps, and understand how Tracy CPU timestamps relate to device profiler cycle counts.
